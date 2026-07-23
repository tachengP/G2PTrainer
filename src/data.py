"""Data discovery, CSV parsing and dataset construction for the multitask G2P.

The training CSV is expected to carry a header whose columns are read directly:

    graphmes, phonemes, separated_graphmes, separated_phonemes, aligned_phonemes

See :data:`src.preprocessing.CSV_COLUMNS` for the canonical mapping of each
column to one of the four training targets (plus the source sequence).

A single CSV file can be large (the real Korean set is ~120 MB).  Two mechanisms
keep things tractable:

* ``max_samples`` caps the number of rows used (handy for quick smoke tests).
* Source-side vocabularies are BPE-compressed so even huge grapheme inventories
  collapse to a small, fixed-size embedding table.
"""

from __future__ import annotations

import csv
import glob
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from src.preprocessing import (
    PIPE,
    SLASH,
    TARGET_NAMES,
    CSV_COLUMNS,
    Vocab,
    build_source_vocab,
    parse_aligned_column,
    parse_no_sep,
    record_targets,
)
from src.utils import make_lang_index, parse_lang_from_filename


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def discover_files(data_dir: str, file_glob: str) -> List[Tuple[str, str]]:
    """Return [(csv_path, lang), ...] for every matching dataset file.

    A file named ``dataset-ko.csv`` contributes the language label ``"ko"``.
    """
    pattern = os.path.join(data_dir, file_glob)
    paths = sorted(glob.glob(pattern))
    out: List[Tuple[str, str]] = []
    for p in paths:
        lang = parse_lang_from_filename(p)
        if lang is None:
            continue
        out.append((p, lang))
    return out


# --------------------------------------------------------------------------- #
# CSV reading (header-driven, column-name aware)
# --------------------------------------------------------------------------- #
def read_records(path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts keyed by *canonical* column names.

    The header is inspected and matched (case-insensitively) against
    :data:`CSV_COLUMNS` so the loader stays in sync with the real dataset
    column names (``graphmes``, ``phonemes``, ``separated_graphmes``,
    ``separated_phonemes``, ``aligned_phonemes``). ``limit`` stops reading
    after that many data rows (useful for quick smoke tests on huge files).
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        header_lc = [h.strip().lower() for h in header]
        idx: Dict[str, int] = {}
        for canonical, candidates in CSV_COLUMNS.items():
            for cand in candidates:
                if cand.lower() in header_lc:
                    idx[canonical] = header_lc.index(cand.lower())
                    break
        required = ("src", "phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes")
        missing = [k for k in required if k not in idx]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)}: missing required columns {missing}; "
                f"found header {header}"
            )
        records: List[Dict[str, str]] = []
        for row in reader:
            if len(row) <= max(idx.values()):
                continue
            records.append({k: row[i] for k, i in idx.items()})
            if limit is not None and len(records) >= limit:
                break
    return records


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(
    data_dir: str,
    file_glob: str,
    bpe_merges: int,
    min_freq: int,
    max_src_len: int,
    max_tgt_len: int,
    phoneme_set: Optional[List[str]] = None,
    val_split: float = 0.05,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple:
    """Build train/val splits and return the training artifacts as a tuple.

    Returns
    -------
    (train_ds, val_ds, src_vocab, phoneme_vocab, grapheme_tgt_vocab,
     tokenizer, pad_idx_dict, meta)
    """
    files = discover_files(data_dir, file_glob)
    if not files:
        raise RuntimeError(f"No dataset files matched {file_glob!r} in {data_dir}")

    # 1. Read every record (capped per language file) and remember the language.
    raw_recs: List[Dict[str, str]] = []
    langs: List[str] = []
    for path, lang in files:
        for rec in read_records(path, limit=max_samples):
            raw_recs.append(rec)
            langs.append(lang)

    if max_samples is not None and max_samples > 0:
        raw_recs = raw_recs[:max_samples]
        langs = langs[:max_samples]

    # 2. Source vocab + BPE tokenizer (trained on the grapheme source column).
    tokenizer, src_vocab = build_source_vocab(
        (r["src"] for r in raw_recs), bpe_merges, min_freq
    )

    # 3. Phoneme vocab (union of the three phoneme tasks) + separators.
    phoneme_syms: Counter = Counter()
    for r in raw_recs:
        phoneme_syms.update(parse_no_sep(r["phonemes"]))
        phoneme_syms.update(parse_aligned_column(r["separated_phonemes"], PIPE, split_within=True))
        phoneme_syms.update(parse_aligned_column(r["aligned_phonemes"], SLASH, split_within=True))
    ph_symbols = [s for s, _ in phoneme_syms.most_common() if s]
    if phoneme_set is not None:
        ph_symbols = [s for s in ph_symbols if s in set(phoneme_set)]
    phoneme_vocab = Vocab(ph_symbols + [PIPE, SLASH])

    # 4. Grapheme-target vocab (the 'separated_graphmes' task) + separator.
    gr_syms: Counter = Counter()
    for r in raw_recs:
        gr_syms.update(parse_aligned_column(r["separated_graphmes"], PIPE, split_within=False))
    gr_symbols = [s for s, _ in gr_syms.most_common() if s]
    grapheme_tgt_vocab = Vocab(gr_symbols + [PIPE])

    # 5. Encode samples.
    vocab_for = {
        "phonemes": phoneme_vocab,
        "separated_graphmes": grapheme_tgt_vocab,
        "separated_phonemes": phoneme_vocab,
        "aligned_phonemes": phoneme_vocab,
    }
    lang2id, _ = make_lang_index(langs)

    samples: List[Dict] = []
    # monitoring payload (original grapheme text) is kept OUT of the worker
    # dataset: only the main process needs it for TensorBoard / saved
    # predictions, so keeping it here would waste (num_workers+1) copies of
    # every source string in host RAM.
    monitor: List[Dict] = []
    for r, lang in zip(raw_recs, langs):
        units = tokenizer.tokenize(r["src"])
        if not units:
            continue
        targets = record_targets(r)
        # drop rows whose targets are empty or exceed the length budget
        if any(len(targets[n]) == 0 for n in TARGET_NAMES):
            continue
        if len(units) > max_src_len:
            continue
        if any(len(targets[n]) + 1 > max_tgt_len for n in TARGET_NAMES):  # +1 for EOS
            continue
        samples.append(
            {
                # tensors are stored directly so the dataset can be placed in
                # shared memory (one physical copy shared by all workers).
                "src": torch.tensor(src_vocab.encode(units), dtype=torch.long),
                "lang": lang2id[lang],
                "targets": {
                    n: torch.tensor(vocab_for[n].encode(targets[n]), dtype=torch.long)
                    for n in TARGET_NAMES
                },
            }
        )
        monitor.append({"text": r["src"], "lang": lang})

    # 6. Train / val split (deterministic).  Shuffle samples and their
    #    monitoring entries together so indices stay aligned.
    rng = random.Random(seed)
    paired = list(zip(samples, monitor))
    rng.shuffle(paired)
    samples, monitor = zip(*paired) if paired else ([], [])
    samples = list(samples)
    monitor = list(monitor)
    n_val = max(1, int(len(samples) * val_split))
    val = samples[:n_val]
    train = samples[n_val:]
    val_monitor = monitor[:n_val]

    pad_idx_dict = {n: vocab_for[n].pad_idx for n in TARGET_NAMES}
    meta = {
        "langs": sorted(set(langs)),
        "num_langs": len(lang2id),
        "lang2id": lang2id,
        "id2lang": {i: l for l, i in lang2id.items()},
        "src_vocab_size": len(src_vocab),
        "phoneme_vocab_size": len(phoneme_vocab),
        "grapheme_tgt_vocab_size": len(grapheme_tgt_vocab),
    }

    return (
        G2PDataset(train),
        G2PDataset(val),
        src_vocab,
        phoneme_vocab,
        grapheme_tgt_vocab,
        tokenizer,
        pad_idx_dict,
        meta,
        val_monitor,
    )


# --------------------------------------------------------------------------- #
# PyTorch dataset + loader
# --------------------------------------------------------------------------- #
class G2PDataset(Dataset):
    def __init__(self, samples: List[Dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "src": s["src"],
            "src_len": s["src"].size(0),
            "lang": s["lang"],
            "targets": s["targets"],
        }

    def share_memory(self) -> "G2PDataset":
        """Put every stored tensor into shared memory so the (num_workers)
        DataLoader subprocesses map the *same* physical buffers instead of
        each copying the whole dataset.  Call before constructing the loader.
        """
        for s in self.samples:
            s["src"].share_memory_()
            for t in s["targets"].values():
                t.share_memory_()
        return self

    def to_device(self, device) -> "G2PDataset":
        """Move every stored tensor onto ``device`` in place (used to keep the
        whole dataset resident on CUDA so host RAM is freed).  Only valid with
        ``num_workers=0`` (a single process owns the tensors).
        """
        for s in self.samples:
            s["src"] = s["src"].to(device)
            for n in list(s["targets"].keys()):
                s["targets"][n] = s["targets"][n].to(device)
        return self


def collate_fn(batch: List[Dict], pad_idx_dict: Dict[str, int]):
    """Collate a list of samples into a batch (module-level, picklable).

    Returns ``(src, src_len, lang, targets, tgt_lens)`` where ``src`` is
    batch-first ``[B, S]`` and ``targets`` is batch-first ``[B, T]``;
    :meth:`src.model.G2PModel.forward` transposes both to seq-first internally.

    Padding is done with :func:`torch.nn.utils.rnn.pad_sequence` (single C-level
    copy) instead of a Python per-sample scatter loop, which keeps the main-thread
    collate cheap so the GPU is not left idle between steps.
    """
    # Build output tensors on the same device as the samples, so this works
    # for both CPU (shared-memory) and CUDA-resident datasets without host
    # round-trips or device-mismatch errors.
    device = batch[0]["src"].device
    src_lens = torch.tensor([b["src_len"] for b in batch], dtype=torch.long, device=device)
    langs = torch.tensor([b["lang"] for b in batch], dtype=torch.long, device=device)

    src = pad_sequence([b["src"] for b in batch], batch_first=True, padding_value=0).to(device)
    targets = {}
    tgt_lens = {}
    for n in TARGET_NAMES:
        seqs = [b["targets"][n] for b in batch]
        tgt_lens[n] = torch.tensor([s.size(0) for s in seqs], dtype=torch.long, device=device)
        targets[n] = pad_sequence(seqs, batch_first=True, padding_value=pad_idx_dict[n]).to(device)
    return src, src_lens, langs, targets, tgt_lens


class LengthSortedBatchSampler(Sampler):
    """Yield batches of ``batch_size`` *consecutive* indices after sorting every
    sample by length, so each batch groups similar-length sequences together.

    Why this is essential: the encoder BiLSTM (``src/model.py``) and the decoder
    cross-attention both run over the *padded* length of the longest member of a
    batch.  With random batches the max length grows with ``batch_size`` (more
    chances of a long outlier), so per-step compute -- and therefore per-epoch
    wall-time -- grows **super-linearly** and a *larger* batch can be *slower*
    (the bug the user hit: batch 1024 took 2x longer than batch 128 while VRAM
    was still half empty).  Sorting by length makes padding ~constant regardless
    of batch size, so bigger batches just mean fewer steps and a faster epoch.

    To keep training stochastic we split the length-sorted list into fixed-size
    "buckets", shuffle the *order* of the buckets each epoch (with a fresh seed),
    but preserve the sorted order *within* a bucket -- so padding stays minimal
    while the per-epoch sample order is randomised.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # length key = max(src_len, longest target) so BOTH encoder and decoder
        # padding shrink when samples of similar length are batched together.
        self.lengths = [self._len_of(i) for i in range(len(dataset))]
        self._sorted = sorted(range(len(dataset)), key=lambda i: self.lengths[i], reverse=True)
        # Split the length-sorted list into contiguous, length-ordered buckets.
        # Buckets stay in length order (so batch boundaries are only ~1 length
        # unit apart -> minimal padding), while samples are shuffled *within*
        # each bucket every epoch to keep SGD stochastic.
        n = len(self._sorted)
        n_batches = max(1, n // batch_size)
        nbuckets = max(1, min(n_batches, 50))
        self.bucket_size = max(batch_size, (n + nbuckets - 1) // nbuckets)
        self.buckets = [
            self._sorted[k * self.bucket_size:(k + 1) * self.bucket_size]
            for k in range((len(self._sorted) + self.bucket_size - 1) // self.bucket_size)
        ]

    def _len_of(self, i: int) -> int:
        item = self.dataset[i]
        L = int(item["src_len"])
        for t in item["targets"].values():
            tl = int(t.size(0))
            if tl > L:
                L = tl
        return L

    def __iter__(self):
        rng = random.Random()
        indices = []
        for b in self.buckets:
            b = list(b)
            if self.shuffle:
                rng.shuffle(b)  # randomise sample identity within a length range
            indices.extend(b)
        for s in range(0, len(indices), self.batch_size):
            batch = indices[s:s + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = len(self._sorted)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


def make_loader(dataset, batch_size, pad_idx_dict, num_workers, shuffle, pin_memory=True,
                sort_by_length=True):
    """Return a DataLoader yielding (src, src_len, lang, targets, tgt_lens).

    Data-side knobs that keep a fast GPU (e.g. RTX 5080) fed and maximise
    utilisation:
      * ``sort_by_length`` -- group similar-length samples in each batch so
        padding (and therefore BiLSTM + attention cost) stops exploding with
        batch size; this is what lets a larger batch actually run faster.
      * ``pin_memory``  -- page-locks CPU tensors so the H2D copy is async/fast.
      * ``persistent_workers`` -- keep workers alive across epochs (avoids the
        fork/spawn cost every epoch, which otherwise leaves the GPU idle).
      * ``prefetch_factor`` -- let each worker stage more batches ahead so the
        GPU never stalls waiting on the next one.
    """
    from functools import partial

    common = dict(
        num_workers=num_workers,
        collate_fn=partial(collate_fn, pad_idx_dict=pad_idx_dict),
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if sort_by_length:
        # batch_sampler drives the order; DataLoader must not also shuffle.
        sampler = LengthSortedBatchSampler(
            dataset, batch_size, shuffle=shuffle, drop_last=shuffle
        )
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,  # skip the ragged last batch in training (fuller kernels)
        **common,
    )
