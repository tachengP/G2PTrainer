"""Offline binarisation: convert the CSV datasets into compact, mmap-able numpy
binaries so the training loop *streams* rows from disk instead of materialising
the full Python intermediate representation (raw_recs + BPE Counters) in host RAM.

A single binarise pass:

1. scans the CSVs to train the BPE tokenizer and build the vocabularies.  This is
   the only memory-heavy step and it now happens *once*, up front, instead of
   every time training starts (the ~90% RAM spike the user observed at the very
   start of training used to be :func:`src.data.build_dataset` doing exactly
   this).  The encode step (step 2) can run in parallel via ``binarize_workers``.
2. writes CSR-style flat ``int32`` arrays (+ ``int64`` offsets) into
   ``{data_dir}/binary/`` together with the vocab / BPE / metadata files.

Training then loads the binaries with ``np.load(..., mmap_mode="r")`` and builds
tensors on the fly per sample (``num_workers=0`` -- no extra RAM copies, the OS
page cache serves the reads).  The on-disk footprint is a few hundred MB for the
whole Korean set, matching the "binarize then stream" pattern used by DiffSinger.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing import (
    PIPE,
    SLASH,
    TARGET_NAMES,
    Vocab,
    build_source_vocab,
    load_bpe,
    parse_aligned_column,
    parse_no_sep,
    record_targets,
    save_bpe,
)
from src.utils import make_lang_index
from src.data import discover_files, read_records

# Bump when the on-disk format changes so stale binaries are never reloaded.
BINARIZE_VERSION = 1


# --------------------------------------------------------------------------- #
# Multiprocessing worker (module-level so it is picklable on Windows/spawn).
# --------------------------------------------------------------------------- #
_STATE: Dict = {}


def _init_worker(tokenizer, src_vocab, vocab_for, max_src_len, max_tgt_len):
    _STATE["tok"] = tokenizer
    _STATE["src"] = src_vocab
    _STATE["vf"] = vocab_for
    _STATE["max_src"] = max_src_len
    _STATE["max_tgt"] = max_tgt_len


def _encode_row(payload):
    """Encode one CSV record into integer token ids. Returns None if dropped."""
    rec, lang = payload
    tok = _STATE["tok"]
    src_vocab = _STATE["src"]
    vf = _STATE["vf"]
    max_src = _STATE["max_src"]
    max_tgt = _STATE["max_tgt"]

    units = tok.tokenize(rec["src"])
    if not units:
        return None
    if len(units) > max_src:
        return None
    targets = record_targets(rec)
    if any(len(targets[n]) == 0 for n in TARGET_NAMES):
        return None
    if any(len(targets[n]) + 1 > max_tgt for n in TARGET_NAMES):  # +1 for EOS
        return None
    src_ids = src_vocab.encode(units)
    tgt_ids = {n: vf[n].encode(targets[n]) for n in TARGET_NAMES}
    return (src_ids, lang, tgt_ids, rec["src"])


# --------------------------------------------------------------------------- #
# Staleness signature
# --------------------------------------------------------------------------- #
def _signature(cfg, files: List[Tuple[str, str]]) -> str:
    """Hash of every config field that changes the binary content, plus the
    mtime/size of every source CSV so edits trigger a rebuild."""
    h = hashlib.sha256()
    fields = {
        "version": BINARIZE_VERSION,
        "file_glob": cfg.file_glob,
        "bpe_merges": cfg.bpe_merges,
        "min_freq": cfg.min_freq,
        "max_src_len": cfg.max_src_len,
        "max_tgt_len": cfg.max_tgt_len,
        "phoneme_set": cfg.phoneme_set,
        "val_split": cfg.val_split,
        "seed": cfg.seed,
        "max_samples": cfg.max_samples,
    }
    h.update(repr(fields).encode("utf-8"))
    for p, lang in sorted(files):
        h.update(p.encode("utf-8"))
        h.update(lang.encode("utf-8"))
        st = os.stat(p)
        h.update(str(st.st_mtime).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Binarise
# --------------------------------------------------------------------------- #
def run_binarize(cfg, binary_dir: str, binarize_workers: int = 8) -> None:
    """Read the CSVs, build vocab/BPE, encode every row (optionally in parallel)
    and write compact numpy binaries + vocab/BPE/meta into ``binary_dir``."""
    from tqdm import tqdm

    os.makedirs(binary_dir, exist_ok=True)
    files = discover_files(cfg.data_dir, cfg.file_glob)
    if not files:
        raise RuntimeError(f"No dataset files matched {cfg.file_glob!r} in {cfg.data_dir}")

    # ---- Pass 1: read every record (the only big RAM holder) ----
    raw_recs: List[Dict[str, str]] = []
    langs: List[str] = []
    for path, lang in files:
        for rec in read_records(path, limit=cfg.max_samples):
            raw_recs.append(rec)
            langs.append(lang)
    if cfg.max_samples is not None and cfg.max_samples > 0:
        raw_recs = raw_recs[: cfg.max_samples]
        langs = langs[: cfg.max_samples]
    print(f"[binarize] read {len(raw_recs)} raw records from {len(files)} file(s)")

    # ---- build source vocab + BPE tokenizer ----
    tokenizer, src_vocab = build_source_vocab(
        (r["src"] for r in raw_recs), cfg.bpe_merges, cfg.min_freq
    )

    # ---- phoneme vocab (union of the three phoneme tasks) + separators ----
    phoneme_syms: Counter = Counter()
    for r in raw_recs:
        phoneme_syms.update(parse_no_sep(r["phonemes"]))
        phoneme_syms.update(parse_aligned_column(r["separated_phonemes"], PIPE, split_within=True))
        phoneme_syms.update(parse_aligned_column(r["aligned_phonemes"], SLASH, split_within=True))
    ph_symbols = [s for s, _ in phoneme_syms.most_common() if s]
    if cfg.phoneme_set is not None:
        ph_symbols = [s for s in ph_symbols if s in set(cfg.phoneme_set)]
    phoneme_vocab = Vocab(ph_symbols + [PIPE, SLASH])

    # ---- grapheme-target vocab (the 'separated_graphmes' task) + separator ----
    gr_syms: Counter = Counter()
    for r in raw_recs:
        gr_syms.update(parse_aligned_column(r["separated_graphmes"], PIPE, split_within=False))
    gr_symbols = [s for s, _ in gr_syms.most_common() if s]
    grapheme_tgt_vocab = Vocab(gr_symbols + [PIPE])

    vocab_for = {
        "phonemes": phoneme_vocab,
        "separated_graphmes": grapheme_tgt_vocab,
        "separated_phonemes": phoneme_vocab,
        "aligned_phonemes": phoneme_vocab,
    }
    lang2id, _ = make_lang_index(langs)

    # ---- Pass 2: encode every row (parallel via multiprocessing) ----
    payloads = list(zip(raw_recs, langs))
    del raw_recs  # free the large list wrapper; dicts stay referenced by payloads

    results: List[Optional[Tuple]] = []
    if binarize_workers and binarize_workers > 1:
        import multiprocessing as mp

        with mp.Pool(
            processes=binarize_workers,
            initializer=_init_worker,
            initargs=(tokenizer, src_vocab, vocab_for, cfg.max_src_len, cfg.max_tgt_len),
        ) as pool:
            for res in tqdm(
                pool.imap(_encode_row, payloads, chunksize=256),
                total=len(payloads),
                desc="binarize encode",
                unit="row",
            ):
                results.append(res)
    else:
        for pl in tqdm(payloads, desc="binarize encode", unit="row"):
            results.append(_encode_row(pl))

    samples: List[Tuple] = [r for r in results if r is not None]
    dropped = len(results) - len(samples)
    if dropped:
        print(f"[binarize] dropped {dropped} empty / over-length rows")
    # map the string language label -> compact int id (the dataset resolves the
    # id back to the string label on load, as the model expects string labels)
    samples = [(s[0], lang2id[s[1]], s[2], s[3], s[1]) for s in samples]
    del results, payloads

    # ---- deterministic train / val split (shuffling rows, text stays paired) ----
    rng = random.Random(cfg.seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * cfg.val_split))
    val_rows = samples[:n_val]
    train_rows = samples[n_val:]
    print(f"[binarize] {len(train_rows)} train / {len(val_rows)} val rows")

    _write_split(os.path.join(binary_dir, "train.npz"), train_rows)
    _write_split(os.path.join(binary_dir, "val.npz"), val_rows)

    # val monitoring payload (original grapheme text) for TensorBoard / saved preds
    with open(os.path.join(binary_dir, "val_monitor.jsonl"), "w", encoding="utf-8") as f:
        for _, _lang_id, _tgt, text, lang_str in val_rows:
            f.write(json.dumps({"text": text, "lang": lang_str}, ensure_ascii=False) + "\n")

    # ---- vocab / BPE / meta ----
    src_vocab.to_file(os.path.join(binary_dir, "src_vocab.txt"))
    phoneme_vocab.to_file(os.path.join(binary_dir, "phoneme_vocab.txt"))
    grapheme_tgt_vocab.to_file(os.path.join(binary_dir, "grapheme_tgt_vocab.txt"))
    save_bpe(tokenizer, os.path.join(binary_dir, "bpe.txt"))

    meta = {
        "langs": sorted(set(langs)),
        "num_langs": len(lang2id),
        "lang2id": lang2id,
        "id2lang": {i: l for l, i in lang2id.items()},
        "src_vocab_size": len(src_vocab),
        "phoneme_vocab_size": len(phoneme_vocab),
        "grapheme_tgt_vocab_size": len(grapheme_tgt_vocab),
        "binarize_version": BINARIZE_VERSION,
    }
    with open(os.path.join(binary_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[binarize] wrote binaries to {binary_dir}")


def _write_split(npz_path: str, rows: List[Tuple]) -> None:
    """Write CSR-style flat int32 arrays (+ int64 offsets) for one split."""
    src_flat: List[int] = []
    src_off: List[int] = [0]
    lang_arr: List[int] = []
    tgt_flat = {n: [] for n in TARGET_NAMES}
    tgt_off = {n: [0] for n in TARGET_NAMES}

    for src_ids, lang, tgt_ids, _text, _lang_str in rows:
        src_flat.extend(src_ids)
        src_off.append(len(src_flat))
        lang_arr.append(lang)
        for n in TARGET_NAMES:
            tgt_flat[n].extend(tgt_ids[n])
            tgt_off[n].append(len(tgt_flat[n]))

    arrays = {
        "src": np.asarray(src_flat, dtype=np.int32),
        "src_off": np.asarray(src_off, dtype=np.int64),
        "lang": np.asarray(lang_arr, dtype=np.int32),
    }
    for n in TARGET_NAMES:
        arrays[f"tgt_{n}"] = np.asarray(tgt_flat[n], dtype=np.int32)
        arrays[f"tgt_{n}_off"] = np.asarray(tgt_off[n], dtype=np.int64)
    np.savez(npz_path, **arrays)


# --------------------------------------------------------------------------- #
# Loading (streaming)
# --------------------------------------------------------------------------- #
class BinaryG2PDataset(Dataset):
    """A Dataset that reads pre-encoded rows straight from an mmap'd ``.npz``.

    Only one sample's worth of tensors is materialised per ``__getitem__`` call,
    so host RAM stays flat regardless of dataset size (the OS page cache serves
    the disk reads).  The mmap is opened lazily per process, so it also works
    with ``num_workers > 0`` (parallel decode + prefetch keep the GPU fed).
    """

    def __init__(self, npz_path: str, monitor_path: Optional[str] = None) -> None:
        # Only store paths; the mmap is opened lazily (per worker) so the dataset
        # stays picklable and works with num_workers > 0. Each worker process
        # opens its OWN mmap, enabling parallel decode + prefetch that keeps the
        # GPU fed instead of stalling between batches.
        self.npz_path = npz_path
        self._npz = None
        self._mem = None
        self.N: Optional[int] = None
        self._materialized: bool = False
        self.samples: List[Dict] = []

        self.monitor: List[Dict] = []
        if monitor_path and os.path.exists(monitor_path):
            with open(monitor_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.monitor.append(json.loads(line))

    def _ensure_open(self) -> None:
        if self._materialized:
            return
        if self._npz is None:
            self._npz = np.load(self.npz_path, mmap_mode="r")
            self._mem = {
                "src": self._npz["src"],
                "src_off": self._npz["src_off"],
                "lang": self._npz["lang"],
                **{f"tgt_{n}": self._npz[f"tgt_{n}"] for n in TARGET_NAMES},
                **{f"tgt_{n}_off": self._npz[f"tgt_{n}_off"] for n in TARGET_NAMES},
            }
        self.N = int(self._mem["lang"].shape[0])

    def __getstate__(self):
        # The open mmap handle / memmaps can't be pickled.  Worker processes
        # reopen them lazily via _ensure_open(), so drop them before pickling.
        state = self.__dict__.copy()
        state["_npz"] = None
        state["_mem"] = None
        return state

    def __len__(self) -> int:
        self._ensure_open()
        return self.N

    def _get(self, i: int) -> Dict:
        self._ensure_open()
        m = self._mem
        s0, s1 = int(m["src_off"][i]), int(m["src_off"][i + 1])
        src = torch.from_numpy(np.asarray(m["src"][s0:s1], dtype=np.int64))
        targets = {}
        for n in TARGET_NAMES:
            o0, o1 = int(m[f"tgt_{n}_off"][i]), int(m[f"tgt_{n}_off"][i + 1])
            targets[n] = torch.from_numpy(np.asarray(m[f"tgt_{n}"][o0:o1], dtype=np.int64))
        lang_id = int(m["lang"][i])
        return {
            "src": src,
            "src_len": src.size(0),
            "lang": lang_id,
            "targets": targets,
        }

    def __getitem__(self, i: int) -> Dict:
        if self._materialized:
            s = self.samples[i]
            return {
                "src": s["src"],
                "src_len": s["src"].size(0),
                "lang": s["lang"],
                "targets": s["targets"],
            }
        return self._get(i)

    def to_device(self, device) -> "BinaryG2PDataset":
        """Materialise every row onto ``device`` (used for ``data_on_gpu``)."""
        self._ensure_open()
        self.samples = [self._get(i) for i in range(self.N)]
        self._materialized = True
        self._npz = None  # release the mmap handle
        self._mem = None
        for s in self.samples:
            s["src"] = s["src"].to(device)
            for n in list(s["targets"].keys()):
                s["targets"][n] = s["targets"][n].to(device)
        return self


def load_binary(binary_dir: str, device: str = "cpu"):
    """Load a binarised dataset (streaming) and return the same tuple as
    :func:`src.data.build_dataset`."""
    src_vocab = Vocab.from_file(os.path.join(binary_dir, "src_vocab.txt"))
    phoneme_vocab = Vocab.from_file(os.path.join(binary_dir, "phoneme_vocab.txt"))
    grapheme_tgt_vocab = Vocab.from_file(os.path.join(binary_dir, "grapheme_tgt_vocab.txt"))
    tokenizer = load_bpe(os.path.join(binary_dir, "bpe.txt"))
    with open(os.path.join(binary_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    pad_idx_dict = {
        "phonemes": phoneme_vocab.pad_idx,
        "separated_graphmes": grapheme_tgt_vocab.pad_idx,
        "separated_phonemes": phoneme_vocab.pad_idx,
        "aligned_phonemes": phoneme_vocab.pad_idx,
    }
    train_ds = BinaryG2PDataset(os.path.join(binary_dir, "train.npz"))
    val_ds = BinaryG2PDataset(
        os.path.join(binary_dir, "val.npz"),
        monitor_path=os.path.join(binary_dir, "val_monitor.jsonl"),
    )
    return (
        train_ds,
        val_ds,
        src_vocab,
        phoneme_vocab,
        grapheme_tgt_vocab,
        tokenizer,
        pad_idx_dict,
        meta,
        val_ds.monitor,
    )


# --------------------------------------------------------------------------- #
# Auto-binarize orchestration (called from train.py)
# --------------------------------------------------------------------------- #
def ensure_binary(cfg) -> str:
    """Return the binary dir, (re)building it only when missing or stale."""
    binary_dir = cfg.binary_dir or os.path.join(cfg.data_dir, "binary")
    files = discover_files(cfg.data_dir, cfg.file_glob)
    sig = _signature(cfg, files)
    sig_path = os.path.join(binary_dir, "binarize_sig.json")

    fresh = (
        os.path.isfile(sig_path)
        and os.path.isfile(os.path.join(binary_dir, "train.npz"))
        and os.path.isfile(os.path.join(binary_dir, "val.npz"))
        and os.path.isfile(os.path.join(binary_dir, "meta.json"))
    )
    if fresh:
        with open(sig_path, "r", encoding="utf-8") as f:
            if json.load(f).get("signature") == sig:
                print(f"[binarize] up to date -> {binary_dir}")
                return binary_dir

    print(f"[binarize] (re)building -> {binary_dir}")
    run_binarize(cfg, binary_dir, binarize_workers=cfg.binarize_workers)
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump({"signature": sig}, f)
    return binary_dir


def binarize_from_config(cfg) -> None:
    """Standalone CLI entry: ``python -m src.binarize``."""
    binary_dir = cfg.binary_dir or os.path.join(cfg.data_dir, "binary")
    run_binarize(cfg, binary_dir, binarize_workers=cfg.binarize_workers)


if __name__ == "__main__":
    from src.config import build_config

    binarize_from_config(build_config())
