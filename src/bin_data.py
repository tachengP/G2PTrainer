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
    parse_no_sep,
    parse_phoneme_counts,
    parse_grapheme_counts,
    parse_counts,
    record_targets,
    CountCodec,
    save_bpe,
)
from src.utils import make_lang_index
from src.data import discover_files, read_records

# Bump when the on-disk format changes so stale binaries are never reloaded.
BINARIZE_VERSION = 2


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
    targets = record_targets(rec, units)
    # Drop rows whose derived-task segment counts disagree with the base
    # sequences (a pre-existing source-CSV glitch: e.g. the `separated_phonemes`
    # column's segment sizes sum to 19 while `phonemes` holds 20 tokens). Such
    # rows would train the model to emit counts that cannot regroup the predicted
    # phonemes, so they are filtered out rather than silently kept.
    # NOTE: `phonemes` in `targets` is SOS/EOS-framed, so compare against the raw
    # phoneme token count, not the framed length.
    ph_tokens = parse_no_sep(rec["phonemes"])
    if sum(targets["separated_phonemes"]) != len(ph_tokens):
        return None
    if sum(targets["aligned_phonemes"]) != len(ph_tokens):
        return None
    if sum(targets["separated_graphmes"]) != len(units):
        return None
    tgt_ids = {}
    for n in TARGET_NAMES:
        enc = vf[n].encode(targets[n])
        if len(enc) == 0 or len(enc) > max_tgt:
            return None
        tgt_ids[n] = enc
    src_ids = src_vocab.encode(units)
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
    for p, lang, _sic in sorted(files):
        h.update(p.encode("utf-8"))
        h.update(lang.encode("utf-8"))
        st = os.stat(p)
        h.update(str(st.st_mtime).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Binarise
# --------------------------------------------------------------------------- #
# Minimum confidence (votes / total) and frequency support a phoneme needs to be
# called an "obvious" vowel from the aligned_phonemes signal.  In CJK-like
# syllable languages the vowel and consonant symbol sets are disjoint, so the
# post-`/` first-token vote cleanly separates the two (vowels ~1.0, consonants
# ~0.0); a modest threshold keeps the decision robust.
VOWEL_CONF_THRESHOLD = 0.5
MIN_VOWEL_SUPPORT = 5


def _derive_vowels(raw_recs, langs, override, max_iters=50):
    """Derive per-language vowel phoneme sets.

    Two stages, matching the geometry of CJK-like syllable languages (one
    character == one independent syllable, so every ``|``/``/`` group carries
    exactly one vowel nucleus):

    STAGE 1 -- vowel confidence from ``aligned_phonemes``.
        The aligned column writes each syllable as ``onset / vowel[ #coda]``,
        i.e. the token *immediately after* every ``/`` separator is the vowel
        nucleus (e.g. ``/a #k r/i #l`` -> ``a`` and ``i`` follow a ``/``, so they
        accumulate vowel confidence).  The count-form aligned column stores the
        size of every ``/`` segment, so we reconstruct the segments from
        ``aligned_phonemes`` counts + the flat ``phonemes`` sequence and tally,
        per symbol, how often it is the first token of a segment that *follows*
        a ``/``.  Dividing by the symbol's total frequency yields a clean vowel
        confidence: a nucleus symbol scores ~1.0, an onset/coda symbol ~0.0.

        The leading onset cluster of a word sits in segment 0 (before any ``/``)
        and is skipped; every genuine vowel still accumulates votes via the
        C-initial syllables that occur elsewhere in the corpus, so none is
        missed.

    STAGE 2 -- apply the CJK per-syllable constraint to ``separated_phonemes``.
        Each ``|``-separated group is one syllable and must contain exactly one
        vowel from the Stage-1 set.  We (a) report the fraction of groups that
        violate this as a diagnostic (``vowel_constraint_violation`` in meta),
        and (b) prune any Stage-1 vowel that *never* serves as the unique vowel
        of any group -- a sure sign it is an onset/consonant that slipped past
        Stage 1.  Real vowels are always the unique vowel of their groups, so
        this pruning only removes genuine errors.

    Returns ``{lang: (set_of_symbols, confidence_dict, violation_rate)}`` where
    ``confidence_dict`` maps every symbol to its Stage-1 vowel confidence.
    ``override`` (a ``{lang: [symbols]}`` map) short-circuits the derivation.
    """
    from src.preprocessing import parse_no_sep, parse_counts

    # ---- Stage 1: collect post-`/` first-token votes from aligned_phonemes ----
    vowel_votes = {}   # lang -> {sym: #times it is the first token after a '/'}
    sym_total = {}     # lang -> {sym: total frequency}
    for rec, lang in zip(raw_recs, langs):
        if lang not in vowel_votes:
            vowel_votes[lang] = {}
            sym_total[lang] = {}
        ph = parse_no_sep(rec["phonemes"])
        for sym in ph:
            if sym:
                sym_total[lang][sym] = sym_total[lang].get(sym, 0) + 1
        counts = parse_counts(rec["aligned_phonemes"])
        idx = 0
        for ci, c in enumerate(counts):
            c = int(c)
            if c <= 0:
                continue
            seg = ph[idx:idx + c]
            idx += c
            # The first token of every segment that follows a '/' is the vowel.
            # Segment 0 is the leading onset cluster (before any '/') -> skip it.
            if ci >= 1 and seg:
                head = seg[0]
                vowel_votes[lang][head] = vowel_votes[lang].get(head, 0) + 1

    # confidence + obvious-vowel selection
    conf: Dict[str, Dict[str, float]] = {}
    candidates: Dict[str, set] = {}
    for lang in set(langs):
        if override and lang in override and override[lang]:
            vs = set(s for s in override[lang] if s)
            candidates[lang] = vs
            conf[lang] = {s: 1.0 for s in vs}
            continue
        lang_conf = {}
        for sym, tot in sym_total.get(lang, {}).items():
            v = vowel_votes.get(lang, {}).get(sym, 0)
            lang_conf[sym] = (v / tot) if tot > 0 else 0.0
        conf[lang] = lang_conf
        vs = {s for s, c in lang_conf.items()
              if c >= VOWEL_CONF_THRESHOLD and sym_total.get(lang, {}).get(s, 0) >= MIN_VOWEL_SUPPORT}
        candidates[lang] = vs

    # ---- Stage 2: CJK one-vowel-per-`|`-group constraint on separated_phonemes ----
    violation: Dict[str, float] = {}
    for lang in candidates:
        if override and lang in override and override[lang]:
            violation[lang] = 0.0
            continue
        vs = candidates[lang]
        n_groups = 0
        n_bad = 0
        ever_unique = set()  # vowels that serve as the unique vowel of some group
        for rec, rlang in zip(raw_recs, langs):
            if rlang != lang:
                continue
            ph = parse_no_sep(rec["phonemes"])
            counts = parse_counts(rec["separated_phonemes"])
            idx = 0
            for c in counts:
                c = int(c)
                if c <= 0:
                    continue
                grp = ph[idx:idx + c]
                idx += c
                if not grp:
                    continue
                n_groups += 1
                v_in = [t for t in grp if t in vs]
                if len(v_in) == 1:
                    ever_unique.add(v_in[0])
                elif len(v_in) != 1:
                    n_bad += 1
        # prune vowels that never act as the unique vowel of any group
        pruned = {v for v in vs if v not in ever_unique}
        if pruned:
            vs = vs - pruned
            candidates[lang] = vs
        violation[lang] = (n_bad / n_groups) if n_groups else 0.0

    result = {}
    for lang in candidates:
        result[lang] = (candidates[lang], conf[lang], violation.get(lang, 0.0))
    return result


def run_binarize(cfg, binary_dir: str, binarize_workers: int = 8) -> None:
    """Read the CSVs, build vocab/BPE, encode every row (optionally in parallel)
    and write compact numpy binaries + vocab/BPE/meta into ``binary_dir``."""
    from tqdm import tqdm

    if binary_dir is None:
        binary_dir = os.path.join(cfg.data_dir, "binary")
    os.makedirs(binary_dir, exist_ok=True)
    files = discover_files(cfg.data_dir, cfg.lang_define, cfg.file_glob)
    if not files:
        raise RuntimeError(f"No dataset files matched in {cfg.data_dir}")

    # per-language "one character == one independent syllable" flag
    lang_syllable_char = {}
    for _p, _l, _sic in files:
        lang_syllable_char[_l] = lang_syllable_char.get(_l, False) or _sic

    # ---- Pass 1: read every record (the only big RAM holder) ----
    raw_recs: List[Dict[str, str]] = []
    langs: List[str] = []
    for path, lang, _sic in files:
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

    # ---- phoneme vocab (base phonemes only; separators live in the count tasks) ----
    phoneme_syms: Counter = Counter()
    for r in raw_recs:
        phoneme_syms.update(parse_no_sep(r["phonemes"]))
    ph_symbols = [s for s, _ in phoneme_syms.most_common() if s]
    if cfg.phoneme_set is not None:
        ph_symbols = [s for s in ph_symbols if s in set(cfg.phoneme_set)]
    phoneme_vocab = Vocab(ph_symbols)  # no PIPE/SLASH anymore

    # ---- count vocab size: max segment length across the three derived tasks ----
    #    Grapheme counts are computed char-wise here (an upper bound on the true
    #    token-wise counts used in pass 2); phoneme counts are already exact.
    max_count = 1
    for r in raw_recs:
        for c in parse_counts(r["separated_phonemes"]):
            if c > max_count:
                max_count = c
        for c in parse_counts(r["aligned_phonemes"]):
            if c > max_count:
                max_count = c
        for c in parse_counts(r["separated_graphmes"]):
            if c > max_count:
                max_count = c
    count_codec = CountCodec(max_count)

    vocab_for = {
        "phonemes": phoneme_vocab,
        "separated_graphmes": count_codec,
        "separated_phonemes": count_codec,
        "aligned_phonemes": count_codec,
    }
    lang2id, _ = make_lang_index(langs)

    # ---- derive vowel phoneme sets (one vowel nucleus per syllable) ----
    # For syllable_is_char languages this lets inference enforce that every
    # phoneme group (between `|`/`/`) contains exactly one vowel -- the structural
    # rule the count heads struggle to learn on their own.
    vowels = _derive_vowels(raw_recs, langs, cfg.vowel_phonemes)

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
    save_bpe(tokenizer, os.path.join(binary_dir, "bpe.txt"))

    meta = {
        "langs": sorted(set(langs)),
        "num_langs": len(lang2id),
        "lang2id": lang2id,
        "id2lang": {i: l for l, i in lang2id.items()},
        "src_vocab_size": len(src_vocab),
        "phoneme_vocab_size": len(phoneme_vocab),
        "count_vocab_size": count_codec.vocab_size,
        "max_count": count_codec.max_count,
        "binarize_version": BINARIZE_VERSION,
        # model hyper-params needed to rebuild the exact architecture at infer/export
        "embed_dim": cfg.embed_dim,
        "enc_layers": cfg.enc_layers,
        "dec_layers": cfg.dec_layers,
        "enc_heads": cfg.enc_heads,
        "dec_hidden": cfg.dec_hidden,
        "ffn_dim": cfg.ffn_dim,
        "lang_embed_dim": cfg.lang_embed_dim,
        # length budgets -- inference reads these so its generation length matches
        # what training used for the val monitor (avoids truncating long inputs)
        "max_src_len": cfg.max_src_len,
        "max_tgt_len": cfg.max_tgt_len,
        # structural constraints used to fix phoneme-group separators
        "syllable_is_char": {lang: bool(lang_syllable_char.get(lang, False))
                              for lang in set(langs)},
        "vowel_phonemes": {lang: sorted(vowels[lang][0]) for lang in vowels},
        "vowel_confidence": {lang: vowels[lang][1] for lang in vowels},
        "vowel_constraint_violation": {lang: vowels[lang][2] for lang in vowels},
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
    if binary_dir is None:
        raise ValueError("load_binary requires an explicit binary_dir "
                         "(set binary_dir in config or pass --binary_dir)")
    src_vocab = Vocab.from_file(os.path.join(binary_dir, "src_vocab.txt"))
    phoneme_vocab = Vocab.from_file(os.path.join(binary_dir, "phoneme_vocab.txt"))
    tokenizer = load_bpe(os.path.join(binary_dir, "bpe.txt"))
    with open(os.path.join(binary_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    count_codec = CountCodec(meta["max_count"])
    # all tasks pad at index 0 (phoneme vocab and the count codec share PAD/SOS/EOS)
    pad_idx_dict = {n: 0 for n in TARGET_NAMES}
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
        count_codec,
        tokenizer,
        pad_idx_dict,
        meta,
        val_ds.monitor,
    )


# --------------------------------------------------------------------------- #
# Standalone vocab/codec loaders (used by inference + export_onnx)
# --------------------------------------------------------------------------- #
def load_src_vocab(path: str) -> Vocab:
    return Vocab.from_file(path)


def load_phoneme_vocab(path: str) -> Vocab:
    return Vocab.from_file(path)


def load_count_codec(meta: dict) -> CountCodec:
    return CountCodec(meta["max_count"])


# --------------------------------------------------------------------------- #
# Auto-binarize orchestration (called from train.py)
# --------------------------------------------------------------------------- #
def ensure_binary(cfg) -> str:
    """Return the binary dir, (re)building it only when missing or stale."""
    binary_dir = cfg.binary_dir or os.path.join(cfg.data_dir, "binary")
    files = discover_files(cfg.data_dir, cfg.lang_define, cfg.file_glob)
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
