"""Inference helpers: reconstruct the four G2P fields from a trained model.

The model emits token *ids* per task.  ``phonemes`` ids are decoded by the
phoneme vocab; the three derived tasks emit *count* ids which are turned into
the separator-joined strings via :func:`src.preprocessing.reconstruct_groups`,
regrouping the appropriate base sequence (source units for ``separated_graphmes``,
predicted phoneme tokens for ``separated_/aligned_phonemes``).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model import G2PModel
from src.preprocessing import (
    PIPE,
    SLASH,
    load_bpe,
    reconstruct_groups,
    resegment_by_vowels,
)
from src.utils import load_model_weights

SEP_UNIT = {
    "separated_graphmes": ("|", ""),
    "separated_phonemes": ("|", " "),
    "aligned_phonemes": ("/", " "),
}


def _resolve_binary_dir(binary_dir: str) -> str:
    """Pick a binary dir that actually contains ``meta.json``.

    The command-line default (``data/binary``) is a placeholder; the real output
    lives under ``<data_dir>/binary`` (e.g. ``data/Korean/binary``).  If the given
    path has no ``meta.json``, fall back to a few common locations so a bare
    ``python -m src.inference --text ...`` works without guessing the path.
    """
    if os.path.exists(os.path.join(binary_dir, "meta.json")):
        return binary_dir
    candidates = [
        binary_dir,
        os.path.join("data", "Korean", "binary"),
        os.path.join("data", "binary", "binary"),
        "data/binary",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "meta.json")):
            print(f"[inference] '{binary_dir}' has no meta.json; using '{c}'")
            return c
    return binary_dir  # give up; let the original error surface


def _resolve_model_path(binary_dir: str, model_dir: str = None) -> str:
    """Pick the checkpoint file holding the trained weights.

    Training now writes the full checkpoint to ``{output_dir}/{model_name}/ckpt``
    (config-driven).  When ``model_dir`` is given, prefer ``best_model.pt`` there,
    falling back to the deployment-only ``model_best.pt``; otherwise fall back to
    the legacy ``{binary_dir}/best_model.pt`` for backwards compatibility.
    """
    if model_dir:
        best = os.path.join(model_dir, "best_model.pt")
        if os.path.exists(best):
            return best
        alt = os.path.join(model_dir, "model_best.pt")
        if os.path.exists(alt):
            return alt
    return os.path.join(binary_dir, "best_model.pt")


def load_inferer(binary_dir: str, model_dir: str = None, device: str = "cpu"):
    binary_dir = _resolve_binary_dir(binary_dir)
    with open(os.path.join(binary_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = G2PModel(
        src_vocab_size=meta["src_vocab_size"],
        phoneme_vocab_size=meta["phoneme_vocab_size"],
        count_vocab_size=meta["count_vocab_size"],
        num_langs=meta["num_langs"],
        embed_dim=meta["embed_dim"],
        enc_layers=meta["enc_layers"],
        dec_layers=meta["dec_layers"],
        enc_heads=meta["enc_heads"],
        dec_hidden=meta["dec_hidden"],
        ffn_dim=meta["ffn_dim"],
        dropout=0.0,
        lang_embed_dim=meta["lang_embed_dim"],
    )
    load_model_weights(_resolve_model_path(binary_dir, model_dir), model, device)
    model.to(device).eval()
    return Inferer(model, binary_dir, meta, device)


class Inferer:
    def __init__(self, model: G2PModel, binary_dir: str, meta: dict, device: str):
        self.model = model
        self.meta = meta
        self.device = device
        self.lang2id = meta["lang2id"]
        # load src vocab + bpe + phoneme vocab + count codec
        from src.bin_data import (  # noqa: F401
            load_src_vocab, load_phoneme_vocab, load_count_codec,
        )
        self.src_vocab = load_src_vocab(os.path.join(binary_dir, "src_vocab.txt"))
        self.phoneme_vocab = load_phoneme_vocab(os.path.join(binary_dir, "phoneme_vocab.txt"))
        self.count_codec = load_count_codec(meta)
        self.tokenizer = load_bpe(os.path.join(binary_dir, "bpe.txt")) \
            if os.path.exists(os.path.join(binary_dir, "bpe.txt")) else None
        if self.tokenizer is None:
            from src.preprocessing import build_source_vocab
            self.tokenizer, _ = build_source_vocab(self.src_vocab.symbols())
        # structural constraint metadata (one vowel nucleus per syllable)
        self.vowel_symbols = meta.get("vowel_phonemes") or {}
        self.syllable_is_char = meta.get("syllable_is_char") or {}

    def predict(self, grapheme_text: str, lang: str,
                tasks=None, max_len: int = None):
        if tasks is None:
            tasks = ["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"]
        # Match the length budget used during training (cfg.max_tgt_len) so long
        # inputs are not truncated earlier in inference than they were at
        # validation time -- otherwise inference would drop trailing separators
        # that the training val monitor still produced.
        if max_len is None:
            max_len = self.meta.get("max_tgt_len", 80)
        lang_id = self.lang2id.get(lang, 0)
        units = self.tokenizer.tokenize(grapheme_text)
        if not units:
            return {t: "" for t in tasks}
        src_ids = self.src_vocab.encode(units)
        src_t = torch.tensor([src_ids], dtype=torch.long, device=self.device).transpose(0, 1)
        src_len_t = torch.tensor([len(src_ids)], dtype=torch.long, device=self.device)
        lang_t = torch.tensor([lang_id], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(src_t, src_len_t, lang_t, max_len, 1)
        result = {}
        # phonemes first (other tasks regroup it)
        ph_ids = out["phonemes"][0].tolist()
        ph_ids = [x for x in ph_ids if x not in (0, 1, 2)]
        ph_tokens = self.phoneme_vocab.decode(ph_ids)
        result["phonemes"] = " ".join(ph_tokens)
        for task in ("separated_graphmes", "separated_phonemes", "aligned_phonemes"):
            if task not in tasks:
                continue
            sep, unit = SEP_UNIT[task]
            # For one-char-one-syllable languages, re-segment the (correct) phoneme
            # sequence by vowel nucleus instead of trusting the count head -- this
            # guarantees each `|`/`/` group has exactly one vowel, fixing the
            # separator errors the count heads under-fit.
            if (task in ("separated_phonemes", "aligned_phonemes")
                    and self.syllable_is_char.get(lang)
                    and lang in self.vowel_symbols and self.vowel_symbols[lang]):
                result[task] = resegment_by_vowels(
                    ph_tokens, set(self.vowel_symbols[lang]), sep, unit)
            else:
                counts = self.count_codec.decode([int(i) for i in out[task][0].tolist()])
                base = units if task == "separated_graphmes" else ph_tokens
                result[task] = reconstruct_groups(base, counts, sep, unit)
        return result


# --------------------------------------------------------------------------- #
# ONNX-backed inferer (same contract, runs the exported graph)
# --------------------------------------------------------------------------- #
class ONNXInferer:
    def __init__(self, onnx_path: str, binary_dir: str):
        import onnxruntime as ort
        import numpy as np
        self.np = np
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with open(os.path.join(binary_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        self.lang2id = meta["lang2id"]
        from src.bin_data import load_src_vocab, load_phoneme_vocab, load_count_codec
        self.src_vocab = load_src_vocab(os.path.join(binary_dir, "src_vocab.txt"))
        self.phoneme_vocab = load_phoneme_vocab(os.path.join(binary_dir, "phoneme_vocab.txt"))
        self.count_codec = load_count_codec(meta)
        self.tokenizer = load_bpe(os.path.join(binary_dir, "bpe.txt")) \
            if os.path.exists(os.path.join(binary_dir, "bpe.txt")) else None
        if self.tokenizer is None:
            from src.preprocessing import build_source_vocab
            self.tokenizer, _ = build_source_vocab(self.src_vocab.symbols())

    def predict(self, grapheme_text: str, lang: str, max_len: int = 80):
        lang_id = self.lang2id.get(lang, 0)
        units = self.tokenizer.tokenize(grapheme_text)
        if not units:
            return {t: "" for t in ["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"]}
        src_ids = self.src_vocab.encode(units)
        S = len(src_ids)
        src_t = self.np.array([src_ids], dtype="int64")
        src_len_t = self.np.array([S], dtype="int64")
        lang_t = self.np.array([lang_id], dtype="int64")
        outs = self.session.run(None, {
            "graphemes": src_t, "src_lens": src_len_t, "langs": lang_t,
        })
        ph, sgr, sph, alp = outs
        result = {}
        ph_ids = [int(x) for x in ph[0] if x not in (0, 1, 2)]
        ph_tokens = self.phoneme_vocab.decode(ph_ids)
        result["phonemes"] = " ".join(ph_tokens)
        for task, raw in (("separated_graphmes", sgr), ("separated_phonemes", sph),
                          ("aligned_phonemes", alp)):
            counts = self.count_codec.decode([int(x) for x in raw[0]])
            base = units if task == "separated_graphmes" else ph_tokens
            sep, unit = SEP_UNIT[task]
            result[task] = reconstruct_groups(base, counts, sep, unit)
        return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dir", default="data/Korean/binary")
    ap.add_argument("--model_dir", default="checkpoints/Korean/ckpt",
                    help="dir holding best_model.pt (training run output)")
    ap.add_argument("--onnx", default=None, help="use ONNX inferer at this path")
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.onnx:
        inf = ONNXInferer(a.onnx, a.binary_dir)
    else:
        inf = load_inferer(a.binary_dir, a.model_dir, a.device)
    res = inf.predict(a.text, a.lang)
    for k, v in res.items():
        print(f"{k}: {v}")
