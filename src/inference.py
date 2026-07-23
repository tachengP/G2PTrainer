"""Inference CLI -- works with a PyTorch checkpoint or an exported ONNX model.

Examples
--------
PyTorch:
    python src/inference.py --checkpoint checkpoints/ckpt_best.pt --lang ko --text "안녕하세요"

ONNX:
    python src/inference.py --onnx g2p_multitask.onnx --lang ko --text "안녕하세요"
    python src/inference.py --onnx g2p_multitask.onnx --lang en --input-file phrases.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import preprocessing as pp
from src.utils import greedy_decode


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
def load_artifacts(vocab_dir: str):
    tokenizer = pp.load_bpe(os.path.join(vocab_dir, "bpe.txt"))
    src_vocab = pp.Vocab.from_file(os.path.join(vocab_dir, "src_vocab.txt"))
    phoneme_vocab = pp.Vocab.from_file(os.path.join(vocab_dir, "phoneme_vocab.txt"))
    grapheme_tgt_vocab = pp.Vocab.from_file(os.path.join(vocab_dir, "grapheme_tgt_vocab.txt"))
    import json
    with open(os.path.join(vocab_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    lang2id = meta["lang2id"]
    return tokenizer, src_vocab, phoneme_vocab, grapheme_tgt_vocab, lang2id, meta


def _prepare_batch(texts, lang, tokenizer, src_vocab, lang2id, device):
    if lang not in lang2id:
        raise ValueError(f"unknown language {lang!r}; known: {sorted(lang2id)}")
    lang_id = lang2id[lang]
    units_list = [tokenizer.tokenize(t) for t in texts]
    ids_list = [src_vocab.encode(u) for u in units_list]
    max_len = max((len(i) for i in ids_list), default=1)
    max_len = max(max_len, 1)
    padded = torch.full((len(ids_list), max_len), src_vocab.pad_idx, dtype=torch.long)
    src_lens = torch.zeros(len(ids_list), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        if len(ids) == 0:
            ids = [src_vocab.unk_idx]
        padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        src_lens[i] = len(ids)
    lang_ids = torch.full((len(texts),), lang_id, dtype=torch.long)
    return padded.to(device), src_lens.to(device), lang_ids.to(device)


def _decode_outputs(preds: Dict[str, List[List[int]]], vocab_map) -> List[Dict[str, str]]:
    """preds: task -> list of token-id lists (per sample). Build readable strings."""
    out: List[Dict[str, str]] = []
    n = len(next(iter(preds.values())))
    for i in range(n):
        d = {}
        for task, vocab in vocab_map.items():
            toks = preds[task][i]
            syms = [
                vocab.itos(t)
                for t in toks
                if t not in (vocab.pad_idx, vocab.eos_idx, vocab.sos_idx)
            ]
            # bare phonemes has no in-string separator -> re-insert spaces to
            # match the CSV; separated_*/aligned_* already carry | and /.
            sep = " " if task == "phonemes" else ""
            d[task] = sep.join(syms)
        out.append(d)
    return out


# --------------------------------------------------------------------------- #
# PyTorch inferer
# --------------------------------------------------------------------------- #
class PyTorchInferer:
    def __init__(self, checkpoint: str, vocab_dir: str, device: str = "cpu"):
        from src.utils import load_model_weights
        self.tokenizer, self.src_vocab, self.phoneme_vocab, self.grapheme_tgt_vocab, \
            self.lang2id, self.meta = load_artifacts(vocab_dir)
        ckpt = torch.load(checkpoint, map_location=device)
        if "config" in ckpt and "meta" in ckpt:
            # full training checkpoint: build from its embedded config/meta
            self.model = _build_model(ckpt["config"], ckpt["meta"]).to(device)
            load_model_weights(checkpoint, self.model, device)
        else:
            # weights-only deployment file (model_best/model_last.pt):
            # rebuild from the meta.json in vocab_dir, then load weights
            self.model = _build_model(self.meta, self.meta).to(device)
            load_model_weights(checkpoint, self.model, device)
        self.model.eval()
        self.device = device
        self.vocab_map = {
            "phonemes": self.phoneme_vocab,
            "separated_graphmes": self.grapheme_tgt_vocab,
            "separated_phonemes": self.phoneme_vocab,
            "aligned_phonemes": self.phoneme_vocab,
        }

    @torch.no_grad()
    def run(self, texts: List[str], lang: str, max_len: int = 80) -> List[Dict[str, str]]:
        src, src_lens, lang_ids = _prepare_batch(
            texts, lang, self.tokenizer, self.src_vocab, self.lang2id, self.device
        )
        src_t = src.transpose(0, 1).contiguous()
        logits = self.model.generate(src_t, src_lens, lang_ids, max_len, sos_idx=1)
        # model.generate returns [T, B, V]; greedy_decode expects [B, T, V]
        logits = {k: v.transpose(0, 1).contiguous() for k, v in logits.items()}
        preds: Dict[str, List[List[int]]] = {}
        for task in logits:
            decoded = greedy_decode(logits[task], eos_idx=2, pad_idx=0)  # 2=EOS
            preds[task] = decoded
        return _decode_outputs(preds, self.vocab_map)


# --------------------------------------------------------------------------- #
# ONNX inferer
# --------------------------------------------------------------------------- #
class ONNXInferer:
    """Drives any exported ``G2PModel`` graph (any language / phoneme scheme).

    The four output names are fixed and identical across all exports
    (``phonemes``, ``separated_graphmes``, ``separated_phonemes``,
    ``aligned_phonemes``), so this single class works for every ONNX produced by
    :mod:`src.export_onnx`.  The only data-dependent dimension is ``V`` (the last
    dim of each output), which is baked into the graph weights -- therefore each
    ONNX must be paired with its own vocab directory.  We verify that contract via
    the ``metadata_props`` embedded at export time and raise if it does not match.
    """

    OUTPUT_NAMES = ["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"]

    def __init__(self, onnx_path: str, vocab_dir: str, device: str = "cpu"):
        import onnxruntime as ort
        self.tokenizer, self.src_vocab, self.phoneme_vocab, self.grapheme_tgt_vocab, \
            self.lang2id, self.meta = load_artifacts(vocab_dir)
        self.sess = ort.InferenceSession(
            onnx_path,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            ),
        )
        self.device = device
        self.vocab_map = {
            "phonemes": self.phoneme_vocab,
            "separated_graphmes": self.grapheme_tgt_vocab,
            "separated_phonemes": self.phoneme_vocab,
            "aligned_phonemes": self.phoneme_vocab,
        }
        self.meta_props = self._read_metadata(onnx_path)
        self._verify_compat()

    # ---- metadata / compatibility check ---------------------------------- #
    def _read_metadata(self, onnx_path: str) -> Dict[str, str]:
        import onnx

        try:
            m = onnx.load(onnx_path)
        except Exception:
            return {}
        return {p.key: p.value for p in m.metadata_props}

    def _verify_compat(self) -> None:
        mp = self.meta_props
        if not mp:
            # legacy export without metadata: trust the vocab dir silently.
            return
        expected = {
            "g2p.src_vocab_size": len(self.src_vocab),
            "g2p.phoneme_vocab_size": len(self.phoneme_vocab),
            "g2p.grapheme_tgt_vocab_size": len(self.grapheme_tgt_vocab),
            "g2p.num_langs": len(self.lang2id),
        }
        mism = [
            f"{k}={mp[k]} but vocab has {expected[k]}"
            for k, v in expected.items()
            if v != int(mp[k])
        ]
        if mism:
            raise ValueError(
                "ONNX vocab contract mismatch -- this .onnx is paired with the "
                "wrong vocab_dir. " + "; ".join(mism)
            )
        # expose the graph's own max_len for the caller
        self.max_len_graph = int(mp.get("g2p.max_len", 80))
        self.eos_idx_graph = int(mp.get("g2p.eos_idx", 2))

    def run(self, texts: List[str], lang: str, max_len: int = 80) -> List[Dict[str, str]]:
        max_len = min(max_len, getattr(self, "max_len_graph", max_len))
        src, src_lens, lang_ids = _prepare_batch(
            texts, lang, self.tokenizer, self.src_vocab, self.lang2id, "cpu"
        )
        feeds = {
            "graphemes": src.numpy().astype(np.int64),
            "src_lens": src_lens.numpy().astype(np.int64),
            "langs": lang_ids.numpy().astype(np.int64),
        }
        outs = self.sess.run(self.OUTPUT_NAMES, feeds)
        preds: Dict[str, List[List[int]]] = {}
        eos_idx = getattr(self, "eos_idx_graph", 2)
        for name, arr in zip(self.OUTPUT_NAMES, outs):
            arr = np.argmax(arr, axis=-1)  # [B, T]
            preds[name] = [list(row) for row in arr]
        return _decode_outputs(preds, self.vocab_map)


# --------------------------------------------------------------------------- #
# model factory
# --------------------------------------------------------------------------- #
def _build_model(cfg_d, meta):
    from src.model import G2PModel
    return G2PModel(
        src_vocab_size=meta["src_vocab_size"],
        phoneme_vocab_size=meta["phoneme_vocab_size"],
        grapheme_tgt_vocab_size=meta["grapheme_tgt_vocab_size"],
        num_langs=meta["num_langs"],
        embed_dim=cfg_d["embed_dim"],
        enc_layers=cfg_d["enc_layers"],
        dec_layers=cfg_d["dec_layers"],
        enc_heads=cfg_d["enc_heads"],
        dec_hidden=cfg_d["dec_hidden"],
        ffn_dim=cfg_d["ffn_dim"],
        dropout=0.0,
        lang_embed_dim=cfg_d["lang_embed_dim"],
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    # Ensure UTF-8 stdout so non-ASCII text (e.g. Korean) prints correctly on
    # Windows consoles whose default encoding is GBK.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--vocab_dir", default=None, help="dir with exported vocabs; defaults to checkpoint dir")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--input-file", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    assert args.checkpoint or args.onnx, "provide --checkpoint or --onnx"

    if args.onnx:
        vocab_dir = args.vocab_dir or os.path.dirname(os.path.abspath(args.onnx))
        inferer = ONNXInferer(args.onnx, vocab_dir)
    else:
        vocab_dir = args.vocab_dir or os.path.dirname(os.path.abspath(args.checkpoint))
        inferer = PyTorchInferer(args.checkpoint, vocab_dir, device=args.device)

    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        texts = [args.text] if args.text else []

    results = inferer.run(texts, args.lang, max_len=args.max_len)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for text, r in zip(texts, results):
            print(f"input ({args.lang}): {text}")
            print(f"  phonemes          : {r['phonemes']}")
            print(f"  separated_graphmes: {r['separated_graphmes']}")
            print(f"  separated_phonemes: {r['separated_phonemes']}")
            print(f"  aligned_phonemes  : {r['aligned_phonemes']}")


if __name__ == "__main__":
    main()
