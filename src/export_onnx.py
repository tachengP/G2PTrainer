"""Export a trained G2PModel to ONNX (single forward, greedy decode loop)."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.model import G2PModel
from src.preprocessing import TARGET_NAMES
from src.utils import load_model_weights


class GreedyWrapper(torch.nn.Module):
    """Wrap the model so the ONNX graph includes the greedy decode loop.

    Inputs: graphemes [B, S], src_lens [B], langs [B].
    Outputs: phonemes [B, T], separated_graphmes [B, T], separated_phonemes [B, T],
             aligned_phonemes [B, T]  (count ids; decode + regroup at call site).
    """

    def __init__(self, model: G2PModel, max_len: int, sos_idx: int):
        super().__init__()
        self.model = model
        self.max_len = max_len
        self.sos_idx = sos_idx

    def forward(self, graphemes, src_lens, langs):
        # generate() transposes batch-first [B, S] internally, so pass as-is.
        return self.model.generate(graphemes, src_lens, langs,
                                   self.max_len, self.sos_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary_dir", default="data/Korean/binary")
    ap.add_argument("--model_dir", default="checkpoints/Korean/ckpt",
                    help="dir holding best_model.pt (training run output)")
    ap.add_argument("--out", default="data/Korean/binary/g2p.onnx")
    ap.add_argument("--max_len", type=int, default=80)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    with open(os.path.join(a.binary_dir, "meta.json"), "r", encoding="utf-8") as f:
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
    best = os.path.join(a.model_dir, "best_model.pt")
    if not os.path.exists(best):
        best = os.path.join(a.model_dir, "model_best.pt")
    if not os.path.exists(best):
        best = os.path.join(a.binary_dir, "best_model.pt")
    load_model_weights(best, model, "cpu")
    model.eval()

    wrapper = GreedyWrapper(model, a.max_len, 1)
    wrapper.eval()

    B = 1
    S = 8
    dummy_g = torch.zeros(B, S, dtype=torch.long)
    dummy_len = torch.ones(B, dtype=torch.long)
    dummy_lang = torch.zeros(B, dtype=torch.long)

    # The greedy decode loop uses python-level control flow + a dynamic batch
    # size, which some torch versions cannot capture via torch.export
    # (``torch.zeros`` with a symbolic batch raises). The PyTorch inference path
    # (src.inference.load_inferer) always works; ONNX is a best-effort that
    # succeeds on recent torch builds that support dynamic_shapes export.
    dynamic_shapes = {
        "graphemes": {0: torch.export.Dim("B"), 1: torch.export.Dim("S")},
        "src_lens": {0: torch.export.Dim("B")},
        "langs": {0: torch.export.Dim("B")},
    }
    try:
        torch.onnx.export(
            wrapper,
            (dummy_g, dummy_len, dummy_lang),
            a.out,
            input_names=["graphemes", "src_lens", "langs"],
            output_names=["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"],
            dynamic_axes={
                "graphemes": {0: "B", 1: "S"},
                "src_lens": {0: "B"},
                "langs": {0: "B"},
                "phonemes": {0: "B", 1: "T"},
                "separated_graphmes": {0: "B", 1: "T"},
                "separated_phonemes": {0: "B", 1: "T"},
                "aligned_phonemes": {0: "B", 1: "T"},
            },
            opset_version=a.opset,
            do_constant_folding=True,
            dynamo=True,
            dynamic_shapes=dynamic_shapes,
        )
        print(f"[onnx] exported -> {a.out}")
    except Exception as e:  # pragma: no cover - backend dependent
        print(f"[onnx] export failed: {type(e).__name__}: {e}")
        print("[onnx] The greedy-decoding loop is not capturable by this torch "
              "build. Use the PyTorch inference path instead:")
        print("        python -m src.inference --binary_dir <dir> --text <graphmes>")


if __name__ == "__main__":
    main()
