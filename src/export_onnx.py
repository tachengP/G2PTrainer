"""Export a single ONNX model with exactly four output nodes.

The exported graph takes

    * ``graphemes`` : int64  [B, S]   source (sub-word / per-char) ids
    * ``langs``     : int64  [B]      language ids
    * ``src_lens``  : int64  [B]      valid length per source sequence

and returns four logit tensors (greedy decoding already applied internally,
but logits are returned so the consumer can re-decide):

    * ``phonemes``          : float32 [B, T, V_ph]
    * ``separated_graphmes``: float32 [B, T, V_gr]
    * ``separated_phonemes`` : float32 [B, T, V_ph]
    * ``aligned_phonemes``   : float32 [B, T, V_ph]

``T`` is fixed to ``--max_len`` at export time (the generate path runs a fixed
number of auto-regressive steps, which is what makes the graph traceable).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.model import G2PModel


class ONNXWrapper(torch.nn.Module):
    def __init__(self, model: G2PModel, max_len: int, sos_idx: int):
        super().__init__()
        self.model = model
        self.max_len = max_len
        self.sos_idx = sos_idx

    def forward(self, graphemes: torch.Tensor, src_lens: torch.Tensor, langs: torch.Tensor):
        # model.generate expects [S, B]; inputs here are [B, S] -> transpose
        src = graphemes.transpose(0, 1).contiguous()
        out = self.model.generate(src, src_lens, langs, self.max_len, self.sos_idx)
        # outputs are [T, B, V] -> transpose to [B, T, V]
        return (
            out["phonemes"].transpose(0, 1).contiguous(),
            out["separated_graphmes"].transpose(0, 1).contiguous(),
            out["separated_phonemes"].transpose(0, 1).contiguous(),
            out["aligned_phonemes"].transpose(0, 1).contiguous(),
        )


def _write_metadata(onnx_path: str, meta: dict, max_len: int, eos_idx: int = 2) -> None:
    """Embed the vocabulary-shape contract into the ONNX file.

    Different languages / phoneme schemes produce ONNX graphs whose only
    data-dependent dimension is ``V`` (the last dim of every output).  We bake
    the expected vocabulary sizes into ``metadata_props`` so an inferer can
    verify that the ONNX and its accompanying vocab files belong together and
    refuse to silently decode garbage if they do not.

    The four output names are ALWAYS fixed (see ``OUTPUT_NAMES``), which is what
    lets a single :class:`src.inference.ONNXInferer` drive any language's graph.
    """
    import onnx

    m = onnx.load(onnx_path)
    props = {
        "g2p.src_vocab_size": str(meta["src_vocab_size"]),
        "g2p.phoneme_vocab_size": str(meta["phoneme_vocab_size"]),
        "g2p.grapheme_tgt_vocab_size": str(meta["grapheme_tgt_vocab_size"]),
        "g2p.num_langs": str(meta["num_langs"]),
        "g2p.max_len": str(max_len),
        "g2p.eos_idx": str(eos_idx),
        "g2p.sos_idx": "1",
        "g2p.pad_idx": "0",
        "g2p.outputs": ",".join(OUTPUT_NAMES),
    }
    # onnx.metadata_props is a repeated field; clear then set
    m.metadata_props.clear()
    for k, v in props.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, onnx_path)


OUTPUT_NAMES = ["phonemes", "separated_graphmes", "separated_phonemes", "aligned_phonemes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="g2p_multitask.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--max_len", type=int, default=80, help="fixed auto-regressive steps baked into the graph")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from src.utils import load_model_weights
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "config" in ckpt and "meta" in ckpt:
        cfg_d, meta = ckpt["config"], ckpt["meta"]
    else:
        # weights-only deployment file: rebuild from meta.json next to it
        import json, os
        meta_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        cfg_d = meta  # meta.json now carries the architecture fields too

    model = G2PModel(
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
    load_model_weights(args.checkpoint, model, "cpu")
    model.eval()

    # SOS index is shared across vocabularies (always 1)
    sos_idx = 1
    wrapper = ONNXWrapper(model, args.max_len, sos_idx).to(args.device)
    wrapper.eval()

    # dummy inputs: batch=1, src_len=10 (single-sample dummy avoids the ONNX
    # LSTM variable-length/batch!=1 caveat; batch dim stays dynamic via axes)
    B, S = 1, 10
    dummy_src = torch.zeros(B, S, dtype=torch.long, device=args.device)
    dummy_lens = torch.full((B,), S, dtype=torch.long, device=args.device)
    dummy_lang = torch.zeros(B, dtype=torch.long, device=args.device)

    dynamic_axes = {
        "graphemes": {0: "B", 1: "S"},
        "src_lens": {0: "B"},
        "langs": {0: "B"},
        "phonemes": {0: "B"},
        "separated_graphmes": {0: "B"},
        "separated_phonemes": {0: "B"},
        "aligned_phonemes": {0: "B"},
    }

    torch.onnx.export(
        wrapper,
        (dummy_src, dummy_lens, dummy_lang),
        args.output,
        input_names=["graphemes", "src_lens", "langs"],
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # Embed the vocab-shape contract so any inferer (any language) can verify
    # it paired this graph with the matching vocab files.
    _write_metadata(args.output, meta, args.max_len)

    print(f"[done] exported -> {args.output}")
    print("       inputs : graphemes[int64 B,S], src_lens[int64 B], langs[int64 B]")
    print(f"       outputs: {', '.join(OUTPUT_NAMES)} [float32 B,{args.max_len},V]")
    print(f"       V_ph={meta['phoneme_vocab_size']} V_gr={meta['grapheme_tgt_vocab_size']} "
          f"V_src={meta['src_vocab_size']} langs={meta['num_langs']}")


if __name__ == "__main__":
    main()
