"""Shared helpers: lang parsing from file names, vocab IO, greedy decoding."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

# Language is encoded in the dataset file name, e.g. "dataset-ko.csv" -> "ko".
# We take the token after the LAST '-' and before the extension.
_LANG_RE = re.compile(r"dataset-([A-Za-z0-9_-]+)\.csv$", re.IGNORECASE)


def parse_lang_from_filename(path: str) -> Optional[str]:
    """Parse a language label from a dataset file name.

    ``dataset-ko.csv`` -> ``"ko"``; ``dataset-en-US.csv`` -> ``"en-US"``.
    Returns ``None`` when the pattern does not match.
    """
    name = os.path.basename(path)
    m = _LANG_RE.search(name)
    if m:
        return m.group(1)
    # fallback: any "<something>-<lang>.csv"
    m = re.search(r"-([A-Za-z0-9_-]+)\.csv$", name, re.IGNORECASE)
    return m.group(1) if m else None


def greedy_decode(
    logits: "torch.Tensor",  # [B, T, V]
    eos_idx: int,
    pad_idx: int,
) -> List[List[int]]:
    """Greedy-argmax decode, stopping each sequence at the first EOS.

    Returns a list of token-index lists (EOS excluded).
    """
    import torch

    preds = logits.argmax(dim=-1).tolist()  # [B, T]
    out: List[List[int]] = []
    for seq in preds:
        toks: List[int] = []
        for t in seq:
            if t == eos_idx:
                break
            if t == pad_idx:
                continue
            toks.append(t)
        out.append(toks)
    return out


def make_lang_index(langs: List[str]) -> Tuple[dict, dict]:
    """Map language labels to contiguous ids (and back)."""
    lang2id = {l: i for i, l in enumerate(sorted(set(langs)))}
    id2lang = {i: l for l, i in lang2id.items()}
    return lang2id, id2lang


def load_model_weights(checkpoint: str, model, device: str = "cpu"):
    """Load state dict from any of our checkpoint formats into ``model``.

    Supports both the full training checkpoint (``{"model": ..., "optimizer": ...}``)
    and the deployment-only file written by ``train._save_model_only``
    (``{"model": sd, "dtype": "fp32"|"fp16"}``).  When the file was exported as
    fp16 the model is cast to half precision so ``load_state_dict`` matches dtypes.
    """
    import torch

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
        if ckpt.get("dtype") == "fp16":
            model.half()
    else:
        sd = ckpt  # raw state_dict
    model.load_state_dict(sd)
    return model
