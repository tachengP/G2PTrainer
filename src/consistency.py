"""Content-consistency losses for the multi-task G2P model.

The CSV defines four targets. Space, ``|`` and ``/`` are all just separators of
varying granularity:

    separated_graphmes  = graphmes  with ``|`` inserted between grapheme groups
    separated_phonemes  = phonemes  with ``|`` inserted between phoneme groups
    aligned_phonemes    = phonemes  with ``/`` inserted between note groups

Therefore the separator-stripped variants must reproduce the base sequences:

    separated_graphmes  (drop ``|``) == graphmes
    separated_phonemes  (drop ``|``) == phonemes
    aligned_phonemes    (drop ``/``) == phonemes

These invariants are turned into soft, differentiable training losses:

* Phoneme consistency (cross-decoder): the ``separated_phonemes`` /
  ``aligned_phonemes`` decoders, at their non-separator frames, should agree with
  the plain ``phonemes`` decoder distribution. All three share the phoneme
  vocabulary, so we KL-diverge their frame-wise softmaxes directly.

  Because the separator tokens ``|`` / ``/`` shift the sequence, the two decoders
  are NOT position-aligned; instead the *k*-th content frame of the separator
  decoder corresponds to the *k*-th (content) frame of the base decoder. We
  therefore gather the content frames in order and compare them 1:1.

* Grapheme consistency: the ``separated_graphmes`` decoder, at its non-separator
  frames, should reproduce the original grapheme sequence (i.e. the
  ``separated_graphmes`` content, which with ``|`` removed equals ``graphmes``).
  We KL-diverge the decoder softmax against a one-hot over those content tokens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# A decoder at logits frame ``i`` predicts ``target[i + 1]`` (teacher-forcing
# shift). Valid prediction frames are therefore ``0 .. T-2``, which correspond to
# ``target[1 .. T-1]``.
_PRED_FRAMES = slice(0, -1)


def phoneme_consistency(logits_base, logits_sep, tgt_sep, sep_id, pad_id, eos_id):
    """KL( p_base || p_sep ) over the content frames of the separator decoder.

    Args:
        logits_base: seq-first ``[T_base, B, V]`` logits of the ``phonemes`` decoder.
        logits_sep:  seq-first ``[T_sep, B, V]`` logits of a separator decoder
                     (``separated_phonemes`` or ``aligned_phonemes``). Shares ``V``.
        tgt_sep:     batch-first ``[B, T_sep]`` ground-truth of the separator task.
        sep_id:      vocab id of the separator token (``|`` or ``/``) in ``V``.
        pad_id:      vocab id of the padding token in ``V``.
        eos_id:      vocab id of the stop token; excluded from content frames so the
                     trailing ``<eos>`` is not treated as a phoneme to align.
    """
    B = tgt_sep.size(0)
    losses = []
    for b in range(B):
        pred = tgt_sep[b, 1:]                       # [T_sep-1] symbols predicted by frames 0..T_sep-2
        content = (pred != sep_id) & (pred != pad_id) & (pred != eos_id)
        idx = content.nonzero(as_tuple=False).squeeze(-1)   # [K] logits-frame indices of content
        K = idx.numel()
        if K == 0:
            continue
        ls = logits_sep[_PRED_FRAMES, b][idx]       # [K, V] separator decoder content frames (in order)
        lb = logits_base[:K, b]                     # [K, V] base decoder frames 0..K-1 (in order)
        p_base = F.softmax(lb, dim=-1)
        p_sep = F.log_softmax(ls, dim=-1)
        # KL(p_base || p_sep): pulls the separator decoder towards the base decoder so
        # the two agree on the underlying phoneme content.
        losses.append(F.kl_div(p_sep, p_base, reduction="batchmean"))
    if not losses:
        return torch.zeros((), device=logits_base.device, dtype=logits_base.dtype)
    return torch.stack(losses).mean()


def grapheme_consistency(logits_sep, tgt_sep, sep_id, pad_id, num_classes, eos_id):
    """KL( one_hot(content) || p_sep ) over the content frames of ``separated_graphmes``.

    Args:
        logits_sep:  seq-first ``[T_sep, B, V_gr]`` logits of ``separated_graphmes``.
        tgt_sep:     batch-first ``[B, T_sep]`` ground-truth of ``separated_graphmes``.
        sep_id:      vocab id of ``|`` in the grapheme-target vocab.
        pad_id:      vocab id of the padding token.
        num_classes: size of the grapheme-target vocab (``V_gr``).
        eos_id:      vocab id of the stop token; excluded from content frames.
    """
    B = tgt_sep.size(0)
    losses = []
    for b in range(B):
        pred = tgt_sep[b, 1:]
        content = (pred != sep_id) & (pred != pad_id) & (pred != eos_id)
        idx = content.nonzero(as_tuple=False).squeeze(-1)   # [K]
        K = idx.numel()
        if K == 0:
            continue
        ls = logits_sep[_PRED_FRAMES, b][idx]        # [K, V_gr] content frames (in order)
        tc = tgt_sep[b, 1:][idx]                      # [K] content tokens (== graphmes units)
        ref = F.one_hot(tc, num_classes=num_classes).to(ls.dtype)
        p_sep = F.log_softmax(ls, dim=-1)
        # KL(one_hot || p_sep): pulls the grapheme decoder towards reproducing the
        # original grapheme sequence (separators removed).
        losses.append(F.kl_div(p_sep, ref, reduction="batchmean"))
    if not losses:
        return torch.zeros((), device=logits_sep.device, dtype=logits_sep.dtype)
    return torch.stack(losses).mean()
