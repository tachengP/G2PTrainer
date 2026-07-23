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


def phoneme_consistency(logits_base, logits_sep, tgt_sep, sep_id, pad_id, eos_id, space_id=1):
    """KL( p_base || p_sep ) over the content frames of the separator decoder.

    Fully vectorised over the batch (no Python loop) so it stays cheap at large
    batch sizes -- the old per-sample ``for b`` loop dominated wall-time when
    ``B`` was large (e.g. 2048) even though the model itself is tiny.

    Alignment: the base decoder has no separators, so its frame ``i`` is its
    *i*-th (content) frame; the separator decoder's *i*-th content frame lives at
    its true position ``idx[i]`` in the (separator-padded) sequence. We therefore
    gather the separator logits at the content positions and compare them 1:1 with
    the base decoder's first ``K`` frames, exactly mirroring the old loop.

    Args:
        logits_base: seq-first ``[T_base, B, V]`` logits of the ``phonemes`` decoder.
        logits_sep:  seq-first ``[T_sep, B, V]`` logits of a separator decoder
                     (``separated_phonemes`` or ``aligned_phonemes``). Shares ``V``.
        tgt_sep:     batch-first ``[B, T_sep]`` ground-truth of the separator task.
        sep_id:      vocab id of the separator token (``|`` or ``/``) in ``V``.
        pad_id:      vocab id of the padding token in ``V``.
        eos_id:      vocab id of the stop token; excluded from content frames so the
                     trailing ``<eos>`` is not treated as a phoneme to align.
        space_id:    vocab id of the space token (`` ``). Space is ALSO a separator
                     (a finer-grained one than ``|``/``/``); it must be excluded from
                     content frames so the model learns to predict separator *positions*
                     rather than treating a space as phoneme content. See refine.md /
                     the user note: "空格也是一种分隔符".
    """
    B = tgt_sep.size(0)
    pred = tgt_sep[:, 1:]                                      # [B, T_sep-1]
    content = (pred != sep_id) & (pred != space_id) & (pred != pad_id) & (pred != eos_id)
    Kb = content.sum(dim=1)                                    # [B] content frames per sample
    Tbase = logits_base.size(0)
    # base content frames = its first K_b frames (base has no separators)
    base_mask = torch.arange(Tbase, device=logits_base.device).unsqueeze(0) < Kb.unsqueeze(1)
    base_sel = logits_base.transpose(0, 1)[base_mask]          # [N, V]
    # separator content frames = their true positions in the sep sequence
    sep_t = logits_sep[_PRED_FRAMES].transpose(0, 1)           # [B, T_sep-1, V]
    idx = content.nonzero(as_tuple=False)                      # [N, 2] (b, t)
    if idx.numel() == 0:
        return torch.zeros((), device=logits_base.device, dtype=logits_base.dtype)
    sep_sel = sep_t[idx[:, 0], idx[:, 1]]                      # [N, V]
    p_base = F.softmax(base_sel, dim=-1)
    p_sep = F.log_softmax(sep_sel, dim=-1)
    kl = F.kl_div(p_sep, p_base, reduction="none")             # [N, V]
    # PyTorch's kl_div(reduction="batchmean") divides by K (the number of frames),
    # NOT by K*V -- it already sums over the vocab axis. Match that exactly.
    invK = (1.0 / Kb[idx[:, 0]]).unsqueeze(-1)                 # [N, 1]
    return (kl * invK).sum() / B


def grapheme_consistency(logits_sep, tgt_sep, sep_id, pad_id, num_classes, eos_id, space_id=1):
    """KL( one_hot(content) || p_sep ) over the content frames of ``separated_graphmes``.

    Fully vectorised over the batch (no Python loop). See :func:`phoneme_consistency`.
    """
    B = tgt_sep.size(0)
    pred = tgt_sep[:, 1:]                                      # [B, T_sep-1]
    content = (pred != sep_id) & (pred != space_id) & (pred != pad_id) & (pred != eos_id)
    Kb = content.sum(dim=1)                                    # [B]
    sep_t = logits_sep[_PRED_FRAMES].transpose(0, 1)           # [B, T_sep-1, V]
    idx = content.nonzero(as_tuple=False)                      # [N, 2] (b, t)
    if idx.numel() == 0:
        return torch.zeros((), device=logits_sep.device, dtype=logits_sep.dtype)
    sep_sel = sep_t[idx[:, 0], idx[:, 1]]                      # [N, V]
    tc = tgt_sep[:, 1:][idx[:, 0], idx[:, 1]]                  # [N] content tokens
    ref = F.one_hot(tc, num_classes=num_classes).to(sep_sel.dtype)
    p_sep = F.log_softmax(sep_sel, dim=-1)
    kl = F.kl_div(p_sep, ref, reduction="none")                # [N, V]
    # PyTorch's kl_div(reduction="batchmean") divides by K (frames), not K*V.
    invK = (1.0 / Kb[idx[:, 0]]).unsqueeze(-1)                 # [N, 1]
    return (kl * invK).sum() / B
