"""Multi-task G2P model.

Architecture
------------
* A **Transformer encoder** consumes the (sub-word / per-character) grapheme
  sequence.  An optional **language embedding** is added to every encoder
  position so the same encoder can serve multiple languages.
* Four **independent LSTM + attention decoders** share the encoder output and
  each specialises in one of the four targets:
    - ``phonemes``         (no separator)
    - ``separated_graphmes`` (``|`` separated graphemes)
    - ``separated_phonemes`` (``|`` separated phonemes)
    - ``aligned_phonemes``   (``/`` separated phonemes)
  Each decoder carries its own embedding table, LSTM, attention and a learnable
  task-specific output bias -- this is what makes the task genuinely multi-task
  while keeping a single shared encoder.

The training path uses teacher forcing; the ``generate`` path runs greedy
auto-regressive decoding with a fixed number of steps, which is what gets
exported to ONNX (a single graph with 4 output nodes).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data import TARGET_NAMES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, 1, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, B, D]
        return x + self.pe[: x.size(0)]

    def at(self, t: int) -> torch.Tensor:
        return self.pe[t : t + 1]  # [1, 1, D]


class Encoder(nn.Module):
    """Stacked bidirectional-LSTM encoder.

    We deliberately avoid ``nn.TransformerEncoder``: its internal reshapes do not
    export cleanly to ONNX with a dynamic sequence length.  A (padded) BiLSTM
    exports without trouble and still gives a strong, CUDA-trainable context.
    The language embedding is added to every time step before the LSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        enc_hidden: int,
        num_layers: int,
        dropout: float,
        num_langs: int,
        lang_embed_dim: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim)
        self.use_lang = lang_embed_dim > 0
        if self.use_lang:
            self.lang_embed = nn.Embedding(num_langs, lang_embed_dim)
            self.lang_proj = nn.Linear(lang_embed_dim, embed_dim)
        else:
            self.lang_embed = None
            self.lang_proj = None

        self.bilstm = nn.LSTM(
            embed_dim,
            enc_hidden,
            num_layers,
            dropout=dropout,
            batch_first=False,
            bidirectional=True,
        )
        # 2*enc_hidden (bi) -> enc_dim consumed by the decoders' attention
        self.proj = nn.Linear(2 * enc_hidden, embed_dim)

    def forward(
        self, src: torch.Tensor, src_len: torch.Tensor, lang: torch.Tensor
    ) -> torch.Tensor:
        x = self.embed(src) * math.sqrt(self.embed_dim)  # [S, B, D]
        if self.use_lang:
            lv = self.lang_proj(self.lang_embed(lang))  # [B, D]
            x = x + lv.unsqueeze(0)  # broadcast over seq
        x = self.pos(x)
        # Pack by length so the BiLSTM skips the padded tail of every sequence.
        # Combined with length-sorted batches (src/data.py) this makes the
        # encoder cost ~ O(B * avg_src_len) instead of O(B * max_src_len), i.e. it
        # stops growing super-linearly with batch size (the bug that made batch
        # 1024 slower than batch 128).
        packed = pack_padded_sequence(
            x, src_len.cpu(), enforce_sorted=False, batch_first=False
        )
        out, _ = self.bilstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=False)  # [S, B, 2*H]
        out = self.proj(out)  # [S, B, D]
        return out  # [S, B, D]


class TaskDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dec_hidden: int,
        num_layers: int,
        dropout: float,
        enc_dim: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos = PositionalEncoding(embed_dim)
        # PyTorch's LSTM only applies dropout *between* layers, so a single-layer
        # LSTM silently ignores `dropout` (and warns). For 1-layer decoders we
        # therefore zero the LSTM's internal dropout and apply an explicit
        # nn.Dropout on the output in the training (teacher-forcing) path, so the
        # configured `dropout` actually regularises the decoder.
        self.lstm = nn.LSTM(
            embed_dim, dec_hidden, num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=False
        )
        self.rec_drop = nn.Dropout(dropout) if num_layers == 1 else None
        self.attn_query = nn.Linear(dec_hidden, enc_dim)
        self.attn_combine = nn.Linear(dec_hidden + enc_dim, dec_hidden)
        self.out = nn.Linear(dec_hidden, vocab_size)
        # weight tying: share the output projection with the input embedding
        # (both are [vocab_size, dec_hidden]). Cuts the decoder parameter count
        # roughly in half with no change to the exportable ONNX graph.
        if dec_hidden == embed_dim:
            self.out.weight = self.embed.weight
        self.task_bias = nn.Parameter(torch.zeros(vocab_size))

    def _attend(self, dec_h: torch.Tensor, enc_out: torch.Tensor, src_mask: Optional[torch.Tensor]):
        # dec_h: [T, B, H]; enc_out: [S, B, enc_dim]
        q = self.attn_query(dec_h)  # [T, B, enc_dim]
        q_t = q.transpose(0, 1)            # [B, T, enc_dim]
        enc_t = enc_out.transpose(0, 1)    # [B, S, enc_dim]
        scores = torch.bmm(q_t, enc_t.transpose(1, 2))  # [B, T, S]
        if src_mask is not None:
            scores = scores.masked_fill(src_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, enc_t)   # [B, T, enc_dim]
        context = context.transpose(0, 1)  # [T, B, enc_dim]
        combined = torch.tanh(self.attn_combine(torch.cat([dec_h, context], dim=-1)))
        return combined

    def forward_teacher(
        self,
        tgt: torch.Tensor,
        tgt_len: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # training targets arrive batch-first [B, T]; decoders are seq-first
        tgt = tgt.transpose(0, 1).contiguous()  # [T, B]
        emb = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        emb = self.pos(emb)
        packed = pack_padded_sequence(
            emb, tgt_len.cpu(), enforce_sorted=False, batch_first=False
        )
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=False)  # [T, B, H]
        if self.rec_drop is not None:
            out = self.rec_drop(out)
        combined = self._attend(out, enc_out, src_mask)
        logits = self.out(combined) + self.task_bias  # [T, B, V]
        return logits

    def generate_step(
        self,
        inp: torch.Tensor,
        step: int,
        hidden,
        enc_out: torch.Tensor,
        src_mask: Optional[torch.Tensor],
    ):
        emb = self.embed(inp) * math.sqrt(self.embed.embedding_dim)  # [1, B, E]
        emb = emb + self.pos.at(step)
        out, hidden = self.lstm(emb, hidden)  # out: [1, B, H]
        combined = self._attend(out, enc_out, src_mask)  # [1, B, H]
        logits = self.out(combined) + self.task_bias  # [1, B, V]
        return logits, hidden


class G2PModel(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        phoneme_vocab_size: int,
        grapheme_tgt_vocab_size: int,
        num_langs: int,
        embed_dim: int = 256,
        enc_layers: int = 3,
        dec_layers: int = 2,
        enc_heads: int = 4,
        dec_hidden: int = 256,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        lang_embed_dim: int = 16,
    ):
        super().__init__()
        self.num_langs = num_langs
        self.encoder = Encoder(
            src_vocab_size, embed_dim, dec_hidden, enc_layers,
            dropout, num_langs, lang_embed_dim,
        )
        enc_dim = embed_dim
        self.dec_phonemes = TaskDecoder(phoneme_vocab_size, embed_dim, dec_hidden, dec_layers, dropout, enc_dim)
        self.dec_separated_graphmes = TaskDecoder(grapheme_tgt_vocab_size, embed_dim, dec_hidden, dec_layers, dropout, enc_dim)
        self.dec_separated_phonemes = TaskDecoder(phoneme_vocab_size, embed_dim, dec_hidden, dec_layers, dropout, enc_dim)
        self.dec_aligned_phonemes = TaskDecoder(phoneme_vocab_size, embed_dim, dec_hidden, dec_layers, dropout, enc_dim)

    # ---- training ----
    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        lang: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        target_lens: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        src = src.transpose(0, 1).contiguous()   # [S, B]
        src_mask = _make_src_mask(src, src_len)  # [B, S] bool (src is seq-first)
        enc_out = self.encoder(src, src_len, lang)  # [S, B, D]
        logits: Dict[str, torch.Tensor] = {}
        logits["phonemes"] = self.dec_phonemes.forward_teacher(
            targets["phonemes"], target_lens["phonemes"], enc_out, src_mask
        )
        logits["separated_graphmes"] = self.dec_separated_graphmes.forward_teacher(
            targets["separated_graphmes"], target_lens["separated_graphmes"], enc_out, src_mask
        )
        logits["separated_phonemes"] = self.dec_separated_phonemes.forward_teacher(
            targets["separated_phonemes"], target_lens["separated_phonemes"], enc_out, src_mask
        )
        logits["aligned_phonemes"] = self.dec_aligned_phonemes.forward_teacher(
            targets["aligned_phonemes"], target_lens["aligned_phonemes"], enc_out, src_mask
        )
        return logits

    # ---- inference / ONNX (free-run greedy, fixed steps) ----
    def generate(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        lang: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: Optional[int] = None,
        early_stop: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = src.size(1)
        device = src.device
        src_mask = _make_src_mask(src, src_len)
        enc_out = self.encoder(src, src_len, lang)

        decoders = {
            "phonemes": self.dec_phonemes,
            "separated_graphmes": self.dec_separated_graphmes,
            "separated_phonemes": self.dec_separated_phonemes,
            "aligned_phonemes": self.dec_aligned_phonemes,
        }
        # store outputs per step, one tensor [1, B, V] each
        outputs: Dict[str, List[torch.Tensor]] = {k: [] for k in decoders}
        hidden = {k: None for k in decoders}
        inp = {k: torch.full((1, B), sos_idx, dtype=torch.long, device=device) for k in decoders}

        for t in range(max_len):
            all_eos = early_stop and eos_idx is not None
            for name, dec in decoders.items():
                logits, h = dec.generate_step(inp[name], t, hidden[name], enc_out, src_mask)
                hidden[name] = h
                outputs[name].append(logits)
                nxt = logits.argmax(dim=-1)  # [1, B]
                inp[name] = nxt
                if all_eos:
                    all_eos = all_eos and bool((nxt == eos_idx).all())
            if all_eos:
                break
        return {k: torch.cat(v, dim=0) for k, v in outputs.items()}  # [T, B, V]


def _make_src_mask(src: torch.Tensor, src_len: torch.Tensor) -> torch.Tensor:
    """Return a [B, S] boolean padding mask (True where padded).

    ``src`` MUST be seq-first ``[S, B]``.  Avoids any Python-level branch on
    tensor sizes so the graph traces cleanly for ONNX export (no baked-in
    sequence length).
    """
    B = src_len.size(0)
    S = src.size(0)
    idx = torch.arange(S, device=src.device).unsqueeze(0).expand(B, S)
    return idx >= src_len.unsqueeze(1)  # True where padded
