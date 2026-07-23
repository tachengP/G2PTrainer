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


class SharedTaskDecoder(nn.Module):
    """One decoder *body* shared by all four sequence tasks.

    It mirrors the old per-task LSTM decoder (teacher-forcing path with packed
    sequences + cross-attention to the encoder), but it does NOT own an embedding
    table or an output projection -- those are task-specific (the input tokens of
    ``separated_graphmes`` live in the *grapheme* vocab, the other three in the
    *phoneme* vocab).  Instead the body receives an already-embedded sequence and
    returns the combined hidden, which the per-task :class:`TaskHead` projects to
    its own vocabulary.  This removes the 4x LSTM + 4x attention redundancy of the
    old design (refine.md) while keeping all four outputs.

    (One shared embedding table is shared by the three phoneme tasks; the grapheme
    task uses its own table -- see :class:`G2PModel`.)
    """

    def __init__(
        self,
        embed_dim: int,
        dec_hidden: int,
        num_layers: int,
        dropout: float,
        enc_dim: int,
    ):
        super().__init__()
        self.pos = PositionalEncoding(embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, dec_hidden, num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=False
        )
        self.rec_drop = nn.Dropout(dropout) if num_layers == 1 else None
        self.attn_query = nn.Linear(dec_hidden, enc_dim)
        self.attn_combine = nn.Linear(dec_hidden + enc_dim, dec_hidden)

    def _run(self, emb: torch.Tensor, tgt_len: torch.Tensor,
             enc_out: torch.Tensor, src_mask: Optional[torch.Tensor],
             hidden=None) -> torch.Tensor:
        emb = self.pos(emb)
        if tgt_len is not None:
            # teacher-forcing path: pack the padded target sequence
            packed = pack_padded_sequence(
                emb, tgt_len.cpu(), enforce_sorted=False, batch_first=False
            )
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=False)  # [T, B, H]
        else:
            out, hidden = self.lstm(emb, hidden)  # [1, B, H]
        if self.rec_drop is not None:
            out = self.rec_drop(out)
        combined = self._attend(out, enc_out, src_mask)
        return combined  # [T, B, H] or [1, B, H] hidden

    def forward_teacher(self, emb, tgt_len, enc_out, src_mask):
        return self._run(emb, tgt_len, enc_out, src_mask)

    def generate_step(self, emb, step, hidden, enc_out, src_mask):
        emb = emb + self.pos.at(step)
        combined = self._run(emb, None, enc_out, src_mask, hidden=hidden)
        return combined, hidden  # combined: [1, B, H]

    def _attend(self, dec_h: torch.Tensor, enc_out: torch.Tensor, src_mask: Optional[torch.Tensor]):
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


class TaskHead(nn.Module):
    """Linear head: shared decoder hidden -> task logits.

    Weight-tying with the task's input embedding (both ``[vocab, dec_hidden]``)
    keeps the head parameter count tiny when shapes line up, and links input and
    output projections of the same vocabulary.
    """

    def __init__(self, dec_hidden: int, vocab_size: int, embed_weight: nn.Parameter):
        super().__init__()
        self.out = nn.Linear(dec_hidden, vocab_size, bias=True)
        if dec_hidden == embed_weight.size(1):
            self.out.weight = embed_weight  # tie with the task's input embedding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.out(hidden)  # [T, B, V]


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
        # Input embedding tables. The three phoneme tasks share one table; the
        # grapheme task uses its own (they live in different vocabularies).
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, embed_dim, padding_idx=0)
        self.grapheme_embed = nn.Embedding(grapheme_tgt_vocab_size, embed_dim, padding_idx=0)
        # ONE shared decoder body for all four sequence tasks.
        self.shared_decoder = SharedTaskDecoder(
            embed_dim, dec_hidden, dec_layers, dropout, enc_dim
        )
        # Four cheap task heads (weight-tied to their own input embedding table).
        self.heads = nn.ModuleDict({
            "phonemes": TaskHead(dec_hidden, phoneme_vocab_size, self.phoneme_embed.weight),
            "separated_phonemes": TaskHead(dec_hidden, phoneme_vocab_size, self.phoneme_embed.weight),
            "aligned_phonemes": TaskHead(dec_hidden, phoneme_vocab_size, self.phoneme_embed.weight),
            "separated_graphmes": TaskHead(dec_hidden, grapheme_tgt_vocab_size, self.grapheme_embed.weight),
        })
        # Graph boundary: PAD/SEP classification straight from the encoder hidden
        # (no autoregressive decoding needed -- it is a per-source-timestep tag).
        self.graph_boundary = nn.Linear(enc_dim, 2)

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

        # Graph boundary over the source sequence.
        glogits = self.graph_boundary(enc_out)   # [S, B, 2]

        # One shared decoder body, four linear heads. Each task embeds its own
        # target with its own vocabulary table, then all share the SAME decoder
        # LSTM/attention weights (refine.md: 4x decoder params -> 1x).
        logits: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            tgt = targets[name]                      # [B, T] batch-first
            emb = self._embed_task(name, tgt)
            hidden = self.shared_decoder.forward_teacher(
                emb, target_lens[name], enc_out, src_mask
            )  # [T, B, H]
            logits[name] = head(hidden)
        logits["graph_boundary"] = glogits
        logits["graph_boundary_len"] = src_len
        return logits

    def _embed_task(self, name: str, tgt: torch.Tensor) -> torch.Tensor:
        """Embed a batch-first target with the correct vocabulary table."""
        table = self.grapheme_embed if name == "separated_graphmes" else self.phoneme_embed
        tgt = tgt.transpose(0, 1).contiguous()  # [T, B]
        return table(tgt) * math.sqrt(table.embedding_dim)  # [T, B, E]

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

        # Greedy decode each task independently (different vocabularies), but all
        # through the SAME shared decoder body.
        outputs: Dict[str, List[torch.Tensor]] = {k: [] for k in self.heads}
        hiddens = {k: None for k in self.heads}
        prev = {k: torch.full((1, B), sos_idx, dtype=torch.long, device=device) for k in self.heads}

        for t in range(max_len):
            step_eos = early_stop and eos_idx is not None
            for name in self.heads:
                table = self.grapheme_embed if name == "separated_graphmes" else self.phoneme_embed
                emb = table(prev[name]) * math.sqrt(table.embedding_dim)  # [1, B, E]
                combined, hiddens[name] = self.shared_decoder.generate_step(
                    emb, t, hiddens[name], enc_out, src_mask
                )  # [1, B, H]
                logits = self.heads[name](combined)       # [1, B, V]
                outputs[name].append(logits)
                prev[name] = logits.argmax(dim=-1)        # [1, B]
                if step_eos:
                    step_eos = step_eos and bool((prev[name] == eos_idx).all())
            if step_eos:
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
