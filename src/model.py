"""Multitask seq2seq G2P model (PyTorch, CPU-friendly).

Shared encoder + per-task decoder heads.  The four training targets are:

  * ``phonemes``          -- a phoneme *symbol* sequence (embedded via the
                            phoneme embedding table, predicted by ``phoneme_head``).
  * ``separated_graphmes``/``separated_phonemes``/``aligned_phonemes`` -- these are
    derived from the source graphmes / phonemes and are encoded as flat *count*
    sequences (segment lengths) by :class:`src.preprocessing.CountCodec`, so they
    share one ``count_embed`` table and one ``count_head``.  At inference the
    predicted counts regroup the base sequence, which removes the old KL
    consistency supervision entirely.

The model is deliberately small (single-layer LSTMs, no attention on the
encoder) so it trains in minutes on CPU and in seconds on a GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.preprocessing import TARGET_NAMES


def _syllable_violation(logits, ph_target, ph_pad, vowel_ids, lang, mask):
    """Diagnostic (NOT a loss): fraction of predicted count-groups whose phoneme
    content violates the one-vowel-per-syllable rule.

    Computed from argmax predictions, so it carries no gradient -- it only
    reports how far the raw count head is from the structural constraint.  A
    group is a segment of the predicted phoneme sequence (split by the predicted
    count ids); it "violates" if it contains != 1 vowel nucleus.
    """
    pred = logits.argmax(dim=-1)  # [T-1, B]
    B = pred.size(1)
    viol = 0
    tot = 0
    for b in range(B):
        lid = int(lang[b].item())
        if lid >= len(mask) or not bool(mask[lid]):
            continue
        # base phoneme ids, stripping SOS=1 / EOS=2 / PAD=0
        base = [int(x) for x in ph_target[b].tolist() if x not in (0, 1, 2)]
        if not base:
            continue
        counts = [int(i) - 3 for i in pred[:, b].tolist() if int(i) >= 3]
        groups = []
        idx = 0
        for c in counts:
            if c <= 0:
                continue
            groups.append(base[idx:idx + c])
            idx += c
            if idx >= len(base):
                break
        if idx < len(base):
            groups.append(base[idx:])
        for g in groups:
            v = sum(1 for tok in g if tok in vowel_ids)
            if v != 1:
                viol += 1
            tot += 1
    if tot == 0:
        return None
    return torch.tensor(viol / tot, dtype=logits.dtype, device=logits.device)


class Encoder(nn.Module):
    def __init__(self, src_vocab_size: int, embed_dim: int, hidden: int,
                 num_layers: int, dropout: float, lang_embed_dim: int,
                 num_langs: int, dec_num_layers: int = 1):
        super().__init__()
        self.dec_num_layers = dec_num_layers
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        if lang_embed_dim and num_langs > 0:
            self.lang_embed = nn.Embedding(num_langs, lang_embed_dim, padding_idx=0)
            self.lang_proj = nn.Linear(lang_embed_dim, embed_dim)
        else:
            self.lang_embed = None
            self.lang_proj = None
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim, hidden, num_layers=num_layers, batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )

    def forward(self, src: torch.Tensor, src_len: torch.Tensor, lang: torch.Tensor):
        # src: [S, B]
        x = self.embed(src)
        if self.lang_embed is not None:
            le = self.lang_embed(lang)            # [B, lang_embed_dim]
            x = x + self.lang_proj(le).unsqueeze(0)  # broadcast over time
        x = self.dropout(x)
        # pack for the BiLSTM so padding frames don't pollute the hidden state
        packed = nn.utils.rnn.pack_padded_sequence(
            x, src_len.cpu().clamp(min=1).to("cpu"), enforce_sorted=False
        )
        out, (h, c) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)   # [S, B, 2*hidden]

        # concat the two directions of the top layer -> [B, 2*hidden]
        h_top = torch.cat([h[-2], h[-1]], dim=-1)
        c_top = torch.cat([c[-2], c[-1]], dim=-1)
        # repeat per decoder layer (decoder may have a different depth than enc)
        h_dec = h_top.unsqueeze(0).repeat(self.dec_num_layers, 1, 1)
        c_dec = c_top.unsqueeze(0).repeat(self.dec_num_layers, 1, 1)
        return out, (h_dec, c_dec)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=num_layers, batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, y_emb: torch.Tensor, state):
        # y_emb: [T, B, input_dim]
        out, state = self.lstm(y_emb, state)
        return self.dropout(out), state


class G2PModel(nn.Module):
    def __init__(self, src_vocab_size: int, phoneme_vocab_size: int,
                 count_vocab_size: int, num_langs: int,
                 embed_dim: int = 128, enc_layers: int = 1, dec_layers: int = 1,
                 enc_heads: int = 4, dec_hidden: int = 256, ffn_dim: int = 512,
                 dropout: float = 0.1, lang_embed_dim: int = 8):
        super().__init__()
        self.num_langs = num_langs
        self.enc = Encoder(src_vocab_size, embed_dim, dec_hidden // 2, enc_layers,
                           dropout, lang_embed_dim, num_langs, dec_layers)

        # phoneme symbol pathway
        self.phoneme_embed = nn.Embedding(phoneme_vocab_size, dec_hidden, padding_idx=0)
        self.phoneme_head = nn.Linear(dec_hidden, phoneme_vocab_size)

        # shared count pathway for the three derived tasks
        self.count_embed = nn.Embedding(count_vocab_size, dec_hidden, padding_idx=0)
        self.count_head = nn.Linear(dec_hidden, count_vocab_size)

        # decoder input is the embedding of the previously predicted token; its
        # dimensionality equals dec_hidden regardless of which task is active.
        self.dec = Decoder(dec_hidden, dec_hidden, dec_layers, dropout)

        # cross-attention over the encoder output (manual SDPA so it traces
        # cleanly to ONNX; key_padding_mask is applied as additive -inf).
        self.attn_q = nn.Linear(dec_hidden, dec_hidden)
        self.attn_k = nn.Linear(dec_hidden, dec_hidden)
        self.attn_v = nn.Linear(dec_hidden, dec_hidden)
        self.attn_heads = enc_heads
        self.attn_scale = (dec_hidden // enc_heads) ** -0.5
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_proj = nn.Linear(dec_hidden * 2, dec_hidden)

    # ----- helpers -------------------------------------------------------- #
    def _embed_prev(self, prev: torch.Tensor, task: str) -> torch.Tensor:
        return self.phoneme_embed(prev) if task == "phonemes" else self.count_embed(prev)

    def _cross_attn(self, h_dec, enc_out, key_pad):
        # h_dec: [T, B, H] (queries), enc_out: [S, B, H] (keys/values)
        # key_pad: [B, S] boolean-ish mask (True where padded)
        T, B, H = h_dec.shape
        n = self.attn_heads
        d = H // n
        q = self.attn_q(h_dec).view(T, B, n, d).transpose(0, 2)   # [n, B, T, d]
        k = self.attn_k(enc_out).view(-1, B, n, d).transpose(0, 2)  # [n, B, S, d]
        v = self.attn_v(enc_out).view(-1, B, n, d).transpose(0, 2)  # [n, B, S, d]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale  # [n, B, T, S]
        if key_pad is not None:
            # key_pad: [B, S] -> [n, B, 1, S] to broadcast over heads & queries
            m = key_pad.unsqueeze(0).unsqueeze(2)                  # [1, B, 1, S]
            scores = scores.masked_fill(m, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v)                               # [n, B, T, d]
        ctx = ctx.transpose(0, 2).contiguous().view(T, B, H)      # [T, B, H]
        return self.attn_proj(torch.cat([h_dec, ctx], dim=-1))

    def _head(self, h: torch.Tensor, task: str) -> torch.Tensor:
        return self.phoneme_head(h) if task == "phonemes" else self.count_head(h)

    def forward(self, src, src_len, lang, targets, tgt_lens, pad_idx_dict,
                tasks=None, vowel_ids=None, syllable_lang_mask=None,
                syllable_constraint_weight=0.0):
        if tasks is None:
            tasks = list(TARGET_NAMES)
        # collate yields batch-first [B, S]; the encoder wants seq-first [S, B]
        src = src.transpose(0, 1).contiguous()
        enc_out, dec_state = self.enc(src, src_len, lang)   # enc_out: [S, B, H]
        B = src.size(1)
        total = 0.0
        loss_parts = {}
        task_logits = {}
        for task in tasks:
            tgt = targets[task]            # [B, T]
            tlen = tgt_lens[task]          # [B]
            T = tgt.size(1)
            pad_idx = pad_idx_dict[task]
            # teacher forcing: feed tgt[:, :-1], predict tgt[:, 1:]
            prev = tgt[:, :-1].transpose(0, 1).contiguous()   # [T-1, B]
            y_emb = self._embed_prev(prev, task)
            h, state = self.dec(y_emb, dec_state)
            # cross attention: query = decoder hidden, kv = encoder output
            h = self._cross_attn(h, enc_out, (src == 0).transpose(0, 1))
            logits = self._head(h, task)   # [T-1, B, V]
            task_logits[task] = logits
            gold = tgt[:, 1:].transpose(0, 1).contiguous()    # [T-1, B]
            mask = (gold != pad_idx)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), gold.reshape(-1),
                reduction="none",
            ).reshape_as(gold)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            loss_parts[task] = loss
            total = total + loss
        # ---- diagnostic: fraction of count-groups violating one-vowel/syllable ----
        # argmax-based, so it carries no gradient -- purely a monitoring signal of
        # how far the raw count head is from the structural constraint.
        if (syllable_constraint_weight > 0.0 and vowel_ids is not None
                and syllable_lang_mask is not None):
            ph_target = targets.get("phonemes")
            if ph_target is not None:
                for task in ("separated_phonemes", "aligned_phonemes"):
                    lg = task_logits.get(task)
                    if lg is None:
                        continue
                    viol = _syllable_violation(
                        lg, ph_target, pad_idx_dict.get("phonemes", 0),
                        vowel_ids, lang, syllable_lang_mask)
                    if viol is not None:
                        loss_parts[task + "_syllable_viol"] = viol
        return total, loss_parts

    # ----- greedy generation --------------------------------------------- #
    def generate(self, src, src_len, lang, max_len: int, sos_idx: int,
                 tasks=None, pad_idx_dict=None, repetition_penalty: float = 1.5):
        if tasks is None:
            tasks = list(TARGET_NAMES)
        eos_idx = 2  # PAD=0, SOS=1, EOS=2 (shared across vocabs)
        enc_out, dec_state = self.enc(src, src_len, lang)
        B = src.size(1)
        # Each task decodes independently from the shared encoder state, so it
        # keeps its own decoder state, accumulated id sequence and EOS flag.
        states = {t: dec_state for t in tasks}
        seqs = {t: [[] for _ in range(B)] for t in tasks}
        done = {t: [False] * B for t in tasks}
        key_pad = (src == 0).transpose(0, 1)
        for _ in range(max_len):
            still_active = False
            for task in tasks:
                if all(done[task]):
                    continue
                still_active = True
                prev = torch.tensor(
                    [s[-1] if (s and not done[task][b]) else sos_idx
                     for b, s in enumerate(seqs[task])],
                    dtype=torch.long, device=src.device)
                y_emb = self._embed_prev(prev.unsqueeze(0), task)  # [1, B, H]
                h, states[task] = self.dec(y_emb, states[task])
                h = self._cross_attn(h, enc_out, key_pad)
                logits = self._head(h[0], task)                   # [B, V]
                logits = logits.clone()
                # Repetition penalty is only meaningful for the free-text phoneme
                # stream.  Count tasks (separated/aligned graphemes & phonemes)
                # legitimately repeat small values -- especially "1" -- so
                # penalising repetition forces the decoder to emit larger counts
                # that *merge* segments, dropping separators in the reconstruction.
                # Hence the penalty is skipped for count tasks.
                apply_rp = (task == "phonemes") and (repetition_penalty != 1.0)
                if apply_rp:
                    for b in range(B):
                        if done[task][b]:
                            continue
                        for tok in set(seqs[task][b]):
                            logits[b, tok] /= repetition_penalty
                nxt = logits.argmax(dim=-1)                       # [B]
                for b in range(B):
                    if done[task][b]:
                        continue
                    tid = int(nxt[b])
                    if tid == eos_idx:
                        done[task][b] = True
                        continue
                    seqs[task][b].append(tid)
            if not still_active:
                break
        # build [B, T] id tensors, trimmed to the actual generated length so the
        # output is NOT force-padded to max_len (decoding already stops at EOS).
        out = {}
        for task in tasks:
            T = max((len(s) for s in seqs[task]), default=0)
            T = min(T, max_len)
            ids = torch.zeros(B, T, dtype=torch.long, device=src.device)
            for b in range(B):
                row = seqs[task][b]
                if not row:
                    continue
                n = min(len(row), T)
                ids[b, :n] = torch.tensor(row[:n], dtype=torch.long, device=src.device)
            out[task] = ids
        return out
