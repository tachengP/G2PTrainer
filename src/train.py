"""Training entrypoint (PyTorch, CPU- or GPU-friendly)."""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; degrade gracefully to a plain iterator
    tqdm = None

from src.config import build_config
from src.model import G2PModel
from src.bin_data import run_binarize, load_binary
from src.data import make_loader
from src.preprocessing import (
    PIPE,
    SLASH,
    TARGET_NAMES,
    reconstruct_groups,
    resegment_by_vowels,
    record_targets,
    parse_no_sep,
)

SEP_UNIT = {
    "separated_graphmes": ("|", ""),
    "separated_phonemes": ("|", " "),
    "aligned_phonemes": ("/", " "),
}


def build_model(cfg, meta, count_vocab_size):
    model = G2PModel(
        src_vocab_size=meta["src_vocab_size"],
        phoneme_vocab_size=meta["phoneme_vocab_size"],
        count_vocab_size=count_vocab_size,
        num_langs=meta["num_langs"],
        embed_dim=cfg.embed_dim,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        enc_heads=cfg.enc_heads,
        dec_hidden=cfg.dec_hidden,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        lang_embed_dim=cfg.lang_embed_dim,
    )
    return model


def reconstruct_prediction(src_units, phoneme_tokens, counts, task, vowel_set=None):
    """Turn predicted counts back into the separator-joined string for monitoring.

    When ``vowel_set`` is given (a CJK-like language with one-char-one-syllable),
    the phoneme-group tasks are re-segmented by vowel nucleus instead of trusting
    the count head -- which is exactly what inference does, so the training log
    shows the same corrected output the user will actually get.
    """
    sep, unit = SEP_UNIT[task]
    if task == "separated_graphmes":
        base = src_units
    else:
        base = phoneme_tokens
    if vowel_set is not None and task in ("separated_phonemes", "aligned_phonemes"):
        return resegment_by_vowels(base, vowel_set, sep, unit)
    return reconstruct_groups(base, counts, sep, unit)


def counts_from_ids(ids, count_codec):
    return count_codec.decode([int(i) for i in ids])


def _deploy_state(model, dtype: str) -> dict:
    """Return a weights-only checkpoint, optionally downcast to fp16.

    The exported key is ``dtype`` (not ``export_dtype``) so that
    :func:`src.utils.load_model_weights` can detect and recast an fp16 file.
    """
    sd = model.state_dict()
    if dtype == "fp16":
        sd = {k: v.half() for k, v in sd.items()}
    return {"model": sd, "dtype": dtype}


def main():
    # All CLI flags (--config, --device, --force_rebinarize, --max_samples, ...)
    # are registered by build_config(); train.py itself adds no argparse parser.
    cfg = build_config()
    # resolve the binary dir (data artifacts: vocab/codec/npz) and the run dir
    # (training artifacts: checkpoints + tensorboard). These are intentionally
    # separate: data lives under {data_dir}/binary; training outputs go to
    # {output_dir}/{model_name} (config-driven, never hardcoded).
    if cfg.binary_dir is None:
        cfg.binary_dir = os.path.join(cfg.data_dir, "binary")
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    need_bin = cfg.force_rebinarize or not os.path.exists(
        os.path.join(cfg.binary_dir, "meta.json"))
    if need_bin:
        print(f"[binarize] building {cfg.binary_dir} from {cfg.data_dir} "
              f"({cfg.file_glob}, max_samples={cfg.max_samples}) ...")
        run_binarize(cfg, cfg.binary_dir)
    else:
        print(f"[binarize] reusing existing {cfg.binary_dir}")

    (train_ds, val_ds, src_vocab, phoneme_vocab, count_codec, tokenizer,
     pad_idx_dict, meta, val_monitor) = load_binary(cfg.binary_dir)

    print(f"[vocab] src={len(src_vocab)} phoneme={len(phoneme_vocab)} "
          f"count={count_codec.vocab_size} (max_count={count_codec.max_count}) "
          f"langs={meta['num_langs']}")

    # ---- structural constraint metadata (one vowel nucleus per syllable) ----
    # Derived in binarize from the aligned groups; lets us both fix the phoneme
    # separators at inference/monitor time and report a violation diagnostic.
    vowel_symbols = meta.get("vowel_phonemes") or {}
    syllable_is_char = meta.get("syllable_is_char") or {}
    # boolean mask over lang ids: which languages are one-char-one-syllable
    # NOTE: JSON round-trips dict keys to strings, so `id2lang` comes back keyed
    # by "0" (str) rather than 0 (int); coerce before indexing by lang id.
    id2lang = {int(k): v for k, v in meta["id2lang"].items()}
    syllable_lang_mask = torch.tensor(
        [bool(syllable_is_char.get(id2lang[i], False))
         for i in range(meta["num_langs"])],
        dtype=torch.bool)
    # global set of vowel phoneme vocab-ids (diagnostic only; the mask gates it)
    all_vowel_ids = set()
    for _lang, _syms in vowel_symbols.items():
        for _s in _syms:
            _idx = phoneme_vocab.stoi(_s)
            if _idx is not None:
                all_vowel_ids.add(_idx)
    use_syllable_aux = cfg.syllable_constraint_weight > 0 and bool(all_vowel_ids)
    if use_syllable_aux:
        print(f"[syllable] constraining separators for {len(all_vowel_ids)} vowel ids; "
              f"languages={[l for l, v in syllable_is_char.items() if v]}")

    model = build_model(cfg, meta, count_codec.vocab_size).to(device)

    # ---- run (training artifact) directory ----
    run_dir = os.path.join(cfg.output_dir, cfg.model_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    tb_dir = cfg.tb_log_dir or os.path.join(run_dir, "tb-logs")
    cfg.save(os.path.join(run_dir, "config.yaml"))

    # pin_memory is enabled for the train loader when on CUDA (and the dataset is
    # host-resident), which lets the H2D copies below run with non_blocking=True.
    pin_memory = device != "cpu" and not cfg.data_on_gpu
    non_blocking = pin_memory
    loader = make_loader(
        train_ds, cfg.batch_size, pad_idx_dict,
        num_workers=0 if cfg.data_on_gpu else cfg.num_workers,
        shuffle=True,
        pin_memory=pin_memory)
    val_loader = make_loader(val_ds, cfg.batch_size, pad_idx_dict, 0, shuffle=False,
                             pin_memory=False)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2)

    # ---- optional mixed precision (AMP) ----
    use_amp = bool(cfg.fp16) and device == "cuda"
    if use_amp:
        print("[amp] training with mixed precision (fp16)")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- tensorboard (only if enabled) ----
    writer = SummaryWriter(log_dir=tb_dir) if cfg.tensorboard else None

    # ---- fixed val samples (deterministic across runs via seeded RNG) ----
    if cfg.num_fixed_samples and len(val_monitor) > 0:
        _rng = random.Random(cfg.fixed_samples_seed)
        fixed_samples = _rng.sample(val_monitor, min(cfg.num_fixed_samples, len(val_monitor)))
    else:
        fixed_samples = val_monitor[:min(5, len(val_monitor))]

    # ---- optionally keep the whole dataset resident on CUDA ----
    if cfg.data_on_gpu and device.startswith("cuda"):
        print(f"[data_on_gpu] uploading dataset to {device} ...")
        train_ds.to_device(device)
        val_ds.to_device(device)

    start_epoch = 1
    best_val = float("inf")
    if cfg.resume and os.path.exists(cfg.resume):
        ck = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck.get("epoch", 1) + 1
        best_val = ck.get("best_val", best_val)
        print(f"[resume] epoch {start_epoch-1} from {cfg.resume}")

    global_step = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        # Accumulate the loss as a (detached) tensor and only convert to a Python
        # number at the end of the epoch.  `float(loss.detach())` each step would
        # force a device->host sync every batch, stalling the GPU pipeline.
        running = torch.zeros((), device=device)
        nbatches = 0
        train_iter = tqdm(loader, desc=f"epoch {epoch}", unit="batch",
                          leave=False) if tqdm is not None else loader
        for src, src_len, lang, targets, tgt_lens in train_iter:
            src = src.to(device, non_blocking=non_blocking)
            lang = lang.to(device, non_blocking=non_blocking)
            targets = {k: v.to(device, non_blocking=non_blocking) for k, v in targets.items()}
            tgt_lens = {k: v.to(device, non_blocking=non_blocking) for k, v in tgt_lens.items()}
            src_len = src_len.to(device, non_blocking=non_blocking)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss, parts = model(
                    src, src_len, lang, targets, tgt_lens, pad_idx_dict,
                    vowel_ids=all_vowel_ids if use_syllable_aux else None,
                    syllable_lang_mask=syllable_lang_mask if use_syllable_aux else None,
                    syllable_constraint_weight=cfg.syllable_constraint_weight,
                )
            optim.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()

            running += loss.detach()
            nbatches += 1
            global_step += 1
            if writer is not None and (cfg.log_every <= 0 or global_step % cfg.log_every == 0):
                for t, v in parts.items():
                    writer.add_scalar(f"train/{t}", float(v.detach()), global_step)
                if tqdm is not None:
                    train_iter.set_postfix(loss=f"{running.item() / nbatches:.4f}")

        train_loss = running.item() / max(1, nbatches)

        # ---- validation ----
        model.eval()
        vrunning = torch.zeros((), device=device)
        vbatches = 0
        val_iter = tqdm(val_loader, desc="val", unit="batch",
                        leave=False) if tqdm is not None else val_loader
        with torch.no_grad():
            for src, src_len, lang, targets, tgt_lens in val_iter:
                src = src.to(device)
                lang = lang.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                tgt_lens = {k: v.to(device) for k, v in tgt_lens.items()}
                src_len = src_len.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, parts = model(src, src_len, lang, targets, tgt_lens, pad_idx_dict)
                vrunning += loss.detach()
                vbatches += 1
        val_loss = vrunning.item() / max(1, vbatches)
        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"[epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} "
              f"time={dt:.1f}s lr={optim.param_groups[0]['lr']:.5f}")

        # ---- fixed-sample inspection: reconstruct strings from predicted counts ----
        with torch.no_grad():
            sample = fixed_samples
            if sample:
                inp = []
                for m in sample:
                    units = tokenizer.tokenize(m["text"])
                    inp.append((units, src_vocab.encode(units)))
                maxlen = max(len(u) for u, _ in inp)
                src_t = torch.tensor([e + [0] * (maxlen - len(e)) for _, e in inp],
                                     dtype=torch.long, device=device).transpose(0, 1)
                src_len_t = torch.tensor([len(e) for _, e in inp], dtype=torch.long,
                                         device=device)
                lang_t = torch.tensor([meta["lang2id"][m["lang"]] for m in sample],
                                      dtype=torch.long, device=device)
                out = model.generate(src_t, src_len_t, lang_t, cfg.max_tgt_len, 1)
                for i, m in enumerate(sample):
                    ph_ids = out["phonemes"][i].tolist()
                    ph_ids = [x for x in ph_ids if x not in (0, 1, 2)]
                    ph_tokens = phoneme_vocab.decode(ph_ids)
                    ph_str = " ".join(ph_tokens)
                    line = f"    {m['text']} -> phonemes: {ph_str}"
                    for task in ("separated_graphmes", "separated_phonemes", "aligned_phonemes"):
                        counts = counts_from_ids(out[task][i].tolist(), count_codec)
                        vset = None
                        if task in ("separated_phonemes", "aligned_phonemes"):
                            mlang = m["lang"]
                            if syllable_is_char.get(mlang) and mlang in vowel_symbols:
                                vset = set(vowel_symbols[mlang])
                        recon = reconstruct_prediction(
                            tokenizer.tokenize(m["text"]), ph_tokens, counts, task,
                            vowel_set=vset)
                        line += f" | {task}: {recon}"
                    print(line)

        # ---- checkpoint ----
        # best is always persisted (resumable: model + optim + meta). Intermediate
        # epoch checkpoints are gated by save_every (0 => never, only best+last).
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": cfg.to_dict(),
                "meta": meta,
            }, os.path.join(ckpt_dir, "best_model.pt"))
        if cfg.save_every > 0 and (epoch % cfg.save_every == 0 or epoch == cfg.epochs):
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "config": cfg.to_dict(),
                "meta": meta,
            }, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
        # light-weight deploy artifacts (weights only, optional fp16)
        if cfg.save_model_only:
            if improved:
                torch.save(_deploy_state(model, cfg.export_dtype),
                           os.path.join(ckpt_dir, "model_best.pt"))
            torch.save(_deploy_state(model, cfg.export_dtype),
                       os.path.join(ckpt_dir, "model_last.pt"))
        if writer is not None:
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("lr", optim.param_groups[0]["lr"], epoch)

    if writer is not None:
        writer.close()
    print(f"[done] best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
