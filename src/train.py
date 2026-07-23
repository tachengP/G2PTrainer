"""Training loop: CUDA-aware, optional mixed precision, checkpointing."""

from __future__ import annotations

import os
import sys
import time
import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import build_config, resolve_device
from src import data as data_mod
from src import preprocessing as pp
from src.model import G2PModel
from src.data import TARGET_NAMES, make_loader
from src.utils import greedy_decode
from tqdm import tqdm
from src.consistency import phoneme_consistency, grapheme_consistency


_SPECIAL_TOKENS = {"<pad>", "<sos>", "<eos>", "<unk>"}


def _decode_pred(indices, vocab, sep="") -> str:
    """Decode model output ids to a string, dropping padding/SOS/EOS/UNK.

    ``sep`` joins the decoded units -- the bare ``phonemes`` task has no in-string
    separator token (its vocab is whitespace-split), so we re-insert a space to
    match the CSV; ``separated_*`` / ``aligned_*`` already carry ``|`` / ``/``.
    """
    syms = vocab.decode(indices)  # List[str]
    return sep.join(s for s in syms if s not in _SPECIAL_TOKENS)


@torch.no_grad()
def _infer_fixed_samples(model, fixed_samples, tokenizer, src_vocab, phoneme_vocab,
                         grapheme_tgt_vocab, lang2id, device, max_tgt, sos_idx=1):
    """Run the model on the fixed val samples; return a list of result dicts.

    Each result is ``{"input", "lang", "pred": {target_name: decoded_str}}``.
    """
    results: List[Dict] = []
    for fs in fixed_samples:
        units = tokenizer.tokenize(fs["text"])
        ids = src_vocab.encode(units) if units else []
        if not ids:
            continue
        src = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [S,1]
        src_len = torch.tensor([len(ids)], dtype=torch.long, device=device)
        lang = torch.tensor([lang2id[fs["lang"]]], dtype=torch.long, device=device)
        logits = model.generate(
            src, src_len, lang, max_tgt, sos_idx,
            eos_idx=phoneme_vocab.eos_idx, early_stop=True,
        )  # {k:[T,1,V]}
        pred: Dict[str, str] = {}
        for k, lg in logits.items():
            lg = lg.permute(1, 0, 2)  # [1,T,V] -> greedy_decode expects batch-first
            vocab = grapheme_tgt_vocab if k == "separated_graphmes" else phoneme_vocab
            toks = greedy_decode(lg, vocab.eos_idx, vocab.pad_idx)[0]
            pred[k] = _decode_pred(toks, vocab, sep=" " if k == "phonemes" else "")
        results.append({"input": fs["text"], "lang": fs["lang"], "pred": pred})
    return results


def _render_fixed_text(results: List[Dict]) -> str:
    lines: List[str] = []
    for i, r in enumerate(results):
        lines.append(f"[{i}] lang={r['lang']}  in : {r['input']}")
        for k in TARGET_NAMES:
            lines.append(f"     {k}: {r['pred'][k]}")
    return "\n".join(lines)


def _write_fixed_predictions(path: str, results: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render_fixed_text(results) + "\n")


def main():
    cfg = build_config()
    device = resolve_device(cfg.device)
    print(f"[info] device = {device}")

    # All artifacts for this run live under checkpoints/{model_name}
    run_dir = os.path.join(cfg.output_dir, cfg.model_name)
    os.makedirs(run_dir, exist_ok=True)

    # ----------------------------------------------------------------- build dataset
    if cfg.binarize:
        from src.binarize import ensure_binary, load_binary
        binary_dir = ensure_binary(cfg)
        (train_ds, val_ds, src_vocab, phoneme_vocab, grapheme_tgt_vocab,
         tokenizer, pad_idx_dict, meta, val_monitor) = load_binary(binary_dir, device)
        use_binary = True
        print(f"[info] loaded binarized dataset from {binary_dir}")
    else:
        (train_ds, val_ds, src_vocab, phoneme_vocab, grapheme_tgt_vocab,
         tokenizer, pad_idx_dict, meta, val_monitor) = data_mod.build_dataset(
            data_dir=cfg.data_dir,
            file_glob=cfg.file_glob,
            bpe_merges=cfg.bpe_merges,
            min_freq=cfg.min_freq,
            max_src_len=cfg.max_src_len,
            max_tgt_len=cfg.max_tgt_len,
            phoneme_set=cfg.phoneme_set,
            val_split=cfg.val_split,
            seed=cfg.seed,
            max_samples=cfg.max_samples,
        )
        use_binary = False
    print(f"[info] langs={meta['langs']} src_vocab={meta['src_vocab_size']} "
          f"phoneme_vocab={meta['phoneme_vocab_size']} "
          f"grapheme_tgt_vocab={meta['grapheme_tgt_vocab_size']} "
          f"train={len(train_ds)} val={len(val_ds)}")

    # Data residency:
    #  * binarize=True + data_on_gpu=False (default): streaming mmap reader with
    #    num_workers prefetch -> host RAM stays flat AND the GPU is kept fed.
    #  * binarize=True + data_on_gpu=True: whole dataset resident on CUDA
    #    (num_workers=0).
    #  * binarize=False + data_on_gpu=True: whole dataset resident on CUDA
    #    (num_workers=0).
    #  * binarize=False + default: dataset on CPU in shared memory so the
    #    (num_workers) subprocesses map ONE physical copy (full preprocessing par.).
    if use_binary:
        if cfg.data_on_gpu and device.startswith("cuda"):
            # Whole dataset resident on CUDA: no parallel workers needed.
            loader_num_workers = 0
            loader_pin_memory = False
            train_ds = train_ds.to_device(device)
            val_ds = val_ds.to_device(device)
            print(f"[info] binary dataset resident on {device} (num_workers=0)")
        else:
            # Prefetch batches in parallel worker processes so the GPU is never
            # starved between steps. Each worker opens its own mmap (lazy) and the
            # H2D copy is async via pinned memory + non_blocking.
            loader_num_workers = cfg.num_workers
            loader_pin_memory = device.startswith("cuda")
            print(f"[info] binary dataset prefetched via {loader_num_workers} workers "
                  f"(pin_memory={loader_pin_memory})")
    elif cfg.data_on_gpu and device.startswith("cuda"):
        train_ds = train_ds.to_device(device)
        val_ds = val_ds.to_device(device)
        loader_num_workers = 0
        loader_pin_memory = False
        print(f"[info] dataset resident on {device} (num_workers=0)")
    else:
        train_ds.share_memory()
        val_ds.share_memory()
        loader_num_workers = cfg.num_workers
        loader_pin_memory = True

    # ---- fixed val samples for monitoring (deterministic selection) ----
    fixed_samples: List[Dict] = []
    if cfg.num_fixed_samples > 0 and len(val_monitor) > 0:
        frng = random.Random(cfg.fixed_samples_seed)
        k = min(cfg.num_fixed_samples, len(val_monitor))
        fidx = sorted(frng.sample(range(len(val_monitor)), k))
        for i in fidx:
            fixed_samples.append({"text": val_monitor[i]["text"],
                                  "lang": val_monitor[i]["lang"]})
    print(f"[info] fixed monitoring samples = {len(fixed_samples)} "
          f"(seed={cfg.fixed_samples_seed})")

    # ---- TensorBoard writer ----
    writer = None
    if cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        # tensorboard logs go under checkpoints/{model_name}/tb-logs
        tb_dir = cfg.tb_log_dir or os.path.join(run_dir, "tb-logs")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[info] tensorboard -> {tb_dir}")

    sep_ids = (
        phoneme_vocab.stoi("|"),
        phoneme_vocab.stoi("/"),
        grapheme_tgt_vocab.stoi("|"),
    )

    model = G2PModel(
        src_vocab_size=meta["src_vocab_size"],
        phoneme_vocab_size=meta["phoneme_vocab_size"],
        grapheme_tgt_vocab_size=meta["grapheme_tgt_vocab_size"],
        num_langs=meta["num_langs"],
        embed_dim=cfg.embed_dim,
        enc_layers=cfg.enc_layers,
        dec_layers=cfg.dec_layers,
        enc_heads=cfg.enc_heads,
        dec_hidden=cfg.dec_hidden,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        lang_embed_dim=cfg.lang_embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler = GradScaler(enabled=(cfg.fp16 and device == "cuda"))
    crit = nn.CrossEntropyLoss(ignore_index=pad_idx_dict["phonemes"])

    start_epoch = 0
    best_val = float("inf")
    if cfg.resume:
        resume_path = cfg.resume
        # allow a bare checkpoint filename to resolve inside run_dir
        if not os.path.isabs(resume_path) and not os.path.exists(resume_path):
            resume_path = os.path.join(run_dir, resume_path)
        print(f"[info] resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", float("inf"))

    # per-epoch lr decay (e.g. 0.8 -> lr at epoch e is lr0 * 0.8**e). Created after a
    # possible resume so its base lr reflects the already-decayed lr from the checkpoint.
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay_gamma)

    train_loader = make_loader(train_ds, cfg.batch_size, pad_idx_dict, loader_num_workers, True, loader_pin_memory, cfg.sort_by_length)
    val_loader = make_loader(val_ds, cfg.batch_size, pad_idx_dict, loader_num_workers, False, loader_pin_memory, cfg.sort_by_length)

    # persist artifacts needed for inference / export
    src_vocab.to_file(os.path.join(run_dir, "src_vocab.txt"))
    phoneme_vocab.to_file(os.path.join(run_dir, "phoneme_vocab.txt"))
    grapheme_tgt_vocab.to_file(os.path.join(run_dir, "grapheme_tgt_vocab.txt"))
    pp.save_bpe(tokenizer, os.path.join(run_dir, "bpe.txt"))
    _save_meta(run_dir, meta, cfg)

    use_amp = cfg.fp16 and device == "cuda"
    global_step = 0
    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        model.train()
        running = torch.zeros((), device=device)
        running_cons = torch.zeros((), device=device)
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", unit="batch")
        for step, (src, src_len, lang, targets, tgt_lens) in enumerate(pbar):
            src = src.to(device, non_blocking=True)
            src_len = src_len.to(device, non_blocking=True)
            lang = lang.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            tgt_lens = {k: v.to(device, non_blocking=True) for k, v in tgt_lens.items()}

            with autocast(device_type=device, enabled=use_amp):
                task_loss, cons_loss = _task_and_consistency_loss(
                    model, src, src_len, lang, targets, tgt_lens, cfg, crit,
                    phoneme_vocab, grapheme_tgt_vocab, pad_idx_dict, sep_ids,
                )
                loss = task_loss + cons_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # accumulate detached tensors (NO per-step .item() -> no forced CUDA
            # sync every step, which would otherwise stall the pipeline and create
            # the "GPU idle between batches" sawtooth).
            running = running + loss.detach()
            running_cons = running_cons + cons_loss.detach()
            n_batches += 1
            global_step += 1
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                loss_val = loss.detach().item()
                cons_val = cons_loss.detach().item()
                pbar.set_postfix(loss=f"{loss_val:.3f}", cons=f"{cons_val:.3f}")
                print(f"  epoch {epoch+1} step {step+1} loss={loss_val:.4f} "
                      f"cons={cons_val:.4f}")
                # transient loss shown in TensorBoard at the per-step granularity
                if writer is not None:
                    writer.add_scalar("loss/step_task", task_loss.detach().item(), global_step)
                    writer.add_scalar("loss/step_total", loss_val, global_step)
                    writer.add_scalar("loss/step_cons", cons_val, global_step)

        # validation
        val_task, val_cons = _evaluate(
            model, val_loader, crit, device, use_amp, pad_idx_dict,
            cfg, phoneme_vocab, grapheme_tgt_vocab, sep_ids,
        )
        dt = time.time() - t0
        train_loss_avg = (running / max(n_batches, 1)).item()
        train_cons_avg = (running_cons / max(n_batches, 1)).item()
        print(f"[epoch {epoch+1}/{cfg.epochs}] train_loss={train_loss_avg:.4f} "
              f"train_cons={train_cons_avg:.4f} "
              f"val_loss={val_task:.4f} val_cons={val_cons:.4f} time={dt:.1f}s")

        # ---- per-epoch lr decay ----
        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  -> lr = {cur_lr:.3e}")
        if writer is not None:
            writer.add_scalar("lr/epoch", cur_lr, epoch)

        # ---- TensorBoard scalars (loss / consistency) ----
        if writer is not None:
            writer.add_scalar("loss/train", train_loss_avg, epoch)
            writer.add_scalar("loss/cons_train", train_cons_avg, epoch)
            writer.add_scalar("loss/val", val_task, epoch)
            writer.add_scalar("loss/cons_val", val_cons, epoch)

        # ---- fixed-sample monitoring: every epoch, also dumped at save ----
        fixed_results: List[Dict] = []
        if fixed_samples:
            fixed_results = _infer_fixed_samples(
                model, fixed_samples, tokenizer, src_vocab, phoneme_vocab,
                grapheme_tgt_vocab, meta["lang2id"], device, cfg.max_tgt_len,
            )
            if writer is not None:
                writer.add_text("fixed_samples", _render_fixed_text(fixed_results), epoch)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_task,
            "val_cons": val_cons,
            "config": cfg.to_dict(),
            "meta": meta,
        }
        torch.save(ckpt, os.path.join(run_dir, "ckpt_last.pt"))
        # synchronize the fixed samples' grapheme -> model predictions with the pt
        if fixed_results:
            _write_fixed_predictions(
                os.path.join(run_dir, "fixed_samples_predictions.txt"), fixed_results
            )
        if val_task < best_val:
            best_val = val_task
            torch.save(ckpt, os.path.join(run_dir, "ckpt_best.pt"))
            print(f"  -> saved ckpt_best.pt (val_loss={val_task:.4f} val_cons={val_cons:.4f})")

        # deployment artifact: model weights only (no optimizer) -> stays small
        if cfg.save_model_only:
            _save_model_only(os.path.join(run_dir, "model_last.pt"), model, cfg.export_dtype)
            if val_task < best_val:  # this epoch is a new best -> refresh model_best
                _save_model_only(os.path.join(run_dir, "model_best.pt"), model, cfg.export_dtype)

    if writer is not None:
        writer.close()

    print("[done] training complete. Best checkpoint at",
          os.path.join(run_dir, "ckpt_best.pt"))


def _task_and_consistency_loss(model, src, src_len, lang, targets, tgt_lens,
                                cfg, crit, phoneme_vocab, grapheme_tgt_vocab,
                                pad_idx_dict, sep_ids):
    """Return ``(task_loss, cons_loss)`` on already device-moved tensors."""
    logits = model(src, src_len, lang, targets, tgt_lens)
    task_loss = torch.zeros((), device=src.device)
    for name in TARGET_NAMES:
        lg = logits[name].permute(1, 0, 2)  # [B, T, V]
        tgt = targets[name]                # [B, T]
        # decoder at position i predicts tgt[i+1] -> shift target
        task_loss = task_loss + crit(
            lg[:, :-1].reshape(-1, lg.size(-1)),
            tgt[:, 1:].reshape(-1),
        )

    cons_loss = torch.zeros((), device=src.device)
    if cfg.consistency_weight or cfg.grapheme_consistency_weight:
        pipe_id, slash_id, graph_id = sep_ids
        eos_ph = phoneme_vocab.eos_idx
        eos_gr = grapheme_tgt_vocab.eos_idx
        c_ph = (
            phoneme_consistency(logits["phonemes"], logits["separated_phonemes"],
                                targets["separated_phonemes"], pipe_id,
                                pad_idx_dict["phonemes"], eos_ph)
            + phoneme_consistency(logits["phonemes"], logits["aligned_phonemes"],
                                  targets["aligned_phonemes"], slash_id,
                                  pad_idx_dict["phonemes"], eos_ph)
        ) * 0.5
        c_gr = grapheme_consistency(
            logits["separated_graphmes"], targets["separated_graphmes"], graph_id,
            pad_idx_dict["separated_graphmes"], len(grapheme_tgt_vocab), eos_gr,
        )
        cons_loss = cfg.consistency_weight * c_ph + cfg.grapheme_consistency_weight * c_gr
    return task_loss, cons_loss


@torch.no_grad()
def _evaluate(model, loader, crit, device, use_amp, pad_idx_dict, cfg,
              phoneme_vocab, grapheme_tgt_vocab, sep_ids):
    model.eval()
    total = torch.zeros((), device=device)
    cons_total = torch.zeros((), device=device)
    n = 0
    for src, src_len, lang, targets, tgt_lens in tqdm(loader, desc="eval", unit="batch", leave=False):
        src = src.to(device, non_blocking=True)
        src_len = src_len.to(device, non_blocking=True)
        lang = lang.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
        tgt_lens = {k: v.to(device, non_blocking=True) for k, v in tgt_lens.items()}
        with autocast(device_type=device, enabled=use_amp):
            task_loss, cons_loss = _task_and_consistency_loss(
                model, src, src_len, lang, targets, tgt_lens, cfg, crit,
                phoneme_vocab, grapheme_tgt_vocab, pad_idx_dict, sep_ids,
            )
        total = total + task_loss.detach()
        cons_total = cons_total + cons_loss.detach()
        n += 1
    return (total / max(n, 1)).item(), (cons_total / max(n, 1)).item()


def _save_model_only(path: str, model, dtype: str) -> None:
    """Write a deployment checkpoint with model weights only (no optimizer).

    Kept tiny so it fits comfortably under typical size budgets (e.g. 20 MB):
    fp32 ~ params*4 bytes, fp16 ~ params*2 bytes.  ``inference.py`` / ONNX
    export consume this together with the vocab files already in ``run_dir``.
    """
    sd = model.state_dict()
    if dtype == "fp16":
        sd = {k: v.half() for k, v in sd.items()}
    torch.save({"model": sd, "dtype": dtype}, path)


def _save_meta(output_dir, meta, cfg):
    import json
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                # lang / length info
                "lang2id": meta["lang2id"],
                "id2lang": {str(k): v for k, v in meta["id2lang"].items()},
                "max_src_len": cfg.max_src_len, "max_tgt_len": cfg.max_tgt_len,
                "lang_embed_dim": cfg.lang_embed_dim,
                # vocab sizes (needed to (re)build the model from this file alone)
                "src_vocab_size": meta["src_vocab_size"],
                "phoneme_vocab_size": meta["phoneme_vocab_size"],
                "grapheme_tgt_vocab_size": meta["grapheme_tgt_vocab_size"],
                "num_langs": meta["num_langs"],
                # architecture (lets inference/export rebuild without the full ckpt)
                "embed_dim": cfg.embed_dim, "enc_layers": cfg.enc_layers,
                "dec_layers": cfg.dec_layers, "enc_heads": cfg.enc_heads,
                "dec_hidden": cfg.dec_hidden, "ffn_dim": cfg.ffn_dim,
            },
            f, ensure_ascii=False, indent=2,
        )
    cfg.save(os.path.join(output_dir, "config.yaml"))


if __name__ == "__main__":
    main()
