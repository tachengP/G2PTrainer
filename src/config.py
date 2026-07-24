"""Configuration: YAML defaults + CLI overrides."""

from __future__ import annotations

import argparse
import copy
import typing
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    # data
    data_dir: str = "data"
    output_dir: str = "checkpoints"
    model_name: str = "default"        # sub-directory under output_dir for this run's artifacts
    # Language definitions: each entry is {id, syllable_is_char}.  `id` selects the
    # dataset file `dataset-{id}.csv`; `syllable_is_char` flags CJK-like languages
    # where one character == one independent syllable (so each phoneme group must
    # contain exactly one vowel nucleus).  Supersedes the legacy `file_glob`.
    lang_define: Optional[List[Dict[str, Any]]] = None
    file_glob: str = "dataset-*.csv"   # legacy fallback when lang_define is empty
    # Optional explicit vowel-phoneme sets per language (list of phoneme symbols).
    # When omitted, vowels are derived from the training data by confidence.
    vowel_phonemes: Optional[Dict[str, List[str]]] = None
    # Diagnostic switch (NOT added to the loss): when > 0, the training loop logs
    # the fraction of predicted count-groups that violate the one-vowel-per-syllable
    # rule -- a direct read on whether the separator learning is improving.
    syllable_constraint_weight: float = 0.5
    phoneme_set: Optional[str] = None
    max_src_len: int = 120
    max_tgt_len: int = 120
    val_split: float = 0.05
    seed: int = 42

    # binarize: preprocess CSVs into compact mmap-able numpy binaries so the
    # training loop streams rows from disk (no host-RAM blow-up at train start).
    binarize: bool = True                 # auto-binarize before training if missing/stale
    binarize_workers: int = 8             # parallel encode workers during the binarize pass
    binary_dir: Optional[str] = None      # null -> {data_dir}/binary

    # vocabulary / subword
    bpe_merges: int = 3000
    min_freq: int = 1
    max_samples: Optional[int] = None  # cap rows read per dataset (None = all)

    # model
    embed_dim: int = 256
    enc_layers: int = 3
    dec_layers: int = 2
    enc_heads: int = 4
    dec_hidden: int = 256
    ffn_dim: int = 512
    dropout: float = 0.1
    lang_embed_dim: int = 16
    max_langs: int = 64

    # training
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    lr_decay_gamma: float = 0.8       # per-epoch lr multiplier; 1.0 = no decay, 0.8 = decay to 80% each epoch
    fp16: bool = False
    sort_by_length: bool = True       # group similar-length samples per batch so padding (and thus BiLSTM + attention cost) stops exploding with batch size
    device: str = "auto"
    num_workers: int = 2
    log_every: int = 50
    save_every: int = 500

    # derived tasks are encoded as *count* sequences and regrouped at inference,
    # so no cross-task consistency supervision is needed.

    # tensorboard monitoring
    tensorboard: bool = True
    tb_log_dir: Optional[str] = None          # null -> checkpoints/{model_name}/tb-logs
    num_fixed_samples: int = 5                # fixed val samples tracked every epoch / at save
    fixed_samples_seed: int = 1337            # deterministic selection of the fixed samples

    # resume
    resume: Optional[str] = None

    # deployment artifact (kept small: model weights only, optional fp16)
    save_model_only: bool = True    # also write model-only weights (no optimizer) for deploy
    export_dtype: str = "fp32"       # dtype of the model-only file: fp32 | fp16

    # data residency
    data_on_gpu: bool = False        # keep the dataset resident on CUDA (num_workers=0), freeing host RAM

    force_rebinarize: bool = False   # ignore existing binary dir and rebuild from CSV

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def _dataclass_defaults() -> Dict[str, Any]:
    return asdict(Config())


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = _dataclass_defaults()
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if k in cfg})
    return cfg


def _cli_type(key: str, default):
    """Best-effort CLI type for a config field.

    ``type(None)`` (Optional fields whose default is None) cannot be used by
    argparse, so we recover the real underlying type from the dataclass
    annotation (e.g. ``max_samples: Optional[int]`` -> ``int``).
    """
    if default is not None:
        return type(default)
    hints = typing.get_type_hints(Config)
    hint = hints.get(key)
    if hint is not None:
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        if non_none:
            return non_none[0]
    return str


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G2P multi-task trainer")
    p.add_argument("--config", type=str, default=None, help="path to YAML config")
    # allow overriding any top-level field via CLI
    for key, val in _dataclass_defaults().items():
        if isinstance(val, bool):
            # nargs="?" + const=True so `--flag` alone sets True; `--flag false`
            # sets False; omitted leaves None (config default is respected).
            p.add_argument(f"--{key}", type=_str2bool, nargs="?", const=True, default=None)
        else:
            p.add_argument(f"--{key}", type=_cli_type(key, val), default=None)
    return p.parse_args(argv)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    return str(v).lower() in ("1", "true", "yes", "y")


def build_config(argv: Optional[list] = None) -> Config:
    args = parse_args(argv)
    cfg_dict = load_config(args.config)
    for key in cfg_dict:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg_dict[key] = cli_val
    if args.resume is not None:
        cfg_dict["resume"] = args.resume
    return Config(**cfg_dict)


def resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
