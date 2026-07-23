# G2P Multi-Task Trainer

Grapheme-to-Phoneme (G2P) multi-task training & inference toolkit.

A single model maps a **grapheme sequence** (possibly with a **language embedding**)
to **four** parallel targets:

| # | Output | Description |
|---|--------|-------------|
| 1 | `phonemes`          | phoneme sequence, **no** separators |
| 2 | `separated_graphmes`| grapheme sequence separated by `\|`  |
| 3 | `separated_phonemes`| phoneme sequence separated by `\|`   |
| 4 | `aligned_phonemes`  | phoneme sequence separated by `/`    |

## Highlights

- **Multi-task**: one encoder / four decoders trained jointly.
- **CUDA training** with optional mixed precision (AMP).
- **Custom phoneme set**: pass your own phoneme inventory; vocabularies are built dynamically.
- **Dimension explosion control** for huge grapheme sets:
  - Latin / space-separated text is **sub-word tokenized with BPE** (so the input
    vocabulary stays small even for millions of orthographic forms).
  - CJK-like scripts are kept as **per-syllable characters** (already a small,
    closed set) — no BPE needed.
- **Single ONNX export** with exactly **4 named output nodes** for production serving.
- **Language embedding**: the language label is parsed from the dataset file name
  (e.g. `dataset-ko.csv` → language `ko`). At inference the model takes
  `(grapheme sequence, language id)` and produces the target language's G2P.
- **Script-aware preprocessing**:
  - Latin-family input (`hello fantastic world`) → split on spaces, then BPE.
  - CJK-family input (`안녕하세요`, `你好谢谢`) → split into independent syllables/characters.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# optional, for ONNX export:
pip install onnx onnxruntime
```

## Data format

Drop one or more CSV files into a data directory. The file name encodes the
language: `dataset-<lang>.csv` (the part after the **last** `-` and before `.csv`).
Example: `data/dataset-ko.csv` → language `ko`.

Each CSV must carry a **header**. The loader matches the following columns by
name (case-insensitively, so the real-world `graphmes` typo is accepted):

| CSV column           | Role                                  | Maps to target       |
|----------------------|---------------------------------------|----------------------|
| `graphmes`           | source grapheme sequence              | (model input)        |
| `phonemes`           | space-separated phonemes              | `phonemes` (no `\|`) |
| `separated_graphmes` | grapheme groups joined by `\|`        | `separated_graphmes` |
| `separated_phonemes` | phoneme groups joined by `\|`         | `separated_phonemes` |
| `aligned_phonemes`   | phoneme groups joined by `/`          | `aligned_phonemes`   |

Example (`data/dataset-ko.csv`):

```csv
graphmes,phonemes,separated_graphmes,separated_phonemes,aligned_phonemes
ㄴㅐㄱㅏ,n e g a,ㄴㅐ|ㄱㅏ,n e|g a,n/e g/a
내가,nega,내|가,ne|ga,n/e|ga
```

The four targets are built **directly from these columns** (see
`CSV_COLUMNS` in `src/preprocessing.py`):

- `phonemes`          = phonemes with separators removed (no `|`)
- `separated_graphmes`= grapheme groups joined with `|`
- `separated_phonemes`= phoneme groups joined with `|`
- `aligned_phonemes`  = phoneme groups joined with `/`

Because space, `|` and `/` are all just separators of varying granularity, the
targets share content by construction:

- `separated_graphmes` (drop `|`) == `graphmes`
- `separated_phonemes` (drop `|`) == `phonemes`
- `aligned_phonemes`   (drop `/`) == `phonemes`

Training adds lightweight **content-consistency losses** (weights
`consistency_weight` / `grapheme_consistency_weight` in `configs/base.yaml`) that
pull the separator decoders towards this invariant, helping the model learn the
grapheme/phoneme association across tasks.

For very large grapheme inventories the **source** vocab is still BPE-compressed
(see below); the target vocabularies are built from the symbols actually seen in
these columns.

## Train

```bash
python src/train.py --config configs/base.yaml
```

Or with defaults (no config file):

```bash
python src/train.py --data_dir data --output_dir checkpoints
```

Resume / override:

```bash
python src/train.py --config configs/base.yaml --resume checkpoints/ckpt_best.pt --epochs 50
```

Key flags: `--batch_size`, `--lr`, `--epochs`, `--embed_dim`, `--enc_layers`,
`--dec_layers`, `--bpe_merges`, `--fp16`, `--device cuda`.

## Export to ONNX

```bash
python src/export_onnx.py --checkpoint checkpoints/ckpt_best.pt \
    --output g2p_multitask.onnx --opset 17
```

The exported graph has inputs `graphemes` (int64 [B, S]), `langs` (int64 [B])
and outputs `phonemes`, `separated_graphmes`, `separated_phonemes`, `aligned_phonemes`
(each float [B, T, V] logits).

## Inference

PyTorch:

```bash
python src/inference.py --checkpoint checkpoints/ckpt_best.pt \
    --lang ko --text "안녕하세요"
```

ONNX (faster, no PyTorch needed at serving time):

```bash
python src/inference.py --onnx g2p_multitask.onnx --lang ko --text "안녕하세요"
```

Batch from a file (one sentence per line):

```bash
python src/inference.py --onnx g2p_multitask.onnx --lang en --input-file phrases.txt
```

## Repository layout

```
configs/            YAML configs
data/               sample multi-language datasets
src/
  config.py         config dataclass + argparse merging
  preprocessing.py  script detection, tokenization, BPE, vocab
  data.py           dataset discovery, lang parsing, 4-target building
  model.py          encoder (BiLSTM + lang embed) + multi-task decoders
  train.py          training loop (CUDA, AMP, checkpointing)
  export_onnx.py    single-model 4-output ONNX export
  inference.py      CLI for PyTorch / ONNX inference
  utils.py          tokenization helpers, vocab IO, greedy decode
```

## Notes on the vocabularies

- **Grapheme (source) vocab**: a BPE-trained sub-word vocabulary. Latin words are
  merged; CJK characters are single units. Because BPE keeps the vocabulary at a
  fixed, small size (e.g. 2000–8000), we never explode to one embedding per
  orthographic word.
- **Phoneme (target) vocab**: built from the unique phoneme symbols you provide /
  observe. Fully customizable — point `--phoneme_set` at a file with one symbol
  per line, or let it be discovered from data.
