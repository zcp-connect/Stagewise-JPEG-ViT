# Stage-wise JPEG ViT

> **Note:** For the PRICAI 2026 paper reproduction commands, final stage names (`huff_*`, `*_direct`), and reference results, see the repository root [`README.md`](../README.md).

This project studies **stage-wise representation learning along the JPEG codec pipeline** with ViT-style encoders.

The main research question is:

> Which JPEG pipeline stage exposes the most learnable representation geometry for ViT?

The project compares continuous tensor stages, syntax-event stages, and entropy-byte stages under a unified ViT training framework.

## Target JPEG Stages

| Stage | Representation | Data Type | Planned Tokenizer |
|---|---|---|---|
| Pixel RGB | RGB image tensor | fixed-size numeric | standard ViT patch embedding |
| Pixel YCbCr | YCbCr image tensor | fixed-size numeric | standard ViT patch embedding |
| DCT coefficients | DCT block coefficients | fixed-size numeric | frequency/channel projection |
| Quantized DCT | quantized DCT coefficients | fixed-size integer numeric | per-frequency normalization + projection |
| Pre-Huffman events | DC diff + AC run-size-amplitude | variable-length hybrid | symbolic embeddings + amplitude projection |
| Huffman/VLC stream | entropy bitstream | variable-length discrete | bit/byte embedding + local encoder |
| Raw MCU bytes | MCU payload bytes | padded byte sequence | byte embedding + Conv/Transformer pooling |

## Scientific Position

This project separates three different claims:

1. **Numeric tensor stages** should use numeric projection because values have metric meaning.
2. **Entropy byte stages** should use discrete lookup embeddings because byte ids are symbolic codec outputs.
3. **Pre-Huffman RLC/DPCM stages** need hybrid symbolic-numeric tokenization because fields such as `run` and `size` are categorical, while `amplitude` and `DC diff` are numeric.

The goal is not simply to beat RGB, but to understand how JPEG pipeline stage choice changes representation learning.

## Current Entry Points

The training pipeline now accepts a stage selector:

```bash
python train.py \
  --model_arch=vits \
  --stage=rlc_dcac \
  --datapath=/path/to/imagenet1k/imagenet1k \
  --temp_datapath=/path/to/imagenet1k/imagenet1k \
  --indexpaths=<train_index.csv>,assets/indexbase_val.csv \
  --num_gpus=1 \
  --num_cpus=16 \
  --train --eval
```

Supported `--stage` values:

| Stage | Input shape |
|---|---|
| `rgb` | `[B,3,224,224]` |
| `ycbcr` | `[B,3,224,224]` |
| `dct_float` | `[B,196,384]` |
| `quant_dct` | `[B,196,384]` |
| `raw_byte` | `(payload, mask)`, each `[B,196,320]` |
| `bitstream` | `(payload, mask)`, each `[B,196,320]` |
| `rlc_dc` | 8 event fields, each `[B,196,384]` |
| `rlc_dcac` | 8 event fields, each `[B,196,384]` |

For `raw_byte`, `--embed_type` means:

| Value | Meaning |
|---:|---|
| 1 | raw byte float linear |
| 2 | discrete byte mean pooling |
| 3 | discrete byte Conv1d pooling |
| 4 | bit-vector Conv1d pooling |

For pilot runs, use `--epochs=30 --scheduler_epochs=90` to run 30 epochs while keeping the formal 90-epoch cosine horizon.

## Smoke Tests

```bash
python tests/smoke_stage_dataset_helpers.py
python tests/smoke_rlc_parser.py
python tests/smoke_stage_tokenizers.py
```

## License and Acknowledgements

This code is derived from the MIT-licensed DCT-ViT project by Jeongsoo Park (see `LICENSE`), and inherits obligations from RGB-no-more, TorchJPEG and PyTorch-derived code.

## Key Files

| File | Role |
|---|---|
| `train.py` | distributed training entrypoint |
| `datasets.py` | dataset and transform entrypoint |
| `models/plainvit.py` | ViT and current MCU-byte tokenizers |
| `dct_manip/` | modified JPEG/DCT/MCU extraction extension |
| `utils/configs.py` | experiment configuration defaults |
