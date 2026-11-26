# Where Does JPEG Become Hard to Learn? A Stage-wise Evaluation of JPEG Representations for Visual Recognition

Official code for the PRICAI 2026 paper by Changpei Zha, Tomo Miyazaki and Shinichiro Omachi
(Graduate School of Engineering, Tohoku University).

We build MCU-aligned inputs for successive stages of the JPEG pipeline (RGB, DCT, quantized DCT,
RLE events, Huffman events, raw bytes), train a separate ViT-Small from scratch on each on ImageNet-1k,
and study where recognition becomes hard. The steepest drop occurs at byte packing, when event boundaries
are no longer explicit.

## Repository layout

| Path | Content |
|---|---|
| `stagewise_jpeg_vit/` | Training / evaluation code for every JPEG stage (see its README for stage names and tokenizers). |
| `stagewise_jpeg_vit/dct_manip/` | C++/pybind11 extension for JPEG parsing, DCT, event and MCU-byte extraction (needs libjpeg headers). |
| `stagewise_jpeg_vit/tools/` | Statistics, quality-robustness and linear-probe utilities. |
| `scripts/reproduce_paper.sh`, `scripts/eval_sensitivity.sh` | One command per paper row; sensitivity analyses. |
| `results/` | Reference numbers of the original runs. |

## Setup

```bash
pip install -r requirements.txt
# build the JPEG extension (libjpeg headers/library are auto-detected in the conda env, /usr and
# Linux multiarch dirs; otherwise set JPEG_INCLUDE_DIR / JPEG_LIB_DIR)
cd stagewise_jpeg_vit/dct_manip && pip install . && cd ../..
```

## Data

ImageNet-1k (not included) arranged as `train/<wnid>/*.JPEG` and `val/<wnid>/*.JPEG`.
`stagewise_jpeg_vit/assets/` contains the validation index and a gzipped training index
(`indexbase_train.csv.gz`, extracted automatically by `scripts/reproduce_paper.sh`).

## Reproducing the paper

Each row of Table 2 / Table 3 / Table 4 has one command (90 epochs; run on a GPU node):

```bash
DATA_DIR=/path/to/imagenet bash scripts/reproduce_paper.sh <row>
# batch 1024 on 1 GPU needs ~75-88 GB GPU memory; on 24-48 GB cards use GRAD_ACCUM or multi-GPU NUM_GPUS
# (effective batch stays 1024, micro-batch = 1024 / (NUM_GPUS * GRAD_ACCUM)), e.g.:
#   GRAD_ACCUM=8 DATA_DIR=/path/to/imagenet bash scripts/reproduce_paper.sh <row>
#   NUM_GPUS=4   DATA_DIR=/path/to/imagenet bash scripts/reproduce_paper.sh <row>
# rows: dct quant_dct rle_linear huff_linear raw_linear rle_conv huff_conv raw_conv abl_no_dc abl_dpcm abl_amp_only
```

| Paper row | `<row>` | Paper Top-1 |
|---|---|---:|
| DCT | `dct` | 74.48 |
| Quant_DCT | `quant_dct` | 74.34 |
| RLE event | `rle_linear` | 69.06 |
| Huffman event | `huff_linear` | 69.44 |
| Raw byte | `raw_linear` | 48.89 |
| RLE + conv embedding | `rle_conv` | 70.20 |
| Huffman + conv embedding | `huff_conv` | 71.45 |
| Raw byte + conv embedding | `raw_conv` | 57.91 |
| Quant_DCT (no DC) | `abl_no_dc` | 74.70 |
| Quant_DCT (DPCM) | `abl_dpcm` | 74.85 |
| Amplitude_only | `abl_amp_only` | 62.71 |

Common recipe: ViT-Small, 196 tokens, AdamW (lr 1e-3, wd 1e-4; raw-byte rows 3e-3 / 3e-4), batch 1024, 90 epochs, cosine decay after 10K warmup steps,
RandAugment N=2 / magnitude 3, Lmax = 320 for event/byte stages, DCT stages with frequency normalization (statistics shipped in
`stagewise_jpeg_vit/outputs/dct_stats/`; for `abl_dpcm`, generate typed stats with `tools/compute_dct_frequency_stats_typed.py` and pass `TYPED_STATS`). RGB and the DCT rows of [25] in Table 2 are quoted from the cited papers.

Sensitivity analyses (after training):

```bash
DATA_DIR=/path/to/imagenet VAL_ZIP=/path/to/imagenet-val.zip bash scripts/eval_sensitivity.sh quality   # cross-quality (Fig. 5)
DATA_DIR=/path/to/imagenet bash scripts/eval_sensitivity.sh lmax     # Fig. 6 (rle_conv huff_conv raw_conv)
```

`results/` contains the reference numbers of the original runs and their mapping to the paper (see `results/README.md`).

## Smoke tests

```bash
cd stagewise_jpeg_vit && python tests/smoke_stage_tokenizers.py && python tests/smoke_rlc_parser.py
```

## Citation

```bibtex
@inproceedings{zha2026jpeg,
  title     = {Where Does {JPEG} Become Hard to Learn? A Stage-wise Evaluation of {JPEG} Representations for Visual Recognition},
  author    = {Zha, Changpei and Miyazaki, Tomo and Omachi, Shinichiro},
  booktitle = {Pacific Rim International Conference on Artificial Intelligence (PRICAI)},
  year      = {2026},
  note      = {To appear}
}
```

## License

MIT. Derived from the DCT-ViT project by Jeongsoo Park; the additions and modifications for this paper are by Changpei Zha, and the whole repository remains under MIT. See `LICENSE`. Inherits obligations from
RGB-no-more, TorchJPEG and PyTorch-derived code.
