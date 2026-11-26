#!/usr/bin/env bash
# Train + evaluate one row of the paper's Table 2 / Table 3 / Table 4 (single node, ~hours per row; run on a GPU node).
#   DATA_DIR=/path/to/imagenet bash scripts/reproduce_paper.sh <row>
# Rows (paper numbers = ImageNet-1k val Top-1):
#   dct             Table 2  DCT                    74.48
#   quant_dct       Table 2  Quant_DCT              74.34
#   rle_linear      Table 2  RLE event              69.06
#   huff_linear     Table 2  Huffman event          69.44
#   raw_linear      Table 2  Raw byte               48.89
#   rle_conv        Table 3  RLE + conv embedding   70.20
#   huff_conv       Table 3  Huffman + conv emb.    71.45
#   raw_conv        Table 3  Raw byte + conv emb.   57.91
#   abl_no_dc       Table 4  Quant_DCT (no DC)      74.70
#   abl_dpcm        Table 4  Quant_DCT (DPCM)       74.85   (needs TYPED_STATS from tools/compute_dct_frequency_stats_typed.py)
#   abl_amp_only    Table 4  Amplitude_only         62.71
#   rle_2d/huff_2d  Auxiliary 2D-pooled event baselines (for eval_sensitivity.sh lmax)
# Optional env: SAVE_DIR (default ./models), NUM_GPUS (1), NUM_CPUS (16), INDEX (train,val index csv),
#   GRAD_ACCUM (default 1). Effective batch size is always 1024 (micro-batch = 1024 / (NUM_GPUS * GRAD_ACCUM)).
#   At NUM_GPUS=1 GRAD_ACCUM=1, batch=1024 needs ~75-88 GB GPU memory (GH200-class);
#   on 24-48 GB cards use e.g. GRAD_ACCUM=4 or 8 (single GPU) or NUM_GPUS=4 (multi-GPU DDP).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${DATA_DIR:?set DATA_DIR to the ImageNet-1k root (train/ and val/)}"
ROW="${1:?usage: reproduce_paper.sh <row>}"
ROOT="$(cd "$(dirname "$0")/../stagewise_jpeg_vit" && pwd)"
INDEX="${INDEX:-assets/indexbase_train.csv,assets/indexbase_val.csv}"
SAVE_DIR="${SAVE_DIR:-$ROOT/models}"; mkdir -p "$SAVE_DIR"
cd "$ROOT"
[ -f assets/indexbase_train.csv ] || gunzip -k assets/indexbase_train.csv.gz

STATS=outputs/dct_stats/imagenet1k_train_dct_freq_stats.pt   # shipped; regenerate with tools/compute_dct_frequency_stats.py
LR=1e-3; WD=1e-4; AMP=(--amp 1 --ampdtype bf16); EXTRA=()
case "$ROW" in
  dct)         STAGE=dct_float;  AMP=(); EXTRA=(--freqnorm --freqnorm_stats "$STATS") ;;
  quant_dct)   STAGE=quant_dct;  AMP=(); EXTRA=(--freqnorm --freqnorm_stats "$STATS") ;;
  rle_linear)  STAGE=rlc_dcac_flat_direct;  EXTRA=(--rlc_lmax 320) ;;
  huff_linear) STAGE=huff_dcac_flat_direct; EXTRA=(--rlc_lmax 320) ;;
  rle_2d)      STAGE=rlc_dcac_2d_direct;  EXTRA=(--rlc_lmax 320) ;;
  huff_2d)     STAGE=huff_dcac_2d_direct; EXTRA=(--rlc_lmax 320) ;;
  rle_conv)    STAGE=rlc_dcac_2d_cp_concat_conv_direct;  EXTRA=(--rlc_lmax 320 --numeric_feature_dim 128) ;;
  huff_conv)   STAGE=huff_dcac_2d_cp_concat_conv_direct; EXTRA=(--rlc_lmax 320 --numeric_feature_dim 128) ;;
  raw_linear)  STAGE=raw_byte; AMP=(); LR=3e-3; WD=3e-4; EXTRA=(--embed_type 1 --rlc_lmax 320) ;;
  raw_conv)    STAGE=raw_byte; AMP=(); LR=3e-3; WD=3e-4; EXTRA=(--embed_type 3 --rlc_lmax 320) ;;
  abl_no_dc)   STAGE=ac_only;    AMP=(); EXTRA=(--freqnorm --freqnorm_stats "$STATS") ;;
  abl_dpcm)    STAGE=dc_diff_ac; AMP=(); EXTRA=(--freqnorm --freqnorm_stats "${TYPED_STATS:?run tools/compute_dct_frequency_stats_typed.py and set TYPED_STATS to the resulting .pt}") ;;
  abl_amp_only) STAGE=event_amp_only_flat_direct; EXTRA=(--rlc_lmax 320) ;;
  *) echo "unknown row: $ROW" >&2; exit 2 ;;
esac

python train.py --train --eval \
  --model_arch vits --stage "$STAGE" --classes 1000 \
  --datapath "$DATA_DIR" --temp_datapath "$DATA_DIR" --indexpaths "$INDEX" \
  --num_gpus "${NUM_GPUS:-1}" --num_cpus "${NUM_CPUS:-16}" \
  --batch 1024 --grad_accum_steps "${GRAD_ACCUM:-1}" --epochs 90 --scheduler_epochs 90 \
  --lr "$LR" --wd "$WD" --warmup_steps 10000 \
  --num_ops 2 --ops_magnitude 3 \
  --no_extract --no_resize \
  "${AMP[@]}" "${EXTRA[@]}" \
  --savepath "$SAVE_DIR/${ROW}.pth" --verbose 1
