#!/usr/bin/env bash
# Sensitivity analyses of the paper (run on a GPU node, after training the rows with reproduce_paper.sh).
#   quality : cross-quality generalization (q100..q10 re-encodings of the ImageNet val set)
#   lmax    : inference-time truncation below Lmax (128/196/256/320)
#   DATA_DIR=/path/to/imagenet [VAL_ZIP=/path/to/imagenet-val.zip] SAVE_DIR=... bash scripts/eval_sensitivity.sh {quality|lmax}
set -euo pipefail
: "${DATA_DIR:?set DATA_DIR}"
ROOT="$(cd "$(dirname "$0")/../stagewise_jpeg_vit" && pwd)"; cd "$ROOT"
SAVE_DIR="${SAVE_DIR:-$ROOT/models}"
case "${1:?quality|lmax}" in
  quality)
    : "${VAL_ZIP:?set VAL_ZIP (zip of the ImageNet val images)}"
    QDIR="${QROBUST_DIR:-$ROOT/imagenet_qrobust}"
    python tools/build_imagenet_val_qrobust.py --zip "$VAL_ZIP" --out "$QDIR"
    STATS=outputs/dct_stats/imagenet1k_train_dct_freq_stats.pt
    ev() { # name stage amp ampdtype extra...
      local name=$1 stage=$2 amp=$3 dt=$4; shift 4
      if [ ! -f "$SAVE_DIR/$name.pth" ]; then
        echo "Skipping $name (missing $SAVE_DIR/$name.pth)"
        return 0
      fi
      for q in orig q100 q95 q75 q50 q30 q10; do
        python train.py --eval --model_arch vits --stage "$stage" --classes 1000 \
          --datapath "$QDIR/$q" --temp_datapath "$QDIR/$q" --indexpaths assets/indexbase_val.csv,assets/indexbase_val.csv \
          --num_gpus 1 --num_cpus 16 --batch 1024 --rlc_lmax 320 --amp "$amp" --ampdtype "$dt" \
          --no_extract --no_resize --loadpath "$SAVE_DIR/$name.pth" --savepath "/tmp/${name}_$q.pth" "$@"
      done
    }
    ev dct dct_float 1 bf16 --freqnorm --freqnorm_stats "$STATS"
    ev quant_dct quant_dct 1 bf16 --freqnorm --freqnorm_stats "$STATS"
    ev abl_no_dc ac_only 1 bf16 --freqnorm --freqnorm_stats "$STATS"
    if [ -n "${TYPED_STATS:-}" ]; then
      ev abl_dpcm dc_diff_ac 1 bf16 --freqnorm --freqnorm_stats "$TYPED_STATS"
    fi
    ev rle_linear rlc_dcac_flat_direct 1 bf16
    ev huff_linear huff_dcac_flat_direct 1 bf16
    ev rle_conv rlc_dcac_2d_cp_concat_conv_direct 1 bf16 --numeric_feature_dim 128
    ev huff_conv huff_dcac_2d_cp_concat_conv_direct 1 bf16 --numeric_feature_dim 128
    ev raw_linear raw_byte 0 bf16 --embed_type 1
    ev raw_conv raw_byte 0 bf16 --embed_type 3
    ;;
  lmax)
    # tools/eval_event_lmax_truncation.py looks for these checkpoint names under models/
    # (train rle_conv huff_conv raw_conv for Fig. 6; optionally rle_2d huff_2d)
    mkdir -p models
    STAGES=()
    link_stage() {
      local src=$1 dst=$2 stage=$3
      if [ -f "$SAVE_DIR/$src" ]; then
        ln -sf "$SAVE_DIR/$src" "models/$dst"
        STAGES+=(--stage "$stage")
      fi
    }
    link_stage rle_conv.pth  stagewise_vits_rlc_dcac_2d_cp_concat_conv_direct_d128_lmax320_ep90.pth  rlc_dcac_2d_cp_concat_conv_direct_d128
    link_stage huff_conv.pth stagewise_vits_huff_dcac_2d_cp_concat_conv_direct_d128_lmax320_ep90.pth huff_dcac_2d_cp_concat_conv_direct_d128
    link_stage raw_conv.pth  stagewise_vits_raw_byte_discrete_conv_lmax320_ep90.pth                  raw_byte_discrete_conv
    link_stage rle_2d.pth    stagewise_vits_rlc_dcac_2d_direct_lmax320_ep90.pth                      rlc_dcac_2d_direct
    link_stage huff_2d.pth   stagewise_vits_huff_dcac_2d_direct_lmax320_ep90.pth                     huff_dcac_2d_direct
    if [ "${#STAGES[@]}" -eq 0 ]; then
      echo "No checkpoints found in $SAVE_DIR for lmax evaluation." >&2
      exit 1
    fi
    python tools/eval_event_lmax_truncation.py \
      "${STAGES[@]}" \
      --lmax 128 196 256 320 --datapath "$DATA_DIR" --indexpath assets/indexbase_val.csv \
      --topk 5 --output outputs/lmax_inference_truncation/results.csv
    ;;
esac
