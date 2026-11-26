# Reference results (from the original runs)

Final ImageNet-1k validation Top-1 of the runs reported in the paper (TensorBoard `Acc/Test` of the original training runs).

| Paper row | `reproduce_paper.sh` row | Stage / settings | Top-1 (original run) | Paper |
|---|---|---|---:|---:|
| DCT | `dct` | `dct_float`, `--freqnorm` | 74.482 | 74.48 |
| Quant_DCT | `quant_dct` | `quant_dct`, `--freqnorm` | 74.338 | 74.34 |
| RLE event | `rle_linear` | `rlc_dcac_flat_direct`, bf16, Lmax 320 | 69.06 | 69.06 |
| Huffman event | `huff_linear` | `huff_dcac_flat_direct`, bf16, Lmax 320 | 69.44 | 69.44 |
| Raw byte | `raw_linear` | `raw_byte`, `--embed_type 1`, Lmax 320, lr 3e-3, wd 3e-4 | 48.894 | 48.89 |
| RLE + conv emb. | `rle_conv` | `rlc_dcac_2d_cp_concat_conv_direct`, `--numeric_feature_dim 128`, Lmax 320 | 70.198 | 70.20 |
| Huffman + conv emb. | `huff_conv` | `huff_dcac_2d_cp_concat_conv_direct`, `--numeric_feature_dim 128`, Lmax 320 | 71.454 | 71.45 |
| Raw byte + conv emb. | `raw_conv` | `raw_byte`, `--embed_type 3`, Lmax 320, lr 3e-3, wd 3e-4 | 57.914 | 57.91 |
| Quant_DCT (no DC) | `abl_no_dc` | `ac_only`, `--freqnorm` | 74.684 | 74.70 |
| Quant_DCT (DPCM) | `abl_dpcm` | `dc_diff_ac`, `--freqnorm` (typed stats) | 74.852 | 74.85 |
| Amplitude_only | `abl_amp_only` | `event_amp_only_flat_direct`, bf16, Lmax 320 | — | 62.71 |

RGB (76.51) and the DCT rows of [25] in Table 2 are results quoted from the cited papers, not produced by this code.

- `qrobust_results_wide.*`: cross-quality evaluation (q100 ... q10, Fig. 5).
- `lmax_truncation.csv`: inference-time truncation below Lmax (Fig. 6).
