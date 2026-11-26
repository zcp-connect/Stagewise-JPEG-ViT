# May 3 2022 11:40
# Jeongsoo Park

import torch
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import utils.custom_transforms as ctrans
from utils.custom_sampler import DistributedEvalSampler
import dct_manip as dm # custom DCT coefficient handler
try:
    import byte_manip as bm
except ImportError:
    bm = None

JPEG_FIXED_QTABLES = [
    [8, 6, 6, 7, 6, 5, 8, 7, 7, 7, 9, 9, 8, 10, 12, 20, 13, 12, 11, 11, 12, 25, 18, 19, 15, 20, 29, 26, 31, 30, 29, 26, 28, 28, 32, 36, 46, 39, 32, 34, 44, 35, 28, 28, 40, 55, 41, 44, 48, 49, 52, 52, 52, 31, 39, 57, 61, 56, 50, 60, 46, 51, 52, 50],
    [8, 9, 9, 12, 11, 12, 24, 13, 13, 24, 50, 33, 28, 33, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50],
]

def get_mcu_coeff_transform(type='train', dtype_resize=torch.float32):
    if type == 'train':
        return transforms.Compose([
            ctrans.RandomResizedCrop_DCT(28, scale=(0.05, 1.0), ratio=(1, 1), dtype_resize=dtype_resize),
            ctrans.RandomFlip_DCT(p=0.5, direction='horizontal'),
        ])
    return transforms.Compose([
        ctrans.ResizedCenterCrop_DCT(32, 28, dtype_resize=dtype_resize),
    ])


def get_requant_dct_coeff_transform(type='train', dtype_resize=torch.float32, ops_list=None, num_ops=2, ops_magnitude=3):
    if type == 'train':
        return transforms.Compose([
            ctrans.RandomResizedCrop_DCT(28, scale=(0.05, 1.0), ratio=(1, 1), dtype_resize=dtype_resize),
            ctrans.RandomFlip_DCT(p=0.5, direction='horizontal'),
            ctrans.RandAugment_dct(num_ops=num_ops, magnitude=ops_magnitude, num_magnitude_bins=11, ops_list=ops_list),
        ])
    return transforms.Compose([
        ctrans.ResizedCenterCrop_DCT(32, 28, dtype_resize=dtype_resize),
    ])


def get_requant_dct_spatial_transform(type='train', dtype_resize=torch.float32):
    if type == 'train':
        return transforms.Compose([
            ctrans.RandomResizedCrop_DCT(28, scale=(0.05, 1.0), ratio=(1, 1), dtype_resize=dtype_resize),
            ctrans.RandomFlip_DCT(p=0.5, direction='horizontal'),
        ])
    return transforms.Compose([
        ctrans.ResizedCenterCrop_DCT(32, 28, dtype_resize=dtype_resize),
    ])



STAGE_ALIASES = {
    "mcu_bytes": "raw_byte",
    "raw_bytes": "raw_byte",
    "byte_mcu": "raw_byte",
    "raw_byte": "raw_byte",
    "bitstream": "bitstream",
    "rgb": "rgb",
    "ycbcr": "ycbcr",
    "dct": "dct_float",
    "dct_float": "dct_float",
    "quant_dct": "quant_dct",
    "quant_dct_group": "quant_dct_group",
    "quant_group": "quant_dct_group",
    "ac_only": "ac_only",
    "quant_dct_ac_only": "ac_only",
    "qdct_ac_only": "ac_only",
    "quant_dct_zigzag": "quant_dct_zigzag_linear",
    "quant_dct_zigzag_linear": "quant_dct_zigzag_linear",
    "zigzag": "quant_dct_zigzag_linear",
    "dc_abs": "dc_abs",
    "dc_diff": "dc_diff",
    "dc_diff_ac": "dc_diff_ac",
    "dc_diff_ac_group": "dc_diff_ac_group",
    "dcdiff_ac_group": "dc_diff_ac_group",
    "dc_dpcm_ac_group": "dc_diff_ac_group",
    "dcdiff_ac": "dc_diff_ac",
    "dc_dpcm_ac": "dc_diff_ac",
    "dc_absdiff": "dc_absdiff",
    "rlc_dc": "rlc_dc",
    "rlc_dcac": "rlc_dcac",
    "rlc_dc_simple": "rlc_dc_simple",
    "rlc_dcac_simple": "rlc_dcac_simple",
    "rlc_dc_attnpool": "rlc_dc_attnpool",
    "rlc_dcac_attnpool": "rlc_dcac_attnpool",
    "rlc_dcac_attn": "rlc_dcac_attnpool",
    "rlc_dc_2d": "rlc_dc_2d",
    "rlc_dcac_2d": "rlc_dcac_2d",
    "rlc_2d": "rlc_dcac_2d",
    "rlc_dc_2d_max": "rlc_dc_2d_max",
    "rlc_dcac_2d_max": "rlc_dcac_2d_max",
    "rlc_dc_flat": "rlc_dc_flat",
    "rlc_dcac_flat": "rlc_dcac_flat",
    "rlc_dc_2d_cp": "rlc_dc_2d_cp",
    "rlc_dcac_2d_cp": "rlc_dcac_2d_cp",
    "rlc_dc_2d_direct": "rlc_dc_2d_direct",
    "rlc_dcac_2d_direct": "rlc_dcac_2d_direct",
    "rlc_2d_direct": "rlc_dcac_2d_direct",
    "rlc_dc_2d_max_direct": "rlc_dc_2d_max_direct",
    "rlc_dcac_2d_max_direct": "rlc_dcac_2d_max_direct",
    "rlc_dc_flat_direct": "rlc_dc_flat_direct",
    "rlc_dcac_flat_direct": "rlc_dcac_flat_direct",
    "rlc_dc_2d_cp_direct": "rlc_dc_2d_cp_direct",
    "rlc_dcac_2d_cp_direct": "rlc_dcac_2d_cp_direct",
    "huff_dcac_2d_direct": "huff_dcac_2d_direct",
    "huffman_dcac_2d_direct": "huff_dcac_2d_direct",
    "huff_2d_direct": "huff_dcac_2d_direct",
    "huff_dcac_flat_direct": "huff_dcac_flat_direct",
    "huffman_dcac_flat_direct": "huff_dcac_flat_direct",
    "huff_flat_direct": "huff_dcac_flat_direct",
    "event_amp_only_flat_direct": "event_amp_only_flat_direct",
    "amp_only_flat_direct": "event_amp_only_flat_direct",
    "huff_dcac_2d_cp_direct": "huff_dcac_2d_cp_direct",
    "huffman_dcac_2d_cp_direct": "huff_dcac_2d_cp_direct",
    "huff_2d_cp_direct": "huff_dcac_2d_cp_direct",

    "rlc_dcac_2d_cp_concat_mean_direct": "rlc_dcac_2d_cp_concat_mean_direct",
    "rlc_dcac_2d_cp_add_conv_direct": "rlc_dcac_2d_cp_add_conv_direct",
    "rlc_dcac_2d_cp_concat_conv_direct": "rlc_dcac_2d_cp_concat_conv_direct",
    "rlc_dcac_2d_cp_film_conv_direct": "rlc_dcac_2d_cp_film_conv_direct",
    "huff_dcac_2d_cp_concat_mean_direct": "huff_dcac_2d_cp_concat_mean_direct",
    "huff_dcac_2d_cp_add_conv_direct": "huff_dcac_2d_cp_add_conv_direct",
    "huff_dcac_2d_cp_concat_conv_direct": "huff_dcac_2d_cp_concat_conv_direct",
    "huff_dcac_2d_cp_film_conv_direct": "huff_dcac_2d_cp_film_conv_direct",
}

JPEG_STAGE_DATASETS = {f"imagenet_{stage}" for stage in STAGE_ALIASES.values()}
JPEG_STAGE_DATASETS.update({"imagenet_mcu_bytes"})

QUANT_DCT_DERIVED_STAGES = {
    "quant_dct",
    "ac_only",
    "quant_dct_zigzag_linear",
    "dc_abs",
    "dc_diff",
    "dc_diff_ac",
    "quant_dct_group",
    "dc_diff_ac_group",
    "dc_absdiff",
}
NUMERIC_DCT_STAGES = {"dct_float", *QUANT_DCT_DERIVED_STAGES}
GROUP_DCT_STAGES = {"dct_float", "quant_dct", "quant_dct_group", "ac_only", "dc_diff_ac", "dc_diff_ac_group"}
FLAT_NUMERIC_DCT_STAGES = NUMERIC_DCT_STAGES - GROUP_DCT_STAGES
DCDIFF_AC_STAGES = {"dc_diff_ac", "dc_diff_ac_group"}
FREQNORM_STAGES = {"dct_float", "quant_dct", "quant_dct_group", "ac_only", "dc_diff_ac", "dc_diff_ac_group"}
DCDIFF_POST_AUG_OPS = {"Cutout", "BlockDropout", "FreqBandDropout", "ACQuantNoise"}
DCDIFF_CHROMA_AUG_OPS = {"Grayscale", "Color", "AutoSaturation", "ChromaDrop"}
FREQNORM_CACHE = {}
RLC_CP_DIRECT_STAGES = {
    "rlc_dc_2d_cp_direct",
    "rlc_dcac_2d_cp_direct",
    "rlc_dcac_2d_cp_concat_mean_direct",
    "rlc_dcac_2d_cp_add_conv_direct",
    "rlc_dcac_2d_cp_concat_conv_direct",
    "rlc_dcac_2d_cp_film_conv_direct",
}
HUFFMAN_CP_DIRECT_STAGES = {
    "huff_dcac_2d_cp_direct",
    "huff_dcac_2d_cp_concat_mean_direct",
    "huff_dcac_2d_cp_add_conv_direct",
    "huff_dcac_2d_cp_concat_conv_direct",
    "huff_dcac_2d_cp_film_conv_direct",
}


def normalize_stage_name(stage):
    key = str(stage).lower()
    if key not in STAGE_ALIASES:
        raise UnsupportedDataset(msg=f"Unsupported JPEG stage: {stage}")
    return STAGE_ALIASES[key]


def stage_to_dataset(stage):
    stage = normalize_stage_name(stage)
    return "imagenet_mcu_bytes" if stage == "raw_byte" else f"imagenet_{stage}"


def dataset_to_stage(dataset):
    dataset = str(dataset).lower()
    if dataset == "imagenet_mcu_bytes":
        return "raw_byte"
    if dataset.startswith("imagenet_"):
        return normalize_stage_name(dataset[len("imagenet_"):])
    return normalize_stage_name(dataset)


DCT_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]
ZIGZAG_POS = {coeff: idx for idx, coeff in enumerate(DCT_ZIGZAG)}

HUFFMAN_PREFIX_PAD_ID = 0
HUFFMAN_PREFIX_TABLE_STRIDE = 256
HUFFMAN_PREFIX_Y_DC_OFFSET = 1
HUFFMAN_PREFIX_Y_AC_OFFSET = HUFFMAN_PREFIX_Y_DC_OFFSET + HUFFMAN_PREFIX_TABLE_STRIDE
HUFFMAN_PREFIX_C_DC_OFFSET = HUFFMAN_PREFIX_Y_AC_OFFSET + HUFFMAN_PREFIX_TABLE_STRIDE
HUFFMAN_PREFIX_C_AC_OFFSET = HUFFMAN_PREFIX_C_DC_OFFSET + HUFFMAN_PREFIX_TABLE_STRIDE
HUFFMAN_PREFIX_VOCAB_SIZE = 1 + 4 * HUFFMAN_PREFIX_TABLE_STRIDE

# JPEG category LUT indexed by value + offset. 2048 covers the current
# requantized range [-189, 189], DC DPCM range [-378, 378], and the wider
# dequant-clamped/requantized range [-1024, 1016] if the clamp is relaxed.
_JPEG_CAT_OFFSET = 2048
_JPEG_CATEGORY_LUT = [0] * (2 * _JPEG_CAT_OFFSET + 1)
for _jpeg_cat_abs_value in range(1, _JPEG_CAT_OFFSET + 1):
    _jpeg_cat_bit_length = _jpeg_cat_abs_value.bit_length()
    _JPEG_CATEGORY_LUT[_JPEG_CAT_OFFSET + _jpeg_cat_abs_value] = _jpeg_cat_bit_length
    _JPEG_CATEGORY_LUT[_JPEG_CAT_OFFSET - _jpeg_cat_abs_value] = _jpeg_cat_bit_length


def normalize_rgb_tensor(img):
    return TF.to_tensor(img.convert("RGB"))


def normalize_ycbcr_tensor(img):
    ycbcr = TF.to_tensor(img.convert("YCbCr"))
    return ycbcr


def load_stage_image_tensor(imgpath, stage, type="train"):
    img = PIL.Image.open(imgpath)
    if type == "train":
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.05, 1.0), ratio=(1.0, 1.0))
        img = TF.resized_crop(img, i, j, h, w, (224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        if random.random() < 0.5:
            img = TF.hflip(img)
    else:
        img = TF.resize(img, 256, interpolation=transforms.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, 224)
    if stage == "rgb":
        return normalize_rgb_tensor(img)
    if stage == "ycbcr":
        return normalize_ycbcr_tensor(img)
    raise UnsupportedDataset(msg=f"Image tensor stage expected rgb/ycbcr, got {stage}")


def _component_grid_to_14x14(component, target_h=14, target_w=14):
    # component: [C, H, W, 8, 8] or [H, W, 8, 8]
    if component.dim() == 4:
        component = component.unsqueeze(0)
    c, h, w, kh, kw = component.shape
    if h < target_h or w < target_w:
        raise UnsupportedDataset(msg=f"DCT component too small after transform: {component.shape}")
    return component[:, :target_h, :target_w].reshape(c, target_h, target_w, kh * kw)


def _dct_16x16_cell_features(y_coeff, cbcr_coeff=None):
    # One visual token corresponds to a 16x16 image cell:
    # 4 luminance 8x8 blocks + 1 Cb 8x8 block + 1 Cr 8x8 block = 384 values.
    y = y_coeff.float()
    if y.dim() == 4:
        y = y.unsqueeze(0)
    _, y_h, y_w, kh, kw = y.shape
    if y_h < 28 or y_w < 28:
        raise UnsupportedDataset(msg=f"Y DCT component too small for 14x14 16x16 cells: {y.shape}")

    y = y[:, :28, :28]
    y = y.reshape(1, 14, 2, 14, 2, kh, kw)
    y = y.permute(1, 3, 0, 2, 4, 5, 6).reshape(14, 14, 4 * kh * kw)

    if cbcr_coeff is None:
        cbcr = torch.zeros((14, 14, 2 * kh * kw), dtype=y.dtype)
    else:
        cbcr = cbcr_coeff.float()
        if cbcr.dim() == 4:
            cbcr = cbcr.unsqueeze(0)
        if cbcr.shape[1] < 14 or cbcr.shape[2] < 14:
            raise UnsupportedDataset(msg=f"CbCr DCT component too small for 14x14 cells: {cbcr.shape}")
        if cbcr.shape[0] == 1:
            cbcr = cbcr.repeat(2, 1, 1, 1, 1)
        cbcr = cbcr[:2, :14, :14].permute(1, 2, 0, 3, 4).reshape(14, 14, 2 * kh * kw)

    return torch.cat((y, cbcr), dim=-1).reshape(14 * 14, -1).contiguous()


def _dct_16x16_cell_blocks(y_coeff, cbcr_coeff=None):
    return _dct_16x16_cell_features(y_coeff, cbcr_coeff).reshape(14 * 14, 6, 64)


def _dct_16x16_quant_vector(quant, dtype=torch.float32):
    q = quant.float()
    y_q = q[0].reshape(-1).repeat(4)
    if q.shape[0] >= 3:
        cbcr_q = q[1:3].reshape(-1)
    elif q.shape[0] >= 2:
        cbcr_q = q[1].reshape(-1).repeat(2)
    else:
        cbcr_q = q[0].reshape(-1).repeat(2)
    return torch.cat((y_q, cbcr_q), dim=0).to(dtype=dtype)


def _dequantize_coefficients(quant, y_coeff, cbcr_coeff):
    y_float = torch.clamp(y_coeff * quant[0], min=-2**10, max=2**10 - 8).short().contiguous()
    if cbcr_coeff is None:
        return (y_float,)
    cbcr_float = torch.clamp(
        cbcr_coeff * quant[1:3].unsqueeze(1).unsqueeze(1),
        min=-2**10,
        max=2**10 - 8,
    ).short().contiguous()
    return y_float, cbcr_float


def _requantize_coefficients(quant, coeffs):
    coeffs = ctrans.dct_prep(coeffs, mode='start')
    y_quant = torch.round(coeffs[0].float() / quant[0].float()).clamp(min=-189, max=189).short().contiguous()
    if len(coeffs) == 1:
        return y_quant, None
    cbcr_quant = torch.round(
        coeffs[1].float() / quant[1:3].float().unsqueeze(1).unsqueeze(1)
    ).clamp(min=-189, max=189).short().contiguous()
    return y_quant, cbcr_quant


def _sample_dcdiff_aug_ops(ops_list, num_ops):
    if num_ops <= 0:
        return [], []
    if ops_list is None:
        ops_pool = ctrans.RandAugment_dct(num_ops=0).ops_list.copy()
        ops_pool.extend(op for op in DCDIFF_POST_AUG_OPS if op not in ops_pool)
    else:
        ops_pool = list(ops_list)
    pre_ops = []
    post_ops = []
    for _ in range(num_ops):
        if len(ops_pool) == 0:
            break
        op_index = int(torch.randint(len(ops_pool), (1,)).item())
        op_name = ops_pool[op_index]
        if op_name in DCDIFF_POST_AUG_OPS:
            post_ops.append(op_name)
        else:
            pre_ops.append(op_name)
        if op_name in DCDIFF_CHROMA_AUG_OPS:
            if op_name == "Grayscale":
                ops_pool = list(set(ops_pool).difference(DCDIFF_CHROMA_AUG_OPS))
            else:
                ops_pool = list(set(ops_pool).difference({"Grayscale"}))
    return pre_ops, post_ops


def _magnitude_from_meta(op_meta, op_name, ops_magnitude):
    magnitudes, signed = op_meta[op_name]
    if magnitudes.ndim > 0:
        magnitude_index = max(0, min(int(ops_magnitude), magnitudes.numel() - 1))
        magnitude = float(magnitudes[magnitude_index].item())
    else:
        magnitude = float(magnitudes.item())
    if signed and torch.randint(2, (1,)).item():
        magnitude *= -1.0
    return magnitude


def _apply_exact_dct_ops(coeffs, selected_ops, ops_magnitude):
    if not selected_ops:
        return coeffs
    coeffs = ctrans.dct_prep(coeffs, mode='start')
    _, height, width, _, _ = coeffs[0].shape
    augmenter = ctrans.RandAugment_dct(num_ops=0, magnitude=ops_magnitude, num_magnitude_bins=11, ops_list=selected_ops)
    op_meta = augmenter._augmentation_space(augmenter.num_magnitude_bins, (height, width))
    for component_index in range(len(coeffs)):
        coeffs[component_index] = coeffs[component_index].clamp(min=-2**10, max=2**10-8)
    for op_name in selected_ops:
        magnitude = _magnitude_from_meta(op_meta, op_name, ops_magnitude)
        coeffs = ctrans._apply_op_dct(
            coeffs,
            op_name,
            magnitude,
            pad=augmenter.pad,
            conv_Ls=augmenter.conv_Ls,
            conv_Ms=augmenter.conv_Ms,
        )
        assert coeffs[0].dtype != torch.float32, (
            f"Coeff dtype should not be float32. Current augmentation: {op_name}, dtype: {coeffs[0].dtype}"
        )
    return ctrans.dct_prep(coeffs, mode='end')


def _apply_exact_quant_dct_ops(coeffs, selected_ops, ops_magnitude):
    if not selected_ops:
        return coeffs
    coeffs = ctrans.dct_prep(coeffs, mode='start')
    dtypes = [component.dtype for component in coeffs]
    _, height, width, _, _ = coeffs[0].shape
    augmenter = ctrans.RandAugment_QuantDCT(num_ops=0, magnitude=ops_magnitude, num_magnitude_bins=11, ops_list=selected_ops)
    op_meta = augmenter._augmentation_space(augmenter.num_magnitude_bins, (height, width))
    for op_name in selected_ops:
        magnitude = _magnitude_from_meta(op_meta, op_name, ops_magnitude)
        coeffs = augmenter._apply_op(coeffs, op_name, magnitude)
        coeffs = augmenter._finalize(coeffs, dtypes)
    return ctrans.dct_prep(coeffs, mode='end')


def _apply_dcdiff_requant_pipeline(
    quant, y_coeff, cbcr_coeff, type='train', dtype_resize=torch.float32,
    ops_list=None, num_ops=2, ops_magnitude=3,
):
    coeffs = _dequantize_coefficients(quant, y_coeff, cbcr_coeff)
    coeffs = get_requant_dct_spatial_transform(type=type, dtype_resize=dtype_resize)(coeffs)

    if type == 'train':
        pre_ops, post_ops = _sample_dcdiff_aug_ops(ops_list, num_ops)
        coeffs = _apply_exact_dct_ops(coeffs, pre_ops, ops_magnitude)
    else:
        post_ops = []

    y_out, cbcr_out = _requantize_coefficients(quant, coeffs)
    coeffs = ctrans.DCDiffQuantDCT(mode="standard")((y_out, cbcr_out) if cbcr_out is not None else (y_out,))
    if type == 'train':
        coeffs = _apply_exact_quant_dct_ops(coeffs, post_ops, ops_magnitude)
    coeffs = ctrans.dct_prep(coeffs, mode='start')
    y_out = coeffs[0].short().contiguous()
    cbcr_out = coeffs[1].short().contiguous() if len(coeffs) > 1 else None
    return y_out, cbcr_out


def _load_freqnorm_stats(stats_path):
    path = str(Path(stats_path).expanduser())
    if path not in FREQNORM_CACHE:
        stats = torch.load(path, map_location="cpu")
        cache = {}
        if "std_q_y_dc_start" in stats or "std_Q_Y_DC_START" in stats:
            for component in ("y", "cb", "cr"):
                for value_type in ("dc_start", "dc_delta", "ac"):
                    for stat_name in ("mean", "std"):
                        lower_key = f"{stat_name}_q_{component}_{value_type}"
                        upper_key = f"{stat_name}_Q_{component.upper()}_{value_type.upper()}"
                        source_key = lower_key if lower_key in stats else upper_key
                        tensor = stats[source_key].float()
                        if stat_name == "std":
                            tensor = tensor.clamp_min(1e-6)
                        cache[lower_key] = tensor
            cache["schema"] = "dcdiff_typed"
        else:
            cache = {
                "schema": "standard",
                "mean_f_y": stats["mean_f_y"].float(),
                "mean_f_cb": stats["mean_f_cb"].float(),
                "mean_f_cr": stats["mean_f_cr"].float(),
                "std_f_y": stats["std_f_y"].float().clamp_min(1e-6),
                "std_f_cb": stats["std_f_cb"].float().clamp_min(1e-6),
                "std_f_cr": stats["std_f_cr"].float().clamp_min(1e-6),
                "mean_q_y": stats["mean_q_y"].float(),
                "mean_q_cb": stats["mean_q_cb"].float(),
                "mean_q_cr": stats["mean_q_cr"].float(),
                "std_q_y": stats["std_q_y"].float().clamp_min(1e-6),
                "std_q_cb": stats["std_q_cb"].float().clamp_min(1e-6),
                "std_q_cr": stats["std_q_cr"].float().clamp_min(1e-6),
            }
        FREQNORM_CACHE[path] = cache
    return FREQNORM_CACHE[path]


def _normalize_dcdiff_component(component, stats, component_name, is_y):
    output = component.float().clone()
    mean_start = stats[f"mean_q_{component_name}_dc_start"].to(device=component.device)
    std_start = stats[f"std_q_{component_name}_dc_start"].to(device=component.device)
    mean_delta = stats[f"mean_q_{component_name}_dc_delta"].to(device=component.device)
    std_delta = stats[f"std_q_{component_name}_dc_delta"].to(device=component.device)
    mean_ac = stats[f"mean_q_{component_name}_ac"].to(device=component.device)
    std_ac = stats[f"std_q_{component_name}_ac"].to(device=component.device)

    output = (output - mean_ac) / std_ac
    output[..., 0, 0] = (component.float()[..., 0, 0] - mean_delta[0, 0]) / std_delta[0, 0]
    if is_y or component.ndim == 5:
        output[:, 0, 0, 0, 0] = (component.float()[:, 0, 0, 0, 0] - mean_start[0, 0]) / std_start[0, 0]
    else:
        output[0, 0, 0, 0] = (component.float()[0, 0, 0, 0] - mean_start[0, 0]) / std_start[0, 0]
    return output.contiguous()


def _apply_dcdiff_typed_frequency_norm(y_coeff, cbcr_coeff, stats):
    if stats.get("schema") != "dcdiff_typed":
        raise UnsupportedDataset(msg="dc_diff_ac freqnorm requires typed DPCM stats")
    y = _normalize_dcdiff_component(y_coeff, stats, "y", is_y=True)
    if cbcr_coeff is None:
        return y, None
    cbcr = cbcr_coeff.float().clone()
    cbcr[0] = _normalize_dcdiff_component(cbcr_coeff[0], stats, "cb", is_y=False)
    cbcr[1] = _normalize_dcdiff_component(cbcr_coeff[1], stats, "cr", is_y=False)
    return y.contiguous(), cbcr.contiguous()


def _apply_frequency_norm(y_coeff, cbcr_coeff, stage, stats_path):
    stage = normalize_stage_name(stage)
    if stats_path is None:
        raise UnsupportedDataset(msg="freqnorm enabled but no frequency stats path was provided")
    if stage not in FREQNORM_STAGES:
        raise UnsupportedDataset(msg=f"freqnorm is only defined for {sorted(FREQNORM_STAGES)}, got {stage}")
    stats = _load_freqnorm_stats(stats_path)
    if stage in DCDIFF_AC_STAGES:
        return _apply_dcdiff_typed_frequency_norm(y_coeff, cbcr_coeff, stats)
    if stats.get("schema") != "standard":
        raise UnsupportedDataset(msg=f"{stage} freqnorm requires standard DCT stats")
    prefix = "f" if stage == "dct_float" else "q"
    mean_y = stats[f"mean_{prefix}_y"].to(device=y_coeff.device)
    std_y = stats[f"std_{prefix}_y"].to(device=y_coeff.device)
    y = (y_coeff.float() - mean_y) / std_y
    if cbcr_coeff is None:
        return y.contiguous(), None
    cbcr = cbcr_coeff.float().clone()
    mean_cb = stats[f"mean_{prefix}_cb"].to(device=cbcr_coeff.device)
    mean_cr = stats[f"mean_{prefix}_cr"].to(device=cbcr_coeff.device)
    std_cb = stats[f"std_{prefix}_cb"].to(device=cbcr_coeff.device)
    std_cr = stats[f"std_{prefix}_cr"].to(device=cbcr_coeff.device)
    cbcr[0] = (cbcr[0] - mean_cb) / std_cb
    cbcr[1] = (cbcr[1] - mean_cr) / std_cr
    return y.contiguous(), cbcr.contiguous()


def _zero_dc_coefficients(y_coeff, cbcr_coeff):
    y_coeff = y_coeff.clone()
    y_coeff[..., 0, 0] = 0
    if cbcr_coeff is None:
        return y_coeff, None
    cbcr_coeff = cbcr_coeff.clone()
    cbcr_coeff[..., 0, 0] = 0
    return y_coeff, cbcr_coeff


def _quant_dct_zigzag_features(y_coeff, cbcr_coeff):
    blocks = _dct_16x16_cell_blocks(y_coeff, cbcr_coeff)
    zigzag_index = torch.as_tensor(DCT_ZIGZAG, device=blocks.device, dtype=torch.long)
    return blocks.index_select(dim=-1, index=zigzag_index).reshape(14 * 14, -1).contiguous() / 189.0


def _quant_dct_dc_abs_features(y_coeff, cbcr_coeff):
    return _dct_16x16_cell_blocks(y_coeff, cbcr_coeff)[:, :, 0].contiguous()


def _component_raster_diff(values):
    diffs = torch.empty_like(values)
    diffs[0] = values[0]
    diffs[1:] = values[1:] - values[:-1]
    return diffs


def _quant_dct_dc_diff_features(y_coeff, cbcr_coeff):
    dc_abs = _quant_dct_dc_abs_features(y_coeff, cbcr_coeff)
    dc_diff = torch.empty_like(dc_abs)

    # Crop-local JPEG-style DC DPCM approximation:
    # Y has four blocks per MCU in scan order; Cb/Cr have one block per MCU.
    # The first value of each component is differenced against a zero predictor.
    y_sequence = dc_abs[:, :4].reshape(-1)
    dc_diff[:, :4] = _component_raster_diff(y_sequence).reshape(14 * 14, 4)
    dc_diff[:, 4] = _component_raster_diff(dc_abs[:, 4])
    dc_diff[:, 5] = _component_raster_diff(dc_abs[:, 5])
    return dc_diff.contiguous()


def _quant_dct_dc_diff_ac_features(y_coeff, cbcr_coeff):
    return _dct_16x16_cell_features(y_coeff, cbcr_coeff)


def stage_dct_features_from_coeffs(quant, y_coeff, cbcr_coeff, stage):
    stage = normalize_stage_name(stage)
    features = _dct_16x16_cell_features(y_coeff, cbcr_coeff)
    if stage == "dct_float":
        pass
    elif stage == "quant_dct":
        features = features / 189.0
    elif stage == "quant_dct_group":
        features = features / 189.0
    elif stage == "ac_only":
        features = _dct_16x16_cell_features(*_zero_dc_coefficients(y_coeff, cbcr_coeff)) / 189.0
    elif stage == "quant_dct_zigzag_linear":
        features = _quant_dct_zigzag_features(y_coeff, cbcr_coeff)
    elif stage == "dc_abs":
        features = _quant_dct_dc_abs_features(y_coeff, cbcr_coeff) / 189.0
    elif stage == "dc_diff":
        features = _quant_dct_dc_diff_features(y_coeff, cbcr_coeff) / 189.0
    elif stage == "dc_diff_ac":
        features = _quant_dct_dc_diff_ac_features(y_coeff, cbcr_coeff) / 189.0
    elif stage == "dc_diff_ac_group":
        features = _quant_dct_dc_diff_ac_features(y_coeff, cbcr_coeff) / 189.0
    elif stage == "dc_absdiff":
        dc_abs = _quant_dct_dc_abs_features(y_coeff, cbcr_coeff)
        dc_diff = _quant_dct_dc_diff_features(y_coeff, cbcr_coeff)
        features = torch.cat((dc_abs, dc_diff), dim=-1) / 189.0
    else:
        raise UnsupportedDataset(msg=f"DCT feature stage expected one of {sorted(NUMERIC_DCT_STAGES)}, got {stage}")
    return features


def load_stage_dct_features(
    imgpath, stage, type="train", dtype_resize=torch.float32,
    ops_list=None, num_ops=2, ops_magnitude=3,
    freqnorm=False, freqnorm_stats=None,
):
    new_dim, quant, y_out, cbcr_out = load_augmented_mcu_coefficients(
        imgpath, type=type, dtype_resize=dtype_resize, stage=stage,
        ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
    )
    if freqnorm:
        y_out, cbcr_out = _apply_frequency_norm(y_out, cbcr_out, stage, freqnorm_stats)
    return stage_dct_features_from_coeffs(quant, y_out, cbcr_out, stage)


def load_stage_dct_group_tensors(
    imgpath, stage, type="train", dtype_resize=torch.float32,
    ops_list=None, num_ops=2, ops_magnitude=3,
    freqnorm=False, freqnorm_stats=None,
):
    _, _, y_out, cbcr_out = load_augmented_mcu_coefficients(
        imgpath, type=type, dtype_resize=dtype_resize, stage=stage,
        ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
    )
    if cbcr_out is None:
        _, h, w, kh, kw = y_out.shape
        cbcr_out = torch.zeros((2, h // 2, w // 2, kh, kw), dtype=y_out.dtype, device=y_out.device)
    if freqnorm:
        y_out, cbcr_out = _apply_frequency_norm(y_out, cbcr_out, stage, freqnorm_stats)
    if normalize_stage_name(stage) == "ac_only":
        y_out, cbcr_out = _zero_dc_coefficients(y_out, cbcr_out)
    if freqnorm:
        return {"y": y_out, "cbcr": cbcr_out}
    if normalize_stage_name(stage) == "dct_float":
        return {"y": y_out.float() / 1024.0, "cbcr": cbcr_out.float() / 1024.0}
    return {
        "y": y_out.float() / 189.0,
        "cbcr": cbcr_out.float() / 189.0,
    }


def _event_kind_id(token_type):
    if token_type == "DC":
        return 1
    if token_type == "AC":
        return 2
    if token_type == "EOB":
        return 3
    if token_type == "ZRL":
        return 4
    return 0


def _append_rlc_event(fields, pos, token, block, dc_only=False):
    kind, component, block_id, zigzag, run, size, amplitude, mask = fields
    token_type = str(token["token_type"])
    if dc_only and token_type != "DC":
        return pos
    if pos >= kind.shape[0]:
        return pos
    coeff_index = int(token.get("coeff_index", -1))
    component_index = int(block.get("component_index", 0))
    block_in_mcu = int(block.get("block_in_mcu", 0))
    kind[pos] = _event_kind_id(token_type)
    component[pos] = min(component_index + 1, 3)
    block_id[pos] = min(block_in_mcu + 1, 6)
    zigzag[pos] = min(ZIGZAG_POS.get(coeff_index, 0) + 1 if coeff_index >= 0 else 0, 64)
    run[pos] = min(int(token.get("run", 0)) + 1, 16)
    size[pos] = min(int(token.get("size", 0)) + 1, 16)
    amplitude[pos] = int(token.get("value", 0))
    mask[pos] = True
    return pos + 1


def pack_rlc_event_tensors(syntax_mcus, lmax=384, dc_only=False):
    mcu_count = len(syntax_mcus)
    kind = torch.zeros((mcu_count, lmax), dtype=torch.long)
    component = torch.zeros((mcu_count, lmax), dtype=torch.long)
    block_id = torch.zeros((mcu_count, lmax), dtype=torch.long)
    zigzag = torch.zeros((mcu_count, lmax), dtype=torch.long)
    run = torch.zeros((mcu_count, lmax), dtype=torch.long)
    size = torch.zeros((mcu_count, lmax), dtype=torch.long)
    amplitude = torch.zeros((mcu_count, lmax), dtype=torch.float32)
    mask = torch.zeros((mcu_count, lmax), dtype=torch.bool)
    event_len = torch.zeros((mcu_count,), dtype=torch.int32)
    for mcu_i, mcu in enumerate(syntax_mcus):
        pos = 0
        fields = (kind[mcu_i], component[mcu_i], block_id[mcu_i], zigzag[mcu_i], run[mcu_i], size[mcu_i], amplitude[mcu_i], mask[mcu_i])
        for block in mcu["blocks"]:
            for token in block["tokens"]:
                pos = _append_rlc_event(fields, pos, token, block, dc_only=dc_only)
                if pos >= lmax:
                    break
            if pos >= lmax:
                break
        event_len[mcu_i] = pos
    return kind, component, block_id, zigzag, run, size, amplitude, mask, event_len


def _append_rlc_2d_event(run_row, value_row, mask_row, pos, token):
    token_type = str(token["token_type"])
    if token_type == "EOB":
        return pos
    if pos >= run_row.shape[0]:
        return pos
    if token_type == "ZRL":
        run_value = 15
        coeff_value = 0.0
    elif token_type == "DC":
        run_value = 0
        coeff_value = float(token.get("value", 0))
    else:
        run_value = int(token.get("run", 0))
        coeff_value = float(token.get("value", 0))
    run_row[pos] = max(0, min(run_value, 15)) / 15.0
    value_row[pos] = coeff_value / 189.0
    mask_row[pos] = True
    return pos + 1


def pack_rlc_event_tensors_2d(syntax_mcus, lmax=320, dc_only=False, dtype=torch.float32):
    mcu_count = len(syntax_mcus)
    run = torch.zeros((mcu_count, lmax), dtype=dtype)
    value = torch.zeros((mcu_count, lmax), dtype=dtype)
    mask = torch.zeros((mcu_count, lmax), dtype=torch.bool)
    event_len = torch.zeros((mcu_count,), dtype=torch.int32)
    for mcu_i, mcu in enumerate(syntax_mcus):
        pos = 0
        for block in mcu["blocks"]:
            for token in block["tokens"]:
                token_type = str(token["token_type"])
                if dc_only and token_type != "DC":
                    continue
                pos = _append_rlc_2d_event(run[mcu_i], value[mcu_i], mask[mcu_i], pos, token)
                if pos >= lmax:
                    break
            if pos >= lmax:
                break
        event_len[mcu_i] = pos
    return run, value, mask, event_len


def pack_rlc_event_tensors_cp(syntax_mcus, lmax=320, dc_only=False, dtype=torch.float32):
    mcu_count = len(syntax_mcus)
    run_int = torch.zeros((mcu_count, lmax), dtype=torch.long)
    value = torch.zeros((mcu_count, lmax), dtype=dtype)
    mask = torch.zeros((mcu_count, lmax), dtype=torch.bool)
    event_len = torch.zeros((mcu_count,), dtype=torch.int32)
    for mcu_i, mcu in enumerate(syntax_mcus):
        pos = 0
        for block in mcu["blocks"]:
            for token in block["tokens"]:
                token_type = str(token["token_type"])
                if dc_only and token_type != "DC":
                    continue
                if token_type == "EOB":
                    continue
                if pos >= lmax:
                    break
                if token_type == "ZRL":
                    run_value = 15
                    coeff_value = 0.0
                elif token_type == "DC":
                    run_value = 0
                    coeff_value = float(token.get("value", 0))
                else:
                    run_value = int(token.get("run", 0))
                    coeff_value = float(token.get("value", 0))
                run_int[mcu_i, pos] = max(0, min(run_value, 15))
                value[mcu_i, pos] = coeff_value / 189.0
                mask[mcu_i, pos] = True
                pos += 1
            if pos >= lmax:
                break
        event_len[mcu_i] = pos
    return run_int, value, mask, event_len


def _zigzag_blocks_for_rlc(y_coeff, cbcr_coeff):
    blocks = _dct_16x16_cell_blocks(y_coeff, cbcr_coeff)
    zigzag_index = torch.as_tensor(DCT_ZIGZAG, device=blocks.device, dtype=torch.long)
    return blocks.index_select(dim=-1, index=zigzag_index).contiguous()


def pack_rlc_event_tensors_direct(
    blocks_zz, lmax=320, dc_only=False, dtype=torch.float32, cp=False,
):
    if blocks_zz.dim() != 3 or tuple(blocks_zz.shape) != (196, 6, 64):
        raise UnsupportedDataset(msg=f"expected RLC blocks with shape (196, 6, 64), got {tuple(blocks_zz.shape)}")

    if blocks_zz.is_floating_point():
        blocks_zz = blocks_zz.round().to(torch.long)
    blocks_list = blocks_zz.tolist()
    dc_pred = [0, 0, 0]
    comp_of_block = [0, 0, 0, 0, 1, 2]
    run_rows = []
    value_rows = []
    event_lens = []

    for mcu_blocks in blocks_list:
        run_row = [0] * lmax if cp else [0.0] * lmax
        value_row = [0.0] * lmax
        pos = 0
        for block_i, block in enumerate(mcu_blocks):
            if pos >= lmax:
                break

            comp = comp_of_block[block_i]
            dc_cur = int(block[0])
            dc_diff = dc_cur - dc_pred[comp]
            dc_pred[comp] = dc_cur

            run_row[pos] = 0
            value_row[pos] = dc_diff / 189.0
            pos += 1
            if pos >= lmax or dc_only:
                continue

            zero_run = 0
            for ac in block[1:]:
                ac = int(ac)
                if ac == 0:
                    zero_run += 1
                    continue
                while zero_run >= 16:
                    if pos >= lmax:
                        break
                    run_row[pos] = 15 if cp else 1.0
                    value_row[pos] = 0.0
                    pos += 1
                    zero_run -= 16
                if pos >= lmax:
                    break
                run_row[pos] = zero_run if cp else (zero_run / 15.0)
                value_row[pos] = ac / 189.0
                pos += 1
                zero_run = 0

        run_rows.append(run_row)
        value_rows.append(value_row)
        event_lens.append(pos)

    run = torch.tensor(run_rows, dtype=torch.long if cp else dtype)
    value = torch.tensor(value_rows, dtype=dtype)
    event_len = torch.tensor(event_lens, dtype=torch.int32)
    mask = torch.arange(lmax).unsqueeze(0) < event_len.unsqueeze(1)
    return run, value, mask, event_len


def _jpeg_category(value):
    value = abs(int(value))
    return 0 if value == 0 else value.bit_length()


def _huffman_prefix_id(component, is_dc, symbol):
    symbol = max(0, min(int(symbol), 255))
    if component == 0:
        offset = HUFFMAN_PREFIX_Y_DC_OFFSET if is_dc else HUFFMAN_PREFIX_Y_AC_OFFSET
    else:
        offset = HUFFMAN_PREFIX_C_DC_OFFSET if is_dc else HUFFMAN_PREFIX_C_AC_OFFSET
    return offset + symbol


def pack_huffman_event_tensors_direct(blocks_zz, lmax=320, dtype=torch.float32):
    if blocks_zz.dim() != 3 or tuple(blocks_zz.shape) != (196, 6, 64):
        raise UnsupportedDataset(msg=f"expected Huffman blocks with shape (196, 6, 64), got {tuple(blocks_zz.shape)}")

    if blocks_zz.is_floating_point():
        blocks_zz = blocks_zz.round().to(torch.long)
    blocks_list = blocks_zz.tolist()
    dc_pred = [0, 0, 0]
    comp_of_block = [0, 0, 0, 0, 1, 2]
    prefix_rows = []
    amplitude_rows = []
    event_lens = []
    category_lut = _JPEG_CATEGORY_LUT
    category_offset = _JPEG_CAT_OFFSET
    y_dc_offset = HUFFMAN_PREFIX_Y_DC_OFFSET
    y_ac_offset = HUFFMAN_PREFIX_Y_AC_OFFSET
    c_dc_offset = HUFFMAN_PREFIX_C_DC_OFFSET
    c_ac_offset = HUFFMAN_PREFIX_C_AC_OFFSET
    pad_id = HUFFMAN_PREFIX_PAD_ID

    for mcu_blocks in blocks_list:
        prefix_row = [pad_id] * lmax
        amplitude_row = [0.0] * lmax
        pos = 0
        for block_i, block in enumerate(mcu_blocks):
            if pos >= lmax:
                break

            comp = comp_of_block[block_i]
            if comp == 0:
                dc_offset = y_dc_offset
                ac_offset = y_ac_offset
            else:
                dc_offset = c_dc_offset
                ac_offset = c_ac_offset

            dc_cur = block[0]
            dc_diff = dc_cur - dc_pred[comp]
            dc_pred[comp] = dc_cur
            prefix_row[pos] = dc_offset + category_lut[dc_diff + category_offset]
            amplitude_row[pos] = dc_diff / 189.0
            pos += 1
            if pos >= lmax:
                continue

            zero_run = 0
            for ac in block[1:]:
                if ac == 0:
                    zero_run += 1
                    continue
                while zero_run >= 16:
                    if pos >= lmax:
                        break
                    prefix_row[pos] = ac_offset + 0xF0
                    amplitude_row[pos] = 0.0
                    pos += 1
                    zero_run -= 16
                if pos >= lmax:
                    break
                symbol = (zero_run << 4) | category_lut[ac + category_offset]
                prefix_row[pos] = ac_offset + symbol
                amplitude_row[pos] = ac / 189.0
                pos += 1
                zero_run = 0

            if pos < lmax and zero_run > 0:
                prefix_row[pos] = ac_offset
                amplitude_row[pos] = 0.0
                pos += 1

        prefix_rows.append(prefix_row)
        amplitude_rows.append(amplitude_row)
        event_lens.append(pos)

    prefix_id = torch.tensor(prefix_rows, dtype=torch.long)
    amplitude = torch.tensor(amplitude_rows, dtype=dtype)
    event_len = torch.tensor(event_lens, dtype=torch.int32)
    mask = torch.arange(lmax).unsqueeze(0) < event_len.unsqueeze(1)
    return prefix_id, amplitude, mask, event_len


def load_augmented_rlc_event_tensors(imgpath, stage, type="train", lmax=384, dtype_resize=torch.float32):
    if bm is None:
        raise UnsupportedDataset(msg="byte_manip is required for RLC stages")
    if not hasattr(bm, "read_syntax_patches_from_buffer"):
        raise UnsupportedDataset(msg="byte_manip.read_syntax_patches_from_buffer is required for RLC stages")
    jpeg_buffer = load_augmented_mcu_buffer(imgpath, type=type, dtype_resize=dtype_resize)
    syntax = bm.read_syntax_patches_from_buffer(jpeg_buffer)
    normalized_stage = normalize_stage_name(stage)
    return pack_rlc_event_tensors(
        syntax["mcus"],
        lmax=lmax,
        dc_only=(normalized_stage in ("rlc_dc", "rlc_dc_simple", "rlc_dc_attnpool")),
    )


def load_augmented_rlc_event_tensors_2d(imgpath, stage, type="train", lmax=320, dtype_resize=torch.float32):
    if bm is None:
        raise UnsupportedDataset(msg="byte_manip is required for RLC 2D stages")
    if not hasattr(bm, "read_syntax_patches_from_buffer"):
        raise UnsupportedDataset(msg="byte_manip.read_syntax_patches_from_buffer is required for RLC 2D stages")
    jpeg_buffer = load_augmented_mcu_buffer(imgpath, type=type, dtype_resize=dtype_resize)
    syntax = bm.read_syntax_patches_from_buffer(jpeg_buffer)
    normalized_stage = normalize_stage_name(stage)
    run, value, mask, _ = pack_rlc_event_tensors_2d(
        syntax["mcus"],
        lmax=lmax,
        dc_only=(normalized_stage in ("rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat")),
        dtype=dtype_resize,
    )
    return run, value, mask


def load_augmented_rlc_event_tensors_cp(imgpath, stage, type="train", lmax=320, dtype_resize=torch.float32):
    if bm is None:
        raise UnsupportedDataset(msg="byte_manip is required for RLC CP stages")
    if not hasattr(bm, "read_syntax_patches_from_buffer"):
        raise UnsupportedDataset(msg="byte_manip.read_syntax_patches_from_buffer is required for RLC CP stages")
    jpeg_buffer = load_augmented_mcu_buffer(imgpath, type=type, dtype_resize=dtype_resize)
    syntax = bm.read_syntax_patches_from_buffer(jpeg_buffer)
    normalized_stage = normalize_stage_name(stage)
    run_int, value, mask, _ = pack_rlc_event_tensors_cp(
        syntax["mcus"],
        lmax=lmax,
        dc_only=(normalized_stage == "rlc_dc_2d_cp"),
        dtype=dtype_resize,
    )
    return run_int, value, mask


def load_augmented_rlc_event_tensors_direct(
    imgpath, stage, type="train", lmax=320, dtype_resize=torch.float32,
    use_randaugment=True, ops_list=None, num_ops=2, ops_magnitude=3,
):
    normalized_stage = normalize_stage_name(stage)
    dc_only = normalized_stage in (
        "rlc_dc_2d_direct", "rlc_dc_2d_max_direct",
        "rlc_dc_flat_direct", "rlc_dc_2d_cp_direct",
    )
    cp = normalized_stage in RLC_CP_DIRECT_STAGES

    stage_for_loader = "quant_dct" if use_randaugment else None
    _, _quant, y_out, cbcr_out = load_augmented_mcu_coefficients(
        imgpath, type=type, dtype_resize=dtype_resize, stage=stage_for_loader,
        ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
    )
    blocks_zz = _zigzag_blocks_for_rlc(y_out, cbcr_out)
    run, value, mask, _ = pack_rlc_event_tensors_direct(
        blocks_zz, lmax=lmax, dc_only=dc_only, dtype=dtype_resize, cp=cp,
    )
    return run, value, mask


def load_augmented_huffman_event_tensors_direct(
    imgpath, stage, type="train", lmax=320, dtype_resize=torch.float32,
    use_randaugment=True, ops_list=None, num_ops=2, ops_magnitude=3,
):
    normalized_stage = normalize_stage_name(stage)
    if normalized_stage not in (
        "huff_dcac_2d_direct",
        "huff_dcac_flat_direct",
        "event_amp_only_flat_direct",
        *HUFFMAN_CP_DIRECT_STAGES,
    ):
        raise UnsupportedDataset(msg=f"Unsupported Huffman direct stage: {stage}")

    stage_for_loader = "quant_dct" if use_randaugment else None
    _, _quant, y_out, cbcr_out = load_augmented_mcu_coefficients(
        imgpath, type=type, dtype_resize=dtype_resize, stage=stage_for_loader,
        ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
    )
    blocks_zz = _zigzag_blocks_for_rlc(y_out, cbcr_out)
    prefix_id, amplitude, mask, _ = pack_huffman_event_tensors_direct(
        blocks_zz, lmax=lmax, dtype=dtype_resize,
    )
    if normalized_stage not in HUFFMAN_CP_DIRECT_STAGES:
        prefix_id = prefix_id.to(dtype=dtype_resize) / float(HUFFMAN_PREFIX_VOCAB_SIZE - 1)
    return prefix_id, amplitude, mask

MCU_OUTPUT_DIM = torch.tensor([[224, 224]], dtype=torch.int32)


def load_augmented_mcu_coefficients(
    imgpath, type='train', dtype_resize=torch.float32, stage=None,
    ops_list=None, num_ops=2, ops_magnitude=3,
):
    _, quant, Y, cbcr = dm.read_coefficients(imgpath)
    normalized_stage = normalize_stage_name(stage or "raw_byte")
    if normalized_stage == "dct_float":
        coeffs = _dequantize_coefficients(quant, Y, cbcr)
        transform = get_requant_dct_coeff_transform(
            type=type, dtype_resize=dtype_resize,
            ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
        )
        coeffs = transform(coeffs)
        coeffs = ctrans.dct_prep(coeffs, mode='start')
        Y_out = coeffs[0].float().contiguous()
        cbcr_out = coeffs[1].float().contiguous() if len(coeffs) > 1 else None
    elif normalized_stage in QUANT_DCT_DERIVED_STAGES:
        if normalized_stage in DCDIFF_AC_STAGES:
            Y_out, cbcr_out = _apply_dcdiff_requant_pipeline(
                quant, Y, cbcr, type=type, dtype_resize=dtype_resize,
                ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
            )
        else:
            coeffs = _dequantize_coefficients(quant, Y, cbcr)
            transform = get_requant_dct_coeff_transform(
                type=type, dtype_resize=dtype_resize,
                ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude,
            )
            coeffs = transform(coeffs)
            Y_out, cbcr_out = _requantize_coefficients(quant, coeffs)
    else:
        coeffs = (Y, cbcr) if cbcr is not None else (Y,)
        transform = get_mcu_coeff_transform(type=type, dtype_resize=dtype_resize)
        coeffs = transform(coeffs)
        if isinstance(coeffs, tuple):
            Y_out = coeffs[0].short().contiguous()
            cbcr_out = coeffs[1].short().contiguous() if len(coeffs) > 1 else None
        else:
            Y_out = coeffs.short().contiguous()
            cbcr_out = None
    quant = quant.short().contiguous()
    return MCU_OUTPUT_DIM, quant, Y_out, cbcr_out


def load_augmented_mcu_buffer(imgpath, type='train', dtype_resize=torch.float32):
    new_dim, quant, Y_out, cbcr_out = load_augmented_mcu_coefficients(imgpath, type=type, dtype_resize=dtype_resize)
    return dm.write_coefficients_to_buffer(new_dim, quant, Y_out, cbcr_out)


def load_augmented_mcu_records(imgpath, type='train', dtype_resize=torch.float32):
    if bm is None:
        raise UnsupportedDataset(msg="byte_manip is required for raw byte/bitstream stages")
    jpeg_buffer = load_augmented_mcu_buffer(imgpath, type=type, dtype_resize=dtype_resize)
    packed = bm.read_mcu_packed_bytes_from_buffer(jpeg_buffer)
    return packed['mcus']


def coefficients_to_mcu_structured_tensors(imgpath, type='train', lmax=320, dtype_resize=torch.float32):
    return pack_mcu_structured_tensors(
        load_augmented_mcu_records(imgpath, type=type, dtype_resize=dtype_resize),
        lmax=lmax,
    )


def coefficients_to_mcu_payload_tensor(imgpath, type='train', lmax=320, dtype_resize=torch.float32):
    new_dim, quant, Y_out, cbcr_out = load_augmented_mcu_coefficients(imgpath, type=type, dtype_resize=dtype_resize)
    return dm.write_coefficients_to_payload_tensor(new_dim, quant, Y_out, cbcr_out, lmax)


def pack_mcu_payload_tensor(mcu_dicts, lmax=320):
    mcu_count = len(mcu_dicts)
    payload_bytes = torch.zeros((mcu_count, lmax), dtype=torch.uint8)
    for i, mcu in enumerate(mcu_dicts):
        payload = list(mcu['payload_list'])
        n = min(len(payload), lmax)
        if n > 0:
            payload_bytes[i, :n] = torch.tensor(payload[:n], dtype=torch.uint8)
    return payload_bytes


def pack_mcu_structured_tensors(mcu_dicts, lmax=320):
    mcu_count = len(mcu_dicts)
    payload_bytes = torch.zeros((mcu_count, lmax), dtype=torch.uint8)
    byte_mask = torch.zeros((mcu_count, lmax), dtype=torch.bool)
    valid_bits = torch.zeros((mcu_count,), dtype=torch.int32)
    byte_len = torch.zeros((mcu_count,), dtype=torch.int32)
    tail_bits = torch.zeros((mcu_count,), dtype=torch.int32)
    for i, mcu in enumerate(mcu_dicts):
        orig_byte_len = int(mcu['byte_len'])
        valid_bits[i] = min(int(mcu['valid_bits']), lmax * 8)
        byte_len[i] = min(orig_byte_len, lmax)
        tail_bits[i] = int(mcu['tail_bits']) if orig_byte_len <= lmax else int(valid_bits[i].item() % 8)
        payload = list(mcu['payload_list'])
        n = min(len(payload), lmax)
        if n > 0:
            payload_bytes[i, :n] = torch.tensor(payload[:n], dtype=torch.uint8)
            byte_mask[i, :n] = True
    return payload_bytes, byte_mask, valid_bits, byte_len, tail_bits

from utils.mp_scripts import tar_extractor
import random
from functools import partial

import pandas as pd
import numpy as np
import os
import tarfile
import shutil
import sysrsync
import PIL.Image
import torchvision.transforms.functional as TF
from time import time
import logging

class UnsupportedDataset(Exception):
    def __init__(self, msg='Unsupported dataset exception', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class imagenet_tar_handler():
    def __init__(
        self, 
        tarpath:str="Path_above_tar", 
        tmppath:str="/tmp/tempdata/imagenet",
        tmpprefix:str="imgnet_data",
        tarnames=['ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'], 
        valprepname='valprep.sh',
        num_proc=4,
        verbose=True
        ):
        """
        tarpath: Path containing original tar files and valprep.sh file
        tmppath: Path to temporary dataset (parent to 'imgnet_data' and 'tarfiles')
        tmpprefix: Prefix for unpacked imagenet data. imagenet data will be unpacked to: <tmppath>/<tmpprefix>/<train or val>/*
        tarnames: [<train tar file name>, <val tar file name>]
        valprepname: .sh file that contains the information for processing val data. Can be downloaded from https://github.com/soumith/imagenet-multiGPU.torch -- moves validation images to appropriate subfolders.
        num_proc: number of processes to use when unpacking
        verbose: If True, prints progress
        """
        self.tarpath=tarpath
        self.tmppath=tmppath
        self.tmpprefix=tmpprefix
        self.originaldir = os.getcwd() # directory to go back to after operation
        self.tarnames=tarnames
        self.valprepname=valprepname
        self.num_proc=num_proc
        self.verbose=verbose

    def checkfile(self):
        """
        Method that checks if there are correct files in .tar path
        returns True if verified, False if not.
        """
        check_target = len(self.tarnames) + 1
        check_count = 0
        check_files = self.tarnames.copy()
        check_files.append(self.valprepname)
        for file in os.listdir(self.tarpath):
            if file in check_files:
                check_count += 1
        
        if check_count == check_target: 
            logging.info(f"All files verified: {check_files}")
            return True
        else:
            logging.info(f"Files not verified. Files in tarpath: {os.listdir(self.tarpath)}")
            return False

    def tar2tmp(self, copy=True, unpack=True, extract_train=True, extract_val=True):
        """
        Method that copies tar file to tmppath and optionally unpacks them.
        Uses rsync to show progress bar
        
        inputs
            copy: If true, copy tar files from source to destination
            unpack: If true, unpack .tar files after copying
            extract_train: If true, extract <train>.tar
            extract_val: If true, extract <val>.tar

        returns
            1: copying done (no unpacking)
            2: unpacking done
        """
        if copy:
            copy_starttime = time()
            if not os.path.isdir(os.path.join(self.tmppath, 'tarfile')):
                os.makedirs(os.path.join(self.tmppath, 'tarfile')) # /<TMP>/tarfile/
            for tarname in self.tarnames:
                tarfilepath = os.path.join(self.tarpath, tarname) # source (/<SRC>/tarfile.tar)
                tmpfilepath = os.path.join(
                    os.path.join(self.tmppath, 'tarfile')) # dest (/<TMP>/tarfile/)
                sysrsync.run(
                    source=tarfilepath, 
                    destination=tmpfilepath, 
                    options=['--progress']) # copy source to dest
                if self.verbose:
                    logging.info(f"File {tarname} copied to {self.tmppath}")
            copy_endtime=time()
            copytime = copy_endtime - copy_starttime
            if self.verbose:
                logging.info(f"Copying done in {copytime//60:2.0f}m {copytime%60:2.0f}s")


        if unpack:
            unpack_starttime=time()
            if self.tmpprefix != "":
                imgnet_trainpath = os.path.join(self.tmppath, f'{self.tmpprefix}/train') # /<TMP>/<tmpprefix>/train
                imgnet_valpath = os.path.join(self.tmppath, f'{self.tmpprefix}/val') # /<TMP>/<tmpprefix>/val
            else:
                imgnet_trainpath = os.path.join(self.tmppath, 'train') # /<TMP>/train
                imgnet_valpath = os.path.join(self.tmppath, 'val') # /<TMP>/val
            if not os.path.isdir(imgnet_trainpath):
                os.makedirs(imgnet_trainpath)
            if not os.path.isdir(imgnet_valpath):
                os.makedirs(imgnet_valpath)
            sysrsync.run(
                source=os.path.join(self.tarpath, self.valprepname),
                destination=imgnet_valpath, 
                options=['--progress']
            ) # copy valprep.sh to imgnet_valpath
            if self.verbose:
                logging.info(f"File {self.valprepname} copied to {imgnet_valpath}")

            if extract_train:
                if copy:
                    train_tar = tarfile.open(os.path.join(self.tmppath, f'tarfile/{self.tarnames[0]}')) # train data handling
                else:
                    train_tar = tarfile.open(os.path.join(self.tarpath, self.tarnames[0])) # train data handling
                if self.verbose:
                    logging.info(f"Extracting {os.path.join(self.tmppath, f'tarfile/{self.tarnames[0]}')} -- can take up to 1hr")
                train_tar.extractall(path=imgnet_trainpath)
            logging.info(f"Train tar files extracted. Now extracting individual tar files...")
            tarfilelist = [] # list of tar files
            for _, _, files in os.walk(imgnet_trainpath): # extract child .tar files
                for filename in files:
                    if filename[-4:].lower()=='.tar':
                        tarfilelist.append(filename)
            tarextract = tar_extractor(self.num_proc, tarfilelist, imgnet_trainpath, self.verbose)
            tarextract.start_processes()
            tarextract.wait_and_terminate()
            logging.info(f"Extracting training data done.")
            if extract_val:
                if copy:
                    val_tar = tarfile.open(os.path.join(self.tmppath, f'tarfile/{self.tarnames[1]}')) # val data handling
                else:
                    val_tar = tarfile.open(os.path.join(self.tarpath, self.tarnames[1])) # val data handling
                if self.verbose:
                    logging.info(f"Extracting {os.path.join(self.tmppath, f'tarfile/{self.tarnames[1]}')} -- can take up to 1hr")
                val_tar.extractall(path=imgnet_valpath)
            os.chdir(imgnet_valpath) # move to imgnet valpath directory
            os.system(f"sh {self.valprepname}") # run shell script
            os.remove(self.valprepname)
            os.chdir(self.originaldir) # go back to original directory
            logging.info(f"Extracting validation data done.")
            unpack_endtime=time()
            unpacktime = unpack_endtime - unpack_starttime
            if self.verbose:
                logging.info(f"Unpacking done in {unpacktime//60:2.0f}m {unpacktime%60:2.0f}")
            return 2
        else:
            return 1
    
    def remove_data(self):
        logging.warning("=== WARNING: Deleting path: {} ===".format(self.tmppath))
        shutil.rmtree(self.tmppath)
        logging.info("=== Path: {} deleted ===".format(self.tmppath))
        
def run_msrsync(
    msrsync_path='./msrsync',
    source='source_folder',
    dest='destination_folder',
    bucketpath="./msrsync_temp",
    process=1,
    verbose=True,
    ):
    """
    Runs msrsync from shell
    msrsync_path: path to msrsync executable
    source: source folder
    dest: destination folder (source folder gets copied as a child folder into the destination folder)
    bucketpath: path to msrsync bucket
    process: number of processes ro run
    verbose: if true, shows progress
    """
    if not os.path.isdir(bucketpath):
        os.makedirs(bucketpath)
    bucketpath = f" -b {bucketpath}"
        
    show_progress = " -P" if verbose else ""

    command = f"{msrsync_path} -p {process}{show_progress}{bucketpath} {source} {dest}"
    if not os.path.isdir(dest):
        os.makedirs(dest) # make directory to destination
    if verbose:
        print(f"[msrsync] copying {source} to {dest}")
    os.system(command) # runs command

def transcode_to_jpeg(
    filepath='something.jpeg',
):
    """
    Transcodes non-jpeg image to jpeg

    Args:
        filepath: path to not-jpeg file to transcode
    """
    notjpeg_image = PIL.Image.open(filepath).convert("RGB")
    filepath_basename = os.path.basename(filepath).split('.')[0] # filename without extension
    filepath_basename = filepath_basename + ".JPEG" # make it jpeg
    savefilename = os.path.join(
        os.path.dirname(filepath), filepath_basename
    )
    os.remove(filepath) # delete original file
    notjpeg_image.save(savefilename, 'jpeg') # write to jpeg


class imagenet_dataset_indexing(torch.utils.data.Dataset):
    """
    ImageNet folder structure should be: 
        /<turbo_path>/shared_datasets/imagenet
            - train
                - n01440764
                    - n01440764_10026.JPEG
                    - ...
                - ...
            - test
                - n01440764
                    - ILSVRC2012_val_00000293.JPEG
                    - ...
                - ...
    ImageNet label mapping from: https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57 by aaronpolhamus
    
    Imagenet Dataloader
    inputs
    - indexfile: .csv file that contains the indexing for imagenet dataset (format: [<filepath>, <label>])
    - labelmapfile: .txt file that maps labelname to label (txt line format: 'n02119789 1 kit_fox')
    - type: 'train' or 'test' for transform selection
    - baseindex: If True, use base index format (<path_to_trainval>/train/<class>/img.jpg). Otherwise, use raw index format (<path>/img.jpg)
    - load_mode: 'RGB' or 'DCT'. If RGB, decode jpeg into RGB. If DCT, use dct_manip library to decode until DCT coeff.
    - basepath: path containing train/val for imagenet data
    - dtype: data type
    - rank: GPU rank (== gpu device index)

    outputs
    - torch.utils.data.Dataset object for ImageNet
    """
    def __init__(
        self, 
        indexfile='path_to_indexfile.csv', 
        type='train', 
        basepath='', 
        load_mode='MCU_BYTES',
        stage=None,
        rlc_lmax=320,
        ops_list=None,
        num_ops=2,
        ops_magnitude=3,
        freqnorm=False,
        freqnorm_stats=None,
        dtype=torch.float32,
        rank=-1):

        super(imagenet_dataset_indexing).__init__()
        self.imageindex = pd.read_csv(indexfile).to_numpy() # format: [<filepath>, <label>]
        self.type = type
        self.dtype = dtype
        self.basepath = basepath
        self.load_mode = load_mode
        self.stage = normalize_stage_name(stage or load_mode)
        self.rlc_lmax = rlc_lmax
        self.ops_list = ops_list
        self.num_ops = num_ops
        self.ops_magnitude = ops_magnitude
        self.freqnorm = freqnorm
        self.freqnorm_stats = freqnorm_stats
        self.transform = None # this will be handeled separately at SubsetWithTransform() class or get_transform() function
        self.rank = rank

    def __getitem__(self, index):
        # path_to_img = self.imageindex[index, 0] # not part of an actual code (can be though) -- just here to help understand what's going on
        # label = self.imageindex[index, 1]
        if self.basepath != '':
            imgpath = os.path.join(self.basepath, self.imageindex[index, 0])
        else:
            imgpath = self.imageindex[index, 0]
        stage = self.stage
        if stage in ("rgb", "ycbcr"):
            data = load_stage_image_tensor(imgpath, stage, type=self.type)
        elif stage in GROUP_DCT_STAGES:
            data = load_stage_dct_group_tensors(
                imgpath, stage, type=self.type, dtype_resize=self.dtype,
                ops_list=self.ops_list, num_ops=self.num_ops, ops_magnitude=self.ops_magnitude,
                freqnorm=self.freqnorm, freqnorm_stats=self.freqnorm_stats,
            )
        elif stage in NUMERIC_DCT_STAGES:
            data = load_stage_dct_features(
                imgpath, stage, type=self.type, dtype_resize=self.dtype,
                ops_list=self.ops_list, num_ops=self.num_ops, ops_magnitude=self.ops_magnitude,
                freqnorm=self.freqnorm, freqnorm_stats=self.freqnorm_stats,
            )
        elif stage in ("raw_byte", "bitstream"):
            payload_bytes, byte_mask, _, _, _ = coefficients_to_mcu_structured_tensors(
                imgpath, type=self.type, lmax=self.rlc_lmax, dtype_resize=self.dtype,
            )
            data = (payload_bytes, byte_mask)
        elif stage in RLC_CP_DIRECT_STAGES:
            rlc_lmax = 8 if stage == "rlc_dc_2d_cp_direct" else self.rlc_lmax
            data = load_augmented_rlc_event_tensors_direct(
                imgpath, stage, type=self.type, lmax=rlc_lmax, dtype_resize=self.dtype,
                use_randaugment=getattr(self, "rlc_direct_randaug", True),
                ops_list=self.ops_list, num_ops=self.num_ops, ops_magnitude=self.ops_magnitude,
            )
        elif stage in (
            "huff_dcac_2d_direct",
            "huff_dcac_flat_direct",
            "event_amp_only_flat_direct",
            *HUFFMAN_CP_DIRECT_STAGES,
        ):
            data = load_augmented_huffman_event_tensors_direct(
                imgpath, stage, type=self.type, lmax=self.rlc_lmax, dtype_resize=self.dtype,
                use_randaugment=getattr(self, "rlc_direct_randaug", True),
                ops_list=self.ops_list, num_ops=self.num_ops, ops_magnitude=self.ops_magnitude,
            )
        elif stage in (
            "rlc_dc_2d_direct", "rlc_dcac_2d_direct",
            "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct",
            "rlc_dc_flat_direct", "rlc_dcac_flat_direct",
        ):
            rlc_lmax = 8 if stage in ("rlc_dc_2d_direct", "rlc_dc_2d_max_direct", "rlc_dc_flat_direct") else self.rlc_lmax
            data = load_augmented_rlc_event_tensors_direct(
                imgpath, stage, type=self.type, lmax=rlc_lmax, dtype_resize=self.dtype,
                use_randaugment=getattr(self, "rlc_direct_randaug", True),
                ops_list=self.ops_list, num_ops=self.num_ops, ops_magnitude=self.ops_magnitude,
            )
        elif stage in ("rlc_dc_2d_cp", "rlc_dcac_2d_cp"):
            rlc_lmax = 8 if stage == "rlc_dc_2d_cp" else self.rlc_lmax
            data = load_augmented_rlc_event_tensors_cp(
                imgpath, stage, type=self.type, lmax=rlc_lmax, dtype_resize=self.dtype,
            )
        elif stage in ("rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max", "rlc_dc_flat", "rlc_dcac_flat"):
            rlc_lmax = 8 if stage in ("rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat") else self.rlc_lmax
            data = load_augmented_rlc_event_tensors_2d(
                imgpath, stage, type=self.type, lmax=rlc_lmax, dtype_resize=self.dtype,
            )
        elif stage in ("rlc_dc", "rlc_dcac", "rlc_dc_simple", "rlc_dcac_simple", "rlc_dc_attnpool", "rlc_dcac_attnpool"):
            rlc_lmax = 8 if stage in ("rlc_dc", "rlc_dc_simple", "rlc_dc_attnpool") else self.rlc_lmax
            rlc_tensors = load_augmented_rlc_event_tensors(
                imgpath, stage, type=self.type, lmax=rlc_lmax, dtype_resize=self.dtype,
            )
            data = rlc_tensors[:8]
        else:
            raise UnsupportedDataset(msg=f"Unsupported JPEG stage: {stage}")
        return data, int(self.imageindex[index, 1])

    def __len__(self):
        return len(self.imageindex)

def get_transform(dataset='imagenet', type='train', ops_list=None, 
    num_ops=2, ops_magnitude=10, 
    dtype=torch.float32, dtype_resize=torch.float32):
    """
    Stage inputs either load an already-transformed image tensor or apply JPEG/DCT
    augmentation before byte/event packing. DataLoader-level transforms are empty.
    """
    dataset_to_stage(dataset)
    return None

def optimized_stage_collate_fn(batch):
    """
    Optimized collate function for MCU byte tensors.
    
    Args:
        batch: List of (stage_data, label) tuples.
    
    Returns:
        (data_batch, label_batch)
    """
    # Unpack batch
    data_list = []
    labels = []
    for item in batch:
        data, label = item
        data_list.append(data)
        labels.append(label)
    
    # Use default collate for complex structures
    # This handles both tensor stacking and tuple structures gracefully
    try:
        # Try standard torch collate first (works for tensors)
        from torch.utils.data._utils.collate import default_collate
        batch_data = default_collate(data_list)
    except:
        # Fallback for mixed/complex data structures
        batch_data = data_list
    
    # Stack labels
    label_batch = torch.tensor(labels, dtype=torch.int64)
    
    return batch_data, label_batch


optimized_mcu_collate_fn = optimized_stage_collate_fn


class SubsetWithTransform(torch.utils.data.Dataset):
    """
    Custom subset class which allows to customize transform for each subsets got from random_split()
    """
    def __init__(self, subset, dataset=None, transform=None):
        self.subset = subset
        self.transform = transform
        self.dataset_name = dataset
    
    def __getitem__(self, index):
        img, lab = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, lab
    
    def __len__(self):
        return len(self.subset)

def set_seeds_for_data(seed=11997733):
    """
    Set seeds for Python, numpy, and pytorch. Used to split the dataset consistantly across DDP instances.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_seeds_for_worker(seed=11997733, id=0):
    """
    Set seeds for python, and numpy. Default seed=11997733.
    PyTorch seeding is handled by torch.Generator passed into the DataLoader
    """
    seed = seed % (2**31)
    random.seed(seed+id)
    np.random.seed(seed+id)

def worker_seed_reporter(id=None):
    """
    Reports worker seeds
    """
    workerseed = torch.utils.data.get_worker_info().seed
    numwkr = torch.utils.data.get_worker_info().num_workers
    baseseed = torch.initial_seed()
    logging.info(f"Worker id: {id}/{numwkr}, worker seed: {workerseed}, baseseed: {baseseed}, workerseed % 2**31: {workerseed % (2**31)}")

def set_seeds_and_report(seed=11997733, report=False, id=0):
    """
    Debug: set seeds and report
    """
    workerseed = torch.utils.data.get_worker_info().seed
    set_seeds_for_worker(workerseed, id)
    if report:
        worker_seed_reporter(id)

def dataset_selector(
    dataset='imagenet_mcu_bytes',
    type='train',
    indexpath='', 
    basepath='',

    batch_size=128, 
    num_workers=4, 
    shuffle=False, 
    trainval_split=-1,
    return_indices=False,

    distributed=False,
    rank=-1,
    world_size=-1,
    seed=None,
    
    ops_list=None,
    num_ops=2,
    ops_magnitude=10,
    freqnorm=False,
    freqnorm_stats=None,
    dtype=torch.float32,
    rlc_lmax=320,
    prefetch_factor=4,
    in_order=True,
    ):
    """
        Selects dataset and returns dataloader
        input
        - dataset: selects dataset
        - type: 'train' or 'test'
        - indexpath: path to indexfile
        - basepath: path containing train/val for imagenet data

        - batch_size: size of batch
        - num_workers: number of subprocesses to spawn
        - shuffle: shuffle dataset or not
        - trainval_split: split dataset if specified.
        - return_indices: if True, returns the split index (for reproducing results if seeding isn't consistant across machines/versions)

        - distributed: flag to whether it should return distributed dataloader or not (True if using Distributed Data Parallel)
        - rank: index of the current distributed process (index of a gpu)
        - world_size: size of the process group (# of gpus)
        - seed: set seeds to this value if not None

        - ops_list: list of operation name strings used by MCU coefficient augmentation before byte packing
        - num_ops: number of operations to apply
        - ops_magnitude: magnitude of randaugment (default:10 / max 10)
        - dtype: dtype of dataset
        - in_order: If False, allow DataLoader workers to return ready batches out of order

        output
        - (Optional) Tuple of ('trainset', 'valset')
        - 'testset'
    """
    if dataset not in JPEG_STAGE_DATASETS:
        raise UnsupportedDataset(msg="Unsupported dataset: {}".format(dataset))
    stage = dataset_to_stage(dataset)

    dataset_object = imagenet_dataset_indexing(
        indexfile=indexpath, type=type,
        basepath=basepath, load_mode=stage, stage=stage, dtype=dtype,
        rlc_lmax=rlc_lmax,
        ops_list=ops_list if type == 'train' else None,
        num_ops=num_ops,
        ops_magnitude=ops_magnitude,
        freqnorm=freqnorm,
        freqnorm_stats=freqnorm_stats,
    )
    use_persistent_workers = True if not num_workers==0 else False
    eval_num_workers = min(num_workers, 4) if num_workers != 0 else 0
    eval_prefetch_factor = 1
    use_eval_persistent_workers = True if eval_num_workers != 0 else False
    if seed is not None:
        seedftn = partial(set_seeds_and_report, seed+rank, False) #report=False
        seed_generator = torch.Generator(device='cpu')
        seed_generator.manual_seed(seed+rank)
    else:
        seedftn = None
        seed_generator = None
    if trainval_split > 0: # train/val split
        set_seeds_for_data(seed) # Set same seeds for dataset split

        dataset_len = len(dataset_object) # whole dataset length
        vallen = int(trainval_split * dataset_len) # validation length
        trainlen = dataset_len - vallen # train length
        traindata_split, valdata_split = torch.utils.data.random_split(dataset_object, (trainlen, vallen))
        eval_dataset_object = imagenet_dataset_indexing(
            indexfile=indexpath, type='val',
            basepath=basepath, load_mode=stage, stage=stage, dtype=dtype,
            rlc_lmax=rlc_lmax,
            ops_list=None,
            num_ops=num_ops,
            ops_magnitude=ops_magnitude,
            freqnorm=freqnorm,
            freqnorm_stats=freqnorm_stats,
        )
        valdata_split = torch.utils.data.Subset(eval_dataset_object, valdata_split.indices)
        trainval_len = round(trainlen * 0.05)
        trainvaldata_split = torch.utils.data.Subset(eval_dataset_object, traindata_split.indices[:trainval_len]) # also evaluate model performance on 5% train data

        set_seeds_for_data(seed+rank) # after dataset is split, use different seeds for augmentations and shuffling.

        traindata_subset = SubsetWithTransform(traindata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='train', ops_list=ops_list, num_ops=num_ops, ops_magnitude=ops_magnitude, dtype=dtype))
        valdata_subset = SubsetWithTransform(valdata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='val', dtype=dtype))
        trainvaldata_subset = SubsetWithTransform(trainvaldata_split, dataset=dataset, transform=get_transform(dataset=dataset, type='val', dtype=dtype))
        train_subset_index = traindata_split.indices
        val_subset_index = valdata_split.indices

        train_sampler=None
        val_sampler=None
        trainval_sampler=None
        if distributed:
            train_sampler = DistributedSampler(
                traindata_subset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False,
                )
            val_sampler = DistributedEvalSampler(
                valdata_subset, num_replicas=world_size, rank=rank, shuffle=False,
                ) # don't distribute val
            trainval_sampler = DistributedEvalSampler(
                trainvaldata_subset, num_replicas=world_size, rank=rank, shuffle=False
            )
        trainloader_obj = torch.utils.data.DataLoader(traindata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False if distributed else shuffle, num_workers=num_workers, 
                                                sampler=train_sampler, prefetch_factor=None if num_workers==0 else prefetch_factor, 
                                                persistent_workers=use_persistent_workers,
                                                collate_fn = optimized_stage_collate_fn, worker_init_fn=seedftn, generator=seed_generator,
                                                in_order=in_order)
        valloader_obj = torch.utils.data.DataLoader(valdata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False, num_workers=eval_num_workers, sampler=val_sampler,
                                                prefetch_factor=None if eval_num_workers==0 else eval_prefetch_factor, 
                                                persistent_workers=use_eval_persistent_workers,
                                                collate_fn = optimized_stage_collate_fn, worker_init_fn=seedftn, generator=seed_generator,
                                                in_order=in_order)
        trainvalloader_obj = torch.utils.data.DataLoader(trainvaldata_subset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False, num_workers=eval_num_workers, sampler=trainval_sampler,
                                                prefetch_factor=None if eval_num_workers==0 else eval_prefetch_factor, 
                                                persistent_workers=use_eval_persistent_workers,
                                                collate_fn = optimized_stage_collate_fn, worker_init_fn=seedftn, generator=seed_generator,
                                                in_order=in_order)

        if rank==0:
            logging.info("Train/Val split: num_total: {}, num_train: {}, num_val: {}, num_trainval: {}".format(dataset_len, trainlen, vallen, round(trainlen*0.05)))

        if not return_indices:
            return trainloader_obj, valloader_obj, trainvalloader_obj
        else:
            return trainloader_obj, valloader_obj, trainvalloader_obj, train_subset_index, val_subset_index
    else:
        sampler=None
        datasubset = SubsetWithTransform(dataset_object, dataset=dataset, transform=get_transform(dataset=dataset, type=type, ops_list=ops_list if type=='train' else None, num_ops=num_ops, ops_magnitude=ops_magnitude, dtype=dtype))
        loader_num_workers = num_workers if type=='train' else eval_num_workers
        loader_prefetch_factor = prefetch_factor if type=='train' else eval_prefetch_factor
        loader_persistent_workers = use_persistent_workers if type=='train' else use_eval_persistent_workers
        if distributed:
            if type=='train':
                sampler = DistributedSampler(
                    datasubset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False,
                    )
            else:
                sampler = DistributedEvalSampler(
                    datasubset, num_replicas=world_size, rank=rank, shuffle=shuffle,
                )
        dataloader_obj = torch.utils.data.DataLoader(datasubset, batch_size=batch_size, pin_memory=True,
                                                shuffle=False if distributed else shuffle, num_workers=loader_num_workers, 
                                                sampler=sampler, prefetch_factor=None if loader_num_workers==0 else loader_prefetch_factor, 
                                                persistent_workers=loader_persistent_workers,
                                                collate_fn = optimized_stage_collate_fn, worker_init_fn=seedftn, generator=seed_generator,
                                                in_order=in_order)
        return dataloader_obj
