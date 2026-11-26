from yacs.config import CfgNode as CN


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
GROUP_DCT_STAGES = {
    "dct_float",
    "quant_dct",
    "quant_dct_group",
    "ac_only",
    "dc_diff_ac",
    "dc_diff_ac_group",
}
DCDIFF_AC_STAGES = {"dc_diff_ac", "dc_diff_ac_group"}
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
DEFAULT_NUMERIC_FEATURE_DIMS = {
    "dct_float": 384,
    "quant_dct": 384,
    "ac_only": 384,
    "quant_dct_zigzag_linear": 384,
    "dc_abs": 6,
    "dc_diff": 6,
    "dc_diff_ac": 384,
    "quant_dct_group": 384,
    "dc_diff_ac_group": 384,
    "dc_absdiff": 12,
    "rlc_dc_2d": 64,
    "rlc_dcac_2d": 64,
    "rlc_dc_2d_max": 64,
    "rlc_dcac_2d_max": 64,
    "rlc_dc_2d_cp": 64,
    "rlc_dcac_2d_cp": 64,
    "rlc_dc_2d_direct": 64,
    "rlc_dcac_2d_direct": 64,
    "rlc_dc_2d_max_direct": 64,
    "rlc_dcac_2d_max_direct": 64,
    "rlc_dc_2d_cp_direct": 64,
    "rlc_dcac_2d_cp_direct": 64,
    "huff_dcac_2d_direct": 64,
    "huff_dcac_flat_direct": 64,
    "event_amp_only_flat_direct": 64,
    "huff_dcac_2d_cp_direct": 64,
    "rlc_dcac_2d_cp_concat_mean_direct": 64,
    "rlc_dcac_2d_cp_add_conv_direct": 64,
    "rlc_dcac_2d_cp_concat_conv_direct": 64,
    "rlc_dcac_2d_cp_film_conv_direct": 64,
    "huff_dcac_2d_cp_concat_mean_direct": 64,
    "huff_dcac_2d_cp_add_conv_direct": 64,
    "huff_dcac_2d_cp_concat_conv_direct": 64,
    "huff_dcac_2d_cp_film_conv_direct": 64,
}

QUANT_DCT_REQUANT_AUGLIST = (
    "AutoContrast,Posterize,SolarizeAdd,Color,Contrast,Brightness,"
    "MidfreqAug,Cutout,TranslateX,TranslateY,Rotate90,AutoSaturation,"
    "Grayscale,ChromaDrop"
).split(",")
DCDIFF_AC_AUGLIST = QUANT_DCT_REQUANT_AUGLIST + [
    "BlockDropout",
    "FreqBandDropout",
    "ACQuantNoise",
]


def normalize_stage_name(stage):
    key = str(stage).lower()
    if key not in STAGE_ALIASES:
        raise ValueError(f"Unsupported JPEG stage: {stage}. Supported: {sorted(STAGE_ALIASES)}")
    return STAGE_ALIASES[key]


def stage_to_dataset(stage):
    stage = normalize_stage_name(stage)
    return "imagenet_mcu_bytes" if stage == "raw_byte" else f"imagenet_{stage}"


CFG = CN()
CFG.SEED = 11997733
CFG.MSRSYNC = False

CFG.MODEL = CN()
CFG.MODEL.ARCH = "vits"
CFG.MODEL.DOMAIN = "MCU_BYTES"
CFG.MODEL.STAGE = "raw_byte"
CFG.MODEL.PATCHSIZE = 16
CFG.MODEL.VERSION = 3
CFG.MODEL.SUBBLOCK = False
CFG.MODEL.CLASSES = 1000
CFG.MODEL.DTYPE = "fp32"
CFG.MODEL.AMPDTYPE = "fp16"
CFG.MODEL.SUMCOL = ["output_size", "num_params"]
CFG.MODEL.SUMDTYPE = ["fp32"]
CFG.MODEL.INPUTSIZE = (1, 196, 320)
CFG.MODEL.NUMERIC_FEATURE_DIM = 384
CFG.MODEL.RLC_LMAX = 320
CFG.MODEL.HEADS = 6
CFG.MODEL.HEADSIZE = 64
CFG.MODEL.EMBEDSIZE = 384
CFG.MODEL.DEPTH = 12

CFG.TRAIN = CN()
CFG.TRAIN.EPOCHS = 90
CFG.TRAIN.SCHEDULER_EPOCHS = -1
CFG.TRAIN.BATCHSIZE = 1024
CFG.TRAIN.LR = 1e-3
CFG.TRAIN.WD = 1e-4
CFG.TRAIN.DROP = 0.0
CFG.TRAIN.WARMUP = 10000
CFG.TRAIN.PREFETCH_FACTOR = 2
CFG.TRAIN.IN_ORDER = True
CFG.TRAIN.AUGLIST = "AutoContrast,Posterize,Color,Contrast,Brightness,Sharpness,Cutout,TranslateX,TranslateY,Rotate90,AutoSaturation,Grayscale,MidfreqAug,ChromaDrop".split(",")
CFG.TRAIN.NUMOPS = 2
CFG.TRAIN.AUGSTR = 3
CFG.TRAIN.AUGMAX = 10
CFG.TRAIN.SPLIT = 0.01
CFG.TRAIN.DETERMINISTIC = False
CFG.TRAIN.AMP = False
CFG.TRAIN.DATASET = "imagenet_mcu_bytes"
CFG.TRAIN.BATCHPERGPU = 128
CFG.TRAIN.RUNTRAIN = True
CFG.TRAIN.RUNEVAL = True
CFG.TRAIN.GRADACCUMSTEPS = 1
CFG.TRAIN.FREQNORM = False
CFG.TRAIN.FREQNORM_STATS = "outputs/dct_stats/imagenet1k_train_dct_freq_stats.pt"

CFG.TRAIN.SCALER = CN()
CFG.TRAIN.SCALER.GROWTH = 1.6
CFG.TRAIN.SCALER.BACKOFF = 0.625
CFG.TRAIN.SCALER.INTERVAL = 600

CFG.RANK = 0
CFG.WORLDSIZE = 1
CFG.THREADS = 8


def generate_config(
    modelarch="vits", domain="mcu_bytes", stage=None, modelver=None, subblock=None,
    classes=None, numeric_feature_dim=None, rlc_lmax=None, scheduler_epochs=None,
    epochs=None, batchsize=None, lr=None, wd=None, drop=None, warmup_steps=None, auglist=None,
    num_ops=None, ops_magnitude=None, augstr=None, seed=None, amp=None, ampdtype=None, use_msrsync=None,
    freqnorm=None, freqnorm_stats=None, prefetch_factor=None, in_order=None,
):
    cfg = CFG.clone()
    cfg.MODEL.ARCH = modelarch.lower()
    cfg.MODEL.DOMAIN = domain.upper()
    cfg.MODEL.STAGE = normalize_stage_name(stage or domain)
    cfg.TRAIN.DATASET = stage_to_dataset(cfg.MODEL.STAGE)
    if cfg.MODEL.STAGE == "dct_float":
        cfg.TRAIN.AUGLIST = QUANT_DCT_REQUANT_AUGLIST.copy()
    elif cfg.MODEL.STAGE in DCDIFF_AC_STAGES:
        cfg.TRAIN.AUGLIST = DCDIFF_AC_AUGLIST.copy()
    elif cfg.MODEL.STAGE in QUANT_DCT_DERIVED_STAGES:
        cfg.TRAIN.AUGLIST = QUANT_DCT_REQUANT_AUGLIST.copy()
    if cfg.MODEL.STAGE in DEFAULT_NUMERIC_FEATURE_DIMS:
        cfg.MODEL.NUMERIC_FEATURE_DIM = DEFAULT_NUMERIC_FEATURE_DIMS[cfg.MODEL.STAGE]
    if cfg.MODEL.STAGE in (
        "rlc_dc_simple", "rlc_dc_attnpool",
        "rlc_dc_2d", "rlc_dc_2d_max", "rlc_dc_flat", "rlc_dc_2d_cp",
        "rlc_dc_2d_direct", "rlc_dc_2d_max_direct", "rlc_dc_flat_direct", "rlc_dc_2d_cp_direct",
    ):
        cfg.MODEL.RLC_LMAX = 8

    if cfg.MODEL.ARCH == "vitti":
        cfg.MODEL.HEADS = 3
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 192
        cfg.MODEL.DEPTH = 12
        cfg.TRAIN.EPOCHS = 300
        cfg.TRAIN.WD = 1e-4
        cfg.TRAIN.AMP = False
    elif cfg.MODEL.ARCH == "vits":
        cfg.MODEL.HEADS = 6
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 384
        cfg.MODEL.DEPTH = 12
        cfg.TRAIN.EPOCHS = 90
        cfg.TRAIN.AMP = False
    elif cfg.MODEL.ARCH == "vitb":
        cfg.MODEL.HEADS = 12
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 768
        cfg.MODEL.DEPTH = 12
        cfg.TRAIN.AMP = True
        cfg.MODEL.AMPDTYPE = "bf16"
        cfg.TRAIN.LR = 1e-3
        cfg.TRAIN.WD = 1e-4
        cfg.TRAIN.BATCHSIZE = 512
    elif cfg.MODEL.ARCH == "vitl":
        cfg.MODEL.HEADS = 12
        cfg.MODEL.HEADSIZE = 64
        cfg.MODEL.EMBEDSIZE = 1024
        cfg.MODEL.DEPTH = 24
        cfg.TRAIN.AMP = True
        cfg.MODEL.AMPDTYPE = "bf16"
    else:
        raise ValueError(f"Unsupported MCU ViT architecture: {modelarch}")

    if modelver is not None:
        cfg.MODEL.VERSION = modelver
    if subblock is not None:
        cfg.MODEL.SUBBLOCK = subblock
    if classes is not None:
        cfg.MODEL.CLASSES = classes
    if numeric_feature_dim is not None:
        cfg.MODEL.NUMERIC_FEATURE_DIM = numeric_feature_dim
    if rlc_lmax is not None:
        cfg.MODEL.RLC_LMAX = rlc_lmax
    if scheduler_epochs is not None:
        cfg.TRAIN.SCHEDULER_EPOCHS = scheduler_epochs
    if epochs is not None:
        cfg.TRAIN.EPOCHS = epochs
    if batchsize is not None:
        cfg.TRAIN.BATCHSIZE = batchsize
    if lr is not None:
        cfg.TRAIN.LR = lr
    if wd is not None:
        cfg.TRAIN.WD = wd
    if drop is not None:
        cfg.TRAIN.DROP = drop
    if warmup_steps is not None:
        cfg.TRAIN.WARMUP = warmup_steps
    if num_ops is not None:
        cfg.TRAIN.NUMOPS = num_ops
    if ops_magnitude is not None:
        cfg.TRAIN.AUGSTR = ops_magnitude
    if prefetch_factor is not None:
        cfg.TRAIN.PREFETCH_FACTOR = prefetch_factor
    if in_order is not None:
        cfg.TRAIN.IN_ORDER = bool(in_order)
    if augstr is not None:
        cfg.TRAIN.AUGMAX = augstr
    if seed is not None:
        cfg.SEED = seed
    if auglist is not None:
        cfg.TRAIN.AUGLIST = auglist.split(",") if isinstance(auglist, str) else auglist
    if amp is not None:
        cfg.TRAIN.AMP = bool(amp)
    if ampdtype is not None:
        cfg.MODEL.AMPDTYPE = ampdtype
    if use_msrsync is not None:
        cfg.MSRSYNC = use_msrsync
    if freqnorm is not None:
        cfg.TRAIN.FREQNORM = bool(freqnorm)
    if freqnorm_stats is not None:
        cfg.TRAIN.FREQNORM_STATS = freqnorm_stats

    return cfg
