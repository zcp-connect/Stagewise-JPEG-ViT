import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.plainvit import ViT


RLC_CP_VARIANT_STAGES = {
    "rlc_dcac_2d_cp_concat_mean_direct",
    "rlc_dcac_2d_cp_add_conv_direct",
    "rlc_dcac_2d_cp_concat_conv_direct",
    "rlc_dcac_2d_cp_film_conv_direct",
}

HUFFMAN_CP_VARIANT_STAGES = {
    "huff_dcac_2d_cp_concat_mean_direct",
    "huff_dcac_2d_cp_add_conv_direct",
    "huff_dcac_2d_cp_concat_conv_direct",
    "huff_dcac_2d_cp_film_conv_direct",
}


NUMERIC_FEATURE_DIMS = {
    "dct_float": 384,
    "quant_dct_zigzag_linear": 384,
    "dc_abs": 6,
    "dc_diff": 6,
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
    "huff_dcac_2d_cp_direct": 64,
    **{stage: 64 for stage in RLC_CP_VARIANT_STAGES},
    **{stage: 64 for stage in HUFFMAN_CP_VARIANT_STAGES},
}

GROUP_DCT_STAGES = {
    "dct_float",
    "quant_dct",
    "quant_dct_group",
    "ac_only",
    "dc_diff_ac",
    "dc_diff_ac_group",
}


def make_input(stage, batch_size=2, rlc_lmax=64, numeric_feature_dim=384):
    if stage in ("rgb", "ycbcr"):
        return torch.randn(batch_size, 3, 224, 224)
    if stage in GROUP_DCT_STAGES:
        return {
            "y": torch.randn(batch_size, 1, 28, 28, 8, 8),
            "cbcr": torch.randn(batch_size, 2, 14, 14, 8, 8),
        }
    if stage in (
        "rlc_dc_2d", "rlc_dcac_2d", "rlc_dc_2d_max", "rlc_dcac_2d_max",
        "rlc_dc_flat", "rlc_dcac_flat", "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
        "rlc_dc_2d_direct", "rlc_dcac_2d_direct",
        "rlc_dc_2d_max_direct", "rlc_dcac_2d_max_direct",
        "rlc_dc_flat_direct", "rlc_dcac_flat_direct",
        "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
        "huff_dcac_2d_direct", "huff_dcac_flat_direct", "huff_dcac_2d_cp_direct",
        *RLC_CP_VARIANT_STAGES,
        *HUFFMAN_CP_VARIANT_STAGES,
    ):
        if stage in {"huff_dcac_2d_cp_direct", *HUFFMAN_CP_VARIANT_STAGES}:
            run = torch.randint(0, 1025, (batch_size, 196, rlc_lmax), dtype=torch.long)
        elif stage in ("huff_dcac_2d_direct", "huff_dcac_flat_direct"):
            run = torch.rand(batch_size, 196, rlc_lmax)
        elif stage in {
            "rlc_dc_2d_cp", "rlc_dcac_2d_cp",
            "rlc_dc_2d_cp_direct", "rlc_dcac_2d_cp_direct",
            *RLC_CP_VARIANT_STAGES,
        }:
            run = torch.randint(0, 16, (batch_size, 196, rlc_lmax), dtype=torch.long)
        else:
            run = torch.rand(batch_size, 196, rlc_lmax)
        value = torch.randn(batch_size, 196, rlc_lmax) * 0.5
        mask = torch.ones((batch_size, 196, rlc_lmax), dtype=torch.bool)
        return run, value, mask
    if stage in NUMERIC_FEATURE_DIMS:
        return torch.randn(batch_size, 196, numeric_feature_dim)
    if stage in ("raw_byte", "bitstream"):
        payload = torch.randint(0, 256, (batch_size, 196, rlc_lmax), dtype=torch.uint8)
        mask = torch.ones((batch_size, 196, rlc_lmax), dtype=torch.bool)
        return payload, mask
    if stage in ("rlc_dc", "rlc_dcac", "rlc_dc_simple", "rlc_dcac_simple", "rlc_dc_attnpool", "rlc_dcac_attnpool"):
        kind = torch.randint(0, 5, (batch_size, 196, rlc_lmax), dtype=torch.long)
        component = torch.randint(0, 4, (batch_size, 196, rlc_lmax), dtype=torch.long)
        block = torch.randint(0, 7, (batch_size, 196, rlc_lmax), dtype=torch.long)
        zigzag = torch.randint(0, 65, (batch_size, 196, rlc_lmax), dtype=torch.long)
        run = torch.randint(0, 17, (batch_size, 196, rlc_lmax), dtype=torch.long)
        size = torch.randint(0, 17, (batch_size, 196, rlc_lmax), dtype=torch.long)
        amplitude = torch.randn(batch_size, 196, rlc_lmax)
        mask = torch.ones((batch_size, 196, rlc_lmax), dtype=torch.bool)
        return kind, component, block, zigzag, run, size, amplitude, mask
    raise ValueError(stage)


def run_stage(stage, embed_type=3):
    torch.manual_seed(0)
    numeric_feature_dim = NUMERIC_FEATURE_DIMS.get(stage, 384)
    model = ViT(
        emb_size=192,
        depth=1,
        n_classes=10,
        drop_p=0.0,
        num_heads=3,
        head_size=64,
        stage=stage,
        ver=embed_type,
        numeric_feature_dim=numeric_feature_dim,
        rlc_lmax=64,
        device="cpu",
        dtype=torch.float32,
    )
    labels = torch.randint(0, 10, (2,))
    outputs = model(make_input(stage, numeric_feature_dim=numeric_feature_dim))
    assert outputs.shape == (2, 10), (stage, outputs.shape)
    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()


def main():
    stages = [
        "rgb",
        "ycbcr",
        "dct_float",
        "quant_dct",
        "quant_dct_group",
        "ac_only",
        "quant_dct_zigzag_linear",
        "dc_abs",
        "dc_diff",
        "dc_diff_ac",
        "dc_diff_ac_group",
        "dc_absdiff",
        "raw_byte",
        "bitstream",
        "rlc_dc",
        "rlc_dcac",
        "rlc_dc_simple",
        "rlc_dcac_simple",
        "rlc_dc_attnpool",
        "rlc_dcac_attnpool",
        "rlc_dc_2d",
        "rlc_dcac_2d",
        "rlc_dc_2d_max",
        "rlc_dcac_2d_max",
        "rlc_dc_flat",
        "rlc_dcac_flat",
        "rlc_dc_2d_cp",
        "rlc_dcac_2d_cp",
        "rlc_dc_2d_direct",
        "rlc_dcac_2d_direct",
        "rlc_dc_2d_max_direct",
        "rlc_dcac_2d_max_direct",
        "rlc_dc_flat_direct",
        "rlc_dcac_flat_direct",
        "rlc_dc_2d_cp_direct",
        "rlc_dcac_2d_cp_direct",
        "huff_dcac_2d_direct",
        "huff_dcac_flat_direct",
        "huff_dcac_2d_cp_direct",
        "rlc_dcac_2d_cp_concat_mean_direct",
        "rlc_dcac_2d_cp_add_conv_direct",
        "rlc_dcac_2d_cp_concat_conv_direct",
        "rlc_dcac_2d_cp_film_conv_direct",
        "huff_dcac_2d_cp_concat_mean_direct",
        "huff_dcac_2d_cp_add_conv_direct",
        "huff_dcac_2d_cp_concat_conv_direct",
        "huff_dcac_2d_cp_film_conv_direct",
    ]
    for stage in stages:
        run_stage(stage)
        print(f"OK {stage}")
    run_stage("raw_byte", embed_type=1)
    run_stage("raw_byte", embed_type=2)
    print("OK raw_byte float/mean variants")


if __name__ == "__main__":
    main()
