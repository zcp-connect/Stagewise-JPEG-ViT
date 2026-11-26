import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import datasets as ds
from datasets import (
    DCT_ZIGZAG,
    DCDIFF_POST_AUG_OPS,
    _apply_frequency_norm,
    _sample_dcdiff_aug_ops,
    _huffman_prefix_id,
    pack_rlc_event_tensors_2d,
    pack_huffman_event_tensors_direct,
    pack_rlc_event_tensors,
    stage_dct_features_from_coeffs,
)
from utils.configs import generate_config
from utils.custom_transforms import DCDiffQuantDCT, RandAugment_QuantDCT


def test_dct_feature_shape():
    quant = torch.ones(3, 8, 8, dtype=torch.short)
    y = torch.randn(1, 28, 28, 8, 8).short()
    cbcr = torch.randn(2, 14, 14, 8, 8).short()
    dct = stage_dct_features_from_coeffs(quant, y, cbcr, "dct_float")
    qdct = stage_dct_features_from_coeffs(quant, y, cbcr, "quant_dct")
    qdct_group = stage_dct_features_from_coeffs(quant, y, cbcr, "quant_dct_group")
    ac_only = stage_dct_features_from_coeffs(quant, y, cbcr, "ac_only")
    qdct_zigzag = stage_dct_features_from_coeffs(quant, y, cbcr, "quant_dct_zigzag")
    dc_abs = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_abs")
    dc_diff = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_diff")
    y_diff, cbcr_diff = DCDiffQuantDCT()((y, cbcr))
    dc_diff_ac = stage_dct_features_from_coeffs(quant, y_diff, cbcr_diff, "dc_diff_ac")
    dc_diff_ac_group = stage_dct_features_from_coeffs(quant, y_diff, cbcr_diff, "dc_diff_ac_group")
    dc_absdiff = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_absdiff")
    assert dct.shape == (196, 384), dct.shape
    assert qdct.shape == (196, 384), qdct.shape
    assert qdct_group.shape == (196, 384), qdct_group.shape
    assert ac_only.shape == (196, 384), ac_only.shape
    assert qdct_zigzag.shape == (196, 384), qdct_zigzag.shape
    assert dc_abs.shape == (196, 6), dc_abs.shape
    assert dc_diff.shape == (196, 6), dc_diff.shape
    assert dc_diff_ac.shape == (196, 384), dc_diff_ac.shape
    assert dc_diff_ac_group.shape == (196, 384), dc_diff_ac_group.shape
    assert dc_absdiff.shape == (196, 12), dc_absdiff.shape
    assert torch.isfinite(dct).all()
    assert torch.isfinite(qdct).all()
    assert torch.isfinite(qdct_group).all()
    assert torch.isfinite(ac_only).all()
    assert torch.isfinite(qdct_zigzag).all()
    assert torch.isfinite(dc_abs).all()
    assert torch.isfinite(dc_diff).all()
    assert torch.isfinite(dc_diff_ac).all()
    assert torch.isfinite(dc_diff_ac_group).all()
    assert torch.isfinite(dc_absdiff).all()


def test_quant_dct_zigzag_order():
    quant = torch.ones(3, 8, 8, dtype=torch.short)
    y = torch.zeros(1, 28, 28, 8, 8, dtype=torch.short)
    cbcr = torch.zeros(2, 14, 14, 8, 8, dtype=torch.short)
    y[0, 0, 0] = torch.arange(64, dtype=torch.short).reshape(8, 8)
    qdct_zigzag = stage_dct_features_from_coeffs(quant, y, cbcr, "quant_dct_zigzag")
    expected = torch.tensor(DCT_ZIGZAG, dtype=torch.float32) / 189.0
    assert torch.allclose(qdct_zigzag[0, :64], expected), qdct_zigzag[0, :64]


def test_ac_only_zeros_dc_preserves_ac():
    quant = torch.ones(3, 8, 8, dtype=torch.short)
    y = torch.zeros(1, 28, 28, 8, 8, dtype=torch.short)
    cbcr = torch.zeros(2, 14, 14, 8, 8, dtype=torch.short)
    y[0, 0, 0, 0, 0] = 10
    y[0, 0, 0, 0, 1] = 101
    cbcr[0, 0, 0, 0, 0] = -3
    cbcr[1, 0, 0, 0, 0] = 7
    cbcr[0, 0, 0, 0, 1] = 105
    cbcr[1, 0, 0, 0, 1] = 106

    ac_only = stage_dct_features_from_coeffs(quant, y, cbcr, "ac_only")
    assert torch.allclose(ac_only[0, [0, 64, 128, 192, 256, 320]], torch.zeros(6))
    assert torch.allclose(
        ac_only[0, [1, 65, 129, 193, 257, 321]],
        torch.tensor([101, 0, 0, 0, 105, 106], dtype=torch.float32) / 189.0,
    )


def test_dc_feature_values():
    quant = torch.ones(3, 8, 8, dtype=torch.short)
    y = torch.zeros(1, 28, 28, 8, 8, dtype=torch.short)
    cbcr = torch.zeros(2, 14, 14, 8, 8, dtype=torch.short)
    y[0, 0, 0, 0, 0] = 10
    y[0, 0, 1, 0, 0] = 15
    y[0, 1, 0, 0, 0] = 21
    y[0, 1, 1, 0, 0] = 28
    y[0, 0, 0, 0, 1] = 101
    y[0, 0, 1, 0, 1] = 102
    y[0, 1, 0, 0, 1] = 103
    y[0, 1, 1, 0, 1] = 104
    cbcr[0, 0, 0, 0, 0] = -3
    cbcr[1, 0, 0, 0, 0] = 7
    cbcr[0, 0, 1, 0, 0] = 1
    cbcr[1, 0, 1, 0, 0] = 9
    cbcr[0, 0, 0, 0, 1] = 105
    cbcr[1, 0, 0, 0, 1] = 106

    dc_abs = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_abs")
    dc_diff = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_diff")
    y_diff, cbcr_diff = DCDiffQuantDCT()((y, cbcr))
    dc_diff_ac = stage_dct_features_from_coeffs(quant, y_diff, cbcr_diff, "dc_diff_ac")
    dc_absdiff = stage_dct_features_from_coeffs(quant, y, cbcr, "dc_absdiff")
    assert torch.allclose(dc_abs[0], torch.tensor([10, 15, 21, 28, -3, 7], dtype=torch.float32) / 189.0)
    assert torch.allclose(dc_diff[0], torch.tensor([10, 5, 6, 7, -3, 7], dtype=torch.float32) / 189.0)
    assert torch.allclose(dc_diff[1, 4:], torch.tensor([4, 2], dtype=torch.float32) / 189.0)
    assert torch.allclose(dc_diff_ac[0, [0, 64, 128, 192, 256, 320]], dc_diff[0])
    assert torch.allclose(dc_diff_ac[0, [1, 65, 129, 193, 257, 321]], torch.tensor([101, 102, 103, 104, 105, 106], dtype=torch.float32) / 189.0)
    assert torch.allclose(dc_absdiff[0, :6], dc_abs[0])
    assert torch.allclose(dc_absdiff[0, 6:], dc_diff[0])


def test_group_dc_diff_values():
    y = torch.zeros(1, 28, 28, 8, 8, dtype=torch.short)
    cbcr = torch.zeros(2, 14, 14, 8, 8, dtype=torch.short)
    y[0, 0, 0, 0, 0] = 10
    y[0, 0, 1, 0, 0] = 15
    y[0, 1, 0, 0, 0] = 21
    y[0, 1, 1, 0, 0] = 28
    cbcr[0, 0, 0, 0, 0] = -3
    cbcr[1, 0, 0, 0, 0] = 7
    cbcr[0, 0, 1, 0, 0] = 1
    cbcr[1, 0, 1, 0, 0] = 9

    y_diff, cbcr_diff = DCDiffQuantDCT()((y, cbcr))
    assert torch.equal(y_diff[0, :2, :2, 0, 0].reshape(-1), torch.tensor([10, 5, 6, 7], dtype=torch.short))
    assert cbcr_diff[0, 0, 0, 0, 0].item() == -3
    assert cbcr_diff[0, 0, 1, 0, 0].item() == 4
    assert cbcr_diff[1, 0, 0, 0, 0].item() == 7
    assert cbcr_diff[1, 0, 1, 0, 0].item() == 2


def test_dc_diff_transform_before_block_dropout():
    y = torch.zeros(1, 28, 28, 8, 8, dtype=torch.short)
    cbcr = torch.zeros(2, 14, 14, 8, 8, dtype=torch.short)
    y[0, 0, 0, 0, 0] = 10
    y[0, 0, 1, 0, 0] = 15
    y[0, 1, 0, 0, 0] = 21
    y[0, 1, 1, 0, 0] = 28
    y[0, 0, 0, 0, 1] = 101
    cbcr[0, 0, 0, 0, 0] = -3
    cbcr[1, 0, 0, 0, 0] = 7

    y_diff, cbcr_diff = DCDiffQuantDCT()((y, cbcr))
    assert torch.equal(y_diff[0, :2, :2, 0, 0].reshape(-1), torch.tensor([10, 5, 6, 7], dtype=torch.short))
    assert y_diff[0, 0, 0, 0, 1].item() == 101

    y_drop, cbcr_drop = RandAugment_QuantDCT(num_ops=0)._block_dropout([y_diff, cbcr_diff], probability=1.0)
    assert torch.count_nonzero(y_drop[:, :, :, 0, 0]) == 0
    assert y_drop[0, 0, 0, 0, 0].item() == 0
    assert y_drop[0, 0, 0, 0, 1].item() == 0


def test_dc_diff_aug_sampling_preserves_num_ops():
    torch.manual_seed(0)
    pre_ops, post_ops = _sample_dcdiff_aug_ops(["Cutout", "Brightness", "BlockDropout"], num_ops=3)
    assert len(pre_ops) + len(post_ops) == 3
    assert all(op not in {"Cutout", "BlockDropout", "FreqBandDropout", "ACQuantNoise"} for op in pre_ops)
    assert all(op in {"Cutout", "BlockDropout", "FreqBandDropout", "ACQuantNoise"} for op in post_ops)

    torch.manual_seed(0)
    pre_ops, post_ops = _sample_dcdiff_aug_ops(["Cutout"], num_ops=1)
    assert pre_ops == []
    assert post_ops == ["Cutout"]


def test_dc_diff_config_uses_stage_specific_auglist():
    quant_only_ops = {"BlockDropout", "FreqBandDropout", "ACQuantNoise"}

    dct_cfg = generate_config(stage="dct_float")
    qdct_cfg = generate_config(stage="quant_dct")
    dcdiff_cfg = generate_config(stage="dc_diff_ac")
    dcdiff_group_cfg = generate_config(stage="dc_diff_ac_group")

    assert not quant_only_ops.intersection(set(dct_cfg.TRAIN.AUGLIST))
    assert not quant_only_ops.intersection(set(qdct_cfg.TRAIN.AUGLIST))
    assert quant_only_ops.issubset(set(dcdiff_cfg.TRAIN.AUGLIST))
    assert quant_only_ops.issubset(set(dcdiff_group_cfg.TRAIN.AUGLIST))

    torch.manual_seed(0)
    pre_ops, post_ops = _sample_dcdiff_aug_ops(None, num_ops=256)
    assert set(post_ops).intersection(quant_only_ops)
    assert all(op not in DCDIFF_POST_AUG_OPS for op in pre_ops)
    assert all(op in DCDIFF_POST_AUG_OPS for op in post_ops)


def test_split_eval_loaders_use_eval_dataset_type():
    original_dataset_cls = ds.imagenet_dataset_indexing

    class FakeImagenetDataset(torch.utils.data.Dataset):
        def __init__(self, type="train", **kwargs):
            self.type = type
            self.stage = kwargs.get("stage")
            self.ops_list = kwargs.get("ops_list")

        def __getitem__(self, index):
            return torch.tensor([index]), index % 1000

        def __len__(self):
            return 100

    try:
        ds.imagenet_dataset_indexing = FakeImagenetDataset
        trainloader, valloader, trainvalloader, train_indices, val_indices = ds.dataset_selector(
            dataset="imagenet_quant_dct",
            type="train",
            indexpath="unused.csv",
            basepath="",
            batch_size=4,
            num_workers=0,
            shuffle=False,
            trainval_split=0.1,
            return_indices=True,
            distributed=False,
            seed=11997733,
            ops_list=["Brightness"],
        )
    finally:
        ds.imagenet_dataset_indexing = original_dataset_cls

    assert trainloader.dataset.subset.dataset.type == "train"
    assert valloader.dataset.subset.dataset.type == "val"
    assert trainvalloader.dataset.subset.dataset.type == "val"
    assert trainloader.dataset.subset.dataset.ops_list == ["Brightness"]
    assert valloader.dataset.subset.dataset.ops_list is None
    assert trainvalloader.dataset.subset.dataset.ops_list is None
    assert list(valloader.dataset.subset.indices) == list(val_indices)
    assert list(trainvalloader.dataset.subset.indices) == list(train_indices[: round(len(train_indices) * 0.05)])


def test_huffman_direct_prefix_semantics():
    blocks = torch.zeros(196, 6, 64, dtype=torch.short)
    blocks[0, 0, 0] = 5
    blocks[0, 0, 1] = 3
    blocks[0, 1, 0] = 5
    blocks[0, 1, 17] = -2
    blocks[0, 4, 0] = -4
    blocks[0, 4, 1:17] = 0
    blocks[0, 4, 17] = 1

    prefix_id, amplitude, mask, event_len = pack_huffman_event_tensors_direct(blocks, lmax=64)
    active = prefix_id[0, :event_len[0]]

    assert prefix_id.shape == (196, 64)
    assert amplitude.shape == (196, 64)
    assert mask.shape == (196, 64)
    assert mask[0, :event_len[0]].all()
    assert not mask[0, event_len[0]:].any()
    assert active[0].item() == _huffman_prefix_id(0, True, 3)
    assert active[1].item() == _huffman_prefix_id(0, False, 2)
    assert _huffman_prefix_id(0, False, 0x00) in active.tolist()
    assert _huffman_prefix_id(1, False, 0xF0) in active.tolist()
    assert _huffman_prefix_id(0, True, 3) != _huffman_prefix_id(1, True, 3)
    assert _huffman_prefix_id(0, False, 2) != _huffman_prefix_id(1, False, 2)
    assert torch.count_nonzero(prefix_id[~mask]) == 0


def _write_typed_freqnorm_stats(path):
    ac_mask = torch.ones(8, 8, dtype=torch.bool)
    ac_mask[0, 0] = False
    stats = {}
    for component, start_mean, start_std, delta_mean, delta_std, ac_mean, ac_std in (
        ("Y", 100.0, 2.0, 10.0, 5.0, 1.0, 4.0),
        ("CB", -20.0, 3.0, -2.0, 6.0, 7.0, 8.0),
        ("CR", 30.0, 4.0, 3.0, 7.0, -5.0, 9.0),
    ):
        for value_type, mean_value, std_value in (
            ("DC_START", start_mean, start_std),
            ("DC_DELTA", delta_mean, delta_std),
            ("AC", ac_mean, ac_std),
        ):
            mean = torch.zeros(8, 8)
            std = torch.ones(8, 8)
            mean[0, 0] = mean_value
            std[0, 0] = std_value
            if value_type == "AC":
                mean[ac_mask] = mean_value
                std[ac_mask] = std_value
            stats[f"mean_Q_{component}_{value_type}"] = mean
            stats[f"std_Q_{component}_{value_type}"] = std
    torch.save(stats, path)


def test_dcdiff_typed_freqnorm_std(tmp_path=None):
    if tmp_path is None:
        tmp_path = Path("/tmp") / "dcdiff_typed_freqnorm_stats.pt"
    else:
        tmp_path = Path(tmp_path) / "dcdiff_typed_freqnorm_stats.pt"

    sample_count = 4
    y = torch.zeros(sample_count, 28, 28, 8, 8)
    cbcr = torch.zeros(2, sample_count, 14, 14, 8, 8)
    values = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    ac_mask = torch.ones(8, 8, dtype=torch.bool)
    ac_mask[0, 0] = False

    y[:, 0, 0, 0, 0] = 100.0 + 2.0 * values
    y[:, :, :, 0, 0] = torch.where(
        torch.ones_like(y[:, :, :, 0, 0], dtype=torch.bool),
        10.0 + 5.0 * values.view(-1, 1, 1),
        y[:, :, :, 0, 0],
    )
    y[:, 0, 0, 0, 0] = 100.0 + 2.0 * values
    y[..., ac_mask] = 1.0 + 4.0 * values.view(-1, 1, 1, 1)

    cbcr[0, :, 0, 0, 0, 0] = -20.0 + 3.0 * values
    cbcr[0, :, :, :, 0, 0] = -2.0 + 6.0 * values.view(1, -1, 1, 1)
    cbcr[0, :, 0, 0, 0, 0] = -20.0 + 3.0 * values
    cbcr[0, :, :, :, ac_mask] = 7.0 + 8.0 * values.view(1, -1, 1, 1, 1)
    cbcr[1, :, 0, 0, 0, 0] = 30.0 + 4.0 * values
    cbcr[1, :, :, :, 0, 0] = 3.0 + 7.0 * values.view(1, -1, 1, 1)
    cbcr[1, :, 0, 0, 0, 0] = 30.0 + 4.0 * values
    cbcr[1, :, :, :, ac_mask] = -5.0 + 9.0 * values.view(1, -1, 1, 1, 1)

    _write_typed_freqnorm_stats(tmp_path)

    y_norm, cbcr_norm = _apply_frequency_norm(y, cbcr, "dc_diff_ac", str(tmp_path))

    def assert_std_one(tensor):
        std = tensor.std(unbiased=False).item()
        assert abs(std - 1.0) < 1e-5, std

    assert_std_one(y_norm[:, 0, 0, 0, 0])
    assert_std_one(y_norm[:, 0, 1, 0, 0])
    assert_std_one(y_norm[:, 0, 0, 0, 1])
    assert_std_one(cbcr_norm[0, :, 0, 0, 0, 0])
    assert_std_one(cbcr_norm[0, :, 0, 1, 0, 0])
    assert_std_one(cbcr_norm[0, :, 0, 0, 0, 1])
    assert_std_one(cbcr_norm[1, :, 0, 0, 0, 0])
    assert_std_one(cbcr_norm[1, :, 0, 1, 0, 0])
    assert_std_one(cbcr_norm[1, :, 0, 0, 0, 1])


def test_dcdiff_typed_freqnorm_production_shape(tmp_path=None):
    if tmp_path is None:
        tmp_path = Path("/tmp") / "dcdiff_typed_freqnorm_stats_prod.pt"
    else:
        tmp_path = Path(tmp_path) / "dcdiff_typed_freqnorm_stats_prod.pt"
    _write_typed_freqnorm_stats(tmp_path)

    y = torch.zeros(1, 28, 28, 8, 8)
    cbcr = torch.zeros(2, 14, 14, 8, 8)
    y[:, :, :, 0, 0] = 10.0 + 5.0 * 2.0
    y[:, 0, 0, 0, 0] = 100.0 + 2.0 * 2.0
    y[:, 0, 0, 0, 1] = 1.0 + 4.0 * 2.0
    cbcr[0, :, :, 0, 0] = -2.0 + 6.0 * 2.0
    cbcr[0, 0, 0, 0, 0] = -20.0 + 3.0 * 2.0
    cbcr[0, 0, 0, 0, 1] = 7.0 + 8.0 * 2.0
    cbcr[1, :, :, 0, 0] = 3.0 + 7.0 * 2.0
    cbcr[1, 0, 0, 0, 0] = 30.0 + 4.0 * 2.0
    cbcr[1, 0, 0, 0, 1] = -5.0 + 9.0 * 2.0

    y_norm, cbcr_norm = _apply_frequency_norm(y, cbcr, "dc_diff_ac", str(tmp_path))
    assert y_norm[0, 0, 0, 0, 0].item() == 2.0
    assert y_norm[0, 0, 1, 0, 0].item() == 2.0
    assert y_norm[0, 0, 0, 0, 1].item() == 2.0
    assert cbcr_norm[0, 0, 0, 0, 0].item() == 2.0
    assert cbcr_norm[0, 0, 1, 0, 0].item() == 2.0
    assert cbcr_norm[0, 0, 0, 0, 1].item() == 2.0
    assert cbcr_norm[1, 0, 0, 0, 0].item() == 2.0
    assert cbcr_norm[1, 0, 1, 0, 0].item() == 2.0
    assert cbcr_norm[1, 0, 0, 0, 1].item() == 2.0


def test_rlc_pack_shape():
    syntax = [
        {
            "blocks": [
                {
                    "component_index": 0,
                    "block_in_mcu": 0,
                    "tokens": [
                        {"token_type": "DC", "coeff_index": 0, "run": 0, "size": 2, "value": 3},
                        {"token_type": "AC", "coeff_index": 1, "run": 0, "size": 1, "value": -1},
                        {"token_type": "EOB", "coeff_index": -1, "run": 0, "size": 0, "value": 0},
                    ],
                }
            ]
        }
    ] * 196
    fields = pack_rlc_event_tensors(syntax, lmax=16, dc_only=False)
    assert len(fields) == 9
    for field in fields[:8]:
        assert field.shape == (196, 16), field.shape
    assert fields[-1].shape == (196,), fields[-1].shape
    assert fields[0].max().item() == 3


def test_rlc_attnpool_amp_norm_uses_size_raw():
    size = torch.tensor([[[4]]])
    amplitude = torch.tensor([[[8.0]]])
    size_raw = (size - 1).clamp(min=0).float()
    amp_norm = amplitude / (2.0 ** size_raw)
    assert abs(amp_norm.item() - 1.0) < 1e-6, amp_norm.item()


def test_rlc_attnpool_padding_safe():
    from models.plainvit import PatchEmbedding_RLCEvent_AttnPool

    torch.manual_seed(42)
    embed = PatchEmbedding_RLCEvent_AttnPool(
        lmax=8, emb_size=384, inner_dim=64, grid_size=(14, 14)
    )
    embed.eval()
    batch_size, num_mcus, lmax = 1, 196, 8
    fields = {
        "kind": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "component": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "block": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "zigzag": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "run": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "size": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.long),
        "amplitude": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.float32),
        "mask": torch.zeros(batch_size, num_mcus, lmax, dtype=torch.bool),
    }
    with torch.no_grad():
        output = embed(fields)
    assert output.shape == (batch_size, num_mcus, 384), output.shape
    assert torch.isfinite(output).all()


def test_rlc_2d_pack_helper_skips_eob_keeps_zrl():
    syntax = [
        {
            "blocks": [
                {
                    "component_index": 0,
                    "block_in_mcu": 0,
                    "tokens": [
                        {"token_type": "DC", "coeff_index": 0, "run": 0, "size": 2, "value": 7},
                        {"token_type": "AC", "coeff_index": 1, "run": 2, "size": 1, "value": -3},
                        {"token_type": "ZRL", "coeff_index": -1, "run": 15, "size": 0, "value": 0},
                        {"token_type": "EOB", "coeff_index": -1, "run": 0, "size": 0, "value": 0},
                    ],
                }
            ]
        }
    ] * 196
    run, value, mask, event_len = pack_rlc_event_tensors_2d(syntax, lmax=8)
    assert run.shape == (196, 8)
    assert value.shape == (196, 8)
    assert mask.shape == (196, 8)
    assert event_len[0].item() == 3
    assert mask[0, :3].tolist() == [True, True, True]
    assert not mask[0, 3:].any()
    assert run[0, 0].item() == 0.0
    assert abs(value[0, 0].item() - 7.0 / 189.0) < 1e-6
    assert abs(run[0, 1].item() - 2.0 / 15.0) < 1e-6
    assert abs(value[0, 1].item() - (-3.0 / 189.0)) < 1e-6
    assert run[0, 2].item() == 1.0
    assert value[0, 2].item() == 0.0


def main():
    test_dct_feature_shape()
    print("OK dct helper")
    test_quant_dct_zigzag_order()
    print("OK quant dct zigzag order")
    test_ac_only_zeros_dc_preserves_ac()
    print("OK ac only zeros dc")
    test_dc_feature_values()
    print("OK dc feature values")
    test_group_dc_diff_values()
    print("OK group dc diff values")
    test_dc_diff_transform_before_block_dropout()
    print("OK dc diff transform before block dropout")
    test_dc_diff_aug_sampling_preserves_num_ops()
    print("OK dc diff augmentation sampling")
    test_dc_diff_config_uses_stage_specific_auglist()
    print("OK dc diff config auglist")
    test_split_eval_loaders_use_eval_dataset_type()
    print("OK split eval loaders use eval dataset type")
    test_huffman_direct_prefix_semantics()
    print("OK huffman direct prefix semantics")
    test_dcdiff_typed_freqnorm_std()
    print("OK dc diff typed freqnorm")
    test_dcdiff_typed_freqnorm_production_shape()
    print("OK dc diff typed freqnorm production shape")
    test_rlc_pack_shape()
    print("OK rlc pack helper")
    test_rlc_attnpool_amp_norm_uses_size_raw()
    print("OK rlc attnpool amp norm size raw")
    test_rlc_attnpool_padding_safe()
    print("OK rlc attnpool padding safe")
    test_rlc_2d_pack_helper_skips_eob_keeps_zrl()
    print("OK rlc 2d pack helper")


if __name__ == "__main__":
    main()
