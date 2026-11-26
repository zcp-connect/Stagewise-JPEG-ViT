"""
Smoke test for the stage-wise quant_dct / dc_diff_ac entry after the merge:
routed via PatchEmbedding_DCT_Group.

Verifies:
  1. Dataset __getitem__ returns structured {y, cbcr} tensors with expected shapes.
  2. Model forward + backward works end-to-end for both stages.
  3. val-path semantic check:
       Y AC coefficients identical between quant_dct and dc_diff_ac.
       At least one Y DC position differs (DCDiffQuantDCT effective, no double-diff,
       no lost-diff).
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import datasets as ds
import utils.configs as configs
import models.plainvit as plainvit


RLC_CP_VARIANT_STAGES = (
    "rlc_dcac_2d_cp_concat_mean_direct",
    "rlc_dcac_2d_cp_add_conv_direct",
    "rlc_dcac_2d_cp_concat_conv_direct",
    "rlc_dcac_2d_cp_film_conv_direct",
)

HUFFMAN_CP_VARIANT_STAGES = (
    "huff_dcac_2d_cp_concat_mean_direct",
    "huff_dcac_2d_cp_add_conv_direct",
    "huff_dcac_2d_cp_concat_conv_direct",
    "huff_dcac_2d_cp_film_conv_direct",
)


def _build_item(stage, index_path, basepath, type_):
    dset = ds.imagenet_dataset_indexing(
        indexfile=index_path,
        basepath=basepath,
        load_mode=stage,
        stage=stage,
        type=type_,
        rlc_lmax=8,
        ops_list=None,
        num_ops=2,
        ops_magnitude=3,
        dtype=torch.float32,
    )
    return dset[0]


def test_dataset_shape(index_path, basepath):
    print("=== test_dataset_shape ===")
    for stage in ("quant_dct", "ac_only", "dc_diff_ac"):
        (data, label) = _build_item(stage, index_path, basepath, type_="val")
        assert isinstance(data, dict), f"{stage}: expected dict, got {type(data)}"
        assert set(data.keys()) == {"y", "cbcr"}, f"{stage}: keys = {list(data.keys())}"
        y = data["y"]
        c = data["cbcr"]
        assert y.shape == (1, 28, 28, 8, 8), f"{stage}: y.shape = {y.shape}"
        assert c.shape == (2, 14, 14, 8, 8), f"{stage}: cbcr.shape = {c.shape}"
        assert y.dtype == torch.float32, f"{stage}: y.dtype = {y.dtype}"
        print(f"  {stage}: y={tuple(y.shape)} cbcr={tuple(c.shape)} "
              f"y.range=[{y.min():.4f},{y.max():.4f}] label={label}")
    print("  OK")


def test_semantic_dc_differs_ac_identical(index_path, basepath):
    print("=== test_semantic_dc_differs_ac_identical ===")
    # Use val (deterministic: CenterCrop, no RandAugment) so only DCDiffQuantDCT
    # is different between stages.
    q, _ = _build_item("quant_dct", index_path, basepath, type_="val")
    d, _ = _build_item("dc_diff_ac", index_path, basepath, type_="val")

    # Y
    y_q, y_d = q["y"], d["y"]
    # AC mask: everywhere in the 8x8 block except [0,0]
    ac_mask = torch.ones(8, 8, dtype=torch.bool)
    ac_mask[0, 0] = False
    y_q_ac = y_q[..., ac_mask]
    y_d_ac = y_d[..., ac_mask]
    assert torch.allclose(y_q_ac, y_d_ac), "Y AC positions should be identical across stages"

    y_q_dc = y_q[..., 0, 0]   # [1, 28, 28]
    y_d_dc = y_d[..., 0, 0]
    n_diff = (y_q_dc != y_d_dc).sum().item()
    n_total = y_q_dc.numel()
    # First DC in raster stays same (predictor=0). Expect most others to differ.
    assert n_diff > 0, "Expected at least one Y DC to differ after DCDiffQuantDCT"
    print(f"  Y: AC identical OK; DC differ at {n_diff}/{n_total} positions")

    # CbCr
    c_q, c_d = q["cbcr"], d["cbcr"]
    c_q_ac = c_q[..., ac_mask]
    c_d_ac = c_d[..., ac_mask]
    assert torch.allclose(c_q_ac, c_d_ac), "CbCr AC positions should be identical across stages"
    c_q_dc = c_q[..., 0, 0]
    c_d_dc = c_d[..., 0, 0]
    n_diff_c = (c_q_dc != c_d_dc).sum().item()
    n_total_c = c_q_dc.numel()
    assert n_diff_c > 0, "Expected at least one CbCr DC to differ"
    print(f"  CbCr: AC identical OK; DC differ at {n_diff_c}/{n_total_c} positions")
    print("  OK")


def test_ac_only_zeros_dc_preserves_ac(index_path, basepath):
    print("=== test_ac_only_zeros_dc_preserves_ac ===")
    q, _ = _build_item("quant_dct", index_path, basepath, type_="val")
    a, _ = _build_item("ac_only", index_path, basepath, type_="val")

    ac_mask = torch.ones(8, 8, dtype=torch.bool)
    ac_mask[0, 0] = False
    assert torch.allclose(q["y"][..., ac_mask], a["y"][..., ac_mask]), "Y AC should match quant_dct"
    assert torch.allclose(q["cbcr"][..., ac_mask], a["cbcr"][..., ac_mask]), "CbCr AC should match quant_dct"
    assert torch.count_nonzero(a["y"][..., 0, 0]).item() == 0, "Y DC should be zero"
    assert torch.count_nonzero(a["cbcr"][..., 0, 0]).item() == 0, "CbCr DC should be zero"
    print("  AC identical OK; DC zero OK")
    print("  OK")


def test_model_forward_backward(device):
    print("=== test_model_forward_backward ===")
    # Build vits with both stages; feed the right dummy batch; check logits + grad.
    for stage in ("quant_dct", "ac_only", "dc_diff_ac"):
        model = plainvit.ViT(
            patch_size=16,
            emb_size=384,
            depth=12,
            n_classes=1000,
            drop_p=0.0,
            pixel_space="MCU_BYTES",
            ver=3,
            use_subblock=True,
            stage=stage,
            numeric_feature_dim=384,
            rlc_lmax=8,
            device=device,
            dtype=torch.float32,
            num_heads=6, head_size=64,
        ).to(device)
        model.train()

        # Check patchembed type
        assert isinstance(model.patchembed, plainvit.PatchEmbedding_DCT_Group), \
            f"{stage}: expected DCT_Group, got {type(model.patchembed).__name__}"

        B = 2
        y = torch.randn(B, 1, 28, 28, 8, 8, device=device) / 189.0
        c = torch.randn(B, 2, 14, 14, 8, 8, device=device) / 189.0
        labels = torch.randint(0, 1000, (B,), device=device)
        out = model({"y": y, "cbcr": c})
        assert out.shape == (B, 1000), f"{stage}: out.shape = {out.shape}"

        loss = torch.nn.functional.cross_entropy(out, labels)
        loss.backward()
        grad_params = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        total_params = sum(1 for _ in model.parameters())
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {stage}: patchembed=DCT_Group  out={tuple(out.shape)}  "
              f"loss={loss.item():.4f}  grad_params={grad_params}/{total_params}  "
              f"total_params={n_params/1e6:.2f}M")
        assert grad_params == total_params, f"{stage}: only {grad_params}/{total_params} got grads"
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  OK")


def test_rlc_attnpool_dataset_shape(index_path, basepath):
    print("=== test_rlc_attnpool_dataset_shape ===")
    for stage, expected_lmax in (("rlc_dc_attnpool", 8), ("rlc_dcac_attnpool", 384)):
        dset = ds.imagenet_dataset_indexing(
            indexfile=index_path,
            basepath=basepath,
            load_mode=stage,
            stage=stage,
            type="val",
            rlc_lmax=expected_lmax,
            ops_list=None,
            num_ops=2,
            ops_magnitude=3,
            dtype=torch.float32,
        )
        data, label = dset[0]
        assert isinstance(data, (tuple, list)), f"{stage}: expected tuple/list, got {type(data)}"
        assert len(data) == 8, f"{stage}: len(data)={len(data)}"
        assert data[0].shape == (196, expected_lmax), f"{stage}: kind shape={data[0].shape}"
        assert data[-1].shape == (196, expected_lmax), f"{stage}: mask shape={data[-1].shape}"
        assert data[-1].dtype == torch.bool, f"{stage}: mask dtype={data[-1].dtype}"
        print(f"  {stage}: kind={tuple(data[0].shape)} valid_events={int(data[-1].sum())} label={label}")
    print("  OK")


def test_rlc_2d_dataset_shape(index_path, basepath):
    print("=== test_rlc_2d_dataset_shape ===")
    for stage, expected_lmax in (
        ("rlc_dc_2d", 8),
        ("rlc_dcac_2d", 320),
        ("rlc_dc_2d_max", 8),
        ("rlc_dcac_2d_max", 320),
        ("rlc_dc_flat", 8),
        ("rlc_dcac_flat", 320),
        ("rlc_dc_2d_cp", 8),
        ("rlc_dcac_2d_cp", 320),
    ):
        dset = ds.imagenet_dataset_indexing(
            indexfile=index_path,
            basepath=basepath,
            load_mode=stage,
            stage=stage,
            type="val",
            rlc_lmax=expected_lmax,
            ops_list=None,
            num_ops=2,
            ops_magnitude=3,
            dtype=torch.float32,
        )
        data, label = dset[0]
        assert isinstance(data, (tuple, list)), f"{stage}: expected tuple/list, got {type(data)}"
        assert len(data) == 3, f"{stage}: len(data)={len(data)}"
        run, value, mask = data
        assert run.shape == (196, expected_lmax), f"{stage}: run shape={run.shape}"
        assert value.shape == (196, expected_lmax), f"{stage}: value shape={value.shape}"
        assert mask.shape == (196, expected_lmax), f"{stage}: mask shape={mask.shape}"
        assert mask.dtype == torch.bool, f"{stage}: mask dtype={mask.dtype}"
        if stage.endswith("_cp"):
            assert run.dtype == torch.long, f"{stage}: run dtype={run.dtype}"
            assert run.min().item() >= 0
            assert run.max().item() <= 15
        else:
            assert run.dtype == torch.float32, f"{stage}: run dtype={run.dtype}"
            assert run.min().item() >= 0.0
            assert run.max().item() <= 1.0 + 1e-6
        assert value.abs().max().item() < 50.0
        print(
            f"  {stage}: events={int(mask.sum())} label={label} "
            f"run.range=[{run.min():.3f},{run.max():.3f}] "
            f"value.range=[{value.min():.3f},{value.max():.3f}]"
        )
    print("  OK")


def test_direct_cp_variant_dataset_shape(index_path, basepath):
    print("=== test_direct_cp_variant_dataset_shape ===")
    for stage in (*RLC_CP_VARIANT_STAGES, *HUFFMAN_CP_VARIANT_STAGES):
        dset = ds.imagenet_dataset_indexing(
            indexfile=index_path,
            basepath=basepath,
            load_mode=stage,
            stage=stage,
            type="val",
            rlc_lmax=320,
            ops_list=None,
            num_ops=2,
            ops_magnitude=3,
            dtype=torch.float32,
        )
        data, label = dset[0]
        assert isinstance(data, (tuple, list)), f"{stage}: expected tuple/list, got {type(data)}"
        assert len(data) == 3, f"{stage}: len(data)={len(data)}"
        symbol_id, amplitude, mask = data
        assert symbol_id.shape == (196, 320), f"{stage}: symbol shape={symbol_id.shape}"
        assert amplitude.shape == (196, 320), f"{stage}: amplitude shape={amplitude.shape}"
        assert mask.shape == (196, 320), f"{stage}: mask shape={mask.shape}"
        assert symbol_id.dtype == torch.long, f"{stage}: symbol dtype={symbol_id.dtype}"
        assert amplitude.dtype == torch.float32, f"{stage}: amplitude dtype={amplitude.dtype}"
        assert mask.dtype == torch.bool, f"{stage}: mask dtype={mask.dtype}"
        if stage.startswith("rlc_"):
            assert symbol_id.min().item() >= 0
            assert symbol_id.max().item() <= 15
        else:
            assert symbol_id.min().item() >= 0
            assert symbol_id.max().item() <= 1024
        print(
            f"  {stage}: symbol={tuple(symbol_id.shape)} valid_events={int(mask.sum())} "
            f"symbol.range=[{int(symbol_id.min())},{int(symbol_id.max())}] "
            f"amp.range=[{amplitude.min():.3f},{amplitude.max():.3f}] label={label}"
        )
    print("  OK")


def test_rlc_simple_dataset_shape(index_path, basepath):
    print("=== test_rlc_simple_dataset_shape ===")
    for stage, expected_lmax in (("rlc_dc_simple", 8), ("rlc_dcac_simple", 384)):
        dset = ds.imagenet_dataset_indexing(
            indexfile=index_path,
            basepath=basepath,
            load_mode=stage,
            stage=stage,
            type="val",
            rlc_lmax=expected_lmax,
            ops_list=None,
            num_ops=2,
            ops_magnitude=3,
            dtype=torch.float32,
        )
        data, label = dset[0]
        assert isinstance(data, (tuple, list)), f"{stage}: expected tuple/list, got {type(data)}"
        assert len(data) == 8, f"{stage}: len(data)={len(data)}"
        assert data[0].shape == (196, expected_lmax), f"{stage}: kind shape={data[0].shape}"
        assert data[-1].shape == (196, expected_lmax), f"{stage}: mask shape={data[-1].shape}"
        assert data[-1].dtype == torch.bool, f"{stage}: mask dtype={data[-1].dtype}"
        print(f"  {stage}: kind={tuple(data[0].shape)} valid_events={int(data[-1].sum())} label={label}")
    print("  OK")


def _synthetic_rlc_fields(batch_size, lmax, device):
    return {
        "kind": torch.randint(1, 5, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "component": torch.randint(1, 4, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "block": torch.randint(1, 7, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "zigzag": torch.randint(1, 65, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "run": torch.randint(1, 17, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "size": torch.randint(1, 17, (batch_size, 196, lmax), device=device, dtype=torch.long),
        "amplitude": torch.randn(batch_size, 196, lmax, device=device) * 50,
        "mask": torch.ones(batch_size, 196, lmax, dtype=torch.bool, device=device),
    }


def test_rlc_attnpool_forward_backward(device):
    print("=== test_rlc_attnpool_forward_backward ===")
    for stage, lmax in (("rlc_dc_attnpool", 8), ("rlc_dcac_attnpool", 384)):
        model = plainvit.ViT(
            patch_size=16,
            emb_size=384,
            depth=12,
            n_classes=1000,
            drop_p=0.0,
            pixel_space="MCU_BYTES",
            ver=3,
            use_subblock=True,
            stage=stage,
            numeric_feature_dim=384,
            rlc_lmax=lmax,
            device=device,
            dtype=torch.float32,
            num_heads=6,
            head_size=64,
        ).to(device)
        model.train()
        assert isinstance(model.patchembed, plainvit.PatchEmbedding_RLCEvent_AttnPool), (
            f"{stage}: expected AttnPool, got {type(model.patchembed).__name__}"
        )
        batch_size = 2
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        output = model(_synthetic_rlc_fields(batch_size, lmax, device))
        assert output.shape == (batch_size, 1000), f"{stage}: out.shape={output.shape}"
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {stage}: out={tuple(output.shape)} loss={loss.item():.4f} params={n_params/1e6:.2f}M")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  OK")


def test_rlc_2d_forward_backward(device):
    print("=== test_rlc_2d_forward_backward ===")
    cases = (
        ("rlc_dc_2d", 8, plainvit.PatchEmbedding_RLCEvent_2D),
        ("rlc_dcac_2d", 320, plainvit.PatchEmbedding_RLCEvent_2D),
        ("rlc_dc_2d_max", 8, plainvit.PatchEmbedding_RLCEvent_2D),
        ("rlc_dcac_2d_max", 320, plainvit.PatchEmbedding_RLCEvent_2D),
        ("rlc_dc_flat", 8, plainvit.PatchEmbedding_RLCEvent_Flat),
        ("rlc_dcac_flat", 320, plainvit.PatchEmbedding_RLCEvent_Flat),
        ("rlc_dc_2d_cp", 8, plainvit.PatchEmbedding_RLCEvent_CP),
        ("rlc_dcac_2d_cp", 320, plainvit.PatchEmbedding_RLCEvent_CP),
    )
    for stage, lmax, expected_type in cases:
        numeric_feature_dim = 64 if expected_type in (plainvit.PatchEmbedding_RLCEvent_2D, plainvit.PatchEmbedding_RLCEvent_CP) else 384
        model = plainvit.ViT(
            patch_size=16,
            emb_size=384,
            depth=12,
            n_classes=1000,
            drop_p=0.0,
            pixel_space="MCU_BYTES",
            ver=3,
            use_subblock=True,
            stage=stage,
            numeric_feature_dim=numeric_feature_dim,
            rlc_lmax=lmax,
            device=device,
            dtype=torch.float32,
            num_heads=6,
            head_size=64,
        ).to(device)
        model.train()
        assert isinstance(model.patchembed, expected_type), (
            f"{stage}: expected {expected_type.__name__}, got {type(model.patchembed).__name__}"
        )
        if stage.endswith("_max"):
            assert getattr(model.patchembed, "pool", None) == "max", f"{stage}: expected max pool"
        batch_size = 2
        if expected_type is plainvit.PatchEmbedding_RLCEvent_CP:
            run = torch.randint(0, 16, (batch_size, 196, lmax), device=device, dtype=torch.long)
        else:
            run = torch.rand(batch_size, 196, lmax, device=device)
        value = torch.randn(batch_size, 196, lmax, device=device) * 0.5
        mask = torch.ones(batch_size, 196, lmax, dtype=torch.bool, device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        output = model({"run": run, "amplitude": value, "mask": mask})
        assert output.shape == (batch_size, 1000), f"{stage}: out.shape={output.shape}"
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {stage}: out={tuple(output.shape)} loss={loss.item():.4f} params={n_params/1e6:.2f}M")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  OK")


def test_direct_cp_variant_forward_backward(device):
    print("=== test_direct_cp_variant_forward_backward ===")
    for stage in (*RLC_CP_VARIANT_STAGES, *HUFFMAN_CP_VARIANT_STAGES):
        model = plainvit.ViT(
            patch_size=16,
            emb_size=384,
            depth=12,
            n_classes=1000,
            drop_p=0.0,
            pixel_space="MCU_BYTES",
            ver=3,
            use_subblock=True,
            stage=stage,
            numeric_feature_dim=64,
            rlc_lmax=320,
            device=device,
            dtype=torch.float32,
            num_heads=6,
            head_size=64,
        ).to(device)
        model.train()
        assert isinstance(model.patchembed, plainvit.PatchEmbedding_EventCPFusion), (
            f"{stage}: expected EventCPFusion, got {type(model.patchembed).__name__}"
        )
        batch_size = 2
        if stage.startswith("huff_"):
            symbol_id = torch.randint(0, 1025, (batch_size, 196, 320), device=device, dtype=torch.long)
        else:
            symbol_id = torch.randint(0, 16, (batch_size, 196, 320), device=device, dtype=torch.long)
        amplitude = torch.randn(batch_size, 196, 320, device=device) * 0.5
        mask = torch.ones(batch_size, 196, 320, dtype=torch.bool, device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        output = model((symbol_id, amplitude, mask))
        assert output.shape == (batch_size, 1000), f"{stage}: out.shape={output.shape}"
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {stage}: out={tuple(output.shape)} loss={loss.item():.4f} params={n_params/1e6:.2f}M")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  OK")


def test_rlc_simple_forward_backward(device):
    print("=== test_rlc_simple_forward_backward ===")
    for stage, lmax in (("rlc_dc_simple", 8), ("rlc_dcac_simple", 384)):
        model = plainvit.ViT(
            patch_size=16,
            emb_size=384,
            depth=12,
            n_classes=1000,
            drop_p=0.0,
            pixel_space="MCU_BYTES",
            ver=3,
            use_subblock=True,
            stage=stage,
            numeric_feature_dim=384,
            rlc_lmax=lmax,
            device=device,
            dtype=torch.float32,
            num_heads=6,
            head_size=64,
        ).to(device)
        model.train()
        assert isinstance(model.patchembed, plainvit.PatchEmbedding_RLCEvent_Simple), (
            f"{stage}: expected Simple, got {type(model.patchembed).__name__}"
        )
        batch_size = 2
        labels = torch.randint(0, 1000, (batch_size,), device=device)
        output = model(_synthetic_rlc_fields(batch_size, lmax, device))
        assert output.shape == (batch_size, 1000), f"{stage}: out.shape={output.shape}"
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {stage}: out={tuple(output.shape)} loss={loss.item():.4f} params={n_params/1e6:.2f}M")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("  OK")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    index_path = os.path.join(REPO, "assets", "indexbase_val.csv")
    basepath = "/path/to/imagenet1k/imagenet1k"
    assert os.path.isfile(index_path), f"missing index: {index_path}"
    assert os.path.isdir(basepath), f"missing data: {basepath}"

    test_dataset_shape(index_path, basepath)
    test_semantic_dc_differs_ac_identical(index_path, basepath)
    test_ac_only_zeros_dc_preserves_ac(index_path, basepath)
    test_model_forward_backward(device)
    test_rlc_2d_dataset_shape(index_path, basepath)
    test_direct_cp_variant_dataset_shape(index_path, basepath)
    test_rlc_2d_forward_backward(device)
    test_direct_cp_variant_forward_backward(device)
    test_rlc_simple_dataset_shape(index_path, basepath)
    test_rlc_simple_forward_backward(device)
    test_rlc_attnpool_dataset_shape(index_path, basepath)
    test_rlc_attnpool_forward_backward(device)
    print("\nALL SMOKES PASSED")


if __name__ == "__main__":
    main()
