import argparse
import os
import sys

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import datasets as ds


def _first_image(index_path, basepath):
    rows = pd.read_csv(index_path).to_numpy()
    path = str(rows[0, 0])
    return path if os.path.isabs(path) else os.path.join(basepath, path)


def _reference_pack(blocks_zz, lmax=320, dtype=torch.float32):
    if blocks_zz.dim() != 3 or tuple(blocks_zz.shape) != (196, 6, 64):
        raise ds.UnsupportedDataset(
            msg=f"expected Huffman blocks with shape (196, 6, 64), got {tuple(blocks_zz.shape)}"
        )

    blocks_list = blocks_zz.tolist()
    dc_pred = [0, 0, 0]
    comp_of_block = [0, 0, 0, 0, 1, 2]
    prefix_rows = []
    amplitude_rows = []
    event_lens = []

    for mcu_blocks in blocks_list:
        prefix_row = [ds.HUFFMAN_PREFIX_PAD_ID] * lmax
        amplitude_row = [0.0] * lmax
        pos = 0
        for block_i, block in enumerate(mcu_blocks):
            if pos >= lmax:
                break

            comp = comp_of_block[block_i]
            dc_cur = int(block[0])
            dc_diff = dc_cur - dc_pred[comp]
            dc_pred[comp] = dc_cur
            prefix_row[pos] = ds._huffman_prefix_id(comp, True, ds._jpeg_category(dc_diff))
            amplitude_row[pos] = dc_diff / 189.0
            pos += 1
            if pos >= lmax:
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
                    prefix_row[pos] = ds._huffman_prefix_id(comp, False, 0xF0)
                    amplitude_row[pos] = 0.0
                    pos += 1
                    zero_run -= 16
                if pos >= lmax:
                    break
                symbol = (zero_run << 4) | ds._jpeg_category(ac)
                prefix_row[pos] = ds._huffman_prefix_id(comp, False, symbol)
                amplitude_row[pos] = ac / 189.0
                pos += 1
                zero_run = 0

            if pos < lmax and zero_run > 0:
                prefix_row[pos] = ds._huffman_prefix_id(comp, False, 0x00)
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


def _assert_pack_equal(blocks, lmax, label):
    ref_prefix, ref_amplitude, ref_mask, ref_event_len = _reference_pack(blocks, lmax=lmax)
    opt_prefix, opt_amplitude, opt_mask, opt_event_len = ds.pack_huffman_event_tensors_direct(
        blocks, lmax=lmax
    )
    assert torch.equal(ref_prefix, opt_prefix), f"prefix mismatch: {label}"
    assert torch.equal(ref_amplitude, opt_amplitude), f"amplitude mismatch: {label}"
    assert torch.equal(ref_mask, opt_mask), f"mask mismatch: {label}"
    assert torch.equal(ref_event_len, opt_event_len), f"event_len mismatch: {label}"
    return opt_prefix, opt_amplitude, opt_mask, opt_event_len


def _make_sparse_blocks(seed=0):
    generator = torch.Generator().manual_seed(seed)
    blocks = torch.zeros(196, 6, 64, dtype=torch.long)
    blocks[:, :, 0] = torch.randint(-189, 190, (196, 6), generator=generator)
    for mcu_i in range(196):
        for block_i in range(6):
            positions = torch.randperm(63, generator=generator)[:10] + 1
            values = torch.randint(-189, 190, (10,), generator=generator)
            values[values == 0] = 1
            blocks[mcu_i, block_i, positions] = values
    return blocks


def _make_boundary_blocks():
    blocks = torch.zeros(196, 6, 64, dtype=torch.long)
    blocks[:, :, 0] = 189
    blocks[1::2, :, 0] = -189
    blocks[0, 0, 17] = 5
    blocks[0, 0, 34] = -7
    blocks[0, 1, 63] = 9
    blocks[0, 2, 1] = -189
    blocks[0, 3, 16] = 189
    blocks[0, 3, 32] = -3
    blocks[0, 4, 48] = 4
    blocks[0, 5, 62] = -4
    return blocks


def test_synthetic_parity():
    for seed in range(5):
        _assert_pack_equal(_make_sparse_blocks(seed), lmax=320, label=f"synthetic seed={seed}")
    print("synthetic parity OK")


def test_boundary_and_truncation_parity():
    blocks = _make_boundary_blocks()
    for lmax in (6, 8, 64, 320):
        prefix, _amplitude, mask, event_len = _assert_pack_equal(
            blocks, lmax=lmax, label=f"boundary lmax={lmax}"
        )
        assert prefix.shape == (196, lmax)
        assert mask.shape == (196, lmax)
        assert int(event_len.max()) <= lmax
    full_prefix, _amplitude, _mask, _event_len = ds.pack_huffman_event_tensors_direct(blocks, lmax=320)
    assert int((full_prefix == ds.HUFFMAN_PREFIX_Y_AC_OFFSET + 0xF0).sum()) > 0
    assert int((full_prefix == ds.HUFFMAN_PREFIX_Y_AC_OFFSET).sum()) > 0
    print("boundary/ZRL/EOB/truncation parity OK")


def test_lut_reference_parity():
    for value in (-2048, -1024, -378, -189, -1, 0, 1, 189, 378, 1016, 2048):
        assert ds._JPEG_CATEGORY_LUT[value + ds._JPEG_CAT_OFFSET] == ds._jpeg_category(value)
    print("category LUT parity OK")


def test_real_image_parity(img_path):
    _new_dim, _quant, y_out, cbcr_out = ds.load_augmented_mcu_coefficients(
        img_path, type="val", stage="quant_dct"
    )
    blocks = ds._zigzag_blocks_for_rlc(y_out, cbcr_out)
    _assert_pack_equal(blocks, lmax=320, label=img_path)
    print("real image parity OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", default=os.path.join(REPO, "assets", "indexbase_val.csv"))
    parser.add_argument("--basepath", default="/path/to/imagenet1k/imagenet1k")
    parser.add_argument("--image", default=None)
    parser.add_argument("--skip_real", action="store_true")
    args = parser.parse_args()

    test_lut_reference_parity()
    test_synthetic_parity()
    test_boundary_and_truncation_parity()
    if not args.skip_real:
        img_path = args.image or _first_image(args.index_path, args.basepath)
        print(f"image={img_path}")
        test_real_image_parity(img_path)


if __name__ == "__main__":
    main()
