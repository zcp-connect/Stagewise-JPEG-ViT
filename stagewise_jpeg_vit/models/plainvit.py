import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math

from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Reduce, Rearrange
from torch import Tensor, nn

import utils.dct_ops as dops


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

NUMERIC_STAGES = {
    "quant_dct_zigzag_linear",
    "dc_abs",
    "dc_diff",
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
RLC_CP_DIRECT_VARIANT_STAGES = {
    "rlc_dcac_2d_cp_concat_mean_direct",
    "rlc_dcac_2d_cp_add_conv_direct",
    "rlc_dcac_2d_cp_concat_conv_direct",
    "rlc_dcac_2d_cp_film_conv_direct",
}
HUFFMAN_CP_DIRECT_VARIANT_STAGES = {
    "huff_dcac_2d_cp_concat_mean_direct",
    "huff_dcac_2d_cp_add_conv_direct",
    "huff_dcac_2d_cp_concat_conv_direct",
    "huff_dcac_2d_cp_film_conv_direct",
}


def normalize_stage_name(stage):
    key = str(stage).lower()
    if key not in STAGE_ALIASES:
        raise ValueError(f"Unsupported JPEG stage: {stage}. Supported: {sorted(STAGE_ALIASES)}")
    return STAGE_ALIASES[key]


class SinCosEmbedding(nn.Module):
    """
    Fixed 2-D sin-cos positional embedding for image-level 14x14 tokens.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        if len(x.shape) == 4:
            _, h, w, e = x.shape
        elif len(x.shape) == 5:
            _, _, h, w, e = x.shape
        else:
            raise ValueError(f"SinCosEmbedding expects 4-D or 5-D input, got {x.shape}")

        assert e % 4 == 0, "Embedding size should be multiple of 4"
        posemb_pos_h = torch.arange(h, device=x.device)
        posemb_pos_w = torch.arange(w, device=x.device)
        hgrid, wgrid = torch.meshgrid(posemb_pos_h, posemb_pos_w, indexing='ij')

        posemb_freq = torch.log(torch.tensor(10000, dtype=torch.int32, device=x.device)) / (e // 4 - 1)
        posemb_freq = torch.exp(-torch.arange(e // 4, dtype=x.dtype, device=x.device) * posemb_freq)

        posemb_posfreq_h = torch.einsum("p,f->pf", hgrid.flatten(), posemb_freq)
        posemb_posfreq_w = torch.einsum("p,f->pf", wgrid.flatten(), posemb_freq)
        posemb_comb = torch.cat(
            (
                posemb_posfreq_w.sin(),
                posemb_posfreq_w.cos(),
                posemb_posfreq_h.sin(),
                posemb_posfreq_h.cos(),
            ),
            dim=-1,
        ).view((1, h, w, e))

        if len(x.shape) == 5:
            posemb_comb = posemb_comb.unsqueeze(1)

        return x + posemb_comb


def mcu_1d_sincos_pos_embed(length, emb_size, device=None, dtype=torch.float32):
    assert emb_size % 2 == 0, "MCU intra positional embedding size should be even"
    positions = torch.arange(length, device=device, dtype=dtype)
    omega = torch.arange(emb_size // 2, device=device, dtype=dtype)
    omega = torch.exp(
        -omega * (torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(emb_size // 2 - 1, 1))
    )
    pos = torch.einsum("l,d->ld", positions, omega)
    pos = torch.cat((pos.sin(), pos.cos()), dim=-1)
    return pos.view(1, 1, length, emb_size)


def add_spatial_sincos(tokens, grid_size, sincos):
    tokens = rearrange(tokens, 'b (h w) e -> b h w e', h=grid_size[0], w=grid_size[1])
    tokens = sincos(tokens)
    return rearrange(tokens, 'b h w e -> b (h w) e')


def patch2subblock(patch_size, generate_matrix=True, dtype=torch.float32, device='cpu'):
    if patch_size > 8:
        patch_dim = patch_size // 8
        convmat = dops.generate_conversion_matrix(length_small=8, mult=patch_dim, scale=True, dtype=dtype, device=device) if generate_matrix else None
    elif patch_size == 8:
        patch_dim = 1
        convmat = None
    else:
        patch_dim = 8 // patch_size
        convmat = dops.generate_conversion_matrix(length_small=patch_size, mult=patch_dim, scale=True, dtype=dtype, device=device) if generate_matrix else None
    return convmat, patch_dim


def apply_subblock(coeff, convmat, combine=True):
    if convmat is None:
        return coeff
    if combine:
        coeff = torch.einsum("i o, b c h w o j -> b c h w i j", convmat, coeff)
        coeff = torch.einsum("b c h w i o, o j -> b c h w i j", coeff, convmat.T)
    else:
        coeff = torch.einsum("i o, b c h w o j -> b c h w i j", convmat.T, coeff)
        coeff = torch.einsum("b c h w i o, o j -> b c h w i j", coeff, convmat)
    return coeff


def patch2rearrange(patch_size, patch_dim):
    if patch_size >= 8:
        return Rearrange('b c (h pdh) (w pdw) p1 p2 -> b c h w (pdh p1) (pdw p2)', pdh=patch_dim, pdw=patch_dim), True
    return Rearrange('b c h w (p1 pdh) (p2 pdw) -> b c (h pdh) (w pdw) p1 p2', pdh=patch_dim, pdw=patch_dim), False


def unpack_mcu_payload_and_mask(mcu_input, device):
    if isinstance(mcu_input, dict):
        payload_bytes = mcu_input["payload_bytes"]
        byte_mask = mcu_input.get("byte_mask", payload_bytes != 0)
    elif isinstance(mcu_input, (tuple, list)):
        payload_bytes, byte_mask = mcu_input[:2]
    else:
        payload_bytes = mcu_input
        byte_mask = payload_bytes != 0

    payload_bytes = payload_bytes.to(device=device, dtype=torch.long)
    byte_mask = byte_mask.to(device=device, dtype=torch.bool)
    return payload_bytes, byte_mask


def downsample_byte_mask(byte_mask, batch_size, num_mcus):
    byte_mask = rearrange(byte_mask.float(), 'b n l -> (b n) 1 l')
    byte_mask = F.max_pool1d(byte_mask, kernel_size=3, stride=2, padding=1)
    return rearrange(byte_mask > 0, '(b n) 1 l -> b n l 1', b=batch_size, n=num_mcus)


class PatchEmbedding_MCU_Bytes(nn.Module):
    """
    Continuous baseline: uint8 MCU payload -> x / 255.0 -> Linear -> MCU token.
    """
    def __init__(self, lmax=320, emb_size=192, grid_size=(14, 14), device=None, dtype=None):
        super().__init__()
        self.projection = nn.Linear(lmax, emb_size, device=device, dtype=dtype)
        self.grid_size = grid_size
        self.sincos = SinCosEmbedding()

    def forward(self, x):
        payload_bytes, _ = unpack_mcu_payload_and_mask(x, self.projection.weight.device)
        x = payload_bytes.to(dtype=self.projection.weight.dtype)
        x = self.projection(x / 255.0)
        return add_spatial_sincos(x, self.grid_size, self.sincos)


class PatchEmbedding_MCU_DiscreteMean(nn.Module):
    """
    Bag-of-bytes discrete baseline.
    """
    def __init__(self, lmax=320, emb_size=192, inner_dim=64, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.byte_embed = nn.Embedding(257, inner_dim, padding_idx=256, device=device, dtype=dtype)
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def forward(self, mcu_input):
        payload_bytes, byte_mask = unpack_mcu_payload_and_mask(mcu_input, self.byte_embed.weight.device)
        payload_tokens = payload_bytes.masked_fill(~byte_mask, 256)

        byte_features = self.byte_embed(payload_tokens)
        mask = byte_mask.unsqueeze(-1).to(dtype=byte_features.dtype)
        mcu_features = (byte_features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        mcu_features = self.projection(mcu_features)
        return add_spatial_sincos(mcu_features, self.grid_size, self.sincos)


class PatchEmbedding_MCU_DiscreteConv(nn.Module):
    """
    uint8 MCU payload -> byte embedding -> intra-MCU PE -> Conv1d pooling -> MCU token.
    """
    def __init__(self, lmax=320, emb_size=192, inner_dim=32, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.byte_embed = nn.Embedding(257, inner_dim, padding_idx=256, device=device, dtype=dtype)
        self.register_buffer(
            "intra_pos",
            mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype),
            persistent=False,
        )
        self.conv = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            nn.GELU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def forward(self, x):
        payload_bytes, byte_mask = unpack_mcu_payload_and_mask(x, self.byte_embed.weight.device)
        payload_tokens = payload_bytes.masked_fill(~byte_mask, 256)

        x = self.byte_embed(payload_tokens)
        mask = byte_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x + self.intra_pos.to(device=x.device, dtype=x.dtype) * mask
        x = x * mask

        batch_size, num_mcus, _, _ = x.shape
        x = rearrange(x, 'b n l c -> (b n) c l')
        x = self.conv(x)
        x = rearrange(x, '(b n) c l -> b n l c', b=batch_size, n=num_mcus)
        mask = downsample_byte_mask(byte_mask, batch_size, num_mcus).to(device=x.device, dtype=x.dtype)

        x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        x = self.projection(x)
        return add_spatial_sincos(x, self.grid_size, self.sincos)


class PatchEmbedding_MCU_BitVectorConv(nn.Module):
    """
    uint8 MCU payload -> 8-bit MSB-first vector -> Linear -> Conv1d pooling.
    """
    def __init__(self, lmax=320, emb_size=192, inner_dim=32, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.bit_projection = nn.Linear(8, inner_dim, device=device, dtype=dtype)
        self.register_buffer("bit_shifts", torch.arange(7, -1, -1, device=device, dtype=torch.long), persistent=False)
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            nn.GELU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def forward(self, mcu_input):
        payload_bytes, byte_mask = unpack_mcu_payload_and_mask(mcu_input, self.bit_projection.weight.device)
        bit_features = ((payload_bytes.unsqueeze(-1) >> self.bit_shifts) & 1).to(dtype=self.bit_projection.weight.dtype)
        byte_features = self.bit_projection(bit_features)

        mask = byte_mask.unsqueeze(-1).to(dtype=byte_features.dtype)
        byte_features = byte_features + self.intra_pos.to(device=byte_features.device, dtype=byte_features.dtype) * mask
        byte_features = byte_features * mask

        batch_size, num_mcus, _, _ = byte_features.shape
        byte_features = rearrange(byte_features, 'b n l c -> (b n) c l')
        byte_features = self.conv(byte_features)
        byte_features = rearrange(byte_features, '(b n) c l -> b n l c', b=batch_size, n=num_mcus)
        mask = downsample_byte_mask(byte_mask, batch_size, num_mcus).to(device=byte_features.device, dtype=byte_features.dtype)

        mcu_features = (byte_features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        mcu_features = self.projection(mcu_features)
        return add_spatial_sincos(mcu_features, self.grid_size, self.sincos)


class PatchEmbedding_ImagePatch(nn.Module):
    """
    RGB/YCbCr numeric image stage: [B,3,224,224] -> [B,196,D].
    """
    def __init__(self, in_channels=3, patch_size=16, emb_size=384, device=None, dtype=torch.float32):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)
        self.sincos = SinCosEmbedding()

    def forward(self, x):
        if isinstance(x, dict):
            x = x["image"]
        x = x.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype)
        x = self.projection(x)
        x = rearrange(x, 'b e h w -> b h w e')
        x = self.sincos(x)
        return rearrange(x, 'b h w e -> b (h w) e')


class PatchEmbedding_NumericStage(nn.Module):
    """
    Numeric JPEG stage: [B,196,F] -> [B,196,D].
    """
    def __init__(self, feature_dim, emb_size=384, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim, device=device, dtype=dtype),
            nn.Linear(feature_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def forward(self, x):
        if isinstance(x, dict):
            x = x["features"]
        x = x.to(device=self.projection[1].weight.device, dtype=self.projection[1].weight.dtype)
        x = self.projection(x)
        return add_spatial_sincos(x, self.grid_size, self.sincos)


class PatchEmbedding_DCT_Group(nn.Module):
    """
    DCT-aware patch embedding aligned with the original DCT baseline.

    Input is a structured tuple/dict of quantized DCT tensors:
        Y:    [B, 1, 28, 28, 8, 8]
        CbCr: [B, 2, 14, 14, 8, 8]
    Output:
        [B, 196, emb_size]
    """
    def __init__(
            self,
            patch_size: int = 16,
            emb_size: int = 384,
            use_subblock=True,
            chroma_scale=2,
            device=None,
            dtype=torch.float32,
        ):
        super().__init__()
        assert not (patch_size & (patch_size - 1)) and patch_size >= 2
        patch_size_c = patch_size // chroma_scale
        if patch_size < 8:
            assert use_subblock
        if patch_size_c < 8:
            assert use_subblock

        self.conv_y, patch_dim_y = patch2subblock(patch_size, generate_matrix=use_subblock, dtype=dtype, device=device)
        self.conv_c, patch_dim_c = patch2subblock(patch_size_c, generate_matrix=use_subblock, dtype=dtype, device=device)
        self.rearrange_y, self.combine_y = patch2rearrange(patch_size, patch_dim_y)
        self.rearrange_c, self.combine_c = patch2rearrange(patch_size_c, patch_dim_c)
        self.collapser = Rearrange('b c h w i j -> b h w (c i j)')
        self.projection = nn.Sequential(
            nn.Linear(patch_size ** 2 + 2 * patch_size_c ** 2, emb_size, device=device, dtype=dtype),
            SinCosEmbedding(),
            Rearrange('b h w e -> b (h w) e'),
        )

    def _unpack(self, x):
        if isinstance(x, dict):
            y = x["y"]
            cbcr = x["cbcr"]
        elif isinstance(x, (tuple, list)):
            y, cbcr = x[:2]
        else:
            raise ValueError("PatchEmbedding_DCT_Group expects dict or tuple input")
        device = self.projection[0].weight.device
        dtype = self.projection[0].weight.dtype
        return y.to(device=device, dtype=dtype), cbcr.to(device=device, dtype=dtype)

    def forward(self, x):
        y, cbcr = self._unpack(x)
        if self.combine_y:
            y = self.rearrange_y(y)
            y = apply_subblock(y, self.conv_y, combine=True)
        else:
            y = apply_subblock(y, self.conv_y, combine=False)
            y = self.rearrange_y(y)
        if self.combine_c:
            cbcr = self.rearrange_c(cbcr)
            cbcr = apply_subblock(cbcr, self.conv_c, combine=True)
        else:
            cbcr = apply_subblock(cbcr, self.conv_c, combine=False)
            cbcr = self.rearrange_c(cbcr)
        y = self.collapser(y)
        cbcr = self.collapser(cbcr)
        ycbcr = torch.cat([y, cbcr], dim=3)
        return self.projection(ycbcr)


class PatchEmbedding_RLCEvent(nn.Module):
    """
    Pre-Huffman syntax-event tokenizer.

    Input tuple/dict fields:
        kind, component, block_id, zigzag_pos, run, size, amplitude, mask
    Output:
        one ViT token per MCU, preserving the 14x14 token grid.
    """
    def __init__(self, lmax=384, emb_size=384, inner_dim=64, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.kind_embed = nn.Embedding(5, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.component_embed = nn.Embedding(4, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.block_embed = nn.Embedding(7, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.zigzag_embed = nn.Embedding(65, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.run_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.size_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.amp_proj = nn.Sequential(
            nn.Linear(2, inner_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype),
            nn.GELU(),
            nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            fields = [x[k] for k in ("kind", "component", "block", "zigzag", "run", "size", "amplitude", "mask")]
        else:
            fields = list(x)
        kind, component, block, zigzag, run, size, amplitude, mask = fields
        return (
            kind.to(device=device, dtype=torch.long).clamp(0, 4),
            component.to(device=device, dtype=torch.long).clamp(0, 3),
            block.to(device=device, dtype=torch.long).clamp(0, 6),
            zigzag.to(device=device, dtype=torch.long).clamp(0, 64),
            run.to(device=device, dtype=torch.long).clamp(0, 16),
            size.to(device=device, dtype=torch.long).clamp(0, 16),
            amplitude.to(device=device, dtype=self.amp_proj[0].weight.dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.kind_embed.weight.device
        kind, component, block, zigzag, run, size, amplitude, mask = self._unpack(x, device)
        amp = torch.stack((amplitude / 2048.0, torch.sign(amplitude)), dim=-1)
        features = (
            self.kind_embed(kind)
            + self.component_embed(component)
            + self.block_embed(block)
            + self.zigzag_embed(zigzag)
            + self.run_embed(run)
            + self.size_embed(size)
            + self.amp_proj(amp)
        )
        dtype = features.dtype
        event_mask = mask.unsqueeze(-1).to(dtype=dtype)
        features = features + self.intra_pos.to(device=features.device, dtype=dtype) * event_mask
        features = features * event_mask

        batch_size, num_mcus, _, _ = features.shape
        features = rearrange(features, 'b n l c -> (b n) c l')
        features = self.conv(features)
        features = rearrange(features, '(b n) c l -> b n l c', b=batch_size, n=num_mcus)
        conv_mask = downsample_byte_mask(mask, batch_size, num_mcus).to(device=features.device, dtype=dtype)
        features = (features * conv_mask).sum(dim=2) / conv_mask.sum(dim=2).clamp_min(1.0)
        features = self.projection(features)
        return add_spatial_sincos(features, self.grid_size, self.sincos)


class PatchEmbedding_RLCEvent_Simple(nn.Module):
    """
    Minimal RLC event tokenizer: per-field embeddings + scalar amplitude
    projection + masked mean pooling.
    """
    def __init__(self, lmax=384, emb_size=384, inner_dim=64, grid_size=(14, 14), device=None, dtype=torch.float32):
        super().__init__()
        self.grid_size = grid_size
        self.kind_embed = nn.Embedding(5, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.component_embed = nn.Embedding(4, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.block_embed = nn.Embedding(7, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.zigzag_embed = nn.Embedding(65, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.run_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.size_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.amp_proj = nn.Linear(1, inner_dim, device=device, dtype=dtype)
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            fields = [x[k] for k in ("kind", "component", "block", "zigzag", "run", "size", "amplitude", "mask")]
        else:
            fields = list(x)
        kind, component, block, zigzag, run, size, amplitude, mask = fields
        amp_dtype = self.amp_proj.weight.dtype
        return (
            kind.to(device=device, dtype=torch.long).clamp(0, 4),
            component.to(device=device, dtype=torch.long).clamp(0, 3),
            block.to(device=device, dtype=torch.long).clamp(0, 6),
            zigzag.to(device=device, dtype=torch.long).clamp(0, 64),
            run.to(device=device, dtype=torch.long).clamp(0, 16),
            size.to(device=device, dtype=torch.long).clamp(0, 16),
            amplitude.to(device=device, dtype=amp_dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.kind_embed.weight.device
        kind, component, block, zigzag, run, size, amplitude, mask = self._unpack(x, device)
        amp = (amplitude / 189.0).unsqueeze(-1)
        features = (
            self.kind_embed(kind)
            + self.component_embed(component)
            + self.block_embed(block)
            + self.zigzag_embed(zigzag)
            + self.run_embed(run)
            + self.size_embed(size)
            + self.amp_proj(amp)
        )
        event_mask = mask.unsqueeze(-1).to(dtype=features.dtype)
        pooled = (features * event_mask).sum(dim=2) / event_mask.sum(dim=2).clamp_min(1.0)
        features = self.projection(pooled)
        return add_spatial_sincos(features, self.grid_size, self.sincos)


class MaskedAttnPool(nn.Module):
    """
    PMA-style pool: variable-length events -> single token via cross-attention
    with a learnable query.
    """
    def __init__(self, dim, num_heads=4, device=None, dtype=torch.float32):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim, device=device, dtype=dtype) * 0.02)
        self.out_proj = nn.Linear(dim, dim, device=device, dtype=dtype)
        self.norm = nn.LayerNorm(dim, device=device, dtype=dtype)

    def forward(self, x, mask):
        bn, lmax, dim = x.shape
        all_pad = ~mask.any(dim=-1, keepdim=True)
        safe_mask = mask.clone()
        safe_mask[:, 0] = safe_mask[:, 0] | all_pad.squeeze(-1)

        x_heads = x.reshape(bn, lmax, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("bhld,hd->bhl", x_heads, self.query.squeeze(0)) * self.scale
        scores = scores.masked_fill(~safe_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bhl,bhld->bhd", weights, x_heads)
        pooled = pooled.reshape(bn, dim)
        pooled = self.out_proj(pooled)
        pooled = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        return self.norm(pooled)


class PatchEmbedding_RLCEvent_AttnPool(nn.Module):
    """
    RLC event tokenizer with multi-field event embeddings and PMA-style masked
    attention pooling. Keeps the ViT token grid at 14x14 MCUs.
    """
    def __init__(
            self, lmax=384, emb_size=384, inner_dim=64, grid_size=(14, 14),
            num_pool_heads=4, device=None, dtype=torch.float32,
        ):
        super().__init__()
        self.lmax = lmax
        self.grid_size = grid_size
        self.inner_dim = inner_dim
        self.kind_embed = nn.Embedding(5, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.component_embed = nn.Embedding(4, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.block_embed = nn.Embedding(7, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.zigzag_embed = nn.Embedding(65, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.run_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.size_embed = nn.Embedding(17, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.amp_proj = nn.Sequential(
            nn.Linear(3, inner_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.pool = MaskedAttnPool(
            inner_dim, num_heads=num_pool_heads, device=device, dtype=dtype
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(inner_dim, device=device, dtype=dtype),
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype),
        )
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            fields = [x[k] for k in ("kind", "component", "block", "zigzag", "run", "size", "amplitude", "mask")]
        else:
            fields = list(x)
        kind, component, block, zigzag, run, size, amplitude, mask = fields
        amp_dtype = self.amp_proj[0].weight.dtype
        return (
            kind.to(device=device, dtype=torch.long).clamp(0, 4),
            component.to(device=device, dtype=torch.long).clamp(0, 3),
            block.to(device=device, dtype=torch.long).clamp(0, 6),
            zigzag.to(device=device, dtype=torch.long).clamp(0, 64),
            run.to(device=device, dtype=torch.long).clamp(0, 16),
            size.to(device=device, dtype=torch.long).clamp(0, 16),
            amplitude.to(device=device, dtype=amp_dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.kind_embed.weight.device
        kind, component, block, zigzag, run, size, amplitude, mask = self._unpack(x, device)
        batch_size, num_mcus, lmax = kind.shape

        size_raw = (size - 1).clamp(min=0).to(dtype=amplitude.dtype)
        amp_sign = torch.sign(amplitude)
        amp_log = torch.log1p(amplitude.abs()) / math.log(2048.0)
        amp_norm = amplitude / torch.pow(torch.tensor(2.0, device=amplitude.device, dtype=amplitude.dtype), size_raw)
        amp_features = torch.stack((amp_sign, amp_log, amp_norm), dim=-1)
        amp_features = amp_features * mask.unsqueeze(-1).to(dtype=amp_features.dtype)

        features = (
            self.kind_embed(kind)
            + self.component_embed(component)
            + self.block_embed(block)
            + self.zigzag_embed(zigzag)
            + self.run_embed(run)
            + self.size_embed(size)
            + self.amp_proj(amp_features)
        )
        features = features.reshape(batch_size * num_mcus, lmax, self.inner_dim)
        pooled = self.pool(features, mask.reshape(batch_size * num_mcus, lmax))
        pooled = pooled.reshape(batch_size, num_mcus, self.inner_dim)
        features = self.projection(pooled)
        return add_spatial_sincos(features, self.grid_size, self.sincos)


class PatchEmbedding_RLCEvent_2D(nn.Module):
    """
    Minimal 2D RLC tokenizer. Each event is [run, value], already normalized
    by the dataset, projected with one Linear(2, D).
    """
    def __init__(
            self, lmax=320, emb_size=384, inner_dim=None, grid_size=(14, 14),
            drop_p=0.0, pool="mean", device=None, dtype=torch.float32,
        ):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"Unsupported RLC 2D pool: {pool}")
        inner_dim = inner_dim or emb_size
        self.lmax = lmax
        self.grid_size = grid_size
        self.pool = pool
        self.token_proj = nn.Linear(2, inner_dim, device=device, dtype=dtype)
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.projection = (
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype)
            if inner_dim != emb_size else nn.Identity()
        )
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            run = x["run"]
            value = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            run, value, mask = list(x)[:3]
        dtype = self.token_proj.weight.dtype
        return (
            run.to(device=device, dtype=dtype),
            value.to(device=device, dtype=dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.token_proj.weight.device
        run, value, mask = self._unpack(x, device)
        tokens = torch.stack((run, value), dim=-1)
        features = self.token_proj(tokens)
        features = features + self.intra_pos[:features.shape[-2]].to(device=features.device, dtype=features.dtype)
        if self.pool == "max":
            neg_inf = torch.finfo(features.dtype).min
            features.masked_fill_(~mask.unsqueeze(-1), neg_inf)
            pooled = features.max(dim=2).values
            all_pad = ~mask.any(dim=2, keepdim=True)
            pooled = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        else:
            event_mask = mask.unsqueeze(-1).to(dtype=features.dtype)
            pooled = (features * event_mask).sum(dim=2) / event_mask.sum(dim=2).clamp_min(1.0)
        features = self.projection(pooled)
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class PatchEmbedding_RLCEvent_Flat(nn.Module):
    """
    Flattened 2D RLC tokenizer. Each MCU token is formed by zeroing padded
    [run, value] events, flattening [L, 2] to [2L], then applying one Linear.
    """
    def __init__(
            self, lmax=320, emb_size=384, grid_size=(14, 14),
            drop_p=0.0, device=None, dtype=torch.float32,
        ):
        super().__init__()
        self.lmax = lmax
        self.grid_size = grid_size
        self.projection = nn.Linear(lmax * 2, emb_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            run = x["run"]
            value = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            run, value, mask = list(x)[:3]
        dtype = self.projection.weight.dtype
        return (
            run.to(device=device, dtype=dtype),
            value.to(device=device, dtype=dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.projection.weight.device
        run, value, mask = self._unpack(x, device)
        tokens = torch.stack((run, value), dim=-1)
        tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
        features = self.projection(tokens.flatten(-2))
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class PatchEmbedding_EventAmplitude_Flat(nn.Module):
    def __init__(
            self, lmax=320, emb_size=384, grid_size=(14, 14),
            drop_p=0.0, device=None, dtype=torch.float32,
        ):
        super().__init__()
        self.lmax = lmax
        self.grid_size = grid_size
        self.projection = nn.Linear(lmax, emb_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            value = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            fields = list(x)
            value = fields[1]
            mask = fields[2]
        dtype = self.projection.weight.dtype
        return (
            value.to(device=device, dtype=dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.projection.weight.device
        value, mask = self._unpack(x, device)
        value = value * mask.to(dtype=value.dtype)
        features = self.projection(value)
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class PatchEmbedding_RLCEvent_CP(nn.Module):
    """
    CP-Transformer style 2D RLC tokenizer: discrete run -> Embedding,
    continuous value -> Linear, summed per event before masked pooling.
    """
    def __init__(
            self, lmax=320, emb_size=384, inner_dim=None, grid_size=(14, 14),
            drop_p=0.0, pool="mean", device=None, dtype=torch.float32,
        ):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"Unsupported RLC CP pool: {pool}")
        inner_dim = inner_dim or emb_size
        self.lmax = lmax
        self.grid_size = grid_size
        self.pool = pool
        self.run_embed = nn.Embedding(16, inner_dim, device=device, dtype=dtype)
        self.value_proj = nn.Linear(1, inner_dim, device=device, dtype=dtype)
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.projection = (
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype)
            if inner_dim != emb_size else nn.Identity()
        )
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            run_int = x["run"]
            value = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            run_int, value, mask = list(x)[:3]
        return (
            run_int.to(device=device, dtype=torch.long).clamp(0, 15),
            value.to(device=device, dtype=self.value_proj.weight.dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.run_embed.weight.device
        run_int, value, mask = self._unpack(x, device)
        if self.pool == "max":
            features = self.run_embed(run_int) + self.value_proj(value.unsqueeze(-1))
            features = features + self.intra_pos[:features.shape[-2]].to(device=features.device, dtype=features.dtype)
            neg_inf = torch.finfo(features.dtype).min
            features.masked_fill_(~mask.unsqueeze(-1), neg_inf)
            pooled = features.max(dim=2).values
            all_pad = ~mask.any(dim=2, keepdim=True)
            pooled = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        else:
            dtype = self.run_embed.weight.dtype
            event_mask = mask.to(dtype=dtype)
            counts = event_mask.sum(dim=2).clamp_min(1.0)
            pooled = self.value_proj.bias.view(1, 1, -1) * counts.unsqueeze(-1)
            pooled = pooled + event_mask.to(value.dtype).mul(value).sum(dim=2).unsqueeze(-1) * self.value_proj.weight.view(1, 1, -1)
            pos = self.intra_pos[0, 0, :run_int.shape[-1], :].to(device=device, dtype=dtype)
            pooled = pooled + torch.matmul(event_mask, pos)
            run_weight = self.run_embed.weight
            for run_id in range(run_weight.shape[0]):
                run_counts = ((run_int == run_id) & mask).sum(dim=2).to(dtype=dtype)
                pooled = pooled + run_counts.unsqueeze(-1) * run_weight[run_id].view(1, 1, -1)
            pooled = pooled / counts.unsqueeze(-1)
        features = self.projection(pooled)
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class PatchEmbedding_HuffmanEvent_CP(nn.Module):
    """
    CP-style Huffman-symbol tokenizer: discrete prefix id -> Embedding,
    continuous signed amplitude -> Linear, summed per event before masked pooling.
    """
    def __init__(
            self, lmax=320, emb_size=384, inner_dim=None, prefix_vocab_size=1025,
            grid_size=(14, 14), drop_p=0.0, pool="mean", device=None, dtype=torch.float32,
        ):
        super().__init__()
        if pool not in ("mean", "max"):
            raise ValueError(f"Unsupported Huffman CP pool: {pool}")
        inner_dim = inner_dim or emb_size
        self.lmax = lmax
        self.grid_size = grid_size
        self.pool = pool
        self.prefix_embed = nn.Embedding(prefix_vocab_size, inner_dim, padding_idx=0, device=device, dtype=dtype)
        self.amplitude_proj = nn.Linear(1, inner_dim, device=device, dtype=dtype)
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.projection = (
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype)
            if inner_dim != emb_size else nn.Identity()
        )
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            prefix_id = x["prefix_id"]
            amplitude = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            prefix_id, amplitude, mask = list(x)[:3]
        return (
            prefix_id.to(device=device, dtype=torch.long).clamp(0, self.prefix_embed.num_embeddings - 1),
            amplitude.to(device=device, dtype=self.amplitude_proj.weight.dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def forward(self, x):
        device = self.prefix_embed.weight.device
        prefix_id, amplitude, mask = self._unpack(x, device)
        features = self.prefix_embed(prefix_id) + self.amplitude_proj(amplitude.unsqueeze(-1))
        features = features + self.intra_pos[:features.shape[-2]].to(device=features.device, dtype=features.dtype)
        if self.pool == "max":
            neg_inf = torch.finfo(features.dtype).min
            features.masked_fill_(~mask.unsqueeze(-1), neg_inf)
            pooled = features.max(dim=2).values
            all_pad = ~mask.any(dim=2, keepdim=True)
            pooled = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        else:
            event_mask = mask.unsqueeze(-1).to(dtype=features.dtype)
            pooled = (features * event_mask).sum(dim=2) / event_mask.sum(dim=2).clamp_min(1.0)
        features = self.projection(pooled)
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class PatchEmbedding_EventCPFusion(nn.Module):
    """
    CP-style event tokenizer variants for RLC/Huffman direct stages.
    Discrete symbol id -> Embedding, continuous amplitude -> Linear, then
    configurable event fusion and sequence aggregation.
    """
    def __init__(
            self, symbol_vocab_size, lmax=320, emb_size=384, inner_dim=None,
            grid_size=(14, 14), drop_p=0.0, fusion="concat", sequence="mean",
            padding_idx=None, chunk_size=8192, device=None, dtype=torch.float32,
        ):
        super().__init__()
        if fusion not in ("add", "concat", "film"):
            raise ValueError(f"Unsupported CP fusion: {fusion}")
        if sequence not in ("mean", "conv"):
            raise ValueError(f"Unsupported CP sequence aggregation: {sequence}")
        inner_dim = inner_dim or emb_size
        self.lmax = lmax
        self.grid_size = grid_size
        self.fusion = fusion
        self.sequence = sequence
        self.chunk_size = chunk_size
        self.symbol_embed = nn.Embedding(
            symbol_vocab_size, inner_dim, padding_idx=padding_idx, device=device, dtype=dtype
        )
        self.amplitude_proj = nn.Linear(1, inner_dim, device=device, dtype=dtype)
        self.concat_fuse = (
            nn.Linear(inner_dim * 2, inner_dim, device=device, dtype=dtype)
            if fusion == "concat" else None
        )
        self.film = (
            nn.Linear(inner_dim, inner_dim * 2, device=device, dtype=dtype)
            if fusion == "film" else None
        )
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.register_buffer("intra_pos", mcu_1d_sincos_pos_embed(lmax, inner_dim, device=device, dtype=dtype), persistent=False)
        self.conv1 = None
        self.conv2 = None
        self.conv_act = None
        if sequence == "conv":
            self.conv1 = nn.Conv1d(inner_dim, inner_dim, kernel_size=3, stride=2, padding=1, device=device, dtype=dtype)
            self.conv2 = nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=1, device=device, dtype=dtype)
            self.conv_act = nn.GELU()
        self.projection = (
            nn.Linear(inner_dim, emb_size, device=device, dtype=dtype)
            if inner_dim != emb_size else nn.Identity()
        )
        self.dropout = nn.Dropout(drop_p)
        self.sincos = SinCosEmbedding()

    def _unpack(self, x, device):
        if isinstance(x, dict):
            symbol_id = x["prefix_id"] if "prefix_id" in x else x["run"]
            amplitude = x["amplitude"] if "amplitude" in x else x["value"]
            mask = x["mask"]
        else:
            symbol_id, amplitude, mask = list(x)[:3]
        return (
            symbol_id.to(device=device, dtype=torch.long).clamp(0, self.symbol_embed.num_embeddings - 1),
            amplitude.to(device=device, dtype=self.amplitude_proj.weight.dtype),
            mask.to(device=device, dtype=torch.bool),
        )

    def _fuse(self, symbol_features, amplitude_features):
        if self.fusion == "add":
            return symbol_features + amplitude_features
        if self.fusion == "concat":
            return self.concat_fuse(torch.cat((symbol_features, amplitude_features), dim=-1))
        gamma_beta = self.film(symbol_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return symbol_features + (1.0 + gamma) * amplitude_features + beta

    @staticmethod
    def _masked_mean(features, mask):
        event_mask = mask.to(dtype=features.dtype)
        denom = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (features * event_mask.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _downsample_mask(mask):
        pooled = mask.float().unsqueeze(1)
        pooled = F.max_pool1d(pooled, kernel_size=3, stride=2, padding=1)
        return pooled.squeeze(1) > 0

    def _conv_pool(self, features, mask):
        features = features * mask.unsqueeze(-1).to(dtype=features.dtype)
        features = rearrange(features, 'c l d -> c d l')
        mask = self._downsample_mask(mask).to(device=features.device)
        features = self.conv_act(self.conv1(features))
        features = rearrange(features, 'c d l -> c l d')
        features = features * mask.unsqueeze(-1).to(dtype=features.dtype)
        features = rearrange(features, 'c l d -> c d l')
        features = self.conv_act(self.conv2(features))
        features = rearrange(features, 'c d l -> c l d')
        features = features * mask.unsqueeze(-1).to(dtype=features.dtype)
        return self._masked_mean(features, mask)

    def _process_chunk(self, symbol_id, amplitude, mask):
        symbol_features = self.symbol_embed(symbol_id)
        amplitude_features = self.amplitude_proj(amplitude.unsqueeze(-1))
        features = self._fuse(symbol_features, amplitude_features)
        intra_pos = self.intra_pos.reshape(-1, self.intra_pos.shape[-1])[:features.shape[-2]]
        features = features + intra_pos.to(device=features.device, dtype=features.dtype)
        return self._conv_pool(features, mask) if self.sequence == "conv" else self._masked_mean(features, mask)

    def forward(self, x):
        device = self.symbol_embed.weight.device
        symbol_id, amplitude, mask = self._unpack(x, device)
        batch_size, num_mcus, lmax = symbol_id.shape
        symbol_id = symbol_id.reshape(batch_size * num_mcus, lmax)
        amplitude = amplitude.reshape(batch_size * num_mcus, lmax)
        mask = mask.reshape(batch_size * num_mcus, lmax)
        if symbol_id.shape[0] <= self.chunk_size:
            pooled = self._process_chunk(symbol_id, amplitude, mask)
        else:
            process_chunk = (
                lambda symbol_chunk, amplitude_chunk, mask_chunk: checkpoint.checkpoint(
                    self._process_chunk,
                    symbol_chunk,
                    amplitude_chunk,
                    mask_chunk,
                    use_reentrant=False,
                )
            ) if self.training else self._process_chunk
            pooled = torch.cat([
                process_chunk(
                    symbol_id[start:start + self.chunk_size],
                    amplitude[start:start + self.chunk_size],
                    mask[start:start + self.chunk_size],
                )
                for start in range(0, symbol_id.shape[0], self.chunk_size)
            ], dim=0)
        features = self.projection(pooled).reshape(batch_size, num_mcus, -1)
        features = add_spatial_sincos(features, self.grid_size, self.sincos)
        return self.dropout(features)


class MultiHeadAttention(nn.Module):
    def __init__(
            self, emb_size: int = 768, input_embed: int = -1,
            num_heads: int = 8, head_size: int = 64, dropout: float = 0, nohidden=False,
            device='cpu', dtype=torch.float32,
        ):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.nohidden = nohidden
        inner_size = num_heads * head_size
        if input_embed < 0:
            input_embed = emb_size

        if nohidden:
            assert inner_size == emb_size, "num_heads*head_size should equal emb_size without hidden projection."
            self.qkv = nn.Linear(input_embed, inner_size * 3, device=device, dtype=dtype)
            self.att_drop = nn.Dropout(dropout)
        else:
            self.qkv = nn.Linear(input_embed, inner_size * 3, device=device, dtype=dtype)
            self.att_drop = nn.Dropout(dropout)
            self.projection = nn.Linear(inner_size, emb_size, device=device, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)

        scaling = queries.shape[-1] ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        if not self.nohidden:
            out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0., device='cpu', dtype=torch.float32):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size, device=device, dtype=dtype),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
            self,
            emb_size: int = 768,
            drop_p: float = 0.,
            forward_expansion: int = 4,
            forward_drop_p: float = 0.,
            input_embed: int = -1,
            device='cpu',
            dtype=torch.float32,
            **kwargs,
        ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                OrderedDict(
                    [
                        ("eb_lrnorm1", nn.LayerNorm(emb_size, device=device, dtype=dtype)),
                        ("eb_mha", MultiHeadAttention(
                            emb_size, input_embed=input_embed, device=device, dtype=dtype, nohidden=False, **kwargs
                        )),
                        ("eb_drop1", nn.Dropout(drop_p)),
                    ]
                )
            )),
            ResidualAdd(nn.Sequential(
                OrderedDict(
                    [
                        ("eb_lrnorm2", nn.LayerNorm(emb_size, device=device, dtype=dtype)),
                        ("eb_ffb", FeedForwardBlock(
                            emb_size, expansion=forward_expansion, drop_p=forward_drop_p, device=device, dtype=dtype
                        )),
                        ("eb_drop2", nn.Dropout(drop_p)),
                    ]
                )
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, input_embed=-1, **kwargs):
        if input_embed < 0:
            super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        else:
            super().__init__(
                TransformerEncoderBlock(input_embed=input_embed, **kwargs),
                *[TransformerEncoderBlock(**kwargs) for _ in range(depth - 1)]
            )


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000, device='cpu', dtype=torch.float32):
        super().__init__(
            OrderedDict(
                [
                    ("ch_lrnorm", nn.LayerNorm(emb_size, device=device, dtype=dtype)),
                    ("ch_gap", Reduce('b n e -> b e', reduction='mean')),
                    ("ch_linear1", nn.Linear(emb_size, emb_size, device=device, dtype=dtype)),
                    ("ch_tanh", nn.Tanh()),
                    ("ch_linear2", nn.Linear(emb_size, n_classes, device=device, dtype=dtype)),
                ]
            )
        )


class ViT(nn.Module):
    """
    Stage-wise JPEG ViT.
    """
    def __init__(
            self,
            in_channels: int = 3,
            patch_size: int = 16,
            emb_size: int = 768,
            input_embed: int = -1,
            depth: int = 12,
            n_classes: int = 1000,
            drop_p=0.1,
            pixel_space='MCU_BYTES',
            ver=1,
            use_subblock=True,
            stage=None,
            numeric_feature_dim=384,
            rlc_lmax=320,
            device='cpu',
            dtype=torch.float32,
            **kwargs,
        ):
        super().__init__()
        stage = normalize_stage_name(stage or pixel_space)

        if stage == 'rgb':
            self.patchembed = PatchEmbedding_ImagePatch(in_channels=3, patch_size=patch_size, emb_size=emb_size, device=device, dtype=dtype)
        elif stage == 'ycbcr':
            self.patchembed = PatchEmbedding_ImagePatch(in_channels=3, patch_size=patch_size, emb_size=emb_size, device=device, dtype=dtype)
        elif stage in NUMERIC_STAGES:
            self.patchembed = PatchEmbedding_NumericStage(numeric_feature_dim, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
        elif stage in GROUP_DCT_STAGES:
            self.patchembed = PatchEmbedding_DCT_Group(patch_size=patch_size, emb_size=emb_size, use_subblock=use_subblock, device=device, dtype=dtype)
        elif stage == 'raw_byte':
            if ver == 1:
                self.patchembed = PatchEmbedding_MCU_Bytes(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
            elif ver == 2:
                self.patchembed = PatchEmbedding_MCU_DiscreteMean(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
            elif ver == 3:
                self.patchembed = PatchEmbedding_MCU_DiscreteConv(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
            elif ver == 4:
                self.patchembed = PatchEmbedding_MCU_BitVectorConv(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
            else:
                raise ValueError(f"Unsupported raw_byte embedding type: {ver}. Use 1=linear, 2=mean, 3=discrete conv, 4=bit-vector conv.")
        elif stage == 'bitstream':
            self.patchembed = PatchEmbedding_MCU_BitVectorConv(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
        elif stage in ('rlc_dc', 'rlc_dcac'):
            self.patchembed = PatchEmbedding_RLCEvent(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
        elif stage in ('rlc_dc_simple', 'rlc_dcac_simple'):
            self.patchembed = PatchEmbedding_RLCEvent_Simple(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
        elif stage in ('rlc_dc_attnpool', 'rlc_dcac_attnpool'):
            self.patchembed = PatchEmbedding_RLCEvent_AttnPool(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), device=device, dtype=dtype)
        elif stage in (
            'rlc_dc_2d', 'rlc_dcac_2d', 'rlc_dc_2d_direct', 'rlc_dcac_2d_direct',
            'huff_dcac_2d_direct',
        ):
            self.patchembed = PatchEmbedding_RLCEvent_2D(lmax=rlc_lmax, emb_size=emb_size, inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14), drop_p=drop_p, device=device, dtype=dtype)
        elif stage in ('rlc_dc_2d_max', 'rlc_dcac_2d_max', 'rlc_dc_2d_max_direct', 'rlc_dcac_2d_max_direct'):
            self.patchembed = PatchEmbedding_RLCEvent_2D(lmax=rlc_lmax, emb_size=emb_size, inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14), drop_p=drop_p, pool="max", device=device, dtype=dtype)
        elif stage in (
            'rlc_dc_flat', 'rlc_dcac_flat', 'rlc_dc_flat_direct', 'rlc_dcac_flat_direct',
            'huff_dcac_flat_direct',
        ):
            self.patchembed = PatchEmbedding_RLCEvent_Flat(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), drop_p=drop_p, device=device, dtype=dtype)
        elif stage == 'event_amp_only_flat_direct':
            self.patchembed = PatchEmbedding_EventAmplitude_Flat(lmax=rlc_lmax, emb_size=emb_size, grid_size=(14, 14), drop_p=drop_p, device=device, dtype=dtype)
        elif stage in ('rlc_dc_2d_cp', 'rlc_dcac_2d_cp', 'rlc_dc_2d_cp_direct', 'rlc_dcac_2d_cp_direct'):
            self.patchembed = PatchEmbedding_RLCEvent_CP(lmax=rlc_lmax, emb_size=emb_size, inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14), drop_p=drop_p, pool="mean", device=device, dtype=dtype)
        elif stage == 'huff_dcac_2d_cp_direct':
            self.patchembed = PatchEmbedding_HuffmanEvent_CP(lmax=rlc_lmax, emb_size=emb_size, inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14), drop_p=drop_p, pool="mean", device=device, dtype=dtype)
        elif stage in RLC_CP_DIRECT_VARIANT_STAGES:
            if "concat_mean" in stage:
                fusion, sequence = "concat", "mean"
            elif "add_conv" in stage:
                fusion, sequence = "add", "conv"
            elif "concat_conv" in stage:
                fusion, sequence = "concat", "conv"
            elif "film_conv" in stage:
                fusion, sequence = "film", "conv"
            else:
                raise ValueError(f"Unsupported RLC CP variant: {stage}")
            self.patchembed = PatchEmbedding_EventCPFusion(
                symbol_vocab_size=16, lmax=rlc_lmax, emb_size=emb_size,
                inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14),
                drop_p=drop_p, fusion=fusion, sequence=sequence,
                padding_idx=None, device=device, dtype=dtype,
            )
        elif stage in HUFFMAN_CP_DIRECT_VARIANT_STAGES:
            if "concat_mean" in stage:
                fusion, sequence = "concat", "mean"
            elif "add_conv" in stage:
                fusion, sequence = "add", "conv"
            elif "concat_conv" in stage:
                fusion, sequence = "concat", "conv"
            elif "film_conv" in stage:
                fusion, sequence = "film", "conv"
            else:
                raise ValueError(f"Unsupported Huffman CP variant: {stage}")
            self.patchembed = PatchEmbedding_EventCPFusion(
                symbol_vocab_size=1025, lmax=rlc_lmax, emb_size=emb_size,
                inner_dim=numeric_feature_dim or emb_size, grid_size=(14, 14),
                drop_p=drop_p, fusion=fusion, sequence=sequence,
                padding_idx=0, device=device, dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        self.encoder = TransformerEncoder(
            depth, emb_size=emb_size, input_embed=input_embed,
            drop_p=drop_p, forward_drop_p=drop_p, device=device, dtype=dtype, **kwargs
        )
        self.classhead = ClassificationHead(emb_size, n_classes, device=device, dtype=dtype)
        self.pixel_space = pixel_space
        self.stage = stage

    def forward(self, x, cbcr=None):
        out = self.patchembed(x)
        out = self.encoder(out)
        return self.classhead(out)
