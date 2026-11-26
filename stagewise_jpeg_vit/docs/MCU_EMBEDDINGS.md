# MCU Byte Embedding Implementations

This branch studies how to encode each JPEG MCU byte payload into one ViT patch token.

## Shared Interface

All MCU embedding variants consume the same input:

```text
payload_bytes: [B, 196, Lmax] uint8   (default Lmax = 320)
byte_mask:     [B, 196, Lmax] bool
```

and output:

```text
mcu_tokens:    [B, 196, E]
```

where `196 = 14 × 14` is the image-level MCU token count, `Lmax` (default `320`) is the maximum byte payload length per MCU, and `E` is the ViT embedding size. The ViT encoder therefore always sees `196` tokens, so its attention map size remains `[B, heads, 196, 196]`.

Padding is represented with a dedicated token id:

```python
nn.Embedding(257, inner_dim, padding_idx=256)
```

`0..255` are valid JPEG byte values; `256` is reserved for padding. This avoids mixing real `0x00` bytes with padded positions.

## `embed_type=2`: `PatchEmbedding_MCU_DiscreteMean`

Location: `models/plainvit.py`

Purpose: test whether treating bytes as discrete symbols is useful even without modeling byte order.

Pipeline:

```text
[B,196,Lmax] uint8
→ replace invalid positions with PAD=256
→ Embedding(257, 64)
→ masked mean over the Lmax byte positions
→ LayerNorm
→ Linear(64, E)
→ fixed 2D sin-cos spatial PE
→ [B,196,E]
```

Interpretation:

- Bag-of-bytes baseline.
- No intra-MCU positional encoding.
- Cleanly isolates the effect of discrete lookup embedding.

## `embed_type=3`: `PatchEmbedding_MCU_DiscreteConv`

Location: `models/plainvit.py`

Purpose: test whether local order/context in the JPEG byte stream is useful.

Pipeline:

```text
[B,196,Lmax] uint8
→ replace invalid positions with PAD=256
→ Embedding(257, 32)
→ fixed 1D intra-MCU sin-cos PE
→ padding mask
→ Conv1d(32, 32, kernel=3, stride=2)
→ GELU
→ Conv1d(32, 32, kernel=3, stride=1)
→ GELU
→ downsample byte mask with max_pool1d
→ masked mean pooling
→ LayerNorm
→ Linear(32, E)
→ fixed 2D sin-cos spatial PE
→ [B,196,E]
```

Implementation note:

- The original `inner_dim=64, stride=1` version OOMed at `batch=512`.
- The current implementation uses `inner_dim=32` and first-layer `stride=2`, reducing activation memory while preserving local byte context modeling.

## `embed_type=4`: `PatchEmbedding_MCU_BitVectorConv`

Location: `models/plainvit.py`

Purpose: test whether JPEG payload bytes are better represented by their bit patterns rather than byte ids.

Pipeline:

```text
[B,196,Lmax] uint8
→ expand each byte to MSB-first 8-bit vector
→ Linear(8, 32)
→ fixed 1D intra-MCU sin-cos PE
→ padding mask
→ Conv1d(32, 32, kernel=3, stride=2)
→ GELU
→ Conv1d(32, 32, kernel=3, stride=1)
→ GELU
→ downsample byte mask with max_pool1d
→ masked mean pooling
→ LayerNorm
→ Linear(32, E)
→ fixed 2D sin-cos spatial PE
→ [B,196,E]
```

Bit expansion is MSB-first:

```python
bits = ((payload_bytes.unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1)
```

## Current Sanity Validation

Validated on a single RTX 3090 (24 GB) with ViT-Tiny:

```text
batch_size=512
num_workers=16
data: /path/to/imagenet
```

All three variants completed real-batch forward, loss, backward, and optimizer step:

| `embed_type` | embedding | result | peak memory |
|---:|---|---|---:|
| 2 | `DiscreteMean` | OK | 9845.7 MB |
| 3 | `DiscreteConv` | OK | 12057.4 MB |
| 4 | `BitVectorConv` | OK | 12498.3 MB |

Validated on an NVIDIA GH200 (120 GB) with ViT-S:

```text
gpu: NVIDIA GH200 120GB
batch_size=1024
num_workers=16
model: emb_size=384, depth=12, num_heads=6, head_size=64
data: /path/to/imagenet1k/imagenet1k
```

All three variants completed real-batch forward, loss, backward, and optimizer step:

| `embed_type` | embedding | result | step time | peak memory | patch params | total params |
|---:|---|---|---:|---:|---:|---:|
| 2 | `DiscreteMean` | OK | 1.442 s | 75175.2 MB | 41536 | 21868712 |
| 3 | `DiscreteConv` | OK | 1.767 s | 86959.6 MB | 27168 | 21854344 |
| 4 | `BitVectorConv` | OK | 1.479 s | 87841.6 MB | 19232 | 21846408 |
