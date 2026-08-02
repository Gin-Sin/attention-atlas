"""Educational, CPU-friendly Heavily Compressed Attention (HCA).

HCA forms one channel-wise summary for every large, non-overlapping block.  It
then attends densely to every causally closed summary (there is no indexer and
no top-k) alongside a recent raw-token window.  Partial RoPE is explicit, and
raw keys are inverse-rotated into a canonical content frame before compression,
then summaries are rotated once at their block-end positions.

This is not a production/paper-exact kernel: compression uses an eager Python
loop, dense candidate tensors are materialized, standard MHA replaces MLA, and
there is no fused cache/decode implementation.  B=batch, L=raw length,
S=closed summaries, H=heads, and D=head dimension.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn


# [Block 01] Partial and inverse RoPE utilities
def apply_partial_rope(
    x: Tensor,
    positions: Tensor,
    rotary_dim: int,
    *,
    inverse: bool = False,
    base: float = 10_000.0,
) -> Tensor:
    """Apply R(position), or its inverse, to leading channels of [B,H,L,D]."""
    if x.ndim != 4 or positions.ndim != 1 or x.size(2) != positions.numel():
        raise ValueError("expected x [B,H,L,D] and positions [L]")
    if rotary_dim == 0:
        return x
    if rotary_dim < 0 or rotary_dim > x.size(-1) or rotary_dim % 2:
        raise ValueError("rotary_dim must be even and in [0, head_dim]")

    frequency_ids = torch.arange(
        0, rotary_dim, 2, device=x.device, dtype=torch.float32
    )
    inverse_frequencies = base ** (-frequency_ids / rotary_dim)
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None]
    angles = angles * inverse_frequencies[None, :]
    if inverse:
        angles = -angles
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]

    pairs = x[..., :rotary_dim].reshape(*x.shape[:-1], rotary_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
    return torch.cat((rotated.flatten(-2), x[..., rotary_dim:]), dim=-1)


def masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Last-axis softmax with zeros at invalid candidate positions."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    finite_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(finite_logits, dim=-1) * mask.to(logits.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).tiny
    )


def gather_per_query(sequence: Tensor, indices: Tensor) -> Tensor:
    """Gather ``sequence [B,H,L,D]`` using raw indices ``[B,L,W]``."""
    batch, heads, _, width = sequence.shape
    queries, choices = indices.shape[1:]
    expanded = sequence.unsqueeze(2).expand(-1, -1, queries, -1, -1)
    gather_index = indices[:, None, :, :, None].expand(
        batch, heads, queries, choices, width
    )
    return torch.gather(expanded, 3, gather_index)
# [/Block 01]


# [Block 02] Non-overlapping heavy compression
class NonOverlappingChannelCompressor(nn.Module):
    """Pool arbitrary value channels with learned per-channel token weights."""

    def __init__(
        self, routing_dim: int, value_dim: int, block_size: int = 128
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.routing_dim = routing_dim
        self.value_dim = value_dim
        self.block_size = block_size
        self.gate = nn.Linear(routing_dim, value_dim, bias=True)
        self.relative_bias = nn.Parameter(torch.zeros(block_size, value_dim))

    def forward(self, routing_x: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """Compress ``values [B,L,C]`` and return ``[B,S,C]`` plus ends [S]."""
        if (
            routing_x.ndim != 3
            or values.ndim != 3
            or routing_x.shape[:2] != values.shape[:2]
            or routing_x.size(-1) != self.routing_dim
            or values.size(-1) != self.value_dim
        ):
            raise ValueError("routing_x [B,L,routing_dim], values [B,L,value_dim]")

        batch, length, _ = routing_x.shape
        summary_count = length // self.block_size
        if summary_count == 0:
            return values.new_empty(batch, 0, self.value_dim), torch.empty(
                0, dtype=torch.long, device=values.device
            )

        summaries = []
        for block_id in range(summary_count):
            start = block_id * self.block_size
            stop = start + self.block_size
            block_values = values[:, start:stop]
            gate_logits = self.gate(routing_x[:, start:stop])
            gate_logits = gate_logits + self.relative_bias
            channel_weights = torch.softmax(gate_logits, dim=1)
            summaries.append((channel_weights * block_values).sum(dim=1))

        end_positions = (
            torch.arange(1, summary_count + 1, device=values.device)
            * self.block_size
            - 1
        )
        return torch.stack(summaries, dim=1), end_positions
# [/Block 02]


# [Block 03] HCA module configuration and projections
class HeavilyCompressedAttention(nn.Module):
    """Causal HCA over all heavy summaries and a recent uncompressed window."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        compression_block: int = 128,
        local_window: int = 128,
        rotary_dim: int = 0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if local_window <= 0:
            raise ValueError("local_window must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.rotary_dim = rotary_dim
        if rotary_dim < 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.compressor = NonOverlappingChannelCompressor(
            routing_dim=d_model,
            value_dim=2 * d_model,
            block_size=compression_block,
        )
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
    # [/Block 03]

    # [Block 04] Raw query/key/value lanes with partial RoPE
    def forward(
        self, x: Tensor, *, return_aux: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Map ``x [B,L,d_model]`` to a causal output of the same shape."""
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,L,d_model]")
        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence length must be positive")

        raw_positions = torch.arange(length, device=x.device)
        queries = apply_partial_rope(
            self._heads(self.query(x)), raw_positions, self.rotary_dim
        )
        canonical_raw_keys = self._heads(self.key(x))
        positioned_raw_keys = apply_partial_rope(
            canonical_raw_keys, raw_positions, self.rotary_dim
        )
        raw_values = self._heads(self.value(x))
        # [/Block 04]

        # [Block 05] Inverse-RoPE canonicalization and heavy compression
        # This explicit round trip teaches the position-safe compression rule.
        # A production kernel would keep/fuse the canonical content channels.
        keys_for_compression = apply_partial_rope(
            positioned_raw_keys,
            raw_positions,
            self.rotary_dim,
            inverse=True,
        )
        flat_keys = keys_for_compression.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )
        flat_values = raw_values.transpose(1, 2).contiguous().view(
            batch, length, self.d_model
        )
        compression_values = torch.cat((flat_keys, flat_values), dim=-1)
        compressed, summary_positions = self.compressor(x, compression_values)
        compressed_keys_flat, compressed_values_flat = compressed.split(
            self.d_model, dim=-1
        )
        # [/Block 05]

        # [Block 06] Completed summary cache and causal publish gate
        summary_count = compressed.size(1)
        if summary_count:
            summary_keys = apply_partial_rope(
                self._heads(compressed_keys_flat),
                summary_positions,
                self.rotary_dim,
            )
            summary_values = self._heads(compressed_values_flat)
            dense_summary_keys = summary_keys.unsqueeze(2).expand(
                -1, -1, length, -1, -1
            )
            dense_summary_values = summary_values.unsqueeze(2).expand(
                -1, -1, length, -1, -1
            )
            summary_valid = (
                summary_positions[None, None, :]
                <= raw_positions[None, :, None]
            )
            summary_valid = summary_valid.expand(batch, -1, -1)
        else:
            dense_summary_keys = positioned_raw_keys.new_empty(
                batch, self.num_heads, length, 0, self.head_dim
            )
            dense_summary_values = raw_values.new_empty(
                batch, self.num_heads, length, 0, self.head_dim
            )
            summary_valid = torch.empty(
                batch, length, 0, dtype=torch.bool, device=x.device
            )
        # [/Block 06]

        # [Block 07] Recent raw sliding-window candidates
        offsets = torch.arange(self.local_window, device=x.device)
        local_indices = raw_positions[:, None] - offsets[None, :]
        local_valid = local_indices >= 0
        local_indices = local_indices.clamp_min(0)
        batched_local = local_indices[None, :, :].expand(batch, -1, -1)
        local_keys = gather_per_query(positioned_raw_keys, batched_local)
        local_values = gather_per_query(raw_values, batched_local)
        # [/Block 07]

        # [Block 08] Dense softmax over all summaries and local tokens
        candidate_keys = torch.cat((dense_summary_keys, local_keys), dim=3)
        candidate_values = torch.cat((dense_summary_values, local_values), dim=3)
        candidate_valid = torch.cat(
            (
                summary_valid,
                local_valid[None, :, :].expand(batch, -1, -1),
            ),
            dim=-1,
        )
        attention_logits = torch.einsum(
            "bhld,bhlkd->bhlk", queries, candidate_keys
        )
        attention_logits = attention_logits * (self.head_dim ** -0.5)
        probabilities = masked_softmax(
            attention_logits, candidate_valid[:, None, :, :]
        )
        context = torch.einsum(
            "bhlk,bhlkd->bhld", probabilities, candidate_values
        )
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        # [/Block 08]

        # [Block 09] Output projection and auxiliary results
        result = self.output(context)

        if not return_aux:
            return result
        return result, {
            "compressed_kv": compressed,
            "summary_positions": summary_positions,
            "summary_valid": summary_valid,
            "local_indices": local_indices,
            "attention_probs": probabilities,
        }
# [/Block 09]


# [Block 10] Deterministic CPU smoke test
def _smoke_test() -> None:
    torch.manual_seed(19)
    model = HeavilyCompressedAttention(
        d_model=32,
        num_heads=4,
        compression_block=4,
        local_window=5,
        rotary_dim=4,
    )
    model.eval()
    x = torch.randn(2, 13, 32)
    output, aux = model(x, return_aux=True)

    assert output.shape == x.shape
    assert aux["compressed_kv"].shape == (2, 3, 64)
    assert torch.isfinite(output).all()

    sample = torch.randn(1, 2, 6, 8)
    positions = torch.arange(6)
    rotated = apply_partial_rope(sample, positions, rotary_dim=4)
    restored = apply_partial_rope(rotated, positions, rotary_dim=4, inverse=True)
    torch.testing.assert_close(sample, restored)

    changed = x.clone()
    changed[:, 9:] = torch.randn_like(changed[:, 9:]) * 5
    changed_output = model(changed)
    torch.testing.assert_close(output[:, :9], changed_output[:, :9])
    print(
        "HCA smoke test passed:",
        tuple(output.shape),
        f"summaries={aux['compressed_kv'].size(1)}",
    )


if __name__ == "__main__":
    _smoke_test()
# [/Block 10]
