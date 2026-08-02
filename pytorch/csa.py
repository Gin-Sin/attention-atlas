"""Educational, CPU-friendly Compressed Sparse Attention (CSA).

The reference first creates one summary per ``compression_block`` tokens with
two overlapping, channel-wise weighted streams.  A low-dimensional indexer
selects causal summary top-k entries, while a raw local window preserves recent
detail.  Selected summaries and local tokens share one full-dimensional
attention softmax.

This is a teaching implementation, not a production/paper-exact kernel.  It
uses Python loops for compression, eager gathers, standard MHA instead of MLA,
and no fused FP4 indexer or cache-aware decoding path.  B=batch, L=raw sequence,
S=number of closed summaries, H=heads, and D=head dimension.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn


# [Block 01] Partial RoPE and safe attention utilities
def apply_partial_rope(
    x: Tensor,
    positions: Tensor,
    rotary_dim: int,
    *,
    inverse: bool = False,
    base: float = 10_000.0,
) -> Tensor:
    """Apply RoPE to only the first ``rotary_dim`` of ``x [B,H,L,D]``."""
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
    """Last-axis softmax with exact zeros for masked or empty candidate rows."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    finite_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(finite_logits, dim=-1) * mask.to(logits.dtype)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(logits.dtype).tiny
    )


def gather_per_query(sequence: Tensor, indices: Tensor) -> Tensor:
    """Gather ``sequence [B,H,S,D]`` with ``indices [B,L,K]``."""
    batch, heads, _, width = sequence.shape
    queries, choices = indices.shape[1:]
    expanded = sequence.unsqueeze(2).expand(-1, -1, queries, -1, -1)
    gather_index = indices[:, None, :, :, None].expand(
        batch, heads, queries, choices, width
    )
    return torch.gather(expanded, 3, gather_index)
# [/Block 01]


# [Block 02] Overlapping channel-wise sequence compression
class OverlappingChannelCompressor(nn.Module):
    """Compress each closed block using current-A and previous-B streams.

    Summary ``i`` covers the current block through stream A and, when present,
    the previous block through stream B.  Softmax runs over those positions
    independently for every output channel.  Adjacent summaries overlap, but
    their stride remains one block, so compression is approximately L/m.
    """

    def __init__(self, d_model: int, block_size: int) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.d_model = d_model
        self.block_size = block_size
        self.content_a = nn.Linear(d_model, d_model, bias=False)
        self.content_b = nn.Linear(d_model, d_model, bias=False)
        self.gate_a = nn.Linear(d_model, d_model, bias=True)
        self.gate_b = nn.Linear(d_model, d_model, bias=True)
        self.relative_bias = nn.Parameter(torch.zeros(2 * block_size, d_model))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return summaries ``[B,S,d_model]`` and block-end positions ``[S]``."""
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,L,d_model]")
        batch, length, width = x.shape
        summary_count = length // self.block_size
        if summary_count == 0:
            return x.new_empty(batch, 0, width), torch.empty(
                0, dtype=torch.long, device=x.device
            )

        summaries = []
        for block_id in range(summary_count):
            start = block_id * self.block_size
            stop = start + self.block_size
            current = x[:, start:stop]
            current_values = self.content_a(current)
            current_gates = self.gate_a(current)

            if block_id == 0:
                values = current_values
                gates = current_gates + self.relative_bias[self.block_size :]
            else:
                previous = x[:, start - self.block_size : start]
                values = torch.cat((self.content_b(previous), current_values), dim=1)
                gates = torch.cat((self.gate_b(previous), current_gates), dim=1)
                gates = gates + self.relative_bias

            channel_weights = torch.softmax(gates, dim=1)
            summaries.append((channel_weights * values).sum(dim=1))

        summary_tensor = torch.stack(summaries, dim=1)
        end_positions = (
            torch.arange(1, summary_count + 1, device=x.device) * self.block_size - 1
        )
        return summary_tensor, end_positions
# [/Block 02]


# [Block 03] CSA module configuration and projections
class CompressedSparseAttention(nn.Module):
    """Causal CSA with overlapping summaries, indexer top-k, and local tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        compression_block: int = 4,
        top_k: int = 8,
        local_window: int = 16,
        index_dim: int = 16,
        num_index_heads: int = 4,
        rotary_dim: int = 0,
        detach_indexer_input: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if min(top_k, local_window, index_dim, num_index_heads) <= 0:
            raise ValueError("top_k, window, index dimensions, and heads must be > 0")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.top_k = top_k
        self.local_window = local_window
        self.index_dim = index_dim
        self.num_index_heads = num_index_heads
        self.rotary_dim = rotary_dim
        self.detach_indexer_input = detach_indexer_input
        if rotary_dim < 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")

        self.compressor = OverlappingChannelCompressor(
            d_model, compression_block
        )
        self.index_queries = nn.Linear(
            d_model, num_index_heads * index_dim, bias=False
        )
        self.index_key = nn.Linear(d_model, index_dim, bias=False)
        self.index_weights = nn.Linear(d_model, num_index_heads, bias=True)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, projected: Tensor) -> Tensor:
        batch, length, _ = projected.shape
        return projected.view(batch, length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
    # [/Block 03]

    # [Block 04] Compressed indexer scoring and summary top-k
    def _compressed_topk(
        self, x: Tensor, summaries: Tensor, summary_positions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Return logits ``[B,L,S]``, indices ``[B,L,K]``, and validity."""
        batch, length, _ = x.shape
        summary_count = summaries.size(1)
        if summary_count == 0:
            logits = x.new_empty(batch, length, 0)
            empty_indices = torch.empty(
                batch, length, 0, dtype=torch.long, device=x.device
            )
            return logits, empty_indices, empty_indices.to(torch.bool)

        query_source = x.detach() if self.detach_indexer_input else x
        summary_source = (
            summaries.detach() if self.detach_indexer_input else summaries
        )
        queries = self.index_queries(query_source).view(
            batch, length, self.num_index_heads, self.index_dim
        )
        queries = queries.transpose(1, 2)
        keys = self.index_key(summary_source)
        weights = torch.sigmoid(self.index_weights(query_source)).transpose(1, 2)

        per_head = torch.einsum("bhld,bsd->bhls", queries, keys)
        per_head = torch.relu(per_head * (self.index_dim ** -0.5))
        logits = (per_head * weights[..., None]).sum(dim=1)
        query_positions = torch.arange(length, device=x.device)
        causal = summary_positions[None, None, :] <= query_positions[None, :, None]
        logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)

        choices = min(self.top_k, summary_count)
        _, indices = torch.topk(logits, k=choices, dim=-1)
        valid = torch.gather(causal.expand(batch, -1, -1), 2, indices)
        return logits, torch.where(valid, indices, torch.zeros_like(indices)), valid
    # [/Block 04]

    # [Block 05] Summary routing and raw query/key/value lanes
    def forward(
        self, x: Tensor, *, return_aux: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Map ``x [B,L,d_model]`` to a causal output of the same shape."""
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,L,d_model]")
        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence length must be positive")

        summaries, summary_positions = self.compressor(x)
        index_logits, selected, selected_valid = self._compressed_topk(
            x, summaries, summary_positions
        )

        raw_positions = torch.arange(length, device=x.device)
        queries = apply_partial_rope(
            self._heads(self.query(x)), raw_positions, self.rotary_dim
        )
        raw_keys = apply_partial_rope(
            self._heads(self.key(x)), raw_positions, self.rotary_dim
        )
        raw_values = self._heads(self.value(x))
        # [/Block 05]

        # [Block 06] Gather selected compressed keys and values
        if summaries.size(1):
            compressed_keys = apply_partial_rope(
                self._heads(self.key(summaries)),
                summary_positions,
                self.rotary_dim,
            )
            compressed_values = self._heads(self.value(summaries))
            chosen_keys = gather_per_query(compressed_keys, selected)
            chosen_values = gather_per_query(compressed_values, selected)
        else:
            chosen_keys = raw_keys.new_empty(
                batch, self.num_heads, length, 0, self.head_dim
            )
            chosen_values = raw_values.new_empty(
                batch, self.num_heads, length, 0, self.head_dim
            )
        # [/Block 06]

        # [Block 07] Recent raw sliding-window candidates
        offsets = torch.arange(self.local_window, device=x.device)
        local_indices = raw_positions[:, None] - offsets[None, :]
        local_valid = local_indices >= 0
        local_indices = local_indices.clamp_min(0)
        batched_local = local_indices[None, :, :].expand(batch, -1, -1)
        local_keys = gather_per_query(raw_keys, batched_local)
        local_values = gather_per_query(raw_values, batched_local)
        # [/Block 07]

        # [Block 08] One shared softmax over summaries and local tokens
        candidate_keys = torch.cat((chosen_keys, local_keys), dim=3)
        candidate_values = torch.cat((chosen_values, local_values), dim=3)
        candidate_valid = torch.cat(
            (
                selected_valid,
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
            "summaries": summaries,
            "summary_positions": summary_positions,
            "index_logits": index_logits,
            "selected_summary_indices": selected,
            "selected_summary_valid": selected_valid,
            "local_indices": local_indices,
            "attention_probs": probabilities,
        }
# [/Block 09]


# [Block 10] Deterministic CPU smoke test
def _smoke_test() -> None:
    torch.manual_seed(11)
    model = CompressedSparseAttention(
        d_model=32,
        num_heads=4,
        compression_block=3,
        top_k=2,
        local_window=4,
        index_dim=6,
        num_index_heads=3,
        rotary_dim=4,
    )
    model.eval()
    x = torch.randn(2, 11, 32)
    output, aux = model(x, return_aux=True)

    assert output.shape == x.shape
    assert aux["summaries"].shape == (2, 3, 32)
    assert torch.isfinite(output).all()
    if aux["selected_summary_valid"].any():
        selected_ends = aux["summary_positions"][
            aux["selected_summary_indices"]
        ]
        query_positions = torch.arange(x.size(1))[None, :, None]
        assert torch.all(
            selected_ends[aux["selected_summary_valid"]]
            <= query_positions.expand_as(selected_ends)[
                aux["selected_summary_valid"]
            ]
        )

    changed = x.clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 5
    changed_output = model(changed)
    torch.testing.assert_close(output[:, :8], changed_output[:, :8])
    print(
        "CSA smoke test passed:",
        tuple(output.shape),
        f"summaries={aux['summaries'].size(1)}",
    )


if __name__ == "__main__":
    _smoke_test()
# [/Block 10]
