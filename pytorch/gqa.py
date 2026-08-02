"""Educational causal Grouped-Query Attention (GQA) in plain PyTorch.

GQA partitions ``Hq`` query heads into ``Hkv`` groups.  Each group of
``R = Hq / Hkv`` queries shares one key head and one value head.  ``Hkv=1`` is
MQA, while ``Hkv=Hq`` is MHA.

The GQA architecture is independent of a particular position scheme.  This
modern decoder-style example uses RoPE.  Shapes use ``B`` (batch), ``T`` (new
query tokens), ``S`` (all cached tokens), ``Hq`` (query heads), ``Hkv`` (KV
heads), ``R`` (queries per KV head), and ``Dh`` (head dimension).

``forward`` returns ``(output, new_cache)``.  The cache contains two
``[B, Hkv, S, Dh]`` tensors when ``use_cache=True``.
"""

from __future__ import annotations

import torch


KVCache = tuple[torch.Tensor, torch.Tensor]


# [Block 01] Rotary position encoding
def apply_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to ``tensor[B, heads, T, Dh]`` at ``positions[T]``."""
    head_dim = tensor.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if positions.ndim != 1 or positions.numel() != tensor.shape[-2]:
        raise ValueError("positions must contain one index per token")

    frequency_indices = torch.arange(
        0, head_dim, 2, device=tensor.device, dtype=torch.float32
    )
    inverse_frequencies = torch.pow(base, -frequency_indices / head_dim)
    angles = positions.to(device=tensor.device, dtype=torch.float32).unsqueeze(
        1
    ) * inverse_frequencies.unsqueeze(0)
    cosine = angles.cos().to(tensor.dtype).unsqueeze(0).unsqueeze(0)
    sine = angles.sin().to(tensor.dtype).unsqueeze(0).unsqueeze(0)

    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)
# [/Block 01]


class GroupedQueryAttention(torch.nn.Module):
    """Causal self-attention with one shared KV head per query-head group."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if min(d_model, num_query_heads, num_kv_heads) <= 0:
            raise ValueError("dimensions and head counts must be positive")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # [Block 02] Query and grouped-KV projections
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            d_model, num_kv_heads * self.head_dim, bias=False
        )
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        # [/Block 02]

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Apply causal GQA to ``x[B, T, d_model]``.

        Queries are reshaped to ``[B, Hkv, R, T, Dh]`` so grouped einsums can
        address shared K/V directly.  No physical KV repetition or expansion
        is performed.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            expected_prefix = (batch, self.num_kv_heads)
            if (
                past_k.ndim != 4
                or past_v.shape != past_k.shape
                or past_k.shape[:2] != expected_prefix
                or past_k.shape[-1] != self.head_dim
            ):
                raise ValueError("invalid GQA cache shape")
            past_length = past_k.shape[2]

        # [Block 03] Project Q heads and grouped K V heads
        q = self.q_proj(x).view(
            batch, query_length, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        # q: [B, Hq, T, Dh]; k, v: [B, Hkv, T, Dh]
        # [/Block 03]

        # [Block 04] Rotate queries and keys at absolute positions
        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        )
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)
        # [/Block 04]

        # [Block 05] Append grouped KV cache
        if kv_cache is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v) if use_cache else None
        # k, v: [B, Hkv, S, Dh]
        # [/Block 05]

        # [Block 06] Map query groups to shared KV without repetition
        grouped_q = q.reshape(
            batch,
            self.num_kv_heads,
            self.queries_per_kv,
            query_length,
            self.head_dim,
        )
        # grouped_q: [B, Hkv, R, T, Dh]
        scores = torch.einsum("bgrtd,bgsd->bgrts", grouped_q, k)
        scores = scores * (self.head_dim**-0.5)
        # scores: [B, Hkv, R, T, S]
        # [/Block 06]

        # [Block 07] Apply causal mask and read grouped values
        key_positions = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        ).unsqueeze(1)
        future_tokens = key_positions > query_positions
        scores = scores.masked_fill(
            future_tokens.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=-1)
        head_outputs = torch.einsum("bgrts,bgsd->bgrtd", attention, v)
        # head_outputs: [B, Hkv, R, T, Dh]
        # [/Block 07]

        # [Block 08] Concatenate query heads and project output
        merged = head_outputs.permute(0, 3, 1, 2, 4).contiguous().view(
            batch, query_length, self.d_model
        )
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 08]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = GroupedQueryAttention(
        d_model=32,
        num_query_heads=8,
        num_kv_heads=2,
    ).eval()
    x = torch.randn(2, 6, 32)

    full_output, full_cache = model(x, use_cache=True)
    cache = None
    decoded_pieces = []
    for token in range(x.shape[1]):
        piece, cache = model(
            x[:, token : token + 1],
            kv_cache=cache,
            use_cache=True,
        )
        decoded_pieces.append(piece)
    decoded_output = torch.cat(decoded_pieces, dim=1)

    assert full_output.shape == (2, 6, 32)
    assert full_cache is not None and full_cache[0].shape == (2, 2, 6, 4)
    assert torch.allclose(full_output, decoded_output, atol=1e-5, rtol=1e-5)
    print("GQA smoke test passed:", tuple(full_output.shape))


if __name__ == "__main__":
    _smoke_test()
