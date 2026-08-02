"""Educational causal Multi-Query Attention (MQA) in plain PyTorch.

MQA keeps ``H`` independent query heads but stores one shared key head and one
shared value head.  Shapes use ``B`` (batch), ``T`` (new query tokens), ``S``
(all cached tokens), ``H`` (query heads), and ``Dh`` (head dimension).

The 2019 MQA proposal changes KV sharing, not positional encoding.  This file
uses RoPE to show the common modern decoder form; replacing RoPE with the
original Transformer's sinusoidal input encoding would not change MQA's
one-KV-head architecture.

``forward`` returns ``(output, new_cache)``.  The cache contains two
``[B, 1, S, Dh]`` tensors when ``use_cache=True``.
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


class MultiQueryAttention(torch.nn.Module):
    """Causal self-attention with many Q heads and exactly one shared KV head."""

    def __init__(self, d_model: int, num_query_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or num_query_heads <= 0:
            raise ValueError("d_model and num_query_heads must be positive")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.head_dim = d_model // num_query_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        # [Block 02] Many-query and single-KV projections
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(d_model, self.head_dim, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        # [/Block 02]

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Apply causal MQA to ``x[B, T, d_model]``.

        Cached decoding uses contiguous positions.  PyTorch broadcasting lets
        ``[B, H, T, Dh]`` queries multiply ``[B, 1, S, Dh]`` keys directly, so
        this implementation never physically repeats the shared KV tensors.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            expected_prefix = (batch, 1)
            if (
                past_k.ndim != 4
                or past_v.shape != past_k.shape
                or past_k.shape[:2] != expected_prefix
                or past_k.shape[-1] != self.head_dim
            ):
                raise ValueError("invalid MQA cache shape")
            past_length = past_k.shape[2]

        # [Block 03] Project independent Q and shared K V
        q = self.q_proj(x).view(
            batch, query_length, self.num_query_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        # q: [B, H, T, Dh]; k, v: [B, 1, T, Dh]
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

        # [Block 05] Append the single-head KV cache
        if kv_cache is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v) if use_cache else None
        # k, v: [B, 1, S, Dh]
        # [/Block 05]

        # [Block 06] Broadcast shared KV and apply causal attention
        # matmul broadcasts the size-1 KV-head axis; no repeat/expand is used.
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        # scores: [B, H, T, S]
        key_positions = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        ).unsqueeze(1)
        future_tokens = key_positions > query_positions
        scores = scores.masked_fill(
            future_tokens.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attention, v)  # [B, H, T, Dh]
        # [/Block 06]

        # [Block 07] Concatenate query heads and project output
        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 07]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = MultiQueryAttention(d_model=32, num_query_heads=4).eval()
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
    assert full_cache is not None and full_cache[0].shape == (2, 1, 6, 8)
    assert torch.allclose(full_output, decoded_output, atol=1e-5, rtol=1e-5)
    print("MQA smoke test passed:", tuple(full_output.shape))


if __name__ == "__main__":
    _smoke_test()
