"""Educational causal Multi-Head Attention (MHA) in plain PyTorch.

Shapes use ``B`` (batch), ``T`` (new query tokens), ``S`` (all cached tokens),
``H`` (heads), and ``Dh`` (head dimension).  The 2017 Transformer added
sinusoidal absolute positions before its Q/K/V projections; this reference
follows that original convention.  Many modern decoder-only MHA layers use
RoPE instead, but RoPE is not part of the definition of MHA itself.

``forward`` returns ``(output, new_cache)``.  The cache is a ``(key, value)``
tuple with two ``[B, H, S, Dh]`` tensors when ``use_cache=True``.
"""

from __future__ import annotations

import torch


KVCache = tuple[torch.Tensor, torch.Tensor]


# [Block 01] Original sinusoidal position encoding
def sinusoidal_position_encoding(
    length: int,
    d_model: int,
    *,
    offset: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return absolute sinusoidal positions with shape ``[length, d_model]``."""
    if length < 0 or offset < 0:
        raise ValueError("length and offset must be non-negative")

    positions = torch.arange(
        offset, offset + length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    even_dimensions = torch.arange(
        0, d_model, 2, device=device, dtype=torch.float32
    )
    angular_rates = torch.pow(10000.0, -even_dimensions / d_model)
    angles = positions * angular_rates.unsqueeze(0)

    encoding = torch.zeros(length, d_model, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)
# [/Block 01]


class MultiHeadAttention(torch.nn.Module):
    """Causal self-attention with an independent K/V pair for every Q head."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # [Block 02] Independent per-head Q K V projections
        self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=False)
        # [/Block 02]

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Apply causal MHA to ``x`` of shape ``[B, T, d_model]``.

        Cached decoding assumes contiguous positions beginning at zero.  If a
        cache contains ``P`` tokens, the new tokens receive positions
        ``P, ..., P + T - 1``.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            expected_prefix = (batch, self.num_heads)
            if (
                past_k.ndim != 4
                or past_v.shape != past_k.shape
                or past_k.shape[:2] != expected_prefix
                or past_k.shape[-1] != self.head_dim
            ):
                raise ValueError("invalid MHA cache shape")
            past_length = past_k.shape[2]

        # [Block 03] Add absolute positions to token states
        positions = sinusoidal_position_encoding(
            query_length,
            self.d_model,
            offset=past_length,
            device=x.device,
            dtype=x.dtype,
        )
        positioned_x = x + positions.unsqueeze(0)
        # [/Block 03]

        # [Block 04] Project and split into attention heads
        # q, k, v: [B, H, T, Dh]
        q = self._split_heads(self.q_proj(positioned_x))
        k = self._split_heads(self.k_proj(positioned_x))
        v = self._split_heads(self.v_proj(positioned_x))
        # [/Block 04]

        # [Block 05] Append full per-head KV cache
        if kv_cache is not None:
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        new_cache = (k, v) if use_cache else None
        # k, v: [B, H, S, Dh], where S = past_length + T
        # [/Block 05]

        # [Block 06] Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        # scores: [B, H, T, S]
        key_positions = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        ).unsqueeze(1)
        future_tokens = key_positions > query_positions  # [T, S]
        scores = scores.masked_fill(
            future_tokens.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attention, v)  # [B, H, T, Dh]
        # [/Block 06]

        # [Block 07] Concatenate heads and project output
        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch, query_length, self.d_model
        )
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 07]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = MultiHeadAttention(d_model=32, num_heads=4).eval()
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
    assert full_cache is not None and full_cache[0].shape == (2, 4, 6, 8)
    assert torch.allclose(full_output, decoded_output, atol=1e-5, rtol=1e-5)
    print("MHA smoke test passed:", tuple(full_output.shape))


if __name__ == "__main__":
    _smoke_test()
