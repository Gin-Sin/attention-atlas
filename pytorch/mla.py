"""Educational causal Multi-Head Latent Attention (MLA) in plain PyTorch.

This module demonstrates the defining DeepSeek-V2 MLA ideas:

* low-rank query compression ``x -> c_q -> per-head q``;
* joint low-rank KV compression ``x -> c_kv`` followed by per-head K/V
  up-projections;
* decoupled RoPE, with a shared positional key kept outside ``c_kv``; and
* a latent cache containing ``c_kv`` plus that small rotated positional key.

Shapes use ``B`` (batch), ``T`` (new tokens), ``S`` (all cached tokens), ``H``
(heads), ``Dc`` (KV latent rank), ``Dq`` (query latent rank), ``Dn`` (non-RoPE
content dimension), ``Dr`` (RoPE dimension), and ``Dv`` (value dimension).

For clarity, this dense CPU reference omits tensor-parallel layout, quantized
caches, fused kernels, residual/RMSNorm placement around the whole attention
layer, and dropout.  It exactly absorbs the content-key up-projection into the
query for scoring.  It reconstructs values from the cached latent before the
weighted sum; optimized inference kernels can also absorb/rearrange the value
up-projection with the output projection.  These simplifications change
execution strategy, not the demonstrated latent cache or attention equation.

``forward`` returns ``(output, new_cache)``.  The cache is
``(latent[B, S, Dc], rope_key[B, 1, S, Dr])`` when ``use_cache=True``.
"""

from __future__ import annotations

import torch


MLACache = tuple[torch.Tensor, torch.Tensor]


# [Block 01] Normalize compressed latents
class RMSNorm(torch.nn.Module):
    """Minimal RMSNorm used between each down- and up-projection."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance.to(x.dtype) + self.eps)
        return normalized * self.weight
# [/Block 01]


# [Block 02] Rotary position encoding
def apply_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to ``tensor[B, heads, T, Dr]`` at ``positions[T]``."""
    rope_dim = tensor.shape[-1]
    if rope_dim % 2 != 0:
        raise ValueError("RoPE requires an even dimension")
    if positions.ndim != 1 or positions.numel() != tensor.shape[-2]:
        raise ValueError("positions must contain one index per token")

    frequency_indices = torch.arange(
        0, rope_dim, 2, device=tensor.device, dtype=torch.float32
    )
    inverse_frequencies = torch.pow(base, -frequency_indices / rope_dim)
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
# [/Block 02]


class MultiHeadLatentAttention(torch.nn.Module):
    """Causal MLA with query/KV LoRA paths and a decoupled-RoPE latent cache."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_content_dim: int,
        qk_rope_dim: int,
        value_dim: int,
    ) -> None:
        super().__init__()
        positive = (
            d_model,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_content_dim,
            qk_rope_dim,
            value_dim,
        )
        if min(positive) <= 0:
            raise ValueError("all dimensions and head counts must be positive")
        if q_lora_rank >= d_model or kv_lora_rank >= d_model:
            raise ValueError("latent ranks must be smaller than d_model")
        if qk_rope_dim % 2 != 0:
            raise ValueError("qk_rope_dim must be even")

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_content_dim = qk_content_dim
        self.qk_rope_dim = qk_rope_dim
        self.value_dim = value_dim
        self.qk_head_dim = qk_content_dim + qk_rope_dim

        # [Block 03] Low-rank query and KV projection paths
        self.q_down_proj = torch.nn.Linear(
            d_model, q_lora_rank, bias=False
        )
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up_proj = torch.nn.Linear(
            q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
        )

        # One down-projection emits the cacheable KV latent and shared RoPE key.
        self.kv_down_and_rope_proj = torch.nn.Linear(
            d_model,
            kv_lora_rank + qk_rope_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.key_up_proj = torch.nn.Linear(
            kv_lora_rank,
            num_heads * qk_content_dim,
            bias=False,
        )
        self.value_up_proj = torch.nn.Linear(
            kv_lora_rank,
            num_heads * value_dim,
            bias=False,
        )
        self.out_proj = torch.nn.Linear(
            num_heads * value_dim, d_model, bias=False
        )
        # [/Block 03]

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: MLACache | None = None,
        use_cache: bool = False,
        implementation: str = "absorbed",
    ) -> tuple[torch.Tensor, MLACache | None]:
        """Apply causal MLA to ``x[B, T, d_model]``.

        ``implementation="absorbed"`` evaluates content scores directly
        against cached ``c_kv`` after folding the key up-projection into the
        query.  ``"reconstruct"`` explicitly rebuilds content keys and is
        included to verify that the algebra is exact.  Both paths cache only
        the normalized KV latent and the shared, already-rotated RoPE key.
        Cached positions are assumed contiguous and zero-based.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        if implementation not in {"absorbed", "reconstruct"}:
            raise ValueError("implementation must be 'absorbed' or 'reconstruct'")

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_latent, past_rope_key = kv_cache
            if (
                past_latent.ndim != 3
                or past_latent.shape[0] != batch
                or past_latent.shape[-1] != self.kv_lora_rank
                or past_rope_key.ndim != 4
                or past_rope_key.shape[:2] != (batch, 1)
                or past_rope_key.shape[2] != past_latent.shape[1]
                or past_rope_key.shape[-1] != self.qk_rope_dim
            ):
                raise ValueError("invalid MLA cache shape")
            past_length = past_latent.shape[1]

        # [Block 04] Compress and restore per-head queries
        query_latent = self.q_norm(self.q_down_proj(x))  # [B, T, Dq]
        query_heads = self.q_up_proj(query_latent).view(
            batch,
            query_length,
            self.num_heads,
            self.qk_head_dim,
        ).transpose(1, 2)
        q_content, q_rope = torch.split(
            query_heads,
            (self.qk_content_dim, self.qk_rope_dim),
            dim=-1,
        )
        # q_content: [B, H, T, Dn]; q_rope: [B, H, T, Dr]
        # [/Block 04]

        # [Block 05] Jointly compress keys and values into one latent
        compressed = self.kv_down_and_rope_proj(x)
        new_latent, new_rope_key = torch.split(
            compressed,
            (self.kv_lora_rank, self.qk_rope_dim),
            dim=-1,
        )
        new_latent = self.kv_norm(new_latent)  # [B, T, Dc]
        # The same new_latent later generates every head's content K and V.
        # [/Block 05]

        # [Block 06] Decouple and rotate positional query and key channels
        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        )
        q_rope = apply_rope(q_rope, positions)
        new_rope_key = apply_rope(new_rope_key.unsqueeze(1), positions)
        # new_rope_key: [B, 1, T, Dr], shared by all query heads
        # [/Block 06]

        # [Block 07] Append latent and positional-key cache
        if kv_cache is not None:
            latent = torch.cat((past_latent, new_latent), dim=1)
            rope_key = torch.cat((past_rope_key, new_rope_key), dim=2)
        else:
            latent = new_latent
            rope_key = new_rope_key
        new_cache = (latent, rope_key) if use_cache else None
        # Cached width per token is Dc + Dr, not 2 * H * head_dim.
        # [/Block 07]

        # [Block 08] Score content through the latent or rebuilt keys
        if implementation == "absorbed":
            key_up_weight = self.key_up_proj.weight.view(
                self.num_heads,
                self.qk_content_dim,
                self.kv_lora_rank,
            )
            latent_queries = torch.einsum(
                "bhtd,hdc->bhtc", q_content, key_up_weight
            )
            content_scores = torch.einsum(
                "bhtc,bsc->bhts", latent_queries, latent
            )
        else:
            rebuilt_keys = self.key_up_proj(latent).view(
                batch,
                latent.shape[1],
                self.num_heads,
                self.qk_content_dim,
            ).transpose(1, 2)
            content_scores = torch.matmul(
                q_content, rebuilt_keys.transpose(-2, -1)
            )
        # content_scores: [B, H, T, S]
        # [/Block 08]

        # [Block 09] Add decoupled RoPE scores and causal mask
        rope_scores = torch.matmul(q_rope, rope_key.transpose(-2, -1))
        scores = (content_scores + rope_scores) * (self.qk_head_dim**-0.5)

        key_positions = torch.arange(latent.shape[1], device=x.device).unsqueeze(
            0
        )
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
        # [/Block 09]

        # [Block 10] Restore values from latent and perform weighted read
        values = self.value_up_proj(latent).view(
            batch,
            latent.shape[1],
            self.num_heads,
            self.value_dim,
        ).transpose(1, 2)
        head_outputs = torch.matmul(attention, values)
        # values: [B, H, S, Dv]; head_outputs: [B, H, T, Dv]
        # [/Block 10]

        # [Block 11] Concatenate heads and project output
        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch, query_length, self.num_heads * self.value_dim
        )
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 11]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = MultiHeadLatentAttention(
        d_model=32,
        num_heads=4,
        q_lora_rank=12,
        kv_lora_rank=10,
        qk_content_dim=6,
        qk_rope_dim=4,
        value_dim=8,
    ).eval()
    x = torch.randn(2, 6, 32)

    absorbed_output, full_cache = model(
        x, use_cache=True, implementation="absorbed"
    )
    rebuilt_output, _ = model(x, implementation="reconstruct")

    cache = None
    decoded_pieces = []
    for token in range(x.shape[1]):
        piece, cache = model(
            x[:, token : token + 1],
            kv_cache=cache,
            use_cache=True,
            implementation="absorbed",
        )
        decoded_pieces.append(piece)
    decoded_output = torch.cat(decoded_pieces, dim=1)

    assert absorbed_output.shape == (2, 6, 32)
    assert full_cache is not None
    assert full_cache[0].shape == (2, 6, 10)
    assert full_cache[1].shape == (2, 1, 6, 4)
    assert torch.allclose(absorbed_output, rebuilt_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(absorbed_output, decoded_output, atol=1e-5, rtol=1e-5)
    print("MLA smoke test passed:", tuple(absorbed_output.shape))


if __name__ == "__main__":
    _smoke_test()
