"""Educational causal Tensor Product Attention (TPA) in plain PyTorch.

This module demonstrates the defining T6/TPA ideas (arXiv:2501.06425):

* every token dynamically generates head-axis factors ``A(x_t)`` and
  channel-axis factors ``B(x_t)``, and each of Q/K/V is rebuilt as a small
  sum of outer products, e.g. ``K_t = A_K(x_t)^T B_K(x_t) / R_K``;
* RoPE acts row-wise on the channel factors ``B_Q`` / ``B_K`` only, so the
  cached key factors can be pre-rotated while the head-mixing ``A`` factors
  stay untouched; and
* the KV cache stores the low-rank K/V factors, ``(R_K + R_V)(h + d_h)``
  elements per token per layer, instead of full ``2 h d_h`` activations.

Shapes use ``B`` (batch), ``T`` (new query tokens), ``S`` (all cached
tokens), ``H`` (attention heads ``h``), ``D`` (head dimension ``d_h``), and
``Rq`` / ``Rk`` / ``Rv`` (the Q/K/V rank budgets).  ``H * D`` may exceed
``d_model``: T6 widens the attention inner width for parameter parity and
writes back through the output projection.

``implementation="factorized"`` contracts scores and outputs directly from
the cached factors without materializing full historical K/V, which is the
property that FlashTPA decoding exploits; this dense CPU reference is not a
production kernel.  ``implementation="reconstruct"`` explicitly rebuilds
``Q``/``K``/``V`` and is included to verify that the factorization algebra is
exact.  Both paths cache only the factors, with ``B_K`` already rotated.

``forward`` returns ``(output, new_cache)``.  The cache is
``(a_k[B, S, Rk, H], b_k[B, S, Rk, D], a_v[B, S, Rv, H], b_v[B, S, Rv, D])``
when ``use_cache=True``.
"""

from __future__ import annotations

import torch


TPACache = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# [Block 01] Row-wise rotary position encoding
def apply_rope_rows(
    factor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Rotate every rank row of ``factor[B, T, R, D]`` at ``positions[T]``.

    Each of the ``R`` rows of a token's channel factor is a ``D``-dim vector
    rotated by the same position-dependent block-diagonal rotation, so
    ``RoPE(A^T B) = A^T RoPE_rows(B)``.
    """
    rotary_dim = factor.shape[-1]
    if rotary_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    if positions.ndim != 1 or positions.numel() != factor.shape[1]:
        raise ValueError("positions must contain one index per token")

    frequency_indices = torch.arange(
        0, rotary_dim, 2, device=factor.device, dtype=torch.float32
    )
    inverse_frequencies = torch.pow(base, -frequency_indices / rotary_dim)
    angles = positions.to(device=factor.device, dtype=torch.float32).unsqueeze(
        1
    ) * inverse_frequencies.unsqueeze(0)
    cosine = angles.cos().to(factor.dtype).unsqueeze(0).unsqueeze(2)
    sine = angles.sin().to(factor.dtype).unsqueeze(0).unsqueeze(2)
    # cosine, sine: [1, T, 1, D/2] broadcast over batch and rank rows.

    even = factor[..., 0::2]
    odd = factor[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)
# [/Block 01]


class TensorProductAttention(torch.nn.Module):
    """Causal TPA with contextual A/B factors and a factorized KV cache."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        *,
        q_rank: int,
        k_rank: int,
        v_rank: int,
    ) -> None:
        super().__init__()
        if min(d_model, num_heads, head_dim, q_rank, k_rank, v_rank) <= 0:
            raise ValueError("all dimensions and ranks must be positive")
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_rank = q_rank
        self.k_rank = k_rank
        self.v_rank = v_rank

        # [Block 02] Six contextual factor projections
        # A projections emit head-axis factors, B projections emit
        # channel-axis factors; all six depend on the current token.
        self.a_q_proj = torch.nn.Linear(d_model, q_rank * num_heads, bias=False)
        self.b_q_proj = torch.nn.Linear(d_model, q_rank * head_dim, bias=False)
        self.a_k_proj = torch.nn.Linear(d_model, k_rank * num_heads, bias=False)
        self.b_k_proj = torch.nn.Linear(d_model, k_rank * head_dim, bias=False)
        self.a_v_proj = torch.nn.Linear(d_model, v_rank * num_heads, bias=False)
        self.b_v_proj = torch.nn.Linear(d_model, v_rank * head_dim, bias=False)
        # The attention inner width H * D may exceed d_model (T6-XL uses
        # 78 x 64 = 4992 over d_model = 1600); W^O writes it back.
        self.out_proj = torch.nn.Linear(
            num_heads * head_dim, d_model, bias=False
        )
        # [/Block 02]

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: TPACache | None = None,
        use_cache: bool = False,
        implementation: str = "factorized",
    ) -> tuple[torch.Tensor, TPACache | None]:
        """Apply causal TPA to ``x[B, T, d_model]``.

        Cached positions are assumed contiguous and zero-based.  The cached
        ``b_k`` factors are already rotated; ``b_v`` is never rotated.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        if implementation not in {"factorized", "reconstruct"}:
            raise ValueError(
                "implementation must be 'factorized' or 'reconstruct'"
            )

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_a_k, past_b_k, past_a_v, past_b_v = kv_cache
            if (
                past_a_k.shape[:2] != past_b_k.shape[:2]
                or past_a_k.shape[:2] != past_a_v.shape[:2]
                or past_a_k.shape[:2] != past_b_v.shape[:2]
                or past_a_k.shape[0] != batch
                or past_a_k.shape[2:] != (self.k_rank, self.num_heads)
                or past_b_k.shape[2:] != (self.k_rank, self.head_dim)
                or past_a_v.shape[2:] != (self.v_rank, self.num_heads)
                or past_b_v.shape[2:] != (self.v_rank, self.head_dim)
            ):
                raise ValueError("invalid TPA factor-cache shape")
            past_length = past_a_k.shape[1]

        # [Block 03] Generate the six contextual A/B factor streams
        a_q = self.a_q_proj(x).view(
            batch, query_length, self.q_rank, self.num_heads
        )
        b_q = self.b_q_proj(x).view(
            batch, query_length, self.q_rank, self.head_dim
        )
        a_k_new = self.a_k_proj(x).view(
            batch, query_length, self.k_rank, self.num_heads
        )
        b_k_new = self.b_k_proj(x).view(
            batch, query_length, self.k_rank, self.head_dim
        )
        a_v_new = self.a_v_proj(x).view(
            batch, query_length, self.v_rank, self.num_heads
        )
        b_v_new = self.b_v_proj(x).view(
            batch, query_length, self.v_rank, self.head_dim
        )
        # A decides how each rank-1 tile spreads across heads; B carries the
        # channel pattern each tile writes into every selected head.
        # [/Block 03]

        # [Block 04] Row-wise RoPE on the Q/K channel factors only
        positions = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        )
        b_q = apply_rope_rows(b_q, positions)
        b_k_new = apply_rope_rows(b_k_new, positions)
        # RoPE(A^T B) = A^T RoPE_rows(B): pre-rotating b_k before caching
        # preserves the relative-position dot product; b_v stays unrotated.
        # [/Block 04]

        # [Block 05] Append the factorized KV cache
        if kv_cache is not None:
            a_k = torch.cat((past_a_k, a_k_new), dim=1)
            b_k = torch.cat((past_b_k, b_k_new), dim=1)
            a_v = torch.cat((past_a_v, a_v_new), dim=1)
            b_v = torch.cat((past_b_v, b_v_new), dim=1)
        else:
            a_k, b_k, a_v, b_v = a_k_new, b_k_new, a_v_new, b_v_new
        new_cache = (a_k, b_k, a_v, b_v) if use_cache else None
        # Cached width per token is (Rk + Rv)(H + D), not 2 * H * D.
        # [/Block 05]

        scale = self.head_dim**-0.5
        values_full = None
        if implementation == "reconstruct":
            # [Block 06] Reference path: rebuild full Q/K/V activations
            queries_full = torch.einsum("btph,btpd->bhtd", a_q, b_q) / self.q_rank
            keys_full = torch.einsum("bsrh,bsrd->bhsd", a_k, b_k) / self.k_rank
            values_full = (
                torch.einsum("bsrh,bsrd->bhsd", a_v, b_v) / self.v_rank
            )
            scores = torch.matmul(
                queries_full, keys_full.transpose(-2, -1)
            ) * scale
            # This path verifies the algebra but materializes [B, H, S, D]
            # historical K/V, losing the main decode-efficiency benefit.
            # [/Block 06]
        else:
            # [Block 07] Factor-domain score contraction
            channel_gram = torch.einsum("btpd,bsrd->btspr", b_q, b_k)
            scores = torch.einsum(
                "btph,bsrh,btspr->bhts", a_q, a_k, channel_gram
            ) * (scale / (self.q_rank * self.k_rank))
            # q_{t,i} . k_{s,i} = (1/(Rq Rk)) sum_{p,r} A_Q[p,i] A_K[r,i]
            # (B_Q[p] . B_K[r]); no full historical K is ever materialized.
            # [/Block 07]

        # [Block 08] Causal mask and softmax
        key_positions = torch.arange(
            a_k.shape[1], device=x.device
        ).unsqueeze(0)
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
        # attention: [B, H, T, S]
        # [/Block 08]

        # [Block 09] Value aggregation from cached factors
        if implementation == "reconstruct":
            head_outputs = torch.matmul(attention, values_full)
        else:
            head_outputs = torch.einsum(
                "bhts,bsrh,bsrd->bhtd", attention, a_v, b_v
            ) / self.v_rank
        # head_outputs: [B, H, T, D]
        # [/Block 09]

        # [Block 10] Concatenate heads and project the widened stream back
        merged = head_outputs.transpose(1, 2).contiguous().view(
            batch, query_length, self.num_heads * self.head_dim
        )
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 10]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    model = TensorProductAttention(
        d_model=32,
        num_heads=5,
        head_dim=8,
        q_rank=3,
        k_rank=2,
        v_rank=2,
    ).eval()
    x = torch.randn(2, 6, 32)

    factorized_output, full_cache = model(
        x, use_cache=True, implementation="factorized"
    )
    reconstructed_output, _ = model(x, implementation="reconstruct")

    cache = None
    decoded_pieces = []
    for token in range(x.shape[1]):
        piece, cache = model(
            x[:, token : token + 1],
            kv_cache=cache,
            use_cache=True,
            implementation="factorized",
        )
        decoded_pieces.append(piece)
    decoded_output = torch.cat(decoded_pieces, dim=1)

    assert factorized_output.shape == (2, 6, 32)
    assert full_cache is not None
    assert full_cache[0].shape == (2, 6, 2, 5)  # a_k: [B, S, Rk, H]
    assert full_cache[1].shape == (2, 6, 2, 8)  # b_k: [B, S, Rk, D]
    assert full_cache[2].shape == (2, 6, 2, 5)  # a_v: [B, S, Rv, H]
    assert full_cache[3].shape == (2, 6, 2, 8)  # b_v: [B, S, Rv, D]
    assert torch.allclose(
        factorized_output, reconstructed_output, atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        factorized_output, decoded_output, atol=1e-5, rtol=1e-5
    )
    print("TPA smoke test passed:", tuple(factorized_output.shape))


if __name__ == "__main__":
    _smoke_test()
