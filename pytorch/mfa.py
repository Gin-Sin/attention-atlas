"""Educational causal Multi-matrix Factorization Attention (MFA) in PyTorch.

This module demonstrates the defining MFA ideas (arXiv:2412.19255):

* one shared ``C``-dimensional key and one shared ``C``-dimensional value per
  token, produced by shared projections ``S_k`` and ``S_v``;
* head-specific ``C x C`` query transforms ``Q_c`` and output transforms
  ``O_c`` that live only in the weights, so adding heads never replicates the
  historical KV cache; and
* an optional MFA-KR (key-reuse) mode that reparameterizes the value as a
  zero-initialized learnable transform of the shared key, halving the cache
  from ``2C`` to ``C`` elements per token per layer.

Shapes use ``B`` (batch), ``T`` (new query tokens), ``S`` (all cached tokens),
``m`` (attention heads), ``C`` (shared latent width, which is also the
per-head factorization rank), and ``d`` (model width).

For clarity, this dense CPU reference omits tensor parallelism, quantized
caches, and fused kernels.  Both variants cache the *pre-RoPE* shared key
features so that the key-reuse variant can regenerate values from the same
cache; keys are rotated at read time from their known absolute positions.
Plain-MFA deployments may instead pre-rotate keys before caching -- that is a
layout choice, not a change to the demonstrated cache width or attention
equation.  ``rope_base`` defaults to 500,000 to match the paper's common
experimental settings; the base is an experiment choice, not part of the MFA
definition.

``forward`` returns ``(output, new_cache)``.  The cache is
``(shared_key[B, S, C], shared_value[B, S, C])`` in the standard mode and
``(shared_key[B, S, C], None)`` in key-reuse mode when ``use_cache=True``.
"""

from __future__ import annotations

from typing import Optional

import torch


MFACache = tuple[torch.Tensor, Optional[torch.Tensor]]


# [Block 01] Rotary position encoding
def apply_rope(
    tensor: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE to ``tensor[B, heads, T, C]`` at ``positions[T]``."""
    rotary_dim = tensor.shape[-1]
    if rotary_dim % 2 != 0:
        raise ValueError("RoPE requires an even dimension")
    if positions.ndim != 1 or positions.numel() != tensor.shape[-2]:
        raise ValueError("positions must contain one index per token")

    frequency_indices = torch.arange(
        0, rotary_dim, 2, device=tensor.device, dtype=torch.float32
    )
    inverse_frequencies = torch.pow(base, -frequency_indices / rotary_dim)
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


class MultiMatrixFactorizationAttention(torch.nn.Module):
    """Causal MFA with shared C-dim K/V and head-specific C x C circuits."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        shared_dim: int,
        *,
        key_reuse: bool = False,
        rope_base: float = 500_000.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or shared_dim <= 0:
            raise ValueError("d_model, num_heads and shared_dim must be positive")
        if shared_dim % 2 != 0:
            raise ValueError("shared_dim must be even for RoPE")
        if rope_base <= 0:
            raise ValueError("rope_base must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.shared_dim = shared_dim
        self.key_reuse = key_reuse
        # The paper's common settings use RoPE base 500,000; this is the
        # representative experiment configuration, not an MFA requirement.
        self.rope_base = rope_base

        # [Block 02] Shared projections and head-specific C x C transforms
        # Shared S_q / S_k (and S_v in the standard variant) map every token
        # into the same C-dimensional latent space.
        self.q_shared_proj = torch.nn.Linear(d_model, shared_dim, bias=False)
        self.k_shared_proj = torch.nn.Linear(d_model, shared_dim, bias=False)
        if key_reuse:
            # MFA-KR: the value projection is constrained to the key-derived
            # family v = k (I + diag(alpha) N) with alpha initialized to zero.
            self.key_reuse_mix = torch.nn.Parameter(
                torch.empty(shared_dim, shared_dim)
            )
            torch.nn.init.xavier_uniform_(self.key_reuse_mix)
            self.key_reuse_gate = torch.nn.Parameter(torch.zeros(shared_dim))
        else:
            self.v_shared_proj = torch.nn.Linear(
                d_model, shared_dim, bias=False
            )
        # Head-specific Q_c and O_c are full C x C matrices: they raise every
        # head's factorization rank to C but never enter the KV cache.
        self.q_head_transforms = torch.nn.Parameter(
            torch.empty(num_heads, shared_dim, shared_dim)
        )
        self.o_head_transforms = torch.nn.Parameter(
            torch.empty(num_heads, shared_dim, shared_dim)
        )
        torch.nn.init.xavier_uniform_(self.q_head_transforms)
        torch.nn.init.xavier_uniform_(self.o_head_transforms)
        self.out_proj = torch.nn.Linear(shared_dim, d_model, bias=False)
        # [/Block 02]

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: MFACache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, MFACache | None]:
        """Apply causal MFA to ``x[B, T, d_model]``.

        Cached positions are assumed contiguous and zero-based.  The cache
        never gains a head axis: it stores one shared ``C``-dim key (and, in
        the standard variant, one shared ``C``-dim value) per token.
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")

        batch, query_length, _ = x.shape
        past_length = 0
        if kv_cache is not None:
            past_key, past_value = kv_cache
            if (
                past_key.ndim != 3
                or past_key.shape[0] != batch
                or past_key.shape[-1] != self.shared_dim
            ):
                raise ValueError("invalid MFA cache shape")
            if self.key_reuse:
                if past_value is not None:
                    raise ValueError("key-reuse cache must not store values")
            elif past_value is None or past_value.shape != past_key.shape:
                raise ValueError("standard MFA cache must store shared values")
            past_length = past_key.shape[1]

        # [Block 03] Shared C-dimensional query/key/value features
        q_shared = self.q_shared_proj(x)  # [B, T, C], never cached
        k_new = self.k_shared_proj(x)  # [B, T, C], shared by all heads
        v_new = None if self.key_reuse else self.v_shared_proj(x)
        # [/Block 03]

        # [Block 04] Head-specific Q_c query expansion
        # q_{t,c} = (x_t S_q) Q_c: every head applies its own full C x C
        # transform, so head diversity lives in weights, not in the cache.
        query_heads = torch.einsum(
            "btc,hcd->bhtd", q_shared, self.q_head_transforms
        )
        # query_heads: [B, m, T, C]
        # [/Block 04]

        # [Block 05] Append the head-count-independent shared cache
        if kv_cache is not None:
            keys = torch.cat((past_key, k_new), dim=1)
            values = (
                None
                if self.key_reuse
                else torch.cat((past_value, v_new), dim=1)
            )
        else:
            keys = k_new
            values = v_new
        new_cache = (keys, values) if use_cache else None
        # Cached width per token is 2C (standard) or C (key reuse); adding
        # heads leaves this untouched.
        # [/Block 05]

        # [Block 06] Optional key-reuse value reparameterization
        if self.key_reuse:
            mix = torch.eye(
                self.shared_dim, device=x.device, dtype=x.dtype
            ) + self.key_reuse_gate.unsqueeze(1) * self.key_reuse_mix
            values = torch.matmul(keys, mix)
        # v_s = k_s (I + diag(alpha) N); alpha starts at zero, so values
        # begin exactly equal to the shared keys and training learns the rest.
        # [/Block 06]

        # [Block 07] Rotate per-head queries and the shared key
        query_positions_1d = torch.arange(
            past_length,
            past_length + query_length,
            device=x.device,
        )
        key_positions_1d = torch.arange(keys.shape[1], device=x.device)
        rotated_queries = apply_rope(
            query_heads, query_positions_1d, base=self.rope_base
        )
        rotated_keys = apply_rope(
            keys.unsqueeze(1), key_positions_1d, base=self.rope_base
        ).squeeze(1)
        # RoPE touches q and k only; the shared value stays unrotated.
        # [/Block 07]

        # [Block 08] Per-head scores, causal mask, and softmax
        scores = torch.einsum(
            "bhtc,bsc->bhts", rotated_queries, rotated_keys
        ) * (self.shared_dim**-0.5)
        # scores: [B, m, T, S]
        key_positions = key_positions_1d.unsqueeze(0)
        query_positions = query_positions_1d.unsqueeze(1)
        future_tokens = key_positions > query_positions
        scores = scores.masked_fill(
            future_tokens.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=-1)
        # [/Block 08]

        # [Block 09] Shared value read and head-specific O_c write-back
        head_reads = torch.einsum("bhts,bsc->bhtc", attention, values)
        # o_t = sum_c (sum_s a_{t,s}^{(c)} v_s) O_c^T, summed over heads.
        merged = torch.einsum(
            "bhtc,hdc->btd", head_reads, self.o_head_transforms
        )
        # merged: [B, T, C]
        # [/Block 09]

        # [Block 10] Project the merged C-dim stream back to d_model
        output = self.out_proj(merged)  # [B, T, d_model]
        # [/Block 10]
        return output, new_cache


def _smoke_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 6, 32)

    for key_reuse in (False, True):
        model = MultiMatrixFactorizationAttention(
            d_model=32,
            num_heads=3,
            shared_dim=8,
            key_reuse=key_reuse,
        ).eval()

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
        assert full_cache is not None
        assert full_cache[0].shape == (2, 6, 8)
        if key_reuse:
            assert full_cache[1] is None
        else:
            assert full_cache[1] is not None
            assert full_cache[1].shape == (2, 6, 8)
        assert torch.allclose(
            full_output, decoded_output, atol=1e-5, rtol=1e-5
        )
        label = "key-reuse" if key_reuse else "standard"
        print(f"MFA {label} smoke test passed:", tuple(full_output.shape))


if __name__ == "__main__":
    _smoke_test()
