"""Educational, CPU-friendly reference for Dynamic Sparse Attention (DSA).

This file preserves the two-stage idea: a cheap, low-dimensional indexer scans
the causal history, then full-dimensional attention reads only its top-k
choices.  The indexer q/k path mirrors the paper's numeric pipeline —
partial RoPE, an orthonormal Hadamard rotation, then FP8 (e4m3) storage —
implemented as an eager float32 simulation.  It is intentionally not a
reproduction of a production DeepSeek kernel: it uses eager PyTorch gathers
and ordinary multi-head attention instead of MLA, with no fused kernels.

Shapes use B=batch, L=sequence length, H=attention heads, and D=head dimension.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn


# [Block 01] Partial RoPE, Hadamard/FP8, and masked-softmax utilities
def apply_partial_rope(
    x: Tensor,
    positions: Tensor,
    rotary_dim: int,
    *,
    inverse: bool = False,
    base: float = 10_000.0,
) -> Tensor:
    """Rotate the first ``rotary_dim`` channels of ``x``.

    Args:
        x: ``[B, H, L, D]`` tensor.
        positions: Integer or floating positions with shape ``[L]``.
        rotary_dim: Even number of leading channels to rotate; zero disables it.
        inverse: Apply R(position)^-1 instead of R(position).
    """
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
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]

    pairs = x[..., :rotary_dim].reshape(*x.shape[:-1], rotary_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2)
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def hadamard_transform(x: Tensor) -> Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last dimension.

    Being orthonormal, it preserves dot products exactly in real arithmetic;
    its role in DSA is purely numeric: rotating away per-channel outliers so
    the coarse FP8 grid loses less information.
    """
    width = x.size(-1)
    if width & (width - 1):
        raise ValueError("Hadamard transform requires a power-of-two width")
    result = x
    stride = 1
    while stride < width:
        result = result.unflatten(-1, (width // (2 * stride), 2, stride))
        even, odd = result[..., 0, :], result[..., 1, :]
        result = torch.stack((even + odd, even - odd), dim=-2).flatten(-3)
        stride *= 2
    return result * width ** -0.5


def fp8_round_trip(x: Tensor) -> Tensor:
    """Simulate FP8 (e4m3) storage with per-tensor absmax scaling.

    Production DSA keeps the indexer's q/k in FP8 to shrink cache traffic;
    quantize-dequantize reproduces that precision loss in float32 eager mode.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax().clamp_min(torch.finfo(x.dtype).tiny) / fp8_max
    return (x / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax that returns zeros, rather than NaNs, for an all-masked row."""
    mask = mask.to(device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=dim) * mask.to(logits.dtype)
    denominator = probabilities.sum(dim=dim, keepdim=True)
    return probabilities / denominator.clamp_min(torch.finfo(logits.dtype).tiny)


def gather_per_query(sequence: Tensor, indices: Tensor) -> Tensor:
    """Gather ``[B,H,S,D]`` at per-query indices ``[B,L,K]``."""
    batch, heads, _, width = sequence.shape
    if indices.ndim != 3 or indices.size(0) != batch:
        raise ValueError("indices must have shape [B,L,K]")
    queries, choices = indices.shape[1:]
    expanded = sequence.unsqueeze(2).expand(-1, -1, queries, -1, -1)
    gather_index = indices[:, None, :, :, None].expand(
        batch, heads, queries, choices, width
    )
    return torch.gather(expanded, dim=3, index=gather_index)
# [/Block 01]


# [Block 02] Detached indexer-alignment loss
def detached_indexer_kl(
    index_logits: Tensor,
    teacher_probs: Tensor,
    causal_mask: Tensor,
) -> Tensor:
    """KL(teacher || indexer) with a detached dense-attention teacher.

    ``index_logits`` is ``[B,L,L]``. ``teacher_probs`` may be ``[B,L,L]`` or
    ``[B,H,L,L]``; multiple teacher heads are averaged.  Detaching the target
    demonstrates the separate indexer objective used by DSA-style training.
    """
    if teacher_probs.ndim == 4:
        teacher_probs = teacher_probs.mean(dim=1)
    if teacher_probs.shape != index_logits.shape:
        raise ValueError("teacher_probs must match [B,L,L] index logits")

    mask = causal_mask.expand_as(index_logits)
    target = teacher_probs.detach().to(index_logits.dtype).clamp_min(0)
    target = target * mask.to(target.dtype)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(target.dtype).tiny
    )

    finite_logits = index_logits.masked_fill(
        ~mask, torch.finfo(index_logits.dtype).min
    )
    log_indexer = torch.log_softmax(finite_logits, dim=-1)
    target_log = target.clamp_min(torch.finfo(target.dtype).tiny).log()
    return (target * (target_log - log_indexer)).sum(dim=-1).mean()
# [/Block 02]


# [Block 03] Two-stage module configuration and projections
class DynamicSparseAttention(nn.Module):
    """Two-stage causal DSA teaching module.

    The indexer has several low-dimensional query heads and one shared key;
    both sides follow the paper's numeric pipeline: partial RoPE, Hadamard
    rotation, FP8 round trip.  Positive per-head weights combine ReLU dot
    products before causal top-k.  The core uses normal full-dimensional
    Q/K/V projections on gathered tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        index_dim: int = 16,
        num_index_heads: int = 4,
        top_k: int = 8,
        rotary_dim: int = 0,
        index_rotary_dim: Optional[int] = None,
        detach_indexer_input: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if index_dim <= 0 or num_index_heads <= 0 or top_k <= 0:
            raise ValueError("index dimensions, heads, and top_k must be positive")
        if index_dim & (index_dim - 1):
            raise ValueError("index_dim must be a power of two for the Hadamard step")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.index_dim = index_dim
        self.num_index_heads = num_index_heads
        self.top_k = top_k
        self.rotary_dim = rotary_dim
        # Default: rotate half of the indexer width, rounded down to even.
        if index_rotary_dim is None:
            index_rotary_dim = 2 * (index_dim // 4)
        self.index_rotary_dim = index_rotary_dim
        self.detach_indexer_input = detach_indexer_input
        if rotary_dim < 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")
        if index_rotary_dim < 0 or index_rotary_dim > index_dim or index_rotary_dim % 2:
            raise ValueError("index_rotary_dim must be even and no larger than index_dim")

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

    def _causal_mask(self, length: int, device: torch.device) -> Tensor:
        positions = torch.arange(length, device=device)
        return positions[None, :, None] >= positions[None, None, :]
    # [/Block 03]

    # [Block 04] Lightning indexer scoring over the causal history
    def indexer_scores(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return causal index logits ``[B,L,L]`` and mask ``[1,L,L]``."""
        batch, length, _ = x.shape
        source = x.detach() if self.detach_indexer_input else x
        queries = self.index_queries(source).view(
            batch, length, self.num_index_heads, self.index_dim
        )
        queries = queries.transpose(1, 2)  # [B, HI, L, DI]
        keys = self.index_key(source)[:, None]  # [B, 1, L, DI], shared by index heads
        weights = torch.sigmoid(self.index_weights(source)).transpose(1, 2)

        # Paper pipeline for both indexer sides: pRoPE -> Hadamard -> FP8.
        positions = torch.arange(length, device=x.device)
        queries = apply_partial_rope(queries, positions, self.index_rotary_dim)
        keys = apply_partial_rope(keys, positions, self.index_rotary_dim)
        queries = fp8_round_trip(hadamard_transform(queries))
        keys = fp8_round_trip(hadamard_transform(keys)).squeeze(1)

        per_head = torch.einsum("bhld,bsd->bhls", queries, keys)
        per_head = torch.relu(per_head * (self.index_dim ** -0.5))
        scores = (per_head * weights[..., None]).sum(dim=1)
        causal = self._causal_mask(length, x.device)
        return scores.masked_fill(~causal, torch.finfo(scores.dtype).min), causal
    # [/Block 04]

    # [Block 05] Dense teacher distribution for indexer alignment
    def dense_teacher_probs(self, x: Tensor) -> Tensor:
        """Build a detached-target candidate from dense core attention."""
        length = x.size(1)
        positions = torch.arange(length, device=x.device)
        queries = apply_partial_rope(
            self._heads(self.query(x)), positions, self.rotary_dim
        )
        keys = apply_partial_rope(
            self._heads(self.key(x)), positions, self.rotary_dim
        )
        logits = torch.einsum("bhld,bhsd->bhls", queries, keys)
        logits = logits * (self.head_dim ** -0.5)
        causal = self._causal_mask(length, x.device)[:, None, :, :]
        return masked_softmax(logits, causal, dim=-1).detach()
    # [/Block 05]

    # [Block 06] Causal top-k address selection
    def forward(
        self,
        x: Tensor,
        *,
        teacher_probs: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """Attend from ``x [B,L,d_model]`` and return the same leading shape."""
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,L,d_model]")
        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence length must be positive")

        index_logits, causal = self.indexer_scores(x)
        choices = min(self.top_k, length)
        _, selected = torch.topk(index_logits, k=choices, dim=-1)
        selected_valid = torch.gather(causal.expand(batch, -1, -1), 2, selected)

        # Fixed-width top-k has padded choices for early queries. Replace their
        # gather addresses with the causal self position, then mask them out.
        query_positions = torch.arange(length, device=x.device)
        safe_self = query_positions[None, :, None].expand_as(selected)
        selected = torch.where(selected_valid, selected, safe_self)
        # [/Block 06]

        # [Block 07] Gather cached keys and values at selected addresses
        positions = torch.arange(length, device=x.device)
        queries = apply_partial_rope(
            self._heads(self.query(x)), positions, self.rotary_dim
        )
        keys = apply_partial_rope(
            self._heads(self.key(x)), positions, self.rotary_dim
        )
        values = self._heads(self.value(x))
        selected_keys = gather_per_query(keys, selected)
        selected_values = gather_per_query(values, selected)
        # [/Block 07]

        # [Block 08] Candidate-only core attention
        core_logits = torch.einsum("bhld,bhlkd->bhlk", queries, selected_keys)
        core_logits = core_logits * (self.head_dim ** -0.5)
        probabilities = masked_softmax(
            core_logits, selected_valid[:, None, :, :], dim=-1
        )
        context = torch.einsum(
            "bhlk,bhlkd->bhld", probabilities, selected_values
        )
        context = context.transpose(1, 2).contiguous().view(batch, length, -1)
        # [/Block 08]

        # [Block 09] Output projection and auxiliary results
        result = self.output(context)

        if not return_aux:
            return result
        aux: Dict[str, Tensor] = {
            "index_logits": index_logits,
            "selected_indices": selected,
            "selected_valid": selected_valid,
            "attention_probs": probabilities,
        }
        if teacher_probs is not None:
            aux["indexer_kl"] = detached_indexer_kl(
                index_logits, teacher_probs, causal
            )
        return result, aux
# [/Block 09]


# [Block 10] Deterministic CPU smoke test
def _smoke_test() -> None:
    torch.manual_seed(7)
    model = DynamicSparseAttention(
        d_model=32,
        num_heads=4,
        index_dim=8,
        num_index_heads=3,
        top_k=4,
        rotary_dim=4,
        index_rotary_dim=4,
    )
    model.eval()
    x = torch.randn(2, 9, 32)
    teacher = model.dense_teacher_probs(x)
    output, aux = model(x, teacher_probs=teacher, return_aux=True)

    query_position = torch.arange(x.size(1))[None, :, None]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(aux["indexer_kl"])
    assert torch.all(aux["selected_indices"] <= query_position)

    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 5
    changed_output = model(changed)
    torch.testing.assert_close(output[:, :6], changed_output[:, :6])
    print(
        "DSA smoke test passed:",
        tuple(output.shape),
        f"KL={aux['indexer_kl'].item():.6f}",
    )


if __name__ == "__main__":
    _smoke_test()
# [/Block 10]
