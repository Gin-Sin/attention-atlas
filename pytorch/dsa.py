"""Educational, CPU-friendly reference for Dynamic Sparse Attention (DSA).

This file preserves the two-stage idea: a cheap, low-dimensional indexer scans
the causal history, then the core attention reads only its top-k choices.
Both stages follow the paper's structure:

* the indexer q/k path mirrors the numeric pipeline — partial RoPE, an
  orthonormal Hadamard rotation, then FP8 (e4m3) storage — implemented as an
  eager float32 simulation; and
* the core is MLA in absorbed (MQA-mode) form: the gathered candidates are
  the original latent entries ``c_kv`` plus the shared decoupled RoPE key,
  scored by folding the key up-projection into the query and read by folding
  the value up-projection into the output write.

It is intentionally not a reproduction of a production DeepSeek kernel: it
uses eager PyTorch gathers and no fused/quantized kernels.

Structure follows the official ``DeepSeek-V3.2-Exp/inference/model.py``:
the indexer queries are projected from the shared MLA low-rank query latent,
indexer keys are LayerNorm-ed before RoPE, and the per-head combination
weights are an unconstrained projection scaled by ``H_I**-0.5``.  The
official reference realizes sparsity by masking full scores with ``-inf``
and keeps preallocated cache buffers indexed by ``start_pos``; this teaching
version gathers the selected candidates explicitly and threads a growing
cache tuple instead, which changes execution strategy but not the math.

``forward`` returns ``(output, new_cache)``.  When ``use_cache=True`` the
cache is ``(latent[B,S,Dc], rope_key[B,1,S,Dr], index_key[B,S,DI] (FP8),
index_scale[B,S,1] (float32))``: the MLA latent cache and shared positional
key that every MLA decoder keeps, plus the indexer keys that DSA
additionally caches.  Mirroring the official ``k_cache``/``k_scale_cache``
pair, indexer keys are stored as a real ``float8_e4m3fn`` payload with one
absmax scale per token (the official 128-wide scaling block equals its
indexer head dimension).  Cached decoding assumes contiguous positions
beginning at zero.

Shapes use ``B`` (batch), ``T`` (new query tokens), ``S`` (all cached
tokens), ``L`` (sequence length), ``H`` (heads), ``Dq`` (query latent rank),
``Dc`` (KV latent rank), ``Dn`` (non-RoPE content dimension), ``Dr`` (RoPE
dimension), ``Dv`` (value dimension), and ``DI`` (indexer dimension).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn

DSACache = Tuple[Tensor, Tensor, Tensor, Tensor]


# [Block 01] Partial RoPE, Hadamard/FP8, RMSNorm, and masking utilities
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


def act_quant_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize activations to FP8 (e4m3) with per-vector absmax scaling.

    Returns the ``float8_e4m3fn`` payload and a float32 scale.  This mirrors
    the official ``act_quant``: DeepSeek scales per 128-wide block, and the
    indexer head dimension equals that block width, so the official indexer
    cache also carries exactly one scale per vector.  Per-vector scaling
    keeps quantization local to its token, so cache entries written during
    incremental decoding are identical to a full-sequence recompute.
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1, keepdim=True)
    scale = (scale.clamp_min(torch.finfo(x.dtype).tiny) / fp8_max).to(torch.float32)
    payload = (x / scale.to(x.dtype)).to(torch.float8_e4m3fn)
    return payload, scale


def fp8_round_trip(x: Tensor) -> Tensor:
    """Reproduce FP8 storage precision loss in float32 eager mode."""
    payload, scale = act_quant_fp8(x)
    return payload.to(x.dtype) * scale.to(x.dtype)


class RMSNorm(nn.Module):
    """Minimal RMSNorm applied to each compressed latent (checkpoint recipe)."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance.to(x.dtype) + self.eps)
        return normalized * self.weight


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


# [Block 03] Configuration, MLA projection paths, and absorbed-query helpers
class DynamicSparseAttention(nn.Module):
    """Two-stage causal DSA teaching module with an MLA core.

    The indexer has several low-dimensional query heads (projected from the
    shared MLA query latent, as in the official implementation) and one
    LayerNorm-ed shared key; both sides follow the paper's numeric pipeline:
    partial RoPE, Hadamard rotation, FP8 round trip.  Per-head weights
    (an unconstrained projection scaled by ``H_I**-0.5``) combine ReLU dot
    products before causal top-k.  The core is absorbed-form MLA: candidates
    are gathered directly from the latent/RoPE-key caches, the key
    up-projection is folded into the query for scoring, and the value
    up-projection is folded into the output write.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_content_dim: int,
        qk_rope_dim: int,
        value_dim: int,
        index_dim: int = 16,
        num_index_heads: int = 4,
        top_k: int = 8,
        index_rotary_dim: Optional[int] = None,
        detach_indexer_input: bool = True,
    ) -> None:
        super().__init__()
        mla_dims = (q_lora_rank, kv_lora_rank, qk_content_dim, qk_rope_dim, value_dim)
        if d_model <= 0 or num_heads <= 0 or min(mla_dims) <= 0:
            raise ValueError("all dimensions and head counts must be positive")
        if q_lora_rank >= d_model or kv_lora_rank >= d_model:
            raise ValueError("latent ranks must be smaller than d_model")
        if qk_rope_dim % 2:
            raise ValueError("qk_rope_dim must be even")
        if index_dim <= 0 or num_index_heads <= 0 or top_k <= 0:
            raise ValueError("index dimensions, heads, and top_k must be positive")
        if index_dim & (index_dim - 1):
            raise ValueError("index_dim must be a power of two for the Hadamard step")

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_content_dim = qk_content_dim
        self.qk_rope_dim = qk_rope_dim
        self.value_dim = value_dim
        self.qk_head_dim = qk_content_dim + qk_rope_dim
        self.index_dim = index_dim
        self.num_index_heads = num_index_heads
        self.top_k = top_k
        # Default: rotate half of the indexer width, rounded down to even.
        if index_rotary_dim is None:
            index_rotary_dim = 2 * (index_dim // 4)
        self.index_rotary_dim = index_rotary_dim
        self.detach_indexer_input = detach_indexer_input
        if index_rotary_dim < 0 or index_rotary_dim > index_dim or index_rotary_dim % 2:
            raise ValueError("index_rotary_dim must be even and no larger than index_dim")

        # Official layout: indexer queries come from the MLA query latent,
        # keys from the hidden state with LayerNorm, weights from the hidden
        # state without any nonlinearity.
        self.index_queries = nn.Linear(
            q_lora_rank, num_index_heads * index_dim, bias=False
        )
        self.index_key = nn.Linear(d_model, index_dim, bias=False)
        self.index_key_norm = nn.LayerNorm(index_dim)
        self.index_weights = nn.Linear(d_model, num_index_heads, bias=False)

        # MLA low-rank paths: query latent and joint KV latent + shared RoPE key.
        self.q_down_proj = nn.Linear(d_model, q_lora_rank, bias=False)
        self.q_norm = RMSNorm(q_lora_rank)
        self.q_up_proj = nn.Linear(
            q_lora_rank, num_heads * self.qk_head_dim, bias=False
        )
        self.kv_down_and_rope_proj = nn.Linear(
            d_model, kv_lora_rank + qk_rope_dim, bias=False
        )
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.key_up_proj = nn.Linear(
            kv_lora_rank, num_heads * qk_content_dim, bias=False
        )
        self.value_up_proj = nn.Linear(
            kv_lora_rank, num_heads * value_dim, bias=False
        )
        self.out_proj = nn.Linear(num_heads * value_dim, d_model, bias=False)

    def _causal_mask(self, query_positions: Tensor, key_positions: Tensor) -> Tensor:
        return query_positions[None, :, None] >= key_positions[None, None, :]

    def _query_latent(self, x: Tensor) -> Tensor:
        """Normalized low-rank query latent ``[B,T,Dq]``, shared by the MLA
        query up-projection and the indexer query projection."""
        return self.q_norm(self.q_down_proj(x))

    def _mla_queries(
        self, query_latent: Tensor, positions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Return content queries ``[B,H,T,Dn]`` and rotated ``[B,H,T,Dr]``."""
        batch, length, _ = query_latent.shape
        query_heads = self.q_up_proj(query_latent).view(
            batch, length, self.num_heads, self.qk_head_dim
        ).transpose(1, 2)
        q_content, q_rope = torch.split(
            query_heads, (self.qk_content_dim, self.qk_rope_dim), dim=-1
        )
        return q_content, apply_partial_rope(q_rope, positions, self.qk_rope_dim)

    def _mla_latents(self, x: Tensor, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the normalized latent ``[B,T,Dc]`` and rotated shared
        positional key ``[B,1,T,Dr]`` — exactly what enters the cache."""
        compressed = self.kv_down_and_rope_proj(x)
        latent, rope_key = torch.split(
            compressed, (self.kv_lora_rank, self.qk_rope_dim), dim=-1
        )
        latent = self.kv_norm(latent)
        rope_key = apply_partial_rope(
            rope_key.unsqueeze(1), positions, self.qk_rope_dim
        )
        return latent, rope_key

    def _absorbed_queries(self, q_content: Tensor) -> Tensor:
        """Fold the key up-projection into the query: ``[B,H,T,Dn] -> [B,H,T,Dc]``.

        Scores can then hit cached latents directly; content keys ``k_c`` are
        never materialized on the decode path.
        """
        key_up_weight = self.key_up_proj.weight.view(
            self.num_heads, self.qk_content_dim, self.kv_lora_rank
        )
        return torch.einsum("bhtd,hdc->bhtc", q_content, key_up_weight)
    # [/Block 03]

    # [Block 04] Lightning indexer scoring over the causal history
    def indexer_scores(
        self,
        x: Tensor,
        query_latent: Tensor,
        *,
        past_index_keys: Optional[Tensor] = None,
        past_index_scales: Optional[Tensor] = None,
        first_position: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score the new queries against the full (cached + new) history.

        Returns causal index logits ``[B,T,S]``, the mask ``[1,T,S]``, and the
        appended indexer key cache: FP8 payload ``[B,S,DI]`` plus float32
        scales ``[B,S,1]``, mirroring the official k/k_scale cache pair.
        Indexer queries are projected from the shared MLA query latent; keys
        are LayerNorm-ed; both then run pRoPE -> Hadamard -> FP8.
        """
        batch, length, _ = x.shape
        source = x.detach() if self.detach_indexer_input else x
        latent_source = (
            query_latent.detach() if self.detach_indexer_input else query_latent
        )
        queries = self.index_queries(latent_source).view(
            batch, length, self.num_index_heads, self.index_dim
        )
        queries = queries.transpose(1, 2)  # [B, HI, T, DI]
        new_keys = self.index_key_norm(self.index_key(source))[:, None]
        # Unconstrained per-head weights, scaled as in the official code.
        weights = self.index_weights(source) * (self.num_index_heads ** -0.5)
        weights = weights.transpose(1, 2)  # [B, HI, T]

        # Paper pipeline for both indexer sides: pRoPE -> Hadamard -> FP8.
        positions = torch.arange(
            first_position, first_position + length, device=x.device
        )
        queries = apply_partial_rope(queries, positions, self.index_rotary_dim)
        new_keys = apply_partial_rope(new_keys, positions, self.index_rotary_dim)
        queries = fp8_round_trip(hadamard_transform(queries))
        new_payload, new_scales = act_quant_fp8(
            hadamard_transform(new_keys).squeeze(1)
        )
        if past_index_keys is None:
            key_payload, key_scales = new_payload, new_scales
        else:
            key_payload = torch.cat((past_index_keys, new_payload), dim=1)
            key_scales = torch.cat((past_index_scales, new_scales), dim=1)
        keys = key_payload.to(x.dtype) * key_scales.to(x.dtype)

        per_head = torch.relu(torch.einsum("bhld,bsd->bhls", queries, keys))
        scores = (per_head * weights[..., None]).sum(dim=1)
        scores = scores * (self.index_dim ** -0.5)
        key_positions = torch.arange(keys.size(1), device=x.device)
        causal = self._causal_mask(positions, key_positions)
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        return scores, causal, key_payload, key_scales
    # [/Block 04]

    # [Block 05] Dense MLA teacher distribution for indexer alignment
    def dense_teacher_probs(self, x: Tensor) -> Tensor:
        """Build a detached-target candidate from dense (full-history) MLA."""
        length = x.size(1)
        positions = torch.arange(length, device=x.device)
        q_content, q_rope = self._mla_queries(self._query_latent(x), positions)
        latent, rope_key = self._mla_latents(x, positions)
        content_scores = torch.einsum(
            "bhtc,bsc->bhts", self._absorbed_queries(q_content), latent
        )
        rope_scores = torch.matmul(q_rope, rope_key.transpose(-2, -1))
        logits = (content_scores + rope_scores) * (self.qk_head_dim ** -0.5)
        causal = self._causal_mask(positions, positions)[:, None, :, :]
        return masked_softmax(logits, causal, dim=-1).detach()
    # [/Block 05]

    # [Block 06] Causal top-k address selection
    def forward(
        self,
        x: Tensor,
        *,
        kv_cache: Optional[DSACache] = None,
        use_cache: bool = False,
        teacher_probs: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Union[
        Tuple[Tensor, Optional[DSACache]],
        Tuple[Tensor, Optional[DSACache], Dict[str, Tensor]],
    ]:
        """Attend from new tokens ``x [B,T,d_model]``.

        Returns ``(output, new_cache)``, plus an aux dict when requested.
        With ``kv_cache`` the new tokens continue at position ``S`` after the
        cached history; without it this is an ordinary full-sequence pass.
        """
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [B,T,d_model]")
        batch, length, _ = x.shape
        if length == 0:
            raise ValueError("sequence length must be positive")

        past_latent = past_rope_key = past_index_keys = past_index_scales = None
        first_position = 0
        if kv_cache is not None:
            past_latent, past_rope_key, past_index_keys, past_index_scales = kv_cache
            past_length = past_latent.size(1)
            if (
                past_latent.shape != (batch, past_length, self.kv_lora_rank)
                or past_rope_key.shape != (batch, 1, past_length, self.qk_rope_dim)
                or past_index_keys.shape != (batch, past_length, self.index_dim)
                or past_index_keys.dtype != torch.float8_e4m3fn
                or past_index_scales.shape != (batch, past_length, 1)
            ):
                raise ValueError("invalid DSA cache shape")
            first_position = past_length

        query_latent = self._query_latent(x)
        index_logits, causal, index_keys, index_scales = self.indexer_scores(
            x,
            query_latent,
            past_index_keys=past_index_keys,
            past_index_scales=past_index_scales,
            first_position=first_position,
        )
        total_length = index_keys.size(1)
        choices = min(self.top_k, total_length)
        _, selected = torch.topk(index_logits, k=choices, dim=-1)
        selected_valid = torch.gather(causal.expand(batch, -1, -1), 2, selected)

        # Fixed-width top-k has padded choices for early queries. Replace their
        # gather addresses with the causal self position, then mask them out.
        query_positions = torch.arange(
            first_position, first_position + length, device=x.device
        )
        safe_self = query_positions[None, :, None].expand_as(selected)
        selected = torch.where(selected_valid, selected, safe_self)
        # [/Block 06]

        # [Block 07] Append the latent/RoPE/index caches and gather candidates
        q_content, q_rope = self._mla_queries(query_latent, query_positions)
        new_latent, new_rope_key = self._mla_latents(x, query_positions)
        latent = new_latent if past_latent is None else torch.cat(
            (past_latent, new_latent), dim=1
        )
        rope_key = new_rope_key if past_rope_key is None else torch.cat(
            (past_rope_key, new_rope_key), dim=2
        )
        new_cache: Optional[DSACache] = (
            (latent, rope_key, index_keys, index_scales) if use_cache else None
        )
        # Candidates are the original cached entries, not rebuilt K/V:
        # latent c_kv [B,T,K,Dc] and the shared rotated RoPE key [B,T,K,Dr].
        selected_latent = gather_per_query(latent.unsqueeze(1), selected).squeeze(1)
        selected_rope_keys = gather_per_query(rope_key, selected).squeeze(1)
        # [/Block 07]

        # [Block 08] Candidate-only absorbed MLA attention
        latent_queries = self._absorbed_queries(q_content)
        content_scores = torch.einsum(
            "bhtc,btkc->bhtk", latent_queries, selected_latent
        )
        rope_scores = torch.einsum("bhtd,btkd->bhtk", q_rope, selected_rope_keys)
        core_logits = (content_scores + rope_scores) * (self.qk_head_dim ** -0.5)
        probabilities = masked_softmax(
            core_logits, selected_valid[:, None, :, :], dim=-1
        )
        # Read in latent space, then fold the value up-projection into the
        # output write: one projection per head, applied after the sum.
        latent_read = torch.einsum(
            "bhtk,btkc->bhtc", probabilities, selected_latent
        )
        value_up_weight = self.value_up_proj.weight.view(
            self.num_heads, self.value_dim, self.kv_lora_rank
        )
        head_outputs = torch.einsum("bhtc,hvc->bhtv", latent_read, value_up_weight)
        context = head_outputs.transpose(1, 2).contiguous().view(batch, length, -1)
        # [/Block 08]

        # [Block 09] Output projection and auxiliary results
        result = self.out_proj(context)

        if not return_aux:
            return result, new_cache
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
        return result, new_cache, aux
# [/Block 09]


# [Block 10] Deterministic CPU smoke test
def _smoke_test() -> None:
    torch.manual_seed(7)
    model = DynamicSparseAttention(
        d_model=32,
        num_heads=4,
        q_lora_rank=12,
        kv_lora_rank=10,
        qk_content_dim=6,
        qk_rope_dim=4,
        value_dim=8,
        index_dim=8,
        num_index_heads=3,
        top_k=4,
        index_rotary_dim=4,
    )
    model.eval()
    x = torch.randn(2, 9, 32)
    teacher = model.dense_teacher_probs(x)
    output, full_cache, aux = model(
        x, use_cache=True, teacher_probs=teacher, return_aux=True
    )

    query_position = torch.arange(x.size(1))[None, :, None]
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(aux["indexer_kl"])
    assert torch.all(aux["selected_indices"] <= query_position)
    assert full_cache is not None
    assert full_cache[0].shape == (2, 9, 10)  # latent c_kv [B,S,Dc]
    assert full_cache[1].shape == (2, 1, 9, 4)  # shared RoPE key [B,1,S,Dr]
    assert full_cache[2].shape == (2, 9, 8)  # indexer key payload [B,S,DI]
    assert full_cache[2].dtype == torch.float8_e4m3fn
    assert full_cache[3].shape == (2, 9, 1)  # per-token absmax scales [B,S,1]

    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 5
    changed_output, _ = model(changed)
    torch.testing.assert_close(output[:, :6], changed_output[:, :6])

    # Token-by-token cached decoding must reproduce the full-sequence pass.
    cache = None
    steps = []
    for step in range(x.size(1)):
        piece, cache = model(x[:, step : step + 1], kv_cache=cache, use_cache=True)
        steps.append(piece)
    torch.testing.assert_close(torch.cat(steps, dim=1), output)
    assert cache is not None
    torch.testing.assert_close(cache[0], full_cache[0])
    assert torch.equal(
        cache[2].to(torch.float32), full_cache[2].to(torch.float32)
    )
    torch.testing.assert_close(cache[3], full_cache[3])
    print(
        "DSA smoke test passed:",
        tuple(output.shape),
        f"KL={aux['indexer_kl'].item():.6f}",
    )


if __name__ == "__main__":
    _smoke_test()
# [/Block 10]
