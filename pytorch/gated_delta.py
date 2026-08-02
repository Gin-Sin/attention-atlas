"""Educational recurrent Gated Delta attention in plain PyTorch.

The implementation is intentionally token-by-token so the state update is
visible.  It is suitable for CPU experiments, not throughput comparisons.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# [Block 01] Causal short convolution
class CausalDepthwiseConv1d(nn.Module):
    """Depthwise short convolution for ``x [B,T,C]`` with a streaming cache."""

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            bias=False,
        )

    def forward(
        self, x: Tensor, cache: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return convolved ``[B,T,C]`` and the last ``kernel_size-1`` inputs."""

        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError(f"x must have shape [B, T, {self.channels}]")
        batch, length, channels = x.shape
        cache_length = self.kernel_size - 1
        if cache is None:
            cache = x.new_zeros(batch, cache_length, channels)
        elif cache.shape != (batch, cache_length, channels):
            raise ValueError(
                "cache must have shape "
                f"{(batch, cache_length, channels)}, got {tuple(cache.shape)}"
            )

        history = torch.cat((cache, x), dim=1)
        y = self.conv(history.transpose(1, 2)).transpose(1, 2)
        if y.shape[1] != length:
            raise RuntimeError("causal convolution produced an unexpected length")
        new_cache = (
            history[:, -cache_length:]
            if cache_length
            else x.new_empty(batch, 0, channels)
        )
        return y, new_cache
# [/Block 01]


# [Block 02] Scalar-gated delta recurrence
def gated_delta_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    memory: Optional[Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Apply a normalized, scalar-gated delta rule.

    Shapes:
        q, k: ``[B,T,H,Dk]`` (normalized internally).
        v: ``[B,T,H,Dv]``.
        alpha: ``[B,T,H]``, one forget scalar per head and token.
        beta: ``[B,T,H]``, one delta step size per head and token.
        memory: optional tutorial state ``S [B,H,Dk,Dv]``.

    The update is written in a prediction-error form:

    ``S_bar = alpha_t S``
    ``S <- S_bar + beta_t k_t (v_t - S_bar^T k_t)^T``.

    With unit-norm keys this expands to
    ``alpha_t (I - beta_t k_t k_t^T) S + beta_t k_t v_t^T``.

    The paper commonly stores a value-by-key fast-weight matrix ``F``.  This
    tutorial stores the transpose ``S = F^T`` so keys index rows and the code
    can share the ``[Dk,Dv]`` state orientation used by the other chapters.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [B, T, H, D]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k must match and v must share [B, T, H]")
    if alpha.shape != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("alpha and beta must have shape [B, T, H]")

    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    q = F.normalize(q, p=2.0, dim=-1, eps=norm_eps)
    k = F.normalize(k, p=2.0, dim=-1, eps=norm_eps)

    if memory is None:
        memory = q.new_zeros(batch, heads, key_dim, value_dim)
    elif memory.shape != (batch, heads, key_dim, value_dim):
        raise ValueError(
            "memory must have shape "
            f"{(batch, heads, key_dim, value_dim)}, got {tuple(memory.shape)}"
        )

    outputs = []
    for t in range(length):
        key_t = k[:, t]  # [B,H,Dk]
        decayed = alpha[:, t, :, None, None] * memory
        prediction = torch.einsum("bhkv,bhk->bhv", decayed, key_t)
        error = v[:, t] - prediction
        correction = torch.einsum("bhk,bhv->bhkv", key_t, error)
        memory = decayed + beta[:, t, :, None, None] * correction

        query_t = q[:, t]  # [B,H,Dk]
        outputs.append(torch.einsum("bhkv,bhk->bhv", memory, query_t))

    output = torch.stack(outputs, dim=1) if outputs else v.new_empty(
        batch, 0, heads, value_dim
    )
    return output, memory
# [/Block 02]


# [Block 03] Importable Gated Delta module
class GatedDeltaState(NamedTuple):
    """Streaming memory plus the independent q/k/v ShortConv histories."""

    memory: Tensor  # [B,H,Dh,Dh]
    conv_cache: Tuple[Tensor, Tensor, Tensor]  # q, k, v: each [B,K-1,C]


class GatedDeltaAttention(nn.Module):
    """Paper-shaped q/k/v ShortConv paths with scalar GDN retention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_conv1d = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv1d = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv1d = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=False)
        self.output_gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        # softplus(-2.25) is close to 0.1, hence alpha starts near exp(-0.1).
        self.alpha_bias = nn.Parameter(torch.full((num_heads,), -2.25))
        self.output_norm = nn.RMSNorm(self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        state: Optional[GatedDeltaState] = None,
    ) -> Tuple[Tensor, GatedDeltaState]:
        """Map ``x [B,T,C]`` to output ``[B,T,C]`` and a streaming state."""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        memory = None if state is None else state.memory
        q_cache, k_cache, v_cache = (
            (None, None, None) if state is None else state.conv_cache
        )

        q, new_q_cache = self.q_conv1d(self.q_proj(x), cache=q_cache)
        k, new_k_cache = self.k_conv1d(self.k_proj(x), cache=k_cache)
        v, new_v_cache = self.v_conv1d(self.v_proj(x), cache=v_cache)
        q = self._heads(F.silu(q))
        k = self._heads(F.silu(k))
        v = self._heads(F.silu(v))

        # Gates deliberately bypass ShortConv and read the original token x.
        alpha_logits = self.alpha_proj(x)
        log_alpha = -torch.exp(self.A_log)[None, None, :] * F.softplus(
            alpha_logits + self.alpha_bias
        )
        alpha = torch.exp(log_alpha)  # [B,T,H], one retention scalar per head
        beta = torch.sigmoid(self.beta_proj(x))  # [B,T,H], direct write gate

        y, new_memory = gated_delta_recurrence(
            q, k, v, alpha, beta, memory=memory
        )
        output_gate = F.silu(self.output_gate_proj(x))
        output_gate = self._heads(output_gate)  # [B,T,H,Dh]
        y = self.output_norm(y) * output_gate

        batch, length, _, _ = y.shape
        y = self.out_proj(y.reshape(batch, length, self.d_model))
        new_conv_cache = (new_q_cache, new_k_cache, new_v_cache)
        return y, GatedDeltaState(new_memory, new_conv_cache)
# [/Block 03]


# [Block 04] Reference simplifications
REFERENCE_SIMPLIFICATIONS = (
    "The recurrence uses a Python loop rather than a WY/scan chunk kernel.",
    "Projection widths are kept equal to d_model instead of using production expansions.",
    "Gate equations follow GDN, but initialization is simplified and not checkpoint-compatible.",
    "No fused kernels, tensor-parallel layout, mixed-precision policy, or custom backward is used.",
)
# [/Block 04]


# [Block 05] Deterministic smoke test
def _smoke_test() -> None:
    torch.manual_seed(1)
    model = GatedDeltaAttention(d_model=16, num_heads=4, conv_kernel_size=3)
    model.eval()
    x = torch.randn(2, 6, 16)

    with torch.no_grad():
        full, full_state = model(x)
        streamed_parts = []
        streamed_state = None
        for t in range(x.shape[1]):
            part, streamed_state = model(x[:, t : t + 1], streamed_state)
            streamed_parts.append(part)
        streamed = torch.cat(streamed_parts, dim=1)

    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(streamed_state.memory, full_state.memory)
    assert torch.isfinite(full).all()
    print("gated_delta: full sequence == token streaming")
    print("output/memory:", tuple(full.shape), tuple(full_state.memory.shape))


if __name__ == "__main__":
    _smoke_test()
# [/Block 05]
