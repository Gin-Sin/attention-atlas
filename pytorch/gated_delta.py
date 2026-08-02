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
            bias=True,
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
        memory: optional ``S [B,H,Dk,Dv]``.

    The update is written in a prediction-error form:

    ``S_bar = alpha_t S``
    ``S <- S_bar + beta_t k_t (v_t - S_bar^T k_t)^T``.

    With unit-norm keys this expands to
    ``alpha_t (I - beta_t k_t k_t^T) S + beta_t k_t v_t^T``.
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
    """Streaming state for the recurrent memory and causal-conv history."""

    memory: Tensor  # [B,H,Dh,Dh]
    conv_cache: Tensor  # [B,K-1,C]


class GatedDeltaAttention(nn.Module):
    """Short-conv -> normalized delta memory -> output-gated projection."""

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

        self.short_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha_proj = nn.Linear(d_model, num_heads)
        self.beta_proj = nn.Linear(d_model, num_heads)
        self.output_gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Start with fairly persistent memory and a moderate delta step.
        nn.init.constant_(self.alpha_proj.bias, 2.0)
        nn.init.zeros_(self.beta_proj.bias)

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
        conv_cache = None if state is None else state.conv_cache
        memory = None if state is None else state.memory
        mixed, new_cache = self.short_conv(x, cache=conv_cache)

        q = self._heads(self.q_proj(mixed))
        k = self._heads(self.k_proj(mixed))
        v = self._heads(self.v_proj(mixed))
        alpha = torch.sigmoid(self.alpha_proj(mixed))  # [B,T,H], scalar/head
        beta = torch.sigmoid(self.beta_proj(mixed))  # [B,T,H]

        y, new_memory = gated_delta_recurrence(
            q, k, v, alpha, beta, memory=memory
        )
        output_gate = torch.sigmoid(self.output_gate_proj(mixed))
        output_gate = self._heads(output_gate)  # [B,T,H,Dh]
        y = y * output_gate

        batch, length, _, _ = y.shape
        y = self.out_proj(y.reshape(batch, length, self.d_model))
        return y, GatedDeltaState(new_memory, new_cache)
# [/Block 03]


# [Block 04] Reference simplifications
REFERENCE_SIMPLIFICATIONS = (
    "The recurrence uses a Python loop rather than a WY/scan chunk kernel.",
    "A single depthwise convolution stands in for production local mixing.",
    "Gates and projections are pedagogical parameterizations, not checkpoint-compatible.",
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
