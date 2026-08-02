"""Kimi Delta Attention (KDA) concepts in readable PyTorch.

KDA replaces one scalar forget gate per head with a gate per key channel.  The
resulting state transition is diagonal plus a key-tied rank-1 correction.  This
reference exposes that algebra directly; it is not an implementation of the
optimized Kimi Linear kernels.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# [Block 01] Causal short convolution
class CausalDepthwiseConv1d(nn.Module):
    """Depthwise causal convolution over ``x [B,T,C]`` with a small cache."""

    def __init__(self, channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=True,
        )

    def forward(
        self, x: Tensor, cache: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
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


# [Block 02] Channel-gated DPLR recurrence
def kda_recurrence(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    memory: Optional[Tensor] = None,
    norm_eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Apply the KDA diagonal-plus-rank-1 state transition.

    Shapes:
        q, k: ``[B,T,H,Dk]`` (normalized internally).
        v: ``[B,T,H,Dv]``.
        alpha: ``[B,T,H,Dk]`` channel-wise retention in ``(0, 1)``.
        beta: ``[B,T,H]`` rank-1 delta strength.
        memory: optional ``S [B,H,Dk,Dv]``.

    For each token:

    ``S_bar = Diag(alpha_t) S``
    ``S <- S_bar + beta_t k_t (v_t - S_bar^T k_t)^T``.

    Equivalently,
    ``S <- (I - beta_t k_t k_t^T) Diag(alpha_t) S
           + beta_t k_t v_t^T``.
    Thus the transition matrix is a restricted DPLR form: its diagonal and
    rank-1 factors are not arbitrary; the low-rank correction is tied to the
    current key.
    """

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [B, T, H, D]")
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k must match and v must share [B, T, H]")
    if alpha.shape != q.shape or beta.shape != q.shape[:3]:
        raise ValueError("alpha must match q; beta must have shape [B, T, H]")

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
        decayed = alpha[:, t, :, :, None] * memory  # Diag(alpha_t) S
        prediction = torch.einsum("bhkv,bhk->bhv", decayed, key_t)
        error = v[:, t] - prediction
        rank_one = torch.einsum("bhk,bhv->bhkv", key_t, error)
        memory = decayed + beta[:, t, :, None, None] * rank_one

        query_t = q[:, t]
        outputs.append(torch.einsum("bhkv,bhk->bhv", memory, query_t))

    output = torch.stack(outputs, dim=1) if outputs else v.new_empty(
        batch, 0, heads, value_dim
    )
    return output, memory
# [/Block 02]


# [Block 03] Importable KDA module
class KDAState(NamedTuple):
    """Streaming KDA memory and causal-convolution history."""

    memory: Tensor  # [B,H,Dh,Dh]
    conv_cache: Tensor  # [B,K-1,C]


class KimiDeltaAttention(nn.Module):
    """Short-conv KDA with channel retention and an output gate."""

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
        self.alpha_proj = nn.Linear(d_model, d_model)  # one alpha per key channel
        self.beta_proj = nn.Linear(d_model, num_heads)
        self.output_gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        nn.init.constant_(self.alpha_proj.bias, 2.0)
        nn.init.zeros_(self.beta_proj.bias)

    def _heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        state: Optional[KDAState] = None,
    ) -> Tuple[Tensor, KDAState]:
        """Map ``x [B,T,C]`` to output ``[B,T,C]`` and a streaming state."""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        conv_cache = None if state is None else state.conv_cache
        memory = None if state is None else state.memory
        mixed, new_cache = self.short_conv(x, cache=conv_cache)

        q = self._heads(self.q_proj(mixed))
        k = self._heads(self.k_proj(mixed))
        v = self._heads(self.v_proj(mixed))
        alpha = self._heads(torch.sigmoid(self.alpha_proj(mixed)))
        beta = torch.sigmoid(self.beta_proj(mixed))

        y, new_memory = kda_recurrence(
            q, k, v, alpha, beta, memory=memory
        )
        output_gate = self._heads(torch.sigmoid(self.output_gate_proj(mixed)))
        y = y * output_gate

        batch, length, _, _ = y.shape
        y = self.out_proj(y.reshape(batch, length, self.d_model))
        return y, KDAState(new_memory, new_cache)
# [/Block 03]


# [Block 04] Simplified NoPE global attention
NOPE_NOTE = (
    "NoPE means this teaching layer adds no absolute, rotary, or relative "
    "position embedding. Its causal mask still exposes token order through "
    "prefix membership, but the scores contain no explicit position signal."
)


class NoPECausalSelfAttention(nn.Module):
    """Tiny quadratic causal MHA used only as the hybrid's global layer.

    Kimi Linear uses MLA for its periodic global layers.  Plain MHA is used
    here so the global, uncompressed read is recognizable in a few lines.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Run full causal attention on ``x [B,T,C]``; no KV cache is modeled."""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        batch, length, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        def heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q, k, v = heads(q), heads(k), heads(v)  # [B,H,T,Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        y = torch.matmul(weights, v).transpose(1, 2).contiguous()
        return self.out_proj(y.view(batch, length, self.d_model))
# [/Block 04]


# [Block 05] Educational layerwise hybrid
def educational_hybrid_schedule(
    num_layers: int = 4, kda_per_global: int = 3
) -> Tuple[str, ...]:
    """Return a layerwise schedule such as ``("kda","kda","kda","global")``."""

    if num_layers < 1 or kda_per_global < 1:
        raise ValueError("num_layers and kda_per_global must be positive")
    period = kda_per_global + 1
    return tuple(
        "global" if (index + 1) % period == 0 else "kda"
        for index in range(num_layers)
    )


class EducationalKDAHybrid(nn.Module):
    """Pre-norm residual stack following a small 3-KDA:1-global schedule.

    This is layerwise alternation, not a weighted mixture of two attention
    outputs in one layer.  It intentionally omits feed-forward sublayers.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 4,
        kda_per_global: int = 3,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()
        self.schedule = educational_hybrid_schedule(num_layers, kda_per_global)
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in self.schedule)
        self.layers = nn.ModuleList(
            KimiDeltaAttention(d_model, num_heads, conv_kernel_size)
            if kind == "kda"
            else NoPECausalSelfAttention(d_model, num_heads)
            for kind in self.schedule
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the offline teaching stack; recurrent/global caches are omitted."""

        for kind, norm, layer in zip(self.schedule, self.norms, self.layers):
            normalized = norm(x)
            if kind == "kda":
                update, _ = layer(normalized)
            else:
                update = layer(normalized)
            x = x + update
        return x
# [/Block 05]


# [Block 06] Reference simplifications
REFERENCE_SIMPLIFICATIONS = (
    "The KDA recurrence is sequential Python, not a chunkwise UT/WY kernel.",
    "Long-chunk numerical safeguards and higher-precision accumulators are omitted.",
    "Gate parameterizations are illustrative and not checkpoint-compatible.",
    "The global layer is vanilla causal MHA rather than production MLA.",
    "The hybrid omits MLPs, distributed layouts, fused kernels, and global KV caching.",
)
# [/Block 06]


# [Block 07] Deterministic smoke test
def _smoke_test() -> None:
    torch.manual_seed(2)

    # Verify the prediction-error update against the explicit DPLR equation.
    initial = torch.randn(1, 1, 3, 2)
    q = torch.randn(1, 1, 1, 3)
    k = F.normalize(torch.randn(1, 1, 1, 3), dim=-1)
    v = torch.randn(1, 1, 1, 2)
    alpha = torch.tensor([[[[0.95, 0.80, 0.60]]]])
    beta = torch.tensor([[[0.4]]])
    _, updated = kda_recurrence(q, k, v, alpha, beta, memory=initial)

    key = k[0, 0, 0]
    transition = (
        torch.eye(3) - beta.item() * torch.outer(key, key)
    ) @ torch.diag(alpha[0, 0, 0])
    expected = transition @ initial[0, 0] + beta.item() * torch.outer(
        key, v[0, 0, 0]
    )
    torch.testing.assert_close(updated[0, 0], expected)

    # The short-conv and recurrent caches make the KDA layer streamable.
    model = KimiDeltaAttention(d_model=12, num_heads=3, conv_kernel_size=3)
    model.eval()
    x = torch.randn(2, 6, 12)
    with torch.no_grad():
        full, full_state = model(x)
        pieces = []
        stream_state = None
        for t in range(x.shape[1]):
            piece, stream_state = model(x[:, t : t + 1], stream_state)
            pieces.append(piece)
        streamed = torch.cat(pieces, dim=1)
    torch.testing.assert_close(streamed, full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(stream_state.memory, full_state.memory)

    hybrid = EducationalKDAHybrid(d_model=12, num_heads=3)
    hybrid.eval()
    with torch.no_grad():
        hybrid_output = hybrid(x)
    assert hybrid.schedule == ("kda", "kda", "kda", "global")
    assert hybrid_output.shape == x.shape and torch.isfinite(hybrid_output).all()

    print("kda: explicit DPLR equation and token streaming agree")
    print("hybrid schedule:", " -> ".join(hybrid.schedule))
    print("NoPE note:", NOPE_NOTE)


if __name__ == "__main__":
    _smoke_test()
# [/Block 07]
