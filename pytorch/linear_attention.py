"""Readable ELU+1 causal linear attention.

This file favors explicit tensor operations over speed.  It shows both the
prefix-scan view used to explain parallel training and the fixed-state
recurrent view used for decoding.  Production implementations fuse operations
and use parallel scans/chunks instead of materializing every prefix state.
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# [Block 01] Positive kernel feature map
def elu_feature_map(x: Tensor) -> Tensor:
    """Map arbitrary features to positive features with ``ELU(x) + 1``."""

    return F.elu(x) + 1.0


def _check_qkv(q: Tensor, k: Tensor, v: Tensor) -> None:
    """Check the educational layout: q/k [B,T,H,Dk], v [B,T,H,Dv]."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shapes [B, T, H, D]")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v must share [B, T, H]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same feature dimension")
# [/Block 01]


# [Block 02] Prefix reference
def causal_linear_attention_prefix(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute causal kernel attention by explicitly storing every prefix.

    Args:
        q, k: Raw query/key tensors of shape ``[B, T, H, Dk]``.
        v: Value tensor of shape ``[B, T, H, Dv]``.
        eps: Positive denominator floor.

    Returns:
        Attention output of shape ``[B, T, H, Dv]``.

    ``S_prefix`` below has shape ``[B,T,H,Dk,Dv]``.  This makes the
    parallel/prefix interpretation obvious, but defeats the constant-memory
    benefit and is not how a production chunkwise kernel would be written.
    """

    _check_qkv(q, k, v)
    q_phi = elu_feature_map(q)
    k_phi = elu_feature_map(k)

    writes = torch.einsum("bthk,bthv->bthkv", k_phi, v)
    s_prefix = writes.cumsum(dim=1)
    z_prefix = k_phi.cumsum(dim=1)

    numerator = torch.einsum("bthk,bthkv->bthv", q_phi, s_prefix)
    denominator = torch.einsum("bthk,bthk->bth", q_phi, z_prefix)
    return numerator / denominator.clamp_min(eps).unsqueeze(-1)
# [/Block 02]


# [Block 03] Recurrent fixed-size state
class LinearAttentionState(NamedTuple):
    """Decoder state: S [B,H,Dk,Dv] and z [B,H,Dk]."""

    s: Tensor
    z: Tensor


def causal_linear_attention_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    state: Optional[LinearAttentionState] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, LinearAttentionState]:
    """Run the same causal attention as a token-by-token recurrence.

    At token ``t`` this performs

    ``S <- S + phi(k_t) v_t^T``, ``z <- z + phi(k_t)``, then
    ``y_t = phi(q_t)^T S / (phi(q_t)^T z)``.

    Only the final ``S`` and ``z`` are retained, so recurrent-state storage is
    independent of sequence length.
    """

    _check_qkv(q, k, v)
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    q_phi = elu_feature_map(q)
    k_phi = elu_feature_map(k)

    if state is None:
        s = q.new_zeros(batch, heads, key_dim, value_dim)
        z = q.new_zeros(batch, heads, key_dim)
    else:
        s, z = state
        expected_s = (batch, heads, key_dim, value_dim)
        expected_z = (batch, heads, key_dim)
        if s.shape != expected_s or z.shape != expected_z:
            raise ValueError(
                f"state shapes must be {expected_s} and {expected_z}, "
                f"got {tuple(s.shape)} and {tuple(z.shape)}"
            )

    outputs = []
    for t in range(length):
        key_t = k_phi[:, t]  # [B,H,Dk]
        value_t = v[:, t]  # [B,H,Dv]
        s = s + torch.einsum("bhk,bhv->bhkv", key_t, value_t)
        z = z + key_t

        query_t = q_phi[:, t]  # [B,H,Dk]
        numerator = torch.einsum("bhk,bhkv->bhv", query_t, s)
        denominator = torch.einsum("bhk,bhk->bh", query_t, z)
        outputs.append(numerator / denominator.clamp_min(eps).unsqueeze(-1))

    output = torch.stack(outputs, dim=1) if outputs else v.new_empty(
        batch, 0, heads, value_dim
    )
    return output, LinearAttentionState(s=s, z=z)
# [/Block 03]


# [Block 04] Importable multi-head module
class CausalLinearAttention(nn.Module):
    """A small self-attention module exposing prefix and recurrent modes.

    Important positional limitation:
        The state is a sum over key/value writes.  Without a positional
        mechanism, permuting already-seen key/value pairs leaves ``S`` and
        ``z`` unchanged.  Causality tells the model *which prefix* is visible,
        but ELU+1 kernel attention alone does not encode order or relative
        distance inside that prefix.  Real models add positional features,
        local convolutions, or other order-sensitive mixing.
    """

    def __init__(self, d_model: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim)

    def forward(
        self,
        x: Tensor,
        state: Optional[LinearAttentionState] = None,
        mode: str = "recurrent",
    ) -> Tuple[Tensor, LinearAttentionState]:
        """Process ``x [B,T,C]`` and return ``(output [B,T,C], final_state)``."""

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B, T, {self.d_model}]")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if mode == "recurrent":
            y, final_state = causal_linear_attention_recurrent(
                q, k, v, state=state, eps=self.eps
            )
        elif mode == "prefix":
            if state is not None:
                raise ValueError("prefix mode is a full-sequence reference only")
            y = causal_linear_attention_prefix(q, k, v, eps=self.eps)
            k_phi = elu_feature_map(k)
            final_state = LinearAttentionState(
                s=torch.einsum("bthk,bthv->bhkv", k_phi, v),
                z=k_phi.sum(dim=1),
            )
        else:
            raise ValueError("mode must be 'recurrent' or 'prefix'")

        batch, length, _, _ = y.shape
        return self.out_proj(y.reshape(batch, length, self.d_model)), final_state
# [/Block 04]


# [Block 05] Deterministic smoke test
def _smoke_test() -> None:
    torch.manual_seed(0)
    batch, length, heads, key_dim, value_dim = 2, 7, 2, 4, 3
    q = torch.randn(batch, length, heads, key_dim)
    k = torch.randn(batch, length, heads, key_dim)
    v = torch.randn(batch, length, heads, value_dim)

    prefix = causal_linear_attention_prefix(q, k, v)
    recurrent, state = causal_linear_attention_recurrent(q, k, v)
    torch.testing.assert_close(recurrent, prefix, rtol=1e-5, atol=1e-6)

    first, split_state = causal_linear_attention_recurrent(q[:, :3], k[:, :3], v[:, :3])
    second, _ = causal_linear_attention_recurrent(
        q[:, 3:], k[:, 3:], v[:, 3:], state=split_state
    )
    torch.testing.assert_close(torch.cat((first, second), dim=1), recurrent)

    # Swapping completed history writes leaves the final commutative state
    # unchanged: a concrete demonstration of the no-position limitation.
    order = torch.tensor([1, 0, 2, 3, 4, 5, 6])
    _, permuted_state = causal_linear_attention_recurrent(q, k[:, order], v[:, order])
    torch.testing.assert_close(permuted_state.s, state.s)
    torch.testing.assert_close(permuted_state.z, state.z)

    print("linear_attention: prefix == recurrent == streamed")
    print("state shapes:", tuple(state.s.shape), tuple(state.z.shape))
    print("note: ELU+1 state alone is invariant to permutations of past writes")


if __name__ == "__main__":
    _smoke_test()
# [/Block 05]
