from typing import Tuple, Optional
from contextlib import nullcontext

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
import einops
import math

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    from flash_attn import flash_attn_func

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, mlp_dropout: float = 0.0):
        super().__init__()
        
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(mlp_dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.mlp_dropout(F.silu(gate) * up))


class ConvSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        conv_kernel: int = 2,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()

        inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.inter = inter
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.dwconv = nn.Conv1d(
            in_channels=inter,
            out_channels=inter,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=inter,
            bias=True,
        ).to(dtype=torch.bfloat16)

        self.act = nn.SiLU()
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, timer: Optional[object] = None, prefix: str = ""):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up
        x_conv = self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))
        x_conv = x_conv[..., :up.size(1)]
        x_conv = self.act(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_out = self.down_proj(x_conv)

        return x_out


class FullyLinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = round(expansion * hidden_size)

        self.up_proj = nn.Linear(hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class LinearGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(gate + up)


class SiLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.silu(x)
        return self.down_proj(x)


class LinearSwish(nn.Module):
    def __init__(self, hidden_size: int, reverse=False):
        super().__init__()

        self.linear = CastedLinear(hidden_size, hidden_size, bias=False)
        self.reverse = reverse

    def forward(self, x):
        if self.reverse:
            return F.silu(self.linear(x))
        else:
            return self.linear(F.silu(x))


class ReLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size), 256)

        self.up_proj = CastedLinear(hidden_size, inter, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        x = self.up_proj(x)
        x = F.relu(x)
        return self.down_proj(x)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
