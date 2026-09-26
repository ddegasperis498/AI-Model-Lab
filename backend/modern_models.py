from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA low-rank adapter."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 rank: int = 0, alpha: float = 1.0, dropout: float = 0.0,
                 freeze_base: bool = False):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank if self.rank > 0 else 0.0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if self.rank > 0:
            self.A = nn.Parameter(torch.empty(self.rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, self.rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
        if freeze_base and self.rank > 0:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.rank > 0:
            y = y + F.linear(F.linear(self.lora_dropout(x), self.A), self.B) * self.scale
        return y


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError('RoPE head_dim deve essere pari')
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv, persistent=False)

    def cos_sin(self, length: int, device, dtype):
        t = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_full = torch.repeat_interleave(cos, 2, dim=-1)
    sin_full = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos_full + _rotate_half(x) * sin_full


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


class ModernAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 dropout: float, rope_theta: float, bias: bool,
                 lora_rank: int, lora_alpha: float, lora_dropout: float,
                 lora_freeze_base: bool, use_sdpa: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model deve essere divisibile per n_heads')
        if n_heads % n_kv_heads != 0:
            raise ValueError('n_heads deve essere divisibile per n_kv_heads')
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')
        Linear = lambda i, o: LoRALinear(i, o, bias=bias, rank=lora_rank,
                                         alpha=lora_alpha, dropout=lora_dropout,
                                         freeze_base=lora_freeze_base)
        self.q_proj = Linear(d_model, n_heads * self.head_dim)
        self.k_proj = Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = Linear(d_model, n_kv_heads * self.head_dim)
        self.o_proj = Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope.cos_sin(T, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)

        if T <= 64:
            with torch.no_grad():
                scores = (q[:1].float() @ k[:1].float().transpose(-2, -1)) / math.sqrt(self.head_dim)
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(mask, float('-inf'))
                self.last_attention = torch.softmax(scores, dim=-1).detach().cpu()

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int, bias: bool,
                 lora_rank: int, lora_alpha: float, lora_dropout: float,
                 lora_freeze_base: bool, dropout: float):
        super().__init__()
        Linear = lambda i, o: LoRALinear(i, o, bias=bias, rank=lora_rank,
                                         alpha=lora_alpha, dropout=lora_dropout,
                                         freeze_base=lora_freeze_base)
        self.gate = Linear(d_model, hidden)
        self.up = Linear(d_model, hidden)
        self.down = Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class ModernBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = ModernAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout,
            cfg.rope_theta, cfg.bias, cfg.lora_rank, cfg.lora_alpha,
            cfg.lora_dropout, cfg.lora_freeze_base, cfg.use_sdpa,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.mlp_ratio)
        self.mlp = SwiGLU(
            cfg.d_model, hidden, cfg.bias, cfg.lora_rank, cfg.lora_alpha,
            cfg.lora_dropout, cfg.lora_freeze_base, cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ModernTransformerLM(nn.Module):
    """Decoder-only autoregressive Transformer with modern building blocks.

    Features: RMSNorm, RoPE, SwiGLU, grouped-query attention, optional LoRA,
    PyTorch SDPA, gradient checkpointing and tied embeddings.
    """
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.context_length = cfg.context_length
        self.token_embedding = nn.Embedding(self.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([ModernBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, self.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.gradient_checkpointing = bool(cfg.gradient_checkpointing)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.context_length:
            raise ValueError(f'Sequenza {T} > context_length {self.context_length}')
        x = self.dropout(self.token_embedding(idx))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def attention_snapshot(self, max_tokens: int = 48):
        if not self.blocks:
            return None
        a = self.blocks[-1].attn.last_attention
        if a is None:
            return None
        return a[0, :, :max_tokens, :max_tokens].float().tolist()
