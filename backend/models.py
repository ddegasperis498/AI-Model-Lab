from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 activation: str = 'relu', dropout: float = 0.0,
                 task: str = 'regression'):
        super().__init__()
        self.task = task
        act_factory = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }[activation]
        modules: List[nn.Module] = []
        sizes = [input_size, *hidden_sizes, output_size]
        for i in range(len(sizes) - 1):
            modules.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                modules.append(act_factory())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)
        self.activation_stats: Dict[str, Dict[str, float]] = {}
        self._install_hooks()

    def _install_hooks(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                module.register_forward_hook(self._make_hook(name or module.__class__.__name__))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return
            with torch.no_grad():
                t = output.detach().float()
                self.activation_stats[name] = {
                    'mean': float(t.mean().cpu()),
                    'std': float(t.std(unbiased=False).cpu()) if t.numel() > 1 else 0.0,
                    'min': float(t.min().cpu()),
                    'max': float(t.max().cpu()),
                    'zero_fraction': float((t == 0).float().mean().cpu()),
                }
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, context_length: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model deve essere divisibile per n_heads')
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer('causal_mask', mask, persistent=False)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        self.last_attention = attn.detach()
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float,
                 dropout: float, context_length: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """Autoregressive byte-level Transformer language model.

    È un vero language model Transformer addestrabile con PyTorch/autograd.
    Il nome 'Tiny' indica la scala predefinita, non un meccanismo simulato.
    """
    vocab_size = 256

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 context_length: int = 128, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, tie_embeddings: bool = True):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout, context_length)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
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
        pos = torch.arange(T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def attention_snapshot(self, max_tokens: int = 64):
        if not self.blocks:
            return None
        attn = self.blocks[-1].attn.last_attention
        if attn is None:
            return None
        a = attn[0, :, :max_tokens, :max_tokens].float().cpu()
        return a.tolist()


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> List[int]:
        return list(text.encode('utf-8', errors='replace'))

    @staticmethod
    def decode(ids: List[int]) -> str:
        return bytes([int(i) % 256 for i in ids]).decode('utf-8', errors='replace')
