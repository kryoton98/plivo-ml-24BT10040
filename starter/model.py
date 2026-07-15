"""A compact Llama/Mistral-style decoder-only LM in pure PyTorch (CPU-friendly).

Design notes (parameter budget = 2,000,000 hard cap):
  vocab_size = 1024, n_embd = 256, n_layer = 6, n_head = 8 (head_dim = 32),
  MQA (1 KV head), SwiGLU FFN with ffn_hidden = 180, RMSNorm, RoPE,
  tied embedding / lm_head, no biases anywhere.

  embedding            1024*256                    =   262,144
  per block:
    wq                  256*256                    =    65,536
    wk                  256*32                     =     8,192
    wv                  256*32                     =     8,192
    wo                  256*256                    =    65,536
    w_gate/w_up/w_down   3*256*180                 =   138,240
    2x RMSNorm           2*256                     =       512
                                                    ---------
                                                       286,208  x6 = 1,717,248
  final RMSNorm         256                        =       256
  lm_head               tied -> 0
                                                    =========
  TOTAL                                              1,979,648
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8          # query heads; K/V heads = 1 (MQA)
    n_embd: int = 256
    ffn_hidden: int = 180    # SwiGLU inner dim, tuned to the budget
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_weights: bool = True  # non-negotiable here


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


# --------------------------------------------------------------------------- #
# Rotary positional embeddings
# --------------------------------------------------------------------------- #
def build_rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                  # (T, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim); cos/sin: (T, head_dim/2)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# --------------------------------------------------------------------------- #
# Multi-Query Attention
# --------------------------------------------------------------------------- #
class MQAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.wq = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.head_dim, bias=False)   # 1 KV head
        self.wv = nn.Linear(cfg.n_embd, self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.p_drop = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, 1, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # broadcast the single KV head across all query heads
        k = k.expand(B, self.n_head, T, self.head_dim)
        v = v.expand(B, self.n_head, T, self.head_dim)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.p_drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.wo(y))


# --------------------------------------------------------------------------- #
# SwiGLU FFN
# --------------------------------------------------------------------------- #
class SwiGLU(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        h = cfg.ffn_hidden
        self.w_gate = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_down = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = MQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GPT(nn.Module):
    def __init__(self, cfg: Config = None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        head_dim = cfg.n_embd // cfg.n_head

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying (this is why lm_head contributes 0 params)
        self.lm_head.weight = self.wte.weight

        cos, sin = build_rope_cache(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        # scaled init on residual-output projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.count_parameters()

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def count_parameters(self, verbose: bool = True) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if p.requires_grad and id(p) not in seen:   # tied weights counted once
                seen.add(id(p))
                total += p.numel()
        if verbose:
            print(f"[GPT] total trainable parameters: {total:,}")
        return total

    # alias for baseline compatibility
    def n_params(self):
        return self.count_parameters(verbose=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        x = self.drop(self.wte(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


if __name__ == "__main__":
    model = GPT(Config())
    n = model.count_parameters(verbose=False)
    assert 1_900_000 <= n <= 1_999_999, n
    x = torch.randint(0, 1024, (2, 128))
    logits, loss = model(x, targets=x)
    print(logits.shape, loss.item())
    print(model(x)[1])