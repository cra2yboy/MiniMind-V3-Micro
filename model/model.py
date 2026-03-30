"""
MiniMind-V3-Micro 核心架构
- MLA (Multi-head Latent Attention) with 低秩 KV 压缩 + 解耦 RoPE
- GQA (Grouped Query Attention)
- Flash Attention 集成
- Sparse MoE
- MTP (Multi-Token Prediction)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.LMConfig import LMConfig
from model.moe import SparseMoE

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("[WARN] flash-attn not found, falling back to PyTorch SDPA.")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.weight


def precompute_rope(dim: int, max_len: int, theta: float = 10000.0):
    """预计算 RoPE 的 cos/sin 缓存"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    x: (..., seq_len, head_dim)  head_dim must be even
    cos, sin: (max_len, head_dim//2)
    """
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(x.device)
    sin = sin[:seq_len].to(x.device)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention with:
    - 低秩 KV 压缩 (d_c)
    - 解耦 RoPE (独立的 rope_head_dim 维度用于位置编码)
    - GQA (num_key_value_heads < num_attention_heads)
    - Flash Attention 支持
    """
    def __init__(self, config: LMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.kv_comp_dim = config.kv_compression_dim
        self.rope_dim = config.rope_head_dim

        # Q 投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # KV 压缩: hidden -> d_c (低秩瓶颈)
        self.kv_compress = nn.Linear(self.hidden_size, self.kv_comp_dim, bias=False)
        self.k_decompress = nn.Linear(self.kv_comp_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_decompress = nn.Linear(self.kv_comp_dim, self.num_kv_heads * self.head_dim, bias=False)

        # 解耦 RoPE 投影
        self.q_rope_proj = nn.Linear(self.hidden_size, self.num_heads * self.rope_dim, bias=False)
        self.k_rope_proj = nn.Linear(self.kv_comp_dim, self.num_kv_heads * self.rope_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(self.num_heads * (self.head_dim + self.rope_dim), self.hidden_size, bias=False)

        # RoPE 预计算缓存
        cos, sin = precompute_rope(self.rope_dim, config.max_position_embeddings)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        kv_ratio = self.num_heads // self.num_kv_heads

        # Q
        q_nope = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        q_rope = self.q_rope_proj(hidden_states).view(B, S, self.num_heads, self.rope_dim)
        q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)

        # KV 压缩
        c = self.kv_compress(hidden_states)
        k_nope = self.k_decompress(c).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_decompress(c).view(B, S, self.num_kv_heads, self.head_dim)
        k_rope = self.k_rope_proj(c).view(B, S, self.num_kv_heads, self.rope_dim)
        k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)

        # 拼接 nope + rope
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        v_padded = F.pad(v, (0, self.rope_dim))

        # GQA 扩展
        if kv_ratio > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, kv_ratio, -1).reshape(B, S, self.num_heads, -1)
            v_padded = v_padded.unsqueeze(3).expand(-1, -1, -1, kv_ratio, -1).reshape(B, S, self.num_heads, -1)

        full_head_dim = self.head_dim + self.rope_dim

        if HAS_FLASH_ATTN and q.is_cuda:
            q_fa = q.to(torch.bfloat16)
            k_fa = k.to(torch.bfloat16)
            v_fa = v_padded.to(torch.bfloat16)
            attn_out = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
            attn_out = attn_out.to(hidden_states.dtype)
        else:
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v_padded.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True).transpose(1, 2)

        attn_out = attn_out.reshape(B, S, self.num_heads * full_head_dim)
        return self.o_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, config: LMConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = MLAAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SparseMoE(config)

    def forward(self, x: torch.Tensor, attention_mask=None):
        h = x + self.attn(self.attn_norm(x), attention_mask)
        ffn_out, aux_loss = self.ffn(self.ffn_norm(h))
        return h + ffn_out, aux_loss


class MTPHead(nn.Module):
    """Multi-Token Prediction: 同时预测 t+1 和 t+2"""
    def __init__(self, config: LMConfig):
        super().__init__()
        self.num_future = config.mtp_num_future
        self.loss_weight = config.mtp_loss_weight
        self.future_projs = nn.ModuleList([
            nn.Sequential(
                RMSNorm(config.hidden_size, config.rms_norm_eps),
                nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            )
            for _ in range(self.num_future)
        ])

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Linear, labels: torch.Tensor):
        total_loss = 0.0
        count = 0
        for i, proj in enumerate(self.future_projs):
            shift = i + 1
            if labels.size(1) <= shift:
                continue
            projected = proj(hidden_states[:, :-shift, :])
            logits = lm_head(projected)
            target = labels[:, shift:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=-100,
            )
            total_loss += loss
            count += 1
        if count > 0:
            return self.loss_weight * total_loss / count
        return torch.tensor(0.0, device=hidden_states.device)


class MiniMindV3(nn.Module):
    """MiniMind-V3-Micro: DeepSeek-V3 极简复现"""
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp = MTPHead(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(self, input_ids, labels=None, attention_mask=None):
        h = self.embed_tokens(input_ids)
        total_aux_loss = 0.0
        for layer in self.layers:
            h, aux_loss = layer(h, attention_mask)
            total_aux_loss += aux_loss
        h = self.norm(h)
        logits = self.lm_head(h)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            mtp_loss = self.mtp(h, self.lm_head, labels)
            return main_loss + mtp_loss + total_aux_loss, logits

        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_position_embeddings:]
            logits = self(idx_cond)
            if isinstance(logits, tuple):
                logits = logits[1]
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

        return input_ids
