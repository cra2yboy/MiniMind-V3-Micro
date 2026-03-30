"""MiniMind-V3-Micro 模型超参数配置"""
from dataclasses import dataclass


@dataclass
class LMConfig:
    # --- 基础维度 ---
    hidden_size: int = 512
    intermediate_size: int = 1408  # FFN 中间维度 (~2.75x hidden)
    num_hidden_layers: int = 12
    vocab_size: int = 100277        # tiktoken cl100k_base vocab size
    max_position_embeddings: int = 1024

    # --- MLA (Multi-head Latent Attention) ---
    num_attention_heads: int = 8
    num_key_value_heads: int = 2         # GQA: 每 4 个 Q head 共享 1 个 KV head
    kv_compression_dim: int = 64         # d_c: 低秩 KV 压缩维度
    rope_head_dim: int = 32              # 解耦 RoPE 的独立维度

    # --- Sparse MoE ---
    num_shared_experts: int = 1
    num_routed_experts: int = 8
    num_experts_per_tok: int = 2         # Top-K 路由
    moe_aux_loss_coeff: float = 0.01     # 辅助损失系数

    # --- MTP (Multi-Token Prediction) ---
    mtp_num_future: int = 2              # 同时预测 t+1, t+2
    mtp_loss_weight: float = 0.3         # MTP 辅助 loss 权重

    # --- 训练 ---
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_heads_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
