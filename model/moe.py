"""Sparse MoE: 1 Shared Expert + 8 Routed Experts, Top-2 路由 + Aux-Loss"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """单个 Expert: SwiGLU FFN"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_routed = config.num_routed_experts
        self.top_k = config.num_experts_per_tok
        self.aux_loss_coeff = config.moe_aux_loss_coeff

        # Shared expert(s)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            ExpertFFN(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_routed_experts)
        ])

        # Router
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B, S, D)
        Returns: (B, S, D), aux_loss
        """
        B, S, D = x.shape
        h = x.view(-1, D)  # (B*S, D)

        # Shared expert 输出
        shared_out = sum(expert(h) for expert in self.shared_experts)

        # 路由逻辑
        logits = self.gate(h)                          # (B*S, num_routed)
        scores = F.softmax(logits, dim=-1)             # (B*S, num_routed)
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)  # (B*S, K)

        # 归一化 top-k 权重
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # 分发到 expert
        routed_out = torch.zeros_like(h)
        for k in range(self.top_k):
            idx = topk_idx[:, k]          # (B*S,)
            w = topk_weights[:, k]        # (B*S,)
            for e_id in range(self.num_routed):
                mask = (idx == e_id)
                if mask.any():
                    routed_out[mask] += w[mask].unsqueeze(-1) * self.routed_experts[e_id](h[mask])

        output = (shared_out + routed_out).view(B, S, D)

        # Aux loss: 鼓励负载均衡
        f = torch.zeros(self.num_routed, device=x.device)
        for k in range(self.top_k):
            for e_id in range(self.num_routed):
                f[e_id] += (topk_idx[:, k] == e_id).float().sum()
        f = f / (B * S * self.top_k + 1e-9)
        P = scores.mean(dim=0)
        aux_loss = self.aux_loss_coeff * self.num_routed * (f * P).sum()

        return output, aux_loss
