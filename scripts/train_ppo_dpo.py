"""
MiniMind-V3-Micro PPO & DPO 最小可行化原生实现
用法: python scripts/train_ppo_dpo.py --method ppo|dpo
"""
import os, sys, copy
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import MiniMindV3
from model.LMConfig import LMConfig

SAVE_DIR = "<DIR_TO_SAVE_CHECKPOINTS>"
SFT_CKPT = "<PATH_TO_SFT_CHECKPOINT>"


def compute_log_probs(model, input_ids):
    logits = model(input_ids)
    if isinstance(logits, tuple):
        logits = logits[1] if len(logits) > 1 else logits[0]
    lp = F.log_softmax(logits[:, :-1, :], dim=-1)
    return lp.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


class PPOTrainer:
    def __init__(self, model, ref_model, lr=1e-6, clip_eps=0.2, kl_coeff=0.1):
        self.model = model
        self.ref_model = ref_model
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, input_ids, rewards, old_log_probs):
        self.model.train()
        token_lp = compute_log_probs(self.model, input_ids)
        with torch.no_grad():
            ref_lp = compute_log_probs(self.ref_model, input_ids)
        ratio = torch.exp(token_lp - old_log_probs)
        adv = rewards.unsqueeze(-1).expand_as(ratio)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        loss = -torch.min(surr1, surr2).mean() + self.kl_coeff * (ref_lp - token_lp).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()


class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, chosen_ids, rejected_ids):
        self.model.train()
        pi_w = compute_log_probs(self.model, chosen_ids).sum(-1)
        pi_l = compute_log_probs(self.model, rejected_ids).sum(-1)
        with torch.no_grad():
            ref_w = compute_log_probs(self.ref_model, chosen_ids).sum(-1)
            ref_l = compute_log_probs(self.ref_model, rejected_ids).sum(-1)
        logits = self.beta * ((pi_w - ref_w) - (pi_l - ref_l))
        loss = -F.logsigmoid(logits).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), (logits > 0).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ppo", "dpo"], required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LMConfig()
    model = MiniMindV3(config).to(device)
    if os.path.exists(SFT_CKPT):
        model.load_state_dict(torch.load(SFT_CKPT, map_location="cpu", weights_only=True), strict=False)
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.method == "ppo":
        trainer = PPOTrainer(model, ref_model, lr=args.lr)
        for step in (pbar := tqdm(range(args.steps), desc="PPO")):
            ids = torch.randint(0, config.vocab_size, (2, 128)).to(device)
            rewards = torch.randn(2).to(device)
            with torch.no_grad():
                old_lp = compute_log_probs(model, ids)
            loss = trainer.step(ids, rewards, old_lp)
            pbar.set_postfix(loss=f"{loss:.4f}")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ppo_final.pt"))
    else:
        trainer = DPOTrainer(model, ref_model, lr=args.lr)
        for step in (pbar := tqdm(range(args.steps), desc="DPO")):
            chosen = torch.randint(0, config.vocab_size, (2, 128)).to(device)
            rejected = torch.randint(0, config.vocab_size, (2, 128)).to(device)
            loss, acc = trainer.step(chosen, rejected)
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "dpo_final.pt"))

    print(f"{args.method.upper()} training complete.")


if __name__ == "__main__":
    main()
