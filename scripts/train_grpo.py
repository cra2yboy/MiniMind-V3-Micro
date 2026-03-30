"""
MiniMind-V3-Micro GRPO (Group Relative Policy Optimization)
用法: python scripts/train_grpo.py
"""
import os, sys, json, copy
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import MiniMindV3
from model.LMConfig import LMConfig

RL_PROMPT_POOL = "<PATH_TO_RL_PROMPT_POOL>"
SAVE_DIR = "<DIR_TO_SAVE_CHECKPOINTS>"
SFT_CKPT = "<PATH_TO_SFT_CHECKPOINT>"


class PromptDataset:
    def __init__(self, path: str):
        self.prompts = []
        if os.path.exists(path) and not path.startswith("<"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    self.prompts.append(json.loads(line.strip())["prompt"])
        else:
            self.prompts = ["What is 2+3?", "Explain gravity.", "Write a poem."] * 100

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def reward_function(prompt: str, response: str) -> float:
    """通用 reward 框架 - 用户可替换"""
    score = 0.0
    if len(response.strip()) > 0:
        score += 0.5
    if 10 < len(response) < 500:
        score += 0.3
    if any(p in response for p in [".", "!", "?"]):
        score += 0.2
    return score


def simple_tokenize(text: str, max_len: int = 256) -> torch.Tensor:
    """占位 tokenizer - 实际应使用 sentencepiece/tiktoken"""
    tokens = [ord(c) % 32000 for c in text[:max_len]]
    return torch.tensor(tokens, dtype=torch.long)


def compute_log_probs(model, input_ids):
    logits = model(input_ids)
    if isinstance(logits, tuple):
        logits = logits[1] if len(logits) > 1 else logits[0]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    return log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(dim=-1)


class GRPO:
    """Group Relative Policy Optimization"""
    def __init__(self, model, ref_model, group_size=8, kl_coeff=0.1, clip_eps=0.2, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    @torch.no_grad()
    def sample_group(self, prompt_ids, max_new=128):
        self.model.eval()
        responses = []
        for _ in range(self.group_size):
            out = self.model.generate(prompt_ids.unsqueeze(0), max_new_tokens=max_new, temperature=0.8, top_p=0.95)
            responses.append(out[0])
        return responses

    def compute_advantages(self, rewards):
        r = torch.tensor(rewards, dtype=torch.float32)
        return (r - r.mean()) / (r.std() + 1e-8)

    def step(self, prompt, prompt_ids, device):
        prompt_ids = prompt_ids.to(device)
        responses = self.sample_group(prompt_ids)

        rewards = []
        for resp_ids in responses:
            resp_text = "".join([chr(t % 128) for t in resp_ids[len(prompt_ids):].tolist()])
            rewards.append(reward_function(prompt, resp_text))

        advantages = self.compute_advantages(rewards).to(device)
        self.model.train()
        total_loss = 0.0

        for i, resp_ids in enumerate(responses):
            resp_ids = resp_ids.unsqueeze(0).to(device)
            curr_lp = compute_log_probs(self.model, resp_ids)
            with torch.no_grad():
                ref_lp = compute_log_probs(self.ref_model, resp_ids)
            ratio = torch.exp(curr_lp - ref_lp)
            adv = advantages[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2)
            kl = (ref_lp - curr_lp).mean()
            total_loss += policy_loss + self.kl_coeff * kl

        total_loss = total_loss / self.group_size
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item(), sum(rewards) / len(rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=8)
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

    grpo = GRPO(model, ref_model, group_size=args.group_size, lr=args.lr)
    dataset = PromptDataset(RL_PROMPT_POOL)
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(range(len(dataset)), desc=f"GRPO Epoch {epoch+1}")
        for idx in pbar:
            prompt = dataset[idx]
            prompt_ids = simple_tokenize(prompt).to(device)
            loss, avg_r = grpo.step(prompt, prompt_ids, device)
            pbar.set_postfix(loss=f"{loss:.4f}", reward=f"{avg_r:.3f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "grpo_final.pt"))
    print("GRPO training complete.")


if __name__ == "__main__":
    main()
