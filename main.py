"""
MiniMind-V3-Micro 推理入口
用法: python main.py --ckpt models/sft_general/final.pt --prompt "你好"
      python main.py --ckpt models/sft_general/final.pt --interactive
"""
import argparse
import torch
import tiktoken
from model.model import MiniMindV3
from model.LMConfig import LMConfig


@torch.no_grad()
def generate(model, input_ids, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -model.config.max_position_embeddings:]
        logits = model(idx_cond)
        if isinstance(logits, tuple):
            logits = logits[1]
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
    return input_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="模型 checkpoint 路径")
    parser.add_argument("--prompt", type=str, default="Hello!")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LMConfig()
    model = MiniMindV3(config).to(device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True), strict=False)
        print(f"Loaded: {args.ckpt}")

    model.eval()
    print(f"Device: {device} | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    enc = tiktoken.get_encoding("cl100k_base")

    def generate_response(text):
        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        output_ids = generate(model, input_ids, args.max_tokens, args.temperature, args.top_p, args.top_k)
        response = enc.decode(output_ids[0].tolist())
        answer = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        return answer

    if args.interactive:
        print("\n=== MiniMind-V3-Micro Interactive ===\nType 'quit' to exit.\n")
        while True:
            inp = input("You: ").strip()
            if inp.lower() in ("quit", "exit", "q"):
                break
            print(f"AI: {generate_response(inp)}\n")
    else:
        print(f"Prompt: {args.prompt}")
        print(f"Response: {generate_response(args.prompt)}")


if __name__ == "__main__":
    main()
