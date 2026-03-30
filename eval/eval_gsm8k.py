"""
GSM8K 数学推理评测
用法: python eval/eval_gsm8k.py --model_path models/sft_general/final.pt
"""
import os, sys, re, torch, argparse
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LMConfig()
    model = MiniMindV3(config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True), strict=False)
    print(f"Loaded model from {args.model_path}")

    enc = tiktoken.get_encoding("cl100k_base")

    samples = [
        {"q": "Tom has 5 apples. He buys 3 more. How many apples does he have?", "a": "8"},
        {"q": "A shirt costs \ and pants cost \. How much do they cost together?", "a": "50"},
        {"q": "There are 12 cookies. They are shared equally among 4 kids. How many cookies does each kid get?", "a": "3"},
        {"q": "John has 15 baseball cards. He gives 7 to his friend. How many cards does he have left?", "a": "8"},
        {"q": "A bakery made 24 muffins. They sold 18 muffins. How many muffins are left?", "a": "6"},
    ]

    correct = 0
    for s in samples:
        prompt = f"<|im_start|>user\n{s['q']}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        output_ids = generate(model, input_ids, max_new_tokens=128, temperature=0.3)
        output_text = enc.decode(output_ids[0].tolist())
        answer_part = output_text.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        nums = re.findall(r'-?\d+', answer_part)
        pred = nums[-1] if nums else ""
        ok = pred == s["a"]
        correct += ok
        print(f"Q: {s['q']}")
        print(f"A: {answer_part[:100]}")
        print(f"Pred: {pred}, GT: {s['a']} -> {'OK' if ok else 'FAIL'}")
        print("-" * 50)

    print(f"\nAccuracy: {correct}/{len(samples)}")


if __name__ == "__main__":
    main()
