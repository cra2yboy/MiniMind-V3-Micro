"""
MiniMind-V3-Micro 指令微调 (SFT)
支持多文件混合训练，ChatML 格式对话，仅在 assistant 回复上计算 loss
用法: deepspeed --include localhost:0 scripts/train_sft.py --data_files data/sft/openhermes_150k.jsonl --pretrain_ckpt models/pretrain_final.pt --save_dir models/sft_general --epochs 3 --max_len 1024
"""
import os, sys, json, math
import torch
import deepspeed
import argparse
import tiktoken
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import MiniMindV3
from model.LMConfig import LMConfig

IGNORE_INDEX = -100


def format_conversation(conversations, enc):
    """将对话转换为 ChatML 格式的 token ids 和 labels，仅在 assistant 回复上计算 loss"""
    all_input_ids = []
    all_labels = []
    for turn in conversations:
        role = turn["role"]
        content = turn["content"]
        header = f"<|im_start|>{role}\n"
        body = content + "<|im_end|>\n"
        header_ids = enc.encode(header, allowed_special={"<|im_start|>", "<|im_end|>"})
        body_ids = enc.encode(body, allowed_special={"<|im_start|>", "<|im_end|>"})
        all_input_ids.extend(header_ids)
        all_labels.extend([IGNORE_INDEX] * len(header_ids))
        all_input_ids.extend(body_ids)
        if role == "assistant":
            all_labels.extend(body_ids)
        else:
            all_labels.extend([IGNORE_INDEX] * len(body_ids))
    return all_input_ids, all_labels


class SFTDataset(Dataset):
    """预分词所有数据，避免 DataLoader fork 时的 tiktoken 安全问题"""
    def __init__(self, data_file, max_len=1024):
        self.max_len = max_len
        enc = tiktoken.get_encoding("cl100k_base")
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                conversations = item["conversations"] if "conversations" in item else item.get("messages", [])
                input_ids, labels = format_conversation(conversations, enc)
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [0] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len
                self.data.append({
                    "input_ids": input_ids,
                    "labels": labels,
                })
        print(f"Loaded {len(self.data)} samples from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", type=str, required=True, help="逗号分隔的多个 JSONL 文件路径")
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="预训练 checkpoint 路径")
    parser.add_argument("--save_dir", type=str, required=True, help="保存目录")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=2000)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    config = LMConfig()
    model = MiniMindV3(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 加载预训练权重
    if os.path.exists(args.pretrain_ckpt):
        model.load_state_dict(torch.load(args.pretrain_ckpt, map_location="cpu", weights_only=True), strict=False)
        print(f"Loaded pretrained weights from {args.pretrain_ckpt}")

    # 支持多文件混合训练
    data_files = [f.strip() for f in args.data_files.split(",")]
    datasets = [SFTDataset(f, args.max_len) for f in data_files]
    concat_dataset = ConcatDataset(datasets)
    total_samples = len(concat_dataset)
    print(f"Total samples: {total_samples}")

    dataloader = DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # DeepSpeed 配置
    ds_config = os.path.join(os.path.dirname(__file__), "ds_config_sft.json")
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters(),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    steps_per_epoch = len(dataloader)

    for epoch in range(args.epochs):
        model_engine.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            loss, _ = model_engine(input_ids, labels=labels)
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"ckpt_step{global_step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

    # 保存最终模型
    final_path = os.path.join(args.save_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nSFT training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
