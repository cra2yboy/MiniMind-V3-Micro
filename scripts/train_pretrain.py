"""
MiniMind-V3-Micro 从零预训练脚本 (DeepSpeed ZeRO-2)
用法: deepspeed --num_gpus=2 scripts/train_pretrain.py
"""
import os, sys, glob
import torch
import deepspeed
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import MiniMindV3
from model.LMConfig import LMConfig

PRETRAIN_DATA_DIR = "<ABS_PATH_TO_YOUR_PRETRAIN_DATASET_DIR>"
SAVE_DIR = "<DIR_TO_SAVE_CHECKPOINTS>"


class PretrainDataset(Dataset):
    """加载预分词的 .bin 文件 (np.uint16 token ids)"""
    def __init__(self, data_dir: str, max_len: int = 2048):
        self.max_len = max_len
        self.data = []
        bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not bin_files:
            print(f"[WARN] No .bin files in {data_dir}. Using dummy data.")
            self.data = [torch.randint(0, 32000, (max_len,)) for _ in range(1000)]
            return
        for f in bin_files:
            tokens = torch.from_numpy(np.fromfile(f, dtype=np.uint16).astype(np.int64))
            for i in range(0, len(tokens) - max_len, max_len):
                self.data.append(tokens[i:i + max_len])
        print(f"Loaded {len(self.data)} samples from {len(bin_files)} files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return {"input_ids": x, "labels": x.clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    config = LMConfig()
    model = MiniMindV3(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    dataset = PretrainDataset(PRETRAIN_DATA_DIR, config.max_position_embeddings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    ds_config = os.path.join(os.path.dirname(__file__), "ds_config.json")
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, config=ds_config, model_parameters=model.parameters(),
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model_engine.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=(args.local_rank > 0))
        for batch in pbar:
            input_ids = batch["input_ids"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            loss, _ = model_engine(input_ids, labels=labels)
            model_engine.backward(loss)
            model_engine.step()
            global_step += 1
            if args.local_rank <= 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", step=global_step)
            if global_step % args.save_every == 0 and args.local_rank <= 0:
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"pretrain_step{global_step}.pt"))

    if args.local_rank <= 0:
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pretrain_final.pt"))
        print("Pretraining complete.")


if __name__ == "__main__":
    main()
