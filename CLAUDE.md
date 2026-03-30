# MiniMind-V3-Micro 项目

## 常用命令

### 训练相关
- 启动 SFT 训练：conda run --no-capture-output -n minimind env PYTHONUNBUFFERED=1 deepspeed --include localhost:0 scripts/train_sft.py --data_files data/sft/openhermes_150k.jsonl --pretrain_ckpt models/pretrain_final.pt --save_dir models/sft_general --epochs 3 --max_len 1024
- 查看实时日志：tail -f /mnt/data/xfh/claude/sft_train.log
- 查看 GPU 状态：nvidia-smi
- 查看训练进程：ps aux | grep deepspeed
- 终止训练进程：pkill -f train_sft

### Git 相关
- 推送代码：cd /mnt/data/xfh/MiniMind-V3-Micro && git add . && git commit -m "描述" && git push
- 拉取代码：cd /mnt/data/xfh/MiniMind-V3-Micro && git pull

## 已知坑
- DeepSpeed 多 GPU NCCL 不兼容，务必用单 GPU：--include localhost:0

## Plan Mode
复杂任务先 /plan 再执行，出问题立刻切回 plan mode 重新分析

## 子 Agent
需要更多算力时加 use subagents，保持主 agent 上下文干净

