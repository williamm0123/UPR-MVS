# UPR-MVS 快速启动指南

## 🚀 三条命令开始训练

### 1️⃣ 本地开发调试
```bash
bash scripts/train_local.sh
```
- ✅ 小模型，快速验证
- ✅ 5 epochs/阶段，约 30 分钟完成
- ✅ 输出到 `saved/local_training/`

### 2️⃣ 服务器单卡训练（A100 80GB）
```bash
bash scripts/train_server_single.sh
```
- ✅ 完整三阶段自动训练
- ✅ 20 epochs/阶段，约 7 天完成
- ✅ 输出到 `saved/server_single_gpu/`

### 3️⃣ 服务器多卡训练
```bash
bash scripts/train_server_multigpu.sh 4
```
- ✅ 4 卡 DDP 并行
- ✅ 速度提升 ~3.5×
- ✅ 输出到 `saved/server_multi_gpu/`

---

## 📁 配置文件（仅 2 个）

| 文件 | 用途 | 位置 |
|------|------|------|
| `configs/local_training.config` | 本地调试配置 | 小模型、低分辨率 |
| `configs/server_training.config` | 服务器训练配置 | 大模型、高分辨率 |

---

## 🎯 常用操作

### 只训练某个阶段
```bash
# 只训练 Stage A
bash scripts/train_server_single.sh --stage stage_a

# 只训练 Stage B
bash scripts/train_server_single.sh --stage stage_b

# 只训练 Stage C
bash scripts/train_server_single.sh --stage stage_c
```

### 从 checkpoint 恢复
```bash
# 从指定 checkpoint 恢复
bash scripts/train_server_single.sh \
  --resume saved/server_single_gpu/stage_a/checkpoints/best.pth
```

### 自定义输出目录
```bash
bash scripts/train_server_single.sh --work_dir saved/my_experiment
```

---

## 📊 输出目录结构

```
saved/
├── server_single_gpu/          # 服务器单卡训练结果
│   ├── stage_a/                # Stage A 输出
│   │   ├── checkpoints/        # 模型权重
│   │   │   ├── best.pth        # 最佳 checkpoint
│   │   │   └── latest.pth      # 最新 checkpoint
│   │   └── tensorboard/        # TensorBoard 日志
│   ├── stage_b/                # Stage B 输出
│   └── stage_c/                # Stage C 输出
│
└── local_training/             # 本地训练结果
    └── ...
```

---

## 🔍 监控训练

### 查看实时日志
```bash
tail -f saved/server_single_gpu/stage_a/log.txt
```

### 启动 TensorBoard
```bash
tensorboard --logdir saved/server_single_gpu/tensorboard --host 0.0.0.0 --port 6006
```

### 监控 GPU
```bash
watch -n 1 nvidia-smi
```

---

## ⚠️ 故障排查

### OOM（显存不足）
```bash
# 编辑配置文件，降低 batch size
vim configs/server_training.config

# 修改：
train:
  batch_size_per_gpu: 2      # 从 4 降到 2
  grad_accum_steps: 16       # 从 8 增到 16
```

### 训练中断后继续
```bash
# 直接使用 --resume 参数
bash scripts/train_server_single.sh \
  --resume saved/server_single_gpu/stage_a/checkpoints/latest.pth
```

---

## 📚 详细文档

- 📘 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - 完整训练指南
- 📘 [`TRAINING_A100_80GB.md`](TRAINING_A100_80GB.md) - A100 优化说明
- 📘 [`OOM_TROUBLESHOOTING.md`](OOM_TROUBLESHOOTING.md) - OOM 排错指南

---

**就这么简单！** 🎉
