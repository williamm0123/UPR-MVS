# UPR-MVS - 简化的训练流程

## 🚀 快速开始（3 条命令）

### 1️⃣ 本地开发调试
```bash
bash scripts/train_local.sh
```

### 2️⃣ 服务器单卡训练（A100 80GB）
```bash
bash scripts/train_server_single.sh
```

### 3️⃣ 服务器多卡训练
```bash
bash scripts/train_server_multigpu.sh 4
```

---

## 📁 项目结构简化

### 配置文件（仅 2 个）
```
configs/
├── local_training.config      # 本地开发调试配置
└── server_training.config     # 服务器 A100 80GB 配置
```

### 启动脚本（仅 3 个）
```
scripts/
├── train_local.sh             # 本地训练脚本
├── train_server_single.sh     # 服务器单卡脚本
└── train_server_multigpu.sh   # 服务器多卡脚本
```

### 输出目录（统一）
```
saved/
├── local_training/            # 本地训练输出
├── server_single_gpu/         # 单卡训练输出
│   ├── stage_a/
│   ├── stage_b/
│   └── stage_c/
└── server_multi_gpu/          # 多卡训练输出
```

---

## 🎯 自动三阶段训练

只需指定一个配置文件，即可自动完成三个阶段：

1. **Stage A**: Coarse Depth Pretraining
2. **Stage B**: Point Refiner Training  
3. **Stage C**: Joint Fine-tuning

每个阶段自动：
- ✅ 加载上一阶段的最佳 checkpoint
- ✅ 应用对应的超参数
- ✅ 保存独立的训练结果

---

## 📖 详细文档

- 📘 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - 完整训练指南
- 📘 [`QUICKSTART_SIMPLE.md`](QUICKSTART_SIMPLE.md) - 快速启动指南
- 📘 [`CONFIG_INTEGRATION_COMPLETE.md`](CONFIG_INTEGRATION_COMPLETE.md) - 配置整合报告

---

## ⚙️ 常用操作

### 只训练某个阶段
```bash
bash scripts/train_server_single.sh --stage stage_a
```

### 从 checkpoint 恢复
```bash
bash scripts/train_server_single.sh \
  --resume saved/server_single_gpu/stage_a/checkpoints/best.pth
```

### 自定义输出目录
```bash
bash scripts/train_server_single.sh --work_dir saved/my_experiment
```

### 查看 TensorBoard
```bash
tensorboard --logdir saved/server_single_gpu/tensorboard --host 0.0.0.0
```

---

**就这么简单！** 🎉
