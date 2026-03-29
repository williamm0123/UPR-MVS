# UPR-MVS 训练指南

## 📋 快速开始

### 本地开发调试
```bash
bash scripts/train_local.sh
```

### 服务器单卡训练（A100 80GB）
```bash
bash scripts/train_server_single.sh
```

### 服务器多卡训练
```bash
bash scripts/train_server_multigpu.sh <NUM_GPUS>
# 例如：bash scripts/train_server_multigpu.sh 4
```

---

## 🎯 自动分阶段训练

本项目的训练脚本支持**自动三阶段训练**，只需指定一个配置文件即可自动完成所有阶段：

### 训练流程
1. **Stage A**: Coarse Depth Pretraining（粗深度预训练）
2. **Stage B**: Point Refiner Training（点云优化器训练）
3. **Stage C**: Joint Fine-tuning（联合微调）

每个阶段会自动：
- 加载上一阶段的最佳 checkpoint
- 应用对应的超参数配置
- 保存独立的训练结果

### 输出目录结构
```
saved/
├── server_single_gpu/
│   ├── stage_a/
│   │   ├── checkpoints/
│   │   │   ├── best.pth
│   │   │   └── latest.pth
│   │   └── tensorboard/
│   ├── stage_b/
│   │   ├── checkpoints/
│   │   └── tensorboard/
│   └── stage_c/
│       ├── checkpoints/
│       └── tensorboard/
└── local_training/
    └── ...
```

---

## 📁 配置文件说明

### 两个主要配置文件

#### 1. `configs/server_training.config`
用于服务器 A100 80GB 单卡训练的配置：
- ✅ 完整分辨率（1024×1280）
- ✅ 大模型（DINOv3 ViT-B/16）
- ✅ 三阶段优化参数
- ✅ Gradient Checkpointing 启用
- ✅ 梯度累积策略

#### 2. `configs/local_training.config`
用于本地开发调试的配置：
- ✅ 降低分辨率（512×640）
- ✅ 小模型（DINOv3 ViT-B/14）
- ✅ 减少 epoch 数（5 个/阶段）
- ✅ 更小的 batch size
- ✅ 更快的训练速度

---

## 🚀 启动脚本详解

### 1. 本地训练脚本 (`scripts/train_local.sh`)

**用途**: 本地开发、调试代码

**特点**:
- 使用较小的模型和分辨率
- 快速验证代码正确性
- 默认 5 个 epoch 快速跑通流程

**使用方法**:
```bash
# 基本用法
bash scripts/train_local.sh

# 指定配置
bash scripts/train_local.sh --config configs/local_training.config

# 只运行特定阶段
bash scripts/train_local.sh --stage stage_a

# 从 checkpoint 恢复
bash scripts/train_local.sh --resume saved/local_training/stage_a/checkpoints/best.pth

# 自定义输出目录
bash scripts/train_local.sh --work_dir saved/my_debug_run
```

### 2. 服务器单卡脚本 (`scripts/train_server_single.sh`)

**用途**: 服务器 A100 80GB 单卡完整训练

**特点**:
- 优化显存使用（expandable_segments, memory_fraction）
- 自动三阶段训练
- 完整的 20 epochs/阶段

**使用方法**:
```bash
# 基本用法（推荐）
bash scripts/train_server_single.sh

# 只运行特定阶段
bash scripts/train_server_single.sh --stage stage_b

# 从指定 checkpoint 恢复第一阶段
bash scripts/train_server_single.sh --resume /path/to/checkpoint.pth

# 使用自定义配置
bash scripts/train_server_single.sh --config configs/my_custom.config
```

### 3. 服务器多卡脚本 (`scripts/train_server_multigpu.sh`)

**用途**: 服务器多卡 DDP 分布式训练

**特点**:
- 支持 2-8 卡并行
- 自动数据并行（DDP）
- 加速训练过程

**使用方法**:
```bash
# 使用 4 卡训练
bash scripts/train_server_multigpu.sh 4

# 使用 8 卡训练
bash scripts/train_server_multigpu.sh 8

# 指定配置和 GPU 数量
bash scripts/train_server_multigpu.sh configs/server_training.config saved/multi_gpu 8

# 多卡 + 特定阶段
bash scripts/train_server_multigpu.sh 4 --stage stage_a
```

---

## ⚙️ 命令行参数说明

### 通用参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--config` | 配置文件路径 | 根据脚本不同而不同 | `configs/server_training.config` |
| `--work_dir` | 输出目录 | `saved/...` | `saved/my_experiment` |
| `--stage` | 训练阶段 | `auto`（自动三阶段） | `stage_a`, `stage_b`, `stage_c` |
| `--resume` | 恢复的 checkpoint 路径 | 空 | `saved/stage_a/checkpoints/best.pth` |

### 多卡脚本特有参数

| 参数位置 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| 第 1 个参数 | GPU 数量 | 4 | `8` |
| 第 2 个参数 | 配置文件 | `configs/server_training.config` | `configs/my_config.config` |
| 第 3 个参数 | 输出目录 | `saved/server_multi_gpu` | `saved/ddp_run` |

---

## 🔍 监控与调试

### 查看训练日志
```bash
tail -f saved/server_single_gpu/stage_a/log.txt
```

### 启动 TensorBoard
```bash
tensorboard --logdir saved/server_single_gpu/tensorboard --host 0.0.0.0 --port 6006
```

### 监控 GPU 状态
```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细显存信息
nvidia-smi dmon -s m
```

---

## 📊 训练阶段详细说明

### Stage A: Coarse Depth Pretraining
- **目的**: 训练粗深度估计网络
- **训练模块**: CostVolumeTransformer (cvt)
- **冻结模块**: Backbone, Point Refiner
- **Loss**: 仅 coarse depth loss
- **监控指标**: depth_abs_error
- **学习率**: lr_backbone=0.00005, lr_coarse=0.0001
- **Batch Size**: 4 (等效 32 via grad_accum=8)
- **Epochs**: 20

### Stage B: Point Refiner Training
- **目的**: 训练点云优化器
- **训练模块**: Point Refiner, Densifier
- **冻结模块**: Backbone, CVT
- **Loss**: Chamfer + Uncertainty + Feature
- **监控指标**: point_abs_error
- **学习率**: lr_point=0.0002
- **Batch Size**: 4 (等效 24 via grad_accum=6)
- **Epochs**: 20

### Stage C: Joint Fine-tuning
- **目的**: 联合微调所有模块
- **训练模块**: 全部
- **Loss**: 所有 loss 加权组合
- **监控指标**: point_abs_error
- **学习率**: lr_backbone=0.000025, lr_coarse=0.00001, lr_point=0.0002
- **Batch Size**: 2 (等效 16 via grad_accum=8)
- **Epochs**: 20

---

## ⚠️ 常见问题

### Q1: 如何跳过某个阶段？
```bash
# 只运行 Stage B 和 C（跳过 Stage A）
bash scripts/train_server_single.sh --stage stage_b

# 然后手动运行 Stage C
bash scripts/train_server_single.sh --stage stage_c \
  --resume saved/server_single_gpu/stage_b/checkpoints/best.pth
```

### Q2: 如何在已有 checkpoint 基础上继续训练？
```bash
# 从 Stage A 的 checkpoint 继续
bash scripts/train_server_single.sh \
  --resume saved/server_single_gpu/stage_a/checkpoints/best.pth
```

### Q3: 如何修改训练超参数？
直接编辑配置文件：
```bash
vim configs/server_training.config
# 或
vim configs/local_training.config
```

### Q4: 显存不足怎么办？
在配置文件中调整：
```yaml
train:
  batch_size_per_gpu: 2      # 降低 batch size
  grad_accum_steps: 16       # 增加梯度累积
```

---

## 📝 自定义配置示例

### 创建自己的配置文件
```bash
cp configs/server_training.config configs/my_experiment.config
vim configs/my_experiment.config
```

### 修改关键参数
```yaml
# 增加 epoch 数
training_stages:
  stage_a:
    epochs: 30              # 从 20 增加到 30
  
# 调整学习率
optim:
  lr_backbone: 0.0001       # 增大学习率
  
# 修改 loss 权重
loss:
  chamfer_weight: 2.0       # 增加 Chamfer loss 权重
```

### 运行自定义配置
```bash
bash scripts/train_server_single.sh \
  --config configs/my_experiment.config \
  --work_dir saved/my_experiment
```

---

## 🎯 性能优化建议

### 1. 显存优化
- ✅ 启用 `use_checkpoint: true`
- ✅ 使用梯度累积（grad_accum_steps）
- ✅ 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 2. 训练速度优化
- ✅ 使用多卡 DDP 训练
- ✅ 增加 num_workers（但不超过 CPU 核心数）
- ✅ 使用 bfloat16 混合精度（amp_dtype: bf16）

### 3. 收敛优化
- ✅ 充分的 warmup（至少 2-3 epochs）
- ✅ 合适的学习率 scaling
- ✅ gradient clipping（grad_clip=1.0）

---

## 📚 相关文档

- `TRAINING_A100_80GB.md` - A100 80GB 训练详细指南
- `OOM_TROUBLESHOOTING.md` - OOM 问题排查
- `QUICK_REFERENCE.md` - 快速参考卡片

---

**祝训练顺利！** 🚀
