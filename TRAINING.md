# UPR-MVS 训练指南

## 📋 目录
- [快速开始](#快速开始)
- [分阶段训练](#分阶段训练)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 本地测试 (单卡)

```bash
# 使用本地测试配置
python train.py \
  --config configs/dtu_upr_mvs_transformer_local.yaml \
  --work_dir runs/local_test
```

### 服务器训练 (单卡)

```bash
python train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --work_dir runs/server_single
```

### 服务器训练 (多卡 DDP)

```bash
# 修改配置文件 train.use_ddp = true
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --launcher pytorch \
  --work_dir runs/server_ddp
```

---

## 📊 分阶段训练策略

### Stage A: Coarse Depth 预训练 (8 epochs)

**目标**: 训练粗深度估计模块，获得稳定的深度初始化

**配置**:
```yaml
train:
  stage: coarse_only
  epochs: 8

loss:
  coarse_weight: 1.0
  chamfer_weight: 0.0
  uncertainty_weight: 0.0
  feature_weight: 0.0
```

**训练**:
```bash
python train.py \
  --config configs/stage_a_coarse.yaml \
  --work_dir runs/stage_a
```

### Stage B: Point Refiner 训练 (12 epochs)

**目标**: 训练点细化模块，学习点云优化

**配置**:
```yaml
train:
  stage: point_refine
  epochs: 12

loss:
  coarse_weight: 0.0
  chamfer_weight: 1.0
  uncertainty_weight: 0.2
  feature_weight: 0.5
```

**训练**:
```bash
python train.py \
  --config configs/stage_b_point.yaml \
  --resume runs/stage_a/checkpoints/best.pth \
  --work_dir runs/stage_b
```

### Stage C: Joint Fine-tuning (10 epochs)

**目标**: 联合微调所有模块

**配置**:
```yaml
train:
  stage: joint
  epochs: 10

optim:
  lr_backbone: 0.00005
  lr_coarse: 0.00004  # 0.0002 * 0.2
  lr_point: 0.0002
```

**训练**:
```bash
python train.py \
  --config configs/stage_c_joint.yaml \
  --resume runs/stage_b/checkpoints/best.pth \
  --work_dir runs/stage_c
```

---

## ⚙️ 配置说明

### Loss 权重调整

如果 training loss 过大 (>1000)，尝试：

```yaml
loss:
  coarse_weight: 0.05      # 进一步降低
  chamfer_weight: 0.05
  depth_loss_scale: 0.001  # 更小的缩放
```

### 学习率调整

如果训练不稳定：

```yaml
optim:
  lr_backbone: 0.00001     # 降低 5 倍
  lr_coarse: 0.00002
  lr_point: 0.00004
  warmup_epochs: 3         # 增加 warmup
```

### 梯度裁剪

如果梯度爆炸：

```yaml
train:
  grad_clip: 0.5           # 从 1.0 降到 0.5
```

---

## 🔧 常见问题

### Q1: Loss 不下降/震荡

**原因**: 
- Loss 数值未归一化
- 学习率过大
- 没有 warmup

**解决**:
```yaml
loss:
  depth_loss_scale: 0.01
  chamfer_loss_scale: 0.001
  coarse_weight: 0.1
  chamfer_weight: 0.1

optim:
  warmup_epochs: 2
```

### Q2: Loss 变成 NaN

**原因**:
- 梯度爆炸
- AMP 精度问题

**解决**:
```yaml
train:
  grad_clip: 0.5
  amp_dtype: fp32          # 暂时关闭 AMP
  use_loss_nan_check: true
  early_stop_on_nan: true
```

### Q3: 显存不足

**解决**:
```yaml
train:
  batch_size_per_gpu: 1
  grad_accum_steps: 2      # 累积 2 步，等效 batch=2
  use_checkpoint: true     # 启用梯度检查点
```

---

## 📈 预期训练曲线

### Stage A (Coarse Only)
- `loss_coarse`: 300 → 50 (8 epochs)
- `depth_abs_error`: 50mm → 5mm

### Stage B (Point Refine)
- `loss_chamfer`: 500 → 50 (12 epochs)
- `loss_uncertainty`: 0.5 → 0.1

### Stage C (Joint)
- `loss_total`: 100 → 20 (10 epochs)
- 所有 loss 平稳下降

---

## 💾 Checkpoint 使用

### 从 checkpoint 恢复

```bash
# 完整恢复 (optimizer/scheduler/scaler)
python train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --resume runs/exp/checkpoints/epoch_005.pth \
  --resume_mode full \
  --work_dir runs/exp_resume

# 只恢复模型权重
python train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --resume runs/exp/checkpoints/epoch_005.pth \
  --resume_mode model_only \
  --work_dir runs/exp_finetune
```

---

## 🎯 超参数调优建议

### Loss 权重调优顺序

1. **先调 coarse_weight**: 确保 depth loss 在 0.1-10 之间
2. **再调 chamfer_weight**: 确保 chamfer loss 在 0.1-10 之间
3. **最后调其他 loss**: uncertainty, feature, repulsion

### 学习率调优

- **Backbone**: 1e-5 ~ 1e-4 (DINOv3 建议更小)
- **Coarse**: 1e-4 ~ 5e-4
- **Point**: 2e-4 ~ 1e-3

### Batch Size

- 单卡 16GB: `batch_size=1`, `grad_accum_steps=2`
- 单卡 80GB: `batch_size=2`, `grad_accum_steps=1`
