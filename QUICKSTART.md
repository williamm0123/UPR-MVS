# 🚀 UPR-MVS 快速启动指南

## ⚡ 5 分钟快速测试

### 1. 环境检查

```bash
# 确认 conda 环境
conda activate mvs

# 确认 GPU
nvidia-smi
```

### 2. 本地快速测试 (推荐)

```bash
# 方法 1: 使用快速测试脚本
cd /home/william/project/UPR-MVS
./scripts/quick_test_local.sh

# 方法 2: 手动启动
python train.py \
  --config configs/dtu_upr_mvs_transformer_local.yaml \
  --work_dir runs/quick_test
```

**预期**:
- 运行 1 epoch
- 约 10-20 分钟
- Loss 应该从 100+ 降到 50 以内

### 3. 查看结果

```bash
# 启动 TensorBoard
tensorboard --logdir runs/quick_test/tensorboard

# 浏览器访问 http://localhost:6006
```

**关键指标**:
- `train/loss_total`: 应该持续下降
- `train/loss_coarse`: 应该 < 50
- `train/depth_abs_error`: 应该 < 10mm

---

## 📊 如果 Loss 正常下降

### 方案 A: 本地完整训练

```bash
# 修改配置文件 epochs=10
# configs/dtu_upr_mvs_transformer_local.yaml
#   train:
#     epochs: 10

python train.py \
  --config configs/dtu_upr_mvs_transformer_local.yaml \
  --work_dir runs/local_full
```

### 方案 B: 分阶段训练 (推荐)

```bash
# 三阶段训练
./scripts/train_stages.sh runs/stage_exp

# 或单独执行每个阶段
python train.py --config configs/stage_a_coarse.yaml --work_dir runs/stage_a
python train.py --config configs/stage_b_point.yaml --resume runs/stage_a/checkpoints/best.pth --work_dir runs/stage_b
python train.py --config configs/stage_c_joint.yaml --resume runs/stage_b/checkpoints/best.pth --work_dir runs/stage_c
```

---

## 🖥️ 服务器训练

### 单卡训练

```bash
python train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --work_dir runs/server_single
```

### 多卡 DDP (4 GPU)

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --launcher pytorch \
  --work_dir runs/server_ddp
```

---

## 🔧 常见问题快速修复

### Q1: Loss > 1000 或 NaN

```yaml
# 修改配置
loss:
  coarse_weight: 0.05
  chamfer_weight: 0.05
  depth_loss_scale: 0.001

train:
  grad_clip: 0.5
  amp_dtype: fp32
```

### Q2: Loss 不下降

```yaml
optim:
  lr_backbone: 0.0001     # 增加学习率
  lr_coarse: 0.0002
  lr_point: 0.0004
  warmup_epochs: 0        # 关闭 warmup
```

### Q3: 显存不足

```yaml
train:
  grad_accum_steps: 2     # 累积 2 步
  use_checkpoint: true    # 启用检查点
```

---

## 📈 训练进度检查

### 正常训练曲线

```
Epoch 1: loss_total ~ 100-200
Epoch 3: loss_total ~ 50-80
Epoch 8: loss_total ~ 20-40
Epoch 10: loss_total ~ 10-30
```

### 检查点

```bash
# 查看最佳模型
ls -lh runs/exp/checkpoints/best.pth
ls -lh runs/exp/checkpoints/epoch_*.pth
```

---

## 🎯 下一步

1. ✅ 本地快速测试 (1 epoch)
2. ✅ 确认 Loss 下降趋势
3. 📝 调整超参数 (如需要)
4. 🚀 本地完整训练 或 服务器训练

---

## 📚 详细文档

- [TRAINING.md](TRAINING.md) - 完整训练指南
- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 优化说明
- [configs/dtu_upr_mvs_transformer_local.yaml](configs/dtu_upr_mvs_transformer_local.yaml) - 本地配置

---

**关键提示**: 
- 第一次运行务必使用 `epochs: 1` 快速验证
- 观察前 100 步的 loss 趋势
- 如有异常立即停止检查
