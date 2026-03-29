# UPR-MVS Loss 收敛优化总结

## 🔍 问题分析

从第一次训练截图发现以下问题：

1. **Loss 数值巨大**: total_loss 在 500-800 之间震荡
2. **Loss 不收敛**: 没有明显下降趋势
3. **Loss 分量异常**:
   - `loss_coarse`: 300-350 (过大)
   - `loss_chamfer`: 400-450 (过大)
   - `loss_feature`: 快速收敛到 0 (异常)

## 🛠️ 优化措施

### 1. Loss 归一化

**问题**: Depth 和点云坐标数值过大（毫米级别）

**解决**:
```python
# models/losses/consistency.py
depth_loss_scale = 0.01  # 将深度值缩小 100 倍
scaled_target = target_depth * depth_loss_scale
scaled_coarse = coarse_depth * depth_loss_scale

# models/losses/chamfer.py
chamfer_scale = 0.001  # 将毫米转为米
chamfer_loss = pairwise_dist.mean() * chamfer_scale
```

### 2. Loss 权重调整

**优化前**:
```yaml
loss:
  coarse_weight: 1.0
  chamfer_weight: 1.0
```

**优化后**:
```yaml
loss:
  coarse_weight: 0.1      # 降低 10 倍
  chamfer_weight: 0.1     # 降低 10 倍
  depth_loss_scale: 0.01
  chamfer_loss_scale: 0.001
```

### 3. 学习率优化

**优化前**:
```yaml
optim:
  lr_backbone: 0.0001
  lr_coarse: 0.0002
  lr_point: 0.0004
```

**优化后**:
```yaml
optim:
  lr_backbone: 0.00005    # 降低 2 倍
  lr_coarse: 0.0001       # 降低 2 倍
  lr_point: 0.0002        # 降低 2 倍
  warmup_epochs: 1        # 新增 warmup
  warmup_lr_init: 1.0e-6
```

### 4. 梯度裁剪

**优化前**: `grad_clip: 5.0`

**优化后**: `grad_clip: 1.0` (更严格的梯度裁剪)

### 5. 训练稳定性增强

新增配置:
```yaml
train:
  use_loss_nan_check: true     # NaN 检测
  early_stop_on_nan: true      # NaN 提前终止
```

### 6. 分阶段训练策略

**Stage A**: Coarse Depth 预训练 (8 epochs)
- 只训练 coarse depth 模块
- 冻结 backbone 和 point 模块
- 只用 coarse loss

**Stage B**: Point Refiner 训练 (12 epochs)
- 只训练 point 模块
- 冻结 backbone 和 cvt
- 使用 chamfer + uncertainty + feature loss

**Stage C**: Joint Fine-tuning (10 epochs)
- 联合训练所有模块
- coarse 模块使用 0.2 倍学习率

## 📊 预期效果

### 优化前
- `loss_total`: 500-800 (不收敛)
- `loss_coarse`: 300-350
- `loss_chamfer`: 400-450

### 优化后 (预期)

**Stage A**:
- `loss_coarse`: 300 → 10-20 (8 epochs)
- `depth_abs_error`: 50mm → 3-5mm

**Stage B**:
- `loss_chamfer`: 500 → 20-30 (12 epochs)
- `loss_uncertainty`: 0.5 → 0.1

**Stage C**:
- `loss_total`: 50-100 → 10-20 (10 epochs)

## 🚀 使用方法

### 本地快速测试

```bash
# 快速测试 (1 epoch)
./scripts/quick_test_local.sh

# 或手动指定
python train.py \
  --config configs/dtu_upr_mvs_transformer_local.yaml \
  --work_dir runs/local_test
```

### 分阶段训练

```bash
# 完整三阶段训练
./scripts/train_stages.sh runs/stage_exp

# 或单独执行某个阶段
python train.py \
  --config configs/stage_a_coarse.yaml \
  --work_dir runs/stage_a
```

### 服务器训练

```bash
# 单卡训练
python train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --work_dir runs/server_single

# 多卡 DDP (4 GPU)
./scripts/train_server_ddp.sh configs/dtu_upr_mvs_transformer.yaml runs/server_ddp 4
```

## 🔧 调参指南

### 如果 Loss 仍然过大

```yaml
loss:
  coarse_weight: 0.05      # 进一步降低
  chamfer_weight: 0.05
  depth_loss_scale: 0.001  # 更小的缩放
```

### 如果训练不稳定

```yaml
train:
  grad_clip: 0.5           # 更严格的裁剪
  amp_dtype: fp32          # 暂时关闭 AMP

optim:
  lr_backbone: 0.00001     # 降低学习率
  lr_coarse: 0.00002
  lr_point: 0.00004
  warmup_epochs: 3         # 增加 warmup
```

### 如果显存不足

```yaml
train:
  batch_size_per_gpu: 1
  grad_accum_steps: 2      # 梯度累积
  use_checkpoint: true     # 梯度检查点
```

## 📈 监控指标

### 正常训练曲线

- **前 100 steps**: loss 快速下降，可能有波动
- **100-1000 steps**: loss 平稳下降
- **1000+ steps**: loss 缓慢下降，趋于稳定

### 异常检测

**Loss 爆炸**:
- loss_total > 1000
- 出现 NaN/Inf

**解决**: 降低学习率，增加 warmup，检查数据加载

**Loss 不下降**:
- loss 保持在初始值
- 梯度接近 0

**解决**: 增加学习率，检查 loss 权重，验证 GT 数据

## 📝 修改文件清单

1. ✅ `configs/dtu_upr_mvs_transformer_local.yaml` - 本地配置
2. ✅ `configs/dtu_upr_mvs_transformer.yaml` - 服务器配置
3. ✅ `models/losses/consistency.py` - Depth loss 缩放
4. ✅ `models/losses/chamfer.py` - Chamfer loss 缩放
5. ✅ `engine/trainer.py` - Warmup + NaN 检查 + 阶段训练
6. ✅ `TRAINING.md` - 完整训练指南
7. ✅ `scripts/quick_test_local.sh` - 本地测试脚本
8. ✅ `scripts/train_server_ddp.sh` - 服务器 DDP 脚本
9. ✅ `scripts/train_stages.sh` - 分阶段训练脚本

## 🎯 下一步

1. **本地验证**: 先在本地跑 1 epoch，确认 loss 下降
2. **调整参数**: 根据本地结果微调 loss 权重和学习率
3. **服务器训练**: 使用优化后的配置在服务器上训练
4. **监控 TensorBoard**: 实时查看 loss 曲线和中间结果

---

**关键提示**: 
- 第一次训练建议使用 `stage: coarse_only` 先验证 coarse depth 能收敛
- 观察 `depth_abs_error` 指标，应该从 50mm+ 降到 5mm 以内
- 如果 coarse stage 不收敛，不要进入下一阶段
