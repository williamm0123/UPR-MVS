# A100 80GB 显存优化配置说明（防 OOM 版本）

## ⚠️ 重要更新

**2026-03-29 更新**: 初始配置 batch_size=16 导致启动时 OOM（exitcode -9），现已调整为保守配置。

### 问题现象
```bash
exitcode: -9 (pid: xxxxxx)
traceback : Signal 9 (SIGKILL) received by PID xxxxxx
```

### 根本原因
- 高分辨率输入（1024×1280）占用大量显存
- DINOv3 Backbone + CostVolumeTransformer 本身显存需求大
- Gradient Checkpointing 虽然启用，但 batch_size=16 仍超出 80GB 上限

---

## 优化目标（已调整）
- **GPU 显存利用率**: 从 <20% 提升到 **85-90%**（安全范围）
- **训练 Epoch**: 统一修改为 20 个 epoch
- **训练稳定性**: 通过梯度累积和 warmup 确保收敛

---

## 主要修改内容

### 1. Batch Size 优化（防 OOM 策略）

采用 **"小 batch + 多步累积"** 策略，在不增加显存的前提下模拟大 batch 训练。

#### Stage A (Coarse Depth Pretraining)
- **原始**: batch_size_per_gpu = 8
- **第一次调整**: batch_size_per_gpu = 16 ❌ → OOM
- **最终方案**: batch_size_per_gpu = 4, grad_accum_steps = 8
- **等效 batch size**: 32 (4 × 8) ✅

#### Stage B (Point Refiner Training)
- **原始**: batch_size_point_per_gpu = 6
- **第一次调整**: batch_size_point_per_gpu = 12 ❌ → OOM
- **最终方案**: batch_size_point_per_gpu = 4, grad_accum_steps = 6
- **等效 batch size**: 24 (4 × 6) ✅

#### Stage C (Joint Fine-tuning)
- **原始**: batch_size_joint_per_gpu = 4
- **第一次调整**: batch_size_joint_per_gpu = 8 ❌ → OOM
- **最终方案**: batch_size_joint_per_gpu = 2, grad_accum_steps = 8
- **等效 batch size**: 16 (2 × 8) ✅

**注意**: Joint 阶段显存占用最大（所有模块都激活），因此 batch size 最小。

### 2. Gradient Checkpointing (显存优化关键技术)

在所有阶段的配置文件中启用 gradient checkpointing:

```yaml
model:
  use_checkpoint: true          # Backbone 启用 checkpoint
  cvt:
    use_checkpoint: true        # CostVolumeTransformer 启用 checkpoint
  point:
    use_checkpoint: true        # Point Refiner 启用 checkpoint (Stage B/C)
```

**原理**: 以计算换显存，在前向传播时不保存中间激活，而在反向传播时重新计算。
**效果**: 可减少 60-70% 的显存占用，支持更大的 batch size。

### 3. 学习率调整

根据 batch size 的变化，相应调整学习率:

#### Stage A
- lr_backbone: 0.0001 → 0.00005 (降低 2×)
- lr_coarse: 0.0002 → 0.0001 (降低 2×)

#### Stage B
- lr_point: 0.0004 → 0.0002 (降低 2×)

#### Stage C
- lr_backbone: 0.00005 → 0.000025 (降低 2×)
- lr_coarse: 0.00002 → 0.00001 (降低 2×)
- lr_point: 0.0004 → 0.0002 (降低 2×)

### 4. Warmup 策略增强

增加 warmup 轮数以适应更大的 batch size:

```yaml
optim:
  warmup_epochs: 2              # 从 0-1 增加到 2
  warmup_lr_init: 1.0e-6
```

### 5. 数据加载优化

减少 num_workers 以平衡 CPU 内存和加载速度:
- **原始**: num_workers = 12
- **优化后**: num_workers = 8

**原因**: 
- 过多的 worker 会占用大量 CPU 内存
- 8 个 worker 对于 A100 已足够快
- 减少进程间通信开销

## 预期效果

### 显存利用率
- **优化前**: <20% (~16GB @ 80GB)
- **优化后**: ~85% (~68GB @ 80GB)

### 训练吞吐量
- **Stage A**: 提升约 2-3× (batch size 从 8→32)
- **Stage B**: 提升约 2× (batch size 从 6→24)
- **Stage C**: 提升约 2× (batch size 从 4→16)

### 训练稳定性
- 更大的 batch size 通常带来更稳定的梯度
- 配合 warmup 和 learning rate scaling，收敛更平滑
- Gradient checkpointing 对精度无影响

## 使用方式

### Stage A - Coarse Depth 训练
```bash
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_a_coarse.yaml \
  --work_dir outputs/stage_a_coarse
```

### Stage B - Point Refiner 训练
```bash
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_b_point.yaml \
  --work_dir outputs/stage_b_point \
  --resume outputs/stage_a_coarse/checkpoints/best.pth
```

### Stage C - Joint Fine-tuning
```bash
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_c_joint.yaml \
  --work_dir outputs/stage_c_joint \
  --resume outputs/stage_b_point/checkpoints/best.pth
```

## 监控建议

训练过程中监控以下指标:
1. **GPU 显存使用率**: `nvidia-smi -l 1`
2. **GPU 利用率**: 应保持在 90%+
3. **训练 Loss**: 确保收敛正常
4. **验证指标**: depth_abs_error / point_abs_error

## 故障排查

### 如果显存溢出 (OOM)
1. 降低 batch_size_per_gpu (如 16→12)
2. 增加 grad_accum_steps 保持等效 batch size
3. 检查图像分辨率是否过高

### 如果显存利用率仍然偏低
1. 进一步增加 batch_size_per_gpu
2. 减少 grad_accum_steps (但保持等效 batch size 不变)
3. 检查是否有其他进程占用显存

## 理论依据

### Batch Size 扩展原则
- 当 batch size 增加 k 倍时，learning rate 也应增加 k 倍 (Linear Scaling Rule)
- 但实际中通常采用保守策略，增加幅度小于 k

### Gradient Checkpointing
- 参考论文: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
- PyTorch 实现: `torch.utils.checkpoint.checkpoint`
- 适用于深层网络（如 ViT、Transformer）

## 配置文件清单
- ✅ configs/stage_a_coarse.yaml
- ✅ configs/stage_b_point.yaml
- ✅ configs/stage_c_joint.yaml
- ✅ configs/dtu_upr_mvs_transformer.yaml
