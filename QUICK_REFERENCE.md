# A100 80GB 训练配置快速参考

## 📊 核心参数对比表

| 阶段 | 原始 Batch | 优化后 Batch | 梯度累积 | 等效 Batch | Epochs | 显存目标 |
|------|-----------|-------------|---------|-----------|--------|---------|
| **Stage A** (Coarse) | 8 | 16 | ×2 | **32** | 20 | 85% |
| **Stage B** (Point) | 6 | 12 | ×2 | **24** | 20 | 85% |
| **Stage C** (Joint) | 4 | 8 | ×2 | **16** | 20 | 85% |

## 🔧 关键配置项

### Stage A - Coarse Depth Pretraining
```yaml
train:
  epochs: 20
  batch_size_per_gpu: 16
  grad_accum_steps: 2          # 等效 32
  num_workers: 8

model:
  use_checkpoint: true         # ✅ Backbone
  cvt:
    use_checkpoint: true       # ✅ Transformer

optim:
  lr_backbone: 0.00005
  lr_coarse: 0.0001
  warmup_epochs: 2
```

### Stage B - Point Refiner Training
```yaml
train:
  epochs: 20
  batch_size_point_per_gpu: 12
  grad_accum_steps: 2          # 等效 24
  num_workers: 8

model:
  use_checkpoint: true         # ✅ Backbone + CVT
  densify:
    enable: true               # ✅ 启用点云加密

optim:
  lr_point: 0.0002
  warmup_epochs: 1
```

### Stage C - Joint Fine-tuning
```yaml
train:
  epochs: 20
  batch_size_joint_per_gpu: 8
  grad_accum_steps: 2          # 等效 16
  num_workers: 8

model:
  use_checkpoint: true         # ✅ 所有模块
  point:
    use_checkpoint: true

optim:
  lr_backbone: 0.000025
  lr_coarse: 0.00001
  lr_point: 0.0002
  warmup_epochs: 2
```

## 🚀 训练命令

### 单卡训练（推荐）
```bash
# Stage A
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_a_coarse.yaml \
  --work_dir outputs/stage_a

# Stage B
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_b_point.yaml \
  --work_dir outputs/stage_b \
  --resume outputs/stage_a/checkpoints/best.pth

# Stage C
torchrun --nproc_per_node=1 train.py \
  --config configs/stage_c_joint.yaml \
  --work_dir outputs/stage_c \
  --resume outputs/stage_b/checkpoints/best.pth
```

## 📈 预期性能指标

| 指标 | 优化前 | 优化后 | 提升 |
|------|-------|-------|------|
| 显存利用率 | ~20% (16GB) | ~85% (68GB) | **4.25×** |
| Stage A 吞吐量 | ~1.5 iters/s | ~4.0 iters/s | **2.7×** |
| Stage B 吞吐量 | ~1.2 iters/s | ~2.5 iters/s | **2.1×** |
| Stage C 吞吐量 | ~0.8 iters/s | ~1.8 iters/s | **2.2×** |

## ⚠️ 注意事项

### 1. 显存监控
```bash
# 实时监控 GPU 状态
watch -n 1 nvidia-smi

# 查看详细显存使用
nvidia-smi dmon -s m
```

### 2. 如果 OOM (Out Of Memory)
- 降低 `batch_size_per_gpu` (如 16→12)
- 增加 `grad_accum_steps` 保持总 batch size
- 检查是否有其他进程占用显存

### 3. 如果显存利用率仍低
- 增加 `batch_size_per_gpu` (如 16→20)
- 减少 `grad_accum_steps` (但保持等效 batch size)
- 确认 `use_checkpoint: true` 已启用

### 4. 学习率调整原则
当修改 batch size 时:
- batch size 增加 k 倍 → lr 增加约 k/2 倍 (保守策略)
- 必须配合 warmup 使用

## 🔍 调试技巧

### 检查 Gradient Checkpointing 是否生效
```python
# 在训练脚本中添加
for name, module in model.named_modules():
    if hasattr(module, 'gradient_checkpointing'):
        print(f"{name}: checkpointing={module.gradient_checkpointing}")
```

### 检查实际 Batch Size
```python
print(f"Per-GPU batch size: {batch_size}")
print(f"Gradient accumulation steps: {accum_steps}")
print(f"Effective batch size: {batch_size * accum_steps}")
```

### 监控训练进度
```bash
# 查看最新日志
tail -f outputs/stage_a/log.txt

# 查看 TensorBoard
tensorboard --logdir outputs/stage_a/tensorboard
```

## 📝 配置文件位置

- ✅ `configs/stage_a_coarse.yaml` - Stage A 配置
- ✅ `configs/stage_b_point.yaml` - Stage B 配置
- ✅ `configs/stage_c_joint.yaml` - Stage C 配置
- ✅ `configs/dtu_upr_mvs_transformer.yaml` - 主配置
- ✅ `MEMORY_OPTIMIZATION.md` - 详细优化说明

## 💡 为什么这样配置？

### 1. **Gradient Checkpointing**
- 原理：用计算换显存
- 效果：减少 60-70% 显存占用
- 代价：训练速度减慢 20-30%
- 净收益：支持 2-3× 更大的 batch size

### 2. **梯度累积 (Gradient Accumulation)**
- 原理：多次小 batch 梯度累加后更新
- 效果：模拟更大 batch size 的训练
- 优势：不增加显存占用
- 注意：accum_steps 不宜过大（影响收敛）

### 3. **学习率调整**
- Linear Scaling Rule: batch × k → lr × k
- 实际策略：batch × k → lr × (k/2) 更稳定
- Warmup: 防止初期梯度爆炸

### 4. **num_workers = 8**
- 平衡 CPU 内存和加载速度
- A100 通常配 20+ CPU 核心
- 8 worker 足够喂饱 GPU
