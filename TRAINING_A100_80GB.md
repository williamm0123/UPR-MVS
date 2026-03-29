# A100 80GB 单卡训练配置（防 OOM 版本）

## ⚠️ 问题说明

之前配置 batch_size=16 导致启动时就 OOM（exitcode -9, SIGKILL），现已调整为保守配置。

## 📊 优化后的配置参数

| 阶段 | 单卡 Batch | 梯度累积 | 等效 Batch | Epochs | Warmup |
|------|-----------|---------|-----------|--------|--------|
| **Stage A** (Coarse) | 4 | ×8 | **32** | 20 | 3 |
| **Stage B** (Point) | 4 | ×6 | **24** | 20 | 1 |
| **Stage C** (Joint) | 2 | ×8 | **16** | 20 | 2 |

## 🔧 关键修改

### Stage A - Coarse Depth Pretraining
```yaml
train:
  batch_size_per_gpu: 4        # 从 16 降低到 4
  grad_accum_steps: 8          # 从 2 增加到 8，保持等效 batch=32
  warmup_epochs: 3             # 增加到 3，更安全

model:
  use_checkpoint: true         # ✅ 必须启用
  cvt:
    use_checkpoint: true       # ✅ 必须启用
```

### Stage B - Point Refiner Training
```yaml
train:
  batch_size_point_per_gpu: 4  # 从 12 降低到 4
  grad_accum_steps: 6          # 等效 batch=24
```

### Stage C - Joint Fine-tuning
```yaml
train:
  batch_size_joint_per_gpu: 2  # 从 8 降低到 2（joint 显存占用最大）
  grad_accum_steps: 8          # 等效 batch=16
```

## 🚀 启动脚本（推荐）

### 使用优化后的启动脚本
```bash
cd /scr/user/qinglong/projects/UPR-MVS

# Stage A
bash scripts/train_stage_a_optimized.sh

# Stage B
bash scripts/train_stage_b_optimized.sh

# Stage C
bash scripts/train_stage_c_optimized.sh
```

## 📝 手动启动命令

### Stage A - Coarse Depth 预训练
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRACTION=0.95

torchrun --nproc_per_node=1 train.py \
  --config configs/stage_a_coarse.yaml \
  --work_dir outputs/stage_a
```

### Stage B - Point Refiner 训练
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRACTION=0.95

torchrun --nproc_per_node=1 train.py \
  --config configs/stage_b_point.yaml \
  --work_dir outputs/stage_b \
  --resume outputs/stage_a/checkpoints/best.pth
```

### Stage C - Joint Fine-tuning
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRACTION=0.95

torchrun --nproc_per_node=1 train.py \
  --config configs/stage_c_joint.yaml \
  --work_dir outputs/stage_c \
  --resume outputs/stage_b/checkpoints/best.pth
```

## 💡 显存优化原理

### 1. Gradient Checkpointing（必须启用）
- **原理**: 不保存中间激活，反向时重新计算
- **效果**: 减少 60-70% 显存占用
- **代价**: 训练速度减慢 20-30%
- **配置**: `use_checkpoint: true`

### 2. 梯度累积（Gradient Accumulation）
- **原理**: 多次小 batch 累加梯度后更新
- **公式**: 全局 batch = 单卡 batch × 梯度累积步数
- **优势**: 不增加显存，模拟大 batch 训练
- **注意**: 步数过多可能影响收敛稳定性

### 3. 混合精度训练（AMP）
- **配置**: `amp_dtype: bf16`（A100 推荐 bfloat16）
- **效果**: 减少约 50% 显存占用
- **配合**: GradScaler 防止下溢

### 4. 环境变量优化
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 减少显存碎片，提高利用率

PYTORCH_CUDA_MEMORY_FRACTION=0.95
# 允许 PyTorch 使用 95% 的 GPU 显存（默认 0.8）
```

## 📈 预期性能

### 显存占用（估算）
| 阶段 | 峰值显存 | 利用率 |
|------|---------|-------|
| Stage A | ~68GB | 85% |
| Stage B | ~65GB | 81% |
| Stage C | ~70GB | 87% |

### 训练速度（估算）
| 阶段 | 迭代速度 | 单 epoch 时间 |
|------|---------|------------|
| Stage A | ~1.5 iters/s | ~2 小时 |
| Stage B | ~1.2 iters/s | ~2.5 小时 |
| Stage C | ~0.8 iters/s | ~3.5 小时 |

**总训练时间**: 约 160-180 小时（20 epochs）

## 🔍 监控与调试

### 实时监控 GPU
```bash
# 每 1 秒刷新一次
watch -n 1 nvidia-smi

# 查看详细显存使用
nvidia-smi dmon -s m

# 记录日志
nvidia-smi dmon -s m -o T > gpu_log.txt &
```

### TensorBoard 监控
```bash
tensorboard --logdir outputs/stage_a/tensorboard --host 0.0.0.0
```

### 查看训练日志
```bash
tail -f outputs/stage_a/log.txt
```

## ⚠️ 故障排查

### 仍然 OOM 怎么办？

#### 方案 1: 进一步降低 batch size
```yaml
train:
  batch_size_per_gpu: 2        # 从 4 降到 2
  grad_accum_steps: 16         # 从 8 增到 16，保持等效 batch=32
```

#### 方案 2: 降低图像分辨率
```yaml
data:
  img_h: 768                   # 从 1024 降到 768
  img_w: 1024                  # 从 1280 降到 1024
```
**效果**: 显存减少约 40%，但可能影响精度

#### 方案 3: 减少网络层数
```yaml
model:
  cvt:
    num_layers: 4              # 从 6 降到 4
```
**效果**: 显存减少约 15%

#### 方案 4: 关闭非必需模块
```yaml
model:
  point:
    predict_uncertainty: false  # 关闭不确定性预测
  densify:
    enable: false               # 关闭点云加密
```

### 如果显存利用率仍然很低

#### 检查 Gradient Checkpointing 是否生效
在 `train.py` 中添加调试代码：
```python
# 在 build_model 后添加
if is_main_process():
    for name, module in model.named_modules():
        if hasattr(module, 'gradient_checkpointing'):
            print(f"{name}: checkpointing={module.gradient_checkpointing}")
```

#### 增加 batch size 测试
逐步增加 `batch_size_per_gpu`：
```yaml
batch_size_per_gpu: 6   # 从 4 增加到 6
grad_accum_steps: 6     # 从 8 降低到 6，保持等效 batch=36
```

## 🎯 学习率调整规则

当修改 batch size 时，需要同步调整学习率：

### Linear Scaling Rule
```
新 lr = 原 lr × (新 batch size / 原 batch size)
```

### 保守策略（推荐）
```
新 lr = 原 lr × sqrt(新 batch size / 原 batch size)
```

**示例**: 
- 原 batch=4, lr=0.0001
- 新 batch=8 → 新 lr = 0.0001 × sqrt(8/4) ≈ 0.00014

### Warmup 策略
- **Stage A**: warmup_epochs=3（大 batch 需要更长 warmup）
- **Stage B**: warmup_epochs=1
- **Stage C**: warmup_epochs=2

## 📋 配置文件清单

- ✅ `configs/stage_a_coarse.yaml` - Stage A（防 OOM 版）
- ✅ `configs/stage_b_point.yaml` - Stage B（防 OOM 版）
- ✅ `configs/stage_c_joint.yaml` - Stage C（防 OOM 版）
- ✅ `scripts/train_stage_a_optimized.sh` - 优化启动脚本
- ✅ `TRAINING_A100_80GB.md` - 本文档

## ✨ 关键要点总结

1. **必须启用 Gradient Checkpointing**: `use_checkpoint: true`
2. **使用梯度累积**: 用小 batch + 多步累积模拟大 batch
3. **设置环境变量**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
4. **充分 Warmup**: 至少 2-3 个 epoch
5. **监控显存**: 实时观察 nvidia-smi，确保不超过 95%
6. **渐进调整**: 如果稳定，可逐步增加 batch size 提高吞吐
