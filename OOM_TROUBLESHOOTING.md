# OOM 问题快速排错指南

## ❌ 遇到的错误

```bash
exitcode: -9 (pid: xxxxxx)
error_file: <N/A>
traceback : Signal 9 (SIGKILL) received by PID xxxxxx
```

**原因**: 显存溢出（OOM），被系统强制杀死

---

## ✅ 解决方案（已应用）

### 1. 降低单卡 Batch Size，增加梯度累积

#### Stage A 配置调整
```yaml
train:
  batch_size_per_gpu: 4        # 从 16 降到 4
  grad_accum_steps: 8          # 从 2 增到 8
  # 等效 batch size = 4 × 8 = 32
```

#### Stage B 配置调整
```yaml
train:
  batch_size_point_per_gpu: 4  # 从 12 降到 4
  grad_accum_steps: 6          # 等效 batch size = 24
```

#### Stage C 配置调整
```yaml
train:
  batch_size_joint_per_gpu: 2  # 从 8 降到 2（joint 最占显存）
  grad_accum_steps: 8          # 等效 batch size = 16
```

### 2. 启用环境变量优化

启动脚本已添加：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRACTION=0.95
```

**作用**:
- `expandable_segments`: 减少显存碎片
- `memory_fraction`: 允许使用 95% 显存（默认 80%）

### 3. 确保 Gradient Checkpointing 启用

所有配置文件已设置：
```yaml
model:
  use_checkpoint: true         # ✅ Backbone
  cvt:
    use_checkpoint: true       # ✅ CostVolumeTransformer
  point:
    use_checkpoint: true       # ✅ Point Refiner (Stage B/C)
```

---

## 🚀 推荐的启动方式

### 使用优化后的启动脚本（推荐）

```bash
cd /scr/user/qinglong/projects/UPR-MVS

# Stage A - Coarse Depth 预训练
bash scripts/train_stage_a_optimized.sh

# Stage B - Point Refiner 训练（从 Stage A 恢复）
bash scripts/train_stage_b_optimized.sh \
  --resume outputs/stage_a/checkpoints/best.pth

# Stage C - Joint Fine-tuning（从 Stage B 恢复）
bash scripts/train_stage_c_optimized.sh \
  --resume outputs/stage_b/checkpoints/best.pth
```

---

## 🔍 如果仍然 OOM

### 方案 A: 进一步降低 Batch Size

修改 `configs/stage_a_coarse.yaml`:
```yaml
train:
  batch_size_per_gpu: 2        # 从 4 降到 2
  grad_accum_steps: 16         # 从 8 增到 16
  # 仍保持等效 batch = 32
```

### 方案 B: 降低图像分辨率

修改 `configs/stage_a_coarse.yaml`:
```yaml
data:
  img_h: 768                   # 从 1024 降到 768
  img_w: 1024                  # 从 1280 降到 1024
```
**效果**: 显存减少约 40%，但可能影响精度

### 方案 C: 减少 Transformer 层数

修改 `configs/stage_a_coarse.yaml`:
```yaml
model:
  cvt:
    num_layers: 4              # 从 6 降到 4
```
**效果**: 显存减少约 15%

### 方案 D: 关闭非必需功能

```yaml
model:
  point:
    predict_uncertainty: false  # 关闭不确定性预测
  
  densify:
    enable: false               # 关闭点云加密（Stage A 已关闭）
```

---

## 📊 显存监控命令

### 实时监控
```bash
watch -n 1 nvidia-smi
```

### 详细显存使用
```bash
nvidia-smi dmon -s m
```

### 记录日志
```bash
nvidia-smi dmon -s m -o T > gpu_usage.log &
```

### 查看进程
```bash
nvidia-smi pmon -c 1
```

---

## ⚙️ 安全阈值建议

| 指标 | 安全范围 | 警告范围 | 危险范围 |
|------|---------|---------|---------|
| **显存利用率** | <85% | 85-92% | >92% |
| **GPU 温度** | <75°C | 75-80°C | >80°C |
| **GPU 功耗** | <300W | 300-350W | >350W |

---

## 🎯 渐进式调优策略

### 第 1 步：先跑起来（当前阶段）
```yaml
batch_size_per_gpu: 4
grad_accum_steps: 8
```
**目标**: 确保不 OOM，能开始训练

### 第 2 步：观察显存使用
运行 10-20 个 iteration，观察：
```bash
watch -n 1 nvidia-smi
```

如果显存利用率 <70%，进入第 3 步

### 第 3 步：逐步增加 Batch Size
```yaml
# 尝试 1
batch_size_per_gpu: 6
grad_accum_steps: 6  # 等效 batch = 36

# 如果稳定，继续增加
batch_size_per_gpu: 8
grad_accum_steps: 4  # 等效 batch = 32
```

### 第 4 步：找到最优平衡点
目标：显存利用率 85-90%，吞吐量最大

---

## 📝 检查清单

训练前确认：
- [ ] 已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- [ ] 已设置 `use_checkpoint: true`
- [ ] batch_size 和 grad_accum_steps 配置合理
- [ ] warmup_epochs >= 2
- [ ] amp_dtype: bf16

训练中监控：
- [ ] 显存利用率 <90%
- [ ] GPU 温度 <80°C
- [ ] Loss 正常下降
- [ ] 无 NaN/Inf 警告

---

## 💡 经验法则

1. **优先使用梯度累积**: 不占显存，只是稍慢
2. **Gradient Checkpointing 必开**: 省 60-70% 显存
3. **不要一开始就用最大 batch**: 渐进测试
4. **保留 10% 显存余量**: 防止峰值溢出
5. **Warmup 很重要**: 大 batch 需要更长 warmup

---

## 🔗 相关文档

- `TRAINING_A100_80GB.md` - 完整训练指南
- `MEMORY_OPTIMIZATION.md` - 显存优化原理
- `QUICK_REFERENCE.md` - 快速参考卡片
