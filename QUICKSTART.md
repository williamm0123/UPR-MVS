# UPR-MVS 快速启动指南（3 条命令）

## 🚀 本地测试（5060Ti 16GB）

### 一键自动训练
```bash
bash scripts/train_local.sh
```

**会自动执行**：
- ✅ Stage A: Coarse Depth (5 epochs)
- ✅ Stage B: Point Refiner (5 epochs)
- ✅ Stage C: Joint Fine-tuning (5 epochs)
- ⏱️ 总耗时：~30 分钟
- 💾 显存占用：12-15GB

---

## 🖥️ 服务器单卡（A100 80GB）

### 一键自动训练
```bash
bash scripts/train_server_single.sh
```

**会自动执行**：
- ✅ Stage A: Coarse Depth (20 epochs)
- ✅ Stage B: Point Refiner (20 epochs)
- ✅ Stage C: Joint Fine-tuning (20 epochs)
- ⏱️ 总耗时：~7 天
- 💾 显存占用：65-70GB (85% 利用率)

---

## 📊 服务器多卡（DDP 并行）

### 多卡加速训练
```bash
bash scripts/train_server_multigpu.sh 4
```

**参数说明**：
- `4`: GPU 数量（1-8）
- ⚡ 加速比：~3.5×（4 卡）
- 💾 每卡显存：65-70GB

---

## 🔍 配置验证

### 检查配置是否正确
```bash
bash scripts/validate_configs.sh
```

**输出示例**：
```
=== Checking Local Training (5060Ti 16GB) ===
   Image Size: 512x640
   Views: 3
   D bins: 64
   Gradient Checkpointing: true
   
   Stage Settings:
      Stage A: batch=2, accum=8, epochs=5
      Stage B: batch=2, accum=6, epochs=5
      Stage C: batch=1, accum=8, epochs=5

=== Checking Server Training (A100 80GB) ===
   Image Size: 1024x1280
   Views: 5
   D bins: 256  ⬆️ 关键改进！
   Gradient Checkpointing: true
```

---

## 📈 监控训练

### 实时日志
```bash
tail -f saved/local_training/stage_a/log.txt
```

### TensorBoard
```bash
tensorboard --logdir saved/local_training/tensorboard
```

### GPU 状态
```bash
watch -n 1 nvidia-smi
```

---

## ⚠️ 常见问题

### Q1: 本地 OOM 怎么办？
**A**: 降低分辨率和 batch size：
```yaml
img_h: 384
img_w: 512
batch_size_per_gpu: 1
```

### Q2: depth error 很高（>100）？
**A**: 确认 d_bins 设置：
- 本地：64（可以接受）
- 服务器：**必须 256**

### Q3: 服务器显存利用率低？
**A**: 提升 batch size：
```yaml
stage_a.batch_size_per_gpu: 16  # 从 12 提升到 16
```

---

## 📚 详细文档

- 📘 [`TRAINING_CONFIG_COMPLETE.md`](TRAINING_CONFIG_COMPLETE.md) - 完整优化报告
- 📘 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - 训练使用指南
- 📘 [`MAXIMIZE_GPU_MEMORY.md`](MAXIMIZE_GPU_MEMORY.md) - 显存优化详解

---

**就这么简单！** 🎉
