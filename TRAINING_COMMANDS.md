# 🚀 UPR-MVS 简洁训练指令

所有训练参数都已配置在 YAML 文件中，脚本只负责调用！

---

## 📁 配置文件说明

### 本地环境 (RTX 5060Ti 16GB)
- **主配置**: `configs/dtu_upr_mvs_transformer_local.yaml`
- **Stage A**: `configs/stage_a_coarse_local.yaml`
- **Stage B**: `configs/stage_b_point_local.yaml`
- **Stage C**: `configs/stage_c_joint_local.yaml`

### 服务器环境 (A100 80GB ×4)
- **主配置**: `configs/dtu_upr_mvs_transformer.yaml`
- **Stage A**: `configs/stage_a_coarse.yaml`
- **Stage B**: `configs/stage_b_point.yaml`
- **Stage C**: `configs/stage_c_joint.yaml`

---

## 🖥️ 本地训练指令

### 方案 1: 快速测试 (1 epoch)
```bash
cd /home/william/project/UPR-MVS
./scripts/quick_test_local.sh
```

### 方案 2: 分阶段训练 (推荐)
```bash
cd /home/william/project/UPR-MVS
./scripts/train_stages_local.sh runs/local_stages
```

### 方案 3: 手动执行单个阶段
```bash
# Stage A
python train.py --config configs/stage_a_coarse_local.yaml --work_dir runs/stage_a

# Stage B
python train.py --config configs/stage_b_point_local.yaml \
  --resume runs/stage_a/checkpoints/best.pth --work_dir runs/stage_b

# Stage C
python train.py --config configs/stage_c_joint_local.yaml \
  --resume runs/stage_b/checkpoints/best.pth --work_dir runs/stage_c
```

---

## 🖥️ 服务器训练指令

### 方案 1: 端到端训练
```bash
cd /path/to/UPR-MVS

# 单卡
python train.py --config configs/dtu_upr_mvs_transformer.yaml --work_dir runs/server

# 多卡 DDP
torchrun --standalone --nproc_per_node=4 train.py \
  --config configs/dtu_upr_mvs_transformer.yaml \
  --launcher pytorch --work_dir runs/server_ddp
```

### 方案 2: 分阶段训练 (强烈推荐)
```bash
cd /path/to/UPR-MVS
./scripts/train_stages.sh runs/full_training
```

### 方案 3: 手动执行单个阶段
```bash
# Stage A (8-10 epochs, ~2-3 小时)
python train.py --config configs/stage_a_coarse.yaml --work_dir runs/stage_a

# Stage B (12-14 epochs, ~4-5 小时)
python train.py --config configs/stage_b_point.yaml \
  --resume runs/stage_a/checkpoints/best.pth --work_dir runs/stage_b

# Stage C (10-12 epochs, ~3-4 小时)
python train.py --config configs/stage_c_joint.yaml \
  --resume runs/stage_b/checkpoints/best.pth --work_dir runs/stage_c
```

---

## 📊 查看 TensorBoard

```bash
# 本地
tensorboard --logdir runs/local_stages/tensorboard

# 服务器
tensorboard --logdir runs/full_training/tensorboard --port 6006
```

---

## 🔧 调整配置

所有超参数都在 YAML 配置文件中，直接修改即可：

### 修改学习率
```yaml
optim:
  lr_backbone: 0.0001
  lr_coarse: 0.0002
  lr_point: 0.0004
```

### 修改 Epoch 数
```yaml
train:
  epochs: 10  # 增加或减少训练轮数
```

### 修改 Loss 权重
```yaml
loss:
  coarse_weight: 0.1
  chamfer_weight: 1.0
```

### 修改 Batch Size
```yaml
train:
  batch_size_per_gpu: 2  # 根据显存调整
```

---

## ✅ 优势

- ✅ **脚本简洁**: 每个脚本 < 50 行
- ✅ **配置集中**: 所有参数在 YAML 文件
- ✅ **易于调整**: 修改配置无需改代码
- ✅ **环境隔离**: 本地和服务器配置完全独立
- ✅ **快速实验**: 轻松尝试不同超参数组合
