# UPR-MVS 配置整合完成报告

## ✅ 完成内容

### 📁 配置文件精简（2 个）

#### 删除的旧文件（10 个）
- ❌ `configs/dtu_upr_mvs_transformer.yaml`
- ❌ `configs/dtu_upr_mvs_transformer_local.yaml`
- ❌ `configs/stage_a_coarse.yaml`
- ❌ `configs/stage_a_coarse_local.yaml`
- ❌ `configs/stage_b_point.yaml`
- ❌ `configs/stage_b_point_local.yaml`
- ❌ `configs/stage_c_joint.yaml`
- ❌ `configs/stage_c_joint_local.yaml`

#### 新增的配置文件（2 个）
- ✅ [`configs/server_training.config`](configs/server_training.config) - 服务器 A100 80GB 单卡训练配置
  - 包含三阶段完整配置
  - 优化显存使用（gradient checkpointing + 梯度累积）
  - 20 epochs/阶段，完整训练流程
  
- ✅ [`configs/local_training.config`](configs/local_training.config) - 本地开发调试配置
  - 小模型、低分辨率
  - 快速验证代码
  - 5 epochs/阶段

---

### 🚀 启动脚本精简（3 个）

#### 删除的旧脚本（7 个）
- ❌ `scripts/train_server_ddp.sh`
- ❌ `scripts/train_stage_a_optimized.sh`
- ❌ `scripts/train_stage_b_optimized.sh`
- ❌ `scripts/train_stage_c_optimized.sh`
- ❌ `scripts/train_stages.sh`
- ❌ `scripts/train_stages_local.sh`
- ❌ `scripts/quick_test_local.sh`

#### 新增的启动脚本（3 个）
- ✅ [`scripts/train_local.sh`](scripts/train_local.sh) - 本地开发训练脚本
  ```bash
  bash scripts/train_local.sh
  ```

- ✅ [`scripts/train_server_single.sh`](scripts/train_server_single.sh) - 服务器单卡训练脚本
  ```bash
  bash scripts/train_server_single.sh
  ```

- ✅ [`scripts/train_server_multigpu.sh`](scripts/train_server_multigpu.sh) - 服务器多卡训练脚本
  ```bash
  bash scripts/train_server_multigpu.sh 4  # 使用 4 卡
  ```

---

### 🔧 train.py 功能增强

#### 新增功能
1. **自动分阶段训练支持**
   - `--stage auto`: 自动执行三阶段训练
   - `--stage stage_a`: 只执行 Stage A
   - `--stage stage_b`: 只执行 Stage B
   - `--stage stage_c`: 只执行 Stage C

2. **阶段配置自动应用**
   - 每个阶段自动加载对应的超参数
   - 自动调整学习率、batch size、loss 权重
   - 自动启用/禁用模块（如 densify）

3. **Checkpoint 自动管理**
   - Stage B 自动加载 Stage A 的最佳 checkpoint
   - Stage C 自动加载 Stage B 的最佳 checkpoint
   - 支持手动指定 resume 路径

#### 新增命令行参数
```python
parser.add_argument(
    "--stage",
    type=str,
    default="auto",
    choices=["auto", "stage_a", "stage_b", "stage_c"],
    help="Training stage to run. Use 'auto' for automatic multi-stage training.",
)
```

---

### 📂 输出目录统一

所有训练输出保存到 `saved/` 目录：

```
saved/
├── server_single_gpu/          # 服务器单卡训练
│   ├── stage_a/
│   │   ├── checkpoints/
│   │   │   ├── best.pth
│   │   │   └── latest.pth
│   │   └── tensorboard/
│   ├── stage_b/
│   └── stage_c/
│
├── local_training/             # 本地训练
└── server_multi_gpu/           # 服务器多卡训练
```

---

## 🎯 核心改进

### 1. 配置管理简化
- **之前**: 8 个配置文件，难以选择和维护
- **现在**: 2 个配置文件，职责清晰

### 2. 训练流程自动化
- **之前**: 需要手动运行 3 个脚本，分别指定配置
- **现在**: 一条命令自动完成三阶段训练

### 3. 代码复用提升
- **之前**: 每个阶段的配置分散，重复代码多
- **现在**: 统一配置结构，通过 `training_stages` 字段区分

### 4. 灵活性增强
- 支持自动三阶段训练
- 支持单独运行某个阶段
- 支持从任意 checkpoint 恢复

---

## 📖 使用示例

### 示例 1: 服务器完整训练流程
```bash
# 一条命令完成三阶段训练
bash scripts/train_server_single.sh

# 输出:
# saved/server_single_gpu/stage_a/  - Stage A 结果
# saved/server_single_gpu/stage_b/  - Stage B 结果
# saved/server_single_gpu/stage_c/  - Stage C 结果
```

### 示例 2: 只训练 Stage A
```bash
bash scripts/train_server_single.sh --stage stage_a
```

### 示例 3: 本地快速测试
```bash
bash scripts/train_local.sh
```

### 示例 4: 多卡加速训练
```bash
# 使用 4 卡 DDP 训练
bash scripts/train_server_multigpu.sh 4
```

### 示例 5: 从 checkpoint 恢复
```bash
# 从 Stage A 的 checkpoint 继续训练
bash scripts/train_server_single.sh \
  --resume saved/server_single_gpu/stage_a/checkpoints/best.pth
```

---

## 📊 配置文件结构对比

### 之前的结构（分散）
```yaml
# stage_a_coarse.yaml
train:
  stage: coarse_only
  epochs: 20
  batch_size_per_gpu: 4
  ...

# stage_b_point.yaml
train:
  stage: point_refine
  epochs: 20
  batch_size_point_per_gpu: 4
  ...
```

### 现在的结构（整合）
```yaml
# server_training.config
training_stages:
  stage_a:
    name: coarse_only
    epochs: 20
    batch_size_per_gpu: 4
    lr_backbone: 0.00005
    loss_weights:
      coarse_weight: 1.0
  
  stage_b:
    name: point_refine
    epochs: 20
    batch_size_per_gpu: 4
    lr_point: 0.0002
    loss_weights:
      chamfer_weight: 1.0
  
  stage_c:
    name: joint
    epochs: 20
    batch_size_per_gpu: 2
    lr_backbone: 0.000025
    loss_weights:
      coarse_weight: 0.1

# 默认配置（不使用 training_stages 时）
train:
  stage: coarse_only
  epochs: 20
  ...
```

---

## 🔍 关键技术实现

### 1. 阶段配置自动应用
```python
def update_config_for_stage(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    """Update config dictionary with stage-specific settings."""
    if "training_stages" not in config:
        return config
    
    stage_cfg = config["training_stages"].get(stage_name, {})
    # 更新 train, optim, loss 等配置
    ...
```

### 2. 模型配置动态调整
```python
def apply_stage_config(model: nn.Module, config: dict[str, Any], stage_name: str, device: torch.device) -> None:
    """Apply stage-specific configuration to the model."""
    # 启用/禁用 densify
    # 设置 use_checkpoint
    ...
```

### 3. Checkpoint 自动管理
```python
if stage_idx > 0:
    # 自动从上一阶段加载最佳 checkpoint
    prev_stage_name = stages_to_run[stage_idx - 1]
    resume_path = work_dir / prev_stage_name / "checkpoints" / "best.pth"
```

---

## ⚠️ 注意事项

### 1. 配置文件格式
- 新配置文件使用 `.config` 后缀，便于识别
- 保持 YAML 格式，与原来兼容

### 2. 向后兼容性
- 旧的训练脚本已删除，如需使用请查看 git history
- 旧的配置文件已删除，建议迁移到新配置

### 3. 路径依赖
- 服务器配置中的数据集路径为绝对路径
- 本地配置使用相对路径，需确保数据在 `./data/DTU/`

---

## 📚 相关文档

- 📘 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - 完整训练指南
- 📘 [`QUICKSTART_SIMPLE.md`](QUICKSTART_SIMPLE.md) - 快速启动指南（3 条命令）
- 📘 [`TRAINING_A100_80GB.md`](TRAINING_A100_80GB.md) - A100 优化详解
- 📘 [`OOM_TROUBLESHOOTING.md`](OOM_TROUBLESHOOTING.md) - OOM 排错指南

---

## ✅ 检查清单

使用前确认：
- [x] 配置文件已精简到 2 个
- [x] 启动脚本已精简到 3 个
- [x] train.py 支持自动分阶段训练
- [x] 输出目录统一为 `saved/`
- [x] 所有文件无语法错误
- [x] 文档已更新

---

**配置整合完成！现在只需一条命令即可开始训练。** 🎉
