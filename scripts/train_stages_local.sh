#!/bin/bash
set -e

echo "========================================"
echo "  UPR-MVS 分阶段训练脚本 (本地版)"
echo "========================================"
echo ""

WORK_DIR_BASE="${1:-runs/stage_training}"

# 检查数据集路径
echo "正在检查数据集路径..."
DATA_ROOT="/home/william/project/dataset/DTU/dtu_training"
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 警告：数据集路径不存在：$DATA_ROOT"
    echo "   请确保 DTU 数据集已下载到该路径"
    echo ""
fi

# Stage A: Coarse Depth Pretraining
echo "========================================"
echo "  Stage A: Coarse Depth Pretraining"
echo "========================================"
echo ""

python train.py \
  --config configs/stage_a_coarse_local.yaml \
  --work_dir "$WORK_DIR_BASE/stage_a"

# Stage B: Point Refiner Training
echo ""
echo "========================================"
echo "  Stage B: Point Refiner Training"
echo "========================================"
echo ""

python train.py \
  --config configs/stage_b_point_local.yaml \
  --resume "$WORK_DIR_BASE/stage_a/checkpoints/best.pth" \
  --work_dir "$WORK_DIR_BASE/stage_b"

# Stage C: Joint Fine-tuning
echo ""
echo "========================================"
echo "  Stage C: Joint Fine-tuning"
echo "========================================"
echo ""

python train.py \
  --config configs/stage_c_joint_local.yaml \
  --resume "$WORK_DIR_BASE/stage_b/checkpoints/best.pth" \
  --work_dir "$WORK_DIR_BASE/stage_c"

echo ""
echo "========================================"
echo "  所有阶段训练完成！"
echo "========================================"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir $WORK_DIR_BASE/tensorboard"
echo ""