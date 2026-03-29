#!/bin/bash
set -e

echo "========================================"
echo "  UPR-MVS 本地快速测试脚本"
echo "========================================"

# 环境配置
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 配置选择
CONFIG_FILE="${1:-configs/dtu_upr_mvs_transformer_local.yaml}"
WORK_DIR="${2:-runs/local_quick_test}"

echo ""
echo "使用配置：$CONFIG_FILE"
echo "工作目录：$WORK_DIR"
echo ""

# 检查数据集路径
echo "正在检查数据集路径..."
DATA_ROOT="/home/william/project/dataset/DTU/dtu_training"
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 警告：数据集路径不存在：$DATA_ROOT"
    echo "   请确保 DTU 数据集已下载到该路径"
    echo ""
fi

# 运行训练
python train.py \
  --config "$CONFIG_FILE" \
  --work_dir "$WORK_DIR"

echo ""
echo "========================================"
echo "  训练完成！"
echo "========================================"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir $WORK_DIR/tensorboard"
echo ""