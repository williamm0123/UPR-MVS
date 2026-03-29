#!/bin/bash
set -e

echo "========================================"
echo "  UPR-MVS 服务器 DDP 训练脚本"
echo "========================================"

# 环境配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 配置选择
CONFIG_FILE="${1:-configs/dtu_upr_mvs_transformer.yaml}"
WORK_DIR="${2:-runs/server_ddp}"
NUM_GPUS="${3:-4}"

echo ""
echo "使用配置：$CONFIG_FILE"
echo "工作目录：$WORK_DIR"
echo "GPU 数量：$NUM_GPUS"
echo ""

# 运行 DDP 训练
torchrun \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  train.py \
  --config "$CONFIG_FILE" \
  --launcher pytorch \
  --work_dir "$WORK_DIR"

echo ""
echo "========================================"
echo "  训练完成！"
echo "========================================"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir $WORK_DIR/tensorboard"
echo ""
