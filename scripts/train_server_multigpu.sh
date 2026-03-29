#!/bin/bash
# UPR-MVS Server Multi-GPU DDP Training Script
# 用于服务器多卡分布式训练

set -e

echo "========================================"
echo "  UPR-MVS Multi-GPU DDP Training"
echo "========================================"
echo ""

# Environment setup for multi-GPU training
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
CONFIG_FILE="${1:-configs/server_training.config}"
WORK_DIR="${2:-saved/server_multi_gpu}"
NUM_GPUS="${3:-4}"

# Parse additional command line arguments
shift 3 2>/dev/null || true

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config|--work_dir|--stage|--resume)
            EXTRA_ARGS="$EXTRA_ARGS $1 $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration: $CONFIG_FILE"
echo "Work Directory: $WORK_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Check if dataset exists
DATA_ROOT="/scr/user/qinglong/dataset/DTU/dtu_training"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Warning: Dataset not found at $DATA_ROOT"
    echo "   Please ensure DTU dataset is downloaded to this location"
    echo ""
fi

# Run multi-GPU DDP training
torchrun \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  train.py \
  --config "$CONFIG_FILE" \
  --launcher pytorch \
  --work_dir "$WORK_DIR" \
  --stage auto \
  $EXTRA_ARGS

echo ""
echo "========================================"
echo "  Training completed!"
echo "========================================"
echo ""
echo "View TensorBoard:"
echo "  tensorboard --logdir $WORK_DIR/tensorboard"
echo ""
