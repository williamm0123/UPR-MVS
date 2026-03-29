#!/bin/bash
# UPR-MVS Server Single-GPU Training Script (A100 80GB)
# 用于服务器 A100 80GB 单卡三阶段自动训练

set -e

echo "========================================"
echo "  UPR-MVS Server Single-GPU Training"
echo "========================================"
echo ""

# Environment setup for server training
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_MEMORY_FRACTION=0.95

# Configuration
CONFIG_FILE="configs/server_training.config"
WORK_DIR="saved/server_single_gpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --work_dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --stage)
            STAGE_ARG="--stage $2"
            shift 2
            ;;
        --resume)
            RESUME_ARG="--resume $2"
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
echo ""

# Check if dataset exists
DATA_ROOT="/scr/user/qinglong/dataset/DTU/dtu_training"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Warning: Dataset not found at $DATA_ROOT"
    echo "   Please ensure DTU dataset is downloaded to this location"
    echo ""
fi

# Run training
python train.py \
  --config "$CONFIG_FILE" \
  --work_dir "$WORK_DIR" \
  --stage auto \
  ${STAGE_ARG:-} \
  ${RESUME_ARG:-}

echo ""
echo "========================================"
echo "  Training completed!"
echo "========================================"
echo ""
echo "View TensorBoard:"
echo "  tensorboard --logdir $WORK_DIR/tensorboard"
echo ""
