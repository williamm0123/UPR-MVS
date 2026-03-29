#!/bin/bash
# UPR-MVS Local Training Script (Development/Debug)
# 用于本地开发调试的单卡训练

set -e

echo "========================================"
echo "  UPR-MVS Local Training"
echo "========================================"
echo ""

# Environment setup for local training
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
CONFIG_FILE="configs/local_training.config"
WORK_DIR="saved/local_training"

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
DATA_ROOT="./data/DTU/dtu_training"
if [ ! -d "$DATA_ROOT" ]; then
    echo "⚠️  Warning: Dataset not found at $DATA_ROOT"
    echo "   Please download DTU dataset to this location"
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
