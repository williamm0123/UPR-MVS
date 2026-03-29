#!/bin/bash
# UPR-MVS Server Multi-GPU Training Script (DDP)
# 用于服务器多卡 DDP 并行训练

set -e

echo "========================================"
echo "  UPR-MVS Multi-GPU Training (DDP)"
echo "========================================"
echo ""

# Check if number of GPUs is specified
if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_gpus>"
    echo "Example: $0 4"
    exit 1
fi

NUM_GPUS=$1

# Validate number of GPUs
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
    echo "Error: Number of GPUs must be between 1 and 8"
    exit 1
fi

echo "🚀 Starting multi-GPU training with $NUM_GPUS GPUs..."
echo ""

# Activate conda environment if needed
if [ -d "/scr/user/qinglong/.conda/envs/mvs2" ]; then
    source /scr/user/qinglong/.conda/envs/mvs2/bin/activate
    echo "✅ Conda environment activated"
fi

cd /scr/user/qinglong/projects/UPR-MVS

echo ""
echo "📋 Configuration:"
echo "   - Config: configs/server_training.config"
echo "   - GPUs: $NUM_GPUS"
echo "   - Effective Batch Size: ${NUM_GPUS}x per stage"
echo ""

# Start DDP training
torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config configs/server_training.config \
    --work_dir saved/server_multi_gpu \
    --stage auto \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Multi-GPU training completed!"
echo "========================================"
echo ""
