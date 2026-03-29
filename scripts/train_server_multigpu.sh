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

# Environment setup for multi-GPU training
export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
echo "   - DDP: Enabled"
echo "   - Batch Size: per-GPU value comes from config"
echo "   - Global Batch: scales with GPU count x grad_accum_steps"
echo "   - OMP_NUM_THREADS: $OMP_NUM_THREADS per process"
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
echo "📊 Results saved to: saved/server_multi_gpu/"
echo "📈 TensorBoard: tensorboard --logdir saved/server_multi_gpu --host 0.0.0.0 --port 6006"
echo ""
