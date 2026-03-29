#!/bin/bash
# UPR-MVS Server Single-GPU Training Script (A100 80GB)
# 用于服务器 A100 80GB 单卡三阶段自动训练 - 高显存利用率版本

set -e

echo "========================================"
echo "  UPR-MVS Server Training (A100 80GB)"
echo "========================================"
echo ""

# Environment setup for server training with high memory utilization
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting server production training..."
echo "📊 Expected GPU memory usage: ~65-70GB (85% utilization)"
echo "⏱️  Estimated time: ~7 days (20 epochs x 3 stages)"
echo ""

# Activate conda environment if needed
if [ -d "/scr/user/qinglong/.conda/envs/mvs2" ]; then
    source /scr/user/qinglong/.conda/envs/mvs2/bin/activate
    echo "✅ Conda environment activated"
fi

cd /scr/user/qinglong/projects/UPR-MVS

echo ""
echo "📋 Configuration Summary:"
echo "   - Config: configs/server_training.config"
echo "   - Batch Size: 12 (Stage A), 8 (Stage B), 6 (Stage C)"
echo "   - Gradient Accumulation: 3 steps"
echo "   - Image Size: 1024x1280"
echo "   - Views: 5"
echo "   - Epochs: 20 per stage"
echo "   - D bins: 256 (improved from 64)"
echo ""

# Start training with automatic 3-stage execution
echo "🎯 Starting automatic 3-stage training..."
echo "   Stage A: Coarse Depth Pretraining"
echo "   Stage B: Point Refiner Training"
echo "   Stage C: Joint Fine-tuning"
echo ""

torchrun --nproc_per_node=1 train.py \
    --config configs/server_training.config \
    --work_dir saved/server_single_gpu \
    --stage auto \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Server training completed!"
echo "========================================"
echo ""
echo "📊 Results saved to: saved/server_single_gpu/"
echo "📈 TensorBoard: tensorboard --logdir saved/server_single_gpu/tensorboard --host 0.0.0.0"
echo ""
