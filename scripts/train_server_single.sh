#!/bin/bash
# UPR-MVS Server Single-GPU Training Script (A100 80GB)
# 用于服务器 A100 80GB 单卡 DA3 + point refinement 训练

set -e

echo "========================================"
echo "  UPR-MVS Server Training (A100 80GB)"
echo "========================================"
echo ""

# Environment setup for server training
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting server production training..."
echo "📊 Expected GPU memory usage: 取决于 DA3 分辨率与 point 数量"
echo "⏱️  Estimated time: 主要由 DA3 prior 前向和 point branch 决定"
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
echo "   - Stage A: point_refine"
echo "   - Stage B: joint (enable densify)"
echo "   - Image Size: 756x1008"
echo "   - Views: 5"
echo "   - Depth Prior: DA3METRIC-LARGE"
echo ""

# Start training with single-run curriculum execution
echo "🎯 Starting single-run curriculum training..."
echo "   Stage A: Point Refiner Training"
echo "   Stage B: Joint Training with Densify"
echo ""

torchrun --nproc_per_node=1 train.py \
    --config configs/server_training.config \
    --work_dir saved/server_single_gpu \
    --stage curriculum \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Server training completed!"
echo "========================================"
echo ""
echo "📊 Results saved to: saved/server_single_gpu/"
echo "📈 TensorBoard: tensorboard --logdir saved/server_single_gpu --host 0.0.0.0 --port 6006"
echo ""
