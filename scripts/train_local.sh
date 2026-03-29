#!/bin/bash
# UPR-MVS Local Development Training Script (RTX 5060Ti 16GB)
# 用于本地开发调试 - 自动三阶段训练

set -e

echo "========================================"
echo "  UPR-MVS Local Training (5060Ti 16GB)"
echo "========================================"
echo ""

# Environment setup for local training with OOM prevention
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR

# Set memory allocation strategy for better stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting local development training..."
echo "📊 Expected GPU memory usage: ~12-15GB"
echo "⏱️  Estimated time: ~30 minutes (5 epochs x 3 stages)"
echo ""

# Activate conda environment if needed
if [ -d "/home/user/qinglong/.conda/envs/mvs2" ]; then
    source /home/user/qinglong/.conda/envs/mvs2/bin/activate
    echo "✅ Conda environment activated"
fi

cd /home/william/project/UPR-MVS

echo ""
echo "📋 Configuration Summary:"
echo "   - Config: configs/local_training.config"
echo "   - Batch Size: 2 (Stage A/B), 1 (Stage C)"
echo "   - Gradient Accumulation: 8-6 steps"
echo "   - Image Size: 512x640"
echo "   - Views: 3"
echo "   - Epochs: 5 per stage"
echo ""

# Start training with automatic 3-stage execution
echo "🎯 Starting automatic 3-stage training..."
echo "   Stage A: Coarse Depth Pretraining"
echo "   Stage B: Point Refiner Training"
echo "   Stage C: Joint Fine-tuning"
echo ""

torchrun --nproc_per_node=1 train.py \
    --config configs/local_training.config \
    --work_dir saved/local_training \
    --stage auto \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Local training completed!"
echo "========================================"
echo ""
echo "📊 Results saved to: saved/local_training/"
echo "📈 TensorBoard: tensorboard --logdir saved/local_training/tensorboard"
echo ""
echo "🔍 Next steps:"
echo "   1. Check depth_abs_error in logs"
echo "   2. Verify no OOM occurred"
echo "   3. If successful, push to server"
echo "   4. Adjust parameters for A100 80GB"
echo "   5. Run: bash scripts/train_server_single.sh"
echo ""
