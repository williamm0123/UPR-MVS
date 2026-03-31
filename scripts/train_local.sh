#!/bin/bash
# UPR-MVS Local Development Training Script
# 用于本地开发调试 - DA3 + point refinement

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
echo "📊 Expected GPU memory usage: 取决于 DA3 分辨率与 point 数量"
echo "⏱️  Estimated time: 以 point refinement 为主"
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
echo "   - Stage A: point_refine"
echo "   - Stage B: joint (enable densify)"
echo "   - Image Size: 560x756"
echo "   - Views: 3"
echo "   - Epochs: 5 per stage"
echo "   - Depth Prior: DA3METRIC-LARGE"
echo ""

# Start training with automatic curriculum execution
echo "🎯 Starting automatic curriculum training..."
echo "   Stage A: Point Refiner Training"
echo "   Stage B: Joint Training with Densify"
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
echo "   1. Check point_abs_error / final_point_abs_error in logs"
echo "   2. Verify no OOM occurred"
echo "   3. If successful, push to server"
echo "   4. Adjust DA3 checkpoint path and point count"
echo "   5. Run: bash scripts/train_server_single.sh"
echo ""
