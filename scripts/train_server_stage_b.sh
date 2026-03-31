#!/bin/bash
# UPR-MVS Server Single-GPU Stage B Training Script (A100 80GB)

set -e

PROJECT_ROOT=/scr/user/qinglong/projects/UPR-MVS
CONFIG=${CONFIG:-configs/server_training.config}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/saved/server_single_gpu_stage_b}
RESUME=${RESUME:-$PROJECT_ROOT/saved/server_single_gpu_stage_a/best.pth}

echo "========================================"
echo "  UPR-MVS Stage B Training (A100 80GB)"
echo "========================================"
echo ""

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -d "/scr/user/qinglong/.conda/envs/mvs2" ]; then
    source /scr/user/qinglong/.conda/envs/mvs2/bin/activate
    echo "✅ Conda environment activated"
fi

cd "$PROJECT_ROOT"

if [ ! -f "$RESUME" ]; then
    echo "❌ Resume checkpoint not found: $RESUME"
    echo "   Run Stage A first or set RESUME=/path/to/best.pth"
    exit 1
fi

echo ""
echo "📋 Configuration Summary:"
echo "   - Config: $CONFIG"
echo "   - Stage: stage_b"
echo "   - Mode: joint"
echo "   - Work Dir: $WORK_DIR"
echo "   - Resume: $RESUME"
echo ""

torchrun --nproc_per_node=1 train.py \
    --config "$CONFIG" \
    --work_dir "$WORK_DIR" \
    --stage stage_b \
    --resume "$RESUME" \
    --resume_mode model_only \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Stage B completed"
echo "========================================"
echo ""
echo "📊 Results saved to: $WORK_DIR"
