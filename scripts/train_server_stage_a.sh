#!/bin/bash
# UPR-MVS Server Single-GPU Stage A Training Script (A100 80GB)

set -e

PROJECT_ROOT=/scr/user/qinglong/projects/UPR-MVS
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/saved/server_single_gpu_stage_a}
CONFIG=${CONFIG:-configs/server_training.config}

echo "========================================"
echo "  UPR-MVS Stage A Training (A100 80GB)"
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

echo ""
echo "📋 Configuration Summary:"
echo "   - Config: $CONFIG"
echo "   - Stage: stage_a"
echo "   - Mode: point_refine"
echo "   - Work Dir: $WORK_DIR"
echo ""

torchrun --nproc_per_node=1 train.py \
    --config "$CONFIG" \
    --work_dir "$WORK_DIR" \
    --stage stage_a \
    --launcher pytorch

echo ""
echo "========================================"
echo "  ✅ Stage A completed"
echo "========================================"
echo ""
echo "📊 Results saved to: $WORK_DIR"
