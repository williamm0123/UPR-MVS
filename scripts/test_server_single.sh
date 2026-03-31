#!/bin/bash
# UPR-MVS Server Single-GPU Evaluation Script (A100 80GB)

set -e

SPLIT=${1:-val}
PROJECT_ROOT=/scr/user/qinglong/projects/UPR-MVS
CONFIG=${CONFIG:-configs/server_training.config}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/saved/server_single_gpu}
CHECKPOINT=${CHECKPOINT:-$WORK_DIR/best.pth}

if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "val" ] && [ "$SPLIT" != "test" ]; then
    echo "Usage: $0 [train|val|test]"
    exit 1
fi

echo "========================================"
echo "  UPR-MVS Single-GPU Evaluation"
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

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "   Run training first or set CHECKPOINT=/path/to/best.pth"
    exit 1
fi

echo ""
echo "📋 Evaluation Summary:"
echo "   - Config: $CONFIG"
echo "   - Split: $SPLIT"
echo "   - Work Dir: $WORK_DIR"
echo "   - Checkpoint: $CHECKPOINT"
echo ""

torchrun --nproc_per_node=1 test.py \
    --config "$CONFIG" \
    --work_dir "$WORK_DIR" \
    --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" \
    --launcher pytorch
