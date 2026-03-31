#!/bin/bash -l
#SBATCH --job-name=upr_mvs_DDP4
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --chdir=/scr/user/qinglong/projects/UPR-MVS
#SBATCH --output=/scr/user/qinglong/projects/UPR-MVS/logs/%x_%j.out
#SBATCH --error=/scr/user/qinglong/projects/UPR-MVS/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/scr/user/qinglong/projects/UPR-MVS
WORK_DIR=$PROJECT_ROOT/saved/server_multi_gpu_4gpu

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs" "$WORK_DIR"

source ~/.bashrc
conda activate mvs2

export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 train.py \
  --config configs/server_training.config \
  --work_dir "$WORK_DIR" \
  --stage curriculum \
  --launcher pytorch

echo ""
echo "📊 Results saved to: $WORK_DIR"
echo "📈 TensorBoard: tensorboard --logdir $WORK_DIR --host 0.0.0.0 --port 6006"
