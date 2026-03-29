#!/bin/bash -l
#SBATCH --job-name=upr_mvs_8gpu
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scr/user/qinglong/projects/UPR-MVS
mkdir -p logs

source ~/.bashrc
conda activate mvs2

export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 train.py \
  --config configs/server_training.config \
  --work_dir saved/server_multi_gpu_8gpu \
  --stage auto \
  --launcher pytorch

echo ""
echo "📊 Results saved to: saved/server_multi_gpu_8gpu/"
echo "📈 TensorBoard: tensorboard --logdir saved/server_multi_gpu_8gpu --host 0.0.0.0 --port 6006"
