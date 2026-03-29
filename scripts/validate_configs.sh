#!/bin/bash
# UPR-MVS Configuration Validation Script
# 用于验证配置文件的正确性（不需要数据集）

set -e

echo "========================================"
echo "  UPR-MVS Config Validation"
echo "========================================"
echo ""

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR

if [ -d "/home/user/qinglong/.conda/envs/mvs2" ]; then
    source /home/user/qinglong/.conda/envs/mvs2/bin/activate
    echo "✅ Conda environment activated"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."  # Go to project root

echo "📁 Working directory: $(pwd)"
echo ""

echo "🔍 Validating configurations..."
echo ""

# Test 1: Check config file syntax
echo "[Test 1] Checking config file syntax..."
python3 -c "
import yaml
configs = ['configs/local_training.config', 'configs/server_training.config']
for cfg in configs:
    with open(cfg, 'r') as f:
        config = yaml.safe_load(f)
    print(f'  ✓ {cfg} loaded successfully')
    
    # Validate key structures
    assert 'model' in config, 'Missing model config'
    assert 'data' in config, 'Missing data config'
    assert 'training_stages' in config, 'Missing training_stages config'
    
    # Validate training stages
    for stage in ['stage_a', 'stage_b', 'stage_c']:
        assert stage in config['training_stages'], f'Missing {stage}'
        stage_cfg = config['training_stages'][stage]
        assert 'batch_size_per_gpu' in stage_cfg, f'Missing batch_size in {stage}'
        assert 'grad_accum_steps' in stage_cfg, f'Missing grad_accum_steps in {stage}'
        
        # Calculate effective batch size
        effective_batch = stage_cfg['batch_size_per_gpu'] * stage_cfg['grad_accum_steps']
        print(f'    {stage}: Batch={stage_cfg[\"batch_size_per_gpu\"]}, Accum={stage_cfg[\"grad_accum_steps\"]}, Effective={effective_batch}')
    
    print(f'  ✓ {cfg} validation passed')
    print()
"

# Test 2: Model architecture validation (without data loading)
echo "[Test 2] Validating model architecture..."
python3 -c "
import torch
import yaml
from models.upr_mvs_transformer import UPRMVSTransformerModel

with open('configs/server_training.config', 'r') as f:
    config = yaml.safe_load(f)

model_cfg = config['model'].copy()
# Temporarily disable pretrained loading for validation
model_cfg['dinov3_pretrained'] = None
print('  Building model with config (pretrained disabled)...')

try:
    model = UPRMVSTransformerModel(model_cfg)
    print('  ✓ Model created successfully')
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  ✓ Total params: {total_params:,} ({trainable_params:,} trainable)')
    
    print('  ✓ Model architecture validation passed (skip forward pass without CUDA)')
    
except Exception as e:
    print(f'  ✗ Model creation failed: {e}')
    raise
"

echo ""
echo "========================================"
echo "  ✅ All validation tests passed!"
echo "========================================"
echo ""
echo "Configuration summary:"
echo "  - Config files: Valid YAML syntax"
echo "  - Training stages: Properly configured"
echo "  - Model architecture: Can be instantiated"
echo ""
echo "📊 Next steps:"
echo "  1. ✓ Config validation complete"
echo "  2. Run on server with: bash scripts/train_server_single.sh"
echo ""
