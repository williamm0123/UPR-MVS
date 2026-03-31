#!/bin/bash
# UPR-MVS Configuration Validation Script
# 用于验证 DA3-only 配置文件的正确性（不依赖数据集）

set -e

echo "========================================"
echo "  UPR-MVS Config Validation"
echo "========================================"
echo ""

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "📁 Working directory: $(pwd)"
echo ""

echo "🔍 Validating configurations..."
echo ""

python3 - <<'PY'
from pathlib import Path
import sys

try:
    import yaml
except ModuleNotFoundError:
    print("❌ Missing Python dependency: PyYAML")
    print("   Install with: pip install pyyaml")
    sys.exit(1)

configs = ['configs/local_training.config', 'configs/server_training.config']
for cfg in configs:
    with open(cfg, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(f'  ✓ {cfg} loaded successfully')

    assert config['model']['backbone'] == 'depth_anything3', 'Only depth_anything3 is supported now'
    assert 'depth_anything3' in config['model'], 'Missing depth_anything3 config'
    assert 'training_stages' in config, 'Missing training_stages config'
    assert list(config['training_stages'].keys()) == ['stage_a', 'stage_b'], 'Expected exactly stage_a/stage_b'

    prior_path = Path(config['model']['depth_anything3']['pretrained'])
    print(f'    depth_prior = {prior_path}')
    if prior_path.exists():
        print('    ✓ DA3 checkpoint path exists')
    else:
        print('    ⚠ DA3 checkpoint path does not exist yet; set it before training')

    for stage in ['stage_a', 'stage_b']:
        stage_cfg = config['training_stages'][stage]
        assert stage_cfg['name'] in {'point_refine', 'joint'}, f'Unexpected stage name in {stage}'
        assert 'batch_size_per_gpu' in stage_cfg, f'Missing batch_size_per_gpu in {stage}'
        assert 'grad_accum_steps' in stage_cfg, f'Missing grad_accum_steps in {stage}'
        effective_batch = stage_cfg['batch_size_per_gpu'] * stage_cfg['grad_accum_steps']
        print(
            f"    {stage}: mode={stage_cfg['name']}, "
            f"batch={stage_cfg['batch_size_per_gpu']}, "
            f"accum={stage_cfg['grad_accum_steps']}, "
            f"effective={effective_batch}"
        )

    print(f'  ✓ {cfg} validation passed')
    print()
PY

echo "========================================"
echo "  ✅ Validation finished"
echo "========================================"
echo ""
echo "📊 Next steps:"
echo "  1. Fill model.depth_anything3.pretrained with a real DA3METRIC-LARGE checkpoint"
echo "  2. Run connectivity test with: bash scripts/train_server_stage_a.sh"
echo "  3. Run full single-GPU curriculum with: bash scripts/train_server_single.sh"
echo ""
