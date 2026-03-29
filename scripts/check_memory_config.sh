#!/bin/bash
# UPR-MVS 显存优化配置验证脚本
# 用于快速检查配置是否正确设置

set -e

echo "========================================"
echo "  UPR-MVS GPU Memory Configuration Check"
echo "========================================"
echo ""

CONFIG_FILE="configs/server_training.config"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Checking configuration: $CONFIG_FILE"
echo ""

# Function to check and display value
check_value() {
    local key=$1
    local expected=$2
    local current=$(grep "$key" "$CONFIG_FILE" | head -1 | awk '{print $2}')
    
    if [ "$current" == "$expected" ]; then
        echo "✅ $key = $current (expected: $expected)"
    else
        echo "⚠️  $key = $current (expected: $expected)"
    fi
}

echo "=== Stage A Settings ==="
check_value "batch_size_per_gpu:" "16"
check_value "grad_accum_steps:" "2"

echo ""
echo "=== Stage B Settings ==="
# Extract stage_b specific settings
stage_b_batch=$(awk '/stage_b:/,/stage_c:/{if(/batch_size_per_gpu:/) print $2}' "$CONFIG_FILE")
stage_b_accum=$(awk '/stage_b:/,/stage_c:/{if(/grad_accum_steps:/) print $2}' "$CONFIG_FILE")

if [ "$stage_b_batch" == "12" ]; then
    echo "✅ Stage B batch_size = $stage_b_batch"
else
    echo "⚠️  Stage B batch_size = $stage_b_batch (expected: 12)"
fi

if [ "$stage_b_accum" == "2" ]; then
    echo "✅ Stage B grad_accum = $stage_b_accum"
else
    echo "⚠️  Stage B grad_accum = $stage_b_accum (expected: 2)"
fi

echo ""
echo "=== Stage C Settings ==="
stage_c_batch=$(awk '/stage_c:/,/^[^ ]/{if(/batch_size_per_gpu:/) print $2}' "$CONFIG_FILE")
stage_c_accum=$(awk '/stage_c:/,/^[^ ]/{if(/grad_accum_steps:/) print $2}' "$CONFIG_FILE")

if [ "$stage_c_batch" == "8" ]; then
    echo "✅ Stage C batch_size = $stage_c_batch"
else
    echo "⚠️  Stage C batch_size = $stage_c_batch (expected: 8)"
fi

if [ "$stage_c_accum" == "2" ]; then
    echo "✅ Stage C grad_accum = $stage_c_accum"
else
    echo "⚠️  Stage C grad_accum = $stage_c_accum (expected: 2)"
fi

echo ""
echo "=== Gradient Checkpointing ==="
backbone_ckpt=$(grep "use_checkpoint:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
cvt_ckpt=$(grep -A5 "cvt:" "$CONFIG_FILE" | grep "use_checkpoint:" | head -1 | awk '{print $2}')

if [ "$backbone_ckpt" == "true" ]; then
    echo "✅ Backbone checkpointing enabled"
else
    echo "⚠️  Backbone checkpointing = $backbone_ckpt (expected: true)"
fi

if [ "$cvt_ckpt" == "true" ]; then
    echo "✅ CVT checkpointing enabled"
else
    echo "⚠️  CVT checkpointing = $cvt_ckpt (expected: true)"
fi

echo ""
echo "=== Expected Memory Usage ==="
echo "Stage A: ~68GB (85% of 80GB)"
echo "Stage B: ~65GB (81% of 80GB)"
echo "Stage C: ~70GB (87% of 80GB)"
echo ""

echo "========================================"
echo "Configuration check complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  bash scripts/train_server_single.sh"
echo ""
echo "Monitor GPU memory:"
echo "  watch -n 1 nvidia-smi"
echo ""
