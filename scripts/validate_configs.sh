#!/bin/bash
# UPR-MVS Configuration Validation Script
# 用于快速检查本地和服务器配置的正确性

set -e

echo "========================================"
echo "  UPR-MVS Configuration Validator"
echo "========================================"
echo ""

LOCAL_CONFIG="configs/local_training.config"
SERVER_CONFIG="configs/server_training.config"

# Check if config files exist
if [ ! -f "$LOCAL_CONFIG" ]; then
    echo "❌ Local config not found: $LOCAL_CONFIG"
    exit 1
fi

if [ ! -f "$SERVER_CONFIG" ]; then
    echo "❌ Server config not found: $SERVER_CONFIG"
    exit 1
fi

echo "✅ Config files found"
echo ""

# Function to check configuration values
check_config() {
    local config_file=$1
    local config_name=$2
    
    echo "=== Checking $config_name ==="
    
    # Extract key parameters
    local img_h=$(grep "^  img_h:" "$config_file" | head -1 | awk '{print $2}')
    local img_w=$(grep "^  img_w:" "$config_file" | head -1 | awk '{print $2}')
    local n_views=$(grep "^  n_views:" "$config_file" | head -1 | awk '{print $2}')
    local d_bins=$(grep "^    d_bins:" "$config_file" | head -1 | awk '{print $2}')
    local use_checkpoint=$(grep "^  use_checkpoint:" "$config_file" | head -1 | awk '{print $2}')
    
    echo "   Image Size: ${img_h}x${img_w}"
    echo "   Views: $n_views"
    echo "   D bins: $d_bins"
    echo "   Gradient Checkpointing: $use_checkpoint"
    
    # Check stage-specific settings
    echo ""
    echo "   Stage Settings:"
    
    # Stage A
    local stage_a_batch=$(awk '/stage_a:/,/stage_b:/{if(/batch_size_per_gpu:/) print $2}' "$config_file")
    local stage_a_accum=$(awk '/stage_a:/,/stage_b:/{if(/grad_accum_steps:/) print $2}' "$config_file")
    local stage_a_epochs=$(awk '/stage_a:/,/stage_b:/{if(/epochs:/) print $2}' "$config_file")
    echo "      Stage A: batch=$stage_a_batch, accum=$stage_a_accum, epochs=$stage_a_epochs"
    
    # Stage B
    local stage_b_batch=$(awk '/stage_b:/,/stage_c:/{if(/batch_size_per_gpu:/) print $2}' "$config_file")
    local stage_b_accum=$(awk '/stage_b:/,/stage_c:/{if(/grad_accum_steps:/) print $2}' "$config_file")
    local stage_b_epochs=$(awk '/stage_b:/,/stage_c:/{if(/epochs:/) print $2}' "$config_file")
    echo "      Stage B: batch=$stage_b_batch, accum=$stage_b_accum, epochs=$stage_b_epochs"
    
    # Stage C
    local stage_c_batch=$(awk '/stage_c:/,/^[^ ]/{if(/batch_size_per_gpu:/) print $2}' "$config_file")
    local stage_c_accum=$(awk '/stage_c:/,/^[^ ]/{if(/grad_accum_steps:/) print $2}' "$config_file")
    local stage_c_epochs=$(awk '/stage_c:/,/^[^ ]/{if(/epochs:/) print $2}' "$config_file")
    echo "      Stage C: batch=$stage_c_batch, accum=$stage_c_accum, epochs=$stage_c_epochs"
    
    echo ""
}

check_config "$LOCAL_CONFIG" "Local Training (5060Ti 16GB)"
check_config "$SERVER_CONFIG" "Server Training (A100 80GB)"

echo "========================================"
echo "Configuration validation complete!"
echo "========================================"
echo ""

# Quick GPU detection
if command -v nvidia-smi &> /dev/null; then
    echo "📊 Current GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read name memory; do
        echo "   - $name ($memory GB)"
    done
    echo ""
fi

echo "💡 To start training:"
echo "   Local test:     bash scripts/train_local.sh"
echo "   Server single:  bash scripts/train_server_single.sh"
echo "   Server multi:   bash scripts/train_server_multigpu.sh <num_gpus>"
echo ""
