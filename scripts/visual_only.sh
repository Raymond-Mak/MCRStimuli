#!/bin/bash

# Visual-Only LaFTer Training Pipeline
# Usage: ./scripts/visual_only.sh [dataset] [epochs] [batch_size] [learning_rate] [architecture]

DSET="$1"
MM_EPOCHS="50"
MM_BATCH_SIZE="32"
HEAD_LR="1e-4"
ARCH="ViT-B/32"

# Default parameter validation
if [ -z "$DSET" ]; then
    echo "Usage: $0 <dataset> [epochs] [batch_size] [learning_rate] [architecture]"
    echo "Example: $0 Emotion6 20 32 1e-4 ViT-B/32"
    exit 1
fi

# Log configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/visual_only/${DSET}"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/visual_only_${DSET}_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

# Visual encoder configuration
ENCODER_TYPE="clip"
FREEZE_VISION="true"  # Freeze CLIP encoder
DROPOUT_RATE="0.1"

# Data paths (using same structure as original LaFTer)
DATA_ROOT="data/${DSET}"

# Clean architecture name for paths
ARCH_CLEAN=$(echo "${ARCH}" | sed 's/[^a-zA-Z0-9]/_/g')

# Training hyperparameters (optimized for visual-only mode)
VIT_LR="0"  # Visual encoder frozen, so no learning rate needed
MM_BATCH_SIZE="${MM_BATCH_SIZE}"
HEAD_LR="${HEAD_LR}"
MM_EPOCHS="${MM_EPOCHS}"
FUSION_DIM="512"  # CLIP feature dimension
FUSION_TYPE="standard"
VISUAL_ONLY="true"

log "===== Visual-Only Training Configuration ====="
log "Dataset: ${DSET}"
log "Architecture: ${ARCH}"
log "Encoder Type: ${ENCODER_TYPE}"
log "Visual Encoder Frozen: ${FREEZE_VISION}"
log "Batch Size: ${MM_BATCH_SIZE}"
log "Epochs: ${MM_EPOCHS}"
log "Classifier Learning Rate: ${HEAD_LR}"
log "Dropout Rate: ${DROPOUT_RATE}"
log "==========================================="

# Step 1: Visual-Only Training
log "Step 1: Starting visual-only training..."
log "Using frozen CLIP visual encoder + trainable 2-layer classifier"

python LaFTer.py \
    --root "${DATA_ROOT}" \
    --trainer LaFTer \
    --dataset-config-file "configs/datasets/${DSET}.yaml" \
    --config-file "configs/trainers/text_cls/vit_b32.yaml" \
    --visual_only \
    --encoder-type "${ENCODER_TYPE}" \
    --arch "${ARCH}" \
    --head_lr "${HEAD_LR}" \
    --mm_epochs "${MM_EPOCHS}" \
    --mm_batch_size "${MM_BATCH_SIZE}" \
    --print_freq 50 \
    --workers 4 2>&1 | tee -a "${MAIN_LOG}"

# Check training completion status
if [ $? -eq 0 ]; then
    log "Training completed successfully!"
    log "Log file: ${MAIN_LOG}"
    log "===== Training Summary ====="
    log "Configuration: CLIP-frozen + 2-layer-classifier"
    log "Dataset: ${DSET}"
    log "Training completed at: $(date)"
    log "==========================="
else
    log "ERROR: Training failed! Check log for details."
    log "Log file: ${MAIN_LOG}"
    exit 1
fi

echo ""
echo "=== Visual-Only Training Complete ==="
echo "Dataset: ${DSET}"
echo "Architecture: ${ARCH}"
echo "Full log: ${MAIN_LOG}"
echo "================================="
