#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/multimodal_dassl.sh <DATASET> <CAPTION_CSV> [ARCH] [BATCH] [EPOCHS] [SEED]
# Example: bash scripts/multimodal_dassl.sh Emotion6 caption/narracap_extended_Emotion6.csv ViT-B/32 32 20 7777

DATASET=${1:-Emotion6}
CAPTION=${2:-caption/narracap_extended_Emotion6.csv}
ARCH=${3:-ViT-B/32}
BATCH=${4:-32}
EPOCHS=${5:-3}
SEED=${6:-42}
USE_ADAPTER="false"
ADAPTER_TYPE="pfeiffer"       # pfeiffer (FFN-only) or houlsby (attn+FFN)
ADAPTER_REDUCTION=16          # bottleneck factor
ADAPTER_DROPOUT="0.1"
LR_ADAPTER="1e-4"
LR_HEAD="3e-4"

# Logging setup (write all outputs here)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/multimodal_new/${DATASET}"
mkdir -p "${LOG_DIR}"
MAIN_LOG="${LOG_DIR}/multimodal_${DATASET}_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${MAIN_LOG}"
}

# Decide dataset root according to adapter style
# - Datasets whose adapter expects ROOT to be the dataset folder itself (dataset_dir="")
#   -> use data/<DATASET>
# - Datasets whose adapter appends dataset_dir to ROOT
#   -> use data
case "$DATASET" in
  Twitter1|Twitter2|FI|Emoset)
    IMG_ROOT="data/${DATASET}"
    ;;
  Emotion6|FI_new|FI_Probing)
    IMG_ROOT="data"
    ;;
  *)
    # Fallback: assume subfolder
    IMG_ROOT="data/${DATASET}"
    ;;
esac
OUT_DIR=cache
OUT_PREFIX=${DATASET}

log "===== Multimodal (Dassl) Configuration ====="
log "Dataset: ${DATASET}"
log "Architecture: ${ARCH}"
log "Batch Size: ${BATCH}"
log "Epochs: ${EPOCHS}"
log "Caption CSV: ${CAPTION}"
log "Seed: ${SEED}"
log "Log file: ${MAIN_LOG}"
log "============================================"

# Make randomness as deterministic as practical from the shell side
export PYTHONHASHSEED=${SEED}
# Note: Full determinism also needs framework-side settings; the Python entry will set those.

log "=== Building pairs from Dassl dataset splits (${DATASET}) ==="
python build_pairs_dassl.py \
  --dataset "${DATASET}" \
  --img_root "${IMG_ROOT}" \
  --caption_file "${CAPTION}" \
  --out_dir "${OUT_DIR}" \
  --out_prefix "${OUT_PREFIX}" 2>&1 | tee -a "${MAIN_LOG}"

CAP_STEM=$(basename "${CAPTION}")
CAP_STEM=${CAP_STEM%.*}
TRAIN_PAIRS="${OUT_DIR}/${OUT_PREFIX}_${DATASET}_${CAP_STEM}_train.jsonl"
VAL_PAIRS="${OUT_DIR}/${OUT_PREFIX}_${DATASET}_${CAP_STEM}_val.jsonl"

log "=== Training Multimodal (Dassl-CLIP + Roberta) ==="
python LaFTer.py \
  --pipeline multimodal \
  --root "${IMG_ROOT}" \
  --arch "${ARCH}" \
  --encoder-type clip \
  --seed ${SEED} \
  --dataset_name "${DATASET}" \
  --train_pairs "${TRAIN_PAIRS}" \
  --val_pairs "${VAL_PAIRS}" \
  --mm_batch_size ${BATCH} \
  --mm_epochs ${EPOCHS} \
  --vit_lr 1e-6 \
  --bert_lr 5e-6 \
  $(if [[ "${USE_ADAPTER}" == "true" ]]; then
      echo --use_adapter \
            --adapter_type "${ADAPTER_TYPE}" \
            --adapter_reduction_factor "${ADAPTER_REDUCTION}" \
            --adapter_dropout "${ADAPTER_DROPOUT}" \
            --lr_adapter "${LR_ADAPTER}" \
            --lr_head "${LR_HEAD}"
  fi) \
  --head_lr 5e-4 \
  --fusion_dim 768 \
  --enable_wandb \
  --fusion_type standard \
  --mm_save_dir checkpoints/mm_${DATASET} 2>&1 | tee -a "${MAIN_LOG}"

log "Done. Checkpoints saved under checkpoints/mm_${DATASET}"
log "Full log: ${MAIN_LOG}"
