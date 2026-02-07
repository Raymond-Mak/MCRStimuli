#!/bin/bash

# Text-only BERT/Roberta training helper
# Usage:
#   ./scripts/text_only.sh <dataset>
# Example:
#   ./scripts/text_only.sh Twitter1

set -euo pipefail

show_usage() {
    cat <<'EOF'
Usage: ./scripts/text_only.sh <dataset>

Arguments:
  dataset  Emotion6 | Twitter1 | Twitter2 (matches cache/<dataset>_* files)

All hyper-parameters are fixed inside scripts/text_only.sh; edit the script if you need to tweak them.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    show_usage
    exit 0
fi

if [[ $# -lt 1 ]]; then
    echo "ERROR: dataset argument is required."
    show_usage
    exit 1
fi

DATASET="$1"

# ===== Default hyper-parameters for ablation runs =====
CAPTION_VARIANT="full"        # change here to train on other caption variants
MODEL_NAME="roberta-large"    # replace with bert-base-uncased if desired
EPOCHS=50
BATCH_SIZE=32
LR="1e-5"
MAX_LEN=196
WARMUP_RATIO="0.1"
WEIGHT_DECAY="0.01"
SEED=42
SAVE_MODEL="false"             # set to "false" to discard weights after each run
# Adapter hyper-parameters (set USE_ADAPTER to "true" to enable)
USE_ADAPTER="false"
ADAPTER_TYPE="houlsby"       # pfeiffer (FFN-only) or houlsby (attn+FFN)
ADAPTER_REDUCTION=16          # bottleneck factor
ADAPTER_DROPOUT="0.1"
LR_ADAPTER="1e-4"
LR_HEAD="3e-4"                # 修改为 1e-4 (用户要求)
# ======================================================
# Freeze backbone mode (set FREEZE_BACKBONE to "true" to train ONLY classifier)
FREEZE_BACKBONE="false"       # true = linear probing (仅分类器), false = full/adapter finetune
# ======================================================

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
CAPTION=${2:-caption/${DATASET}/gpt_narracap_extended_${DATASET}.csv}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/multimodal_new/${DATASET}"
MAIN_LOG="${LOG_DIR}/multimodal_${DATASET}_${TIMESTAMP}.log"

#  "=== Building pairs from Dassl dataset splits (${DATASET}) ==="
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/cache"

declare -A VARIANT_SUFFIXES=(
    ["full"]=""
    ["cleaned"]="_cleaned"
    ["emotion_only"]="_emotion_only"
    ["highlevel_feature_only"]="_highlevel_feature_only"
    ["lowlevel_feature_only"]="_lowlevel_feature_only"
    ["midhighfeature_mixed"]="_midhighfeature_mixed"
    ["midlevel_feature_only"]="_midlevel_feature_only"
    ["parts"]="_parts"
    ["reasoning_only"]="_reasoning_only"
    ["without_last_sentence"]="_without_last_sentence"
)

if [[ -v VARIANT_SUFFIXES["$CAPTION_VARIANT"] ]]; then
    VARIANT_SUFFIX="${VARIANT_SUFFIXES[$CAPTION_VARIANT]}"
else
    VARIANT_SUFFIX="_${CAPTION_VARIANT}"
fi

BASE_NAME="${DATASET}_${DATASET}_gpt_narracap_extended_${DATASET}${VARIANT_SUFFIX}"
TRAIN_FILE="${CACHE_DIR}/${BASE_NAME}_train.jsonl"
VAL_FILE="${CACHE_DIR}/${BASE_NAME}_val.jsonl"

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "ERROR: Could not find ${TRAIN_FILE} or ${VAL_FILE}."
    echo "Available options in ${CACHE_DIR}:"
    ls "${CACHE_DIR}" | grep "^${DATASET}_${DATASET}_" || true
    exit 1
fi

MODEL_ID=$(basename "${MODEL_NAME}")
RUN_ID="${DATASET}_${CAPTION_VARIANT}_${MODEL_ID}_$(date +%Y%m%d_%H%M%S)"
if [[ "${SAVE_MODEL}" == "true" ]]; then
    OUTPUT_DIR="${REPO_ROOT}/checkpoints/text_only/${RUN_ID}"
else
    OUTPUT_DIR="$(mktemp -d "${REPO_ROOT}/tmp_text_only_output.XXXXXX")"
fi
LOG_DIR="${REPO_ROOT}/log/text_only/${DATASET}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/text_only_${RUN_ID}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "===== Text-only training ====="
log "Dataset: ${DATASET}"
log "Caption variant: ${CAPTION_VARIANT} (${BASE_NAME})"
log "Model: ${MODEL_NAME}"

# 确定训练模式并记录
if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
    log "Mode: FREEZE BACKBONE (classifier only, lr=${LR_HEAD})"
elif [[ "${USE_ADAPTER}" == "true" ]]; then
    log "Mode: ADAPTER (${ADAPTER_TYPE}, adapter_lr=${LR_ADAPTER}, head_lr=${LR_HEAD})"
else
    log "Mode: FULL FINETUNING (lr=${LR})"
fi

log "Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Max Len: ${MAX_LEN}"
log "Warmup ratio: ${WARMUP_RATIO} | Weight decay: ${WEIGHT_DECAY} | Seed: ${SEED}"
log "Train file: ${TRAIN_FILE}"
log "Val file: ${VAL_FILE}"
log "Output dir: ${OUTPUT_DIR}"
log "Log file: ${LOG_FILE}"
log "=============================="

cd "${REPO_ROOT}"

python runBert_simple.py \
    --train_file "${TRAIN_FILE}" \
    --validation_file "${VAL_FILE}" \
    --model_name_or_path "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --text_column "text" \
    --label_column "label" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --max_length "${MAX_LEN}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --seed "${SEED}" \
    $(if [[ "${USE_ADAPTER}" == "true" ]]; then
          echo --use_adapter \
               --adapter_type "${ADAPTER_TYPE}" \
               --adapter_reduction_factor "${ADAPTER_REDUCTION}" \
               --adapter_dropout "${ADAPTER_DROPOUT}" \
               --lr_adapter "${LR_ADAPTER}" \
               --lr_head "${LR_HEAD}"
      fi) \
    $(if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
          echo --freeze_backbone --lr_head "${LR_HEAD}"
      fi) \
    2>&1 | tee -a "${LOG_FILE}"

STATUS=$?

if [[ ${STATUS} -eq 0 ]]; then
    log "Training completed successfully."
    if [[ "${SAVE_MODEL}" == "true" ]]; then
        log "Artifacts saved to ${OUTPUT_DIR}"
    else
        log "SAVE_MODEL is false; deleting ${OUTPUT_DIR}"
        rm -rf "${OUTPUT_DIR}"
    fi
else
    log "Training failed with exit code ${STATUS}."
    if [[ "${SAVE_MODEL}" != "true" ]]; then
        rm -rf "${OUTPUT_DIR}"
    fi
    exit ${STATUS}
fi

log "================================"
