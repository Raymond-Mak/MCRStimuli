#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/multimodal_dassl_batch.sh <DATASET> <CAPTION_DIR> [ARCH] [BATCH] [EPOCHS]
# Example:
#   bash scripts/multimodal_dassl_batch.sh Emotion6 caption/Emotion6 "ViT-B/32" 32 50
#
# Note:
# - Activate the desired environment before running this script (e.g., `conda activate LaFTer`).
# - This script simply loops through caption CSVs in <CAPTION_DIR> and calls
#   scripts/multimodal_dassl.sh for each file with the provided hyper-parameters.
# - Logs from each run are moved into log/multimodal_new/<DATASET>/<caption_name>/.

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/multimodal_dassl_batch.sh <DATASET> <CAPTION_DIR> [ARCH] [BATCH] [EPOCHS]"
  exit 1
fi

DATASET=$1
CAPTION_DIR=$2
ARCH=${3:-ViT-B/32}
BATCH=${4:-32}
EPOCHS=${5:-4}

if [[ ! -d "${CAPTION_DIR}" ]]; then
  echo "Caption directory '${CAPTION_DIR}' does not exist." >&2
  exit 1
fi

mapfile -t CAPTION_FILES < <(find "${CAPTION_DIR}" -maxdepth 1 -type f -name "*.csv" | sort)

if [[ ${#CAPTION_FILES[@]} -eq 0 ]]; then
  echo "No CSV caption files found under '${CAPTION_DIR}'." >&2
  exit 1
fi

BASE_LOG_DIR="log/multimodal_new/${DATASET}"

for CAPTION_PATH in "${CAPTION_FILES[@]}"; do
  CAPTION_FILE=$(basename "${CAPTION_PATH}")
  CAPTION_STEM=${CAPTION_FILE%.*}
  TARGET_LOG_DIR="${BASE_LOG_DIR}/${CAPTION_STEM}"

  echo "==================================================================="
  echo "Dataset   : ${DATASET}"
  echo "Caption   : ${CAPTION_PATH}"
  echo "Arch/Batch/Epochs: ${ARCH} / ${BATCH} / ${EPOCHS}"
  echo "Log folder: ${TARGET_LOG_DIR}"
  echo "==================================================================="

  bash scripts/multimodal_dassl.sh "${DATASET}" "${CAPTION_PATH}" "${ARCH}" "${BATCH}" "${EPOCHS}"

  mkdir -p "${TARGET_LOG_DIR}"
  LATEST_LOG=$(find "${BASE_LOG_DIR}" -maxdepth 1 -type f -name "multimodal_${DATASET}_*.log" | sort | tail -n 1)

  if [[ -n "${LATEST_LOG}" ]]; then
    LOG_BASENAME=$(basename "${LATEST_LOG}")
    LOG_TIME=${LOG_BASENAME#"multimodal_${DATASET}_"}
    LOG_TIME=${LOG_TIME%.log}
    DEST_LOG="${TARGET_LOG_DIR}/multimodal_${DATASET}_${CAPTION_STEM}_${LOG_TIME}.log"
    mv "${LATEST_LOG}" "${DEST_LOG}"
    echo "Saved log to ${DEST_LOG}"
  else
    echo "Warning: could not locate log file for ${CAPTION_FILE}" >&2
  fi
done
