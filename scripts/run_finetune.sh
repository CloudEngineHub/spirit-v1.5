#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parameters (no defaults)
: "${PRETRAINED_PATH:?Please set PRETRAINED_PATH}"
: "${DATA_ROOT:?Please set DATA_ROOT}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "[ERROR] PRETRAINED_PATH not found: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_PATH}/model.safetensors" ]]; then
  echo "[ERROR] model.safetensors not found in PRETRAINED_PATH: ${CKPT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAINED_PATH}/config.json" ]]; then
  echo "[ERROR] config.json not found in PRETRAINED_PATH: ${CKPT_PATH}" >&2
  exit 1
fi

torchrun --nproc_per_node=${NUM_GPUS:-8} \
    "${REPO_ROOT}/train.py" \
    --data_root "${DATA_ROOT:?DATA_ROOT must be set}" \
    --pretrained_path "${PRETRAINED_PATH:?PRETRAINED_PATH must be set}" \
    --output_dir "${OUTPUT_DIR:-${REPO_ROOT}/outputs}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --max_train_steps "${MAX_TRAIN_STEPS:-5}" \
    --log_interval "${LOG_INTERVAL:-1}" \
    --save_steps "${SAVE_STEPS:-2500}" \
    --num_workers "${NUM_WORKERS:-32}" \
    --prefetch_factor "${PREFETCH_FACTOR:-8}" \
    --wandb_mode "${WANDB_MODE:-disabled}"
