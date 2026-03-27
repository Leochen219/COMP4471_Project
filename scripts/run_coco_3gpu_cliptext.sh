#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/ydongbd/COMP4471_Project"
CONFIG="configs/coco_3gpu_cliptext.yaml"
CHECKPOINT_DIR="checkpoints/coco_3gpu_cliptext"
TRAIN_LOG="logs/coco_3gpu_cliptext_train.log"
EVAL_LOG="logs/coco_3gpu_cliptext_eval.log"
SUMMARY_LOG="logs/coco_3gpu_cliptext_summary.txt"

cd "${ROOT_DIR}"

mkdir -p logs "${CHECKPOINT_DIR}"

# The CLIP tokenizer/text model have been cached locally already.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u train.py --config "${CONFIG}" 2>&1 | tee "${TRAIN_LOG}"
python -u evaluate.py --config "${CONFIG}" --checkpoint "${CHECKPOINT_DIR}/best.pt" 2>&1 | tee "${EVAL_LOG}"

python - <<'PY' > "${SUMMARY_LOG}"
from pathlib import Path
import re

train_log = Path("logs/coco_3gpu_cliptext_train.log").read_text(encoding="utf-8")
eval_log = Path("logs/coco_3gpu_cliptext_eval.log").read_text(encoding="utf-8")

epoch_matches = re.findall(
    r"Epoch (\d+) 完毕 \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)",
    train_log,
)

if not epoch_matches:
    raise SystemExit("No epoch summaries found in training log.")

last_epoch, last_train_loss, last_val_loss = epoch_matches[-1]
best_epoch, _, best_val_loss = min(epoch_matches, key=lambda item: float(item[2]))

metric_matches = re.findall(r"\] +([a-z0-9_@]+): ([0-9.]+)", eval_log, flags=re.IGNORECASE)

print("config: configs/coco_3gpu_cliptext.yaml")
print(f"last_epoch: {last_epoch}")
print(f"last_train_loss: {last_train_loss}")
print(f"last_val_loss: {last_val_loss}")
print(f"best_val_epoch: {best_epoch}")
print(f"best_val_loss: {best_val_loss}")

for name, value in metric_matches:
    print(f"{name}: {value}")
PY
