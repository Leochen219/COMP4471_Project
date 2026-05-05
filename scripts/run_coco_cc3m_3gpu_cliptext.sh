#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
python -u train.py --config configs/coco_cc3m_3gpu_cliptext.yaml \
  > logs/coco_cc3m_3gpu_cliptext_train.log 2>&1
