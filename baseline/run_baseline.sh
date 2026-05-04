#!/bin/bash
# baseline/run_baseline.sh
#
# Run the full ResNet-50 baseline pipeline:
#   1. Train on COCO Captions
#   2. COCO retrieval evaluation
#   3. CIFAR-100 zero-shot evaluation
#   4. ImageNet zero-shot evaluation
#
# Usage:
#   bash baseline/run_baseline.sh
#
# You can also run each step individually, e.g.:
#   python baseline/train.py --config baseline/config.yaml
#   python baseline/evaluate.py --config baseline/config.yaml --checkpoint checkpoints/baseline_resnet50/best.pt

set -e  # Exit on error

CONFIG="baseline/config.yaml"
CHECKPOINT_DIR="checkpoints/baseline_resnet50"
BEST_CKPT="${CHECKPOINT_DIR}/best.pt"

echo "=========================================="
echo "  Step 1: Training ResNet-50 Baseline"
echo "=========================================="
python baseline/train.py --config ${CONFIG}

echo ""
echo "=========================================="
echo "  Step 2: COCO Retrieval Evaluation"
echo "=========================================="
python baseline/evaluate.py --config ${CONFIG} --checkpoint ${BEST_CKPT}

echo ""
echo "=========================================="
echo "  Step 3: CIFAR-100 Zero-shot Evaluation"
echo "=========================================="
python baseline/evaluate_transfer.py \
    --config ${CONFIG} \
    --checkpoint ${BEST_CKPT} \
    --dataset cifar100 \
    --output logs/baseline_cifar100_zero_shot.txt

echo ""
echo "=========================================="
echo "  Step 4: ImageNet Zero-shot Evaluation"
echo "=========================================="
echo "Note: Update --imagenet-root and --class-index-json paths before running."
echo ""
echo "Example command:"
echo "  python baseline/evaluate_imagenet.py \\"
echo "      --config ${CONFIG} \\"
echo "      --checkpoint ${BEST_CKPT} \\"
echo "      --imagenet-root /path/to/imagenet/val \\"
echo "      --class-index-json /path/to/imagenet_class_index.json \\"
echo "      --output logs/baseline_imagenet_zero_shot.txt"

echo ""
echo "=========================================="
echo "  Baseline pipeline complete!"
echo "=========================================="
