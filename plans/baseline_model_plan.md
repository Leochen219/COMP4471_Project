# Baseline Model Plan: ResNet-50 Dual-Encoder (CLIP-style)

## 1. Current Model Overview

| Component | Architecture | Params | Trainable? |
|-----------|-------------|--------|------------|
| Image Encoder | ViT-B/16 (torchvision) | ~86M | Yes (lr=1e-5) |
| Text Encoder | openai/clip-vit-base-patch32 (HuggingFace) | ~63M | Frozen |
| Image Projection | Linear(768→256) → GELU → Linear(256→256) | ~0.26M | Yes |
| Text Projection | Linear(512→256) → GELU → Linear(256→256) | ~0.20M | Yes |
| Logit Scale | 1 scalar | 1 | Yes |
| **Total (all)** | | **~149M** | |
| **Total (trainable)** | | **~86.5M** | |

## 2. Proposed Baseline: ResNet-50 Dual-Encoder

### Architecture

This is the **same CLIP-style dual-encoder paradigm** as the current model, but with **ResNet-50** as the image backbone instead of ViT-B/16.

```
Image ──► ResNet-50 ──► Feature Vector (2048)
              │
              │  Image Projection
              │  Linear(2048→256) → GELU → Linear(256→256)
              │
              ▼
        Image Embedding [B, 256] ────┐
                                     │
                              Contrastive Loss
                              (InfoNCE)
                                     │
Text  ──► CLIP Text Encoder ─────────┤
        (frozen)                     │
              │                      │
              │  Text Projection     │
              │  Linear(512→256) → GELU → Linear(256→256)
              │
              ▼
        Text Embedding [B, 256] ─────┘
```

### Why this is a good baseline

| Reason | Detail |
|--------|--------|
| **Same paradigm, different backbone** | Tests whether ViT-B/16 is better than ResNet-50 for this task |
| **Minimal changes** | Almost identical code structure to current model |
| **Fast training** | ResNet-50 is lighter than ViT-B/16 |
| **Fast inference** | Dual-encoder = O(N) retrieval (same as current) |
| **Fair comparison** | Same loss, same training config, same evaluation |

### Parameter Breakdown

| Component | Specification | Params | Trainable? |
|-----------|--------------|--------|------------|
| ResNet-50 backbone | torchvision resnet50 | ~25.6M | Yes (fine-tune) |
| CLIP text encoder | openai/clip-vit-base-patch32 | ~63M | Frozen |
| Image projection | Linear(2048→256) → GELU → Linear(256→256) | ~0.6M | Yes |
| Text projection | Linear(512→256) → GELU → Linear(256→256) | ~0.2M | Yes |
| Logit scale | 1 scalar | 1 | Yes |
| **Total trainable** | | **~26.4M** | |
| **Total frozen** | | **~63M** | |
| **Total all** | | **~89.4M** | |

### Comparison with Current Model

| Aspect | Current (ViT-B/16) | Baseline (ResNet-50) |
|--------|-------------------|---------------------|
| **Image backbone** | ViT-B/16 (transformer) | ResNet-50 (CNN) |
| **Backbone params** | ~86M | ~25.6M |
| **Trainable params** | ~86.5M | ~26.4M |
| **Total params** | ~149M | ~89.4M |
| **Loss** | Symmetric InfoNCE | Symmetric InfoNCE (same) |
| **Embedding dim** | 256 | 256 (same) |
| **Text encoder** | Frozen CLIP | Frozen CLIP (same) |
| **Training paradigm** | Dual-encoder contrastive | Dual-encoder contrastive (same) |

## 3. Training Resources

### Dataset: COCO Captions (same as current)

| Item | Value |
|------|-------|
| Training set | COCO train2017 (~118K images) |
| Validation set | COCO val2017 (~5K images) |
| Image size | 224×224 |
| Text length | 77 tokens |

### Training Config

```yaml
model:
  image_encoder_name: "resnet50"
  pretrained: true
  text_encoder_name: "openai/clip-vit-base-patch32"
  embed_dim: 256
  text_max_length: 77
  freeze_text_encoder: true

training:
  epochs: 30
  batch_size: 128
  lr: 3.0e-4                     # Projection layers LR
  image_encoder_lr: 1.0e-5       # ResNet-50 fine-tune LR
  weight_decay: 0.01
  warmup_steps: 1000
  max_grad_norm: 1.0
  device: "cuda"
```

### Memory & Time Estimate

| Aspect | Current (ViT-B/16) | Baseline (ResNet-50) |
|--------|-------------------|---------------------|
| **Trainable params** | ~86.5M | ~26.4M |
| **Memory per GPU** | ~8-10 GB | **~4-6 GB** |
| **Training time (3×GPU)** | ~4 hours | **~2-3 hours** |
| **Inference (COCO val)** | ~2 min | **~1-2 min** (same O(N)) |

## 4. File Structure: All Changes in `baseline/` Folder

```
baseline/
├── __init__.py                    # Exports BaselineModel
├── dual_encoder_model.py          # ResNet-50 + projection (CLIP-style)
├── config.yaml                    # Training config
├── train.py                       # Training script (same InfoNCE loss)
├── evaluate.py                    # COCO retrieval eval
├── evaluate_imagenet.py           # ImageNet zero-shot eval
├── evaluate_transfer.py           # CIFAR transfer eval
├── data.py                        # Dataset (reuses existing)
├── utils.py                       # Loss functions, metrics
└── run_baseline.sh                # Shell script
```

### [`baseline/dual_encoder_model.py`](baseline/dual_encoder_model.py)

```python
class BaselineModel(nn.Module):
    """
    ResNet-50 dual-encoder model (CLIP-style).
    
    Architecture:
      - ResNet-50 image encoder (fine-tuned)
      - Frozen CLIP text encoder
      - Trainable projection heads → 256-dim shared embedding
      - Symmetric InfoNCE loss
    
    Interface (same as CLIPModel):
        encode_image(images)                              → [B, 256] L2-normalized
        encode_text(input_ids, attention_mask)             → [B, 256] L2-normalized
        forward(images, input_ids, attention_mask)         → scalar loss
        get_param_groups(lr, image_encoder_lr)             → list[dict]
    """
```

This is essentially a copy of [`models/clip_model.py`](models/clip_model.py) but with:
- `build_image_encoder("resnet50")` instead of `build_image_encoder("vit_base_patch16_224")`
- Image projection: `Linear(2048→256)` instead of `Linear(768→256)`
- Same text encoder, same loss, same everything else

### [`baseline/config.yaml`](baseline/config.yaml)

Same structure as [`configs/default.yaml`](configs/default.yaml) but with:
- `image_encoder_name: "resnet50"`
- `save_dir: "checkpoints/baseline_resnet50"`

### [`baseline/train.py`](baseline/train.py)

Nearly identical to [`train.py`](train.py) — just imports `BaselineModel` instead of `CLIPModel`.

### [`baseline/evaluate.py`](baseline/evaluate.py)

Nearly identical to [`evaluate.py`](evaluate.py) — just imports `BaselineModel`.

### [`baseline/evaluate_imagenet.py`](baseline/evaluate_imagenet.py)

Nearly identical to [`evaluate_imagenet.py`](evaluate_imagenet.py) — just imports `BaselineModel`.

### [`baseline/evaluate_transfer.py`](baseline/evaluate_transfer.py)

Nearly identical to [`evaluate_transfer.py`](evaluate_transfer.py) — just imports `BaselineModel`.

### [`baseline/run_baseline.sh`](baseline/run_baseline.sh)

```bash
# Step 1: Train
python baseline/train.py --config baseline/config.yaml

# Step 2: COCO retrieval eval
python baseline/evaluate.py --config baseline/config.yaml --checkpoint checkpoints/baseline_resnet50/best.pt

# Step 3: CIFAR-100 zero-shot
python baseline/evaluate_transfer.py --config baseline/config.yaml --checkpoint checkpoints/baseline_resnet50/best.pt

# Step 4: ImageNet zero-shot
python baseline/evaluate_imagenet.py --config baseline/config.yaml --checkpoint checkpoints/baseline_resnet50/best.pt
```

## 5. What Stays Unchanged

- `models/clip_model.py` — unchanged
- `models/image_encoder.py` — unchanged (but imported by baseline)
- `models/text_encoder.py` — unchanged
- `train.py` — unchanged
- `evaluate.py` — unchanged
- `configs/default.yaml` — unchanged
- All other existing files — unchanged

## 6. Expected Results Comparison

| Metric | Current (ViT-B/16) | Baseline (ResNet-50) |
|--------|-------------------|---------------------|
| **Image backbone** | ViT-B/16 (86M) | ResNet-50 (25.6M) |
| **Trainable params** | ~86.5M | ~26.4M |
| **COCO R@1 (I2T)** | ~24.9% | **~15-20%** |
| **COCO R@5 (I2T)** | ~54.2% | **~40-50%** |
| **COCO R@10 (I2T)** | ~68.0% | **~55-65%** |
| **CIFAR-100 Zero-shot** | 37.2% | **~25-33%** |
| **ImageNet Zero-shot** | 19.6% | **~12-17%** |
| **Training time (3×GPU)** | ~4 hours | **~2-3 hours** |
| **Inference (COCO val)** | ~2 min | **~1-2 min** |

The baseline is expected to perform lower because ResNet-50 has ~3.4× fewer parameters than ViT-B/16. But this is exactly the point — it quantifies **how much the larger ViT backbone contributes** to performance.

## 7. Summary

| Item | Value |
|------|-------|
| **Architecture** | ResNet-50 dual-encoder (CLIP-style) |
| **Image backbone** | ResNet-50 (torchvision, ImageNet-pretrained) |
| **Text backbone** | Frozen CLIP text encoder (same as current) |
| **Loss** | Symmetric InfoNCE (same as current) |
| **Embedding dim** | 256 (same as current) |
| **Trainable params** | ~26.4M |
| **Total params** | ~89.4M |
| **Training data** | COCO Captions (same as current) |
| **Training resources** | 3× GPU, ~2-3 hours, ~4-6 GB/GPU |
| **All code in** | `baseline/` folder — no existing files modified |
