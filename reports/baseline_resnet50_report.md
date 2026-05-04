# Baseline Model Report: ResNet-50 Dual-Encoder (CLIP-style)

## 1. Objective

This report documents the **baseline model** for the COMP4471 image-text alignment project. The baseline replaces the main model's ViT-B/16 image encoder with a **ResNet-50** backbone, keeping all other components identical. The goal is to establish a lower-bound performance reference and quantify the improvement gained by using a transformer-based image encoder (ViT-B/16) over a CNN-based one (ResNet-50).

## 2. Network Architecture

### 2.1 Design Overview

The baseline follows the same **CLIP-style dual-encoder paradigm** as the main model:

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
                                (Symmetric InfoNCE)
                                       │
Text  ──► CLIP Text Encoder ───────────┤
          (frozen)                     │
                │                      │
                │  Text Projection     │
                │  Linear(512→256) → GELU → Linear(256→256)
                │
                ▼
          Text Embedding [B, 256] ─────┘
```

### 2.2 Component Details

| Component | Specification | Params | Trainable? |
|-----------|--------------|--------|------------|
| Image encoder | ResNet-50 (torchvision, ImageNet-pretrained) | ~25.6M | Yes (fine-tune, lr=1e-5) |
| Text encoder | `openai/clip-vit-base-patch32` (HuggingFace) | ~63M | Frozen |
| Image projection | Linear(2048→256) → GELU → Linear(256→256) | ~0.6M | Yes (lr=3e-4) |
| Text projection | Linear(512→256) → GELU → Linear(256→256) | ~0.2M | Yes (lr=3e-4) |
| Logit scale | 1 scalar (initialized to `ln(1/0.07)`) | 1 | Yes (lr=3e-4) |
| **Total** | | **~89.4M** | |
| **Trainable** | | **~26.4M** | |
| **Frozen** | | **~63M** | |

### 2.3 Key Design Decisions

- **Same text encoder**: Uses the same frozen CLIP text encoder as the main model, ensuring any performance difference is attributable solely to the image backbone.
- **Same loss function**: Symmetric InfoNCE loss with learnable temperature, identical to the main model.
- **Same embedding dimension**: 256-dimensional shared embedding space.
- **Differential learning rates**: Image encoder fine-tuned with a smaller LR (1e-5) to preserve pretrained features; projection heads trained with a higher LR (3e-4).

### 2.4 Comparison with Main Model (ViT-B/16)

| Aspect | Main Model (ViT-B/16) | Baseline (ResNet-50) |
|--------|----------------------|---------------------|
| Image backbone | ViT-B/16 (transformer) | ResNet-50 (CNN) |
| Backbone params | ~86M | ~25.6M |
| Trainable params | ~86.5M | ~26.4M |
| Total params | ~149M | ~89.4M |
| Loss | Symmetric InfoNCE | Symmetric InfoNCE (same) |
| Embedding dim | 256 | 256 (same) |
| Text encoder | Frozen CLIP | Frozen CLIP (same) |

## 3. Training Setup

### 3.1 Dataset

| Item | Value |
|------|-------|
| Training set | COCO train2017 (subsampled to 5,000 images) |
| Validation set | COCO val2017 (5,000 images) |
| Image size | 224 × 224 |
| Text length | 77 tokens (CLIP tokenizer) |
| Captions per image | 5 (randomly selected one per training step) |

### 3.2 Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 30 |
| Batch size | 64 |
| Optimizer | AdamW |
| Image encoder LR | 1.0e-5 |
| Projection head LR | 3.0e-4 |
| Weight decay | 0.01 |
| Warmup steps | 40 (~1 epoch) |
| Gradient max norm | 1.0 |
| LR scheduler | Warmup + cosine decay |
| GPU | 1 × NVIDIA GPU (CUDA) |

### 3.3 Training Summary

| Metric | Value |
|--------|-------|
| Completed epochs | 30 |
| Best epoch | 29 (last) |
| Best validation loss | 0.9304 |
| Total training steps | 2,340 (78 steps/epoch × 30 epochs) |

## 4. Evaluation Results

### 4.1 COCO Retrieval (Strict One-Caption Protocol)

Evaluation on COCO `val2017` (5,000 images, one caption per image as positive target).

#### Image → Text Retrieval

| Metric | Baseline (ResNet-50) | Main Model (ViT-B/16) | Gap |
|--------|:--------------------:|:---------------------:|:---:|
| **R@1 (Top-1)** | **11.84** | **24.88** | −13.04 |
| **R@5 (Top-5)** | **33.38** | **54.22** | −20.84 |
| R@10 | 46.56 | 68.00 | −21.44 |

#### Text → Image Retrieval

| Metric | Baseline (ResNet-50) | Main Model (ViT-B/16) | Gap |
|--------|:--------------------:|:---------------------:|:---:|
| **R@1 (Top-1)** | **12.38** | **24.58** | −12.20 |
| **R@5 (Top-5)** | **33.50** | **54.28** | −20.78 |
| R@10 | 46.34 | 68.18 | −21.84 |

#### Summary Metrics

| Metric | Baseline (ResNet-50) | Main Model (ViT-B/16) |
|--------|:--------------------:|:---------------------:|
| Mean Recall@1 | 12.11 | 24.73 |
| Mean Recall@5 | 33.44 | 54.25 |
| Mean Recall@10 | 46.45 | 68.09 |
| **Mean Recall (all)** | **30.67** | **49.02** |

### 4.2 Interpretation

1. **Above random baseline**: Random chance on 5,000 candidates is ~0.02% for R@1. The baseline's ~12% R@1 confirms it has learned meaningful image-text alignment.

2. **Consistent gap**: The ViT-B/16 model outperforms ResNet-50 by ~13 points for R@1 and ~21 points for R@5/R@10. This gap is consistent across both image→text and text→image directions.

3. **Architecture matters**: The performance difference is attributable to two factors:
   - **Model capacity**: ViT-B/16 (86M params) has 3.4× more parameters than ResNet-50 (25.6M), allowing it to learn richer visual representations.
   - **Representation quality**: Vision Transformers capture global contextual relationships through self-attention, while CNNs are inherently local. For image-text alignment tasks that require understanding scene-level semantics, global context is critical.

4. **Data limitation**: Both models were trained on only 5,000 COCO images (subsampled from 118K). Training on the full dataset would likely improve both models, but the relative gap may persist.

## 5. Conclusion

The ResNet-50 baseline achieves **11.84%** image→text R@1 and **12.38%** text→image R@1 on COCO retrieval, well above random chance. Compared to the ViT-B/16 main model (24.88% / 24.58%), the baseline demonstrates that:

- A CNN-based image encoder can learn meaningful cross-modal alignment, but
- Transformer-based architectures (ViT) provide significantly better visual representations for this task,
- The performance gap (~13 points R@1, ~21 points R@5) quantifies the benefit of using ViT over ResNet-50 for image-text contrastive learning.

## 6. Artifacts

| Artifact | Path |
|----------|------|
| Model definition | [`baseline/dual_encoder_model.py`](../baseline/dual_encoder_model.py) |
| Training config | [`baseline/config.yaml`](../baseline/config.yaml) |
| Training script | [`baseline/train.py`](../baseline/train.py) |
| COCO evaluation script | [`baseline/evaluate.py`](../baseline/evaluate.py) |
| Best checkpoint | `checkpoints/baseline_resnet50/best.pt` (epoch 29) |
| Latest checkpoint | `checkpoints/baseline_resnet50/latest.pt` (epoch 29) |
