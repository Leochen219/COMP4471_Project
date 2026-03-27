# CLIP Text Encoder Training Report

## 1. Objective

This experiment studies whether replacing the project's original randomly initialized text encoder with a frozen pretrained CLIP text encoder improves image-text alignment quality and cross-dataset transferability.

The final system trains:

- an ImageNet-pretrained `ViT-B/16` image encoder
- a frozen `openai/clip-vit-base-patch32` text encoder
- trainable projection heads for image and text embeddings
- a trainable temperature parameter for contrastive alignment

## 2. Training Pipeline

### 2.1 Dataset

- Training set: COCO `train2017`
- Validation set: COCO `val2017`
- Image resolution: `224 x 224`
- Text length: `77`

### 2.2 Model Setup

- Image encoder: `vit_base_patch16_224`
- Image encoder initialization: pretrained
- Text encoder: `openai/clip-vit-base-patch32`
- Text encoder policy: frozen
- Text projection head: trainable
- Shared embedding dimension: `256`

### 2.3 Optimization

- Epochs: `30`
- Global batch size: `96`
- GPUs: `[1, 2, 3]`
- CLIP head learning rate: `3e-4`
- Image encoder learning rate: `1e-5`
- Weight decay: `0.01`
- Warmup steps: `1000`
- Gradient clipping: `1.0`
- Scheduler: warmup + cosine decay

### 2.4 Runtime

- Training start: `2026-03-26 20:52:34 HKT`
- Training end: `2026-03-27 01:00:38 HKT`
- Total training time: about `4.13` hours

## 3. Evaluation Protocol

### 3.1 Validation During Training

During training, validation is measured by the same symmetric contrastive loss used for optimization. This is the `val_loss` reported after every epoch.

### 3.2 Retrieval Evaluation After Training

The post-training evaluation is a strict image-text retrieval setup on COCO `val2017`.

- Each image is paired with only the first caption in the annotation list.
- The metric assumes `image_i <-> text_i` is the only positive pair.
- Under this protocol, `Recall@1` is numerically equal to `top-1 retrieval accuracy`.

This protocol is stricter than the standard COCO 5-caption retrieval benchmark, so the reported retrieval scores should be interpreted as conservative.

### 3.3 Transfer Evaluation

To test transfer learning ability, the best checkpoint is evaluated zero-shot on `CIFAR-100` classification.

- Dataset split: `CIFAR-100 test`
- Classes: `100`
- Prompt template: `a photo of a {class}.`
- Metric: `Top-1` and `Top-5` classification accuracy

## 4. Main Results

### 4.1 Training Summary

| Metric | Value |
| --- | ---: |
| Completed epochs | 30 |
| Best epoch | 26 |
| Best validation loss | 0.4421 |
| Final epoch train loss | 0.1043 |
| Final epoch validation loss | 0.4478 |

The best checkpoint was selected from epoch 26 and stored locally as `checkpoints/coco_3gpu_cliptext/best.pt`.

### 4.2 COCO Retrieval Results

| Metric | Value |
| --- | ---: |
| Image-to-Text `R@1` | 24.88 |
| Image-to-Text `R@5` | 54.22 |
| Image-to-Text `R@10` | 68.00 |
| Text-to-Image `R@1` | 24.58 |
| Text-to-Image `R@5` | 54.28 |
| Text-to-Image `R@10` | 68.18 |
| Mean Recall | 49.02 |

### 4.3 Retrieval Accuracy / Top-5 Recall

Because this evaluation uses one positive target per query, `accuracy@1` is identical to `R@1`.

| Metric | Value |
| --- | ---: |
| Image-to-Text retrieval `accuracy@1` | 24.88 |
| Text-to-Image retrieval `accuracy@1` | 24.58 |
| Mean retrieval `accuracy@1` | 24.73 |
| Image-to-Text retrieval `recall@5` | 54.22 |
| Text-to-Image retrieval `recall@5` | 54.28 |

These numbers show that the model usually ranks the correct match near the front of the candidate list, and roughly one quarter of queries retrieve the correct match at the first position under the strict one-caption protocol.

## 5. Transfer Learning Result

### 5.1 CIFAR-100 Zero-Shot Classification

| Metric | Value |
| --- | ---: |
| Top-1 accuracy | 37.21 |
| Top-5 accuracy | 67.03 |

Interpretation:

- Random guess on CIFAR-100 gives about `1%` Top-1 and `5%` Top-5.
- The model reaches `37.21%` / `67.03%` without supervised fine-tuning on CIFAR-100.
- This indicates that the image encoder learned from COCO caption alignment retains meaningful semantic transfer ability beyond the training dataset.

## 6. Discussion

### 6.1 What Worked

- Replacing the old custom text encoder with a pretrained CLIP text encoder produced stable contrastive training.
- Validation loss improved steadily through most of training and reached the best value at epoch 26.
- Retrieval results are strong enough to show clear image-text alignment.
- Transfer performance on CIFAR-100 confirms that the learned representation is not limited to COCO caption matching.

### 6.2 Current Limitations

- The retrieval protocol is stricter than the standard COCO evaluation because only one caption per image is kept as positive during testing.
- The report currently uses a single prompt template for zero-shot transfer evaluation; prompt ensembling may improve CIFAR-100 accuracy further.
- No standard zero-shot classification benchmark such as ImageNet has been run yet.

## 7. Conclusion

The CLIP text encoder version is a clear improvement over the previous setup conceptually and experimentally.

- It trains stably on COCO with best validation loss `0.4421`.
- It reaches about `24.7%` mean top-1 retrieval accuracy under a strict retrieval protocol.
- It transfers to a different dataset, achieving `37.21%` Top-1 and `67.03%` Top-5 zero-shot accuracy on CIFAR-100.

Overall, the experiment supports the claim that pretrained text semantics improve cross-modal alignment and provide useful transferability beyond the original training distribution.

## 8. Artifacts

- Training config committed to the repo: `configs/coco_3gpu_cliptext.yaml`
- Transfer evaluation script committed to the repo: `evaluate_transfer.py`
- Report committed to the repo: `reports/cliptext_experiment_report.md`
- Local training log not committed: `logs/coco_3gpu_cliptext_train.log`
- Local retrieval evaluation log not committed: `logs/coco_3gpu_cliptext_eval.log`
- Local run summary not committed: `logs/coco_3gpu_cliptext_summary.txt`
- Local transfer evaluation output not committed: `logs/cifar100_transfer_eval.txt`
- Local best checkpoint not committed: `checkpoints/coco_3gpu_cliptext/best.pt`
