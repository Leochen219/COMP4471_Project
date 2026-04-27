# ImageNet Zero-shot 评估简短报告

日期：2026-04-27

## 1. 本次完成了什么

本次补齐了项目最终 benchmark：ImageNet validation set 上的 zero-shot classification。

使用的主模型 checkpoint：

```text
checkpoints/coco_3gpu_cliptext/best.pt
```

这个 checkpoint 来自 `configs/coco_3gpu_cliptext.yaml`，最佳 epoch 是 26。

## 2. ImageNet 数据准备

数据位置：

```text
/data/ydongbd/datasets/imagenet/
```

关键文件：

```text
/data/ydongbd/datasets/imagenet/ILSVRC2012_img_val.tar
/data/ydongbd/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz
/data/ydongbd/datasets/imagenet/imagenet_class_index.json
/data/ydongbd/datasets/imagenet/val/
```

最终整理结果：

```text
1000 class folders
50000 validation images
```

注意：第一次整理 ImageNet val 时，使用了 `imagenet_class_index.json` 的类别顺序去解释官方 validation ground-truth。后来检查发现官方 devkit 的 label id 顺序应以 `meta.mat` 中的 `ILSVRC2012_ID` 为准，两者顺序不同。第一次评估得到的 `Top-1 = 0.10%` / `Top-5 = 0.50%` 是标签错位导致的无效随机结果，不能引用。

现在已经修正 `scripts/prepare_imagenet_val.py`，用官方 devkit `meta.mat` 重新整理了 50000 张图片。

## 3. 使用的脚本

### 下载脚本

```text
scripts/download_imagenet_val.py
```

作用：

- 下载 ImageNet validation tar 和 devkit tar。
- 校验 MD5，避免损坏文件被继续使用。

### 整理脚本

```text
scripts/prepare_imagenet_val.py
```

作用：

- 读取 `ILSVRC2012_validation_ground_truth.txt`。
- 从官方 `meta.mat` 读取 `ILSVRC2012_ID -> WNID` 映射。
- 把 50000 张 validation 图片整理为 ImageFolder 格式：

```text
val/<wnid>/ILSVRC2012_val_xxxxxxxx.JPEG
```

### 评估脚本

```text
evaluate_imagenet.py
```

作用：

- 加载训练好的图文对齐模型。
- 用 prompt `a photo of a {label}.` 编码 1000 个 ImageNet 类别名称。
- 编码 ImageNet val 图片。
- 计算图像 embedding 与类别文本 embedding 的相似度。
- 输出 Top-1 和 Top-5 zero-shot accuracy。

## 4. 运行命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u evaluate_imagenet.py \
  --config configs/coco_3gpu_cliptext.yaml \
  --checkpoint checkpoints/coco_3gpu_cliptext/best.pt \
  --imagenet-root /data/ydongbd/datasets/imagenet/val \
  --backend imagefolder \
  --class-index-json /data/ydongbd/datasets/imagenet/imagenet_class_index.json \
  --batch-size 128 \
  --num-workers 8 \
  --output logs/imagenet_zero_shot.txt
```

评估设备：

```text
cuda:1
```

## 5. 最终 ImageNet 结果

结果文件：

```text
logs/imagenet_zero_shot.txt
```

最终有效结果：

| Metric | Value |
| --- | ---: |
| ImageNet val samples | 50000 |
| Classes | 1000 |
| Zero-shot Top-1 | 19.55% |
| Zero-shot Top-5 | 42.08% |

对比随机猜测：

- ImageNet-1K 随机 Top-1 约为 `0.10%`。
- ImageNet-1K 随机 Top-5 约为 `0.50%`。
- 当前模型达到 `19.55%` / `42.08%`，明显高于随机，说明模型具备真实的 ImageNet zero-shot 迁移能力。

## 6. 可以写进报告的结论

我们使用 COCO Captions 训练的 ViT-B/16 图像编码器和冻结 CLIP 文本编码器，在没有使用 ImageNet 标签监督微调的情况下，在 ImageNet validation set 上达到 `19.55%` Top-1 和 `42.08%` Top-5 zero-shot accuracy。该结果补齐了项目 TODO 中标记的最终 benchmark，也进一步支持模型学到了跨数据集的图文语义对齐能力。

## 7. 后续可选改进

可以尝试 prompt ensembling，例如同时使用：

```text
a photo of a {label}.
an image of a {label}.
a blurry photo of a {label}.
a cropped photo of a {label}.
```

这可能进一步提高 ImageNet zero-shot Top-1 / Top-5，但当前单 prompt 结果已经可以作为最终 benchmark 写入报告。
