# COMP4471 Project 后续工作说明

本文档记录 2026-04-27 对项目训练结果、评估脚本、可视化图表和后续待办的整理，方便组员快速理解当前项目状态和每个新增文件的作用。

## 1. 当前主模型结果

目前建议作为最终主要结果使用的 checkpoint 是：

```text
checkpoints/coco_3gpu_cliptext/best.pt
```

这个模型来自 `configs/coco_3gpu_cliptext.yaml` 配置，对 COCO train2017 训练 30 个 epoch，最佳模型出现在 epoch 26。

主要结果如下：

| 评估项 | 数值 |
| --- | ---: |
| Best epoch | 26 |
| Best validation loss | 0.4421 |
| Final train loss | 0.1043 |
| Final validation loss | 0.4478 |
| COCO image-to-text R@1 | 24.88 |
| COCO image-to-text R@5 | 54.22 |
| COCO text-to-image R@1 | 24.58 |
| COCO text-to-image R@5 | 54.28 |
| COCO mean recall | 49.02 |
| CIFAR-100 zero-shot Top-1 | 37.21% |
| CIFAR-100 zero-shot Top-5 | 67.03% |
| ImageNet zero-shot Top-1 | 19.55% |
| ImageNet zero-shot Top-5 | 42.08% |

这些结果说明模型已经学到了一定的图文语义对齐能力：在严格的一图一文本检索协议下，Top-1 大约有四分之一能直接匹配正确，Top-5 超过一半能匹配正确；同时 CIFAR-100 和 ImageNet zero-shot 结果都明显高于随机猜测。

## 2. 新增/整理的图片

图片都放在：

```text
reports/figures/
```

### 2.1 Loss 曲线

```text
reports/figures/coco_3gpu_cliptext_loss_curve.png
```

作用：

- 展示 COCO 正式训练过程中 train loss 和 validation loss 的变化。
- 可以直接放入报告或 presentation，用来说明训练过程稳定收敛。
- 图中标注了 best validation loss：epoch 26，val loss = 0.4421。

含义：

- 蓝线是训练 loss，下降很快，说明模型在训练集上持续学习。
- 红线是验证 loss，整体下降后趋于平稳，中后期有轻微波动。
- 最佳模型不是最后一个 epoch，而是 epoch 26，因此最终评估应使用 `best.pt`，而不是 `latest.pt`。

### 2.2 相似度热力图

```text
reports/figures/coco_similarity_heatmap.png
```

作用：

- 展示一小批 COCO validation 图片和 caption 之间的相似度矩阵。
- 行表示图片，列表示文本 caption。
- 黑框代表正确配对的图文。

含义：

- 如果模型图文对齐学得好，黑框所在的对角线区域应该相对更红。
- 图中能看到不少对角线位置相似度较高，说明模型能把正确图片和对应文本拉近。
- 也能看到部分非对角线位置较红，说明相似场景或容易混淆的文本仍会被模型认为相近。

### 2.3 Top-k 检索样例

```text
reports/figures/coco_topk_retrieval_examples.png
```

作用：

- 展示若干张 query image 的 Top-5 文本检索结果。
- 绿色框表示正确 caption。
- 可以用于 qualitative analysis，说明模型具体检索效果。

含义：

- 当前展示的前 4 个 query image 中，Top-1 都是正确 caption。
- 例子包括 kitchen/pizza、kitchen/table、street/shopping cart、skate park 等。
- Top-5 中有时会混入不相关 caption，例如 kitchen 图可能召回 toilet/bucket 类文本，这说明模型仍有语义混淆空间。

## 3. 新增/整理的脚本

### 3.1 ImageNet zero-shot 评估脚本

```text
evaluate_imagenet.py
```

作用：

- 用训练好的图文对齐模型在 ImageNet validation set 上做 zero-shot classification。
- 构造 prompt，例如：

```text
a photo of a {label}.
```

- 将 1000 个 ImageNet 类别名称编码为文本 embedding。
- 将 ImageNet validation 图片编码为图像 embedding。
- 计算图像 embedding 和类别文本 embedding 的相似度。
- 输出 Top-1 和 Top-5 accuracy。

运行示例：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python evaluate_imagenet.py \
  --config configs/coco_3gpu_cliptext.yaml \
  --checkpoint checkpoints/coco_3gpu_cliptext/best.pt \
  --imagenet-root /data/ydongbd/datasets/imagenet/val \
  --backend imagefolder \
  --class-index-json /data/ydongbd/datasets/imagenet/imagenet_class_index.json \
  --output logs/imagenet_zero_shot.txt
```

当前状态：

- 已完成 ImageNet validation set 下载、MD5 校验和 ImageFolder 整理。
- 数据位置是 `/data/ydongbd/datasets/imagenet/val`。
- 整理后共有 1000 个 class folders 和 50000 张 validation images。
- 评估结果记录在：

```text
logs/imagenet_zero_shot.txt
```

当前有效结果：

```text
top1_accuracy: 19.55
top5_accuracy: 42.08
num_samples: 50000
```

注意：ImageNet val 的整理必须使用官方 devkit `meta.mat` 里的 `ILSVRC2012_ID -> WNID` 映射。不能直接用 `imagenet_class_index.json` 的 0-999 顺序解释 validation ground-truth，否则会导致标签错位，评估结果会退化到随机水平。

### 3.1.1 ImageNet 数据下载脚本

```text
scripts/download_imagenet_val.py
```

作用：

- 下载 `ILSVRC2012_img_val.tar` 和 `ILSVRC2012_devkit_t12.tar.gz`。
- 校验官方 MD5，避免损坏 tar 文件继续参与解压或评估。

### 3.1.2 ImageNet validation 整理脚本

```text
scripts/prepare_imagenet_val.py
```

作用：

- 读取官方 validation ground-truth。
- 从 devkit `meta.mat` 中读取正确的 `ILSVRC2012_ID -> WNID` 映射。
- 将 50000 张 validation 图片整理成 `torchvision.datasets.ImageFolder` 可直接读取的目录结构。

### 3.2 Loss 曲线脚本

```text
scripts/plot_loss_curve.py
```

作用：

- 从训练日志中解析每个 epoch 的 train loss 和 validation loss。
- 用 PIL 生成 loss 曲线图片，不依赖 matplotlib。
- 默认输入：

```text
logs/coco_3gpu_cliptext_train.log
```

- 默认输出：

```text
reports/figures/coco_3gpu_cliptext_loss_curve.png
```

运行命令：

```bash
python scripts/plot_loss_curve.py \
  --log logs/coco_3gpu_cliptext_train.log \
  --output reports/figures/coco_3gpu_cliptext_loss_curve.png
```

### 3.3 Qualitative visualization 脚本

```text
scripts/visualize_retrieval.py
```

作用：

- 从 COCO validation set 中取少量图片和对应 caption。
- 使用训练好的模型编码图像和文本。
- 生成两类 qualitative visualization：
  - image-text similarity heatmap
  - image-to-text Top-k retrieval examples

默认输出：

```text
reports/figures/coco_similarity_heatmap.png
reports/figures/coco_topk_retrieval_examples.png
```

运行命令：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/visualize_retrieval.py \
  --config configs/coco_3gpu_cliptext.yaml \
  --checkpoint checkpoints/coco_3gpu_cliptext/best.pt \
  --num-samples 12 \
  --queries 4 \
  --topk 5 \
  --heatmap-output reports/figures/coco_similarity_heatmap.png \
  --retrieval-output reports/figures/coco_topk_retrieval_examples.png
```

## 4. 原有关键文件说明

### 4.1 主训练脚本

```text
train.py
```

作用：

- 加载 COCO dataset。
- 构建 ViT image encoder + frozen CLIP text encoder。
- 使用对称 InfoNCE contrastive loss 训练图文对齐模型。
- 保存 `latest.pt` 和 `best.pt`。

### 4.2 COCO 检索评估脚本

```text
evaluate.py
```

作用：

- 对 COCO validation set 做 image-to-text 和 text-to-image retrieval。
- 输出 R@1、R@5、R@10 和 mean recall。
- 当前主结果就是由该脚本评估得到的。

### 4.3 CIFAR zero-shot 迁移评估脚本

```text
evaluate_transfer.py
```

作用：

- 在 CIFAR-10 或 CIFAR-100 上做 zero-shot classification。
- 用 prompt `a photo of a {class}.` 编码类别名称。
- 输出 Top-1 和 Top-5 accuracy。

当前已完成的是 CIFAR-100：

```text
logs/cifar100_transfer_eval.txt
```

结果：

```text
top1_accuracy: 37.21
top5_accuracy: 67.03
```

### 4.4 正式实验配置

```text
configs/coco_3gpu_cliptext.yaml
```

作用：

- 正式 COCO 训练配置。
- 使用 `vit_base_patch16_224` image encoder。
- 使用 frozen `openai/clip-vit-base-patch32` text encoder。
- 训练 30 epoch。
- 使用 GPU `[1, 2, 3]`。
- checkpoint 保存到：

```text
checkpoints/coco_3gpu_cliptext/
```

## 5. 4 月 26 日 dev5000 小实验说明

另一个 checkpoint 目录是：

```text
checkpoints/coco_train_dev5000_gpu3_cliptext/
```

这个实验使用从 COCO train2017 中划出的 5000 张 dev set 做验证，配置是：

```text
configs/coco_train_dev5000_gpu3_cliptext.yaml
```

状态：

- `latest.pt` 到 epoch 15。
- `best.pt` 在 epoch 14。
- 目标是 30 epoch，所以这个实验没有完整跑完。
- 对应训练日志是空文件。
- 临时评估结果 mean recall = 46.67，低于正式主模型的 49.02。

结论：

- 不建议把这组结果作为主结果。
- 最终报告仍建议使用 `checkpoints/coco_3gpu_cliptext/best.pt`。

## 6. 当前还缺什么

最重要的必做项已经补齐：

1. ImageNet validation set 已准备完成。
2. ImageNet zero-shot Top-1 / Top-5 已完成评估。
3. Loss curve、similarity heatmap、Top-k retrieval examples 已生成。

可选增强：

1. 尝试多个 prompt template 做 prompt ensembling，例如：

```text
a photo of a {label}.
an image of a {label}.
a blurry photo of a {label}.
```

2. 增加更多 qualitative examples。
3. 整理 slides，把 loss curve、heatmap、Top-k retrieval examples 放进去。

## 7. 推荐给报告的表述

可以在报告中这样总结：

> We trained a ViT-B/16 image encoder with a frozen pretrained CLIP text encoder on COCO Captions using symmetric contrastive learning. The best checkpoint was selected at epoch 26 based on validation loss. On COCO retrieval, the model achieved 24.88 image-to-text R@1, 54.22 image-to-text R@5, 24.58 text-to-image R@1, and 54.28 text-to-image R@5. For zero-shot classification, it reached 37.21% Top-1 and 67.03% Top-5 accuracy on CIFAR-100, and 19.55% Top-1 and 42.08% Top-5 accuracy on ImageNet validation. Qualitative retrieval examples show that the model often retrieves semantically correct captions at Top-1, while some visually or semantically related false positives remain.

中文解释：

> 我们使用 COCO Captions 训练了一个 ViT-B/16 图像编码器，并冻结预训练 CLIP 文本编码器，通过对称对比学习完成图文对齐。最佳 checkpoint 根据 validation loss 选自 epoch 26。在 COCO 检索任务上，模型达到 image-to-text R@1 = 24.88、R@5 = 54.22，以及 text-to-image R@1 = 24.58、R@5 = 54.28。在 CIFAR-100 zero-shot 迁移评估中，模型达到 Top-1 = 37.21%、Top-5 = 67.03%；在 ImageNet validation zero-shot 评估中，模型达到 Top-1 = 19.55%、Top-5 = 42.08%。定性检索结果显示，模型通常能在 Top-1 找到语义正确的 caption，但仍存在一些语义相近或场景相似导致的错误召回。
