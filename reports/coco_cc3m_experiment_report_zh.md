# COCO + CC3M 图文对齐实验报告

## 1. 实验目的

本次实验在原有 COCO Captions 训练流程基础上加入 CC3M，目标是验证更大规模网页图文数据是否能提升模型的 open-vocabulary 泛化能力。

原 COCO-only 模型在 COCO 检索上表现较好，但训练数据域较窄。CC3M 提供更丰富的自然图像和网页 caption，因此本实验重点观察两个方向：

- COCO 图文检索是否保持稳定。
- CIFAR-100、ImageNet、Flickr30k 等跨数据集评估是否提升。

## 2. 数据与训练设置

训练数据：

- COCO Captions `train2017`
- CC3M WebDataset 版本：`pixparse/cc3m-wds`
- 本地路径：`data/cc3m_wds`
- 下载规模：576 个 train shards，16 个 validation shards，总计 592 个 shards，约 263 GB

训练配置：

- 配置文件：`configs/coco_cc3m_3gpu_cliptext.yaml`
- 图像编码器：`vit_base_patch16_224`
- 文本编码器：`openai/clip-vit-base-patch32`
- 文本编码器冻结，projection head 可训练
- 全局 batch size：96
- GPU：`[1, 2, 3]`
- epoch：10
- 每个 epoch 训练样本数：240000
- COCO:CC3M 混合权重：`1:3`

模型保存：

- 最佳 checkpoint：`checkpoints/coco_cc3m_3gpu_cliptext/best.pt`
- 最后 checkpoint：`checkpoints/coco_cc3m_3gpu_cliptext/latest.pt`
- 最佳模型来自 epoch 7

## 3. 训练结果

训练从 epoch 0 跑到 epoch 9 正常结束。验证 loss 在 epoch 7 达到最低，随后轻微回升，因此最终评估使用 `best.pt`。

| Epoch | Train Loss | Val Loss | 备注 |
|---:|---:|---:|---|
| 0 | 0.8305 | 0.7546 | best saved |
| 1 | 0.3982 | 0.7029 | best saved |
| 2 | 0.3461 | 0.6828 | best saved |
| 3 | 0.3145 | 0.6490 | best saved |
| 4 | 0.2911 | 0.6462 | best saved |
| 5 | 0.2701 | 0.6125 | best saved |
| 6 | 0.2609 | 0.5977 | best saved |
| 7 | 0.2449 | 0.5911 | best saved |
| 8 | 0.2328 | 0.5990 | validation 回升 |
| 9 | 0.2224 | 0.6134 | validation 回升 |

训练日志：

```text
logs/coco_cc3m_3gpu_cliptext_train.log
```

## 4. 评估协议

本次评估使用四类指标：

1. COCO val 图文检索：沿用项目原有严格一图一 caption 协议。
2. Flickr30k test 图文检索：新增跨数据集 retrieval benchmark，使用多 caption 正样本协议。
3. CIFAR-100 zero-shot 分类：测试细粒度物体类别迁移。
4. ImageNet val zero-shot 分类：测试大规模 open-vocabulary 分类迁移。

新增 Flickr30k 评估脚本：

```text
evaluate_flickr30k.py
```

Flickr30k 数据来源：

```text
clip-benchmark/wds_flickr30k
```

本地只下载 test split：

```text
data/flickr30k_wds
```

## 5. 评估结果

### 5.1 COCO val Retrieval

| Metric | Score |
|---|---:|
| Image-to-Text R@1 | 18.52 |
| Image-to-Text R@5 | 42.90 |
| Image-to-Text R@10 | 56.54 |
| Text-to-Image R@1 | 19.78 |
| Text-to-Image R@5 | 43.82 |
| Text-to-Image R@10 | 57.40 |
| Mean Acc@1 | 19.15 |
| Mean Recall | 39.83 |

日志：

```text
logs/coco_cc3m_cliptext_eval.log
```

### 5.2 Flickr30k Test Retrieval

Flickr30k test 包含 1000 张图和 5000 条 caption。这里按多 caption 正样本协议计算 retrieval recall。

| Metric | Score |
|---|---:|
| Image-to-Text R@1 | 44.20 |
| Image-to-Text R@5 | 71.90 |
| Image-to-Text R@10 | 82.40 |
| Text-to-Image R@1 | 35.36 |
| Text-to-Image R@5 | 66.26 |
| Text-to-Image R@10 | 77.40 |
| Mean Recall | 62.92 |

结果文件：

```text
logs/coco_cc3m_flickr30k_retrieval.txt
```

### 5.3 CIFAR-100 Zero-shot Classification

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 49.41 |
| Top-5 Accuracy | 82.89 |

结果文件：

```text
logs/coco_cc3m_cifar100_transfer_eval.txt
```

### 5.4 ImageNet Zero-shot Classification

| Metric | Score |
|---|---:|
| Top-1 Accuracy | 34.40 |
| Top-5 Accuracy | 63.54 |
| Samples | 50000 |

结果文件：

```text
logs/coco_cc3m_imagenet_zero_shot.txt
```

## 6. 与 COCO-only 模型对比

此前 COCO-only 模型的主要结果如下：

| Benchmark | COCO-only | COCO + CC3M |
|---|---:|---:|
| COCO mean retrieval | 49.02 | 39.83 |
| CIFAR-100 Top-1 | 37.21 | 49.41 |
| CIFAR-100 Top-5 | 67.03 | 82.89 |
| ImageNet Top-1 | 19.55 | 34.40 |
| ImageNet Top-5 | 42.08 | 63.54 |

可以看到，加入 CC3M 后，COCO 严格 retrieval 指标下降，但跨数据集 zero-shot 分类显著提升。这个现象说明模型从更大规模网页图文数据中学到了更强的通用视觉语义表示，但对 COCO val 的一图一 caption 精确匹配能力不如 COCO-only 训练那么专门。

新增的 Flickr30k retrieval 结果进一步支持这一点：模型在未参与训练的跨数据集图文检索上达到 `62.92` mean recall，说明 CC3M 对图文语义泛化是有帮助的。

## 7. 结论

本次 COCO + CC3M 训练成功提升了模型的 open-vocabulary 泛化能力。虽然 COCO val 的严格单 caption retrieval 指标低于 COCO-only 模型，但 CIFAR-100、ImageNet 和 Flickr30k 的结果更适合说明 CC3M 带来的跨域收益。

报告中建议把两个模型定位清楚：

- COCO-only：更适合展示 COCO 域内 retrieval。
- COCO + CC3M：更适合展示跨数据集泛化和 zero-shot 能力。

最终如果只选一个 checkpoint 用于 open-vocabulary demo 和跨域评估，建议使用：

```text
checkpoints/coco_cc3m_3gpu_cliptext/best.pt
```

如果报告重点是 COCO val retrieval，则保留 COCO-only checkpoint 作为对照会更公平。
