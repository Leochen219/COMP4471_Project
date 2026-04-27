# TODO — Zero-Shot ViT Contrastive Learning Project

> 项目目标：训练 ViT 图像编码器，通过对比学习与冻结的文本编码器对齐，实现 Zero-Shot 图像分类。

## 最新进展（2026-04-27）

- [x] 已切换到冻结的预训练 CLIP text encoder
- [x] 已完成 COCO `train2017` / `val2017` 正式训练
- [x] 已完成 3-GPU 正式训练与 best checkpoint 选取
- [x] 已完成 COCO 严格检索评估
- [x] 已完成 `CIFAR-100` zero-shot 迁移测试
- [x] 已完成 `ImageNet val` zero-shot 最终 benchmark
- [x] 已生成 loss curve、similarity heatmap 和 Top-k 检索可视化
- [x] 已整理实验报告到 `reports/cliptext_experiment_report.md`

### 当前最佳结果

- `best_val_loss = 0.4421` at epoch `26`
- COCO retrieval:
  - `i2t_R@1 = 24.88`
  - `i2t_R@5 = 54.22`
  - `t2i_R@1 = 24.58`
  - `t2i_R@5 = 54.28`
  - `mean_recall = 49.02`
- CIFAR-100 zero-shot transfer:
  - `Top-1 = 37.21%`
  - `Top-5 = 67.03%`
- ImageNet zero-shot transfer:
  - `Top-1 = 19.55%`
  - `Top-5 = 42.08%`

---

## Phase 1: 环境与数据准备 ✅ / 🔲

- [x] COCO 数据集下载脚本 (`Dataset/download_COCO.py`)
- [x] COCO Dataset & DataLoader (`Dataset/DataLoader.py`)
- [x] 创建 `requirements.txt`（torch, torchvision, transformers, timm, tqdm 等）

### 1.1 训练数据

| 数据集 | 规模 | 优先级 | 用途 |
|--------|------|--------|------|
| **COCO Captions** | 118K 图 / 590K 图文对 | **必须** | 主训练集（已有加载器） |
| **CC3M (Conceptual Captions)** | ~300 万图文对 | **强烈推荐** | 补充训练集，性价比最高 |
| **Flickr30k** | 31K 图 / 155K 图文对 | 可选 | 可作为额外验证集 |

- [x] 下载 COCO train2017 / val2017 图片和标注到本地
- [ ] 下载 CC3M 数据集（需自行根据 TSV 中 URL 下载图片，部分 URL 已失效，实际约 250 万对）
- [ ] 编写 CC3M DataLoader
- [ ] 实现 COCO + CC3M 混合训练的 ConcatDataset

### 1.2 评估数据

| 数据集 | 类别数 | 验证集大小 | 优先级 | 用途 |
|--------|--------|-----------|--------|------|
| **ImageNet val** | 1000 类 | 50K 张 | **必须** | 最终 zero-shot 评估指标 |
| **CIFAR-10** | 10 类 | 10K 张 | **推荐** | 训练中快速 sanity check（秒级完成） |
| **CIFAR-100** | 100 类 | 10K 张 | 可选 | 中等难度补充评估 |
| **Caltech-101** | 101 类 | ~3K 张 | 可选 | 轻量级补充评估 |

- [x] 准备 ImageNet validation set（下载验证集图片 + 1000 类标签映射）
- [x] 编写 ImageNet zero-shot 评估脚本
- [x] 运行 ImageNet zero-shot 最终 benchmark
- [ ] 准备 CIFAR-10（torchvision 可直接下载）
- [ ] 编写 CIFAR-10 zero-shot 评估脚本（用于训练过程中快速检查）
- [x] 准备 CIFAR-100（torchvision 可直接下载）
- [x] 编写 CIFAR-100 zero-shot 迁移评估脚本
- [x] 运行 CIFAR-100 zero-shot 迁移评估

---

## Phase 2: 模型搭建 🔲

### 2.1 图像编码器
- [x] 实现 ViT 图像编码器（可基于 `timm` 库加载 ViT-B/16 或 ViT-B/32）
- [x] 提取 `[CLS]` token 输出作为图像全局特征

### 2.2 文本编码器（冻结）
- [x] 加载预训练文本编码器（如 CLIP text encoder 或 DistilBERT）
- [x] 冻结全部参数（`requires_grad = False`）
- [x] 提取 `[EOS]` / `[CLS]` token 输出作为文本全局特征

### 2.3 投影层
- [x] 实现 Image Projection Head（Linear / MLP，将图像特征映射到 d 维共享空间）
- [x] 实现 Text Projection Head（Linear / MLP，将文本特征映射到 d 维共享空间）
- [x] 对输出做 L2 归一化

### 2.4 整体模型
- [x] 封装为统一的 `CLIPModel` 类
- [x] 包含可学习的温度参数 `logit_scale`

---

## Phase 3: 训练 🔲

### 3.1 损失函数
- [x] 实现对称的 InfoNCE / NT-Xent 对比损失
  - `loss = (loss_img2txt + loss_txt2img) / 2`

### 3.2 训练脚本
- [x] 创建 `train.py` 主训练脚本
- [ ] 配置超参数（建议起步值）：
  - batch size: 256
  - learning rate: 3e-4（配合 cosine scheduler）
  - embedding dim: 512
  - epochs: 30+
  - optimizer: AdamW（weight_decay=0.01）
- [x] 实现学习率调度器（warmup + cosine decay）
- [x] 添加训练日志（loss 曲线、学习率变化）
- [x] 支持 checkpoint 保存与恢复
- [x] 支持多 GPU 训练（DataParallel / DDP，可选）

### 3.3 训练策略（推荐）

```
阶段 1（调试）：只用 COCO，跑几个 epoch，确认整体流程跑通
阶段 2（正式）：COCO + CC3M 混合训练，跑 10-30 epochs
评估节奏：每 N epoch 用 CIFAR-10 快速 sanity check → 最终用 ImageNet 出结果
```

- [x] 阶段 1：仅 COCO 训练，验证流程
- [ ] 阶段 2：COCO + CC3M 混合训练（~350 万对），正式出结果

---

## Phase 4: Zero-Shot 评估 ✅ / 🔲

### 4.1 ImageNet Zero-Shot 分类
- [x] 构建 1000 类的文本 prompt（如 `"a photo of a {class_name}"`）
- [x] 对所有类别 prompt 提前编码为文本嵌入
- [x] 对验证集图片编码为图像嵌入
- [x] 计算余弦相似度，取 Top-k 类别作为预测
- [x] 报告 **Top-1 Accuracy** 和 **Top-5 Accuracy**

### 4.2 Prompt Engineering（可选增强）
- [ ] 尝试多种 prompt 模板（如 `"a photo of a {class}"`, `"an image of {class}"` 等）
- [ ] 对多个 prompt 取平均嵌入，看是否提升准确率

---

## Phase 5: 可视化与分析 ✅ / 🔲

### 5.1 定性可视化
- [x] **Similarity Heatmap**：展示图像与多个文本的相似度矩阵
- [x] **Top-k 检索**：给定图片，展示模型检索出的前 k 个文本结果
- [ ] **Attention Map**：可视化 ViT 关注的图像区域（可选）

### 5.2 消融实验 (Ablation Study)
- [ ] 训练一个同等规模的监督分类模型（如 ViT 在 COCO 类别上做分类）
- [ ] 对比 zero-shot 方法 vs 监督方法的 ImageNet 性能
- [ ] 分析不同 embedding 维度 / projection 层结构对性能的影响

### 5.3 结果整理
- [x] 制作结果表格（accuracy 对比）
- [x] 绘制 loss 曲线图
- [x] 准备项目报告 / 演示用的图表

---

## 分工建议

| 成员 | 建议负责模块 |
|------|-------------|
| 成员 A | Phase 2 模型搭建 + Phase 3 训练 |
| 成员 B | Phase 1 数据准备 + Phase 4 评估 |
| 成员 C | Phase 5 可视化 + 消融实验 + 报告 |

---

## 关键设计决策（待讨论）

1. **ViT 是否用预训练权重初始化？** 从头训练 vs ImageNet pretrained ViT fine-tune
2. **文本编码器选哪个？** CLIP text encoder vs DistilBERT vs 其他
3. **Projection 层用线性层还是 MLP？** Linear vs 2-layer MLP with ReLU
4. **是否只训练 COCO？** 还是额外加入 CC3M 等数据集

---

## 数据集建议说明

### 为什么推荐 COCO + CC3M？

1. **COCO 单独训练大概率不够**。59 万图文对对于对比学习来说偏少，即使用了预训练 ViT，对齐效果也有限。
2. **CC3M 是最佳性价比选择**。~300 万对真实网络图文数据，质量不错；加上 COCO 总共约 350 万对，这个规模能学到有意义的对齐。
3. **CIFAR-10 作为训练中的快速检查**。只有 10 类，评估秒级完成，能快速判断模型是否在学到东西，不用每次都跑完整 ImageNet。
4. **ImageNet 作为最终报告指标**。这是 zero-shot 分类的标准 benchmark，报告里必须有。

### 参考：原版 CLIP 使用的数据

| | 原版 CLIP | 本项目（推荐方案） |
|---|----------|-------------------|
| **训练数据** | WIT 4 亿对（私有） | COCO + CC3M ~350 万对 |
| **Image Encoder** | 从头训练 | Pretrained ViT fine-tune |
| **Text Encoder** | 从头训练 | 冻结预训练模型 |
| **Batch Size** | 32,768 | 128~512（受限于显存） |
| **GPU** | 256x V100 | 1~4 张 GPU |

> 注意：由于数据和算力差距巨大，zero-shot 准确率不要期望达到 CLIP 水平（CLIP ImageNet Top-1 ~76%），但能展示出 zero-shot 能力即可，作为课程项目已经很有价值。
