# evaluate.py

import logging
import argparse

import torch

from configs import load_config
from models import CLIPModel
from data import build_eval_dataloader
from utils.metrics import compute_retrieval_metrics
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="[PRTS %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_all_embeddings(model, loader, device):
    """遍历 dataloader，提取全部图文嵌入"""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        image_embeds = model.encode_image(images)
        text_embeds = model.encode_text(input_ids, attention_mask)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)

    return all_image_embeds, all_text_embeds


def main():
    parser = argparse.ArgumentParser(description="[PRTS] CLIP Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- 数据（eval_mode=True → 固定第一条 caption）----------
    eval_loader = build_eval_dataloader(cfg)
    logger.info(f"评估样本数: {len(eval_loader.dataset)}") # type: ignore[arg-type]

    # ---------- 模型 ----------
    model = CLIPModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,       # 加载权重，不需要 pretrained backbone
        embed_dim=cfg.embed_dim,
        vocab_size=cfg.vocab_size,
        text_hidden_dim=cfg.text_hidden_dim,
        text_num_layers=cfg.text_num_layers,
        text_num_heads=cfg.text_num_heads,
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    logger.info(f"已加载权重: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # ---------- 提取嵌入 ----------
    logger.info("提取嵌入中...")
    image_embeds, text_embeds = extract_all_embeddings(model, eval_loader, device)
    logger.info(f"图像嵌入: {image_embeds.shape}  文本嵌入: {text_embeds.shape}")

    # ---------- 计算指标 ----------
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)

    logger.info("=" * 50)
    logger.info("检索指标 (Recall@K)")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info(f"  {name:>15s}: {value:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()