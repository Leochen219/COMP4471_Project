# baseline/evaluate.py
#
# COCO retrieval evaluation for ResNet-50 baseline model.
# Adapted from evaluate.py - imports BaselineModel instead of CLIPModel.

import logging
import argparse

import os
# Force HuggingFace offline mode to avoid network hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

import sys

# Allow importing from project root
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import load_config
from baseline import BaselineModel
from baseline.data import build_baseline_eval_dataloader
from utils.metrics import compute_retrieval_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[BASELINE %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_all_embeddings(model, loader, device):
    """Iterate over dataloader and extract all image/text embeddings."""
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
    parser = argparse.ArgumentParser(description="[BASELINE] ResNet-50 Evaluation")
    parser.add_argument("--config", default="baseline/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model weights")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    eval_loader = build_baseline_eval_dataloader(cfg)
    logger.info(f"Evaluation samples: {len(eval_loader.dataset)}")

    # ---------- Model ----------
    model = BaselineModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,  # Loading weights, no need for pretrained backbone
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(
            cfg, "text_encoder_name", "openai/clip-vit-base-patch32"
        ),
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint incompatible with baseline model."
        ) from exc
    logger.info(f"Loaded weights: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    # ---------- Extract embeddings ----------
    logger.info("Extracting embeddings...")
    image_embeds, text_embeds = extract_all_embeddings(model, eval_loader, device)
    logger.info(f"Image embeddings: {image_embeds.shape}  Text embeddings: {text_embeds.shape}")

    # ---------- Compute metrics ----------
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)

    logger.info("=" * 50)
    logger.info("Retrieval Metrics (Recall@K)")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info(f"  {name:>15s}: {value:.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
