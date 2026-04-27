# inference.py

import argparse
import logging

import torch
from PIL import Image
from transformers import CLIPTokenizer

from configs import load_config
from models import CLIPModel
from data.transforms import get_val_transform

logging.basicConfig(
    level=logging.INFO,
    format="[PRTS %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(cfg, checkpoint_path, device):
    model = CLIPModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(
            cfg, "text_encoder_name", "openai/clip-vit-base-patch32"
        ),
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # ====== 新增：兼容队友权重名称差异的代码 ======
    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # 如果字典里有多余的 'text_model.' 前缀，我们把它去掉，以匹配你本地的代码
        if "text_encoder.model.text_model." in k:
            new_k = k.replace("text_encoder.model.text_model.", "text_encoder.model.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    # ==============================================

    try:
        # 注意这里改成了加载 new_state_dict
        model.load_state_dict(new_state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint 与当前 CLIP 文本编码器模型不兼容。"
        ) from exc
    model.eval()
    logger.info(f"模型加载完毕: {checkpoint_path}")
    return model


@torch.no_grad()
def encode_image(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    return model.encode_image(image)                    # [1, D]


@torch.no_grad()
def encode_texts(model, texts, tokenizer, max_length, device):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)            # [N, 77]
    attention_mask = inputs["attention_mask"].to(device)   # [N, 77]
    return model.encode_text(input_ids, attention_mask)    # [N, D]


def main():
    parser = argparse.ArgumentParser(description="[PRTS] CLIP Inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--texts", nargs="+", required=True, help="候选文本列表")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---------- 加载模型 ----------
    model = load_model(cfg, args.checkpoint, device)

    # ---------- 编码图像 ----------
    transform = get_val_transform(cfg.image_size)
    image_embed = encode_image(model, args.image, transform, device)  # [1, D]

    # ---------- 编码文本 ----------
    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )
    text_embeds = encode_texts(
        model, args.texts, tokenizer, cfg.text_max_length, device
    )  # [N, D]

    # ---------- 计算相似度 ----------
    similarities = (image_embed @ text_embeds.t()).squeeze(0)  # [N]

    # ---------- 排序输出 ----------
    sorted_indices = similarities.argsort(descending=True)

    logger.info("")
    logger.info(f"图片: {args.image}")
    logger.info(f"{'排名':>4s}   {'相似度':>8s}   文本")
    logger.info("-" * 60)
    for rank, idx in enumerate(sorted_indices, 1):
        score = similarities[idx].item()
        text = args.texts[idx]
        logger.info(f"  {rank:>2d}    {score:>8.4f}   {text}")
    logger.info("")


if __name__ == "__main__":
    main()
