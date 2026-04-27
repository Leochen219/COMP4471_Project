#!/usr/bin/env python

import argparse
import json
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from data.transforms import get_val_transform
from models import CLIPModel


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_model(cfg, checkpoint_path: str, device: torch.device) -> CLIPModel:
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
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_coco_samples(json_path: str, img_dir: str, limit: int):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    img_to_captions = {}
    for ann in data["annotations"]:
        img_to_captions.setdefault(ann["image_id"], []).append(clean_text(ann["caption"]))

    samples = []
    for img in data["images"]:
        if len(samples) >= limit:
            break
        captions = img_to_captions.get(img["id"], [])
        path = Path(img_dir) / img["file_name"]
        if captions and path.exists():
            samples.append(
                {
                    "id": img["id"],
                    "file_name": img["file_name"],
                    "path": path,
                    "caption": captions[0],
                }
            )
    return samples


@torch.no_grad()
def encode_samples(model, tokenizer, transform, samples, max_length, device):
    images = []
    for sample in samples:
        image = Image.open(sample["path"]).convert("RGB")
        images.append(transform(image))
    image_tensor = torch.stack(images).to(device)

    captions = [sample["caption"] for sample in samples]
    text_inputs = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)

    image_embeds = model.encode_image(image_tensor)
    text_embeds = model.encode_text(input_ids, attention_mask)
    return image_embeds.cpu(), text_embeds.cpu()


def sim_to_color(value: float, min_value: float, max_value: float):
    span = max(max_value - min_value, 1e-8)
    t = max(0.0, min(1.0, (value - min_value) / span))
    # Blue-white-red.
    if t < 0.5:
        k = t / 0.5
        r = int(245 * k + 35 * (1 - k))
        g = int(245 * k + 90 * (1 - k))
        b = int(245 * k + 170 * (1 - k))
    else:
        k = (t - 0.5) / 0.5
        r = int(210 * k + 245 * (1 - k))
        g = int(65 * k + 245 * (1 - k))
        b = int(65 * k + 245 * (1 - k))
    return (r, g, b)


def draw_heatmap(sim: torch.Tensor, samples, output: Path):
    n = sim.size(0)
    cell = 44
    left = 140
    top = 95
    width = left + n * cell + 45
    height = top + n * cell + 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(26)
    small_font = load_font(15)

    values = sim.flatten().tolist()
    min_value, max_value = min(values), max(values)
    draw.text((30, 28), "COCO Val Image-Text Similarity Heatmap", font=title_font, fill=(20, 20, 20))

    for i in range(n):
        label = str(i + 1)
        draw.text((left + i * cell + 15, top - 28), label, font=small_font, fill=(70, 70, 70))
        draw.text((left - 35, top + i * cell + 13), label, font=small_font, fill=(70, 70, 70))
        for j in range(n):
            x0 = left + j * cell
            y0 = top + i * cell
            color = sim_to_color(float(sim[i, j]), min_value, max_value)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(255, 255, 255))
            if i == j:
                draw.rectangle((x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2), outline=(20, 20, 20), width=2)

    draw.text((left, height - 55), "Rows: images, columns: captions. Black boxes mark the paired caption.", font=small_font, fill=(40, 40, 40))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def thumbnail(path: Path, size: int = 170):
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (245, 245, 245))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def wrapped_lines(text: str, width: int, max_lines: int):
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".") + "..."
    return lines


def draw_retrieval_grid(sim: torch.Tensor, samples, output: Path, queries: int, topk: int):
    queries = min(queries, len(samples))
    topk = min(topk, len(samples))
    row_h = 245
    width = 1420
    height = 90 + queries * row_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    header_font = load_font(18)
    small_font = load_font(15)

    draw.text((28, 25), "COCO Val Image-to-Text Retrieval Examples", font=title_font, fill=(20, 20, 20))
    for qi in range(queries):
        y = 80 + qi * row_h
        sample = samples[qi]
        image.paste(thumbnail(sample["path"]), (28, y + 30))
        draw.text((28, y + 8), f"Query image {qi + 1}", font=header_font, fill=(20, 20, 20))

        ranked = torch.argsort(sim[qi], descending=True).tolist()[:topk]
        x = 225
        for rank, idx in enumerate(ranked, start=1):
            is_match = idx == qi
            box = (x, y + 30, x + 220, y + 205)
            fill = (236, 248, 240) if is_match else (248, 248, 248)
            outline = (40, 130, 70) if is_match else (205, 205, 205)
            draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=2)
            draw.text((x + 12, y + 42), f"Top {rank} | caption {idx + 1}", font=header_font, fill=(20, 20, 20))
            draw.text((x + 12, y + 68), f"sim={float(sim[qi, idx]):.3f}", font=small_font, fill=(80, 80, 80))
            for line_i, line in enumerate(wrapped_lines(samples[idx]["caption"], 28, 5)):
                draw.text((x + 12, y + 98 + line_i * 19), line, font=small_font, fill=(35, 35, 35))
            x += 238

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description="Create qualitative retrieval visualizations")
    parser.add_argument("--config", default="configs/coco_3gpu_cliptext.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/coco_3gpu_cliptext/best.pt")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--heatmap-output", default="reports/figures/coco_similarity_heatmap.png")
    parser.add_argument("--retrieval-output", default="reports/figures/coco_topk_retrieval_examples.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    samples = load_coco_samples(cfg.val_json, cfg.val_img, args.num_samples)
    if not samples:
        raise ValueError("No COCO samples found")

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )
    model = load_model(cfg, args.checkpoint, device)
    transform = get_val_transform(cfg.image_size)
    image_embeds, text_embeds = encode_samples(
        model,
        tokenizer,
        transform,
        samples,
        cfg.text_max_length,
        device,
    )
    sim = image_embeds @ text_embeds.t()
    draw_heatmap(sim, samples, Path(args.heatmap_output))
    draw_retrieval_grid(sim, samples, Path(args.retrieval_output), args.queries, args.topk)
    print(f"saved: {args.heatmap_output}")
    print(f"saved: {args.retrieval_output}")


if __name__ == "__main__":
    main()
