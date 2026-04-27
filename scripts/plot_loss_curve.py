#!/usr/bin/env python

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"Epoch (\d+) 完毕 \| Train Loss: ([0-9.]+) \| Val Loss: ([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError(f"No epoch summaries found in {path}")
    epochs = [int(item[0]) for item in matches]
    train_losses = [float(item[1]) for item in matches]
    val_losses = [float(item[2]) for item in matches]
    return epochs, train_losses, val_losses


def scale_points(xs, ys, box, x_min, x_max, y_min, y_max):
    left, top, right, bottom = box
    x_span = max(x_max - x_min, 1)
    y_span = max(y_max - y_min, 1e-8)
    points = []
    for x, y in zip(xs, ys):
        px = left + (x - x_min) / x_span * (right - left)
        py = bottom - (y - y_min) / y_span * (bottom - top)
        points.append((px, py))
    return points


def draw_polyline(draw, points, color, width=4):
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def draw_plot(epochs, train_losses, val_losses, output: Path):
    width, height = 1200, 760
    margin_left, margin_top, margin_right, margin_bottom = 110, 90, 70, 105
    plot_box = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    label_font = load_font(22)
    small_font = load_font(18)

    x_min, x_max = min(epochs), max(epochs)
    all_losses = train_losses + val_losses
    y_min = max(0.0, min(all_losses) * 0.85)
    y_max = max(all_losses) * 1.08

    left, top, right, bottom = plot_box
    draw.rectangle(plot_box, outline=(30, 30, 30), width=2)

    for i in range(6):
        y = top + i * (bottom - top) / 5
        value = y_max - i * (y_max - y_min) / 5
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((20, y - 11), f"{value:.3f}", font=small_font, fill=(70, 70, 70))

    for i in range(0, len(epochs), max(1, len(epochs) // 10)):
        x = left + (epochs[i] - x_min) / max(x_max - x_min, 1) * (right - left)
        draw.line((x, bottom, x, bottom + 8), fill=(30, 30, 30), width=2)
        draw.text((x - 10, bottom + 14), str(epochs[i]), font=small_font, fill=(70, 70, 70))

    train_points = scale_points(epochs, train_losses, plot_box, x_min, x_max, y_min, y_max)
    val_points = scale_points(epochs, val_losses, plot_box, x_min, x_max, y_min, y_max)
    draw_polyline(draw, train_points, (32, 116, 183))
    draw_polyline(draw, val_points, (220, 70, 70))

    best_idx = min(range(len(val_losses)), key=lambda i: val_losses[i])
    bx, by = val_points[best_idx]
    draw.ellipse((bx - 8, by - 8, bx + 8, by + 8), outline=(120, 0, 0), width=4)
    best_text = f"best val {val_losses[best_idx]:.4f} @ epoch {epochs[best_idx]}"
    text_width = draw.textlength(best_text, font=small_font)
    text_x = bx + 14
    if text_x + text_width > right - 8:
        text_x = bx - text_width - 14
    draw.text(
        (text_x, by - 32),
        best_text,
        font=small_font,
        fill=(120, 0, 0),
    )

    draw.text((margin_left, 28), "COCO CLIP-text Training Loss", font=title_font, fill=(20, 20, 20))
    draw.text((width // 2 - 45, height - 55), "Epoch", font=label_font, fill=(20, 20, 20))
    draw.text((20, 24), "Loss", font=label_font, fill=(20, 20, 20))

    legend_x = right - 260
    legend_y = top + 24
    draw.line((legend_x, legend_y, legend_x + 55, legend_y), fill=(32, 116, 183), width=5)
    draw.text((legend_x + 70, legend_y - 12), "Train loss", font=small_font, fill=(40, 40, 40))
    draw.line((legend_x, legend_y + 34, legend_x + 55, legend_y + 34), fill=(220, 70, 70), width=5)
    draw.text((legend_x + 70, legend_y + 22), "Val loss", font=small_font, fill=(40, 40, 40))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description="Plot training/validation losses from train.py logs")
    parser.add_argument("--log", default="logs/coco_3gpu_cliptext_train.log")
    parser.add_argument("--output", default="reports/figures/coco_3gpu_cliptext_loss_curve.png")
    args = parser.parse_args()

    epochs, train_losses, val_losses = parse_log(Path(args.log))
    draw_plot(epochs, train_losses, val_losses, Path(args.output))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
