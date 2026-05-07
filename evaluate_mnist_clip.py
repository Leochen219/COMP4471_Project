# evaluate_mnist_clip.py
#
# Evaluate CLIP model (ViT-B/16) on MNIST zero-shot classification.
#
# Approach:
#   - Load MNIST test set (10 classes: digits 0-9)
#   - Create text prompts: "a photo of the digit {n}" for n=0..9
#   - Encode all images and all text prompts via CLIP
#   - Classify each image by finding the text prompt with highest cosine similarity
#   - Report accuracy per class and overall

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import CLIPTokenizer

# Allow importing from project root
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import load_config
from models import CLIPModel

logging.basicConfig(
    level=logging.INFO,
    format="[MNIST-CLIP %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================
#  MNIST transforms
# ================================================================
# MNIST images are 28x28 grayscale. CLIP expects 3-channel RGB.
# We resize to the model's expected input size (224 for ViT-B/16)
# and replicate the single channel to 3 channels.
MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

def get_mnist_transform(image_size: int = 224):
    """Transform MNIST grayscale -> 3-channel RGB resized to model input size."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),       # e.g. 256
        transforms.CenterCrop(image_size),         # e.g. 224
        transforms.Grayscale(num_output_channels=3),  # 1ch -> 3ch
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.1307, 0.1307, 0.1307],
            std=[0.3081, 0.3081, 0.3081],
        ),
    ])


# ================================================================
#  Text prompts for MNIST digits
# ================================================================
def build_mnist_prompts() -> list[str]:
    """Create a list of text prompts, one per digit class."""
    templates = [
        "a photo of the digit {n}",
        "the number {n}",
        "digit {n}",
        "a handwritten digit {n}",
    ]
    prompts = []
    for n in range(10):
        # Use the first template as the primary prompt
        prompts.append(templates[0].format(n=n))
    return prompts


# ================================================================
#  Zero-shot classification
# ================================================================
@torch.no_grad()
def classify_mnist_zero_shot(
    model,
    test_loader,
    text_embeds: torch.Tensor,  # [10, D] L2-normalized
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classify all MNIST test images via zero-shot CLIP.
    
    Returns:
        predictions: [N] predicted class indices
        targets:     [N] ground truth labels
    """
    model.eval()
    all_preds = []
    all_targets = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)  # [B, 3, H, W]
        
        # Encode images -> [B, D]
        image_embeds = model.encode_image(images)
        
        # Compute similarity with all text prompts -> [B, 10]
        logits = image_embeds @ text_embeds.t()  # cosine similarity
        
        # Predict the class with highest similarity
        preds = logits.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.clone())

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return predictions, targets


# ================================================================
#  Metrics
# ================================================================
def compute_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 10,
) -> dict:
    """Compute per-class accuracy and overall accuracy."""
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    overall_acc = 100.0 * correct / total

    per_class_acc = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            acc = 100.0 * (predictions[mask] == c).sum().item() / mask.sum().item()
            per_class_acc[c] = acc
        else:
            per_class_acc[c] = 0.0

    metrics = {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "correct": correct,
        "total": total,
    }
    return metrics


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="[MNIST-CLIP] Evaluate CLIP model on MNIST zero-shot"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to CLIP model weights")
    parser.add_argument(
        "--data_root", default="./data/mnist",
        help="Directory to store/download MNIST dataset"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Test batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- Data ----------
    transform = get_mnist_transform(cfg.image_size)
    test_dataset = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"MNIST test set: {len(test_dataset)} images")

    # ---------- Model ----------
    model = CLIPModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32"),
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    # Handle potential key name mismatches
    new_state_dict = {}
    for k, v in state_dict.items():
        if "text_encoder.model.text_model." in k:
            new_k = k.replace("text_encoder.model.text_model.", "text_encoder.model.")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint incompatible with CLIP model."
        ) from exc
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ---------- Build text prompts ----------
    prompts = build_mnist_prompts()
    logger.info(f"Text prompts: {prompts}")

    tokenizer = CLIPTokenizer.from_pretrained(
        getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
    )

    # Encode all text prompts
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=cfg.text_max_length,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs["attention_mask"].to(device)
    text_embeds = model.encode_text(input_ids, attention_mask)  # [10, D]

    # ---------- Zero-shot classification ----------
    logger.info("Running zero-shot classification on MNIST...")
    predictions, targets = classify_mnist_zero_shot(
        model, test_loader, text_embeds, device
    )

    # ---------- Metrics ----------
    metrics = compute_classification_metrics(predictions, targets, num_classes=10)

    logger.info("=" * 50)
    logger.info("MNIST Zero-Shot Classification Results (CLIP)")
    logger.info("=" * 50)
    logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    logger.info(f"  Correct / Total:  {metrics['correct']} / {metrics['total']}")
    logger.info("")
    logger.info("  Per-Class Accuracy:")
    for c in range(10):
        acc = metrics["per_class_accuracy"][c]
        bar = "#" * int(acc / 2)
        logger.info(f"    Digit {c}: {acc:>6.2f}%  |{bar}")
    logger.info("=" * 50)

    # ---------- Confusion matrix (summary) ----------
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for t, p in zip(targets, predictions):
        confusion[t, p] += 1

    logger.info("  Confusion Matrix (rows=true, cols=pred):")
    header = "       " + "".join(f"{c:>6d}" for c in range(10))
    logger.info(header)
    for c in range(10):
        row = "  " + "".join(f"{confusion[c, p]:>6d}" for p in range(10))
        logger.info(f"  {c}: {row}")

    # Save results to file
    results_dir = "reports/mnist_eval"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "clip_mnist_results.txt")
    with open(results_path, "w") as f:
        f.write("MNIST Zero-Shot Classification Results (CLIP)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%\n")
        f.write(f"Correct / Total: {metrics['correct']} / {metrics['total']}\n\n")
        f.write("Per-Class Accuracy:\n")
        for c in range(10):
            f.write(f"  Digit {c}: {metrics['per_class_accuracy'][c]:.2f}%\n")
        f.write("\nConfusion Matrix:\n")
        f.write(header + "\n")
        for c in range(10):
            row = "  ".join(f"{confusion[c, p]:>5d}" for p in range(10))
            f.write(f"  {c}: {row}\n")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
