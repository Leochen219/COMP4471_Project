# evaluate_mnist_baseline.py
#
# Evaluate Baseline ResNet-50 model on MNIST.
#
# Two evaluation modes:
#   1. Zero-shot (CLIP-style): Use text prompts to classify digits via cosine similarity
#   2. Linear probe: Train a linear classifier on top of frozen ResNet-50 image features
#      (this is the more standard approach for comparing vision backbones)
#
# The linear probe provides a fairer comparison of the image encoder quality.

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import CLIPTokenizer

# Allow importing from project root
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs import load_config
from baseline import BaselineModel

logging.basicConfig(
    level=logging.INFO,
    format="[MNIST-BASELINE %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ================================================================
#  MNIST transforms
# ================================================================
def get_mnist_transform(image_size: int = 224):
    """Transform MNIST grayscale -> 3-channel RGB resized to model input size."""
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.1307, 0.1307, 0.1307],
            std=[0.3081, 0.3081, 0.3081],
        ),
    ])


# ================================================================
#  Text prompts for MNIST digits (zero-shot mode)
# ================================================================
def build_mnist_prompts() -> list[str]:
    templates = [
        "a photo of the digit {n}",
        "the number {n}",
        "digit {n}",
        "a handwritten digit {n}",
    ]
    prompts = []
    for n in range(10):
        prompts.append(templates[0].format(n=n))
    return prompts


# ================================================================
#  Zero-shot classification (same as CLIP approach)
# ================================================================
@torch.no_grad()
def classify_zero_shot(model, test_loader, text_embeds, device):
    """Classify MNIST via zero-shot image-text similarity."""
    model.eval()
    all_preds = []
    all_targets = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        image_embeds = model.encode_image(images)
        logits = image_embeds @ text_embeds.t()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(labels.clone())

    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return predictions, targets


# ================================================================
#  Linear probe on frozen image encoder features
# ================================================================
class LinearProbe(nn.Module):
    """Simple linear classifier on top of frozen image features."""
    def __init__(self, feature_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract image features from the frozen image encoder (before projection)."""
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        # Get features from the image encoder backbone (before projection)
        features = model.image_encoder(images)  # [B, 2048] for ResNet-50
        all_features.append(features.cpu())
        all_labels.append(labels.clone())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    feature_dim: int,
    num_classes: int = 10,
    epochs: int = 50,
    lr: float = 0.01,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Train a linear classifier on frozen features and return test accuracy."""
    probe = LinearProbe(feature_dim, num_classes).to(device)

    optimizer = optim.SGD(probe.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    best_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        outputs = probe(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            probe.eval()
            with torch.no_grad():
                test_outputs = probe(test_features)
                preds = test_outputs.argmax(dim=1)
                acc = 100.0 * (preds == test_labels).sum().item() / test_labels.size(0)
                best_acc = max(best_acc, acc)
            logger.info(
                f"  Linear probe epoch {epoch+1:>3d}/{epochs} | "
                f"Loss: {loss.item():.4f} | Test Acc: {acc:.2f}%"
            )

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        test_outputs = probe(test_features)
        preds = test_outputs.argmax(dim=1)
        final_acc = 100.0 * (preds == test_labels).sum().item() / test_labels.size(0)
        best_acc = max(best_acc, final_acc)

    return best_acc


# ================================================================
#  Metrics
# ================================================================
def compute_classification_metrics(predictions, targets, num_classes=10):
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

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "correct": correct,
        "total": total,
    }


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="[MNIST-BASELINE] Evaluate Baseline ResNet-50 on MNIST"
    )
    parser.add_argument("--config", default="baseline/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to baseline model weights")
    parser.add_argument(
        "--data_root", default="./data/mnist",
        help="Directory to store/download MNIST dataset"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["zero_shot", "linear_probe", "both"],
        default="both",
        help="Evaluation mode"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---------- Data ----------
    transform = get_mnist_transform(cfg.image_size)
    
    test_dataset = datasets.MNIST(
        root=args.data_root, train=False, download=True, transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    train_dataset = datasets.MNIST(
        root=args.data_root, train=True, download=True, transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    logger.info(f"MNIST train set: {len(train_dataset)}  test set: {len(test_dataset)}")

    # ---------- Model ----------
    model = BaselineModel(
        image_encoder_name=cfg.image_encoder_name,
        pretrained=False,
        embed_dim=cfg.embed_dim,
        text_encoder_name=getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32"),
        text_max_length=cfg.text_max_length,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint incompatible with baseline model."
        ) from exc
    model.eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    results = {}

    # ========== Zero-shot evaluation ==========
    if args.mode in ("zero_shot", "both"):
        logger.info("\n" + "=" * 50)
        logger.info("Mode: Zero-shot (CLIP-style text prompts)")
        logger.info("=" * 50)

        prompts = build_mnist_prompts()
        logger.info(f"Text prompts: {prompts}")

        tokenizer = CLIPTokenizer.from_pretrained(
            getattr(cfg, "text_encoder_name", "openai/clip-vit-base-patch32")
        )
        text_inputs = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=cfg.text_max_length, return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        text_embeds = model.encode_text(input_ids, attention_mask)

        predictions, targets = classify_zero_shot(
            model, test_loader, text_embeds, device
        )
        metrics = compute_classification_metrics(predictions, targets, num_classes=10)

        logger.info(f"  Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
        logger.info(f"  Correct / Total:  {metrics['correct']} / {metrics['total']}")
        logger.info("  Per-Class Accuracy:")
        for c in range(10):
            logger.info(f"    Digit {c}: {metrics['per_class_accuracy'][c]:>6.2f}%")

        results["zero_shot"] = metrics

    # ========== Linear probe evaluation ==========
    if args.mode in ("linear_probe", "both"):
        logger.info("\n" + "=" * 50)
        logger.info("Mode: Linear Probe (frozen ResNet-50 features)")
        logger.info("=" * 50)

        # Extract features from frozen image encoder
        logger.info("Extracting training features...")
        train_features, train_labels = extract_features(model, train_loader, device)
        logger.info(f"Train features: {train_features.shape}")

        logger.info("Extracting test features...")
        test_features, test_labels = extract_features(model, test_loader, device)
        logger.info(f"Test features: {test_features.shape}")

        # The image encoder feature dim for ResNet-50 is 2048
        feature_dim = train_features.size(1)

        logger.info("Training linear probe...")
        best_acc = train_linear_probe(
            train_features, train_labels,
            test_features, test_labels,
            feature_dim=feature_dim,
            num_classes=10,
            epochs=50,
            lr=0.01,
            device=device,
        )

        logger.info(f"  Linear Probe Best Accuracy: {best_acc:.2f}%")
        results["linear_probe"] = {"overall_accuracy": best_acc}

    # ========== Summary ==========
    logger.info("\n" + "=" * 50)
    logger.info("MNIST Evaluation Summary (Baseline ResNet-50)")
    logger.info("=" * 50)
    if "zero_shot" in results:
        logger.info(f"  Zero-shot Accuracy:  {results['zero_shot']['overall_accuracy']:.2f}%")
    if "linear_probe" in results:
        logger.info(f"  Linear Probe Acc:   {results['linear_probe']['overall_accuracy']:.2f}%")
    logger.info("=" * 50)

    # Save results
    results_dir = "reports/mnist_eval"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "baseline_mnist_results.txt")
    with open(results_path, "w") as f:
        f.write("MNIST Evaluation Results (Baseline ResNet-50)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        if "zero_shot" in results:
            f.write("Zero-shot Classification:\n")
            f.write(f"  Overall Accuracy: {results['zero_shot']['overall_accuracy']:.2f}%\n")
            f.write(f"  Correct / Total: {results['zero_shot']['correct']} / {results['zero_shot']['total']}\n")
            f.write("  Per-Class Accuracy:\n")
            for c in range(10):
                f.write(f"    Digit {c}: {results['zero_shot']['per_class_accuracy'][c]:.2f}%\n")
            f.write("\n")
        if "linear_probe" in results:
            f.write("Linear Probe (frozen ResNet-50 features):\n")
            f.write(f"  Best Accuracy: {results['linear_probe']['overall_accuracy']:.2f}%\n")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
