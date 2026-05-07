# compare_mnist.py
#
# Run both CLIP and Baseline evaluations on MNIST and produce a comparison report.
#
# Usage:
#   python compare_mnist.py \
#       --clip_checkpoint <path> \
#       --baseline_checkpoint <path>
#
# Optional:
#   --clip_config    configs/default.yaml (default)
#   --baseline_config baseline/config.yaml (default)
#   --data_root      ./data/mnist (default)

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[COMPARE %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_script(script_name: str, args: list[str]) -> bool:
    """Run a Python script and return True if successful."""
    cmd = [sys.executable, script_name] + args
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="[COMPARE] Compare CLIP vs Baseline on MNIST"
    )
    parser.add_argument("--clip_checkpoint", required=True, help="CLIP model checkpoint")
    parser.add_argument("--baseline_checkpoint", required=True, help="Baseline model checkpoint")
    parser.add_argument("--clip_config", default="configs/default.yaml")
    parser.add_argument("--baseline_config", default="baseline/config.yaml")
    parser.add_argument("--data_root", default="./data/mnist")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # ========== 1. Run CLIP evaluation ==========
    logger.info("\n" + "=" * 60)
    logger.info("Step 1/2: Evaluating CLIP model on MNIST (zero-shot)")
    logger.info("=" * 60)

    clip_success = run_script("evaluate_mnist_clip.py", [
        f"--config={args.clip_config}",
        f"--checkpoint={args.clip_checkpoint}",
        f"--data_root={args.data_root}",
        f"--batch_size={args.batch_size}",
    ])

    if not clip_success:
        logger.error("CLIP evaluation failed!")
        sys.exit(1)

    # ========== 2. Run Baseline evaluation ==========
    logger.info("\n" + "=" * 60)
    logger.info("Step 2/2: Evaluating Baseline ResNet-50 on MNIST")
    logger.info("=" * 60)

    baseline_success = run_script("evaluate_mnist_baseline.py", [
        f"--config={args.baseline_config}",
        f"--checkpoint={args.baseline_checkpoint}",
        f"--data_root={args.data_root}",
        f"--batch_size={args.batch_size}",
        "--mode=both",
    ])

    if not baseline_success:
        logger.error("Baseline evaluation failed!")
        sys.exit(1)

    # ========== 3. Parse results and generate comparison ==========
    logger.info("\n" + "=" * 60)
    logger.info("Generating comparison report...")
    logger.info("=" * 60)

    # Read CLIP results
    clip_results_file = "reports/mnist_eval/clip_mnist_results.txt"
    baseline_results_file = "reports/mnist_eval/baseline_mnist_results.txt"

    clip_acc = None
    baseline_zs_acc = None
    baseline_lp_acc = None

    if os.path.exists(clip_results_file):
        with open(clip_results_file) as f:
            for line in f:
                if "Overall Accuracy" in line:
                    clip_acc = float(line.split(":")[-1].strip().replace("%", ""))

    if os.path.exists(baseline_results_file):
        with open(baseline_results_file) as f:
            content = f.read()
        # Parse zero-shot accuracy: "  Overall Accuracy: 9.96%" (under Zero-shot section)
        zs_section = content.split("Zero-shot Classification:")[1] if "Zero-shot Classification:" in content else ""
        if "Linear Probe" in content:
            zs_section = content.split("Zero-shot Classification:")[1].split("Linear Probe")[0] if "Zero-shot Classification:" in content else ""
        for line in zs_section.split("\n"):
            if "Overall Accuracy" in line:
                baseline_zs_acc = float(line.split(":")[-1].strip().replace("%", ""))
                break
        # Parse linear probe accuracy: "  Best Accuracy: 81.29%"
        lp_section = content.split("Linear Probe")[1] if "Linear Probe" in content else ""
        for line in lp_section.split("\n"):
            if "Best Accuracy" in line:
                baseline_lp_acc = float(line.split(":")[-1].strip().replace("%", ""))
                break

    # Generate comparison report
    report_path = "reports/mnist_eval/comparison_report.txt"
    os.makedirs("reports/mnist_eval", exist_ok=True)

    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MNIST Evaluation: CLIP (ViT-B/16) vs Baseline (ResNet-50)\n")
        f.write("=" * 60 + "\n\n")

        f.write("Models:\n")
        f.write(f"  CLIP:     ViT-B/16 image encoder + frozen CLIP text encoder\n")
        f.write(f"  Baseline: ResNet-50 image encoder + frozen CLIP text encoder\n\n")

        f.write("Evaluation Protocol:\n")
        f.write("  - Zero-shot: Classify MNIST digits via image-text similarity\n")
        f.write("    (text prompts: 'a photo of the digit {0-9}')\n")
        f.write("  - Linear Probe (Baseline only): Train linear classifier on\n")
        f.write("    frozen image encoder features (more standard MNIST benchmark)\n\n")

        f.write("-" * 60 + "\n")
        f.write("Results\n")
        f.write("-" * 60 + "\n\n")

        f.write(f"{'Method':<40s} {'Accuracy':>10s}\n")
        f.write("-" * 50 + "\n")

        if clip_acc is not None:
            f.write(f"{'CLIP (ViT-B/16) - Zero-shot':<40s} {clip_acc:>8.2f}%\n")
        else:
            f.write(f"{'CLIP (ViT-B/16) - Zero-shot':<40s} {'N/A':>10s}\n")

        if baseline_zs_acc is not None:
            f.write(f"{'Baseline (ResNet-50) - Zero-shot':<40s} {baseline_zs_acc:>8.2f}%\n")
        else:
            f.write(f"{'Baseline (ResNet-50) - Zero-shot':<40s} {'N/A':>10s}\n")

        if baseline_lp_acc is not None:
            f.write(f"{'Baseline (ResNet-50) - Linear Probe':<40s} {baseline_lp_acc:>8.2f}%\n")
        else:
            f.write(f"{'Baseline (ResNet-50) - Linear Probe':<40s} {'N/A':>10s}\n")

        f.write("-" * 50 + "\n\n")

        # Analysis
        f.write("Analysis:\n")
        if clip_acc is not None and baseline_zs_acc is not None:
            diff = clip_acc - baseline_zs_acc
            if diff > 0:
                f.write(f"  CLIP outperforms Baseline by {diff:.2f}% in zero-shot.\n")
            elif diff < 0:
                f.write(f"  Baseline outperforms CLIP by {abs(diff):.2f}% in zero-shot.\n")
            else:
                f.write(f"  Both models achieve the same zero-shot accuracy.\n")

        if baseline_lp_acc is not None:
            f.write(f"  Linear probe on frozen ResNet-50 features provides a\n")
            f.write(f"  more standard MNIST benchmark for the image backbone.\n")

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write(f"Report generated by compare_mnist.py\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Comparison report saved to {report_path}")

    # Print summary to console
    logger.info("\n" + "=" * 60)
    logger.info("MNIST Comparison Summary")
    logger.info("=" * 60)
    if clip_acc is not None:
        logger.info(f"  CLIP (ViT-B/16) - Zero-shot:       {clip_acc:.2f}%")
    if baseline_zs_acc is not None:
        logger.info(f"  Baseline (ResNet-50) - Zero-shot:  {baseline_zs_acc:.2f}%")
    if baseline_lp_acc is not None:
        logger.info(f"  Baseline (ResNet-50) - Linear Probe: {baseline_lp_acc:.2f}%")
    logger.info("=" * 60)
    logger.info(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
