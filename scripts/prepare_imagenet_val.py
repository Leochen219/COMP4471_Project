#!/usr/bin/env python

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def read_validation_ground_truth(devkit_tar: Path) -> list[int]:
    target_name = "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    with tarfile.open(devkit_tar, "r:gz") as tar:
        member = tar.getmember(target_name)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Cannot read {target_name} from {devkit_tar}")
        return [int(line) for line in file_obj.read().decode("utf-8").splitlines() if line.strip()]


def load_idx_to_wnid(devkit_tar: Path) -> dict[int, str]:
    """Map official ILSVRC2012 class ids to WNIDs from the devkit meta.mat."""
    try:
        import scipy.io
    except ImportError as exc:
        raise RuntimeError(
            "Preparing ImageNet val requires scipy to read devkit meta.mat. "
            "Install scipy or run this script in an environment that has it."
        ) from exc

    target_name = "ILSVRC2012_devkit_t12/data/meta.mat"
    with tarfile.open(devkit_tar, "r:gz") as tar:
        member = tar.getmember(target_name)
        file_obj = tar.extractfile(member)
        if file_obj is None:
            raise RuntimeError(f"Cannot read {target_name} from {devkit_tar}")
        synsets = scipy.io.loadmat(file_obj, squeeze_me=True)["synsets"]

    mapping = {}
    for synset in synsets:
        if int(synset["num_children"]) != 0:
            continue
        mapping[int(synset["ILSVRC2012_ID"])] = str(synset["WNID"])
    return mapping


def load_class_index_wnids(class_index_json: Path) -> set[str]:
    if not class_index_json.exists():
        return set()
    data = json.loads(class_index_json.read_text(encoding="utf-8"))
    return {str(value[0]) for value in data.values()}


def extract_val_images(val_tar: Path, flat_dir: Path) -> None:
    flat_dir.mkdir(parents=True, exist_ok=True)
    existing = list(flat_dir.glob("ILSVRC2012_val_*.JPEG"))
    if len(existing) == 50000:
        print(f"skip extract: {flat_dir} already contains 50000 images")
        return
    with tarfile.open(val_tar, "r:") as tar:
        tar.extractall(flat_dir)


def organize(flat_dir: Path, output_dir: Path, val_labels: list[int], idx_to_wnid: dict[int, str]) -> None:
    images = sorted(flat_dir.glob("ILSVRC2012_val_*.JPEG"))
    if len(images) != len(val_labels):
        raise RuntimeError(
            f"Expected {len(val_labels)} validation images, found {len(images)} in {flat_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for wnid in sorted(set(idx_to_wnid.values())):
        (output_dir / wnid).mkdir(exist_ok=True)

    moved = 0
    for image_path, label_idx in zip(images, val_labels):
        wnid = idx_to_wnid[label_idx]
        target = output_dir / wnid / image_path.name
        if target.exists():
            continue
        shutil.move(str(image_path), str(target))
        moved += 1
    print(f"organized {len(images)} images into {output_dir} ({moved} moved)")


def reorganize_existing(output_dir: Path, val_labels: list[int], idx_to_wnid: dict[int, str]) -> None:
    current_paths = {
        path.name: path
        for path in output_dir.glob("*/*.JPEG")
    }
    if len(current_paths) != len(val_labels):
        raise RuntimeError(
            f"Expected {len(val_labels)} existing images, found {len(current_paths)} in {output_dir}"
        )

    moved = 0
    for image_idx, label_idx in enumerate(val_labels, start=1):
        image_name = f"ILSVRC2012_val_{image_idx:08d}.JPEG"
        source = current_paths.get(image_name)
        if source is None:
            raise RuntimeError(f"Missing validation image: {image_name}")
        target_dir = output_dir / idx_to_wnid[label_idx]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image_name
        if source == target:
            continue
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite existing file: {target}")
        shutil.move(str(source), str(target))
        moved += 1
    print(f"reorganized existing ImageNet val folders ({moved} moved)")


def count_images(output_dir: Path) -> int:
    return sum(1 for _ in output_dir.glob("*/*.JPEG"))


def main():
    parser = argparse.ArgumentParser(description="Prepare ImageNet val as ImageFolder")
    parser.add_argument("--root", default="/data/ydongbd/datasets/imagenet")
    parser.add_argument("--val-tar", default="")
    parser.add_argument("--devkit-tar", default="")
    parser.add_argument("--class-index-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--keep-flat", action="store_true")
    parser.add_argument(
        "--force-reorganize",
        action="store_true",
        help="Rebuild existing ImageFolder labels using official devkit meta.mat.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    val_tar = Path(args.val_tar) if args.val_tar else root / "ILSVRC2012_img_val.tar"
    devkit_tar = Path(args.devkit_tar) if args.devkit_tar else root / "ILSVRC2012_devkit_t12.tar.gz"
    class_index_json = (
        Path(args.class_index_json)
        if args.class_index_json
        else root / "imagenet_class_index.json"
    )
    output_dir = Path(args.output_dir) if args.output_dir else root / "val"
    flat_dir = root / "val_flat"

    for path in [val_tar, devkit_tar]:
        if not path.exists():
            raise FileNotFoundError(path)

    val_labels = read_validation_ground_truth(devkit_tar)
    idx_to_wnid = load_idx_to_wnid(devkit_tar)
    if len(val_labels) != 50000:
        raise RuntimeError(f"Expected 50000 labels, got {len(val_labels)}")
    if len(idx_to_wnid) != 1000:
        raise RuntimeError(f"Expected 1000 ImageNet classes, got {len(idx_to_wnid)}")
    class_index_wnids = load_class_index_wnids(class_index_json)
    if class_index_wnids and set(idx_to_wnid.values()) != class_index_wnids:
        raise RuntimeError(
            "Devkit WNIDs and class-index WNIDs do not match; refusing to prepare data."
        )

    if count_images(output_dir) == 50000:
        if args.force_reorganize:
            reorganize_existing(output_dir, val_labels, idx_to_wnid)
            print(f"final image count: {count_images(output_dir)}")
            return
        print(f"skip organize: {output_dir} already contains 50000 images")
        return

    extract_val_images(val_tar, flat_dir)
    organize(flat_dir, output_dir, val_labels, idx_to_wnid)

    total = count_images(output_dir)
    print(f"final image count: {total}")
    if total != 50000:
        raise RuntimeError(f"Expected 50000 organized images, got {total}")

    if not args.keep_flat and flat_dir.exists() and not any(flat_dir.iterdir()):
        flat_dir.rmdir()


if __name__ == "__main__":
    main()
