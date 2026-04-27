import argparse
import json
import os
import random
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def build_split(
    source_json: Path,
    image_dir: Path,
    output_dir: Path,
    dev_size: int,
    seed: int,
    prefix: str,
) -> dict:
    source = load_json(source_json)

    img_to_info = {img["id"]: img for img in source["images"]}
    ann_by_img: dict[int, list[dict]] = {}
    for ann in source["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    candidate_ids = []
    for img_id, img_info in img_to_info.items():
        file_path = image_dir / img_info["file_name"]
        if file_path.exists() and img_id in ann_by_img:
            candidate_ids.append(img_id)

    candidate_ids.sort()
    if dev_size <= 0 or dev_size >= len(candidate_ids):
        raise ValueError(
            f"dev_size must be in [1, {len(candidate_ids) - 1}], got {dev_size}"
        )

    rng = random.Random(seed)
    dev_ids = set(rng.sample(candidate_ids, dev_size))
    train_ids = set(candidate_ids) - dev_ids

    def build_payload(selected_ids: set[int]) -> dict:
        images = [img_to_info[i] for i in candidate_ids if i in selected_ids]
        annotations = []
        next_ann_id = 1
        for img in images:
            for ann in ann_by_img[img["id"]]:
                annotations.append({**ann, "id": next_ann_id})
                next_ann_id += 1
        payload = {
            "info": source.get("info", {}),
            "licenses": source.get("licenses", []),
            "images": images,
            "annotations": annotations,
        }
        return payload

    train_payload = build_payload(train_ids)
    dev_payload = build_payload(dev_ids)

    train_path = output_dir / f"{prefix}_train.json"
    dev_path = output_dir / f"{prefix}_dev.json"
    summary_path = output_dir / f"{prefix}_summary.json"

    dump_json(train_path, train_payload)
    dump_json(dev_path, dev_payload)

    summary = {
        "source_json": str(source_json),
        "image_dir": str(image_dir),
        "seed": seed,
        "dev_size": dev_size,
        "train_size": len(train_ids),
        "dev_actual_size": len(dev_ids),
        "source_images": len(source["images"]),
        "source_annotations": len(source["annotations"]),
        "candidate_images": len(candidate_ids),
        "train_annotations": len(train_payload["annotations"]),
        "dev_annotations": len(dev_payload["annotations"]),
        "train_json": str(train_path),
        "dev_json": str(dev_path),
    }
    dump_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed COCO train/dev split from captions_train2017.json"
    )
    parser.add_argument(
        "--source-json",
        default="data/coco/annotations/captions_train2017.json",
        help="Path to COCO captions_train2017.json",
    )
    parser.add_argument(
        "--image-dir",
        default="data/coco/train2017",
        help="Path to COCO train2017 image directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/coco_splits",
        help="Directory for generated split files",
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=5000,
        help="Number of images to sample for dev split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4471,
        help="Random seed for image-level sampling",
    )
    parser.add_argument(
        "--prefix",
        default="captions_train2017_dev5000",
        help="Output filename prefix",
    )
    args = parser.parse_args()

    summary = build_split(
        source_json=Path(args.source_json),
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        dev_size=args.dev_size,
        seed=args.seed,
        prefix=args.prefix,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
