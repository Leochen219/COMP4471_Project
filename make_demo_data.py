import os
import json
import numpy as np
from PIL import Image

CAPTIONS = [
    "a photo of a cat sitting on a couch",
    "a dog playing fetch in the park",
    "a beautiful sunset over the ocean",
    "a person riding a bicycle on a road",
    "a red sports car parked on the street",
    "a cup of coffee on a wooden table",
    "a snow covered mountain landscape",
    "children playing in a playground",
    "a bird flying across the blue sky",
    "a plate of delicious spaghetti pasta",
]


def generate(num_samples, img_dir, json_path):
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    images_info = []
    annotations = []

    for i in range(num_samples):
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        fname = f"{i:06d}.jpg"
        save_path = os.path.join(img_dir, fname)
        Image.fromarray(arr).save(save_path)

        images_info.append({"id": i, "file_name": fname})
        annotations.append({
            "image_id": i,
            "id": i,
            "caption": CAPTIONS[i % len(CAPTIONS)],
        })

    coco = {"images": images_info, "annotations": annotations}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    # ===== 验证 =====
    for img in images_info:
        full = os.path.join(img_dir, img["file_name"])
        assert os.path.exists(full), f"文件不存在: {full}"

    print(f"  ✅ {num_samples} 条数据 → {json_path}")
    print(f"     图片目录: {os.path.abspath(img_dir)}")
    print(f"     示例文件: {images_info[0]['file_name']}")


def main():
    print("[PRTS] 正在生成 demo 数据...\n")

    generate(
        num_samples=64,
        img_dir="data/demo/images",
        json_path="data/demo/annotations/captions_train.json",
    )
    generate(
        num_samples=16,
        img_dir="data/demo/images",
        json_path="data/demo/annotations/captions_val.json",
    )

    print("\n[PRTS] 全部生成完毕。")


if __name__ == "__main__":
    main()