# data/dataset.py

import os
import json
import random
import logging
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


class CleanCOCODataset(Dataset):
    """
    COCO Captions 数据集

    __getitem__ 返回:
        {
            "image":          [3, H, W]   float32
            "input_ids":      [max_length] int64
            "attention_mask": [max_length] int64
        }
    """

    def __init__(
        self,
        json_path: str,
        img_dir: str,
        transform=None,
        tokenizer=None,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        eval_mode: bool = False,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.max_length = max_length
        self.eval_mode = eval_mode  # True → 始终取第一条 caption

        # tokenizer: 支持外部传入（多 worker 共享）
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained(
            tokenizer_name
        )

        # ---------- 读取 JSON ----------
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # caption 按 image_id 归类
        self.img_to_captions: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            img_id: int = ann["image_id"]
            caption = self._clean_text(ann["caption"])
            if img_id not in self.img_to_captions:
                self.img_to_captions[img_id] = []
            self.img_to_captions[img_id].append(caption)

        # 过滤：必须有 caption 且文件存在
        self.img_data: list[dict] = []
        skipped = 0
        for img in data["images"]:
            path = os.path.join(self.img_dir, img["file_name"])
            if img["id"] in self.img_to_captions and os.path.exists(path):
                self.img_data.append(
                    {"file_name": img["file_name"], "id": img["id"]}
                )
            else:
                skipped += 1

        logger.info(
            f"[PRTS] 数据集就绪: {len(self.img_data)} 有效样本, "
            f"{skipped} 跳过"
        )

    # ----------------------------------------------------------
    @staticmethod
    def _clean_text(text) -> str:
        if not isinstance(text, str):
            text = str(text)
        return " ".join(text.strip().lower().split())

    def __len__(self) -> int:
        return len(self.img_data)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.img_data[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # 读取图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning(f"[PRTS] 图片损坏: {img_path}, 跳转替代样本")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        # 选取 caption
        captions = self.img_to_captions[img_info["id"]]
        if self.eval_mode:
            selected_caption = captions[0]
        else:
            selected_caption = random.choice(captions)

        # Tokenize
        text_inputs = self.tokenizer(
            selected_caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
        }
