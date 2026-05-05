# data/dataset.py

import os
import json
import random
import logging
import tarfile
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import CLIPTokenizer

logger = logging.getLogger(__name__)


def _tokenize_caption(tokenizer, caption: str, max_length: int) -> dict:
    text_inputs = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": text_inputs["input_ids"].squeeze(0),
        "attention_mask": text_inputs["attention_mask"].squeeze(0),
    }


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
        text_inputs = _tokenize_caption(
            self.tokenizer, selected_caption, self.max_length
        )

        return {
            "image": image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }


class ImageTextManifestDataset(Dataset):
    """
    Generic image-text dataset backed by a local manifest.

    Supported manifest formats:
      - JSONL: {"image": "relative/or/absolute/path.jpg", "caption": "..."}
        Aliases such as file_name/path/text are also accepted.
      - TSV: caption<TAB>local_image_path, or caption<TAB>url<TAB>local_image_path

    URL-only TSV rows are skipped because training needs local image files.
    Use scripts/prepare_cc3m.py to download CC3M URLs and create JSONL manifests.
    """

    def __init__(
        self,
        manifest_path: str,
        img_root: str = "",
        transform=None,
        tokenizer=None,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        max_samples: int = 0,
        dataset_name: str = "manifest",
    ):
        self.manifest_path = manifest_path
        self.img_root = img_root
        self.transform = transform
        self.max_length = max_length
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained(
            tokenizer_name
        )

        self.samples = self._load_samples(max_samples=max_samples)
        if not self.samples:
            raise ValueError(
                f"[PRTS] {dataset_name} manifest has no valid local images: "
                f"{manifest_path}"
            )

        logger.info(
            f"[PRTS] {dataset_name} 数据集就绪: "
            f"{len(self.samples)} 有效样本"
        )

    @staticmethod
    def _is_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    def _resolve_path(self, value: str) -> str:
        value = value.strip()
        if os.path.isabs(value):
            return value
        if self.img_root:
            return os.path.join(self.img_root, value)
        return value

    def _append_if_valid(self, samples: list[dict], image_value, caption) -> None:
        if not image_value or not caption:
            return
        image_value = str(image_value).strip()
        if self._is_url(image_value):
            return
        image_path = self._resolve_path(image_value)
        if os.path.exists(image_path):
            samples.append(
                {
                    "image_path": image_path,
                    "caption": CleanCOCODataset._clean_text(caption),
                }
            )

    def _load_jsonl(self, max_samples: int) -> list[dict]:
        samples: list[dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if max_samples and len(samples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                caption = (
                    item.get("caption")
                    or item.get("text")
                    or item.get("title")
                    or item.get("description")
                )
                image_value = (
                    item.get("image")
                    or item.get("image_path")
                    or item.get("path")
                    or item.get("file_name")
                    or item.get("filename")
                )
                self._append_if_valid(samples, image_value, caption)
        return samples

    def _load_tsv(self, max_samples: int) -> list[dict]:
        samples: list[dict] = []
        with open(self.manifest_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if max_samples and len(samples) >= max_samples:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue

                # Official CC3M TSV is caption<TAB>url. If a local image path is
                # present, it is usually in the last column.
                caption = parts[0]
                image_value = parts[-1]
                if self._is_url(caption) and not self._is_url(parts[1]):
                    caption, image_value = parts[1], parts[0]
                self._append_if_valid(samples, image_value, caption)
        return samples

    def _load_samples(self, max_samples: int) -> list[dict]:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(self.manifest_path)

        suffix = os.path.splitext(self.manifest_path)[1].lower()
        if suffix in {".jsonl", ".json"}:
            return self._load_jsonl(max_samples=max_samples)
        if suffix in {".tsv", ".txt", ".csv"}:
            return self._load_tsv(max_samples=max_samples)
        raise ValueError(f"[PRTS] Unsupported manifest format: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            logger.warning(
                f"[PRTS] 图片损坏: {sample['image_path']}, 跳转替代样本"
            )
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        text_inputs = _tokenize_caption(
            self.tokenizer, sample["caption"], self.max_length
        )

        return {
            "image": image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }


class CC3MDataset(ImageTextManifestDataset):
    def __init__(
        self,
        manifest_path: str,
        img_root: str = "data/cc3m",
        transform=None,
        tokenizer=None,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        max_samples: int = 0,
    ):
        super().__init__(
            manifest_path=manifest_path,
            img_root=img_root,
            transform=transform,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            max_samples=max_samples,
            dataset_name="cc3m",
        )


class CC3MWebDataset(Dataset):
    """
    Random-access reader for pixparse/cc3m-wds tar shards.

    Each sample in the shard has .jpg, .json and .txt members with the same key.
    The caption is read from the JSON "caption" field, falling back to .txt.
    """

    def __init__(
        self,
        root: str = "data/cc3m_wds",
        split: str = "train",
        transform=None,
        tokenizer=None,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_length: int = 77,
        max_samples: int = 0,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained(
            tokenizer_name
        )

        pattern = f"cc3m-{split}-*.tar"
        self.shards = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if name.startswith(f"cc3m-{split}-") and name.endswith(".tar")
        )
        if not self.shards:
            raise FileNotFoundError(f"[PRTS] no CC3M WDS shards: {root}/{pattern}")

        self.samples = self._index_shards(max_samples=max_samples)
        if not self.samples:
            raise ValueError(f"[PRTS] no valid CC3M WDS samples in {root}")

        self._tar_cache: dict[int, tarfile.TarFile] = {}
        self._member_cache: dict[int, dict[str, tarfile.TarInfo]] = {}
        logger.info(
            f"[PRTS] cc3m_wds 数据集就绪: {len(self.samples)} 样本, "
            f"{len(self.shards)} shards, split={split}"
        )

    def _index_shards(self, max_samples: int) -> list[dict]:
        info_path = os.path.join(self.root, "_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            split_key = self.split
            if split_key not in info.get("splits", {}) and split_key == "val":
                split_key = "validation"
            split_info = info.get("splits", {}).get(split_key)
            if split_info:
                filenames = split_info.get("filenames", [])
                shard_lengths = split_info.get("shard_lengths", [])
                if len(filenames) == len(shard_lengths):
                    samples: list[dict] = []
                    for shard_idx, (filename, shard_len) in enumerate(
                        zip(filenames, shard_lengths)
                    ):
                        expected = os.path.join(self.root, filename)
                        if shard_idx >= len(self.shards) or self.shards[shard_idx] != expected:
                            logger.warning(
                                "[PRTS] _info.json shard order differs from local files; "
                                "falling back to tar scan"
                            )
                            break
                        for offset in range(int(shard_len)):
                            samples.append({"shard_idx": shard_idx, "offset": offset})
                            if max_samples and len(samples) >= max_samples:
                                return samples
                    else:
                        return samples

        samples: list[dict] = []
        for shard_idx, path in enumerate(self.shards):
            try:
                with tarfile.open(path, "r") as tar:
                    keys: dict[str, set[str]] = {}
                    for member in tar:
                        if not member.isfile():
                            continue
                        stem, ext = os.path.splitext(member.name)
                        ext = ext.lower()
                        if ext in {".jpg", ".jpeg", ".json", ".txt"}:
                            keys.setdefault(stem, set()).add(ext)
                    for key in sorted(keys):
                        exts = keys[key]
                        if (".jpg" in exts or ".jpeg" in exts) and (
                            ".json" in exts or ".txt" in exts
                        ):
                            samples.append({"shard_idx": shard_idx, "key": key})
                            if max_samples and len(samples) >= max_samples:
                                return samples
            except tarfile.TarError as exc:
                logger.warning(f"[PRTS] 跳过损坏 shard: {path} ({exc})")
        return samples

    def _get_tar(self, shard_idx: int) -> tarfile.TarFile:
        tar = self._tar_cache.get(shard_idx)
        if tar is None:
            tar = tarfile.open(self.shards[shard_idx], "r")
            self._tar_cache[shard_idx] = tar
        return tar

    def _get_members(self, shard_idx: int) -> dict[str, tarfile.TarInfo]:
        members = self._member_cache.get(shard_idx)
        if members is None:
            tar = self._get_tar(shard_idx)
            members = {member.name: member for member in tar if member.isfile()}
            self._member_cache[shard_idx] = members
        return members

    def __len__(self) -> int:
        return len(self.samples)

    def _read_member(self, shard_idx: int, name: str) -> bytes | None:
        tar = self._get_tar(shard_idx)
        member = self._get_members(shard_idx).get(name)
        if member is None:
            return None
        file_obj = tar.extractfile(member)
        if file_obj is None:
            return None
        return file_obj.read()

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        shard_idx = sample["shard_idx"]
        key = sample.get("key")
        if key is None:
            members = self._get_members(shard_idx)
            image_names = sorted(
                name
                for name in members
                if name.endswith(".jpg") or name.endswith(".jpeg")
            )
            image_name = image_names[int(sample["offset"])]
            key = os.path.splitext(image_name)[0]

        try:
            image_bytes = self._read_member(shard_idx, f"{key}.jpg")
            if image_bytes is None:
                image_bytes = self._read_member(shard_idx, f"{key}.jpeg")
            if image_bytes is None:
                raise ValueError("missing image")
            from io import BytesIO

            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            caption = ""
            json_bytes = self._read_member(shard_idx, f"{key}.json")
            if json_bytes is not None:
                metadata = json.loads(json_bytes.decode("utf-8", errors="ignore"))
                caption = metadata.get("caption", "")
            if not caption:
                txt_bytes = self._read_member(shard_idx, f"{key}.txt")
                if txt_bytes is not None:
                    caption = txt_bytes.decode("utf-8", errors="ignore")
            caption = CleanCOCODataset._clean_text(caption)
            if not caption:
                raise ValueError("missing caption")
        except Exception as exc:
            logger.warning(f"[PRTS] CC3M WDS 样本损坏: {key} ({exc})")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        text_inputs = _tokenize_caption(self.tokenizer, caption, self.max_length)

        return {
            "image": image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tar_cache"] = {}
        state["_member_cache"] = {}
        return state


class StreamingCOCOCC3MDataset(IterableDataset):
    """
    Stream CC3M WDS shards sequentially and mix in random COCO samples.

    This avoids random tar access and long upfront indexing while still keeping
    COCO in the training stream.
    """

    def __init__(
        self,
        coco_dataset: CleanCOCODataset,
        cc3m_root: str = "data/cc3m_wds",
        split: str = "train",
        transform=None,
        tokenizer=None,
        max_length: int = 77,
        mix_weights: list[float] | None = None,
        epoch_size: int = 240000,
    ):
        self.coco_dataset = coco_dataset
        self.cc3m_root = cc3m_root
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.epoch_size = int(epoch_size)
        self.mix_weights = mix_weights or [1.0, 3.0]
        self.cc3m_probability = self.mix_weights[1] / sum(self.mix_weights)

        self.shards = sorted(
            os.path.join(cc3m_root, name)
            for name in os.listdir(cc3m_root)
            if name.startswith(f"cc3m-{split}-") and name.endswith(".tar")
        )
        if not self.shards:
            raise FileNotFoundError(f"[PRTS] no CC3M WDS shards in {cc3m_root}")
        logger.info(
            f"[PRTS] streaming COCO+CC3M: coco={len(coco_dataset)} | "
            f"cc3m_shards={len(self.shards)} | epoch_size={self.epoch_size}"
        )

    def __len__(self) -> int:
        return self.epoch_size

    @staticmethod
    def _key_from_name(name: str) -> str:
        return os.path.splitext(os.path.basename(name))[0]

    def _iter_cc3m_samples(self, worker_id: int, num_workers: int):
        from io import BytesIO

        shard_indices = list(range(worker_id, len(self.shards), num_workers))
        random.shuffle(shard_indices)
        while True:
            for shard_idx in shard_indices:
                shard_path = self.shards[shard_idx]
                try:
                    with tarfile.open(shard_path, "r") as tar:
                        pending: dict[str, dict[str, bytes]] = {}
                        for member in tar:
                            if not member.isfile():
                                continue
                            ext = os.path.splitext(member.name)[1].lower()
                            if ext not in {".jpg", ".jpeg", ".json", ".txt"}:
                                continue
                            file_obj = tar.extractfile(member)
                            if file_obj is None:
                                continue
                            key = self._key_from_name(member.name)
                            item = pending.setdefault(key, {})
                            item[ext] = file_obj.read()
                            if (
                                (".jpg" in item or ".jpeg" in item)
                                and (".json" in item or ".txt" in item)
                            ):
                                try:
                                    image_bytes = item.get(".jpg") or item.get(".jpeg")
                                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                                    caption = ""
                                    if ".json" in item:
                                        metadata = json.loads(
                                            item[".json"].decode(
                                                "utf-8", errors="ignore"
                                            )
                                        )
                                        caption = metadata.get("caption", "")
                                    if not caption and ".txt" in item:
                                        caption = item[".txt"].decode(
                                            "utf-8", errors="ignore"
                                        )
                                    caption = CleanCOCODataset._clean_text(caption)
                                    if not caption:
                                        continue
                                    yield image, caption
                                except Exception as exc:
                                    logger.warning(
                                        f"[PRTS] CC3M WDS 流式样本损坏: "
                                        f"{key} ({exc})"
                                    )
                                finally:
                                    pending.pop(key, None)
                except tarfile.TarError as exc:
                    logger.warning(f"[PRTS] 跳过损坏 shard: {shard_path} ({exc})")

    def _format_sample(self, image: Image.Image, caption: str) -> dict:
        if self.transform:
            image = self.transform(image)
        text_inputs = _tokenize_caption(self.tokenizer, caption, self.max_length)
        return {
            "image": image,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        per_worker = self.epoch_size // num_workers
        if worker_id < self.epoch_size % num_workers:
            per_worker += 1

        cc3m_iter = self._iter_cc3m_samples(worker_id, num_workers)
        for _ in range(per_worker):
            if random.random() < self.cc3m_probability:
                image, caption = next(cc3m_iter)
                yield self._format_sample(image, caption)
            else:
                idx = random.randrange(len(self.coco_dataset))
                yield self.coco_dataset[idx]


class RandomMixDataset(Dataset):
    """
    Sample from multiple datasets with configurable probabilities.

    This avoids letting a large web dataset completely drown out COCO while still
    keeping a fixed number of training samples per epoch.
    """

    def __init__(
        self,
        datasets: list[Dataset],
        weights: list[float] | None = None,
        epoch_size: int | None = None,
    ):
        if not datasets:
            raise ValueError("[PRTS] RandomMixDataset requires at least one dataset")
        self.datasets = datasets
        if weights is None:
            weights = [float(len(dataset)) for dataset in datasets]
        if len(weights) != len(datasets):
            raise ValueError("[PRTS] train_mix_weights length must match datasets")
        if any(weight <= 0 for weight in weights):
            raise ValueError("[PRTS] train_mix_weights must be positive")

        total_weight = float(sum(weights))
        running = 0.0
        self.cum_weights = []
        for weight in weights:
            running += weight / total_weight
            self.cum_weights.append(running)
        self.epoch_size = int(epoch_size or sum(len(dataset) for dataset in datasets))

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        del idx
        r = random.random()
        dataset_idx = 0
        for i, threshold in enumerate(self.cum_weights):
            if r <= threshold:
                dataset_idx = i
                break
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randrange(len(dataset))
        return dataset[sample_idx]
