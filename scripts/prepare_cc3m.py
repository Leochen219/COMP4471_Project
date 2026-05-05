#!/usr/bin/env python

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger("prepare_cc3m")

DEFAULT_URLS = {
    "train": "https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv",
    "val": "https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv",
}


def md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def download_tsv(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        LOGGER.info("tsv exists: %s", output)
        return

    tmp = output.with_suffix(output.suffix + ".part")
    cmd = ["wget", "-c", "-O", str(tmp), url]
    LOGGER.info("downloading tsv: %s", url)
    subprocess.run(cmd, check=True)
    tmp.rename(output)


def iter_tsv_rows(path: Path, start_index: int, limit: int):
    produced = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if idx < start_index:
                continue
            if limit and produced >= limit:
                break
            if len(row) < 2:
                continue

            first = row[0].strip()
            second = row[1].strip()
            if first.startswith("http://") or first.startswith("https://"):
                url, caption = first, second
            else:
                caption, url = first, second
            if not caption or not url.startswith(("http://", "https://")):
                continue

            produced += 1
            yield idx, caption, url


def target_relpath(split: str, idx: int, url: str) -> str:
    digest = md5_text(url)[:10]
    shard = f"{idx // 10000:05d}"
    return f"images/{split}/{shard}/{idx:09d}_{digest}.jpg"


def load_done_indices(manifest_path: Path) -> set[int]:
    done = set()
    if not manifest_path.exists():
        return done
    with manifest_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in item:
                done.add(int(item["index"]))
    return done


def fetch_image(
    idx: int,
    caption: str,
    url: str,
    root: Path,
    split: str,
    timeout: float,
    min_size: int,
    retries: int,
):
    relpath = target_relpath(split, idx, url)
    output = root / relpath
    if output.exists() and output.stat().st_size > 0:
        return {
            "index": idx,
            "caption": caption,
            "url": url,
            "image": relpath,
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    last_error = ""
    for _ in range(max(1, retries)):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                stream=False,
                allow_redirects=True,
            )
            if response.status_code != 200:
                last_error = f"http_{response.status_code}"
                continue
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            if image.width < min_size or image.height < min_size:
                last_error = "too_small"
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            image.save(output, format="JPEG", quality=95)
            return {
                "index": idx,
                "caption": caption,
                "url": url,
                "image": relpath,
            }
        except Exception as exc:
            last_error = type(exc).__name__

    return {
        "index": idx,
        "caption": caption,
        "url": url,
        "error": last_error or "failed",
    }


def prepare_split(args) -> None:
    root = Path(args.root)
    split = args.split
    tsv_path = Path(args.tsv) if args.tsv else root / "metadata" / f"{split}.tsv"
    manifest_path = (
        Path(args.output)
        if args.output
        else root / f"{split}_manifest.jsonl"
    )

    if args.download_tsv:
        download_tsv(args.tsv_url or DEFAULT_URLS[split], tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"Missing CC3M TSV: {tsv_path}. Pass --tsv, or try --download-tsv."
        )

    done_indices = load_done_indices(manifest_path) if args.resume else set()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "split=%s tsv=%s manifest=%s already_done=%d",
        split,
        tsv_path,
        manifest_path,
        len(done_indices),
    )

    submitted = 0
    succeeded = 0
    failed = 0
    pending = set()

    def submit_next(executor, rows):
        nonlocal submitted
        for idx, caption, url in rows:
            if idx in done_indices:
                continue
            future = executor.submit(
                fetch_image,
                idx,
                caption,
                url,
                root,
                split,
                args.timeout,
                args.min_size,
                args.retries,
            )
            pending.add(future)
            submitted += 1
            return True
        return False

    rows = iter_tsv_rows(tsv_path, args.start_index, args.limit)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor, manifest_path.open(
        "a", encoding="utf-8"
    ) as out:
        for _ in range(args.num_workers * 4):
            if not submit_next(executor, rows):
                break

        while pending:
            done, pending_left = wait(pending, return_when=FIRST_COMPLETED)
            pending = pending_left
            for future in done:
                item = future.result()
                if "image" in item:
                    out.write(json.dumps(item, ensure_ascii=False) + "\n")
                    succeeded += 1
                else:
                    failed += 1

                if (succeeded + failed) % args.log_interval == 0:
                    out.flush()
                    os.fsync(out.fileno())
                    LOGGER.info(
                        "processed=%d success=%d failed=%d submitted=%d",
                        succeeded + failed,
                        succeeded,
                        failed,
                        submitted,
                    )
                submit_next(executor, rows)

    LOGGER.info(
        "done split=%s submitted=%d success=%d failed=%d manifest=%s",
        split,
        submitted,
        succeeded,
        failed,
        manifest_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download CC3M images from the official Conceptual Captions TSV."
    )
    parser.add_argument("--root", default="data/cc3m")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--tsv", default="")
    parser.add_argument("--tsv-url", default="")
    parser.add_argument("--download-tsv", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[CC3M %(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    prepare_split(args)


if __name__ == "__main__":
    main()
