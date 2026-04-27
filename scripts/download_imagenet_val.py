#!/usr/bin/env python

import argparse
import hashlib
import subprocess
from pathlib import Path


VAL_URLS = [
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    "https://image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar",
]
DEVKIT_URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz"
CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

VAL_MD5 = "29b22e2961454d5413ddabcf34fc5622"
DEVKIT_MD5 = "fa75699e90414af021442c21a62c3abf"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_wget(url: str, output: Path) -> bool:
    cmd = ["wget", "-O", str(output), url]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def ensure_file(urls: list[str], output: Path, expected_md5: str) -> None:
    if output.exists():
        current = md5sum(output)
        if current == expected_md5:
            print(f"exists and md5 ok: {output}")
            return
        print(f"existing file md5 mismatch: {output} {current} != {expected_md5}")
        bad = output.with_suffix(output.suffix + ".bad")
        output.rename(bad)
        print(f"renamed bad file to: {bad}")

    for url in urls:
        tmp = output.with_suffix(output.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        if not run_wget(url, tmp):
            continue
        current = md5sum(tmp)
        if current == expected_md5:
            tmp.rename(output)
            print(f"downloaded and verified: {output}")
            return
        print(f"md5 mismatch from {url}: {current} != {expected_md5}")
        tmp.rename(output.with_suffix(output.suffix + ".bad"))

    raise RuntimeError(f"Failed to download a verified copy of {output.name}")


def main():
    parser = argparse.ArgumentParser(description="Download ImageNet val files with md5 verification")
    parser.add_argument("--root", default="/data/ydongbd/datasets/imagenet")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    ensure_file(VAL_URLS, root / "ILSVRC2012_img_val.tar", VAL_MD5)
    ensure_file([DEVKIT_URL], root / "ILSVRC2012_devkit_t12.tar.gz", DEVKIT_MD5)

    class_index = root / "imagenet_class_index.json"
    if not class_index.exists():
        run_wget(CLASS_INDEX_URL, class_index)


if __name__ == "__main__":
    main()
