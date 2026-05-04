# baseline/download_cifar.py
#
# Manual CIFAR-10/100 download helper with resume support.
# Use this if torchvision's auto-download fails (HTTP 503, etc.)
#
# Usage:
#   python baseline/download_cifar.py cifar100
#   python baseline/download_cifar.py cifar10
#
# The script will:
#   1. Try the official cs.toronto.edu server first
#   2. Fall back to the fast-ai S3 mirror if the official server is down
#   3. Extract the archive into the format torchvision expects

import argparse
import hashlib
import os
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path


# ── URLs ──────────────────────────────────────────────────────────────────
OFFICIAL_URLS = {
    "cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}

# fast-ai S3 mirror (same pickle format as official)
MIRROR_URLS = {
    "cifar10": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    "cifar100": "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz",
}

# Expected archive names
ARCHIVE_NAMES = {
    "cifar10": "cifar-10-python.tar.gz",
    "cifar100": "cifar-100-python.tar.gz",
}

# Expected directories inside the archive
INNER_DIRS = {
    "cifar10": "cifar-10-batches-py",
    "cifar100": "cifar-100-python",
}

# Expected files inside the archive (for validation)
EXPECTED_FILES = {
    "cifar10": [
        "cifar-10-batches-py/data_batch_1",
        "cifar-10-batches-py/data_batch_2",
        "cifar-10-batches-py/data_batch_3",
        "cifar-10-batches-py/data_batch_4",
        "cifar-10-batches-py/data_batch_5",
        "cifar-10-batches-py/test_batch",
        "cifar-10-batches-py/batches.meta",
    ],
    "cifar100": [
        "cifar-100-python/train",
        "cifar-100-python/test",
        "cifar-100-python/meta",
    ],
}


class DownloadProgress:
    """Simple progress callback for urlretrieve."""

    def __init__(self, name: str):
        self.name = name
        self.last_pct = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = int(100 * downloaded / total_size)
        if pct != self.last_pct and pct % 5 == 0:
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {self.name}: {pct}% ({mb:.1f}/{total_mb:.1f} MB)", end="")
            self.last_pct = pct
        if pct == 100:
            print()


def check_url(url: str, timeout: int = 5) -> bool:
    """Check if a URL is accessible."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def download_with_resume(url: str, dest: str, timeout: int = 300):
    """Download a file with resume support and progress reporting."""
    name = os.path.basename(url)

    # Check if partial download exists
    existing_size = 0
    mode = "wb"
    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)
        if existing_size > 0:
            mode = "ab"
            print(f"  Resuming download from {existing_size / (1024*1024):.1f} MB")

    req = urllib.request.Request(url)
    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + existing_size
            if resp.status == 206:  # Partial Content
                total = int(resp.headers.get("Content-Length", 0)) + existing_size
            elif resp.status == 200:
                total = int(resp.headers.get("Content-Length", 0))
                mode = "wb"  # Server doesn't support resume, start over
                existing_size = 0

            progress = DownloadProgress(name)
            downloaded = existing_size

            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress(downloaded // 8192, 8192, total)

        print(f"  Download complete: {dest}")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def verify_archive(path: str, name: str) -> bool:
    """Verify that the archive contains the expected files."""
    try:
        with tarfile.open(path, "r:gz") as tar:
            members = {m.name for m in tar.getmembers()}
            expected = set(EXPECTED_FILES[name])
            missing = expected - members
            if missing:
                print(f"  Archive is missing files: {missing}")
                return False
            print(f"  Archive verified: {len(members)} members, all expected files present")
            return True
    except Exception as e:
        print(f"  Archive verification failed: {e}")
        return False


def download_cifar(name: str, root: str = "./data", timeout: int = 300):
    """Download CIFAR dataset, trying official URL first, then mirrors."""
    root = os.path.abspath(root)
    archive_name = ARCHIVE_NAMES[name]
    archive_path = os.path.join(root, archive_name)
    target_dir = os.path.join(root, INNER_DIRS[name])

    # Check if already extracted
    if os.path.isdir(target_dir):
        # Verify it has the expected files
        all_ok = all(os.path.isfile(os.path.join(target_dir, f.replace(INNER_DIRS[name] + "/", "")))
                     for f in EXPECTED_FILES[name])
        if all_ok:
            print(f"✓ {name} already exists and is valid at {target_dir}")
            return True
        else:
            print(f"  {target_dir} exists but is incomplete, re-downloading...")

    os.makedirs(root, exist_ok=True)

    # ── Try official URL first ──
    official_url = OFFICIAL_URLS[name]
    print(f"\nTrying official URL: {official_url}")
    if check_url(official_url):
        print("  Server is accessible, downloading...")
        success = download_with_resume(official_url, archive_path, timeout)
    else:
        print("  Official server is down (HTTP 503 or timeout)")

    # ── Fall back to mirror ──
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
        mirror_url = MIRROR_URLS[name]
        print(f"\nTrying mirror: {mirror_url}")
        if check_url(mirror_url):
            print("  Mirror is accessible, downloading...")
            success = download_with_resume(mirror_url, archive_path, timeout)
        else:
            print("  Mirror also unavailable!")
            return False

    # ── Verify archive ──
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
        print("  Download failed: no data received")
        return False

    if not verify_archive(archive_path, name):
        print("  Archive is corrupted, deleting...")
        os.remove(archive_path)
        return False

    # ── Extract ──
    print(f"\nExtracting {archive_path}...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root)
        print(f"  Extracted to {target_dir}")
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False

    # ── Clean up archive ──
    os.remove(archive_path)
    print(f"  Removed archive {archive_path}")

    # ── Verify final structure ──
    all_ok = all(
        os.path.isfile(os.path.join(target_dir, Path(f).name))
        for f in EXPECTED_FILES[name]
    )
    if all_ok:
        print(f"✓ {name} ready at {target_dir}")
        return True
    else:
        print(f"  Warning: {target_dir} may be incomplete")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download CIFAR-10/100 manually with fallback mirrors"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="cifar100",
        choices=list(OFFICIAL_URLS.keys()),
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    success = download_cifar(args.dataset, args.data_root, args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
