import os

import requests
from tqdm import tqdm


def download_file(url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(file_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                size = f.write(data)
                bar.update(size)


base_path = os.path.join("data", "coco")

urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]

for url in urls:
    download_file(url, base_path)
