import os
import sys
import urllib.error
import urllib.request
import zipfile

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
TIMEOUT = 30


@hydra.main(version_base=None, config_path="../configs/data", config_name="weather")
def main(cfg: DictConfig) -> None:
    root = cfg.path
    if not os.path.isabs(root):
        root = os.path.join(get_original_cwd(), root)
    dest = os.path.abspath(root)
    os.makedirs(dest, exist_ok=True)

    csv_path = os.path.join(dest, cfg.name)

    if os.path.exists(csv_path):
        print(f"[download_weather] {csv_path} already exists, skipping")
        return

    part_path = os.path.join(dest, cfg.name + ".zip.part")

    print(f"[download_weather] fetching {URL} -> {csv_path}")
    try:
        with urllib.request.urlopen(URL, timeout=TIMEOUT) as resp:
            with open(part_path, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
    except urllib.error.URLError as e:
        print(f"[download_weather] download failed: {e.reason} ({URL})")
        sys.exit(1)

    print(f"[download_weather] extracting to {dest}")
    try:
        with zipfile.ZipFile(part_path) as zf:
            zf.extract(cfg.name, dest)
    except zipfile.BadZipFile:
        print("[download_weather] failed: downloaded file is not a valid zip")
        sys.exit(1)
    finally:
        os.remove(part_path)

    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        print("[download_weather] failed: extracted CSV missing or empty")
        sys.exit(1)

    print(f"[download_weather] done -> {csv_path}")


if __name__ == "__main__":
    main()