"""
Creates a reproducible train/val/test split over image IDs and saves it to
data/splits.json.
"""
import json
import random
from pathlib import Path

import yaml

from caption_parser import parse_caption_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)

    fl = cfg["flickr8k"]
    split_cfg = cfg["split"]

    captions_path = REPO_ROOT / fl["captions_file"]
    captions = parse_caption_file(captions_path)
    image_ids = sorted(captions.keys())

    rng = random.Random(split_cfg["seed"])
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}

    out_path = REPO_ROOT / fl["splits_file"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

    print(f"Total images: {n}")
    print(f"Train: {len(train_ids)}  Val: {len(val_ids)}  Test: {len(test_ids)}")
    print(f"Saved split to {out_path}")


if __name__ == "__main__":
    main()
