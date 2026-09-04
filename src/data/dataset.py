"""
PyTorch Dataset for Flickr8k-Amharic image captioning.
"""
import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

from caption_parser import parse_caption_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Flickr8kAmharicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        split: str = "train",
        splits_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        tokenizer: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.tokenizer = tokenizer

        all_captions = parse_caption_file(Path(captions_file))

        if splits_file is not None and Path(splits_file).exists():
            with open(splits_file, "r", encoding="utf-8") as f:
                splits = json.load(f)
            if split not in splits:
                raise ValueError(f"split={split!r} not in splits file keys: {list(splits.keys())}")
            allowed_ids = set(splits[split])
            captions = {k: v for k, v in all_captions.items() if k in allowed_ids}
        else:
            captions = all_captions

        # Validate EVERY image_id actually exists on disk before building
        # samples — full validation, not a spot check, so malformed
        # caption-file lines never reach training.
        existing_files = set(f.name for f in self.images_dir.iterdir()) if self.images_dir.exists() else set()
        valid_captions = {}
        dropped_images = 0
        for img_id, caps in captions.items():
            if img_id in existing_files:
                valid_captions[img_id] = caps
            else:
                dropped_images += 1

        if dropped_images:
            print(f"[WARN] Dropped {dropped_images} image_id(s) with no matching file in "
                  f"{self.images_dir} (likely malformed caption-file lines). "
                  f"Kept {len(valid_captions)} valid images.")

        self.samples: List[Tuple[str, str]] = [
            (img_id, cap) for img_id, caps in valid_captions.items() for cap in caps
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, caption = self.samples[idx]
        img_path = self.images_dir / img_id
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


if __name__ == "__main__":
    import yaml

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)["flickr8k"]

    ds = Flickr8kAmharicDataset(
        images_dir=REPO_ROOT / cfg["images_dir"],
        captions_file=REPO_ROOT / cfg["captions_file"],
        split="train",
        splits_file=REPO_ROOT / cfg["splits_file"],
    )
    print(f"Dataset size: {len(ds)}")
    image, caption = ds[0]
    print(f"Sample image: {image.size}, mode={image.mode}")
    print(f"Sample caption: {caption}")
