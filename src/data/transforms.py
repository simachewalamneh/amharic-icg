"""
Image transforms and collate function for batching variable-length caption
sequences.
"""
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 299) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 299) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def collate_fn(batch: List[Tuple[torch.Tensor, List[int]]], pad_id: int = 0):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)

    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
    caption_tensors = [torch.tensor(c, dtype=torch.long) for c in captions]
    captions_padded = pad_sequence(caption_tensors, batch_first=True, padding_value=pad_id)

    return images, captions_padded, lengths


if __name__ == "__main__":
    import json
    from pathlib import Path

    import yaml
    from torch.utils.data import DataLoader

    from dataset import Flickr8kAmharicDataset
    from tokenizer import Vocabulary, WordTokenizer, PAD

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)["flickr8k"]

    vocab_path = REPO_ROOT / "data" / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"{vocab_path} not found — run 'python src/data/tokenizer.py' first to build the vocab."
        )
    vocab = Vocabulary.load(vocab_path)
    tokenizer = WordTokenizer(vocab)
    pad_id = vocab.token_to_id[PAD]

    ds = Flickr8kAmharicDataset(
        images_dir=REPO_ROOT / cfg["images_dir"],
        captions_file=REPO_ROOT / cfg["captions_file"],
        split="train",
        splits_file=REPO_ROOT / cfg["splits_file"],
        transform=get_train_transform(image_size=299),
        tokenizer=tokenizer,
    )

    loader = DataLoader(
        ds, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    images, captions, lengths = next(iter(loader))
    print(f"Vocab size: {len(vocab)}")
    print(f"Batch images shape: {images.shape}")
    print(f"Batch captions shape: {captions.shape}")
    print(f"Lengths: {lengths.tolist()}")
    print(f"First caption ids: {captions[0].tolist()}")
    print(f"First caption decoded: {' '.join(vocab.decode(captions[0].tolist()))}")
