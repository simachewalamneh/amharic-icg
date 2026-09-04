"""
Trains the Bi-GRU baseline model.

Usage:
    python scripts/train_baseline.py --debug
    python scripts/train_baseline.py --epochs 20 --batch_size 32
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from dataset import Flickr8kAmharicDataset          # noqa: E402
from tokenizer import Vocabulary, WordTokenizer, PAD  # noqa: E402
from transforms import get_train_transform, get_eval_transform, collate_fn  # noqa: E402
from baseline_bigru import BiGRUBaselineModel        # noqa: E402


def build_dataloaders(cfg, vocab, pad_id, batch_size, debug=False):
    tokenizer = WordTokenizer(vocab)

    train_ds = Flickr8kAmharicDataset(
        images_dir=REPO_ROOT / cfg["images_dir"],
        captions_file=REPO_ROOT / cfg["captions_file"],
        split="train",
        splits_file=REPO_ROOT / cfg["splits_file"],
        transform=get_train_transform(image_size=299),
        tokenizer=tokenizer,
    )
    val_ds = Flickr8kAmharicDataset(
        images_dir=REPO_ROOT / cfg["images_dir"],
        captions_file=REPO_ROOT / cfg["captions_file"],
        split="val",
        splits_file=REPO_ROOT / cfg["splits_file"],
        transform=get_eval_transform(image_size=299),
        tokenizer=tokenizer,
    )

    if debug:
        train_ds.samples = train_ds.samples[:32]
        val_ds.samples = val_ds.samples[:16]

    collate = lambda b: collate_fn(b, pad_id=pad_id)  # noqa: E731

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, captions, lengths in loader:
        images, captions = images.to(device), captions.to(device)

        optimizer.zero_grad()
        logits = model(images, captions)

        targets = captions[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for images, captions, lengths in loader:
        images, captions = images.to(device), captions.to(device)
        logits = model(images, captions)
        targets = captions[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fine_tune_encoder", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu" and not args.debug:
        print("[WARN] No GPU detected and --debug not set. This will be slow. "
              "Consider running on Colab/Kaggle, or add --debug to sanity-check first.")

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)["flickr8k"]

    vocab_path = REPO_ROOT / "data" / "vocab.json"
    vocab = Vocabulary.load(vocab_path)
    pad_id = vocab.token_to_id[PAD]

    batch_size = 4 if args.debug else args.batch_size
    epochs = 1 if args.debug else args.epochs

    train_loader, val_loader = build_dataloaders(
        cfg, vocab, pad_id, batch_size=batch_size, debug=args.debug
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = BiGRUBaselineModel(
        vocab_size=len(vocab), pad_id=pad_id, fine_tune_encoder=args.fine_tune_encoder
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    checkpoint_dir = REPO_ROOT / "experiments" / "baseline_bigru"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        dt = time.time() - t0

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  Saved new best checkpoint (val_loss={val_loss:.4f})")

    print("Done." if not args.debug else "Debug run complete — pipeline verified end to end.")


if __name__ == "__main__":
    main()
