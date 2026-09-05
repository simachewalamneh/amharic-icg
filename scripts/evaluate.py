"""
Evaluates a trained checkpoint using REAL autoregressive generation
(model.generate(), never teacher forcing) and computes BLEU-1..4 against
the reference captions — this is the number to actually compare against
the baseline paper's reported 60.6 / 50.1 / 43.7 / 38.8.

Usage:
    python scripts/evaluate.py --checkpoint experiments/baseline_bigru/best_model.pt --split test
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "data"))
sys.path.insert(0, str(REPO_ROOT / "src" / "models"))

from dataset import Flickr8kAmharicDataset          # noqa: E402
from tokenizer import Vocabulary, WordTokenizer, PAD, SOS, EOS, basic_tokenize  # noqa: E402
from transforms import get_eval_transform, collate_fn  # noqa: E402
from baseline_bigru import BiGRUBaselineModel        # noqa: E402
from caption_parser import parse_caption_file        # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)["flickr8k"]

    vocab = Vocabulary.load(REPO_ROOT / "data" / "vocab.json")
    pad_id = vocab.token_to_id[PAD]
    sos_id = vocab.token_to_id[SOS]
    eos_id = vocab.token_to_id[EOS]

    # We need ALL reference captions per image (not flattened one-per-sample)
    # for a fair multi-reference BLEU score.
    all_captions = parse_caption_file(REPO_ROOT / cfg["captions_file"])
    with open(REPO_ROOT / cfg["splits_file"]) as f:
        splits = json.load(f)
    split_ids = splits[args.split]

    tokenizer = WordTokenizer(vocab)
    ds = Flickr8kAmharicDataset(
        images_dir=REPO_ROOT / cfg["images_dir"],
        captions_file=REPO_ROOT / cfg["captions_file"],
        split=args.split,
        splits_file=REPO_ROOT / cfg["splits_file"],
        transform=get_eval_transform(image_size=299),
        tokenizer=None,  # we tokenize references separately below; images only needed here
    )

    # Deduplicate to ONE sample per image (dataset normally repeats each
    # image once per caption — for generation we only need to run the
    # model once per image, then compare against ALL its references).
    seen = set()
    unique_image_ids = []
    for img_id, _ in ds.samples:
        if img_id not in seen:
            seen.add(img_id)
            unique_image_ids.append(img_id)

    model = BiGRUBaselineModel(vocab_size=len(vocab), pad_id=pad_id).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    hypotheses = []
    references = []

    print(f"Generating captions for {len(unique_image_ids)} images in '{args.split}' split...")
    batch_size = args.batch_size
    for i in range(0, len(unique_image_ids), batch_size):
        batch_ids = unique_image_ids[i:i + batch_size]
        images = torch.stack([
            get_eval_transform(299)(Image.open(ds.images_dir / img_id).convert("RGB"))
            for img_id in batch_ids
        ]).to(device)

        with torch.no_grad():
            generated = model.generate(images, sos_id=sos_id, eos_id=eos_id, max_len=args.max_len)

        for row, img_id in zip(generated, batch_ids):
            ids = row.tolist()
            # strip sos, stop at eos
            if ids[0] == sos_id:
                ids = ids[1:]
            if eos_id in ids:
                ids = ids[:ids.index(eos_id)]
            tokens = [vocab.id_to_token.get(t, "<unk>") for t in ids]
            hypotheses.append(tokens)

            ref_tokens = [basic_tokenize(c) for c in all_captions[img_id]]
            references.append(ref_tokens)

        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch_ids)}/{len(unique_image_ids)} done")

    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    print("\n=== Results (real autoregressive generation, not teacher-forced) ===")
    print(f"BLEU-1: {bleu1*100:.1f}")
    print(f"BLEU-2: {bleu2*100:.1f}")
    print(f"BLEU-3: {bleu3*100:.1f}")
    print(f"BLEU-4: {bleu4*100:.1f}")
    print("\nPaper's reported baseline (Flickr8k+BNATURE): BLEU-1 60.6, BLEU-2 50.1, BLEU-3 43.7, BLEU-4 38.8")

    print("\n--- Sample generated captions ---")
    for k in range(min(5, len(hypotheses))):
        print(f"[{unique_image_ids[k]}]")
        print(f"  Generated: {' '.join(hypotheses[k])}")
        print(f"  Reference: {' '.join(references[k][0])}")


if __name__ == "__main__":
    main()
