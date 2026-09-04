# Amharic Image Captioning — Morphology-Aware CLIP/ViT-Transformer

Research project extending Amharic image caption generation beyond the
CNN-BiGRU baseline (Solomon & Abebe, 2023) using a CLIP/ViT visual encoder,
a transformer decoder, and a **morphology-aware Amharic tokenizer** (rather
than word-level or plain subword/BPE tokenization).

## Motivation

- Existing Amharic ICG baseline: Inception-v3 CNN encoder + visual attention
  + Bi-GRU attention decoder, trained on translated Flickr8k/BNATURE.
- Existing Amharic multimodal work (Amharic LLaMA/LLaVA) pairs a CLIP encoder
  with an Amharic LLM decoder, but notes the tokenizer is poorly suited to
  Amharic's morphology.
- **Gap this project targets**: nobody has tested whether morphology-aware
  segmentation (vs. word-level or standard BPE/WordPiece) improves Amharic
  caption quality when paired with a CLIP/ViT + transformer decoder.

## Project structure

```
amharic-icg/
├── data/               # datasets (gitignored — see data/README.md)
├── src/
│   ├── models/         # encoder, decoder, full model definitions
│   ├── data/            # dataset loaders, preprocessing, tokenizer training
│   └── utils/            # metrics (BLEU/METEOR/CIDEr), logging helpers
├── scripts/            # train.py, evaluate.py, download_data.sh
├── notebooks/          # exploratory analysis
├── configs/            # experiment config files (yaml)
├── experiments/        # run outputs, checkpoints (gitignored)
└── requirements.txt
```

## Baselines to reproduce/compare against

1. Solomon & Abebe (2023) — Inception-v3 + Bi-GRU + attention (word-level)
2. Our variant A — ViT/CLIP encoder + transformer decoder, word-level tokenizer
3. Our variant B — ViT/CLIP encoder + transformer decoder, standard Amharic
   BPE/WordPiece tokenizer
4. Our proposed model — ViT/CLIP encoder + transformer decoder,
   morphology-aware tokenizer

Comparing 2/3/4 isolates whether the *tokenizer* is what drives any
improvement, rather than just the encoder/decoder swap — this is the core
ablation for the novelty claim.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

See [`data/README.md`](data/README.md). Datasets are not committed to this
repo — place them locally under `data/`.

## Status

- [ ] Data loading + preprocessing pipeline
- [ ] Baseline reproduction (Bi-GRU) for a fair comparison point
- [ ] ViT/CLIP encoder + transformer decoder (word-level)
- [ ] Morphology-aware Amharic tokenizer
- [ ] Full ablation + evaluation (BLEU, METEOR, CIDEr)
