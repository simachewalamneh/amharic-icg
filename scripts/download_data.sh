#!/usr/bin/env bash
# Fetches the Amharic Flickr8k and BNATURE datasets into data/.
# Fill in the actual source (URL, gdrive id, etc.) once confirmed —
# these datasets came from the Solomon & Abebe (2023) baseline paper.
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR/flickr8k_amharic" "$DATA_DIR/bnature_amharic"

echo "TODO: add actual download commands here, e.g.:"
echo "  wget <flickr8k_amharic_url> -O $DATA_DIR/flickr8k_amharic.zip"
echo "  wget <bnature_amharic_url> -O $DATA_DIR/bnature_amharic.zip"
echo "For now, place the datasets you already downloaded manually into:"
echo "  $DATA_DIR/flickr8k_amharic/"
echo "  $DATA_DIR/bnature_amharic/"
