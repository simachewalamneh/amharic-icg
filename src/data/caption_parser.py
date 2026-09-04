"""
Parses the Amharic/English caption txt files into a normalized structure:
    { image_filename: [caption1, caption2, ...], ... }
"""
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png)", re.IGNORECASE)

# A clean image filename: no embedded tabs, commas, or newlines — this is
# what catches lines where the delimiter-guessing logic in _split_line
# picked the wrong separator and merged the filename with caption text.
CLEAN_IMG_ID_RE = re.compile(r"^[\w\-.]+\.(jpg|jpeg|png)$", re.IGNORECASE)


def _split_line(line: str):
    line = line.rstrip("\n").rstrip("\r")
    if not line.strip():
        return None
    if "\t" in line:
        parts = line.split("\t", 1)
    elif "," in line:
        parts = line.split(",", 1)
    else:
        parts = line.split(None, 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_caption_file(path: Path) -> Dict[str, List[str]]:
    captions = defaultdict(list)
    bad_lines = 0
    total_lines = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total_lines += 1
            split = _split_line(line)
            if split is None:
                bad_lines += 1
                continue
            left, caption = split

            img_id = re.sub(r"#\d+$", "", left).strip()

            # STRICT check: reject anything that isn't cleanly a filename.
            if not CLEAN_IMG_ID_RE.match(img_id):
                bad_lines += 1
                continue

            captions[img_id].append(caption)

    print(f"File: {path}")
    print(f"  Total lines read: {total_lines}")
    print(f"  Unparseable lines: {bad_lines}")
    print(f"  Unique images with captions: {len(captions)}")
    if captions:
        sample_key = next(iter(captions))
        print(f"  Sample image_id: {sample_key!r}")
        print(f"  Sample captions for it ({len(captions[sample_key])}):")
        for c in captions[sample_key][:5]:
            print(f"    - {c}")
        cap_counts = [len(v) for v in captions.values()]
        print(f"  Captions-per-image: min={min(cap_counts)}, max={max(cap_counts)}, "
              f"avg={sum(cap_counts)/len(cap_counts):.2f}")
    return dict(captions)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python caption_parser.py <path_to_captions.txt>")
        sys.exit(1)
    parse_caption_file(Path(sys.argv[1]))
