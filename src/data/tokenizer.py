"""
Word-level Amharic tokenizer + vocabulary (baseline-comparable).
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]

AMHARIC_PUNCT = "።፣፤፥፦፧፨"
PUNCT_RE = re.compile(rf"([{re.escape(AMHARIC_PUNCT)}!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{{|}}~])")


def basic_tokenize(text: str) -> List[str]:
    text = text.strip()
    text = PUNCT_RE.sub(r" \1 ", text)
    tokens = text.split()
    return tokens


class Vocabulary:
    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    def encode(self, tokens: List[str]) -> List[int]:
        unk_id = self.token_to_id[UNK]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, UNK) for i in ids]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return cls(token_to_id)

    @classmethod
    def build(cls, captions: List[str], min_freq: int = 2) -> "Vocabulary":
        counter = Counter()
        for cap in captions:
            counter.update(basic_tokenize(cap))

        token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for token, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
            if freq >= min_freq:
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id)


class WordTokenizer:
    def __init__(self, vocab: Vocabulary, max_len: int = 30):
        self.vocab = vocab
        self.max_len = max_len

    def __call__(self, text: str) -> List[int]:
        tokens = basic_tokenize(text)
        ids = [self.vocab.token_to_id[SOS]] + self.vocab.encode(tokens) + [self.vocab.token_to_id[EOS]]
        ids = ids[: self.max_len]
        return ids


if __name__ == "__main__":
    import yaml
    from caption_parser import parse_caption_file

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        cfg = yaml.safe_load(f)["flickr8k"]

    with open(REPO_ROOT / cfg["splits_file"]) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])

    all_captions = parse_caption_file(REPO_ROOT / cfg["captions_file"])
    train_captions = [c for img_id, caps in all_captions.items() if img_id in train_ids for c in caps]

    vocab = Vocabulary.build(train_captions, min_freq=2)
    print(f"Vocab size (min_freq=2): {len(vocab)}")

    out_path = REPO_ROOT / "data" / "vocab.json"
    vocab.save(out_path)
    print(f"Saved vocab to {out_path}")

    tok = WordTokenizer(vocab)
    sample = train_captions[0]
    ids = tok(sample)
    print(f"Sample caption: {sample}")
    print(f"Tokenized ids: {ids}")
    print(f"Decoded back: {' '.join(vocab.decode(ids))}")
