"""
CatalogUpdate AI - Text Preprocessing & Character Tokenization.

Preprocesses raw specification text:
- Case normalization (lowercase).
- Whitespace and punctuation normalization.
- Deduplication and empty line removal.
- Non-printable character removal.
- Character-level vocabulary extraction and sequence generation.
- Dataset statistics logging to outputs/text/dataset_stats.json.
"""

import re
import json
import string
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch

PRINTABLE_SET = set(string.printable)


def clean_text(raw_text: str) -> str:
    """
    Cleans raw specification text:
    - Lowercases all characters.
    - Strips non-printable characters.
    - Normalizes punctuation (collapses repeated periods, commas, colons, exclamation marks).
    - Normalizes whitespace (replaces tabs/multiple spaces with single space, strips leading/trailing).
    - Drops empty lines and removes duplicate lines while preserving order.

    Args:
        raw_text: Raw string containing multiple lines.

    Returns:
        Cleaned text string with newline-separated lines.
    """
    lines = raw_text.splitlines()
    seen = set()
    cleaned_lines: List[str] = []

    for line in lines:
        # 1. Lowercase
        s = line.lower()

        # 2. Filter non-printable characters
        s = "".join(ch for ch in s if ch in PRINTABLE_SET and ch not in "\r\t\x0b\x0c")

        # 3. Normalize punctuation repetitions
        s = re.sub(r"[.!?]{2,}", ".", s)
        s = re.sub(r"\.\!", ".", s)
        s = re.sub(r"\!\.", ".", s)
        s = re.sub(r",{2,}", ",", s)
        s = re.sub(r":{2,}", ":", s)
        s = re.sub(r";\s*;", ";", s)
        s = re.sub(r";{2,}", ";", s)
        s = re.sub(r"\s+([,.:;!?])", r"\1", s)  # Remove space before punctuation

        # 4. Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()

        # 5. Drop empty or duplicate lines
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            cleaned_lines.append(s)

    return "\n".join(cleaned_lines)


def tokenize(cleaned_text: Union[str, List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds a deterministic character-level vocabulary from cleaned text.

    Args:
        cleaned_text: Cleaned text string or list of lines.

    Returns:
        Tuple of (char2idx, idx2char).
    """
    if isinstance(cleaned_text, list):
        full_text = "\n".join(cleaned_text)
    else:
        full_text = cleaned_text

    unique_chars = sorted(list(set(full_text)))
    char2idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx2char = {idx: ch for idx, ch in enumerate(unique_chars)}

    return char2idx, idx2char


def save_vocab(char2idx: Dict[str, int], idx2char: Dict[int, str], output_path: Union[str, Path]) -> None:
    """Saves character vocabulary mappings to JSON."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "char2idx": char2idx,
        "idx2char": {str(k): v for k, v in idx2char.items()},
        "vocab_size": len(char2idx),
        "chars": [idx2char[i] for i in range(len(idx2char))],
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_vocab(vocab_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Loads character vocabulary from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    char2idx = data["char2idx"]
    idx2char = {int(k): v for k, v in data["idx2char"].items()}
    return char2idx, idx2char


def make_sequences(
    text: str,
    seq_len: int = 40,
    stride: int = 3,
    char2idx: Optional[Dict[str, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts fixed-length overlapping character sequences and next-character targets.

    Args:
        text: Input text corpus.
        seq_len: Length of input sequence window (default: 40).
        stride: Step size between sliding windows (default: 3).
        char2idx: Character-to-index mapping dictionary.

    Returns:
        Tuple of (X, y) as PyTorch LongTensors:
        - X: shape (num_samples, seq_len)
        - y: shape (num_samples,)
    """
    if char2idx is None:
        char2idx, _ = tokenize(text)

    x_seqs: List[List[int]] = []
    y_chars: List[int] = []

    for i in range(0, len(text) - seq_len, stride):
        seq = text[i : i + seq_len]
        target = text[i + seq_len]
        x_seqs.append([char2idx[ch] for ch in seq])
        y_chars.append(char2idx[target])

    X = torch.tensor(x_seqs, dtype=torch.long)
    y = torch.tensor(y_chars, dtype=torch.long)
    return X, y


def split_sequences(
    X: torch.Tensor,
    y: torch.Tensor,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits sequence tensors into train and validation subsets."""
    torch.manual_seed(seed)
    num_samples = len(X)
    val_size = int(num_samples * val_ratio)

    indices = torch.randperm(num_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    return X_train, y_train, X_val, y_val


def preprocess_corpus(
    input_path: Union[str, Path] = "data/text/furniture_specs_raw.txt",
    output_dir: Union[str, Path] = "outputs/text",
    seq_len: int = 40,
    stride: int = 3,
) -> Dict[str, Any]:
    """
    Executes end-to-end preprocessing on raw corpus:
    - Reads raw text.
    - Cleans and deduplicates text.
    - Builds character vocabulary and saves outputs/text/vocab.json.
    - Generates sequence tensors and computes dataset stats.
    - Saves outputs/text/dataset_stats.json.

    Returns:
        Dataset statistics dictionary.
    """
    in_file = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as f:
        raw_content = f.read()

    raw_lines = raw_content.splitlines()
    cleaned_content = clean_text(raw_content)
    clean_lines = cleaned_content.splitlines()

    char2idx, idx2char = tokenize(cleaned_content)
    save_vocab(char2idx, idx2char, out_dir / "vocab.json")

    # Save cleaned corpus for transparency and inspection
    with open(out_dir / "furniture_specs_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(cleaned_content + "\n")

    X, y = make_sequences(cleaned_content, seq_len=seq_len, stride=stride, char2idx=char2idx)
    X_train, _, X_val, _ = split_sequences(X, y, val_ratio=0.10)

    stats = {
        "num_raw_lines": len(raw_lines),
        "num_clean_lines": len(clean_lines),
        "vocab_size": len(char2idx),
        "total_characters": len(cleaned_content),
        "num_sequences": len(X),
        "train_sequences": len(X_train),
        "val_sequences": len(X_val),
        "seq_len": seq_len,
        "stride": stride,
    }

    stats_file = out_dir / "dataset_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess specification text corpus")
    parser.add_argument("--input", default="data/text/furniture_specs_raw.txt", help="Input raw text corpus file")
    parser.add_argument("--output", default="outputs/text", help="Output directory for vocab and stats")
    parser.add_argument("--seq-len", type=int, default=40, help="Sliding sequence length (default: 40)")
    parser.add_argument("--stride", type=int, default=3, help="Sliding window stride (default: 3)")

    args = parser.parse_args()
    stats = preprocess_corpus(input_path=args.input, output_dir=args.output, seq_len=args.seq_len, stride=args.stride)

    print("\nDataset Preprocessing Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nSaved vocab.json and dataset_stats.json to {args.output}")


if __name__ == "__main__":
    main()
