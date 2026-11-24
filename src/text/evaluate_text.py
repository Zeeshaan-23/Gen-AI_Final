"""
CatalogUpdate AI - Text Generation Evaluation Suite.

Evaluates generated specification text across coherence, relevance, and diversity:
- Generates 150 samples (5 seeds x 3 temperatures [0.5, 0.8, 1.1] x 10 samples each).
- Evaluates:
  * Validation perplexity.
  * Format validity rate (regex match against specification templates).
  * Vocabulary validity rate (ratio of generated words in training vocabulary).
  * Relevance score (presence of known furniture part + material/color).
  * Diversity metrics (distinct-1, distinct-2 ratios, and duplicate rate vs training).
  * Qualitative 3 best and 3 worst samples.
- Saves outputs/text/text_eval.csv and outputs/text/text_eval_summary.md.
"""

import re
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

import torch

from src.text.model import CharLSTM
from src.text.generate import generate
from src.text.make_corpus import PARTS, MATERIALS, COLORS, FINISHES

EVAL_SEEDS = [
    "Update: chair legs changed to",
    "Update: sofa cushions updated to",
    "Update: table top modified to",
    "Change: replace armrests with",
    "Spec update: cabinet handles to",
]

EVAL_TEMPERATURES = [0.5, 0.8, 1.1]

FORMAT_REGEX = re.compile(
    r"^(update|change|spec update):\s*[\w\s]+\s*(changed to|updated to|modified to|replace .* with|to)\s*[\w\s,]+",
    re.IGNORECASE,
)


def extract_words(text: str) -> List[str]:
    """Tokenizes alphanumeric words in text."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def build_training_vocab(clean_corpus_path: Path) -> Tuple[Set[str], Set[str]]:
    """Builds word set and line set from the cleaned training corpus."""
    if not clean_corpus_path.exists():
        # Fallback to predefined lists
        words = set()
        for group in (PARTS, MATERIALS, COLORS, FINISHES):
            for item in group:
                words.update(item.split())
        words.update(["update", "change", "spec", "changed", "updated", "modified", "replace", "with", "to", "in", "finish"])
        return words, set()

    with open(clean_corpus_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = set(line.strip().lower() for line in content.splitlines() if line.strip())
    words = set(extract_words(content))
    return words, lines


def compute_diversity(texts: List[str]) -> Tuple[float, float]:
    """Computes distinct-1 and distinct-2 n-gram diversity ratios."""
    all_unigrams: List[str] = []
    all_bigrams: List[Tuple[str, str]] = []

    for t in texts:
        words = extract_words(t)
        all_unigrams.extend(words)
        for i in range(len(words) - 1):
            all_bigrams.append((words[i], words[i + 1]))

    distinct_1 = len(set(all_unigrams)) / max(1, len(all_unigrams))
    distinct_2 = len(set(all_bigrams)) / max(1, len(all_bigrams))
    return round(distinct_1, 4), round(distinct_2, 4)


def evaluate_sample(
    sample_text: str,
    train_words: Set[str],
    train_lines: Set[str],
) -> Dict[str, Any]:
    """Evaluates a single generated text sample."""
    line = sample_text.splitlines()[0].strip() if "\n" in sample_text else sample_text.strip()
    words = extract_words(line)

    # Format validity (regex match)
    is_valid_format = bool(FORMAT_REGEX.search(line))

    # Vocabulary validity (fraction of generated words in training word list)
    in_vocab_count = sum(1 for w in words if w in train_words)
    vocab_validity = (in_vocab_count / max(1, len(words)))

    # Relevance: contains at least one known part AND at least one known material/color
    line_lower = line.lower()
    has_part = any(part.lower() in line_lower for part in PARTS)
    has_material_or_color = any(m.lower() in line_lower for m in MATERIALS) or any(c.lower() in line_lower for c in COLORS)
    is_relevant = bool(has_part and has_material_or_color)

    # Duplicate vs training
    is_duplicate = line_lower in train_lines

    # Overall score for best/worst ranking
    score = (
        (1.5 if is_valid_format else 0.0)
        + (2.0 if is_relevant else 0.0)
        + (vocab_validity * 1.5)
        - (1.0 if is_duplicate else 0.0)
        + (0.2 if len(words) >= 5 else -1.0)
    )

    return {
        "text": line,
        "is_valid_format": is_valid_format,
        "vocab_validity": round(vocab_validity, 4),
        "is_relevant": is_relevant,
        "is_duplicate": is_duplicate,
        "score": round(score, 3),
    }


def run_evaluation(
    checkpoint_path: str = "outputs/text/charlstm.pt",
    output_dir: str = "outputs/text",
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """
    Executes full evaluation:
    5 seeds x 3 temperatures x 10 samples = 150 samples.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = Path(checkpoint_path).resolve()

    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}. Train model first.")

    device = torch.device(device_name)
    ckpt = torch.load(ckpt_file, map_location=device)
    val_ppl = ckpt.get("val_ppl", 0.0)

    char2idx = ckpt["char2idx"]
    idx2char = {int(k): v for k, v in ckpt["idx2char"].items()}
    cfg = ckpt.get("config", {})

    model = CharLSTM(
        vocab_size=len(char2idx),
        embedding_dim=cfg.get("embedding_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_layers=cfg.get("num_layers", 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    clean_corpus_file = out_dir / "furniture_specs_cleaned.txt"
    train_words, train_lines = build_training_vocab(clean_corpus_file)

    eval_records: List[Dict[str, Any]] = []
    all_evaluated: List[Dict[str, Any]] = []

    print("\nRunning Text Generation Evaluation (150 samples across 5 seeds & 3 temperatures)...")

    for temp in EVAL_TEMPERATURES:
        temp_samples: List[str] = []

        for seed in EVAL_SEEDS:
            for sample_idx in range(10):
                generated_str = generate(
                    model=model,
                    char2idx=char2idx,
                    idx2char=idx2char,
                    seed=seed,
                    length=90,
                    temperature=temp,
                    device=device,
                    stop_at_newline=True,
                )
                temp_samples.append(generated_str)

                metrics = evaluate_sample(generated_str, train_words, train_lines)
                record = {
                    "seed": seed,
                    "temperature": temp,
                    "sample_idx": sample_idx + 1,
                    "generated_text": metrics["text"],
                    "format_valid": int(metrics["is_valid_format"]),
                    "vocab_validity": metrics["vocab_validity"],
                    "relevant": int(metrics["is_relevant"]),
                    "duplicate_of_train": int(metrics["is_duplicate"]),
                    "score": metrics["score"],
                }
                eval_records.append(record)
                all_evaluated.append({**metrics, "seed": seed, "temperature": temp})

    # Diversity metrics across all generated samples
    all_texts = [r["generated_text"] for r in eval_records]
    dist1, dist2 = compute_diversity(all_texts)

    # Aggregate rates
    format_rate = round(sum(r["format_valid"] for r in eval_records) / len(eval_records), 4)
    vocab_rate = round(sum(r["vocab_validity"] for r in eval_records) / len(eval_records), 4)
    relevance_rate = round(sum(r["relevant"] for r in eval_records) / len(eval_records), 4)
    dup_rate = round(sum(r["duplicate_of_train"] for r in eval_records) / len(eval_records), 4)

    # 1. Write outputs/text/text_eval.csv
    csv_file = out_dir / "text_eval.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "seed", "temperature", "sample_idx", "generated_text",
            "format_valid", "vocab_validity", "relevant", "duplicate_of_train", "score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(eval_records)

    # Rank 3 best and 3 worst samples
    sorted_samples = sorted(all_evaluated, key=lambda x: x["score"], reverse=True)
    best_3 = sorted_samples[:3]
    worst_3 = sorted_samples[-3:]

    # 2. Write outputs/text/text_eval_summary.md
    summary_md = f"""# CharLSTM Text Generation Evaluation Summary

## 1. Quantitative Benchmark Metrics
- **Validation Perplexity (PPL)**: {val_ppl}
- **Total Samples Evaluated**: {len(eval_records)} (5 seeds x 3 temperatures x 10 runs)
- **Format Validity Rate**: {format_rate * 100:.1f}% (matches specification update syntax)
- **Vocabulary Validity Rate**: {vocab_rate * 100:.1f}% (words present in training corpus)
- **Catalog Relevance Rate**: {relevance_rate * 100:.1f}% (contains target part + material/color)
- **Lexical Diversity**:
  - Distinct-1 (Unigrams): {dist1}
  - Distinct-2 (Bigrams): {dist2}
- **Training Duplicate Rate**: {dup_rate * 100:.1f}% (novel variations vs. exact memorization)

---

## 2. Temperature Comparison
| Temperature | Format Valid % | Vocab Valid % | Relevance % |
| :---: | :---: | :---: | :---: |
| **0.5 (Conservative)** | {sum(r['format_valid'] for r in eval_records if r['temperature'] == 0.5) / 50 * 100:.1f}% | {sum(r['vocab_validity'] for r in eval_records if r['temperature'] == 0.5) / 50 * 100:.1f}% | {sum(r['relevant'] for r in eval_records if r['temperature'] == 0.5) / 50 * 100:.1f}% |
| **0.8 (Optimal)** | {sum(r['format_valid'] for r in eval_records if r['temperature'] == 0.8) / 50 * 100:.1f}% | {sum(r['vocab_validity'] for r in eval_records if r['temperature'] == 0.8) / 50 * 100:.1f}% | {sum(r['relevant'] for r in eval_records if r['temperature'] == 0.8) / 50 * 100:.1f}% |
| **1.1 (Exploratory)** | {sum(r['format_valid'] for r in eval_records if r['temperature'] == 1.1) / 50 * 100:.1f}% | {sum(r['vocab_validity'] for r in eval_records if r['temperature'] == 1.1) / 50 * 100:.1f}% | {sum(r['relevant'] for r in eval_records if r['temperature'] == 1.1) / 50 * 100:.1f}% |

---

## 3. Qualitative Samples: Best vs. Worst

### Best 3 Samples (High Syntax Fidelity, Coherence & Material Plausibility)
| # | Temp | Generated Specification | Format | Vocab |
| :---: | :---: | :--- | :---: | :---: |
| 1 | {best_3[0]['temperature']} | `{best_3[0]['text']}` | {best_3[0]['is_valid_format']} | {best_3[0]['vocab_validity']} |
| 2 | {best_3[1]['temperature']} | `{best_3[1]['text']}` | {best_3[1]['is_valid_format']} | {best_3[1]['vocab_validity']} |
| 3 | {best_3[2]['temperature']} | `{best_3[2]['text']}` | {best_3[2]['is_valid_format']} | {best_3[2]['vocab_validity']} |

### Worst 3 Samples (High Temperature Drift, Truncation, or Non-Standard Syntax)
| # | Temp | Generated Specification | Format | Vocab |
| :---: | :---: | :--- | :---: | :---: |
| 1 | {worst_3[0]['temperature']} | `{worst_3[0]['text']}` | {worst_3[0]['is_valid_format']} | {worst_3[0]['vocab_validity']} |
| 2 | {worst_3[1]['temperature']} | `{worst_3[1]['text']}` | {worst_3[1]['is_valid_format']} | {worst_3[1]['vocab_validity']} |
| 3 | {worst_3[2]['temperature']} | `{worst_3[2]['text']}` | {worst_3[2]['is_valid_format']} | {worst_3[2]['vocab_validity']} |
"""

    summary_file = out_dir / "text_eval_summary.md"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"Evaluation complete. Saved CSV to {csv_file} and Summary to {summary_file}")
    return {
        "validation_ppl": val_ppl,
        "format_validity_rate": format_rate,
        "vocabulary_validity_rate": vocab_rate,
        "relevance_rate": relevance_rate,
        "distinct_1": dist1,
        "distinct_2": dist2,
        "duplicate_rate": dup_rate,
        "csv_path": str(csv_file),
        "summary_path": str(summary_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CharLSTM text generation quality")
    parser.add_argument("--checkpoint", default="outputs/text/charlstm.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", default="outputs/text", help="Directory for evaluation CSV and summary")
    parser.add_argument("--device", default="cpu", help="Device for evaluation (cpu or cuda)")

    args = parser.parse_args()
    run_evaluation(checkpoint_path=args.checkpoint, output_dir=args.output_dir, device_name=args.device)


if __name__ == "__main__":
    main()
