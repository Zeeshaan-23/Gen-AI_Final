"""
CatalogUpdate AI - Raw Furniture Specification Corpus Generator.

Generates a realistic, deliberately messy corpus of furniture specification changes
(400-600 lines) from structured vocabulary lists and phrasing templates.
Introduces realistic noise (random casing, extraneous whitespace, stray punctuation,
duplicate lines, empty lines, and non-printable characters) so the preprocessing stage
has meaningful work to perform.
"""

import random
import argparse
from pathlib import Path
from typing import List, Optional

PARTS = [
    "chair legs",
    "armrests",
    "table top",
    "cabinet handles",
    "sofa cushions",
    "backrest",
    "dining chair seat",
    "coffee table surface",
    "drawer pulls",
    "couch fabric",
]

MATERIALS = [
    "walnut",
    "oak",
    "brushed brass",
    "black leather",
    "linen",
    "marble",
    "velvet",
    "boucle",
    "matte steel",
    "rattan",
    "dark wood",
    "cherry wood",
]

COLORS = [
    "dark",
    "charcoal",
    "white",
    "beige",
    "emerald",
    "matte black",
    "warm brown",
    "sand",
    "navy blue",
    "gold",
]

FINISHES = [
    "matte",
    "glossy",
    "satin",
    "brushed",
    "textured",
    "polished",
    "oiled",
]

TEMPLATES = [
    "Update: {part} changed to {color} {material}, {finish} finish.",
    "Update: change {part} to {color} {material} with {finish} finish.",
    "Change: replace {part} with {color} {material}, {finish} finish.",
    "Spec update: {part} updated to {color} {material}, {finish} finish.",
    "Update: {part} modified to {material} in {color}, {finish} finish.",
    "Update: {part} changed to {color} {material}.",
]


def add_realistic_noise(line: str, rng: random.Random) -> str:
    """Injects deliberate realistic noise into a generated specification line."""
    p = rng.random()

    # 1. Random casing (all upper, title, or alternating case)
    if p < 0.15:
        line = line.upper()
    elif p < 0.28:
        # Mess up casing on individual characters
        line = "".join(c.upper() if rng.random() < 0.3 else c.lower() for c in line)

    # 2. Extra spaces between words or leading/trailing
    if rng.random() < 0.30:
        words = line.split(" ")
        line = "   ".join(words)
    if rng.random() < 0.25:
        line = "  " + line + "   "

    # 3. Stray punctuation
    if rng.random() < 0.20:
        stray = rng.choice(["..", "...", "!!", ",,", " #", " ;", " :", " - "])
        pos = rng.randint(0, len(line))
        line = line[:pos] + stray + line[pos:]

    # 4. Occasional non-printable character (e.g. zero-width space)
    if rng.random() < 0.08:
        pos = rng.randint(0, len(line))
        line = line[:pos] + "\u200b" + line[pos:]

    return line


def generate_corpus(
    output_path: str = "data/text/furniture_specs_raw.txt",
    num_lines: int = 500,
    seed: int = 42,
    append_text: Optional[str] = None,
) -> List[str]:
    """
    Generates a deterministic raw corpus of furniture spec changes.

    Args:
        output_path: File path to save the generated raw corpus.
        num_lines: Target number of lines (default: 500).
        seed: Random seed for exact reproducibility.
        append_text: Optional custom text to append to the end of the file.

    Returns:
        List of generated raw lines.
    """
    rng = random.Random(seed)
    lines: List[str] = []

    # Generate core clean templates
    base_pool: List[str] = []
    for _ in range(num_lines - 40):
        part = rng.choice(PARTS)
        material = rng.choice(MATERIALS)
        color = rng.choice(COLORS)
        finish = rng.choice(FINISHES)
        template = rng.choice(TEMPLATES)

        clean_line = template.format(part=part, material=material, color=color, finish=finish)
        base_pool.append(clean_line)

    # Introduce duplicates (10-15 duplicate entries)
    num_duplicates = rng.randint(12, 18)
    for _ in range(num_duplicates):
        dup = rng.choice(base_pool)
        base_pool.append(dup)

    rng.shuffle(base_pool)

    # Apply realistic noise to a significant portion of lines (~40%)
    for line in base_pool:
        if rng.random() < 0.45:
            noisy_line = add_realistic_noise(line, rng)
            lines.append(noisy_line)
        else:
            lines.append(line)

    # Insert a few empty/blank lines (10-15 lines)
    num_blanks = rng.randint(10, 15)
    for _ in range(num_blanks):
        idx = rng.randint(0, len(lines))
        blank = rng.choice(["", "   ", " \t \n", "  "])
        lines.insert(idx, blank)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Append custom user text if provided
    if append_text:
        lines.append(append_text.strip())

    with open(out_file, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    return lines


def main():
    parser = argparse.ArgumentParser(description="Generate raw furniture specification corpus")
    parser.add_argument("--output", default="data/text/furniture_specs_raw.txt", help="Path to output raw text file")
    parser.add_argument("--lines", type=int, default=500, help="Number of lines to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--append", default=None, help="Optional user text line to append to the corpus")

    args = parser.parse_args()
    lines = generate_corpus(output_path=args.output, num_lines=args.lines, seed=args.seed, append_text=args.append)
    print(f"Generated raw corpus with {len(lines)} lines at {args.output}")


if __name__ == "__main__":
    main()
