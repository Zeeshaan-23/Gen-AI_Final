"""
CatalogUpdate AI - Character-Level LSTM Text Generation.

Samples specification update text from a trained CharLSTM model with:
- Configurable seed text prompting.
- Softmax temperature scaling (0.2 - 1.5).
- Optional top-k logit filtering.
- Early stop on newline (--one-line).
- Standalone CLI execution.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from src.text.model import CharLSTM


def generate(
    model: CharLSTM,
    char2idx: Dict[str, int],
    idx2char: Dict[int, str],
    seed: str,
    length: int = 120,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    stop_at_newline: bool = False,
) -> str:
    """
    Autoregressively generates text from seed using the CharLSTM model.

    Args:
        model: Trained CharLSTM model instance.
        char2idx: Character to index dictionary.
        idx2char: Index to character dictionary.
        seed: Initial prefix prompt string.
        length: Maximum number of characters to generate beyond seed.
        temperature: Softmax sampling temperature (>0; lower = more conservative, higher = more creative).
        top_k: Optional top-k logit truncation.
        device: Torch execution device.
        stop_at_newline: If True, terminates generation immediately upon generating a newline character.

    Returns:
        Generated string (seed + newly generated characters).
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    # Clean seed into valid characters
    clean_seed = seed.lower()
    valid_chars = [ch for ch in clean_seed if ch in char2idx]
    if not valid_chars:
        # Fallback if seed has no matching chars
        valid_chars = [list(char2idx.keys())[0]]

    # Initialize hidden states
    hidden = None
    generated = list(seed)

    # 1. Warm up LSTM hidden states with seed sequence
    seed_tensor = torch.tensor([[char2idx[ch] for ch in valid_chars]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, hidden = model(seed_tensor, hidden)
        last_logit = logits[:, -1, :]

    # 2. Autoregressive sampling loop
    for _ in range(length):
        scaled_logits = last_logit / max(temperature, 1e-4)

        if top_k is not None and top_k > 0:
            k = min(top_k, scaled_logits.size(-1))
            val, _ = torch.topk(scaled_logits, k)
            min_val = val[:, -1].unsqueeze(-1)
            scaled_logits = torch.where(scaled_logits < min_val, torch.tensor(-float("Inf"), device=device), scaled_logits)

        probs = F.softmax(scaled_logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx2char.get(next_idx, "")

        if stop_at_newline and next_char == "\n":
            break

        generated.append(next_char)

        # Feed the sampled character back for next step
        inp = torch.tensor([[next_idx]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, hidden = model(inp, hidden)
            last_logit = logits[:, -1, :]

    return "".join(generated)


def generate_from_checkpoint(
    checkpoint_path: Union[str, Path] = "outputs/text/charlstm.pt",
    seed: str = "Update: chair legs changed to",
    length: int = 120,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    stop_at_newline: bool = False,
) -> str:
    """Loads CharLSTM from checkpoint file and generates text."""
    ckpt_file = Path(checkpoint_path).resolve()
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}. Please train the model first.")

    ckpt = torch.load(ckpt_file, map_location=device)
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

    return generate(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        seed=seed,
        length=length,
        temperature=temperature,
        top_k=top_k,
        device=device,
        stop_at_newline=stop_at_newline,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate furniture specification text with CharLSTM")
    parser.add_argument("--checkpoint", default="outputs/text/charlstm.pt", help="Path to charlstm.pt checkpoint")
    parser.add_argument("--seed", default="Update: chair legs changed to", help="Initial text seed prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--length", type=int, default=120, help="Maximum generated characters (default: 120)")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k logit truncation")
    parser.add_argument("--one-line", action="store_true", help="Stop generation at the first newline character")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")

    args = parser.parse_args()

    result = generate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        stop_at_newline=args.one_line,
    )
    print("\n--- Generated Furniture Specification ---")
    print(result)
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
