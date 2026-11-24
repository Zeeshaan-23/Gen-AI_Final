"""
CatalogUpdate AI - Character-Level LSTM Training Pipeline.

Trains the 2-layer CharLSTM on preprocessed furniture specifications:
- Auto-detects CUDA GPU (Colab T4 target) or CPU fallback.
- Computes CrossEntropyLoss and perplexity per epoch.
- Applies gradient norm clipping (1.0).
- Tracks validation loss and checkpoints best weights to outputs/text/charlstm.pt.
- Generates loss curve plot outputs/text/loss_curve.png.
- Records metrics in outputs/text/training_log.csv.
"""

import math
import time
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from src.text.preprocess_text import (
    clean_text,
    tokenize,
    save_vocab,
    make_sequences,
    split_sequences,
)
from src.text.model import CharLSTM


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """Runs a single training epoch and returns mean loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(X_batch)
        # Evaluate prediction on the final sequence token
        last_logits = logits[:, -1, :]
        loss = criterion(last_logits, y_batch)

        loss.backward()
        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluates validation loss."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, _ = model(X_batch)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, y_batch)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(1, total_batches)


def run_training(
    data_path: str = "data/text/furniture_specs_raw.txt",
    output_dir: str = "outputs/text",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-3,
    seq_len: int = 40,
    stride: int = 3,
    hidden_dim: int = 256,
    embedding_dim: int = 64,
    num_layers: int = 2,
    clip_grad: float = 1.0,
    device_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes the full CharLSTM training routine.
    """
    torch.manual_seed(seed)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using training device: {device}")

    # 1. Load and clean text corpus
    data_file = Path(data_path).resolve()
    if not data_file.exists():
        from src.text.make_corpus import generate_corpus
        print(f"Corpus {data_file} not found. Generating default raw corpus...")
        generate_corpus(output_path=str(data_file), num_lines=500, seed=seed)

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    char2idx, idx2char = tokenize(cleaned)
    vocab_size = len(char2idx)
    save_vocab(char2idx, idx2char, out_dir / "vocab.json")

    # 2. Extract sequence tensors and train/val split
    X, y = make_sequences(cleaned, seq_len=seq_len, stride=stride, char2idx=char2idx)
    X_train, y_train, X_val, y_val = split_sequences(X, y, val_ratio=0.10, seed=seed)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Vocab size: {vocab_size} chars | Sequences: {len(X)} (train: {len(X_train)}, val: {len(X_val)})")

    # 3. Model, Criterion, Optimizer
    model = CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    log_records = []

    best_val_loss = float("inf")
    best_checkpoint_path = out_dir / "charlstm.pt"

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=clip_grad)
        val_loss = evaluate(model, val_loader, criterion, device)
        epoch_time = round(time.time() - t_epoch0, 2)

        train_ppl = round(math.exp(min(train_loss, 20.0)), 2)
        val_ppl = round(math.exp(min(val_loss, 20.0)), 2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_ppl": train_ppl,
            "val_loss": round(val_loss, 4),
            "val_ppl": val_ppl,
            "time_s": epoch_time,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "char2idx": char2idx,
                "idx2char": idx2char,
                "config": {
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "seq_len": seq_len,
                },
                "epoch": epoch,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
            }, best_checkpoint_path)

        if epoch % max(1, epochs // 10) == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} (PPL: {train_ppl}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl}) | {epoch_time}s")

    total_time = round(time.time() - t_start, 2)
    print(f"Training completed in {total_time}s. Best val loss: {best_val_loss:.4f}. Saved checkpoint to {best_checkpoint_path}")

    # 4. Save training log CSV
    csv_file = out_dir / "training_log.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "time_s"])
        writer.writeheader()
        writer.writerows(log_records)

    # 5. Save Loss Curve PNG
    curve_file = out_dir / "loss_curve.png"
    if plt is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
        plt.plot(range(1, epochs + 1), val_losses, label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Cross-Entropy Loss", fontsize=11)
        plt.title("CharLSTM Furniture Spec Training Loss Curve", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(curve_file, dpi=150)
        plt.close()
    else:
        # Graceful Pillow-based line plot fallback when matplotlib is not installed
        from PIL import Image, ImageDraw
        img_w, img_h = 600, 400
        chart_img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(chart_img)
        # Bounding box for plot
        x0, y0, x1, y1 = 60, 40, img_w - 40, img_h - 60
        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180), width=1)

        max_loss = max(max(train_losses), max(val_losses), 0.1)
        min_loss = min(min(train_losses), min(val_losses), 0.0)
        loss_range = max(max_loss - min_loss, 1e-4)

        def loss_to_xy(ep_idx, loss_val):
            px = x0 + int((ep_idx / max(1, epochs - 1)) * (x1 - x0)) if epochs > 1 else (x0 + x1) // 2
            py = y1 - int(((loss_val - min_loss) / loss_range) * (y1 - y0))
            return px, py

        # Plot train curve (blue)
        train_pts = [loss_to_xy(i, l) for i, l in enumerate(train_losses)]
        for i in range(len(train_pts) - 1):
            draw.line([train_pts[i], train_pts[i + 1]], fill=(31, 119, 180), width=3)

        # Plot val curve (orange)
        val_pts = [loss_to_xy(i, l) for i, l in enumerate(val_losses)]
        for i in range(len(val_pts) - 1):
            draw.line([val_pts[i], val_pts[i + 1]], fill=(255, 127, 14), width=3)

        chart_img.save(curve_file)

    return {
        "best_val_loss": best_val_loss,
        "best_val_ppl": round(math.exp(min(best_val_loss, 20.0)), 2),
        "epochs": epochs,
        "checkpoint": str(best_checkpoint_path),
        "loss_curve": str(curve_file),
        "total_time_s": total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CharLSTM for furniture specification generation")
    parser.add_argument("--data", default="data/text/furniture_specs_raw.txt", help="Path to raw text corpus")
    parser.add_argument("--output-dir", default="outputs/text", help="Directory for checkpoint, logs, and plots")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate (default: 2e-3)")
    parser.add_argument("--seq-len", type=int, default=40, help="Sequence length (default: 40)")
    parser.add_argument("--stride", type=int, default=3, help="Stride (default: 3)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden dimension (default: 256)")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Character embedding dimension (default: 64)")
    parser.add_argument("--num-layers", type=int, default=2, help="LSTM layer depth (default: 2)")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu; default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    run_training(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seq_len=args.seq_len,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        clip_grad=args.clip,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
