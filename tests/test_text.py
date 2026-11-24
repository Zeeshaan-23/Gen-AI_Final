"""
Unit tests for the character-level LSTM text generation module:
- clean_text edge cases (extra whitespace, casing, non-printable chars, punctuation, deduplication).
- Character vocabulary round-trip and persistence.
- make_sequences tensor extraction and shape checks.
- CharLSTM model forward pass output dimensions on CPU.
- 1-epoch training smoke test on CPU (< 10 seconds).
- generate function string output and length preservation.
- spec_parser template and fallback extraction.
"""

from pathlib import Path
import torch

from src.text.preprocess_text import (
    clean_text,
    tokenize,
    save_vocab,
    load_vocab,
    make_sequences,
    split_sequences,
)
from src.text.model import CharLSTM
from src.text.train import run_training
from src.text.generate import generate
from src.text.spec_parser import parse_spec_line
from src.prompts import build_prompt_from_text


def test_clean_text_edge_cases():
    """Verifies text cleaning: casing, whitespace, punctuation, duplicates, and non-printables."""
    raw = (
        "  UPDATE:   chair   legs   changed   to   DARK   WALNUT,,   matte   finish..!!  \n"
        "\n"
        "   \t  \n"
        "update: chair legs changed to dark walnut, matte finish.\n"  # Exact duplicate once normalized
        "Spec UPDATE: \u200b TABLE TOP modified to marble in white ; ; polished finish!\n"
    )

    cleaned = clean_text(raw)
    lines = cleaned.splitlines()

    # Empty lines and duplicates must be dropped -> exactly 2 unique clean lines
    assert len(lines) == 2

    # Line 1 checks
    assert lines[0] == "update: chair legs changed to dark walnut, matte finish."
    assert lines[0].islower()
    assert "  " not in lines[0]
    assert ",," not in lines[0]

    # Line 2 checks (non-printable zero-width space stripped)
    assert "\u200b" not in lines[1]
    assert lines[1] == "spec update: table top modified to marble in white; polished finish!"


def test_vocab_round_trip(tmp_path):
    """Verifies character vocabulary creation, JSON persistence, and reloading."""
    text = "update: chair legs changed to walnut wood, matte finish."
    char2idx, idx2char = tokenize(text)

    assert len(char2idx) == len(idx2char)
    # Every character must map back to itself
    for ch, idx in char2idx.items():
        assert idx2char[idx] == ch

    vocab_file = tmp_path / "vocab.json"
    save_vocab(char2idx, idx2char, vocab_file)
    assert vocab_file.exists()

    loaded_c2i, loaded_i2c = load_vocab(vocab_file)
    assert loaded_c2i == char2idx
    assert loaded_i2c == idx2char


def test_make_sequences_shapes():
    """Verifies sliding window sequence extraction produces expected tensor shapes."""
    text = "update: chair legs changed to dark walnut, matte finish.\n" * 5
    seq_len = 30
    stride = 2

    X, y = make_sequences(text, seq_len=seq_len, stride=stride)

    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.dtype == torch.long
    assert y.dtype == torch.long

    expected_samples = len(range(0, len(text) - seq_len, stride))
    assert X.shape == (expected_samples, seq_len)
    assert y.shape == (expected_samples,)

    # Test train/val split
    X_train, y_train, X_val, y_val = split_sequences(X, y, val_ratio=0.10, seed=42)
    assert len(X_train) + len(X_val) == len(X)
    assert len(y_train) + len(y_val) == len(y)


def test_char_lstm_model_forward_shape():
    """Verifies CharLSTM forward pass dimensions on CPU."""
    vocab_size = 45
    batch_size = 4
    seq_len = 32
    hidden_dim = 256
    num_layers = 2

    model = CharLSTM(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, (hn, cn) = model(x)

    # Output logits must match (batch_size, seq_len, vocab_size)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert hn.shape == (num_layers, batch_size, hidden_dim)
    assert cn.shape == (num_layers, batch_size, hidden_dim)


def test_training_smoke_test_cpu(tmp_path):
    """Executes a 1-epoch training smoke test on a tiny synthetic corpus on CPU (< 10 seconds)."""
    raw_corpus = (
        "Update: chair legs changed to dark walnut, matte finish.\n"
        "Update: sofa cushions updated to beige linen, textured finish.\n"
        "Change: replace armrests with black leather, satin finish.\n"
        "Spec update: table top modified to marble in white, polished finish.\n"
        "Update: cabinet handles changed to brushed brass, metallic finish.\n"
    ) * 4  # 20 lines

    data_file = tmp_path / "tiny_specs.txt"
    data_file.write_text(raw_corpus, encoding="utf-8")
    out_dir = tmp_path / "output_test"

    results = run_training(
        data_path=str(data_file),
        output_dir=str(out_dir),
        epochs=1,
        batch_size=16,
        lr=1e-3,
        seq_len=25,
        stride=4,
        hidden_dim=64,
        embedding_dim=32,
        num_layers=1,
        device_name="cpu",
    )

    assert results["epochs"] == 1
    assert "best_val_loss" in results
    assert Path(results["checkpoint"]).exists()
    assert Path(results["loss_curve"]).exists()
    assert (out_dir / "training_log.csv").exists()


def test_generate_string_output():
    """Verifies autoregressive character generation produces expected string lengths."""
    vocab = list("abcdefghijklmnopqrstuvwxyz :,-.")
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    model = CharLSTM(vocab_size=len(vocab), embedding_dim=32, hidden_dim=64, num_layers=1)

    seed = "update: chair legs"
    gen_len = 25
    result = generate(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        seed=seed,
        length=gen_len,
        temperature=0.8,
        device="cpu",
        stop_at_newline=False,
    )

    assert isinstance(result, str)
    assert result.startswith(seed)
    assert len(result) == len(seed) + gen_len


def test_spec_parser_examples():
    """Verifies specification line parsing into structured fields and prompt composition."""
    # Example 1: Standard template
    line1 = "Update: chair legs changed to dark walnut, matte finish."
    p1 = parse_spec_line(line1)
    assert p1["part"] == "chair legs"
    assert p1["color"] == "dark"
    assert p1["material"] == "walnut"
    assert p1["finish"] == "matte"

    # Example 2: Change/replace variant
    line2 = "Change: replace sofa cushions with beige linen, textured finish."
    p2 = parse_spec_line(line2)
    assert p2["part"] == "sofa cushions"
    assert p2["color"] == "beige"
    assert p2["material"] == "linen"
    assert p2["finish"] == "textured"

    # Example 3: Fallback extraction on informal phrasing
    line3 = "we want gold brushed brass on cabinet handles right now"
    p3 = parse_spec_line(line3)
    assert p3["part"] == "cabinet handles"
    assert p3["material"] == "brushed brass"
    assert p3["color"] == "gold"

    # Example 4: Integration with build_prompt_from_text
    pos, neg = build_prompt_from_text(line1)
    assert "chair legs" in pos
    assert "walnut" in pos
    assert "dark" in pos
    assert "matte finish" in pos
    assert "studio product photo" in pos
    assert len(neg) > 10
