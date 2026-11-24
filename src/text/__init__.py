"""
CatalogUpdate AI - Text Generation Module (Character-Level LSTM).

Provides:
- Synthetic corpus generation with intentional realistic noise.
- Text cleaning, character tokenization, and sequence creation.
- 2-layer character-level LSTM model architecture.
- Training pipeline with loss curves and validation perplexity logging.
- Sampling and text generation with temperature control.
- Quantitative and qualitative evaluation (perplexity, format validity, vocabulary validity, relevance, diversity).
- Specification line parser connecting generated text to the catalog diffusion prompt builder.
"""

def __getattr__(name: str):
    if name == "CharLSTM":
        from src.text.model import CharLSTM
        return CharLSTM
    if name == "parse_spec_line":
        from src.text.spec_parser import parse_spec_line
        return parse_spec_line
    raise AttributeError(f"module 'src.text' has no attribute '{name}'")

__all__ = [
    "CharLSTM",
    "parse_spec_line",
]
