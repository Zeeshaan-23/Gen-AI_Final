"""
CatalogUpdate AI - Package Initialization.
"""

from src.engine import CatalogEditor

def __getattr__(name: str):
    if name in ("build_prompt", "FURNITURE_PRESETS", "DEFAULT_NEGATIVE_PROMPT"):
        import src.prompts as _prompts
        return getattr(_prompts, name)
    if name in ("run_batch", "zip_outputs", "MockCatalogEditor"):
        import src.batch as _batch
        return getattr(_batch, name)
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = [
    "CatalogEditor",
    "run_batch",
    "zip_outputs",
    "build_prompt",
    "FURNITURE_PRESETS",
    "DEFAULT_NEGATIVE_PROMPT",
]
