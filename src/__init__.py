"""
CatalogUpdate AI - Package Initialization.
"""

from src.engine import CatalogEditor

def __getattr__(name: str):
    if name in ("build_prompt", "build_prompt_from_text", "FURNITURE_PRESETS", "DEFAULT_NEGATIVE_PROMPT"):
        import src.prompts as _prompts
        return getattr(_prompts, name)
    if name in ("run_batch", "zip_outputs", "MockCatalogEditor"):
        import src.batch as _batch
        return getattr(_batch, name)
    if name in ("make_control_image",):
        import src.engine as _engine
        return getattr(_engine, name)
    if name in ("compare_controlnet",):
        import src.compare as _compare
        return getattr(_compare, name)
    if name in ("load_and_clean", "normalize_background", "pad_to_square", "resize_for_model", "preprocess_dir"):
        import src.preprocess as _preprocess
        return getattr(_preprocess, name)
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = [
    "CatalogEditor",
    "make_control_image",
    "run_batch",
    "zip_outputs",
    "build_prompt",
    "build_prompt_from_text",
    "FURNITURE_PRESETS",
    "DEFAULT_NEGATIVE_PROMPT",
    "compare_controlnet",
    "load_and_clean",
    "normalize_background",
    "pad_to_square",
    "resize_for_model",
    "preprocess_dir",
]
