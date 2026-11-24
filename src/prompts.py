import sys
import importlib.util
from pathlib import Path

_target_path = Path(__file__).resolve().parent.parent / "catalogupdate-ai" / "src" / "prompts.py"
_spec = importlib.util.spec_from_file_location("_catalogupdate_prompts", _target_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_prompt = _mod.build_prompt
FURNITURE_PRESETS = _mod.FURNITURE_PRESETS
DEFAULT_NEGATIVE_PROMPT = _mod.DEFAULT_NEGATIVE_PROMPT

__all__ = ["build_prompt", "FURNITURE_PRESETS", "DEFAULT_NEGATIVE_PROMPT"]

if __name__ == "__main__":
    _mod.main()
