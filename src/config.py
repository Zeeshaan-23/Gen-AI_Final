import sys
import importlib.util
from pathlib import Path

_target_path = Path(__file__).resolve().parent.parent / "catalogupdate-ai" / "src" / "config.py"
_spec = importlib.util.spec_from_file_location("_catalogupdate_config", _target_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

DEFAULT_STEPS = _mod.DEFAULT_STEPS
DEFAULT_GUIDANCE = _mod.DEFAULT_GUIDANCE
DEFAULT_STRENGTH = _mod.DEFAULT_STRENGTH
DEFAULT_SEED = _mod.DEFAULT_SEED
DEFAULT_MASK_THRESHOLD = _mod.DEFAULT_MASK_THRESHOLD
DEFAULT_RETRY_MASK_THRESHOLD = _mod.DEFAULT_RETRY_MASK_THRESHOLD
DEFAULT_MASK_DILATION_PX = _mod.DEFAULT_MASK_DILATION_PX
DEFAULT_MASK_BLUR_RADIUS = _mod.DEFAULT_MASK_BLUR_RADIUS
MIN_MASK_COVERAGE_PCT = _mod.MIN_MASK_COVERAGE_PCT
