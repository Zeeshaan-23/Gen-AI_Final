import sys
import importlib.util
from pathlib import Path

_target_path = Path(__file__).resolve().parent.parent / "catalogupdate-ai" / "src" / "engine.py"
_spec = importlib.util.spec_from_file_location("_catalogupdate_engine", _target_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CatalogEditor = _mod.CatalogEditor

__all__ = ["CatalogEditor"]
