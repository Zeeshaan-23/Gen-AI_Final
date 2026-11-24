import sys
import importlib.util
from pathlib import Path

_target_path = Path(__file__).resolve().parent.parent / "catalogupdate-ai" / "src" / "batch.py"
_spec = importlib.util.spec_from_file_location("_catalogupdate_batch", _target_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_batch = _mod.run_batch
zip_outputs = _mod.zip_outputs
MockCatalogEditor = _mod.MockCatalogEditor

__all__ = ["run_batch", "zip_outputs", "MockCatalogEditor"]

if __name__ == "__main__":
    _mod.main()
