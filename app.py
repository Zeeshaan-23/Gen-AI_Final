import sys
import importlib.util
from pathlib import Path

_target_path = Path(__file__).resolve().parent / "catalogupdate-ai" / "app.py"
_spec = importlib.util.spec_from_file_location("_catalogupdate_app", _target_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

demo = _mod.demo

if __name__ == "__main__":
    demo.queue().launch()
