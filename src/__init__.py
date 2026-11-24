import sys
from pathlib import Path

# Add catalogupdate-ai to sys.path so src imports from catalogupdate-ai/src
_pkg_root = Path(__file__).resolve().parent.parent / "catalogupdate-ai"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.engine import CatalogEditor

__all__ = ["CatalogEditor"]
