from __future__ import annotations

import pkgutil
from pathlib import Path

__path__ = pkgutil.extend_path(__path__, __name__)

_nested_root = Path(__file__).resolve().parent.parent / "asparagus" / "preprocessing" / "asparagus_preprocessing"
if _nested_root.is_dir():
    nested_str = str(_nested_root)
    if nested_str in __path__:
        __path__.remove(nested_str)
    __path__.insert(0, nested_str)
