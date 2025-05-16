"""Grace top‑level package.

Provides backward‑compat shim so existing 'from utils import ...' calls
continue to work until every import is migrated to
`from grace.utils.common import ...`.
"""

from importlib import import_module as _imp
import sys as _sys

_legacy = _imp("grace.utils.common")
_sys.modules[__name__ + ".utils"] = _legacy

# Re‑export convenience names
from grace.utils.common import *
