"""Make the project root importable when Python starts inside backend/."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
project_root = str(PROJECT_ROOT)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
