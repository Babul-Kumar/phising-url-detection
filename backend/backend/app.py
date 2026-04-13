"""Compatibility ASGI entrypoint for `uvicorn backend.app:app` inside backend/."""

from __future__ import annotations

import sys
from pathlib import Path


PARENT_DIR = Path(__file__).resolve().parent.parent
parent_dir = str(PARENT_DIR)

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app_runtime import app
