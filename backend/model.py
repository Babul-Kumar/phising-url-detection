"""Inference helpers for the phishing URL detection API."""

from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional

sys.dont_write_bytecode = True

try:
    from .phishing_pipeline import (
        DEFAULT_MODEL_PATH,
        get_model_metadata,
        load_model_bundle,
        predict_url,
    )
except ImportError:  # pragma: no cover - allows running from inside backend/
    from phishing_pipeline import (
        DEFAULT_MODEL_PATH,
        get_model_metadata,
        load_model_bundle,
        predict_url,
    )


def load_model(path: str = str(DEFAULT_MODEL_PATH)) -> Mapping[str, Any]:
    """Load the trained phishing detector bundle."""
    return load_model_bundle(path)


def batch_predict(
    urls: Iterable[str],
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> List[Dict[str, Any]]:
    """Predict a batch of URLs using the saved phishing detector."""
    bundle = load_model(model_path)
    return [predict_url(url, bundle_or_path=bundle) for url in urls]
