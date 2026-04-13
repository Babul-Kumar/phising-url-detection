"""
Production-ready phishing website detection pipeline.

The bundled training data is the classic phishing websites feature dataset in
ARFF form. Because the original dataset does not include raw URLs, this module
derives a URL-servable feature space from the legacy columns so the deployed
model can score real URLs using features extracted directly from the URL string.
"""

from __future__ import annotations

import ipaddress
import json
import math
import random
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

sys.dont_write_bytecode = True

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# matplotlib backend must be set before pyplot is imported
matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency guard
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "output"

DEFAULT_ARFF_PATH = DATA_DIR / "Training Dataset.arff"
DEFAULT_CSV_PATH = DATA_DIR / "phishing_websites.csv"
DEFAULT_MODEL_PATH = ARTIFACTS_DIR / "phishing_website_detector.joblib"
LEGACY_MODEL_PATH = MODELS_DIR / "phishing_website_detector.joblib"

TARGET_COLUMN = "Result"
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MAX_FPR = 0.03
CV_SPLITS = 5
CALIBRATION_SPLITS = 3

URL_KEYWORDS = ("login", "verify", "secure", "bank", "account")
SHORTENER_DOMAINS = {
    "bit.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
}
TRUSTED_DOMAINS = {
    "amazon.com",
    "apple.com",
    "bankofamerica.com",
    "github.com",
    "google.com",
    "microsoft.com",
    "openai.com",
    "paypal.com",
    "wikipedia.org",
}
RISKY_TLDS = {
    ".biz",
    ".click",
    ".country",
    ".gq",
    ".info",
    ".link",
    ".live",
    ".loan",
    ".ml",
    ".monster",
    ".online",
    ".ru",
    ".support",
    ".tk",
    ".top",
    ".work",
    ".xyz",
}
BRAND_DOMAINS = {
    "amazon": "amazon.com",
    "apple": "apple.com",
    "bankofamerica": "bankofamerica.com",
    "google": "google.com",
    "github": "github.com",
    "microsoft": "microsoft.com",
    "openai": "openai.com",
    "paypal": "paypal.com",
}
HEURISTIC_REASON_LABELS = {
    "at_symbol": "@ symbol used",
    "double_slash": "extra double slash in URL",
    "fake_https_token": "misleading https token",
    "hyphenated_domain": "hyphenated domain",
    "ip_address": "IP address used",
    "long_url": "very long URL",
    "many_dots": "many dots in hostname",
    "many_subdomains": "many subdomains",
    "no_https": "no HTTPS",
    "non_standard_port": "non-standard port used",
    "risky_tld": "suspicious domain",
    "shortener": "URL shortener used",
    "suspicious_keywords": "security-themed keywords",
}
LEGITIMATE_MAX_PROBABILITY = 0.40
PHISHING_MIN_PROBABILITY = 0.75
ADAPTIVE_HEURISTIC_TRIGGER = 0.40
ADAPTIVE_PHISHING_MIN_PROBABILITY = 0.60
HEURISTIC_MAX_PROBABILITY = 0.60
HYBRID_ML_WEIGHT = 0.70
HYBRID_HEURISTIC_WEIGHT = 0.30
MAX_COMBO_HEURISTIC_ADJUSTMENT = 0.30
MAX_HYBRID_PROBABILITY = 0.95
MAX_NEGATIVE_HEURISTIC_ADJUSTMENT = 0.15
HARD_RULE_PHISHING_PROBABILITY = 0.90
CONFIDENCE_MEDIUM_MIN = 0.35
CONFIDENCE_HIGH_MIN = 0.75
DEFAULT_EXAMPLE_URLS = [
    "https://secure-paypal-login.verify-account-update.xyz/session/confirm",
    "https://www.microsoft.com/security",
    "http://185.23.54.11/bank/login.php?redirect=secure",
]

TRAINING_FEATURES = [
    "url_length",
    "has_ip_address",
    "uses_https",
    "num_dots",
    "suspicious_keywords",
    "has_hyphen",
    "subdomain_count",
    "has_at_symbol",
    "has_double_slash",
    "has_shortener",
    "has_port",
    "contains_https_token",
]

FEATURE_DESCRIPTIONS = {
    "url_length": "Long URLs can hide malicious paths and confuse users.",
    "has_ip_address": "Using a raw IP address is a strong phishing signal.",
    "uses_https": "HTTPS usage lowers risk but is not sufficient by itself.",
    "num_dots": "Many dots often indicate deceptive nesting and fake subdomains.",
    "suspicious_keywords": "Words like login, verify, secure, bank, and account are common bait.",
    "has_hyphen": "Hyphenated domains are frequently used for brand impersonation.",
    "subdomain_count": "Too many subdomains can hide the true registrable domain.",
    "has_at_symbol": "The @ symbol is unusual and often used in phishing obfuscation.",
    "has_double_slash": "Double slashes in the path can indicate redirect tricks.",
    "has_shortener": "Shorteners obscure the final destination.",
    "has_port": "Non-standard ports are uncommon on legitimate public websites.",
    "contains_https_token": "Embedding 'https' inside the hostname/path is a common deception tactic.",
}

BundleLike = Union[str, Path, Mapping[str, Any]]


def _set_random_seed(seed: int = RANDOM_STATE) -> None:
    """Set deterministic seeds used throughout the pipeline."""
    random.seed(seed)
    np.random.seed(seed)


def _safe_float(value: Any, digits: int = 4) -> float:
    """Convert numpy scalar types into builtin floats."""
    return round(float(value), digits)


def _ensure_parent_dirs(*paths: Path) -> None:
    """Create parent directories for all supplied file paths."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_url(url: str) -> str:
    """Normalize a URL string so parsing works consistently."""
    cleaned = str(url).strip()
    if not cleaned:
        raise ValueError("A non-empty URL is required.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = f"http://{cleaned}"
    return cleaned


def _root_domain(hostname: str) -> str:
    """Extract a simple root-domain representation from a hostname."""
    hostname = hostname.lower().strip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def _is_ip_address(hostname: str) -> bool:
    """Return True when a hostname is an IPv4 or IPv6 literal."""
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _subdomain_count_from_hostname(hostname: str) -> int:
    """Estimate the number of subdomains from a hostname."""
    if not hostname or _is_ip_address(hostname):
        return 0
    hostname = hostname.lower().strip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    parts = hostname.split(".")
    return max(len(parts) - 2, 0)


def convert_arff_to_csv(
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
) -> Path:
    """Convert the bundled numeric ARFF dataset into CSV."""
    arff_path = Path(arff_path)
    csv_path = Path(csv_path)
    _ensure_parent_dirs(csv_path)

    if not arff_path.exists():
        raise FileNotFoundError(f"ARFF dataset not found at {arff_path}")

    attributes: List[str] = []
    rows: List[List[Any]] = []
    in_data = False

    with arff_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            lowered = line.lower()
            if lowered.startswith("@attribute"):
                match = re.match(
                    r"@attribute\s+('([^']+)'|\"([^\"]+)\"|([^\s]+))",
                    line,
                    flags=re.IGNORECASE,
                )
                if not match:
                    raise ValueError(f"Could not parse attribute line: {line}")
                name = next(group for group in match.groups()[1:] if group)
                attributes.append(name)
                continue

            if lowered.startswith("@data"):
                in_data = True
                continue

            if in_data and not lowered.startswith("@"):
                values = [value.strip() for value in line.split(",")]
                if len(values) != len(attributes):
                    raise ValueError(
                        f"Unexpected column count in ARFF row ({len(values)} vs {len(attributes)})."
                    )
                rows.append([np.nan if value == "?" else value for value in values])

    dataset = pd.DataFrame(rows, columns=attributes).apply(pd.to_numeric, errors="coerce")
    dataset.to_csv(csv_path, index=False)
    return csv_path


def load_data(
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
) -> Tuple[pd.DataFrame, Path]:
    """Load the phishing dataset from CSV, converting from ARFF when needed."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        csv_path = convert_arff_to_csv(arff_path=arff_path, csv_path=csv_path)

    dataset = pd.read_csv(csv_path).replace("?", np.nan)
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Dataset must include '{TARGET_COLUMN}'.")
    return dataset, csv_path


def _derive_training_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Derive URL-servable features from the legacy phishing dataset.

    The source dataset contains hand-engineered website signals rather than raw
    URLs. This projection keeps the model aligned with features we can extract
    directly from a URL string at prediction time.

    All binary features (has_hyphen, has_double_slash, suspicious_keywords) are
    clamped to 0/1 here to match the binary values produced by extract_url_features
    at inference time, preventing a train/inference distribution mismatch.
    """
    return pd.DataFrame(
        {
            "url_length": dataset["URL_Length"].map({1: 40, 0: 65, -1: 90}).astype(float),
            "has_ip_address": (dataset["having_IP_Address"] == -1).astype(int),
            "uses_https": (dataset["SSLfinal_State"] >= 0).astype(int),
            "num_dots": dataset["having_Sub_Domain"].map({1: 1, 0: 2, -1: 4}).astype(float),
            # FIX: clamp to 0/1 — extract_url_features produces a boolean hit count (0 or 1)
            # from URL_KEYWORDS, not a multi-valued integer. Keeping binary avoids a
            # train/inference distribution mismatch.
            "suspicious_keywords": (
                (dataset["HTTPS_token"] == 1)
                | (dataset["Request_URL"] == -1)
                | (dataset["Abnormal_URL"] == 1)
                | (dataset["Prefix_Suffix"] == -1)
            ).astype(int),
            # FIX: clamp to 0/1 — extract_url_features returns int(hyphen_count > 0),
            # not the raw count, so training must also use a binary indicator.
            "has_hyphen": (dataset["Prefix_Suffix"] == -1).astype(int),
            "subdomain_count": dataset["having_Sub_Domain"].map({1: 0, 0: 1, -1: 2}).astype(float),
            "has_at_symbol": (dataset["having_At_Symbol"] == -1).astype(int),
            # FIX: clamp to 0/1 — extract_url_features counts occurrences but the
            # training proxy must match; using a boolean keeps distributions aligned.
            "has_double_slash": (dataset["double_slash_redirecting"] == 1).astype(int),
            "has_shortener": (dataset["Shortining_Service"] == 1).astype(int),
            "has_port": (dataset["port"] == -1).astype(int),
            "contains_https_token": (dataset["HTTPS_token"] == 1).astype(int),
        }
    )


def preprocess_data(
    dataset: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Prepare train/test splits and metadata for model training."""
    features = _derive_training_features(dataset)
    target = (dataset[TARGET_COLUMN].astype(int) == -1).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )

    return {
        "features": features,
        "target": target,
        "feature_names": list(features.columns),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "missing_values_before_imputation": int(features.isna().sum().sum()),
    }


def _build_pipeline(classifier: BaseEstimator, use_scaler: bool) -> Pipeline:
    """Create a preprocessing + classifier pipeline."""
    steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps=steps)


def _build_model_catalog(random_state: int = RANDOM_STATE) -> Dict[str, Dict[str, Any]]:
    """Create the candidate model catalog."""
    models: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "estimator": _build_pipeline(
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=random_state,
                ),
                use_scaler=True,
            ),
            "tree_model": False,
        },
        "random_forest": {
            "estimator": _build_pipeline(
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=1,
                ),
                use_scaler=False,
            ),
            "tree_model": True,
        },
    }

    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models["xgboost"] = {
            "estimator": _build_pipeline(
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=1,
                ),
                use_scaler=False,
            ),
            "tree_model": True,
        }

    return models


def _cv_strategy(
    y: pd.Series,
    max_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Build a safe stratified CV strategy based on the minority class size."""
    min_class_count = int(y.value_counts().min())
    n_splits = max(2, min(max_splits, min_class_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _predict_oof_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    cv_splits: int = CV_SPLITS,
) -> np.ndarray:
    """Collect out-of-fold predicted probabilities for threshold selection."""
    cv = _cv_strategy(target, max_splits=cv_splits)
    return cross_val_predict(
        estimator,
        features,
        target,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def _roc_threshold_frame(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """Build a ROC threshold table for diagnostics and plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "fpr": fpr,
            "recall": tpr,
        }
    )
    return frame[np.isfinite(frame["threshold"])].reset_index(drop=True)


def find_best_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    max_fpr: float = DEFAULT_MAX_FPR,
) -> float:
    """
    Select the threshold under an FPR constraint while maximizing recall.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_threshold = 0.5
    best_recall = 0.0

    for index in range(len(thresholds)):
        if np.isfinite(thresholds[index]) and fpr[index] <= max_fpr and tpr[index] > best_recall:
            best_recall = float(tpr[index])
            best_threshold = float(thresholds[index])

    return float(best_threshold)


def calibrate_model(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int = CALIBRATION_SPLITS,
) -> CalibratedClassifierCV:
    """Calibrate a model with sigmoid calibration for reliable probabilities."""
    calibrator = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=_cv_strategy(y_train, max_splits=cv_splits),
    )
    calibrator.fit(x_train, y_train)
    return calibrator


def _compute_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute the requested classification metrics."""
    predictions = (np.asarray(y_probs) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": _safe_float(threshold, 6),
        "accuracy": _safe_float(accuracy_score(y_true, predictions)),
        "precision": _safe_float(precision_score(y_true, predictions, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": _safe_float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": _safe_float(roc_auc_score(y_true, y_probs)),
        "average_precision": _safe_float(average_precision_score(y_true, y_probs)),
        "false_positive_rate": _safe_float(fp / max(fp + tn, 1), 6),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _compute_metrics_from_predictions(
    y_true: Union[pd.Series, np.ndarray],
    predictions: Union[pd.Series, np.ndarray, List[int]],
    probabilities: Union[pd.Series, np.ndarray, List[float]],
    threshold: float,
) -> Dict[str, Any]:
    """Compute metrics for already-decided operational predictions."""
    y_true_array = np.asarray(y_true, dtype=int)
    prediction_array = np.asarray(predictions, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true_array, prediction_array, labels=[0, 1]).ravel()

    try:
        roc_auc = _safe_float(roc_auc_score(y_true_array, probability_array))
    except ValueError:
        roc_auc = float("nan")

    try:
        average_precision = _safe_float(average_precision_score(y_true_array, probability_array))
    except ValueError:
        average_precision = float("nan")

    return {
        "threshold": _safe_float(threshold, 6),
        "accuracy": _safe_float(accuracy_score(y_true_array, prediction_array)),
        "precision": _safe_float(precision_score(y_true_array, prediction_array, zero_division=0)),
        "recall": _safe_float(recall_score(y_true_array, prediction_array, zero_division=0)),
        "f1_score": _safe_float(f1_score(y_true_array, prediction_array, zero_division=0)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "false_positive_rate": _safe_float(fp / max(fp + tn, 1), 6),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_probs: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save a confusion matrix plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    predictions = (y_probs >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["legitimate", "phishing"],
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_precision_recall_curve(
    y_true: pd.Series,
    y_probs: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save a precision-recall curve with the operating point highlighted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_precision_recall_curve.png"
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    predictions = (y_probs >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, color="#1f77b4")
    ax.scatter(
        recall_score(y_true, predictions, zero_division=0),
        precision_score(y_true, predictions, zero_division=0),
        color="#d62728",
        s=80,
        label=f"threshold={threshold:.3f}",
    )
    ax.set_title(f"{model_name} Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_roc_curve(
    y_true: pd.Series,
    y_probs: np.ndarray,
    threshold: float,
    threshold_curve: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> Path:
    """Save the ROC curve and threshold-selected operating point."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_roc_curve.png"
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    operating_point = threshold_curve.iloc[
        (threshold_curve["threshold"] - threshold).abs().argsort()[:1]
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, color="#2ca02c", label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.axvline(max_fpr, linestyle=":", color="#d62728", label="max FPR")
    ax.scatter(
        operating_point["fpr"],
        operating_point["recall"],
        color="#d62728",
        s=80,
        label=f"threshold={threshold:.3f}",
    )
    ax.set_title(f"{model_name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_model(
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> Dict[str, Any]:
    """Fit, threshold, and evaluate a model."""
    output_dir = Path(output_dir)
    oof_probabilities = _predict_oof_probabilities(estimator, x_train, y_train)
    threshold_curve = _roc_threshold_frame(y_train, oof_probabilities)
    threshold = find_best_threshold(y_train, oof_probabilities, max_fpr=max_fpr)

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(x_train, y_train)
    test_probabilities = fitted_estimator.predict_proba(x_test)[:, 1]
    metrics = _compute_metrics(y_test, test_probabilities, threshold)

    artifacts = {
        "confusion_matrix": str(
            _plot_confusion_matrix(y_test, test_probabilities, threshold, model_name, output_dir)
        ),
        "precision_recall_curve": str(
            _plot_precision_recall_curve(y_test, test_probabilities, threshold, model_name, output_dir)
        ),
        "roc_curve": str(
            _plot_roc_curve(
                y_test,
                test_probabilities,
                threshold,
                threshold_curve,
                model_name,
                output_dir,
                max_fpr=max_fpr,
            )
        ),
    }

    threshold_csv = output_dir / f"{model_name}_roc_thresholds.csv"
    threshold_curve.to_csv(threshold_csv, index=False)
    artifacts["threshold_curve_csv"] = str(threshold_csv)

    return {
        "model_name": model_name,
        "model": fitted_estimator,
        "threshold": float(threshold),
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _tree_model_names(model_catalog: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Return the names of tree-based models in the catalog."""
    return [name for name, meta in model_catalog.items() if meta["tree_model"]]


def _select_best_tree_model(results: Mapping[str, Mapping[str, Any]]) -> str:
    """Choose the best tree model using precision-first ordering."""
    comparison = pd.DataFrame(
        {
            "model_name": name,
            "precision": result["metrics"]["precision"],
            "false_positives": result["metrics"]["false_positives"],
            "recall": result["metrics"]["recall"],
            "f1_score": result["metrics"]["f1_score"],
            "accuracy": result["metrics"]["accuracy"],
        }
        for name, result in results.items()
    )
    comparison = comparison.sort_values(
        by=["precision", "false_positives", "recall", "f1_score", "accuracy"],
        ascending=[False, True, False, False, False],
    )
    return str(comparison.iloc[0]["model_name"])


def _transform_for_explanation(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline preprocessing while keeping feature names intact."""
    transformed = frame.copy()
    if "imputer" in model.named_steps:
        transformed = pd.DataFrame(
            model.named_steps["imputer"].transform(transformed),
            columns=frame.columns,
            index=frame.index,
        )
    if "scaler" in model.named_steps:
        transformed = pd.DataFrame(
            model.named_steps["scaler"].transform(transformed),
            columns=frame.columns,
            index=frame.index,
        )
    return transformed


def _normalize_shap_values(shap_values: Any) -> np.ndarray:
    """Normalize SHAP output shapes across model implementations."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    return shap_values


def _plot_feature_importance(
    explanation_model: Pipeline,
    feature_names: List[str],
    model_name: str,
    output_dir: Path,
    top_n: int = 12,
) -> Optional[Path]:
    """Save global feature importance for a tree model."""
    classifier = explanation_model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_feature_importance.png"

    importance = np.asarray(classifier.feature_importances_, dtype=float)
    order = np.argsort(importance)[-min(top_n, len(feature_names)):]
    selected_names = np.asarray(feature_names)[order]
    selected_scores = importance[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(selected_names, selected_scores, color="#1f77b4")
    ax.set_title(f"{model_name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_shap_summary(
    explanation_model: Pipeline,
    reference_frame: pd.DataFrame,
    model_name: str,
    output_dir: Path,
) -> Optional[Path]:
    """Save a SHAP summary plot for the final tree model."""
    classifier = explanation_model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return None

    sampled = reference_frame.sample(min(len(reference_frame), 800), random_state=RANDOM_STATE)
    transformed = _transform_for_explanation(explanation_model, sampled)
    explainer = shap.TreeExplainer(classifier)
    shap_values = _normalize_shap_values(explainer.shap_values(transformed))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_shap_summary.png"
    plt.figure(figsize=(10, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        shap.summary_plot(shap_values, transformed, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    return output_path


def _single_prediction_explanation(
    explanation_model: Pipeline,
    aligned_features: pd.DataFrame,
    output_dir: Optional[Path] = None,
    plot_name: str = "prediction_shap_explanation.png",
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Generate top SHAP risk factors for a single prediction."""
    classifier = explanation_model.named_steps["classifier"]
    transformed = _transform_for_explanation(explanation_model, aligned_features)
    explainer = shap.TreeExplainer(classifier)
    shap_values = _normalize_shap_values(explainer.shap_values(transformed))[0]

    contributions = pd.DataFrame(
        {
            "feature": transformed.columns,
            "feature_value": transformed.iloc[0].values,
            "shap_value": shap_values,
            "abs_shap_value": np.abs(shap_values),
            "description": [FEATURE_DESCRIPTIONS.get(name, name) for name in transformed.columns],
        }
    )
    positive = contributions[contributions["shap_value"] > 0].sort_values(
        by="shap_value",
        ascending=False,
    )
    ranked = positive if not positive.empty else contributions.sort_values(by="abs_shap_value", ascending=False)
    top_rows = ranked.head(top_n)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / plot_name
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(top_rows["feature"], top_rows["shap_value"], color="#d62728")
        ax.set_title("Top SHAP Risk Factors")
        ax.set_xlabel("SHAP contribution toward phishing")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return [
        {
            "feature": str(row.feature),
            "feature_value": _safe_float(row.feature_value, 6),
            "contribution": _safe_float(row.shap_value, 6),
            "description": str(row.description),
        }
        for row in top_rows.itertuples(index=False)
    ]


def _confidence_score(probability: float) -> float:
    """Measure certainty from distance to the ambiguous 0.5 decision boundary."""
    return round(abs(float(probability) - 0.5) * 2.0, 6)


def _confidence_bucket(probability: float) -> str:
    """Map a probability into a low/medium/high certainty bucket."""
    score = _confidence_score(probability)
    if score < CONFIDENCE_MEDIUM_MIN:
        return "low"
    if score < CONFIDENCE_HIGH_MIN:
        return "medium"
    return "high"


def _triage_label(
    probability: float,
    phishing_min_probability: float = PHISHING_MIN_PROBABILITY,
) -> str:
    """Return the coarse triage band from the blended probability."""
    phishing_threshold = min(
        max(float(phishing_min_probability), LEGITIMATE_MAX_PROBABILITY),
        MAX_HYBRID_PROBABILITY,
    )
    if probability >= phishing_threshold:
        return "phishing"
    if probability >= LEGITIMATE_MAX_PROBABILITY:
        return "suspicious"
    return "legitimate"


def _effective_phishing_threshold(heuristic_probability: float) -> float:
    """Lower the phishing band only when independent heuristic evidence is strong."""
    if heuristic_probability >= ADAPTIVE_HEURISTIC_TRIGGER:
        return ADAPTIVE_PHISHING_MIN_PROBABILITY
    return PHISHING_MIN_PROBABILITY


def _display_label(risk_band: str, confidence_score: float) -> str:
    """Return a user-facing label that combines risk band and confidence."""
    if risk_band == "phishing":
        return "phishing"
    if risk_band == "suspicious":
        return "suspicious (uncertain)" if confidence_score < 0.30 else "suspicious"
    return "legitimate" if confidence_score > 0.70 else "likely legitimate"


def _risk_band_config() -> Dict[str, float]:
    """Expose the probability bands used for final triage decisions."""
    return {
        "legitimate_max_probability": LEGITIMATE_MAX_PROBABILITY,
        "phishing_min_probability": PHISHING_MIN_PROBABILITY,
        "adaptive_heuristic_trigger": ADAPTIVE_HEURISTIC_TRIGGER,
        "adaptive_phishing_min_probability": ADAPTIVE_PHISHING_MIN_PROBABILITY,
        "hybrid_ml_weight": HYBRID_ML_WEIGHT,
        "hybrid_heuristic_weight": HYBRID_HEURISTIC_WEIGHT,
        "max_combo_heuristic_adjustment": MAX_COMBO_HEURISTIC_ADJUSTMENT,
        "max_hybrid_probability": MAX_HYBRID_PROBABILITY,
        "hard_rule_phishing_probability": HARD_RULE_PHISHING_PROBABILITY,
    }


def _reason_entry(reason: str, weight: float, signal: str = "risk") -> Dict[str, Any]:
    """Build a structured explanation item for heuristic adjustments."""
    return {
        "reason": reason,
        "weight": round(float(weight), 4),
        "signal": signal,
    }


def _add_capped_combo_reasons(
    reason_details: List[Dict[str, Any]],
    combo_reasons: Iterable[Tuple[str, float]],
) -> None:
    """Add overlapping combo heuristics while capping their total influence."""
    remaining_weight = MAX_COMBO_HEURISTIC_ADJUSTMENT
    for reason, weight in combo_reasons:
        if remaining_weight <= 0:
            break
        applied_weight = min(float(weight), remaining_weight)
        reason_details.append(_reason_entry(reason, applied_weight))
        remaining_weight -= applied_weight


def _format_reason_details(reason_details: Iterable[Mapping[str, Any]]) -> List[str]:
    """Render structured reason details into compact demo-friendly strings."""
    return [
        f"{detail['reason']} ({float(detail['weight']):+.2f})"
        for detail in reason_details
    ]


def _bounded_heuristic_probability(reason_details: Iterable[Mapping[str, Any]]) -> float:
    """Cap stacked heuristic evidence and prevent trust signals from overcorrecting."""
    positive_score = 0.0
    negative_score = 0.0
    for detail in reason_details:
        weight = float(detail.get("weight", 0.0))
        if weight >= 0:
            positive_score += weight
        else:
            negative_score += abs(weight)

    adjusted_score = positive_score - min(negative_score, MAX_NEGATIVE_HEURISTIC_ADJUSTMENT)
    return min(max(adjusted_score, 0.0), HEURISTIC_MAX_PROBABILITY)


def _fuse_hybrid_probability(base_probability: float, heuristic_probability: float) -> float:
    """Fuse ML and heuristic scores with damping so rules cannot dominate."""
    blended_probability = (
        (HYBRID_ML_WEIGHT * float(base_probability))
        + (HYBRID_HEURISTIC_WEIGHT * float(heuristic_probability))
    )
    return min(
        max(blended_probability, 0.0),
        MAX_HYBRID_PROBABILITY,
    )


def _dual_threshold_decision(
    probability: float,
    extracted_features: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str, float]:
    """Return triage label/confidence using dual thresholds."""
    dual_score = float(probability)
    if extracted_features is not None:
        # Triage-only boost to improve recall for obvious phishing lexical signals.
        has_ip_address = int(extracted_features.get("has_ip_address", 0))
        suspicious_keywords = int(extracted_features.get("suspicious_keywords", 0))
        has_hyphen = int(extracted_features.get("has_hyphen", 0))
        subdomain_count = float(extracted_features.get("subdomain_count", 0))

        if has_ip_address:
            dual_score = max(dual_score, 0.90)
        if suspicious_keywords >= 1 and has_hyphen >= 1:
            dual_score = max(dual_score, 0.60)
        if suspicious_keywords >= 2 and subdomain_count >= 1:
            dual_score = max(dual_score, 0.80)

    dual_score = min(dual_score, 1.0)
    return _triage_label(dual_score), _confidence_bucket(dual_score), dual_score


def _sigmoid_risk_score(probability: float) -> float:
    """Optional nonlinear risk score mapping."""
    return 100.0 / (1.0 + math.exp(-5.0 * (probability - 0.5)))


def _heuristic_cyber_risk(
    extracted_features: Mapping[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute a conservative URL-risk overlay with weighted explanation entries."""
    reason_details: List[Dict[str, Any]] = []

    def add_reason(reason: str, weight: float) -> None:
        reason_details.append(_reason_entry(reason, weight))

    url_length = float(extracted_features.get("url_length", 0))
    hyphen_count = int(extracted_features.get("has_hyphen", 0))
    subdomain_count = float(extracted_features.get("subdomain_count", 0))

    if url_length >= 75:
        add_reason("long URL", 0.18)
    elif url_length >= 54:
        add_reason("long URL", 0.10)

    if int(extracted_features.get("has_ip_address", 0)):
        add_reason("IP address used", 0.50)

    if not int(extracted_features.get("uses_https", 0)):
        add_reason("no HTTPS", 0.08)

    if hyphen_count > 0:
        add_reason("hyphenated domain", min(0.14, 0.08 + (hyphen_count - 1) * 0.02))

    if subdomain_count >= 2:
        add_reason("many subdomains", 0.18)
    elif subdomain_count >= 1:
        add_reason("many subdomains", 0.08)

    if int(extracted_features.get("has_at_symbol", 0)):
        add_reason("@ symbol used", 0.12)

    if int(extracted_features.get("has_double_slash", 0)):
        add_reason("extra double slash in URL", 0.10)

    if int(extracted_features.get("has_shortener", 0)):
        add_reason("URL shortener used", 0.15)

    if int(extracted_features.get("has_port", 0)):
        add_reason("non-standard port used", 0.10)

    if int(extracted_features.get("contains_https_token", 0)):
        add_reason("misleading https token", 0.12)

    # Return the bounded probability computed from reason_details (not a running local sum)
    # so that the caller always receives a consistent, capped value.
    return _bounded_heuristic_probability(reason_details), reason_details


def _hybrid_decision_from_feature_row(
    base_probability: float,
    extracted_features: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate the deployed hybrid logic from held-out feature rows.

    The training dataset does not contain raw URLs, so this excludes URL-only
    signals like brand strings, suspicious TLDs, and exact login/verify tokens.
    """
    _, reason_details = _heuristic_cyber_risk(extracted_features)

    def add_reason(reason: str, weight: float, signal: str = "risk") -> None:
        reason_details.append(_reason_entry(reason, weight, signal=signal))

    has_ip_address = bool(int(extracted_features.get("has_ip_address", 0)))
    suspicious_keywords = int(extracted_features.get("suspicious_keywords", 0))
    has_hyphen = int(extracted_features.get("has_hyphen", 0))
    subdomain_count = float(extracted_features.get("subdomain_count", 0))

    if suspicious_keywords > 0:
        add_reason("security-themed keywords", 0.12)
    combo_reasons: List[Tuple[str, float]] = []
    if has_ip_address and suspicious_keywords > 0:
        combo_reasons.append(("IP address with suspicious keyword", 0.30))
    if suspicious_keywords > 0 and has_hyphen > 0:
        combo_reasons.append(("keyword and hyphen combo", 0.25))
    if suspicious_keywords > 0 and subdomain_count >= 1:
        combo_reasons.append(("keyword and subdomain combo", 0.20))
    _add_capped_combo_reasons(reason_details, combo_reasons)

    heuristic_probability = _bounded_heuristic_probability(reason_details)
    probability = _fuse_hybrid_probability(base_probability, heuristic_probability)
    effective_threshold = _effective_phishing_threshold(heuristic_probability)
    risk_band = _triage_label(probability, phishing_min_probability=effective_threshold)
    if risk_band == "legitimate" and heuristic_probability >= LEGITIMATE_MAX_PROBABILITY:
        risk_band = "suspicious"

    return {
        "probability": probability,
        "heuristic_probability": heuristic_probability,
        "risk_band": risk_band,
        "effective_phishing_min_probability": effective_threshold,
    }


def _hybrid_heuristic_probability(
    url: str,
    extracted_features: Mapping[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Convert lexical heuristics into a bounded 0..1 support signal."""
    # Start from the base heuristic risk; receive reason_details by reference for mutation.
    _, reason_details = _heuristic_cyber_risk(extracted_features)

    # FIX: removed the nonlocal heuristic_prob accumulation that was updating a local
    # variable never returned. All additional reasons are appended to reason_details and
    # the final bounded probability is recomputed from the complete list at the end.
    def add_reason(reason: str, weight: float, signal: str = "risk") -> None:
        reason_details.append(_reason_entry(reason, weight, signal=signal))

    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower().strip(".")
    root_domain = _root_domain(hostname)
    url_lower = normalized.lower()
    hostname_tokens = {token for token in re.split(r"[.\-]+", hostname) if token}
    is_trusted_domain = root_domain in TRUSTED_DOMAINS or root_domain.endswith((".gov", ".edu"))
    has_ip_address = bool(int(extracted_features.get("has_ip_address", 0)))
    combo_reasons: List[Tuple[str, float]] = []

    if float(extracted_features.get("num_dots", 0)) > 3 and not is_trusted_domain:
        add_reason("many dots in hostname", 0.08)

    login_present = "login" in url_lower
    verify_present = "verify" in url_lower
    if login_present:
        add_reason("login keyword", 0.12)
    if verify_present:
        add_reason("verify keyword", 0.12)
    if login_present and verify_present:
        combo_reasons.append(("login and verify keyword combo", 0.25))
    if has_ip_address and login_present:
        combo_reasons.append(("IP address with login path", 0.30))

    keyword_hits = int(extracted_features.get("suspicious_keywords", 0))
    other_keyword_hits = max(keyword_hits - int(login_present) - int(verify_present), 0)
    if other_keyword_hits > 0:
        add_reason(
            "security-themed keywords",
            min(0.12, 0.08 + (other_keyword_hits - 1) * 0.02),
        )

    risky_tld = any(root_domain.endswith(tld) for tld in RISKY_TLDS)
    if risky_tld:
        add_reason("suspicious domain", 0.15)

    hyphen_count = int(extracted_features.get("has_hyphen", 0))
    subdomain_count = float(extracted_features.get("subdomain_count", 0))
    for brand_keyword, trusted_domain in BRAND_DOMAINS.items():
        if brand_keyword in hostname_tokens and root_domain != trusted_domain:
            brand_bonus = 0.40 if (risky_tld or hyphen_count > 0 or subdomain_count > 0) else 0.25
            add_reason(f"{brand_keyword} brand impersonation", brand_bonus)
            if risky_tld:
                combo_reasons.append(("brand impersonation on suspicious domain", 0.30))
            break

    _add_capped_combo_reasons(reason_details, combo_reasons)

    if is_trusted_domain:
        add_reason("trusted domain", -MAX_NEGATIVE_HEURISTIC_ADJUSTMENT, signal="trust")

    # Recompute the bounded probability from the complete, extended reason_details list.
    return _bounded_heuristic_probability(reason_details), reason_details


def _hybrid_decision_from_url(
    url: str,
    extracted_features: Mapping[str, Any],
    base_probability: float,
) -> Dict[str, Any]:
    """Apply the same final URL risk engine used by predict_url."""
    heuristic_prob, reason_details = _hybrid_heuristic_probability(url, extracted_features)
    probability = _fuse_hybrid_probability(base_probability, heuristic_prob)

    normalized_url = _normalize_url(url)
    parsed_url = urlparse(normalized_url)
    hostname = (parsed_url.hostname or "").lower().strip(".")
    root_domain = _root_domain(hostname)
    url_lower = normalized_url.lower()
    ip_login_hard_rule = bool(int(extracted_features.get("has_ip_address", 0))) and "login" in url_lower
    paypal_hard_rule = "paypal" in hostname and root_domain != BRAND_DOMAINS["paypal"]
    hard_rule_phishing = ip_login_hard_rule or paypal_hard_rule
    if ip_login_hard_rule:
        reason_details.append(_reason_entry("hard rule: IP address login URL", 0.0, signal="rule"))
    if paypal_hard_rule:
        reason_details.append(_reason_entry("hard rule: paypal impersonation", 0.0, signal="rule"))
    if hard_rule_phishing:
        probability = max(probability, HARD_RULE_PHISHING_PROBABILITY)

    has_brand_impersonation = any(
        str(reason["reason"]).endswith("brand impersonation")
        for reason in reason_details
    )
    strong_heuristic_signal = heuristic_prob >= LEGITIMATE_MAX_PROBABILITY
    effective_phishing_min_probability = _effective_phishing_threshold(heuristic_prob)
    risk_band = _triage_label(
        probability,
        phishing_min_probability=effective_phishing_min_probability,
    )
    if hard_rule_phishing:
        risk_band = "phishing"
    if risk_band == "legitimate" and (has_brand_impersonation or strong_heuristic_signal):
        risk_band = "suspicious"

    if probability > 0.9:
        risk_level = "critical"
    elif probability > 0.7:
        risk_level = "high"
    elif probability > 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    if risk_band == "suspicious" and risk_level == "low":
        risk_level = "medium"

    confidence_score = _confidence_score(probability)
    confidence = _confidence_bucket(probability)

    return {
        "probability": probability,
        "heuristic_probability": heuristic_prob,
        "effective_phishing_min_probability": effective_phishing_min_probability,
        "hard_rule_phishing": hard_rule_phishing,
        "risk_band": risk_band,
        "risk_level": risk_level,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "prediction": _display_label(risk_band, confidence_score),
        "reasons": _format_reason_details(reason_details),
        "reason_details": reason_details,
    }


def preprocess_single(
    extracted_features: Mapping[str, Any],
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Preprocess one extracted URL feature dictionary for inference."""
    return align_feature_columns(extracted_features, feature_names)


def align_feature_columns(
    extracted_features: Mapping[str, Any],
    feature_names: Iterable[str],
) -> pd.DataFrame:
    """Align extracted URL features with the training feature columns."""
    ordered = {name: extracted_features.get(name, 0) for name in feature_names}
    return pd.DataFrame([ordered], columns=list(feature_names))


def extract_url_features(url: str) -> Dict[str, Union[int, float]]:
    """
    Extract real lexical features directly from the URL string.

    Binary features (has_hyphen, has_double_slash) are clamped to 0/1 to match
    the training distribution produced by _derive_training_features.
    """
    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("The provided URL does not contain a valid hostname.")

    try:
        port_value = parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc

    is_ip = _is_ip_address(hostname)
    subdomain_count = _subdomain_count_from_hostname(hostname)
    root_domain = _root_domain(hostname)
    remainder = normalized.split("://", 1)[-1]
    lowered = normalized.lower()
    keyword_hits = sum(keyword in lowered for keyword in URL_KEYWORDS)
    hyphen_count = hostname.count("-")
    double_slash_count = remainder.count("//")

    return {
        "url_length": float(len(normalized)),
        "has_ip_address": int(is_ip),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "num_dots": float(hostname.count(".")),
        # FIX: clamp suspicious_keywords to 0/1 to match the binary training proxy.
        "suspicious_keywords": int(min(keyword_hits, 1)),
        # FIX: clamp has_hyphen to 0/1 (presence indicator, not count) to match training.
        "has_hyphen": int(min(hyphen_count, 1)),
        "subdomain_count": float(subdomain_count),
        "has_at_symbol": int("@" in normalized),
        # FIX: clamp has_double_slash to 0/1 to match the binary training proxy.
        "has_double_slash": int(min(double_slash_count, 1)),
        "has_shortener": int(root_domain in SHORTENER_DOMAINS),
        "has_port": int(port_value not in (None, 80, 443)),
        "contains_https_token": int("https" in hostname),
    }


def load_model_bundle(bundle_or_path: BundleLike = DEFAULT_MODEL_PATH) -> Mapping[str, Any]:
    """Load a saved model bundle from disk."""
    if isinstance(bundle_or_path, Mapping):
        return bundle_or_path

    bundle_path = Path(bundle_or_path)
    if (
        bundle_path == DEFAULT_MODEL_PATH
        and not bundle_path.exists()
        and LEGACY_MODEL_PATH.exists()
    ):
        bundle_path = LEGACY_MODEL_PATH

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Trained model bundle not found at {bundle_path}. Run train.py first."
        )
    return joblib.load(bundle_path)


def predict_from_features(
    extracted_features: Mapping[str, Any],
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    save_explanation_plot: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Predict phishing risk from extracted URL features."""
    bundle = load_model_bundle(bundle_or_path)
    feature_names = list(bundle["feature_names"])
    aligned = preprocess_single(extracted_features, feature_names)

    calibrated_model = bundle["model"]
    if hasattr(calibrated_model, "classes_"):
        classes = list(getattr(calibrated_model, "classes_"))
        if classes != [0, 1]:
            raise ValueError(
                f"Unexpected class mapping {classes}. Expected class 0=legitimate, 1=phishing."
            )

    proba = calibrated_model.predict_proba(aligned)
    probability = float(proba[0][1])
    threshold = float(bundle["threshold"])
    prediction = "phishing" if probability >= threshold else "legitimate"
    risk_score = round(probability * 100.0, 2)
    confidence = _confidence_bucket(probability)
    confidence_score = _confidence_score(probability)
    dual_threshold_label, dual_threshold_confidence, dual_threshold_score = _dual_threshold_decision(
        probability,
        extracted_features,
    )
    if debug:
        print(f"DEBUG -> prob={probability}, threshold={threshold}, decision={probability >= threshold}")

    # FIX: guard against legacy bundles that were saved without an explainability_model key.
    # predict_from_features is also called by predict_url which provides a fully-populated
    # bundle from run_training_pipeline, but callers may load older bundles.
    top_risk_factors: List[Dict[str, Any]] = []
    if "explainability_model" in bundle:
        explanation_output_dir = (
            Path(bundle["artifacts_directory"]) if save_explanation_plot and "artifacts_directory" in bundle else None
        )
        top_risk_factors = _single_prediction_explanation(
            explanation_model=bundle["explainability_model"],
            aligned_features=aligned,
            output_dir=explanation_output_dir,
        )

    return {
        "prediction": prediction,
        "risk_score": risk_score,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "probability": round(probability, 6),
        "risk_score_sigmoid": round(_sigmoid_risk_score(probability), 2),
        "model_threshold": round(threshold, 6),
        "risk_bands": _risk_band_config(),
        "dual_threshold_score": round(dual_threshold_score, 6),
        "dual_threshold_label": dual_threshold_label,
        "dual_threshold_confidence": dual_threshold_confidence,
        "top_risk_factors": [factor["feature"] for factor in top_risk_factors[:5]],
        "feature_contributions": top_risk_factors,
        "extracted_features": {key: _safe_float(value, 6) for key, value in extracted_features.items()},
        "model_name": bundle["model_name"],
    }


def predict_url(
    url: str,
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    save_explanation_plot: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Extract URL features and apply weighted ML + heuristic decisioning."""
    extracted = extract_url_features(url)
    result = predict_from_features(
        extracted_features=extracted,
        bundle_or_path=bundle_or_path,
        save_explanation_plot=save_explanation_plot,
        debug=debug,
    )

    base_prob = float(result["probability"])
    model_threshold = float(result["model_threshold"])
    decision = _hybrid_decision_from_url(
        url=url,
        extracted_features=extracted,
        base_probability=base_prob,
    )
    prob = float(decision["probability"])
    heuristic_prob = float(decision["heuristic_probability"])

    if debug:
        print(
            f"DEBUG -> base_prob={base_prob}, heuristic_prob={heuristic_prob}, "
            f"prob={prob}, model_threshold={model_threshold}, "
            f"effective_phishing_min_probability="
            f"{decision['effective_phishing_min_probability']}, "
            f"hard_rule_phishing={decision['hard_rule_phishing']}, "
            f"risk_band={decision['risk_band']}, "
            f"risk_level={decision['risk_level']}, "
            f"confidence_score={decision['confidence_score']}"
        )

    result["url"] = url
    result["prediction"] = decision["prediction"]
    result["risk_band"] = decision["risk_band"]
    result["confidence"] = decision["confidence"]
    result["confidence_score"] = decision["confidence_score"]
    result["base_model_probability"] = round(base_prob, 6)
    result["heuristic_probability"] = round(heuristic_prob, 6)
    result["risk_score"] = round(prob * 100.0, 2)
    result["risk_score_sigmoid"] = round(_sigmoid_risk_score(prob), 2)
    result["risk_level"] = decision["risk_level"]
    result["probability"] = round(prob, 6)
    result["model_threshold"] = round(model_threshold, 6)
    result["effective_phishing_min_probability"] = round(
        float(decision["effective_phishing_min_probability"]),
        6,
    )
    result["risk_bands"] = _risk_band_config()
    result["hard_rule_phishing"] = bool(decision["hard_rule_phishing"])
    result["dual_threshold_score"] = round(prob, 6)
    result["dual_threshold_label"] = decision["prediction"]
    result["dual_threshold_confidence"] = decision["confidence"]
    result["reasons"] = decision["reasons"]
    result["reason_details"] = decision["reason_details"]
    # NOTE: "threshold" is not a key in the predict_from_features result dict;
    # the key is "model_threshold". The original pop() was a no-op and is removed.
    return result


def run_example_predictions(
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    urls: Optional[Iterable[str]] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Run example predictions for smoke-testing and demo output."""
    return [
        predict_url(
            url,
            bundle_or_path=bundle_or_path,
            save_explanation_plot=index == 0,
            debug=debug,
        )
        for index, url in enumerate(list(urls or DEFAULT_EXAMPLE_URLS))
    ]


def run_sanity_tests(
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Run targeted URL sanity checks for basic phishing/legitimate behavior."""
    cases = [
        {
            "url": "http://192.168.1.1/login",
            "expected": "suspicious_or_phishing",
            "accept": {"suspicious", "phishing"},
        },
        {
            "url": "https://google.com",
            "expected": "legitimate",
            "accept": {"legitimate"},
        },
        {
            "url": "https://secure-amazon-login.com",
            "expected": "suspicious_or_phishing",
            "accept": {"suspicious", "phishing"},
        },
    ]

    results: List[Dict[str, Any]] = []
    for case in cases:
        output = predict_url(case["url"], bundle_or_path=bundle_or_path, debug=debug)
        observed = output["risk_band"]
        passed = observed in case["accept"]
        results.append(
            {
                "url": case["url"],
                "expected": case["expected"],
                "observed": observed,
                "prediction": output["prediction"],
                "probability": output["probability"],
                "model_threshold": output["model_threshold"],
                "risk_bands": output["risk_bands"],
                "risk_score": output["risk_score"],
                "passed": passed,
            }
        )
    return results


def _serializable_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop fitted estimator objects before writing JSON summaries."""
    serializable = {
        "model_name": result["model_name"],
        "threshold": _safe_float(result["threshold"], 6),
        "metrics": result["metrics"],
        "artifacts": result["artifacts"],
    }
    if "full_system_evaluation" in result:
        serializable["full_system_evaluation"] = result["full_system_evaluation"]
    return serializable


def evaluate_full_system_on_features(
    base_probabilities: Union[pd.Series, np.ndarray, List[float]],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate the deployed risk engine on the held-out feature dataset."""
    # FIX: validate that lengths match before zipping to prevent silent truncation.
    base_prob_list = list(base_probabilities)
    if len(base_prob_list) != len(x_test):
        raise ValueError(
            f"base_probabilities length ({len(base_prob_list)}) must match "
            f"x_test row count ({len(x_test)})."
        )

    probabilities: List[float] = []
    heuristic_probabilities: List[float] = []
    risk_bands: List[str] = []
    effective_thresholds: List[float] = []

    for base_probability, (_, row) in zip(base_prob_list, x_test.iterrows()):
        decision = _hybrid_decision_from_feature_row(
            base_probability=float(base_probability),
            extracted_features=row.to_dict(),
        )
        probabilities.append(float(decision["probability"]))
        heuristic_probabilities.append(float(decision["heuristic_probability"]))
        risk_bands.append(str(decision["risk_band"]))
        effective_thresholds.append(float(decision["effective_phishing_min_probability"]))

    alert_predictions = np.asarray([band in {"suspicious", "phishing"} for band in risk_bands], dtype=int)
    phishing_predictions = np.asarray([band == "phishing" for band in risk_bands], dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)

    return {
        "status": "proxy_only",
        "is_full_system_evaluation": False,
        "evaluation_mode": "hybrid_proxy_diagnostic",
        "evaluation_scope": "held_out_feature_rows",
        "note": (
            "The source dataset has engineered features, not raw URLs. True "
            "hybrid_system(url) holdout metrics require a labeled raw-URL test "
            "set; the proxy below is diagnostic only and excludes URL-only "
            "brand/TLD/login hard rules."
        ),
        "feature_proxy_evaluation": {
            "risk_band_counts": {
                "legitimate": int(sum(band == "legitimate" for band in risk_bands)),
                "suspicious": int(sum(band == "suspicious" for band in risk_bands)),
                "phishing": int(sum(band == "phishing" for band in risk_bands)),
            },
            "average_hybrid_probability": _safe_float(np.mean(probability_array), 6),
            "average_heuristic_probability": _safe_float(np.mean(heuristic_probabilities), 6),
            "adaptive_threshold_count": int(
                sum(
                    threshold == ADAPTIVE_PHISHING_MIN_PROBABILITY
                    for threshold in effective_thresholds
                )
            ),
            "alert_metrics": _compute_metrics_from_predictions(
                y_true=y_test,
                predictions=alert_predictions,
                probabilities=probability_array,
                threshold=LEGITIMATE_MAX_PROBABILITY,
            ),
            "phishing_metrics": _compute_metrics_from_predictions(
                y_true=y_test,
                predictions=phishing_predictions,
                probabilities=probability_array,
                threshold=PHISHING_MIN_PROBABILITY,
            ),
        },
    }


def evaluate_full_system_on_urls(
    urls: Iterable[str],
    y_true: Union[pd.Series, np.ndarray, List[int]],
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Evaluate the actual deployed URL pipeline on labeled raw URLs."""
    bundle = load_model_bundle(bundle_or_path)
    feature_names = list(bundle["feature_names"])
    calibrated_model = bundle["model"]
    url_list = list(urls)
    y_true_array = np.asarray(list(y_true), dtype=int)
    if set(np.unique(y_true_array)).issubset({-1, 1}):
        y_true_array = (y_true_array == -1).astype(int)
    if len(url_list) != len(y_true_array):
        raise ValueError("URL evaluation requires the same number of URLs and labels.")

    probabilities: List[float] = []
    risk_bands: List[str] = []
    for url in url_list:
        extracted = extract_url_features(url)
        aligned = preprocess_single(extracted, feature_names)
        base_probability = float(calibrated_model.predict_proba(aligned)[0][1])
        decision = _hybrid_decision_from_url(
            url=url,
            extracted_features=extracted,
            base_probability=base_probability,
        )
        probabilities.append(float(decision["probability"]))
        risk_bands.append(str(decision["risk_band"]))

    probability_array = np.asarray(probabilities, dtype=float)
    alert_predictions = np.asarray([band in {"suspicious", "phishing"} for band in risk_bands], dtype=int)
    phishing_predictions = np.asarray([band == "phishing" for band in risk_bands], dtype=int)

    return {
        "status": "available",
        "is_full_system_evaluation": True,
        "evaluation_mode": "hybrid_full_url_pipeline",
        "evaluation_scope": "labeled_raw_urls",
        "risk_band_counts": {
            "legitimate": int(sum(band == "legitimate" for band in risk_bands)),
            "suspicious": int(sum(band == "suspicious" for band in risk_bands)),
            "phishing": int(sum(band == "phishing" for band in risk_bands)),
        },
        "alert_metrics": _compute_metrics_from_predictions(
            y_true=y_true_array,
            predictions=alert_predictions,
            probabilities=probability_array,
            threshold=LEGITIMATE_MAX_PROBABILITY,
        ),
        "phishing_metrics": _compute_metrics_from_predictions(
            y_true=y_true_array,
            predictions=phishing_predictions,
            probabilities=probability_array,
            threshold=PHISHING_MIN_PROBABILITY,
        ),
    }


def _normalize_binary_labels(labels: Iterable[Any]) -> np.ndarray:
    """Normalize common binary label formats to 0=legitimate, 1=phishing."""
    series = pd.Series(list(labels))
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique_values = set(numeric.astype(int).unique())
        if unique_values.issubset({0, 1}):
            return numeric.astype(int).to_numpy()
        if unique_values.issubset({-1, 1}):
            return (numeric.astype(int) == -1).astype(int).to_numpy()
        raise ValueError("Numeric URL-evaluation labels must be 0/1 or -1/1.")

    label_map = {
        "benign": 0,
        "clean": 0,
        "good": 0,
        "legit": 0,
        "legitimate": 0,
        "safe": 0,
        "bad": 1,
        "malicious": 1,
        "phish": 1,
        "phishing": 1,
        "suspicious": 1,
        "unsafe": 1,
    }
    normalized: List[int] = []
    for value in series:
        key = str(value).strip().lower()
        if key not in label_map:
            raise ValueError(f"Unsupported URL-evaluation label: {value!r}")
        normalized.append(label_map[key])
    return np.asarray(normalized, dtype=int)


def evaluate_full_system_from_csv(
    csv_path: Union[str, Path],
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    url_column: str = "url",
    label_column: str = "label",
) -> Dict[str, Any]:
    """Evaluate the deployed URL pipeline from a labeled URL CSV file."""
    csv_path = Path(csv_path)
    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in (url_column, label_column) if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"URL evaluation CSV is missing required column(s): {', '.join(missing_columns)}"
        )

    urls = frame[url_column].dropna().astype(str)
    labels = _normalize_binary_labels(frame.loc[urls.index, label_column])
    evaluation = evaluate_full_system_on_urls(
        urls=urls.tolist(),
        y_true=labels,
        bundle_or_path=bundle_or_path,
    )
    evaluation["dataset_path"] = str(csv_path)
    evaluation["row_count"] = int(len(urls))
    evaluation["url_column"] = url_column
    evaluation["label_column"] = label_column
    return evaluation


def train_models(
    preprocessed: Mapping[str, Any],
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    max_fpr: float = DEFAULT_MAX_FPR,
) -> Dict[str, Any]:
    """Train all models, calibrate the best tree model, and save artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_catalog = _build_model_catalog()

    base_results: Dict[str, Dict[str, Any]] = {}
    for name, meta in model_catalog.items():
        base_results[name] = evaluate_model(
            estimator=meta["estimator"],
            x_train=preprocessed["x_train"],
            x_test=preprocessed["x_test"],
            y_train=preprocessed["y_train"],
            y_test=preprocessed["y_test"],
            model_name=name,
            output_dir=output_dir,
            max_fpr=max_fpr,
        )

    tree_names = _tree_model_names(model_catalog)
    best_tree_name = _select_best_tree_model({name: base_results[name] for name in tree_names})
    calibrated_template = CalibratedClassifierCV(
        estimator=clone(model_catalog[best_tree_name]["estimator"]),
        method="sigmoid",
        cv=_cv_strategy(preprocessed["y_train"], max_splits=CALIBRATION_SPLITS),
    )
    calibrated_oof = _predict_oof_probabilities(
        calibrated_template,
        preprocessed["x_train"],
        preprocessed["y_train"],
        cv_splits=CALIBRATION_SPLITS,
    )
    calibrated_curve = _roc_threshold_frame(preprocessed["y_train"], calibrated_oof)
    calibrated_threshold = find_best_threshold(
        preprocessed["y_train"],
        calibrated_oof,
        max_fpr=max_fpr,
    )

    calibrated_model = calibrate_model(
        estimator=model_catalog[best_tree_name]["estimator"],
        x_train=preprocessed["x_train"],
        y_train=preprocessed["y_train"],
        cv_splits=CALIBRATION_SPLITS,
    )
    calibrated_probabilities = calibrated_model.predict_proba(preprocessed["x_test"])[:, 1]
    deployed_metrics = _compute_metrics(
        preprocessed["y_test"],
        calibrated_probabilities,
        calibrated_threshold,
    )
    full_system_evaluation = evaluate_full_system_on_features(
        base_probabilities=calibrated_probabilities,
        x_test=preprocessed["x_test"],
        y_test=preprocessed["y_test"],
    )

    deployed_name = f"{best_tree_name}_calibrated"
    deployed_artifacts = {
        "confusion_matrix": str(
            _plot_confusion_matrix(
                preprocessed["y_test"],
                calibrated_probabilities,
                calibrated_threshold,
                deployed_name,
                output_dir,
            )
        ),
        "precision_recall_curve": str(
            _plot_precision_recall_curve(
                preprocessed["y_test"],
                calibrated_probabilities,
                calibrated_threshold,
                deployed_name,
                output_dir,
            )
        ),
        "roc_curve": str(
            _plot_roc_curve(
                preprocessed["y_test"],
                calibrated_probabilities,
                calibrated_threshold,
                calibrated_curve,
                deployed_name,
                output_dir,
                max_fpr=max_fpr,
            )
        ),
    }
    calibrated_curve_path = output_dir / f"{deployed_name}_roc_thresholds.csv"
    calibrated_curve.to_csv(calibrated_curve_path, index=False)
    deployed_artifacts["threshold_curve_csv"] = str(calibrated_curve_path)

    explainability_model = clone(model_catalog[best_tree_name]["estimator"])
    explainability_model.fit(preprocessed["x_train"], preprocessed["y_train"])
    feature_importance_path = _plot_feature_importance(
        explainability_model,
        preprocessed["feature_names"],
        deployed_name,
        output_dir,
    )
    shap_summary_path = _plot_shap_summary(
        explainability_model,
        preprocessed["x_train"],
        deployed_name,
        output_dir,
    )
    if feature_importance_path is not None:
        deployed_artifacts["feature_importance"] = str(feature_importance_path)
    if shap_summary_path is not None:
        deployed_artifacts["shap_summary"] = str(shap_summary_path)

    all_results = dict(base_results)
    all_results[deployed_name] = {
        "model_name": deployed_name,
        "model": calibrated_model,
        "threshold": float(calibrated_threshold),
        "metrics": deployed_metrics,
        "full_system_evaluation": full_system_evaluation,
        "artifacts": deployed_artifacts,
    }

    comparison = pd.DataFrame(
        {
            "model_name": name,
            "threshold": result["threshold"],
            "accuracy": result["metrics"]["accuracy"],
            "precision": result["metrics"]["precision"],
            "recall": result["metrics"]["recall"],
            "f1_score": result["metrics"]["f1_score"],
            "false_positives": result["metrics"]["false_positives"],
            "false_positive_rate": result["metrics"]["false_positive_rate"],
            "roc_auc": result["metrics"]["roc_auc"],
        }
        for name, result in all_results.items()
    ).sort_values(
        by=["precision", "false_positives", "recall", "f1_score", "accuracy"],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)

    return {
        "all_results": all_results,
        "comparison": comparison,
        "selected_model_name": deployed_name,
        "selected_model": calibrated_model,
        "selected_threshold": float(calibrated_threshold),
        "selected_metrics": deployed_metrics,
        "full_system_evaluation": full_system_evaluation,
        "explainability_model": explainability_model,
        "artifacts": deployed_artifacts,
    }


def run_training_pipeline(
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
    model_output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    max_fpr: float = DEFAULT_MAX_FPR,
    url_eval_csv_path: Optional[Union[str, Path]] = None,
    url_column: str = "url",
    label_column: str = "label",
    debug: bool = False,
) -> Dict[str, Any]:
    """Execute the full training workflow and persist the best calibrated model."""
    _set_random_seed()

    dataset, dataset_path = load_data(csv_path=csv_path, arff_path=arff_path)
    preprocessed = preprocess_data(dataset, test_size=test_size, random_state=RANDOM_STATE)
    training_output = train_models(preprocessed, output_dir=output_dir, max_fpr=max_fpr)

    model_output_path = Path(model_output_path)
    output_dir = Path(output_dir)
    _ensure_parent_dirs(model_output_path)

    # Use the feature-proxy evaluation as the default; may be replaced below.
    full_system_evaluation = training_output["full_system_evaluation"]

    bundle = {
        "model_name": training_output["selected_model_name"],
        "model": training_output["selected_model"],
        "threshold": training_output["selected_threshold"],
        "feature_names": list(preprocessed["feature_names"]),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "artifacts_directory": str(output_dir),
        "dataset_path": str(dataset_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_mapping": {"0": "legitimate", "1": "phishing"},
        "metrics": training_output["selected_metrics"],
        "full_system_evaluation": full_system_evaluation,
        "comparison": training_output["comparison"].to_dict(orient="records"),
        "explainability_model": training_output["explainability_model"],
    }

    if url_eval_csv_path is not None:
        # FIX: capture the url-eval result and propagate it to both the bundle and the
        # summary dict so the JSON output reflects the true full-system evaluation.
        full_system_evaluation = evaluate_full_system_from_csv(
            csv_path=url_eval_csv_path,
            bundle_or_path=bundle,
            url_column=url_column,
            label_column=label_column,
        )
        bundle["full_system_evaluation"] = full_system_evaluation

    evaluation_modes = {
        "ml_only": {
            "status": "available",
            "evaluation_mode": "ml_only_holdout",
            "evaluation_scope": "held_out_feature_rows",
            "metrics": training_output["selected_metrics"],
        },
        "hybrid": full_system_evaluation,
    }
    bundle["evaluation_modes"] = evaluation_modes

    joblib.dump(bundle, model_output_path)

    comparison_path = output_dir / "model_comparison.csv"
    training_output["comparison"].to_csv(comparison_path, index=False)

    summary = {
        "dataset_path": str(dataset_path),
        "dataset_rows": int(len(dataset)),
        "class_distribution": {
            "legitimate": int((preprocessed["target"] == 0).sum()),
            "phishing": int((preprocessed["target"] == 1).sum()),
        },
        "feature_count": len(preprocessed["feature_names"]),
        "feature_names": list(preprocessed["feature_names"]),
        "missing_values_before_imputation": int(preprocessed["missing_values_before_imputation"]),
        "selected_model": training_output["selected_model_name"],
        "selected_threshold": _safe_float(training_output["selected_threshold"], 6),
        "selected_metrics": training_output["selected_metrics"],
        # FIX: use the (possibly url-eval-updated) full_system_evaluation variable so
        # the JSON summary always reflects the most complete evaluation available.
        "full_system_evaluation": full_system_evaluation,
        "evaluation_modes": evaluation_modes,
        "comparison": training_output["comparison"].to_dict(orient="records"),
        "models": {
            name: _serializable_result(result)
            for name, result in training_output["all_results"].items()
        },
        "model_output_path": str(model_output_path),
        "artifacts_directory": str(output_dir),
        "xgboost_available": bool(XGBOOST_AVAILABLE),
    }

    example_predictions = run_example_predictions(bundle_or_path=bundle, debug=debug)
    summary["example_predictions"] = example_predictions
    summary["sanity_tests"] = run_sanity_tests(bundle_or_path=bundle, debug=debug)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def get_model_metadata(bundle_or_path: BundleLike = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Return bundle metadata for API health checks."""
    bundle = load_model_bundle(bundle_or_path)
    metadata = {
        "model_name": bundle["model_name"],
        "threshold": _safe_float(bundle["threshold"], 6),
        "feature_names": list(bundle["feature_names"]),
        "feature_count": len(bundle["feature_names"]),
        "metrics": bundle["metrics"],
        "created_at_utc": bundle["created_at_utc"],
        "dataset_path": bundle["dataset_path"],
    }
    if "full_system_evaluation" in bundle:
        metadata["f ull_system_evaluation"] = bundle["full_system_evaluation"]
    if "evaluation_modes" in bundle:
        metadata["evaluation_modes"] = bundle["evaluation_modes"]
    return metadata
