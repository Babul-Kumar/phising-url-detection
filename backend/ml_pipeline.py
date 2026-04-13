"""
Reusable phishing website training and inference pipeline.

This module converts the bundled ARFF dataset to CSV, trains multiple models,
optimizes the decision threshold for high precision, saves artifacts, and
provides helpers for website prediction and cyber risk scoring.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

# matplotlib backend must be set before pyplot is imported
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

sys.dont_write_bytecode = True


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
DEFAULT_MIN_PRECISION = 0.95

SUSPICIOUS_KEYWORDS = {
    "account",
    "alert",
    "bank",
    "billing",
    "bonus",
    "confirm",
    "credential",
    "free",
    "login",
    "mfa",
    "password",
    "pay",
    "paypal",
    "recover",
    "reset",
    "secure",
    "session",
    "signin",
    "suspend",
    "unlock",
    "update",
    "urgent",
    "verify",
    "wallet",
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

FEATURE_DESCRIPTIONS = {
    "having_IP_Address": "URL uses a raw IP address instead of a domain",
    "URL_Length": "Short, suspicious, or very long URL length bucket",
    "Shortining_Service": "URL uses a shortening service",
    "having_At_Symbol": "URL contains '@'",
    "double_slash_redirecting": "URL contains an extra redirect-like double slash",
    "Prefix_Suffix": "Domain contains a hyphen",
    "having_Sub_Domain": "Suspicious number of subdomains",
    "SSLfinal_State": "SSL/HTTPS quality signal",
    "Domain_registeration_length": "Estimated domain registration horizon",
    "Favicon": "Favicon loading trust signal",
    "port": "URL contains a non-standard port",
    "HTTPS_token": "The token 'https' appears in the domain/path",
    "Request_URL": "External request pattern signal",
    "URL_of_Anchor": "Anchor-link trust signal",
    "Links_in_tags": "External links inside HTML tags signal",
    "SFH": "Server form handler safety signal",
    "Submitting_to_email": "Suspicious email submission signal",
    "Abnormal_URL": "Abnormal URL pattern signal",
    "Redirect": "Redirect behavior signal",
    "on_mouseover": "Mouseover JavaScript signal",
    "RightClick": "Right-click blocking signal",
    "popUpWidnow": "Pop-up window signal",
    "Iframe": "Iframe embedding signal",
    "age_of_domain": "Estimated domain age signal",
    "DNSRecord": "DNS availability signal",
    "web_traffic": "Estimated web traffic signal",
    "Page_Rank": "Estimated popularity/PageRank signal",
    "Google_Index": "Estimated search-engine indexing signal",
    "Links_pointing_to_page": "Estimated backlink count signal",
    "Statistical_report": "Statistical blacklist/report signal",
}

DEFAULT_EXAMPLE_URLS = [
    "https://secure-paypal-login.verify-account-update.xyz/session/confirm",
    "https://www.microsoft.com/security",
    "http://185.23.54.11/bank/login.php?redirect=secure",
]

BundleLike = Union[str, Path, Mapping[str, Any]]


def _ensure_parent_dirs(*paths: Path) -> None:
    """Create the parent directories required by the supplied paths."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _safe_float(value: Any, digits: int = 4) -> float:
    """Convert numpy and pandas scalar types to builtin floats."""
    return round(float(value), digits)


def _root_domain(hostname: str) -> str:
    """Return the registrable-looking root domain from a hostname."""
    hostname = hostname.lower().strip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def _normalize_url(url: str) -> str:
    """Add a default scheme so urlparse can extract the hostname reliably."""
    cleaned = str(url).strip()
    if not cleaned:
        raise ValueError("A non-empty URL is required.")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = f"http://{cleaned}"
    return cleaned


def _is_ip_address(hostname: str) -> bool:
    """Return True when the hostname looks like an IPv4 literal."""
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname.strip().lower()))


def _subdomain_count_from_hostname(hostname: str) -> int:
    """Estimate the number of meaningful subdomains from a hostname."""
    hostname = hostname.strip().lower().strip(".")
    if not hostname or _is_ip_address(hostname):
        return 0
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return max(len(hostname.split(".")) - 2, 0)


def convert_arff_to_csv(
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
) -> Path:
    """
    Convert a simple numeric ARFF file into CSV.

    The bundled dataset uses only numeric nominal values, so a lightweight parser
    is sufficient and avoids introducing an extra dependency.
    """
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
                        "Encountered a row with an unexpected number of columns "
                        f"({len(values)} vs {len(attributes)})."
                    )
                rows.append([np.nan if value == "?" else value for value in values])

    if not attributes or not rows:
        raise ValueError(f"No usable data was found in {arff_path}")

    dataset = pd.DataFrame(rows, columns=attributes)
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset.to_csv(csv_path, index=False)
    return csv_path


def load_dataset(
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
) -> Tuple[pd.DataFrame, Path]:
    """Load the phishing dataset from CSV, converting from ARFF when needed."""
    csv_path = Path(csv_path)
    arff_path = Path(arff_path)

    if not csv_path.exists():
        csv_path = convert_arff_to_csv(arff_path=arff_path, csv_path=csv_path)

    dataset = pd.read_csv(csv_path)
    dataset = dataset.replace("?", np.nan)
    dataset = dataset.apply(pd.to_numeric, errors="coerce")

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Dataset must include a '{TARGET_COLUMN}' target column. "
            f"Found columns: {list(dataset.columns)}"
        )

    return dataset, csv_path


def prepare_training_data(
    dataset: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, List[str], int]:
    """
    Split raw dataframe into features and target.

    In this dataset, `Result = -1` denotes phishing and `Result = 1`
    denotes legitimate. The pipeline maps phishing to the positive class.
    """
    features = dataset.drop(columns=[target_column]).copy()
    target = (dataset[target_column].astype(int) == -1).astype(int)
    missing_values = int(features.isna().sum().sum())
    return features, target, list(features.columns), missing_values


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create an 80/20 stratified train-test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=random_state,
    )


def _build_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Build a preprocessing + classifier pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_model_catalog(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Create the model zoo required for training."""
    models: Dict[str, Pipeline] = {
        "logistic_regression": _build_pipeline(
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=random_state,
            )
        ),
        "random_forest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=random_state,
            )
        ),
    }

    if XGBOOST_AVAILABLE and XGBClassifier is not None:
        models["xgboost"] = _build_pipeline(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=1,
            )
        )

    return models


def _calculate_fbeta(
    precision: pd.Series,
    recall: pd.Series,
    beta: float,
) -> pd.Series:
    """Compute the F-beta score safely for a precision/recall series."""
    beta_sq = beta**2
    denominator = (beta_sq * precision) + recall
    denominator = denominator.replace(0, np.nan)
    return ((1 + beta_sq) * precision * recall / denominator).fillna(0.0)


def tune_decision_threshold(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    min_precision: float = DEFAULT_MIN_PRECISION,
    cv_splits: int = 5,
) -> Tuple[float, pd.DataFrame]:
    """
    Tune the classification threshold from out-of-fold predictions.

    The selection rule prioritizes high precision first, then picks the threshold
    with the highest recall among thresholds that satisfy the minimum precision.
    If no threshold reaches the target precision, a precision-focused F0.5 score
    is used as the fallback.
    """
    min_class_count = int(y_train.value_counts().min())
    n_splits = max(2, min(cv_splits, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    oof_probabilities = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_train, oof_probabilities)
    if thresholds.size == 0:
        fallback = pd.DataFrame(
            {
                "threshold": [0.5],
                "precision": [
                    precision_score(
                        y_train,
                        (oof_probabilities >= 0.5).astype(int),
                        zero_division=0,
                    )
                ],
                "recall": [
                    recall_score(
                        y_train,
                        (oof_probabilities >= 0.5).astype(int),
                        zero_division=0,
                    )
                ],
            }
        )
        fallback["f1"] = _calculate_fbeta(fallback["precision"], fallback["recall"], beta=1.0)
        fallback["f0_5"] = _calculate_fbeta(fallback["precision"], fallback["recall"], beta=0.5)
        return 0.5, fallback

    tradeoff = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    tradeoff["f1"] = _calculate_fbeta(tradeoff["precision"], tradeoff["recall"], beta=1.0)
    tradeoff["f0_5"] = _calculate_fbeta(tradeoff["precision"], tradeoff["recall"], beta=0.5)

    precision_first = tradeoff[tradeoff["precision"] >= min_precision]
    if not precision_first.empty:
        chosen_row = precision_first.sort_values(
            by=["recall", "precision", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        chosen_row = tradeoff.sort_values(
            by=["precision", "f0_5", "recall", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]

    return float(chosen_row["threshold"]), tradeoff


def find_best_threshold(
    y_true: Union[pd.Series, np.ndarray],
    y_probs: Union[pd.Series, np.ndarray],
    max_fpr: float = 0.03,
) -> float:
    """
    Select threshold under an FPR constraint and maximize recall.

    Falls back to 0.5 when no threshold satisfies the FPR condition.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_threshold = 0.5
    best_recall = 0.0

    for i in range(len(thresholds)):
        if np.isfinite(thresholds[i]) and fpr[i] <= max_fpr and tpr[i] > best_recall:
            best_recall = float(tpr[i])
            best_threshold = float(thresholds[i])

    return float(best_threshold)


def compute_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """Compute all requested classification metrics for a given threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    metrics = {
        "threshold": _safe_float(threshold, digits=6),
        "accuracy": _safe_float(accuracy_score(y_true, predictions)),
        "precision": _safe_float(precision_score(y_true, predictions, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": _safe_float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": _safe_float(average_precision_score(y_true, probabilities)),
        "false_positive_rate": _safe_float(fp / max(fp + tn, 1)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }
    return metrics


def _get_classifier(model: Pipeline) -> BaseEstimator:
    """Return the fitted classifier from a pipeline."""
    return model.named_steps["classifier"]


def plot_precision_recall_tradeoff(
    y_true: pd.Series,
    probabilities: np.ndarray,
    tradeoff: pd.DataFrame,
    threshold: float,
    model_name: str,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
) -> Path:
    """Save a precision-recall tradeoff visualization for one trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_precision_recall_tradeoff.png"

    precision_curve, recall_curve, _ = precision_recall_curve(y_true, probabilities)
    predicted_labels = (probabilities >= threshold).astype(int)
    operating_precision = precision_score(y_true, predicted_labels, zero_division=0)
    operating_recall = recall_score(y_true, predicted_labels, zero_division=0)
    average_precision = average_precision_score(y_true, probabilities)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(recall_curve, precision_curve, color="#1f77b4", linewidth=2)
    axes[0].scatter(
        operating_recall,
        operating_precision,
        color="#d62728",
        s=80,
        label=f"Selected threshold = {threshold:.3f}",
    )
    axes[0].set_title(f"{model_name} Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1.01)
    axes[0].set_ylim(0, 1.01)
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].text(
        0.03,
        0.05,
        f"Average precision: {average_precision:.3f}",
        transform=axes[0].transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    axes[1].plot(
        tradeoff["threshold"],
        tradeoff["precision"],
        label="Precision",
        color="#2ca02c",
        linewidth=2,
    )
    axes[1].plot(
        tradeoff["threshold"],
        tradeoff["recall"],
        label="Recall",
        color="#ff7f0e",
        linewidth=2,
    )
    axes[1].axvline(
        threshold,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )
    axes[1].set_title(f"{model_name} Threshold Tradeoff")
    axes[1].set_xlabel("Classification threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.01)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    model_name: str,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    top_n: int = 15,
) -> Optional[Path]:
    """Save a feature-importance chart for tree models or LR coefficients."""
    classifier = _get_classifier(model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_feature_importance.png"

    signed_values: Optional[np.ndarray]
    if hasattr(classifier, "feature_importances_"):
        signed_values = None
        importance = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        signed_values = np.asarray(classifier.coef_).ravel()
        importance = np.abs(signed_values)
    else:
        return None

    n_features = min(top_n, len(feature_names))
    order = np.argsort(importance)[-n_features:]
    selected_names = np.asarray(feature_names)[order]
    selected_scores = importance[order]

    if signed_values is not None:
        signed_subset = signed_values[order]
        colors = ["#d62728" if value > 0 else "#1f77b4" for value in signed_subset]
        plot_values = signed_subset
        xlabel = "Coefficient value (positive pushes toward phishing)"
    else:
        colors = ["#1f77b4"] * len(selected_scores)
        plot_values = selected_scores
        xlabel = "Feature importance"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(selected_names, plot_values, color=colors)
    ax.set_title(f"{model_name} Feature Importance")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix_artifact(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
) -> Path:
    """Save the confusion matrix for the current operating threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_confusion_matrix.png"

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Phishing"],
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def train_and_evaluate_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    min_precision: float = DEFAULT_MIN_PRECISION,
) -> Dict[str, Dict[str, Any]]:
    """Train all requested models and collect metrics + artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for model_name, model in build_model_catalog().items():
        min_class_count = int(y_train.value_counts().min())
        n_splits = max(2, min(5, min_class_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        oof_probabilities = cross_val_predict(
            model,
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        # Use the FPR-constrained threshold for final evaluation.
        threshold = find_best_threshold(y_train, oof_probabilities, max_fpr=0.03)

        # Produce the precision/recall tradeoff table for diagnostic reporting.
        _, tradeoff = tune_decision_threshold(
            model=model,
            x_train=x_train,
            y_train=y_train,
            min_precision=min_precision,
        )

        fitted_model = model.fit(x_train, y_train)
        test_probabilities = fitted_model.predict_proba(x_test)[:, 1]

        tuned_metrics = compute_metrics(y_test, test_probabilities, threshold)
        default_metrics = compute_metrics(y_test, test_probabilities, threshold=0.5)

        tradeoff_csv_path = output_dir / f"{model_name}_precision_recall_tradeoff.csv"
        tradeoff.to_csv(tradeoff_csv_path, index=False)

        pr_plot_path = plot_precision_recall_tradeoff(
            y_true=y_test,
            probabilities=test_probabilities,
            tradeoff=tradeoff,
            threshold=threshold,
            model_name=model_name,
            output_dir=output_dir,
        )
        feature_plot_path = plot_feature_importance(
            model=fitted_model,
            feature_names=feature_names,
            model_name=model_name,
            output_dir=output_dir,
        )
        confusion_plot_path = plot_confusion_matrix_artifact(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=threshold,
            model_name=model_name,
            output_dir=output_dir,
        )

        results[model_name] = {
            "model_name": model_name,
            "model": fitted_model,
            "threshold": float(threshold),
            "tuned_metrics": tuned_metrics,
            "default_threshold_metrics": default_metrics,
            "artifacts": {
                "tradeoff_csv": str(tradeoff_csv_path),
                "precision_recall_plot": str(pr_plot_path),
                "feature_importance_plot": str(feature_plot_path) if feature_plot_path else None,
                "confusion_matrix_plot": str(confusion_plot_path),
            },
        }

    return results


def build_model_comparison(results: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten per-model metrics into a comparison dataframe."""
    rows: List[Dict[str, Any]] = []
    for name, result in results.items():
        tuned = result["tuned_metrics"]
        default = result["default_threshold_metrics"]
        rows.append(
            {
                "model_name": name,
                "threshold": result["threshold"],
                "accuracy": tuned["accuracy"],
                "precision": tuned["precision"],
                "recall": tuned["recall"],
                "f1_score": tuned["f1_score"],
                "average_precision": tuned["average_precision"],
                "false_positives": tuned["false_positives"],
                "false_positive_rate": tuned["false_positive_rate"],
                "default_precision": default["precision"],
                "default_recall": default["recall"],
                "default_f1_score": default["f1_score"],
            }
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=["precision", "false_positives", "recall", "f1_score", "accuracy"],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)
    return comparison


def _serializable_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop non-serializable model objects before writing JSON output."""
    return {
        "model_name": result["model_name"],
        "threshold": _safe_float(result["threshold"], digits=6),
        "tuned_metrics": result["tuned_metrics"],
        "default_threshold_metrics": result["default_threshold_metrics"],
        "artifacts": result["artifacts"],
    }


def save_training_outputs(
    results: Mapping[str, Mapping[str, Any]],
    comparison: pd.DataFrame,
    feature_names: List[str],
    dataset_path: Union[str, Path],
    dataset_rows: int,
    missing_values_before_imputation: int,
    model_output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
) -> Dict[str, Any]:
    """Persist the best model bundle and the supporting training summary."""
    model_output_path = Path(model_output_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_parent_dirs(model_output_path)

    best_model_name = str(comparison.iloc[0]["model_name"])
    best_result = results[best_model_name]

    bundle = {
        "model_name": best_model_name,
        "model": best_result["model"],
        "threshold": float(best_result["threshold"]),
        "feature_names": feature_names,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_mapping": {"0": "legitimate", "1": "phishing"},
        "dataset_path": str(dataset_path),
        "metrics": best_result["tuned_metrics"],
        "default_threshold_metrics": best_result["default_threshold_metrics"],
        "comparison": comparison.to_dict(orient="records"),
        "feature_descriptions": FEATURE_DESCRIPTIONS,
    }
    joblib.dump(bundle, model_output_path)

    summary = {
        "dataset_path": str(dataset_path),
        "dataset_rows": int(dataset_rows),
        "feature_count": len(feature_names),
        "missing_values_before_imputation": int(missing_values_before_imputation),
        "selected_model": best_model_name,
        "selected_threshold": _safe_float(best_result["threshold"], digits=6),
        "selected_metrics": best_result["tuned_metrics"],
        "comparison": comparison.to_dict(orient="records"),
        "models": {name: _serializable_result(result) for name, result in results.items()},
        "model_output_path": str(model_output_path),
        "artifacts_directory": str(output_dir),
        "xgboost_available": bool(XGBOOST_AVAILABLE),
    }

    comparison_path = output_dir / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _load_bundle(bundle_or_path: BundleLike = DEFAULT_MODEL_PATH) -> Mapping[str, Any]:
    """Load a trained model bundle from disk or pass an already-loaded bundle through."""
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
            f"Trained model bundle not found at {bundle_path}. "
            "Run train.py first."
        )
    return joblib.load(bundle_path)


def predict_from_features(
    features: Mapping[str, Any],
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    debug: bool = False,
) -> Dict[str, Any]:
    """Predict phishing risk from a feature dictionary."""
    bundle = _load_bundle(bundle_or_path)
    feature_names = list(bundle["feature_names"])
    classifier = bundle["model"]
    threshold = float(bundle["threshold"])

    if hasattr(classifier, "classes_"):
        classes = list(getattr(classifier, "classes_"))
        if classes != [0, 1]:
            raise ValueError(
                f"Unexpected class mapping {classes}. Expected class 0=legitimate, 1=phishing."
            )

    ordered_features = {
        feature_name: features.get(feature_name, np.nan)
        for feature_name in feature_names
    }
    input_frame = pd.DataFrame([ordered_features], columns=feature_names)
    phishing_probability = float(classifier.predict_proba(input_frame)[0, 1])
    if debug:
        print(
            f"DEBUG -> prob={phishing_probability}, threshold={threshold}, "
            f"decision={phishing_probability >= threshold}"
        )
    cyber_risk_score = round(phishing_probability * 100, 2)
    is_phishing = phishing_probability >= threshold
    confidence_score = _confidence_score(phishing_probability)

    return {
        "prediction": "phishing" if is_phishing else "legitimate",
        "is_phishing": bool(is_phishing),
        "confidence": _confidence_from_risk(phishing_probability),
        "confidence_score": confidence_score,
        "phishing_probability": round(phishing_probability, 6),
        "cyber_risk_score": cyber_risk_score,
        "model_threshold": round(threshold, 6),
        "risk_bands": _risk_band_config(),
        "model_name": bundle["model_name"],
        "features": ordered_features,
    }


def simulate_feature_extraction(
    url: str,
    feature_names: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """
    Simulate the legacy phishing-website feature vector from a URL.

    The underlying dataset includes browser-, content-, and popularity-based
    signals that are not directly observable from a raw URL. This helper uses
    conservative heuristics and neutral defaults so that example predictions can
    still be produced from a URL string alone.
    """
    normalized_url = _normalize_url(url)
    parsed = urlparse(normalized_url)
    hostname = (parsed.hostname or "").lower()
    domain = _root_domain(hostname)
    path = parsed.path or ""
    query = parsed.query or ""
    query_params = {key.lower(): value for key, value in parse_qs(query).items()}
    lowered_url = normalized_url.lower()

    subdomains = _subdomain_count_from_hostname(hostname)
    is_ip_address = bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", hostname))
    has_shortener = domain in SHORTENER_DOMAINS
    has_at_symbol = "@" in normalized_url
    extra_double_slash = "//" in (path or "")
    hyphen_in_domain = "-" in hostname
    suspicious_keyword_hits = sum(keyword in lowered_url for keyword in SUSPICIOUS_KEYWORDS)
    nonstandard_port = parsed.port not in (None, 80, 443)
    misplaced_https_token = "https" in hostname.replace("https", "", 1) or "https" in path.lower()
    has_redirect_hint = bool(
        extra_double_slash
        or {"redirect", "redir", "next", "continue", "url", "target"} & set(query_params)
    )
    trusted = domain in TRUSTED_DOMAINS or domain.endswith((".gov", ".edu"))
    risky_tld = any(domain.endswith(tld) for tld in RISKY_TLDS)
    looks_obfuscated = bool(
        is_ip_address
        or has_shortener
        or has_at_symbol
        or suspicious_keyword_hits
        or risky_tld
        or sum(character.isdigit() for character in hostname) >= 4
        or "%" in normalized_url
    )

    features: Dict[str, int] = {
        "having_IP_Address": -1 if is_ip_address else 1,
        "URL_Length": 1 if len(normalized_url) < 54 else 0 if len(normalized_url) <= 75 else -1,
        "Shortining_Service": 1 if has_shortener else -1,
        "having_At_Symbol": -1 if has_at_symbol else 1,
        "double_slash_redirecting": 1 if extra_double_slash else -1,
        "Prefix_Suffix": -1 if hyphen_in_domain else 1,
        "having_Sub_Domain": 1 if subdomains <= 1 else -1 if subdomains == 2 else 0,
        "SSLfinal_State": 1 if parsed.scheme == "https" and not looks_obfuscated else 0 if parsed.scheme == "https" else -1,
        "Domain_registeration_length": 1 if looks_obfuscated and not trusted else -1,
        "Favicon": -1 if looks_obfuscated and suspicious_keyword_hits >= 2 else 1,
        "port": -1 if nonstandard_port else 1,
        "HTTPS_token": 1 if misplaced_https_token else -1,
        "Request_URL": -1 if looks_obfuscated else 1,
        "URL_of_Anchor": -1 if suspicious_keyword_hits >= 2 else 0 if looks_obfuscated else 1,
        "Links_in_tags": -1 if looks_obfuscated else 0 if has_redirect_hint else 1,
        "SFH": -1 if has_redirect_hint and suspicious_keyword_hits >= 1 else 0 if has_redirect_hint else 1,
        "Submitting_to_email": -1 if "mailto:" in lowered_url or "email" in query.lower() else 1,
        "Abnormal_URL": 1 if is_ip_address or has_shortener or has_at_symbol else -1,
        "Redirect": 1 if has_redirect_hint else 0,
        "on_mouseover": -1 if "mouseover" in lowered_url else 1,
        "RightClick": -1 if "rightclick" in lowered_url else 1,
        "popUpWidnow": -1 if "popup" in lowered_url else 1,
        # FIX: Iframe in dataset convention: -1 = suspicious (uses iframe/embedding tricks).
        # An iframe token in the URL path is a suspicious signal → return -1.
        # The original code returned 1 when "iframe" was found, which was inverted.
        "Iframe": -1 if "iframe" in lowered_url else 1,
        "age_of_domain": -1 if looks_obfuscated and not trusted else 1,
        "DNSRecord": -1 if is_ip_address else 1,
        "web_traffic": 1 if trusted else -1 if looks_obfuscated else 0,
        "Page_Rank": 1 if trusted else -1 if looks_obfuscated else 1,
        "Google_Index": 1 if trusted else -1 if looks_obfuscated else 1,
        "Links_pointing_to_page": -1 if trusted else 0 if looks_obfuscated else 1,
        "Statistical_report": -1 if is_ip_address or risky_tld else 1,
    }

    if feature_names is not None:
        selected = list(feature_names)
        return {name: int(features.get(name, 0)) for name in selected}
    return {name: int(value) for name, value in features.items()}


def extract_url_features(
    url: str,
    feature_names: Optional[Iterable[str]] = None,
) -> Dict[str, Union[int, float]]:
    """
    Compatibility URL feature extraction.

    Supports both the legacy 30-feature schema and the newer 12 lexical
    URL-feature schema used by the calibrated production model.
    """
    normalized_url = _normalize_url(url)
    parsed = urlparse(normalized_url)
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("The provided URL does not contain a valid hostname.")
    try:
        port_value = parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc
    domain = _root_domain(hostname)
    path = parsed.path or ""
    lowered_url = normalized_url.lower()
    is_ip_address = _is_ip_address(hostname)
    subdomain_count = _subdomain_count_from_hostname(hostname)

    modern_features: Dict[str, Union[int, float]] = {
        "url_length": float(len(normalized_url)),
        "has_ip_address": int(is_ip_address),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "num_dots": float(hostname.count(".")),
        "suspicious_keywords": int(sum(keyword in lowered_url for keyword in SUSPICIOUS_KEYWORDS)),
        "has_hyphen": int(hostname.count("-")),
        "subdomain_count": float(subdomain_count),
        "has_at_symbol": int("@" in normalized_url),
        "has_double_slash": int((normalized_url.split("://", 1)[-1]).count("//")),
        "has_shortener": int(domain in SHORTENER_DOMAINS),
        "has_port": int(port_value not in (None, 80, 443)),
        "contains_https_token": int("https" in hostname),
    }

    legacy_features = simulate_feature_extraction(url)
    combined: Dict[str, Union[int, float]] = {**legacy_features, **modern_features}

    if feature_names is not None:
        selected = list(feature_names)
        return {name: combined.get(name, 0) for name in selected}
    return combined


def preprocess_single(
    features: Mapping[str, Any],
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Prepare one feature dict for model inference."""
    bundle = _load_bundle(bundle_or_path)
    feature_names = list(bundle["feature_names"])
    ordered_features = {feature_name: features.get(feature_name, np.nan) for feature_name in feature_names}
    return pd.DataFrame([ordered_features], columns=feature_names)


def _confidence_score(probability: float) -> float:
    """Measure certainty from distance to the ambiguous 0.5 decision boundary."""
    return round(abs(float(probability) - 0.5) * 2.0, 6)


def _confidence_from_risk(probability: float) -> str:
    """Map a probability into a low / medium / high confidence bucket."""
    score = _confidence_score(probability)
    if score < CONFIDENCE_MEDIUM_MIN:
        return "low"
    if score < CONFIDENCE_HIGH_MIN:
        return "medium"
    return "high"


def _triage_label_from_probability(
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
    return min(max(blended_probability, 0.0), MAX_HYBRID_PROBABILITY)


def _dual_threshold_label(
    probability: float,
    extracted_features: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str, float]:
    """
    Dual-threshold triage output for better operational recall.

    The binary decision still uses the calibrated model probability and saved
    threshold. This triage score only helps surface obvious risky URLs that may
    otherwise fall below a very strict operating threshold.
    """
    dual_score = float(probability)
    if extracted_features is not None:
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
    return (
        _triage_label_from_probability(dual_score),
        _confidence_from_risk(dual_score),
        dual_score,
    )


def _hybrid_heuristic_probability(
    url: str,
    extracted_features: Mapping[str, Any],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Convert lexical heuristics into a bounded 0..1 support signal."""
    score = 0.0
    reason_details: List[Dict[str, Any]] = []

    def add_reason(reason: str, weight: float, signal: str = "risk") -> None:
        nonlocal score
        score += weight
        reason_details.append(_reason_entry(reason, weight, signal=signal))

    url_length = float(extracted_features.get("url_length", 0))
    hyphen_count = int(extracted_features.get("has_hyphen", 0))
    subdomain_count = float(extracted_features.get("subdomain_count", 0))
    num_dots = float(extracted_features.get("num_dots", 0))

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

    normalized = _normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower().strip(".")
    root_domain = _root_domain(hostname)
    url_lower = normalized.lower()
    hostname_tokens = {token for token in re.split(r"[.\-]+", hostname) if token}
    is_trusted_domain = root_domain in TRUSTED_DOMAINS or root_domain.endswith((".gov", ".edu"))
    has_ip_address = bool(int(extracted_features.get("has_ip_address", 0)))
    combo_reasons: List[Tuple[str, float]] = []

    if num_dots > 3 and not is_trusted_domain:
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

    for brand_keyword, trusted_domain in BRAND_DOMAINS.items():
        if brand_keyword in hostname_tokens and root_domain != trusted_domain:
            add_reason(
                f"{brand_keyword} brand impersonation",
                0.40 if (risky_tld or hyphen_count > 0 or subdomain_count > 0) else 0.25,
            )
            if risky_tld:
                combo_reasons.append(("brand impersonation on suspicious domain", 0.30))
            break

    _add_capped_combo_reasons(reason_details, combo_reasons)

    if is_trusted_domain:
        add_reason("trusted domain", -MAX_NEGATIVE_HEURISTIC_ADJUSTMENT, signal="trust")

    return _bounded_heuristic_probability(reason_details), reason_details


def predict_url(
    url: str,
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    debug: bool = False,
) -> Dict[str, Any]:
    """Predict URL risk using weighted ML + heuristic scoring."""
    bundle = _load_bundle(bundle_or_path)

    extracted = extract_url_features(url)
    result = predict_from_features(extracted, bundle, debug=debug)

    # Keep original model output; do not rescale.
    base_prob = float(result["phishing_probability"])
    model_threshold = float(result["model_threshold"])

    heuristic_prob, reason_details = _hybrid_heuristic_probability(url, extracted)
    prob = _fuse_hybrid_probability(base_prob, heuristic_prob)

    normalized_url = _normalize_url(url)
    parsed_url = urlparse(normalized_url)
    hostname = (parsed_url.hostname or "").lower().strip(".")
    root_domain = _root_domain(hostname)
    url_lower = normalized_url.lower()
    ip_login_hard_rule = bool(int(extracted.get("has_ip_address", 0))) and "login" in url_lower
    paypal_hard_rule = "paypal" in hostname and root_domain != BRAND_DOMAINS["paypal"]
    hard_rule_phishing = ip_login_hard_rule or paypal_hard_rule
    if ip_login_hard_rule:
        reason_details.append(_reason_entry("hard rule: IP address login URL", 0.0, signal="rule"))
    if paypal_hard_rule:
        reason_details.append(_reason_entry("hard rule: paypal impersonation", 0.0, signal="rule"))
    if hard_rule_phishing:
        prob = max(prob, HARD_RULE_PHISHING_PROBABILITY)

    has_brand_impersonation = any(
        str(reason["reason"]).endswith("brand impersonation")
        for reason in reason_details
    )
    strong_heuristic_signal = heuristic_prob >= LEGITIMATE_MAX_PROBABILITY
    effective_phishing_min_probability = _effective_phishing_threshold(heuristic_prob)

    risk_band = _triage_label_from_probability(
        prob,
        phishing_min_probability=effective_phishing_min_probability,
    )
    if hard_rule_phishing:
        risk_band = "phishing"
    if risk_band == "legitimate" and (has_brand_impersonation or strong_heuristic_signal):
        risk_band = "suspicious"

    if prob > 0.9:
        risk_level = "critical"
    elif prob > 0.7:
        risk_level = "high"
    elif prob > 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    if risk_band == "suspicious" and risk_level == "low":
        risk_level = "medium"

    confidence_score = _confidence_score(prob)
    confidence = _confidence_from_risk(prob)
    prediction = _display_label(risk_band, confidence_score)
    reasons = _format_reason_details(reason_details)

    if debug:
        print(
            f"DEBUG -> base_prob={base_prob}, heuristic_prob={heuristic_prob}, "
            f"prob={prob}, model_threshold={model_threshold}, "
            f"effective_phishing_min_probability={effective_phishing_min_probability}, "
            f"hard_rule_phishing={hard_rule_phishing}, risk_band={risk_band}, "
            f"confidence_score={confidence_score}"
        )

    return {
        "url": url,
        "prediction": prediction,
        "risk_band": risk_band,
        "risk_score": round(prob * 100.0, 2),
        "risk_level": risk_level,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "probability": round(prob, 6),
        "model_threshold": round(model_threshold, 6),
        "effective_phishing_min_probability": round(effective_phishing_min_probability, 6),
        "risk_bands": _risk_band_config(),
        "model": result["model_name"],
        "reasons": reasons,
        "reason_details": reason_details,
    }


def run_example_predictions(
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    urls: Optional[Iterable[str]] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Run example predictions for demonstration and smoke-testing."""
    candidate_urls = list(urls or DEFAULT_EXAMPLE_URLS)
    return [predict_url(url, bundle_or_path=bundle_or_path, debug=debug) for url in candidate_urls]


def run_sanity_tests(
    bundle_or_path: BundleLike = DEFAULT_MODEL_PATH,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Run quick sanity checks on known URL patterns."""
    cases = [
        ("http://192.168.1.1/login", {"suspicious", "phishing"}),
        ("https://google.com", {"legitimate"}),
        ("https://secure-amazon-login.com", {"suspicious", "phishing"}),
    ]
    results: List[Dict[str, Any]] = []
    for url, expected_labels in cases:
        result = predict_url(url, bundle_or_path=bundle_or_path, debug=debug)
        observed = result["risk_band"]
        results.append(
            {
                "url": url,
                "expected_labels": sorted(expected_labels),
                "observed_label": observed,
                "prediction": result["prediction"],
                "passed": observed in expected_labels,
                "probability": result["probability"],
                "model_threshold": result["model_threshold"],
                "risk_bands": result["risk_bands"],
            }
        )
    return results


def run_training_pipeline(
    csv_path: Union[str, Path] = DEFAULT_CSV_PATH,
    arff_path: Union[str, Path] = DEFAULT_ARFF_PATH,
    model_output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    output_dir: Union[str, Path] = ARTIFACTS_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    min_precision: float = DEFAULT_MIN_PRECISION,
    debug: bool = False,
) -> Dict[str, Any]:
    """Execute the complete ML workflow end to end."""
    dataset, dataset_path = load_dataset(csv_path=csv_path, arff_path=arff_path)
    features, target, feature_names, missing_values = prepare_training_data(dataset)
    x_train, x_test, y_train, y_test = split_dataset(
        features=features,
        target=target,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    results = train_and_evaluate_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        output_dir=output_dir,
        min_precision=min_precision,
    )
    comparison = build_model_comparison(results)
    summary = save_training_outputs(
        results=results,
        comparison=comparison,
        feature_names=feature_names,
        dataset_path=dataset_path,
        dataset_rows=len(dataset),
        missing_values_before_imputation=missing_values,
        model_output_path=model_output_path,
        output_dir=output_dir,
    )

    bundle = _load_bundle(model_output_path)
    summary["class_distribution"] = {
        "legitimate": int((target == 0).sum()),
        "phishing": int((target == 1).sum()),
    }
    summary["example_predictions"] = run_example_predictions(bundle, debug=debug)
    summary["sanity_tests"] = run_sanity_tests(bundle, debug=debug)
    return summary