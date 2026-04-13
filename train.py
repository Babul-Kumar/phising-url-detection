"""
Train a phishing website detector and export the best model bundle.

Usage:
    python train.py
    python train.py --max-fpr 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import pandas as pd

from backend.phishing_pipeline import (
    ARTIFACTS_DIR,
    DEFAULT_ARFF_PATH,
    DEFAULT_CSV_PATH,
    DEFAULT_MAX_FPR,
    DEFAULT_MODEL_PATH,
    run_training_pipeline,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI used to train the phishing detector."""
    parser = argparse.ArgumentParser(
        description="Train a phishing website detection pipeline and export artifacts."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the CSV dataset. If it does not exist, the ARFF file is converted automatically.",
    )
    parser.add_argument(
        "--arff-path",
        type=Path,
        default=DEFAULT_ARFF_PATH,
        help="Fallback ARFF source used when the CSV dataset is missing.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained best-model bundle.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory where plots, CSV summaries, and JSON reports will be saved.",
    )
    parser.add_argument(
        "--max-fpr",
        type=float,
        default=DEFAULT_MAX_FPR,
        help="Maximum allowed false positive rate used when selecting the decision threshold.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Train/test split ratio for the hold-out test set.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logs for example predictions and sanity checks.",
    )
    parser.add_argument(
        "--url-eval-csv",
        type=Path,
        default=None,
        help="Optional labeled raw-URL CSV for true full-system evaluation.",
    )
    parser.add_argument(
        "--url-column",
        default="url",
        help="URL column name for --url-eval-csv.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Label column name for --url-eval-csv.",
    )
    return parser


def _print_training_summary(summary: dict) -> None:
    """Print a concise console summary after training finishes."""
    comparison = pd.DataFrame(summary["comparison"])[
        [
            "model_name",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "false_positives",
            "false_positive_rate",
        ]
    ]

    print("\n=== Dataset ===")
    print(f"CSV dataset: {summary['dataset_path']}")
    print(f"Rows: {summary['dataset_rows']}")
    print(f"Class distribution: {summary['class_distribution']}")
    print(f"Missing feature values handled during preprocessing: {summary['missing_values_before_imputation']}")

    print("\n=== Model Comparison (sorted for low false positives) ===")
    print(comparison.to_string(index=False))

    print("\n=== Selected Model ===")
    print(f"Model: {summary['selected_model']}")
    print(f"Model Threshold (raw ML reference): {summary['selected_threshold']}")
    print("Risk Bands: legitimate < 0.40, suspicious >= 0.40, phishing >= 0.75")
    print("Adaptive Band: strong heuristic evidence lowers phishing band to >= 0.60")
    print(
        "Evaluation Modes: ML-only holdout metrics are authoritative for this dataset; "
        "hybrid metrics require raw URLs."
    )
    print(f"ML-only Metrics: {summary['selected_metrics']}")

    full_system = summary.get("full_system_evaluation")
    if full_system:
        if full_system.get("is_full_system_evaluation"):
            print(f"Full-System Alert Metrics: {full_system['alert_metrics']}")
            print(f"Full-System Phishing Metrics: {full_system['phishing_metrics']}")
            print(f"Full-System Risk Bands: {full_system['risk_band_counts']}")
        else:
            print("Full-System URL Metrics: unavailable for this dataset (no raw URL column)")
            print(f"Full-System Note: {full_system['note']}")
            proxy = full_system.get("feature_proxy_evaluation")
            if proxy:
                print(f"Feature-Proxy Alert Metrics (diagnostic only): {proxy['alert_metrics']}")
                print(f"Feature-Proxy Phishing Metrics (diagnostic only): {proxy['phishing_metrics']}")
                print(f"Feature-Proxy Risk Bands: {proxy['risk_band_counts']}")

    print(f"Saved bundle: {summary['model_output_path']}")
    print(f"Artifacts directory: {summary['artifacts_directory']}")

    print("\n=== Example Predictions ===")
    for prediction in summary["example_predictions"]:
        confidence_suffix = (
            f"/{prediction['confidence_score']:.3f}"
            if "confidence_score" in prediction
            else ""
        )
        print(
            f"{prediction['url']} -> {prediction['prediction']} "
            f"(risk={prediction['risk_score']}%, "
            f"band={prediction.get('risk_band', prediction['prediction'])}, "
            f"confidence={prediction['confidence']}{confidence_suffix}, "
            f"p={prediction['probability']})"
        )
        reasons = prediction.get("reasons") or []
        if reasons:
            print(f"  reasons: {', '.join(reasons)}")


def main() -> None:
    """Train the phishing pipeline using CLI arguments."""
    parser = build_argument_parser()
    args = parser.parse_args()

    summary = run_training_pipeline(
        csv_path=args.csv_path,
        arff_path=args.arff_path,
        model_output_path=args.model_output,
        output_dir=args.artifacts_dir,
        test_size=args.test_size,
        max_fpr=args.max_fpr,
        url_eval_csv_path=args.url_eval_csv,
        url_column=args.url_column,
        label_column=args.label_column,
        debug=args.debug,
    )
    _print_training_summary(summary)


if __name__ == "__main__":
    main()