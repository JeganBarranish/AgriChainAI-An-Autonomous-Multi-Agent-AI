"""
Anomaly Detection — AI4I Predictive Maintenance

Trains and compares Random Forest, XGBoost, and LightGBM classifiers
to detect machine failures (anomalies) from preprocessed sensor data.

Outputs:
  - anomaly_model.pkl          (best model by F1-score)
  - results/model_comparison.png
  - results/confusion_matrix_<model>.png
  - results/feature_importance.png
  - results/evaluation_metrics.json
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset" / "Preprocessed_Predictive_Maintenance.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = PROJECT_ROOT / "anomaly_model.pkl"

TARGET_COLUMN = "Machine failure"
FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
# XGBoost requires feature names without [, ], or <
FEATURE_COLUMNS_SAFE = [
    "Type",
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class ModelResult:
    """Container for a trained model and its evaluation metrics."""

    name: str
    model: Any
    y_pred: np.ndarray
    metrics: dict[str, float]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the preprocessed AI4I predictive maintenance CSV."""
    df = pd.read_csv(path)
    missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split to preserve the rare failure class ratio.

    Failure-type columns (TWF, HDF, PWF, OSF, RNF) are excluded from
    features because they directly encode the target and would leak labels.
    """
    x = df[FEATURE_COLUMNS].copy()
    x.columns = FEATURE_COLUMNS_SAFE
    y = df[TARGET_COLUMN]

    return train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models() -> dict[str, Any]:
    """
    Define the three candidate classifiers with imbalance-aware settings.

    scale_pos_weight / class_weight up-weight the minority (failure) class.
    """
    # Approximate imbalance ratio: ~9661 normal vs ~339 failure
    pos_weight = 9661 / 339

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def evaluate_model(name: str, model: Any, x_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    """Predict on the test set and compute classification metrics."""
    y_pred = model.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    print(f"\n{'=' * 55}")
    print(f"{name.upper()} — TEST SET EVALUATION")
    print("=" * 55)
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title():12s}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return ModelResult(name=name, model=model, y_pred=y_pred, metrics=metrics)


def plot_metrics_comparison(results: list[ModelResult], output_dir: Path) -> None:
    """Bar chart comparing Accuracy, Precision, Recall, and F1 across models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    model_names = [r.name for r in results]

    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, result in enumerate(results):
        values = [result.metrics[m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=result.name)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Anomaly Detection Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison → {path}")


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> None:
    """Save a labelled confusion-matrix heatmap for one model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Anomaly (1)"],
        yticklabels=["Normal (0)", "Anomaly (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    path = output_dir / f"confusion_matrix_{safe_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {path}")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    output_dir: Path,
) -> None:
    """Plot and save feature-importance bar chart for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"Warning: {model_name} has no feature_importances_ attribute.")
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance → {path}")


def save_best_model(best: ModelResult, path: Path) -> None:
    """Persist the winning model and its metadata to disk."""
    artifact = {
        "model_name": best.name,
        "model": best.model,
        "feature_columns": FEATURE_COLUMNS_SAFE,
        "target_column": TARGET_COLUMN,
        "metrics": best.metrics,
    }
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)
    print(f"\nSaved best model ({best.name}) → {path}")


def main() -> None:
    """Run the full anomaly detection pipeline."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load preprocessed dataset
    print(f"Loading dataset from {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df):,} rows | Failure rate: {df[TARGET_COLUMN].mean():.2%}")

    # Step 2 — Train/test split
    x_train, x_test, y_train, y_test = split_data(df)
    print(f"Train: {len(x_train):,} | Test: {len(x_test):,}")

    # Step 3 — Train all candidate models
    models = build_models()
    results: list[ModelResult] = []

    for name, model in models.items():
        print(f"\nTraining {name} …")
        model.fit(x_train, y_train)
        results.append(evaluate_model(name, model, x_test, y_test))
        plot_confusion_matrix(y_test, results[-1].y_pred, name, RESULTS_DIR)

    # Step 4 — Compare metrics and select best model (by F1-score)
    plot_metrics_comparison(results, RESULTS_DIR)

    best = max(results, key=lambda r: r.metrics["f1_score"])

    print(f"\nBest model: {best.name} (F1 = {best.metrics['f1_score']:.4f})")

    # Step 5 — Feature importance for the best model
    plot_feature_importance(best.model, FEATURE_COLUMNS_SAFE, best.name, RESULTS_DIR)

    # Step 6 — Save best model and metrics summary
    save_best_model(best, MODEL_PATH)

    summary = {
        "best_model": best.name,
        "models": {r.name: r.metrics for r in results},
    }
    metrics_path = RESULTS_DIR / "anomaly_evaluation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved evaluation summary → {metrics_path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
