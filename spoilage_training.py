"""
AgriChain Cold-Chain Spoilage Network (AC-CSN) — Training & Model Comparison

Trains the custom AC-CSN neural network alongside baseline models
(Random Forest, XGBoost, LightGBM, GRU baseline) on the real Paneer
cold-chain supervised dataset, evaluates all models with identical
methodology, and saves the best model for inference.

Train/test methodology:
  Samples are grouped by storage_day (replicates within a day share
  identical sensor features). Whole days are assigned to train or test
  to prevent replicate leakage. The test days cover all three classes.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier

from spoilage_preprocessing import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    OUTPUT_PATH as SUPERVISED_PATH,
    run_preprocessing,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
AC_CSN_PATH = MODELS_DIR / "ac_csn_spoilage_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "spoilage_preprocessor.pkl"
BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"

RANDOM_STATE = 42
SEQUENCE_LENGTH = 3  # days of cold-chain history fed to the temporal encoder

# Test days chosen to cover all classes while keeping replicate groups intact:
#   FRESH: day 2  | WARNING: day 6 | SPOILED: days 9, 12
TEST_DAYS = [2, 6, 9, 12]

# Feature grouping for the AC-CSN dual-branch design
ENVIRONMENTAL_FEATURES = [
    "mean_temperature",
    "min_temperature",
    "max_temperature",
    "temperature_std",
    "temperature_range",
    "temperature_excursion_count",
    "time_above_safe_threshold_ratio",
    "mean_rh",
    "min_rh",
    "max_rh",
    "rh_std",
    "rh_range",
]
OPERATIONAL_FEATURES = [
    "compressor_on_ratio",
    "compressor_cycle_count",
    "fridge_open_count",
    "fridge_open_ratio",
    "storage_day",
]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def load_supervised() -> pd.DataFrame:
    """Load the processed dataset, regenerating it if missing."""
    if not SUPERVISED_PATH.exists():
        run_preprocessing()
    return pd.read_csv(SUPERVISED_PATH)


def build_sequence_lookup(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    For each storage day, build the sequence of daily feature vectors covering
    the previous SEQUENCE_LENGTH days (inclusive of the current day).
    Days before the experiment start are padded with the day-0 vector.
    """
    daily = df.drop_duplicates("storage_day").set_index("storage_day", drop=False)[FEATURE_COLUMNS]
    lookup = {}
    for day in daily.index:
        seq_days = [max(0, day - offset) for offset in range(SEQUENCE_LENGTH - 1, -1, -1)]
        lookup[day] = daily.loc[seq_days].values.astype(np.float32)
    return lookup


def split_by_day(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware split: whole storage days go to either train or test."""
    test_mask = df["storage_day"].isin(TEST_DAYS)
    return df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# AC-CSN architecture
# ---------------------------------------------------------------------------
def build_ac_csn(n_env: int, n_ops: int, seq_len: int, n_features: int) -> tf.keras.Model:
    """
    AgriChain Cold-Chain Spoilage Network (AC-CSN).

    Dual-branch design sized for a small dataset:
      Temporal branch : GRU(12) over the last SEQUENCE_LENGTH days of all
                        cold-chain features (captures deterioration dynamics).
      Environmental   : Dense(8) on current-day temperature/humidity stats.
      Operational     : Dense(6) on compressor/fridge/storage-duration signals.
      Fusion          : concatenate → Dense(16) → Dropout → Softmax(3).

    BatchNormalization is intentionally omitted: with ~50 training samples
    and batch size 8, batch statistics are too noisy and destabilized
    validation performance.
    """
    seq_input = layers.Input(shape=(seq_len, n_features), name="sequence_input")
    env_input = layers.Input(shape=(n_env,), name="environmental_input")
    ops_input = layers.Input(shape=(n_ops,), name="operational_input")

    # Temporal encoder — small GRU appropriate for 14 distinct day sequences
    t = layers.GRU(12, name="temporal_encoder")(seq_input)
    t = layers.Dropout(0.3)(t)

    # Environmental representation (no BatchNorm — unstable with ~50 samples)
    e = layers.Dense(8, activation="relu", name="environmental_repr")(env_input)

    # Operational representation
    o = layers.Dense(6, activation="relu", name="operational_repr")(ops_input)

    # Feature fusion
    fused = layers.Concatenate(name="feature_fusion")([t, e, o])
    x = layers.Dense(16, activation="relu")(fused)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation="softmax", name="spoilage_class")(x)

    model = models.Model(
        inputs=[seq_input, env_input, ops_input],
        outputs=output,
        name="AC_CSN",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru_baseline(seq_len: int, n_features: int) -> tf.keras.Model:
    """Plain GRU baseline for comparison against AC-CSN."""
    model = models.Sequential(
        [
            layers.Input(shape=(seq_len, n_features)),
            layers.GRU(16),
            layers.Dropout(0.3),
            layers.Dense(3, activation="softmax"),
        ],
        name="GRU_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "fresh_recall": float(per_class_recall[0]) if len(per_class_recall) > 0 else 0.0,
        "warning_recall": float(per_class_recall[1]) if len(per_class_recall) > 1 else 0.0,
        "spoiled_recall": float(per_class_recall[2]) if len(per_class_recall) > 2 else 0.0,
    }


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n{'=' * 55}\n{name} — TEST EVALUATION\n{'=' * 55}")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2],
                                target_names=CLASS_LABELS, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_comparison(all_metrics: dict[str, dict], path: Path) -> None:
    metric_keys = ["accuracy", "f1_macro", "f1_weighted", "spoiled_recall"]
    labels = ["Accuracy", "Macro F1", "Weighted F1", "Spoiled Recall"]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(all_metrics)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, m) in enumerate(all_metrics.items()):
        ax.bar(x + i * width, [m[k] for k in metric_keys], width, label=name)
    ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Spoilage Model Comparison — Paneer Cold-Chain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison chart → {path}")


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix → {path}")


def plot_history(history, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("AC-CSN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("AC-CSN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training history → {path}")


def plot_feature_importance(model, path: Path, model_name: str) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imp = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp["feature"], imp["importance"], color="seagreen")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance → {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    set_seeds()
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_supervised()
    print(f"Supervised dataset: {len(df)} samples, {df['storage_day'].nunique()} storage days")

    train_df, test_df = split_by_day(df)
    print(f"Train: {len(train_df)} samples (days {sorted(train_df['storage_day'].unique())})")
    print(f"Test : {len(test_df)} samples (days {sorted(test_df['storage_day'].unique())})")

    # Flat features for tree models
    scaler = StandardScaler()
    x_train_flat = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    x_test_flat = scaler.transform(test_df[FEATURE_COLUMNS])
    y_train = train_df["spoilage_class"].values
    y_test = test_df["spoilage_class"].values

    # Sequence features for temporal models (scaled with the same scaler)
    seq_lookup = build_sequence_lookup(df)
    def to_sequences(frame: pd.DataFrame) -> np.ndarray:
        seqs = np.stack([seq_lookup[d] for d in frame["storage_day"]])
        flat = seqs.reshape(-1, len(FEATURE_COLUMNS))
        return scaler.transform(flat).reshape(seqs.shape).astype(np.float32)

    x_train_seq = to_sequences(train_df)
    x_test_seq = to_sequences(test_df)

    # Branch inputs for AC-CSN (scaled flat features, split by group)
    env_idx = [FEATURE_COLUMNS.index(f) for f in ENVIRONMENTAL_FEATURES]
    ops_idx = [FEATURE_COLUMNS.index(f) for f in OPERATIONAL_FEATURES]
    x_train_env, x_test_env = x_train_flat[:, env_idx], x_test_flat[:, env_idx]
    x_train_ops, x_test_ops = x_train_flat[:, ops_idx], x_test_flat[:, ops_idx]

    all_metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    # ---------------- Baseline tree models ----------------
    baselines = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, verbose=-1),
    }
    trained_baselines = {}
    for name, model in baselines.items():
        print(f"\nTraining {name} …")
        model.fit(x_train_flat, y_train)
        y_pred = model.predict(x_test_flat)
        all_metrics[name] = report(name, y_test, y_pred)
        predictions[name] = y_pred
        trained_baselines[name] = model

    # ---------------- GRU baseline ----------------
    print("\nTraining GRU baseline …")
    gru = build_gru_baseline(SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    gru.fit(
        x_train_seq, y_train,
        validation_split=0.2, epochs=200, batch_size=8, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)],
    )
    y_pred = np.argmax(gru.predict(x_test_seq, verbose=0), axis=1)
    all_metrics["GRU baseline"] = report("GRU baseline", y_test, y_pred)
    predictions["GRU baseline"] = y_pred

    # ---------------- AC-CSN ----------------
    print("\nTraining AC-CSN …")
    ac_csn = build_ac_csn(
        len(ENVIRONMENTAL_FEATURES), len(OPERATIONAL_FEATURES),
        SEQUENCE_LENGTH, len(FEATURE_COLUMNS),
    )
    ac_csn.summary()
    class_weight = {
        c: len(y_train) / (3 * np.sum(y_train == c)) for c in np.unique(y_train)
    }
    history = ac_csn.fit(
        [x_train_seq, x_train_env, x_train_ops], y_train,
        validation_split=0.2, epochs=300, batch_size=8, verbose=0,
        class_weight=class_weight,
        callbacks=[EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)],
    )
    y_pred = np.argmax(
        ac_csn.predict([x_test_seq, x_test_env, x_test_ops], verbose=0), axis=1)
    all_metrics["AC-CSN"] = report("AC-CSN", y_test, y_pred)
    predictions["AC-CSN"] = y_pred

    # ---------------- Comparison & selection ----------------
    print(f"\n{'Model':<20}{'Accuracy':<12}{'Macro-F1':<12}{'Spoiled Recall':<15}")
    print("-" * 59)
    for name, m in all_metrics.items():
        print(f"{name:<20}{m['accuracy']:<12.4f}{m['f1_macro']:<12.4f}{m['spoiled_recall']:<15.4f}")

    # Selection: Macro F1 primary, Spoiled recall tiebreaker
    best_name = max(all_metrics, key=lambda n: (all_metrics[n]["f1_macro"],
                                                all_metrics[n]["spoiled_recall"]))
    print(f"\nBest model: {best_name} "
          f"(Macro F1 = {all_metrics[best_name]['f1_macro']:.4f}, "
          f"Spoiled recall = {all_metrics[best_name]['spoiled_recall']:.4f})")

    # ---------------- Save artifacts ----------------
    plot_comparison(all_metrics, RESULTS_DIR / "spoilage_model_comparison.png")
    plot_confusion(y_test, predictions[best_name], best_name,
                   RESULTS_DIR / "spoilage_confusion_matrix.png")
    plot_history(history, RESULTS_DIR / "spoilage_training_history.png")
    if best_name in trained_baselines:
        plot_feature_importance(trained_baselines[best_name],
                                RESULTS_DIR / "spoilage_feature_importance.png", best_name)
    else:
        # Save importance chart from the strongest tree model for interpretability
        tree_best = max(trained_baselines,
                        key=lambda n: all_metrics[n]["f1_macro"])
        plot_feature_importance(trained_baselines[tree_best],
                                RESULTS_DIR / "spoilage_feature_importance.png", tree_best)

    with (RESULTS_DIR / "spoilage_evaluation_metrics.json").open("w") as fh:
        json.dump({
            "best_model": best_name,
            "selection_criteria": "Macro F1 (primary), Spoiled-class recall (tiebreaker)",
            "test_days": TEST_DAYS,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "models": all_metrics,
        }, fh, indent=2)
    print(f"Saved metrics → {RESULTS_DIR / 'spoilage_evaluation_metrics.json'}")

    # Save AC-CSN model (always, for inference) and preprocessor
    ac_csn.save(AC_CSN_PATH)
    print(f"Saved AC-CSN model → {AC_CSN_PATH}")

    # Save best baseline too if it beat AC-CSN
    best_artifact = None
    if best_name in trained_baselines:
        best_artifact = MODELS_DIR / "best_spoilage_model.pkl"
        with best_artifact.open("wb") as fh:
            pickle.dump(trained_baselines[best_name], fh)
        print(f"Saved best baseline model → {best_artifact}")

    preprocessor = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "environmental_features": ENVIRONMENTAL_FEATURES,
        "operational_features": OPERATIONAL_FEATURES,
        "sequence_length": SEQUENCE_LENGTH,
        "class_labels": CLASS_LABELS,
        "risk_weights": {"FRESH": 0.0, "WARNING": 50.0, "SPOILED": 100.0},
    }
    with PREPROCESSOR_PATH.open("wb") as fh:
        pickle.dump(preprocessor, fh)
    print(f"Saved preprocessor → {PREPROCESSOR_PATH}")

    with BEST_MODEL_INFO_PATH.open("w") as fh:
        json.dump({
            "best_model": best_name,
            "ac_csn_path": str(AC_CSN_PATH),
            "best_baseline_path": str(best_artifact) if best_artifact else None,
            "metrics": all_metrics[best_name],
        }, fh, indent=2)

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
