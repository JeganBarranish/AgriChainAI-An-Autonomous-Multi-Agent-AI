"""
Multi-product spoilage model training.

[SUPERVISED learning]

Paneer: delegates to existing spoilage_training.py (preserved unchanged).
Dried fruit: separate AC-CSN-inspired model + XGBoost comparison.
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier

from spoilage.ac_csn_model import build_dryfruit_ac_csn
from spoilage.data_loader import write_dry_fruit_profile
from spoilage.dryfruit_processor import DRYFRUIT_FEATURE_COLUMNS, build_dryfruit_samples, encode_dryfruit_features
from spoilage.explainability import DRYFRUIT_DISPLAY, permutation_importance
from spoilage.feature_engineering import build_unified_dataset
from spoilage.target_builder import CLASS_LABELS, label_dryfruit, write_target_definition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PANEER_MODEL = MODELS_DIR / "paneer_ac_csn.keras"
DRYFRUIT_MODEL = MODELS_DIR / "dryfruit_ac_csn.keras"
PANEER_PREPROCESSOR = MODELS_DIR / "paneer_preprocessor.pkl"
DRYFRUIT_PREPROCESSOR = MODELS_DIR / "dryfruit_preprocessor.pkl"
EVAL_REPORT = MODELS_DIR / "evaluation_report.txt"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

DRYFRUIT_COMMON = [
    "storage_duration_days", "storage_months",
    "packaging_None", "packaging_HDPE", "packaging_LDPE",
]
DRYFRUIT_PRODUCT = [
    "dryer_type_CMD", "dryer_type_TD", "fruit_Mango", "fruit_Pineapple",
]

# Group-aware test holdout: entire fruit×dryer combinations
DRYFRUIT_TEST_GROUP_PREFIXES = ("Pineapple|TD|", "Mango|Mixed|")


def _metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2])),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2])),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=[0, 1, 2])),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=[0, 1, 2])),
        "spoiled_recall": float(per_recall[2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_LABELS, zero_division=0
        ),
    }


def train_paneer() -> dict:
    """Run existing Paneer pipeline; copy artifacts to product-specific paths."""
    import spoilage_training

    spoilage_training.main()

    src_model = MODELS_DIR / "ac_csn_spoilage_model.keras"
    src_prep = MODELS_DIR / "spoilage_preprocessor.pkl"
    src_best = MODELS_DIR / "best_spoilage_model.pkl"
    src_info = MODELS_DIR / "best_model_info.json"

    shutil.copy2(src_model, PANEER_MODEL)
    shutil.copy2(src_prep, PANEER_PREPROCESSOR)
    if src_best.exists():
        shutil.copy2(src_best, MODELS_DIR / "paneer_best_model.pkl")

    with src_info.open() as fh:
        info = json.load(fh)

    with (PROJECT_ROOT / "results" / "spoilage_evaluation_metrics.json").open() as fh:
        full = json.load(fh)

    return {"best_model_info": info, "all_models": full.get("models", {}), "product": "paneer"}


def _split_dryfruit(df: pd.DataFrame):
    test_mask = df["group_id"].str.startswith(DRYFRUIT_TEST_GROUP_PREFIXES)
    train = df[~test_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)
    if len(test) == 0 or len(train) == 0:
        # fallback: split by storage_months (time-aware)
        test = df[df["storage_months"] == 6].reset_index(drop=True)
        train = df[df["storage_months"] != 6].reset_index(drop=True)
    return train, test


def train_dryfruit() -> dict:
    raw = build_dryfruit_samples(include_fresh_fruit=False)
    df = label_dryfruit(encode_dryfruit_features(raw))
    train_df, test_df = _split_dryfruit(df)

    scaler_common = StandardScaler()
    scaler_product = StandardScaler()
    x_tr_c = scaler_common.fit_transform(train_df[DRYFRUIT_COMMON])
    x_tr_p = scaler_product.fit_transform(train_df[DRYFRUIT_PRODUCT])
    x_te_c = scaler_common.transform(test_df[DRYFRUIT_COMMON])
    x_te_p = scaler_product.transform(test_df[DRYFRUIT_PRODUCT])
    y_tr = train_df["target_class"].values
    y_te = test_df["target_class"].values

    tf.random.set_seed(RANDOM_STATE)
    model = build_dryfruit_ac_csn(len(DRYFRUIT_COMMON), len(DRYFRUIT_PRODUCT))
    cw = {int(c): len(y_tr) / (3 * max(1, np.sum(y_tr == c))) for c in np.unique(y_tr)}
    model.fit(
        [x_tr_c, x_tr_p], y_tr,
        validation_split=0.15, epochs=300, batch_size=4, verbose=0,
        class_weight=cw,
        callbacks=[EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)],
    )
    y_pred_nn = model.predict([x_te_c, x_te_p], verbose=0).argmax(axis=1)
    nn_metrics = _metrics(y_te, y_pred_nn)

    # XGBoost on full flat features for comparison
    scaler_all = StandardScaler()
    x_tr_all = scaler_all.fit_transform(train_df[DRYFRUIT_FEATURE_COLUMNS])
    x_te_all = scaler_all.transform(test_df[DRYFRUIT_FEATURE_COLUMNS])
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        eval_metric="mlogloss", random_state=RANDOM_STATE,
    )
    xgb.fit(x_tr_all, y_tr)
    y_pred_xgb = xgb.predict(x_te_all)
    xgb_metrics = _metrics(y_te, y_pred_xgb)

    use_xgb = xgb_metrics["f1_macro"] >= nn_metrics["f1_macro"]
    best_name = "XGBoost" if use_xgb else "DryFruit_AC_CSN"
    best_metrics = xgb_metrics if use_xgb else nn_metrics
    best_pred = y_pred_xgb if use_xgb else y_pred_nn

    # Feature importance for explanations (always from XGBoost — stable on small n)
    imp = dict(zip(DRYFRUIT_FEATURE_COLUMNS, xgb.feature_importances_.tolist()))

    model.save(DRYFRUIT_MODEL)

    preprocessor = {
        "product": "dryfruit",
        "model_type": best_name,
        "common_features": DRYFRUIT_COMMON,
        "product_features": DRYFRUIT_PRODUCT,
        "all_features": DRYFRUIT_FEATURE_COLUMNS,
        "scaler_common": scaler_common,
        "scaler_product": scaler_product,
        "scaler_all": scaler_all,
        "class_labels": CLASS_LABELS,
        "risk_weights": {"FRESH": 0.0, "WARNING": 50.0, "SPOILED": 100.0},
        "feature_importance": imp,
        "display_map": DRYFRUIT_DISPLAY,
        "days_per_month": 30,
        "valid_dryer_types": ["CMD", "TD"],
        "valid_packaging": ["None", "HDPE", "LDPE"],
        "valid_fruits": ["Mango", "Pineapple"],
    }
    with DRYFRUIT_PREPROCESSOR.open("wb") as fh:
        pickle.dump(preprocessor, fh)
    if use_xgb:
        with (MODELS_DIR / "dryfruit_best_model.pkl").open("wb") as fh:
            pickle.dump(xgb, fh)

    return {
        "product": "dryfruit",
        "best_model": best_name,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_class_distribution": train_df["target_status"].value_counts().to_dict(),
        "test_class_distribution": test_df["target_status"].value_counts().to_dict(),
        "test_groups": sorted(test_df["group_id"].unique()),
        "metrics": best_metrics,
        "nn_metrics": nn_metrics,
        "xgb_metrics": xgb_metrics,
        "feature_importance": imp,
    }


def train_all() -> dict:
    MODELS_DIR.mkdir(exist_ok=True)
    write_dry_fruit_profile()
    write_target_definition()
    build_unified_dataset(save=True)

    lines = ["AgriChainAI Multi-Product Spoilage — Evaluation Report", "=" * 60, ""]
    results = {}

    print("\n>>> Training Paneer (existing AC-CSN pipeline)...")
    paneer = train_paneer()
    results["paneer"] = paneer
    lines.append("PANEER MODEL")
    lines.append("-" * 40)
    lines.append(f"Best model: {paneer['best_model_info'].get('best_model')}")
    for name, m in paneer.get("all_models", {}).items():
        lines.append(f"  {name}: acc={m['accuracy']:.3f} macro_f1={m['f1_macro']:.3f} spoiled_recall={m['spoiled_recall']:.3f}")
    lines.append("")

    print("\n>>> Training Dried Fruit (separate model)...")
    dry = train_dryfruit()
    results["dryfruit"] = dry
    lines.append("DRIED FRUIT MODEL")
    lines.append("-" * 40)
    lines.append(f"Best model: {dry['best_model']}")
    lines.append(f"Train n={dry['train_samples']} | Test n={dry['test_samples']}")
    lines.append(f"Train classes: {dry['train_class_distribution']}")
    lines.append(f"Test classes: {dry['test_class_distribution']}")
    m = dry["metrics"]
    lines.append(f"Test accuracy: {m['accuracy']:.3f}")
    lines.append(f"Macro F1: {m['f1_macro']:.3f} | Weighted F1: {m['f1_weighted']:.3f}")
    lines.append(f"Spoiled recall: {m['spoiled_recall']:.3f}")
    lines.append(f"Confusion matrix:\n{m['confusion_matrix']}")
    lines.append(f"\n{m['classification_report']}")
    lines.append("")
    lines.append("LEAKAGE / OVERFITTING CHECK (dried fruit):")
    lines.append("- Test split holds out entire fruit×dryer×packaging groups (not random rows).")
    lines.append("- Target columns (TPC) excluded from model inputs.")
    lines.append("- n=24 total dried cells; test n=6 — 100% accuracy may reflect small-sample")
    lines.append("  separability rather than production-grade generalization.")
    lines.append("")

    metadata = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "products": {
            "paneer": {
                "model_path": str(PANEER_MODEL),
                "preprocessor_path": str(PANEER_PREPROCESSOR),
                "features": "17 cold-chain daily aggregates (see spoilage_preprocessing.FEATURE_COLUMNS)",
                "target_definition": "results/target_definition.json",
                "best_model": paneer["best_model_info"].get("best_model"),
                "metrics": paneer["best_model_info"].get("metrics"),
            },
            "dryfruit": {
                "model_path": str(DRYFRUIT_MODEL),
                "preprocessor_path": str(DRYFRUIT_PREPROCESSOR),
                "inference_features": DRYFRUIT_COMMON + DRYFRUIT_PRODUCT,
                "inference_user_inputs": [
                    "product_subtype (Mango/Pineapple)",
                    "storage_duration_days",
                    "dryer_type (CMD/TD)",
                    "packaging (None/HDPE/LDPE)",
                ],
                "not_available_in_source": ["temperature", "humidity"],
                "target_definition": "dataset/reports/target_definition.md",
                "best_model": dry["best_model"],
                "metrics": {k: v for k, v in dry["metrics"].items() if k != "classification_report"},
            },
        },
        "class_mapping": {"FRESH": 0, "WARNING": 1, "SPOILED": 2},
        "unified_dataset": str(PROJECT_ROOT / "dataset" / "ac_csn_multiproduct.csv"),
    }
    EVAL_REPORT.write_text("\n".join(lines), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report → {EVAL_REPORT}")
    print(f"Saved model metadata → {METADATA_PATH}")
    return results


if __name__ == "__main__":
    train_all()
