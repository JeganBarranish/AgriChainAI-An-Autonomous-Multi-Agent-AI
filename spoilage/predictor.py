"""
Product-specific spoilage prediction API.

[SUPERVISED inference — loads trained artifacts]
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.spoilage_risk import compute_spoilage_risk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEER_MODEL = PROJECT_ROOT / "models" / "paneer_ac_csn.keras"
PANEER_PREPROCESSOR = PROJECT_ROOT / "models" / "paneer_preprocessor.pkl"
PANEER_LEGACY_MODEL = PROJECT_ROOT / "models" / "ac_csn_spoilage_model.keras"
PANEER_LEGACY_PREP = PROJECT_ROOT / "models" / "spoilage_preprocessor.pkl"
PANEER_BEST = PROJECT_ROOT / "models" / "paneer_best_model.pkl"
PANEER_BEST_LEGACY = PROJECT_ROOT / "models" / "best_spoilage_model.pkl"

DRYFRUIT_MODEL = PROJECT_ROOT / "models" / "dryfruit_ac_csn.keras"
DRYFRUIT_PREPROCESSOR = PROJECT_ROOT / "models" / "dryfruit_preprocessor.pkl"
DRYFRUIT_BEST = PROJECT_ROOT / "models" / "dryfruit_best_model.pkl"

READINGS_PER_DAY = 139
_CLASS_CACHE: dict = {}


class PredictionError(RuntimeError):
    pass


def _load_pickle(path: Path):
    if not path.exists():
        raise PredictionError(f"Missing artifact: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def _resolve_paneer_artifacts():
    prep_path = PANEER_PREPROCESSOR if PANEER_PREPROCESSOR.exists() else PANEER_LEGACY_PREP
    prep = _load_pickle(prep_path)
    best_path = PANEER_BEST if PANEER_BEST.exists() else PANEER_BEST_LEGACY
    use_ac_csn = not best_path.exists()
    return prep, best_path, use_ac_csn


def predict_paneer(
    temperature: float,
    humidity: float,
    storage_duration: float,
    compressor_activity: float | None = None,
    fridge_opening_count: int | None = None,
) -> dict:
    if storage_duration < 0:
        raise PredictionError("Storage duration must be >= 0 days.")
    if not (-30 <= temperature <= 40):
        raise PredictionError("Temperature must be between -30 and 40 °C.")
    if not (0 <= humidity <= 100):
        raise PredictionError("Humidity must be between 0 and 100 %.")
    """
    Paneer spoilage prediction.

    compressor_activity and fridge_opening_count are optional; when omitted,
    median training values are used (documented in spoilage_predict.py).
    """
    prep, best_path, use_ac_csn = _resolve_paneer_artifacts()
    feature_columns = prep["feature_columns"]

    # Load daily stats from supervised dataset
    supervised = PROJECT_ROOT / "dataset" / "Paneer_ColdChain_Supervised.csv"
    stats = {
        "temperature_std": 0.99, "temperature_range": 3.4,
        "max_minus_mean_temp": 1.7, "mean_minus_min_temp": 1.71,
        "rh_std": 0.99, "rh_range": 3.4,
        "max_minus_mean_rh": 1.7, "mean_minus_min_rh": 1.71,
        "compressor_cycle_count": 3.0,
        "compressor_on_ratio": 0.72,
        "fridge_open_ratio": 0.01,
    }
    daily_history = None
    if supervised.exists():
        daily = pd.read_csv(supervised).drop_duplicates("storage_day")
        stats.update({
            "temperature_std": float(daily["temperature_std"].median()),
            "temperature_range": float(daily["temperature_range"].median()),
            "max_minus_mean_temp": float((daily["max_temperature"] - daily["mean_temperature"]).median()),
            "mean_minus_min_temp": float((daily["mean_temperature"] - daily["min_temperature"]).median()),
            "rh_std": float(daily["rh_std"].median()),
            "rh_range": float(daily["rh_range"].median()),
            "max_minus_mean_rh": float((daily["max_rh"] - daily["mean_rh"]).median()),
            "mean_minus_min_rh": float((daily["mean_rh"] - daily["min_rh"]).median()),
            "compressor_cycle_count": float(daily["compressor_cycle_count"].median()),
            "compressor_on_ratio": float(daily["compressor_on_ratio"].median()),
            "fridge_open_ratio": float(daily["fridge_open_ratio"].median()),
        })
        daily_history = daily.set_index("storage_day", drop=False)[feature_columns]

    if compressor_activity is None:
        compressor_activity = stats["compressor_on_ratio"] * 100.0
    if fridge_opening_count is None:
        fridge_opening_count = int(round(stats["fridge_open_ratio"] * READINGS_PER_DAY))

    safe_temp = 4.0
    sigma = max(stats["temperature_std"], 1e-6)
    z = (safe_temp - temperature) / (sigma * math.sqrt(2.0))
    excursion_ratio = 0.5 * math.erfc(z)

    row = {
        "mean_temperature": temperature,
        "min_temperature": temperature - stats["mean_minus_min_temp"],
        "max_temperature": temperature + stats["max_minus_mean_temp"],
        "temperature_std": stats["temperature_std"],
        "temperature_range": stats["temperature_range"],
        "temperature_excursion_count": excursion_ratio * READINGS_PER_DAY,
        "time_above_safe_threshold_ratio": excursion_ratio,
        "mean_rh": humidity,
        "min_rh": humidity - stats["mean_minus_min_rh"],
        "max_rh": humidity + stats["max_minus_mean_rh"],
        "rh_std": stats["rh_std"],
        "rh_range": stats["rh_range"],
        "compressor_on_ratio": compressor_activity / 100.0,
        "compressor_cycle_count": stats["compressor_cycle_count"],
        "fridge_open_count": fridge_opening_count,
        "fridge_open_ratio": fridge_opening_count / READINGS_PER_DAY,
        "storage_day": storage_duration,
    }
    frame = pd.DataFrame([row])[feature_columns]
    scaled = prep["scaler"].transform(frame)

    if use_ac_csn:
        import tensorflow as tf
        model_path = PANEER_MODEL if PANEER_MODEL.exists() else PANEER_LEGACY_MODEL
        model = tf.keras.models.load_model(model_path)
        seq_len = prep["sequence_length"]
        rows = []
        for offset in range(seq_len - 1, 0, -1):
            if daily_history is not None:
                day = int(min(max(storage_duration - offset, 0), daily_history.index.max()))
                rows.append(daily_history.loc[[day]][feature_columns])
            else:
                rows.append(frame)
        rows.append(frame)
        seq = pd.concat(rows, ignore_index=True)
        x_seq = prep["scaler"].transform(seq).reshape(1, seq_len, len(feature_columns)).astype(np.float32)
        env_idx = [feature_columns.index(f) for f in prep["environmental_features"]]
        ops_idx = [feature_columns.index(f) for f in prep["operational_features"]]
        proba = model.predict([x_seq, scaled[:, env_idx], scaled[:, ops_idx]], verbose=0)[0]
        model_name = "Paneer AC-CSN"
    else:
        model = _load_pickle(best_path)
        proba = model.predict_proba(scaled)[0]
        model_name = "Paneer XGBoost"

    labels = prep["class_labels"]
    idx = {l: i for i, l in enumerate(labels)}
    pred = int(np.argmax(proba))
    risk = compute_spoilage_risk(proba, labels, prep["risk_weights"])

    return {
        "product_type": "paneer",
        "product_subtype": "Paneer",
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "storage_duration_days": int(storage_duration),
        "compressor_activity": round(compressor_activity, 1),
        "fridge_opening_count": int(fridge_opening_count),
        "fresh_probability": float(proba[idx["FRESH"]]),
        "warning_probability": float(proba[idx["WARNING"]]),
        "spoiled_probability": float(proba[idx["SPOILED"]]),
        "spoilage_risk": round(float(proba[idx["SPOILED"]]) * 100, 1),
        "status": labels[pred],
        "confidence": round(float(np.max(proba)), 4),
        "model_used": model_name,
        "inputs_used": {
            "compressor_activity": "user" if compressor_activity != stats["compressor_on_ratio"] * 100 else "default_from_training_median",
            "fridge_opening_count": "user" if fridge_opening_count != int(round(stats["fridge_open_ratio"] * READINGS_PER_DAY)) else "default_from_training_median",
        },
    }


def predict_dryfruit(
    product_subtype: str,
    storage_duration_days: float,
    dryer_type: str,
    packaging: str,
) -> dict:
    """Dried mango/pineapple prediction — no temp/RH (not in source dataset)."""
    if storage_duration_days < 0:
        raise PredictionError("Storage duration must be >= 0 days.")
    prep = _load_pickle(DRYFRUIT_PREPROCESSOR)

    fruit = product_subtype.strip().title()
    dryer = dryer_type.strip().upper()
    if dryer in ("CMD", "MIXED"):
        dryer = "CMD"
    elif dryer in ("TD", "TUNNEL", "TUNEL"):
        dryer = "TD"
    pkg_map = {"NONE": "None", "NO": "None", "HDPE": "HDPE", "HIGH DENSITY": "HDPE",
               "LDPE": "LDPE", "LOW DENSITY": "LDPE"}
    pkg = pkg_map.get(packaging.strip().upper(), packaging.strip())

    if fruit not in prep["valid_fruits"]:
        raise PredictionError(f"Invalid fruit '{fruit}'. Choose: {prep['valid_fruits']}")
    if dryer not in prep["valid_dryer_types"]:
        raise PredictionError(f"Invalid dryer type '{dryer}'. Choose: {prep['valid_dryer_types']}")
    if pkg not in prep["valid_packaging"]:
        raise PredictionError(f"Invalid packaging '{pkg}'. Choose: {prep['valid_packaging']}")

    storage_months = storage_duration_days / prep["days_per_month"]
    row = {
        "storage_duration_days": storage_duration_days,
        "storage_months": storage_months,
        "dryer_type_CMD": float(dryer == "CMD"),
        "dryer_type_TD": float(dryer == "TD"),
        "packaging_None": float(pkg == "None"),
        "packaging_HDPE": float(pkg == "HDPE"),
        "packaging_LDPE": float(pkg == "LDPE"),
        "fruit_Mango": float(fruit == "Mango"),
        "fruit_Pineapple": float(fruit == "Pineapple"),
    }
    common = prep["common_features"]
    product = prep["product_features"]
    x_c = prep["scaler_common"].transform(pd.DataFrame([row])[common])
    x_p = prep["scaler_product"].transform(pd.DataFrame([row])[product])

    if prep.get("model_type") == "XGBoost" and DRYFRUIT_BEST.exists():
        model = _load_pickle(DRYFRUIT_BEST)
        x_all = prep["scaler_all"].transform(pd.DataFrame([row])[prep["all_features"]])
        proba = model.predict_proba(x_all)[0]
        model_name = "DryFruit XGBoost"
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(DRYFRUIT_MODEL)
        proba = model.predict([x_c, x_p], verbose=0)[0]
        model_name = "DryFruit AC-CSN"

    labels = prep["class_labels"]
    idx = {l: i for i, l in enumerate(labels)}
    pred = int(np.argmax(proba))

    return {
        "product_type": "dried_fruit",
        "product_subtype": f"Dried {fruit}",
        "storage_duration_days": int(storage_duration_days),
        "dryer_type": dryer,
        "packaging": pkg,
        "fresh_probability": float(proba[idx["FRESH"]]),
        "warning_probability": float(proba[idx["WARNING"]]),
        "spoiled_probability": float(proba[idx["SPOILED"]]),
        "spoilage_risk": round(float(proba[idx["SPOILED"]]) * 100, 1),
        "status": labels[pred],
        "confidence": round(float(np.max(proba)), 4),
        "model_used": model_name,
        "feature_importance": prep.get("feature_importance", {}),
        "display_map": prep.get("display_map", {}),
    }
