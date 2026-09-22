"""
AgriChainAI — Spoilage Prediction (User Inference Layer)

Loads the trained spoilage models and saved preprocessing artifacts, asks
the user for current cold-chain conditions, and prints a Cold Chain
Analysis report. Inference only — no training happens here.

CLI:
    python3 spoilage_predict.py          # Paneer-only (legacy entry point)
    python3 -m spoilage.inference        # Multi-product (Paneer + Dried Fruit)

Programmatic (future FastAPI/Flask):
    from spoilage_predict import predict_spoilage
    result = predict_spoilage(temperature=4.2, humidity=37.1,
                              storage_duration=8, compressor_activity=72,
                              fridge_opening_count=3)

Environment variables:
    SPOILAGE_MODEL=ac_csn   use the AC-CSN neural network instead of the
                            selected best model (XGBoost)
    SPOILAGE_DEBUG=1        show full tracebacks on errors

--------------------------------------------------------------------------
Which model answers the prediction?
--------------------------------------------------------------------------
Training selected XGBoost as the best model (Macro F1 = 1.0 on the held-out
test days vs 0.78 for AC-CSN). XGBoost is also far more robust for user
inference: the training experiment kept conditions nearly constant
(daily mean temp 3.7-4.4 degC, RH ~37 %), so tree models degrade gracefully
for inputs outside those ranges while the neural network's extrapolation
becomes unreliable. The inference layer therefore defaults to XGBoost and
offers AC-CSN via SPOILAGE_MODEL=ac_csn.

--------------------------------------------------------------------------
How 5 user inputs become the 17-feature model input
--------------------------------------------------------------------------
The models were trained on DAILY AGGREGATES of a 10-minute sensor log
(~139 readings/day), not on single instantaneous readings. Day-level
features are reconstructed around the user's values using the sub-daily
variation observed in the training experiment (temperature std ~1.0 degC,
range ~3.4 degC per day) so the constructed vector stays in-distribution:

  mean temperature / RH     = user values
  min / max                 = mean -/+ median training offsets (~1.7)
  std / range               = median training values (~1.0 / ~3.4)
  excursion ratio           = P(reading > 4 degC) under Normal(temp, std)
  compressor_on_ratio       = user compressor activity / 100
  compressor_cycle_count    = median of training days
  fridge_open_count         = user count; ratio = count / 139 readings/day
  storage_day               = user storage duration

For the AC-CSN temporal branch, the 3-day sequence uses the real daily
history recorded in the training experiment for the two preceding days
(clamped to days 0-13) with the current day replaced by the user-derived
features. History is never fabricated.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from utils.spoilage_risk import compute_spoilage_risk

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
AC_CSN_PATH = PROJECT_ROOT / "models" / "ac_csn_spoilage_model.keras"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "best_spoilage_model.pkl"
BEST_MODEL_INFO_PATH = PROJECT_ROOT / "models" / "best_model_info.json"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "spoilage_preprocessor.pkl"
TARGET_DEF_PATH = PROJECT_ROOT / "results" / "target_definition.json"
SUPERVISED_PATH = PROJECT_ROOT / "dataset" / "Paneer_ColdChain_Supervised.csv"

# Sensor cadence observed in the training log: 1,950 readings over 14 days
READINGS_PER_DAY = 139

DEBUG = os.environ.get("SPOILAGE_DEBUG", "0") == "1"

# Lazily initialised singletons so repeated predict_spoilage() calls are cheap
_STATE: dict = {}

# Fallback sub-daily statistics (medians of the training experiment's daily
# aggregates) used when the supervised dataset is unavailable.
_DAILY_STATS_FALLBACK = {
    "temperature_std": 0.99,
    "temperature_range": 3.4,
    "max_minus_mean_temp": 1.7,
    "mean_minus_min_temp": 1.71,
    "rh_std": 0.99,
    "rh_range": 3.4,
    "max_minus_mean_rh": 1.7,
    "mean_minus_min_rh": 1.71,
    "compressor_cycle_count": 3.0,
}


class PredictionError(RuntimeError):
    """User-friendly inference failure."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _requested_model() -> str:
    return "AC-CSN" if os.environ.get("SPOILAGE_MODEL", "").lower() == "ac_csn" else "best"


def load_model() -> dict:
    """Load the trained model(s) and preprocessing artifacts (once)."""
    if _STATE:
        return _STATE

    if not PREPROCESSOR_PATH.exists():
        raise PredictionError(
            "ERROR: preprocessor could not be loaded.\n"
            f"Check that {PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)} exists "
            "(it is created by spoilage_training.py)."
        )
    try:
        with PREPROCESSOR_PATH.open("rb") as fh:
            prep = pickle.load(fh)
    except Exception as exc:
        raise PredictionError(f"ERROR: preprocessor file is unreadable: {exc}") from exc

    required = {"scaler", "feature_columns", "environmental_features",
                "operational_features", "sequence_length", "class_labels",
                "risk_weights"}
    missing = required - set(prep)
    if missing:
        raise PredictionError(
            f"ERROR: preprocessor is missing keys {sorted(missing)} — "
            "it may come from an incompatible training version."
        )
    _STATE["preprocessor"] = prep

    # Which model to serve
    if _requested_model() == "AC-CSN":
        if not AC_CSN_PATH.exists():
            raise PredictionError(
                "ERROR: AC-CSN model could not be loaded.\n"
                f"Check that {AC_CSN_PATH.relative_to(PROJECT_ROOT)} exists "
                "(it is created by spoilage_training.py)."
            )
        try:
            import tensorflow as tf

            _STATE["model"] = tf.keras.models.load_model(AC_CSN_PATH)
            _STATE["model_name"] = "AC-CSN"
        except Exception as exc:
            raise PredictionError(
                "ERROR: AC-CSN model could not be loaded "
                f"({exc.__class__.__name__}: {exc}).\n"
                "The file may be corrupted or saved with an incompatible "
                "TensorFlow version."
            ) from exc
    else:
        if not BEST_MODEL_PATH.exists():
            raise PredictionError(
                "ERROR: best spoilage model could not be loaded.\n"
                f"Check that {BEST_MODEL_PATH.relative_to(PROJECT_ROOT)} exists "
                "(it is created by spoilage_training.py)."
            )
        try:
            with BEST_MODEL_PATH.open("rb") as fh:
                _STATE["model"] = pickle.load(fh)
            name = "best model"
            if BEST_MODEL_INFO_PATH.exists():
                with BEST_MODEL_INFO_PATH.open() as fh:
                    name = json.load(fh).get("best_model", name)
            _STATE["model_name"] = name
        except PredictionError:
            raise
        except Exception as exc:
            raise PredictionError(
                f"ERROR: best spoilage model is unreadable: {exc}") from exc

    # Safe-temperature threshold — read from the documented target definition
    safe_temp = 4.0
    if TARGET_DEF_PATH.exists():
        try:
            with TARGET_DEF_PATH.open() as fh:
                safe_temp = float(
                    json.load(fh)["thresholds"]["safe_temperature_c"]["value"])
        except Exception:
            pass  # keep documented fallback
    _STATE["safe_temp"] = safe_temp

    # Sub-daily variation statistics + real daily history for AC-CSN sequences
    stats = dict(_DAILY_STATS_FALLBACK)
    daily_history = None
    if SUPERVISED_PATH.exists():
        try:
            daily = pd.read_csv(SUPERVISED_PATH).drop_duplicates("storage_day")
            stats = {
                "temperature_std": float(daily["temperature_std"].median()),
                "temperature_range": float(daily["temperature_range"].median()),
                "max_minus_mean_temp": float(
                    (daily["max_temperature"] - daily["mean_temperature"]).median()),
                "mean_minus_min_temp": float(
                    (daily["mean_temperature"] - daily["min_temperature"]).median()),
                "rh_std": float(daily["rh_std"].median()),
                "rh_range": float(daily["rh_range"].median()),
                "max_minus_mean_rh": float(
                    (daily["max_rh"] - daily["mean_rh"]).median()),
                "mean_minus_min_rh": float(
                    (daily["mean_rh"] - daily["min_rh"]).median()),
                "compressor_cycle_count": float(
                    daily["compressor_cycle_count"].median()),
            }
            daily_history = (
                daily.set_index("storage_day", drop=False)[prep["feature_columns"]])
        except Exception:
            pass  # keep documented fallbacks
    _STATE["daily_stats"] = stats
    _STATE["daily_history"] = daily_history

    return _STATE


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def _build_day_features(
    temperature: float,
    humidity: float,
    storage_duration: float,
    compressor_activity: float,
    fridge_opening_count: int,
) -> pd.DataFrame:
    """Reconstruct the 17 daily-aggregate features from the 5 user inputs."""
    state = load_model()
    stats = state["daily_stats"]

    # Fraction of the day's readings expected above the safe limit, modelling
    # readings as Normal(user_temp, training_std) — consistent with how the
    # fridge's natural cycling produced excursions in the training data.
    sigma = max(stats["temperature_std"], 1e-6)
    z = (state["safe_temp"] - temperature) / (sigma * math.sqrt(2.0))
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
    feature_columns = state["preprocessor"]["feature_columns"]
    return pd.DataFrame([row])[feature_columns]


def preprocess_input(
    temperature: float,
    humidity: float,
    storage_duration: float,
    compressor_activity: float,
    fridge_opening_count: int,
):
    """
    Build the model input using the SAVED StandardScaler (transform only —
    never refit) and the saved feature order.

    Returns scaled flat features for tree models, or the three-input tensor
    set [(1, 3, 17), (1, 12), (1, 5)] when serving AC-CSN.
    """
    state = load_model()
    prep = state["preprocessor"]
    feature_columns = prep["feature_columns"]

    frame = _build_day_features(
        temperature, humidity, storage_duration,
        compressor_activity, fridge_opening_count,
    )
    scaled = prep["scaler"].transform(frame)

    if state["model_name"] != "AC-CSN":
        return scaled

    # AC-CSN sequence: real recorded history for the two preceding days
    # (clamped to the experiment's day range), current day from user input.
    seq_len = prep["sequence_length"]
    history = state["daily_history"]
    rows = []
    for offset in range(seq_len - 1, 0, -1):
        if history is not None:
            day = int(min(max(storage_duration - offset, 0), history.index.max()))
            rows.append(history.loc[[day]][feature_columns])
        else:
            rows.append(frame)  # no history file available — degrade gracefully
    rows.append(frame)
    seq = pd.concat(rows, ignore_index=True)
    x_seq = prep["scaler"].transform(seq).reshape(
        1, seq_len, len(feature_columns)).astype(np.float32)

    env_idx = [feature_columns.index(f) for f in prep["environmental_features"]]
    ops_idx = [feature_columns.index(f) for f in prep["operational_features"]]
    return [x_seq,
            scaled[:, env_idx].astype(np.float32),
            scaled[:, ops_idx].astype(np.float32)]


# ---------------------------------------------------------------------------
# Prediction API (reusable from FastAPI/Flask)
# ---------------------------------------------------------------------------
def predict_spoilage(
    temperature: float,
    humidity: float,
    storage_duration: float,
    compressor_activity: float,
    fridge_opening_count: int,
) -> dict:
    """
    Run spoilage inference for one cold-chain observation.

    Returns a JSON-serializable dict with class probabilities, weighted
    spoilage risk (0-100 %), status, and prediction confidence.
    """
    state = load_model()
    prep = state["preprocessor"]
    inputs = preprocess_input(
        temperature, humidity, storage_duration,
        compressor_activity, fridge_opening_count,
    )

    try:
        if state["model_name"] == "AC-CSN":
            proba = state["model"].predict(inputs, verbose=0)[0]
        else:
            proba = state["model"].predict_proba(inputs)[0]
    except Exception as exc:
        raise PredictionError(
            f"ERROR: prediction failed ({exc.__class__.__name__}: {exc}). "
            "The model input shape may be incompatible with the saved model."
        ) from exc

    class_labels = prep["class_labels"]  # trained mapping: index -> label
    label_index = {label: i for i, label in enumerate(class_labels)}
    predicted_class = int(np.argmax(proba))

    risk = compute_spoilage_risk(proba, class_labels, prep["risk_weights"])

    return {
        "temperature": round(float(temperature), 2),
        "humidity": round(float(humidity), 2),
        "storage_duration_days": int(storage_duration),
        "compressor_activity": round(float(compressor_activity), 1),
        "fridge_opening_count": int(fridge_opening_count),
        "fresh_probability": round(float(proba[label_index["FRESH"]]), 4),
        "warning_probability": round(float(proba[label_index["WARNING"]]), 4),
        "spoiled_probability": round(float(proba[label_index["SPOILED"]]), 4),
        "spoilage_risk": round(risk, 1),
        "status": class_labels[predicted_class],
        "predicted_class": predicted_class,
        "confidence": round(float(np.max(proba)), 4),
        "model_used": state["model_name"],
    }


# ---------------------------------------------------------------------------
# Terminal UI
# ---------------------------------------------------------------------------
def _ask(prompt: str, error_msg: str, cast, check) -> float:
    """Prompt until the input parses with `cast` and satisfies `check`."""
    while True:
        raw = input(prompt).strip()
        try:
            value = cast(raw)
            if check(value):
                return value
        except (ValueError, TypeError):
            pass
        print(error_msg)


def collect_user_inputs() -> dict:
    """Interactive, validated collection of the five cold-chain inputs."""
    return {
        "temperature": _ask(
            "Enter Temperature (°C): ",
            "Invalid temperature.\nPlease enter a numeric value between -30 and 40 °C.",
            float, lambda v: -30.0 <= v <= 40.0),
        "humidity": _ask(
            "Enter Humidity (%): ",
            "Invalid humidity.\nPlease enter a numeric value between 0 and 100.",
            float, lambda v: 0.0 <= v <= 100.0),
        "storage_duration": _ask(
            "Enter Storage Duration (days): ",
            "Invalid storage duration.\nPlease enter a numeric value >= 0.",
            float, lambda v: v >= 0),
        "compressor_activity": _ask(
            "Enter Compressor Activity (%): ",
            "Invalid compressor activity.\nPlease enter a numeric value between 0 and 100.",
            float, lambda v: 0.0 <= v <= 100.0),
        "fridge_opening_count": _ask(
            "Enter Fridge Opening Count: ",
            "Invalid fridge opening count.\nPlease enter a whole number >= 0.",
            int, lambda v: v >= 0),
    }


def display_prediction(result: dict) -> None:
    """Render the Cold Chain Analysis report from a real prediction."""
    print()
    print("┌─────────────────────────────────┐")
    print("│       COLD CHAIN ANALYSIS       │")
    print("├─────────────────────────────────┤")
    print(f"│ Temperature:     {result['temperature']:>7.1f} °C     │")
    print(f"│ Humidity:        {result['humidity']:>7.1f} %      │")
    print(f"│ Storage duration:{result['storage_duration_days']:>5d} days     │")
    print("├─────────────────────────────────┤")
    print(f"│ Spoilage Risk:   {result['spoilage_risk']:>7.1f} %      │")
    print(f"│ Status:          {result['status']:<12s}   │")
    print("│                                 │")
    print("│ Predicted class:                │")
    print(f"│   Fresh    {result['fresh_probability'] * 100:>6.1f} %             │")
    print(f"│   Warning  {result['warning_probability'] * 100:>6.1f} %             │")
    print(f"│   Spoiled  {result['spoiled_probability'] * 100:>6.1f} %             │")
    print("└─────────────────────────────────┘")
    print(f"  Confidence: {result['confidence'] * 100:.1f} %  |  Model: {result['model_used']}")


def main() -> None:
    print("\n========== AgriChainAI Spoilage Prediction ==========\n")
    try:
        load_model()  # fail fast with a clear message before asking for input
        user_inputs = collect_user_inputs()
        print("\nPredicting...")
        result = predict_spoilage(**user_inputs)
        display_prediction(result)
    except PredictionError as exc:
        print(f"\n{exc}")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: unexpected failure ({exc.__class__.__name__}). "
              "Set SPOILAGE_DEBUG=1 for details.")
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
