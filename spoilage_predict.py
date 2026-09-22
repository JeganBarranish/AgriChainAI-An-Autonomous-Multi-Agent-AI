"""
Paneer Cold-Chain Spoilage — Inference Interface

Loads the trained spoilage models (best baseline + AC-CSN), the fitted
preprocessor/scaler, and model metadata, and provides a reusable
prediction API for CLI use or future FastAPI/Flask integration.

CLI usage:
    python3 spoilage_predict.py            # predicts on a real processed sample

Programmatic usage:
    from spoilage_predict import SpoilagePredictor
    predictor = SpoilagePredictor()
    result = predictor.predict_spoilage(input_data)   # dict of feature values
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
AC_CSN_PATH = MODELS_DIR / "ac_csn_spoilage_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "spoilage_preprocessor.pkl"
BEST_MODEL_INFO_PATH = MODELS_DIR / "best_model_info.json"
BEST_BASELINE_PATH = MODELS_DIR / "best_spoilage_model.pkl"
SUPERVISED_PATH = PROJECT_ROOT / "dataset" / "Paneer_ColdChain_Supervised.csv"


class SpoilagePredictor:
    """
    Reusable spoilage prediction interface.

    Uses the best model selected during training (see best_model_info.json).
    If the best model is the AC-CSN neural network, temporal sequences are
    reconstructed from the supervised dataset; otherwise the flat-feature
    baseline model is used.

    Spoilage risk formula (documented, weighted by class severity):
        risk % = 0 * P(FRESH) + 50 * P(WARNING) + 100 * P(SPOILED)
    """

    RISK_WEIGHTS = np.array([0.0, 50.0, 100.0])

    def __init__(self) -> None:
        with PREPROCESSOR_PATH.open("rb") as fh:
            self.preprocessor = pickle.load(fh)
        with BEST_MODEL_INFO_PATH.open("r") as fh:
            self.best_info = json.load(fh)

        self.feature_columns = self.preprocessor["feature_columns"]
        self.class_labels = self.preprocessor["class_labels"]
        self.scaler = self.preprocessor["scaler"]
        self.best_name = self.best_info["best_model"]

        if self.best_name == "AC-CSN" or not BEST_BASELINE_PATH.exists():
            import tensorflow as tf  # deferred import — only needed for NN inference

            self.model = tf.keras.models.load_model(AC_CSN_PATH)
            self.model_kind = "ac_csn"
        else:
            with BEST_BASELINE_PATH.open("rb") as fh:
                self.model = pickle.load(fh)
            self.model_kind = "baseline"

    def _predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for one or more feature rows."""
        x_flat = self.scaler.transform(features[self.feature_columns])

        if self.model_kind == "baseline":
            return self.model.predict_proba(x_flat)

        # AC-CSN path: rebuild the 3-day sequence from the supervised dataset
        seq_len = self.preprocessor["sequence_length"]
        env_cols = self.preprocessor["environmental_features"]
        ops_cols = self.preprocessor["operational_features"]
        env_idx = [self.feature_columns.index(f) for f in env_cols]
        ops_idx = [self.feature_columns.index(f) for f in ops_cols]

        daily = (
            pd.read_csv(SUPERVISED_PATH)
            .drop_duplicates("storage_day")
            .set_index("storage_day", drop=False)[self.feature_columns]
        )
        sequences = []
        for _, row in features.iterrows():
            day = int(row["storage_day"])
            seq_days = [max(0, min(day, daily.index.max()) - o)
                        for o in range(seq_len - 1, -1, -1)]
            seq = daily.loc[seq_days].values.astype(np.float32)
            flat = self.scaler.transform(seq)
            sequences.append(flat)
        x_seq = np.stack(sequences)

        return self.model.predict(
            [x_seq, x_flat[:, env_idx], x_flat[:, ops_idx]], verbose=0)

    def predict_spoilage(self, input_data: dict | pd.DataFrame) -> dict:
        """
        Predict spoilage for one cold-chain observation.

        input_data: dict (or single-row DataFrame) containing the cold-chain
        feature columns listed in models/spoilage_preprocessor.pkl.
        """
        if isinstance(input_data, dict):
            features = pd.DataFrame([input_data])
        else:
            features = input_data.head(1).copy()

        proba = self._predict_proba(features)[0]
        predicted_class = int(np.argmax(proba))
        risk = float(np.dot(proba, self.RISK_WEIGHTS))

        return {
            "temperature": round(float(features["mean_temperature"].iloc[0]), 2),
            "humidity": round(float(features["mean_rh"].iloc[0]), 2),
            "storage_duration_days": int(features["storage_day"].iloc[0]),
            "spoilage_risk": round(risk, 1),
            "status": self.class_labels[predicted_class],
            "fresh_probability": round(float(proba[0]), 4),
            "warning_probability": round(float(proba[1]), 4),
            "spoiled_probability": round(float(proba[2]), 4),
            "predicted_class": predicted_class,
            "model_used": self.best_name,
        }


def print_analysis(result: dict) -> None:
    """Render the Cold Chain Analysis box from a real prediction result."""
    lines = [
        "┌─────────────────────────────────────┐",
        "│         COLD CHAIN ANALYSIS         │",
        "├─────────────────────────────────────┤",
        f"│ Temperature:       {result['temperature']:>6.1f} °C       │",
        f"│ Humidity:          {result['humidity']:>6.1f} %        │",
        f"│ Storage duration:  {result['storage_duration_days']:>4d} days       │",
        "├─────────────────────────────────────┤",
        f"│ Spoilage Risk:     {result['spoilage_risk']:>6.1f} %        │",
        f"│ Status:            {result['status']:<12s}     │",
        "│                                     │",
        "│ Class probabilities:                │",
        f"│   Fresh    {result['fresh_probability'] * 100:>6.1f} %                 │",
        f"│   Warning  {result['warning_probability'] * 100:>6.1f} %                 │",
        f"│   Spoiled  {result['spoiled_probability'] * 100:>6.1f} %                 │",
        "└─────────────────────────────────────┘",
        f"  Model: {result['model_used']}",
    ]
    print("\n".join(lines))


def main() -> None:
    predictor = SpoilagePredictor()
    print(f"Loaded best model: {predictor.best_name}\n")

    # Run inference on a real processed sample (mid-storage observation)
    df = pd.read_csv(SUPERVISED_PATH)
    sample = df[df["storage_day"] == 8].head(1)
    if sample.empty:
        sample = df.tail(1)

    result = predictor.predict_spoilage(sample)
    print_analysis(result)
    print("\nRaw prediction output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
