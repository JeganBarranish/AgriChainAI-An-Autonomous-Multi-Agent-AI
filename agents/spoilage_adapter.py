"""
Spoilage Agent adapters.

Wrap the EXISTING spoilage.predictor.predict_paneer / predict_dryfruit. Model
loading, preprocessing, class order and the returned probability/risk values
are all produced by those functions; this module only (1) checks artifact
availability, (2) defines the input forms, (3) validates, and (4) adds a
cautious interpretation next to -- never in place of -- the model output.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from agents.fields import Field

ROOT = Path(__file__).resolve().parents[1]


def _range(csv: Path, col: str):
    try:
        import pandas as pd
        s = pd.read_csv(csv)[col].dropna()
        return float(s.min()), float(s.max())
    except Exception:
        return None


class _Base:
    agent_name = "Spoilage Agent"
    product_type = ""
    status = "Initializing"
    error: str | None = None

    def check(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _set(self, status, error=None):
        self.status, self.error = status, error


class PaneerService(_Base):
    product_type = "paneer"
    api_name = "spoilage.predictor.predict_paneer()"

    def __init__(self):
        self._soft = {}
        self.check()

    def check(self):
        try:
            from spoilage import predictor as P
            prep, best_path, use_ac_csn = P._resolve_paneer_artifacts()
            if use_ac_csn:
                import tensorflow  # noqa: F401  (only needed when no XGBoost pickle exists)
                if not (P.PANEER_MODEL.exists() or P.PANEER_LEGACY_MODEL.exists()):
                    raise FileNotFoundError("No Paneer model artifact found in models/")
            else:
                P._load_pickle(best_path)
            sup = ROOT / "dataset" / "Paneer_ColdChain_Supervised.csv"
            self._soft = {
                "storage_duration": _range(sup, "storage_day"),
                "temperature": _range(sup, "mean_temperature"),
                "humidity": _range(sup, "mean_rh"),
            }
            self._set("Ready")
        except Exception as exc:
            self._set("Error", f"{type(exc).__name__}: {exc}")
        return self.status

    def fields(self, collected: dict | None = None) -> list[Field]:
        s = self._soft
        return [
            Field("storage_duration", "Storage duration", "float", "Storage duration (days)?",
                  minimum=0, maximum=365, unit="days", soft_range=s.get("storage_duration")),
            Field("temperature", "Temperature", "float", "Average storage temperature (°C)?",
                  minimum=-30, maximum=40, unit="°C", soft_range=s.get("temperature")),
            Field("humidity", "Humidity", "float", "Average relative humidity (%)?",
                  minimum=0, maximum=100, unit="%", soft_range=s.get("humidity")),
            Field("compressor_activity", "Compressor activity", "float",
                  "Compressor activity (% of time ON)? Type 'skip' to use the training-data median.",
                  minimum=0, maximum=100, unit="%", optional=True, skip_note="training-data median"),
            Field("fridge_opening_count", "Fridge opening count", "int",
                  "Fridge door openings per day? Type 'skip' to use the training-data median.",
                  minimum=0, maximum=1000, optional=True, skip_note="training-data median"),
        ]

    def predict(self, collected: dict) -> dict:
        from spoilage.predictor import predict_paneer
        kwargs = {k: v for k, v in collected.items() if v is not None}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = predict_paneer(**kwargs)
        return normalise(result, collected, self._soft)


class DryFruitService(_Base):
    product_type = "dried_fruit"
    api_name = "spoilage.predictor.predict_dryfruit()"

    def __init__(self):
        self.valid = {}
        self._soft = {}
        self.check()

    def check(self):
        try:
            from spoilage import predictor as P
            if not P.DRYFRUIT_PREPROCESSOR.exists():
                self._set("Not configured", f"Missing artifact: {P.DRYFRUIT_PREPROCESSOR.name}")
                return self.status
            prep = P._load_pickle(P.DRYFRUIT_PREPROCESSOR)
            if prep.get("model_type") == "XGBoost" and P.DRYFRUIT_BEST.exists():
                P._load_pickle(P.DRYFRUIT_BEST)
            else:
                import tensorflow  # noqa: F401
                if not P.DRYFRUIT_MODEL.exists():
                    raise FileNotFoundError(f"Missing artifact: {P.DRYFRUIT_MODEL.name}")
            # Options come from the deployed preprocessing artifact, not hard-coded.
            self.valid = {"fruit": prep["valid_fruits"], "dryer": prep["valid_dryer_types"],
                          "packaging": prep["valid_packaging"]}
            self._soft = {"storage_duration_days": _range(ROOT / "dataset" / "ac_csn_multiproduct.csv", "storage_duration_days")}
            self._set("Ready")
        except Exception as exc:
            self._set("Error", f"{type(exc).__name__}: {exc}")
        return self.status

    def fields(self, collected: dict | None = None) -> list[Field]:
        v = self.valid
        return [
            Field("product_subtype", "Dried fruit", "choice", f"Which dried fruit? ({' / '.join(v['fruit'])})",
                  choices=tuple(v["fruit"])),
            Field("storage_duration_days", "Storage duration", "float", "Storage duration (days)?",
                  minimum=0, maximum=730, unit="days", soft_range=self._soft.get("storage_duration_days")),
            Field("dryer_type", "Dryer type", "choice",
                  f"Dryer type? ({' / '.join(v['dryer'])}) — CMD = cabinet mixed-mode solar dryer, TD = tunnel dryer",
                  choices=tuple(v["dryer"]),
                  aliases={"mixed": "CMD", "tunnel": "TD", "tunel": "TD"}),
            Field("packaging", "Packaging", "choice", f"Packaging condition? ({' / '.join(v['packaging'])})",
                  choices=tuple(v["packaging"]),
                  aliases={"no": "None", "no packaging": "None", "high density": "HDPE", "low density": "LDPE"}),
        ]

    def predict(self, collected: dict) -> dict:
        from spoilage.predictor import predict_dryfruit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = predict_dryfruit(**collected)
        return normalise(result, collected, self._soft)


def normalise(result: dict, collected: dict, soft: dict) -> dict:
    """Display structure around the UNMODIFIED model output."""
    status = result["status"]
    paneer = result["product_type"] == "paneer"
    inputs = {k: v for k, v in result.items() if k in (
        "storage_duration_days", "temperature", "humidity", "compressor_activity",
        "fridge_opening_count", "dryer_type", "packaging")}
    notes = []
    if paneer:
        for k, src in (result.get("inputs_used") or {}).items():
            if src != "user":
                notes.append(f"{k.replace('_', ' ').capitalize()} was not provided, so the training-data median was used.")
        notes.append("The existing Paneer predictor fills the daily variability features (temperature std/range, "
                     "compressor cycles) from training-data medians; only the values you entered are real inputs.")
    else:
        notes.append("Temperature and humidity are not available in the dried-fruit source data and are not model inputs.")
    meaning = {
        "FRESH": "The model rates this batch as most likely still FRESH under the inputs given.",
        "WARNING": "The model rates this batch as most likely in the WARNING stage: quality is declining; inspect it and prioritise it for use or sale.",
        "SPOILED": "The model rates this batch as most likely SPOILED: inspect it before any use or sale.",
    }[status]
    notes.append("Trained on a very small experimental dataset (single batch); treat this as decision support, "
                 "not a food-safety determination.")
    return {
        "product_type": result["product_type"], "product_subtype": result["product_subtype"],
        "inputs": inputs, "result": result,
        "probabilities": {"FRESH": result["fresh_probability"], "WARNING": result["warning_probability"],
                          "SPOILED": result["spoiled_probability"]},
        "status": status, "spoilage_risk": result["spoilage_risk"], "model_used": result["model_used"],
        "confidence": result["confidence"], "interpretation": meaning, "notes": notes,
    }
