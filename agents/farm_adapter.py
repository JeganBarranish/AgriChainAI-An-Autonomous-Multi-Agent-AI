"""
Farm Agent adapter.

Wraps the EXISTING farm_agent.FarmAgent. Nothing here re-implements crop
prediction: it (1) trains the agent once (FarmAgent has no persistence API),
(2) derives the input form from the agent's ACTIVE dataset spec, and
(3) calls FarmAgent.recommend() and normalises the dict it returns.
"""

from __future__ import annotations

import contextlib
import io
import threading
from datetime import date

from agents.fields import Field, parse_value

# Physical sanity limits for the crop_reco features (UI validation only).
_HARD = {
    "N": (0, 300, "kg/ha"), "P": (0, 300, "kg/ha"), "K": (0, 400, "kg/ha"),
    "ph": (2, 12, ""), "temperature": (-10, 60, "°C"),
    "humidity": (0, 100, "%"), "rainfall": (0, 1500, "mm"),
}
_LABELS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "ph": "Soil pH",
    "temperature": "Temperature", "humidity": "Relative humidity", "rainfall": "Rainfall",
}


def initialize_farm_agent():
    """Train the existing FarmAgent once. Returns (agent, training_report).

    Uses the agent's own defaults (crop_reco dataset, 150 epochs, validation
    hyper-parameter search). Only the k-fold CV *report* is skipped, because it
    does not change the deployed model. Console output of train() is captured.
    """
    from farm_agent import FarmAgent

    agent = FarmAgent()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report = agent.train(data_source="crop_reco", search=True, verbose=False, cv_folds=0)
    return agent, report


class FarmService:
    """Holds the trained FarmAgent plus its initialisation status."""

    api_name = "FarmAgent.recommend()"
    agent_name = "Farm Agent"

    def __init__(self, agent=None, report=None, error: str | None = None, ranges: dict | None = None):
        self.agent, self.report, self.error = agent, report, error
        self._ranges = ranges or {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def initialize(cls) -> "FarmService":
        try:
            agent, report = initialize_farm_agent()
        except Exception as exc:  # shown verbatim in the sidebar
            return cls(error=f"{type(exc).__name__}: {exc}")
        ranges = {}
        try:
            from dataset import load_crop_reco
            df, spec = load_crop_reco()
            df = df[df[spec.label_col].isin(agent.crop_list)]
            for c in spec.scaled_numeric:
                ranges[c] = (float(df[c].min()), float(df[c].max()))
        except Exception:
            pass  # soft-range warnings are optional
        return cls(agent=agent, report=report, ranges=ranges)

    @property
    def status(self) -> str:
        return "Error" if self.error or self.agent is None or self.agent.model is None else "Ready"

    # --------------------------------------------------------------- schema
    def fields(self) -> list[Field]:
        """Input form derived from the ACTIVE FarmAgent.spec."""
        spec = self.agent.spec
        if spec.name == "crop_reco":
            out = []
            for key in spec.scaled_numeric:
                lo, hi, unit = _HARD[key]
                out.append(Field(
                    key, _LABELS[key], "float",
                    f"{_LABELS[key]}{f' ({unit})' if unit else ''}?",
                    minimum=lo, maximum=hi, unit=unit, soft_range=self._ranges.get(key)))
            out.append(Field("land_area_ha", "Land area", "float", "Land area (hectares)?",
                             minimum=0.01, maximum=10000, unit="ha"))
            out.append(Field("sowing_date", "Sowing date", "date",
                             "Planned sowing date (YYYY-MM-DD)? Type 'skip' to use today's date.",
                             optional=True, skip_note="today's date"))
            return out
        if spec.name == "legacy":
            from dataset import REGIONS, SOIL_TYPES
            return [
                Field("region", "Region", "choice", f"Region? ({', '.join(REGIONS)})", choices=tuple(REGIONS)),
                Field("soil_type", "Soil type", "choice", f"Soil type? ({', '.join(SOIL_TYPES)})", choices=tuple(SOIL_TYPES)),
                Field("month", "Sowing month", "int", "Sowing month (1-12)?", minimum=1, maximum=12),
                Field("irrigation_available", "Irrigation", "bool", "Is irrigation available? (yes/no)"),
                Field("fertilizer_planned", "Fertilizer", "bool", "Will fertilizer be used? (yes/no)"),
                Field("water_availability_mm", "Water availability", "float",
                      "Water availability in mm? Type 'skip' to use the month's seasonal default.",
                      minimum=0, maximum=3000, unit="mm", optional=True, skip_note="seasonal default for the month"),
                Field("land_area_ha", "Land area", "float", "Land area (hectares)?", minimum=0.01, maximum=10000, unit="ha"),
            ]
        raise RuntimeError(f"Unsupported FarmAgent spec '{spec.name}'")

    # ----------------------------------------------------------- invocation
    def recommend(self, collected: dict) -> dict:
        """Call FarmAgent.recommend() with validated inputs; normalise the result."""
        kwargs = {k: v for k, v in collected.items() if v is not None}
        if kwargs.get("sowing_date") is None:
            kwargs.pop("sowing_date", None)
        with self._lock:
            raw = self.agent.recommend(**kwargs)
        best = raw["best_crop"]
        return {
            "dataset": self.report.get("source") if self.report else self.agent.spec.name,
            "inputs": dict(collected),
            "best": best,
            "alternatives": raw["alternatives"],
            "raw": raw,
            "caveats": caveats(best, self.agent.spec),
        }


def caveats(best: dict, spec) -> list[str]:
    out = []
    if best.get("estimate_source") == "reference_table_estimate":
        out.append("Expected yield and profit come from the government reference table (MSP/CACP), "
                   "not from a yield prediction by the model: this dataset has no yield column.")
    elif best.get("estimate_source") == "model_prediction":
        out.append("Expected yield is the model's own prediction.")
    out.append("Confidence is the classifier's probability for the crop; the recommended crop is chosen "
               "by expected profit among plausible candidates, so it need not be the most probable crop.")
    if spec.name == "crop_reco":
        out.append("The model only knows 8 crops: rice, maize, cotton, pigeonpeas, mungbean, blackgram, chickpea, jute.")
    return out
