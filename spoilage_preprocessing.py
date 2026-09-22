"""
Paneer Cold-Chain Spoilage — Data Preprocessing Pipeline

Loads real experimental datasets:
  - 20241118 DataLog.csv       (cold-chain sensor readings)
  - 20241230 Testing Data.xlsx (Paneer quality observations)

Synchronizes sensor aggregates with quality replicates, derives
scientifically documented spoilage target classes, and saves the
supervised dataset for model training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SENSOR_PATH = PROJECT_ROOT / "dataset" / "20241118 DataLog.csv"
QUALITY_PATH = PROJECT_ROOT / "dataset" / "20241230 Testing Data.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "Paneer_ColdChain_Supervised.csv"
TARGET_DEF_PATH = PROJECT_ROOT / "results" / "target_definition.json"

# Configurable spoilage thresholds (documented in target_definition.json)
TARGET_CONFIG = {
    "fresh_tpc_log10_max": 3.5,
    "spoiled_tpc_log10_min": 4.5,
    "fresh_acceptability_min": 7.5,
    "spoiled_acceptability_max": 6.0,
    "safe_temperature_c": 4.0,
}

CLASS_NAMES = {0: "FRESH", 1: "WARNING", 2: "SPOILED"}
CLASS_LABELS = ["FRESH", "WARNING", "SPOILED"]

# Columns used ONLY for target construction — never as model inputs
TARGET_SOURCE_COLUMNS = [
    "tpc_log10",
    "overall_acceptability",
    "coliform_log10",
    "yeast_mold_log10",
    "psychrophilic_log10",
    "ph",
    "moisture_pct",
    "colour_appearance",
    "body_texture",
    "flavour",
]

# Legitimate cold-chain prediction features
FEATURE_COLUMNS = [
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
    "compressor_on_ratio",
    "compressor_cycle_count",
    "fridge_open_count",
    "fridge_open_ratio",
    "storage_day",
]


def load_sensor_data(path: Path) -> pd.DataFrame:
    """Load and clean cold-chain sensor CSV."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Time_Stamp"], format="%d-%m-%Y %H:%M")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Assign storage day relative to experiment start (Day 0 = first calendar date)
    start_date = df["timestamp"].dt.normalize().min()
    df["storage_day"] = (df["timestamp"].dt.normalize() - start_date).dt.days

    return df


def aggregate_sensor_by_day(df: pd.DataFrame, safe_temp: float) -> pd.DataFrame:
    """Aggregate high-frequency sensor readings to daily cold-chain features."""
    records = []

    for day, group in df.groupby("storage_day"):
        temps = group["Temperature"].values
        rh = group["RH"].values
        compressor = group["Compressor"].values
        fridge_open = group["Fridge_Open"].values

        # Compressor cycle count: count 0→1 transitions
        comp_diff = np.diff(compressor, prepend=compressor[0])
        cycle_count = int(np.sum(comp_diff == 1))

        above_safe = temps > safe_temp

        records.append(
            {
                "storage_day": int(day),
                "mean_temperature": float(np.mean(temps)),
                "min_temperature": float(np.min(temps)),
                "max_temperature": float(np.max(temps)),
                "temperature_std": float(np.std(temps)),
                "temperature_range": float(np.max(temps) - np.min(temps)),
                "temperature_excursion_count": int(np.sum(above_safe)),
                "time_above_safe_threshold_ratio": float(np.mean(above_safe)),
                "mean_rh": float(np.mean(rh)),
                "min_rh": float(np.min(rh)),
                "max_rh": float(np.max(rh)),
                "rh_std": float(np.std(rh)),
                "rh_range": float(np.max(rh) - np.min(rh)),
                "compressor_on_ratio": float(np.mean(compressor)),
                "compressor_cycle_count": cycle_count,
                "fridge_open_count": int(np.sum(fridge_open)),
                "fridge_open_ratio": float(np.mean(fridge_open)),
            }
        )

    return pd.DataFrame(records).sort_values("storage_day").reset_index(drop=True)


def load_quality_data(path: Path) -> pd.DataFrame:
    """Parse the Excel quality/testing file into tidy replicate-level rows."""
    raw = pd.read_excel(path, sheet_name="Data", header=None)

    rows = []
    for i in range(2, len(raw)):
        day_val = raw.iloc[i, 0]
        repl_val = raw.iloc[i, 1]

        # Skip summary rows (Mean, SD) and blank rows
        if pd.isna(day_val) or str(repl_val).strip() in ("Mean", "SD", "nan"):
            continue
        try:
            day = int(day_val)
            replicate = int(repl_val)
        except (ValueError, TypeError):
            continue

        def _to_float(val):
            try:
                return float(val) if pd.notna(val) else np.nan
            except (ValueError, TypeError):
                return np.nan

        rows.append(
            {
                "storage_day": day,
                "replicate": replicate,
                "tpc_log10": _to_float(raw.iloc[i, 4]),
                "coliform_log10": _to_float(raw.iloc[i, 7]),
                "yeast_mold_log10": _to_float(raw.iloc[i, 10]),
                "psychrophilic_log10": _to_float(raw.iloc[i, 13]),
                "colour_appearance": _to_float(raw.iloc[i, 15]),
                "body_texture": _to_float(raw.iloc[i, 16]),
                "flavour": _to_float(raw.iloc[i, 17]),
                "overall_acceptability": _to_float(raw.iloc[i, 18]),
                "ph": _to_float(raw.iloc[i, 20]),
                "moisture_pct": _to_float(raw.iloc[i, 21]),
            }
        )

    return pd.DataFrame(rows)


def assign_spoilage_class(row: pd.Series, cfg: dict) -> int:
    """
    Derive 3-class spoilage label from microbiological + sensory evidence.

    Primary indicator: TPC log10 (Total Plate Count, cfu/ml).
    Secondary indicator: Overall Acceptability (9-point hedonic scale).

    Rules (configurable via TARGET_CONFIG):
      SPOILED (2): TPC >= spoiled_tpc_min  OR  acceptability <= spoiled_accept_max
      FRESH   (0): TPC < fresh_tpc_max  AND  acceptability >= fresh_accept_min
      WARNING (1): all other cases

    When sensory data is missing (days 10–13), classification relies on TPC only:
      SPOILED: TPC >= spoiled_tpc_min
      FRESH:   TPC < fresh_tpc_max
      WARNING: otherwise
    """
    tpc = row["tpc_log10"]
    accept = row["overall_acceptability"]

    if pd.notna(accept):
        if tpc >= cfg["spoiled_tpc_log10_min"] or accept <= cfg["spoiled_acceptability_max"]:
            return 2
        if tpc < cfg["fresh_tpc_log10_max"] and accept >= cfg["fresh_acceptability_min"]:
            return 0
        return 1

    # Sensory unavailable — TPC-only fallback
    if tpc >= cfg["spoiled_tpc_log10_min"]:
        return 2
    if tpc < cfg["fresh_tpc_log10_max"]:
        return 0
    return 1


def build_target_definition(cfg: dict) -> dict:
    """Document target methodology for reproducibility."""
    return {
        "classes": CLASS_LABELS,
        "class_encoding": {"FRESH": 0, "WARNING": 1, "SPOILED": 2},
        "methodology": (
            "Three-class spoilage labels derived from real Paneer quality measurements. "
            "Total Plate Count (TPC, log10 cfu/ml) is the primary microbiological indicator "
            "of microbial deterioration. Overall Acceptability (9-point hedonic scale) provides "
            "secondary sensory confirmation when available."
        ),
        "thresholds": {
            "fresh_tpc_log10_max": {
                "value": cfg["fresh_tpc_log10_max"],
                "rationale": (
                    "TPC log10 < 3.5 corresponds to early storage (Days 0–2) where mean TPC "
                    "remained below 2.8 and sensory acceptability exceeded 8.3 in the "
                    "experimental dataset, consistent with satisfactory microbiological quality "
                    "for fresh paneer."
                ),
            },
            "spoiled_tpc_log10_min": {
                "value": cfg["spoiled_tpc_log10_min"],
                "rationale": (
                    "TPC log10 >= 4.5 aligns with marked microbial proliferation observed "
                    "from Day 8 onward (mean TPC >= 4.49) and declining sensory scores, "
                    "consistent with ICMSF/food-microbiology spoilage thresholds for "
                    "refrigerated dairy products."
                ),
            },
            "fresh_acceptability_min": {
                "value": cfg["fresh_acceptability_min"],
                "rationale": (
                    "Overall acceptability >= 7.5 represents the lower bound of fresh-product "
                    "sensory scores observed during Days 0–3 (mean >= 8.0)."
                ),
            },
            "spoiled_acceptability_max": {
                "value": cfg["spoiled_acceptability_max"],
                "rationale": (
                    "Overall acceptability <= 6.0 corresponds to sensory rejection threshold "
                    "observed from Day 7 onward when panellists rated samples below acceptable "
                    "freshness."
                ),
            },
            "safe_temperature_c": {
                "value": cfg["safe_temperature_c"],
                "rationale": (
                    "4 °C is the standard cold-chain upper limit for refrigerated dairy "
                    "products (including paneer) per FSSAI/industry cold-chain guidelines."
                ),
            },
        },
        "source_columns_for_target": [
            "tpc_log10 (Total Plate Count, log10 cfu/ml)",
            "overall_acceptability (9-point hedonic scale, when available)",
        ],
        "fallback_when_sensory_missing": (
            "Days 10–13 lack individual replicate sensory scores in the source Excel file. "
            "For those samples, TPC-only rules apply."
        ),
        "leakage_prevention": (
            "Target source columns (TPC, sensory, pH, moisture) are excluded from model "
            "input features. The model predicts spoilage from cold-chain sensor aggregates only."
        ),
        "limitations": [
            "Small sample size (65 replicates across 14 storage days).",
            "Single experimental batch — generalization beyond this study is not claimed.",
            "Thresholds are configurable via TARGET_CONFIG and should be validated with "
            "domain experts before production deployment.",
            "No explicit study documentation was bundled with the dataset; thresholds are "
            "justified from observed data trends and standard food-microbiology practice.",
        ],
    }


def merge_and_label(
    quality: pd.DataFrame,
    sensor_daily: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Join quality replicates with daily sensor aggregates and assign target class."""
    merged = quality.merge(sensor_daily, on="storage_day", how="inner")
    merged["spoilage_class"] = merged.apply(lambda r: assign_spoilage_class(r, cfg), axis=1)
    merged["spoilage_status"] = merged["spoilage_class"].map(CLASS_NAMES)
    return merged


def run_preprocessing() -> pd.DataFrame:
    """Execute the full preprocessing pipeline."""
    TARGET_DEF_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading sensor data …")
    sensor = load_sensor_data(SENSOR_PATH)
    print(f"  {len(sensor):,} readings | {sensor['storage_day'].nunique()} storage days")

    print("Aggregating sensor features by storage day …")
    sensor_daily = aggregate_sensor_by_day(sensor, TARGET_CONFIG["safe_temperature_c"])

    print("Loading Paneer quality data …")
    quality = load_quality_data(QUALITY_PATH)
    print(f"  {len(quality):,} replicate observations")

    print("Merging and assigning spoilage classes …")
    supervised = merge_and_label(quality, sensor_daily, TARGET_CONFIG)

    class_counts = supervised["spoilage_status"].value_counts().to_dict()
    print(f"  Class distribution: {class_counts}")

    # Save outputs
    supervised.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved supervised dataset → {OUTPUT_PATH}")

    target_def = build_target_definition(TARGET_CONFIG)
    target_def["observed_class_distribution"] = class_counts
    target_def["total_samples"] = len(supervised)
    with TARGET_DEF_PATH.open("w", encoding="utf-8") as fh:
        json.dump(target_def, fh, indent=2)
    print(f"Saved target definition → {TARGET_DEF_PATH}")

    return supervised


if __name__ == "__main__":
    run_preprocessing()
