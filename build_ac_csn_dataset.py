"""
AC-CSN Merged Dataset Builder — AgriChainAI Paneer Cold-Chain

Creates ONE clean, leakage-aware training dataset by combining:
  - dataset/20241118 DataLog.csv        (high-frequency cold-chain sensor log)
  - dataset/20241230 Testing Data.xlsx  (Paneer quality observations)

Strategy: TEMPORAL AGGREGATION + STORAGE-DAY BASED FEATURE MERGING.
Sensor readings are aggregated to daily cold-chain features, then joined
to Paneer replicates on storage_day. Original datasets are never modified.

Target labels: the project already has a validated target-generation
method (spoilage_preprocessing.assign_spoilage_class, documented in
results/target_definition.json — TPC log10 + sensory acceptability
thresholds). That existing method is PRESERVED here; no new thresholds
are invented.

Outputs:
  data/ac_csn_merged_dataset.csv
  data/ac_csn_feature_metadata.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the project's validated target method and configuration — do NOT
# invent a new one (requirement 8).
from spoilage_preprocessing import (
    TARGET_CONFIG,
    assign_spoilage_class,
    load_quality_data,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SENSOR_PATH = PROJECT_ROOT / "dataset" / "20241118 DataLog.csv"
QUALITY_PATH = PROJECT_ROOT / "dataset" / "20241230 Testing Data.xlsx"
DATA_DIR = PROJECT_ROOT / "data"
MERGED_PATH = DATA_DIR / "ac_csn_merged_dataset.csv"
METADATA_PATH = DATA_DIR / "ac_csn_feature_metadata.csv"

# Existing project threshold (results/target_definition.json:
# safe_temperature_c = 4.0, FSSAI/industry cold-chain limit for dairy).
SAFE_TEMPERATURE_C = TARGET_CONFIG["safe_temperature_c"]

# Feature order expected by the AC-CSN training code
# (spoilage_preprocessing.FEATURE_COLUMNS). Training selects columns BY NAME,
# so CSV column order is not load-bearing, but we match it exactly anyway.
AC_CSN_FEATURE_ORDER = [
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

# Quality/testing columns that must NEVER be model inputs (leakage guard)
EXCLUDED_QUALITY_COLUMNS = [
    "tpc_log10", "coliform_log10", "yeast_mold_log10", "psychrophilic_log10",
    "colour_appearance", "body_texture", "flavour", "overall_acceptability",
    "ph", "moisture_pct",
]


def hr(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Inspection
# ---------------------------------------------------------------------------
def inspect_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    hr("STEP 1 — DATASET INSPECTION")

    sensor = pd.read_csv(SENSOR_PATH)
    print(f"\nDataLog.csv shape: {sensor.shape}")
    print(f"Columns: {list(sensor.columns)}")
    print(f"Dtypes:\n{sensor.dtypes.to_string()}")
    print(f"Missing values:\n{sensor.isnull().sum().to_string()}")
    print(f"Duplicate rows: {sensor.duplicated().sum()}")

    sensor["timestamp"] = pd.to_datetime(sensor["Time_Stamp"], format="%d-%m-%Y %H:%M")
    print(f"Date/time range: {sensor['timestamp'].min()} → {sensor['timestamp'].max()}")

    gaps = sensor.sort_values("timestamp")["timestamp"].diff().dropna()
    print(f"Reading interval: median {gaps.median()}, "
          f"min {gaps.min()}, max {gaps.max()} "
          f"(~10-minute cadence: {'YES' if abs(gaps.median().total_seconds() - 600) < 120 else 'NO'})")

    quality = load_quality_data(QUALITY_PATH)
    print(f"\nTesting Data.xlsx (tidy replicate rows) shape: {quality.shape}")
    print(f"Columns: {list(quality.columns)}")
    print(f"Missing values:\n{quality.isnull().sum().to_string()}")
    print(f"Duplicate rows: {quality.duplicated().sum()}")
    print(f"Unique Day values: {sorted(quality['storage_day'].unique())}")
    print(f"Replicates per day:\n"
          f"{quality.groupby('storage_day')['replicate'].count().to_string()}")

    return sensor, quality


# ---------------------------------------------------------------------------
# 2-3. Storage day + daily aggregation
# ---------------------------------------------------------------------------
def aggregate_sensor(sensor: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hr("STEP 2-3 — STORAGE DAY + DAILY SENSOR AGGREGATION")

    sensor = sensor.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    first_date = sensor["timestamp"].dt.normalize().min()
    sensor["storage_day"] = (sensor["timestamp"].dt.normalize() - first_date).dt.days
    print(f"First sensor date (Day 0): {first_date.date()}")
    print(f"storage_day range: {sensor['storage_day'].min()} → {sensor['storage_day'].max()}")
    print(f"Readings per day:\n{sensor.groupby('storage_day').size().to_string()}")

    records = []
    for day, g in sensor.groupby("storage_day"):
        temps, rh = g["Temperature"].values, g["RH"].values
        comp, fridge = g["Compressor"].values, g["Fridge_Open"].values
        above = temps > SAFE_TEMPERATURE_C
        records.append({
            "storage_day": int(day),
            "mean_temperature": float(np.mean(temps)),
            "min_temperature": float(np.min(temps)),
            "max_temperature": float(np.max(temps)),
            "temperature_std": float(np.std(temps)),
            "temperature_range": float(np.max(temps) - np.min(temps)),
            "temperature_excursion_count": int(np.sum(above)),
            "time_above_safe_threshold_ratio": float(np.mean(above)),
            "mean_rh": float(np.mean(rh)),
            "min_rh": float(np.min(rh)),
            "max_rh": float(np.max(rh)),
            "rh_std": float(np.std(rh)),
            "rh_range": float(np.max(rh) - np.min(rh)),
            # Percentages per merge specification
            "compressor_on_ratio": float(np.mean(comp)) * 100.0,
            "compressor_cycle_count": int(np.sum(np.diff(comp, prepend=comp[0]) == 1)),
            "fridge_open_count": int(np.sum(fridge)),
            "fridge_open_ratio": float(np.mean(fridge)) * 100.0,
            "n_sensor_readings": len(g),
        })
    daily = pd.DataFrame(records).sort_values("storage_day").reset_index(drop=True)
    print(f"\nDaily aggregate table: {daily.shape[0]} days × {daily.shape[1]} columns")
    print(f"Safe-temperature threshold used: {SAFE_TEMPERATURE_C} °C "
          "(existing project threshold from results/target_definition.json)")
    return daily, sensor


# ---------------------------------------------------------------------------
# 4. RH correlation check
# ---------------------------------------------------------------------------
def check_rh_correlation(sensor: pd.DataFrame) -> float:
    hr("STEP 4 — TEMPERATURE ↔ RH RELATIONSHIP")
    corr = float(sensor["Temperature"].corr(sensor["RH"]))
    diff = (sensor["RH"] - sensor["Temperature"]).describe()
    print(f"Pearson correlation Temperature vs RH: {corr:.6f}")
    print(f"RH − Temperature offset: mean {diff['mean']:.3f}, "
          f"std {diff['std']:.3f}, min {diff['min']:.3f}, max {diff['max']:.3f}")
    if corr > 0.999:
        print("FINDING: RH is (near-)perfectly derived from Temperature in this "
              "log (constant offset). RH features are KEPT as valid physical "
              "cold-chain variables, but they carry no independent signal in "
              "THIS experiment — flagged for review.")
    return corr


# ---------------------------------------------------------------------------
# 5-8. Merge + existing validated target
# ---------------------------------------------------------------------------
def merge_datasets(daily: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    hr("STEP 5-8 — MERGE + EXISTING VALIDATED TARGET")

    sensor_days = set(daily["storage_day"])
    quality_days = set(quality["storage_day"])
    print(f"Sensor days:  {sorted(sensor_days)}")
    print(f"Quality days: {sorted(quality_days)}")
    print(f"Days in both (INNER merge): {sorted(sensor_days & quality_days)}")
    print(f"Sensor-only days dropped: {sorted(sensor_days - quality_days)}")

    merged = quality.merge(daily, on="storage_day", how="inner")

    # Existing validated target method (TPC log10 + sensory acceptability;
    # documented thresholds in results/target_definition.json). PRESERVED.
    merged["spoilage_class"] = merged.apply(
        lambda r: assign_spoilage_class(r, TARGET_CONFIG), axis=1)
    merged["spoilage_status"] = merged["spoilage_class"].map(
        {0: "FRESH", 1: "WARNING", 2: "SPOILED"})
    print("\nTarget: EXISTING validated method preserved "
          "(TPC >= 4.5 log10 OR acceptability <= 6.0 → SPOILED; "
          "TPC < 3.5 AND acceptability >= 7.5 → FRESH; else WARNING).")
    print(f"Class distribution:\n{merged['spoilage_status'].value_counts().to_string()}")

    return merged


# ---------------------------------------------------------------------------
# 9. Leakage check + final column selection
# ---------------------------------------------------------------------------
def leakage_check_and_select(merged: pd.DataFrame) -> pd.DataFrame:
    hr("STEP 9 — DATA LEAKAGE CHECK")

    final_columns = (["storage_day", "Rep"]
                     + [c for c in AC_CSN_FEATURE_ORDER if c != "storage_day"]
                     + ["spoilage_class", "spoilage_status"])
    final = (merged
             .rename(columns={"replicate": "Rep"})
             [final_columns + ["n_sensor_readings"]]
             .copy())

    leaked = [c for c in EXCLUDED_QUALITY_COLUMNS if c in final.columns]
    print(f"Input features:\n{AC_CSN_FEATURE_ORDER}")
    print(f"\nExcluded quality variables (target-defining, never inputs):\n"
          f"{EXCLUDED_QUALITY_COLUMNS}")
    print("\nTarget:\n['spoilage_class', 'spoilage_status'] — derived from "
          "tpc_log10 + overall_acceptability (both excluded from inputs)")
    print("\nMetadata (not ML features): ['Rep', 'n_sensor_readings']")
    assert not leaked, f"LEAKAGE: quality columns present in final dataset: {leaked}"
    print("\nLeakage check PASSED: no target-defining quality column is "
          "present in the final dataset.")
    return final


# ---------------------------------------------------------------------------
# 12. Metadata file
# ---------------------------------------------------------------------------
def build_metadata() -> pd.DataFrame:
    S, Q = "20241118 DataLog.csv", "20241230 Testing Data.xlsx"
    rows = [
        ("storage_day", "Elapsed days since first sensor date (Day 0)", S,
         "(date - first_date).days", "days", "yes"),
        ("Rep", "Paneer replicate number within a storage day", Q,
         "as recorded", "-", "no (grouping metadata only)"),
        ("mean_temperature", "Daily mean storage temperature", S, "mean(Temperature)", "degC", "yes"),
        ("min_temperature", "Daily minimum temperature", S, "min(Temperature)", "degC", "yes"),
        ("max_temperature", "Daily maximum temperature", S, "max(Temperature)", "degC", "yes"),
        ("temperature_std", "Daily temperature standard deviation", S, "std(Temperature)", "degC", "yes"),
        ("temperature_range", "Daily temperature spread", S, "max - min", "degC", "yes"),
        ("temperature_excursion_count", "Readings above safe threshold", S,
         f"count(Temperature > {SAFE_TEMPERATURE_C})", "readings", "yes"),
        ("time_above_safe_threshold_ratio", "Fraction of day above safe threshold", S,
         f"mean(Temperature > {SAFE_TEMPERATURE_C})", "fraction 0-1", "yes"),
        ("mean_rh", "Daily mean relative humidity", S, "mean(RH)", "%RH", "yes"),
        ("min_rh", "Daily minimum RH", S, "min(RH)", "%RH", "yes"),
        ("max_rh", "Daily maximum RH", S, "max(RH)", "%RH", "yes"),
        ("rh_std", "Daily RH standard deviation", S, "std(RH)", "%RH", "yes"),
        ("rh_range", "Daily RH spread", S, "max - min", "%RH", "yes"),
        ("compressor_on_ratio", "Share of readings with compressor ON", S,
         "count(Compressor==1)/total * 100", "percent 0-100", "yes"),
        ("compressor_cycle_count", "Compressor OFF->ON transitions per day", S,
         "count(diff(Compressor)==1)", "cycles", "yes"),
        ("fridge_open_count", "Door-open readings per day", S, "count(Fridge_Open==1)", "readings", "yes"),
        ("fridge_open_ratio", "Share of readings with door open", S,
         "count(Fridge_Open==1)/total * 100", "percent 0-100", "yes"),
        ("n_sensor_readings", "Sensor readings aggregated for the day", S, "count", "readings",
         "no (audit metadata only)"),
        ("spoilage_class", "Target: 0=FRESH 1=WARNING 2=SPOILED", Q,
         "existing validated method (TPC log10 + overall acceptability)", "-", "no (TARGET)"),
        ("spoilage_status", "Target label text", Q, "class name mapping", "-", "no (TARGET)"),
    ]
    return pd.DataFrame(rows, columns=[
        "feature_name", "meaning", "source_dataset", "calculation", "unit", "model_input"])


# ---------------------------------------------------------------------------
# 13-14. Summary + merge verification
# ---------------------------------------------------------------------------
def summarize_and_verify(final: pd.DataFrame, sensor: pd.DataFrame,
                         quality: pd.DataFrame) -> None:
    hr("STEP 13 — FINAL DATASET SUMMARY")
    print(f"Rows: {len(final)} | Columns: {len(final.columns)}")
    print(f"Column names: {list(final.columns)}")
    print(f"Missing values (total): {final.isnull().sum().sum()}")
    if final.isnull().sum().sum():
        print(final.isnull().sum()[lambda s: s > 0].to_string())
    print(f"Storage-day range: {final['storage_day'].min()} → {final['storage_day'].max()}")
    print(f"Replicates per day:\n{final.groupby('storage_day')['Rep'].count().to_string()}")
    print(f"Sensor observations used: "
          f"{sensor[sensor['storage_day'].isin(final['storage_day'].unique())].shape[0]:,} "
          f"of {len(sensor):,} total")
    print(f"Sensor observations per merged day:\n"
          f"{final.drop_duplicates('storage_day').set_index('storage_day')['n_sensor_readings'].to_string()}")
    print(f"Merged Paneer observations: {len(final)} (of {len(quality)} quality rows)")

    print("\nFINAL AC-CSN FEATURES")
    ordered = ["storage_day"] + [f for f in AC_CSN_FEATURE_ORDER if f != "storage_day"]
    for i, f in enumerate(ordered, 1):
        print(f"{i:>2}. {f}")

    hr("STEP 14 — MERGE VERIFICATION (sample rows)")
    cols = ["storage_day", "Rep", "mean_temperature", "mean_rh",
            "compressor_on_ratio", "fridge_open_count"]
    print(final[cols].head(12).to_string(index=False))
    same = (final.groupby("storage_day")[AC_CSN_FEATURE_ORDER]
            .nunique().max().max())
    print(f"\nAll replicates of a day share identical daily sensor features: "
          f"{'YES' if same == 1 else 'NO — PROBLEM'}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    sensor, quality = inspect_datasets()
    daily, sensor = aggregate_sensor(sensor)
    check_rh_correlation(sensor)
    merged = merge_datasets(daily, quality)
    final = leakage_check_and_select(merged)

    final.to_csv(MERGED_PATH, index=False)
    metadata = build_metadata()
    metadata.to_csv(METADATA_PATH, index=False)

    summarize_and_verify(final, sensor, quality)

    hr("SAVED")
    print(f"Merged dataset  → {MERGED_PATH}")
    print(f"Feature metadata → {METADATA_PATH}")
    print("\nOriginal datasets were NOT modified.")


if __name__ == "__main__":
    main()
