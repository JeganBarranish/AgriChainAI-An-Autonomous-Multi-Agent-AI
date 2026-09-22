"""
Dried mango/pineapple data processing for the Mendeley Tanzania dataset.

[UNSUPERVISED preprocessing + target assignment via target_builder]

Inference-time features (available without laboratory QC):
  - product_subtype (Mango / Pineapple)
  - storage_duration_days (derived from storage months in source)
  - dryer_type (CMD / TD — normalized from Mixed, Tunel, Tunnel)
  - packaging (None / HDPE / LDPE)

NOT available in source (never fabricated):
  - storage temperature
  - relative humidity
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from spoilage.data_loader import MICRO_PATH, PHYSICO_PATH

# Normalized categorical vocabularies
DRYER_TYPES = ["CMD", "TD"]  # cabinet mixed-mode / tunnel — excludes Fresh raw fruit
PACKAGING_TYPES = ["None", "HDPE", "LDPE"]
FRUITS = ["Mango", "Pineapple"]

DRYER_MAP = {
    "mixed": "CMD",
    "cabinet mixed-mode dryer (cmd)": "CMD",
    "tunel": "TD",
    "tunnel": "TD",
    "tunnel ": "TD",
}

PACKAGING_MAP = {
    "no": "None",
    "high density": "HDPE",
    "low density": "LDPE",
    "low density ": "LDPE",
}

TIME_MONTHS_MAP = {"zero": 0, "three": 3, "six": 6}
DAYS_PER_MONTH = 30  # project convention for month→day conversion (documented)


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def parse_cfu(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(",", "")
    if s.lower() in ("nil", "0", "-", "na", "nan"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def normalize_dryer(raw: str) -> str | None:
    key = _norm(raw)
    if key == "fresh":
        return "FRESH"
    return DRYER_MAP.get(key)


def normalize_packaging(raw: str) -> str:
    return PACKAGING_MAP.get(_norm(raw), _norm(raw))


def normalize_fruit(raw: str) -> str:
    return _norm(raw).title()


def load_physicochemical() -> pd.DataFrame:
    df = pd.read_excel(PHYSICO_PATH, sheet_name="COMBINED (MANGO AND PINEAPPLE)")
    df.columns = [c.strip() for c in df.columns]
    df["fruit"] = df["Fruit"].apply(normalize_fruit)
    df["dryer_type"] = df["Drying method"].apply(normalize_dryer)
    df = df.rename(columns={
        "Moisture Content (g/100 g WB": "moisture_pct",
        "Water Activity (Aw)": "water_activity",
        "pH": "ph",
        "TTA (g of ca/100 g DM)": "tta",
    })
    return df


def load_microbial_tpc() -> pd.DataFrame:
    df = pd.read_excel(MICRO_PATH, sheet_name="COMBINED (MANGO&PINEAPPLE)")
    df.columns = [c.strip() for c in df.columns]
    df["fruit"] = df["Fruit"].apply(normalize_fruit)
    df["dryer_type"] = df["Dryer type"].apply(normalize_dryer)
    df["packaging"] = df["Packaging material"].apply(normalize_packaging)
    df["storage_months"] = df["Time (Months)"].apply(lambda x: TIME_MONTHS_MAP.get(_norm(x)))
    df["organism"] = df["Microrganism"].str.strip()
    df["tpc_cfu_g"] = df["Microbial load (CFU/g)"].apply(parse_cfu)
    df = df[df["organism"] == "Total Plate Count"].copy()
    return df


def build_dryfruit_samples(include_fresh_fruit: bool = False) -> pd.DataFrame:
    """
    One row per experimental cell (fruit × dryer × packaging × storage month).
    Excludes raw Fresh fruit unless include_fresh_fruit=True.
    """
    tpc = load_microbial_tpc()
    if not include_fresh_fruit:
        tpc = tpc[tpc["dryer_type"] != "FRESH"].copy()

    # Aggregate replicates (typically 1 per cell in published tables)
    grouped = (
        tpc.groupby(["fruit", "dryer_type", "packaging", "storage_months"], as_index=False)
        .agg(tpc_cfu_g=("tpc_cfu_g", "mean"), replicate_count=("tpc_cfu_g", "count"))
    )
    grouped["tpc_log10"] = np.log10(grouped["tpc_cfu_g"].clip(lower=1.0))
    grouped["storage_duration_days"] = grouped["storage_months"] * DAYS_PER_MONTH

    # Attach reference physicochemical values by fruit+dryer (POST-PREDICTION metadata)
    phys = load_physicochemical()
    phys_ref = (
        phys.groupby(["fruit", "dryer_type"], as_index=False)
        .agg(
            ref_moisture_pct=("moisture_pct", "mean"),
            ref_water_activity=("water_activity", "mean"),
            ref_ph=("ph", "mean"),
        )
    )
    grouped = grouped.merge(phys_ref, on=["fruit", "dryer_type"], how="left")

    grouped["product_type"] = "dried_fruit"
    grouped["product_subtype"] = grouped["fruit"]
    grouped["group_id"] = grouped.apply(
        lambda r: f"{r['fruit']}|{r['dryer_type']}|{r['packaging']}", axis=1
    )
    return grouped


# Model input columns — NO target/leakage columns
DRYFRUIT_FEATURE_COLUMNS = [
    "storage_duration_days",
    "storage_months",
    "dryer_type_CMD",
    "dryer_type_TD",
    "packaging_None",
    "packaging_HDPE",
    "packaging_LDPE",
    "fruit_Mango",
    "fruit_Pineapple",
]

# Identifier / metadata (not model inputs)
DRYFRUIT_METADATA_COLUMNS = [
    "product_type", "product_subtype", "fruit", "dryer_type", "packaging",
    "group_id", "replicate_count",
    "ref_moisture_pct", "ref_water_activity", "ref_ph",
]

# Target sources — NEVER model inputs
DRYFRUIT_TARGET_SOURCE_COLUMNS = ["tpc_cfu_g", "tpc_log10"]


def encode_dryfruit_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals for modeling."""
    out = df.copy()
    for dt in DRYER_TYPES:
        out[f"dryer_type_{dt}"] = (out["dryer_type"] == dt).astype(float)
    for pkg in PACKAGING_TYPES:
        out[f"packaging_{pkg}"] = (out["packaging"] == pkg).astype(float)
    for fruit in FRUITS:
        out[f"fruit_{fruit}"] = (out["fruit"] == fruit).astype(float)
    return out
