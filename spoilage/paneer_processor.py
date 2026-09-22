"""
Paneer cold-chain processing — wraps the existing validated pipeline.

[SUPERVISED dataset construction uses spoilage_preprocessing]

Preserves the original spoilage_preprocessing.py logic without modification.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spoilage_preprocessing import (
    FEATURE_COLUMNS,
    OUTPUT_PATH,
    TARGET_SOURCE_COLUMNS,
    run_preprocessing,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = PROJECT_ROOT / "data" / "ac_csn_merged_dataset.csv"


def load_paneer_supervised(regenerate: bool = False) -> pd.DataFrame:
    """Load Paneer supervised dataset via existing preprocessing."""
    if regenerate or not OUTPUT_PATH.exists():
        run_preprocessing()
    df = pd.read_csv(OUTPUT_PATH)
    df["product_type"] = "paneer"
    df["product_subtype"] = "paneer"
    return df


def build_paneer_unified_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Map Paneer rows into the multi-product unified schema."""
    out = pd.DataFrame({
        "product_type": "paneer",
        "product_subtype": "paneer",
        "storage_duration": df["storage_day"],
        "temperature": df["mean_temperature"],
        "humidity": df["mean_rh"],
        "packaging_condition": "refrigerated",
        "target_class": df["spoilage_class"],
        "target_status": df["spoilage_status"],
        "group_id": df["storage_day"].astype(str),
    })
    for col in FEATURE_COLUMNS:
        out[f"paneer_{col}"] = df[col].values
    return out


# Re-export for convenience
PANEER_FEATURE_COLUMNS = FEATURE_COLUMNS
PANEER_TARGET_SOURCE = TARGET_SOURCE_COLUMNS
