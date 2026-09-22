"""
Unified multi-product dataset builder.

[UNSUPERVISED merge + SUPERVISED labels via target_builder]
Original source files are never modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from spoilage.dryfruit_processor import (
    DRYFRUIT_FEATURE_COLUMNS,
    build_dryfruit_samples,
    encode_dryfruit_features,
)
from spoilage.paneer_processor import build_paneer_unified_rows, load_paneer_supervised
from spoilage.target_builder import label_dryfruit, label_paneer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_PATH = PROJECT_ROOT / "dataset" / "ac_csn_multiproduct.csv"


def build_unified_dataset(save: bool = True) -> pd.DataFrame:
    """Build dataset/ac_csn_multiproduct.csv from Paneer + dried fruit."""
    paneer_raw = load_paneer_supervised()
    paneer = label_paneer(paneer_raw)
    paneer_u = build_paneer_unified_rows(paneer)

    dry_raw = build_dryfruit_samples(include_fresh_fruit=False)
    dry = label_dryfruit(dry_raw)
    dry = encode_dryfruit_features(dry)
    dry_u = pd.DataFrame({
        "product_type": dry["product_type"],
        "product_subtype": dry["product_subtype"],
        "storage_duration": dry["storage_duration_days"],
        "temperature": pd.NA,  # not in Mendeley source
        "humidity": pd.NA,
        "packaging_condition": dry["packaging"],
        "target_class": dry["target_class"],
        "target_status": dry["target_status"],
        "group_id": dry["group_id"],
        "dryer_type": dry["dryer_type"],
        "storage_months": dry["storage_months"],
    })
    for col in DRYFRUIT_FEATURE_COLUMNS:
        dry_u[col] = dry[col].values

    unified = pd.concat([paneer_u, dry_u], ignore_index=True, sort=False)
    if save:
        UNIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(UNIFIED_PATH, index=False)
    return unified
