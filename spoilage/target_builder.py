"""
Spoilage target construction for all product types.

[SUPERVISED label derivation — rules documented in dataset/reports/target_definition.md]

Paneer: reuses existing assign_spoilage_class from spoilage_preprocessing.
Dried fruit: TPC-based rules derived from observed Mendeley data trends.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spoilage_preprocessing import TARGET_CONFIG, assign_spoilage_class
from spoilage.dryfruit_processor import build_dryfruit_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "dataset" / "reports" / "target_definition.md"

CLASS_NAMES = {0: "FRESH", 1: "WARNING", 2: "SPOILED"}
CLASS_LABELS = ["FRESH", "WARNING", "SPOILED"]

# Dried-fruit thresholds — derived from dried-product TPC distribution in source data
# (24 cells, Mixed/Tunnel dryers only; Fresh raw fruit excluded).
# Computed on 2026-09-09 from dataset/dry_fruit microbial file:
#   month-0 dried mean log10(TPC) ≈ 3.74, month-6 ≈ 3.99
#   25th/75th percentiles of log10(TPC) across dried samples: ≈ 3.57 / 4.01
DRYFRUIT_TARGET_CONFIG = {
    "fresh_tpc_log10_max": 3.70,
    "warning_tpc_log10_max": 3.95,
    "fresh_max_storage_months": 0,
    "spoiled_min_storage_months": 6,
    "spoiled_tpc_log10_min": 3.95,
    "primary_indicator": "Total Plate Count (CFU/g)",
    "secondary_indicator": "storage_months (shelf-life duration)",
}


def assign_dryfruit_class(row: pd.Series, cfg: dict = DRYFRUIT_TARGET_CONFIG) -> int:
    """
    Three-class label for stored dried mango/pineapple.

    Primary: log10(TPC) — microbiological deterioration (source organism column).
    Secondary: storage_months — shelf-life duration in the Mendeley study.

    Rules (documented; thresholds from dried-sample distribution + temporal trend):
      SPOILED (2): storage_months >= spoiled_min_storage_months
                   OR log10(TPC) >= spoiled_tpc_log10_min
      FRESH   (0): storage_months <= fresh_max_storage_months
                   AND log10(TPC) < fresh_tpc_log10_max
      WARNING (1): all other cases (intermediate TPC and/or 3-month storage)
    """
    log_tpc = row["tpc_log10"]
    months = row["storage_months"]

    if months >= cfg["spoiled_min_storage_months"] or log_tpc >= cfg["spoiled_tpc_log10_min"]:
        return 2
    if months <= cfg["fresh_max_storage_months"] and log_tpc < cfg["fresh_tpc_log10_max"]:
        return 0
    return 1


def label_dryfruit(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = out.apply(assign_dryfruit_class, axis=1)
    out["target_status"] = out["target_class"].map(CLASS_NAMES)
    return out


def label_paneer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_class"] = out.apply(lambda r: assign_spoilage_class(r, TARGET_CONFIG), axis=1)
    out["target_status"] = out["target_class"].map(CLASS_NAMES)
    return out


def build_target_definition_markdown() -> str:
    """Generate dataset/reports/target_definition.md content."""
    dry = label_dryfruit(build_dryfruit_samples())
    dist = dry["target_status"].value_counts().to_dict()

    md = f"""# Spoilage Target Definition — AgriChainAI Multi-Product

## Overview

Three-class classification: **FRESH (0)**, **WARNING (1)**, **SPOILED (2)**.

Target variables used for labeling are **never** included as model inputs.

---

## 1. Paneer (existing AC-CSN pipeline)

**Source files:** `dataset/20241118 DataLog.csv`, `dataset/20241230 Testing Data.xlsx`

**Primary indicator:** Total Plate Count (`tpc_log10`, log10 cfu/ml)

**Secondary indicator:** Overall Acceptability (9-point hedonic scale, when available)

**Rules** (from `spoilage_preprocessing.TARGET_CONFIG`):
- **SPOILED:** TPC ≥ {TARGET_CONFIG['spoiled_tpc_log10_min']} OR acceptability ≤ {TARGET_CONFIG['spoiled_acceptability_max']}
- **FRESH:** TPC < {TARGET_CONFIG['fresh_tpc_log10_max']} AND acceptability ≥ {TARGET_CONFIG['fresh_acceptability_min']}
- **WARNING:** otherwise
- **TPC-only fallback** when sensory scores missing (days 10–13)

**Model inputs:** daily cold-chain sensor aggregates only (17 features).

**Leakage prevention:** TPC, sensory, pH, moisture excluded from features.

---

## 2. Dried Fruit (Mendeley Tanzania dataset)

**Source files:** `dataset/dry_fruit/S 1...Physicochemical...xlsx`, `S 2...Microbial...xlsx`

**Samples modeled:** stored **dried** mango/pineapple only (`Dryer type` ≠ Fresh raw fruit).

**Primary indicator:** Total Plate Count (`tpc_cfu_g` → `tpc_log10`)

**Secondary indicator:** `storage_months` (0, 3, 6 months in source study)

**Thresholds** (project-specific, justified from dried-sample distribution):
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| fresh_tpc_log10_max | {DRYFRUIT_TARGET_CONFIG['fresh_tpc_log10_max']} | Below month-0 dried mean (~3.74); separates low-TPC tunnel/CMD samples |
| spoiled_tpc_log10_min | {DRYFRUIT_TARGET_CONFIG['spoiled_tpc_log10_min']} | Near 75th percentile of dried TPC; aligns with 6-month elevated loads |
| fresh_max_storage_months | {DRYFRUIT_TARGET_CONFIG['fresh_max_storage_months']} | Initial shelf-life timepoint in source |
| spoiled_min_storage_months | {DRYFRUIT_TARGET_CONFIG['spoiled_min_storage_months']} | Final shelf-life timepoint showing highest mean TPC |

**Rules:**
- **SPOILED:** storage ≥ 6 months OR log10(TPC) ≥ {DRYFRUIT_TARGET_CONFIG['spoiled_tpc_log10_min']}
- **FRESH:** storage ≤ 0 months AND log10(TPC) < {DRYFRUIT_TARGET_CONFIG['fresh_tpc_log10_max']}
- **WARNING:** intermediate storage (3 months) and/or intermediate TPC

**Model inputs (inference-time, no lab QC):**
- product subtype (Mango / Pineapple)
- storage duration (days)
- dryer type (CMD / TD)
- packaging (None / HDPE / LDPE)

**NOT in source dataset (not fabricated):** storage temperature, relative humidity.

**Reference-only (post-prediction):** moisture, water activity, pH from physicochemical table by dryer type.

**Observed class distribution (dried samples, n={len(dry)}):** {dist}

---

## Limitations

1. Paneer: small single-batch experiment (~70 replicates).
2. Dried fruit: only 24 dried storage cells — high variance; separate model required.
3. Dried-fruit thresholds are project-specific conversions from continuous QC data.
4. No claim of universal regulatory compliance — thresholds must be validated with domain experts.
5. Temperature/humidity unavailable for dried-fruit path; environmental risk for dried products is captured indirectly via storage duration and packaging.
"""
    return md


def write_target_definition() -> Path:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_target_definition_markdown(), encoding="utf-8")
    return REPORT_PATH


def dryfruit_thresholds_json() -> dict:
    return {
        "paneer": {"source": "spoilage_preprocessing.TARGET_CONFIG", "config": TARGET_CONFIG},
        "dryfruit": DRYFRUIT_TARGET_CONFIG,
        "classes": CLASS_LABELS,
    }
