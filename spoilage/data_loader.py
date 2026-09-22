"""
Dataset paths and inspection utilities.

[UNSUPERVISED] — loads and profiles raw files; no model training.
Original source files are never modified.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRY_FRUIT_DIR = PROJECT_ROOT / "dataset" / "dry_fruit"
REPORTS_DIR = PROJECT_ROOT / "dataset" / "reports"

PHYSICO_PATH = DRY_FRUIT_DIR / "S 1. Mango and Pineaplle Physicochemical Charcteristics Data.xlsx"
MICRO_PATH = DRY_FRUIT_DIR / "S 2. Mango and Pinepple Microbial Data.xlsx"

PANEER_SENSOR = PROJECT_ROOT / "dataset" / "20241118 DataLog.csv"
PANEER_QUALITY = PROJECT_ROOT / "dataset" / "20241230 Testing Data.xlsx"


def inspect_dry_fruit_dataset() -> str:
    """Profile the Mendeley dried-fruit files; return report text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DRY FRUIT DATA PROFILE — Mendeley Dataset")
    lines.append("Dataset on Physicochemical Properties, Microbiological Quality")
    lines.append("and Shelf Life Prediction of Solar Dried Mangoes and Pineapples")
    lines.append("=" * 72)

    for path in (PHYSICO_PATH, MICRO_PATH):
        lines.append(f"\nFILE: {path.name}")
        lines.append(f"Exists: {path.exists()}")
        if not path.exists():
            continue
        xl = pd.ExcelFile(path)
        lines.append(f"Sheets: {xl.sheet_names}")
        for sheet in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            lines.append(f"\n  Sheet: {sheet}")
            lines.append(f"  Rows: {len(df)} | Cols: {len(df.columns)}")
            lines.append(f"  Columns: {list(df.columns)}")
            lines.append(f"  Dtypes:\n{df.dtypes.to_string()}")
            miss = (df.isnull().mean() * 100).round(1)
            lines.append(f"  Missing %:\n{miss.to_string()}")
            for col in df.select_dtypes(include="number").columns:
                s = df[col].dropna()
                if len(s):
                    lines.append(
                        f"  {col}: min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}"
                    )
            for col in df.select_dtypes(include="object").columns:
                uniq = df[col].astype(str).str.strip().unique()
                lines.append(f"  {col} unique ({len(uniq)}): {list(uniq)[:20]}")

    lines.append("\n" + "=" * 72)
    lines.append("KEY FINDINGS")
    lines.append("=" * 72)
    lines.append("- NO temperature or relative humidity columns exist in any file.")
    lines.append("- Physicochemical: Moisture, Water Activity (Aw), pH, TTA by drying method.")
    lines.append("- Microbial: TPC, Fungi, Coliform by dryer, packaging, storage time.")
    lines.append("- Storage time levels: Zero (0), Three (3), Six (6) months.")
    lines.append("- Dryer types: Fresh (raw fruit), Mixed/CMD, Tunel/Tunnel.")
    lines.append("- Packaging: No, High Density, Low Density.")
    lines.append("- Coliform mostly absent/zero for dried samples.")
    return "\n".join(lines)


def write_dry_fruit_profile() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = inspect_dry_fruit_dataset()
    out = REPORTS_DIR / "dry_fruit_data_profile.txt"
    out.write_text(report, encoding="utf-8")
    return out
