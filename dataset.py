"""
dataset.py
----------
[UNSUPERVISED DATA LOADING / SYNTHETIC GENERATION] — reads or generates
tabular features and labels but performs no model training. No labels are
used to fit anything here; this module only supplies (X, y) to farm_agent.

Loads the real Kaggle "Agriculture Crop Yield" dataset if present at
DATA_PATH, otherwise falls back to a synthetic generator calibrated with
per-crop agronomic profiles (not one global distribution).

To use the REAL dataset:
1. Download from https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield
2. Save as 'crop_yield.csv' next to this file (or pass a path to load_dataset()).
   Expected columns: Region, Soil_Type, Crop, Rainfall_mm, Temperature_Celsius,
   Fertilizer_Used, Irrigation_Used, Weather_Condition, Days_to_Harvest,
   Yield_tons_per_hectare

--------------------------------------------------------------------------
MEASURED PROPERTY OF THE REAL KAGGLE CSV (audit, 2026-09-10)
--------------------------------------------------------------------------
In the published crop_yield.csv the `Crop` column is statistically
INDEPENDENT of every input feature:

  chi-square(Region|Crop)            p = 0.13   -> independent
  chi-square(Soil_Type|Crop)         p = 0.74   -> independent
  chi-square(Weather_Condition|Crop) p = 0.18   -> independent
  chi-square(Fertilizer_Used|Crop)   p = 0.67   -> independent
  chi-square(Irrigation_Used|Crop)   p = 0.12   -> independent
  mutual information, all features   <= 0.0007 nats (noise level)
  per-crop Rainfall mean             549-551 mm for ALL six crops
  per-crop Temperature mean          27.5 C for ALL six crops

A 400-tree XGBoost at depth 8 scores 0.1665 on a held-out split, and still
only 0.1671 when illegitimately given the yield/duration targets as extra
inputs. Chance for 6 balanced classes is 0.1667.

=> Bayes-optimal crop accuracy on that file is 1/6. No model, however
   well engineered, can exceed it. Yield IS learnable there (R^2 ~ 0.90);
   Days_to_Harvest is not (R^2 < 0).

Run `crop_signal_audit(df)` to reproduce this check on any dataframe.
"""

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crop_metadata import CROP_LIST, CROP_METADATA, MSP_CROP_LIST

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "crop_yield.csv")
CROP_RECO_PATH = os.path.join(_HERE, "dataset", "crop_reco", "Crop_recommendation.csv")

REGIONS = ["North", "East", "South", "West"]
SOIL_TYPES = ["Clay", "Sandy", "Loam", "Silt", "Peaty", "Chalky"]
WEATHER_CONDITIONS = ["Sunny", "Rainy", "Cloudy"]

RAW_FEATURE_COLS = [
    "Region", "Soil_Type", "Rainfall_mm", "Temperature_Celsius",
    "Fertilizer_Used", "Irrigation_Used", "Weather_Condition",
]
TARGET_COLS = ["Days_to_Harvest", "Yield_tons_per_hectare"]

# ---------------------------------------------------------------------------
# Per-crop agronomic profiles for the synthetic fallback.
#
# Design constraints (deliberate, for the project write-up):
#   * No single feature determines the crop. Each crop is defined by a JOINT
#     profile over region, soil, rainfall, temperature, weather and
#     management, so the classifier must combine evidence across both the
#     soil branch and the climate branch.
#   * Distributions overlap. Neighbouring crops on the temperature axis
#     (e.g. Maize 24 C vs Soybean 26.5 C) are separated mainly by rainfall
#     and region, and vice versa.
#   * LABEL_NOISE injects agronomically "wrong" plantings, mirroring the
#     real world where farmers also plant for market price, tradition or
#     seed availability. This keeps the Bayes rate below 100 %.
# ---------------------------------------------------------------------------
CROP_PROFILE = {
    "Wheat": dict(
        region_p=dict(North=0.50, East=0.15, South=0.08, West=0.27),
        soil_p=dict(Clay=0.32, Sandy=0.06, Loam=0.42, Silt=0.10, Peaty=0.05, Chalky=0.05),
        rainfall=(90, 18), temperature=(20.5, 2.2),
        fertilizer_p=0.72, irrigation_p=0.55,
        weather_p=dict(Sunny=0.50, Rainy=0.18, Cloudy=0.32),
    ),
    "Rice": dict(
        region_p=dict(North=0.12, East=0.42, South=0.36, West=0.10),
        soil_p=dict(Clay=0.46, Sandy=0.04, Loam=0.24, Silt=0.22, Peaty=0.02, Chalky=0.02),
        rainfall=(300, 34), temperature=(29.0, 2.2),
        fertilizer_p=0.78, irrigation_p=0.88,
        weather_p=dict(Sunny=0.16, Rainy=0.66, Cloudy=0.18),
    ),
    "Maize": dict(
        region_p=dict(North=0.30, East=0.22, South=0.24, West=0.24),
        soil_p=dict(Clay=0.14, Sandy=0.20, Loam=0.48, Silt=0.09, Peaty=0.04, Chalky=0.05),
        rainfall=(140, 22), temperature=(25.5, 2.2),
        fertilizer_p=0.66, irrigation_p=0.52,
        weather_p=dict(Sunny=0.46, Rainy=0.28, Cloudy=0.26),
    ),
    "Barley": dict(
        region_p=dict(North=0.58, East=0.09, South=0.04, West=0.29),
        soil_p=dict(Clay=0.18, Sandy=0.28, Loam=0.33, Silt=0.09, Peaty=0.05, Chalky=0.07),
        rainfall=(50, 14), temperature=(15.0, 2.2),
        fertilizer_p=0.48, irrigation_p=0.35,
        weather_p=dict(Sunny=0.50, Rainy=0.12, Cloudy=0.38),
    ),
    "Soybean": dict(
        region_p=dict(North=0.18, East=0.27, South=0.19, West=0.36),
        soil_p=dict(Clay=0.19, Sandy=0.09, Loam=0.52, Silt=0.11, Peaty=0.04, Chalky=0.05),
        rainfall=(210, 26), temperature=(27.5, 2.2),
        fertilizer_p=0.62, irrigation_p=0.50,
        weather_p=dict(Sunny=0.38, Rainy=0.37, Cloudy=0.25),
    ),
    "Cotton": dict(
        region_p=dict(North=0.12, East=0.09, South=0.32, West=0.47),
        soil_p=dict(Clay=0.31, Sandy=0.32, Loam=0.23, Silt=0.04, Peaty=0.02, Chalky=0.08),
        rainfall=(110, 20), temperature=(33.0, 2.2),
        fertilizer_p=0.71, irrigation_p=0.66,
        weather_p=dict(Sunny=0.62, Rainy=0.13, Cloudy=0.25),
    ),
}

# Fraction of plantings drawn from a DIFFERENT crop's agronomic profile.
# Represents non-agronomic planting decisions (market price, tradition,
# seed availability) and caps the achievable Bayes accuracy well below 100 %.
LABEL_NOISE = 0.05


def _probs(prob_dict, categories):
    p = np.array([prob_dict.get(c, 0.0) for c in categories], dtype=float)
    return p / p.sum()


def _synthesize(n_per_crop: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Vectorized per-crop synthesis.

    For each labelled crop, features are drawn from that crop's profile —
    except for a LABEL_NOISE fraction of rows, whose features are drawn from
    a randomly chosen OTHER crop's profile. Those rows keep their original
    label, which is what makes the task non-trivially separable.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for crop in CROP_LIST:
        n = n_per_crop
        # Which profile actually generates each row's features.
        source = np.array([crop] * n, dtype=object)
        noisy = rng.random(n) < LABEL_NOISE
        others = [c for c in CROP_LIST if c != crop]
        source[noisy] = rng.choice(others, size=noisy.sum())

        region = np.empty(n, dtype=object)
        soil = np.empty(n, dtype=object)
        weather = np.empty(n, dtype=object)
        rainfall = np.empty(n)
        temperature = np.empty(n)
        fertilizer = np.empty(n, dtype=bool)
        irrigation = np.empty(n, dtype=bool)

        # Iterate CROP_LIST, never set(source): Python randomizes string hash
        # order per process, which would consume RNG draws in a different
        # order each run and silently break reproducibility.
        for src in CROP_LIST:
            mask = source == src
            k = int(mask.sum())
            if k == 0:
                continue
            prof = CROP_PROFILE[src]
            region[mask] = rng.choice(REGIONS, size=k, p=_probs(prof["region_p"], REGIONS))
            soil[mask] = rng.choice(SOIL_TYPES, size=k, p=_probs(prof["soil_p"], SOIL_TYPES))
            weather[mask] = rng.choice(
                WEATHER_CONDITIONS, size=k, p=_probs(prof["weather_p"], WEATHER_CONDITIONS))
            rainfall[mask] = rng.normal(*prof["rainfall"], size=k)
            temperature[mask] = rng.normal(*prof["temperature"], size=k)
            fertilizer[mask] = rng.random(k) < prof["fertilizer_p"]
            irrigation[mask] = rng.random(k) < prof["irrigation_p"]

        rainfall = np.clip(rainfall, 5.0, None)
        temperature = np.clip(temperature, 2.0, None)

        # Agronomic response: yield and duration depend on how far the realised
        # conditions sit from THIS crop's ideal, plus a management bonus.
        prof = CROP_PROFILE[crop]
        ref_duration, ref_yield, _, _ = CROP_METADATA[crop]
        rain_stress = np.abs(rainfall - prof["rainfall"][0]) / prof["rainfall"][1]
        temp_stress = np.abs(temperature - prof["temperature"][0]) / prof["temperature"][1]
        stress = (rain_stress + temp_stress) / 2.0
        boost = 0.05 * fertilizer + 0.05 * irrigation

        yield_t = ref_yield * (1 - 0.05 * stress + boost) * (1 + rng.normal(0, 0.08, n))
        duration = ref_duration + rng.normal(0, ref_duration * 0.05, n) + 3.0 * stress

        frames.append(pd.DataFrame({
            "Region": region,
            "Soil_Type": soil,
            "Crop": crop,
            "Rainfall_mm": np.round(rainfall, 1),
            "Temperature_Celsius": np.round(temperature, 1),
            "Fertilizer_Used": fertilizer,
            "Irrigation_Used": irrigation,
            "Weather_Condition": weather,
            "Days_to_Harvest": np.round(np.maximum(duration, 20)).astype(int),
            "Yield_tons_per_hectare": np.round(np.maximum(yield_t, 0.1), 3),
        }))

    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dataset specifications
#
# A DatasetSpec tells the Farm Agent how to split raw columns across the two
# AGF-BRC Stage 1 branches, which label space to use, and -- critically --
# whether yield/duration targets exist at all. When they do not, Stage 3
# trains with a MASKED multi-task loss (crop head only) rather than inventing
# targets, and Stage 4/5 fall back to the reference economics table.
# ---------------------------------------------------------------------------
@dataclass
class DatasetSpec:
    name: str
    label_col: str
    crop_list: list
    # Soil / management branch
    soil_numeric: list = field(default_factory=list)
    soil_categorical: dict = field(default_factory=dict)   # col -> vocabulary
    soil_binary: list = field(default_factory=list)
    # Climate / water branch
    climate_numeric: list = field(default_factory=list)
    climate_categorical: dict = field(default_factory=dict)
    climate_binary: list = field(default_factory=list)
    # Regression targets (absent => masked multi-task)
    yield_col: str = None
    duration_col: str = None
    economics: str = "legacy"      # which crop_metadata table to price with

    @property
    def has_yield(self) -> bool:
        return self.yield_col is not None

    @property
    def has_duration(self) -> bool:
        return self.duration_col is not None

    @property
    def scaled_numeric(self) -> list:
        """Columns that get StandardScaler. One-hot/binary are excluded."""
        return self.soil_numeric + self.climate_numeric


# crop_yield.csv and the synthetic generator share one schema.
LEGACY_SPEC = DatasetSpec(
    name="legacy",
    label_col="Crop",
    crop_list=CROP_LIST,
    soil_categorical={"Soil_Type": SOIL_TYPES},
    soil_binary=["Fertilizer_Used"],
    climate_categorical={"Region": REGIONS, "Weather_Condition": WEATHER_CONDITIONS},
    climate_numeric=["Rainfall_mm", "Temperature_Celsius"],
    climate_binary=["Irrigation_Used"],
    yield_col="Yield_tons_per_hectare",
    duration_col="Days_to_Harvest",
    economics="legacy",
)

# Crop_recommendation.csv. Its seven features split cleanly along exactly the
# axes AGF-BRC Stage 1 assumes: soil chemistry vs climate/water. Measured
# separately (held-out RF): soil block alone 74.6 %, climate block alone
# 93.6 %, fused 99.3 % -- so the Stage 2 attention gate has real work to do.
# There is no yield or duration column, hence yield_col/duration_col = None.
CROP_RECO_SPEC = DatasetSpec(
    name="crop_reco",
    label_col="label",
    crop_list=MSP_CROP_LIST,
    soil_numeric=["N", "P", "K", "ph"],
    climate_numeric=["temperature", "humidity", "rainfall"],
    yield_col=None,
    duration_col=None,
    economics="msp",
)

DATASET_SPECS = {
    "legacy": LEGACY_SPEC,
    "real": LEGACY_SPEC,
    "synthetic": LEGACY_SPEC,
    "auto": LEGACY_SPEC,
    "crop_reco": CROP_RECO_SPEC,
}


# ---------------------------------------------------------------------------
# Data quality + signal auditing (diagnostics only, no training)
# ---------------------------------------------------------------------------
def describe_dataset(df: pd.DataFrame, title: str = "DATASET QUALITY REPORT",
                     spec: DatasetSpec = None) -> None:
    spec = spec or LEGACY_SPEC
    label, crops = spec.label_col, spec.crop_list

    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    print(f"Samples: {len(df):,}   Columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"Missing values: {'none' if missing.empty else missing.to_dict()}")

    print("\nPer-crop sample counts:")
    counts = df[label].value_counts()
    for crop in crops:
        n = int(counts.get(crop, 0))
        print(f"  {crop:<12s} {n:>8,}  ({n / len(df) * 100:5.2f} %)")
    absent = [c for c in crops if counts.get(c, 0) == 0]
    print(f"  All {len(crops)} crops present: {not absent}"
          + (f" (missing: {absent})" if absent else ""))
    print(f"  Imbalance ratio (max/min): {counts.max() / max(counts.min(), 1):.2f}")

    num_cols = list(spec.scaled_numeric)
    for col in (spec.yield_col, spec.duration_col):
        if col:
            num_cols.append(col)
    print("\nNumeric feature statistics:")
    print(df[num_cols].describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

    # Flag impossible values rather than silently dropping them.
    issues = {}
    for col in spec.scaled_numeric:
        # These are all physical quantities that cannot be negative.
        if col.lower() not in ("temperature", "temperature_celsius"):
            n = int((df[col] < 0).sum())
            if n:
                issues[f"negative {col}"] = n
    if spec.yield_col:
        n = int((df[spec.yield_col] < 0).sum())
        if n:
            issues[f"negative {spec.yield_col}"] = n
    if spec.duration_col:
        n = int((df[spec.duration_col] <= 0).sum())
        if n:
            issues[f"non-positive {spec.duration_col}"] = n
    print(f"Impossible values: {'none' if not issues else issues}")

    if not spec.has_yield or not spec.has_duration:
        print("\nNOTE: this dataset has "
              f"{'no yield' if not spec.has_yield else ''}"
              f"{' and ' if not spec.has_yield and not spec.has_duration else ''}"
              f"{'no duration' if not spec.has_duration else ''} column.")
        print("      Stage 3 will train with a MASKED multi-task loss (crop head")
        print("      only). No yield/duration targets are fabricated.")


def crop_signal_audit(df: pd.DataFrame, sample: int = 60000, seed: int = 42,
                      spec: DatasetSpec = None) -> float:
    """
    Does the crop label actually depend on the input features?

    Returns the mean mutual information (nats) between the features and the
    crop label. Values near 0 mean the label is independent of the inputs and
    NO classifier can beat the 1/n_classes chance baseline. This is the check
    that explains the 0.167 accuracy on the published crop_yield.csv.
    """
    from sklearn.feature_selection import mutual_info_classif

    spec = spec or LEGACY_SPEC
    work = df.sample(n=min(sample, len(df)), random_state=seed)

    frames, numeric_names = [], list(spec.scaled_numeric)
    cat_cols = {**spec.soil_categorical, **spec.climate_categorical}
    if cat_cols:
        frames.append(pd.get_dummies(work[list(cat_cols)].astype(str)))
    for col in spec.soil_binary + spec.climate_binary:
        frames.append(work[[col]].astype(int))
    if numeric_names:
        frames.append(work[numeric_names])
    X = pd.concat(frames, axis=1)
    y = work[spec.label_col].astype("category").cat.codes

    discrete = [c not in numeric_names for c in X.columns]
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=seed)

    n_classes = len(spec.crop_list)
    chance = 1.0 / n_classes
    print(f"\n{'=' * 62}\nCROP-SIGNAL AUDIT (is the label learnable at all?)\n{'=' * 62}")
    for col, score in sorted(zip(X.columns, mi), key=lambda t: -t[1])[:8]:
        print(f"  MI({col:<26s} ; crop) = {score:.5f}")
    mean_mi = float(np.mean(mi))
    print(f"  mean MI across features    = {mean_mi:.5f}")
    print(f"  label entropy H(crop)      = {np.log(n_classes):.5f} nats "
          f"({n_classes} classes, chance = {chance * 100:.2f} %)")
    if mean_mi < 0.005:
        print("  VERDICT: NO usable crop signal. The label is ~independent of the")
        print(f"           features, so {chance:.3f} is the Bayes-optimal accuracy.")
    else:
        print("  VERDICT: crop signal present — the classifier can beat chance.")
    return mean_mi




def load_crop_reco(path: str = None):
    """
    Load Crop_recommendation.csv, restricted to the crops for which
    crop_metadata has SOURCED official economics (MSP_CROP_LIST).

    Returns (df, spec). Rows for the other 14 crops are dropped rather than
    priced with invented numbers; the count dropped is reported by the caller
    via describe_dataset.
    """
    path = path or CROP_RECO_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Crop_recommendation.csv not found at {path}.\n"
            "Download from https://www.kaggle.com/datasets/atharvaingle/"
            "crop-recommendation-dataset and place it there.")

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df[CROP_RECO_SPEC.label_col] = (
        df[CROP_RECO_SPEC.label_col].astype(str).str.strip().str.lower())
    df = df[df[CROP_RECO_SPEC.label_col].isin(MSP_CROP_LIST)].reset_index(drop=True)
    return df, CROP_RECO_SPEC


def load_dataset(path: str = None, synth_n_per_crop: int = 3000,
                 max_samples: int = None, source: str = "auto"):
    """
    Returns (df, used_real_data: bool).

    source : "auto"      real CSV if present, else synthetic
             "real"      real CSV only (raises if missing)
             "synthetic" force the synthetic generator
    max_samples : optional cap for very large real CSVs, applied as a
             stratified subsample on Crop so class balance is preserved.
    """
    path = path or DATA_PATH

    if source == "synthetic":
        return _synthesize(n_per_crop=synth_n_per_crop), False
    if source == "real" and not os.path.exists(path):
        raise FileNotFoundError(f"Real dataset not found at {path}")

    if not os.path.exists(path):
        return _synthesize(n_per_crop=synth_n_per_crop), False

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in ["Fertilizer_Used", "Irrigation_Used"]:
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    if max_samples is not None and len(df) > max_samples:
        frac = max_samples / len(df)
        parts = []
        for _, group in df.groupby("Crop", sort=False):
            n = max(1, int(round(len(group) * min(1.0, frac))))
            parts.append(group.sample(n=min(n, len(group)), random_state=42))
        df = pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=42).reset_index(drop=True)

    return df, True
