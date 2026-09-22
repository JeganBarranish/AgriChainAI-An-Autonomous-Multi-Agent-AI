"""
farm_agent.py
--------------
Farm Agent (FA) for AgriChainAI -- entry point of the pipeline.

Paradigm map (for viva classification):
  - _encode_frame / StandardScaler:  UNSUPERVISED preprocessing (no labels)
  - train() Stages 1-4:              SUPERVISED (via hybrid_model.HybridFarmModel)
  - recommend() Stage 5:             RULE-BASED / DETERMINISTIC re-ranking

Usage:
    python3 farm_agent.py                     # default: Crop_recommendation.csv
    python3 farm_agent.py --data crop_reco    # 8 MSP-priced crops, real signal
    python3 farm_agent.py --data real         # legacy crop_yield.csv (no signal)
    python3 farm_agent.py --data synthetic    # legacy synthetic generator
    python3 farm_agent.py --no-search         # skip the hyperparameter search

==========================================================================
DATASETS AND WHY THERE ARE THREE
==========================================================================
1. crop_reco  (DEFAULT) -- dataset/crop_reco/Crop_recommendation.csv
   22-crop Kaggle "Crop Recommendation" data, restricted to the 8 crops for
   which crop_metadata.py has SOURCED official MSP/CACP economics. Its seven
   features split cleanly across the two AGF-BRC Stage 1 branches:
       soil branch    = N, P, K, ph
       climate branch = temperature, humidity, rainfall
   Measured signal: mean MI 0.94 nats against H(crop)=2.08; a held-out
   RandomForest scores 99.5 % and the true-vs-shuffled label gap is
   +95.9 points. This dataset genuinely supports high accuracy.
   It has NO yield and NO duration column, so Stage 3 trains with a MASKED
   multi-task loss (crop head only) and Stage 5 prices every candidate from
   the reference table. No yield or duration target is ever fabricated.

2. real -- crop_yield.csv  [KEPT ONLY AS A NEGATIVE CONTROL]
   Its `Crop` column is statistically INDEPENDENT of every input feature:
   chi-square p = 0.12-0.74 for all five categoricals, mutual information
   <= 0.0007 nats, per-crop Rainfall/Temperature means identical to three
   significant figures, and a 400-tree XGBoost at depth 8 scores 0.1665
   (0.1671 even when illegitimately handed the yield/duration targets).
   Chance is 1/6 = 0.1667, so 16.7 % IS the Bayes-optimal accuracy there.
   Training on shuffled labels scores the same as on true labels.
   Full forensic write-up: results/crop_dataset_diagnostic.txt
   Yield is genuinely learnable in that file (R^2 ~ 0.90); Days_to_Harvest
   is not (R^2 < 0).

3. synthetic -- dataset.py generator, ~88 % held-out. Useful for testing the
   pipeline when no CSV is present.

==========================================================================
DIAGNOSTIC: implementation defects found and fixed during the audit
==========================================================================
   a. StandardScaler was fit on the FULL dataset before the train/test split
      (preprocessing leakage). Now the split happens first and only the
      training split calls fit().
   b. No validation split existed (80/20 only), so there was no honest basis
      for early stopping or model selection. Now 70/15/15 stratified.
   c. No early stopping and no best-checkpoint restore.
   d. Yield MSE was computed on RAW tons/hectare while the crop loss is O(1),
      letting the regression term dominate the shared trunk. Both regression
      targets are now rescaled to roughly unit variance using TRAIN-only
      statistics.
   e. Branch/trunk widths were 16/16/48 -- far too narrow.
   f. Row-by-row .iterrows() encoding, now vectorized.
   g. dataset._synthesize iterated `set(source)`; Python randomizes string
      hash order per process, so RNG draws were consumed in a different order
      every run and results were not reproducible. Now iterates CROP_LIST.
Item (a) and (g) are correctness bugs; (b)-(e) were the accuracy limiters.
"""

import argparse
import calendar
import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from crop_metadata import (
    get_expected_profit,
    get_profit_weight_vector,
    get_reference_duration,
    get_reference_yield,
    get_source_note,
)
from dataset import (
    CROP_RECO_SPEC,
    LEGACY_SPEC,
    REGIONS,
    SOIL_TYPES,
    crop_signal_audit,
    describe_dataset,
    load_crop_reco,
    load_dataset,
)
from hybrid_model import HybridFarmModel

RANDOM_SEED = 42

# Stage 5 output contract: exactly 1 best crop + exactly 3 alternatives,
# drawn from the top-N crops by classification probability.
CANDIDATE_POOL_SIZE = 4
N_ALTERNATIVES = CANDIDATE_POOL_SIZE - 1

# Month -> typical (Rainfall_mm, Temperature_Celsius, Weather_Condition) for a
# generic Indian agro-climatic zone. Used only by the LEGACY recommend() path,
# where the farmer gives a month instead of numeric readings. Replace with
# local IMD / weather-API data for a production deployment.
MONTH_CLIMATE_DEFAULTS = {
    1:  (15, 18, "Sunny"),   2:  (12, 21, "Sunny"),   3:  (18, 26, "Sunny"),
    4:  (12, 31, "Sunny"),   5:  (22, 34, "Cloudy"),  6:  (110, 32, "Rainy"),
    7:  (250, 29, "Rainy"),  8:  (230, 28, "Rainy"),  9:  (170, 27, "Rainy"),
    10: (70, 25, "Cloudy"),  11: (20, 21, "Cloudy"),  12: (10, 18, "Sunny"),
}

# Small, controlled search grid. Selection uses VALIDATION accuracy only;
# the test split is untouched until the single final evaluation.
SEARCH_GRID = [
    dict(branch_hidden=64, branch_out=64, trunk_hidden=128, dropout=0.2, lr=1e-3),
    dict(branch_hidden=64, branch_out=64, trunk_hidden=128, dropout=0.1, lr=5e-4),
    dict(branch_hidden=32, branch_out=32, trunk_hidden=64, dropout=0.2, lr=1e-3),
    dict(branch_hidden=128, branch_out=64, trunk_hidden=128, dropout=0.3, lr=5e-4),
]
DEFAULT_CONFIG = SEARCH_GRID[0]

# Below this many samples a single 15 % test split is too small to trust, so
# stratified k-fold CV is additionally reported.
SMALL_DATASET_THRESHOLD = 5000


def _one_hot_matrix(series, categories) -> np.ndarray:
    """Vectorized one-hot with a FIXED category order (deterministic columns)."""
    codes = pd.Categorical(series, categories=categories).codes
    out = np.zeros((len(series), len(categories)), dtype=np.float32)
    valid = codes >= 0
    out[np.flatnonzero(valid), codes[valid]] = 1.0
    return out


def _hstack(parts, n_rows):
    parts = [p for p in parts if p is not None and p.shape[1] > 0]
    if not parts:
        return np.zeros((n_rows, 0), dtype=np.float32)
    return np.hstack(parts).astype(np.float32)


class FarmAgent:
    def __init__(self, seed: int = RANDOM_SEED, spec=None):
        self.seed = seed
        self.spec = spec or LEGACY_SPEC
        self.crop_list = list(self.spec.crop_list)
        self.crop_to_idx = {c: i for i, c in enumerate(self.crop_list)}
        self.idx_to_crop = {i: c for c, i in self.crop_to_idx.items()}
        # Scales ONLY the continuous columns named by the spec. One-hot and
        # binary columns are left untouched -- standardizing them would
        # distort their 0/1 semantics without any benefit.
        self.numeric_scaler = StandardScaler()
        self.model: HybridFarmModel = None
        self.report: dict = {}

    # ------------------------------------------------------------- encoding
    def _encode_frame(self, df: pd.DataFrame, fit_scaler: bool = False):
        """
        Build (soil_x, climate_x) [UNSUPERVISED preprocessing -- no labels].

        fit_scaler must be True ONLY for the training split. Validation, test
        and inference all call transform(), never fit(), which is what keeps
        preprocessing free of leakage.
        """
        spec, n = self.spec, len(df)

        soil_num = clim_num = None
        if spec.scaled_numeric:
            numeric = df[spec.scaled_numeric].to_numpy(dtype=np.float64)
            numeric = (self.numeric_scaler.fit_transform(numeric) if fit_scaler
                       else self.numeric_scaler.transform(numeric)).astype(np.float32)
            k = len(spec.soil_numeric)   # scaled_numeric == soil_numeric + climate_numeric
            soil_num, clim_num = numeric[:, :k], numeric[:, k:]

        soil_parts = [_one_hot_matrix(df[c], vocab)
                      for c, vocab in spec.soil_categorical.items()]
        soil_parts.append(soil_num)
        soil_parts += [df[c].to_numpy(dtype=np.float32)[:, None]
                       for c in spec.soil_binary]

        clim_parts = [_one_hot_matrix(df[c], vocab)
                      for c, vocab in spec.climate_categorical.items()]
        clim_parts.append(clim_num)
        clim_parts += [df[c].to_numpy(dtype=np.float32)[:, None]
                       for c in spec.climate_binary]

        return _hstack(soil_parts, n), _hstack(clim_parts, n)

    def feature_names(self):
        """
        (soil_names, climate_names) in the exact column order produced by
        _encode_frame. Exposed so diagnostics can locate a column by NAME
        instead of hardcoding offsets, which silently break whenever the
        spec's column ordering changes.
        """
        spec = self.spec
        soil = [f"{c}={v}" for c, vocab in spec.soil_categorical.items() for v in vocab]
        soil += list(spec.soil_numeric) + list(spec.soil_binary)
        climate = [f"{c}={v}" for c, vocab in spec.climate_categorical.items()
                   for v in vocab]
        climate += list(spec.climate_numeric) + list(spec.climate_binary)
        return soil, climate

    def _encode_single(self, **feature_values):
        """Inference-time encoding -- identical transform path as training."""
        spec = self.spec
        needed = (list(spec.soil_categorical) + spec.soil_binary
                  + list(spec.climate_categorical) + spec.climate_binary
                  + spec.scaled_numeric)
        missing = [c for c in needed if c not in feature_values]
        if missing:
            raise ValueError(f"missing required features for {spec.name}: {missing}")
        row = pd.DataFrame([{c: feature_values[c] for c in needed}])
        for c in spec.scaled_numeric:
            row[c] = row[c].astype(float)
        for c in spec.soil_binary + spec.climate_binary:
            row[c] = row[c].astype(float)
        return self._encode_frame(row, fit_scaler=False)

    # ---------------------------------------------------------------- train
    def train(self, epochs: int = 150, data_source: str = "crop_reco",
              max_samples: int = 60000, search: bool = True, verbose: bool = True,
              cv_folds: int = 5):
        # ---- load, and adopt the dataset's spec
        if data_source == "crop_reco":
            df, spec = load_crop_reco()
            source_name = "Crop_recommendation.csv (8 MSP-priced crops)"
        else:
            df, used_real = load_dataset(max_samples=max_samples, source=data_source)
            spec = LEGACY_SPEC
            source_name = ("crop_yield.csv [NEGATIVE CONTROL - no crop signal]"
                           if used_real else "synthetic generator (dataset.py)")

        self.__init__(seed=self.seed, spec=spec)   # re-key label space to the spec
        print(f"\n[FarmAgent] Data source: {source_name}  (seed={self.seed})")

        describe_dataset(df, f"DATASET QUALITY REPORT - {source_name}", spec=spec)
        mean_mi = crop_signal_audit(df, seed=self.seed, spec=spec)

        df = df[df[spec.label_col].isin(self.crop_list)].reset_index(drop=True)
        crop_y = df[spec.label_col].map(self.crop_to_idx).to_numpy(dtype=np.int64)
        n_classes = len(self.crop_list)
        chance = 1.0 / n_classes

        # ---- Stratified 70 / 15 / 15. Test is set aside and NOT touched
        # again until the single final evaluation at the end of train().
        idx = np.arange(len(df))
        idx_train, idx_hold = train_test_split(
            idx, test_size=0.30, random_state=self.seed, stratify=crop_y)
        idx_val, idx_test = train_test_split(
            idx_hold, test_size=0.50, random_state=self.seed, stratify=crop_y[idx_hold])

        parts = {}
        for name, part_idx, fit in (("train", idx_train, True),
                                    ("val", idx_val, False),
                                    ("test", idx_test, False)):
            sub = df.iloc[part_idx]
            # fit_scaler=True happens for "train" FIRST, so val/test are
            # transformed with training-set statistics only.
            soil, clim = self._encode_frame(sub, fit_scaler=fit)
            parts[name] = dict(
                soil=soil, clim=clim, crop=crop_y[part_idx],
                yield_=(sub[spec.yield_col].to_numpy(dtype=np.float32)
                        if spec.has_yield else None),
                dur=(sub[spec.duration_col].to_numpy(dtype=np.float32)
                     if spec.has_duration else None),
            )

        print(f"\n{'=' * 62}\nSPLIT (stratified, seed={self.seed})\n{'=' * 62}")
        for name in ("train", "val", "test"):
            print(f"  {name:<6s} {len(parts[name]['crop']):>7,} samples")
        print(f"  soil branch dim {parts['train']['soil'].shape[1]}"
              f"  | climate branch dim {parts['train']['clim'].shape[1]}")

        train_counts = np.bincount(parts["train"]["crop"], minlength=n_classes)
        print("\n  Training-set class distribution:")
        for i, crop in enumerate(self.crop_list):
            print(f"    {crop:<12s} {train_counts[i]:>7,}")
        imbalance = train_counts.max() / max(train_counts.min(), 1)
        print(f"  Imbalance ratio (max/min): {imbalance:.2f}"
              f"  -> {'balanced, no resampling needed' if imbalance < 1.5 else 'IMBALANCED'}")

        # ---- PWCE profit weights (economic weighting, not class balancing)
        profit_weights = get_profit_weight_vector(self.crop_list)
        raw_weights = get_profit_weight_vector(self.crop_list, clip_range=None)
        pw, raw = np.asarray(profit_weights), np.asarray(raw_weights)
        print(f"\n{'=' * 62}\nPWCE PROFIT WEIGHTS (mean must be 1.0)\n{'=' * 62}")
        print(f"  {'crop':<12s} {'raw':>8s} {'capped':>8s}   source")
        for crop, r, w in zip(self.crop_list, raw, pw):
            print(f"  {crop:<12s} {r:>8.3f} {w:>8.3f}   {get_source_note(crop)[:44]}")
        print(f"  raw    mean={raw.mean():.3f} min={raw.min():.3f} "
              f"max={raw.max():.3f} ratio={raw.max() / raw.min():.2f}")
        print(f"  capped mean={pw.mean():.3f} min={pw.min():.3f} "
              f"max={pw.max():.3f} ratio={pw.max() / pw.min():.2f}")
        print("  Cap = [0.5, 2.0] then renormalized to mean 1.0, so profit can")
        print("  break ties between plausible crops without overriding agronomy.")
        assert np.all(pw > 0) and np.all(np.isfinite(pw)), "weights must be positive+finite"

        # ---- Stage 3 target scaling, fitted on TRAINING data only
        y_mean, y_std = 0.0, 1.0
        if spec.has_yield:
            y_mean = float(parts["train"]["yield_"].mean())
            y_std = float(parts["train"]["yield_"].std()) or 1.0

        val_tuple = (parts["val"]["soil"], parts["val"]["clim"], parts["val"]["crop"],
                     parts["val"]["yield_"], parts["val"]["dur"])

        def build_and_fit(cfg, soil, clim, crop, yld, dur, val, log):
            model = HybridFarmModel(
                n_classes=n_classes, soil_dim=soil.shape[1], climate_dim=clim.shape[1],
                profit_weights=profit_weights,
                branch_hidden=cfg["branch_hidden"], branch_out=cfg["branch_out"],
                trunk_hidden=cfg["trunk_hidden"], dropout=cfg["dropout"],
                yield_mean=y_mean, yield_std=y_std, seed=self.seed,
            )
            acc = model.fit_neural(soil, clim, crop, yld, dur, val_data=val,
                                   epochs=epochs, lr=cfg["lr"], batch_size=256,
                                   patience=15, verbose=log)
            return model, acc

        tr = parts["train"]

        # ---- Stages 1-3: hyperparameter search on VALIDATION only
        print(f"\n{'=' * 62}\nSTAGES 1-3: AGF-BRC training (PWCE crop head)\n{'=' * 62}")
        if search:
            best_model, best_cfg, best_val = None, None, -1.0
            for i, cfg in enumerate(SEARCH_GRID, 1):
                print(f"\n  [config {i}/{len(SEARCH_GRID)}] {cfg}")
                model, val_acc = build_and_fit(cfg, tr["soil"], tr["clim"], tr["crop"],
                                               tr["yield_"], tr["dur"], val_tuple, verbose)
                print(f"  -> validation accuracy {val_acc:.4f}")
                if val_acc > best_val:
                    best_model, best_cfg, best_val = model, cfg, val_acc
            print(f"\n  Selected config (by VALIDATION accuracy {best_val:.4f}): {best_cfg}")
        else:
            best_cfg = DEFAULT_CONFIG
            print(f"\n  Using default config: {best_cfg}")
            best_model, best_val = build_and_fit(best_cfg, tr["soil"], tr["clim"],
                                                 tr["crop"], tr["yield_"], tr["dur"],
                                                 val_tuple, verbose)
            print(f"  -> validation accuracy {best_val:.4f}")

        self.model = best_model

        # ---- Stage 2 sanity check
        gate = self.model.gate_statistics(parts["val"]["soil"], parts["val"]["clim"])
        print(f"\n{'=' * 62}\nSTAGE 2: ATTENTION GATE DIAGNOSTICS\n{'=' * 62}")
        print(f"  mean gate activation      : {gate['mean']:.4f}")
        print(f"  std within a sample       : {gate['std_within_sample']:.4f}")
        print(f"  std across samples        : {gate['std_across_samples']:.4f}")
        print(f"  saturated <0.05 / >0.95   : {gate['frac_saturated_low']:.3f} / "
              f"{gate['frac_saturated_high']:.3f}")
        # Graded verdict. A gate that is pinned near 1.0 for every sample is
        # mathematically fine but functionally an identity pass-through, which
        # means Stage 2 is contributing little and should be reported as such
        # rather than being called "active" on a technicality.
        near_constant = gate["std_across_samples"] < 0.01
        saturated = gate["frac_saturated_high"] > 0.8 or gate["frac_saturated_low"] > 0.8
        if saturated and near_constant:
            verdict = ("NEAR-IDENTITY: gate is saturated and almost constant across "
                       "samples,\n     so Stage 2 fusion is close to plain "
                       "concatenation on this dataset")
        elif near_constant:
            verdict = ("WEAK: gate varies within a sample but barely between samples "
                       "(not input-adaptive)")
        elif saturated:
            verdict = "PARTIALLY SATURATED but still sample-dependent"
        elif 0.05 < gate["mean"] < 0.95:
            verdict = "ACTIVE and sample-dependent"
        else:
            verdict = "DEGENERATE - inspect"
        print(f"  -> gate is {verdict}")

        # ---- Stage 4: residual correction (yield/duration only)
        print(f"\n{'=' * 62}\nSTAGE 4: gradient-boosted residual correction\n{'=' * 62}")
        fitted = self.model.fit_residual_correctors(
            tr["soil"], tr["clim"], tr["yield_"], tr["dur"])
        if fitted:
            print(f"  XGBoost correctors fitted for: {', '.join(fitted)}")
        else:
            print("  SKIPPED - this dataset has no yield/duration targets, so there")
            print("  is no residual to correct. Stage 4 is a documented no-op here.")
        print("  (Crop classification is never affected by Stage 4.)")

        # ---- Baselines on the SAME split
        baselines = self._train_baselines(parts)

        # ---- k-fold CV for small datasets
        cv = None
        if cv_folds and len(df) < SMALL_DATASET_THRESHOLD:
            cv = self._cross_validate(df, crop_y, best_cfg, y_mean, y_std,
                                      profit_weights, n_classes, epochs, cv_folds)

        # ---- Single final evaluation on the untouched test split
        self.report = self._final_evaluation(parts, best_cfg, best_val, baselines,
                                             source_name, mean_mi, gate, chance, cv)
        return self.report

    def _train_baselines(self, parts) -> dict:
        """Reference classifiers on the identical train/test split."""
        Xtr = np.hstack([parts["train"]["soil"], parts["train"]["clim"]])
        Xte = np.hstack([parts["test"]["soil"], parts["test"]["clim"]])
        ytr, yte = parts["train"]["crop"], parts["test"]["crop"]

        out = {}
        lr = LogisticRegression(max_iter=2000, random_state=self.seed)
        lr.fit(Xtr, ytr)
        out["Logistic Regression"] = float((lr.predict(Xte) == yte).mean())

        rf = RandomForestClassifier(n_estimators=300, random_state=self.seed, n_jobs=-1)
        rf.fit(Xtr, ytr)
        out["Random Forest"] = float((rf.predict(Xte) == yte).mean())
        return out

    def _cross_validate(self, df, crop_y, cfg, y_mean, y_std, profit_weights,
                        n_classes, epochs, folds) -> dict:
        """
        Stratified k-fold CV of the full crop classifier, for datasets where a
        single 15 % test split is too small to be a stable estimate.

        DISCLOSURE: `cfg` was selected on the validation split of the single
        70/15/15 partition, whose rows reappear inside these CV folds. The CV
        number is therefore mildly OPTIMISTIC as a model-selection-inclusive
        estimate. The single-split test accuracy reported alongside it is the
        strictly unbiased figure. Preprocessing is still re-fit per fold.
        """
        print(f"\n{'=' * 62}\n{folds}-FOLD STRATIFIED CV (small dataset)\n{'=' * 62}")
        print("  Baselines are scored on the SAME folds, because a single 15 %")
        print("  test split of this dataset is only ~120 rows -- one sample there")
        print("  moves accuracy by ~0.8 pts, which is too noisy to rank models.")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        scores, base_scores = [], {"Logistic Regression": [], "Random Forest": []}
        for k, (i_tr, i_te) in enumerate(skf.split(np.zeros(len(df)), crop_y), 1):
            agent = FarmAgent(seed=self.seed, spec=self.spec)
            soil_tr, clim_tr = agent._encode_frame(df.iloc[i_tr], fit_scaler=True)
            soil_te, clim_te = agent._encode_frame(df.iloc[i_te], fit_scaler=False)

            spec = self.spec
            yld = (df.iloc[i_tr][spec.yield_col].to_numpy(dtype=np.float32)
                   if spec.has_yield else None)
            dur = (df.iloc[i_tr][spec.duration_col].to_numpy(dtype=np.float32)
                   if spec.has_duration else None)

            model = HybridFarmModel(
                n_classes=n_classes, soil_dim=soil_tr.shape[1],
                climate_dim=clim_tr.shape[1], profit_weights=profit_weights,
                branch_hidden=cfg["branch_hidden"], branch_out=cfg["branch_out"],
                trunk_hidden=cfg["trunk_hidden"], dropout=cfg["dropout"],
                yield_mean=y_mean, yield_std=y_std, seed=self.seed,
            )
            model.fit_neural(soil_tr, clim_tr, crop_y[i_tr], yld, dur,
                             val_data=(soil_te, clim_te, crop_y[i_te], None, None),
                             epochs=epochs, lr=cfg["lr"], patience=15, verbose=False)
            pred, _, _, _ = model.predict(soil_te, clim_te)
            acc = float((pred == crop_y[i_te]).mean())
            scores.append(acc)

            # Same folds, same encoded features, for a fair comparison.
            Xtr = np.hstack([soil_tr, clim_tr])
            Xte = np.hstack([soil_te, clim_te])
            for name, mdl in (
                ("Logistic Regression",
                 LogisticRegression(max_iter=2000, random_state=self.seed)),
                ("Random Forest",
                 RandomForestClassifier(n_estimators=300, random_state=self.seed,
                                        n_jobs=-1)),
            ):
                mdl.fit(Xtr, crop_y[i_tr])
                base_scores[name].append(float((mdl.predict(Xte) == crop_y[i_te]).mean()))

            print(f"  fold {k}/{folds}: n_test={len(i_te):>4,}  AGF-BRC {acc:.4f}  "
                  f"| LR {base_scores['Logistic Regression'][-1]:.4f}"
                  f"  RF {base_scores['Random Forest'][-1]:.4f}")

        scores = np.asarray(scores)
        print(f"\n  {'AGF-BRC':<22s} mean {scores.mean():.4f}  std {scores.std():.4f}  "
              f"min {scores.min():.4f}  max {scores.max():.4f}")
        base_summary = {}
        for name, vals in base_scores.items():
            v = np.asarray(vals)
            base_summary[name] = dict(mean=float(v.mean()), std=float(v.std()))
            print(f"  {name:<22s} mean {v.mean():.4f}  std {v.std():.4f}  "
                  f"min {v.min():.4f}  max {v.max():.4f}")
        print("\n  NOTE: the config was chosen on the single split's validation set,")
        print("  whose rows recur in these folds, so treat the AGF-BRC number as")
        print("  mildly optimistic. The single-split test accuracy is the unbiased")
        print("  one. Baselines had no comparable tuning, so if a baseline wins here")
        print("  that result stands.")
        return dict(mean=float(scores.mean()), std=float(scores.std()),
                    folds=folds, scores=scores.tolist(), baselines=base_summary)

    def _final_evaluation(self, parts, cfg, val_acc, baselines, source_name,
                          mean_mi, gate, chance, cv) -> dict:
        test = parts["test"]
        crop_pred, _, yield_pred, dur_pred = self.model.predict(test["soil"], test["clim"])
        acc = float((crop_pred == test["crop"]).mean())

        yield_mae = dur_mae = None
        if yield_pred is not None and test["yield_"] is not None:
            yield_mae = float(np.mean(np.abs(yield_pred - test["yield_"])))
        if dur_pred is not None and test["dur"] is not None:
            dur_mae = float(np.mean(np.abs(dur_pred - test["dur"])))

        print(f"\n{'=' * 62}\nAGF-BRC FINAL EVALUATION\n{'=' * 62}")
        print(f"\nDataset:              {source_name}")
        print(f"  Training samples:   {len(parts['train']['crop']):,}")
        print(f"  Validation samples: {len(parts['val']['crop']):,}")
        print(f"  Test samples:       {len(test['crop']):,}")
        print(f"  Classes:            {len(self.crop_list)}")
        print(f"\nBest configuration (selected on validation): {cfg}")
        print(f"  Validation crop accuracy: {val_acc * 100:.2f} %")
        print(f"\nCrop Classification Accuracy: {acc * 100:.2f} %   "
              f"(chance = {chance * 100:.2f} %)")
        if cv:
            print(f"{cv['folds']}-fold CV accuracy:            "
                  f"{cv['mean'] * 100:.2f} % +/- {cv['std'] * 100:.2f} %")
        print(f"Yield MAE:                    "
              f"{f'{yield_mae:.3f} tons/hectare' if yield_mae is not None else 'n/a (no yield target in this dataset)'}")
        print(f"Duration MAE:                 "
              f"{f'{dur_mae:.2f} days' if dur_mae is not None else 'n/a (no duration target in this dataset)'}")

        print(f"\n{'=' * 62}\nPER-CLASS PERFORMANCE\n{'=' * 62}")
        print(classification_report(test["crop"], crop_pred,
                                    labels=range(len(self.crop_list)),
                                    target_names=self.crop_list, digits=3,
                                    zero_division=0))

        print(f"{'=' * 62}\nCONFUSION MATRIX (rows = actual, cols = predicted)\n{'=' * 62}")
        cm = confusion_matrix(test["crop"], crop_pred, labels=range(len(self.crop_list)))
        print(pd.DataFrame(cm, index=self.crop_list, columns=self.crop_list).to_string())

        print(f"\n{'=' * 62}\nPREDICTION DISTRIBUTION (collapse check)\n{'=' * 62}")
        pred_counts = np.bincount(crop_pred, minlength=len(self.crop_list))
        true_counts = np.bincount(test["crop"], minlength=len(self.crop_list))
        for i, crop in enumerate(self.crop_list):
            print(f"  {crop:<12s} predicted {pred_counts[i]:>6,}   (actual {true_counts[i]:>6,})")
        n_used = int((pred_counts > 0).sum())
        collapsed = (n_used < len(self.crop_list)
                     or pred_counts.max() > 0.6 * len(test["crop"]))
        print(f"  -> {n_used}/{len(self.crop_list)} classes predicted; "
              f"{'COLLAPSE DETECTED' if collapsed else 'no collapse'}")
        if collapsed and mean_mi < 0.005:
            # Expected, not a defect: with zero mutual information the crop
            # loss is flat across classes, so the PWCE weight vector is the
            # only remaining gradient signal and the optimum is to always
            # predict the highest-profit crop.
            weights = get_profit_weight_vector(self.crop_list)
            top = self.crop_list[int(np.argmax(weights))]
            print("     Cause: no crop signal + PWCE. With a flat likelihood the")
            print("     profit weights dominate, so the loss-optimal policy is to")
            print(f"     always predict the max-weight crop ({top}). This")
            print("     collapse disappears on data that contains crop signal.")

        print(f"\n{'=' * 62}\nBASELINE VS AGF-BRC (identical split)\n{'=' * 62}")
        print(f"  {'model':<22s} {'single split':>13s}" + (f" {'k-fold CV':>18s}" if cv else ""))
        cv_base = (cv or {}).get("baselines", {})
        for name, score in baselines.items():
            line = f"  {name:<22s} {score * 100:12.2f} %"
            if name in cv_base:
                line += (f" {cv_base[name]['mean'] * 100:12.2f} % "
                         f"+/-{cv_base[name]['std'] * 100:.2f}")
            print(line)
        line = f"  {'AGF-BRC':<22s} {acc * 100:12.2f} %"
        if cv:
            line += f" {cv['mean'] * 100:12.2f} % +/-{cv['std'] * 100:.2f}"
        print(line)

        # Report honestly when a plain baseline beats the proposed model.
        if cv_base:
            winner = max(list(cv_base.items()) + [("AGF-BRC", {"mean": cv["mean"]})],
                         key=lambda kv: kv[1]["mean"])
            if winner[0] != "AGF-BRC":
                margin = (winner[1]["mean"] - cv["mean"]) * 100
                print(f"\n  HONEST NOTE: {winner[0]} outperforms AGF-BRC by "
                      f"{margin:.2f} pts on k-fold CV.")
                print("  With only a few hundred training rows and 7 features, tree")
                print("  ensembles typically beat neural nets on tabular data. AGF-BRC's")
                print("  justification on this dataset is therefore NOT raw accuracy; it")
                print("  is the multi-task heads plus the economic coupling (PWCE at")
                print("  training time, guarded profit re-ranking at inference), which")
                print("  the baselines do not provide. Do not claim an accuracy win.")
        if mean_mi < 0.005:
            print("\n  NOTE: the crop-signal audit found no usable signal in this")
            print("  dataset, so every model -- baselines included -- is pinned to")
            print("  the chance rate. That is a data ceiling, not a model defect.")
        print(f"{'=' * 62}\n")

        return dict(
            source=source_name, accuracy=acc, val_accuracy=val_acc, chance=chance,
            yield_mae=yield_mae, duration_mae=dur_mae, config=cfg, cv=cv,
            baselines=baselines, mean_mutual_information=mean_mi,
            confusion_matrix=cm.tolist(), gate=gate, crops=self.crop_list,
            n_train=len(parts["train"]["crop"]), n_val=len(parts["val"]["crop"]),
            n_test=len(test["crop"]),
        )

    # -------------------------------------------------------------- Stage 5
    def _stage5(self, probs, yield_pred, dur_pred, land_area_ha, sowing_date,
                plausibility_floor, plausibility_ratio):
        """
        STAGE 5 - confidence-guarded economic re-ranking
        [RULE-BASED, deterministic: no learned parameters below this line].

        Output contract is FIXED: exactly 1 best crop + exactly 3 alternatives.
        """
        pool = min(CANDIDATE_POOL_SIZE, len(self.crop_list))

        # Step 1: restrict to the top-N crops by classification probability.
        ranked_idx = [int(i) for i in np.argsort(-probs)[:pool]]
        argmax_idx = ranked_idx[0]
        top_prob = float(probs[argmax_idx])

        # Step 2: plausibility guardrail, so an expensive but low-confidence
        # crop can never win purely on price.
        threshold = max(plausibility_floor, top_prob * plausibility_ratio)

        # Step 3: candidate economics. The argmax crop uses the model's own
        # residual-corrected yield IF the model was trained to predict yield;
        # otherwise every candidate is priced from the reference table and
        # labelled as such, so a table value is never passed off as a
        # model prediction.
        model_has_yield = yield_pred is not None
        candidates = []
        for idx in ranked_idx:
            crop = self.idx_to_crop[idx]
            prob = float(probs[idx])
            is_plausible = prob >= threshold

            if idx == argmax_idx and model_has_yield:
                yield_est = float(yield_pred[0])
                duration_days = (int(round(float(dur_pred[0]))) if dur_pred is not None
                                 else get_reference_duration(crop))
                source = "model_prediction"
            else:
                yield_est = get_reference_yield(crop)
                duration_days = get_reference_duration(crop)
                source = ("reference_table_estimate" if is_plausible
                          else "not_plausible_for_profit_ranking")

            candidates.append(dict(
                crop=crop, confidence=round(prob, 3),
                expected_yield_t_per_ha=round(yield_est, 2),
                expected_profit_inr=get_expected_profit(crop, yield_est, land_area_ha),
                estimate_source=source, duration_days=duration_days,
                is_plausible=is_plausible,
            ))

        # Step 4: best crop = highest expected profit among the PLAUSIBLE
        # subset only. The argmax crop is plausible by construction.
        plausible = [c for c in candidates if c["is_plausible"]]
        best = max(plausible, key=lambda c: c["expected_profit_inr"])

        # Step 5: alternatives = the remaining candidates, in probability order.
        alternatives = [c for c in candidates if c["crop"] != best["crop"]]

        harvest_date = sowing_date + dt.timedelta(days=best["duration_days"])

        def _public(candidate, include_harvest):
            out = {
                "crop": candidate["crop"],
                "confidence": candidate["confidence"],
                "expected_yield_t_per_ha": candidate["expected_yield_t_per_ha"],
                "expected_profit_inr": candidate["expected_profit_inr"],
                "estimate_source": candidate["estimate_source"],
            }
            if include_harvest:
                out["harvest_month"] = calendar.month_name[harvest_date.month]
                out["harvest_date_estimate"] = harvest_date.isoformat()
            return out

        return {
            "best_crop": _public(best, include_harvest=True),
            "alternatives": [_public(c, include_harvest=False) for c in alternatives],
        }

    # -------------------------------------------------------------- predict
    def recommend(self, land_area_ha: float = 1.0, sowing_date: dt.date = None,
                  plausibility_floor: float = 0.03, plausibility_ratio: float = 0.15,
                  **features):
        """
        Main entry point. Feature keyword arguments depend on the dataset spec:

          crop_reco : N, P, K, ph, temperature, humidity, rainfall
          legacy    : region, soil_type, month, irrigation_available,
                      fertilizer_planned, and optionally water_availability_mm
                      (see recommend_legacy, which this delegates to)

        Returns a dict with a FIXED shape:
            {
              "best_crop":    {crop, confidence, expected_yield_t_per_ha,
                               harvest_month, harvest_date_estimate,
                               expected_profit_inr, estimate_source},
              "alternatives": [3 dicts: {crop, confidence,
                               expected_yield_t_per_ha, expected_profit_inr,
                               estimate_source}],
            }

        NOTE for the viva: the crop chosen here is an ECONOMIC decision, not
        the classifier's accuracy metric. Held-out classification accuracy is
        measured separately in train(), directly from the crop head's argmax,
        and is never computed from this profit-ranked output.
        """
        if self.model is None:
            raise RuntimeError("Call .train() before .recommend()")

        if self.spec is LEGACY_SPEC or self.spec.name == "legacy":
            return self.recommend_legacy(
                land_area_ha=land_area_ha, sowing_date=sowing_date,
                plausibility_floor=plausibility_floor,
                plausibility_ratio=plausibility_ratio, **features)

        soil_x, climate_x = self._encode_single(**features)
        _, crop_probs, yield_pred, dur_pred = self.model.predict(soil_x, climate_x)
        sowing_date = sowing_date or dt.date.today()
        return self._stage5(crop_probs[0], yield_pred, dur_pred, land_area_ha,
                            sowing_date, plausibility_floor, plausibility_ratio)

    def recommend_legacy(self, region: str, soil_type: str, month: int,
                         irrigation_available: bool, fertilizer_planned: bool,
                         water_availability_mm: float = None,
                         land_area_ha: float = 1.0, sowing_date: dt.date = None,
                         plausibility_floor: float = 0.03,
                         plausibility_ratio: float = 0.15):
        """
        Legacy farmer-facing interface for the crop_yield.csv / synthetic
        schema, where the farmer supplies a sowing month rather than numeric
        climate readings. `month` selects seasonal climate defaults.
        """
        if region not in REGIONS or soil_type not in SOIL_TYPES:
            raise ValueError(f"region must be one of {REGIONS}; "
                             f"soil_type must be one of {SOIL_TYPES}")
        if month not in MONTH_CLIMATE_DEFAULTS:
            raise ValueError("month must be an integer 1-12")

        rain_default, temp_default, weather_default = MONTH_CLIMATE_DEFAULTS[month]
        rainfall = water_availability_mm if water_availability_mm is not None else rain_default

        soil_x, climate_x = self._encode_single(
            Region=region, Soil_Type=soil_type,
            Rainfall_mm=rainfall, Temperature_Celsius=temp_default,
            Fertilizer_Used=fertilizer_planned, Irrigation_Used=irrigation_available,
            Weather_Condition=weather_default,
        )
        _, crop_probs, yield_pred, dur_pred = self.model.predict(soil_x, climate_x)
        sowing_date = sowing_date or dt.date(dt.date.today().year, month, 15)
        return self._stage5(crop_probs[0], yield_pred, dur_pred, land_area_ha,
                            sowing_date, plausibility_floor, plausibility_ratio)


# ---------------------------------------------------------------------------
# Post-training test harness [deterministic, scriptable]
# ---------------------------------------------------------------------------
# Fixed lists of farmer inputs rather than interactive input(), so runs stay
# reproducible and can be executed in CI. Values are round numbers chosen to
# sit inside each dataset's observed feature ranges; they are NOT copied rows.
CROP_RECO_TEST_INSTANCES = [
    dict(N=90,  P=42, K=43, ph=6.5, temperature=21, humidity=82, rainfall=200,
         land_area_ha=2.0),
    dict(N=80,  P=45, K=20, ph=6.8, temperature=24, humidity=65, rainfall=90,
         land_area_ha=1.0),
    dict(N=120, P=45, K=20, ph=6.2, temperature=26, humidity=80, rainfall=80,
         land_area_ha=1.5),
    dict(N=20,  P=68, K=20, ph=6.0, temperature=28, humidity=50, rainfall=60,
         land_area_ha=3.25),
    dict(N=40,  P=60, K=80, ph=7.0, temperature=18, humidity=20, rainfall=70,
         land_area_ha=0.75),
]

LEGACY_TEST_INSTANCES = [
    dict(region="North", soil_type="Loam", month=7, irrigation_available=True,
         fertilizer_planned=True, land_area_ha=2.0),
    dict(region="South", soil_type="Clay", month=11, irrigation_available=False,
         fertilizer_planned=False, land_area_ha=1.0),
    dict(region="West", soil_type="Sandy", month=6, irrigation_available=True,
         fertilizer_planned=False, land_area_ha=1.5, water_availability_mm=60),
    dict(region="East", soil_type="Silt", month=1, irrigation_available=True,
         fertilizer_planned=True, land_area_ha=3.25),
    dict(region="North", soil_type="Chalky", month=4, irrigation_available=False,
         fertilizer_planned=True, land_area_ha=0.75, water_availability_mm=200),
]

# Kept for backwards compatibility with earlier scripts.
TEST_INSTANCES = LEGACY_TEST_INSTANCES


def main():
    parser = argparse.ArgumentParser(description="AgriChainAI Farm Agent (AGF-BRC)")
    parser.add_argument("--data", choices=["crop_reco", "auto", "real", "synthetic"],
                        default="crop_reco",
                        help="which dataset to train on (default: crop_reco)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=60000)
    parser.add_argument("--no-search", action="store_true",
                        help="skip the validation hyperparameter search")
    parser.add_argument("--no-cv", action="store_true",
                        help="skip k-fold CV on small datasets")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = FarmAgent(seed=RANDOM_SEED)
    report = agent.train(
        epochs=args.epochs, data_source=args.data, max_samples=args.max_samples,
        search=not args.no_search, verbose=not args.quiet,
        cv_folds=0 if args.no_cv else 5,
    )

    instances = (CROP_RECO_TEST_INSTANCES if args.data == "crop_reco"
                 else LEGACY_TEST_INSTANCES)

    print("=" * 62)
    print("POST-TRAINING TEST HARNESS (Stage 5 economic recommendation)")
    print("=" * 62)
    for i, instance in enumerate(instances, 1):
        result = agent.recommend(**instance)
        print(f"\n--- Test instance {i}: {instance} ---")
        print(json.dumps(result, indent=2))
        assert len(result["alternatives"]) == N_ALTERNATIVES, (
            f"expected {N_ALTERNATIVES} alternatives, got {len(result['alternatives'])}")

    print(f"\nAll {len(instances)} test instances returned "
          f"1 best crop + {N_ALTERNATIVES} alternatives.")
    print(f"\nHeld-out crop classification accuracy: {report['accuracy'] * 100:.2f} % "
          f"(chance {report['chance'] * 100:.2f} %)")
    if report.get("cv"):
        print(f"{report['cv']['folds']}-fold CV accuracy: "
              f"{report['cv']['mean'] * 100:.2f} % +/- {report['cv']['std'] * 100:.2f} %")
    if report["mean_mutual_information"] < 0.005:
        print("\n" + "!" * 62)
        print("This dataset contains NO crop signal (see the crop-signal audit),")
        print(f"so {report['chance'] * 100:.1f} % is its Bayes-optimal ceiling and the")
        print("reported number is honest, not a defect. Baselines match it.")
        print("For a dataset that DOES contain crop signal, run:")
        print("    python3 farm_agent.py --data crop_reco")
        print("!" * 62)


if __name__ == "__main__":
    main()
