"""
Model-supported prediction explanations.

[SUPERVISED explainability — permutation importance on held-out data]
Only reports factors present in the trained feature set.
"""

from __future__ import annotations

import numpy as np


def permutation_importance(model, X, y, feature_names: list[str], n_repeats: int = 5) -> dict[str, float]:
    """Mean decrease in accuracy when each feature is shuffled."""
    baseline = float((model.predict(X, verbose=0).argmax(axis=1) == y).mean())

    importances = {}
    X = np.asarray(X, dtype=np.float32)
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            np.random.shuffle(Xp[:, j])
            acc = float((model.predict(Xp, verbose=0).argmax(axis=1) == y).mean())
            drops.append(baseline - acc)
        importances[name] = float(np.mean(drops))
    return importances


def format_explanation(
    importances: dict[str, float],
    feature_display_map: dict[str, str],
    top_k: int = 4,
    min_importance: float = 0.01,
) -> list[str]:
    """Human-readable bullets from permutation importances."""
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    lines = []
    for name, score in ranked[:top_k]:
        if score < min_importance:
            continue
        label = feature_display_map.get(name, name)
        lines.append(f"- {label} (importance {score:.2f})")
    return lines or ["- model features did not show strong single-factor dominance"]


# Display names for dried-fruit features
DRYFRUIT_DISPLAY = {
    "storage_duration_days": "storage duration",
    "storage_months": "months in storage",
    "dryer_type_CMD": "cabinet mixed-mode drying (CMD)",
    "dryer_type_TD": "tunnel drying (TD)",
    "packaging_None": "unpackaged storage",
    "packaging_HDPE": "HDPE packaging",
    "packaging_LDPE": "LDPE packaging",
    "fruit_Mango": "mango product type",
    "fruit_Pineapple": "pineapple product type",
}

PANEER_DISPLAY = {
    "storage_day": "storage duration",
    "mean_temperature": "mean storage temperature",
    "mean_rh": "mean relative humidity",
    "temperature_range": "temperature variation",
    "time_above_safe_threshold_ratio": "time above safe temperature threshold",
    "compressor_on_ratio": "compressor activity level",
    "fridge_open_ratio": "fridge opening frequency",
}
