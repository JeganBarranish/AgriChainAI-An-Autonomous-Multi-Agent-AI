"""
Shared spoilage-risk calculation.

Single source of truth for converting AC-CSN class probabilities into the
0-100 % spoilage-risk score, so training/evaluation and inference can never
drift apart.

The class weights are defined at training time and persisted inside
models/spoilage_preprocessor.pkl (key: "risk_weights"). Inference must pass
those saved weights here rather than redefining its own.

Formula:
    risk % = P(FRESH) * w_fresh + P(WARNING) * w_warning + P(SPOILED) * w_spoiled

with the trained configuration:
    w_fresh = 0.0, w_warning = 50.0, w_spoiled = 100.0
"""

from __future__ import annotations

import numpy as np


def compute_spoilage_risk(
    probabilities: np.ndarray,
    class_labels: list[str],
    risk_weights: dict[str, float],
) -> float:
    """
    Weighted spoilage risk (0-100 %) from class probabilities.

    probabilities : 1-D array of class probabilities, ordered like class_labels
    class_labels  : label names in model output order (e.g. FRESH, WARNING, SPOILED)
    risk_weights  : per-label weights saved by the training pipeline
    """
    weights = np.array([risk_weights[label] for label in class_labels], dtype=float)
    return float(np.dot(np.asarray(probabilities, dtype=float), weights))
