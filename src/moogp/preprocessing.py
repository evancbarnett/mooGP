"""Dependency-light preprocessing shared by models and benchmark adapters."""

from __future__ import annotations

import numpy as np


def resolve_q_from_var_threshold(Y, var_threshold):
    """Choose the smallest latent rank whose variance strictly exceeds a target."""
    if not (0.0 < var_threshold < 1.0):
        raise ValueError(f"var_threshold must lie in (0, 1); got {var_threshold}.")
    Y = np.asarray(Y, float)
    svals = np.linalg.svd(Y, compute_uv=False)
    cumvar = np.cumsum(svals**2) / np.sum(svals**2)
    return int(np.argmax(cumvar > var_threshold) + 1)


def _normalize_standardize_y_mode(standardize_y):
    """Normalize user-facing output-standardization options."""
    if standardize_y in (False, None):
        return False
    if standardize_y is True:
        return "zscore"
    if isinstance(standardize_y, str):
        mode = standardize_y.lower()
        if mode in {"zscore", "robust"}:
            return mode
    raise ValueError("standardize_y must be one of False, True, 'zscore', or 'robust'.")


def _compute_y_center_scale(Y, mode):
    """Return per-output center and spread for the working response scale."""
    Y = np.asarray(Y, float)
    p = Y.shape[1]

    if mode is False:
        return np.zeros(p, dtype=float), np.ones(p, dtype=float)

    if mode == "zscore":
        center = np.mean(Y, axis=0)
        scale = np.std(Y, axis=0, ddof=1)
    elif mode == "robust":
        center = np.median(Y, axis=0)
        scale = np.median(np.abs(Y - center[None, :]), axis=0)
    else:  # pragma: no cover - guarded by _normalize_standardize_y_mode
        raise ValueError(f"Unsupported standardize_y mode: {mode}")

    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray(center, float), np.asarray(scale, float)


def compute_working_y(Y, standardize_y):
    """Project raw outputs onto the model's internal working scale."""
    mode = _normalize_standardize_y_mode(standardize_y)
    center, scale = _compute_y_center_scale(Y, mode)
    Y_work = (np.asarray(Y, float) - center[None, :]) / scale[None, :]
    return Y_work, center, scale
