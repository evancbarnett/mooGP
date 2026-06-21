"""End-to-end smoke test for the MOOGP fit + predict pipeline.

Numerical kernels and fast paths are covered in test_fast_paths.py; this file
verifies the full public API on a small real dataset. Tests for the example
scripts (Forrester illustrations, nuclear-data helpers) live alongside the
scripts in examples/ and aren't part of the package regression suite.
"""

from __future__ import annotations

import numpy as np

from moogp.datasets import generate_borehole_data_nd
from moogp.model import MOOGP


def test_moogp_borehole_fit_and_predict():
    """Small fixed-Psi borehole fit should converge and reproduce training data."""
    data = generate_borehole_data_nd(n=10, p=2, seed=11)
    X = data["X_scaled"]
    Y = data["Y"]
    d = X.shape[1]
    p = Y.shape[1]
    q = p

    rng = np.random.default_rng(0)
    Psi = rng.standard_normal((p, q))
    Psi /= np.maximum(np.linalg.norm(Psi, axis=0, keepdims=True), 1e-12)

    theta0 = []
    bounds = []
    for _ in range(q):
        theta0.append(np.log(1.0))
        theta0.extend(list(np.log(0.5) * np.ones(d)))
        bounds.append((np.log(1e-3), np.log(1e3)))
        bounds.extend([(np.log(0.05), np.log(5.0))] * d)
    theta0 = np.array(theta0)

    model = MOOGP(
        terms=[None] + list(range(1, d + 1)),
        q=q,
        Psi=Psi,
        learn_Psi=False,
        learn_sigma_eps=False,
        use_reml=False,
        jitter=1e-8,
        standardize_x=False,
        standardize_y=False,
    )
    model.fit(data=data, theta0=theta0, bounds=bounds,
              optimizer_opts={"maxiter": 200})

    Y_pred, Y_std = model.predict(X, return_std=True)
    assert Y_pred.shape == Y.shape == Y_std.shape
    assert np.all(np.isfinite(Y_pred))
    assert np.all(np.isfinite(Y_std))
    assert np.max(np.abs(Y_pred - Y)) < 2e-2
