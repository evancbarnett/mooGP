"""End-to-end smoke tests for MOOGP on the included example problems.

These cover full fit + predict pipelines (not individual numerical kernels —
those live in test_fast_paths.py). One test per example dataset:

  * Borehole: small multi-output fit reproduces the training response.
  * Forrester: orthogonal MOOGP recovers the trend better than non-orthogonal
    MOGP on the log-LHS design (the central claim of the orthogonality work).
  * Nuclear data: index parsing and block-spec construction helpers.
"""

from __future__ import annotations

import numpy as np

from moogp.datasets import (
    generate_borehole_data_nd,
    generate_forrester_data,
    log_lhs_1d_rescaled,
)
from moogp.forrester_illustration import (
    fit_moogp_forrester,
    get_model_trend_betas_raw,
)
from moogp.model import MOOGP
from moogp.nuclear_data_experiment import (
    DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
    load_nuclear_dataset,
    make_block_specs,
    train_test_split_indices,
)


# ---------------------------------------------------------------------------
# Borehole pipeline
# ---------------------------------------------------------------------------


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
    # Fit interpolates training data closely (small but nonzero jitter/noise).
    assert np.max(np.abs(Y_pred - Y)) < 2e-2


# ---------------------------------------------------------------------------
# Forrester orthogonality recovery
# ---------------------------------------------------------------------------


def test_orthogonal_moogp_recovers_loglhs_trend_better_than_non_orthogonal():
    """On the log-LHS Forrester design, orthogonal MOOGP must recover the
    third-output linear trend better than the non-orthogonal MOGP baseline."""
    n_train = 25
    seed = 0
    true_beta_output3 = np.array([-1.5, -5.0], dtype=float)

    X_log = log_lhs_1d_rescaled(
        n_train, seed=seed, xmin=1e-3, cluster="right",
        include_x0=False, shuffle=False,
    )
    data_log = generate_forrester_data(
        n=n_train, seed=seed, with_error=True,
        error_per_output=[10.0, 1.0, 0.05], X_override=X_log,
    )

    moogp, _, _, _ = fit_moogp_forrester(
        n_train=n_train, seed=seed, orthogonal=True, Psi=np.eye(3),
        use_fast=False, data=data_log, maxiter=300,
    )
    mogp, _, _, _ = fit_moogp_forrester(
        n_train=n_train, seed=seed, orthogonal=False, Psi=np.eye(3),
        use_fast=False, data=data_log, maxiter=300,
    )

    beta_moogp = get_model_trend_betas_raw(moogp)[:, 2]
    beta_mogp = get_model_trend_betas_raw(mogp)[:, 2]
    err_moogp = np.linalg.norm(beta_moogp - true_beta_output3)
    err_mogp = np.linalg.norm(beta_mogp - true_beta_output3)

    assert moogp.cache["used_fast"] is False
    assert mogp.cache["used_fast"] is False
    assert err_moogp < err_mogp


# ---------------------------------------------------------------------------
# Nuclear-data experiment helpers
# ---------------------------------------------------------------------------


def test_nuclear_dataset_output_index_columns():
    dataset = load_nuclear_dataset()
    families = dataset["output_index"]
    names = [f.name for f in families]
    assert names == [
        "dNch_deta", "dET_deta", "dN_dy_pion", "dN_dy_kaon", "dN_dy_proton",
        "mean_pT_pion", "mean_pT_kaon", "mean_pT_proton", "pT_fluct", "v22",
    ]
    assert families[-1].start == 90
    assert families[-1].end == 98


def test_nuclear_block_specs_default_and_fixed_q():
    specs = make_block_specs(98, blocks=DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
                              q_fraction=0.25)
    assert [(s["start"], s["end"], s["q"]) for s in specs] == [
        (0, 30, 8), (30, 54, 6), (54, 78, 6), (78, 98, 5),
    ]
    assert all(s["q_rule"] == "ceil(0.25 * p_block)" for s in specs)

    specs_fixed = make_block_specs(98, blocks=DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
                                    q_fraction=0.25, fixed_q=4)
    assert [s["q"] for s in specs_fixed] == [4, 4, 4, 4]
    assert all(s["q_rule"] == "fixed_q=4" for s in specs_fixed)


def test_nuclear_train_test_split_reproducible_and_disjoint():
    train_idx, test_idx = train_test_split_indices(541, train_fraction=0.8, seed=42)
    assert train_idx.size == 432
    assert test_idx.size == 109
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.array_equal(
        np.sort(np.concatenate([train_idx, test_idx])), np.arange(541)
    )
