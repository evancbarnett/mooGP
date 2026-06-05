"""Tests for the ``standardize_x`` mode added to ``MOOGP``."""

from __future__ import annotations

import numpy as np
import pytest

from src.moogp.model import (
    MOOGP,
    _compute_x_center_scale,
    _normalize_standardize_x_mode,
    compute_working_x,
)


def test_default_standardization_is_unitcube_and_zscore():
    """Regression guard for the documented MOOGP defaults.

    The kernel theory and the default length-scale bounds assume X is on the
    unit cube; the fast Psi/sigma_eps parameterization assumes Y is centered
    and unit-scaled. Both defaults are part of the public contract.
    """
    m = MOOGP(terms=[None], q=1)
    assert m.standardize_x == "unitcube"
    assert m.standardize_y == "zscore"
    # x_margin default was picked via a sweep on VAH fold 1 (see
    # notebooks/vah_moogp_diagnostic.ipynb section 8).
    assert m.x_margin == pytest.approx(0.1)


def test_normalize_standardize_x_mode_accepts_documented_values():
    assert _normalize_standardize_x_mode(False) is False
    assert _normalize_standardize_x_mode(None) is False
    assert _normalize_standardize_x_mode(True) == "unitcube"
    assert _normalize_standardize_x_mode("unitcube") == "unitcube"
    assert _normalize_standardize_x_mode("UnitCube") == "unitcube"


def test_normalize_standardize_x_mode_rejects_unknown_modes():
    with pytest.raises(ValueError, match="standardize_x"):
        _normalize_standardize_x_mode("zscore")


def test_compute_x_center_scale_identity_when_disabled():
    X = np.array([[0.0, 100.0], [3.0, 200.0]])
    center, scale = _compute_x_center_scale(X, False)
    np.testing.assert_array_equal(center, np.zeros(2))
    np.testing.assert_array_equal(scale, np.ones(2))


def test_compute_x_center_scale_maps_unit_cube_to_pm_one():
    X = np.array([[0.0, 100.0], [4.0, 300.0], [2.0, 200.0]])
    center, scale = _compute_x_center_scale(X, "unitcube")
    np.testing.assert_allclose(center, np.array([2.0, 200.0]))
    np.testing.assert_allclose(scale, np.array([2.0, 100.0]))

    X_work, c2, s2 = compute_working_x(X, True)
    np.testing.assert_allclose(c2, center)
    np.testing.assert_allclose(s2, scale)
    np.testing.assert_allclose(X_work.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(X_work.max(axis=0), [1.0, 1.0])


def test_x_margin_widens_unitcube_scale_proportionally():
    X = np.array([[0.0, 10.0], [4.0, 20.0], [2.0, 14.0]])
    base_center, base_scale = _compute_x_center_scale(X, "unitcube", margin=0.0)
    padded_center, padded_scale = _compute_x_center_scale(X, "unitcube", margin=0.25)
    np.testing.assert_allclose(padded_center, base_center)
    np.testing.assert_allclose(padded_scale, 1.25 * base_scale)

    X_work, _, _ = compute_working_x(X, "unitcube", margin=0.25)
    # Training points land in [-1/(1+m), 1/(1+m)] instead of [-1, 1].
    np.testing.assert_allclose(X_work.min(axis=0), [-1.0 / 1.25, -1.0 / 1.25])
    np.testing.assert_allclose(X_work.max(axis=0), [1.0 / 1.25, 1.0 / 1.25])


def test_negative_x_margin_rejected():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="non-negative"):
        _compute_x_center_scale(X, "unitcube", margin=-0.01)
    with pytest.raises(ValueError, match="non-negative"):
        MOOGP(terms=[None], q=1, x_margin=-0.1)


def test_compute_x_center_scale_constant_column_does_not_divide_by_zero():
    X = np.array([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
    center, scale = _compute_x_center_scale(X, "unitcube")
    assert scale[0] == 1.0  # constant column collapses to scale = 1 (avoid /0)
    X_work, _, _ = compute_working_x(X, "unitcube")
    np.testing.assert_allclose(X_work[:, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(X_work[:, 1], [-1.0, 0.0, 1.0])


def _toy_two_output_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    # Raw X in an awkward range to confirm internal scaling cleans it up.
    n, d, _ = 25, 2, 2
    X_raw = rng.uniform(low=-3.0, high=7.0, size=(n, d))
    Y_raw = np.column_stack(
        [
            np.sin(X_raw[:, 0]) + 0.1 * X_raw[:, 1],
            np.cos(X_raw[:, 1]) - 0.05 * X_raw[:, 0],
        ]
    )
    return X_raw, Y_raw


def test_fit_predict_with_unitcube_x_matches_a_pre_scaled_fit():
    """A model with ``standardize_x='unitcube'`` on raw X should reproduce a
    model fit on the same X pre-scaled to ``[-1, 1]`` (no internal scaling)."""
    X_raw, Y_raw = _toy_two_output_problem(seed=42)

    x_center = 0.5 * (X_raw.min(axis=0) + X_raw.max(axis=0))
    x_half = 0.5 * (X_raw.max(axis=0) - X_raw.min(axis=0))
    X_pre = (X_raw - x_center) / x_half

    n, d = X_raw.shape
    q = 2

    theta0 = []
    bounds = []
    for _ in range(q):
        theta0.append(float(np.log(1.0)))
        theta0.extend([float(np.log(0.5))] * d)
        bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
        bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
    theta0 = np.asarray(theta0, dtype=float)
    y_var = np.maximum(1e-12, np.var(Y_raw, axis=0, ddof=1))
    theta0 = np.concatenate([theta0, np.log(1e-2 * y_var / y_var)])  # sigma in working scale
    bounds = bounds + [(float(np.log(1e-6)), float(np.log(0.5)))] * Y_raw.shape[1]

    common = dict(
        terms=[None] + list(range(1, d + 1)),
        q=q,
        Psi=None,
        orthogonal=True,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=1e-6,
        one_based=True,
        normalize_cols=True,
        use_diagonalized_interaction=True,
        use_slow_kyinv=False,
        standardize_y="zscore",
    )

    # Force margin=0 on both sides so internal unit-cube scaling matches the
    # manual pre-scaling exactly. (The library default x_margin > 0 widens the
    # cube by 1+m, which would offset the two fits relative to each other.)
    m_raw = MOOGP(**common, standardize_x="unitcube", x_margin=0.0)
    m_raw.fit({"X": X_raw, "Y": Y_raw}, theta0=theta0.copy(), bounds=list(bounds),
              optimizer_opts={"maxiter": 50})

    m_pre = MOOGP(**common, standardize_x=False)
    m_pre.fit({"X_scaled": X_pre, "Y": Y_raw}, theta0=theta0.copy(), bounds=list(bounds),
              optimizer_opts={"maxiter": 50})

    rng = np.random.default_rng(7)
    Xstar_raw = rng.uniform(low=-3.0, high=7.0, size=(8, d))
    Xstar_pre = (Xstar_raw - x_center) / x_half

    mean_raw, std_raw = m_raw.predict(Xstar_raw, return_std=True)
    mean_pre, std_pre = m_pre.predict(Xstar_pre, return_std=True)

    np.testing.assert_allclose(mean_raw, mean_pre, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(std_raw, std_pre, atol=1e-8, rtol=1e-6)


def test_x_keys_accepts_both_x_and_x_scaled_aliases():
    X_raw, Y_raw = _toy_two_output_problem(seed=0)
    d = X_raw.shape[1]
    q = 2

    theta0 = []
    bounds = []
    for _ in range(q):
        theta0.append(float(np.log(1.0)))
        theta0.extend([float(np.log(0.5))] * d)
        bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
        bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
    theta0 = np.asarray(theta0, dtype=float)
    theta0 = np.concatenate([theta0, np.log(np.full(Y_raw.shape[1], 1e-2))])
    bounds = bounds + [(float(np.log(1e-6)), float(np.log(0.5)))] * Y_raw.shape[1]

    common = dict(
        terms=[None] + list(range(1, d + 1)),
        q=q, Psi=None, orthogonal=True, learn_Psi=False, learn_sigma_eps=True,
        jitter=1e-6, one_based=True, normalize_cols=True,
        use_diagonalized_interaction=True, use_slow_kyinv=False,
        standardize_y="zscore", standardize_x="unitcube",
    )
    m_a = MOOGP(**common)
    m_a.fit({"X": X_raw, "Y": Y_raw}, theta0=theta0.copy(), bounds=list(bounds),
            optimizer_opts={"maxiter": 5})
    m_b = MOOGP(**common)
    m_b.fit({"X_scaled": X_raw, "Y": Y_raw}, theta0=theta0.copy(), bounds=list(bounds),
            optimizer_opts={"maxiter": 5})
    rng = np.random.default_rng(0)
    Xstar = rng.uniform(low=-3.0, high=7.0, size=(4, d))
    np.testing.assert_allclose(m_a.predict(Xstar), m_b.predict(Xstar), atol=1e-12)
