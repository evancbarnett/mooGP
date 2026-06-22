"""Tests for the diagonal error grouping (LCGP-style ``diag_error_structure``).

The grouping partitions the ``p`` outputs into blocks
``[p_1, ..., p_G]`` (with ``sum(p_g) == p``) so that
``Sigma_eps = bdiag(sigma_1^2 I_{p_1}, ..., sigma_G^2 I_{p_G})`` and only ``G``
free variances are estimated instead of ``p``.

The general analytical-grad vs autograd equivalence is covered in
``test_fast_paths.py``; this file focuses on the grouping-specific behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
from autograd import value_and_grad

from moogp.model import (
    MOOGP,
    aggregate_per_output_to_groups,
    expand_grouped_sigma_eps2,
    group_indices_for_outputs,
    normalize_diag_error_structure,
    unpack_theta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n=12, d=3, p=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(X[:, j0]) + 0.3 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _build_theta(q, d, sigma_groups, seed=0):
    rng = np.random.default_rng(seed + 1000)
    theta = []
    for _ in range(q):
        theta.append(float(np.log(rng.uniform(0.5, 1.5))))
        theta.extend([float(np.log(rng.uniform(0.4, 0.9))) for _ in range(d)])
    theta.extend(list(np.log(np.asarray(sigma_groups, dtype=float))))
    return np.asarray(theta, dtype=float)


def _make_model(X, Y, q, *, diag_error_structure=None, use_analytical_grad=True):
    d = X.shape[1]
    model = MOOGP(
        terms=[None] + list(range(1, d + 1)),
        q=q, Psi=None, orthogonal=True,
        learn_Psi=False, learn_sigma_eps=True,
        diag_error_structure=diag_error_structure,
        jitter=0.0, one_based=True, normalize_cols=True,
        use_diagonalized_interaction=True, use_slow_kyinv=False,
        standardize_y=False, use_analytical_grad=use_analytical_grad,
    )
    model._prepare_data({"X_scaled": X, "y": Y})
    return model


# ---------------------------------------------------------------------------
# Grouping helpers and validators
# ---------------------------------------------------------------------------


class TestGroupingHelpers:
    def test_normalize_default_is_one_per_output(self):
        assert normalize_diag_error_structure(None, 4) == (1, 1, 1, 1)

    def test_normalize_passthrough_when_valid(self):
        assert normalize_diag_error_structure([3, 2, 4], 9) == (3, 2, 4)

    def test_normalize_rejects_wrong_sum(self):
        with pytest.raises(ValueError, match="sum"):
            normalize_diag_error_structure([2, 2], 5)

    def test_normalize_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="positive"):
            normalize_diag_error_structure([2, 0, 1], 3)

    def test_expand_broadcasts_to_p(self):
        out = expand_grouped_sigma_eps2(np.array([1.0, 5.0, 9.0]), [2, 1, 3])
        np.testing.assert_array_equal(out, np.array([1.0, 1.0, 5.0, 9.0, 9.0, 9.0]))

    def test_aggregate_sums_per_output_grads_inside_groups(self):
        # Chain-rule identity used inside _nll_and_grad_fast: if sigma_l is the
        # group variance of output l, then dg/dtheta_g = sum_{l in g} df_l/dsigma_l.
        agg = aggregate_per_output_to_groups(np.array([0.1, 0.2, -0.3, 0.4, 0.5]), (2, 3))
        np.testing.assert_allclose(agg, np.array([0.3, 0.6]))

    def test_group_indices(self):
        np.testing.assert_array_equal(
            group_indices_for_outputs([2, 1, 3]), np.array([0, 0, 1, 2, 2, 2])
        )


class TestUnpackTheta:
    def test_grouped_theta_has_g_entries_not_p(self):
        # latent block length q*(d+1) = 3, then 2 grouped log-variances
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5),
                          np.log(0.04), np.log(0.25)])
        _, _, sigma = unpack_theta(theta, d=2, q=1, p=5,
                                   learn_sigma_eps=True, diag_error_structure=(2, 3))
        np.testing.assert_allclose(sigma, np.array([0.04, 0.04, 0.25, 0.25, 0.25]))

    def test_default_recovers_per_output_packing(self):
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5),
                          np.log(0.01), np.log(0.02), np.log(0.03)])
        _, _, sigma = unpack_theta(theta, d=2, q=1, p=3, learn_sigma_eps=True)
        np.testing.assert_allclose(sigma, np.array([0.01, 0.02, 0.03]))

    def test_too_short_theta_raises(self):
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5), np.log(0.04)])
        with pytest.raises(ValueError, match="Sigma_eps"):
            unpack_theta(theta, d=2, q=1, p=4, learn_sigma_eps=True,
                         diag_error_structure=(2, 2))


# ---------------------------------------------------------------------------
# Grouped NLL must reproduce the per-output NLL when groups are consistent
# ---------------------------------------------------------------------------


class TestGroupedNLLEquivalence:
    def test_ones_grouping_equals_default(self):
        """``[1, 1, ..., 1]`` is the trivial grouping and must equal default."""
        X, Y = _make_dataset(n=10, d=2, p=4, seed=3)
        q = 2
        sigma0 = np.array([0.05, 0.07, 0.04, 0.06])
        theta = _build_theta(q, X.shape[1], sigma0, seed=3)
        m_default = _make_model(X, Y, q, diag_error_structure=None)
        m_ones = _make_model(X, Y, q, diag_error_structure=[1] * Y.shape[1])
        nll_default = float(m_default._nll(theta, build_cache=False))
        nll_ones = float(m_ones._nll(theta, build_cache=False))
        np.testing.assert_allclose(nll_ones, nll_default, atol=1e-12)

    def test_grouped_equals_per_output_when_outputs_share_variance(self):
        X, Y = _make_dataset(n=12, d=2, p=6, seed=11)
        q = 2
        es = (2, 1, 3)
        sigma_groups = np.array([0.05, 0.10, 0.02])
        sigma_per_output = np.array([0.05, 0.05, 0.10, 0.02, 0.02, 0.02])
        m_grp = _make_model(X, Y, q, diag_error_structure=es)
        m_full = _make_model(X, Y, q, diag_error_structure=None)
        nll_grp = float(m_grp._nll(_build_theta(q, X.shape[1], sigma_groups, seed=11),
                                    build_cache=False))
        nll_full = float(m_full._nll(_build_theta(q, X.shape[1], sigma_per_output, seed=11),
                                      build_cache=False))
        np.testing.assert_allclose(nll_grp, nll_full, atol=1e-12)


# ---------------------------------------------------------------------------
# Analytical gradient under grouping (grouping-specific, complements the
# ungrouped equivalence test in test_fast_paths.py)
# ---------------------------------------------------------------------------


class TestAnalyticalGradientUnderGrouping:
    @pytest.mark.parametrize("es", [(2, 1, 3), (3, 3), (1, 5), (6,)])
    def test_grad_matches_autograd(self, es):
        p = sum(es)
        X, Y = _make_dataset(n=10, d=2, p=p, seed=7)
        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)
        sigma_groups = 0.01 + 0.005 * np.arange(len(es))
        theta = _build_theta(q, X.shape[1], sigma_groups, seed=7)
        nll_a, grad_a = model._nll_and_grad_fast(theta)
        nll_auto, grad_auto = value_and_grad(
            lambda th: model._nll(th, build_cache=False)
        )(theta)
        np.testing.assert_allclose(nll_a, float(nll_auto), atol=1e-12)
        np.testing.assert_allclose(grad_a, np.asarray(grad_auto, float), atol=1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_mismatched_sum_raises_at_prepare():
    X, Y = _make_dataset(n=8, d=2, p=4, seed=0)
    model = MOOGP(
        terms=[None, 1, 2], q=2, learn_sigma_eps=True,
        diag_error_structure=[2, 1],  # sums to 3 but p=4
        jitter=0.0,
    )
    with pytest.raises(ValueError, match="sum"):
        model._prepare_data({"X_scaled": X, "y": Y})


# ---------------------------------------------------------------------------
# End-to-end fit: outputs inside a group share variance, groups separate noise
# ---------------------------------------------------------------------------


def test_grouped_fit_ties_within_group_and_separates_across_groups():
    """Two groups of two outputs each, with very different true noise levels:
    fit must (a) tie outputs inside each group, (b) recover a noticeably larger
    variance for the noisier group, (c) produce sensible predictions."""
    n, d, p = 60, 1, 4
    es = (2, 2)
    small, large = 1e-3, 0.5
    true_per_output = expand_grouped_sigma_eps2(np.array([small, large]), es)

    rng = np.random.default_rng(2026)
    X_phys = rng.uniform(0.0, 1.0, size=(n, 1))
    X = 2.0 * X_phys - 1.0
    s1 = np.sin(3.0 * X_phys.ravel())
    s2 = np.cos(3.0 * X_phys.ravel())
    Y_clean = np.column_stack([s1, 0.7 * s1, s2, 0.7 * s2])
    Y = Y_clean + rng.normal(0.0, np.sqrt(true_per_output), size=Y_clean.shape)

    q = 2
    model = _make_model(X, Y, q, diag_error_structure=es)
    theta0 = _build_theta(q, d, np.array([0.05, 0.05]), seed=2026)
    lat_bounds = []
    for _ in range(q):
        lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
        lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
    bounds = lat_bounds + [(float(np.log(1e-8)), float(np.log(10.0)))] * len(es)

    nll0 = float(model._nll(theta0, build_cache=False))
    model.fit(data={"X_scaled": X, "y": Y}, theta0=theta0, bounds=bounds,
              optimizer_opts={"maxiter": 200})

    # NLL improved.
    assert model.nll_hat <= nll0 + 1e-6
    # theta_hat has G sigma entries, not p.
    assert model.theta_hat.size == q * (d + 1) + len(es)

    sigma_hat = model.cache["sigma_eps2"]
    # Outputs inside each group share fitted variance exactly.
    np.testing.assert_allclose(sigma_hat[0], sigma_hat[1], atol=1e-12)
    np.testing.assert_allclose(sigma_hat[2], sigma_hat[3], atol=1e-12)
    # Noisier group's variance is at least 5x larger.
    assert sigma_hat[2] > 5.0 * sigma_hat[0], (sigma_hat[0], sigma_hat[2])

    # Predict runs and returns sensible shapes/values.
    Xs = np.linspace(-1, 1, 7).reshape(-1, 1)
    mean, std = model.predict(Xs, return_std=True)
    assert mean.shape == (7, p)
    assert std.shape == (7, p)
    assert np.all(std >= 0)
