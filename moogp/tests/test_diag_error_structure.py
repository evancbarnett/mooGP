"""Tests for the diagonal error grouping (LCGP-style ``diag_error_structure``).

The grouping partitions the ``p`` outputs into blocks
``[p1, p2, ..., pG]`` (with ``sum(p_g) == p``) so that
``Sigma_eps = bdiag(sigma_1^2 I_{p1}, ..., sigma_G^2 I_{pG})`` and only ``G``
free variances are estimated instead of ``p``.
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
        q=q,
        Psi=None,
        orthogonal=True,
        learn_Psi=False,
        learn_sigma_eps=True,
        diag_error_structure=diag_error_structure,
        jitter=0.0,
        one_based=True,
        normalize_cols=True,
        use_diagonalized_interaction=True,
        use_slow_kyinv=False,
        standardize_y=False,
        use_analytical_grad=use_analytical_grad,
    )
    model._prepare_data({"X_scaled": X, "y": Y})
    return model


# ---------------------------------------------------------------------------
# Helpers / validators
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_default_is_one_per_output(self):
        assert normalize_diag_error_structure(None, 4) == (1, 1, 1, 1)

    def test_passthrough_when_valid(self):
        assert normalize_diag_error_structure([3, 2, 4], 9) == (3, 2, 4)

    def test_rejects_wrong_sum(self):
        with pytest.raises(ValueError, match="sum"):
            normalize_diag_error_structure([2, 2], 5)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="positive"):
            normalize_diag_error_structure([2, 0, 1], 3)


class TestExpandAndAggregate:
    def test_expand_broadcasts_to_p(self):
        out = expand_grouped_sigma_eps2(np.array([1.0, 5.0, 9.0]), [2, 1, 3])
        np.testing.assert_array_equal(out, np.array([1.0, 1.0, 5.0, 9.0, 9.0, 9.0]))

    def test_aggregate_is_inverse_of_expand_for_constant_groups(self):
        es = (3, 2, 4)
        per_group = np.array([2.5, -1.0, 7.0])
        per_output = expand_grouped_sigma_eps2(per_group, es)
        # ``aggregate`` sums per-output values inside each group, so a constant
        # vector recovers the per-group value scaled by the group size.
        agg = aggregate_per_output_to_groups(per_output, es)
        np.testing.assert_allclose(agg, per_group * np.array(es))

    def test_aggregate_matches_chain_rule_for_grouped_log_var(self):
        # Reproduce the chain-rule identity used inside _nll_and_grad_fast:
        # if g(theta) = sum_l f_l(sigma_l(theta)) and sigma_l = sigma_{group(l)},
        # then dg/dtheta_g = sum_{l in group g} df_l/dsigma_l.
        es = (2, 3)
        per_output_grad = np.array([0.1, 0.2, -0.3, 0.4, 0.5])
        agg = aggregate_per_output_to_groups(per_output_grad, es)
        np.testing.assert_allclose(agg, np.array([0.1 + 0.2, -0.3 + 0.4 + 0.5]))

    def test_group_indices(self):
        np.testing.assert_array_equal(
            group_indices_for_outputs([2, 1, 3]),
            np.array([0, 0, 1, 2, 2, 2]),
        )


# ---------------------------------------------------------------------------
# unpack_theta with grouping
# ---------------------------------------------------------------------------


class TestUnpackTheta:
    def test_grouped_theta_has_g_entries_not_p(self):
        d, q, p = 2, 1, 5
        es = (2, 3)
        # latent block of length q*(d+1) = 3, then 2 grouped log-variances
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5), np.log(0.04), np.log(0.25)])
        _, _, sigma = unpack_theta(
            theta, d, q, p, learn_sigma_eps=True, diag_error_structure=es
        )
        np.testing.assert_allclose(sigma, np.array([0.04, 0.04, 0.25, 0.25, 0.25]))

    def test_default_recovers_per_output_packing(self):
        d, q, p = 2, 1, 3
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5),
                          np.log(0.01), np.log(0.02), np.log(0.03)])
        _, _, sigma = unpack_theta(theta, d, q, p, learn_sigma_eps=True)
        np.testing.assert_allclose(sigma, np.array([0.01, 0.02, 0.03]))

    def test_too_short_theta_raises(self):
        d, q, p = 2, 1, 4
        es = (2, 2)
        # Latent block (length 3) + only 1 group entry (need 2)
        theta = np.array([np.log(1.0), np.log(0.5), np.log(0.5), np.log(0.04)])
        with pytest.raises(ValueError, match="Sigma_eps"):
            unpack_theta(theta, d, q, p, learn_sigma_eps=True, diag_error_structure=es)


# ---------------------------------------------------------------------------
# Model behavior — equivalence and gradients
# ---------------------------------------------------------------------------


class TestEquivalenceWithDefault:
    """A grouping of ``[1] * p`` must reproduce the ungrouped (default) model
    exactly."""

    def test_nll_matches_default(self):
        X, Y = _make_dataset(n=10, d=2, p=4, seed=3)
        q = 2
        model_default = _make_model(X, Y, q, diag_error_structure=None)
        model_ones = _make_model(X, Y, q, diag_error_structure=[1] * Y.shape[1])

        sigma0 = np.array([0.05, 0.07, 0.04, 0.06])
        theta = _build_theta(q, X.shape[1], sigma0, seed=3)

        nll_default = float(model_default._nll(theta, build_cache=False))
        nll_ones = float(model_ones._nll(theta, build_cache=False))
        np.testing.assert_allclose(nll_ones, nll_default, rtol=1e-12, atol=1e-12)


class TestNLLEquivalenceUnderGrouping:
    """When all per-output entries inside each group are equal, the grouped
    parameterization must produce the same NLL as the per-output one."""

    def test_grouped_vs_per_output_nll(self):
        X, Y = _make_dataset(n=12, d=2, p=6, seed=11)
        q = 2

        es = (2, 1, 3)
        sigma_groups = np.array([0.05, 0.10, 0.02])
        sigma_per_output = np.array([0.05, 0.05, 0.10, 0.02, 0.02, 0.02])

        # Grouped model takes one log-variance per group.
        model_grp = _make_model(X, Y, q, diag_error_structure=es)
        theta_grp = _build_theta(q, X.shape[1], sigma_groups, seed=11)
        nll_grp = float(model_grp._nll(theta_grp, build_cache=False))

        # Equivalent per-output model.
        model_full = _make_model(X, Y, q, diag_error_structure=None)
        theta_full = _build_theta(q, X.shape[1], sigma_per_output, seed=11)
        nll_full = float(model_full._nll(theta_full, build_cache=False))

        np.testing.assert_allclose(nll_grp, nll_full, rtol=1e-12, atol=1e-12)


class TestAnalyticalGradient:
    """The analytical fast-path gradient must agree with autograd under
    arbitrary ``diag_error_structure``."""

    @pytest.mark.parametrize("es", [(2, 1, 3), (3, 3), (1, 5), (6,)])
    def test_grad_matches_autograd(self, es):
        p = sum(es)
        X, Y = _make_dataset(n=10, d=2, p=p, seed=7)
        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)

        sigma_groups = 0.01 + 0.005 * np.arange(len(es))
        theta = _build_theta(q, X.shape[1], sigma_groups, seed=7)

        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        nll_auto, grad_auto = vg(theta)
        nll_ana, grad_ana = model._nll_and_grad_fast(theta)

        np.testing.assert_allclose(nll_ana, float(nll_auto), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(grad_ana, np.asarray(grad_auto, float),
                                   rtol=1e-8, atol=1e-9)

    def test_grad_matches_finite_differences(self):
        es = (2, 3)
        p = sum(es)
        X, Y = _make_dataset(n=10, d=2, p=p, seed=2)
        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)
        theta = _build_theta(q, X.shape[1], np.array([0.02, 0.05]), seed=2)

        _, grad_ana = model._nll_and_grad_fast(theta)

        eps = 1e-6
        grad_fd = np.zeros_like(theta)
        for i in range(theta.size):
            plus = theta.copy()
            minus = theta.copy()
            plus[i] += eps
            minus[i] -= eps
            grad_fd[i] = (
                float(model._nll(plus, build_cache=False))
                - float(model._nll(minus, build_cache=False))
            ) / (2 * eps)

        np.testing.assert_allclose(grad_ana, grad_fd, rtol=5e-5, atol=5e-6)


# ---------------------------------------------------------------------------
# Validation — bad inputs caught at prepare-time
# ---------------------------------------------------------------------------


class TestValidation:
    def test_mismatched_sum_raises_at_prepare(self):
        X, Y = _make_dataset(n=8, d=2, p=4, seed=0)
        model = MOOGP(
            terms=[None, 1, 2],
            q=2,
            learn_sigma_eps=True,
            diag_error_structure=[2, 1],  # sums to 3, but p=4
            jitter=0.0,
        )
        with pytest.raises(ValueError, match="sum"):
            model._prepare_data({"X_scaled": X, "y": Y})


# ---------------------------------------------------------------------------
# End-to-end fit
# ---------------------------------------------------------------------------


class TestEndToEndFit:
    """Fitting should improve the NLL and recover sensible per-group variances
    when the underlying noise actually shares values across each group."""

    def test_fit_ties_within_group_and_separates_across_groups(self):
        # 4 outputs, 2 groups of 2.  Group 0 has small noise (signal-dominated),
        # group 1 has much larger noise (noise-dominated).  The grouped fit
        # must (a) tie outputs inside each group exactly, and (b) recover a
        # noticeably larger variance for group 1 than for group 0.
        n, d, p = 60, 1, 4
        es = (2, 2)
        small, large = 1e-3, 0.5
        true_per_output = expand_grouped_sigma_eps2(np.array([small, large]), es)

        rng = np.random.default_rng(2026)
        X_phys = rng.uniform(0.0, 1.0, size=(n, 1))
        X = 2.0 * X_phys - 1.0
        # Smooth low-magnitude signals so the noise-to-signal ratios in the
        # two groups are clearly different.
        s1 = np.sin(3.0 * X_phys.ravel())
        s2 = np.cos(3.0 * X_phys.ravel())
        Y_clean = np.column_stack([s1, 0.7 * s1, s2, 0.7 * s2])
        Y = Y_clean + rng.normal(0.0, np.sqrt(true_per_output), size=Y_clean.shape)

        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)

        sigma0 = np.array([0.05, 0.05])
        theta0 = _build_theta(q, d, sigma0, seed=2026)

        lat_bounds = []
        for _ in range(q):
            lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
            lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
        sigma_bounds = [(float(np.log(1e-8)), float(np.log(10.0)))] * len(es)
        bounds = lat_bounds + sigma_bounds

        nll0 = float(model._nll(theta0, build_cache=False))
        model.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0,
            bounds=bounds,
            optimizer_opts={"maxiter": 200},
        )
        assert model.nll_hat <= nll0 + 1e-6

        sigma_hat = model.cache["sigma_eps2"]
        # Outputs inside the same group must share the same fitted variance.
        np.testing.assert_allclose(sigma_hat[0], sigma_hat[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(sigma_hat[2], sigma_hat[3], rtol=1e-10, atol=1e-12)
        # The optimizer should distinguish the two noise levels by at least 5x.
        assert sigma_hat[2] > 5.0 * sigma_hat[0], (sigma_hat[0], sigma_hat[2])

    def test_fewer_parameters_than_per_output_fit(self):
        # The grouped parameterization must expose len(es) free sigma entries,
        # not p.  Verify by inspecting the fitted theta length.
        es = (2, 3)
        p = sum(es)
        X, Y = _make_dataset(n=20, d=2, p=p, seed=33)
        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)

        theta0 = _build_theta(q, X.shape[1], np.array([0.05, 0.05]), seed=33)
        lat_bounds = []
        for _ in range(q):
            lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
            lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * X.shape[1])
        bounds = lat_bounds + [(float(np.log(1e-6)), float(np.log(1.0)))] * len(es)

        model.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0,
            bounds=bounds,
            optimizer_opts={"maxiter": 50},
        )

        assert model.theta_hat.size == q * (X.shape[1] + 1) + len(es)
        # ungrouped baseline: should have p sigma entries
        model_full = _make_model(X, Y, q, diag_error_structure=None)
        theta0_full = _build_theta(q, X.shape[1], np.full(p, 0.05), seed=33)
        bounds_full = lat_bounds + [(float(np.log(1e-6)), float(np.log(1.0)))] * p
        model_full.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0_full,
            bounds=bounds_full,
            optimizer_opts={"maxiter": 50},
        )
        assert model_full.theta_hat.size == q * (X.shape[1] + 1) + p
        assert model.theta_hat.size < model_full.theta_hat.size

    def test_predict_runs_and_shapes_match(self):
        es = (1, 2)
        p = sum(es)
        X, Y = _make_dataset(n=25, d=2, p=p, seed=14)
        q = 2
        model = _make_model(X, Y, q, diag_error_structure=es)

        theta0 = _build_theta(q, X.shape[1], np.array([0.05, 0.05]), seed=14)
        lat_bounds = []
        for _ in range(q):
            lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
            lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * X.shape[1])
        bounds = lat_bounds + [(float(np.log(1e-6)), float(np.log(1.0)))] * len(es)

        model.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0,
            bounds=bounds,
            optimizer_opts={"maxiter": 100},
        )

        Xs = np.linspace(-1, 1, 7).reshape(-1, 1).repeat(X.shape[1], axis=1)
        mean, std = model.predict(Xs, return_std=True)
        assert mean.shape == (Xs.shape[0], p)
        assert std.shape == (Xs.shape[0], p)
        assert np.all(std >= 0)
