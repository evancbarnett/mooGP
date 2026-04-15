"""Tests for the analytical fast-path NLL gradient (Fix 4).

These tests ensure that the hand-coded ``MOOGP._nll_and_grad_fast`` is
bit-identical (to floating-point roundoff) with both:

  1. The NLL value produced by the existing ``_nll`` method.
  2. The autograd gradient of ``_nll`` wrapped in ``value_and_grad``.

They also verify a convergence speedup against the autograd path at a
moderate problem size.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from autograd import value_and_grad

from moogp.model import MOOGP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n=12, d=3, p=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(X[:, j0]) + 0.3 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _build_theta(q, d, sigma_eps2, seed=0):
    rng = np.random.default_rng(seed + 1000)
    theta = []
    for _ in range(q):
        theta.append(float(np.log(rng.uniform(0.5, 1.5))))
        theta.extend([float(np.log(rng.uniform(0.4, 0.9))) for _ in range(d)])
    theta.extend(list(np.log(sigma_eps2)))
    return np.asarray(theta, dtype=float)


def _make_model(X, Y, q, *, learn_sigma_eps=True, orthogonal=True, use_analytical_grad=True):
    d = X.shape[1]
    model = MOOGP(
        terms=[None] + list(range(1, d + 1)),
        q=q,
        Psi=None,
        orthogonal=orthogonal,
        learn_Psi=False,
        learn_sigma_eps=learn_sigma_eps,
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
# Exactness — NLL value
# ---------------------------------------------------------------------------


class TestAnalyticalNLLValue:
    @pytest.mark.parametrize("seed,q", [(0, 2), (1, 3), (7, 2)])
    def test_matches_existing_nll(self, seed, q):
        X, Y = _make_dataset(n=14, d=3, p=3, seed=seed)
        model = _make_model(X, Y, q)
        theta = _build_theta(q, X.shape[1], np.array([1e-2, 2e-2, 1.5e-2]), seed=seed)

        nll_ref = float(model._nll(theta, build_cache=False))
        nll_new, _ = model._nll_and_grad_fast(theta)

        np.testing.assert_allclose(nll_new, nll_ref, rtol=1e-12, atol=1e-12)

    def test_learn_sigma_eps_false(self):
        X, Y = _make_dataset(n=10, d=2, p=2, seed=2)
        # When learn_sigma_eps=False we must supply a fixed sigma_eps2.
        model = MOOGP(
            terms=[None, 1, 2],
            q=2,
            Psi=None,
            orthogonal=True,
            learn_Psi=False,
            sigma_eps2=np.array([5e-3, 5e-3]),
            learn_sigma_eps=False,
            jitter=0.0,
            one_based=True,
            normalize_cols=True,
            use_diagonalized_interaction=True,
            use_slow_kyinv=False,
            standardize_y=False,
            use_analytical_grad=True,
        )
        model._prepare_data({"X_scaled": X, "y": Y})

        # theta has only latent kernel params in this mode.
        theta = _build_theta(q=2, d=X.shape[1], sigma_eps2=np.array([]), seed=3)
        nll_ref = float(model._nll(theta, build_cache=False))
        nll_new, grad_new = model._nll_and_grad_fast(theta)

        np.testing.assert_allclose(nll_new, nll_ref, rtol=1e-12, atol=1e-12)
        assert grad_new.shape == theta.shape


# ---------------------------------------------------------------------------
# Exactness — gradient vs autograd
# ---------------------------------------------------------------------------


class TestAnalyticalGradientVsAutograd:
    @pytest.mark.parametrize("seed,q,p,d", [
        (0, 2, 3, 3),
        (1, 3, 4, 2),
        (5, 2, 2, 4),
        (11, 3, 3, 3),
    ])
    def test_gradient_matches_autograd(self, seed, q, p, d):
        X, Y = _make_dataset(n=10, d=d, p=p, seed=seed)
        model = _make_model(X, Y, q)
        sigma0 = np.full(p, 1e-2) + 1e-3 * np.arange(p)
        theta = _build_theta(q, d, sigma0, seed=seed)

        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        nll_auto, grad_auto = vg(theta)
        grad_auto = np.asarray(grad_auto, dtype=float)

        nll_analytic, grad_analytic = model._nll_and_grad_fast(theta)

        np.testing.assert_allclose(nll_analytic, float(nll_auto), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(grad_analytic, grad_auto, rtol=1e-8, atol=1e-9)

    def test_gradient_matches_autograd_non_orthogonal(self):
        X, Y = _make_dataset(n=10, d=3, p=3, seed=4)
        model = _make_model(X, Y, q=2, orthogonal=False)
        theta = _build_theta(q=2, d=3, sigma_eps2=np.array([1e-2, 1e-2, 1e-2]), seed=4)

        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        _, grad_auto = vg(theta)
        _, grad_analytic = model._nll_and_grad_fast(theta)

        np.testing.assert_allclose(grad_analytic, np.asarray(grad_auto, float), rtol=1e-8, atol=1e-9)

    def test_gradient_matches_finite_differences(self):
        X, Y = _make_dataset(n=8, d=2, p=2, seed=1)
        model = _make_model(X, Y, q=2)
        theta = _build_theta(q=2, d=2, sigma_eps2=np.array([5e-3, 5e-3]), seed=1)

        _, grad_analytic = model._nll_and_grad_fast(theta)

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

        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=5e-5, atol=5e-6)


# ---------------------------------------------------------------------------
# Exactness — fit() produces the same optimum via both paths
# ---------------------------------------------------------------------------


class TestFitEquivalence:
    def test_fit_matches_autograd_path(self):
        X, Y = _make_dataset(n=20, d=3, p=3, seed=9)
        q, d, p = 3, X.shape[1], Y.shape[1]
        sigma0 = np.full(p, 5e-3)
        theta0 = _build_theta(q, d, sigma0, seed=9)

        lat_bounds = []
        for _ in range(q):
            lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
            lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
        sigma_bounds = [(float(np.log(1e-6)), float(np.log(1.0)))] * p
        bounds = lat_bounds + sigma_bounds

        model_auto = _make_model(X, Y, q, use_analytical_grad=False)
        model_auto.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0.copy(),
            bounds=bounds,
            optimizer_opts={"maxiter": 200},
        )
        model_ana = _make_model(X, Y, q, use_analytical_grad=True)
        model_ana.fit(
            data={"X_scaled": X, "y": Y},
            theta0=theta0.copy(),
            bounds=bounds,
            optimizer_opts={"maxiter": 200},
        )

        # Fitted NLLs should agree tightly — the optimizer path may differ
        # slightly, but both are solving the same problem with the same exact
        # gradient (the analytical gradient is bit-equivalent to autograd's).
        np.testing.assert_allclose(model_ana.nll_hat, model_auto.nll_hat, rtol=1e-6, atol=1e-6)

    def test_predict_equivalent_after_fit(self):
        X, Y = _make_dataset(n=15, d=2, p=2, seed=21)
        q, d, p = 2, X.shape[1], Y.shape[1]
        sigma0 = np.full(p, 5e-3)
        theta0 = _build_theta(q, d, sigma0, seed=21)

        lat_bounds = []
        for _ in range(q):
            lat_bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
            lat_bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
        sigma_bounds = [(float(np.log(1e-6)), float(np.log(1.0)))] * p
        bounds = lat_bounds + sigma_bounds

        model_auto = _make_model(X, Y, q, use_analytical_grad=False)
        model_auto.fit({"X_scaled": X, "y": Y}, theta0=theta0.copy(), bounds=bounds,
                       optimizer_opts={"maxiter": 200})
        model_ana = _make_model(X, Y, q, use_analytical_grad=True)
        model_ana.fit({"X_scaled": X, "y": Y}, theta0=theta0.copy(), bounds=bounds,
                      optimizer_opts={"maxiter": 200})

        rng = np.random.default_rng(99)
        Xs = rng.uniform(-1, 1, size=(5, d))
        mean_auto, std_auto = model_auto.predict(Xs, return_std=True)
        mean_ana, std_ana = model_ana.predict(Xs, return_std=True)

        np.testing.assert_allclose(mean_ana, mean_auto, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(std_ana, std_auto, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Speedup
# ---------------------------------------------------------------------------


class TestAnalyticalSpeedup:
    def test_per_call_speedup(self):
        """The analytical grad should be meaningfully faster per (value, grad)
        call than autograd's ``value_and_grad(_nll)`` at a moderate size.
        """
        X, Y = _make_dataset(n=400, d=4, p=4, seed=33)
        q, d, p = 3, X.shape[1], Y.shape[1]
        model = _make_model(X, Y, q)

        theta = _build_theta(q, d, np.full(p, 1e-2), seed=33)

        # Warm-up: ensure caches/JIT paths are hit once.
        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        vg(theta)
        model._nll_and_grad_fast(theta)

        def _time(fn, repeats=6):
            # Take the best of `repeats` runs so that system jitter can't
            # drag the timing down. Speedup should be measured against the
            # representative steady-state cost, not the worst outlier.
            samples = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn(theta)
                samples.append(time.perf_counter() - t0)
            return min(samples)

        t_auto = _time(lambda th: vg(th))
        t_ana = _time(lambda th: model._nll_and_grad_fast(th))

        speedup = t_auto / t_ana
        # At this size the Cholesky backward dominates autograd's cost and
        # the analytical path consistently delivers ≥1.8× on a laptop. The
        # structural win is ~2× because we skip the autograd Cholesky
        # backward entirely; the remaining cost is the kernel rebuild inside
        # per-latent `tr(M_k C_k)` autograd calls.
        assert t_ana < t_auto, (
            f"analytical ({t_ana:.4f}s) not faster than autograd ({t_auto:.4f}s)"
        )
        assert speedup >= 1.8, (
            f"speedup {speedup:.2f}x below 1.8x threshold "
            f"(auto={t_auto:.4f}s, ana={t_ana:.4f}s)"
        )
