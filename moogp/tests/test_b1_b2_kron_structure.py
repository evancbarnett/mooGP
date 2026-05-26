"""Exactness tests for the B1 / B2 Kronecker-structure optimizations.

B1 — ``_profiled_gls_terms_fast``: reuse of ``Q_k`` applied only to the design
matrix ``G`` (n × r) instead of solving on the full ``G_y = I_p ⊗ G`` (n p × r p).

B2 — ``_predict_variance_diag_fast``: closed-form diagonal of
``K_*X K_y^{-1} K_X*`` via the Woodbury identity
``K_y^{-1} K_X* = Σ_j (ψ̃_j ψ_j^T) ⊗ A_j^{-1} C_j(X, X_*)``, avoiding the
dense (n p × n_* p) intermediate.

Each test independently constructs a *reference* answer using the slow,
unstructured path and compares to the optimized path produced by the model.
"""

from __future__ import annotations

import numpy as np
import pytest
from autograd import value_and_grad
from scipy.linalg import cho_factor, cho_solve

from moogp.design import build_Gy, make_G, vecF, unvecF
from moogp.kernels import make_c_star_diag, make_c_star_matrix
from moogp.model import (
    MOOGP,
    _profiled_gls_terms,
    _profiled_gls_terms_fast,
    _predict_variance_diag_fast,
    build_Ky,
    build_cross_K,
    init_phi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n=14, d=3, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(1.7 * X[:, j0]) + 0.5 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _theta_for(q, d, sigma_eps2_init, seed=0):
    rng = np.random.default_rng(seed + 100)
    theta = []
    for _ in range(q):
        theta.append(float(np.log(rng.uniform(0.6, 1.4))))
        theta.extend([float(np.log(rng.uniform(0.4, 1.2))) for _ in range(d)])
    theta.extend(list(np.log(sigma_eps2_init)))
    return np.asarray(theta, dtype=float)


def _make_fitted_model(X, Y, q, *, orthogonal=True, learn_sigma_eps=True):
    d = X.shape[1]
    p = Y.shape[1]
    sigma_eps2_init = 1e-2 * np.ones(p)
    theta0 = _theta_for(q, d, sigma_eps2_init, seed=1)
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
        use_analytical_grad=True,
        standardize_x=False,
    )
    # Use a single _nll call rather than full optimisation to keep tests fast
    # and deterministic; this still populates the cache and exercises the fast
    # path identically to the post-fit state.
    model._prepare_data({"X_scaled": X, "y": Y})
    model._nll(theta0, build_cache=True)
    model.theta_hat = theta0
    model.fitted = True
    return model, theta0


def _build_reference_dense_Ky(X, Y, theta_raw, q, d, p, terms, orthogonal=True):
    """Build the dense Ky (no jitter) using the manuscript fast Psi parameterization."""
    n = X.shape[0]
    Phi, _ = init_phi(Y, q, n)
    sigma_eps2 = np.maximum(np.exp(theta_raw[-p:]), 1e-10)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi  # (p, q)

    Cjs = []
    for k in range(q):
        s2 = float(np.exp(theta_raw[k * (d + 1)]))
        ell = np.exp(theta_raw[k * (d + 1) + 1 : (k + 1) * (d + 1)])
        Ck = make_c_star_matrix(X, X, ell=ell, sigma2=s2, terms=terms,
                                orthogonal=orthogonal, one_based=True)
        Cjs.append(np.asarray(Ck, dtype=float))

    Ky = build_Ky(Cjs, Psi, sigma_eps2=sigma_eps2)
    return Ky, Psi, sigma_eps2, Cjs


# ---------------------------------------------------------------------------
# B1 — profiled GLS terms via Kronecker structure
# ---------------------------------------------------------------------------


class TestProfiledGLSFastEqualsReference:
    """The new fast path should match the original ``_profiled_gls_terms``
    closure-style call exactly (up to floating-point roundoff)."""

    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (1, 3, 4), (5, 1, 1), (7, 3, 6)])
    def test_qf_bhat_rvec_kyinvr_match(self, seed, q, p):
        X, Y = _make_dataset(n=12, d=3, p=p, seed=seed)
        model, theta0 = _make_fitted_model(X, Y, q=q)
        cache = model.cache
        fast_info = cache["fast_diag_info"]
        assert fast_info is not None, "Fast path expected for this configuration."

        G = cache["G"]
        Gy = cache["Gy"]
        vecY = vecF(model.Y)
        n = X.shape[0]

        # --- Reference (un-optimised) path ---
        solve_Ky = lambda rhs: model._apply_Ky_inv_fast(rhs, fast_info)
        qf_ref, bhat_ref, rvec_ref, kir_ref = _profiled_gls_terms(
            solve_Ky, G, Gy, vecY, p, build_cache=True
        )

        # --- Optimised structure-exploiting path ---
        alpha_vec = solve_Ky(vecY)
        alpha_mat = unvecF(alpha_vec, n, p)
        qf_fast, bhat_fast, rvec_fast, kir_fast = _profiled_gls_terms_fast(
            fast_info, G, vecY, alpha_mat, p, build_cache=True,
        )

        np.testing.assert_allclose(qf_fast, qf_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(bhat_fast, bhat_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(rvec_fast, rvec_ref, rtol=0, atol=1e-9)
        np.testing.assert_allclose(kir_fast, kir_ref, rtol=0, atol=1e-9)

    def test_no_cache_path_returns_qf_only(self):
        X, Y = _make_dataset(n=10, d=2, p=3, seed=2)
        model, _ = _make_fitted_model(X, Y, q=2)
        fast_info = model.cache["fast_diag_info"]
        G = model.cache["G"]
        vecY = vecF(model.Y)
        n = X.shape[0]
        p = Y.shape[1]

        alpha_vec = model._apply_Ky_inv_fast(vecY, fast_info)
        alpha_mat = unvecF(alpha_vec, n, p)
        qf, b, r, kir = _profiled_gls_terms_fast(
            fast_info, G, vecY, alpha_mat, p, build_cache=False,
        )
        assert b is None and r is None and kir is None

        # And the qf value must agree with the build_cache=True branch.
        qf2, *_ = _profiled_gls_terms_fast(
            fast_info, G, vecY, alpha_mat, p, build_cache=True,
        )
        np.testing.assert_allclose(qf, qf2, rtol=0, atol=1e-12)


class TestNLLAfterFix:
    """End-to-end exactness: NLL via the fast Kronecker path matches NLL via
    a totally independent dense-Ky reference (slow-path style construction)."""

    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (3, 3, 5), (4, 1, 1)])
    def test_nll_value_matches_dense_reference(self, seed, q, p):
        X, Y = _make_dataset(n=10, d=3, p=p, seed=seed)
        d = X.shape[1]
        terms = [None] + list(range(1, d + 1))
        sigma_eps2_init = 5e-3 * np.ones(p)
        theta = _theta_for(q, d, sigma_eps2_init, seed=seed)

        model = MOOGP(
            terms=terms, q=q, Psi=None, orthogonal=True,
            learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
            use_diagonalized_interaction=True, use_slow_kyinv=False,
            standardize_y=False,
            standardize_x=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})
        nll_fast = float(model._nll(theta, build_cache=False))

        # Independent reference: build Ky densely and do the GLS profile by hand.
        Ky, _Psi, _sigma_eps2, _Cjs = _build_reference_dense_Ky(
            X, Y, theta, q, d, p, terms, orthogonal=True
        )
        n = X.shape[0]
        G = make_G({"X_scaled": X}, terms, one_based=True, return_names=False)
        Gy = build_Gy(G, p)
        vecY = vecF(Y)

        L_full, lo = cho_factor(Ky, lower=True, check_finite=False)
        z = cho_solve((L_full, lo), Gy, check_finite=False)
        alpha = cho_solve((L_full, lo), vecY, check_finite=False)
        A_gls = Gy.T @ z
        b_gls = Gy.T @ alpha
        beta = np.linalg.solve(A_gls, b_gls)
        qf = float(vecY @ alpha - b_gls @ beta)
        logdetK = 2.0 * float(np.sum(np.log(np.diag(L_full))))
        # ``MOOGP._nll`` returns the per-row negative log-likelihood
        # (``NLL / n``); match that here so the dense reference is comparable.
        nll_ref = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi)) / float(n)

        np.testing.assert_allclose(nll_fast, nll_ref, rtol=1e-10, atol=1e-10)

    def test_autograd_through_fast_path_still_matches_finite_difference(self):
        X, Y = _make_dataset(n=10, d=2, p=3, seed=11)
        d = X.shape[1]
        p = Y.shape[1]
        q = 2
        sigma_eps2_init = 5e-3 * np.ones(p)
        theta = _theta_for(q, d, sigma_eps2_init, seed=11)

        model = MOOGP(
            terms=[None, 1, 2], q=q, Psi=None, orthogonal=True,
            learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
            use_diagonalized_interaction=True, use_slow_kyinv=False,
            standardize_y=False, use_analytical_grad=False,
            standardize_x=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})

        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        _, g_auto = vg(theta)

        eps = 1e-6
        g_num = np.zeros_like(theta)
        for i in range(theta.size):
            plus = theta.copy()
            minus = theta.copy()
            plus[i] += eps
            minus[i] -= eps
            g_num[i] = (
                float(model._nll(plus, build_cache=False))
                - float(model._nll(minus, build_cache=False))
            ) / (2 * eps)

        np.testing.assert_allclose(g_auto, g_num, rtol=1e-4, atol=1e-5)

    def test_analytical_grad_still_matches_autograd(self):
        """The hand-coded `_nll_and_grad_fast` must remain consistent with the
        autograd value/gradient of `_nll` after the B1 rewrite."""
        X, Y = _make_dataset(n=12, d=3, p=4, seed=5)
        d = X.shape[1]
        p = Y.shape[1]
        q = 3
        sigma_eps2_init = 5e-3 * np.ones(p)
        theta = _theta_for(q, d, sigma_eps2_init, seed=5)

        model = MOOGP(
            terms=[None] + list(range(1, d + 1)), q=q, Psi=None,
            orthogonal=True, learn_Psi=False, learn_sigma_eps=True,
            jitter=0.0, use_diagonalized_interaction=True,
            use_slow_kyinv=False, standardize_y=False,
            use_analytical_grad=True,
            standardize_x=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})

        nll_anal, g_anal = model._nll_and_grad_fast(theta)
        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        nll_auto, g_auto = vg(theta)

        np.testing.assert_allclose(nll_anal, float(nll_auto), rtol=0, atol=1e-9)
        np.testing.assert_allclose(g_anal, np.asarray(g_auto), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# B2 — Predict variance diagonal via Woodbury structure
# ---------------------------------------------------------------------------


class TestPredictVarianceDiagFastEqualsDense:
    """The closed-form variance diagonal must equal the brute-force computation
    that builds K_XsX explicitly and solves K_y on it (the original path)."""

    @pytest.mark.parametrize("seed,q,p,n_star", [(0, 2, 3, 9), (3, 3, 5, 17), (7, 1, 1, 5)])
    def test_diag_matches_dense(self, seed, q, p, n_star):
        X, Y = _make_dataset(n=14, d=3, p=p, seed=seed)
        model, _ = _make_fitted_model(X, Y, q=q)
        cache = model.cache
        fast_info = cache["fast_diag_info"]
        assert fast_info is not None

        rng = np.random.default_rng(seed + 42)
        Xs = rng.uniform(-1, 1, size=(n_star, X.shape[1]))

        Psi = cache["Psi"]
        lat_params = cache["lat_params"]
        terms = cache["terms"]
        sigma_eps2 = cache["sigma_eps2"]

        # Cross kernels and test-test diagonals (computed once and reused by both paths)
        Cj_XsX = [
            make_c_star_matrix(Xs, X, ell=ell_j, sigma2=sigma2_j,
                               terms=terms, orthogonal=model.orthogonal,
                               one_based=cache["one_based"])
            for (sigma2_j, ell_j) in lat_params
        ]
        Cj_diag_star = [
            make_c_star_diag(Xs, ell=ell_j, sigma2=sigma2_j, terms=terms,
                             orthogonal=model.orthogonal, one_based=cache["one_based"])
            for (sigma2_j, ell_j) in lat_params
        ]

        # --- Reference (dense) ---
        K_XsX = build_cross_K(Psi, Cj_XsX)
        diag_prior = np.zeros(n_star * p)
        for j in range(q):
            diag_prior += np.kron(Psi[:, j] ** 2, Cj_diag_star[j])
        V = model._solve_with_cached_Ky(K_XsX.T)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag_ref_signal = diag_prior - diag_cross
        diag_ref_obs = diag_ref_signal + np.repeat(sigma_eps2, n_star)

        # --- Optimised path ---
        diag_fast_signal = _predict_variance_diag_fast(
            fast_info, Cj_XsX, Psi,
            Cj_diag_star_list=Cj_diag_star,
            predict_observation=False,
        )
        diag_fast_obs = _predict_variance_diag_fast(
            fast_info, Cj_XsX, Psi,
            Cj_diag_star_list=Cj_diag_star,
            predict_observation=True,
        )

        np.testing.assert_allclose(diag_fast_signal, diag_ref_signal, rtol=0, atol=1e-9)
        np.testing.assert_allclose(diag_fast_obs, diag_ref_obs, rtol=0, atol=1e-9)


class TestPredictEndToEndUnchanged:
    """``model.predict`` must return the same mean and std after the rewrite —
    we compare the fast path to the slow path (`use_slow_kyinv=True`) for ground
    truth."""

    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (1, 3, 5), (4, 1, 1)])
    def test_mean_and_std_match_slow_path(self, seed, q, p):
        X, Y = _make_dataset(n=14, d=3, p=p, seed=seed)
        d = X.shape[1]
        sigma_eps2_init = 5e-3 * np.ones(p)
        theta = _theta_for(q, d, sigma_eps2_init, seed=seed)
        terms = [None] + list(range(1, d + 1))

        # Fast model: bypass optimization by manually evaluating at theta.
        m_fast = MOOGP(
            terms=terms, q=q, Psi=None, orthogonal=True,
            learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
            use_diagonalized_interaction=True, use_slow_kyinv=False,
            standardize_y=False, use_analytical_grad=True,
            standardize_x=False,
        )
        m_fast._prepare_data({"X_scaled": X, "y": Y})
        m_fast._nll(theta, build_cache=True)
        m_fast.theta_hat = theta
        m_fast.fitted = True

        # Slow model: same theta, but force the dense Ky path (Psi must be
        # provided because the fast SVD-of-Y construction is what use_slow_kyinv
        # bypasses).
        Phi, _ = init_phi(m_fast.Y, q, X.shape[0])
        sigma_eps2 = np.maximum(np.exp(theta[-p:]), 1e-10)
        Psi_fixed = np.diag(np.sqrt(sigma_eps2)) @ Phi

        m_slow = MOOGP(
            terms=terms, q=q, Psi=Psi_fixed, orthogonal=True,
            learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
            use_diagonalized_interaction=False, use_slow_kyinv=True,
            standardize_y=False, use_analytical_grad=False,
            standardize_x=False,
        )
        m_slow._prepare_data({"X_scaled": X, "y": Y})
        m_slow._nll(theta, build_cache=True)
        m_slow.theta_hat = theta
        m_slow.fitted = True

        rng = np.random.default_rng(seed + 1234)
        Xs = rng.uniform(-1, 1, size=(11, d))

        mean_fast, std_fast = m_fast.predict(Xs, return_std=True)
        mean_slow, std_slow = m_slow.predict(Xs, return_std=True)

        np.testing.assert_allclose(mean_fast, mean_slow, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(std_fast, std_slow, rtol=1e-7, atol=1e-9)

    def test_predict_observation_toggle_consistent(self):
        """``predict_observation=True`` should add exactly the per-output noise
        (in raw scale) to the variance vs ``predict_observation=False``."""
        X, Y = _make_dataset(n=12, d=2, p=4, seed=8)
        m, _ = _make_fitted_model(X, Y, q=2)
        rng = np.random.default_rng(0)
        Xs = rng.uniform(-1, 1, size=(7, 2))

        _, std_signal = m.predict(Xs, return_std=True, predict_observation=False)
        _, std_obs = m.predict(Xs, return_std=True, predict_observation=True)

        var_signal = std_signal ** 2
        var_obs = std_obs ** 2
        diff = var_obs - var_signal
        # Working-scale Σ_eps in cache; predict applies y_scale^2 to take it back
        # to raw scale. With standardize_y=False this is identity.
        sigma_raw = np.asarray(m.cache["sigma_eps2_raw"], dtype=float).ravel()
        expected = np.broadcast_to(sigma_raw[None, :], diff.shape)
        np.testing.assert_allclose(diff, expected, rtol=1e-6, atol=1e-9)
