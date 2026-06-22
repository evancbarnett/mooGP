"""Exactness tests for MOOGP's fast paths.

Each fast path has a slower reference implementation that it must reproduce
to floating-point tolerance. This file consolidates those equivalence checks:

    Kernel-level (kernels.py / design.py):
        * parse_terms_to_index_sets LRU cache
        * se_kernel_matrix with precomputed sqdist
        * h_matrix_se vectorized vs per-column reference
        * make_c_star_matrix orthogonal/non-orthogonal + X-is-Xp shortcut

    Model-level (model.py):
        * _profiled_gls_terms_fast       (B1) vs _profiled_gls_terms
        * _predict_variance_diag_fast    (B2) vs dense Woodbury
        * _latent_kernel_logtheta_grad   (B3) vs autograd of the trace functional
        * _nll_and_grad_fast             (analytical) vs value_and_grad(_nll)
        * MOOGP._nll                      vs an independent dense-Ky reference
        * MOOGP.predict fast path        vs use_slow_kyinv=True
        * MOOGP.fit                       analytical_grad on/off equivalence
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad as ag_grad, value_and_grad
from scipy.linalg import cho_factor, cho_solve

from moogp.design import (
    _parse_terms_cached,
    _parse_terms_impl,
    build_Gy,
    make_G,
    parse_terms_to_index_sets,
    unvecF,
    vecF,
)
from moogp.kernels import (
    H_diag_se,
    L_gauss,
    M_gauss,
    h_matrix_se,
    make_c_star_diag,
    make_c_star_matrix,
    se_kernel_matrix,
)
from moogp.model import (
    MOOGP,
    _latent_kernel_logtheta_grad,
    _predict_variance_diag_fast,
    _profiled_gls_terms,
    _profiled_gls_terms_fast,
    build_Ky,
    build_cross_K,
    init_phi,
)


# ---------------------------------------------------------------------------
# Shared helpers
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


def _make_model(X, Y, q, *, terms=None, orthogonal=True, learn_sigma_eps=True,
                use_analytical_grad=True, prepare=True):
    d = X.shape[1]
    if terms is None:
        terms = [None] + list(range(1, d + 1))
    model = MOOGP(
        terms=terms,
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
        standardize_x=False,
    )
    if prepare:
        model._prepare_data({"X_scaled": X, "y": Y})
    return model


def _bounds_for(q, d, p):
    lat = []
    for _ in range(q):
        lat.append((float(np.log(1e-3)), float(np.log(1e3))))
        lat.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)
    sigma = [(float(np.log(1e-6)), float(np.log(1.0)))] * p
    return lat + sigma


def _ref_se_kernel(X, Xp, ell, sigma2=1.0):
    diff = (X[:, None, :] - Xp[None, :, :]) / ell
    return sigma2 * np.exp(-np.sum(diff * diff, axis=2))


def _ref_h_matrix(X, ell, sigma2, terms, one_based=True):
    n, d = X.shape
    J_sets = _parse_terms_impl(list(terms), d, one_based, False)
    if len(J_sets) == 0:
        return np.empty((n, 0), float)
    M_all = np.column_stack([M_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])
    L_all = np.column_stack([L_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])
    all_idx = set(range(d))
    cols = []
    for Ji in J_sets:
        Ji_s = set(Ji)
        notJ = list(all_idx - Ji_s)
        col = np.ones(n)
        if notJ:
            col = col * np.prod(M_all[:, notJ], axis=1)
        if Ji_s:
            col = col * np.prod(L_all[:, list(Ji_s)], axis=1)
        cols.append(col)
    return np.column_stack(cols) * sigma2


def _dense_Ky_from_theta(X, Y, theta_raw, q, d, p, terms, orthogonal=True):
    """Independent dense reference Ky built without touching MOOGP fast paths."""
    n = X.shape[0]
    Phi, _ = init_phi(Y, q, n)
    sigma_eps2 = np.maximum(np.exp(theta_raw[-p:]), 1e-10)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi
    Cjs = []
    for k in range(q):
        s2 = float(np.exp(theta_raw[k * (d + 1)]))
        ell = np.exp(theta_raw[k * (d + 1) + 1 : (k + 1) * (d + 1)])
        Ck = make_c_star_matrix(X, X, ell=ell, sigma2=s2, terms=terms,
                                orthogonal=orthogonal, one_based=True)
        Cjs.append(np.asarray(Ck, dtype=float))
    Ky = build_Ky(Cjs, Psi, sigma_eps2=sigma_eps2)
    return Ky, Psi, sigma_eps2, Cjs


# ===========================================================================
# Kernel-level fast paths
# ===========================================================================


class TestParseTermsCaching:
    def test_values_match_uncached_reference(self):
        terms = [None, 1, 2, 3, (1, 2), (2, 3)]
        cached = parse_terms_to_index_sets(terms, 3, one_based=True)
        reference = _parse_terms_impl(terms, 3, True, False)
        assert list(cached) == list(reference)

    def test_cache_hit_returns_same_object(self):
        terms = [None, 1, 2]
        _parse_terms_cached.cache_clear()
        a = parse_terms_to_index_sets(terms, 2, one_based=True)
        b = parse_terms_to_index_sets(terms, 2, one_based=True)
        assert a is b
        assert _parse_terms_cached.cache_info().hits >= 1

    def test_cached_result_is_immutable_tuple(self):
        result = parse_terms_to_index_sets([None, 1, 2], 2, one_based=True)
        assert isinstance(result, tuple)

    def test_distinct_keys_do_not_collide(self):
        a = parse_terms_to_index_sets([None, 1], 2, one_based=True)
        b = parse_terms_to_index_sets([None, 2], 2, one_based=True)
        assert a != b

    def test_zero_based_vs_one_based(self):
        assert parse_terms_to_index_sets([1], 3, one_based=True) == ((0,),)
        assert parse_terms_to_index_sets([1], 3, one_based=False) == ((1,),)


class TestSEKernelFastPaths:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_sqdist_path_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(7, 4))
        Xp = rng.uniform(-1, 1, size=(5, 4))
        ell = rng.uniform(0.3, 1.5, size=4)
        sigma2 = 0.7
        expected = _ref_se_kernel(X, Xp, ell, sigma2)
        no_sqdist = se_kernel_matrix(X, Xp, ell, sigma2=sigma2)
        diff = X[:, None, :] - Xp[None, :, :]
        with_sqdist = se_kernel_matrix(X, Xp, ell, sigma2=sigma2, sqdist=diff * diff)
        np.testing.assert_allclose(no_sqdist, expected, atol=1e-14)
        np.testing.assert_allclose(with_sqdist, expected, atol=1e-14)

    def test_sqdist_when_X_equals_Xp(self):
        rng = np.random.default_rng(11)
        X = rng.uniform(-1, 1, size=(9, 3))
        ell = rng.uniform(0.2, 1.0, size=3)
        diff = X[:, None, :] - X[None, :, :]
        got = se_kernel_matrix(X, X, ell, sigma2=1.1, sqdist=diff * diff)
        np.testing.assert_allclose(got, _ref_se_kernel(X, X, ell, 1.1), atol=1e-14)


class TestHMatrixVectorization:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_matches_per_column_reference(self, seed):
        rng = np.random.default_rng(seed)
        d = 4
        X = rng.uniform(-1, 1, size=(6, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        terms = [None, 1, 2, 3, 4, (1, 2), (3, 4)]
        expected = _ref_h_matrix(X, ell, 0.9, terms, one_based=True)
        got = h_matrix_se(X, ell, 0.9, terms, one_based=True)
        np.testing.assert_allclose(got, expected, atol=1e-14)

    def test_intercept_only(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(-1, 1, size=(5, 2))
        ell = np.array([0.4, 0.9])
        expected = _ref_h_matrix(X, ell, 1.0, [None])
        np.testing.assert_allclose(h_matrix_se(X, ell, 1.0, [None]), expected, atol=1e-14)


class TestMakeCStarMatrix:
    @pytest.mark.parametrize("orthogonal", [True, False])
    def test_train_train_matches_reference(self, orthogonal):
        rng = np.random.default_rng(5)
        d = 3
        X = rng.uniform(-1, 1, size=(8, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        sigma2 = 1.3
        terms = [None, 1, 2, 3]
        expected = _ref_se_kernel(X, X, ell, sigma2)
        if orthogonal:
            hX = _ref_h_matrix(X, ell, sigma2, terms)
            Hdiag = H_diag_se(ell, sigma2, terms)
            expected = expected - (hX * (1.0 / Hdiag)) @ hX.T
        got_no_sqdist = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2,
                                           terms=terms, orthogonal=orthogonal)
        diff = X[:, None, :] - X[None, :, :]
        got_sqdist = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms,
                                        orthogonal=orthogonal, sqdist=diff * diff)
        np.testing.assert_allclose(got_no_sqdist, expected, atol=1e-13)
        np.testing.assert_allclose(got_sqdist, expected, atol=1e-13)

    def test_X_is_Xp_shortcut_matches_two_calls(self):
        rng = np.random.default_rng(9)
        X = rng.uniform(-1, 1, size=(7, 3))
        ell = rng.uniform(0.3, 1.5, size=3)
        terms = [None, 1, 2, 3]
        shortcut = make_c_star_matrix(X, X, ell=ell, sigma2=0.8, terms=terms)
        # Separate copy defeats the `is` shortcut.
        two_calls = make_c_star_matrix(X, X.copy(), ell=ell, sigma2=0.8, terms=terms)
        np.testing.assert_allclose(shortcut, two_calls, atol=1e-14)

    def test_cross_kernel_X_neq_Xp(self):
        rng = np.random.default_rng(13)
        d = 3
        X = rng.uniform(-1, 1, size=(6, d))
        Xp = rng.uniform(-1, 1, size=(4, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        terms = [None, 1, 2, 3]
        expected = _ref_se_kernel(X, Xp, ell, 1.1)
        hX = _ref_h_matrix(X, ell, 1.1, terms)
        hXp = _ref_h_matrix(Xp, ell, 1.1, terms)
        Hdiag = H_diag_se(ell, 1.1, terms)
        expected = expected - (hX * (1.0 / Hdiag)) @ hXp.T
        got = make_c_star_matrix(X, Xp, ell=ell, sigma2=1.1, terms=terms)
        np.testing.assert_allclose(got, expected, atol=1e-13)


# ===========================================================================
# B1 — profiled GLS terms via Kronecker structure
# ===========================================================================


def _fit_model_via_single_nll(X, Y, q, **kwargs):
    """Drive a model into a 'fitted' state via one _nll call (faster than fit())."""
    d, p = X.shape[1], Y.shape[1]
    theta = _build_theta(q, d, 1e-2 * np.ones(p), seed=1)
    model = _make_model(X, Y, q, **kwargs)
    model._nll(theta, build_cache=True)
    model.theta_hat = theta
    model.fitted = True
    return model, theta


class TestProfiledGLSFast:
    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (1, 3, 4), (5, 1, 1), (7, 3, 6)])
    def test_fast_matches_reference(self, seed, q, p):
        X, Y = _make_dataset(n=12, d=3, p=p, seed=seed)
        model, _ = _fit_model_via_single_nll(X, Y, q)
        cache = model.cache
        fast_info = cache["fast_diag_info"]
        assert fast_info is not None

        G = cache["G"]
        Gy = cache["Gy"]
        vecY = vecF(model.Y)
        n = X.shape[0]

        def solve_Ky(rhs): return model._apply_Ky_inv_fast(rhs, fast_info)
        qf_ref, b_ref, r_ref, kir_ref = _profiled_gls_terms(
            solve_Ky, G, Gy, vecY, p, build_cache=True
        )
        alpha_mat = unvecF(solve_Ky(vecY), n, p)
        qf, b, r, kir = _profiled_gls_terms_fast(
            fast_info, G, vecY, alpha_mat, p, build_cache=True,
        )
        np.testing.assert_allclose(qf, qf_ref, atol=1e-9)
        np.testing.assert_allclose(b, b_ref, atol=1e-9)
        np.testing.assert_allclose(r, r_ref, atol=1e-9)
        np.testing.assert_allclose(kir, kir_ref, atol=1e-9)

    def test_no_cache_branch_returns_qf_only(self):
        X, Y = _make_dataset(n=10, d=2, p=3, seed=2)
        model, _ = _fit_model_via_single_nll(X, Y, q=2)
        fast_info = model.cache["fast_diag_info"]
        vecY = vecF(model.Y)
        alpha_mat = unvecF(model._apply_Ky_inv_fast(vecY, fast_info), X.shape[0], 3)
        qf, b, r, kir = _profiled_gls_terms_fast(
            fast_info, model.cache["G"], vecY, alpha_mat, 3, build_cache=False,
        )
        assert b is None and r is None and kir is None
        qf2, *_ = _profiled_gls_terms_fast(
            fast_info, model.cache["G"], vecY, alpha_mat, 3, build_cache=True,
        )
        np.testing.assert_allclose(qf, qf2, atol=1e-12)


# ===========================================================================
# NLL: fast path vs independent dense reference
# ===========================================================================


class TestNLLExactness:
    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (2, 2, 2), (3, 3, 5), (4, 1, 1)])
    def test_fast_nll_matches_dense_reference(self, seed, q, p):
        X, Y = _make_dataset(n=10, d=3, p=p, seed=seed)
        d = X.shape[1]
        terms = [None] + list(range(1, d + 1))
        theta = _build_theta(q, d, 5e-3 * np.ones(p), seed=seed)
        model = _make_model(X, Y, q, terms=terms, use_analytical_grad=False)
        nll_fast = float(model._nll(theta, build_cache=False))

        Ky, _, _, _ = _dense_Ky_from_theta(X, Y, theta, q, d, p, terms, orthogonal=True)
        n = X.shape[0]
        G = make_G({"X_scaled": X}, terms, one_based=True, return_names=False)
        Gy = build_Gy(G, p)
        vecY = vecF(Y)
        L, lo = cho_factor(Ky, lower=True, check_finite=False)
        z = cho_solve((L, lo), Gy, check_finite=False)
        alpha = cho_solve((L, lo), vecY, check_finite=False)
        A_gls = Gy.T @ z
        b_gls = Gy.T @ alpha
        beta = np.linalg.solve(A_gls, b_gls)
        qf = float(vecY @ alpha - b_gls @ beta)
        logdetK = 2.0 * float(np.sum(np.log(np.diag(L))))
        # _nll returns NLL / n (per-row).
        nll_ref = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi)) / float(n)
        np.testing.assert_allclose(nll_fast, nll_ref, atol=1e-10)


# ===========================================================================
# Gradients: B3 closed-form latent kernel, analytical NLL grad, autograd, FD
# ===========================================================================


def _autograd_trace_grad(M_k, X, sqdist, ell, sigma2, *, terms, orthogonal):
    d = X.shape[1]

    def trace_fn(log_theta):
        s2 = anp.exp(log_theta[0])
        ell_ = anp.exp(log_theta[1:1 + d])
        Ck = make_c_star_matrix(
            X, X, ell=ell_, sigma2=s2, terms=terms,
            orthogonal=orthogonal, one_based=True, sqdist=sqdist,
        )
        return anp.sum(M_k * Ck)

    log_theta = np.concatenate([[float(np.log(sigma2))], np.log(np.asarray(ell))])
    return np.asarray(ag_grad(trace_fn)(log_theta), dtype=float)


class TestLatentKernelGrad:
    @pytest.mark.parametrize(
        "seed,n,d,p,q,orthogonal,interaction",
        [
            (0, 12, 3, 3, 2, True, False),
            (1, 14, 4, 3, 3, True, False),
            (5, 16, 3, 4, 2, False, False),  # orthogonal=False
            (7, 12, 3, 3, 2, True, True),    # with interaction
            (11, 8, 1, 1, 1, True, False),   # q=p=d=1 corner
        ],
    )
    def test_closed_form_matches_autograd(self, seed, n, d, p, q, orthogonal, interaction):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, d))
        diff = X[:, None, :] - X[None, :, :]
        sqdist = diff * diff
        terms = [None] + list(range(1, d + 1))
        if interaction and d >= 2:
            terms.append((1, 2))
        A = rng.standard_normal((n, n))
        M_k = 0.5 * (A + A.T)
        sigma2 = float(np.exp(rng.uniform(-0.4, 0.4)))
        ell = np.exp(rng.uniform(-0.5, 0.5, size=d))

        g_ana = _latent_kernel_logtheta_grad(
            M_k, X, sqdist, ell, sigma2, terms,
            orthogonal=orthogonal, one_based=True,
        )
        g_auto = _autograd_trace_grad(
            M_k, X, sqdist, ell, sigma2, terms=terms, orthogonal=orthogonal,
        )
        assert g_ana.shape == (1 + d,)
        np.testing.assert_allclose(g_ana, g_auto, atol=1e-10)

    @pytest.mark.parametrize("seed,n,d,orthogonal", [(0, 10, 3, True), (5, 14, 2, False)])
    def test_logsigma_derivative_equals_trace(self, seed, n, d, orthogonal):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, d))
        diff = X[:, None, :] - X[None, :, :]
        sqdist = diff * diff
        terms = [None] + list(range(1, d + 1))
        A = rng.standard_normal((n, n))
        M_k = 0.5 * (A + A.T)
        sigma2 = float(np.exp(0.3))
        ell = np.exp(rng.uniform(-0.4, 0.4, size=d))
        Ck = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms,
                                orthogonal=orthogonal, one_based=True, sqdist=sqdist)
        expected = float(np.sum(M_k * Ck))
        g = _latent_kernel_logtheta_grad(M_k, X, sqdist, ell, sigma2, terms,
                                          orthogonal=orthogonal, one_based=True)
        np.testing.assert_allclose(g[0], expected, atol=1e-12)


class TestAnalyticalNLLGrad:
    @pytest.mark.parametrize(
        "seed,n,d,p,q,orthogonal,interaction",
        [
            (0, 14, 3, 3, 2, True, False),
            (1, 12, 4, 3, 3, True, False),
            (3, 10, 2, 2, 2, True, False),
            (4, 16, 3, 4, 2, False, False),
            (7, 14, 3, 3, 2, True, True),
            (9, 10, 1, 1, 1, True, False),
        ],
    )
    def test_value_and_grad_match_autograd(self, seed, n, d, p, q, orthogonal, interaction):
        """_nll_and_grad_fast (analytical) == value_and_grad(_nll) (autograd-traced)."""
        X, Y = _make_dataset(n, d, p, seed)
        terms = [None] + list(range(1, d + 1))
        if interaction and d >= 2:
            terms.append((1, 2))
        model = _make_model(X, Y, q, terms=terms, orthogonal=orthogonal)
        theta = _build_theta(q, d, np.full(p, 5e-3), seed=seed)

        nll_a, grad_a = model._nll_and_grad_fast(theta)
        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        nll_auto, grad_auto = vg(theta)
        np.testing.assert_allclose(nll_a, float(nll_auto), atol=1e-10)
        np.testing.assert_allclose(grad_a, np.asarray(grad_auto, float), atol=1e-9)

    def test_autograd_grad_matches_finite_difference(self):
        """One FD sanity check that the autograd-traced grad is correct."""
        X, Y = _make_dataset(n=8, d=2, p=2, seed=1)
        model = _make_model(X, Y, q=2, terms=[None, 1, 2], use_analytical_grad=False)
        theta = _build_theta(2, 2, np.full(2, 5e-3), seed=1)
        _, g_auto = value_and_grad(lambda th: model._nll(th, build_cache=False))(theta)
        eps = 1e-6
        g_num = np.zeros_like(theta)
        for i in range(theta.size):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += eps
            minus[i] -= eps
            g_num[i] = (
                float(model._nll(plus, build_cache=False))
                - float(model._nll(minus, build_cache=False))
            ) / (2 * eps)
        np.testing.assert_allclose(np.asarray(g_auto, float), g_num, rtol=1e-4, atol=1e-5)

    def test_learn_sigma_eps_false_mode(self):
        """When learn_sigma_eps=False, theta has only latent params."""
        X, Y = _make_dataset(n=10, d=2, p=2, seed=2)
        model = MOOGP(
            terms=[None, 1, 2], q=2, Psi=None, orthogonal=True,
            learn_Psi=False, sigma_eps2=np.array([5e-3, 5e-3]),
            learn_sigma_eps=False, jitter=0.0, one_based=True,
            normalize_cols=True, use_diagonalized_interaction=True,
            use_slow_kyinv=False, standardize_y=False,
            use_analytical_grad=True, standardize_x=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})
        theta = _build_theta(q=2, d=X.shape[1], sigma_eps2=np.array([]), seed=3)
        nll_ref = float(model._nll(theta, build_cache=False))
        nll, grad = model._nll_and_grad_fast(theta)
        np.testing.assert_allclose(nll, nll_ref, atol=1e-12)
        assert grad.shape == theta.shape


# ===========================================================================
# B2 — predict variance diagonal via Woodbury
# ===========================================================================


class TestPredictVarianceDiagFast:
    @pytest.mark.parametrize("seed,q,p,n_star", [(0, 2, 3, 9), (3, 3, 5, 17), (7, 1, 1, 5)])
    def test_diag_matches_dense_woodbury(self, seed, q, p, n_star):
        X, Y = _make_dataset(n=14, d=3, p=p, seed=seed)
        model, _ = _fit_model_via_single_nll(X, Y, q)
        cache = model.cache
        fast_info = cache["fast_diag_info"]
        rng = np.random.default_rng(seed + 42)
        Xs = rng.uniform(-1, 1, size=(n_star, X.shape[1]))

        Psi = cache["Psi"]
        lat_params = cache["lat_params"]
        terms = cache["terms"]
        sigma_eps2 = cache["sigma_eps2"]

        Cj_XsX = [
            make_c_star_matrix(Xs, X, ell=ell_j, sigma2=s2_j, terms=terms,
                               orthogonal=model.orthogonal, one_based=cache["one_based"])
            for (s2_j, ell_j) in lat_params
        ]
        Cj_diag_star = [
            make_c_star_diag(Xs, ell=ell_j, sigma2=s2_j, terms=terms,
                             orthogonal=model.orthogonal, one_based=cache["one_based"])
            for (s2_j, ell_j) in lat_params
        ]

        K_XsX = build_cross_K(Psi, Cj_XsX)
        diag_prior = np.zeros(n_star * p)
        for j in range(q):
            diag_prior += np.kron(Psi[:, j] ** 2, Cj_diag_star[j])
        V = model._solve_with_cached_Ky(K_XsX.T)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag_ref_signal = diag_prior - diag_cross
        diag_ref_obs = diag_ref_signal + np.repeat(sigma_eps2, n_star)

        diag_fast_signal = _predict_variance_diag_fast(
            fast_info, Cj_XsX, Psi, Cj_diag_star_list=Cj_diag_star,
            predict_observation=False,
        )
        diag_fast_obs = _predict_variance_diag_fast(
            fast_info, Cj_XsX, Psi, Cj_diag_star_list=Cj_diag_star,
            predict_observation=True,
        )
        np.testing.assert_allclose(diag_fast_signal, diag_ref_signal, atol=1e-9)
        np.testing.assert_allclose(diag_fast_obs, diag_ref_obs, atol=1e-9)


# ===========================================================================
# End-to-end: predict and fit equivalence across paths
# ===========================================================================


class TestPredictEquivalence:
    @pytest.mark.parametrize("seed,q,p", [(0, 2, 3), (1, 3, 5), (4, 1, 1)])
    def test_fast_path_predict_matches_slow_path(self, seed, q, p):
        X, Y = _make_dataset(n=14, d=3, p=p, seed=seed)
        d = X.shape[1]
        terms = [None] + list(range(1, d + 1))
        theta = _build_theta(q, d, 5e-3 * np.ones(p), seed=seed)

        m_fast = _make_model(X, Y, q, terms=terms)
        m_fast._nll(theta, build_cache=True)
        m_fast.theta_hat = theta
        m_fast.fitted = True

        # Slow path: requires explicit Psi (the SVD-of-Y construction is what
        # use_slow_kyinv bypasses).
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
        np.testing.assert_allclose(mean_fast, mean_slow, atol=1e-9)
        np.testing.assert_allclose(std_fast, std_slow, atol=1e-9)

    def test_predict_observation_toggle_adds_noise(self):
        X, Y = _make_dataset(n=12, d=2, p=4, seed=8)
        m, _ = _fit_model_via_single_nll(X, Y, q=2)
        rng = np.random.default_rng(0)
        Xs = rng.uniform(-1, 1, size=(7, 2))
        _, std_signal = m.predict(Xs, return_std=True, predict_observation=False)
        _, std_obs = m.predict(Xs, return_std=True, predict_observation=True)
        diff = std_obs ** 2 - std_signal ** 2
        sigma_raw = np.asarray(m.cache["sigma_eps2_raw"], dtype=float).ravel()
        expected = np.broadcast_to(sigma_raw[None, :], diff.shape)
        np.testing.assert_allclose(diff, expected, atol=1e-9)


class TestFitEquivalence:
    def test_fit_analytical_grad_matches_autograd_path(self):
        X, Y = _make_dataset(n=20, d=3, p=3, seed=9)
        q, d, p = 3, X.shape[1], Y.shape[1]
        theta0 = _build_theta(q, d, np.full(p, 5e-3), seed=9)
        bounds = _bounds_for(q, d, p)

        m_auto = _make_model(X, Y, q, use_analytical_grad=False)
        m_auto.fit(data={"X_scaled": X, "y": Y}, theta0=theta0.copy(),
                   bounds=bounds, optimizer_opts={"maxiter": 200})
        m_ana = _make_model(X, Y, q, use_analytical_grad=True)
        m_ana.fit(data={"X_scaled": X, "y": Y}, theta0=theta0.copy(),
                  bounds=bounds, optimizer_opts={"maxiter": 200})
        np.testing.assert_allclose(m_ana.nll_hat, m_auto.nll_hat, atol=1e-6)

        rng = np.random.default_rng(99)
        Xs = rng.uniform(-1, 1, size=(5, d))
        mean_auto, std_auto = m_auto.predict(Xs, return_std=True)
        mean_ana, std_ana = m_ana.predict(Xs, return_std=True)
        np.testing.assert_allclose(mean_ana, mean_auto, atol=1e-6)
        np.testing.assert_allclose(std_ana, std_auto, atol=1e-6)
