"""Tests that the Fix 1/2/3 fast-path optimizations are numerically exact
matches of the previous implementation.

Each test constructs an independent "reference" computation using the original
formulas (pre-optimization) and compares bit-for-bit (with a tight tolerance)
against the optimized code path now shipped in ``moogp.kernels`` /
``moogp.design`` / ``moogp.model``.
"""

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad
from autograd.numpy import exp, pi, sqrt
from autograd.scipy.special import erf

from moogp.design import (
    _parse_terms_cached,
    _parse_terms_impl,
    parse_terms_to_index_sets,
)
from moogp.kernels import (
    L_gauss,
    M_gauss,
    h_matrix_se,
    make_c_star_matrix,
    se_kernel_matrix,
)
from moogp.model import MOOGP


# ---------------------------------------------------------------------------
# Reference implementations (copies of the *pre-optimization* code paths).
# ---------------------------------------------------------------------------


def ref_se_kernel_matrix(X, Xp, ell, sigma2=1.0):
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    dif = (X[:, None, :] - Xp[None, :, :]) / ell
    D2 = np.sum(dif * dif, axis=2)
    return sigma2 * np.exp(-D2)


def ref_h_matrix_se(X, ell, sigma2, terms, one_based=True):
    X = np.asarray(X)
    n, d = X.shape
    J_sets = _parse_terms_impl(list(terms), d, one_based, False)
    r = len(J_sets)
    if r == 0:
        return np.empty((n, 0), float)

    M_all = np.column_stack([M_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])
    L_all = np.column_stack([L_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])

    all_idx = set(range(d))
    cols = []
    for Ji in J_sets:
        Ji_s = set(Ji)
        notJ = list(all_idx - Ji_s)
        Ji_l = list(Ji_s)
        col = np.ones(n)
        if notJ:
            col = col * np.prod(M_all[:, notJ], axis=1)
        if Ji_l:
            col = col * np.prod(L_all[:, Ji_l], axis=1)
        cols.append(col)
    return np.column_stack(cols) * sigma2


# ---------------------------------------------------------------------------
# Fix 3 — parse_terms_to_index_sets caching
# ---------------------------------------------------------------------------


class TestParseTermsCaching:
    def test_values_match_uncached_reference(self):
        terms = [None, 1, 2, 3, (1, 2), (2, 3)]
        d = 3
        cached = parse_terms_to_index_sets(terms, d, one_based=True)
        reference = _parse_terms_impl(terms, d, True, False)
        assert list(cached) == list(reference)

    def test_cache_hit_returns_same_object(self):
        terms = [None, 1, 2]
        _parse_terms_cached.cache_clear()
        a = parse_terms_to_index_sets(terms, 2, one_based=True)
        b = parse_terms_to_index_sets(terms, 2, one_based=True)
        # tuple identity means the cached entry is being reused
        assert a is b
        info = _parse_terms_cached.cache_info()
        assert info.hits >= 1

    def test_cached_result_is_tuple_not_mutable_list(self):
        terms = [None, 1, 2]
        result = parse_terms_to_index_sets(terms, 2, one_based=True)
        assert isinstance(result, tuple)

    def test_distinct_keys_do_not_collide(self):
        a = parse_terms_to_index_sets([None, 1], 2, one_based=True)
        b = parse_terms_to_index_sets([None, 2], 2, one_based=True)
        assert a != b

    def test_zero_based_vs_one_based(self):
        a = parse_terms_to_index_sets([1], 3, one_based=True)
        b = parse_terms_to_index_sets([1], 3, one_based=False)
        assert a == ((0,),)
        assert b == ((1,),)


# ---------------------------------------------------------------------------
# Fix 1 — se_kernel_matrix with precomputed sqdist
# ---------------------------------------------------------------------------


class TestSqdistFastPath:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_se_kernel_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(7, 4))
        Xp = rng.uniform(-1, 1, size=(5, 4))
        ell = rng.uniform(0.3, 1.5, size=4)
        sigma2 = 0.7

        expected = ref_se_kernel_matrix(X, Xp, ell, sigma2)
        # Path A: current no-sqdist call (still recomputes subtraction internally).
        got_nosqdist = se_kernel_matrix(X, Xp, ell, sigma2=sigma2)
        # Path B: supply precomputed sqdist explicitly.
        diff = X[:, None, :] - Xp[None, :, :]
        sqdist = diff * diff
        got_sqdist = se_kernel_matrix(X, Xp, ell, sigma2=sigma2, sqdist=sqdist)

        np.testing.assert_allclose(got_nosqdist, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(got_sqdist, expected, rtol=0, atol=1e-14)

    def test_sqdist_matches_when_X_eq_Xp(self):
        rng = np.random.default_rng(11)
        X = rng.uniform(-1, 1, size=(9, 3))
        ell = rng.uniform(0.2, 1.0, size=3)
        sigma2 = 1.1

        expected = ref_se_kernel_matrix(X, X, ell, sigma2)
        diff = X[:, None, :] - X[None, :, :]
        sqdist = diff * diff
        got = se_kernel_matrix(X, X, ell, sigma2=sigma2, sqdist=sqdist)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# Fix 2 — vectorized h_matrix_se and make_c_star_matrix X-is-Xp shortcut
# ---------------------------------------------------------------------------


class TestHMatrixVectorization:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    def test_matches_per_column_reference(self, seed):
        rng = np.random.default_rng(seed)
        d = 4
        X = rng.uniform(-1, 1, size=(6, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        sigma2 = 0.9
        terms = [None, 1, 2, 3, 4, (1, 2), (3, 4)]

        expected = ref_h_matrix_se(X, ell, sigma2, terms, one_based=True)
        got = h_matrix_se(X, ell, sigma2, terms, one_based=True)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)

    def test_intercept_only(self):
        rng = np.random.default_rng(3)
        X = rng.uniform(-1, 1, size=(5, 2))
        ell = np.array([0.4, 0.9])
        terms = [None]
        expected = ref_h_matrix_se(X, ell, 1.0, terms)
        got = h_matrix_se(X, ell, 1.0, terms)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)


class TestMakeCStarMatrix:
    @pytest.mark.parametrize("orthogonal", [True, False])
    def test_train_train_matches_reference(self, orthogonal):
        rng = np.random.default_rng(5)
        d = 3
        X = rng.uniform(-1, 1, size=(8, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        sigma2 = 1.3
        terms = [None, 1, 2, 3]

        expected = ref_se_kernel_matrix(X, X, ell, sigma2)
        if orthogonal:
            hX = ref_h_matrix_se(X, ell, sigma2, terms)
            from moogp.kernels import H_diag_se

            Hdiag = H_diag_se(ell, sigma2, terms)
            expected = expected - (hX * (1.0 / Hdiag)) @ hX.T

        got_no_sqdist = make_c_star_matrix(
            X, X, ell=ell, sigma2=sigma2, terms=terms, orthogonal=orthogonal
        )
        diff = X[:, None, :] - X[None, :, :]
        sqdist = diff * diff
        got_sqdist = make_c_star_matrix(
            X, X, ell=ell, sigma2=sigma2, terms=terms, orthogonal=orthogonal,
            sqdist=sqdist,
        )

        np.testing.assert_allclose(got_no_sqdist, expected, rtol=0, atol=1e-13)
        np.testing.assert_allclose(got_sqdist, expected, rtol=0, atol=1e-13)

    def test_X_is_Xp_shortcut_matches_two_calls(self):
        """When X is the same python object as Xp, the optimized path reuses
        h(X) once. Verify the result is bit-identical to the two-call path.
        """
        rng = np.random.default_rng(9)
        d = 3
        X = rng.uniform(-1, 1, size=(7, d))
        # Separate copy defeats the `is` shortcut.
        X_copy = X.copy()
        ell = rng.uniform(0.3, 1.5, size=d)
        sigma2 = 0.8
        terms = [None, 1, 2, 3]

        shortcut = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms)
        two_calls = make_c_star_matrix(X, X_copy, ell=ell, sigma2=sigma2, terms=terms)

        np.testing.assert_allclose(shortcut, two_calls, rtol=0, atol=1e-14)

    def test_cross_kernel_unaffected(self):
        """Cross kernels (X != Xp) should match the reference exactly."""
        rng = np.random.default_rng(13)
        d = 3
        X = rng.uniform(-1, 1, size=(6, d))
        Xp = rng.uniform(-1, 1, size=(4, d))
        ell = rng.uniform(0.3, 1.5, size=d)
        sigma2 = 1.1
        terms = [None, 1, 2, 3]

        expected = ref_se_kernel_matrix(X, Xp, ell, sigma2)
        hX = ref_h_matrix_se(X, ell, sigma2, terms)
        hXp = ref_h_matrix_se(Xp, ell, sigma2, terms)
        from moogp.kernels import H_diag_se

        Hdiag = H_diag_se(ell, sigma2, terms)
        expected = expected - (hX * (1.0 / Hdiag)) @ hXp.T

        got = make_c_star_matrix(X, Xp, ell=ell, sigma2=sigma2, terms=terms)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# End-to-end: MOOGP NLL + gradient reproducibility
# ---------------------------------------------------------------------------


def _make_small_dataset(n=12, d=3, p=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    # Simple nonlinear multi-output response with correlated noise.
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(X[:, j0]) + 0.3 * X[:, j1])
    Y = np.column_stack(cols)
    Y = Y + 1e-3 * rng.standard_normal(Y.shape)
    return X, Y


def _theta0_for(n_q, d, sigma_eps2_init):
    theta = []
    for _ in range(n_q):
        theta.append(float(np.log(1.0)))
        theta.extend([float(np.log(0.5))] * d)
    theta.extend(list(np.log(sigma_eps2_init)))
    return np.asarray(theta)


class TestMOOGPNLLExactness:
    def test_nll_value_matches_reference(self):
        """Compare NLL at a fixed theta under a reference kernel path (slow,
        using ``ref_*`` functions and dense Ky) against the optimized fast
        path exercised by ``MOOGP._nll``.
        """
        X, Y = _make_small_dataset(n=10, d=3, p=2, seed=2)
        d = X.shape[1]
        p = Y.shape[1]
        q = 2

        sigma_eps2_init = np.array([1e-2, 1e-2])
        theta0 = _theta0_for(q, d, sigma_eps2_init)

        model = MOOGP(
            terms=[None, 1, 2, 3],
            q=q,
            Psi=None,
            orthogonal=True,
            learn_Psi=False,
            learn_sigma_eps=True,
            jitter=0.0,
            one_based=True,
            normalize_cols=True,
            use_diagonalized_interaction=True,
            use_slow_kyinv=False,
            standardize_y=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})
        nll_fast = float(model._nll(theta0, build_cache=False))

        # Reference: construct Ky exactly from the reference C matrices and
        # sigma_eps^{1/2} Phi(Y) basis, then compute the NLL the long way.
        from moogp.model import build_Ky, init_phi, vecF

        Phi, _ = init_phi(Y, q, X.shape[0])
        sigma_eps2 = np.maximum(np.exp(theta0[-p:]), 1e-10)
        Psi_ref = np.diag(np.sqrt(sigma_eps2)) @ Phi

        Cjs = []
        for k in range(q):
            s2 = np.exp(theta0[k * (d + 1)])
            ell = np.exp(theta0[k * (d + 1) + 1 : k * (d + 1) + 1 + d])
            # Use reference (unchanged) kernel path
            C_ref = ref_se_kernel_matrix(X, X, ell, s2)
            hX = ref_h_matrix_se(X, ell, s2, [None, 1, 2, 3])
            from moogp.kernels import H_diag_se

            Hdiag = H_diag_se(ell, s2, [None, 1, 2, 3])
            C_ref = C_ref - (hX * (1.0 / Hdiag)) @ hX.T
            Cjs.append(C_ref)

        Ky = build_Ky(Cjs, Psi_ref, sigma_eps2=sigma_eps2)
        # Profile out trend coefficients using the same GLS closed form the
        # model uses, then evaluate 0.5*(logdet + qf + n*p*log(2 pi)).
        from moogp.design import build_Gy, make_G

        G = make_G({"X_scaled": X}, [None, 1, 2, 3], one_based=True, return_names=False)
        Gy = build_Gy(G, p)
        vecY = vecF(Y)
        L = np.linalg.cholesky(Ky)
        z = np.linalg.solve(L.T, np.linalg.solve(L, Gy))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, vecY))
        A_gls = Gy.T @ z
        b_gls = Gy.T @ alpha
        beta = np.linalg.solve(A_gls, b_gls)
        qf = float(vecY @ alpha - b_gls @ beta)
        logdetK = 2.0 * float(np.sum(np.log(np.diag(L))))
        n = X.shape[0]
        nll_ref = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi))

        np.testing.assert_allclose(nll_fast, nll_ref, rtol=1e-10, atol=1e-10)

    def test_nll_gradient_agrees_with_finite_difference(self):
        """Autograd gradient of the optimized NLL matches central differences."""
        from autograd import value_and_grad

        X, Y = _make_small_dataset(n=8, d=2, p=2, seed=1)
        d = X.shape[1]
        p = Y.shape[1]
        q = 2
        sigma_eps2_init = np.array([5e-3, 5e-3])
        theta0 = _theta0_for(q, d, sigma_eps2_init)

        model = MOOGP(
            terms=[None, 1, 2],
            q=q,
            Psi=None,
            orthogonal=True,
            learn_Psi=False,
            learn_sigma_eps=True,
            jitter=0.0,
            one_based=True,
            normalize_cols=True,
            use_diagonalized_interaction=True,
            use_slow_kyinv=False,
            standardize_y=False,
        )
        model._prepare_data({"X_scaled": X, "y": Y})

        vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
        _, g_analytic = vg(theta0)

        eps = 1e-6
        g_num = np.zeros_like(theta0)
        for i in range(theta0.size):
            plus = theta0.copy()
            minus = theta0.copy()
            plus[i] += eps
            minus[i] -= eps
            g_num[i] = (
                float(model._nll(plus, build_cache=False))
                - float(model._nll(minus, build_cache=False))
            ) / (2 * eps)

        np.testing.assert_allclose(g_analytic, g_num, rtol=1e-4, atol=1e-5)
