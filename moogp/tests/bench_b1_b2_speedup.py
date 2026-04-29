"""Quick wall-clock micro-benchmark for the B1 / B2 optimizations.

Not a pytest run-to-pass benchmark — invoke directly with the venv:
    .venv/bin/python -m moogp.tests.bench_b1_b2_speedup
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import cho_solve

from moogp.design import build_Gy, vecF, unvecF
from moogp.kernels import make_c_star_diag, make_c_star_matrix
from moogp.model import (
    MOOGP,
    _profiled_gls_terms,
    _profiled_gls_terms_fast,
    _predict_variance_diag_fast,
    build_cross_K,
)


def _make_data(n, p, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(1.7 * X[:, j0]) + 0.5 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _theta(q, d, p, seed=1):
    rng = np.random.default_rng(seed)
    th = []
    for _ in range(q):
        th.append(float(np.log(rng.uniform(0.6, 1.4))))
        th.extend([float(np.log(rng.uniform(0.4, 1.2))) for _ in range(d)])
    th.extend(list(np.log(5e-3 * np.ones(p))))
    return np.asarray(th, dtype=float)


def _make_model(X, Y, q, theta):
    d = X.shape[1]
    m = MOOGP(
        terms=[None] + list(range(1, d + 1)), q=q, Psi=None,
        orthogonal=True, learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
        use_diagonalized_interaction=True, use_slow_kyinv=False,
        standardize_y=False, use_analytical_grad=True,
    )
    m._prepare_data({"X_scaled": X, "y": Y})
    m._nll(theta, build_cache=True)
    m.theta_hat = theta
    m.fitted = True
    return m


def time_b1(n, p, q=5, d=4, n_iter=10):
    X, Y = _make_data(n, p, d=d, seed=0)
    theta = _theta(q, d, p)
    model = _make_model(X, Y, q, theta)
    cache = model.cache
    fast_info = cache["fast_diag_info"]
    G = cache["G"]
    Gy = cache["Gy"]
    vecY = vecF(model.Y)

    solve_Ky = lambda rhs: model._apply_Ky_inv_fast(rhs, fast_info)

    # Warm-up
    _profiled_gls_terms(solve_Ky, G, Gy, vecY, p, build_cache=True)
    alpha_vec = solve_Ky(vecY)
    alpha_mat = unvecF(alpha_vec, n, p)
    _profiled_gls_terms_fast(fast_info, G, vecY, alpha_mat, p, build_cache=True)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _profiled_gls_terms(solve_Ky, G, Gy, vecY, p, build_cache=True)
    t_old = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        a = solve_Ky(vecY)
        am = unvecF(a, n, p)
        _profiled_gls_terms_fast(fast_info, G, vecY, am, p, build_cache=True)
    t_new = (time.perf_counter() - t0) / n_iter
    return t_old, t_new


def time_b2(n, p, q=5, d=4, n_star=200, n_iter=5):
    X, Y = _make_data(n, p, d=d, seed=0)
    theta = _theta(q, d, p)
    model = _make_model(X, Y, q, theta)
    cache = model.cache
    fast_info = cache["fast_diag_info"]
    Psi = cache["Psi"]
    lat_params = cache["lat_params"]
    terms = cache["terms"]
    sigma_eps2 = cache["sigma_eps2"]

    rng = np.random.default_rng(0)
    Xs = rng.uniform(-1, 1, size=(n_star, X.shape[1]))

    Cj_XsX = [
        make_c_star_matrix(Xs, X, ell=ell_j, sigma2=sigma2_j, terms=terms,
                           orthogonal=True, one_based=True)
        for (sigma2_j, ell_j) in lat_params
    ]
    Cj_diag_star = [
        make_c_star_diag(Xs, ell=ell_j, sigma2=sigma2_j, terms=terms,
                         orthogonal=True, one_based=True)
        for (sigma2_j, ell_j) in lat_params
    ]

    # Old dense path
    def dense_path():
        K_XsX = build_cross_K(Psi, Cj_XsX)
        diag_prior = np.zeros(n_star * p)
        for j in range(q):
            diag_prior += np.kron(Psi[:, j] ** 2, Cj_diag_star[j])
        V = model._solve_with_cached_Ky(K_XsX.T)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag = diag_prior - diag_cross + np.repeat(sigma_eps2, n_star)
        return diag

    def fast_path():
        return _predict_variance_diag_fast(
            fast_info, Cj_XsX, Psi,
            Cj_diag_star_list=Cj_diag_star,
            predict_observation=True,
        )

    # Warm up + sanity
    a = dense_path()
    b = fast_path()
    assert np.allclose(a, b, rtol=0, atol=1e-9)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        dense_path()
    t_old = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_path()
    t_new = (time.perf_counter() - t0) / n_iter
    return t_old, t_new


if __name__ == "__main__":
    print("=== B1 — profiled GLS terms (per likelihood call) ===")
    print(f"{'n':>5} {'p':>4} {'q':>3}  {'old (ms)':>11} {'new (ms)':>11}  {'speedup':>9}")
    for n, p, q in [(100, 4, 3), (200, 4, 3), (200, 20, 5), (400, 20, 5), (800, 20, 5)]:
        t_old, t_new = time_b1(n, p, q=q, d=4, n_iter=5)
        print(f"{n:>5} {p:>4} {q:>3}  {1e3*t_old:>11.2f} {1e3*t_new:>11.2f}  {t_old/t_new:>8.2f}x")

    print()
    print("=== B2 — predict variance diagonal (per call) ===")
    print(f"{'n':>5} {'p':>4} {'q':>3} {'n*':>4}  {'old (ms)':>11} {'new (ms)':>11}  {'speedup':>9}")
    for n, p, q, ns in [(200, 4, 3, 100), (200, 20, 5, 200), (400, 20, 5, 200), (800, 20, 5, 200)]:
        t_old, t_new = time_b2(n, p, q=q, d=4, n_star=ns, n_iter=3)
        print(f"{n:>5} {p:>4} {q:>3} {ns:>4}  {1e3*t_old:>11.2f} {1e3*t_new:>11.2f}  {t_old/t_new:>8.2f}x")
