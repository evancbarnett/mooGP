"""Wall-clock micro-benchmark for the B3 closed-form latent-kernel gradient.

The B3 change replaces the autograd-traced trace functional
``tr(M_k * C*_k(theta_k))`` per latent inside ``_nll_and_grad_fast`` with a
closed-form gradient. This script times only the per-latent gradient block
(M_k assembly is shared, so we keep it identical between the two paths and
time only the bit that actually changed).

Run with:

    .venv/bin/python -m moogp.tests.bench_b3_latent_grad
"""

from __future__ import annotations

import time

import autograd.numpy as anp
import numpy as np
from autograd import grad as ag_grad

from moogp.kernels import make_c_star_matrix
from moogp.model import (
    MOOGP,
    _latent_kernel_logtheta_grad,
)


def _make_data(n, p, d, seed=0):
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
    return m


def _autograd_block(M_k, X, sqdist, theta_k, *, terms, orthogonal, one_based, d):
    """Same trace-functional autograd path the old _nll_and_grad_fast used."""
    def _tr_Mk_Ck(lat_k_raw):
        s2 = anp.exp(lat_k_raw[0])
        ell = anp.exp(lat_k_raw[1:1 + d])
        Ck = make_c_star_matrix(
            X, X, ell=ell, sigma2=s2,
            terms=terms, orthogonal=orthogonal, one_based=one_based, sqdist=sqdist,
        )
        return anp.sum(M_k * Ck)
    return ag_grad(_tr_Mk_Ck)(theta_k)


def time_b3(n, p, q=3, d=4, n_iter=10):
    X, Y = _make_data(n, p, d=d, seed=0)
    theta = _theta(q, d, p, seed=1)
    model = _make_model(X, Y, q, theta)

    sqdist = model._train_sqdist
    terms = model.terms
    orthogonal = model.orthogonal
    one_based = model.one_based

    # Build M_k once per latent (independent of B3 — same in both paths).
    rng = np.random.default_rng(11)
    M_list = []
    theta_blocks = []
    for k in range(q):
        A = rng.standard_normal((n, n))
        M_list.append(0.5 * (A + A.T))
        theta_blocks.append(theta[k * (d + 1):(k + 1) * (d + 1)].astype(float))

    sigmas = [float(np.exp(t[0])) for t in theta_blocks]
    ells = [np.exp(t[1:1 + d]) for t in theta_blocks]

    # Warm-up
    for k in range(q):
        _ = _autograd_block(M_list[k], X, sqdist, theta_blocks[k],
                            terms=terms, orthogonal=orthogonal, one_based=one_based, d=d)
        _ = _latent_kernel_logtheta_grad(M_list[k], X, sqdist, ells[k], sigmas[k],
                                         terms, orthogonal=orthogonal, one_based=one_based)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for k in range(q):
            _autograd_block(M_list[k], X, sqdist, theta_blocks[k],
                            terms=terms, orthogonal=orthogonal, one_based=one_based, d=d)
    t_old = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for k in range(q):
            _latent_kernel_logtheta_grad(M_list[k], X, sqdist, ells[k], sigmas[k],
                                         terms, orthogonal=orthogonal, one_based=one_based)
    t_new = (time.perf_counter() - t0) / n_iter
    return t_old, t_new


if __name__ == "__main__":
    print("=== B3 — latent-kernel gradient block (per likelihood call, all q latents) ===")
    print(f"{'n':>5} {'p':>4} {'q':>3} {'d':>3}  {'old (ms)':>11} {'new (ms)':>11}  {'speedup':>9}")
    for n, p, q, d in [
        (100, 4, 3, 4),
        (200, 4, 3, 4),
        (200, 20, 5, 4),
        (400, 20, 5, 4),
        (800, 20, 5, 4),
        (200, 4, 3, 6),
        (400, 4, 3, 6),
    ]:
        t_old, t_new = time_b3(n, p, q=q, d=d, n_iter=5)
        print(f"{n:>5} {p:>4} {q:>3} {d:>3}  {1e3*t_old:>11.2f} {1e3*t_new:>11.2f}  {t_old/t_new:>8.2f}x")
