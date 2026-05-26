"""Exactness tests for the B3 closed-form latent-kernel gradient.

The B3 change replaces the autograd-traced trace functional
``tr(M_k * C*_k(theta_k))`` in ``_nll_and_grad_fast`` with a closed-form
gradient assembled from the elementary M / L / IM / ILL pieces of the
orthogonal SE kernel and their log-ell derivatives.

These tests verify the new helper ``_latent_kernel_logtheta_grad`` against:

  1. ``autograd.grad`` applied to the same trace functional through
     ``make_c_star_matrix`` (the gradient the old block used to compute).
  2. ``value_and_grad(_nll)`` end-to-end (via the existing
     ``_nll_and_grad_fast`` path).

Configurations exercised:
  * ``orthogonal=True`` with intercept + main-effects design.
  * ``orthogonal=False`` (no h-correction; bare SE kernel only).
  * ``orthogonal=True`` with an interaction term in ``terms``.
  * ``q=1, p=1`` corner case.
  * Several ``(n, d, p, q)`` sizes.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
import pytest
from autograd import grad as ag_grad, value_and_grad

from moogp.kernels import make_c_star_matrix
from moogp.model import MOOGP, _latent_kernel_logtheta_grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n, d, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(X[:, j0]) + 0.3 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _build_theta(q, d, sigma_eps2, seed):
    rng = np.random.default_rng(seed + 1000)
    theta = []
    for _ in range(q):
        theta.append(float(np.log(rng.uniform(0.5, 1.5))))
        theta.extend([float(np.log(rng.uniform(0.4, 0.9))) for _ in range(d)])
    theta.extend(list(np.log(sigma_eps2)))
    return np.asarray(theta, dtype=float)


def _make_model(X, Y, q, *, terms, orthogonal, learn_sigma_eps=True,
                use_analytical_grad=True):
    d = X.shape[1]
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
    model._prepare_data({"X_scaled": X, "y": Y})
    return model


def _autograd_trace_grad(M_k, X, sqdist, ell, sigma2, *, terms, orthogonal,
                         one_based):
    """Reference: autograd ∂/∂(log_sigma2, log_ell) of tr(M_k * C*_k)."""
    d = X.shape[1]

    def trace_fn(log_theta):
        s2 = anp.exp(log_theta[0])
        ell_ = anp.exp(log_theta[1:1 + d])
        Ck = make_c_star_matrix(
            X, X,
            ell=ell_,
            sigma2=s2,
            terms=terms,
            orthogonal=orthogonal,
            one_based=one_based,
            sqdist=sqdist,
        )
        return anp.sum(M_k * Ck)

    log_theta = np.concatenate([[float(np.log(sigma2))], np.log(np.asarray(ell))])
    return np.asarray(ag_grad(trace_fn)(log_theta), dtype=float)


# ---------------------------------------------------------------------------
# Per-latent trace functional matches autograd ref (the function the old
# block used to differentiate).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,n,d,p,q,orthogonal,terms",
    [
        (0, 12, 3, 3, 2, True, "intercept_main"),
        (1, 14, 4, 3, 3, True, "intercept_main"),
        (3, 10, 2, 2, 2, True, "intercept_main"),
        (5, 16, 3, 4, 2, False, "intercept_main"),
        (7, 12, 3, 3, 2, True, "with_interaction"),
        (9, 10, 4, 3, 1, True, "intercept_main"),
        (11, 8, 1, 1, 1, True, "intercept_main"),
    ],
)
def test_latent_kernel_grad_matches_autograd(seed, n, d, p, q, orthogonal, terms):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    diff = X[:, None, :] - X[None, :, :]
    sqdist = diff * diff

    if terms == "intercept_main":
        terms_list = [None] + list(range(1, d + 1))
    elif terms == "with_interaction":
        terms_list = [None] + list(range(1, d + 1))
        if d >= 2:
            terms_list.append((1, 2))
    else:
        raise ValueError(terms)

    # Random symmetric M_k (the gradient functional only assumes symmetry).
    A = rng.standard_normal((n, n))
    M_k = 0.5 * (A + A.T)

    sigma2 = float(np.exp(rng.uniform(-0.4, 0.4)))
    ell = np.exp(rng.uniform(-0.5, 0.5, size=d))

    g_ana = _latent_kernel_logtheta_grad(
        M_k, X, sqdist, ell, sigma2, terms_list,
        orthogonal=orthogonal, one_based=True,
    )
    g_auto = _autograd_trace_grad(
        M_k, X, sqdist, ell, sigma2,
        terms=terms_list, orthogonal=orthogonal, one_based=True,
    )

    assert g_ana.shape == g_auto.shape == (1 + d,)
    np.testing.assert_allclose(g_ana, g_auto, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end: _nll_and_grad_fast (now using B3) matches value_and_grad(_nll).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,n,d,p,q,orthogonal,interaction",
    [
        (0, 14, 3, 3, 2, True, False),
        (1, 12, 4, 3, 3, True, False),
        (3, 10, 2, 2, 2, True, False),
        (5, 16, 3, 4, 2, False, False),
        (7, 14, 3, 3, 2, True, True),
        (9, 10, 1, 1, 1, True, False),
    ],
)
def test_full_grad_matches_autograd_value_and_grad(seed, n, d, p, q, orthogonal,
                                                   interaction):
    X, Y = _make_dataset(n, d, p, seed)
    terms = [None] + list(range(1, d + 1))
    if interaction and d >= 2:
        terms.append((1, 2))
    model = _make_model(X, Y, q, terms=terms, orthogonal=orthogonal)
    sigma_eps2 = np.full(p, 5e-3)
    theta = _build_theta(q, d, sigma_eps2, seed=seed)

    nll_ana, grad_ana = model._nll_and_grad_fast(theta)

    vg = value_and_grad(lambda th: model._nll(th, build_cache=False))
    nll_auto, grad_auto = vg(theta)
    grad_auto = np.asarray(grad_auto, dtype=float)

    np.testing.assert_allclose(nll_ana, float(nll_auto), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(grad_ana, grad_auto, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Sigma2-derivative shortcut: ∂tr(M C*)/∂log(sigma2) = tr(M C*).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,n,d,orthogonal", [
    (0, 10, 3, True),
    (2, 12, 4, True),
    (5, 14, 2, False),
])
def test_logsigma_grad_equals_trace(seed, n, d, orthogonal):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    diff = X[:, None, :] - X[None, :, :]
    sqdist = diff * diff
    terms = [None] + list(range(1, d + 1))

    A = rng.standard_normal((n, n))
    M_k = 0.5 * (A + A.T)
    sigma2 = float(np.exp(0.3))
    ell = np.exp(rng.uniform(-0.4, 0.4, size=d))

    Ck = make_c_star_matrix(
        X, X, ell=ell, sigma2=sigma2,
        terms=terms, orthogonal=orthogonal, one_based=True, sqdist=sqdist,
    )
    expected = float(np.sum(M_k * Ck))

    g = _latent_kernel_logtheta_grad(
        M_k, X, sqdist, ell, sigma2, terms,
        orthogonal=orthogonal, one_based=True,
    )
    np.testing.assert_allclose(g[0], expected, rtol=1e-12, atol=1e-12)
