# moogp/tests/test_learn_psi.py
import numpy as np

from src.moogp.datasets import generate_forrester_data
from src.moogp.model import MOOGP, unpack_theta


def test_learn_psi():
    # Use Forrester toy: 1D input, 3 outputs (nice and small/fast)
    data = generate_forrester_data(n=18, seed=0)
    X = data["X_scaled"]
    Y = data["y"]
    n, d = X.shape
    p = Y.shape[1]

    # Mean basis: intercept + main effect
    terms = [None] + list(range(1, d + 1))

    # Choose q < p so Psi learning is non-trivial
    q = 2

    # Model with learn_Psi=True. Psi=None here: it will be learned from theta.
    model = MOOGP(
        terms=terms,
        q=q,
        Psi=None,
        learn_Psi=True,
        learn_sigma_eps=False,
        use_reml=False,
        jitter=1e-6,
        one_based=True,
        normalize_cols=True,
        standardize_x=False,
        standardize_y=False,
    )

    # Prepare internal X/Y/n/d/p
    model._prepare_data(data)

    # ---- Build theta0: [latent params ... | Psi_free entries ...] ----
    theta_latent = []
    bounds = []

    # For each latent j: [log_sigma2_j, log_ell_j1, ..., log_ell_jd]
    for _ in range(q):
        theta_latent.append(np.log(1.0))              # log sigma^2
        theta_latent.extend([np.log(0.5)] * d)        # log lengthscales
        bounds.append((np.log(1e-3), np.log(1e3)))    # sigma^2 bounds
        bounds.extend([(np.log(0.05), np.log(5.0))] * d)  # ell bounds

    theta_latent = np.array(theta_latent)

    # Free Psi parameters (unconstrained in theta; columns normalized in unpack)
    rng = np.random.default_rng(0)
    Psi0_free = rng.standard_normal((p, q))
    theta0 = np.concatenate([theta_latent, Psi0_free.ravel()])

    # Bounds for Psi entries: effectively wide box constraints
    bounds.extend([(-5.0, 5.0)] * (p * q))
    bounds = list(bounds)

    # ---- NLL at initial theta0 (for comparison) ----
    nll0 = model._nll(theta0)

    # ---- Fit with learn_Psi=True ----
    model.fit(
        data=data,
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 150},
    )

    assert model.opt_result.success

    # Optimization should not make the NLL worse
    assert model.nll_hat <= nll0 + 1e-6

    # Extract learned Psi from the cache
    Psi_hat = model.cache["Psi"]

    # Shape (p, q)
    assert Psi_hat.shape == (p, q)

    # Each column should be unit norm due to normalize_cols=True
    col_norms = np.linalg.norm(Psi_hat, axis=0)
    assert np.allclose(col_norms, np.ones(q), atol=1e-6)

    # Smoke test: predict runs and shapes match
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == Y.shape
    assert std.shape == Y.shape

    _, psihat, _ = unpack_theta(model.theta_hat, d, q, p, learn_Psi=True, learn_sigma_eps=False)
    assert np.allclose(psihat, Psi_hat)
