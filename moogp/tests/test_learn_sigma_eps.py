# moogp/tests/test_learn_sigma_eps.py
import numpy as np
import matplotlib.pyplot as plt
from moogp.datasets import generate_forrester_data
from moogp.model import MOOGP, unpack_theta
from moogp.forrester_illustration import *

def test_learn_sigma_eps_fixed_psi():

    true_sigma_eps = [10, 1, 0.05]

    data = generate_forrester_data(n=100, seed = 67, with_error=True, error_per_output=true_sigma_eps)
    X = data["X_scaled"]
    Y = data["y"]
    n, d = X.shape
    p = Y.shape[1]

    # Mean basis: intercept + main effect
    terms = [None] + list(range(1, d + 1))

    # Choose q = p 
    q = 3
    Psinot = np.identity(3)

    # Model with learn_Psi=True. Psi=None here: it will be learned from theta.
    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psinot,
        learn_Psi=False,
        learn_sigma_eps=True,
        use_reml=False,
        jitter=None,
        one_based=True,
        normalize_cols=True,
    )


    model._prepare_data(data)

    # ---- Build theta0: [latent params | sigma_eps2 ] ----
    theta_latent = []
    bounds = []

    # For each latent j: [log_sigma2_j, log_ell_j1, ..., log_ell_jd]
    for _ in range(q):
        theta_latent.append(np.log(1.0))              # log sigma^2
        theta_latent.extend([np.log(0.5)] * d)        # log lengthscales
        bounds.append((np.log(1e-3), np.log(1e3)))    # sigma^2 bounds
        bounds.extend([(np.log(0.05), np.log(5.0))] * d)  # ell bounds

    theta_latent = np.array(theta_latent)

    # Calculate output variance of Y (Y is noisy in this case)
    y_var = Y.var(axis=0, ddof=1)  
    print(y_var)

    # Set initial value of sigma_eps2 based on scaled version of per-output variance
    sigma_eps2_init = np.log(1e-2 * y_var)
    print(f"Sigma_eps2 log init: {sigma_eps2_init}")
    theta0 = np.concatenate([theta_latent, sigma_eps2_init])

    # Bounds for sigma_eps2
    lb = np.maximum(1e-12, 1e-6 * y_var)
    ub = np.maximum(lb * 10.0, 0.5 * y_var) 

    log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]
    bounds.extend(log_bounds)
    print(bounds)
    bounds = list(bounds)

    # ---- NLL at initial theta0 (for comparison) ----
    nll0 = model._nll(theta0)

    # fit model with learn_sigma_eps2 true
    model.fit(
        data=data,
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 500},
    )

    assert model.opt_result.success

    # Optimization should not make the NLL worse
    assert model.nll_hat <= nll0 + 1e-6

    # Extract learned Psi from the cache
    Psi_hat = model.cache["Psi"]

    # Shape (p, q)
    assert Psi_hat.shape == (p, q)
    assert np.allclose(Psi_hat, Psinot, 1e-8)

    # Each column should be unit norm due to normalize_cols=True
    col_norms = np.linalg.norm(Psi_hat, axis=0)
    assert np.allclose(col_norms, np.ones(q), atol=1e-6)

    # Get sigma from cache
    sigma_eps2_hat = model.cache['sigma_eps2']

    # Check all learned noise parameters are positive
    assert (sigma_eps2_hat > 0).all()

    diff = true_sigma_eps - sigma_eps2_hat
    print(diff)

    mean, std = model.predict(X, return_std=True)
    assert mean.shape == Y.shape
    assert std.shape == Y.shape
    
    thetanot, _, sigmanot = unpack_theta(theta0, d,q,p,learn_Psi=False, learn_sigma_eps=True)
    thetahat, _, sigmahat = unpack_theta(model.theta_hat, d,q,p,learn_Psi=False, learn_sigma_eps=True)
    
    print(f"True sigma_eps2: {true_sigma_eps}")
    print(f"Starting sigma_eps2: {sigmanot}")
    print(f"Fitted sigma_eps2: {sigma_eps2_hat}")

    print("\n ")
    print(f"unpacked matches cached?: {np.allclose(sigma_eps2_hat, sigmahat)}")
    print("\n ")

    print(f"Starting param values: {thetanot}")
    print(f"Fitted param values: {thetahat}")

    plot_forrester_fit(model, data['X'], X, Y, n_plot=200)
    plot_trend_vs_ls(
        model, data['X'], X, Y,
        title_suffix=""
    )
    plt.show()

