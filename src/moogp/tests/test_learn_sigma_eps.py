import numpy as np
from moogp.datasets import generate_forrester_data
from moogp.model import MOOGP, unpack_theta

def test_learn_sigma_eps_fixed_psi():

    true_sigma_eps = np.array([10.0, 1.0, 0.05], dtype=float)

    data = generate_forrester_data(n=60, seed=67, with_error=True, error_per_output=true_sigma_eps)
    X = data["X_scaled"]
    Y = data["y"]
    n, d = X.shape
    p = Y.shape[1]

    # Mean basis: intercept + main effect
    terms = [None] + list(range(1, d + 1))

    # Choose q = p 
    q = 3
    Psinot = np.identity(3)

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
        standardize_x=False,
        standardize_y=False,
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
    # Set initial value of sigma_eps2 based on scaled version of per-output variance
    sigma_eps2_init = np.log(1e-2 * y_var)
    theta0 = np.concatenate([theta_latent, sigma_eps2_init])

    # Bounds for sigma_eps2
    lb = np.maximum(1e-12, 1e-6 * y_var)
    ub = np.maximum(lb * 10.0, 0.5 * y_var) 

    log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]
    bounds.extend(log_bounds)
    bounds = list(bounds)

    # ---- NLL at initial theta0 (for comparison) ----
    nll0 = model._nll(theta0)

    # fit model with learn_sigma_eps2 true
    model.fit(
        data=data,
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 150},
    )

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
    init_sigma_eps2 = np.exp(sigma_eps2_init)
    assert abs(sigma_eps2_hat[2] - true_sigma_eps[2]) < abs(init_sigma_eps2[2] - true_sigma_eps[2])

    mean, std = model.predict(X, return_std=True)
    assert mean.shape == Y.shape
    assert std.shape == Y.shape
    
    _, _, sigmahat = unpack_theta(model.theta_hat, d, q, p, learn_Psi=False, learn_sigma_eps=True)
    assert np.allclose(sigma_eps2_hat, sigmahat)
