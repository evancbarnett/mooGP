"""
sigma_eps_demo.py

Sanity checks to confirm that diagonal measurement noise Sigma_eps is being used as intended:

Model:
    vec(Y) | B  ~  N( (I_p ⊗ G) vec(B),  K_y )
    K_y = K_f + (Sigma_eps ⊗ I_n)
    K_f = sum_{k=1}^q (psi_k psi_k^T) ⊗ C_k

This script:
  1) Generates a small synthetic multi-output dataset with known noise variances.
  2) Fits MOOGP with (a) sigma_eps2 = 0 and (b) sigma_eps2 = true values.
  3) Verifies (numerically) that the model's cached Ky equals the manual formula.
  4) Demonstrates how sigma_eps affects training-point interpolation and predictive std.


python -m moogp.sigma_eps_demo
"""

from __future__ import annotations

import numpy as np
from moogp.model import MOOGP


def scale_to_unit_box(X: np.ndarray) -> np.ndarray:
    """
    Convenience: map X in [0,1]^d to X_scaled in [-1,1]^d.
    """
    return 2.0 * (X - 0.5)


def true_functions(X: np.ndarray) -> np.ndarray:
    """
    Define a smooth multi-output ground truth.
    X is (n,1) in [0,1]
    Returns F_true of shape (n,p)
    """
    x = X[:, 0]
    f1 = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    f2 = 0.7 * np.cos(8 * x) + 0.2 * f1
    f3 = 0.4 * np.sin(2 * np.pi * x) - 0.15 * f1
    return np.vstack([f1, f2, f3]).T


def make_theta0_and_bounds(
    q: int,
    p: int,
    d: int,
    learn_psi: bool = False,
    learn_sigma_eps2: bool = False,
    y_var=None,                 # <-- new (optional): length-p variances of each Y output
    eps_init_frac: float = 1e-3,
    eps_lb_frac: float = 1e-8,
    eps_ub_frac: float = 0.5,
):
    """
    theta layout per latent k:
        [log(sigma2_k), log(ell_{k1}), ..., log(ell_{kd})]
    if learn_psi:
        append Psi_free (p*q entries, unconstrained)
    if learn_sigma_eps2:
        append log(sigma_eps2_j) for j=1..p

    Notes:
      - If Y is NOT standardized, pass y_var = Var(Y, axis=0) (shape (p,)).
      - Bounds for sigma_eps2_j are [eps_lb_frac * y_var[j], eps_ub_frac * y_var[j]].
      - Initial sigma_eps2_j is eps_init_frac * y_var[j].
    """
    theta0 = []
    bounds = []

    # latent hyperparameter starting values and bounds
    for _ in range(q):
        theta0.append(np.log(1.0))               # log sigma2
        theta0.extend([np.log(0.4)] * d)         # log ell
        bounds.append((np.log(1e-6), np.log(1e3)))
        bounds.extend([(np.log(1e-3), np.log(10.0))] * d)

    if learn_psi:
        rng = np.random.default_rng(0)
        Psi0_free = rng.standard_normal((p, q))

        theta0 = np.concatenate([np.array(theta0, float), Psi0_free.ravel()])
        bounds.extend([(-5.0, 5.0)] * (p * q))

    if learn_sigma_eps2:
        # Use per-output variance scaling when Y isn't standardized
        if y_var is None:
            y_var = np.ones(p, dtype=float)  # fallback
        else:
            y_var = np.asarray(y_var, dtype=float)
            if y_var.shape != (p,):
                raise ValueError(f"y_var must have shape (p,), got {y_var.shape}")

        # initial values
        sigma0 = eps_init_frac * y_var

        # bounds: [eps_lb_frac * var, eps_ub_frac * var]
        lb = np.maximum(1e-12, eps_lb_frac * y_var)
        ub = np.maximum(lb * 10.0, eps_ub_frac * y_var)  # ensure ub > lb

        log_sigma0 = np.log(np.maximum(sigma0, lb))
        log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]

        if isinstance(theta0, list):
            theta0.extend(log_sigma0.tolist())
        else:
            theta0 = np.concatenate([theta0, log_sigma0])

        bounds.extend(log_bounds)

    return np.array(theta0, float), bounds



def manual_Ky_from_cache(cache: dict, sigma_eps2: np.ndarray, jitter: float) -> np.ndarray:
    """
    Build Ky manually:
        Ky = sum_k kron(psi_k psi_k^T, Ck) + kron(diag(sigma_eps2), I_n) + jitter I
    using cached Cj_list and Psi.
    """
    Cj_list = cache["Cj_list"]
    Psi = cache["Psi"]
    n = Cj_list[0].shape[0]
    p, q = Psi.shape

    Ky = np.zeros((n * p, n * p))
    for k in range(q):
        Wk = np.outer(Psi[:, k], Psi[:, k])
        Ky += np.kron(Wk, Cj_list[k])

    sigma_eps2 = np.asarray(sigma_eps2, float).ravel()
    Ky += np.kron(np.diag(sigma_eps2), np.eye(n))

    if jitter and jitter > 0:
        Ky += jitter * np.eye(n * p)
    return Ky


def fit_model(X_scaled: np.ndarray, Y: np.ndarray, *, sigma_eps2, learn_sigma_eps: bool, jitter: float):
    n, d = X_scaled.shape
    p = Y.shape[1]

    # Mean basis: intercept + main effect(s)
    terms = [None] + list(range(1, d + 1))

    # Choose q = p and Psi = I so each latent maps to one output (simplest sanity check)
    q = p
    Psi = np.eye(p)

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=learn_sigma_eps,
        jitter=jitter,
        one_based=True,
        normalize_cols=True,
    )

    y_var = Y.var(axis=0, ddof=1)  
    theta0, bounds = make_theta0_and_bounds(
                                    q=q, 
                                    p=p, 
                                    d=d,
                                    learn_psi=False,
                                    learn_sigma_eps2=learn_sigma_eps,
                                    y_var=y_var)

    data = {"X_scaled": X_scaled, "Y": Y}
    model.fit(
        data=data,
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 250, "disp": False},
    )
    return model


def main(seed: int = 0):
    rng = np.random.default_rng(seed)

    # Synthetic design in [0,1]
    n = 100
    X = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    X_scaled = scale_to_unit_box(X)

    # Generate multi-output truth and add diagonal measurement noise
    F_true = true_functions(X)
    p = F_true.shape[1]

    # Pick clearly different noise levels to make effects obvious
    sigma_eps2_true = np.array([1e-4, 2e-2, 8e-2], float)  # variances per output
    E = rng.normal(0.0, np.sqrt(sigma_eps2_true), size=F_true.shape)
    Y = F_true + E

    jitter = 1e-8

    print("\n=== True noise variances sigma_eps2_true ===")
    print(sigma_eps2_true)

    # Fit model with no measurement noise (sigma_eps2 = 0)
    print("\n=== Fit A: sigma_eps2 = 0 (interpolating / near-interpolating) ===")
    model0 = fit_model(X_scaled, Y, sigma_eps2=np.zeros(p), learn_sigma_eps=False, jitter=jitter)

    # Fit model with known measurement noise
    print("\n=== Fit B: sigma_eps2 = true values (noisy observations) ===")
    modelN = fit_model(X_scaled, Y, sigma_eps2=sigma_eps2_true, learn_sigma_eps=False, jitter=jitter)

    # ---------------------------------------------------------------------
    # 1) Check Ky matches the manual construction
    # ---------------------------------------------------------------------
    for label, model, sig in [("A", model0, np.zeros(p)), ("B", modelN, sigma_eps2_true)]:
        Ky_cache = model.cache["Ky"]
        Ky_manual = manual_Ky_from_cache(model.cache, sig, jitter=model.jitter)
        max_abs = np.max(np.abs(Ky_cache - Ky_manual))
        rel = max_abs / max(1.0, np.max(np.abs(Ky_manual)))
        print(f"\n[Ky check {label}] max|Ky_cache - Ky_manual| = {max_abs:.3e} (rel {rel:.3e})")
        # This should be ~1e-10 to 1e-7 depending on jitter/conditioning
        if rel > 1e-6:
            print("  WARNING: Ky mismatch is larger than expected — check Sigma_eps ⊗ I_n implementation.")

        # Spot-check diagonal inflation by sigma_eps2
        n = X_scaled.shape[0]
        Ky_no_noise = Ky_manual - np.kron(np.diag(sig), np.eye(n))
        inflated = np.diag(Ky_cache - Ky_no_noise)
        # inflated should be (sigma_eps2 repeated n times)
        target = np.tile(sig, n)
        max_diag_err = np.max(np.abs(inflated - target))
        print(f"[Ky diag check {label}] max|diag(Ky - (Ky-no-noise)) - tile(sigma_eps2)| = {max_diag_err:.3e}")

    # ---------------------------------------------------------------------
    # 2) Compare predictions at training points
    # ---------------------------------------------------------------------
    print("\n=== Prediction sanity checks at training inputs (Xstar = X) ===")

    # Predict the observed output y = f + eps (default predict_observation=True)
    mean0_y, std0_y = model0.predict(X_scaled, return_std=True, predict_observation=True)
    meanN_y, stdN_y = modelN.predict(X_scaled, return_std=True, predict_observation=True)

    # Predict latent function f only (predict_observation=False)
    mean0_f, std0_f = model0.predict(X_scaled, return_std=True, predict_observation=False)
    meanN_f, stdN_f = modelN.predict(X_scaled, return_std=True, predict_observation=False)

    # Training residual norms
    rmse0 = np.sqrt(np.mean((mean0_y - Y) ** 2, axis=0))
    rmseN = np.sqrt(np.mean((meanN_y - Y) ** 2, axis=0))
    print("RMSE at training points (predicting y):")
    print("  Fit A (sigma_eps2=0):", rmse0)
    print("  Fit B (sigma_eps2=true):", rmseN)
    print("Expected: Fit A should be much closer to 0 (near-interpolation). Fit B should not interpolate.\n")

    # Check std behavior at training points
    print("Median predictive std at training points:")
    print("  Fit A std(y):", np.median(std0_y, axis=0))
    print("  Fit B std(y):", np.median(stdN_y, axis=0))
    print("  Fit B std(f):", np.median(stdN_f, axis=0))
    print("Expected: For Fit B, std(y) should be noticeably larger than std(f),")
    print("          and outputs with larger sigma_eps2 should have larger std(y).\n")

    # ---------------------------------------------------------------------
    # 3) Optional: learning sigma_eps2 (if your model supports it cleanly)
    # ---------------------------------------------------------------------
    print("=== Optional: learn sigma_eps2 (can be slow / sometimes ill-conditioned) ===")
    try:
        modelL = fit_model(X_scaled, Y, sigma_eps2=None, learn_sigma_eps=True, jitter=jitter)
        # After fitting, sigma_eps2 is stored in the cache used for predictions.
        # If you expose it as an attribute, you can print it here; otherwise we show posterior std at training points.
        meanL_y, stdL_y = modelL.predict(X_scaled, return_std=True, predict_observation=True)
        print("Learned model fitted. Median std(y) at training points:", np.median(stdL_y, axis=0))
        print("Compare with true sqrt(sigma_eps2_true):", np.sqrt(sigma_eps2_true))
        print("Note: Identifiability with latent sigma2 can make estimates imperfect; use as a smoke test.")
    except Exception as e:
        print("Learning sigma_eps2 path raised an exception (this is OK if you haven't finished that branch):")
        print(" ", repr(e))


if __name__ == "__main__":
    main()
