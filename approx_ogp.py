# OGP demo: y(x) = sin(2x) on [-1, 1], polynomial mean m(x) = [1, x, x^2], Gaussian kernel.
# We'll approximate the OGP integrals with a simple trapezoidal quadrature on [-1, 1].
# Then do profiled MLE over a small grid of lengthscales, compute GLS beta, and plot predictions.

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ----- Data -----
n = 60
X = np.linspace(-1, 1, n)
true_f = np.sin(2 * X)
sigma_noise = 0.05
y = true_f + sigma_noise * rng.standard_normal(n)

# ----- Mean basis m(x) = [1, x, x^2] -----
def M_basis(x):
    x = np.asarray(x)
    return np.vstack([np.ones_like(x), x, x**2]).T  # shape (len(x), 3)

G = M_basis(X)  # (n, r), r=3

# ----- Base Gaussian kernel (variance fixed to 1 for simplicity) -----
def K_se(Xa, Xb, ell):
    Xa = np.asarray(Xa)[:, None]
    Xb = np.asarray(Xb)[None, :]
    d2 = (Xa - Xb) ** 2
    return np.exp(-0.5 * d2 / (ell**2))

# ----- Quadrature setup on [-1, 1] -----
m = 400
Xq = np.linspace(-1, 1, m)
dx = Xq[1] - Xq[0]
# trapezoid weights
W = np.ones(m) * dx
W[0] *= 0.5
W[-1] *= 0.5
W = np.diag(W)

Mq = M_basis(Xq)  # (m, r)

def ogp_components(X_train, ell):
    # base covariances
    Kxx = K_se(X_train, X_train, ell)  # (n, n)
    Kq  = K_se(Xq, Xq, ell)            # (m, m)
    Kxq = K_se(X_train, Xq, ell)       # (n, m)
    # h(x) ≈ ∫ k(x, ξ) m(ξ) dξ ≈ K(x, Xq) W Mq -> shape (n, r)
    Hx = Kxq @ (W @ Mq)
    # H ≈ Mq^T W Kq W Mq -> (r, r)
    H = Mq.T @ (W @ (Kq @ (W @ Mq)))
    return Kxx, Hx, H

def C_star(X_train, ell, nugget=1e-8):
    Kxx, Hx, H = ogp_components(X_train, ell)
    # Regularize H a touch for numerical stability
    H_reg = H + 1e-10 * np.eye(H.shape[0])
    Cstar = Kxx - Hx @ np.linalg.solve(H_reg, Hx.T)
    # add small nugget (and observation noise)
    return Cstar + nugget * np.eye(len(X_train))

def profiled_nll(X_train, y, G, ell, noise_var):
    # Build C_* and add noise variance to diagonal
    Cstar = C_star(X_train, ell) + noise_var * np.eye(len(X_train))
    # GLS beta hat
    CiG = np.linalg.solve(Cstar, G)
    GT_Ci_G = G.T @ CiG
    beta_hat = np.linalg.solve(GT_Ci_G, G.T @ np.linalg.solve(Cstar, y))
    r = y - G @ beta_hat
    # Profiled NLL up to constants (variance profiled out)
    # Here we include logdet and Mahalanobis with scale; noise_var already included
    sign, logdet = np.linalg.slogdet(Cstar)
    if sign <= 0:
        return np.inf, beta_hat
    quad = r.T @ np.linalg.solve(Cstar, r)
    return 0.5 * (logdet + quad), beta_hat

# ----- Grid search over lengthscale (and fix noise variance to known sigma^2) -----
ell_grid = np.linspace(0.05, 1.2, 40)
best = (np.inf, None, None)
noise_var = sigma_noise**2  # here we assume known; could also profile it
for ell in ell_grid:
    nll, beta_hat = profiled_nll(X, y, G, ell, noise_var)
    if nll < best[0]:
        best = (nll, ell, beta_hat)

nll_best, ell_best, beta_hat_best = best

# ----- Build predictor using OGP kernel -----
def make_predictor(X_train, y, G, ell, noise_var, beta_hat):
    Cstar = C_star(X_train, ell) + noise_var * np.eye(len(X_train))
    alpha = np.linalg.solve(Cstar, (y - G @ beta_hat))

    # For new x*, need c_*(x*, X)
    Kxx, Hx, H = ogp_components(X_train, ell)
    H_reg = H + 1e-10 * np.eye(H.shape[0])

    # helper to compute h(x*) ≈ ∫ k(x*, ξ) m(ξ) dξ
    def h_star(xstar):
        kxq = K_se(np.atleast_1d(xstar), Xq, ell)  # (1, m)
        return (kxq @ (W @ Mq)).reshape(-1)  # (r,)

    def cstar_row(xstar):
        # c_*(x*, X) = k(x*, X) - h(x*) H^{-1} Hx^T
        kxX = K_se(np.atleast_1d(xstar), X_train, ell)  # (1, n)
        hx = h_star(xstar)  # (r,)
        return (kxX - hx @ np.linalg.solve(H_reg, Hx.T)).ravel()  # (n,)

    def predict(xstar):
        xstar = np.atleast_1d(xstar)
        Gstar = M_basis(xstar)  # (m, r)
        mean_spline = Gstar @ beta_hat
        # build c_*(x*, X) rows
        Crows = np.vstack([cstar_row(x) for x in xstar])  # (m, n)
        mean_gp = Crows @ alpha
        return mean_spline + mean_gp

    return predict

predict = make_predictor(X, y, G, ell_best, noise_var, beta_hat_best)

# ----- Evaluate on dense grid and plot -----
Xs = np.linspace(-1, 1, 400)
ys_true = np.sin(2 * Xs)
ys_pred = predict(Xs)

plt.figure(figsize=(7,4))
plt.plot(Xs, ys_true, label="true sin(2x)")
plt.plot(Xs, ys_pred, label="OGP mean")
plt.scatter(X, y, s=12, label="observations")
plt.title(f"OGP fit with polynomial mean [1, x, x^2], ell*={ell_best:.3f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
