# Re-run with careful vectorization and no external deps beyond numpy/matplotlib.
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, erf

# Domain and basis
a, b = -1.0, 1.0

erf_vec = np.vectorize(erf)

def I0(mu, sigma):
    mu = np.asarray(mu)
    s = sqrt(2.0) * sigma
    return np.sqrt(np.pi/2.0) * sigma * (erf_vec((b-mu)/s) - erf_vec((a-mu)/s))

def I1(mu, sigma):
    mu = np.asarray(mu)
    A = I0(mu, sigma)
    eb = np.exp(-0.5*((b-mu)/sigma)**2)
    ea = np.exp(-0.5*((a-mu)/sigma)**2)
    return mu * A - (sigma**2) * (eb - ea)

def I2(mu, sigma):
    mu = np.asarray(mu)
    A = I0(mu, sigma)
    eb = np.exp(-0.5*((b-mu)/sigma)**2)
    ea = np.exp(-0.5*((a-mu)/sigma)**2)
    tb = (b - mu)
    ta = (a - mu)
    return (mu**2 + sigma**2) * A - 2*mu*(sigma**2)*(eb - ea) - (sigma**2)*(tb*eb - ta*ea)

def h_vec(x, ell):
    return np.array([I0(x, ell), I1(x, ell), I2(x, ell)])  # (3,)

# Outer 1D integration by Gauss-Legendre (exact inner integrals)
import numpy.polynomial.legendre as LG

def gauss_legendre_integrate(f, a=-1.0, b=1.0, n=200):
    x, w = LG.leggauss(n)
    xm = 0.5*(b+a)
    xr = 0.5*(b-a)
    xp = xm + xr*x
    return np.sum(w * f(xp)) * xr

def H_matrix(ell):
    def F(mu):
        return np.array([I0(mu, ell), I1(mu, ell), I2(mu, ell)])  # (3, len(mu)) if mu array
    H = np.zeros((3,3))
    # We'll integrate component-wise
    for i in range(3):
        for j in range(3):
            if j == 0:
                integrand = lambda mu: (F(mu)[i] * (1.0))
            elif j == 1:
                integrand = lambda mu: (F(mu)[i] * mu)
            else:
                integrand = lambda mu: (F(mu)[i] * (mu**2))
            H[i,j] = gauss_legendre_integrate(integrand, a, b, n=200)
    return H

def c_star(x, xp, ell):
    k = np.exp(-0.5*((x - xp)/ell)**2)
    hx = h_vec(x, ell)
    hxp = h_vec(xp, ell)
    H = H_matrix(ell)
    H_reg = H + 1e-12 * np.eye(3)
    return k - hx @ np.linalg.solve(H_reg, hxp)

# Data
rng = np.random.default_rng(2)
n = 60
X = np.linspace(-1, 1, n)
y_true = np.sin(2*X)
sigma_noise = 0.05
y = y_true + sigma_noise * rng.standard_normal(n)
G = np.vstack([np.ones_like(X), X, X**2]).T

def build_Cstar(X, ell, noise_var):
    n = len(X)
    # Precompute h(X) for efficiency
    HX = np.stack([h_vec(xi, ell) for xi in X], axis=0)  # (n,3)
    H = H_matrix(ell)
    H_reg = H + 1e-12 * np.eye(3)
    invH_HT = np.linalg.solve(H_reg, HX.T)              # (3,n)
    # Base kernel
    XX = X[:,None] - X[None,:]
    K = np.exp(-0.5*(XX/ell)**2)
    # c_* = K - HX @ invH @ HX^T
    C = K - HX @ invH_HT
    C += noise_var * np.eye(n)
    return C, HX, invH_HT, H_reg

def profiled_nll(X, y, G, ell, noise_var):
    C, HX, invH_HT, H_reg = build_Cstar(X, ell, noise_var)
    CiG = np.linalg.solve(C, G)
    GT_Ci_G = G.T @ CiG
    beta_hat = np.linalg.solve(GT_Ci_G, G.T @ np.linalg.solve(C, y))
    r = y - G @ beta_hat
    sign, logdet = np.linalg.slogdet(C)
    if sign <= 0:
        return np.inf, beta_hat, (C, HX, invH_HT, H_reg)
    quad = r.T @ np.linalg.solve(C, r)
    return 0.5 * (logdet + quad), beta_hat, (C, HX, invH_HT, H_reg)

# Grid search over ell
ell_grid = np.linspace(0.08, 1.2, 35)
best = (np.inf, None, None, None)
noise_var = sigma_noise**2
cache = None
for ell in ell_grid:
    nll, beta_hat, cache = profiled_nll(X, y, G, ell, noise_var)
    if nll < best[0]:
        best = (nll, ell, beta_hat, cache)
nll_best, ell_best, beta_hat_best, cache_best = best

C_best, HX_best, invH_HT_best, H_reg_best = cache_best

def predictor(X_train, y, G, ell, noise_var, beta_hat, HX, invH_HT, H_reg):
    # reuse already-built pieces
    # alpha = C^{-1}(y - G beta)
    # Solve with the already computed C
    C, _, _, _ = build_Cstar(X_train, ell, noise_var)
    alpha = np.linalg.solve(C, y - G @ beta_hat)

    def cstar_row(x):
        kx = np.exp(-0.5*((x - X_train)/ell)**2)                 # (n,)
        hx = h_vec(x, ell)                                       # (3,)
        return kx - hx @ invH_HT                                 # (n,)

    def predict(xstar):
        xstar = np.atleast_1d(xstar)
        Gs = np.vstack([np.ones_like(xstar), xstar, xstar**2]).T
        mean_spline = Gs @ beta_hat
        Crows = np.vstack([cstar_row(x) for x in xstar])
        mean_gp = Crows @ alpha
        return mean_spline + mean_gp

    return predict

pred = predictor(X, y, G, ell_best, noise_var, beta_hat_best, HX_best, invH_HT_best, H_reg_best)

Xs = np.linspace(-1,1,400)
ys_true = np.sin(2*Xs)
ys_pred = pred(Xs)

plt.figure(figsize=(7,4))
plt.plot(Xs, ys_true, label="true sin(2x)")
plt.plot(Xs, ys_pred, label="OGP (exact inner integrals)")
plt.scatter(X, y, s=12, label="observations")
plt.title(f"Exact c* via Eq (3.2)-(3.3) structure, ell*={ell_best:.3f}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.show()
