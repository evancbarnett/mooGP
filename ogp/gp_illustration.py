import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Gaussian process figures with notation matching the poster:
#
#   f(\cdot) ~ GP(m(\cdot), c(\cdot,\cdot))
#   f(x) | \mathbf{f} ~ N(\hat{\mu}(x), \hat{\sigma}^2(x))
# ============================================================

rng = np.random.default_rng(7)

# -----------------------------
# Mean and covariance functions
# -----------------------------
def m(x):
    """Mean function m(x)."""
    x = np.asarray(x)
    return np.zeros_like(x)

def c(x1, x2, ell=1.2, sigma2=1.0):
    """Covariance function c(x, x')."""
    x1 = np.atleast_1d(x1)[:, None]
    x2 = np.atleast_1d(x2)[None, :]
    sqdist = (x1 - x2) ** 2
    return sigma2 * np.exp(-0.5 * sqdist / ell**2)

def gp_posterior(X, f_obs, x_star, ell=1.2, sigma2=1.0, noise_var=1e-8):
    """
    Posterior for f(x_star) given observed values \mathbf{f} at X.
    """
    C = c(X, X, ell=ell, sigma2=sigma2) + noise_var * np.eye(len(X))
    c_star = c(X, x_star, ell=ell, sigma2=sigma2)
    C_star = c(x_star, x_star, ell=ell, sigma2=sigma2) + 1e-10 * np.eye(len(x_star))

    m_X = m(X)
    m_star = m(x_star)

    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, f_obs - m_X))

    mu_hat = m_star + c_star.T @ alpha
    v = np.linalg.solve(L, c_star)
    Sigma_hat = C_star - v.T @ v
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)

    return mu_hat, Sigma_hat

# -----------------------------
# Plot style for poster column
# -----------------------------
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 1.3,
})

x = np.linspace(0, 10, 350)

# ============================================================
# Figure 1: GP prior draws
# ============================================================
C_prior = c(x, x, ell=1.15, sigma2=1.15) + 1e-8 * np.eye(len(x))
prior_draws = rng.multivariate_normal(mean=m(x), cov=C_prior, size=5)

fig1, ax1 = plt.subplots(figsize=(3.5, 2.35), dpi=300)

for draw in prior_draws:
    ax1.plot(x, draw, linestyle="--", alpha=0.75)

ax1.set_xlim(0, 10)
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")
ax1.set_title(
    "Gaussian Process Prior\n"
    + r"$f(\cdot)\sim \mathsf{GP} \: \left(m(\cdot),\,c(\cdot,\cdot)\right)$",
    pad=6
)
ax1.grid(False)

fig1.tight_layout()
fig1.savefig("ogp/figs/gp_prior_leftcol.pdf", bbox_inches="tight")
fig1.savefig("ogp/figs/gp_prior_leftcol.png", bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2: GP posterior prediction
# ============================================================
def f_true(x):
    """Smooth function used to generate example observations."""
    return (
        0.95 * np.sin(0.62 * x - 0.15)
        + 0.22 * np.cos(1.22 * x + 0.45)
        - 0.04 * (x - 8.0)
    )

X = np.array([1.0, 1.6, 2.2, 3.0, 5.1, 7.8])
f_obs = f_true(X)

mu_hat, Sigma_hat = gp_posterior(X, f_obs, x, ell=1.3, sigma2=1.0, noise_var=1e-8)
sigma_hat = np.sqrt(np.clip(np.diag(Sigma_hat), 0, None))

posterior_draws = rng.multivariate_normal(
    mean=mu_hat,
    cov=Sigma_hat + 1e-10 * np.eye(len(x)),
    size=4
)

fig2, ax2 = plt.subplots(figsize=(3.5, 2.35), dpi=300)

ax2.fill_between(
    x,
    mu_hat - 2 * sigma_hat,
    mu_hat + 2 * sigma_hat,
    alpha=0.16
)

for draw in posterior_draws:
    ax2.plot(x, draw, linestyle="--", alpha=0.7)

ax2.plot(x, mu_hat, color="0.25", linewidth=2.0)
ax2.scatter(X, f_obs, color="red", s=20, zorder=5)

ax2.set_xlim(0, 10)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")
ax2.set_title(
    "Posterior Prediction\n"
    + r"$f(x)\mid \mathbf{f}\sim N \left(\hat{\mu}(x),\,\hat{\sigma}^2(x)\right)$",
    pad=6
)
ax2.grid(False)

fig2.tight_layout()
fig2.savefig("ogp/figs/gp_posterior_leftcol.pdf", bbox_inches="tight")
fig2.savefig("ogp/figs/gp_posterior_leftcol.png", bbox_inches="tight")
plt.show()

# ============================================================
# Combined stacked figure for the left column
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(3.5, 6), dpi=300)

# Top: prior
for draw in prior_draws:
    axes[0].plot(x, draw, linestyle="--", alpha=0.75)
axes[0].set_xlim(0, 10)
axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$f(x)$")
axes[0].set_title(
    "Gaussian Process Prior\n"
    + r"$f(\cdot)\sim \mathsf{GP} \: \left(m(\cdot),\,c(\cdot,\cdot)\right)$",
    pad=5
)
axes[0].grid(False)

# Bottom: posterior
axes[1].fill_between(
    x,
    mu_hat - 2 * sigma_hat,
    mu_hat + 2 * sigma_hat,
    alpha=0.16
)
for draw in posterior_draws:
    axes[1].plot(x, draw, linestyle="--", alpha=0.7)
axes[1].plot(x, mu_hat, color="0.25", linewidth=2.0)
axes[1].scatter(X, f_obs, color="red", s=20, zorder=5)
axes[1].set_xlim(0, 10)
axes[1].set_xlabel(r"$x$")
axes[1].set_ylabel(r"$f(x)$")
axes[1].set_title(
    "Posterior Prediction\n"
    + r"$f(x)\mid \mathbf{f}\sim N \left(\hat{\mu}(x),\,\hat{\sigma}^2(x)\right)$",
    pad=5
)
axes[1].grid(False)

fig.tight_layout(h_pad=1.0)
fig.savefig("ogp/figs/gp_left_column_combined.pdf", bbox_inches="tight")
fig.savefig("ogp/figs/gp_left_column_combined.png", bbox_inches="tight")
plt.show()