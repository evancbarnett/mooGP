import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp
from scipy.special import erf
from scipy.stats import qmc
from scipy.linalg import cho_factor, cho_solve, cholesky, solve, solve_triangular
from scipy.optimize import minimize

# Algorithm design

# 1. Generate Data
# - 1a. Pick true function y(x) = ...
# - 1b. Determine number of inputs, d,
# - 1c. Simulate data
# - 1d. Scale X to [-1, 1]^d


# 2. Model Structure
# - 2a. Determine structure of g(x), i.e, choose p
# - 2b. Calculate structure of h(x) -- determine what each h_i will be
# - 2c. Same for H(x)
# - 2d. Calculate Y, G, \beta, C

# 3. Training 
# - 3a. Calculate individual parts C*\inv, \logdet C, etc.
# - 3b. Assemble nll
# - 3c. Optimization LBFGS


def generate_data(n, seed=67):
    # Borehole function: 8 x_j variables
    BOUNDS = np.array([
        [0.05, 0.15],
        [100.0, 5_000.0],
        [63_070., 115_600.0],
        [990.0, 1_110.0],
        [63.1, 116.0],
        [700.0, 820.0],
        [1_120.0, 1680.0],
        [9_855.0, 12_045.0],
    ])

    def borehole_y(X_phys):
        x1, x2, x3, x4, x5, x6, x7, x8 = X_phys.T
        log_term = np.log(x2 / x1)
        denom = log_term * (1.0 + (2.0 * x7 * x3) / (log_term * x1**2 * x8) + (x3 / x5))
        return (2.0 * np.pi * x3 * (x4 - x6)) / denom
    


    def lhs_in_bounds(n, bounds=BOUNDS, seed=seed):
        d = bounds.shape[0]
        sampler = qmc.LatinHypercube(d=d, scramble=True, rng=seed)
        U = sampler.random(n)                                  # in [0,1]^d
        return qmc.scale(U, bounds[:, 0], bounds[:, 1]) 
    
    def to_m1_1(X_phys, bounds=BOUNDS):
        a, b = bounds[:, 0], bounds[:, 1]
        return 2.0 * (X_phys - a) / (b - a) - 1.0

    
    X_phys = lhs_in_bounds(n, seed=seed)
    y = borehole_y(X_phys)
    X_scaled = to_m1_1(X_phys)
    out = {"X_phys": X_phys, "X_scaled": X_scaled, "y": y}
    
    return out

# Create Design Matrix
def make_G(data, terms, one_based=True, return_names=False):
    # Creates n x p design matrix of the form

    X = np.asarray(data['X_scaled'])
    n, d = X.shape
    
    cols = []
    names = []

    def add_col(name, col):
            cols.append(col.reshape(n, 1))
            names.append(name)

    for t in terms:
        # Intercept
        if t is None:
            add_col('1', np.ones(n))
            continue

        # Single Effect
        if isinstance(t, (int, np.integer)):
            j = t - 1 if one_based else t
            if j < 0 or j >= d:
                raise ValueError(f"Index {t} out of range for d={d} (one_based={one_based}).")
            add_col(f'x{j+1}', X[:, j])
            continue

        # Interaction
        else:
            try:
                idxs = [(i - 1) if one_based else int(i) for i in t]
            except Exception as e:
                raise ValueError(f"Unrecognized term: {t}") from e

        if len(idxs) < 2:
            raise ValueError(f"Interactions must have length >= 2; got {t}")
        if any((i < 0 or i >= d) for i in idxs):
            raise ValueError(f"Some indices in {t} are out of range for d={d}.")


        prod = np.prod(X[:, idxs], axis=1)
        name = '*'.join([f'x{i+1}' for i in idxs])
        add_col(name, prod)

    G = np.hstack(cols) if cols else np.empty((n, 0))
    return (G, names) if return_names else G


# Covariance analytical integrals
def M_gauss(x, psi, sigma2=1.0):
    x = np.asarray(x)
    return (sqrt(pi) * sigma2 * psi / 2.0) * (erf((x + 1.0)/psi) - erf((x - 1.0)/psi))

def L_gauss(x, psi, sigma2=1.0):
    x = np.asarray(x)
    return (sigma2 * (psi**2) / 2.0) * (exp(-((x + 1.0)/psi)**2) - exp(-((x - 1.0)/psi)**2)) + x * M_gauss(x, psi, sigma2)

def IM_gauss(psi, sigma2=1.0):
    return 2.0 * sqrt(pi) * sigma2 * psi * erf(2.0/psi) - sigma2 * (psi**2) * (1.0 - exp(-4.0/(psi**2)))

def IL_gauss(psi, sigma2=1.0):
    return 0.0

def ILL_gauss(psi, sigma2=1.0):
    return (sigma2 * (psi**4) / 6.0) * (1.0 - exp(-4.0/(psi**2))) \
         - (sigma2 * (psi**2) / 3.0) * (3.0 - exp(-4.0/(psi**2))) \
         + (2.0 * sqrt(pi) * sigma2 * psi / 3.0) * erf(2.0/psi)


def make_c_star_matrix(X, Xp, psi, sigma2, terms, one_based=True):
    """
    Compute C*(X, X') with Eq (2.3) c* = c - h^T H \inv h' (matrix version)
    Inputs:
      X, Xp : (n,d), (m,d) in [-1,1]^d
      psi   : (d,) lengthscales
      sigma2: variance (scale)
      terms : list defining g(x) structure (same format as in make_G)
    """
    def _parse_terms_to_sets(terms, d, one_based=True):
        """
        Map each term to its index set J_i ⊆ {0,...,d-1}.
        None  -> empty set
        int j -> {j-1} (if one_based) or {j} (if not)
        list of ints -> set of indices
        """
        J_sets = []
        for t in terms:
            if t is None:
                J_sets.append(tuple())                  # intercept = empty set
                continue
            if isinstance(t, (int, np.integer)):        # single term = int
                j = t-1 if one_based else t
                if not (0 <= j < d):
                    raise ValueError(f'Index {t} out of range for d={d}')
                J_sets.append((j,))
                continue
            else:                                       # list = interaction
                idxs = [(i-1) if one_based else int(i) for i in t]

            if any(i < 0 or i >= d for i in idxs):
                raise ValueError(f'Indices {idxs} out of range for d={d}')
            if len(set(idxs)) != len(idxs):
                raise ValueError(f'Duplicate indices in interaction {t}')
            if len(idxs) < 2:
                raise ValueError(f'Interactions must have length ≥2; got {t}')
            J_sets.append(tuple(sorted(idxs)))
        return J_sets  # list of tuples
    
    def _base_se_kernel(X, Xp, psi, sigma2=1.0):
        """Separable squared-exp with lengthscales psi (shape: (d,))."""

        X = np.asarray(X)
        Xp = np.asarray(Xp)
        psi = np.asarray(psi)
        dif = (X[:, None, :] - Xp[None, :, :]) / psi  # (n, m, d)
        D2 = np.sum(dif * dif, axis=2)

        return sigma2 * np.exp(-D2)
    

    def _h_matrix(X, psi, sigma2, J_sets):
        """
        Build h(X) matrix (shape: n x p)
        Each column is h(x_i), i=1,...,n where is i the data index (x_i \in \R^d)
        See formulas (2.2) and (3.3)
        """
        X = np.asarray(X)
        n, d = X.shape
        p = len(J_sets)

        # Precompute M_j and L_j for all coords (shape: n×d)
        M_all = np.empty((n, d))
        L_all = np.empty((n, d))
        for j in range(d):
            M_all[:, j] = M_gauss(X[:, j], psi[j], sigma2=1.0)
            L_all[:, j] = L_gauss(X[:, j], psi[j], sigma2=1.0)

        Hcols = np.empty((n, p))
        all_idx = set(range(d))
        for i, Ji in enumerate(J_sets):
            Ji = set(Ji)
            notJ = list(all_idx - Ji)
            Ji   = list(Ji)
            col = np.ones(n)
            if notJ:
                col *= np.prod(M_all[:, notJ], axis=1)
            if Ji:
                col *= np.prod(L_all[:, Ji], axis=1)
            Hcols[:, i] = col
        return Hcols * sigma2  # n×p
    
    def _H_diag(psi, sigma2, J_sets):
        """
        Build H(X) matrix. See formula (2.2) & (3.3)
        """
        psi = np.asarray(psi)
        d = len(psi)
        IM = np.array([IM_gauss(psi[j], sigma2=1.0)  for j in range(d)])
        ILL= np.array([ILL_gauss(psi[j], sigma2=1.0) for j in range(d)])

        Hdiag = []
        all_idx = set(range(d))
        for Ji in J_sets:
            Ji = set(Ji)
            notJ = list(all_idx - Ji)
            Ji   = list(Ji)
            val = 1.0
            if notJ:
                val *= np.prod(IM[notJ])
            if Ji:
                val *= np.prod(ILL[Ji])
            Hdiag.append(val)
        Hdiag = np.asarray(Hdiag)
        if np.any(Hdiag <= 0):
            raise RuntimeError("Non-positive H diagonal (check psi/domain).")
        return Hdiag * sigma2  # shape (p,)
    
    
    X  = np.asarray(X)  
    Xp = np.asarray(Xp)
    d  = X.shape[1]
    J_sets = _parse_terms_to_sets(terms, d, one_based=one_based)

    # Base covariance
    C = _base_se_kernel(X, Xp, psi, sigma2=sigma2)

    if len(J_sets) == 0:
        return C 

    # h(X), h(X')
    hX  = _h_matrix(X,  psi, sigma2, J_sets)     # n×p
    hXp = _h_matrix(Xp, psi, sigma2, J_sets)     # m×p

    # H diagonal and H^{-1}
    Hdiag = _H_diag(psi, sigma2, J_sets)         # (p,)
    Hinvd = 1.0 / Hdiag                          # (p,)

    # c* = c - h(X) diag(H^{-1}) h(X')^T
    C_corr = (hX * Hinvd) @ hXp.T          # n×m
    return C - C_corr

def make_beta(y, G, C_chol, lower=True):
    """
    \hat{\beta} = (G^T C_*^{-1} G)^{-1} G^T C_*^{-1} y
    See Eq (2.4)
    Parameters
    ----------
    y    : (n,)
    G    : (n,p)
    chol : Cholesky factor of C_*
    lower: whether `chol` is lower-triangular
    """

    if G.size == 0:  # no mean terms
        return np.zeros(0, dtype=float)

    Cinv = lambda v: cho_solve((C_chol, lower), v, check_finite=False)
    GT_Cinv_G = G.T @ Cinv(G)
    GT_Cinv_y = G.T @ Cinv(y)
    beta_hat  = solve(GT_Cinv_G, GT_Cinv_y, assume_a='sym')
    return beta_hat


def nll_ogp(theta_raw, data, terms, use_reml=True, jitter=1e-8, one_based=True):
    """
    Negative log-likelihood that optimizes over (sigma^2, psi).

    Parameters
    ----------
    theta_raw : array-like, shape (1 + d,)
        Packed as [log_sigma2, log_psi_1, ..., log_psi_d] (unconstrained).
    data : dict
        Contains 'X_scaled' (n,d) in [-1,1]^d and 'y' (n,).
    terms : list
        Structure of g(x): None for intercept, ints for main effects, lists/tuples for interactions.
    use_reml : bool
        If True, add the REML correction (+ 0.5 log|G^T C^{-1} G|) and profile with n-p dof.
        If False, standard ML (no correction).
    jitter : float
        Small diagonal added to C for numerical stability.
    """
    X = np.asarray(data['X_scaled'])
    y = np.asarray(data['y'])
    n, d = X.shape

    # Build design matrix
    G = make_G(data, terms, one_based=one_based, return_names=False)
    p = G.shape[1] if G.size else 0

    # Unpack parameters and enforce positivity via exp
    theta_raw = np.asarray(theta_raw).ravel()
    if theta_raw.size != 1 + d:
        raise ValueError(f"theta_raw must have length {1+d} = 1 + d; got {theta_raw.size}.")
    log_sigma2, log_psi = theta_raw[0], theta_raw[1:]
    sigma2 = float(np.exp(log_sigma2))
    psi    = np.exp(log_psi)

    # Full covariance: C_*(psi, sigma2)
    C = make_c_star_matrix(X, X, psi=psi, sigma2=sigma2, terms=terms, one_based=one_based)
    if jitter:
        C = C + jitter * np.eye(n)

    # Cholesky and log|C|
    L, lower = cho_factor(C, lower=True, check_finite=False)
    logdetC = 2.0 * np.sum(np.log(np.diag(L)))
    Cinv = lambda v: cho_solve((L, lower), v, check_finite=False)

    # beta and residuals
    beta_hat = make_beta(y, G, C_chol=L, lower=lower) if p else np.zeros(0)
    r = y - (G @ beta_hat if p else 0.0)

    # Quadratic form
    qf = float(r.T @ Cinv(r))

    # Full (R)ML objective with (sigma^2, psi)
    # ML:   0.5*(log|C| + r^T C^{-1} r + n log 2π)
    # REML: + 0.5*log|G^T C^{-1} G|
    nll = 0.5 * (logdetC + qf + n * np.log(2.0 * np.pi))
    if use_reml and p:
        GT_Cinv_G = G.T @ Cinv(G)
        nll += 0.5 * np.linalg.slogdet(GT_Cinv_G)[1]

    return nll


def fit_ogp(data, terms, theta0, bounds, use_reml=False, jitter=1e-8, one_based=True, maxiter=200):
    """
    Fits OGP by ML over (sigma^2, psi). Returns a fitted-model dict.
    """
    X = np.asarray(data['X_scaled'])
    y = np.asarray(data['y'])
    n, d = X.shape

    # Optimize NLL (your nll_ogp)
    obj = lambda th: nll_ogp(th, data, terms, use_reml=use_reml, jitter=jitter, one_based=one_based)
    res = minimize(obj, theta0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    theta_hat = np.asarray(res.x).ravel()
    sigma2 = float(np.exp(theta_hat[0]))
    psi    = np.exp(theta_hat[1:])

    # Build final training covariance and its Cholesky
    C = make_c_star_matrix(X, X, psi=psi, sigma2=sigma2, terms=terms, one_based=one_based)
    if jitter:
        C = C + jitter * np.eye(n)

    L, lower = cho_factor(C, lower=True, check_finite=False)

    # Design matrix and GLS beta
    G = make_G(data, terms, one_based=one_based, return_names=False)
    p = G.shape[1] if G.size else 0
    if p:
        Cinv = lambda v: cho_solve((L, lower), v, check_finite=False)
        GT_Cinv = G.T @ Cinv(G)
        GT_Cinv_y = G.T @ Cinv(y)
        beta_hat = solve(GT_Cinv, GT_Cinv_y)
        r = y - G @ beta_hat
        alpha = Cinv(r)  # C^{-1}(y - G beta)
        CinvG = Cinv(G)
        A = GT_Cinv            # = G^T C^{-1} G
        LA = cholesky(A, lower=True)
    else:
        beta_hat = np.zeros(0)
        Cinv = lambda v: cho_solve((L, lower), v, check_finite=False)
        alpha = Cinv(y)
        CinvG = None
        LA = None

    return {
        "success": res.success,
        "message": res.message,
        "theta_hat": theta_hat,
        "sigma2": sigma2,
        "psi": psi,
        "jitter": jitter,
        "X": X,
        "y": y,
        "terms": terms,
        "one_based": one_based,
        "G": G,
        "beta_hat": beta_hat,
        "L": L,
        "lower": lower,
        "alpha": alpha,
        "CinvG": CinvG,
        "LA": LA,        # Cholesky of A = G^T C^{-1} G (None if p=0)
        "res": res,
    }

def predict_ogp(model, X_new_scaled, return_std=True):
    """
    Predict at new scaled points Z = X_new_scaled (in [-1,1]^d).
    Returns mean and variance (and std if return_std).
    """
    X = model["X"] 
    y = model["y"]
    psi = model["psi"]
    sigma2 = model["sigma2"]
    terms = model["terms"]
    one_based = model["one_based"]
    G = model["G"]
    beta_hat = model["beta_hat"]
    L = model["L"]
    lower = model["lower"]
    alpha = model["alpha"]
    CinvG = model["CinvG"]
    LA = model["LA"]

    Z = np.asarray(X_new_scaled)
    m = Z.shape[0]

    # Cross-covariances and self-covariances
    C_s  = make_c_star_matrix(Z, X, psi=psi, sigma2=sigma2, terms=terms, one_based=one_based)   # (m,n)
    C_ss = make_c_star_matrix(Z, Z, psi=psi, sigma2=sigma2, terms=terms, one_based=one_based)   # (m,m)

    # Mean basis for new points
    G_new = make_G({"X_scaled": Z}, terms, one_based=one_based, return_names=False)

    # Predictive mean: mu = G_* \beta + C_s \alpha
    mu = (G_new @ beta_hat if beta_hat.size else 0.0) + C_s @ alpha

    # Predictive variance: diag(C_ss - C_s C^{-1} C_s^T + (B A^{-1} B^T)),
    # where B = G_* - C_s C^{-1} G and A = G^T C^{-1} G
    # Compute V = L^{-1} C_s^T  => C^{-1} C_s^T = (L^{-T} L^{-1}) C_s^T
    V = solve_triangular(L, C_s.T, lower=True, check_finite=False)  # (n,m)
    base_var = np.clip(np.diag(C_ss) - np.sum(V*V, axis=0), 0.0, np.inf)

    if beta_hat.size:
        # B = G_* - C_s C^{-1} G
        CinvG = cho_solve((L, lower), G, check_finite=False)  # (n,p)
        B = G_new - C_s @ CinvG                               # (m,p)
        # add rowwise quadratic form with A^{-1} via LA
        T = solve_triangular(LA, B.T, lower=True, check_finite=False)  # (p,m)
        adj = np.sum(T*T, axis=0)
        var = base_var + adj
    else:
        var = base_var

    if return_std:
        std = np.sqrt(np.maximum(var, 0.0))
        return mu, var, std
    return mu, var





if __name__ == "__main__":
    # 1) Train set
    data_tr = generate_data(n=80, seed=123)
    d = data_tr['X_scaled'].shape[1]
    terms = [None] + list(range(1, d+1))  # intercept + main effects

    theta0 = np.r_[np.log(1.0), np.log(0.5)*np.ones(d)]
    bounds = [(np.log(1e-6), np.log(1e3))] + [(np.log(0.05), np.log(5.0))]*d

    model = fit_ogp(data_tr, terms, theta0, bounds, use_reml=False, jitter=1e-8, one_based=True, maxiter=300)

    # 2) Test set (fresh LHS)
    data_test = generate_data(n=200, seed=67)
    Z = data_test["X_scaled"]
    y_true = data_test["y"]

    # 3) Predict
    y_pred, v_pred, s_pred = predict_ogp(model, Z, return_std=True)

    # 4) Metrics & plot
    rmspe = float(np.sqrt(np.mean(((y_pred - y_true)/y_true)**2))) * 100
    r2 = float(1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

    print(f"Fitted sigma^2 = {model['sigma2']:.6g}")
    print(f"Fitted psi     = {model['psi']}")
    print(f"RMSE (test)    = {rmspe:.4g}   |   R^2 = {r2:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=20, alpha=0.7)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title("OGP: True vs Predicted")
    plt.tight_layout()
    plt.show()

