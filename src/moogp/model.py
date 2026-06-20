import autograd.numpy as np
import numpy as onp  # real numpy for non-traced theta0/bounds construction

from autograd.scipy.linalg import solve_triangular as ag_solve_triangular
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.linalg.lapack import dpotri
from scipy.optimize import minimize

from .design import make_G, build_Gy, vecF, unvecF, parse_terms_to_index_sets
from .kernels import (
    make_c_star_matrix,
    make_c_star_diag,
    se_kernel_matrix,
    M_gauss,
    L_gauss,
    IM_gauss,
    ILL_gauss,
    M_dlogell_gauss,
    L_dlogell_gauss,
    IM_dlogell_gauss,
    ILL_dlogell_gauss,
)
from autograd import value_and_grad

def normalize_diag_error_structure(diag_error_structure, p):
    """Validate and normalize a diagonal error grouping spec.

    A grouping specifies how the ``p`` outputs are partitioned into blocks that
    each share a single ``sigma^2`` parameter, giving
    ``Sigma_eps = bdiag(sigma_1^2 I_{p1}, ..., sigma_G^2 I_{pG})``.

    ``None`` is interpreted as ``[1] * p`` (no grouping; one parameter per
    output).
    """
    if diag_error_structure is None:
        return tuple([1] * int(p))
    sizes = tuple(int(s) for s in diag_error_structure)
    if any(s <= 0 for s in sizes):
        raise ValueError(
            f"diag_error_structure entries must be positive integers; got {diag_error_structure}"
        )
    if sum(sizes) != int(p):
        raise ValueError(
            f"sum(diag_error_structure)={sum(sizes)} must equal p={p}; got {diag_error_structure}"
        )
    return sizes


def group_indices_for_outputs(diag_error_structure):
    """Per-output integer index into a length-G group vector.

    Output ``l`` belongs to group ``group_indices_for_outputs(es)[l]``.  The
    returned array can be used as ``values_per_group[idx]`` to broadcast a
    grouped vector to length ``p`` in an autograd-friendly way (autograd's
    ``np.repeat`` does not accept an array-valued ``repeats`` argument).
    """
    sizes = np.asarray(diag_error_structure, dtype=int)
    out = np.empty(int(sizes.sum()), dtype=int)
    col = 0
    for k, sz in enumerate(sizes):
        out[col:col + int(sz)] = k
        col += int(sz)
    return out


def expand_grouped_sigma_eps2(values_per_group, diag_error_structure):
    """Expand a length-G vector of group variances to a length-p per-output vector.

    ``values_per_group[k]`` is broadcast across ``diag_error_structure[k]`` outputs,
    matching the LCGP ``built_lsigma2s`` construction.
    """
    values_per_group = np.asarray(values_per_group)
    if values_per_group.size != len(diag_error_structure):
        raise ValueError(
            f"values_per_group length {values_per_group.size} != number of groups "
            f"{len(diag_error_structure)}"
        )
    idx = group_indices_for_outputs(diag_error_structure)
    return values_per_group[idx]


def aggregate_per_output_to_groups(per_output_values, diag_error_structure):
    """Sum a length-p per-output vector into a length-G group vector.

    Used when assembling gradients with respect to the grouped log-variance
    parameters: each group's gradient is the sum of per-output gradients over
    the outputs in that group.
    """
    per_output_values = np.asarray(per_output_values, dtype=float).ravel()
    if per_output_values.size != int(np.sum(diag_error_structure)):
        raise ValueError(
            f"per_output_values length {per_output_values.size} does not match "
            f"sum(diag_error_structure)={int(np.sum(diag_error_structure))}"
        )
    out = np.zeros(len(diag_error_structure), dtype=float)
    col = 0
    for k, sz in enumerate(diag_error_structure):
        out[k] = float(np.sum(per_output_values[col:col + sz]))
        col += sz
    return out


def unpack_theta(
    theta_raw,
    d,
    q,
    p,
    *,
    learn_Psi=False,
    learn_sigma_eps=False,
    normalize_cols=True,
    diag_error_structure=None,
):
    """
    Unpack the unconstrained optimization vector

    r: number of elements in g(x)
    q: number of gaussian processes in g(x)
    d: length of g(x)
    p: number of output locations in data

    Packing order:
      1) Latent kernel hyperparams for each k=1..q:
           [log(sigma2_k), log(ell_{k1}), ..., log(ell_{kd})]
         => total length q*(d+1)
      2) Optional Psi (if ``learn_Psi=True``): p*q free entries
      3) Optional diagonal measurement noise variances (if ``learn_sigma_eps=True``):
           [log(sigma^2_{eps,1}), ..., log(sigma^2_{eps,G})]
         where G = len(diag_error_structure).  When ``diag_error_structure``
         is ``None`` the default grouping ``[1] * p`` recovers one log-variance
         per output (G == p).  The returned ``sigma_eps2`` is always a
         length-p vector with entries broadcast across each group.

    """
     # flatten array to 1d
    t = theta_raw
    base = q * (d + 1)  # number of latent-kernel hyperparameters
    if t.size < base:
        raise ValueError(f"theta length {t.size} < required latent {base}")

    lat = t[:base]
    lat_params = []
    off = 0
    for j in range(q):
        log_s2 = lat[off]
        off += 1
        log_ell = lat[off:off+d]
        off += d
        lat_params.append((np.exp(log_s2), np.exp(log_ell)))
    off = base

    Psi = None
    if learn_Psi:
        need = p * q
        if t.size < off + need:
            raise ValueError(f"theta length {t.size} < required latent+Psi {off+need}")
        Psi_free = t[off:off + need].reshape(p, q)
        off += need

        # Keep this autograd-friendly when theta_raw carries ArrayBox values.
        Psi = 1.0 * Psi_free
        if normalize_cols:
            nrms = np.linalg.norm(Psi, axis=0)
            nrms = np.where(nrms == 0, 1.0, nrms) # prevent divide by zero
            Psi = Psi / nrms

    sigma_eps2 = None
    if learn_sigma_eps:
        err_struct = normalize_diag_error_structure(diag_error_structure, p)
        n_groups = len(err_struct)
        need = n_groups
        if t.size < off + need:
            raise ValueError(f"theta length {t.size} < required +Sigma_eps {off+need}")
        log_sigma = t[off:off + need]
        off += need
        sigma_grouped = np.exp(log_sigma)
        # Broadcast each group's variance across its outputs via fancy
        # indexing. Autograd traces through ``arr[idx]`` cleanly, while
        # ``np.repeat`` with an array ``repeats`` argument does not.
        idx = group_indices_for_outputs(err_struct)
        sigma_eps2 = sigma_grouped[idx]

    return lat_params, Psi, sigma_eps2

def build_Ky(Cjs, Psi, sigma_eps2 = None):
    n = Cjs[0].shape[0]
    p,q = Psi.shape

    assert q == len(Cjs)

    # build covariance without measurement noise
    Ky = np.zeros((n * p, n * p))
    for j in range(q):
        Wj = np.outer(Psi[:,j], Psi[:,j])
        Ky += np.kron(Wj, Cjs[j])

    # add diagonal measurement noise 
    if sigma_eps2 is not None:
        sigma_eps2 = np.asarray(sigma_eps2, float).ravel()
        if sigma_eps2.size != p:
            raise ValueError(f"sigma_eps2 length {sigma_eps2.size} needs to equal p={p}")
        Ky += np.kron(np.diag(sigma_eps2), np.eye(n))
    return Ky

def build_cross_K(Psi, Cj_list):
    # train-test covariance matrix
    nstar, m = Cj_list[0].shape
    p,q = Psi.shape
    K = np.zeros((nstar*p, m*p))
    for j in range(q):
        Wj = np.outer(Psi[:,j], Psi[:,j])  # (p×p)
        K += np.kron(Wj, Cj_list[j])
    return K


def gls_bhat(Y, G, Ky_chol):
    """Generalized least-squares estimate of the trend coefficients."""
    Y = np.asarray(Y, float)
    G = np.asarray(G, float)

    if Y.ndim != 2 or G.ndim != 2:
        raise ValueError("Y and G must both be rank-2 arrays.")
    if Y.shape[0] != G.shape[0]:
        raise ValueError(f"Y and G must share the same number of rows; got {Y.shape[0]} and {G.shape[0]}.")

    p = Y.shape[1]
    r = G.shape[1]
    Gy = build_Gy(G, p)
    vecY = vecF(Y)

    z = cho_solve(Ky_chol, Gy, check_finite=False)
    alpha = cho_solve(Ky_chol, vecY, check_finite=False)

    A_gls = Gy.T @ z
    b_gls = Gy.T @ alpha
    beta_vec = np.linalg.solve(A_gls, b_gls)
    return unvecF(beta_vec, r, p)

def init_phi(Y, q, n):
    """
    Build ``Phi`` and ``d`` from the SVD of ``Y``.
    """
    Y = np.asarray(Y, float)
    
    # SVD of Y returns U: (n,n), S: (min(n,p)), Vt: (p,p)
    _, svals, Vt = np.linalg.svd(Y, full_matrices=False)
    
    V = Vt.T 
    s_q = svals[:q]

    # Prevent divide by zero
    s_q = np.maximum(svals[:q], 1e-12)
    
    # Phi is (p, q)
    Phi = V[:, :q] * np.sqrt(n) / s_q[None, :]
    d = n / (s_q ** 2)
    return Phi, d


def _normalize_standardize_y_mode(standardize_y):
    """Normalize user-facing output-standardization options."""
    if standardize_y in (False, None):
        return False
    if standardize_y is True:
        return "zscore"
    if isinstance(standardize_y, str):
        mode = standardize_y.lower()
        if mode in {"zscore", "robust"}:
            return mode
    raise ValueError("standardize_y must be one of False, True, 'zscore', or 'robust'.")


def _compute_y_center_scale(Y, mode):
    """Return per-output center and spread for the working response scale."""
    Y = np.asarray(Y, float)
    p = Y.shape[1]

    if mode is False:
        return np.zeros(p, dtype=float), np.ones(p, dtype=float)

    if mode == "zscore":
        center = np.mean(Y, axis=0)
        scale = np.std(Y, axis=0, ddof=1)
    elif mode == "robust":
        center = np.median(Y, axis=0)
        scale = np.median(np.abs(Y - center[None, :]), axis=0)
    else:  # pragma: no cover - guarded by _normalize_standardize_y_mode
        raise ValueError(f"Unsupported standardize_y mode: {mode}")

    scale = np.where(scale > 1e-12, scale, 1.0)
    return np.asarray(center, float), np.asarray(scale, float)


def compute_working_y(Y, standardize_y):
    """Project raw outputs onto the model's internal working scale."""
    mode = _normalize_standardize_y_mode(standardize_y)
    center, scale = _compute_y_center_scale(Y, mode)
    Y_work = (np.asarray(Y, float) - center[None, :]) / scale[None, :]
    return Y_work, center, scale


def _normalize_standardize_x_mode(standardize_x):
    """Normalize user-facing input-standardization options."""
    if standardize_x in (False, None):
        return False
    if standardize_x is True:
        return "unitcube"
    if isinstance(standardize_x, str):
        mode = standardize_x.lower()
        if mode in {"unitcube"}:
            return mode
    raise ValueError("standardize_x must be one of False, True, or 'unitcube'.")


def _compute_x_center_scale(X, mode, margin=0.0):
    """Return per-dim center / half-width for the model's internal X scale.

    ``margin`` is a non-negative multiplicative padding applied in unit-cube
    mode: the working half-width becomes ``(1 + margin) * (x_max - x_min) / 2``,
    so training X lands in ``[-1/(1+margin), 1/(1+margin)]`` instead of exactly
    on ``[-1, 1]``. This leaves headroom for test points whose coordinates
    fall slightly outside the training range and keeps gradient evaluations
    away from the corners of the cube where the kernel has zero curvature.
    """
    X = np.asarray(X, float)
    d = X.shape[1]

    if mode is False:
        return np.zeros(d, dtype=float), np.ones(d, dtype=float)

    if mode == "unitcube":
        margin = float(margin)
        if margin < 0.0:
            raise ValueError(f"x margin must be non-negative; got {margin}.")
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        center = 0.5 * (x_min + x_max)
        half_width = 0.5 * (x_max - x_min) * (1.0 + margin)
        # Constant columns can't be rescaled to [-1, 1]; leave them at the
        # center (X_work = 0) by setting scale = 1.
        half_width = np.where(half_width > 1e-12, half_width, 1.0)
        return center, half_width

    # pragma: no cover - guarded by _normalize_standardize_x_mode
    raise ValueError(f"Unsupported standardize_x mode: {mode}")


def compute_working_x(X, standardize_x, margin=0.0):
    """Project raw inputs onto the model's internal working scale."""
    mode = _normalize_standardize_x_mode(standardize_x)
    center, scale = _compute_x_center_scale(X, mode, margin=margin)
    X_work = (np.asarray(X, float) - center[None, :]) / scale[None, :]
    return X_work, center, scale


def _solve_spd_from_cholesky(chol, rhs):
    """Solve ``A x = rhs`` given a lower-triangular Cholesky factor of ``A``."""
    y = ag_solve_triangular(chol, rhs, lower=True)
    return ag_solve_triangular(chol.T, y, lower=False)


def _apply_factorized_qk(chol, d_val, rhs):
    """Apply ``Q = (I + d C)^{-1} C`` to ``rhs`` using the factor of ``I + d C``."""
    solved_rhs = _solve_spd_from_cholesky(chol, rhs)
    return (rhs - solved_rhs) / d_val


def _block_design_t_matmul(G, rhs, p):
    """Compute ``(I_p ⊗ G)^T rhs`` without forming the Kronecker product."""
    was_vec = rhs.ndim == 1
    rhs2 = rhs[:, None] if was_vec else rhs

    n, _ = G.shape
    if rhs2.shape[0] != n * p:
        raise ValueError(f"rhs has incompatible leading dimension {rhs2.shape[0]} (expected {n * p}).")

    out = np.vstack([G.T @ rhs2[j * n:(j + 1) * n, :] for j in range(p)])
    return out[:, 0] if was_vec else out


def _block_design_matvec(G, beta_vec, p):
    """Compute ``(I_p ⊗ G) beta_vec`` without forming the Kronecker product."""
    bhat = unvecF(beta_vec, G.shape[1], p)
    return vecF(G @ bhat)


def _profiled_gls_terms(solve_Ky, G, Gy, vecY, p, *, build_cache):
    """Profile out the trend coefficients using exact GLS identities."""
    if Gy.size == 0:
        alpha = solve_Ky(vecY)
        qf = np.dot(vecY, alpha)
        if build_cache:
            return qf, np.zeros((0, p)), vecY, alpha
        return qf, None, None, None

    solved_rhs = solve_Ky(np.concatenate((Gy, vecY[:, None]), axis=1))
    z = solved_rhs[:, :-1]
    alpha = solved_rhs[:, -1]

    A_gls = _block_design_t_matmul(G, z, p)
    b_gls = _block_design_t_matmul(G, alpha, p)
    beta_vec = np.linalg.solve(A_gls, b_gls)
    qf = np.dot(vecY, alpha) - np.dot(b_gls, beta_vec)

    if not build_cache:
        return qf, None, None, None

    bhat = unvecF(beta_vec, G.shape[1], p)
    rvec = vecY - _block_design_matvec(G, beta_vec, p)
    Ky_inv_rvec = alpha - z @ beta_vec
    return qf, bhat, rvec, Ky_inv_rvec


def _profiled_gls_terms_fast(fast_diag_info, G, vecY, alpha_mat, p, *, build_cache):
    """Fast (Kronecker-structured) profiled GLS terms.

    Equivalent to ``_profiled_gls_terms`` when ``solve_Ky`` is the diagonalized
    Woodbury form, but exploits ``G_y = I_p ⊗ G`` so we only apply
    ``Q_k = (I + d_k C_k)^{-1} C_k`` to the (n×r) design ``G``, never to the
    full (n p × r p) Kronecker block.

    Parameters
    ----------
    fast_diag_info : dict
        Output of the fast-path covariance build (``sigma_eps2``,
        ``latent_factors``, ``n``, ``p``).
    G : (n, r) array
    vecY : (n p,) array
    alpha_mat : (n, p) array
        Already-applied ``unvec(K_y^{-1} vec(Y))``.
    p : int
    build_cache : bool

    Returns
    -------
    qf, bhat, rvec, Ky_inv_rvec
    """
    r = G.shape[1] if G.size else 0
    sigma_eps2 = fast_diag_info["sigma_eps2"]
    latent_factors = fast_diag_info["latent_factors"]

    alpha = vecF(alpha_mat)

    if r == 0:
        qf = np.dot(vecY, alpha)
        if build_cache:
            return qf, np.zeros((0, p)), vecY, alpha
        return qf, None, None, None

    # b_gls = G_y^T α = vec(G^T α_mat)
    b_gls = vecF(G.T @ alpha_mat)

    # W_k = Q_k G   (n × r),     T_k = G^T W_k = G^T Q_k G   (r × r)
    W_list = []
    T_list = []
    for f in latent_factors:
        W_k = _apply_factorized_qk(f["chol"], f["d"], G)
        W_list.append(W_k)
        T_list.append(G.T @ W_k)

    GTG = G.T @ G

    # A_gls = Σ_eps^{-1} ⊗ G^T G  -  Σ_k (ψ̃_k ψ̃_k^T) ⊗ T_k
    A_gls = np.kron(np.diag(1.0 / sigma_eps2), GTG)
    for k, T_k in enumerate(T_list):
        tk = latent_factors[k]["psi"]
        A_gls = A_gls - np.kron(np.outer(tk, tk), T_k)

    beta_vec = np.linalg.solve(A_gls, b_gls)
    qf = np.dot(vecY, alpha) - np.dot(b_gls, beta_vec)

    if not build_cache:
        return qf, None, None, None

    bhat = unvecF(beta_vec, r, p)
    GB = G @ bhat  # (n, p)
    rvec = vecY - vecF(GB)

    # K_y^{-1} (G_y β) = vec( G B / σ_eps² - Σ_k W_k (B ψ̃_k) ψ̃_k^T )
    KyinvGbeta_mat = GB / sigma_eps2[None, :]
    for k, W_k in enumerate(W_list):
        tk = latent_factors[k]["psi"]
        Btk = bhat @ tk  # (r,)
        Wk_Btk = W_k @ Btk  # (n,)
        KyinvGbeta_mat = KyinvGbeta_mat - np.outer(Wk_Btk, tk)

    Ky_inv_rvec = alpha - vecF(KyinvGbeta_mat)
    return qf, bhat, rvec, Ky_inv_rvec


def _predict_variance_diag_fast(
    fast_diag_info,
    Cj_XsX,
    psi,
    *,
    Cj_diag_star_list,
    predict_observation,
):
    """Diagonal of the predictive covariance via the Woodbury-Kronecker form.

    Computes ``diag(K_** - K_*X K_y^{-1} K_X*)`` (plus measurement noise if
    requested) without ever forming the dense ``K_*X`` Kronecker block.

    Returns the same column-major (F-order) layout as the dense path:
    index ``l * n_star + m`` for output ``l`` and test point ``m``.

    Parameters
    ----------
    fast_diag_info : dict
        ``sigma_eps2``, ``latent_factors`` (each with ``chol``, ``d``, ``psi``).
    Cj_XsX : list of (n_star, n) arrays
        Per-latent test-train kernel matrices (orthogonalized).
    psi : (p, q) array
        Working-scale ψ_j = Σ_eps^{1/2} φ_j (the cached ``Psi``).
    Cj_diag_star_list : list of (n_star,) arrays
        Per-latent diagonals of ``C_j(X_*, X_*)``.
    predict_observation : bool
        Whether to add ``Σ_eps`` (working-scale) for predictive of an
        *observation* rather than the mean function.
    """
    sigma_eps2 = np.asarray(fast_diag_info["sigma_eps2"], float).ravel()
    latent_factors = fast_diag_info["latent_factors"]
    n_star = Cj_XsX[0].shape[0]
    p, q = psi.shape

    diag = np.zeros(n_star * p)
    for j in range(q):
        psi_sq = psi[:, j] ** 2  # (p,)
        diag = diag + np.kron(psi_sq, Cj_diag_star_list[j])

        chol = latent_factors[j]["chol"]
        d_j = latent_factors[j]["d"]
        # diag(C_j(X_*, X) A_j^{-1} C_j(X, X_*))
        #   = diag( (L^{-1} C_j(X, X_*))^T (L^{-1} C_j(X, X_*)) )
        # so a single lower-triangular solve plus a column-wise sum-of-squares
        # replaces the previous solve(L^T, solve(L, C^T)) + einsum(C, S).
        v = solve_triangular(chol, Cj_XsX[j].T, lower=True, check_finite=False)
        diag_AjCj = np.einsum("nm,nm->m", v, v)
        diag = diag - d_j * np.kron(psi_sq, diag_AjCj)

    if predict_observation:
        diag = diag + np.repeat(sigma_eps2, n_star)
    return diag


def _latent_kernel_logtheta_grad(
    M_k,
    X,
    sqdist,
    ell,
    sigma2,
    terms,
    *,
    orthogonal,
    one_based,
):
    """Closed-form ``d/dlog(theta_k) tr(M_k * C*_k(theta_k))`` for one latent.

    Returns a (1 + d,) array
    ``[d/dlog(sigma2), d/dlog(ell_1), ..., d/dlog(ell_d)]`` matching the
    packing used by ``_nll_and_grad_fast`` for each latent block.

    Mathematics
    -----------
    ``C*_k = sigma2 * (c_0(X,X;ell) - h_0(X;ell) Hd_0(ell)^{-1} h_0(X;ell)^T)``
    is linear in ``sigma2``, so

        d/dlog(sigma2) tr(M_k C*_k) = tr(M_k C*_k).

    For each coordinate ``c``,

        d/dlog(ell_c) c_k = (2/ell_c^2) * sqdist[:,:,c] * c_k,
        d/dlog(ell_c) (h_0 Hd_0^{-1} h_0^T) =
            replace M(x_c, ell_c) with ell_c * dM/dell in the per-term product
            (resp. L) for h_0; replace IM, ILL similarly for Hd_0.

    Only ``M`` and ``L`` along the c-th coordinate, and ``IM``/``ILL`` at
    ``ell_c``, depend on ``ell_c`` because the per-coordinate factors are
    multiplicative and independent across coordinates.
    """
    n, d = X.shape
    sigma2 = float(sigma2)
    ell = np.asarray(ell, dtype=float).reshape(-1)

    # Bare SE kernel and its log-ell gradient traces.
    c_se_k = se_kernel_matrix(X, X, ell, sigma2=sigma2, sqdist=sqdist)
    M_cse = M_k * c_se_k  # (n, n)
    # tr_se[c] = sum_{i,j} M_k[i,j] * c_se[i,j] * sqdist[i,j,c]
    tr_se = np.einsum("ij,ijc->c", M_cse, sqdist)
    grad_logell_se = (2.0 / (ell * ell)) * tr_se  # (d,)

    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based)
    do_correction = bool(orthogonal) and (len(J_sets) > 0)

    if not do_correction:
        # C*_k = c_se_k.  d/dlog(sigma2) tr(M_k C*_k) = tr(M_k c_se_k).
        grad_logsig = float(np.sum(M_cse))
        return np.concatenate([[grad_logsig], grad_logell_se])

    # Per-coordinate building blocks (sigma2 = 1 inside the M, L, IM, ILL
    # factories — sigma2 enters as a single linear multiplier on h, Hd).
    ell_row = ell.reshape(1, d)
    M_all = M_gauss(X, ell_row, sigma2=1.0)        # (n, d)
    L_all = L_gauss(X, ell_row, sigma2=1.0)        # (n, d)
    dM_all = M_dlogell_gauss(X, ell_row, sigma2=1.0)  # (n, d)
    dL_all = L_dlogell_gauss(X, ell_row, sigma2=1.0)  # (n, d)
    IM_arr = np.array([float(IM_gauss(ell[j], sigma2=1.0)) for j in range(d)])
    ILL_arr = np.array([float(ILL_gauss(ell[j], sigma2=1.0)) for j in range(d)])
    dIM_arr = np.asarray(IM_dlogell_gauss(ell, sigma2=1.0), dtype=float)
    dILL_arr = np.asarray(ILL_dlogell_gauss(ell, sigma2=1.0), dtype=float)

    # h_0(X), Hd_0 (sigma2 = 1 versions).
    r_design = len(J_sets)
    h0 = np.empty((n, r_design))
    Hd0 = np.empty(r_design)
    Ji_indicator = np.zeros((d, r_design), dtype=bool)
    for i, J_i in enumerate(J_sets):
        Ji_set = set(J_i)
        col = np.ones(n)
        hd = 1.0
        for j in range(d):
            in_Ji = j in Ji_set
            Ji_indicator[j, i] = in_Ji
            if in_Ji:
                col = col * L_all[:, j]
                hd *= ILL_arr[j]
            else:
                col = col * M_all[:, j]
                hd *= IM_arr[j]
        h0[:, i] = col
        Hd0[i] = hd

    # h_k = sigma2 * h_0; Hd_k = sigma2 * Hd_0; tr(M_k C_corr_k) is a single
    # sum that lets us recover ∂/∂log(sigma2) tr(M_k C*_k) = tr(M_k C*_k).
    h_k = sigma2 * h0
    Hd_k = sigma2 * Hd0
    Hi = 1.0 / Hd_k             # (r,)

    Mh = M_k @ h_k              # (n, r)
    v = np.sum(h_k * Mh, axis=0)  # (r,) = h_k[:,i].T M_k h_k[:,i]
    tr_M_Ccorr = float(np.sum(Hi * v))

    grad_logsig = float(np.sum(M_cse)) - tr_M_Ccorr  # = tr(M_k C*_k)

    # Per-coordinate ∂tr(M_k C_corr_k)/∂log(ell_c).
    grad_logell_corr = np.empty(d)
    for c in range(d):
        # dh0_c (n, r) = product over j with j == c factor swapped to derivative.
        dh0_c = np.empty((n, r_design))
        dHd0_c = np.empty(r_design)
        for i, _J_i in enumerate(J_sets):
            col = np.ones(n)
            hd = 1.0
            for j in range(d):
                in_Ji = bool(Ji_indicator[j, i])
                if j == c:
                    if in_Ji:
                        col = col * dL_all[:, j]
                        hd *= dILL_arr[j]
                    else:
                        col = col * dM_all[:, j]
                        hd *= dIM_arr[j]
                else:
                    if in_Ji:
                        col = col * L_all[:, j]
                        hd *= ILL_arr[j]
                    else:
                        col = col * M_all[:, j]
                        hd *= IM_arr[j]
            dh0_c[:, i] = col
            dHd0_c[i] = hd
        dh_c = sigma2 * dh0_c
        dHd_c = sigma2 * dHd0_c
        # d/dlog(ell_c) [tr(M_k C_corr_k)]
        #   = sum_i [-Hi_i^2 * dHd_c_i * v_i + 2 * Hi_i * Mh[:,i].T dh_c[:,i]]
        # M_k is symmetric so the squared form contributes a factor of 2.
        term1 = -np.sum((Hi * Hi) * dHd_c * v)
        term2 = 2.0 * np.sum(Hi * np.sum(Mh * dh_c, axis=0))
        grad_logell_corr[c] = term1 + term2

    grad_logell = grad_logell_se - grad_logell_corr
    return np.concatenate([[grad_logsig], grad_logell])


class MOOGP:
    """
    Multi-output Orthogonal Gaussian Process Model
    """

    # Default box bounds used by :meth:`_default_theta0_and_bounds` when the
    # caller does not pass explicit ``theta0`` / ``bounds`` to :meth:`fit`.
    # They assume X is standardized to ~[-1, 1] (the default ``standardize_x``)
    # and match the production benchmark init. Override on an instance or
    # subclass to widen/narrow the search box without hand-building bounds.
    DEFAULT_LATENT_SIGMA2_BOUNDS = (1e-3, 1e6)   # latent kernel variance sigma^2_k
    DEFAULT_LATENT_ELL_BOUNDS = (0.05, 100.0)    # latent length-scales ell_{k,j}
    DEFAULT_PSI_BOUNDS = (-5.0, 5.0)             # free Psi entries (learn_Psi=True)
    DEFAULT_SIGMA_EPS2_LB_FRAC = 1e-6            # noise lower bound = frac * Var(Y)
    DEFAULT_SIGMA_EPS2_UB_FRAC = 0.5             # noise upper bound = frac * Var(Y)

    def __init__(self,
                 terms,
                 q,
                 Psi=None,
                 *,
                 orthogonal=True,
                 learn_Psi=False,
                 sigma_eps2=None,
                 learn_sigma_eps=None,
                 diag_error_structure=None,
                 min_sigma_eps2=1e-10,
                 use_reml=False,
                 jitter=1e-6,
                 one_based=True,
                 normalize_cols=True,
                 use_diagonalized_interaction=True,
                 use_slow_kyinv=False,
                 store_dense_ky=False,
                 standardize_y="zscore",
                 standardize_x="unitcube",
                 x_margin=0.1,
                 use_analytical_grad=True):
        """
        Parameters
        ----------
        terms : list
            Basis specification for g(x) (same as make_G).
        q : int
            Number of latent orthogonal GPs. (q <= p)
        Psi : (p, q) array or None # TODO Change Psi arg options
            Initial / fixed mixing matrix. If learn_Psi=False, this must be provided.
            If learn_Psi=True, it's used for shape.
        orthogonal : bool
            If true, use orthogonalized squared exponential kernel detailed in 
            #TODO put paper link. If false, use standard squared exponential kernel.
        learn_Psi : bool
            If True, theta also parameterizes Psi.
        sigma_eps2: (p,) array or None
            Vector for measurement error
        learn_sigma_eps2: bool
            If True, optimize finds values for diagonal heterogeneous error parameters
        diag_error_structure: list[int] or None
            Optional grouping of the ``p`` outputs into blocks that share a
            single ``sigma^2`` parameter, giving
            ``Sigma_eps = bdiag(sigma_1^2 I_{p1}, ..., sigma_G^2 I_{pG})``.
            For example, ``[3, 2, 4]`` requires ``p == 9`` and fits three free
            variances. ``None`` (the default) is equivalent to ``[1] * p`` and
            recovers a unique parameter per output. The grouping affects only
            the parameterization used when ``learn_sigma_eps=True``; the
            covariance always sees a length-``p`` per-output vector with each
            group's variance broadcast across its outputs.
        min_sigma_eps2: float
            Minimum value for output measurement error
        use_reml : bool
            Placeholder for REML; currently same as ML if True.
        jitter : float
            Diagonal jitter for Ky.
        one_based : bool
            Interpret integers in `terms` as 1-based indices.
        normalize_cols : bool
            Whether to normalize each column of Psi when learning.
        use_diagonalized_interaction : bool
            If True, request the manuscript fast path based on
            ``Psi = Sigma_eps^{1/2} Phi`` and block-diagonal Woodbury updates.
            In this mode ``Phi`` (and thus ``Psi``) is rebuilt from the SVD of
            ``Y`` each likelihood call. The supplied ``Psi`` is only used in the
            dense slow path.
        use_slow_kyinv : bool
            Escape hatch for regression/debugging. If True, force the dense
            covariance/cholesky path.
        store_dense_ky : bool
            If True, keep dense ``Ky`` in cache for debugging.
        standardize_y : bool | str
            Output standardization applied internally before fitting. Default
            ``"zscore"``. Supported values are ``False`` (no scaling),
            ``True``/``"zscore"`` (per-output mean / sample-std), and
            ``"robust"`` (median / MAD). Predictions are always returned on the
            original output scale.
        standardize_x : bool | str
            Input standardization applied internally before fitting. Default
            ``"unitcube"`` (per-dim mapping to ``[-1, 1]`` using the training
            min/max) — the kernel theory and the default length-scale bounds
            assume X lives on the unit cube. Pass ``False`` to disable the
            transform when X is already on the right scale. When enabled the
            same transform is applied to ``Xstar`` at predict time, so callers
            can pass raw, unscaled X to both ``fit`` and ``predict``.
        x_margin : float
            Non-negative multiplicative padding for ``standardize_x="unitcube"``.
            With ``x_margin = m`` the per-dim half-width is widened by
            ``(1 + m)``, so training points land in
            ``[-1/(1+m), 1/(1+m)]`` instead of exactly on the cube boundary.
            Useful when test inputs may fall slightly outside the training
            range and to keep the optimizer away from the corners of the cube
            where the kernel has zero curvature. Default ``0.1`` — calibrated
            on the VAH fold-1 margin sweep (see
            ``notebooks/vah_moogp_diagnostic.ipynb``). Ignored when
            ``standardize_x`` is disabled.
        """
        self.terms = terms
        self.q = q
        self.Psi = None if Psi is None else np.asarray(Psi, float)
        self.orthogonal = orthogonal
        self.learn_Psi = learn_Psi
        self.sigma_eps2 = None if sigma_eps2 is None else np.asarray(sigma_eps2, float).ravel()
        
        # If the user doesn't specify, default to learning Sigma_eps when not provided.
        if learn_sigma_eps is None:
            learn_sigma_eps = (self.sigma_eps2 is None)
        self.learn_sigma_eps = bool(learn_sigma_eps)

        # Stored as-is; the data-dependent normalization (filling in the
        # ``[1] * p`` default and verifying ``sum == p``) happens in
        # ``_prepare_data`` once ``p`` is known.
        self._diag_error_structure_arg = diag_error_structure
        self.diag_error_structure = None

        self.min_sigma_eps2 = float(min_sigma_eps2)
        self.use_reml = use_reml
        self.jitter = jitter
        self.one_based = one_based
        self.normalize_cols = normalize_cols
        self.use_diagonalized_interaction = bool(use_diagonalized_interaction)
        self.use_slow_kyinv = bool(use_slow_kyinv)
        self.use_analytical_grad = bool(use_analytical_grad)
        self.store_dense_ky = bool(store_dense_ky)
        self.standardize_y = _normalize_standardize_y_mode(standardize_y)
        self.standardize_x = _normalize_standardize_x_mode(standardize_x)
        self.x_margin = float(x_margin)
        if self.x_margin < 0.0:
            raise ValueError(f"x_margin must be non-negative; got {self.x_margin}.")

        self._data = None
        self.cache = None
        self.theta_hat = None
        self.nll_hat = None
        self.opt_result = None
        self.fitted = False
        self.Ky_inv_rvec_ = None
        self.Y_raw = None
        self.y_center_ = None
        self.y_scale_ = None
        self.X_raw = None
        self.x_center_ = None
        self.x_scale_ = None
        self.Psi_work = None
        self.sigma_eps2_work = None

    def _to_working_x(self, X):
        return (np.asarray(X, float) - self.x_center_[None, :]) / self.x_scale_[None, :]

    def _to_working_y(self, Y):
        return (np.asarray(Y, float) - self.y_center_[None, :]) / self.y_scale_[None, :]

    def _from_working_mean(self, mean):
        return np.asarray(mean, float) * self.y_scale_[None, :] + self.y_center_[None, :]

    def _from_working_std(self, std):
        return np.asarray(std, float) * self.y_scale_[None, :]

    def _psi_to_working_scale(self, Psi):
        if Psi is None:
            return None
        return np.asarray(Psi, float) / self.y_scale_[:, None]

    def _psi_from_working_scale(self, Psi):
        if Psi is None:
            return None
        # Keep this compatible with autograd-traced arrays inside _nll.
        return Psi * self.y_scale_[:, None]

    def _sigma_eps2_to_working_scale(self, sigma_eps2):
        if sigma_eps2 is None:
            return None
        return np.asarray(sigma_eps2, float).ravel() / (self.y_scale_ ** 2)

    def _sigma_eps2_from_working_scale(self, sigma_eps2):
        if sigma_eps2 is None:
            return None
        # Keep this compatible with autograd-traced arrays inside _nll.
        return sigma_eps2 * (self.y_scale_ ** 2)
    
    def _prepare_data(self, data):
        # Accept both legacy ``X_scaled`` and the more honest ``X`` key; if both
        # are present we prefer the raw ``X`` so callers can migrate cleanly.
        if "X" in data:
            X_raw = np.asarray(data["X"])
        else:
            X_raw = np.asarray(data["X_scaled"])
        Y_raw = np.asarray(data.get("Y", data.get("y")))
        n, d = X_raw.shape
        p = Y_raw.shape[1]

        X, self.x_center_, self.x_scale_ = compute_working_x(
            X_raw, self.standardize_x, margin=self.x_margin,
        )
        Y, self.y_center_, self.y_scale_ = compute_working_y(Y_raw, self.standardize_y)

        self._data = data
        self.X = X
        self.X_raw = X_raw
        self.Y = Y
        self.Y_raw = Y_raw
        self.n = n
        self.d = d
        self.p = p
        self.Ky_inv_rvec_ = None
        self.Psi_work = self._psi_to_working_scale(self.Psi)
        self.sigma_eps2_work = self._sigma_eps2_to_working_scale(self.sigma_eps2)

        if self.Psi_work is not None:
            if self.Psi_work.shape != (p, self.q):
                raise ValueError(f"Psi shape {self.Psi_work.shape} ≠ (p={p}, q={self.q})")

        if (self.sigma_eps2_work is not None) and (self.sigma_eps2_work.size != p):
            raise ValueError(f"sigma_eps2 length {self.sigma_eps2_work.size} ≠ p={p}")

        # Resolve and validate the diagonal error grouping now that ``p`` is known.
        self.diag_error_structure = normalize_diag_error_structure(
            self._diag_error_structure_arg, p
        )

        # Build the fast low-rank basis from the working-scale outputs.
        if self.use_diagonalized_interaction:
            self.Phi_fast, self.d_vals_fast = init_phi(self.Y, self.q, self.n)
        else:
            self.Phi_fast, self.d_vals_fast = None, None

        # Cache Design matrices
        self.G = make_G({"X_scaled": self.X}, self.terms, one_based=self.one_based, return_names=False)
        self.r = self.G.shape[1] if self.G.size else 0
        self.Gy = build_Gy(self.G, self.p) if self.r > 0 else np.empty((self.n * self.p, 0))

        # Precompute per-coordinate squared differences for the train-train
        # kernel matrix so autograd does not retrace the (n, n, d) subtraction
        # each likelihood call (X is constant during optimization).
        X_np = np.asarray(self.X, dtype=float)
        diff_train = X_np[:, None, :] - X_np[None, :, :]
        self._train_sqdist = diff_train * diff_train

    def _apply_Ky_inv_fast(self, rhs, info):
        """Apply the diagonalized Woodbury form for ``K_y^{-1}`` to ``rhs``."""

        was_vec = (rhs.ndim == 1)
        rhs2 = rhs[:, None] if was_vec else rhs

        n = info["n"]
        p = info["p"]
        sigma_eps2 = info["sigma_eps2"]
        latent_factors = info["latent_factors"]

        if rhs2.shape[0] != n * p:
            raise ValueError(f"rhs has incompatible leading dimension {rhs2.shape[0]} (expected {n*p}).")

        # vecF stacks output columns sequentially, so a C-order reshape to
        # (p, n, m) gives Xrhs[i, j, k] = V_k[j, i] where V_k is the (n, p)
        # matrix carrying the k-th rhs. Putting the output dimension first
        # makes the per-output divide and the latent-projection contraction
        # operate on contiguous (n, m) slabs.
        m = rhs2.shape[1]
        rhs2_c = np.ascontiguousarray(rhs2)
        Xrhs = rhs2_c.reshape(p, n, m)

        # 1. Base measurement noise application: (Sigma_eps^{-1} \otimes I_n)
        out = Xrhs / sigma_eps2[:, None, None]

        # 2. Subtract latent GP updates. We keep the Cholesky factors of
        # ``I + d_k C_k`` and apply ``Q_k @ rhs`` lazily to avoid caching dense
        # ``Q_k`` matrices that can be reconstructed from ``C_k`` and the solve.
        for factor in latent_factors:
            uk = factor["psi"]      # shape (p,)
            d_val = factor["d"]     # scalar
            chol = factor["chol"]   # shape (n, n), lower triangular

            # Step A: project outputs into the latent space -> shape (n, m).
            M_uk = np.tensordot(uk, Xrhs, axes=(0, 0))

            # Step B: apply Q_k = (I + d_k C_k)^{-1} C_k via the exact identity
            # Q_k = (I - (I + d_k C_k)^{-1}) / d_k.
            Qk_M_uk = _apply_factorized_qk(chol, d_val, M_uk)

            # Step C: subtract uk[i] * Qk_M_uk[j, k] from out[i, j, k].
            out = out - uk[:, None, None] * Qk_M_uk[None, :, :]

        # Re-vectorize back to (np, m) in vecF order: out2[i*n + j, k] = out[i, j, k]
        # which equals vecF(V_k)[i*n + j].
        out2 = out.reshape(n * p, m)
        return out2[:, 0] if was_vec else out2

    def _solve_with_cached_Ky(self, rhs):
        """Apply ``K_y^{-1}`` using the active cache representation."""
        cache = self.cache
        fast_info = cache.get("fast_diag_info")
        if fast_info is not None:
            return self._apply_Ky_inv_fast(rhs, fast_info)
        Ky_chol = cache.get("Ky_chol")
        if Ky_chol is None:
            raise RuntimeError("No cached Ky solver is available.")
        return cho_solve(Ky_chol, rhs, check_finite=False)
        
    def _nll(self, theta_raw, *, build_cache=True):
        """
        Negative log-likelihood for the MOOGP model.

        Parameters
        ----------
        ``theta_raw`` follows the packing described in :func:`unpack_theta`.
        """
        X, Y, n, d, p = self.X, self.Y, self.n, self.d, self.p
        terms = self.terms

        use_fast_requested = self.use_diagonalized_interaction and (not self.use_slow_kyinv)
        # Fast path requires learn_Psi=False
        use_fast = bool(use_fast_requested and (not self.learn_Psi))

        lat_params, Psi_th, sigma_eps2_th = unpack_theta(
            theta_raw,
            d,
            self.q,
            p,
            learn_Psi=self.learn_Psi,
            learn_sigma_eps=self.learn_sigma_eps,
            normalize_cols=self.normalize_cols,
            diag_error_structure=self.diag_error_structure,
        )

        Psi = Psi_th if self.learn_Psi else self.Psi_work
        sigma_eps2 = sigma_eps2_th if self.learn_sigma_eps else self.sigma_eps2_work

        if (Psi is None) and (not use_fast):
            raise ValueError("Psi is None. Provide Psi or set learn_Psi=True.")

        if sigma_eps2 is None:
            sigma_eps2 = np.zeros(p, dtype=float)
        sigma_eps2 = np.maximum(sigma_eps2, self.min_sigma_eps2)

        # A fixed Psi only matches the diagonalized fast path when it agrees
        # with Psi = Sigma_eps^{1/2} Phi(Y) for the current sigma_eps2.
        if use_fast and (self.Psi_work is not None):
            Phi_fast = self.Phi_fast
            if Phi_fast is None:
                use_fast = False
            else:
                Psi_fast = np.diag(np.sqrt(sigma_eps2)) @ Phi_fast
                use_fast = bool(np.allclose(self.Psi_work, Psi_fast, rtol=1e-8, atol=1e-10))

        # Design
        G = self.G
        r = self.r
        Gy = self.Gy

        # Covariance per latent: Cj_list[k]
        train_sqdist = getattr(self, "_train_sqdist", None)
        Cj_list = []
        for (sigma2_j, ell_j) in lat_params:
            Cj = make_c_star_matrix(X, X, ell=ell_j, sigma2=sigma2_j,
                                    terms=terms, orthogonal=self.orthogonal,
                                    one_based=self.one_based,
                                    sqdist=train_sqdist)
            Cj_list.append(Cj)

        fast_diag_info = None
        Ky = None
        Ky_chol = None
        Psi_cache = Psi

        if use_fast:
            # 1. Phi from SVD of Y
            Phi = self.Phi_fast
            d_vals = self.d_vals_fast

            # 2. Standardized output basis (q,p)
            psi_c = Phi.T / np.sqrt(sigma_eps2)

            # 3. Precompute Cholesky factors for the Woodbury updates and the
            # corresponding log-determinant terms.
            latent_factors = []
            logdetK = n * np.sum(np.log(sigma_eps2))

            for k in range(self.q):
                Ck_star = Cj_list[k]
                Dk = d_vals[k]

                A = np.eye(n) + Dk * Ck_star 
                L = np.linalg.cholesky(A)
                # log(det(A)) = 2 * sum(log(diag(L)))
                logdetK += 2.0 * np.sum(np.log(np.diag(L)))

                latent_factors.append(
                    {
                        "psi": psi_c[k],
                        "d": Dk,
                        "chol": L,
                    }
                )

            fast_diag_info = dict(
                n=n,
                p=p,
                latent_factors=latent_factors,
                sigma_eps2=sigma_eps2
            )

            def solve_Ky(rhs): return self._apply_Ky_inv_fast(rhs, fast_diag_info)
            Psi_cache = (np.sqrt(sigma_eps2)[:, None] * Phi) if build_cache else None
        else:
            Ky = build_Ky(Cj_list, Psi, sigma_eps2=sigma_eps2)
            if self.jitter:
                Ky = Ky + self.jitter * np.eye(n * p)
            Ky_chol = cho_factor(Ky, lower=True, check_finite=False)
            logdetK = 2.0 * np.sum(np.log(np.diag(Ky_chol[0])))
            def solve_Ky(rhs): return cho_solve(Ky_chol, rhs, check_finite=False)
            Psi_cache = Psi

        vecY = vecF(Y)
        if use_fast:
            # Apply K_y^{-1} to vec(Y) once (single np-vector solve), then use
            # the Kronecker structure of G_y = I_p ⊗ G to assemble A_gls and
            # K_y^{-1} G_y β analytically — no dense (np × rp) solve.
            alpha_vec = self._apply_Ky_inv_fast(vecY, fast_diag_info)
            alpha_mat = unvecF(alpha_vec, n, p)
            qf, bhat, rvec, Ky_inv_rvec = _profiled_gls_terms_fast(
                fast_diag_info,
                G,
                vecY,
                alpha_mat,
                p,
                build_cache=build_cache,
            )
        elif r > 0:
            z = solve_Ky(Gy)
            alpha = solve_Ky(vecY)

            A_gls = Gy.T @ z
            b_gls = Gy.T @ alpha
            beta_vec = np.linalg.solve(A_gls, b_gls)

            bhat = unvecF(beta_vec, r, p)
            rvec = vecY - Gy @ vecF(bhat)
            Ky_inv_rvec = alpha - z @ beta_vec
            qf = np.dot(rvec, Ky_inv_rvec)
        else:
            bhat = np.zeros((0, p))
            rvec = vecY
            Ky_inv_rvec = solve_Ky(rvec)
            qf = np.dot(rvec, Ky_inv_rvec)
        nll = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi))
        # Normalize to per-row NLL (mirrors LCGP's ``neglpost /= n``). The
        # converged hyperparameters are invariant under positive rescaling
        # of the objective; this only keeps the L-BFGS-B function value and
        # its gradient O(1) regardless of dataset size, so a single set of
        # ``ftol`` / ``gtol`` defaults is meaningful across the sweep.
        nll = nll / float(n)

        if self.use_reml:
            A = None  # you can fill in REML here later
        else:
            A = None

        # cache for predictions
        if build_cache:
            self.cache = dict(
                Ky=(Ky if self.store_dense_ky else None),
                Ky_chol=Ky_chol,
                Cj_list=Cj_list,
                Psi=Psi_cache,
                Psi_raw=self._psi_from_working_scale(Psi_cache),
                sigma_eps2=sigma_eps2,
                sigma_eps2_raw=self._sigma_eps2_from_working_scale(sigma_eps2),
                used_fast=use_fast,
                fast_diag_info=fast_diag_info,
                G=G,
                Gy=Gy,
                bhat=bhat,
                r_vec=rvec,
                Ky_inv_rvec=Ky_inv_rvec,
                residual_vec=rvec,
                qf=qf,
                logdetK=logdetK,
                A=A,
                lat_params=lat_params,
                terms=terms,
                one_based=self.one_based,
                X=X,
                Y=Y,
                Y_raw=self.Y_raw,
                y_center=self.y_center_,
                y_scale=self.y_scale_,
                standardize_y=self.standardize_y,
                theta_raw=np.asarray(theta_raw, dtype=float).copy(),
            )
            self.Ky_inv_rvec_ = Ky_inv_rvec
        return nll
    
    def _nll_and_grad_fast(self, theta_raw):
        """Exact NLL + analytical gradient on the fast Woodbury path.

        This replaces ``value_and_grad(_nll)`` when the fast path is active and
        ``use_analytical_grad=True``. It is bit-identical to the autograd
        gradient up to floating-point roundoff — no approximations.

        Preconditions (enforced by ``fit`` before this method is selected):
          - ``use_diagonalized_interaction=True``
          - ``use_slow_kyinv=False``
          - ``learn_Psi=False`` and ``self.Psi is None`` (so Psi_work is None)

        REML adjustment is left as a placeholder (``A=None``) matching the
        current ``_nll`` implementation.
        """
        X = self.X
        Y = self.Y
        n = self.n
        d = self.d
        p = self.p
        q = self.q
        terms = self.terms
        orthogonal = self.orthogonal
        one_based = self.one_based
        sqdist = getattr(self, "_train_sqdist", None)

        theta_raw = np.asarray(theta_raw, dtype=float)

        # --- Unpack parameters (plain numpy, not traced) ---
        lat_params, _, sigma_eps2_th = unpack_theta(
            theta_raw,
            d,
            q,
            p,
            learn_Psi=False,
            learn_sigma_eps=self.learn_sigma_eps,
            normalize_cols=self.normalize_cols,
            diag_error_structure=self.diag_error_structure,
        )
        sigma_eps2 = sigma_eps2_th if self.learn_sigma_eps else self.sigma_eps2_work
        if sigma_eps2 is None:
            sigma_eps2 = np.zeros(p, dtype=float)
        sigma_eps2 = np.maximum(np.asarray(sigma_eps2, dtype=float), self.min_sigma_eps2)

        Phi = self.Phi_fast
        d_vals = self.d_vals_fast
        if Phi is None or d_vals is None:
            raise RuntimeError("Fast-path Phi/d not initialized; call _prepare_data first.")
        Psi = np.sqrt(sigma_eps2)[:, None] * Phi  # (p, q)

        # --- Build C_k and Cholesky factors (plain numpy, no autograd tape) ---
        Cj_list = []
        for (sigma2_j, ell_j) in lat_params:
            Cj = make_c_star_matrix(
                X, X,
                ell=np.asarray(ell_j, dtype=float),
                sigma2=float(sigma2_j),
                terms=terms,
                orthogonal=orthogonal,
                one_based=one_based,
                sqdist=sqdist,
            )
            Cj_list.append(np.asarray(Cj, dtype=float))

        L_list = []
        logdetK = float(n) * float(np.sum(np.log(sigma_eps2)))
        for k in range(q):
            Ak = np.eye(n) + d_vals[k] * Cj_list[k]
            Lk = np.linalg.cholesky(Ak)
            L_list.append(Lk)
            logdetK += 2.0 * float(np.sum(np.log(np.diag(Lk))))

        # --- Forward pass: apply K_y^{-1} via the existing fast Woodbury helper ---
        fast_info = dict(
            n=n,
            p=p,
            sigma_eps2=sigma_eps2,
            latent_factors=[
                dict(
                    psi=Phi[:, k] / np.sqrt(sigma_eps2),
                    d=d_vals[k],
                    chol=L_list[k],
                )
                for k in range(q)
            ],
        )
        vecY_ = vecF(Y)
        # Single np-vector apply for α; fast path then exploits I_p ⊗ G.
        alpha_vec = self._apply_Ky_inv_fast(vecY_, fast_info)
        alpha_mat_for_terms = unvecF(alpha_vec, n, p)
        qf, bhat, rvec, Ky_inv_rvec = _profiled_gls_terms_fast(
            fast_info,
            self.G,
            vecY_,
            alpha_mat_for_terms,
            p,
            build_cache=True,
        )
        nll = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi))

        # Populate the prediction cache so ``fit`` does not need a separate
        # post-optimization ``_nll`` rebuild. The fields mirror what ``_nll``
        # writes on the fast path.
        self.cache = dict(
            Ky=None,
            Ky_chol=None,
            Cj_list=Cj_list,
            Psi=Psi,
            Psi_raw=self._psi_from_working_scale(Psi),
            sigma_eps2=sigma_eps2,
            sigma_eps2_raw=self._sigma_eps2_from_working_scale(sigma_eps2),
            used_fast=True,
            fast_diag_info=fast_info,
            G=self.G,
            Gy=self.Gy,
            bhat=bhat,
            r_vec=rvec,
            Ky_inv_rvec=Ky_inv_rvec,
            residual_vec=rvec,
            qf=qf,
            logdetK=logdetK,
            A=None,
            lat_params=lat_params,
            terms=terms,
            one_based=one_based,
            X=X,
            Y=Y,
            Y_raw=self.Y_raw,
            y_center=self.y_center_,
            y_scale=self.y_scale_,
            standardize_y=self.standardize_y,
            theta_raw=theta_raw.copy(),
        )
        self.Ky_inv_rvec_ = Ky_inv_rvec

        # --- Gradient assembly ---
        grad_theta = np.zeros_like(theta_raw)
        alpha_mat = unvecF(Ky_inv_rvec, n, p)  # (n, p)

        # (1) Latent kernel params: closed-form gradient of the trace functional
        #     tr(M_k * C*_k(theta_k)). C*_k is linear in sigma2 and the
        #     orthogonalized SE kernel decomposes coordinate-wise, so each
        #     latent block is assembled from elementary M, L, IM, ILL pieces
        #     and their log-ell derivatives — no autograd tape, no rebuild of
        #     the (n × n) kernel under reverse-mode tracing.
        if sqdist is None:
            diff = X[:, None, :] - X[None, :, :]
            sqdist_local = diff * diff
        else:
            sqdist_local = sqdist
        for k in range(q):
            # dpotri exploits symmetry of A^{-1} and is ~2x faster than
            # cho_solve(L, I_n) at the cost of returning only one triangle.
            # Pass a copy of L so dpotri's overwrite does not corrupt L_list[k]
            # (still needed by _apply_Ky_inv_fast and the cache).
            A_inv_lo, info_inv = dpotri(L_list[k], lower=1, overwrite_c=0)
            if info_inv != 0:
                raise RuntimeError(f"dpotri failed for latent {k} (info={info_inv})")
            # Mirror the lower triangle into the upper triangle in-place.
            i_hi = np.triu_indices(n, k=1)
            A_inv_lo[i_hi] = A_inv_lo.T[i_hi]
            B_k = d_vals[k] * A_inv_lo
            g_k = alpha_mat @ Psi[:, k]               # (n,)
            M_k = 0.5 * (B_k - np.outer(g_k, g_k))    # symmetric (n, n)

            sigma2_k = float(lat_params[k][0])
            ell_k = np.asarray(lat_params[k][1], dtype=float)
            grad_k = _latent_kernel_logtheta_grad(
                M_k,
                X,
                sqdist_local,
                ell_k,
                sigma2_k,
                terms,
                orthogonal=orthogonal,
                one_based=one_based,
            )
            grad_theta[k * (d + 1):(k + 1) * (d + 1)] = grad_k

        # (2) Sigma_eps params: fully analytical.
        #     ∂NLL/∂(log σ²_l) = 0.5 (n - σ²_l ||α_l||² - Σ_k ψ_{k,l} · (α_mat^T C_k g_k)_l)
        # When ``diag_error_structure`` groups outputs, the free parameter is
        # ``log σ²_g`` shared across the outputs in group g; by the chain rule
        # the group gradient is the sum of per-output gradients in that group.
        if self.learn_sigma_eps:
            grad_sigma_per_output = 0.5 * (float(n) - sigma_eps2 * np.sum(alpha_mat ** 2, axis=0))
            for k in range(q):
                g_k = alpha_mat @ Psi[:, k]           # (n,)
                Ck_gk = Cj_list[k] @ g_k              # (n,)
                HA_k = alpha_mat.T @ Ck_gk            # (p,)
                grad_sigma_per_output = grad_sigma_per_output - 0.5 * Psi[:, k] * HA_k

            err_struct = self.diag_error_structure
            grad_sigma = aggregate_per_output_to_groups(grad_sigma_per_output, err_struct)
            n_groups = len(err_struct)

            base = q * (d + 1)
            grad_theta[base:base + n_groups] = grad_sigma

        # Match the ``_nll`` per-row normalization: scale both the value and
        # the gradient by 1/n so the analytical-grad path stays bit-equivalent
        # to ``value_and_grad(_nll)``.
        inv_n = 1.0 / float(n)
        return float(nll) * inv_n, grad_theta * inv_n

    def _default_theta0_and_bounds(self):
        """Data-aware initial hyperparameters and box bounds for :meth:`fit`.

        Built automatically from the prepared working-scale data so callers do
        not have to hand-construct ``theta0`` / ``bounds``. Must be called after
        :meth:`_prepare_data` (``self.X`` / ``self.Y`` hold the working-scale
        arrays, ``self.diag_error_structure`` is resolved).

        The packing matches :func:`unpack_theta`:

          1. latent kernel block of length ``q * (d + 1)`` (always),
          2. ``Psi`` block of length ``p * q`` (only when ``learn_Psi``),
          3. ``Sigma_eps`` block of ``len(diag_error_structure)`` log-variances
             (only when ``learn_sigma_eps``).

        Seeds mirror LCGP's ``init_params`` (and the previous external
        ``make_data_aware_theta0_and_bounds`` benchmark helper):

        * latent ``sigma^2_k = s_k^2 / n`` from the singular values of
          ``Y_work`` (undoes the ``Phi`` SVD normalization in the fast path),
        * length-scales ``ell_{k,j} = exp(0.5*log(d) + log(std(X_work[:, j])))``
          (the ``sqrt(d) * std(X)`` heuristic, shared across latents),
        * ``Psi`` (if learned) from the leading right singular vectors of
          ``Y_work`` (column normalization makes only their direction matter),
        * ``sigma^2_eps`` from the per-group variance of the rank-``q`` SVD
          reconstruction residual of ``Y_work`` (a data-implied noise floor).

        All seeds are clipped strictly inside the box so L-BFGS-B starts
        feasible. Uses real numpy (not autograd) since this is never traced.
        """
        q, d, p, n = self.q, self.d, self.p, self.n
        X_work = onp.asarray(self.X, dtype=float)
        Y_work = onp.asarray(self.Y, dtype=float)

        U, svals, Vt = onp.linalg.svd(Y_work, full_matrices=False)
        s_q = onp.maximum(svals[:q], 1e-12)
        s2_k = (s_q ** 2) / float(n)  # = 1 / d_vals in init_phi

        std_x = onp.std(X_work, axis=0)
        std_x = onp.where(std_x > 1e-12, std_x, 1.0)
        log_ell = 0.5 * onp.log(d) + onp.log(std_x)  # shared across latents

        theta0: list[float] = []
        bounds: list[tuple[float, float]] = []

        # 1. Latent kernel block: data-aware seeds + constant box.
        sig2_lb, sig2_ub = self.DEFAULT_LATENT_SIGMA2_BOUNDS
        ell_lb, ell_ub = self.DEFAULT_LATENT_ELL_BOUNDS
        latent_box = [(float(onp.log(sig2_lb)), float(onp.log(sig2_ub)))] + (
            [(float(onp.log(ell_lb)), float(onp.log(ell_ub)))] * d
        )
        for k in range(q):
            theta0.append(float(onp.log(max(s2_k[k], 1e-8))))
            theta0.extend(float(v) for v in log_ell)
            bounds.extend(latent_box)

        # 2. Optional Psi block (slow path; learn_Psi=True).
        if self.learn_Psi:
            Psi0 = onp.asarray(Vt[:q].T, dtype=float)  # (p, q), orthonormal columns
            theta0.extend(float(v) for v in Psi0.ravel())
            bounds.extend([tuple(self.DEFAULT_PSI_BOUNDS)] * (p * q))

        # 3. Optional Sigma_eps block (learn_sigma_eps=True). When
        #    ``diag_error_structure`` groups outputs, each group carries one
        #    shared log-variance, so per-output seeds/bounds are collapsed to
        #    per-group means (a no-op for the default [1]*p grouping).
        if self.learn_sigma_eps:
            sizes = onp.asarray(self.diag_error_structure, dtype=int)
            bp = onp.concatenate([[0], onp.cumsum(sizes)[:-1]])

            Y_hat = U[:, :q] @ onp.diag(svals[:q]) @ Vt[:q, :]
            resid_var = onp.maximum(onp.var(Y_work - Y_hat, axis=0, ddof=1), 1e-12)
            resid_var = onp.add.reduceat(resid_var, bp) / sizes
            theta0.extend(float(v) for v in onp.log(resid_var))

            y_var = onp.maximum(1e-12, onp.var(Y_work, axis=0, ddof=1))
            y_var = onp.add.reduceat(y_var, bp) / sizes
            lb = onp.maximum(1e-12, self.DEFAULT_SIGMA_EPS2_LB_FRAC * y_var)
            ub = onp.maximum(lb * 10.0, self.DEFAULT_SIGMA_EPS2_UB_FRAC * y_var)
            bounds.extend(
                (float(onp.log(lbi)), float(onp.log(ubi))) for lbi, ubi in zip(lb, ub)
            )

        # Clip every seed strictly inside its box so L-BFGS-B starts feasible.
        theta0_arr = onp.asarray(theta0, dtype=float)
        lo = onp.array([b[0] for b in bounds], dtype=float)
        hi = onp.array([b[1] for b in bounds], dtype=float)
        theta0_arr = onp.minimum(onp.maximum(theta0_arr, lo), hi)
        return theta0_arr, bounds

    def fit(self, data, theta0=None, bounds=None, optimizer_opts=None):
        """
        Fit the model by ML (or REML placeholder) given data and initial hyperparameters.

        Parameters
        ----------
        data : dict
            Must contain 'X_scaled' (n,d) and 'Y' or 'y' (n,p).
        theta0 : 1D array or None
            Initial parameter vector (same packing as :func:`unpack_theta`).
            When ``None`` (the default) a data-aware vector is built
            automatically by :meth:`_default_theta0_and_bounds` so callers do
            not have to set it explicitly.
        bounds : list of (low, high) or None
            L-BFGS-B bounds. When ``None`` (the default) the matching box bounds
            from :meth:`_default_theta0_and_bounds` are used.
        optimizer_opts : dict or None
            Extra options passed to scipy.optimize.minimize.
        """
        self._prepare_data(data)

        # Auto-build a data-aware theta0 / bounds when the caller omits either.
        if theta0 is None or bounds is None:
            default_theta0, default_bounds = self._default_theta0_and_bounds()
            if theta0 is None:
                theta0 = default_theta0
            if bounds is None:
                bounds = default_bounds

        def obj(th): return self._nll(th)

        use_fast = (
            self.use_diagonalized_interaction
            and (not self.use_slow_kyinv)
            and (not self.learn_Psi)
            and (self.Psi is None)
        )

        if use_fast and self.use_analytical_grad:
            target_obj = self._nll_and_grad_fast
            use_jac = True
        elif use_fast:
            target_obj = value_and_grad(lambda th: self._nll(th, build_cache=False))
            use_jac = True
        else:
            target_obj = obj
            use_jac = False
        

        res = minimize(
            target_obj,
            np.asarray(theta0),
            method="L-BFGS-B",
            jac=use_jac,
            bounds=bounds,
            options={"maxiter": 200, **(optimizer_opts or {})} if optimizer_opts else {"maxiter": 200},
        )

        self.theta_hat = res.x
        self.nll_hat = float(res.fun)
        self.opt_result = res
        self.fitted = True

        # Skip the post-fit rebuild when the optimizer's last evaluation already
        # populated ``self.cache`` for ``theta_hat``. ``_nll_and_grad_fast`` and
        # ``_nll(build_cache=True)`` both stamp ``theta_raw`` on the cache.
        cache_theta = (self.cache or {}).get("theta_raw") if self.cache else None
        if cache_theta is None or not np.array_equal(np.asarray(cache_theta), np.asarray(self.theta_hat)):
            self._nll(self.theta_hat)
        return self
    
    def predict(
        self,
        Xstar,
        *,
        return_std=False,
        diag_only=True,
        include_mean_uncertainty=False,
        predict_observation=True,
    ):
        """
        Predict at new inputs ``Xstar``.

        ``Xstar`` should be on the same scale as the X the model was fit on:
        raw when ``standardize_x`` is set (the model applies its training-data
        ``[-1, 1]`` map to ``Xstar`` here), and already in ``[-1, 1]`` when
        ``standardize_x=False`` (the historical convention).

        Returns
        -------
        mean : (n*, p)
        std  : (n*, p) if return_std=True
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        cache = self.cache
        terms = cache["terms"]
        one_based = cache["one_based"]
        use_reml = self.use_reml

        X = cache["X"]
        Psi = cache["Psi"]
        sigma_eps2 = cache.get("sigma_eps2", None)
        lat_params = cache["lat_params"]
        bhat = cache["bhat"]
        Gy = cache["Gy"]
        A = cache["A"]
        
        n = X.shape[0]
        p = Psi.shape[0]
        Ky_inv_rvec = cache["Ky_inv_rvec"]

        Xs = self._to_working_x(np.asarray(Xstar))
        nstar = Xs.shape[0]

        # 1. Evaluate spatial cross-kernels
        Cj_XsX = []
        for (sigma2_j, ell_j) in lat_params:
            Cj1 = make_c_star_matrix(Xs, X, ell=ell_j, sigma2=sigma2_j,
                                     terms=terms, orthogonal=self.orthogonal, one_based=one_based)
            Cj_XsX.append(Cj1)

        # 2. FAST MEAN PREDICTION (Zero Kronecker Products)
        Gs = make_G({"X_scaled": Xs}, terms, one_based=one_based, return_names=False)
        mean_mat = (Gs @ bhat) if Gs.size else np.zeros((nstar, p))
        
        # Unvectorize the pre-computed K_y^-1 R vector
        alpha_mat = unvecF(Ky_inv_rvec, n, p)
        
        for j in range(self.q):
            psi_j = Psi[:, j]
            # Spatial projection: Cj_XsX @ (alpha_mat @ psi_j) -> shape (nstar,)
            latent_mean = Cj_XsX[j] @ (alpha_mat @ psi_j)
            # Output projection: outer product -> shape (nstar, p)
            mean_mat += np.outer(latent_mean, psi_j)
            
        mean = mean_mat
        mean_raw = self._from_working_mean(mean)

        # LAZY EVALUATION: Stop here if standard deviation is not requested
        if not return_std:
            return mean_raw

        # 3. VARIANCE PREDICTION
        fast_diag_info = cache.get("fast_diag_info")

        # Per-latent test-test diagonals are needed by both paths.
        Cj_diag_star_list = [
            make_c_star_diag(Xs, ell=ell_j, sigma2=sigma2_j,
                             terms=terms, orthogonal=self.orthogonal, one_based=one_based)
            for (sigma2_j, ell_j) in lat_params
        ]

        if fast_diag_info is not None:
            # Structure-exploiting: never form the dense K_XsX (np × n*p) block
            # nor solve K_y^{-1} on it. Use the closed Woodbury form
            #   K_y^{-1} K_X* = Σ_j (ψ̃_j ψ_j^T) ⊗ A_j^{-1} C_j(X, X_*)
            # whose diagonal contribution to K_*X K_y^{-1} K_X* is
            #   Σ_j d_j (ψ_j ⊙ ψ_j) ⊗ diag(C_j(X_*, X) A_j^{-1} C_j(X, X_*)).
            diag = _predict_variance_diag_fast(
                fast_diag_info,
                Cj_XsX,
                Psi,
                Cj_diag_star_list=Cj_diag_star_list,
                predict_observation=bool(predict_observation and sigma_eps2 is not None),
            )

            if include_mean_uncertainty and Gs.size and use_reml and (A is not None):
                # Rare REML branch — falls back to the dense form to keep parity.
                K_XsX = build_cross_K(Psi, Cj_XsX)
                Gs_y = build_Gy(Gs, p)
                M = Gs_y - K_XsX @ self._solve_with_cached_Ky(Gy)
                W = solve(A, M.T, assume_a="sym")
                diag = diag + np.sum(M * W.T, axis=1)
        else:
            # Dense fallback (slow path, e.g. learn_Psi=True or use_slow_kyinv=True).
            K_XsX = build_cross_K(Psi, Cj_XsX)

            diag_prior = np.zeros(nstar * p)
            for j, _ in enumerate(lat_params):
                diag_prior = diag_prior + np.kron(Psi[:, j] ** 2, Cj_diag_star_list[j])

            V = self._solve_with_cached_Ky(K_XsX.T)
            diag_cross = np.sum(K_XsX * V.T, axis=1)
            diag = diag_prior - diag_cross

            if predict_observation and (sigma_eps2 is not None):
                diag = diag + np.repeat(np.asarray(sigma_eps2, float).ravel(), nstar)

            if include_mean_uncertainty and Gs.size and use_reml and (A is not None):
                Gs_y = build_Gy(Gs, p)
                M = Gs_y - K_XsX @ self._solve_with_cached_Ky(Gy)
                W = solve(A, M.T, assume_a="sym")
                diag = diag + np.sum(M * W.T, axis=1)

        std = np.sqrt(np.maximum(diag, 0.0))
        std = unvecF(std, nstar, p)
        std_raw = self._from_working_std(std)

        return mean_raw, std_raw
