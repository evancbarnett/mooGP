import autograd.numpy as np

from autograd.scipy.linalg import solve_triangular as ag_solve_triangular
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.optimize import minimize

from .design import make_G, build_Gy, vecF, unvecF
from .kernels import make_c_star_matrix, make_c_star_diag
from autograd import value_and_grad

def unpack_theta(
    theta_raw,
    d,
    q,
    p,
    *,
    learn_Psi=False,
    learn_sigma_eps=False,
    normalize_cols=True,
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
           [log(sigma^2_{eps,1}), ..., log(sigma^2_{eps,p})]

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
        need = p
        if t.size < off + need:
            raise ValueError(f"theta length {t.size} < required +Sigma_eps {off+need}")
        log_sigma = t[off:off + need]
        off += need
        sigma_eps2 = np.exp(log_sigma)

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
    p = Y.shape[1]
    
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

class MOOGP:
    """
    Multi-output Orthogonal Gaussian Process Model
    """

    def __init__(self,
                 terms,
                 q,
                 Psi=None,
                 *,
                 orthogonal=True,
                 learn_Psi=False,
                 sigma_eps2=None,
                 learn_sigma_eps=None,
                 min_sigma_eps2=1e-10,
                 use_reml=False,
                 jitter=1e-6,
                 one_based=True,
                 normalize_cols=True,
                 use_diagonalized_interaction=True,
                 use_slow_kyinv=False,
                 store_dense_ky=False,
                 standardize_y=False):
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
            Optional output standardization applied internally before fitting.
            Supported values are ``False``, ``True``/``"zscore"``, and
            ``"robust"``. Predictions are always returned on the original
            output scale.
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
        
        self.min_sigma_eps2 = float(min_sigma_eps2)
        self.use_reml = use_reml
        self.jitter = jitter
        self.one_based = one_based
        self.normalize_cols = normalize_cols
        self.use_diagonalized_interaction = bool(use_diagonalized_interaction)
        self.use_slow_kyinv = bool(use_slow_kyinv)
        self.store_dense_ky = bool(store_dense_ky)
        self.standardize_y = _normalize_standardize_y_mode(standardize_y)

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
        self.Psi_work = None
        self.sigma_eps2_work = None

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
        X = np.asarray(data["X_scaled"])
        Y_raw = np.asarray(data.get("Y", data.get("y")))
        n, d = X.shape
        p = Y_raw.shape[1]

        Y, self.y_center_, self.y_scale_ = compute_working_y(Y_raw, self.standardize_y)

        self._data = data
        self.X = X
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

        # Build the fast low-rank basis from the working-scale outputs.
        if self.use_diagonalized_interaction:
            self.Phi_fast, self.d_vals_fast = init_phi(self.Y, self.q, self.n)
        else:
            self.Phi_fast, self.d_vals_fast = None, None

        # Cache Design matrices
        self.G = make_G({"X_scaled": self.X}, self.terms, one_based=self.one_based, return_names=False)
        self.r = self.G.shape[1] if self.G.size else 0
        self.Gy = build_Gy(self.G, self.p) if self.r > 0 else np.empty((self.n * self.p, 0))

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

        # Un-vectorize: each RHS column becomes an (n x p) matrix
        Xrhs = rhs2.reshape(n, p, rhs2.shape[1], order="F")

        # 1. Base measurement noise application: (Sigma_eps^{-1} \otimes I_n)
        # Broadcasting sigma_eps2 across the n and m dimensions
        out = Xrhs / sigma_eps2[None, :, None]

        # 2. Subtract latent GP updates. We keep the Cholesky factors of
        # ``I + d_k C_k`` and apply ``Q_k @ rhs`` lazily to avoid caching dense
        # ``Q_k`` matrices that can be reconstructed from ``C_k`` and the solve.
        for factor in latent_factors:
            uk = factor["psi"]      # shape (p,)
            d_val = factor["d"]     # scalar
            chol = factor["chol"]   # shape (n, n), lower triangular

            # Step A: Project outputs into the latent space -> shape (n, m)
            M_uk = np.sum(Xrhs * uk[None, :, None], axis=1)

            # Step B: Apply Q_k = (I + d_k C_k)^{-1} C_k via the exact identity
            # Q_k = (I - (I + d_k C_k)^{-1}) / d_k.
            Qk_M_uk = _apply_factorized_qk(chol, d_val, M_uk)

            # Step C: Project back to the output space and subtract -> shape (n, p, m)
            out = out - (Qk_M_uk[:, None, :] * uk[None, :, None])

        # Re-vectorize back to (np, m)
        out2 = out.reshape(n * p, rhs2.shape[1], order="F")
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
        Cj_list = []
        for (sigma2_j, ell_j) in lat_params:
            Cj = make_c_star_matrix(X, X, ell=ell_j, sigma2=sigma2_j,
                                    terms=terms, orthogonal=self.orthogonal, one_based=self.one_based)
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

            solve_Ky = lambda rhs: self._apply_Ky_inv_fast(rhs, fast_diag_info)
            Psi_cache = (np.sqrt(sigma_eps2)[:, None] * Phi) if build_cache else None
        else:
            Ky = build_Ky(Cj_list, Psi, sigma_eps2=sigma_eps2)
            if self.jitter:
                Ky = Ky + self.jitter * np.eye(n * p)
            Ky_chol = cho_factor(Ky, lower=True, check_finite=False)
            logdetK = 2.0 * np.sum(np.log(np.diag(Ky_chol[0])))
            solve_Ky = lambda rhs: cho_solve(Ky_chol, rhs, check_finite=False)
            Psi_cache = Psi

        vecY = vecF(Y)
        if use_fast:
            qf, bhat, rvec, Ky_inv_rvec = _profiled_gls_terms(
                solve_Ky,
                G,
                Gy,
                vecY,
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
            )
            self.Ky_inv_rvec_ = Ky_inv_rvec
        return nll
    
    def fit(self, data, theta0, bounds=None, optimizer_opts=None):
        """
        Fit the model by ML (or REML placeholder) given data and initial hyperparameters.

        Parameters
        ----------
        data : dict
            Must contain 'X_scaled' (n,d) and 'Y' or 'y' (n,p).
        theta0 : 1D array
            Initial parameter vector (same packing as before).
        bounds : list of (low, high) or None
            L-BFGS-B bounds.
        optimizer_opts : dict or None
            Extra options passed to scipy.optimize.minimize.
        """
        self._prepare_data(data)

        obj = lambda th: self._nll(th)

        use_fast = (
            self.use_diagonalized_interaction
            and (not self.use_slow_kyinv)
            and (not self.learn_Psi)
            and (self.Psi is None)
        )

        if use_fast:
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

        # Make sure cache corresponds to theta_hat
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
        Predict at new inputs Xstar (scaled in [-1,1]^d).

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

        Xs = np.asarray(Xstar)
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
        
        # We only build this massive matrix if variance is explicitly requested
        K_XsX = build_cross_K(Psi, Cj_XsX) 

        # FAST PRIOR VARIANCE: Compute diagonal analytically without building K_XsXs
        diag_prior = np.zeros(nstar * p)
        for j, (sigma2_j, ell_j) in enumerate(lat_params):
            # Compute only diagonal of test-test matrix
            Cj2_diag = make_c_star_diag(Xs, ell=ell_j, sigma2=sigma2_j,
                                        terms=terms, orthogonal=self.orthogonal, one_based=one_based)
            # The diagonal of A kron B is kron(diag(A), diag(B))
            diag_prior += np.kron(Psi[:, j]**2, Cj2_diag)

        # Cross variance and final computation
        V = self._solve_with_cached_Ky(K_XsX.T)  # (n p × n* p)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag = diag_prior - diag_cross

        if predict_observation and (sigma_eps2 is not None):
            diag += np.repeat(np.asarray(sigma_eps2, float).ravel(), nstar)

        if include_mean_uncertainty and Gs.size and use_reml and (A is not None):
            Gs_y = build_Gy(Gs, p)
            M = Gs_y - K_XsX @ self._solve_with_cached_Ky(Gy)
            W = solve(A, M.T, assume_a="sym")
            diag += np.sum(M * W.T, axis=1)

        std = np.sqrt(np.maximum(diag, 0.0))
        std = unvecF(std, nstar, p)
        std_raw = self._from_working_std(std)
        
        return mean_raw, std_raw
