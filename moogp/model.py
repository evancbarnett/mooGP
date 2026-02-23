import numpy as np
from numpy.linalg import LinAlgError, norm
from scipy.linalg import cho_factor, cho_solve, solve, solve_triangular
from scipy.optimize import minimize

from .design import make_G, build_Gy, vecF, unvecF
from .kernels import make_c_star_matrix, make_c_star_diag


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
    t = np.asarray(theta_raw).ravel()
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
        lat_params.append((float(np.exp(log_s2)), np.exp(log_ell)))
    off = base

    Psi = None
    if learn_Psi:
        need = p * q
        if t.size < off + need:
            raise ValueError(f"theta length {t.size} < required latent+Psi {off+need}")
        Psi_free = t[off:off + need].reshape(p, q)
        off += need

        Psi = Psi_free.copy()
        if normalize_cols:
            for j in range(q):
                nrm = norm(Psi[:, j]) or 1.0 # prevent divide by zero 
                Psi[:, j] /= nrm

    sigma_eps2 = None
    if learn_sigma_eps:
        need = p
        if t.size < off + need:
            raise ValueError(f"theta length {t.size} < required +Sigma_eps {off+need}")
        log_sigma = t[off:off + need]
        off += need
        sigma_eps2 = np.exp(log_sigma).astype(float)

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

class MOOGP:
    """
    Multi-output Orthogonal Gaussian Process Model
    """

    def __init__(self,
                 terms,
                 q,
                 Psi=None,
                 *,
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
                 store_dense_ky=False):
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
            In this mode ``Phi`` (and thus ``Psi``) is rebuilt from SVD of
            ``Y^T Y`` each likelihood call. If a fixed ``Psi`` is incompatible
            with this parameterization, the model falls back to slow mode.
        use_slow_kyinv : bool
            Escape hatch for regression/debugging. If True, force the dense
            covariance/cholesky path.
        store_dense_ky : bool
            If True, keep dense ``Ky`` in cache for debugging.
        """
        self.terms = terms
        self.q = q
        self.Psi = None if Psi is None else np.asarray(Psi, float)
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

        self._data = None
        self.cache = None
        self.theta_hat = None
        self.nll_hat = None
        self.opt_result = None
        self.fitted = False
        self.Ky_inv_rvec_ = None
    
    def _prepare_data(self, data):
        X = np.asarray(data["X_scaled"])
        Y = np.asarray(data.get("Y", data.get("y")))
        n, d = X.shape
        p = Y.shape[1]

        self._data = data
        self.X = X
        self.Y = Y
        self.n = n
        self.d = d
        self.p = p
        self.Ky_inv_rvec_ = None

        if self.Psi is not None:
            if self.Psi.shape != (p, self.q):
                raise ValueError(f"Psi shape {self.Psi.shape} ≠ (p={p}, q={self.q})")

        # Cache SVD
        if (self.sigma_eps2 is not None) and (self.sigma_eps2.size != p):
            raise ValueError(f"sigma_eps2 length {self.sigma_eps2.size} ≠ p={p}")
        
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
        rhs = np.asarray(rhs, float)
        was_vec = (rhs.ndim == 1)
        rhs2 = rhs[:, None] if was_vec else rhs

        n = info["n"]
        p = info["p"]
        sigma_eps2 = info["sigma_eps2"]
        psi_c = info["psi_c"]
        Q_list = info["Q_list"]

        if rhs2.shape[0] != n * p:
            raise ValueError(f"rhs has incompatible leading dimension {rhs2.shape[0]} (expected {n*p}).")

        # Un-vectorize: each RHS column becomes an (n x p) matrix
        Xrhs = rhs2.reshape(n, p, rhs2.shape[1], order="F")

        # 1. Base measurement noise application: (Sigma_eps^{-1} \otimes I_n)
        # Broadcasting sigma_eps2 across the n and m dimensions
        out = Xrhs / sigma_eps2[None, :, None]

        # 2. Subtract Latent GP Components
        # mathematically: \sum_k [ Q_k @ M @ P_k ]
        # Since P_k = psi_c[k] psi_c[k]^T, we optimize M @ P_k to (M @ psi_c[k]) @ psi_c[k]^T
        for k in range(len(Q_list)):
            uk = psi_c[k]  # shape (p,)
            Qk = Q_list[k] # shape (n, n)
            
            # Step A: Project outputs into the latent space -> shape (n, m)
            M_uk = np.sum(Xrhs * uk[None, :, None], axis=1)
            
            # Step B: Apply spatial covariance inversion -> shape (n, m)
            Qk_M_uk = Qk @ M_uk
            
            # Step C: Project back to the output space and subtract -> shape (n, p, m)
            out -= Qk_M_uk[:, None, :] * uk[None, :, None]

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
        
    def _nll(self, theta_raw):
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

        Psi = Psi_th if self.learn_Psi else self.Psi
        sigma_eps2 = sigma_eps2_th if self.learn_sigma_eps else self.sigma_eps2

        if (Psi is None) and (not use_fast):
            raise ValueError("Psi is None. Provide Psi or set learn_Psi=True.")

        if sigma_eps2 is None:
            sigma_eps2 = np.zeros(p, dtype=float)
        sigma_eps2 = np.maximum(np.asarray(sigma_eps2, float).ravel(), self.min_sigma_eps2)

        # Design
        G = self.G
        r = self.r
        Gy = self.Gy

        # Covariance per latent: Cj_list[k] corresponds to manuscript C_k^*.
        Cj_list = []
        for (sigma2_j, ell_j) in lat_params:
            Cj = make_c_star_matrix(X, X, ell=ell_j, sigma2=sigma2_j,
                                    terms=terms, one_based=self.one_based)
            Cj_list.append(Cj)

        fast_diag_info = None
        Ky = None
        Ky_chol = None

        if use_fast:
            # 1. Phi from SVD of Y
            Phi = self.Phi_fast
            d_vals = self.d_vals_fast

            # 2. Standardized output basis (q,p)
            psi_c = Phi.T / np.sqrt(sigma_eps2)

            # 3. Precompute Q_k and determinant
            Q_list =  []
            logdetK = n * np.sum(np.log(sigma_eps2))

            for k in range(self.q):
                Ck_star = Cj_list[k]
                Dk = d_vals[k]

                # Eigendecomp of C_k*
                Wk, Uk = np.linalg.eigh(Ck_star)

                Wk = np.maximum(Wk, 1e-10)

                # sum(log|I_n + d_k C_k*|) - sum of log eigenvalues equals log det
                logdetK += np.sum(np.log(1.0 + Dk * Wk))

                # Qk = (d_k I_n + (C_k*)^-1)^-1
                inner_diag = 1.0 / (Dk + 1.0 / Wk)
                Qk = (Uk * inner_diag) @ Uk.T
                Q_list.append(Qk)

            fast_diag_info = dict(
                n=n,
                p=p,
                Q_list=Q_list,
                psi_c=psi_c,
                sigma_eps2=sigma_eps2
            )

            solve_Ky = lambda rhs: self._apply_Ky_inv_fast(rhs, fast_diag_info)
            Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi
        else:
            Ky = build_Ky(Cj_list, Psi, sigma_eps2=sigma_eps2)
            if self.jitter:
                Ky = Ky + self.jitter * np.eye(n * p)
            Ky_chol = cho_factor(Ky, lower=True, check_finite=False)
            logdetK = 2.0 * np.sum(np.log(np.diag(Ky_chol[0])))
            solve_Ky = lambda rhs: cho_solve(Ky_chol, rhs, check_finite=False)

        vecY = vecF(Y)
        if r > 0:        
            z = solve_Ky(Gy)
            alpha = solve_Ky(vecY)

            A_gls = Gy.T @ z
            b_gls = Gy.T @ alpha
            beta_vec = solve(A_gls, b_gls, assume_a="sym")

            bhat = unvecF(beta_vec, r, p)
            rvec = vecY - Gy @ vecF(bhat)
            Ky_inv_rvec = alpha - z @ beta_vec
        else:
            bhat = np.zeros((0, p))
            rvec = vecY
            Ky_inv_rvec = solve_Ky(rvec)
    
        qf = float(rvec @ Ky_inv_rvec)
        nll = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi))

        if self.use_reml:
            A = None  # you can fill in REML here later
        else:
            A = None

        # cache for predictions
        self.cache = dict(
            Ky=(Ky if self.store_dense_ky else None),
            Ky_chol=Ky_chol,
            Cj_list=Cj_list,
            Psi=Psi,
            sigma_eps2=sigma_eps2,
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

        res = minimize(
            obj,
            np.asarray(theta0),
            method="L-BFGS-B",
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
                                     terms=terms, one_based=one_based)
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

        # LAZY EVALUATION: Stop here if standard deviation is not requested
        if not return_std:
            return mean

        # 3. VARIANCE PREDICTION 
        
        # We only build this massive matrix if variance is explicitly requested
        K_XsX = build_cross_K(Psi, Cj_XsX) 

        # FAST PRIOR VARIANCE: Compute diagonal analytically without building K_XsXs
        diag_prior = np.zeros(nstar * p)
        for j, (sigma2_j, ell_j) in enumerate(lat_params):
            # Compute only diagonal of test-test matrix
            Cj2_diag = make_c_star_diag(Xs, ell=ell_j, sigma2=sigma2_j,
                                        terms=terms, one_based=one_based)
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
        
        return mean, std