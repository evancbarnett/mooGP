import numpy as np
from numpy.linalg import norm
from scipy.linalg import cho_factor, cho_solve, solve
from scipy.optimize import minimize

from .design import make_G, build_Gy, vecF, unvecF
from .kernels import make_c_star_matrix


def unpack_theta(theta_raw, d, q, p, learn_Psi=False, normalize_rows=True):
    """
    flatten theta to 1d array

    r: number of elements in g(x)
    q: number of gaussian processes in g(x)
    p: number of output locations in data
    """
    t = np.asarray(theta_raw).ravel()
    base = q*(d+1) # number of params excluding Psi
    if t.size < base:
        raise ValueError(f"theta length {t.size} < required {base}")

    lat = t[:base]
    lat_params = []
    off = 0
    for j in range(q): 
        log_s2 = lat[off]
        off += 1
        log_ell = lat[off:off+d]
        off += d
        lat_params.append((float(np.exp(log_s2)), np.exp(log_ell)))
    if not learn_Psi:
        return lat_params, None
    
    # learn Psi
    Psi_free = t[base:].reshape(p,q)
    Psi = Psi_free.copy()
    if normalize_rows:
        for j in range(q):
            nrm = norm(Psi[:,j]) or 1.0
            Psi[:,j] /= nrm
    return lat_params, Psi

def build_Ky(Cjs, Psi):
    n = Cjs[0].shape[0]
    p,q = Psi.shape

    assert q == len(Cjs)

    Ky = np.zeros((n*p, n*p))
    for j in range(q):
        Wj = np.outer(Psi[:,j], Psi[:,j])
        Ky += np.kron(Wj, Cjs[j])
    return Ky

def build_cross_K(Psi, Cj_list):
    nstar, m = Cj_list[0].shape
    p,q = Psi.shape
    K = np.zeros((nstar*p, m*p))
    for j in range(q):
        Wj = np.outer(Psi[:,j], Psi[:,j])  # (p×p)
        K += np.kron(Wj, Cj_list[j])
    return K

def gls_bhat(Y, G, Ky_chol):
    """
    \widehat{\vec{\mat{B}}} &= \left((\mat{I}_p \otimes \mat{G})\transpose \mat{K}_y\inv (\mat{I}_p \otimes \mat{G})  \right) \inv (\mat{I}_p \otimes 
    See Eq (8) of Ortho LMC
    Parameters
    ----------
    Y    : (n,p)
    G    : (n,r)
    Ky_chol: (np, np) -> Cov(vec(y))
    
    """

    n, p = Y.shape
    r = G.shape[1] 
    Gy = build_Gy(G,p)

    z = cho_solve(Ky_chol, Gy) # z = K_y \inv (I_p \otimes G)
    alpha = cho_solve(Ky_chol, vecF(Y)) # \alpha = K_y \inv vec(Y)

    A = Gy.T @ z     # = (Ip \otimes G)\transpose K_y \inv (Ip \otimes G)
    b = Gy.T @ alpha # (Ip \otimes G)\transpose K_y \inv vec(Y)

    # Solve A \hat{B} = b
    beta_vec = solve(A, b, assume_a='sym')
    return unvecF(beta_vec,r,p)

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
                 use_reml=False,
                 jitter=1e-6,
                 one_based=True,
                 normalize_rows=True):
        """
        Parameters
        ----------
        terms : list
            Basis specification for g(x) (same as make_G).
        q : int
            Number of latent orthogonal GPs.
        Psi : (p, q) array or None
            Initial / fixed mixing matrix. If learn_Psi=False, this must be provided.
            If learn_Psi=True, it's used for shape.
        learn_Psi : bool
            If True, theta also parameterizes Psi (like your original unpack_theta).
        use_reml : bool
            Placeholder for REML; currently same as ML if True.
        jitter : float
            Diagonal jitter for Ky.
        one_based : bool
            Interpret integers in `terms` as 1-based indices.
        normalize_rows : bool
            Whether to normalize each column of Psi when learning.
        """
        self.terms = terms
        self.q = q
        self.Psi = None if Psi is None else np.asarray(Psi, float)
        self.learn_Psi = learn_Psi
        self.use_reml = use_reml
        self.jitter = jitter
        self.one_based = one_based
        self.normalize_rows = normalize_rows

        self._data = None
        self.cache = None
        self.theta_hat = None
        self.nll_hat = None
        self.opt_result = None
        self.fitted = False
    
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

        if self.Psi is not None:
            if self.Psi.shape != (p, self.q):
                raise ValueError(f"Psi shape {self.Psi.shape} ≠ (p={p}, q={self.q})")

    def _nll(self, theta_raw):
        """
        Negative log-likelihood that optimizes over (sigma^2, ell).

        Parameters
        ----------
        theta_raw : array-like, shape (1 + d,)
            Packed as [log_sigma2, log_ell_1, ..., log_ell_d] (unconstrained).
        """
        X, Y, n, d, p = self.X, self.Y, self.n, self.d, self.p
        terms = self.terms

        # unpack hyperparameters
        if self.learn_Psi:
            if self.Psi is None:
                # we only need Psi shape for unpacking
                Psi_shape = (p, self.q)
            else:
                Psi_shape = self.Psi.shape
            lat_params, Psi = unpack_theta(
                theta_raw, d, self.q, p,
                learn_Psi=True,
                normalize_rows=self.normalize_rows,
            )
        else:
            if self.Psi is None:
                raise ValueError("Provide Psi when learn_Psi=False.")
            lat_params, _ = unpack_theta(
                theta_raw, d, self.q, p,
                learn_Psi=False,
            )
            Psi = self.Psi

        # Design
        G = make_G({"X_scaled": X}, terms, one_based=self.one_based, return_names=False)
        r = G.shape[1] if G.size else 0
        Gy = build_Gy(G, p) if r > 0 else np.empty((n * p, 0))

        # Covariance per latent
        Cj_list = []
        for (sigma2_j, ell_j) in lat_params:
            Cj = make_c_star_matrix(X, X, ell=ell_j, sigma2=sigma2_j,
                                    terms=terms, one_based=self.one_based)
            Cj_list.append(Cj)

        Ky = build_Ky(Cj_list, Psi)
        if self.jitter:
            Ky = Ky + self.jitter * np.eye(n * p)

        Ky_chol = cho_factor(Ky, lower=True, check_finite=False)

        if r > 0:
            bhat = gls_bhat(Y, G, Ky_chol)
            rvec = vecF(Y) - Gy @ vecF(bhat)
        else:
            bhat = np.zeros((0, p))
            rvec = vecF(Y)

        qf = float(rvec @ cho_solve(Ky_chol, rvec, check_finite=False))
        logdetK = 2.0 * np.sum(np.log(np.diag(Ky_chol[0])))

        nll = 0.5 * (logdetK + qf + (n * p) * np.log(2.0 * np.pi))

        if self.use_reml:
            A = None  # you can fill in REML here later
        else:
            A = None

        # cache for predictions
        self.cache = dict(
            Ky=Ky,
            Ky_chol=Ky_chol,
            Cj_list=Cj_list,
            Psi=Psi,
            G=G,
            Gy=Gy,
            bhat=bhat,
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
    
    def predict(self, Xstar, return_std=False, diag_only=True, include_mean_uncertainty=False):
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
        Y = cache["Y"]
        Psi = cache["Psi"]
        lat_params = cache["lat_params"]
        Gy = cache["Gy"]
        bhat = cache["bhat"]
        Ky_chol = cache["Ky_chol"]
        A = cache["A"]
        p = Y.shape[1]

        Xs = np.asarray(Xstar)
        nstar = Xs.shape[0]

        # cross and test covariances per latent
        Cj_XsX = []
        Cj_XsXs = []
        for (sigma2_j, ell_j) in lat_params:
            Cj1 = make_c_star_matrix(Xs, X, ell=ell_j, sigma2=sigma2_j,
                                     terms=terms, one_based=one_based)
            Cj2 = make_c_star_matrix(Xs, Xs, ell=ell_j, sigma2=sigma2_j,
                                     terms=terms, one_based=one_based)
            Cj_XsX.append(Cj1)
            Cj_XsXs.append(Cj2)

        K_XsX = build_cross_K(Psi, Cj_XsX)    # (n* p × n p)
        K_XsXs = build_cross_K(Psi, Cj_XsXs)  # (n* p × n* p)

        # mean
        Gs = make_G({"X_scaled": Xs}, terms, one_based=one_based, return_names=False)
        Gs_y = build_Gy(Gs, p) if Gs.size else np.empty((nstar * p, 0))
        rvec = cache["residual_vec"]

        mean_vec = (Gs_y @ vecF(bhat)) if Gs.size else np.zeros(nstar * p)
        mean_vec += K_XsX @ cho_solve(Ky_chol, rvec, check_finite=False)
        mean = unvecF(mean_vec, nstar, p)

        if not return_std:
            return mean

        # variance (diag_only currently, like your code)
        V = cho_solve(Ky_chol, K_XsX.T, check_finite=False)  # (n p × n* p)
        diag_prior = np.einsum("ii->i", K_XsXs)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag = diag_prior - diag_cross

        if include_mean_uncertainty and Gs.size and use_reml and (A is not None):
            M = Gs_y - K_XsX @ cho_solve(Ky_chol, Gy, check_finite=False)
            W = solve(A, M.T, assume_a="sym")
            diag += np.sum(M * W.T, axis=1)

        std = np.sqrt(np.maximum(diag, 0.0))
        std = unvecF(std, nstar, p)
        return mean, std
