"""Experimental OILMM-style grouped-projection path for MOOGP-shape models.

This is a *standalone* alternative training/inference path for the same
multi-output orthogonal GP family the existing :class:`moogp.model.MOOGP`
fits. It does not modify any existing module, kernel, or design helper —
it only imports them.

Mathematical idea (see ``the private grouped-projection research note`` for a longer note).
Under the fast MOOGP parameterisation

    Psi = Sigma_eps^{1/2} Phi,   Phi^T Phi = D = diag(d_1, ..., d_q),

the joint (vec-shaped) Gaussian likelihood of ``Y - GB`` factorises as

    NLL_joint(Y; B, theta_1..q, Sigma_eps, Phi, D)
       =  sum_k NLL_k( t_k ;  theta_k, b_proj_k )
       +  L_complement(B, Sigma_eps, Phi, D),

where

    T = (Y - GB) * Sigma_eps^{-1/2} * Phi * D^{-1}    in R^{n x q}

and each scalar likelihood ``NLL_k`` is a textbook univariate GP regression
with kernel ``C*_k(theta_k) + (1/d_k) I_n`` and design ``G``. The complement
term is closed form and depends only on the trend ``B`` and on the fixed
``(Sigma_eps, Phi, D)``.

This module holds ``(Sigma_eps, Phi, D)`` *fixed* at initialisation and
fits the q latent kernel hyperparameters by q independent scalar
optimisations — the OILMM-style "never do the joint" approach — without
touching the existing :class:`MOOGP` pipeline.

The implementation deliberately reuses the orthogonalised SE kernel
``make_c_star_matrix`` and the regression-design helpers ``make_G`` /
``vecF`` from :mod:`moogp.kernels` and :mod:`moogp.design`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .design import make_G, unvecF, vecF
from .kernels import make_c_star_diag, make_c_star_matrix
from .model import init_phi


__all__ = [
    "GroupedProjection",
    "GroupedProjectionMOOGP",
]


# ---------------------------------------------------------------------------
# (Sigma_eps, Phi, D) projection container
# ---------------------------------------------------------------------------


class GroupedProjection:
    """Container for the fixed ``(Sigma_eps, Phi, D)`` projection.

    The projection acts on a residual matrix ``R = Y - GB`` of shape ``(n, p)``
    via

        T = R * Sigma_eps^{-1/2} * Phi * D^{-1}        (n x q)

    Whitened-space decomposition:

        W      = R * Sigma_eps^{-1/2}                  (n x p)
        W_proj = T * Phi^T                             (n x p, in span(Phi))
        W_perp = W - W_proj                            (n x p, orthogonal complement)

    so that ``||W||_F^2 = sum_k d_k * ||T[:,k]||^2 + ||W_perp||_F^2`` exactly.

    Parameters
    ----------
    sigma_eps2 : (p,) array
        Per-output observation noise variance. Must be strictly positive.
    Phi : (p, q) array
        Loading matrix with ``Phi^T Phi`` diagonal. The diagonal must equal
        ``d_vals``.
    d_vals : (q,) array
        Diagonal entries of ``D = Phi^T Phi``. Must be strictly positive.
    diag_rtol : float, default 1e-8
        Tolerance for the ``Phi^T Phi`` diagonality check.
    """

    def __init__(self, sigma_eps2, Phi, d_vals, *, diag_rtol: float = 1e-8):
        sigma_eps2 = np.asarray(sigma_eps2, dtype=float).ravel()
        Phi = np.asarray(Phi, dtype=float)
        d_vals = np.asarray(d_vals, dtype=float).ravel()

        if Phi.ndim != 2:
            raise ValueError(f"Phi must be 2D; got shape {Phi.shape}")
        p, q = Phi.shape

        if sigma_eps2.size != p:
            raise ValueError(
                f"sigma_eps2 size {sigma_eps2.size} must equal p={p} (Phi rows)"
            )
        if d_vals.size != q:
            raise ValueError(
                f"d_vals size {d_vals.size} must equal q={q} (Phi columns)"
            )
        if np.any(~np.isfinite(sigma_eps2)) or np.any(sigma_eps2 <= 0.0):
            raise ValueError("sigma_eps2 must be finite and strictly positive")
        if np.any(~np.isfinite(d_vals)) or np.any(d_vals <= 0.0):
            raise ValueError("d_vals must be finite and strictly positive")
        if np.any(~np.isfinite(Phi)):
            raise ValueError("Phi must be finite")

        gram = Phi.T @ Phi  # (q, q)
        diag = np.diag(gram)
        off = gram - np.diag(diag)
        scale = float(np.max(np.abs(diag))) + 1e-12
        if np.max(np.abs(off)) > diag_rtol * scale:
            raise ValueError(
                "Phi^T Phi must be diagonal; off-diagonal magnitude "
                f"{np.max(np.abs(off)):.2e} exceeds {diag_rtol:.0e} * "
                f"{scale:.2e}."
            )
        diag_err = np.max(np.abs(diag - d_vals))
        if diag_err > diag_rtol * (float(np.max(np.abs(d_vals))) + 1e-12):
            raise ValueError(
                "diag(Phi^T Phi) must equal d_vals; mismatch "
                f"{diag_err:.2e}."
            )

        self.sigma_eps2 = sigma_eps2
        self.Phi = Phi
        self.d_vals = d_vals
        self.p = p
        self.q = q

    # ------------------------------------------------------------------
    # Convenience constructors / matrices
    # ------------------------------------------------------------------

    @classmethod
    def from_y(
        cls,
        Y,
        q: int,
        *,
        sigma_eps2=None,
        residual: Optional[np.ndarray] = None,
        eps_floor: float = 1e-12,
    ) -> "GroupedProjection":
        """Construct from data ``Y`` via the same SVD that ``init_phi`` uses.

        Parameters
        ----------
        Y : (n, p) array
            Working-scale outputs. Used to derive ``Phi, D`` via the SVD,
            matching ``moogp.model.init_phi``.
        q : int
            Number of latent components. Must satisfy ``1 <= q <= min(n, p)``.
        sigma_eps2 : (p,) array or None
            Per-output noise variance. If ``None``, set to per-output sample
            variance of ``residual`` (or of ``Y`` if ``residual`` is None),
            floored at ``eps_floor``.
        residual : (n, p) array or None
            Optional residual ``Y - GB`` to use for the noise estimate.
        eps_floor : float
            Lower clamp on sigma_eps2 when computed from data.
        """
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D; got shape {Y.shape}")
        n, p = Y.shape
        if not (1 <= q <= min(n, p)):
            raise ValueError(
                f"q={q} must satisfy 1 <= q <= min(n, p) = {min(n, p)}"
            )
        Phi, d_vals = init_phi(Y, q, n)

        if sigma_eps2 is None:
            ref = Y if residual is None else np.asarray(residual, dtype=float)
            if ref.shape != (n, p):
                raise ValueError(
                    f"residual shape {ref.shape} must equal Y shape {(n, p)}"
                )
            ddof = 1 if n > 1 else 0
            sigma_eps2 = np.maximum(np.var(ref, axis=0, ddof=ddof), eps_floor)
        return cls(sigma_eps2, Phi, d_vals)

    @property
    def Psi(self) -> np.ndarray:
        """``Psi = Sigma_eps^{1/2} Phi`` (p x q)."""
        return np.sqrt(self.sigma_eps2)[:, None] * self.Phi

    @property
    def projection_matrix(self) -> np.ndarray:
        """``Sigma_eps^{-1/2} Phi D^{-1}`` (p x q); ``T = R @ projection_matrix``."""
        return (self.Phi / np.sqrt(self.sigma_eps2)[:, None]) / self.d_vals[None, :]

    @property
    def reconstruction_matrix(self) -> np.ndarray:
        """Same as ``Psi``. ``R_proj = T @ reconstruction_matrix.T``."""
        return self.Psi

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def _check_residual(self, R) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if R.ndim != 2 or R.shape[1] != self.p:
            raise ValueError(
                f"residual must have shape (n, {self.p}); got {R.shape}"
            )
        return R

    def project(self, R) -> np.ndarray:
        """Map residual ``R`` (n x p) to projected latents ``T`` (n x q)."""
        R = self._check_residual(R)
        return R @ self.projection_matrix

    def reconstruct(self, T) -> np.ndarray:
        """Inverse direction: ``T @ Psi^T`` (n x p) — projected component only."""
        T = np.asarray(T, dtype=float)
        if T.ndim != 2 or T.shape[1] != self.q:
            raise ValueError(f"T must have shape (n, {self.q}); got {T.shape}")
        return T @ self.reconstruction_matrix.T

    def whitened(self, R) -> np.ndarray:
        """``W = R * Sigma_eps^{-1/2}`` (n x p)."""
        R = self._check_residual(R)
        return R / np.sqrt(self.sigma_eps2)[None, :]

    def complement_residual_white(self, R) -> np.ndarray:
        """``W_perp = W - T @ Phi^T`` (n x p) — whitened complement."""
        R = self._check_residual(R)
        T = self.project(R)
        W = self.whitened(R)
        return W - T @ self.Phi.T

    def complement_log_density_term(self, R) -> float:
        """Closed-form complement contribution to the joint NLL:

            L_complement = 1/2 * [ n * sum_l log sigma_l^2
                                + n * sum_k log d_k
                                + ||W_perp||_F^2
                                + n * (p - q) * log(2 pi) ]
        """
        R = self._check_residual(R)
        n = R.shape[0]
        W_perp = self.complement_residual_white(R)
        frob_sq = float(np.sum(W_perp ** 2))
        return 0.5 * (
            float(n) * float(np.sum(np.log(self.sigma_eps2)))
            + float(n) * float(np.sum(np.log(self.d_vals)))
            + frob_sq
            + float(n) * float(self.p - self.q) * float(np.log(2.0 * np.pi))
        )


# ---------------------------------------------------------------------------
# Per-latent scalar GP NLL helper
# ---------------------------------------------------------------------------


def _scalar_orthogonal_gp_nll(
    theta_k,
    *,
    X,
    t_k,
    G,
    d_k: float,
    terms,
    orthogonal: bool,
    one_based: bool,
    sqdist,
    jitter: float,
):
    """NLL for a single scalar orthogonal GP regression on ``t_k``.

    Model: ``t_k ~ N(G @ b, C*_k(theta_k) + (1/d_k) I_n)``, ``b`` profiled out
    via GLS. ``theta_k = [log sigma2_k, log ell_k_1, ..., log ell_k_d]``.
    """
    n, d = X.shape
    if theta_k.shape != (1 + d,):
        raise ValueError(
            f"theta_k must have shape (1 + d,) = ({1+d},); got {theta_k.shape}"
        )
    sigma2 = float(np.exp(theta_k[0]))
    ell = np.exp(np.asarray(theta_k[1:1 + d], dtype=float))
    Ck = make_c_star_matrix(
        X, X, ell=ell, sigma2=sigma2, terms=terms,
        orthogonal=orthogonal, one_based=one_based, sqdist=sqdist,
    )
    K = Ck + (1.0 / d_k) * np.eye(n)
    if jitter:
        K = K + jitter * np.eye(n)
    cf = cho_factor(K, lower=True, check_finite=False)
    L = cf[0]
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))

    if G is None or G.size == 0:
        alpha = cho_solve(cf, t_k, check_finite=False)
        qf = float(t_k @ alpha)
    else:
        rhs = np.column_stack([G, t_k[:, None]])
        sol = cho_solve(cf, rhs, check_finite=False)
        z = sol[:, :-1]
        alpha = sol[:, -1]
        Agls = G.T @ z
        bgls = G.T @ alpha
        beta = np.linalg.solve(Agls, bgls)
        qf = float(t_k @ alpha - bgls @ beta)
    return 0.5 * (log_det + qf + n * float(np.log(2.0 * np.pi)))


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


@dataclass
class _LatentDiagnostics:
    success: bool
    nit: int
    nfev: int
    message: str = ""


class GroupedProjectionMOOGP:
    """OILMM-style training of a MOOGP-shape model via fixed projection.

    Parameters
    ----------
    terms : list
        Regression-basis specification (same convention as
        :func:`moogp.design.make_G`). Use ``[]`` for no trend.
    q : int
        Number of latent components. Must satisfy ``1 <= q <= min(n, p)``
        once data is supplied.
    orthogonal : bool, default True
        Use the orthogonalised SE kernel ``C*_k`` from
        :func:`moogp.kernels.make_c_star_matrix`. If False, falls back to the
        bare SE kernel.
    one_based : bool, default True
        Index convention forwarded to ``make_G`` and ``make_c_star_matrix``.
    jitter : float, default 1e-6
        Diagonal jitter added to each per-latent ``K_proj_k`` for numerical
        stability.
    sigma_eps2 : (p,) array or None
        Optional override for the per-output noise variance held fixed
        during fit. If None, computed from the OLS residual variance per
        output.

    Attributes (after ``fit``)
    --------------------------
    projection : :class:`GroupedProjection`
        Frozen ``(Sigma_eps, Phi, D)`` used by the fit.
    theta_hat_per_latent : (q, 1+d) array
        Per-latent ``[log sigma2, log ell_1, ..., log ell_d]`` at the optimum.
    b_proj_hat : (r, q) array
        Per-latent profiled trend coefficients (zero up to numerical noise
        when an OLS trend has already been removed).
    nll_hat : float
        Total grouped NLL = sum of per-latent NLLs + complement.
    """

    def __init__(
        self,
        terms,
        q: int,
        *,
        orthogonal: bool = True,
        one_based: bool = True,
        jitter: float = 1e-6,
        sigma_eps2=None,
        beta_method: str = "gls",
    ):
        if int(q) < 1:
            raise ValueError(f"q must be >= 1; got {q}")
        if beta_method not in {"ols", "gls"}:
            raise ValueError(
                f"beta_method must be 'ols' or 'gls'; got {beta_method!r}"
            )
        self.terms = list(terms) if terms is not None else []
        self.q = int(q)
        self.orthogonal = bool(orthogonal)
        self.one_based = bool(one_based)
        self.jitter = float(jitter)
        self.beta_method = beta_method
        self._sigma_eps2_user = (
            None if sigma_eps2 is None
            else np.asarray(sigma_eps2, dtype=float).ravel()
        )
        # Filled by _prepare / fit:
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.G: Optional[np.ndarray] = None
        self.B_ols: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.projection: Optional[GroupedProjection] = None
        self._train_sqdist: Optional[np.ndarray] = None
        # Filled by fit:
        self.theta_hat_per_latent: Optional[np.ndarray] = None
        self.b_proj_hat: Optional[np.ndarray] = None
        self.B_gls: Optional[np.ndarray] = None
        self.B_hat: Optional[np.ndarray] = None
        self.nll_per_latent_: Optional[np.ndarray] = None
        self.complement_nll_: Optional[float] = None
        self.nll_hat: Optional[float] = None
        self.per_latent_diag_: list = []
        self.fitted: bool = False

    # ------------------------------------------------------------------

    def _prepare(self, data) -> None:
        if "X_scaled" not in data:
            raise KeyError("data must contain 'X_scaled'")
        X = np.asarray(data["X_scaled"], dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X_scaled must be 2D; got shape {X.shape}")
        if "Y" in data:
            Y = np.asarray(data["Y"], dtype=float)
        elif "y" in data:
            Y = np.asarray(data["y"], dtype=float)
        else:
            raise KeyError("data must contain 'Y' or 'y'")
        if Y.ndim != 2:
            raise ValueError(f"Y must be 2D; got shape {Y.shape}")
        n, d = X.shape
        if Y.shape[0] != n:
            raise ValueError(
                f"X_scaled and Y row counts disagree: {n} vs {Y.shape[0]}"
            )
        p = Y.shape[1]
        if not (1 <= self.q <= min(n, p)):
            raise ValueError(
                f"q={self.q} must satisfy 1 <= q <= min(n, p) = {min(n, p)}"
            )

        # OLS trend removal (per-output independent regression on G).
        G = make_G(
            {"X_scaled": X}, self.terms,
            one_based=self.one_based, return_names=False,
        )
        if G.size:
            B_ols, *_ = np.linalg.lstsq(G, Y, rcond=None)
            R = Y - G @ B_ols
        else:
            B_ols = np.zeros((0, p))
            R = Y.copy()

        if self._sigma_eps2_user is not None:
            sigma_eps2 = np.asarray(self._sigma_eps2_user, dtype=float).ravel()
            if sigma_eps2.size != p:
                raise ValueError(
                    f"sigma_eps2 size {sigma_eps2.size} != p={p}"
                )
        else:
            ddof = 1 if n > 1 else 0
            sigma_eps2 = np.maximum(np.var(R, axis=0, ddof=ddof), 1e-12)

        # Phi, D from SVD of the residual R, matching init_phi semantics
        # (no centering — init_phi is itself uncentered).
        Phi, d_vals = init_phi(R, self.q, n)
        projection = GroupedProjection(sigma_eps2, Phi, d_vals)

        diff = X[:, None, :] - X[None, :, :]
        self.X = X
        self.Y = Y
        self.G = G
        self.B_ols = B_ols
        self.R = R
        self.projection = projection
        self._train_sqdist = diff * diff

    # ------------------------------------------------------------------

    def fit(
        self,
        data,
        theta0_per_latent=None,
        bounds_per_latent=None,
        optimizer_opts=None,
    ) -> "GroupedProjectionMOOGP":
        """Fit the q decoupled scalar GPs and stamp the fitted state.

        Parameters
        ----------
        data : dict
            Must contain ``'X_scaled'`` (n, d) and ``'Y'`` or ``'y'`` (n, p).
        theta0_per_latent : (q, 1+d) array or None
            Per-latent initial ``[log sigma2, log ell_1, ..., log ell_d]``.
            Default: ``log sigma2 = 0``, ``log ell = log(0.5)``.
        bounds_per_latent : list of length q of lists of (lo, hi), or None
            Per-latent bounds. Default: log sigma2 in [log 1e-3, log 1e3],
            log ell in [log 0.05, log 5.0].
        optimizer_opts : dict
            Forwarded to ``scipy.optimize.minimize`` per-latent.
        """
        self._prepare(data)
        n, d = self.X.shape
        q = self.q

        T = self.projection.project(self.R)  # (n, q)

        if theta0_per_latent is None:
            theta0_per_latent = np.tile(
                np.concatenate([[0.0], np.full(d, np.log(0.5))]),
                (q, 1),
            )
        else:
            theta0_per_latent = np.asarray(theta0_per_latent, dtype=float)
            if theta0_per_latent.shape != (q, 1 + d):
                raise ValueError(
                    f"theta0_per_latent shape {theta0_per_latent.shape} != "
                    f"({q}, {1 + d})"
                )

        if bounds_per_latent is None:
            default_one = [
                (float(np.log(1e-3)), float(np.log(1e3)))
            ] + [(float(np.log(0.05)), float(np.log(5.0)))] * d
            bounds_per_latent = [default_one for _ in range(q)]
        else:
            if len(bounds_per_latent) != q:
                raise ValueError(
                    f"bounds_per_latent must have length q={q}; "
                    f"got {len(bounds_per_latent)}"
                )

        opt_opts = {"maxiter": 200}
        if optimizer_opts:
            opt_opts.update(dict(optimizer_opts))

        sqdist = self._train_sqdist
        d_vals = self.projection.d_vals
        theta_hat = np.empty((q, 1 + d), dtype=float)
        nll_per_latent = np.empty(q, dtype=float)
        r_design = self.G.shape[1] if self.G.size else 0
        b_proj_hat = np.zeros((r_design, q), dtype=float)
        diagnostics: list[_LatentDiagnostics] = []

        for k in range(q):
            t_k = T[:, k]
            d_k = float(d_vals[k])

            def obj(tk):
                return _scalar_orthogonal_gp_nll(
                    np.asarray(tk, dtype=float),
                    X=self.X, t_k=t_k, G=self.G, d_k=d_k,
                    terms=self.terms, orthogonal=self.orthogonal,
                    one_based=self.one_based, sqdist=sqdist,
                    jitter=self.jitter,
                )

            res = minimize(
                obj,
                theta0_per_latent[k],
                method="L-BFGS-B",
                bounds=bounds_per_latent[k],
                options=opt_opts,
            )
            theta_hat[k] = res.x
            nll_per_latent[k] = float(res.fun)
            diagnostics.append(_LatentDiagnostics(
                success=bool(res.success),
                nit=int(res.nit),
                nfev=int(res.nfev),
                message=str(res.message) if res.message is not None else "",
            ))

            if r_design > 0:
                sigma2_k = float(np.exp(res.x[0]))
                ell_k = np.exp(res.x[1:1 + d])
                Ck = make_c_star_matrix(
                    self.X, self.X, ell=ell_k, sigma2=sigma2_k,
                    terms=self.terms, orthogonal=self.orthogonal,
                    one_based=self.one_based, sqdist=sqdist,
                )
                Kk = Ck + (1.0 / d_k) * np.eye(n)
                if self.jitter:
                    Kk = Kk + self.jitter * np.eye(n)
                cf = cho_factor(Kk, lower=True, check_finite=False)
                z_G = cho_solve(cf, self.G, check_finite=False)
                z_t = cho_solve(cf, t_k, check_finite=False)
                Agls = self.G.T @ z_G
                bgls = self.G.T @ z_t
                b_proj_hat[:, k] = np.linalg.solve(Agls, bgls)

        L_complement = self.projection.complement_log_density_term(self.R)
        self.theta_hat_per_latent = theta_hat
        self.b_proj_hat = b_proj_hat
        self.nll_per_latent_ = nll_per_latent
        self.complement_nll_ = float(L_complement)
        # Mirror MOOGP._nll's per-row normalization so this NLL stays
        # comparable to ``MOOGP._nll(theta_raw) / n`` parameter-for-parameter.
        self.nll_hat = float(np.sum(nll_per_latent) + L_complement) / float(self.X.shape[0])
        self.per_latent_diag_ = diagnostics
        self.fitted = True

        # If requested, replace the OLS trend B with the joint GLS estimate
        # evaluated at the fitted theta. The latent kernel optima themselves
        # were obtained from OLS-detrended residuals, so this is a one-pass
        # update — sufficient for B-interpretability without iterating back
        # through the per-latent fits. When there is no trend (r_design == 0),
        # the GLS B is trivially the empty (0, p) array — same as OLS.
        if self.beta_method == "gls":
            self.B_gls = self._compute_gls_beta()
            self.B_hat = self.B_gls
        else:
            self.B_gls = None
            self.B_hat = self.B_ols
        return self

    # ------------------------------------------------------------------

    def evaluate_nll(self, theta_per_latent) -> tuple[float, np.ndarray, float]:
        """Evaluate the grouped NLL at user-supplied per-latent parameters.

        Returns ``(total_nll, per_latent_nlls, complement_nll)``. Profiles
        the trend per-latent via GLS at the supplied theta.
        """
        if self.X is None:
            raise RuntimeError(
                "Call fit() (or _prepare(data) for evaluation only) first."
            )
        theta_per_latent = np.asarray(theta_per_latent, dtype=float)
        d = self.X.shape[1]
        if theta_per_latent.shape != (self.q, 1 + d):
            raise ValueError(
                f"theta_per_latent shape {theta_per_latent.shape} != "
                f"({self.q}, {1 + d})"
            )
        T = self.projection.project(self.R)
        sqdist = self._train_sqdist
        d_vals = self.projection.d_vals
        per_lat = np.empty(self.q, dtype=float)
        for k in range(self.q):
            per_lat[k] = _scalar_orthogonal_gp_nll(
                theta_per_latent[k], X=self.X, t_k=T[:, k], G=self.G,
                d_k=float(d_vals[k]), terms=self.terms,
                orthogonal=self.orthogonal, one_based=self.one_based,
                sqdist=sqdist, jitter=self.jitter,
            )
        L_complement = self.projection.complement_log_density_term(self.R)
        # Per-row normalization (matches MOOGP._nll). per_lat and L_complement
        # are returned unscaled so callers can still inspect raw per-block NLLs.
        n_total = float(self.X.shape[0])
        return (float(np.sum(per_lat) + L_complement) / n_total,
                per_lat, float(L_complement))

    # ------------------------------------------------------------------

    def _compute_gls_beta(self) -> np.ndarray:
        """Joint GLS estimate of the trend coefficient matrix ``B`` (Y-space).

        Mirrors the formula MOOGP's fast path uses inside
        :func:`moogp.model._profiled_gls_terms_fast`, evaluated at this
        model's fitted ``theta_hat_per_latent`` and the frozen
        ``(Sigma_eps, Phi, D)``. Unlike the OLS estimate, this uses the
        full ``K_y^{-1}`` weighting through the fast Woodbury form, which
        exploits the kernel-induced row covariance and is generally more
        resilient at small ``n`` and on non-uniform input designs.

        Cost
        ----
        ``q`` Cholesky factorisations of ``A_k = I + d_k C*_k`` (one
        per latent, reusing the kernel build), ``q`` triangular solves
        on the ``(n, r)`` design matrix, plus one ``(rp, rp)`` linear
        solve. Asymptotically ``O(q n^3 + q n^2 r + (rp)^3)`` — i.e. on
        the same order as one MOOGP per-call cost, run once at the end
        of training.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        n = self.X.shape[0]
        d = self.X.shape[1]
        p = self.projection.p
        q = self.q
        G = self.G
        r_design = G.shape[1] if G.size else 0
        if r_design == 0:
            return np.zeros((0, p), dtype=float)

        Y = self.Y
        sigma_eps2 = self.projection.sigma_eps2
        Phi = self.projection.Phi
        d_vals = self.projection.d_vals
        sqdist = self._train_sqdist

        # Build per-latent Cholesky of A_k = I + d_k C*_k at the fitted theta.
        L_list: list[np.ndarray] = []
        for k in range(q):
            theta_k = self.theta_hat_per_latent[k]
            sigma2_k = float(np.exp(theta_k[0]))
            ell_k = np.exp(theta_k[1:1 + d])
            Ck = make_c_star_matrix(
                self.X, self.X, ell=ell_k, sigma2=sigma2_k,
                terms=self.terms, orthogonal=self.orthogonal,
                one_based=self.one_based, sqdist=sqdist,
            )
            A_k = np.eye(n) + d_vals[k] * Ck
            if self.jitter:
                A_k = A_k + self.jitter * np.eye(n)
            L_list.append(np.linalg.cholesky(A_k))

        # psi_tilde[:, k] = Sigma_eps^{-1/2} phi_k = Sigma_eps^{-1} psi_k.
        psi_tilde = Phi / np.sqrt(sigma_eps2)[:, None]  # (p, q)

        # alpha_mat = unvec(K_y^{-1} vec(Y)) using fast Woodbury, in matrix form:
        #   alpha_mat = Y / sigma_eps^2  -  sum_k outer(Q_k (Y psi_tilde_k), psi_tilde_k)
        alpha_mat = Y / sigma_eps2[None, :]
        for k in range(q):
            uk = psi_tilde[:, k]              # (p,)
            d_k = float(d_vals[k])
            L_k = L_list[k]
            v = Y @ uk                        # (n,)
            solved = cho_solve((L_k, True), v, check_finite=False)
            Qk_v = (v - solved) / d_k         # Q_k v
            alpha_mat = alpha_mat - np.outer(Qk_v, uk)

        # b_GLS_vec = vec(G^T alpha_mat).
        b_gls_mat = G.T @ alpha_mat            # (r, p)
        b_gls_vec = vecF(b_gls_mat)            # (rp,)

        # A_GLS = Sigma_eps^{-1} (x) (G^T G)  -  sum_k psi_tilde_k psi_tilde_k^T (x) T_k,
        # with T_k = G^T Q_k G.
        GTG = G.T @ G                          # (r, r)
        A_gls = np.kron(np.diag(1.0 / sigma_eps2), GTG)
        for k in range(q):
            d_k = float(d_vals[k])
            L_k = L_list[k]
            AinvG = cho_solve((L_k, True), G, check_finite=False)  # (n, r)
            QkG = (G - AinvG) / d_k                                 # (n, r)
            T_k = G.T @ QkG                                         # (r, r)
            uk = psi_tilde[:, k]
            A_gls = A_gls - np.kron(np.outer(uk, uk), T_k)

        beta_vec = np.linalg.solve(A_gls, b_gls_vec)
        return unvecF(beta_vec, r_design, p)

    def predict(self, Xstar, *, return_std: bool = False):
        """Predict at ``Xstar`` (n*, d).

        Returns
        -------
        mean : (n*, p) array
            Reconstructed Y-space predictive mean.
        std : (n*, p) array
            Per-output marginal predictive std. Only if ``return_std`` is True.
            Includes observation noise contribution from ``Sigma_eps``.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        Xstar = np.asarray(Xstar, dtype=float)
        if Xstar.ndim != 2 or Xstar.shape[1] != self.X.shape[1]:
            raise ValueError(
                f"Xstar must have shape (n*, {self.X.shape[1]}); got {Xstar.shape}"
            )
        n = self.X.shape[0]
        d = self.X.shape[1]
        nstar = Xstar.shape[0]
        sqdist = self._train_sqdist

        T = self.projection.project(self.R)
        Gs = make_G(
            {"X_scaled": Xstar}, self.terms,
            one_based=self.one_based, return_names=False,
        )

        # Trend reconstruction in Y-space using the active B (OLS or GLS).
        B_active = self.B_hat if self.B_hat is not None else self.B_ols
        if self.G.size and B_active.size:
            mean_y = Gs @ B_active
        else:
            mean_y = np.zeros((nstar, self.projection.p), dtype=float)

        # Per-latent posterior mean and variance at Xstar.
        latent_mean = np.zeros((nstar, self.q), dtype=float)
        latent_var = np.zeros((nstar, self.q), dtype=float)
        for k in range(self.q):
            theta_k = self.theta_hat_per_latent[k]
            sigma2_k = float(np.exp(theta_k[0]))
            ell_k = np.exp(theta_k[1:1 + d])
            d_k = float(self.projection.d_vals[k])
            t_k = T[:, k]

            Ck_train = make_c_star_matrix(
                self.X, self.X, ell=ell_k, sigma2=sigma2_k,
                terms=self.terms, orthogonal=self.orthogonal,
                one_based=self.one_based, sqdist=sqdist,
            )
            K = Ck_train + (1.0 / d_k) * np.eye(n)
            if self.jitter:
                K = K + self.jitter * np.eye(n)
            cf = cho_factor(K, lower=True, check_finite=False)

            Ck_cross = make_c_star_matrix(
                Xstar, self.X, ell=ell_k, sigma2=sigma2_k,
                terms=self.terms, orthogonal=self.orthogonal,
                one_based=self.one_based,
            )

            if self.G.size:
                z_G = cho_solve(cf, self.G, check_finite=False)
                z_t = cho_solve(cf, t_k, check_finite=False)
                Agls = self.G.T @ z_G
                bgls_t = self.G.T @ z_t
                beta_k = np.linalg.solve(Agls, bgls_t)
                resid_t = t_k - self.G @ beta_k
                Kinv_resid = cho_solve(cf, resid_t, check_finite=False)
                gs_mean = (Gs @ beta_k) if Gs.size else np.zeros(nstar)
                latent_mean[:, k] = gs_mean + Ck_cross @ Kinv_resid
            else:
                Kinv_t = cho_solve(cf, t_k, check_finite=False)
                latent_mean[:, k] = Ck_cross @ Kinv_t

            Ck_diag_star = make_c_star_diag(
                Xstar, ell=ell_k, sigma2=sigma2_k,
                terms=self.terms, orthogonal=self.orthogonal,
                one_based=self.one_based,
            )
            v = cho_solve(cf, Ck_cross.T, check_finite=False)  # (n, n*)
            cross_var = np.einsum("nm,nm->m", Ck_cross.T, v)
            latent_var[:, k] = np.maximum(Ck_diag_star - cross_var, 0.0)

        Psi = self.projection.Psi
        mean_y = mean_y + latent_mean @ Psi.T

        if not return_std:
            return mean_y

        # var_y_l(x*) = sum_k Psi[l,k]^2 * var_z_k(x*) + sigma_l^2
        Psi_sq = Psi ** 2
        var_y = latent_var @ Psi_sq.T  # (n*, p)
        var_y = var_y + self.projection.sigma_eps2[None, :]
        std_y = np.sqrt(np.maximum(var_y, 0.0))
        return mean_y, std_y
