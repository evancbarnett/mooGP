import autograd.numpy as np
from autograd.numpy import sqrt, pi, exp
from autograd.scipy.special import erf

from .design import parse_terms_to_index_sets

def M_gauss(x, ell, sigma2=1.0):
    x = np.asarray(x)
    return (sqrt(pi) * sigma2 * ell / 2.0) * (erf((x + 1.0)/ell) - erf((x - 1.0)/ell))

def L_gauss(x, ell, sigma2=1.0):
    x = np.asarray(x)
    return (sigma2 * (ell**2) / 2.0) * (exp(-((x + 1.0)/ell)**2) - exp(-((x - 1.0)/ell)**2)) + x * M_gauss(x, ell, sigma2)

def IM_gauss(ell, sigma2=1.0):
    return 2.0 * sqrt(pi) * sigma2 * ell * erf(2.0/ell) - sigma2 * (ell**2) * (1.0 - exp(-4.0/(ell**2)))

def IL_gauss(ell, sigma2=1.0):
    return 0.0

def ILL_gauss(ell, sigma2=1.0):
    return (sigma2 * (ell**4) / 6.0) * (1.0 - exp(-4.0/(ell**2))) \
         - (sigma2 * (ell**2) / 3.0) * (3.0 - exp(-4.0/(ell**2))) \
         + (2.0 * sqrt(pi) * sigma2 * ell / 3.0) * erf(2.0/ell)

def se_kernel_matrix(X, Xp, ell, sigma2=1.0):
    """Separable squared-exponential kernel.

    Parameters
    ----------
    X, Xp : array_like
        Shapes ``(n,d)`` and ``(m,d)``.
    ell : array_like
        Lengthscales, shape ``(d,)``.
    sigma2 : float
        Marginal variance.

    Returns
    -------
    ndarray
        Kernel matrix shape ``(n,m)``.
    """
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    
    dif = (X[:, None, :] - Xp[None, :, :]) / ell  # (n, m, d)
    D2 = np.sum(dif * dif, axis=2)
    return sigma2 * np.exp(-D2)


def h_matrix_se(X, ell, sigma2, terms, one_based=True):
    r"""Build the *h* design for the orthogonal kernel.

    For regression basis functions of the form
    ``g_i(x) = ∏_{j∈J_i} x_j`` (including the intercept with ``J_i=∅``), the
    orthogonal GP kernel for a stationary product kernel can be written as

    ``c*(x,x') = c(x,x') - h(x)^T H^{-1} h(x')``.

    This function returns the matrix
    ``h(X) = [h(x_1), ..., h(x_n)]^T`` with shape ``(n, r)``.

    Notes
    -----
    This implementation is specialized to:
      * inputs scaled to ``[-1,1]^d``
      * separable squared-exponential kernel
      * Lebesgue reference measure on ``[-1,1]^d``

    Parameters
    ----------
    X : (n,d) array
    ell : (d,) array
    sigma2 : float
    terms : list
        Same encoding as :func:`design.make_G`.
    """
    X = np.asarray(X)
    n, d = X.shape
    

    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based)
    r = len(J_sets)
    if r == 0:
        return np.empty((n, 0), float)

    # Precompute M_j and L_j for all coords (shape: n×d)
    M_all = np.column_stack([M_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])
    L_all = np.column_stack([L_gauss(X[:, j], ell[j], sigma2=1.0) for j in range(d)])

    Hcols_list = []
    all_idx = set(range(d))
    for i, Ji in enumerate(J_sets):
        Ji = set(Ji)
        notJ = list(all_idx - Ji)
        Ji = list(Ji)
        col = np.ones(n)
        if notJ:
            col = col * np.prod(M_all[:, notJ], axis=1)
        if Ji:
            col = col * np.prod(L_all[:, Ji], axis=1)
        Hcols_list.append(col)

    Hcols = np.column_stack(Hcols_list)

    # sigma2 scales the kernel, so it scales both h and H
    return Hcols * sigma2


def H_diag_se(ell, sigma2, terms, one_based=True):
    r"""Return the diagonal of ``H`` for the SE orthogonal kernel.

    Under the same assumptions as :func:`h_matrix_se`, the Gram matrix
    ``H = ∬ g(u) c(u,v) g(v)^T du dv`` is diagonal for the monomial-by-subset
    basis used in the MOOGP/OGP construction.

    Returns
    -------
    ndarray
        Shape ``(r,)``.
    """
    
    d = len(ell)
    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based)
    r = len(J_sets)
    if r == 0:
        return np.empty((0,), float)

    IM = np.stack([IM_gauss(ell[j], sigma2=1.0) for j in range(d)])
    ILL = np.stack([ILL_gauss(ell[j], sigma2=1.0) for j in range(d)])

    Hdiag = []
    all_idx = set(range(d))
    for Ji in J_sets:
        Ji = set(Ji)
        notJ = list(all_idx - Ji)
        Ji = list(Ji)
        val = 1.0
        if notJ:
            val *= np.prod(IM[notJ])
        if Ji:
            val *= np.prod(ILL[Ji])
        Hdiag.append(val)

    Hdiag = np.stack(Hdiag)
    if np.any(Hdiag <= 0):
        raise RuntimeError("Non-positive H diagonal (check ell/domain).")
    return Hdiag * sigma2


def make_c_star_matrix(X, Xp, ell, sigma2, terms, orthogonal=True,one_based=True):
    """Compute the orthogonalized kernel matrix ``C*(X, X')``.

    This is the matrix version of
    ``c*(x,x') = c(x,x') - h(x)^T H^{-1} h(x')``.

    Notes
    -----
    Measurement noise (``Sigma_eps``) is **not** part of this kernel; it is added
    at the observation level via ``K_y = K_f + (Sigma_eps ⊗ I_n)``.
    """
    X = np.asarray(X)
    Xp = np.asarray(Xp)
    d = X.shape[1]

    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based)

    # Base covariance
    C = se_kernel_matrix(X, Xp, ell, sigma2=sigma2)
    if (len(J_sets) == 0) or (not orthogonal):
        return C

    # h(X), h(X')
    hX = h_matrix_se(X, ell, sigma2, terms, one_based=one_based)    # n×r
    hXp = h_matrix_se(Xp, ell, sigma2, terms, one_based=one_based)  # m×r

    # H diagonal and H^{-1}
    Hdiag = H_diag_se(ell, sigma2, terms, one_based=one_based)
    Hinvd = 1.0 / Hdiag

    C_corr = (hX * Hinvd) @ hXp.T
    return C - C_corr

def make_c_star_diag(X, ell, sigma2, terms, orthogonal=True ,one_based=True):
    """Compute ONLY the diagonal of the orthogonalized kernel matrix ``C*(X, X)``.

    This avoids computing the full (n, n) matrix when only the variance
    is needed, dropping memory complexity from O(n^2) to O(n).
    """
    X = np.asarray(X)
    n, d = X.shape

    J_sets = parse_terms_to_index_sets(terms, d, one_based=one_based)

    # 1. Base covariance diagonal (For SE, it is just sigma2 everywhere)
    C_diag = np.full(n, sigma2)
    if (len(J_sets) == 0) or (not orthogonal):
        return C_diag

    # 2. Compute h(X) -> shape (n, r)
    hX = h_matrix_se(X, ell, sigma2, terms, one_based=one_based)

    # 3. Compute H diagonal and H^{-1} -> shape (r,)
    Hdiag = H_diag_se(ell, sigma2, terms, one_based=one_based)
    Hinvd = 1.0 / Hdiag

    # 4. Compute the diagonal of the correction term: diag(hX @ Hinvd @ hX.T)
    # We do this quickly by squaring hX, multiplying by Hinvd, and summing across the rows
    C_corr_diag = np.sum((hX ** 2) * Hinvd, axis=1)

    return C_diag - C_corr_diag