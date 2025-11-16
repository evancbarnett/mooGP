import numpy as np
from scipy.integrate import quad

from moogp.kernels import (
    M_gauss,
    L_gauss,
    IM_gauss,
    IL_gauss,
    ILL_gauss,
    make_c_star_matrix,
)

def test_M_gauss_matches_numeric_integral():
    sigma2 = 1.3
    ell = 0.7

    def k(z, x):
        return sigma2 * np.exp(-((z - x) / ell) ** 2)

    xs = np.linspace(-1.0, 1.0, 5)
    for x in xs:
        num, _ = quad(lambda z: k(z, x), -1.0, 1.0)
        ana = M_gauss(x, ell, sigma2)
        print(f"diff: {ana-num}")
        assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)


def test_L_gauss_matches_numeric_first_moment():
    sigma2 = 0.9
    ell = 0.4

    def k(z, x):
        return sigma2 * np.exp(-((z - x) / ell) ** 2)

    xs = np.linspace(-1.0, 1.0, 5)
    for x in xs:
        num, _ = quad(lambda z: z * k(z, x), -1.0, 1.0)
        ana = L_gauss(x, ell, sigma2)
        print(f"diff: {ana-num}")
        assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)

def test_IM_gauss_matches_numeric_double_integral():
    for ell in [0.2, 0.7, 1.5]:
        sigma2 = 1.5

        def Mx(x):
            return M_gauss(x, ell, sigma2)

        num, _ = quad(Mx, -1.0, 1.0)
        ana = IM_gauss(ell, sigma2)
        assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)


def test_ILL_gauss_matches_numeric_cross_moment(): 
    for ell in [0.2, 0.7, 1.5]:
        sigma2 = 1.5

        def xLx(x):
            return x * L_gauss(x, ell, sigma2)

        num, _ = quad(xLx, -1.0, 1.0)
        ana = ILL_gauss(ell, sigma2)
        assert np.allclose(ana, num, rtol=1e-5, atol=1e-7)



def _grid_interval(n=2001):
    xs = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    return xs


def test_cstar_orthogonal_to_intercept():
    ell = np.array([0.7])
    sigma2 = 1.0
    terms = [None]  # only intercept in g(x)

    X = _grid_interval()
    Cstar = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms)

    # Approximating \int_{-1}^1 c(x,x') via trapezoidal rule
    I0 = np.trapz(Cstar, x=X[:,0], axis=0)
    assert np.max(np.abs(I0)) < 1e-3


def test_cstar_orthogonal_to_intercept_and_linear():
    ell = np.array([0.5])
    sigma2 = 1.0
    terms = [None, 1]  # intercept + linear term in 1D

    X = _grid_interval()
    xs = X[:, 0]

    Cstar = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms)

    # \int c(x,x')
    I0 = np.trapz(Cstar, x=xs, axis=0)
    
    # \int x * c(x,x')
    I1 = np.trapz(Cstar * xs[:, None], x=xs, axis=0)

    assert np.max(np.abs(I0)) < 1e-3
    assert np.max(np.abs(I1)) < 1e-3

