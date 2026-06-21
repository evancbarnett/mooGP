import numpy as np
from scipy.integrate import quad

from moogp.design import parse_terms_to_index_sets
from moogp.kernels import (
    M_gauss,
    L_gauss,
    IM_gauss,
    ILL_gauss,
    make_c_star_matrix,
    se_kernel_matrix,
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
    I0 = np.trapezoid(Cstar, x=X[:,0], axis=0)
    assert np.max(np.abs(I0)) < 1e-3


def test_cstar_orthogonal_to_intercept_and_linear():
    ell = np.array([0.5])
    sigma2 = 1.0
    terms = [None, 1]  # intercept + linear term in 1D

    X = _grid_interval()
    xs = X[:, 0]

    Cstar = make_c_star_matrix(X, X, ell=ell, sigma2=sigma2, terms=terms)

    # \int c(x,x')
    I0 = np.trapezoid(Cstar, x=xs, axis=0)
    
    # \int x * c(x,x')
    I1 = np.trapezoid(Cstar * xs[:, None], x=xs, axis=0)

    assert np.max(np.abs(I0)) < 1e-3
    assert np.max(np.abs(I1)) < 1e-3


def _numeric_h_and_H_full(X_row, ell, sigma2, terms, grid_n=401):
    """Numerically approximate the manuscript h(x) and H matrices."""
    x = np.asarray(X_row, dtype=float).ravel()
    d = x.size
    J_sets = parse_terms_to_index_sets(terms, d, one_based=True)

    h_moments = []
    H_moments = []
    for j in range(d):
        grid = np.linspace(-1.0, 1.0, grid_n)
        kernel_x = np.exp(-((x[j] - grid) / ell[j]) ** 2)
        h0 = np.trapezoid(kernel_x, x=grid)
        h1 = np.trapezoid(grid * kernel_x, x=grid)
        h_moments.append((h0, h1))

        K = np.exp(-((grid[:, None] - grid[None, :]) / ell[j]) ** 2)
        H00 = np.trapezoid(np.trapezoid(K, x=grid, axis=0), x=grid)
        H10 = np.trapezoid(np.trapezoid(grid[:, None] * K, x=grid, axis=0), x=grid)
        H01 = np.trapezoid(np.trapezoid(K * grid[None, :], x=grid, axis=0), x=grid)
        H11 = np.trapezoid(np.trapezoid(grid[:, None] * K * grid[None, :], x=grid, axis=0), x=grid)
        H_moments.append(np.array([[H00, H01], [H10, H11]], dtype=float))

    h = []
    for J in J_sets:
        val = sigma2
        J = set(J)
        for j in range(d):
            val *= h_moments[j][1 if j in J else 0]
        h.append(val)

    H = np.zeros((len(J_sets), len(J_sets)), dtype=float)
    for a, Ja in enumerate(J_sets):
        set_a = set(Ja)
        for b, Jb in enumerate(J_sets):
            set_b = set(Jb)
            val = sigma2
            for j in range(d):
                val *= H_moments[j][int(j in set_a), int(j in set_b)]
            H[a, b] = val

    return np.asarray(h, dtype=float), H


def test_orthogonal_flag_false_returns_standard_kernel():
    rng = np.random.default_rng(11)
    X = rng.uniform(-1.0, 1.0, size=(5, 2))
    Xp = rng.uniform(-1.0, 1.0, size=(4, 2))
    ell = np.array([0.7, 1.1])
    sigma2 = 1.4
    terms = [None, 1, 2, (1, 2)]

    C_plain = make_c_star_matrix(X, Xp, ell=ell, sigma2=sigma2, terms=terms, orthogonal=False)
    C_se = se_kernel_matrix(X, Xp, ell=ell, sigma2=sigma2)

    assert np.allclose(C_plain, C_se, atol=1e-12)


def test_cstar_matches_full_manuscript_definition_in_2d():
    ell = np.array([0.7, 0.9])
    sigma2 = 1.3
    terms = [None, 1, 2, (1, 2)]

    X = np.array([[0.1, -0.2]])
    Xp = np.array([[0.4, 0.3]])

    C_impl = make_c_star_matrix(X, Xp, ell=ell, sigma2=sigma2, terms=terms, orthogonal=True)[0, 0]
    hX, H_full = _numeric_h_and_H_full(X[0], ell, sigma2, terms)
    hXp, _ = _numeric_h_and_H_full(Xp[0], ell, sigma2, terms)
    C_full = se_kernel_matrix(X, Xp, ell=ell, sigma2=sigma2)[0, 0] - hX @ np.linalg.solve(H_full, hXp)

    assert np.allclose(C_impl, C_full, rtol=1e-4, atol=3e-3)
