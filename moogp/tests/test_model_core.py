import numpy as np
from numpy.linalg import cholesky

from moogp.model import build_Ky, gls_bhat, unpack_theta

def test_build_Ky_single_latent_matches_kron():
    n = 3
    p = 2
    C = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.4],
        [0.2, 0.4, 1.0],
    ])
    Psi = np.array([
        [1.0],
        [2.0],
    ])  # (p × 1)

    Ky = build_Ky([C], Psi)
    W = Psi @ Psi.T
    Ky_manual = np.kron(W, C)

    assert Ky.shape == (n * p, n * p)
    assert np.allclose(Ky, Ky_manual)

def test_gls_bhat_equals_ols_when_cov_is_identity():
    rng = np.random.default_rng(0)
    n, r, p = 20, 3, 2
    G = rng.normal(size=(n, r))
    true_B = rng.normal(size=(r, p))
    Y = G @ true_B

    Ky = np.eye(n * p)
    chol = cholesky(Ky)
    Ky_chol = (chol, False)  # upper-triangular

    B_hat = gls_bhat(Y, G, Ky_chol)
    B_ols = np.linalg.lstsq(G, Y, rcond=None)[0]

    assert np.allclose(B_hat, B_ols, atol=1e-10)


def test_unpack_theta_basic_splits():
    # d dims, q latents, p outputs
    d, q, p = 3, 2, 4

    # log_sigma2 = 0, log_ell = log(0.5)
    theta_raw = []
    for j in range(q):
        theta_raw += [0.0]               # log sigma2
        theta_raw += list(np.log(0.5) * np.ones(d))  # log ells
    theta_raw = np.array(theta_raw)

    lat_params, Psi, _ = unpack_theta(theta_raw, d=d, q=q, p=p, learn_Psi=False, learn_sigma_eps=False)

    assert len(lat_params) == q
    for (sigma2_j, ell_j) in lat_params:
        assert np.allclose(sigma2_j, 1.0)
        assert ell_j.shape == (d,)
        assert np.allclose(ell_j, 0.5)

    assert Psi is None