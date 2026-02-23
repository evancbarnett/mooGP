import numpy as np
from numpy.linalg import cholesky

from moogp.model import MOOGP, build_Ky, gls_bhat, unpack_theta, init_phi

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


def test_diagonalized_interaction_fast_matches_general():
    rng = np.random.default_rng(123)
    n, d, p, q = 9, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = rng.normal(size=(n, p))
    data = {"X_scaled": X, "Y": Y}

    terms = [None, 1, 2]
    sigma_eps2 = np.array([0.2, 0.35, 0.5], dtype=float)
    
    Phi, d_vals = init_phi(Y, q, n)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi

    theta = []
    for _ in range(q):
        theta.append(np.log(1.0))
        theta.extend([np.log(0.7)] * d)
    theta = np.asarray(theta, float)

    model_general = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=False,
    )
    model_general._prepare_data(data)
    nll_general = model_general._nll(theta)

    model_fast = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=True,
    )
    model_fast._prepare_data(data)
    nll_fast = model_fast._nll(theta)

    assert np.isclose(nll_fast, nll_general, rtol=1e-8, atol=1e-10)

    v = rng.normal(size=n * p)
    Kv_general = model_general._solve_with_cached_Ky(v)
    Kv_fast = model_fast._solve_with_cached_Ky(v)
    assert np.allclose(Kv_fast, Kv_general, rtol=1e-8, atol=1e-10)

    model_general.fitted = True
    model_fast.fitted = True
    mean_g, std_g = model_general.predict(X, return_std=True)
    mean_f, std_f = model_fast.predict(X, return_std=True)
    assert np.allclose(mean_f, mean_g, rtol=1e-8, atol=1e-10)
    assert np.allclose(std_f, std_g, rtol=1e-8, atol=1e-10)


def test_fast_path_is_default_and_caches_kyinv_rvec():
    rng = np.random.default_rng(7)
    n, d, p, q = 10, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = rng.normal(size=(n, p))
    data = {"X_scaled": X, "Y": Y}

    terms = [None, 1, 2]
    sigma_eps2 = np.array([0.2, 0.4, 0.6], dtype=float)
    Phi, d_vals = init_phi(Y, q, n)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi

    theta = []
    for _ in range(q):
        theta.append(np.log(1.0))
        theta.extend([np.log(0.8)] * d)
    theta = np.asarray(theta, float)

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
    )
    model._prepare_data(data)
    model._nll(theta)

    assert model.cache["fast_diag_info"] is not None
    assert model.cache["Ky"] is None
    assert model.cache["Ky_inv_rvec"].shape == (n * p,)
    assert np.allclose(model.Ky_inv_rvec_, model.cache["Ky_inv_rvec"])


def test_predict_mean_uses_cached_kyinv_rvec():
    rng = np.random.default_rng(19)
    n, d, p, q = 8, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = rng.normal(size=(n, p))
    data = {"X_scaled": X, "Y": Y}

    terms = [None, 1, 2]
    sigma_eps2 = np.array([0.25, 0.3, 0.55], dtype=float)
    
    Phi, d_vals = init_phi(Y, q, n)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi

    theta = []
    for _ in range(q):
        theta.append(np.log(1.0))
        theta.extend([np.log(0.9)] * d)
    theta = np.asarray(theta, float)

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
    )
    model._prepare_data(data)
    model._nll(theta)
    model.fitted = True

    def _raise_if_called(_rhs):
        raise RuntimeError("solver should not be called for mean-only predict")

    model._solve_with_cached_Ky = _raise_if_called
    pred = model.predict(X, return_std=False)
    assert pred.shape == Y.shape
