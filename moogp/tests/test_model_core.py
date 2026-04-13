import numpy as np
from numpy.linalg import cholesky

from moogp.design import build_Gy, vecF
from moogp.model import (
    MOOGP,
    _apply_factorized_qk,
    _block_design_matvec,
    _block_design_t_matmul,
    _profiled_gls_terms,
    build_Ky,
    gls_bhat,
    init_phi,
    unpack_theta,
)

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

    M = rng.normal(size=(n * p, 4))
    KM_general = model_general._solve_with_cached_Ky(M)
    KM_fast = model_fast._solve_with_cached_Ky(M)
    assert np.allclose(KM_fast, KM_general, rtol=1e-8, atol=1e-10)

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

    fast_info = model.cache["fast_diag_info"]
    assert fast_info is not None
    assert model.cache["Ky"] is None
    assert model.cache["Ky_inv_rvec"].shape == (n * p,)
    assert np.allclose(model.Ky_inv_rvec_, model.cache["Ky_inv_rvec"])
    assert "Q_list" not in fast_info
    assert len(fast_info["latent_factors"]) == q
    for factor in fast_info["latent_factors"]:
        assert factor["psi"].shape == (p,)
        assert factor["chol"].shape == (n, n)
        assert np.isscalar(factor["d"]) or np.ndim(factor["d"]) == 0


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


def test_block_design_helpers_match_explicit_kron_products():
    rng = np.random.default_rng(21)
    n, r, p = 7, 3, 4
    G = rng.normal(size=(n, r))
    Gy = build_Gy(G, p)

    vec_rhs = rng.normal(size=n * p)
    mat_rhs = rng.normal(size=(n * p, r * p + 2))
    beta_vec = rng.normal(size=r * p)

    assert np.allclose(_block_design_t_matmul(G, vec_rhs, p), Gy.T @ vec_rhs)
    assert np.allclose(_block_design_t_matmul(G, mat_rhs, p), Gy.T @ mat_rhs)
    assert np.allclose(_block_design_matvec(G, beta_vec, p), Gy @ beta_vec)


def test_apply_factorized_qk_matches_explicit_solve():
    rng = np.random.default_rng(31)
    n, d, p, q = 8, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = rng.normal(size=(n, p))
    sigma_eps2 = np.array([0.25, 0.35, 0.45], dtype=float)
    Phi, d_vals = init_phi(Y, q, n)
    Psi = np.diag(np.sqrt(sigma_eps2)) @ Phi

    theta = []
    for _ in range(q):
        theta.append(np.log(1.0))
        theta.extend([np.log(0.6)] * d)
    theta = np.asarray(theta, float)

    model = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=Psi,
        learn_Psi=False,
        sigma_eps2=sigma_eps2,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=True,
    )
    model._prepare_data({"X_scaled": X, "Y": Y})
    model._nll(theta)

    rhs = rng.normal(size=(n, 5))
    factor = model.cache["fast_diag_info"]["latent_factors"][0]
    C = model.cache["Cj_list"][0]
    d_val = factor["d"]
    A = np.eye(n) + d_val * C

    explicit = np.linalg.solve(A, C @ rhs)
    factorized = _apply_factorized_qk(factor["chol"], d_val, rhs)
    assert np.allclose(factorized, explicit, rtol=1e-8, atol=1e-10)


def test_profiled_gls_terms_match_dense_formulas_and_non_caching_path():
    rng = np.random.default_rng(41)
    n, d, p, q = 9, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = rng.normal(size=(n, p))
    data = {"X_scaled": X, "Y": Y}

    model = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=None,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=1e-6,
        one_based=True,
        use_diagonalized_interaction=True,
        standardize_y="zscore",
    )
    model._prepare_data(data)

    theta = []
    for _ in range(q):
        theta.append(np.log(1.0))
        theta.extend([np.log(0.7)] * d)
    theta.extend([np.log(0.2)] * p)
    theta = np.asarray(theta, float)

    nll_cached = model._nll(theta)
    cache_before = model.cache
    ky_inv_before = model.Ky_inv_rvec_.copy()

    vecY = vecF(model.Y)
    Gy = model.Gy
    solve_Ky = model._solve_with_cached_Ky
    qf, bhat, rvec, ky_inv_rvec = _profiled_gls_terms(
        solve_Ky,
        model.G,
        Gy,
        vecY,
        p,
        build_cache=True,
    )

    z = solve_Ky(Gy)
    alpha = solve_Ky(vecY)
    A_gls = Gy.T @ z
    b_gls = Gy.T @ alpha
    beta_vec = np.linalg.solve(A_gls, b_gls)
    dense_bhat = beta_vec.reshape((model.r, p), order="F")
    dense_rvec = vecY - Gy @ beta_vec
    dense_ky_inv_rvec = alpha - z @ beta_vec
    dense_qf = np.dot(dense_rvec, dense_ky_inv_rvec)

    assert np.allclose(qf, dense_qf, rtol=1e-8, atol=1e-10)
    assert np.allclose(bhat, dense_bhat, rtol=1e-8, atol=1e-10)
    assert np.allclose(rvec, dense_rvec, rtol=1e-8, atol=1e-10)
    assert np.allclose(ky_inv_rvec, dense_ky_inv_rvec, rtol=1e-8, atol=1e-10)

    nll_no_cache = model._nll(theta, build_cache=False)
    assert np.isclose(nll_no_cache, nll_cached, rtol=1e-8, atol=1e-10)
    assert model.cache is cache_before
    assert np.allclose(model.Ky_inv_rvec_, ky_inv_before, rtol=1e-12, atol=1e-12)
