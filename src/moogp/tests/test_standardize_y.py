import numpy as np

from moogp.model import init_phi
from moogp.model import MOOGP


def _latent_theta(q: int, d: int, sigma2: float = 1.0, ell: float = 0.7) -> np.ndarray:
    theta = []
    for _ in range(q):
        theta.append(np.log(sigma2))
        theta.extend([np.log(ell)] * d)
    return np.asarray(theta, dtype=float)


def test_standardize_y_zscore_matches_manual_scaled_dense_path():
    rng = np.random.default_rng(42)
    n, d, p, q = 8, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y_base = rng.normal(size=(n, p))
    output_scale = np.array([1.0, 7.5, 55.0], dtype=float)
    output_shift = np.array([2.0, -3.0, 10.0], dtype=float)
    Y_raw = Y_base * output_scale + output_shift

    y_center = Y_raw.mean(axis=0)
    y_scale = Y_raw.std(axis=0, ddof=1)
    Y_std = (Y_raw - y_center) / y_scale

    Psi_raw = rng.normal(size=(p, q))
    sigma_eps2_raw = np.array([0.2, 0.8, 5.5], dtype=float)
    Psi_std = Psi_raw / y_scale[:, None]
    sigma_eps2_std = sigma_eps2_raw / (y_scale ** 2)

    theta = _latent_theta(q=q, d=d, sigma2=1.1, ell=0.9)
    Xstar = rng.uniform(-1.0, 1.0, size=(5, d))

    model_auto = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=Psi_raw,
        learn_Psi=False,
        sigma_eps2=sigma_eps2_raw,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=False,
        standardize_y="zscore",
    )
    model_auto._prepare_data({"X_scaled": X, "Y": Y_raw})
    nll_auto = model_auto._nll(theta)
    model_auto.fitted = True

    model_manual = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=Psi_std,
        learn_Psi=False,
        sigma_eps2=sigma_eps2_std,
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=False,
        standardize_y=False,
    )
    model_manual._prepare_data({"X_scaled": X, "Y": Y_std})
    nll_manual = model_manual._nll(theta)
    model_manual.fitted = True

    mean_auto, std_auto = model_auto.predict(Xstar, return_std=True)
    mean_manual, std_manual = model_manual.predict(Xstar, return_std=True)
    mean_manual_raw = mean_manual * y_scale[None, :] + y_center[None, :]
    std_manual_raw = std_manual * y_scale[None, :]

    assert np.allclose(model_auto.y_center_, y_center, atol=1e-12)
    assert np.allclose(model_auto.y_scale_, y_scale, atol=1e-12)
    assert np.allclose(model_auto.cache["Psi_raw"], Psi_raw, atol=1e-12)
    assert np.allclose(model_auto.cache["sigma_eps2_raw"], sigma_eps2_raw, atol=1e-12)
    assert np.isclose(nll_auto, nll_manual, rtol=1e-8, atol=1e-10)
    assert np.allclose(mean_auto, mean_manual_raw, rtol=1e-8, atol=1e-10)
    assert np.allclose(std_auto, std_manual_raw, rtol=1e-8, atol=1e-10)


def test_standardize_y_robust_uses_per_output_median_and_mad():
    X = np.linspace(-1.0, 1.0, 5, dtype=float).reshape(-1, 1)
    Y = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 2.0, 50.0],
            [0.0, 3.0, 1.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=float,
    )

    model = MOOGP(
        terms=[None, 1],
        q=1,
        Psi=np.ones((3, 1), dtype=float),
        learn_Psi=False,
        sigma_eps2=np.ones(3, dtype=float),
        learn_sigma_eps=False,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=False,
        standardize_y="robust",
    )
    model._prepare_data({"X_scaled": X, "Y": Y})

    expected_center = np.median(Y, axis=0)
    expected_scale = np.median(np.abs(Y - expected_center[None, :]), axis=0)
    expected_scale = np.where(expected_scale > 1e-12, expected_scale, 1.0)

    assert np.allclose(model.y_center_, expected_center, atol=1e-12)
    assert np.allclose(model.y_scale_, expected_scale, atol=1e-12)
    assert np.allclose(model.Y, (Y - expected_center[None, :]) / expected_scale[None, :], atol=1e-12)


def test_fast_fit_smoke_handles_cache_scaling_for_raw_and_standardized_outputs():
    rng = np.random.default_rng(7)
    n, d, p, q = 6, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y_base = rng.normal(size=(n, p))
    Y = Y_base * np.array([1.0, 12.0, 80.0], dtype=float) + np.array([0.0, 3.0, -5.0], dtype=float)

    latent_bounds = []
    for _ in range(q):
        latent_bounds.append((np.log(1e-3), np.log(1e3)))
        latent_bounds.extend([(np.log(0.05), np.log(5.0))] * d)
    sigma_bounds = [(np.log(1e-6), np.log(10.0))] * p
    theta0 = np.concatenate([_latent_theta(q=q, d=d), np.log(1e-2 * np.ones(p, dtype=float))])

    for standardize_y in (False, "zscore"):
        model = MOOGP(
            terms=[None, 1, 2],
            q=q,
            Psi=None,
            learn_Psi=False,
            learn_sigma_eps=True,
            jitter=1e-6,
            one_based=True,
            use_diagonalized_interaction=True,
            standardize_y=standardize_y,
        )
        model.fit(
            {"X_scaled": X, "Y": Y},
            theta0=theta0,
            bounds=latent_bounds + sigma_bounds,
            optimizer_opts={"maxiter": 1},
        )

        mean, std = model.predict(X[:3], return_std=True)
        assert model.fitted
        assert model.cache["used_fast"] is True
        assert model.cache["Psi_raw"].shape == (p, q)
        assert model.cache["sigma_eps2_raw"].shape == (p,)
        assert mean.shape == (3, p)
        assert std.shape == (3, p)
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(std))


def test_fast_standardize_y_preserves_raw_sigma_scale_but_rebuilds_fast_basis():
    rng = np.random.default_rng(17)
    n, d, p, q = 9, 2, 3, 2
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y_base = rng.normal(size=(n, p))
    output_scale = np.array([1.0, 9.0, 40.0], dtype=float)
    output_shift = np.array([3.0, -2.0, 15.0], dtype=float)
    Y_raw = Y_base * output_scale + output_shift

    y_scale = Y_raw.std(axis=0, ddof=1)
    sigma_eps2_raw = np.array([0.15, 1.2, 8.0], dtype=float)

    theta_raw = np.concatenate([_latent_theta(q=q, d=d, sigma2=1.1, ell=0.8), np.log(sigma_eps2_raw)])
    theta_std = theta_raw.copy()
    theta_std[-p:] = np.log(sigma_eps2_raw / (y_scale ** 2))

    Xstar = rng.uniform(-1.0, 1.0, size=(4, d))

    model_raw = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=None,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=True,
        standardize_y=False,
    )
    model_raw._prepare_data({"X_scaled": X, "Y": Y_raw})
    model_raw._nll(theta_raw)
    model_raw.fitted = True

    model_std = MOOGP(
        terms=[None, 1, 2],
        q=q,
        Psi=None,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=0.0,
        one_based=True,
        use_diagonalized_interaction=True,
        standardize_y="zscore",
    )
    model_std._prepare_data({"X_scaled": X, "Y": Y_raw})
    model_std._nll(theta_std)
    model_std.fitted = True

    mean_raw, std_raw = model_raw.predict(Xstar, return_std=True)
    mean_std, std_std = model_std.predict(Xstar, return_std=True)

    phi_expected, d_expected = init_phi(model_std.Y, q=q, n=n)
    raw_ratio = np.linalg.norm(model_raw.Phi_fast, axis=1).max() / np.linalg.norm(model_raw.Phi_fast, axis=1).min()
    std_ratio = np.linalg.norm(model_std.Phi_fast, axis=1).max() / np.linalg.norm(model_std.Phi_fast, axis=1).min()

    assert not np.allclose(model_std.Phi_fast, model_raw.Phi_fast, atol=1e-12)
    assert not np.allclose(model_std.d_vals_fast, model_raw.d_vals_fast, atol=1e-12)
    assert np.allclose(model_std.Phi_fast, phi_expected, atol=1e-12)
    assert np.allclose(model_std.d_vals_fast, d_expected, atol=1e-12)
    assert raw_ratio > std_ratio
    assert not np.allclose(model_std.cache["Psi_raw"], model_raw.cache["Psi_raw"], atol=1e-12)
    assert np.allclose(model_std.cache["sigma_eps2_raw"], model_raw.cache["sigma_eps2_raw"], atol=1e-12)
    assert not np.allclose(mean_std, mean_raw, rtol=1e-8, atol=1e-10)
    assert not np.allclose(std_std, std_raw, rtol=1e-8, atol=1e-10)
