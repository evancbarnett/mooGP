import numpy as np

from moogp.model import init_phi
from moogp.model import MOOGP

from ..benchmark_lib import (
    PredictionBundle,
    append_sigma_eps_theta0_and_bounds,
    build_dataset_bundle,
    compute_metrics,
    make_latent_theta0_and_bounds,
    stable_seed,
)


BASE_SEED = 20260308


def _borehole_bundle_rep3() -> tuple[dict, np.ndarray]:
    seed_data = stable_seed(BASE_SEED, "borehole", 100, 4, 3, "data")
    bundle = build_dataset_bundle(
        function="borehole",
        n=100,
        p=4,
        n_test=50,
        seed_data=seed_data,
        noise_var_frac=0.05,
    )
    y_train = np.asarray(bundle.train_data["Y"], dtype=float)
    return bundle, y_train


def _standardize_outputs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    center = y.mean(axis=0)
    spread = np.std(y, axis=0, ddof=1)
    spread = np.where(spread > 1e-12, spread, 1.0)
    return (y - center) / spread, center, spread


def test_borehole_diagnostic_cell_has_large_output_scale_imbalance():
    _, y_train = _borehole_bundle_rep3()

    y_var = np.var(y_train, axis=0, ddof=1)
    var_ratio = y_var.max() / y_var.min()

    assert var_ratio > 1e3


def test_standardizing_outputs_balances_sigma_eps_initialization_and_bounds():
    _, y_train = _borehole_bundle_rep3()
    y_std, _, _ = _standardize_outputs(y_train)

    theta0, bounds = make_latent_theta0_and_bounds(q=4, d=4, seed_model=0)
    theta_raw, bounds_raw = append_sigma_eps_theta0_and_bounds(theta0, bounds, y_train)
    theta_std, bounds_std = append_sigma_eps_theta0_and_bounds(theta0, bounds, y_std)

    sigma_raw = theta_raw[-4:]
    sigma_std = theta_std[-4:]
    raw_span = sigma_raw.max() - sigma_raw.min()
    std_span = sigma_std.max() - sigma_std.min()

    assert raw_span > 5.0
    assert std_span < 1e-10
    assert np.allclose(sigma_std, np.log(1e-2), atol=1e-10)

    lb_raw = np.array([lo for lo, _ in bounds_raw[-4:]], dtype=float)
    ub_raw = np.array([hi for _, hi in bounds_raw[-4:]], dtype=float)
    lb_std = np.array([lo for lo, _ in bounds_std[-4:]], dtype=float)
    ub_std = np.array([hi for _, hi in bounds_std[-4:]], dtype=float)

    assert np.ptp(ub_raw - lb_raw) < 1e-10
    assert np.ptp(ub_std - lb_std) < 1e-10
    assert (lb_raw.max() - lb_raw.min()) > 5.0
    assert (ub_raw.max() - ub_raw.min()) > 5.0
    assert (lb_std.max() - lb_std.min()) < 1e-10
    assert (ub_std.max() - ub_std.min()) < 1e-10


def test_standardizing_outputs_rebalances_fast_phi_basis():
    _, y_train = _borehole_bundle_rep3()
    y_std, _, _ = _standardize_outputs(y_train)

    phi_raw, _ = init_phi(y_train, q=4, n=y_train.shape[0])
    phi_std, _ = init_phi(y_std, q=4, n=y_std.shape[0])

    row_norms_raw = np.linalg.norm(phi_raw, axis=1)
    row_norms_std = np.linalg.norm(phi_std, axis=1)

    raw_ratio = row_norms_raw.max() / row_norms_raw.min()
    std_ratio = row_norms_std.max() / row_norms_std.min()

    assert raw_ratio > 10.0
    assert std_ratio < 2.0


def test_moogp_standardize_y_rebuilds_fast_phi_basis_from_standardized_outputs():
    bundle, y_train = _borehole_bundle_rep3()
    data = {
        "X_scaled": np.asarray(bundle.train_data["X_scaled"], dtype=float),
        "Y": y_train,
    }

    model_raw = MOOGP(
        terms=[None, 1, 2, 3, 4],
        q=4,
        Psi=None,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=1e-6,
        one_based=True,
        use_diagonalized_interaction=True,
        standardize_y=False,
    )
    model_raw._prepare_data(data)

    model_std = MOOGP(
        terms=[None, 1, 2, 3, 4],
        q=4,
        Psi=None,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=1e-6,
        one_based=True,
        use_diagonalized_interaction=True,
        standardize_y="zscore",
    )
    model_std._prepare_data(data)

    raw_ratio = np.linalg.norm(model_raw.Phi_fast, axis=1).max() / np.linalg.norm(model_raw.Phi_fast, axis=1).min()
    std_ratio = np.linalg.norm(model_std.Phi_fast, axis=1).max() / np.linalg.norm(model_std.Phi_fast, axis=1).min()
    phi_expected, d_expected = init_phi(model_std.Y, q=4, n=model_std.n)

    assert raw_ratio > 10.0
    assert std_ratio < 2.0
    assert not np.allclose(model_std.Phi_fast, model_raw.Phi_fast, atol=1e-12)
    assert not np.allclose(model_std.d_vals_fast, model_raw.d_vals_fast, atol=1e-12)
    assert np.allclose(model_std.Phi_fast, phi_expected, atol=1e-12)
    assert np.allclose(model_std.d_vals_fast, d_expected, atol=1e-12)
    assert np.allclose(np.mean(model_std.Y, axis=0), 0.0, atol=1e-12)
    assert np.allclose(np.std(model_std.Y, axis=0, ddof=1), 1.0, atol=1e-12)


def test_clean_and_noisy_rmse_capture_different_targets():
    bundle, _ = _borehole_bundle_rep3()
    y_clean = np.asarray(bundle.extra["Y_test_clean"], dtype=float)
    prediction = PredictionBundle(mean=y_clean)

    noisy_metrics = compute_metrics(bundle.test_Y_true, prediction)
    clean_metrics = compute_metrics(y_clean, prediction)

    assert noisy_metrics["rmse"] > 0.0
    assert np.isclose(clean_metrics["rmse"], 0.0)
