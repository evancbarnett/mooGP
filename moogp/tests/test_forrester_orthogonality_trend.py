import numpy as np

from moogp.datasets import generate_forrester_data, log_lhs_1d_rescaled
from moogp.forrester_illustration import fit_moogp_forrester, get_model_trend_betas_raw


def test_orthogonal_moogp_recovers_loglhs_trend_better_than_nonorthogonal():
    n_train = 25
    seed = 0
    true_beta_output3 = np.array([-1.5, -5.0], dtype=float)

    X_log = log_lhs_1d_rescaled(
        n_train,
        seed=seed,
        xmin=1e-3,
        cluster="right",
        include_x0=False,
        shuffle=False,
    )
    data_log = generate_forrester_data(
        n=n_train,
        seed=seed,
        with_error=True,
        error_per_output=[10.0, 1.0, 0.05],
        X_override=X_log,
    )

    moogp, _, _, _ = fit_moogp_forrester(
        n_train=n_train,
        seed=seed,
        orthogonal=True,
        Psi=np.eye(3),
        use_fast=False,
        data=data_log,
        maxiter=300,
    )
    mogp, _, _, _ = fit_moogp_forrester(
        n_train=n_train,
        seed=seed,
        orthogonal=False,
        Psi=np.eye(3),
        use_fast=False,
        data=data_log,
        maxiter=300,
    )

    beta_moogp = get_model_trend_betas_raw(moogp)[:, 2]
    beta_mogp = get_model_trend_betas_raw(mogp)[:, 2]

    err_moogp = np.linalg.norm(beta_moogp - true_beta_output3)
    err_mogp = np.linalg.norm(beta_mogp - true_beta_output3)

    assert moogp.cache["used_fast"] is False
    assert mogp.cache["used_fast"] is False
    assert err_moogp < err_mogp
