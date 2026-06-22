"""Tests for the in-model data-aware ``theta0`` / ``bounds`` initialization.

``MOOGP.fit`` builds a data-aware starting point and box bounds automatically
when the caller omits them (``MOOGP._default_theta0_and_bounds``). These tests
pin the packing/feasibility contract, verify the noise init is computed on the
standardized working scale (the guarantee the benchmark's append_sigma_eps
capture test used to protect before the logic moved into the model), and check
that omitting the arguments matches passing the default explicitly.
"""

import numpy as np
import pytest

from moogp.datasets import generate_forrester_data
from moogp.model import MOOGP, unpack_theta


def _forrester(n=40, seed=3):
    """Small Forrester block: p=3 outputs, d=1 input."""
    data = generate_forrester_data(
        n=n, seed=seed, with_error=True,
        error_per_output=np.array([10.0, 1.0, 0.05], dtype=float),
    )
    return {"X_scaled": data["X_scaled"], "y": data["y"]}


def _model(q=2, **kw):
    defaults = dict(
        terms=[None, 1],
        q=q,
        Psi=None,
        orthogonal=True,
        learn_Psi=False,
        learn_sigma_eps=True,
        use_diagonalized_interaction=True,
        standardize_x="unitcube",
        standardize_y="zscore",
        jitter=1e-8,
    )
    defaults.update(kw)
    return MOOGP(**defaults)


@pytest.mark.parametrize(
    "learn_Psi,learn_sigma_eps,diag_es",
    [
        (False, True, None),
        (False, False, None),
        (True, True, None),
        (True, False, None),
        (False, True, [2, 1]),
    ],
)
def test_default_theta0_and_bounds_packing_and_feasibility(learn_Psi, learn_sigma_eps, diag_es):
    data = _forrester(n=30, seed=1)
    q = 2
    d = data["X_scaled"].shape[1]
    p = data["y"].shape[1]

    model = _model(
        q=q,
        learn_Psi=learn_Psi,
        learn_sigma_eps=learn_sigma_eps,
        diag_error_structure=diag_es,
    )
    model._prepare_data(data)
    theta0, bounds = model._default_theta0_and_bounds()

    n_groups = (len(diag_es) if diag_es is not None else p)
    expected = q * (d + 1)
    if learn_Psi:
        expected += p * q
    if learn_sigma_eps:
        expected += n_groups

    assert theta0.shape == (expected,)
    assert len(bounds) == expected

    # Every seed must start strictly inside (or on) its box so L-BFGS-B is feasible.
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    assert np.all(theta0 >= lo - 1e-12)
    assert np.all(theta0 <= hi + 1e-12)

    # The packing must round-trip through unpack_theta for the same flags.
    lat_params, Psi, sigma_eps2 = unpack_theta(
        theta0, d, q, p,
        learn_Psi=learn_Psi,
        learn_sigma_eps=learn_sigma_eps,
        diag_error_structure=model.diag_error_structure,
    )
    assert len(lat_params) == q
    assert (Psi is not None) == learn_Psi
    if learn_Psi:
        assert Psi.shape == (p, q)
    assert (sigma_eps2 is not None) == learn_sigma_eps
    if learn_sigma_eps:
        assert np.asarray(sigma_eps2).size == p  # broadcast back to per-output


def test_default_sigma_eps_init_uses_standardized_y():
    """The noise init lives on the standardized scale, not the raw output scale."""
    data = _forrester(n=50, seed=5)
    # Inflate outputs so raw vs. standardized scales differ by orders of magnitude.
    Y_big = data["y"] * 1000.0 + 5000.0
    data_big = {"X_scaled": data["X_scaled"], "y": Y_big}
    q = 2
    d = data["X_scaled"].shape[1]
    p = Y_big.shape[1]

    model = _model(q=q, standardize_y="zscore")
    model._prepare_data(data_big)
    theta0, _ = model._default_theta0_and_bounds()

    base = q * (d + 1)
    sigma_eps2_init = np.exp(theta0[base:])
    assert sigma_eps2_init.size == p

    # zscore makes Var(Y_work) == 1, so the noise upper bound is 0.5 * 1 = 0.5.
    # A raw-scale init (the historical bug) would have produced
    # exp(sigma) ~ 1e-2 * Var(Y_raw) >> 1.
    assert np.all(sigma_eps2_init <= 0.5 + 1e-9)
    raw_var = np.var(Y_big, axis=0, ddof=1)
    assert raw_var.min() > 1e3
    assert np.all(sigma_eps2_init < raw_var.min())


def test_fit_without_theta0_bounds_runs_and_is_deterministic():
    data = _forrester(n=50, seed=7)

    m1 = _model(q=2)
    m1.fit(data=data, optimizer_opts={"maxiter": 100})

    assert m1.fitted
    assert m1.theta_hat is not None
    mean, std = m1.predict(data["X_scaled"], return_std=True)
    assert mean.shape == data["y"].shape
    assert std.shape == data["y"].shape
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
    # Auto-init should reach a sensible fit: train RMSE well below the spread of Y.
    train_rmse = np.sqrt(np.mean((mean - data["y"]) ** 2))
    assert train_rmse < np.std(data["y"])

    # A second identical auto-init fit reproduces theta_hat (no hidden randomness).
    m2 = _model(q=2)
    m2.fit(data=data, optimizer_opts={"maxiter": 100})
    assert np.allclose(m1.theta_hat, m2.theta_hat)


def test_fit_omitting_args_matches_explicit_default():
    """Omitting theta0/bounds must equal passing the data-aware default explicitly."""
    data = _forrester(n=40, seed=9)

    m_auto = _model(q=2)
    m_auto.fit(data=data, optimizer_opts={"maxiter": 80})

    ref = _model(q=2)
    ref._prepare_data(data)
    theta0, bounds = ref._default_theta0_and_bounds()

    m_explicit = _model(q=2)
    m_explicit.fit(data=data, theta0=theta0, bounds=bounds, optimizer_opts={"maxiter": 80})

    assert np.allclose(m_auto.theta_hat, m_explicit.theta_hat)


def test_fit_without_args_learn_psi_slow_path():
    """Auto-init also produces a valid start for the learn_Psi slow path."""
    data = _forrester(n=30, seed=11)

    model = _model(
        q=2,
        learn_Psi=True,
        use_diagonalized_interaction=False,
        jitter=1e-6,
    )
    model.fit(data=data, optimizer_opts={"maxiter": 50})

    assert model.fitted
    mean = model.predict(data["X_scaled"])
    assert mean.shape == data["y"].shape
    assert np.all(np.isfinite(mean))
