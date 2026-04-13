import numpy as np

from moogp.forrester_illustration import (
    generate_forrester_data,
    fit_moogp_forrester,
    FORRESTER_MANUSCRIPT_SIGMA_EPS2
)

def fit_forrester_sigma_eps_case(
    *,
    n_train,
    seed=0,
    true_sigma_eps2=FORRESTER_MANUSCRIPT_SIGMA_EPS2,
    Psi=None,
    orthogonal=True,
    use_fast=False,
    maxiter=500,
    data=None,
):
    """
    Fit a single Forrester sigma-epsilon recovery case.

    The intended baseline for sigma-epsilon recovery uses ``use_fast=False`` and
    a fixed ``Psi = I`` so that the measurement noise estimation is not coupled
    to the SVD-based fast parameterization.
    """
    true_sigma_eps2 = np.asarray(true_sigma_eps2, dtype=float).ravel()
    p = true_sigma_eps2.size
    q = p

    if Psi is None:
        Psi_fit = None if use_fast else np.eye(p, dtype=float)
    else:
        Psi = np.asarray(Psi, dtype=float)
        Psi_fit = Psi

    if (Psi_fit is not None) and (Psi_fit.shape != (p, q)):
        raise ValueError(f"Psi shape {Psi_fit.shape} must equal ({p}, {q}) for the Forrester study.")

    if data is None:
        data = generate_forrester_data(
            n=n_train,
            seed=seed,
            with_error=True,
            error_per_output=true_sigma_eps2,
        )

    model, X, X_scaled, Y = fit_moogp_forrester(
        n_train=n_train,
        seed=seed,
        q=q,
        Psi=Psi_fit,
        orthogonal=orthogonal,
        use_fast=use_fast,
        learn_Psi=False,
        data=data,
        error_per_output=true_sigma_eps2,
        maxiter=maxiter,
    )

    sigma_eps2_hat = np.asarray(model.cache["sigma_eps2"], dtype=float).copy()
    return {
        "n_train": int(n_train),
        "seed": int(seed),
        "model": model,
        "data": data,
        "X": X,
        "X_scaled": X_scaled,
        "Y": Y,
        "true_sigma_eps2": true_sigma_eps2.copy(),
        "sigma_eps2_hat": sigma_eps2_hat,
        "sigma_eps2_error": sigma_eps2_hat - true_sigma_eps2,
        "used_fast": bool(model.cache["used_fast"]),
        "Psi_hat": np.asarray(model.cache["Psi"], dtype=float).copy(),
    }


def run_forrester_sigma_eps_study(
    sample_sizes,
    *,
    seeds=(0,),
    true_sigma_eps2=FORRESTER_MANUSCRIPT_SIGMA_EPS2,
    Psi=None,
    orthogonal=True,
    use_fast=False,
    maxiter=500,
):
    """Run the sigma-epsilon recovery study over sample sizes and seeds."""
    cases = []
    for n_train in sample_sizes:
        for seed in seeds:
            cases.append(
                fit_forrester_sigma_eps_case(
                    n_train=n_train,
                    seed=seed,
                    true_sigma_eps2=true_sigma_eps2,
                    Psi=Psi,
                    orthogonal=orthogonal,
                    use_fast=use_fast,
                    maxiter=maxiter,
                )
            )
    return cases


def summarize_forrester_sigma_eps_study(cases):
    """Aggregate sigma-epsilon recovery cases into notebook-friendly rows."""
    if not cases:
        return []

    summary_rows = []
    sample_sizes = sorted({case["n_train"] for case in cases})

    for n_train in sample_sizes:
        group = [case for case in cases if case["n_train"] == n_train]
        sigma_hats = np.vstack([case["sigma_eps2_hat"] for case in group])
        true_sigma_eps2 = group[0]["true_sigma_eps2"]
        mean_hat = sigma_hats.mean(axis=0)
        std_hat = sigma_hats.std(axis=0, ddof=0)
        rmse_hat = np.sqrt(np.mean((sigma_hats - true_sigma_eps2[None, :]) ** 2, axis=0))

        for output_idx in range(true_sigma_eps2.size):
            summary_rows.append(
                {
                    "n_train": int(n_train),
                    "output": int(output_idx + 1),
                    "true_sigma_eps2": float(true_sigma_eps2[output_idx]),
                    "mean_sigma_eps2_hat": float(mean_hat[output_idx]),
                    "std_sigma_eps2_hat": float(std_hat[output_idx]),
                    "rmse_sigma_eps2_hat": float(rmse_hat[output_idx]),
                    "used_fast": bool(group[0]["used_fast"]),
                }
            )

    return summary_rows
def test_sigma_eps_study_summary_matches_single_seed_cases():
    true_sigma_eps2 = np.array([10.0, 1.0, 0.05], dtype=float)

    cases = run_forrester_sigma_eps_study(
        sample_sizes=[25, 50],
        seeds=(67,),
        true_sigma_eps2=true_sigma_eps2,
        Psi=np.eye(3),
        use_fast=False,
        maxiter=300,
    )
    by_n = {case["n_train"]: case for case in cases}

    err25 = abs(by_n[25]["sigma_eps2_hat"][2] - true_sigma_eps2[2])
    err50 = abs(by_n[50]["sigma_eps2_hat"][2] - true_sigma_eps2[2])

    assert by_n[50]["used_fast"] is False
    assert by_n[50]["model"].cache["fast_diag_info"] is None
    assert np.allclose(by_n[50]["Psi_hat"], np.eye(3), atol=1e-10)
    assert np.all(by_n[50]["sigma_eps2_hat"] > 0.0)
    assert err50 < err25
    assert err50 < 0.02

    summary = summarize_forrester_sigma_eps_study(cases)
    row_50_out3 = next(
        row for row in summary
        if row["n_train"] == 50 and row["output"] == 3
    )

    assert row_50_out3["used_fast"] is False
    assert np.isclose(row_50_out3["mean_sigma_eps2_hat"], by_n[50]["sigma_eps2_hat"][2])
    assert row_50_out3["rmse_sigma_eps2_hat"] < 0.02
