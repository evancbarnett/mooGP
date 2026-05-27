import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moogp.datasets import generate_forrester_data, log_lhs_1d_rescaled
from forrester_illustration import (
    FORRESTER_MANUSCRIPT_SIGMA_EPS2,
    evaluate_moogp,
    fit_moogp_forrester,
    plot_forrester_fit,
    plot_forrester_fit_side_by_side,
    plot_trend_recovery_two_designs,
    print_predictive_table,
    print_trend_comparison_table,
)


def run_forrester_illustration_slow(
    *,
    n_train=25,
    seed=67,
    trend_output_idx=2,
    MOGP=True,
    learn_Psi=False,
    Psi=None,
    outdir=None,
):
    """Reproduce the Forrester illustration figures using the dense slow path."""
    error_per_output = FORRESTER_MANUSCRIPT_SIGMA_EPS2.tolist()
    include_mogp = bool(MOGP)
    if learn_Psi:
        fit_psi = None if Psi is None else np.asarray(Psi, dtype=float)
    else:
        fit_psi = np.eye(len(error_per_output)) if Psi is None else np.asarray(Psi, dtype=float)

    if outdir is None:
        subdir_parts = ["slow"]
        if learn_Psi:
            subdir_parts.append("learn_psi")
        if not include_mogp:
            subdir_parts.append("moogp_only")
        subdir = "_".join(subdir_parts)
        outdir = Path(__file__).resolve().parent / "figs" / subdir
    outdir.mkdir(parents=True, exist_ok=True)

    data_lhs = generate_forrester_data(
        n=n_train,
        seed=seed,
        with_error=True,
        error_per_output=error_per_output,
    )

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
        error_per_output=error_per_output,
        X_override=X_log,
    )

    start_time = time.perf_counter()

    moogp_lhs, X1, Xs1, Y1 = fit_moogp_forrester(
        n_train=n_train,
        seed=seed,
        orthogonal=True,
        Psi=fit_psi,
        use_fast=False,
        learn_Psi=learn_Psi,
        data=data_lhs,
    )
    mogp_lhs = None
    if include_mogp:
        mogp_lhs, _, _, _ = fit_moogp_forrester(
            n_train=n_train,
            seed=seed,
            orthogonal=False,
            Psi=fit_psi,
            use_fast=False,
            learn_Psi=learn_Psi,
            data=data_lhs,
        )

    moogp_log, X2, Xs2, Y2 = fit_moogp_forrester(
        n_train=n_train,
        seed=seed,
        orthogonal=True,
        Psi=fit_psi,
        use_fast=False,
        learn_Psi=learn_Psi,
        data=data_log,
    )
    mogp_log = None
    if include_mogp:
        mogp_log, _, _, _ = fit_moogp_forrester(
            n_train=n_train,
            seed=seed,
            orthogonal=False,
            Psi=fit_psi,
            use_fast=False,
            learn_Psi=learn_Psi,
            data=data_log,
        )

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.3f}s")

    fig_pred = plot_forrester_fit(moogp_lhs, X1, Xs1, Y1, non_ortho_model=mogp_lhs)
    fig_pred.savefig(outdir / "forrester_fit_lhs.png", dpi=600, bbox_inches="tight")
    fig_pred.savefig(outdir / "forrester_fit_lhs.pdf", dpi=600, bbox_inches="tight")

    fig_pred = plot_forrester_fit(moogp_log, X2, Xs2, Y2, non_ortho_model=mogp_log)
    fig_pred.savefig(outdir / "forrester_fit_log.png", dpi=600, bbox_inches="tight")
    fig_pred.savefig(outdir / "forrester_fit_log.pdf", dpi=600, bbox_inches="tight")

    fig = plot_forrester_fit_side_by_side(
        moogp_lhs=moogp_lhs,
        X_lhs=data_lhs["X"],
        Y_lhs=data_lhs["y"],
        mogp_lhs=mogp_lhs,
        moogp_log=moogp_log,
        X_log=data_log["X"],
        Y_log=data_log["y"],
        mogp_log=mogp_log,
    )
    fig.savefig(outdir / "forrester_fit_lhs_vs_loglhs.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "forrester_fit_lhs_vs_loglhs.pdf", dpi=600, bbox_inches="tight")

    fig_trend = plot_trend_recovery_two_designs(
        data_lhs,
        moogp_lhs,
        mogp_lhs,
        data_log,
        moogp_log,
        mogp_log,
        output_idx=trend_output_idx,
        left_label="LHS",
        right_label="log-LHS",
    )
    fig_trend.savefig(
        outdir / f"forrester_trend_recovery_output{trend_output_idx + 1}.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig_trend.savefig(
        outdir / f"forrester_trend_recovery_output{trend_output_idx + 1}.pdf",
        dpi=600,
        bbox_inches="tight",
    )

    results_lhs = evaluate_moogp(
        moogp_lhs,
        data=data_lhs,
        error_per_output=error_per_output,
        non_ortho_model=mogp_lhs,
        ols=True,
        output_idx=trend_output_idx,
        scheme_name="LHS",
    )
    results_log = evaluate_moogp(
        moogp_log,
        data=data_log,
        error_per_output=error_per_output,
        non_ortho_model=mogp_log,
        ols=True,
        output_idx=trend_output_idx,
        scheme_name="log-LHS",
    )

    print_predictive_table(results_lhs, "LHS")
    print()
    print_predictive_table(results_log, "log-LHS")
    print()
    print_trend_comparison_table(
        results_lhs,
        results_log,
        scheme1_label="LHS",
        scheme2_label="log-LHS",
    )

    return {
        "outdir": outdir,
        "moogp_lhs": moogp_lhs,
        "mogp_lhs": mogp_lhs,
        "moogp_log": moogp_log,
        "mogp_log": mogp_log,
        "results_lhs": results_lhs,
        "results_log": results_log,
    }


if __name__ == "__main__":
    run_forrester_illustration_slow(learn_Psi=True, seed=67, n_train=100, MOGP=False)
    plt.show()
