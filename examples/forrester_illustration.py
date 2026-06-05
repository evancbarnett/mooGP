import time

import numpy as np
import matplotlib.pyplot as plt

from moogp.design import make_G
from moogp.datasets import generate_forrester_data, log_lhs_1d_rescaled
from moogp.model import MOOGP
from moogp.evaluation import rmse, normalized_rmse, intervalstats, dss
from pathlib import Path

def get_model_trend_betas_raw(model):
    """
    Convert cached bhat from scaled basis
        m(x_scaled) = b0 + b1 * x_scaled
    to raw-x basis
        m(x) = beta0 + beta1 * x,  x in [0,1],
    where x_scaled = 2x - 1.
    """
    Bhat = np.asarray(model.cache["bhat"], dtype=float)

    if Bhat.shape[0] != 2:
        raise ValueError("Expected a 1D linear trend with terms=[None, 1].")

    b0_scaled = Bhat[0, :]
    b1_scaled = Bhat[1, :]

    beta0_raw = b0_scaled - b1_scaled
    beta1_raw = 2.0 * b1_scaled

    return np.vstack([beta0_raw, beta1_raw])  # shape (2, p)


def get_ols_betas_raw(X_raw, Y):
    """
    OLS fit in the raw-x basis: y = beta0 + beta1 x.
    Returns shape (2, p).
    """
    G = np.column_stack([np.ones(X_raw.shape[0]), X_raw[:, 0]])
    B_ls, *_ = np.linalg.lstsq(G, Y, rcond=None)
    return B_ls

def eval_linear_trend(X_raw, betas_raw, output_idx):
    """
    Evaluate beta0 + beta1 x for one selected output.
    """
    x = X_raw[:, 0]
    return betas_raw[0, output_idx] + betas_raw[1, output_idx] * x


PREDICTIVE_METRIC_LABELS = {
    "rmse_y": "RMSE(y)",
    "rmse_f": "RMSE(f)",
    "nrmse_y": "NRMSE(y)",
    "coverage_95_y": "Coverage95(y)",
    "width_95_y": "Width95(y)",
    "dss_y": "DSS(y)",
}

TREND_GRID_RMSE_KEY = "rmse_f_grid"
TREND_GRID_RMSE_LABEL = "RMSE(f_grid)"


def rmse_1d(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def forrester_true_functions(X):
    """
    X: (n,1) in [0,1]
    Returns y_true: (n,3) matching Forrester functions.
    """
    x = X.reshape(-1, 1)

    f1 = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    f2 = 0.5 * f1 + 5 * (x - 0.5) + 5
    f3 = -0.8 * f1 - 5 * (x - 0.5) - 4

    return np.concatenate([f1, f2, f3], axis=1)

FORRESTER_MANUSCRIPT_SIGMA_EPS2 = np.array([10.0, 1.0, 0.05], dtype=float)


def build_forrester_theta0_bounds(Y, q, d):
    """Construct the default Forrester initialization and box constraints."""
    theta0 = []
    bounds = []
    for _ in range(q):
        theta0.append(np.log(1.0))
        theta0.extend([np.log(0.5)] * d)

        bounds.append((np.log(1e-3), np.log(1e3)))
        bounds.extend([(np.log(0.05), np.log(5.0))] * d)

    theta0 = np.asarray(theta0, dtype=float)

    y_var = Y.var(axis=0, ddof=1)
    sigma_eps2_init = np.log(1e-2 * y_var)
    theta0 = np.concatenate([theta0, sigma_eps2_init])

    lb = np.maximum(1e-12, 1e-6 * y_var)
    ub = np.maximum(lb * 10.0, 0.5 * y_var)
    log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]
    bounds.extend(log_bounds)
    return theta0, bounds


def fit_moogp_forrester(n_train=50,
                        seed=0,
                        q=3,
                        Psi=None,
                        orthogonal=True,
                        use_fast=True,
                        learn_Psi=False,
                        data=None,
                        error_per_output=FORRESTER_MANUSCRIPT_SIGMA_EPS2,
                        maxiter=500):
    if data is None:
        data = generate_forrester_data(
            n=n_train,
            seed=seed,
            with_error=True,
            error_per_output=error_per_output,
        )
    
    X = data["X"]          # in [0,1]
    X_scaled = data["X_scaled"]  # in [-1,1]
    Y = data["y"]          # (n,3)

    n, d = X_scaled.shape
    p = Y.shape[1]

    # 2) Mean basis: intercept + main effect
    terms = [None] + list(range(1, d + 1))

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        orthogonal=orthogonal,
        learn_Psi=learn_Psi,
        learn_sigma_eps=True,
        jitter=1e-10,
        normalize_cols=True,
        use_diagonalized_interaction=use_fast,  # Fast computation
        standardize_x=False,
        standardize_y=False,
    )
    theta0, bounds = build_forrester_theta0_bounds(Y, q, d)

    if learn_Psi:
        if Psi is None:
            Psi0 = np.eye(p, q, dtype=float)
        else:
            Psi0 = np.asarray(Psi, dtype=float)
            if Psi0.shape != (p, q):
                raise ValueError(f"Psi shape {Psi0.shape} must equal ({p}, {q}) when learn_Psi=True.")

        base = q * (d + 1)
        theta_latent = theta0[:base]
        theta_sigma = theta0[base:]
        bounds_latent = bounds[:base]
        bounds_sigma = bounds[base:]

        theta0 = np.concatenate([theta_latent, Psi0.ravel(order="C"), theta_sigma])
        bounds = bounds_latent + [(-5.0, 5.0)] * (p * q) + bounds_sigma

    # 5) Fit
    model.fit(
        data={"X_scaled": X_scaled, "y": Y},
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": maxiter},
    )

    return model, X, X_scaled, Y

def plot_forrester_fit(model, X, X_scaled, Y, n_plot=400, non_ortho_model=None):
    """
    Paper-quality plot: training data + true function + mean/interval for
      - MOOGP (model)
      - MOGP (non_ortho_model), if provided

    Legend is placed in bottom-left of Output 3 panel.
    Coverage bands are shown via fill + boundary lines (not in legend).
    True function is solid black.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    paper_rc = {
        # Fonts 
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        # Lines / markers
        "lines.linewidth": 1.6,
        "lines.markersize": 4,

        # Axes aesthetics
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,

        # Savefig defaults
        "savefig.dpi": 300,
    }

    with mpl.rc_context(paper_rc):
        X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
        X_plot_scaled = 2.0 * (X_plot - 0.5)

        Y_true_plot = forrester_true_functions(X_plot)

        mean_moogp, std_moogp = model.predict(X_plot_scaled, return_std=True)

        mean_mogp = std_mogp = None
        if non_ortho_model is not None:
            mean_mogp, std_mogp = non_ortho_model.predict(X_plot_scaled, return_std=True)

        p = Y.shape[1]
        fig, axes = plt.subplots(
            p, 1,
            figsize=(6.6, 1.9 * p),
            sharex=True,
            constrained_layout=True
        )
        if p == 1:
            axes = [axes]

        legend_ax_idx = min(2, p - 1)

        moogp_style = dict(color="tab:blue", linestyle="--")
        mogp_style  = dict(color="tab:orange", linestyle=":")

        def plot_mean_and_interval_lines(
            ax, x, mean, std, *,
            label_line=None,
            color="tab:blue",
            linestyle="-",
            lw_mean=1.9,
            lw_bound=1.3,
            alpha_fill=0.12
        ):
            upper = mean + 2.0 * std
            lower = mean - 2.0 * std

            ax.fill_between(
                x, lower, upper,
                color=color,
                alpha=alpha_fill,
                linewidth=0.0,
                zorder=1
            )

            ax.plot(x, upper, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)
            ax.plot(x, lower, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)

            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=lw_mean, label=label_line, zorder=4)

        for j, ax in enumerate(axes):
            add_labels = (j == legend_ax_idx)

            ax.scatter(
                X[:, 0], Y[:, j],
                s=18,
                color="black",
                alpha=0.75,
                linewidths=0.0,
                label="Training data" if add_labels else None,
                zorder=5
            )

            ax.plot(
                X_plot[:, 0], Y_true_plot[:, j],
                linestyle="-",
                linewidth=1.4,
                color="black",
                label="True function" if add_labels else None,
                zorder=2
            )

            # MOOGP (label without "mean")
            plot_mean_and_interval_lines(
                ax,
                X_plot[:, 0],
                mean_moogp[:, j],
                std_moogp[:, j],
                label_line="MOOGP" if add_labels else None,
                color=moogp_style["color"],
                linestyle=moogp_style["linestyle"],
                lw_mean=1.9,
                lw_bound=1.3,
                alpha_fill=0.12
            )

            # MOGP (label without "mean")
            if non_ortho_model is not None:
                plot_mean_and_interval_lines(
                    ax,
                    X_plot[:, 0],
                    mean_mogp[:, j],
                    std_mogp[:, j],
                    label_line="MOGP" if add_labels else None,
                    color=mogp_style["color"],
                    linestyle=mogp_style["linestyle"],
                    lw_mean=1.9,
                    lw_bound=1.3,
                    alpha_fill=0.10
                )

            ax.set_ylabel(f"Output {j+1}")
            ax.tick_params(axis="both", which="both", top=False, right=False)

        axes[-1].set_xlabel("x")

        # Legend bottom-left of Output 3 panel; boxed.
        leg_ax = axes[legend_ax_idx]
        handles, labels = leg_ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if l and (l not in seen and not seen.add(l))]
        if uniq:
            leg = leg_ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                loc="lower left",
                frameon=True,          # <-- box ON
                fancybox=False,        # crisp rectangle (journal-y)
                framealpha=1.0,
                handlelength=2.8,
                borderaxespad=0.6
            )
            # Make the box look clean in print
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(0.8)

        return fig

def plot_forrester_fit_side_by_side(
    moogp_lhs, X_lhs, Y_lhs, mogp_lhs=None,
    moogp_log=None, X_log=None, Y_log=None, mogp_log=None,
    n_plot=400,
    left_label="LHS",
    right_label="log-LHS",
):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    # Match your latest styling choices (bigger fonts + boxed legend)
    scale_factor = 1.5
    paper_rc = {
        "font.family": "serif",
        "font.size": 12 * scale_factor,
        "axes.labelsize": 12 * scale_factor,
        "legend.fontsize": 11 * scale_factor,
        "xtick.labelsize": 11 * scale_factor,
        "ytick.labelsize": 11 * scale_factor,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "savefig.dpi": 300,
    }

    with mpl.rc_context(paper_rc):
        # Dense grid
        X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
        X_plot_scaled = 2.0 * (X_plot - 0.5)
        Y_true_plot = forrester_true_functions(X_plot)

        # Colors/linestyles
        moogp_style = dict(color="tab:blue", linestyle="--")
        mogp_style  = dict(color="tab:orange", linestyle=":")

        def plot_mean_and_interval_lines(ax, x, mean, std, *, label_line=None, color="tab:blue",
                                         linestyle="-", lw_mean=1.9, lw_bound=1.3, alpha_fill=0.12):
            upper = mean + 2.0 * std
            lower = mean - 2.0 * std

            # Fill (no legend)
            ax.fill_between(x, lower, upper, color=color, alpha=alpha_fill, linewidth=0.0, zorder=1)
            # Boundary lines (same formatting as trend; no legend)
            ax.plot(x, upper, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)
            ax.plot(x, lower, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)
            # Trend line (legend label here only)
            ax.plot(x, mean,  color=color, linestyle=linestyle, linewidth=lw_mean, label=label_line, zorder=4)

        def draw_panel(ax, X, Y, moogp, mogp, add_legend_labels):
            # training points
            ax.scatter(X[:, 0], Y[:, 0]*0 + Y[:, 0], s=18, color="black", alpha=0.75,
                       linewidths=0.0, label="Training data" if add_legend_labels else None, zorder=5)

        # Create axes: rows = outputs, cols = designs
        p = Y_lhs.shape[1]
        fig, axes = plt.subplots(
            p, 2,
            figsize=(13.2, 1.9 * p),  # ~double width of your single-column figure
            sharex=True,
            sharey="row",
            constrained_layout=True,
        )
        if p == 1:
            axes = np.array([[axes[0], axes[1]]])

        # Precompute predictions for both designs/models
        mean_lhs_moogp, std_lhs_moogp = moogp_lhs.predict(X_plot_scaled, return_std=True)
        mean_lhs_mogp = std_lhs_mogp = None
        if mogp_lhs is not None:
            mean_lhs_mogp, std_lhs_mogp = mogp_lhs.predict(X_plot_scaled, return_std=True)

        mean_log_moogp = std_log_moogp = None
        mean_log_mogp = std_log_mogp = None
        if moogp_log is not None:
            mean_log_moogp, std_log_moogp = moogp_log.predict(X_plot_scaled, return_std=True)
        if mogp_log is not None:
            mean_log_mogp, std_log_mogp = mogp_log.predict(X_plot_scaled, return_std=True)

        # Legend goes in bottom-left of Output 3 graph on the RIGHT column
        legend_row = min(2, p - 1)
        legend_ax = axes[legend_row, 0]

        # Change training point opacity based number of training points.
        if X_lhs.shape[0] > 50:
            alpha = 0.5
        else:
            alpha = 0.75


        for j in range(p):
            # -------- Left column (LHS) --------
            axL = axes[j, 0]

            axL.scatter(X_lhs[:, 0], Y_lhs[:, j], s=18, color="black", alpha=alpha, linewidths=0.0,
                        label="Training data" if axL is legend_ax else None, zorder=5)

            axL.plot(X_plot[:, 0], Y_true_plot[:, j], color="black", linestyle="-", linewidth=1.6,
                     label="True function" if axL is legend_ax else None, zorder=2)

            plot_mean_and_interval_lines(
                axL, X_plot[:, 0], mean_lhs_moogp[:, j], std_lhs_moogp[:, j],
                label_line="MOOGP" if axL is legend_ax else None,
                color=moogp_style["color"], linestyle=moogp_style["linestyle"], alpha_fill=0.12
            )
            if mogp_lhs is not None:
                plot_mean_and_interval_lines(
                    axL, X_plot[:, 0], mean_lhs_mogp[:, j], std_lhs_mogp[:, j],
                    label_line="MOGP" if axL is legend_ax else None,
                    color=mogp_style["color"], linestyle=mogp_style["linestyle"], alpha_fill=0.10
                )

            # y-label only on left column
            axL.set_ylabel(f"Output {j+1}")
            axL.tick_params(axis="both", which="both", top=False, right=False)

            # Panel label (no title)
            if j == 0:
                axL.text(0.02, 0.96, left_label, transform=axL.transAxes, va="top")

            # -------- Right column (log-LHS) --------
            axR = axes[j, 1]

            axR.scatter(X_log[:, 0], Y_log[:, j], s=18, color="black", alpha=alpha, linewidths=0.0,
                        label="Training data" if axR is legend_ax else None, zorder=5)

            axR.plot(X_plot[:, 0], Y_true_plot[:, j], color="black", linestyle="-", linewidth=1.6,
                     label="True function" if axR is legend_ax else None, zorder=2)

            if moogp_log is not None:
                plot_mean_and_interval_lines(
                    axR, X_plot[:, 0], mean_log_moogp[:, j], std_log_moogp[:, j],
                    label_line="MOOGP" if axR is legend_ax else None,
                    color=moogp_style["color"], linestyle=moogp_style["linestyle"], alpha_fill=0.12
                )
            if mogp_log is not None:
                plot_mean_and_interval_lines(
                    axR, X_plot[:, 0], mean_log_mogp[:, j], std_log_mogp[:, j],
                    label_line="MOGP" if axR is legend_ax else None,
                    color=mogp_style["color"], linestyle=mogp_style["linestyle"], alpha_fill=0.10
                )

            axR.tick_params(axis="both", which="both", top=False, right=False)

            if j == 0:
                axR.text(0.02, 0.96, right_label, transform=axR.transAxes, va="top")

        # x labels on bottom row
        axes[-1, 0].set_xlabel("x")
        axes[-1, 1].set_xlabel("x")

        # Legend: only on (Output 3, right col), boxed, deduped
        handles, labels = legend_ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if l and (l not in seen and not seen.add(l))]
        if uniq:
            leg = legend_ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                loc="lower left",
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                handlelength=2.8,
                borderaxespad=0.6,
            )
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(0.8)

        return fig

def plot_trend_recovery_two_designs(
    data_lhs, moogp_lhs, mogp_lhs=None,
    data_log=None, moogp_log=None, mogp_log=None,
    output_idx=0,
    n_plot=400,
    left_label="LHS",
    right_label="log-LHS (rescaled)",
):
    """
    Two-panel trend comparison:
      left: observation scheme 1
      right: observation scheme 2

    Shows:
      - training observations
      - true function
      - OLS fitted linear trend
      - MOOGP fitted linear trend
      - MOGP fitted linear trend, if provided
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    if output_idx not in (0, 1, 2):
        raise ValueError("output_idx must be 0, 1, or 2.")

    paper_rc = {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 600,
    }

    # Change opacity of training points based on # of observations
    if data_lhs['X'].shape[0] > 50:
        pts_alpha = 0.5
    else:
        pts_alpha = 1

    with mpl.rc_context(paper_rc):
        X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
        y_true = forrester_true_functions(X_plot)[:, output_idx]

        # Fit/extract all trends in the SAME raw-x basis
        beta_ls_lhs = get_ols_betas_raw(data_lhs["X"], data_lhs["y"])
        beta_ls_log = get_ols_betas_raw(data_log["X"], data_log["y"])

        beta_moogp_lhs = get_model_trend_betas_raw(moogp_lhs)
        beta_mogp_lhs = get_model_trend_betas_raw(mogp_lhs) if mogp_lhs is not None else None

        beta_moogp_log = get_model_trend_betas_raw(moogp_log)
        beta_mogp_log = get_model_trend_betas_raw(mogp_log) if mogp_log is not None else None

        fig, axes = plt.subplots(
            1, 2, figsize=(7.6, 2.8), sharey=True, constrained_layout=True
        )

        true_style  = dict(color="black", linestyle="-",  linewidth=2.4)
        pts_style   = dict(color="black", s=55, zorder=5, alpha=pts_alpha)
        ls_style    = dict(color="tab:green", linestyle=":",  linewidth=3.0)
        moogp_style = dict(color="tab:blue",  linestyle="--", linewidth=3.0)
        mogp_style  = dict(color="tab:orange",   linestyle="-",  linewidth=3.0, alpha=0.65)

        panel_specs = [
            (
                axes[0], data_lhs, left_label,
                beta_ls_lhs, beta_moogp_lhs, beta_mogp_lhs
            ),
            (
                axes[1], data_log, right_label,
                beta_ls_log, beta_moogp_log, beta_mogp_log
            ),
        ]

        legend_ax = axes[0]  # legend on LHS panel

        for ax, data, panel_label, beta_ls, beta_moogp, beta_mogp in panel_specs:
            Xtr = data["X"]
            Ytr = data["y"]
            add_labels = (ax is legend_ax)

            ax.scatter(
                Xtr[:, 0], Ytr[:, output_idx],
                label="Training data" if add_labels else None,
                **pts_style
            )

            ax.plot(
                X_plot[:, 0], y_true,
                label="True function" if add_labels else None,
                **true_style
            )
            ax.plot(
                X_plot[:, 0],
                eval_linear_trend(X_plot, beta_ls, output_idx),
                label="Least squares" if add_labels else None,
                **ls_style,
            )
            ax.plot(
                X_plot[:, 0],
                eval_linear_trend(X_plot, beta_moogp, output_idx),
                label="MOOGP" if add_labels else None,
                **moogp_style,
            )
            if beta_mogp is not None:
                ax.plot(
                    X_plot[:, 0],
                    eval_linear_trend(X_plot, beta_mogp, output_idx),
                    label="MOGP" if add_labels else None,
                    **mogp_style,
                )

            ax.set_xlabel("Input")
            ax.text(
                0.03, 0.95, panel_label,
                transform=ax.transAxes,
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=1.0,
                    boxstyle="square,pad=0.15",
                ),
                zorder=10,
            )

        axes[0].set_ylabel(f"Output {output_idx + 1}")

        # Boxed legend on LHS panel
        handles, labels = legend_ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if l and (l not in seen and not seen.add(l))]
        if uniq:
            leg = legend_ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                loc="lower left",
                frameon=True,
                fancybox=False,
                framealpha=1.0,
                handlelength=2.8,
                borderaxespad=0.6,
            )
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(0.8)

        return fig

def plot_pred_vs_true(model, n_test=50, seed=123):
    # Generate a test set
    data_test = generate_forrester_data(n=n_test, seed=seed)
    # X_test = data_test["X"]
    X_test_scaled = data_test["X_scaled"]
    Y_test_true = data_test["y"]

    mean_test, _ = model.predict(X_test_scaled, return_std=True)

    p = Y_test_true.shape[1]
    fig, axes = plt.subplots(1, p, figsize=(4 * p, 4))

    if p == 1:
        axes = [axes]

    for j in range(p):
        ax = axes[j]
        ax.scatter(
            Y_test_true[:, j],
            mean_test[:, j],
            alpha=0.7,
        )
        min_y = min(Y_test_true[:, j].min(), mean_test[:, j].min())
        max_y = max(Y_test_true[:, j].max(), mean_test[:, j].max())
        ax.plot([min_y, max_y], [min_y, max_y], "k--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Output {j+1}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

def plot_trend_vs_ls(model, X, X_scaled, Y, n_plot=200, title_suffix=""):
    """
    Plot, for each output:
      - True Forrester function
      - OLS linear trend (y ~ a + b x)
      - MOOGP trend part: g(x)^T B_hat (excluding Psi z(x))

    Parameters
    ----------
    model : fitted MOOGP instance
    X : (n,1) raw input in [0,1]
    X_scaled : (n,1) scaled input in [-1,1]
    Y : (n,p) outputs
    n_plot : number of points in the dense plot grid
    """
    terms = model.terms
    cache = model.cache
    B_hat = cache["bhat"]  # shape (r, p)
    p = Y.shape[1]

    # Dense grid in [0,1] and scaled to [-1,1]
    X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
    X_plot_scaled = 2.0 * (X_plot - 0.5)

    # True Forrester functions at X_plot
    Y_true_plot = forrester_true_functions(X_plot)

    # --- MOOGP trend part: g(x)^T B_hat ---
    G_plot = make_G({"X_scaled": X_plot_scaled}, terms,
                    one_based=model.one_based, return_names=False)
    Y_trend_moogp = G_plot @ B_hat  # (n_plot, p)

    # --- OLS trend: fit y_j ~ a_j + b_j x on raw X ---
    # Design matrix for OLS: [1, x]
    G_ls_train = np.column_stack([np.ones_like(X[:, 0]), X[:, 0]])
    # Fit all outputs at once: B_ls has shape (2, p)
    B_ls, *_ = np.linalg.lstsq(G_ls_train, Y, rcond=None)
    # Evaluate on X_plot
    G_ls_plot = np.column_stack([np.ones_like(X_plot[:, 0]), X_plot[:, 0]])
    Y_trend_ls = G_ls_plot @ B_ls  # (n_plot, p)

    # --- Plot per output ---
    fig, axes = plt.subplots(p, 1, figsize=(8, 3 * p), sharex=True)
    if p == 1:
        axes = [axes]

    for j in range(p):
        ax = axes[j]

        # Training data
        ax.scatter(
            X[:, 0],
            Y[:, j],
            color="black",
            s=25,
            alpha=0.7,
            label="Training data" if j == 0 else None,
        )

        # True function
        ax.plot(
            X_plot[:, 0],
            Y_true_plot[:, j],
            "k--",
            linewidth=1.5,
            label="True function" if j == 0 else None,
        )

        # OLS trend
        ax.plot(
            X_plot[:, 0],
            Y_trend_ls[:, j],
            linewidth=2.0,
            label="OLS trend" if j == 0 else None,
        )

        # MOOGP trend
        ax.plot(
            X_plot[:, 0],
            Y_trend_moogp[:, j],
            linewidth=2.0,
            linestyle=":",
            label="MOOGP trend" if j == 0 else None,
        )

        ax.set_ylabel(f"y_{j+1}")
        ax.set_title(f"Output {j+1}{title_suffix}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    axes[0].legend(loc="best")
    fig.tight_layout()
    return fig

def evaluate_moogp(
    model,
    data,
    error_per_output,
    non_ortho_model=None,
    ols=True,
    n_test=200,
    seed=999,
    output_idx=0,
    n_grid=400,
    scheme_name=None,
):
    """
    Returns:
      - predictive: your original train/test table inputs
      - trend_table: grid RMSE + beta0/beta1 for one observation scheme and all outputs
                     (to be compared later across two schemes)
    """

    X_train_scaled = data["X_scaled"]
    X_train_raw = data["X"]
    Y_train = data["y"]
    F_train = data["f"]

    n_train, d = X_train_scaled.shape
    p = Y_train.shape[1]

    def get_metrics(ftrue, ytrue, ypred, yvar=None, train=False, modelname=""):
        rmse_f_val = rmse(ftrue, ypred)
        rmse_y_val = rmse(ytrue, ypred)
        nrmse_y_val = normalized_rmse(ytrue, ypred)

        if yvar is not None:
            coverage_95_y_val, width_95_y_val = intervalstats(ytrue, ypred, yvar)
            dss_y_val = dss(ytrue, ypred, yvar, use_diag=True)
        else:
            coverage_95_y_val = width_95_y_val = dss_y_val = None

        split_label = "train set metrics" if train else "test set metrics"

        return {
            "modelname": modelname,
            "modeltrain": split_label,
            "rmse_y": rmse_y_val,
            "rmse_f": rmse_f_val,
            "nrmse_y": nrmse_y_val,
            "coverage_95_y": coverage_95_y_val,
            "width_95_y": width_95_y_val,
            "dss_y": dss_y_val,
        }

    # -------------------
    # TRAIN METRICS
    # -------------------
    Y_train_pred, Y_train_std = model.predict(X_train_scaled, return_std=True)
    moogp_train = get_metrics(F_train, Y_train, Y_train_pred, Y_train_std**2, train=True, modelname="MOOGP")

    non_ortho_train = None
    if non_ortho_model is not None:
        ypred, ypredstd = non_ortho_model.predict(X_train_scaled, return_std=True)
        non_ortho_train = get_metrics(F_train, Y_train, ypred, ypredstd**2, train=True, modelname="MOGP")

    ols_train = None
    B_ls = None
    if ols:
        B_ls = get_ols_betas_raw(X_train_raw, Y_train)
        G_ls_train = np.column_stack([np.ones((n_train, 1)), X_train_raw])
        Y_train_pred_ls = G_ls_train @ B_ls
        ols_train = get_metrics(F_train, Y_train, Y_train_pred_ls, yvar=None, train=True, modelname="OLS")

    # -------------------
    # TEST METRICS
    # -------------------
    test_data = generate_forrester_data(
        n=n_test,
        seed=seed,
        with_error=True,
        error_per_output=error_per_output,
    )
    X_test_scaled = test_data["X_scaled"]
    X_test_raw = test_data["X"]
    F_test = test_data["f"]
    Y_test = test_data["y"]

    Y_test_pred, Y_test_std = model.predict(X_test_scaled, return_std=True)
    moogp_test = get_metrics(F_test, Y_test, Y_test_pred, Y_test_std**2, train=False, modelname="MOOGP")

    non_ortho_test = None
    if non_ortho_model is not None:
        ypred, ypredstd = non_ortho_model.predict(X_test_scaled, return_std=True)
        non_ortho_test = get_metrics(F_test, Y_test, ypred, ypredstd**2, train=False, modelname="MOGP")

    ols_test = None
    if ols:
        G_ls_test = np.column_stack([np.ones((n_test, 1)), X_test_raw])
        Y_test_pred_ls = G_ls_test @ B_ls
        ols_test = get_metrics(F_test, Y_test, Y_test_pred_ls, yvar=None, train=False, modelname="OLS")

    predictive = {
        "moogp": {"train": moogp_train, "test": moogp_test},
        "mogp": {"train": non_ortho_train, "test": non_ortho_test} if non_ortho_model is not None else None,
        "ols": {"train": ols_train, "test": ols_test} if ols else None,
    }

    # -------------------
    # PLUMLEE-JOSEPH-STYLE TREND TABLE (single scheme)
    # Grid RMSE over equally spaced points + fitted betas
    # -------------------
    X_grid = np.linspace(0.0, 1.0, n_grid).reshape(-1, 1)
    grid_data = generate_forrester_data(
        n=n_grid,
        seed=seed,
        with_error=False,
        X_override=X_grid,
    )
    X_grid_scaled = grid_data["X_scaled"]
    F_grid = grid_data["f"]

    if not 0 <= output_idx < p:
        raise ValueError(f"output_idx must be between 0 and {p - 1}; got {output_idx}.")

    trend_table = {
        "scheme_name": scheme_name,
        "output_idx": output_idx,
        "n_grid": n_grid,
        "rows": {},
        "outputs": []
    }

    trend_rows_by_output = [{"output_idx": j, "rows": {}} for j in range(p)]

    # OLS
    if ols:
        G_grid_ls = np.column_stack([np.ones((n_grid, 1)), X_grid])
        Y_grid_pred_ls = G_grid_ls @ B_ls
        for j, trend_output in enumerate(trend_rows_by_output):
            trend_output["rows"]["ols"] = {
                TREND_GRID_RMSE_KEY: rmse_1d(F_grid[:, j], Y_grid_pred_ls[:, j]),
                "beta0": float(B_ls[0, j]),
                "beta1": float(B_ls[1, j]),
            }

    # MOOGP
    Y_grid_pred_moogp, _ = model.predict(X_grid_scaled, return_std=True)
    beta_moogp = get_model_trend_betas_raw(model)
    for j, trend_output in enumerate(trend_rows_by_output):
        trend_output["rows"]["moogp"] = {
            TREND_GRID_RMSE_KEY: rmse_1d(F_grid[:, j], Y_grid_pred_moogp[:, j]),
            "beta0": float(beta_moogp[0, j]),
            "beta1": float(beta_moogp[1, j]),
        }

    # MOGP
    if non_ortho_model is not None:
        Y_grid_pred_mogp, _ = non_ortho_model.predict(X_grid_scaled, return_std=True)
        beta_mogp = get_model_trend_betas_raw(non_ortho_model)
        for j, trend_output in enumerate(trend_rows_by_output):
            trend_output["rows"]["mogp"] = {
                TREND_GRID_RMSE_KEY: rmse_1d(F_grid[:, j], Y_grid_pred_mogp[:, j]),
                "beta0": float(beta_mogp[0, j]),
                "beta1": float(beta_mogp[1, j]),
            }

    trend_table["outputs"] = trend_rows_by_output
    trend_table["rows"] = trend_rows_by_output[output_idx]["rows"]

    return {
        "predictive": predictive,
        "trend_table": trend_table,
    }

def print_predictive_table(results, scheme_label):
    """
    Prints your original predictive train/test table.
    Expects results to be the output of evaluate_moogp(...).
    """
    predictive = results["predictive"]
    metric_keys = [
        "rmse_y",
        "rmse_f",
        "nrmse_y",
        "coverage_95_y",
        "width_95_y",
        "dss_y",
    ]

    def format_metric(metrics, key):
        value = metrics[key]
        return f"{value:.4f}" if value is not None else "N/A"

    print("-" * 35 + scheme_label + "-" * 35)
    metric_header = " | ".join(f"{PREDICTIVE_METRIC_LABELS[key]:<12}" for key in metric_keys)
    print(f"{'Model':<8} | {'Split':<6} | {metric_header}")
    print("-" * 111)

    for model_key, splits in predictive.items():
        if splits is None:
            continue

        for split_key, metrics in splits.items():
            split_name = "Train" if "train" in split_key else "Test"
            metric_values = " | ".join(f"{format_metric(metrics, key):<12}" for key in metric_keys)

            print(f"{model_key.upper():<8} | {split_name:<6} | {metric_values}")


def print_trend_comparison_table(results_scheme1, results_scheme2, scheme1_label="LHS", scheme2_label="log-LHS"):
    """
    Prints the Plumlee-Joseph-style comparison table using the two trend_table blocks.
    """
    t1 = results_scheme1["trend_table"]
    t2 = results_scheme2["trend_table"]

    outputs1 = t1.get("outputs", [{"output_idx": t1["output_idx"], "rows": t1["rows"]}])
    outputs2 = t2.get("outputs", [{"output_idx": t2["output_idx"], "rows": t2["rows"]}])

    outputs1_by_idx = {entry["output_idx"]: entry["rows"] for entry in outputs1}
    outputs2_by_idx = {entry["output_idx"]: entry["rows"] for entry in outputs2}
    if outputs1_by_idx.keys() != outputs2_by_idx.keys():
        raise ValueError("Both trend tables must contain the same output indices.")

    print()
    row_order = ["ols", "moogp", "mogp"]
    row_names = {"ols": "LS", "moogp": "MOOGP", "mogp": "MOGP"}

    for table_idx, output_idx in enumerate(sorted(outputs1_by_idx)):
        if table_idx > 0:
            print()

        rows1 = outputs1_by_idx[output_idx]
        rows2 = outputs2_by_idx[output_idx]

        print(f"Trend comparison for Output {output_idx + 1}")
        print(f"{'Model':<8} | "
              f"{scheme1_label + ' ' + TREND_GRID_RMSE_LABEL:<20} | {'beta0':<10} | {'beta1':<10} | "
              f"{scheme2_label + ' ' + TREND_GRID_RMSE_LABEL:<20} | {'beta0':<10} | {'beta1':<10} | "
              f"{'|Δbeta0|':<10} | {'|Δbeta1|':<10}")
        print("-" * 132)

        for key in row_order:
            if key not in rows1 or key not in rows2:
                continue

            a = rows1[key]
            b = rows2[key]

            db0 = abs(b["beta0"] - a["beta0"])
            db1 = abs(b["beta1"] - a["beta1"])

            print(f"{row_names[key]:<8} | "
                  f"{a[TREND_GRID_RMSE_KEY]:<20.6f} | {a['beta0']:<10.4f} | {a['beta1']:<10.4f} | "
                  f"{b[TREND_GRID_RMSE_KEY]:<20.6f} | {b['beta0']:<10.4f} | {b['beta1']:<10.4f} | "
                  f"{db0:<10.4f} | {db1:<10.4f}")

if __name__ == "__main__":
    # ----- Choice Variables -----
    
    # original n_train = 25
    n_train = 100
    # original seed 67
    seed = 1154
    
    trend_output_idx = 2

    # --------------------------------
    # ------- Data Generation --------
    #--------------------------------

    outdir = Path(__file__).resolve().parent / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    # LHS data
    data_lhs = generate_forrester_data(
        n=n_train, seed=seed, with_error=True, error_per_output=[10, 1, 0.05]
    )

    # ---- Design 2: log-LHS using the SAME generator via X_override ----
    X_log = log_lhs_1d_rescaled(n_train, seed=seed, xmin=1e-3, cluster="right", include_x0=False, shuffle=False)

    data_log = generate_forrester_data(
        n=n_train, seed=seed, with_error=True, error_per_output=[10, 1, 0.05], X_override=X_log
    )

    start_time = time.perf_counter()
    
    moogp_lhs, X1, Xs1, Y1 = fit_moogp_forrester(n_train=n_train, seed=seed, orthogonal=True,  data=data_lhs)
    mogp_lhs,  _,  _,  _  = fit_moogp_forrester(n_train=n_train, seed=seed, orthogonal=False, data=data_lhs)

    moogp_log, X2, Xs2, Y2 = fit_moogp_forrester(n_train=n_train, seed=seed, orthogonal=True,  data=data_log)
    mogp_log,  _,  _,  _  = fit_moogp_forrester(n_train=n_train, seed=seed, orthogonal=False, data=data_log)

    elapsed = time.perf_counter() - start_time
    print(f"Done in {elapsed:.3f}s")

    fig_pred = plot_forrester_fit(moogp_lhs, X1, Xs1, Y1, non_ortho_model=mogp_lhs)
    fig_pred.savefig(outdir / "forrester_fit_lhs.png", dpi=600, bbox_inches="tight")
    fig_pred.savefig(outdir / "forrester_fit_lhs.pdf", dpi=600, bbox_inches="tight")
    
    fig_pred = plot_forrester_fit(moogp_log, X2, Xs2, Y2, non_ortho_model=mogp_log)
    fig_pred.savefig(outdir / "forrester_fit_log.png", dpi=600, bbox_inches="tight")
    fig_pred.savefig(outdir / "forrester_fit_log.pdf", dpi=600, bbox_inches="tight")

    fig = plot_forrester_fit_side_by_side(
    moogp_lhs=moogp_lhs, X_lhs=data_lhs["X"], Y_lhs=data_lhs["y"], mogp_lhs=mogp_lhs,
    moogp_log=moogp_log, X_log=data_log["X"], Y_log=data_log["y"], mogp_log=mogp_log,
    )

    # Single side-by-side plot
    fig.savefig(outdir / "forrester_fit_lhs_vs_loglhs.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "forrester_fit_lhs_vs_loglhs.pdf", dpi=600, bbox_inches="tight")
    

    fig_trend = plot_trend_recovery_two_designs(
        data_lhs, moogp_lhs, mogp_lhs,
        data_log, moogp_log, mogp_log,
        output_idx=trend_output_idx,  # 0->Output1, 1->Output2, 2->Output3
        left_label="LHS",
        right_label="log-LHS",
    )

    # for this second graph, maybe do a linspace with n=1000 and OLS to get "true" trend
    # then look at the change in coefficients across the data shifts

    fig_trend.savefig(outdir / f"forrester_trend_recovery_output{trend_output_idx + 1}.png", dpi=600, bbox_inches="tight")
    fig_trend.savefig(outdir / f"forrester_trend_recovery_output{trend_output_idx + 1}.pdf", dpi=600, bbox_inches="tight")

    # -----------------------------
    # Evaluation: both tables
    # -----------------------------
    results_lhs = evaluate_moogp(
        moogp_lhs,
        data=data_lhs,
        error_per_output=[10, 1, 0.05],
        non_ortho_model=mogp_lhs,
        ols=True,
        output_idx=trend_output_idx,
        scheme_name="LHS",
    )

    results_log = evaluate_moogp(
        moogp_log,
        data=data_log,
        error_per_output=[10, 1, 0.05],
        non_ortho_model=mogp_log,
        ols=True,
        output_idx=trend_output_idx,
        scheme_name="log-LHS",
    )

    # Original predictive tables
    print_predictive_table(results_lhs, "LHS")
    print()
    print_predictive_table(results_log, "log-LHS")

    # Trend comparison tables
    print()
    print_trend_comparison_table(
        results_lhs,
        results_log,
        scheme1_label="LHS",
        scheme2_label="log-LHS",
    )

    plt.show()
