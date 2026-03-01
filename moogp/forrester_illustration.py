import time

import numpy as np
import matplotlib.pyplot as plt

from moogp.design import make_G
from moogp.datasets import generate_forrester_data, log_lhs_1d_rescaled
from moogp.model import MOOGP
from moogp.evaluation import *
from pathlib import Path


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

def fit_moogp_forrester(n_train=50, 
                        seed=0,
                        q=3, 
                        Psi = None, 
                        orthogonal=True,
                        use_fast=True, 
                        learn_Psi=False,
                        data=None):
    if data is None:
        data = generate_forrester_data(
            n=n_train,
            seed=seed,
            with_error=True,
            error_per_output=[10, 1, 0.05], 
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
        jitter=1e-6,
        normalize_cols=True,
        use_diagonalized_interaction=use_fast  # Fast computation
    )
    # TODO Decide whether to have default bounds inside MOOGP class
    # 4) Build theta0 and bounds
    theta0 = []
    bounds = []
    for _ in range(q):
        # log sigma^2
        theta0.append(np.log(1.0))
        # log lengthscales (d dims)
        theta0.extend([np.log(0.5)] * d)

        bounds.append((np.log(1e-3), np.log(1e3)))       # sigma^2
        bounds.extend([(np.log(0.05), np.log(5.0))] * d) # ell bounds

    theta0 = np.array(theta0)

    # include parameters for Sigma eps
    y_var = Y.var(axis=0, ddof=1)  
    sigma_eps2_init = np.log(1e-2 * y_var)
    theta0 = np.concatenate([theta0, sigma_eps2_init])

    # Bounds for Sigma eps 
    lb = np.maximum(1e-12, 1e-6 * y_var)
    ub = np.maximum(lb * 10.0, 0.5 * y_var) 

    log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]
    bounds.extend(log_bounds)

    # 5) Fit
    model.fit(
        data={"X_scaled": X_scaled, "y": Y},
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 500},
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
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
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

        # Legend on Output 3 if it exists; otherwise on last axis.
        legend_ax_idx = min(2, p - 1)

        # Model styling (color + linestyle). Keep linestyle difference for B/W safety.
        moogp_style = dict(color="tab:blue", linestyle="--")
        mogp_style  = dict(color="tab:orange", linestyle=":")

        def plot_mean_and_interval_lines(
            ax, x, mean, std, *,
            label_mean=None,
            color="tab:blue",
            linestyle="-",
            lw_mean=1.9,
            lw_bound=1.3,
            alpha_fill=0.12
        ):
            upper = mean + 2.0 * std
            lower = mean - 2.0 * std

            # Fill (no label)
            ax.fill_between(
                x, lower, upper,
                color=color,
                alpha=alpha_fill,
                linewidth=0.0,
                zorder=1
            )

            # Boundary lines (no labels; same formatting as mean)
            ax.plot(x, upper, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)
            ax.plot(x, lower, color=color, linestyle=linestyle, linewidth=lw_bound, zorder=3)

            # Mean line (label only on legend axis)
            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=lw_mean, label=label_mean, zorder=4)

        for j, ax in enumerate(axes):
            add_labels = (j == legend_ax_idx)

            # Training data
            ax.scatter(
                X[:, 0], Y[:, j],
                s=18,
                color="black",
                alpha=0.75,
                linewidths=0.0,
                label="Training data" if add_labels else None,
                zorder=5
            )

            # True function: solid black (requested)
            ax.plot(
                X_plot[:, 0], Y_true_plot[:, j],
                linestyle="-",
                linewidth=1.4,
                color="black",
                label="True function" if add_labels else None,
                zorder=2
            )

            # MOOGP
            plot_mean_and_interval_lines(
                ax,
                X_plot[:, 0],
                mean_moogp[:, j],
                std_moogp[:, j],
                label_mean="MOOGP mean" if add_labels else None,
                color=moogp_style["color"],
                linestyle=moogp_style["linestyle"],
                lw_mean=1.9,
                lw_bound=1.3,
                alpha_fill=0.12
            )

            # MOGP (non-ortho)
            if non_ortho_model is not None:
                plot_mean_and_interval_lines(
                    ax,
                    X_plot[:, 0],
                    mean_mogp[:, j],
                    std_mogp[:, j],
                    label_mean="MOGP mean" if add_labels else None,
                    color=mogp_style["color"],
                    linestyle=mogp_style["linestyle"],
                    lw_mean=1.9,
                    lw_bound=1.3,
                    alpha_fill=0.10
                )

            ax.set_ylabel(f"Output {j+1}")
            ax.tick_params(axis="both", which="both", top=False, right=False)

        axes[-1].set_xlabel("x")

        # Legend bottom-left of Output 3 panel; de-duplicate.
        leg_ax = axes[legend_ax_idx]
        handles, labels = leg_ax.get_legend_handles_labels()
        seen = set()
        uniq = [(h, l) for h, l in zip(handles, labels) if l and (l not in seen and not seen.add(l))]
        if uniq:
            leg_ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                loc="lower left",
                frameon=False,
                handlelength=2.8,
                borderaxespad=0.6
            )

        return fig

def plot_trend_recovery_two_designs(
    data_lhs, moogp_lhs, mogp_lhs,
    data_log, moogp_log, mogp_log,
    output_idx=0,
    n_plot=400,
    left_label="LHS",
    right_label="log-LHS (rescaled)",
):
    """
    Two panels (like the example):
      left: trends fit under LHS
      right: trends fit under log-LHS (rescaled)

    Shows: Training points, True function (solid black),
           Least Squares trend, MOOGP trend, MOGP trend.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

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

    def trend_from_model(model, X_plot_scaled):
        Gp = make_G({"X_scaled": X_plot_scaled}, model.terms,
                    one_based=model.one_based, return_names=False)
        Bhat = model.cache["bhat"]  # (r, p)
        return (Gp @ Bhat)[:, output_idx]

    def ols_trend(X_train, Y_train, X_plot):
        G = np.column_stack([np.ones_like(X_train[:, 0]), X_train[:, 0]])
        b, *_ = np.linalg.lstsq(G, Y_train[:, [output_idx]], rcond=None)  # (2,1)
        Gp = np.column_stack([np.ones_like(X_plot[:, 0]), X_plot[:, 0]])
        return (Gp @ b).ravel()

    with mpl.rc_context(paper_rc):
        X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
        X_plot_scaled = 2.0 * (X_plot - 0.5)
        y_true = forrester_true_functions(X_plot)[:, output_idx]

        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.8), sharey=True, constrained_layout=True)

        # Styles (match the example: distinct line patterns + color)
        true_style  = dict(color="black", linestyle="-",  linewidth=2.8)
        pts_style   = dict(color="black", s=70, zorder=5)
        ls_style    = dict(color="tab:green", linestyle=":", linewidth=3.6)
        moogp_style = dict(color="tab:blue",  linestyle="--", linewidth=3.6)
        mogp_style  = dict(color="tab:red",   linestyle="-",  linewidth=3.6, alpha=0.55)

        panels = [
            (axes[0], data_lhs, moogp_lhs, mogp_lhs, left_label),
            (axes[1], data_log, moogp_log, mogp_log, right_label),
        ]

        for ax, data, moogp, mogp, panel_label in panels:
            Xtr = data["X"]
            Ytr = data["y"]

            ax.scatter(Xtr[:, 0], Ytr[:, output_idx], **pts_style)

            ax.plot(X_plot[:, 0], y_true, label="True function", **true_style)

            y_ls = ols_trend(Xtr, Ytr, X_plot)
            ax.plot(X_plot[:, 0], y_ls, label="Least squares", **ls_style)

            y_moogp = trend_from_model(moogp, X_plot_scaled)
            ax.plot(X_plot[:, 0], y_moogp, label="MOOGP trend", **moogp_style)

            y_mogp = trend_from_model(mogp, X_plot_scaled)
            ax.plot(X_plot[:, 0], y_mogp, label="MOGP trend", **mogp_style)

            ax.set_xlabel("input")
            ax.text(0.03, 0.95, panel_label, transform=ax.transAxes, va="top")

        axes[0].set_ylabel("output")

        # Legend in bottom-left of RIGHT panel (like your earlier preference)
        axes[1].legend(loc="lower right", frameon=False, handlelength=2.8)

        return fig

def plot_pred_vs_true(model, n_test=50, seed=123):
    # Generate a test set
    data_test = generate_forrester_data(n=n_test, seed=seed)
    X_test = data_test["X"]
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

def evaluate_moogp(model, data, error_per_output, non_ortho_model=None, ols=True, n_test=200, seed=999):

    # Generate data
    
    X_train_scaled = data["X_scaled"]
    X_train_raw = data["X"]
    Y_train = data["y"]
    F_train = data["f"]
    
    n_train, d = X_train_scaled.shape
    p = Y_train.shape[1]
    
    def get_metrics(ftrue, ytrue, ypred, yvar=None, train=False, modelname=""):
        rec_rmse_val = rmse(ftrue, ypred)
        pred_rmse_val = rmse(ytrue, ypred)
        nrmse_val = normalized_rmse(ytrue, ypred)
        
        if yvar is not None:
            pcover, pwidth = intervalstats(ytrue, ypred, yvar)
            dss_val = dss(ytrue, ypred, yvar, use_diag=True)
        else:
            pcover = pwidth = dss_val = None

        split_label = "train set metrics" if train else "test set metrics"

        
        return {
            'modelname': modelname,
            'modeltrain': split_label,
            'predrmse': pred_rmse_val,
            'recrmse': rec_rmse_val,
            'nrmse': nrmse_val,
            'pcover': pcover,
            'pwidth': pwidth,
            'dss': dss_val
        }

    # -------------------
    # TRAIN METRICS
    # -------------------
    Y_train_pred, Y_train_std = model.predict(X_train_scaled, return_std=True)
    moogp_train = get_metrics(F_train, Y_train, Y_train_pred, Y_train_std**2, train=True, modelname="MOOGP")
    
    non_ortho_train = None
    if non_ortho_model:
        ypred, ypredstd = non_ortho_model.predict(X_train_scaled, return_std=True)
        non_ortho_train = get_metrics(F_train,Y_train, ypred, ypredstd**2, train=True, modelname="MOGP")
        
    ols_train = None
    if ols:
        G_ls_train = np.column_stack([np.ones((n_train, 1)), X_train_raw])
        B_ls, *_ = np.linalg.lstsq(G_ls_train, Y_train, rcond=None)
        Y_train_pred_ls = G_ls_train @ B_ls
        ols_train = get_metrics(F_train, Y_train, Y_train_pred_ls, yvar=None, train=True, modelname="OLS")
    
    # -------------------
    # TEST METRICS
    # -------------------
    test_data = generate_forrester_data(n=n_test, seed=seed, with_error=True, error_per_output=error_per_output)
    X_test_scaled = test_data["X_scaled"]
    X_test_raw = test_data["X"]
    F_test = test_data["f"] 
    Y_test = test_data["y"]

    Y_test_pred, Y_test_std = model.predict(X_test_scaled, return_std=True)
    moogp_test = get_metrics(F_test,Y_test, Y_test_pred, Y_test_std**2, train=False, modelname="MOOGP")

    non_ortho_test = None
    if non_ortho_model:
        ypred, ypredstd = non_ortho_model.predict(X_test_scaled, return_std=True)
        non_ortho_test = get_metrics(F_test,Y_test, ypred, ypredstd**2, train=False, modelname="MOGP")
    
    ols_test = None
    if ols:
        G_ls_test = np.column_stack([np.ones((n_test, 1)), X_test_raw])
        Y_test_pred_ls = G_ls_test @ B_ls
        # CORRECTED: Evaluated on Y_test instead of Y_train, train=False
        ols_test = get_metrics(F_test,Y_test, Y_test_pred_ls, yvar=None, train=False, modelname="OLS")
    
    # CORRECTED: Standardized return dict keys
    return {
        "moogp": {"train": moogp_train, "test": moogp_test},
        "mogp": {"train": non_ortho_train, "test": non_ortho_test} if non_ortho_model else None,
        "ols": {"train": ols_train, "test": ols_test} if ols else None,
    }


if __name__ == "__main__":
    n_train = 25
    # original seed 67
    seed = 2

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
    # fig_pred.savefig(outdir / "forrester_fit_lhs.png", dpi=600, bbox_inches="tight")
    # fig_pred.savefig(outdir / "forrester_fit_lhs.pdf", dpi=600, bbox_inches="tight")
    
    fig_pred = plot_forrester_fit(moogp_log, X2, Xs2, Y2, non_ortho_model=mogp_log)
    # fig_pred.savefig(outdir / "forrester_fit_log.png", dpi=600, bbox_inches="tight")
    # fig_pred.savefig(outdir / "forrester_fit_log.pdf", dpi=600, bbox_inches="tight")
    

    fig_trend = plot_trend_recovery_two_designs(
        data_lhs, moogp_lhs, mogp_lhs,
        data_log, moogp_log, mogp_log,
        output_idx=1,  # 0->Output1, 1->Output2, 2->Output3
        left_label="LHS",
        right_label="log-LHS (rescaled)",
    )

    # for this second graph, maybe do a linspace with n=1000 and OLS to get "true" trend
    # then look at the change in coefficients across the data shifts

    fig_trend.savefig(outdir / "forrester_trend_recovery.png", dpi=600, bbox_inches="tight")
    fig_trend.savefig(outdir / "forrester_trend_recovery.pdf", dpi=600, bbox_inches="tight")

    
    results = evaluate_moogp(moogp_lhs, data=data_lhs, error_per_output=[10, 1, 0.05],
                                 non_ortho_model=mogp_lhs,ols=True)

    # Print the header
    print("-"*35 + "LHS"+ "-"*35)
    print(f"{'Model':<8} | {'Split':<6} | {'Pred RMSE':<8} | {'Rec RMSE':<8}| {'NRMSE':<8} | {'Coverage':<10} | {'Width':<8} | {'DSS':<8}")
    print("-" * 85)

    # Iterate through the models and their splits
    for model_key, splits in results.items():
        if splits is None:
            continue # Skip if model wasn't run (e.g., MOGP or OLS)
            
        for split_key, metrics in splits.items():
            # Handle cases where OLS might not have variance metrics
            pred_rmse_val = f"{metrics['predrmse']:.4f}" if metrics['predrmse'] is not None else "N/A"
            rec_rmse_val = f"{metrics['recrmse']:.4f}" if metrics['recrmse'] is not None else "N/A"
            nrmse_val = f"{metrics['nrmse']:.4f}" if metrics['nrmse'] is not None else "N/A"
            pcover_val = f"{metrics['pcover']:.4f}" if metrics['pcover'] is not None else "N/A"
            pwidth_val = f"{metrics['pwidth']:.4f}" if metrics['pwidth'] is not None else "N/A"
            dss_val = f"{metrics['dss']:.4f}" if metrics['dss'] is not None else "N/A"
            
            split_name = "Train" if "train" in split_key else "Test"
            
            print(f"{model_key.upper():<8} | {split_name:<6} | {pred_rmse_val:<8} | {rec_rmse_val:<8} | {nrmse_val:<8} | {pcover_val:<10} | {pwidth_val:<8} | {dss_val:<8}")

    results_log = evaluate_moogp(moogp_log, data=data_log, error_per_output=[10, 1, 0.05],
                                 non_ortho_model=mogp_log,ols=True)

    print("-"*30 + "Log"+ "-"*30)

    print(f"{'Model':<8} | {'Split':<6} | {'Pred RMSE':<8} | {'Rec RMSE':<8}| {'NRMSE':<8} | {'Coverage':<10} | {'Width':<8} | {'DSS':<8}")
    print("-" * 85)

    # Iterate through the models and their splits
    for model_key, splits in results_log.items():
        if splits is None:
            continue # Skip if model wasn't run (e.g., MOGP or OLS)
            
        for split_key, metrics in splits.items():
            # Handle cases where OLS might not have variance metrics
            pred_rmse_val = f"{metrics['predrmse']:.4f}" if metrics['predrmse'] is not None else "N/A"
            rec_rmse_val = f"{metrics['recrmse']:.4f}" if metrics['recrmse'] is not None else "N/A"
            nrmse_val = f"{metrics['nrmse']:.4f}" if metrics['nrmse'] is not None else "N/A"
            pcover_val = f"{metrics['pcover']:.4f}" if metrics['pcover'] is not None else "N/A"
            pwidth_val = f"{metrics['pwidth']:.4f}" if metrics['pwidth'] is not None else "N/A"
            dss_val = f"{metrics['dss']:.4f}" if metrics['dss'] is not None else "N/A"
            
            split_name = "Train" if "train" in split_key else "Test"
            
            print(f"{model_key.upper():<8} | {split_name:<6} | {pred_rmse_val:<8} | {rec_rmse_val:<8} | {nrmse_val:<8} | {pcover_val:<10} | {pwidth_val:<8} | {dss_val:<8}")
    
    plt.show()
