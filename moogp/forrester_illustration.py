import numpy as np
import matplotlib.pyplot as plt

from moogp.design import make_G
from moogp.datasets import generate_forrester_data
from moogp.model import MOOGP


def forrester_true_functions(X):
    """
    X: (n,1) in [0,1]
    Returns y_true: (n,3) matching your generate_forrester_data definition.
    """
    x = X.reshape(-1, 1)

    f1 = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
    f2 = 0.5 * f1 + 5 * (x - 0.5) + 5
    f3 = -0.8 * f1 - 5 * (x - 0.5) - 4

    return np.concatenate([f1, f2, f3], axis=1)


def fit_moogp_forrester(n_train=25, seed=0,q=3, learn_psi=False):
    # 1) Generate data
    data = generate_forrester_data(n=n_train, seed=seed, with_error=True)
    X = data["X"]          # in [0,1]
    X_scaled = data["X_scaled"]  # in [-1,1]
    Y = data["y"]          # (n,3)

    n, d = X_scaled.shape
    p = Y.shape[1]

    # 2) Mean basis: intercept + main effect
    terms = [None] + list(range(1, d + 1))

    # 3) Latent dimension and Psi
    if learn_psi:
        Psi = None          # learned from theta
    else:
        Psi = np.eye(p)     # each latent → one output

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=learn_psi,
        use_reml=False,
        jitter=1e-2,
        one_based=True,
        normalize_cols=True,
    )

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

    if learn_psi:
        # add free Psi params at the end
        rng = np.random.default_rng(0)
        Psi0_free = rng.standard_normal((p, q))
        theta0 = np.concatenate([theta0, Psi0_free.ravel()])

        # bounds for Psi entries
        bounds.extend([(-5.0, 5.0)] * (p * q))

    # 5) Fit
    model.fit(
        data={"X_scaled": X_scaled, "y": Y},
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": 100},
    )

    return model, X, X_scaled, Y


def plot_forrester_fit(model, X, X_scaled, Y, n_plot=200):
    # Make a dense grid in [0,1] and scale to [-1,1]
    X_plot = np.linspace(0.0, 1.0, n_plot).reshape(-1, 1)
    X_plot_scaled = 2.0 * (X_plot - 0.5)  # same scaling as generate_forrester_data

    # Predict
    mean, std = model.predict(X_plot_scaled, return_std=True)

    # True functions at the same points
    Y_true_plot = forrester_true_functions(X_plot)

    # Plot per-output
    p = Y.shape[1]
    fig, axes = plt.subplots(p, 1, figsize=(8, 3 * p), sharex=True)

    if p == 1:
        axes = [axes]

    titles = [f"Output {j+1}" for j in range(p)]

    for j in range(p):
        ax = axes[j]

        # Training data
        ax.scatter(
            X[:, 0],
            Y[:, j],
            color="black",
            s=25,
            alpha=0.7,
            label="Training data",
        )

        # True function
        ax.plot(
            X_plot[:, 0],
            Y_true_plot[:, j],
            linestyle="--",
            linewidth=1.5,
            label="True function",
        )

        # Predictive mean
        ax.plot(
            X_plot[:, 0],
            mean[:, j],
            linewidth=2.0,
            label="Predictive mean",
        )

        # ± 2 std band
        upper = mean[:, j] + 2.0 * std[:, j]
        lower = mean[:, j] - 2.0 * std[:, j]
        ax.fill_between(
            X_plot[:, 0],
            lower,
            upper,
            alpha=0.2,
            label="±2 std" if j == 0 else None,
        )

        ax.set_ylabel(f"y_{j+1}")
        ax.set_title(titles[j])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("x")
    axes[0].legend(loc="best")
    fig.tight_layout()
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

def evaluate_moogp(model, label="", n_test=200, seed=999):
    """
    Compute train/test RMSE + RMSPE per output for a fitted MOOGP,
    and compare against a least-squares baseline (linear in x).

    Assumes model was fit on data with keys:
      - 'X_scaled' : (n,d) in [-1,1]^d
      - 'y'        : (n,p)
    Optionally:
      - 'X'        : (n,d) in original physical space (here [0,1] for Forrester)
    """
    print(f"\n=== Evaluation: {label} ===")

    # -----------------------
    # Helpers
    # -----------------------
    def rmse(y_hat, y):
        return np.sqrt(np.mean((y_hat - y) ** 2, axis=0))

    def rmspe(y_hat, y):
        denom = np.maximum(np.abs(y), 1e-8)
        return np.sqrt(np.mean(((y_hat - y) / denom) ** 2, axis=0)) * 100.0

    # -----------------------
    # Train data & MOOGP preds
    # -----------------------
    train_data = model._data  # set inside model.fit via _prepare_data
    X_train_scaled = train_data["X_scaled"]
    Y_train = train_data["y"]
    n_train, d = X_train_scaled.shape
    p = Y_train.shape[1]

    # Recover raw X if available, else invert scaling from [-1,1] to [0,1]
    if "X" in train_data:
        X_train_raw = train_data["X"]
    else:
        # Only correct for Forrester where bounds are [0,1]
        X_train_raw = 0.5 * (X_train_scaled + 1.0)

    # MOOGP predictions on train
    Y_train_pred, Y_train_std = model.predict(X_train_scaled, return_std=True)

    # -----------------------
    # OLS baseline fit (train)
    # Model: y ~ a + sum_k b_k x_k
    # -----------------------
    G_ls_train = np.column_stack([np.ones((n_train, 1)), X_train_raw])  # (n_train, 1 + d)
    # Fit all outputs at once: B_ls shape (1 + d, p)
    B_ls, *_ = np.linalg.lstsq(G_ls_train, Y_train, rcond=None)
    Y_train_pred_ls = G_ls_train @ B_ls  # (n_train, p)

    # -----------------------
    # Train metrics
    # -----------------------
    train_rmse_moogp = rmse(Y_train_pred, Y_train)
    train_rmspe_moogp = rmspe(Y_train_pred, Y_train)

    train_rmse_ls = rmse(Y_train_pred_ls, Y_train)
    train_rmspe_ls = rmspe(Y_train_pred_ls, Y_train)

    print("Train RMSE per output (MOOGP):", train_rmse_moogp)
    print("Train RMSE per output (OLS)  :", train_rmse_ls)
    print("Train RMSE mean   MOOGP vs OLS:",
          train_rmse_moogp.mean(), "vs", train_rmse_ls.mean())

    # print("Train RMSPE (%) per output (MOOGP):", train_rmspe_moogp)
    # print("Train RMSPE (%) per output (OLS)  :", train_rmspe_ls)
    # print("Train RMSPE (%) mean MOOGP vs OLS:",
    #       train_rmspe_moogp.mean(), "vs", train_rmspe_ls.mean())

    # -----------------------
    # Test data & predictions
    # -----------------------
    test_data = generate_forrester_data(n=n_test, seed=seed)
    X_test_scaled = test_data["X_scaled"]
    X_test_raw = test_data["X"]  # Forrester generator gives this
    Y_test = test_data["y"]

    # MOOGP predictions on test
    Y_test_pred, Y_test_std = model.predict(X_test_scaled, return_std=True)

    # OLS baseline predictions on test (using B_ls from train)
    G_ls_test = np.column_stack([np.ones((n_test, 1)), X_test_raw])
    Y_test_pred_ls = G_ls_test @ B_ls

    # -----------------------
    # Test metrics
    # -----------------------
    test_rmse_moogp = rmse(Y_test_pred, Y_test)
    test_rmspe_moogp = rmspe(Y_test_pred, Y_test)

    test_rmse_ls = rmse(Y_test_pred_ls, Y_test)
    test_rmspe_ls = rmspe(Y_test_pred_ls, Y_test)

    print("Test RMSE per output (MOOGP):", test_rmse_moogp)
    print("Test RMSE per output (OLS)  :", test_rmse_ls)
    print("Test RMSE mean   MOOGP vs OLS:",
          test_rmse_moogp.mean(), "vs", test_rmse_ls.mean())

    # print("Test RMSPE (%) per output (MOOGP):", test_rmspe_moogp)
    # print("Test RMSPE (%) per output (OLS)  :", test_rmspe_ls)
    # print("Test RMSPE (%) mean MOOGP vs OLS:",
    #       test_rmspe_moogp.mean(), "vs", test_rmspe_ls.mean())

    # -----------------------
    # Coverage of ±2σ band (MOOGP only, on test)
    # -----------------------
    lower = Y_test_pred - 2.0 * Y_test_std
    upper = Y_test_pred + 2.0 * Y_test_std
    covered = (Y_test >= lower) & (Y_test <= upper)
    coverage_per_output = covered.mean(axis=0)
    print("Test coverage of ±2 std per output (MOOGP):", coverage_per_output)

    return {
        "moogp": {
            "train_rmse": train_rmse_moogp,
            "train_rmspe": train_rmspe_moogp,
            "test_rmse": test_rmse_moogp,
            "test_rmspe": test_rmspe_moogp,
            "coverage": coverage_per_output,
        },
        "ols": {
            "train_rmse": train_rmse_ls,
            "train_rmspe": train_rmspe_ls,
            "test_rmse": test_rmse_ls,
            "test_rmspe": test_rmspe_ls,
        },
    }



if __name__ == "__main__":
    # Example 1: full-rank (q=p, Psi=I) – should be very accurate
    model, X, X_scaled, Y = fit_moogp_forrester(
        n_train=25,
        seed=0,
        learn_psi=False,   # Psi = I_p
    )
    fig1 = plot_forrester_fit(model, X, X_scaled, Y)
    fig2 = plot_pred_vs_true(model)
    fig3 = plot_trend_vs_ls(
        model, X, X_scaled, Y,
        title_suffix=" (Psi = Identity)"
    )
    evaluate_moogp(model, label="Full-rank (q=p, Psi=I)")
    plt.show()

    # Example 2: low-rank with learn_Psi=True – see how capacity changes
    model_lr, X_lr, X_scaled_lr, Y_lr = fit_moogp_forrester(
        n_train=25,
        seed=1,
        q=2,
        learn_psi=True,    # q=2 < p, Psi learned
    )
    fig4 = plot_forrester_fit(model_lr, X_lr, X_scaled_lr, Y_lr)
    fig5 = plot_pred_vs_true(model_lr)
    fig6 = plot_trend_vs_ls(
        model_lr, X_lr, X_scaled_lr, Y_lr,
        title_suffix=" (Psi Learned)"
    )
    evaluate_moogp(model_lr, label="Psi Learned")
    plt.show()
