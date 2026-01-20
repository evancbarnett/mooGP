# moogp/main_borehole_example.py
import time
import numpy as np
from scipy.stats import qmc

from .datasets import (
    generate_borehole_data_nd,
    tstd2theta,
    borehole_vec_physical,
)
from .model import MOOGP

if __name__ == "__main__":
    # 1) data
    data = generate_borehole_data_nd(n=100, p=4, seed=67)
    X = data["X_scaled"]
    Y = data["Y"]
    locations_phys = data["locations_phys"]

    d = X.shape[1]
    p = Y.shape[1]
    terms = [None] + list(range(1, d + 1))

    # 2) mixing matrix
    q = 4
    rng = np.random.default_rng(0)
    Psi = rng.standard_normal((p, q))
    Psi /= np.maximum(np.linalg.norm(Psi, axis=0, keepdims=True), 1e-12)

    # 3) theta0 and bounds (same as before)
    theta0 = []
    bounds = []
    for j in range(q):
        theta0 += [np.log(1.0)]
        theta0 += list(np.log(0.5) * np.ones(d))
        bounds += [(np.log(1e-6), np.log(1e3))] + [(np.log(0.05), np.log(5.0))] * d
    theta0 = np.array(theta0)

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=Psi,
        learn_Psi=False,
        learn_sigma_eps=False,
        use_reml=False,
        jitter=1e-6,
    )

    t0 = time.perf_counter()
    model.fit(data=data, theta0=theta0, bounds=bounds)
    elapsed = time.perf_counter() - t0

    print(f"Training took {elapsed:.3f} s")
    print("Optimization success:", model.opt_result.success, "NLL:", model.nll_hat)

    # 4) predict
    n_star = 30
    theta_star_u = qmc.LatinHypercube(d=d, scramble=True, rng=999).random(n_star)
    theta_star_scaled = 2.0 * (theta_star_u - 0.5)

    Y_mean, Y_std = model.predict(theta_star_scaled, return_std=True)
    print("Pred mean shape:", Y_mean.shape, "Pred std shape:", Y_std.shape)

    # 5) compare to physical borehole function
    theta_star_phys = tstd2theta(theta_star_u)
    theta_stack = np.repeat(theta_star_phys, repeats=p, axis=0)
    x_stack = np.tile(locations_phys, (n_star, 1))
    Y_true = borehole_vec_physical(x_stack, theta_stack).reshape(n_star, p)

    rmse = np.sqrt(np.mean((Y_mean - Y_true) ** 2, axis=0))
    print("RMSE per output:", rmse)
