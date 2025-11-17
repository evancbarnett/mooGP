import numpy as np

from moogp.datasets import generate_borehole_data_nd
from moogp.model import MOOGP


def test_moogp_borehole_small_q_equals_p():
    # small problem so test runs fast
    data = generate_borehole_data_nd(n=15, p=3, seed=11)
    X = data["X_scaled"]
    Y = data["Y"]
    n, d = X.shape
    p = Y.shape[1]

    terms = [None] + list(range(1, d + 1))  # intercept + main effects
    q_latent = p

    rng = np.random.default_rng(0)
    Psi = rng.standard_normal((p, q_latent))
    Psi /= np.maximum(np.linalg.norm(Psi, axis=0, keepdims=True), 1e-12)

    theta0 = []
    bounds = []
    for j in range(q_latent):
        theta0 += [np.log(1.0)]
        theta0 += list(np.log(0.5) * np.ones(d))
        bounds += [(np.log(1e-3), np.log(1e3))] + [(np.log(0.05), np.log(5.0))] * d
    theta0 = np.array(theta0)

    model = MOOGP(
        terms=terms,
        q=q_latent,
        Psi=Psi,
        learn_Psi=False,
        use_reml=False,
        jitter=1e-8,
    )

    model.fit(data=data, theta0=theta0, bounds=bounds,
              optimizer_opts={"maxiter": 50})

    assert model.opt_result.success

    # predictions at training X
    Y_pred, Y_std = model.predict(X, return_std=True)
    assert Y_pred.shape == Y.shape
    assert Y_std.shape == Y.shape

    max_abs_err = np.max(np.abs(Y_pred - Y))
    # not exact (we have jitter + numerical error),
    # but should be very close
    assert max_abs_err < 1e-2
