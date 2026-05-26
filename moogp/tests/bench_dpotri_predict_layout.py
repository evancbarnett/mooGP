"""End-to-end fit + predict benchmark for the dpotri / cache-skip / predict-solve / layout changes.

Run with the moogp venv:

    .venv/bin/python -m moogp.tests.bench_dpotri_predict_layout

The script is written so it can be executed with either the optimized or the
baseline ``moogp/model.py`` on disk. The driver shell script in this same
directory swaps files between runs and parses the printed JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from moogp.model import MOOGP


def _make_data(n, p, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, d))
    cols = []
    for k in range(p):
        j0 = k % d
        j1 = (k + 1) % d
        cols.append(np.sin(1.7 * X[:, j0]) + 0.5 * X[:, j1] + 0.1 * (k + 1))
    Y = np.column_stack(cols) + 1e-3 * rng.standard_normal((n, p))
    return X, Y


def _theta(q, d, p, seed=1):
    rng = np.random.default_rng(seed)
    th = []
    for _ in range(q):
        th.append(float(np.log(rng.uniform(0.6, 1.4))))
        th.extend([float(np.log(rng.uniform(0.4, 1.2))) for _ in range(d)])
    th.extend(list(np.log(5e-3 * np.ones(p))))
    return np.asarray(th, dtype=float)


def _make_model(d):
    return MOOGP(
        terms=[None] + list(range(1, d + 1)), q=5, Psi=None,
        orthogonal=True, learn_Psi=False, learn_sigma_eps=True, jitter=0.0,
        use_diagonalized_interaction=True, use_slow_kyinv=False,
        standardize_y=False, use_analytical_grad=True,
        standardize_x=False,
    )


def _bounds(q, d, p):
    # Keep ell and sigma in sensible ranges so the optimizer cannot wander into
    # numerically degenerate territory and the benchmark stays comparable
    # across baseline / optimized runs.
    bnds = []
    for _ in range(q):
        bnds.append((np.log(1e-2), np.log(1e2)))            # log sigma2
        bnds.extend([(np.log(0.1), np.log(5.0))] * d)         # log ell_j
    bnds.extend([(np.log(1e-6), np.log(1.0))] * p)            # log sigma_eps2
    return bnds


def _time_fit(n, p, d, q, n_iter, seed=0, maxiter=100):
    X, Y = _make_data(n, p, d=d, seed=seed)
    theta0 = _theta(q, d, p, seed=seed + 1)
    bounds = _bounds(q, d, p)

    times_total = []
    times_per_call = []
    last_model = None
    for rep in range(n_iter):
        m = _make_model(d)
        t0 = time.perf_counter()
        m.fit({"X_scaled": X, "y": Y}, theta0, bounds=bounds,
              optimizer_opts={"maxiter": maxiter})
        dt = time.perf_counter() - t0
        n_calls = m.opt_result.nfev
        times_total.append(dt)
        times_per_call.append(dt / max(1, n_calls))
        last_model = m

    return last_model, np.array(times_total), np.array(times_per_call)


def _time_predict(model, n_star, n_iter, seed=2):
    rng = np.random.default_rng(seed)
    d = model.X.shape[1]
    Xs = rng.uniform(-1, 1, size=(n_star, d))

    # warm-up
    model.predict(Xs, return_std=True)

    t_mean = []
    t_full = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.predict(Xs, return_std=False)
        t_mean.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.predict(Xs, return_std=True)
        t_full.append(time.perf_counter() - t0)
    return np.array(t_mean), np.array(t_full)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grid", type=str, default="200,400,800,1200")
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--predict-reps", type=int, default=10)
    parser.add_argument("--n-star", type=int, default=250)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--label", type=str, default="run")
    args = parser.parse_args()

    ns = [int(x) for x in args.n_grid.split(",")]
    print(f"# label={args.label} threads={os.environ.get('OMP_NUM_THREADS', 'default')}", flush=True)
    print(f"{'n':>5} {'fit_med_s':>10} {'fit_per_call_ms':>16} {'pred_mean_ms':>13} {'pred_full_ms':>13}", flush=True)

    rows = []
    for n in ns:
        model, t_total, t_per_call = _time_fit(
            n, args.p, args.d, args.q, n_iter=args.reps, maxiter=args.maxiter
        )
        t_mean, t_full = _time_predict(model, args.n_star, n_iter=args.predict_reps)
        row = dict(
            n=int(n), p=args.p, q=args.q, d=args.d,
            fit_med_s=float(np.median(t_total)),
            fit_per_call_ms=float(1e3 * np.median(t_per_call)),
            pred_mean_ms=float(1e3 * np.median(t_mean)),
            pred_full_ms=float(1e3 * np.median(t_full)),
            nfev_med=int(np.median([model.opt_result.nfev])),
            nll=float(model.nll_hat),
        )
        rows.append(row)
        print(
            f"{row['n']:>5} {row['fit_med_s']:>10.3f} {row['fit_per_call_ms']:>16.2f} "
            f"{row['pred_mean_ms']:>13.3f} {row['pred_full_ms']:>13.3f}",
            flush=True,
        )

    print("JSON " + json.dumps({"label": args.label, "rows": rows}))


if __name__ == "__main__":
    main()
