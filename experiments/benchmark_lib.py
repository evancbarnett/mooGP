from __future__ import annotations

import csv
import hashlib
from importlib import metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

from moogp.datasets import (
    borehole_vec_physical,
    generate_borehole_data_nd,
    generate_forrester_data,
    tstd2theta,
)
from moogp.evaluation import dss, intervalstats, rmse


RESULT_COLUMNS = [
    "run_id",
    "function",
    "method",
    "n",
    "p",
    "rep",
    "seed_data",
    "seed_model",
    "status",
    "error",
    "train_time_sec",
    "pred_time_sec",
    "rmse",
    "coverage_95",
    "interval_len_95",
    "dss_diag",
    "dss_full",
]

SUPPORTED_FUNCTIONS = ("borehole", "forrester_mixed")
SUPPORTED_METHODS = ("MOOGP", "MOGP", "LCGP", "OILMM", "PUQ")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOOGP_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_OILMM_PYTHON = REPO_ROOT / ".venv-oilmm" / "bin" / "python"


@dataclass(frozen=True)
class ExperimentConfig:
    """Serializable configuration for a benchmark sweep."""

    functions: tuple[str, ...]
    methods: tuple[str, ...]
    sample_sizes: tuple[int, ...]
    output_dims: tuple[int, ...]
    reps: int
    n_test: int
    q: int
    maxiter: int
    jitter: float
    noise_var_frac: float
    use_fast: bool
    jobs: int
    base_seed: int
    results_dir: Path
    moogp_python: Path = field(default=DEFAULT_MOOGP_PYTHON)
    oilmm_python: Path = field(default=DEFAULT_OILMM_PYTHON)

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results_dir"] = str(self.results_dir)
        payload["moogp_python"] = str(self.moogp_python)
        payload["oilmm_python"] = str(self.oilmm_python)
        return payload

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            functions=tuple(payload["functions"]),
            methods=tuple(payload["methods"]),
            sample_sizes=tuple(payload["sample_sizes"]),
            output_dims=tuple(payload["output_dims"]),
            reps=int(payload["reps"]),
            n_test=int(payload["n_test"]),
            q=int(payload["q"]),
            maxiter=int(payload["maxiter"]),
            jitter=float(payload["jitter"]),
            noise_var_frac=float(payload["noise_var_frac"]),
            use_fast=bool(payload["use_fast"]),
            jobs=int(payload["jobs"]),
            base_seed=int(payload["base_seed"]),
            results_dir=Path(payload["results_dir"]),
            moogp_python=Path(payload.get("moogp_python", DEFAULT_MOOGP_PYTHON)),
            oilmm_python=Path(payload.get("oilmm_python", DEFAULT_OILMM_PYTHON)),
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Training and test data for one benchmark cell."""

    function: str
    n: int
    p: int
    seed_data: int
    train_data: dict[str, np.ndarray]
    test_X_scaled: np.ndarray
    test_Y_true: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionBundle:
    """Standardized prediction output across methods."""

    mean: np.ndarray
    std: np.ndarray | None = None
    cov: np.ndarray | None = None


@dataclass(frozen=True)
class FittedPredictor:
    """Callable wrapper returned by each method adapter."""

    predict_fn: Callable[[np.ndarray], PredictionBundle]
    status: str = "ok"
    error: str = ""

    def predict(self, Xstar: np.ndarray) -> PredictionBundle:
        return self.predict_fn(Xstar)


def method_python_executable(method: str, config: ExperimentConfig) -> Path | None:
    """Return the dedicated Python executable for methods that need one."""

    if method in {"MOOGP", "MOGP", "LCGP"}:
        return config.moogp_python
    if method == "OILMM":
        return config.oilmm_python
    return None


def stable_seed(*parts: Any) -> int:
    """Create a deterministic 32-bit seed from arbitrary values."""

    key = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32 - 1)


def timestamp_utc() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def build_dataset_bundle(
    function: str,
    n: int,
    p: int,
    n_test: int,
    seed_data: int,
    noise_var_frac: float = 1e-2,
) -> DatasetBundle:
    """Build one deterministic train/test dataset for a benchmark cell."""

    if function == "borehole":
        return _build_borehole_bundle(n=n, p=p, n_test=n_test, seed_data=seed_data, noise_var_frac=noise_var_frac)
    if function == "forrester_mixed":
        return _build_forrester_bundle(n=n, p=p, n_test=n_test, seed_data=seed_data, noise_var_frac=noise_var_frac)
    raise ValueError(f"Unsupported function '{function}'. Choices: {SUPPORTED_FUNCTIONS}.")


def make_noise_variances(y_clean: np.ndarray, noise_var_frac: float) -> np.ndarray:
    """Create per-output Gaussian noise variances from clean signal variance."""

    y_clean = np.asarray(y_clean, dtype=float)
    base_var = np.var(y_clean, axis=0, ddof=1)
    return np.maximum(1e-12, noise_var_frac * np.maximum(base_var, 1e-12))


def add_output_noise(y_clean: np.ndarray, noise_var: np.ndarray, seed: int) -> np.ndarray:
    """Add independent Gaussian noise to each output dimension."""

    y_clean = np.asarray(y_clean, dtype=float)
    noise_var = np.asarray(noise_var, dtype=float)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, np.sqrt(noise_var), size=y_clean.shape)
    return y_clean + noise


def _build_borehole_bundle(
    n: int,
    p: int,
    n_test: int,
    seed_data: int,
    noise_var_frac: float,
) -> DatasetBundle:
    train = generate_borehole_data_nd(n=n, p=p, seed=seed_data)
    d = train["X_scaled"].shape[1]
    seed_test = stable_seed(seed_data, "borehole", "test")
    theta_test_unit = qmc.LatinHypercube(d=d, scramble=True, seed=seed_test).random(n_test)
    theta_test_scaled = 2.0 * (theta_test_unit - 0.5)
    theta_test_phys = tstd2theta(theta_test_unit)

    locations = np.asarray(train["locations_phys"], dtype=float)
    theta_stack = np.repeat(theta_test_phys, repeats=p, axis=0)
    x_stack = np.tile(locations, (n_test, 1))
    y_train_clean = np.asarray(train["Y"], dtype=float)
    y_test_clean = borehole_vec_physical(x_stack, theta_stack).reshape(n_test, p)
    noise_var = make_noise_variances(y_train_clean, noise_var_frac=noise_var_frac)
    y_train = add_output_noise(y_train_clean, noise_var, seed=stable_seed(seed_data, "borehole", "train_noise"))
    y_test = add_output_noise(y_test_clean, noise_var, seed=stable_seed(seed_data, "borehole", "test_noise"))

    train_data = {
        "X_scaled": np.asarray(train["X_scaled"], dtype=float),
        "Y": y_train,
        "y": y_train,
    }
    return DatasetBundle(
        function="borehole",
        n=n,
        p=p,
        seed_data=seed_data,
        train_data=train_data,
        test_X_scaled=np.asarray(theta_test_scaled, dtype=float),
        test_Y_true=np.asarray(y_test, dtype=float),
        extra={
            "locations_phys": locations,
            "Y_train_clean": y_train_clean,
            "Y_test_clean": y_test_clean,
            "noise_var": noise_var,
        },
    )


def _forrester_loadings(p: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, p, dtype=float)
    loadings = np.column_stack(
        [
            1.0 + 0.25 * np.sin(2.0 * np.pi * s),
            np.cos(np.pi * s),
            np.sin(2.0 * np.pi * s),
        ]
    )
    norms = np.maximum(np.linalg.norm(loadings, axis=1, keepdims=True), 1e-12)
    return loadings / norms


def _build_forrester_bundle(
    n: int,
    p: int,
    n_test: int,
    seed_data: int,
    noise_var_frac: float,
) -> DatasetBundle:
    loadings = _forrester_loadings(p)
    train = generate_forrester_data(n=n, seed=seed_data, with_error=False)
    seed_test = stable_seed(seed_data, "forrester_mixed", "test")
    test = generate_forrester_data(n=n_test, seed=seed_test, with_error=False)

    y_train_clean = np.asarray(train["f"], dtype=float) @ loadings.T
    y_test_clean = np.asarray(test["f"], dtype=float) @ loadings.T
    noise_var = make_noise_variances(y_train_clean, noise_var_frac=noise_var_frac)
    y_train = add_output_noise(y_train_clean, noise_var, seed=stable_seed(seed_data, "forrester_mixed", "train_noise"))
    y_test = add_output_noise(y_test_clean, noise_var, seed=stable_seed(seed_data, "forrester_mixed", "test_noise"))
    train_data = {
        "X_scaled": np.asarray(train["X_scaled"], dtype=float),
        "Y": y_train,
        "y": y_train,
    }
    return DatasetBundle(
        function="forrester_mixed",
        n=n,
        p=p,
        seed_data=seed_data,
        train_data=train_data,
        test_X_scaled=np.asarray(test["X_scaled"], dtype=float),
        test_Y_true=np.asarray(y_test, dtype=float),
        extra={
            "loadings": loadings,
            "Y_train_clean": y_train_clean,
            "Y_test_clean": y_test_clean,
            "noise_var": noise_var,
        },
    )


def make_latent_theta0_and_bounds(q: int, d: int, seed_model: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Build the fixed latent initialization used in ``fit_moogp_forrester``."""

    theta0: list[float] = []
    bounds: list[tuple[float, float]] = []

    for _ in range(q):
        theta0.append(float(np.log(1.0)))
        theta0.extend([float(np.log(0.5))] * d)
        bounds.append((float(np.log(1e-3)), float(np.log(1e3))))
        bounds.extend([(float(np.log(0.05)), float(np.log(5.0)))] * d)

    return np.asarray(theta0, dtype=float), bounds


def append_sigma_eps_theta0_and_bounds(
    theta0: np.ndarray,
    bounds: list[tuple[float, float]],
    y_train: np.ndarray,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Append the sigma-eps block exactly as in ``fit_moogp_forrester``."""

    y_var = np.maximum(1e-12, np.var(np.asarray(y_train, dtype=float), axis=0, ddof=1))
    sigma_eps2_init = np.log(1e-2 * y_var)
    theta0_out = np.concatenate([theta0, sigma_eps2_init])

    lb = np.maximum(1e-12, 1e-6 * y_var)
    ub = np.maximum(lb * 10.0, 0.5 * y_var)
    log_bounds = [(float(np.log(lbi)), float(np.log(ubi))) for lbi, ubi in zip(lb, ub)]
    return theta0_out, bounds + log_bounds


def fit_method_local(method: str, bundle: DatasetBundle, seed_model: int, config: ExperimentConfig) -> FittedPredictor:
    """Fit one benchmark method and return a standardized predictor."""

    if method == "MOOGP":
        return _fit_moogp_like(bundle=bundle, seed_model=seed_model, config=config, orthogonal=True)
    if method == "MOGP":
        return _fit_moogp_like(bundle=bundle, seed_model=seed_model, config=config, orthogonal=False)
    if method == "LCGP":
        return _fit_lcgp(bundle=bundle, seed_model=seed_model, config=config)
    if method == "OILMM":
        return _fit_oilmm(bundle=bundle, seed_model=seed_model, config=config)
    if method == "PUQ":
        return _fit_stub("PUQ")
    raise ValueError(f"Unsupported method '{method}'. Choices: {SUPPORTED_METHODS}.")


def _fit_moogp_like(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
    *,
    orthogonal: bool,
) -> FittedPredictor:
    from moogp.model import MOOGP

    y_train = np.asarray(bundle.train_data["Y"], dtype=float)
    n, d = bundle.train_data["X_scaled"].shape
    p = y_train.shape[1]
    q_eff = min(config.q, p, n)
    theta0, bounds = make_latent_theta0_and_bounds(q=q_eff, d=d, seed_model=seed_model)
    theta0, bounds = append_sigma_eps_theta0_and_bounds(theta0=theta0, bounds=bounds, y_train=y_train)

    model = MOOGP(
        terms=[None] + list(range(1, d + 1)),
        q=q_eff,
        Psi=None,
        orthogonal=orthogonal,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=config.jitter,
        one_based=True,
        normalize_cols=True,
        use_diagonalized_interaction=config.use_fast,
        use_slow_kyinv=False,
    )
    model.fit(
        data={"X_scaled": bundle.train_data["X_scaled"], "y": y_train},
        theta0=theta0,
        bounds=bounds,
        optimizer_opts={"maxiter": config.maxiter},
    )

    status = "ok" if bool(model.opt_result.success) else "opt_failed"
    error = "" if not model.opt_result.message else str(model.opt_result.message)

    def _predict(Xstar: np.ndarray) -> PredictionBundle:
        mean, std = model.predict(Xstar, return_std=True)
        return PredictionBundle(mean=np.asarray(mean, dtype=float), std=np.asarray(std, dtype=float))

    return FittedPredictor(predict_fn=_predict, status=status, error=error)


def _fit_lcgp(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
) -> FittedPredictor:
    """Fit LCGP using the package's native `(p, n)` output convention."""

    def _install_pkg_resources_shim() -> None:
        try:
            import pkg_resources  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        import types

        pkg_resources = types.ModuleType("pkg_resources")

        class DistributionNotFound(Exception):
            pass

        class _Distribution:
            def __init__(self, version: str):
                self.parsed_version = version

        def get_distribution(name: str) -> _Distribution:
            try:
                return _Distribution(metadata.version(name))
            except metadata.PackageNotFoundError as exc:
                raise DistributionNotFound(str(exc)) from exc

        pkg_resources.get_distribution = get_distribution
        pkg_resources.DistributionNotFound = DistributionNotFound
        sys.modules["pkg_resources"] = pkg_resources

    _install_pkg_resources_shim()

    try:
        import gpflow
        import tensorflow as tf
        from lcgp import LCGP
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "lcgp"
        raise ModuleNotFoundError(
            f"LCGP benchmark requires `{missing_name}` in {config.moogp_python}. "
            "Install the benchmark dependencies from requirements.txt."
        ) from exc

    np.random.seed(seed_model)
    tf.random.set_seed(seed_model)

    X_train = np.asarray(bundle.train_data["X_scaled"], dtype=float)
    Y_train = np.asarray(bundle.train_data["Y"], dtype=float)

    n, _ = X_train.shape
    p = Y_train.shape[1]
    q_eff = min(config.q, p, n)

    model = LCGP(
        y=Y_train.T,
        x=X_train,
        q=q_eff,
        verbose=False,
    )
    opt_result = gpflow.optimizers.Scipy().minimize(
        model.loss,
        model.trainable_variables,
        options={"maxiter": config.maxiter},
    )

    status = "ok" if bool(opt_result.success) else "opt_failed"
    error = "" if not opt_result.message else str(opt_result.message)

    def _predict(Xstar: np.ndarray) -> PredictionBundle:
        pred = np.asarray(model.predict(np.asarray(Xstar, dtype=float)))
        if pred.ndim != 3 or pred.shape[0] < 2:
            raise ValueError(f"Unexpected LCGP prediction shape {pred.shape}; expected (k, p, n).")

        mean = np.asarray(pred[0], dtype=float).T
        var = np.maximum(np.asarray(pred[1], dtype=float).T, 1e-12)
        return PredictionBundle(mean=mean, std=np.sqrt(var), cov=None)

    return FittedPredictor(
        predict_fn=_predict,
        status=status,
        error=error,
    )


def _fit_oilmm(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
) -> FittedPredictor:
    """
    Fit OILMM using the simple README-style API.

    Returns marginal predictive std via PredictionBundle.std.
    Full predictive covariance is not returned here.
    """

    def _install_plum_parametric_alias() -> None:
        try:
            import plum.parametric  # noqa: F401
        except ModuleNotFoundError as exc:
            if exc.name != "plum.parametric":
                raise
            import plum._parametric as plum_parametric

            sys.modules["plum.parametric"] = plum_parametric

    def _install_tensorflow_probability_stub() -> None:
        if "tensorflow_probability" in sys.modules:
            return

        import types
        import tensorflow as tf

        tfp = types.ModuleType("tensorflow_probability")
        tfp.stats = types.SimpleNamespace(
            percentile=lambda a, q, axis=None, interpolation="linear": tf.experimental.numpy.percentile(
                a,
                q,
                axis=axis,
                method=interpolation,
            )
        )
        sys.modules["tensorflow_probability"] = tfp

    def _coerce_oilmm_output_shape(arr: np.ndarray, n_rows: int, p: int) -> np.ndarray:
        """
        Normalize OILMM outputs to shape (n_rows, p).
        """
        arr = np.asarray(arr, dtype=float)

        if arr.shape == (n_rows, p):
            return arr
        if arr.shape == (p, n_rows):
            return arr.T
        if arr.ndim == 1 and p == 1 and arr.shape[0] == n_rows:
            return arr[:, None]

        raise ValueError(
            f"Unsupported OILMM output shape {arr.shape}; expected {(n_rows, p)} or {(p, n_rows)}."
        )
    
    def _oilmm_prepare_X(X: np.ndarray) -> np.ndarray:
        """
        OILMM examples use a vector for 1D inputs.
        Keep multi-input problems as an (n, d) matrix.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected X with shape (n, d), got {X.shape}.")
        if X.shape[1] == 1:
            return X[:, 0]
        return X

    _install_plum_parametric_alias()

    import tensorflow as tf

    _install_tensorflow_probability_stub()

    from oilmm.tensorflow import OILMM
    from stheno import EQ, GP


    backend_name = None
    to_numpy = None

    dtype = tf.float32
    backend_name = "tensorflow"
    to_numpy = lambda z: z.numpy() if hasattr(z, "numpy") else np.asarray(z, dtype=float)

    X_train = np.asarray(bundle.train_data["X_scaled"], dtype=np.float32)
    Y_train = np.asarray(bundle.train_data["Y"], dtype=np.float32)

    n, d = X_train.shape
    p = Y_train.shape[1]
    q_eff = min(config.q, p, n)

    X_fit = _oilmm_prepare_X(X_train)

    def build_latent_processes(ps):
        return [
            (
                latent_ps.variance.positive(1.0) * GP(EQ().stretch(latent_ps.length_scale.positive(1.0))),
                latent_ps.noise.positive(1e-2),
            )
            for latent_ps, _ in zip(ps, range(q_eff))
        ]

    prior = OILMM(
        dtype,
        build_latent_processes,
        num_outputs=p,
    )

    fit_kwargs = {"trace": False}
    if backend_name == "tensorflow":
        fit_kwargs["jit"] = False

    prior.fit(X_fit, Y_train, **fit_kwargs)
    posterior = prior.condition(X_fit, Y_train)

    def _predict(Xstar: np.ndarray) -> PredictionBundle:
        Xs = _oilmm_prepare_X(np.asarray(Xstar, dtype=np.float32))
        mean, var = posterior.predict(Xs)

        mean_np = _coerce_oilmm_output_shape(
            to_numpy(mean),
            n_rows=Xstar.shape[0],
            p=p,
        )
        var_np = _coerce_oilmm_output_shape(
            to_numpy(var),
            n_rows=Xstar.shape[0],
            p=p,
        )
        var_np = np.maximum(var_np, 1e-12)

        return PredictionBundle(
            mean=mean_np,
            std=np.sqrt(var_np),
            cov=None,
        )

    return FittedPredictor(
        predict_fn=_predict,
        status="ok",
        error="",
    )


def _fit_stub(name: str) -> FittedPredictor:
    raise NotImplementedError(f"{name} wrapper is intentionally left empty in experiments/benchmark_lib.py.")


def compute_metrics(y_true: np.ndarray, prediction: PredictionBundle) -> dict[str, float | None]:
    """Compute benchmark metrics with shape adapters for moogp.evaluation."""

    y_true = np.asarray(y_true, dtype=float)
    y_mean = np.asarray(prediction.mean, dtype=float)
    if y_true.shape != y_mean.shape:
        raise ValueError(f"Prediction mean shape {y_mean.shape} does not match truth shape {y_true.shape}.")

    metrics: dict[str, float | None] = {
        "rmse": float(rmse(y_true, y_mean)),
        "coverage_95": None,
        "interval_len_95": None,
        "dss_diag": None,
        "dss_full": None,
    }

    if prediction.std is not None:
        y_var = np.maximum(np.asarray(prediction.std, dtype=float) ** 2, 1e-12)
        coverage_95, interval_len_95 = intervalstats(y_true, y_mean, y_var)
        metrics["coverage_95"] = float(coverage_95)
        metrics["interval_len_95"] = float(interval_len_95)
        metrics["dss_diag"] = float(dss(y_true.T, y_mean.T, y_var.T, use_diag=True))

    if prediction.cov is not None:
        cov_npp = normalize_covariance(prediction.cov, n=y_true.shape[0], p=y_true.shape[1])
        metrics["dss_full"] = float(dss(y_true.T, y_mean.T, np.moveaxis(cov_npp, 0, -1), use_diag=False))

    return metrics


def normalize_covariance(cov: np.ndarray, n: int, p: int) -> np.ndarray:
    """Normalize full predictive covariance to shape ``(n, p, p)``."""

    arr = np.asarray(cov, dtype=float)
    if arr.shape == (n, p, p):
        out = arr.copy()
    elif arr.shape == (p, p, n):
        out = np.moveaxis(arr, -1, 0).copy()
    else:
        raise ValueError(f"Unsupported covariance shape {arr.shape}; expected {(n, p, p)} or {(p, p, n)}.")

    eye = np.eye(p, dtype=float)
    for idx in range(n):
        out[idx] = 0.5 * (out[idx] + out[idx].T) + 1e-12 * eye
    return out


def make_base_row(
    run_id: str,
    function: str,
    method: str,
    n: int,
    p: int,
    rep: int,
    seed_data: int,
    seed_model: int,
) -> dict[str, Any]:
    """Create a CSV row with the required schema."""

    return {
        "run_id": run_id,
        "function": function,
        "method": method,
        "n": n,
        "p": p,
        "rep": rep,
        "seed_data": seed_data,
        "seed_model": seed_model,
        "status": "",
        "error": "",
        "train_time_sec": None,
        "pred_time_sec": None,
        "rmse": None,
        "coverage_95": None,
        "interval_len_95": None,
        "dss_diag": None,
        "dss_full": None,
    }


def run_single_method_local(
    run_id: str,
    function: str,
    method: str,
    n: int,
    p: int,
    rep: int,
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run one method on one dataset bundle and return one results row."""

    seed_data = bundle.seed_data
    seed_model = stable_seed(config.base_seed, function, method, n, p, rep, "model")
    row = make_base_row(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        rep=rep,
        seed_data=seed_data,
        seed_model=seed_model,
    )

    try:
        t0 = time.perf_counter()
        predictor = fit_method_local(method=method, bundle=bundle, seed_model=seed_model, config=config)
        row["train_time_sec"] = time.perf_counter() - t0
        row["status"] = predictor.status
        row["error"] = predictor.error

        t1 = time.perf_counter()
        prediction = predictor.predict(bundle.test_X_scaled)
        row["pred_time_sec"] = time.perf_counter() - t1
        row.update(compute_metrics(bundle.test_Y_true, prediction))
    except NotImplementedError as exc:
        row["status"] = "not_implemented"
        row["error"] = str(exc)
    except Exception as exc:  # pragma: no cover - exercised in smoke runs instead
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def _parse_subprocess_row(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Child process produced no JSON output.")
    return json.loads(lines[-1])


def run_single_method_subprocess(
    run_id: str,
    function: str,
    method: str,
    n: int,
    p: int,
    rep: int,
    seed_data: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run one method in its dedicated virtualenv and return the result row."""

    seed_model = stable_seed(config.base_seed, function, method, n, p, rep, "model")
    row = make_base_row(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        rep=rep,
        seed_data=seed_data,
        seed_model=seed_model,
    )

    python_executable = method_python_executable(method, config)
    if python_executable is None:
        raise ValueError(f"Method '{method}' does not use an external Python executable.")
    if not python_executable.exists():
        row["status"] = "error"
        row["error"] = f"Missing Python executable for {method}: {python_executable}"
        return row

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(config.results_dir / ".mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(config.results_dir / ".cache"))

    payload = {
        "run_id": run_id,
        "function": function,
        "method": method,
        "n": n,
        "p": p,
        "rep": rep,
        "seed_data": seed_data,
        "config": config.to_metadata(),
    }

    completed = subprocess.run(
        [str(python_executable), "-m", "experiments.method_runner"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        row["status"] = "error"
        row["error"] = stderr or f"Child process exited with code {completed.returncode}."
        return row

    try:
        return _parse_subprocess_row(completed.stdout)
    except Exception as exc:
        stderr = completed.stderr.strip()
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        if stderr:
            row["error"] = f"{row['error']} | stderr: {stderr}"
        return row


def run_single_method_job_local(
    run_id: str,
    function: str,
    method: str,
    n: int,
    p: int,
    rep: int,
    seed_data: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Worker-friendly wrapper that rebuilds the deterministic dataset bundle."""

    bundle = build_dataset_bundle(
        function=function,
        n=n,
        p=p,
        n_test=config.n_test,
        seed_data=seed_data,
        noise_var_frac=config.noise_var_frac,
    )
    return run_single_method_local(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        rep=rep,
        bundle=bundle,
        config=config,
    )


def run_single_method_job(
    run_id: str,
    function: str,
    method: str,
    n: int,
    p: int,
    rep: int,
    seed_data: int,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Run one benchmark row, delegating to a dedicated interpreter when needed."""

    if method_python_executable(method, config) is not None:
        return run_single_method_subprocess(
            run_id=run_id,
            function=function,
            method=method,
            n=n,
            p=p,
            rep=rep,
            seed_data=seed_data,
            config=config,
        )

    return run_single_method_job_local(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        rep=rep,
        seed_data=seed_data,
        config=config,
    )


def append_results_rows(results_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append result rows without clobbering prior runs."""

    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in RESULT_COLUMNS})


def append_run_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    """Append one run metadata record to ``run_metadata.json``."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = {"runs": []}
    data.setdefault("runs", []).append(payload)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def make_run_metadata(run_id: str, config: ExperimentConfig) -> dict[str, Any]:
    """Collect run metadata for reproducibility."""

    return {
        "run_id": run_id,
        "created_at_utc": timestamp_utc(),
        "config": config.to_metadata(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": platform.uname()._asdict(),
    }


def run_benchmarks(config: ExperimentConfig) -> dict[str, Any]:
    """Run the full benchmark grid and persist results incrementally."""

    run_id = f"run_{stable_seed(config.base_seed, timestamp_utc()):010d}"
    results_path = config.results_dir / "results.csv"
    metadata_path = config.results_dir / "run_metadata.json"

    append_run_metadata(metadata_path, make_run_metadata(run_id=run_id, config=config))

    rows: list[dict[str, Any]] = []
    jobs = max(1, int(config.jobs))
    job_specs = []
    for function in config.functions:
        for p in config.output_dims:
            for n in config.sample_sizes:
                for rep in range(1, config.reps + 1):
                    seed_data = stable_seed(config.base_seed, function, n, p, rep, "data")
                    for method in config.methods:
                        job_specs.append((run_id, function, method, n, p, rep, seed_data, config))

    if jobs == 1:
        for spec in job_specs:
            row = run_single_method_job(*spec)
            append_results_rows(results_path, [row])
            rows.append(row)
    else:
        max_workers = min(jobs, len(job_specs), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_single_method_job, *spec) for spec in job_specs]
            for future in as_completed(futures):
                row = future.result()
                append_results_rows(results_path, [row])
                rows.append(row)

    return {
        "run_id": run_id,
        "results_path": results_path,
        "metadata_path": metadata_path,
        "rows": rows,
    }
