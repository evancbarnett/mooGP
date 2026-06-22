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
from moogp.evaluation import dss, intervalstats, normalized_rmse, rmse


OPTIMIZER_DIAGNOSTIC_COLUMNS = ("nit", "njev", "nfev")

RESULT_COLUMNS = [
    "run_id",
    "function",
    "method",
    "n",
    "n_train",
    "p",
    "q",
    "rep",
    "seed_data",
    "seed_model",
    "status",
    "error",
    "train_time_sec",
    "pred_time_sec",
    *OPTIMIZER_DIAGNOSTIC_COLUMNS,
    "rmse",
    "normalized_rmse",
    "train_rmse",
    "train_normalized_rmse",
    "coverage_95",
    "interval_len_95",
    "dss_diag",
    "dss_full",
    "n_folds",
]

SUPPORTED_FUNCTIONS = ("borehole", "forrester_mixed", "vah_nuclear")
SUPPORTED_METHODS = ("MOOGP", "MOGP", "LCGP", "OILMM", "PUQ")
VAH_GROUPING_CHOICES = ("index", "none")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOOGP_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_OILMM_PYTHON = REPO_ROOT / ".venv-oilmm" / "bin" / "python"
DEFAULT_PUQ_PYTHON = REPO_ROOT / ".venv-puq" / "bin" / "python"

# The VAH dataset ships at a fixed shape; the values here must match
# moogp/nuclear_data/all_theta.csv and all_f.csv. They are validated against the
# files at load time in `_load_vah_arrays`.
VAH_DATA_DIR = REPO_ROOT / "moogp" / "nuclear_data"
VAH_THETA_PATH = VAH_DATA_DIR / "all_theta.csv"
VAH_F_PATH = VAH_DATA_DIR / "all_f.csv"
VAH_INDEX_PATH = VAH_DATA_DIR / "all_f_index.csv"
VAH_TOTAL_N = 541
VAH_D = 15
VAH_P = 98

PUQ_INSTALL_HINT = (
    "PUQ is not installed. Install it (with its hetGPy dependency) into the "
    "PUQ benchmark environment, e.g.:\n"
    "    python -m venv .venv-puq && source .venv-puq/bin/activate\n"
    "    pip install git+https://github.com/davidogara/hetGPy.git\n"
    "    pip install git+https://github.com/parallelUQ/PUQ.git\n"
    "See https://github.com/parallelUQ/PUQ for details."
)


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
    puq_python: Path = field(default=DEFAULT_PUQ_PYTHON)
    n_folds: int = 5
    vah_grouping: str = "index"

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results_dir"] = str(self.results_dir)
        payload["moogp_python"] = str(self.moogp_python)
        payload["oilmm_python"] = str(self.oilmm_python)
        payload["puq_python"] = str(self.puq_python)
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
            puq_python=Path(payload.get("puq_python", DEFAULT_PUQ_PYTHON)),
            n_folds=int(payload.get("n_folds", 5)),
            vah_grouping=str(payload.get("vah_grouping", "index")),
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
    train_time_sec: float | None = None
    nit: int | None = None
    njev: int | None = None
    nfev: int | None = None

    def predict(self, Xstar: np.ndarray) -> PredictionBundle:
        return self.predict_fn(Xstar)


def blank_optimizer_diagnostics() -> dict[str, int | None]:
    """Return empty optimiser counters for methods that do not expose them."""

    return {name: None for name in OPTIMIZER_DIAGNOSTIC_COLUMNS}


def _coerce_optional_int(value: Any) -> int | None:
    """Best-effort conversion of optimiser counters to plain ints."""

    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def extract_optimizer_diagnostics(opt_result: Any) -> dict[str, int | None]:
    """Extract standard optimiser counters when a backend exposes them."""

    if opt_result is None:
        return blank_optimizer_diagnostics()

    if isinstance(opt_result, dict):
        getter = opt_result.get
    else:
        def getter(name): return getattr(opt_result, name, None)

    return {
        name: _coerce_optional_int(getter(name))
        for name in OPTIMIZER_DIAGNOSTIC_COLUMNS
    }


def method_python_executable(method: str, config: ExperimentConfig) -> Path | None:
    """Return the dedicated Python executable for methods that need one."""

    if method in {"MOOGP", "MOGP", "LCGP"}:
        return config.moogp_python
    if method == "OILMM":
        return config.oilmm_python
    if method == "PUQ":
        return config.puq_python
    return None


def stable_seed(*parts: Any) -> int:
    """Create a deterministic 32-bit seed from arbitrary values."""

    key = "||".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32 - 1)


def effective_latent_rank(q_requested: int, n: int, p: int) -> int:
    """Return the latent rank actually used for one benchmark cell."""

    return int(min(q_requested, p, n))


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
    *,
    rep: int = 1,
    n_folds: int = 5,
    vah_grouping: str = "index",
    base_seed: int = 0,
) -> DatasetBundle:
    """Build one deterministic train/test dataset for a benchmark cell."""

    if function == "borehole":
        return _build_borehole_bundle(n=n, p=p, n_test=n_test, seed_data=seed_data, noise_var_frac=noise_var_frac)
    if function == "forrester_mixed":
        return _build_forrester_bundle(n=n, p=p, n_test=n_test, seed_data=seed_data, noise_var_frac=noise_var_frac)
    if function == "vah_nuclear":
        return _build_vah_bundle(
            rep=rep,
            n_folds=n_folds,
            vah_grouping=vah_grouping,
            base_seed=base_seed,
        )
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


def _load_vah_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Load the headerless VAH design and output matrices and shape-check them."""

    theta = np.loadtxt(VAH_THETA_PATH, delimiter=",", dtype=float)
    f = np.loadtxt(VAH_F_PATH, delimiter=",", dtype=float)
    if theta.ndim != 2 or theta.shape[1] != VAH_D:
        raise ValueError(
            f"all_theta.csv has shape {theta.shape}; expected (*, {VAH_D})."
        )
    if f.ndim != 2 or f.shape[1] != VAH_P:
        raise ValueError(
            f"all_f.csv has shape {f.shape}; expected (*, {VAH_P})."
        )
    if theta.shape[0] != f.shape[0]:
        raise ValueError(
            f"all_theta.csv ({theta.shape[0]} rows) and all_f.csv "
            f"({f.shape[0]} rows) have mismatched row counts."
        )
    return theta, f


def _load_vah_diag_error_structure(expected_p: int = VAH_P) -> list[int]:
    """Parse `all_f_index.csv` into the diagonal grouping for the kept outputs.

    Skips lines whose group name is empty or that contain `# omitted`, so the
    `pT_fluct` block (referenced in the index file but absent from `all_f.csv`)
    is dropped while every other group keeps its full width.
    """

    widths: list[int] = []
    with VAH_INDEX_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or "# omitted" in text.lower():
                continue
            parts = text.split(",")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            if not name:
                continue
            try:
                start = int(parts[1].strip())
                stop = int(parts[2].split("#")[0].strip())
            except ValueError:
                continue
            widths.append(stop - start)
    total = sum(widths)
    if total != expected_p:
        raise ValueError(
            f"VAH grouping widths sum to {total} but expected {expected_p}; "
            f"check {VAH_INDEX_PATH}."
        )
    return widths


def _kfold_indices(n_total: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return a deterministic shuffled k-fold partition of `range(n_total)`.

    The split depends only on (`n_total`, `n_folds`, `seed`), so every cell in a
    sweep sees the same fold assignment regardless of which rep it processes.
    """

    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if n_total < n_folds:
        raise ValueError(f"n_total={n_total} < n_folds={n_folds}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    return [np.sort(part) for part in np.array_split(perm, n_folds)]


def _build_vah_bundle(
    rep: int,
    n_folds: int,
    vah_grouping: str,
    base_seed: int,
) -> DatasetBundle:
    """K-fold CV bundle for the VAH dataset.

    The bundle ships **raw** simulator inputs and outputs — each method is
    responsible for its own standardization. MOOGP/MOGP read
    ``extra["standardize_x"]`` and ``extra["standardize_y"]`` and apply those
    transforms inside the model. LCGP and PUQ have their own internal
    standardization; OILMM sees raw data and absorbs the scale through its
    learnable kernel parameters.
    """

    if vah_grouping not in VAH_GROUPING_CHOICES:
        raise ValueError(
            f"vah_grouping must be one of {VAH_GROUPING_CHOICES}, got {vah_grouping!r}."
        )

    theta, f = _load_vah_arrays()
    n_total = theta.shape[0]
    fold_seed = stable_seed(base_seed, "vah_nuclear", "kfold", n_folds)
    folds = _kfold_indices(n_total, n_folds, seed=fold_seed)

    fold_index = (int(rep) - 1) % n_folds
    test_idx = folds[fold_index]
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)

    X_train = theta[train_idx]
    X_test = theta[test_idx]
    Y_train = f[train_idx]
    Y_test = f[test_idx]

    diag_error_structure: list[int] | None = None
    if vah_grouping == "index":
        diag_error_structure = _load_vah_diag_error_structure(expected_p=Y_train.shape[1])

    train_data = {
        "X_scaled": np.asarray(X_train, dtype=float),
        "Y": np.asarray(Y_train, dtype=float),
        "y": np.asarray(Y_train, dtype=float),
    }
    return DatasetBundle(
        function="vah_nuclear",
        n=int(X_train.shape[0]),
        p=int(Y_train.shape[1]),
        seed_data=int(fold_seed),
        train_data=train_data,
        test_X_scaled=np.asarray(X_test, dtype=float),
        test_Y_true=np.asarray(Y_test, dtype=float),
        extra={
            "fold_index": fold_index,
            "n_folds": n_folds,
            "diag_error_structure": diag_error_structure,
            "standardize_x": "unitcube",
            "standardize_y": "zscore",
        },
    )


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
        return _fit_puq(bundle=bundle, seed_model=seed_model, config=config)
    raise ValueError(f"Unsupported method '{method}'. Choices: {SUPPORTED_METHODS}.")


def _install_pkg_resources_shim() -> None:
    """Provide the minimal pkg_resources API that LCGP still imports."""

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


def _install_plum_parametric_alias() -> None:
    """Bridge the legacy OILMM import path expected by probmods."""

    try:
        import plum.parametric  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "plum.parametric":
            raise
        import plum._parametric as plum_parametric

        sys.modules["plum.parametric"] = plum_parametric


def _install_tensorflow_probability_stub(tf: Any) -> None:
    """Install the tiny tensorflow_probability surface OILMM needs."""

    if "tensorflow_probability" in sys.modules:
        return

    import types

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


def _get_oilmm_runtime() -> dict[str, Any]:
    """Load OILMM and its compatibility shims once per interpreter."""

    _install_plum_parametric_alias()

    import tensorflow as tf

    _install_tensorflow_probability_stub(tf)

    from oilmm.tensorflow import OILMM
    from stheno import EQ, GP

    return {
        "tf": tf,
        "dtype": tf.float32,
        "OILMM": OILMM,
        "EQ": EQ,
        "GP": GP,
        "to_numpy": lambda z: z.numpy() if hasattr(z, "numpy") else np.asarray(z, dtype=float),
    }


def _fit_moogp_like(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
    *,
    orthogonal: bool,
) -> FittedPredictor:
    from moogp.model import MOOGP

    # Exclude import/setup overhead, but include all model work required before predict.
    train_t0 = time.perf_counter()
    y_train = np.asarray(bundle.train_data["Y"], dtype=float)
    n, d = bundle.train_data["X_scaled"].shape
    p = y_train.shape[1]
    q_eff = effective_latent_rank(config.q, n=n, p=p)
    # MOOGP defaults to standardize_x="unitcube" and standardize_y="zscore",
    # which is what every bundle in this harness wants. A bundle can override
    # via ``extra["standardize_x"]`` / ``extra["standardize_y"]``; passing the
    # defaults through explicitly keeps the adapter behavior auditable.
    standardize_y = bundle.extra.get("standardize_y", "zscore")
    standardize_x = bundle.extra.get("standardize_x", "unitcube")
    x_margin = float(bundle.extra.get("x_margin", 0.1))
    diag_error_structure = bundle.extra.get("diag_error_structure")

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
        standardize_y=standardize_y,
        standardize_x=standardize_x,
        x_margin=x_margin,
        diag_error_structure=diag_error_structure,
    )
    # theta0 / bounds are omitted: MOOGP.fit builds a data-aware initialization
    # internally from the standardized working-scale data (the SVD/input-spread
    # seeds and box bounds the external make_data_aware_theta0_and_bounds helper
    # used to supply). This reaches the same NLL/RMSE/coverage in ~32% fewer
    # L-BFGS-B iterations on VAH (see notebooks/vah_moogp_diagnostic.ipynb §15).
    model.fit(
        data={"X_scaled": bundle.train_data["X_scaled"], "y": y_train},
        optimizer_opts={"maxiter": config.maxiter},
    )
    train_time_sec = time.perf_counter() - train_t0

    status = "ok" if bool(model.opt_result.success) else "opt_failed"
    error = "" if not model.opt_result.message else str(model.opt_result.message)
    diagnostics = extract_optimizer_diagnostics(model.opt_result)

    def _predict(Xstar: np.ndarray) -> PredictionBundle:
        mean, std = model.predict(Xstar, return_std=True)
        return PredictionBundle(mean=np.asarray(mean, dtype=float), std=np.asarray(std, dtype=float))

    return FittedPredictor(
        predict_fn=_predict,
        status=status,
        error=error,
        train_time_sec=train_time_sec,
        **diagnostics,
    )


def _fit_lcgp(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
) -> FittedPredictor:
    """Fit LCGP using the package's native `(p, n)` output convention."""

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

    # Keep imports and env shims out of the benchmark timer.
    train_t0 = time.perf_counter()
    X_train = np.asarray(bundle.train_data["X_scaled"], dtype=float)
    Y_train = np.asarray(bundle.train_data["Y"], dtype=float)

    n, _ = X_train.shape
    p = Y_train.shape[1]
    q_eff = effective_latent_rank(config.q, n=n, p=p)
    diag_error_structure = bundle.extra.get("diag_error_structure")

    lcgp_kwargs: dict[str, Any] = {
        "y": Y_train.T,
        "x": X_train,
        "q": q_eff,
        "verbose": False,
    }
    if diag_error_structure is not None:
        lcgp_kwargs["diag_error_structure"] = list(diag_error_structure)
    model = LCGP(**lcgp_kwargs)
    opt_result = gpflow.optimizers.Scipy().minimize(
        model.loss,
        model.trainable_variables,
        options={"maxiter": config.maxiter},
    )
    train_time_sec = time.perf_counter() - train_t0

    status = "ok" if bool(opt_result.success) else "opt_failed"
    error = "" if not opt_result.message else str(opt_result.message)
    diagnostics = extract_optimizer_diagnostics(opt_result)

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
        train_time_sec=train_time_sec,
        **diagnostics,
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

    try:
        runtime = _get_oilmm_runtime()
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "oilmm"
        raise ModuleNotFoundError(
            f"OILMM benchmark requires `{missing_name}` in {config.oilmm_python}. "
            "Install the OILMM benchmark dependencies in the dedicated environment."
        ) from exc

    OILMM = runtime["OILMM"]
    EQ = runtime["EQ"]
    GP = runtime["GP"]
    dtype = runtime["dtype"]
    to_numpy = runtime["to_numpy"]
    tf = runtime["tf"]

    # Match the other TensorFlow-based adapters: seed model initialisation outside timing.
    np.random.seed(seed_model)
    tf.random.set_seed(seed_model)

    # The dedicated interpreter absorbs import/shim cost. Time only model work.
    train_t0 = time.perf_counter()
    X_train = np.asarray(bundle.train_data["X_scaled"], dtype=np.float32)
    Y_train = np.asarray(bundle.train_data["Y"], dtype=np.float32)

    n, d = X_train.shape
    p = Y_train.shape[1]
    q_eff = effective_latent_rank(config.q, n=n, p=p)

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
    fit_kwargs["jit"] = False
    fit_kwargs["iters"] = config.maxiter

    # OILMM (via varz) ultimately calls ``scipy.optimize.fmin_l_bfgs_b`` and
    # discards the returned ``info`` dict, which is the only place
    # ``nit`` / ``funcalls`` / termination message live. ``varz/minimise.py``
    # does ``from scipy.optimize import fmin_l_bfgs_b`` at module load, so we
    # must patch the **module-level binding inside ``varz.minimise``** — patching
    # ``scipy.optimize.fmin_l_bfgs_b`` is a no-op once varz has already bound
    # the name. Restored in ``finally`` even if ``fit`` raises.
    #
    # Guard the import: the unit-test path mocks ``_get_oilmm_runtime`` and runs
    # in the main moogp venv where ``varz`` is not installed. In that case we
    # skip the patch (diagnostics stay blank) and let the mocked ``fit`` run.
    oilmm_opt_records: list[dict[str, Any]] = []
    try:
        import varz.minimise as _varz_min
    except ModuleNotFoundError:
        _varz_min = None

    if _varz_min is not None:
        _orig_fmin_l_bfgs_b = _varz_min.fmin_l_bfgs_b

        def _capturing_fmin_l_bfgs_b(*args, **kwargs):
            x_opt, val_opt, info = _orig_fmin_l_bfgs_b(*args, **kwargs)
            # ``info`` keys per scipy.optimize.fmin_l_bfgs_b docs:
            #   'nit'      -> number of L-BFGS-B iterations
            #   'funcalls' -> number of objective+gradient evaluations
            #   'task'     -> termination reason (bytes or str)
            #   'warnflag' -> 0=converged, 1=hit maxiter, 2=other failure
            oilmm_opt_records.append({
                "nit": int(info.get("nit", 0)),
                "funcalls": int(info.get("funcalls", 0)),
                "task": info.get("task", b""),
                "warnflag": int(info.get("warnflag", 0)),
            })
            return x_opt, val_opt, info

        _varz_min.fmin_l_bfgs_b = _capturing_fmin_l_bfgs_b
        try:
            prior.fit(X_fit, Y_train, **fit_kwargs)
        finally:
            _varz_min.fmin_l_bfgs_b = _orig_fmin_l_bfgs_b
    else:
        prior.fit(X_fit, Y_train, **fit_kwargs)
    posterior = prior.condition(X_fit, Y_train)
    train_time_sec = time.perf_counter() - train_t0

    # Aggregate per-output L-BFGS-B runs into the standard diagnostic columns.
    # OILMM's IMOGP-style fit calls the optimiser once per output (or once
    # total in the OILMM-proper dispatch); summing keeps the columns honest
    # regardless of which dispatch ran.
    oilmm_nit = sum(r["nit"] for r in oilmm_opt_records) or None
    oilmm_nfev = sum(r["funcalls"] for r in oilmm_opt_records) or None
    # Each L-BFGS-B step calls the joint objective+gradient closure once, so
    # njev equals funcalls under varz's wrapping. Report it explicitly so the
    # column is populated.
    oilmm_njev = oilmm_nfev

    def _decode_task(t: Any) -> str:
        if isinstance(t, (bytes, bytearray)):
            return t.decode("utf-8", errors="replace")
        return str(t)

    if oilmm_opt_records:
        # status: ok unless any sub-fit returned a non-zero warnflag.
        oilmm_status = "ok"
        oilmm_error = ""
        for r in oilmm_opt_records:
            if r["warnflag"] != 0:
                oilmm_status = "opt_failed"
                oilmm_error = _decode_task(r["task"])
                break
    else:
        # Defensive: if the patch never caught a call (e.g. OILMM switched
        # to a different minimiser), keep the legacy "ok / no diagnostics"
        # behaviour rather than fabricating values.
        oilmm_status = "ok"
        oilmm_error = ""

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
        status=oilmm_status,
        error=oilmm_error,
        train_time_sec=train_time_sec,
        nit=oilmm_nit,
        njev=oilmm_njev,
        nfev=oilmm_nfev,
    )


def _fit_puq(
    bundle: DatasetBundle,
    seed_model: int,
    config: ExperimentConfig,
) -> FittedPredictor:
    """
    Fit PUQ's multi-output heteroskedastic GP surrogate (multihetGP).

    PUQ wraps one hetGPy.hetGP per output column, so the predictive distribution
    is per-output: it provides marginal mean and variance at each test point but
    no cross-output covariance. We populate PredictionBundle.std and leave .cov
    as None so dss_full stays blank for this method.
    """

    try:
        from PUQ.surrogate import emulator as puq_emulator
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"PUQ benchmark requires `{exc.name or 'PUQ'}` in {config.puq_python}. "
            f"{PUQ_INSTALL_HINT}"
        ) from exc

    np.random.seed(seed_model)

    # Match the OILMM/LCGP adapters: keep import/shim cost out of the timer,
    # then time only the model work that includes fitting.
    train_t0 = time.perf_counter()
    X_train = np.asarray(bundle.train_data["X_scaled"], dtype=float)
    Y_train = np.asarray(bundle.train_data["Y"], dtype=float)
    p = Y_train.shape[1]

    # PUQ's emulator validates `f.shape[1] == theta.shape[0]`; multihetGP itself
    # never reads `theta`, so a (p, 1) placeholder satisfies the shape check.
    theta_dummy = np.arange(p, dtype=float).reshape(p, 1)

    emu = puq_emulator(
        x=X_train,
        theta=theta_dummy,
        f=Y_train,
        method="multihetGP",
        args={
            "covtype": "Gaussian",
            "maxit": int(config.maxiter),
        },
    )
    train_time_sec = time.perf_counter() - train_t0

    # PUQ's multihetGP wraps one ``hetgpy.hetGP`` per output column. Each
    # per-output emulator's ``_info`` dict carries the optimiser counters
    # populated by ``hetGP.mleHetGP`` (or, when the homoskedastic fallback
    # fires, by ``homGP.mleHomGP`` — see hetGP.py:1256-1264). The shape of
    # ``nit_opt`` differs across the two paths:
    #
    #   hetGP path  -> ``{"nfev": ..., "njev": ...}``  (hetGP.py:1191)
    #   homGP path  -> ``int`` (= nfev only; homGP discards njev)
    #
    # In both cases scipy's ``OptimizeResult.nit`` is dropped, so we report
    # ``njev`` as a proxy for ``nit``: with the analytical gradient L-BFGS-B
    # invokes the objective+gradient closure once per accepted iteration.
    # ``msg`` carries the scipy termination message used for the status.
    puq_nfev_total = 0
    puq_njev_total = 0
    puq_status = "ok"
    puq_error = ""
    sub_emulators = emu._info.get("emulist", []) if hasattr(emu, "_info") else []
    for sub in sub_emulators:
        sub_info = getattr(sub, "_info", {}) or {}
        nit_opt = sub_info.get("nit_opt")
        if isinstance(nit_opt, dict):
            puq_nfev_total += int(nit_opt.get("nfev", 0) or 0)
            puq_njev_total += int(nit_opt.get("njev", 0) or 0)
        elif isinstance(nit_opt, (int, np.integer)):
            # homGP fallback: only nfev is preserved. Use it for both
            # counters so the column is not silently dropped.
            puq_nfev_total += int(nit_opt)
            puq_njev_total += int(nit_opt)
        msg = sub_info.get("msg", "")
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode("utf-8", errors="replace")
        msg_lower = str(msg).lower()
        if msg and "converge" not in msg_lower:
            puq_status = "opt_failed"
            puq_error = puq_error or str(msg)
    # When mleHetGP captured nothing, leave the columns blank rather than
    # fabricate zeros.
    puq_nfev = puq_nfev_total or None
    puq_njev = puq_njev_total or None
    puq_nit = puq_njev  # one gradient eval per L-BFGS-B iteration

    def _predict(Xstar: np.ndarray) -> PredictionBundle:
        Xs = np.asarray(Xstar, dtype=float)
        # thetaprime=None keeps PUQ from building cross-location covariances we
        # don't use here.
        preds = emu.predict(x=Xs, thetaprime=None)
        mean_pn = np.asarray(preds._info["mean"], dtype=float)
        # `var` is the mean-process (epistemic) variance only; `nugs` is the
        # heteroskedastic observation-noise variance. The predictive variance of
        # a new observation is their sum. Using `var` alone yields a confidence
        # band rather than a prediction interval and badly under-covers, unlike
        # the other methods which all return observation-level predictive vars.
        var_pn = np.asarray(preds._info["var"], dtype=float)
        nugs_pn = np.asarray(preds._info["nugs"], dtype=float)
        if mean_pn.shape != (p, Xs.shape[0]):
            raise ValueError(
                f"Unexpected PUQ prediction shape {mean_pn.shape}; expected {(p, Xs.shape[0])}."
            )
        total_var_pn = np.maximum(var_pn + nugs_pn, 1e-12)
        return PredictionBundle(mean=mean_pn.T, std=np.sqrt(total_var_pn).T, cov=None)

    return FittedPredictor(
        predict_fn=_predict,
        status=puq_status,
        error=puq_error,
        train_time_sec=train_time_sec,
        nit=puq_nit,
        njev=puq_njev,
        nfev=puq_nfev,
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
        "normalized_rmse": float(normalized_rmse(y_true, y_mean)),
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
    q: int,
    rep: int,
    seed_data: int,
    seed_model: int,
    n_folds: int | None = None,
) -> dict[str, Any]:
    """Create a CSV row with the required schema."""

    return {
        "run_id": run_id,
        "function": function,
        "method": method,
        "n": n,
        "p": p,
        "q": q,
        "rep": rep,
        "seed_data": seed_data,
        "seed_model": seed_model,
        "n_train": None,
        "status": "",
        "error": "",
        "train_time_sec": None,
        "pred_time_sec": None,
        **blank_optimizer_diagnostics(),
        "rmse": None,
        "normalized_rmse": None,
        "train_rmse": None,
        "train_normalized_rmse": None,
        "coverage_95": None,
        "interval_len_95": None,
        "dss_diag": None,
        "dss_full": None,
        "n_folds": n_folds,
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
    q_eff = effective_latent_rank(config.q, n=n, p=p)
    n_folds_row = config.n_folds if function == "vah_nuclear" else None
    row = make_base_row(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        q=q_eff,
        rep=rep,
        seed_data=seed_data,
        seed_model=seed_model,
        n_folds=n_folds_row,
    )
    # `n` is the grid label (for vah it is the full dataset size); `n_train` is
    # the actual number of rows the model was fit on (post train/test split).
    row["n_train"] = int(bundle.n)

    try:
        predictor = fit_method_local(method=method, bundle=bundle, seed_model=seed_model, config=config)
        row["train_time_sec"] = predictor.train_time_sec
        row["status"] = predictor.status
        row["error"] = predictor.error
        for key in OPTIMIZER_DIAGNOSTIC_COLUMNS:
            row[key] = getattr(predictor, key)

        # Only the predict call belongs in pred_time_sec; scoring stays outside.
        t1 = time.perf_counter()
        prediction = predictor.predict(bundle.test_X_scaled)
        row["pred_time_sec"] = time.perf_counter() - t1
        row.update(compute_metrics(bundle.test_Y_true, prediction))

        train_prediction = predictor.predict(bundle.train_data["X_scaled"])
        y_train_true = np.asarray(bundle.train_data["Y"], dtype=float)
        y_train_mean = np.asarray(train_prediction.mean, dtype=float)
        row["train_rmse"] = float(rmse(y_train_true, y_train_mean))
        row["train_normalized_rmse"] = float(normalized_rmse(y_train_true, y_train_mean))
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
    q_eff = effective_latent_rank(config.q, n=n, p=p)
    n_folds_row = config.n_folds if function == "vah_nuclear" else None
    row = make_base_row(
        run_id=run_id,
        function=function,
        method=method,
        n=n,
        p=p,
        q=q_eff,
        rep=rep,
        seed_data=seed_data,
        seed_model=seed_model,
        n_folds=n_folds_row,
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
        rep=rep,
        n_folds=config.n_folds,
        vah_grouping=config.vah_grouping,
        base_seed=config.base_seed,
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
    if not write_header:
        with results_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_header = next(reader, [])
        if existing_header != RESULT_COLUMNS:
            raise ValueError(
                f"Existing results header in {results_path} does not match the current schema. "
                "Use a fresh results directory or remove the old results.csv before rerunning."
            )
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


def make_run_id(config: ExperimentConfig) -> str:
    """Generate a fresh run identifier from base_seed + current timestamp."""

    return f"run_{stable_seed(config.base_seed, timestamp_utc()):010d}"


def iter_job_cells(config: ExperimentConfig):
    """Yield ``(function, method, n, p, rep)`` tuples covering the full grid.

    Used by both the in-process runner and the per-job emission path so the two
    code paths can never disagree about which cells exist in a sweep.

    The ``vah_nuclear`` function ignores ``sample_sizes`` / ``output_dims``;
    each rep maps to one held-out CV fold, with ``n`` set to the fixed dataset
    row count and ``p`` to the fixed observable count. The number of folds run
    is ``min(config.reps, config.n_folds)`` — pass ``--reps n_folds`` for full
    CV, or a smaller ``--reps`` to run only the first few folds (useful for
    quick smokes).
    """

    for function in config.functions:
        if function == "vah_nuclear":
            n_vah_reps = min(int(config.reps), int(config.n_folds))
            for rep in range(1, n_vah_reps + 1):
                for method in config.methods:
                    yield function, method, VAH_TOTAL_N, VAH_P, rep
            continue
        for p in config.output_dims:
            for n in config.sample_sizes:
                for rep in range(1, config.reps + 1):
                    for method in config.methods:
                        yield function, method, n, p, rep


def per_job_csv_name(function: str, method: str, n: int, p: int, rep: int) -> str:
    """Stable per-job CSV filename for parallel sweeps."""

    return f"{function}__{method}__n{n}__p{p}__rep{rep}.csv"


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    """Write one benchmark row to a fresh CSV using the canonical schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in RESULT_COLUMNS})


def emit_job_list(
    config: ExperimentConfig,
    jobs_path: Path,
    output_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist config + per-cell job list for parallel execution.

    Writes ``<results_dir>/config.json``, appends ``<results_dir>/run_metadata.json``,
    and writes ``jobs_path`` with one whitespace-separated line per cell:
    ``<run_id> <function> <method> <n> <p> <rep>``.
    """

    if run_id is None:
        run_id = make_run_id(config)
    if output_dir is None:
        output_dir = config.results_dir / "jobs"

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.results_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_metadata(), handle, indent=2, sort_keys=True)

    metadata_path = config.results_dir / "run_metadata.json"
    metadata = make_run_metadata(run_id=run_id, config=config)
    metadata["job_list_path"] = str(jobs_path)
    metadata["job_output_dir"] = str(output_dir)
    append_run_metadata(metadata_path, metadata)

    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    n_jobs = 0
    with jobs_path.open("w", encoding="utf-8") as handle:
        for function, method, n, p, rep in iter_job_cells(config):
            handle.write(f"{run_id} {function} {method} {n} {p} {rep}\n")
            n_jobs += 1

    return {
        "run_id": run_id,
        "config_path": config_path,
        "metadata_path": metadata_path,
        "jobs_path": jobs_path,
        "output_dir": output_dir,
        "n_jobs": n_jobs,
    }


def run_benchmarks(config: ExperimentConfig) -> dict[str, Any]:
    """Run the full benchmark grid and persist results incrementally."""

    run_id = make_run_id(config)
    results_path = config.results_dir / "results.csv"
    metadata_path = config.results_dir / "run_metadata.json"

    append_run_metadata(metadata_path, make_run_metadata(run_id=run_id, config=config))

    rows: list[dict[str, Any]] = []
    jobs = max(1, int(config.jobs))
    job_specs = []
    for function, method, n, p, rep in iter_job_cells(config):
        seed_data = stable_seed(config.base_seed, function, n, p, rep, "data")
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
