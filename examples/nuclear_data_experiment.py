from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from moogp.evaluation import dss, intervalstats, normalized_rmse, rmse

if TYPE_CHECKING:
    from moogp.model import MOOGP


DEFAULT_NUCLEAR_OUTPUT_BLOCKS: tuple[tuple[int, int | None], ...] = (
    (0, 30),
    (30, 54),
    (54, 78),
    (78, None),
)

_INT_PATTERN = re.compile(r"-?\d+")


@dataclass(frozen=True)
class OutputFamily:
    """Named output slice from ``all_f_index.csv``."""

    name: str
    start: int
    end: int


@dataclass(frozen=True)
class MinMaxScaler:
    """Column-wise min/max scaler to ``[-1, 1]``."""

    data_min: np.ndarray
    data_max: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        span = np.where(self.data_max > self.data_min, self.data_max - self.data_min, 1.0)
        return 2.0 * (X - self.data_min) / span - 1.0


def default_nuclear_data_dir() -> Path:
    """Return the repository-local nuclear-data directory."""

    return Path(__file__).resolve().parent / "nuclear_data"


def _parse_int_field(value: str) -> int | None:
    match = _INT_PATTERN.search(str(value))
    return None if match is None else int(match.group(0))


def load_output_index(path: str | Path, *, n_outputs: int | None = None) -> list[OutputFamily]:
    """Parse the output-family index file, tolerating comments and trailing rows."""

    families: list[OutputFamily] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue

            name = row[0].strip()
            if not name:
                continue

            start = _parse_int_field(row[1])
            end = _parse_int_field(row[2])
            if start is None or end is None:
                continue

            if n_outputs is not None:
                if start >= n_outputs:
                    continue
                end = min(end, n_outputs)

            if end <= start:
                continue

            families.append(OutputFamily(name=name, start=start, end=end))

    return families


def load_nuclear_dataset(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the nuclear-data inputs, outputs, and parsed output-family metadata."""

    base = Path(data_dir) if data_dir is not None else default_nuclear_data_dir()
    X = np.loadtxt(base / "all_theta.csv", delimiter=",", dtype=float)
    Y = np.loadtxt(base / "all_f.csv", delimiter=",", dtype=float)
    output_index = load_output_index(base / "all_f_index.csv", n_outputs=Y.shape[1])
    return {
        "data_dir": base,
        "X": X,
        "Y": Y,
        "output_index": output_index,
    }


def fit_input_minmax_scaler(X: np.ndarray) -> MinMaxScaler:
    """Fit a per-feature min/max scaler on the training inputs."""

    X = np.asarray(X, dtype=float)
    return MinMaxScaler(data_min=X.min(axis=0), data_max=X.max(axis=0))


def train_test_split_indices(
    n_samples: int,
    *,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible train/test row indices."""

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1); got {train_fraction}.")

    n_train = int(math.floor(train_fraction * n_samples))
    if n_train <= 0 or n_train >= n_samples:
        raise ValueError(
            f"train_fraction={train_fraction} yields n_train={n_train} for n_samples={n_samples}."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    train_idx = np.sort(perm[:n_train])
    test_idx = np.sort(perm[n_train:])
    return train_idx, test_idx


def overlapping_output_families(
    output_index: Sequence[OutputFamily],
    start: int,
    end: int,
) -> list[str]:
    """Return the output-family names that overlap a column block."""

    return [family.name for family in output_index if family.end > start and family.start < end]


def make_block_specs(
    n_outputs: int,
    *,
    blocks: Sequence[tuple[int, int | None]] = DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
    q_fraction: float | None = 0.25,
    fixed_q: int | None = None,
    output_index: Sequence[OutputFamily] = (),
) -> list[dict[str, Any]]:
    """Build the four requested output-block specifications."""

    if fixed_q is None:
        if q_fraction is None or q_fraction <= 0.0:
            raise ValueError(f"q_fraction must be positive when fixed_q is None; got {q_fraction}.")
    else:
        fixed_q = int(fixed_q)
        if fixed_q <= 0:
            raise ValueError(f"fixed_q must be positive; got {fixed_q}.")

    specs: list[dict[str, Any]] = []
    for idx, (start, end) in enumerate(blocks, start=1):
        stop = n_outputs if end is None else int(end)
        start = int(start)
        if not (0 <= start < stop <= n_outputs):
            raise ValueError(f"Invalid block ({start}, {end}) for n_outputs={n_outputs}.")

        p = stop - start
        if fixed_q is not None:
            if fixed_q > p:
                raise ValueError(f"fixed_q={fixed_q} exceeds block output count p={p} for block {idx}.")
            q = fixed_q
            q_rule = f"fixed_q={fixed_q}"
        else:
            q = max(1, int(math.ceil(q_fraction * p)))
            q_rule = f"ceil({q_fraction} * p_block)"
        output_groups = overlapping_output_families(output_index, start, stop)
        specs.append(
            {
                "model": f"Model {idx}",
                "start": start,
                "end": stop,
                "column_range": f"{start + 1}-{stop}",
                "python_slice": f"[{start}:{stop})",
                "n_outputs": p,
                "q": q,
                "q_rule": q_rule,
                "output_groups": output_groups,
                "output_groups_label": ", ".join(output_groups) if output_groups else "Unlabeled",
            }
        )

    return specs


def compute_predictive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_std: np.ndarray | None = None,
    f_true: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute the same metric set used by the repo's numerical-experiment code."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    recovery_target = y_true if f_true is None else np.asarray(f_true, dtype=float)

    metrics: dict[str, float | None] = {
        "predrmse": float(rmse(y_true, y_pred)),
        "recrmse": float(rmse(recovery_target, y_pred)),
        "nrmse": float(normalized_rmse(y_true, y_pred)),
        "coverage": None,
        "width": None,
        "dss": None,
    }

    if y_std is not None:
        yvar = np.maximum(np.asarray(y_std, dtype=float) ** 2, 1e-12)
        coverage, width = intervalstats(y_true, y_pred, yvar)
        metrics["coverage"] = float(coverage)
        metrics["width"] = float(width)
        metrics["dss"] = float(dss(y_true, y_pred, yvar, use_diag=True))

    return metrics


def fit_fast_nuclear_block(
    X_train_scaled: np.ndarray,
    Y_train: np.ndarray,
    *,
    q: int,
    maxiter: int = 10,
    jitter: float = 1e-6,
    standardize_y: bool | str = "zscore",
) -> "MOOGP":
    """Fit a fast-path MOOGP model for one output block."""

    from moogp.model import MOOGP

    X_train_scaled = np.asarray(X_train_scaled, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float)
    _, d = X_train_scaled.shape

    terms = [None] + list(range(1, d + 1))

    model = MOOGP(
        terms=terms,
        q=q,
        Psi=None,
        orthogonal=True,
        learn_Psi=False,
        learn_sigma_eps=True,
        jitter=jitter,
        normalize_cols=True,
        use_diagonalized_interaction=True,
        standardize_y=standardize_y,
    )
    # theta0 / bounds omitted: MOOGP.fit derives a data-aware initialization from
    # the standardized working-scale data. (The previous build_theta0_bounds
    # helper seeded sigma_eps from the *raw* Y variance, which disagreed with the
    # model's internal zscore standardization; the in-model init fixes that.)
    model.fit(
        data={"X_scaled": X_train_scaled, "Y": Y_train},
        optimizer_opts={"maxiter": maxiter},
    )
    return model


def run_nuclear_fast_experiment(
    *,
    data_dir: str | Path | None = None,
    blocks: Sequence[tuple[int, int | None]] = DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
    train_fraction: float = 0.8,
    seed: int = 0,
    q_fraction: float | None = 0.25,
    fixed_q: int | None = None,
    maxiter: int = 10,
    jitter: float = 1e-6,
    standardize_y: bool | str = "zscore",
) -> dict[str, Any]:
    """Run the four requested fast-mode MOOGP fits on the nuclear dataset."""

    dataset = load_nuclear_dataset(data_dir=data_dir)
    X_raw = dataset["X"]
    Y_full = dataset["Y"]
    output_index = dataset["output_index"]

    block_specs = make_block_specs(
        Y_full.shape[1],
        blocks=blocks,
        q_fraction=q_fraction,
        fixed_q=fixed_q,
        output_index=output_index,
    )
    train_idx, test_idx = train_test_split_indices(
        X_raw.shape[0],
        train_fraction=train_fraction,
        seed=seed,
    )

    scaler = fit_input_minmax_scaler(X_raw[train_idx])
    X_train_scaled = scaler.transform(X_raw[train_idx])
    X_test_scaled = scaler.transform(X_raw[test_idx])

    metrics_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    for spec in block_specs:
        block_slice = slice(spec["start"], spec["end"])
        Y_train = Y_full[train_idx, block_slice]
        Y_test = Y_full[test_idx, block_slice]

        model = fit_fast_nuclear_block(
            X_train_scaled,
            Y_train,
            q=spec["q"],
            maxiter=maxiter,
            jitter=jitter,
            standardize_y=standardize_y,
        )
        models[spec["model"]] = model

        model_rows.append(
            {
                **spec,
                "opt_success": bool(model.opt_result.success),
                "opt_message": str(model.opt_result.message),
                "nit": int(getattr(model.opt_result, "nit", -1)),
                "nfev": int(getattr(model.opt_result, "nfev", -1)),
                "used_fast": bool(model.cache["used_fast"]),
                "nll": float(model.nll_hat),
            }
        )

        for split, X_split_scaled, Y_split in (
            ("train", X_train_scaled, Y_train),
            ("test", X_test_scaled, Y_test),
        ):
            Y_pred, Y_std = model.predict(X_split_scaled, return_std=True)
            metrics_rows.append(
                {
                    **spec,
                    "split": split,
                    "train_size": int(train_idx.size),
                    "test_size": int(test_idx.size),
                    "opt_success": bool(model.opt_result.success),
                    "used_fast": bool(model.cache["used_fast"]),
                    **compute_predictive_metrics(
                        Y_split,
                        Y_pred,
                        y_std=Y_std,
                        f_true=Y_split,
                    ),
                }
            )

    return {
        "data_dir": dataset["data_dir"],
        "n_samples": int(X_raw.shape[0]),
        "n_inputs": int(X_raw.shape[1]),
        "n_outputs": int(Y_full.shape[1]),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "q_fraction": None if q_fraction is None else float(q_fraction),
        "fixed_q": None if fixed_q is None else int(fixed_q),
        "maxiter": int(maxiter),
        "standardize_y": standardize_y,
        "assumption_note": (
            "The nuclear dataset provides one observed output matrix only, so "
            "RRMSE is computed against all_f as a deterministic surrogate target."
        ),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "block_specs": block_specs,
        "output_index": output_index,
        "model_summary": model_rows,
        "metrics": metrics_rows,
        "models": models,
    }
