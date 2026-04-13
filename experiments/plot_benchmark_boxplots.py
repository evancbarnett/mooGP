from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch


DEFAULT_METRICS = ("train_time_sec", "rmse", "coverage_95")

METRIC_SPECS = {
    "train_time_sec": {"label": "Train Time (sec)", "slug": "train_time"},
    "rmse": {"label": "RMSE", "slug": "rmse"},
    "coverage_95": {"label": "95% Coverage", "slug": "coverage"},
}

METRIC_ALIASES = {
    "train_time": "train_time_sec",
    "train_time_sec": "train_time_sec",
    "rmse": "rmse",
    "coverage": "coverage_95",
    "coverage_95": "coverage_95",
}


@dataclass(frozen=True)
class GroupedMetricData:
    """Grouped benchmark values for one metric."""

    metric: str
    metric_label: str
    sample_sizes: tuple[int, ...]
    models: tuple[str, ...]
    values: dict[str, dict[int, list[float]]]


def canonical_metric_name(metric: str) -> str:
    """Normalize a user-facing metric alias to a results.csv column name."""

    canonical = METRIC_ALIASES.get(metric, metric)
    if canonical not in METRIC_SPECS:
        choices = ", ".join(sorted(METRIC_SPECS))
        raise ValueError(f"Unsupported metric '{metric}'. Choices: {choices}.")
    return canonical


def read_results_rows(results_csv: str | Path) -> list[dict[str, str]]:
    """Read one benchmark results CSV into a list of dict rows."""

    results_path = Path(results_csv)
    with results_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Results CSV '{results_path}' is missing a header row.")
        return [dict(row) for row in reader]


def prepare_grouped_metric_data(
    results_csv: str | Path,
    metric: str,
    models: Sequence[str] | None = None,
    include_statuses: Iterable[str] = ("ok", "opt_failed"),
) -> GroupedMetricData:
    """Collect per-model box-plot values grouped by sample size."""

    metric_name = canonical_metric_name(metric)
    requested_models = tuple(models) if models is not None else None
    allowed_models = set(requested_models) if requested_models is not None else None
    allowed_statuses = set(include_statuses)

    values: dict[str, dict[int, list[float]]] = {}
    discovered_models: list[str] = []
    sample_sizes: set[int] = set()

    for row in read_results_rows(results_csv):
        model = row.get("method", "")
        if allowed_models is not None and model not in allowed_models:
            continue
        if row.get("status", "") not in allowed_statuses:
            continue

        metric_value = _parse_float(row.get(metric_name))
        if metric_value is None:
            continue

        n_value = row.get("n", "")
        if not n_value:
            continue

        n = int(n_value)
        values.setdefault(model, {}).setdefault(n, []).append(metric_value)
        sample_sizes.add(n)
        if model not in discovered_models:
            discovered_models.append(model)

    if not values:
        raise ValueError(
            f"No usable '{metric_name}' rows found in '{results_csv}' for statuses {sorted(allowed_statuses)}."
        )

    ordered_models = tuple(model for model in (requested_models or tuple(discovered_models)) if model in values)
    if not ordered_models:
        requested = ", ".join(requested_models or ())
        raise ValueError(f"No rows matched the requested models: {requested}.")

    return GroupedMetricData(
        metric=metric_name,
        metric_label=METRIC_SPECS[metric_name]["label"],
        sample_sizes=tuple(sorted(sample_sizes)),
        models=ordered_models,
        values=values,
    )


def plot_grouped_metric_boxplot(
    plot_data: GroupedMetricData,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Create a grouped box plot for one benchmark metric."""

    if ax is None:
        width = max(6.0, 1.6 * len(plot_data.sample_sizes))
        fig, ax = plt.subplots(figsize=(width, 4.5))
    else:
        fig = ax.figure

    centers = np.arange(len(plot_data.sample_sizes), dtype=float)
    offsets, box_width = _boxplot_positions(num_models=len(plot_data.models))
    cmap = plt.get_cmap("tab10")
    legend_handles: list[Patch] = []

    for model_idx, model in enumerate(plot_data.models):
        distributions: list[list[float]] = []
        positions: list[float] = []

        for center, n in zip(centers, plot_data.sample_sizes):
            model_values = plot_data.values.get(model, {}).get(n, [])
            if not model_values:
                continue
            positions.append(float(center + offsets[model_idx]))
            distributions.append(model_values)

        if not distributions:
            continue

        color = cmap(model_idx % cmap.N)
        boxplot = ax.boxplot(
            distributions,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=0.6, linewidth=1.1)
        for whisker in boxplot["whiskers"]:
            whisker.set(color=color, linewidth=1.1)
        for cap in boxplot["caps"]:
            cap.set(color=color, linewidth=1.1)
        for median in boxplot["medians"]:
            median.set(color="black", linewidth=1.4)
        for flier in boxplot["fliers"]:
            flier.set(markerfacecolor=color, markeredgecolor=color, alpha=0.5, markersize=4.0)

        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.6, label=model))

    ax.set_xticks(centers)
    ax.set_xticklabels([str(n) for n in plot_data.sample_sizes])
    ax.set_xlabel("n")
    ax.set_ylabel(plot_data.metric_label)
    ax.set_title(f"{plot_data.metric_label} by n")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Model", frameon=False)
    fig.tight_layout()
    return fig, ax


def create_metric_figures(
    results_csv: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
    models: Sequence[str] | None = None,
    include_statuses: Iterable[str] = ("ok", "opt_failed"),
    output_dir: str | Path | None = None,
    dpi: int = 150,
    show: bool = True,
) -> dict[str, Figure]:
    """Create one grouped box-plot figure per requested metric."""

    results_path = Path(results_csv)
    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Figure] = {}
    for metric in metrics:
        plot_data = prepare_grouped_metric_data(
            results_path,
            metric=metric,
            models=models,
            include_statuses=include_statuses,
        )
        fig, _ = plot_grouped_metric_boxplot(plot_data)
        figures[plot_data.metric] = fig

        if output_path is not None:
            filename = f"{results_path.stem}_{METRIC_SPECS[plot_data.metric]['slug']}_boxplot.png"
            fig.savefig(output_path / filename, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    return figures


def _parse_float(value: str | None) -> float | None:
    """Return a finite float value when possible."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _boxplot_positions(num_models: int) -> tuple[np.ndarray, float]:
    """Compute horizontal offsets and widths for grouped box plots."""

    if num_models <= 1:
        return np.array([0.0]), 0.45

    group_width = min(0.8, 0.22 * num_models)
    box_width = min(0.18, 0.9 * group_width / num_models)
    offsets = np.linspace(
        -group_width / 2.0 + box_width / 2.0,
        group_width / 2.0 - box_width / 2.0,
        num_models,
    )
    return offsets, box_width


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark box plots."""

    parser = argparse.ArgumentParser(description="Create grouped benchmark box plots from a results CSV.")
    parser.add_argument("--results-csv", required=True, help="Path to the benchmark results CSV.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to include, in display order.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=list(DEFAULT_METRICS),
        help="Metrics to plot. Supports aliases: train_time, rmse, coverage.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for saving PNGs. If omitted, figures are only displayed.",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI for saved PNG files.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Create figures without calling matplotlib.show().",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    create_metric_figures(
        results_csv=args.results_csv,
        metrics=args.metrics,
        models=args.models,
        output_dir=args.output_dir,
        dpi=args.dpi,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
