"""
Run a single benchmark cell (function, method, n, p, rep) and write a per-job CSV.

Designed for use under GNU parallel, AWS Batch array jobs, or Slurm so that many
independent processes can run in parallel without contending on a single shared
results.csv.

Usage
-----
    python -m experiments.run_one \\
        --config results/config.json \\
        --run-id run_0123456789 \\
        --function borehole \\
        --method MOOGP \\
        --n 100 \\
        --p 10 \\
        --rep 1 \\
        --output-dir results/jobs

Writes ``<output-dir>/<function>__<method>__n<n>__p<p>__rep<r>.csv`` containing
the canonical CSV header and exactly one row. Each invocation is independent and
deterministic for a fixed config + cell coordinate; reruns overwrite the file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_lib import (
    ExperimentConfig,
    per_job_csv_name,
    run_single_method_job,
    stable_seed,
    write_single_row_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for one-cell execution."""

    parser = argparse.ArgumentParser(description="Run a single benchmark cell.")
    parser.add_argument("--config", type=Path, required=True, help="Path to ExperimentConfig JSON.")
    parser.add_argument("--run-id", type=str, required=True, help="Sweep-wide run identifier.")
    parser.add_argument("--function", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the per-job CSV is written.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the per-job CSV already exists, exit 0 without re-running the cell.",
    )
    return parser


def load_config(config_path: Path) -> ExperimentConfig:
    """Load an ExperimentConfig from the JSON file written by ``--emit-jobs``."""

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig.from_metadata(payload)


def main() -> int:
    """Parse args, run one cell, persist the row, and report status."""

    args = build_parser().parse_args()
    config = load_config(args.config)

    out_path = args.output_dir / per_job_csv_name(
        function=args.function,
        method=args.method,
        n=args.n,
        p=args.p,
        rep=args.rep,
    )
    if args.skip_existing and out_path.exists():
        print(f"skipped_existing={out_path}")
        return 0

    seed_data = stable_seed(config.base_seed, args.function, args.n, args.p, args.rep, "data")
    row = run_single_method_job(
        run_id=args.run_id,
        function=args.function,
        method=args.method,
        n=args.n,
        p=args.p,
        rep=args.rep,
        seed_data=seed_data,
        config=config,
    )

    write_single_row_csv(out_path, row)
    print(f"row_written={out_path}")
    print(f"status={row.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
