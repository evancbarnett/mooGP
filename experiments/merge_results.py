"""
Concatenate per-job CSV outputs (from ``run_one.py``) into a single results.csv
matching the canonical schema produced by the in-process ``run_benchmarks``.

Usage
-----
    python -m experiments.merge_results \\
        --input-dir results/jobs \\
        --output results/results.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .benchmark_lib import RESULT_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the merge step."""

    parser = argparse.ArgumentParser(description="Merge per-job benchmark CSV files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing per-job CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for the merged results.csv.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for per-job files (default: *.csv).",
    )
    return parser


def merge_per_job_csvs(input_dir: Path, output: Path, pattern: str = "*.csv") -> dict[str, int]:
    """Concatenate CSVs matching ``pattern`` under ``input_dir`` into ``output``."""

    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} under {input_dir}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with output.open("w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for file_path in files:
            with file_path.open("r", newline="", encoding="utf-8") as in_handle:
                reader = csv.DictReader(in_handle)
                if reader.fieldnames != RESULT_COLUMNS:
                    raise ValueError(
                        f"File {file_path} has fieldnames {reader.fieldnames!r}; "
                        f"expected {RESULT_COLUMNS!r}."
                    )
                for row in reader:
                    writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
                    n_rows += 1

    return {"merged_files": len(files), "rows_written": n_rows}


def main() -> int:
    """Parse args, merge per-job CSVs, and print a brief summary."""

    args = build_parser().parse_args()
    summary = merge_per_job_csvs(args.input_dir, args.output, pattern=args.pattern)
    print(f"merged_files={summary['merged_files']}")
    print(f"rows_written={summary['rows_written']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
