"""
Examples
--------
# Small smoke run
python -m experiments.run --functions borehole --methods MOOGP MOGP LCGP --ns 50 --ps 10 --reps 1 --n-test 40 --maxiter 5 --jobs 4

# Larger sweep
python -m experiments.run --functions borehole forrester_mixed --methods MOOGP MOGP LCGP OILMM PUQ --ns 50 100 250 1000 2500 --ps 10 20 50 --reps 5

# Emit a job list for parallel execution (no benchmark is run; see run_guide.md)
python -m experiments.run --emit-jobs results/jobs.txt --results-dir results

# Expected outputs (in-process mode)
results/results.csv
results/run_metadata.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_lib import (
    DEFAULT_MOOGP_PYTHON,
    DEFAULT_OILMM_PYTHON,
    DEFAULT_PUQ_PYTHON,
    ExperimentConfig,
    SUPPORTED_FUNCTIONS,
    SUPPORTED_METHODS,
    VAH_GROUPING_CHOICES,
    emit_job_list,
    run_benchmarks,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for experiment sweeps."""

    parser = argparse.ArgumentParser(description="Run MOOGP benchmark experiments.")
    parser.add_argument(
        "--functions",
        nargs="+",
        default=["borehole"],
        choices=SUPPORTED_FUNCTIONS,
        help="Benchmark functions to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["MOOGP", "MOGP", "LCGP", "OILMM", "PUQ"],
        choices=SUPPORTED_METHODS,
        help="Methods to run.",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[50, 100, 250, 1000, 2500],
        help="Training sample sizes.",
    )
    parser.add_argument(
        "--ps",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="Output dimensions.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=5,
        help="Number of replications per (function, p, n) cell.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=250,
        help="Number of test points per benchmark cell (ignored for vah_nuclear).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds for the vah_nuclear function (rep selects the fold).",
    )
    parser.add_argument(
        "--vah-grouping",
        choices=VAH_GROUPING_CHOICES,
        default="index",
        help=(
            "Diagonal error grouping for vah_nuclear: 'index' uses the groups in "
            "all_f_index.csv (passed to MOOGP/MOGP/LCGP); 'none' fits one noise "
            "per output. Ignored by OILMM and PUQ."
        ),
    )
    parser.add_argument(
        "--q",
        type=int,
        default=5,
        help="Maximum latent rank used for MOOGP and MOGP.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Optimizer max iterations for MOOGP and MOGP.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=1e-6,
        help="Jitter passed to MOOGP and MOGP.",
    )
    parser.add_argument(
        "--noise-var-frac",
        type=float,
        default=0.05,
        help="Per-output observation-noise variance as a fraction of the clean signal variance.",
    )
    parser.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the diagonalized-interaction fast path for MOOGP and MOGP.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes for independent benchmark rows.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260308,
        help="Master seed used to derive all dataset and model seeds.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where results.csv and run_metadata.json are written.",
    )
    parser.add_argument(
        "--moogp-python",
        type=Path,
        default=DEFAULT_MOOGP_PYTHON,
        help="Python executable used for MOOGP and MOGP runs.",
    )
    parser.add_argument(
        "--oilmm-python",
        type=Path,
        default=DEFAULT_OILMM_PYTHON,
        help="Python executable used for OILMM runs.",
    )
    parser.add_argument(
        "--puq-python",
        type=Path,
        default=DEFAULT_PUQ_PYTHON,
        help="Python executable used for PUQ runs.",
    )
    parser.add_argument(
        "--emit-jobs",
        type=Path,
        default=None,
        help=(
            "If set, write a per-cell job list to this path and exit instead of "
            "running the benchmark. Each line is `<run_id> <function> <method> "
            "<n> <p> <rep>`, designed for `parallel --colsep ' ' ./run_one.sh "
            "{1} {2} {3} {4} {5} {6} :::: jobs.txt` or AWS Batch array jobs."
        ),
    )
    parser.add_argument(
        "--jobs-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory each per-cell job writes its CSV row into when running "
            "via run_one.sh. Defaults to <results-dir>/jobs."
        ),
    )
    return parser


def main() -> int:
    """Parse CLI arguments, run the benchmark, and print a short summary."""

    args = build_parser().parse_args()
    config = ExperimentConfig(
        functions=tuple(args.functions),
        methods=tuple(args.methods),
        sample_sizes=tuple(args.ns),
        output_dims=tuple(args.ps),
        reps=args.reps,
        n_test=args.n_test,
        q=args.q,
        maxiter=args.maxiter,
        jitter=args.jitter,
        noise_var_frac=args.noise_var_frac,
        use_fast=args.use_fast,
        jobs=args.jobs,
        base_seed=args.base_seed,
        results_dir=args.results_dir,
        moogp_python=args.moogp_python,
        oilmm_python=args.oilmm_python,
        puq_python=args.puq_python,
        n_folds=args.n_folds,
        vah_grouping=args.vah_grouping,
    )

    if args.emit_jobs is not None:
        summary = emit_job_list(
            config=config,
            jobs_path=args.emit_jobs,
            output_dir=args.jobs_output_dir,
        )
        print(f"run_id={summary['run_id']}")
        print(f"config_json={summary['config_path']}")
        print(f"run_metadata={summary['metadata_path']}")
        print(f"jobs_file={summary['jobs_path']}")
        print(f"jobs_output_dir={summary['output_dir']}")
        print(f"n_jobs={summary['n_jobs']}")
        return 0

    summary = run_benchmarks(config)
    print(f"run_id={summary['run_id']}")
    print(f"results_csv={summary['results_path']}")
    print(f"run_metadata={summary['metadata_path']}")
    print(f"rows_written={len(summary['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
