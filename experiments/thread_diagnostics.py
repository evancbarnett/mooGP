"""Print numeric backend and thread-pool diagnostics for a benchmark venv.

Run this inside each benchmark virtualenv, for example:

    .venv/bin/python -m experiments.thread_diagnostics
    .venv-oilmm/bin/python -m experiments.thread_diagnostics
    .venv-puq/bin/python -m experiments.thread_diagnostics
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from typing import Any


THREAD_ENV_VARS = (
    "MOOGP_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
)


def _print_section(title: str) -> None:
    print(f"\n== {title} ==")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _capture_np_show_config(np: Any) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue().rstrip()


def _run_numpy_matmul(np: Any, size: int) -> float:
    rng = np.random.default_rng(12345)
    a = rng.standard_normal((size, size))
    b = rng.standard_normal((size, size))
    c = a @ b
    return float(c[0, 0])


def _print_tensorflow_info() -> None:
    _print_section("TensorFlow")
    if importlib.util.find_spec("tensorflow") is None:
        print("not installed")
        return

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - depends on optional venv state
        print(f"import failed: {type(exc).__name__}: {exc}")
        return

    print(f"version: {tf.__version__}")
    print(f"intra_op_parallelism_threads: {tf.config.threading.get_intra_op_parallelism_threads()}")
    print(f"inter_op_parallelism_threads: {tf.config.threading.get_inter_op_parallelism_threads()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print NumPy/SciPy/BLAS/TensorFlow thread diagnostics.")
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=512,
        help="Square matrix size used to force BLAS thread pools to initialize.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.matmul_size <= 0:
        raise SystemExit("--matmul-size must be a positive integer.")

    _print_section("Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.replace(os.linesep, ' ')}")

    _print_section("Thread Environment")
    _print_json({name: os.environ.get(name) for name in THREAD_ENV_VARS})

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(f"NumPy is not installed in this environment: {exc}") from exc

    _print_section("NumPy/SciPy")
    print(f"numpy: {np.__version__}")
    if importlib.util.find_spec("scipy") is None:
        print("scipy: not installed")
    else:
        import scipy

        print(f"scipy: {scipy.__version__}")

    _print_section("np.show_config()")
    print(_capture_np_show_config(np))

    _print_section("Matrix Multiplication")
    sentinel = _run_numpy_matmul(np, args.matmul_size)
    print(f"matmul_size: {args.matmul_size}")
    print(f"sentinel: {sentinel:.12g}")

    _print_section("threadpoolctl.threadpool_info()")
    if importlib.util.find_spec("threadpoolctl") is None:
        print("threadpoolctl is not installed in this environment.")
    else:
        from threadpoolctl import threadpool_info

        _print_json(threadpool_info())

    _print_tensorflow_info()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
