from __future__ import annotations

import json
import sys

from .benchmark_lib import ExperimentConfig, run_single_method_job_local


def main() -> int:
    """Run one benchmark row from JSON stdin and emit the result row as JSON."""

    payload = json.load(sys.stdin)
    config = ExperimentConfig.from_metadata(payload["config"])
    row = run_single_method_job_local(
        run_id=payload["run_id"],
        function=payload["function"],
        method=payload["method"],
        n=int(payload["n"]),
        p=int(payload["p"]),
        rep=int(payload["rep"]),
        seed_data=int(payload["seed_data"]),
        config=config,
    )
    json.dump(row, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
