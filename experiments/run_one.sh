#!/usr/bin/env bash
# Run one benchmark cell. Designed to be invoked one line at a time from
# jobs.txt under GNU parallel, AWS Batch array jobs, or Slurm.
#
# Each line of jobs.txt has six whitespace-separated fields:
#     <run_id> <function> <method> <n> <p> <rep>
# and run_one.sh expects the matching positional arguments.
#
# Required environment variables:
#   MOOGP_CONFIG        Path to the ExperimentConfig JSON for this sweep.
#                       (Created by `python -m experiments.run --emit-jobs ...`.)
#   MOOGP_OUTPUT_DIR    Directory where per-job CSV files are written.
#
# Optional environment variables:
#   MOOGP_REPO_ROOT     Absolute path to the moogp-codex-play checkout.
#                       Defaults to the parent of this script.
#   MOOGP_PYTHON        Python executable that can import the experiments
#                       package. Defaults to $MOOGP_REPO_ROOT/.venv/bin/python.
#   MOOGP_SKIP_EXISTING If set to "1", skip cells whose CSV already exists.
#
# Example (local, GNU parallel, 6-way):
#   MOOGP_CONFIG=results/config.json \
#   MOOGP_OUTPUT_DIR=results/jobs \
#     parallel -j 6 --colsep ' ' ./experiments/run_one.sh \
#       {1} {2} {3} {4} {5} {6} :::: results/jobs.txt

set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 <run_id> <function> <method> <n> <p> <rep>" >&2
    exit 2
fi

run_id="$1"
function="$2"
method="$3"
n="$4"
p="$5"
rep="$6"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${MOOGP_REPO_ROOT:-$(cd "$script_dir/.." && pwd)}"
python_bin="${MOOGP_PYTHON:-$repo_root/.venv/bin/python}"
config_path="${MOOGP_CONFIG:?MOOGP_CONFIG must point to the saved ExperimentConfig JSON.}"
output_dir="${MOOGP_OUTPUT_DIR:?MOOGP_OUTPUT_DIR must point to the per-job output directory.}"

cd "$repo_root"
if [[ "${MOOGP_SKIP_EXISTING:-0}" == "1" ]]; then
    exec "$python_bin" -m experiments.run_one \
        --config "$config_path" \
        --run-id "$run_id" \
        --function "$function" \
        --method "$method" \
        --n "$n" \
        --p "$p" \
        --rep "$rep" \
        --output-dir "$output_dir" \
        --skip-existing
else
    exec "$python_bin" -m experiments.run_one \
        --config "$config_path" \
        --run-id "$run_id" \
        --function "$function" \
        --method "$method" \
        --n "$n" \
        --p "$p" \
        --rep "$rep" \
        --output-dir "$output_dir"
fi
