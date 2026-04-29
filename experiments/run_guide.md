## In-process example

```
python -m experiments.run \
  --functions borehole \
  --methods LCGP OILMM \
  --ns 100 500 \
  --ps 4 8 \
  --q 3 \
  --reps 10 \
  --results-dir results-oilmm-lcgp-only \
  --base-seed 123
```

**Ensure the backslash is the last character on each line or the terminal command will not work.**

Default values:
 - Function: `borehole`
 - Methods: `["MOOGP", "MOGP", "LCGP", "OILMM", "PUQ"]`
 - Training sizes (`--ns`): `[50, 100, 250, 1000, 2500]`
 - Output dimension (`--ps`): `[10, 20, 50]`
 - Replications (`--reps`): `5`
 - Test size (`--n-test`): `250`
 - Latent dimension (`--q`): `5`
 - Optimizer max iterations (`--maxiter`): `1000`
 - Jitter: `1e-6`
 - Per-output noise: `0.05 * Var(y)`

In-process mode writes:
 - `<results-dir>/results.csv`
 - `<results-dir>/run_metadata.json`

---

## Parallel mode (one process per cell)

The same sweep can be split into one independent process per benchmark cell —
each cell is `(function, method, n, p, rep)`. Each process writes its own CSV
file so parallel writers never contend on a single shared file. This is what
runs on AWS.

The workflow is three steps:
 1. **Emit** the job list and shared config.
 2. **Run** each line of the job list as an independent process.
 3. **Merge** the per-job CSVs into one `results.csv` matching the in-process
    schema.

### 1. Emit the job list

```
python -m experiments.run \
  --functions borehole \
  --methods MOOGP MOGP LCGP \
  --ns 100 500 \
  --ps 4 8 \
  --q 3 \
  --reps 10 \
  --base-seed 123 \
  --results-dir results \
  --emit-jobs results/jobs.txt
```

Writes:
 - `results/config.json` — the `ExperimentConfig` shared by every job.
 - `results/run_metadata.json` — sweep metadata (run_id, host, etc.).
 - `results/jobs.txt` — one whitespace-separated line per cell:
   `<run_id> <function> <method> <n> <p> <rep>`.

The default per-job output directory is `<results-dir>/jobs`; override with
`--jobs-output-dir`.

### 2a. Run locally with GNU parallel

```
MOOGP_CONFIG=results/config.json \
MOOGP_OUTPUT_DIR=results/jobs \
parallel -j 6 --colsep ' ' ./experiments/run_one.sh \
  {1} {2} {3} {4} {5} {6} :::: results/jobs.txt
```

Each invocation writes
`results/jobs/<function>__<method>__n<n>__p<p>__rep<r>.csv` containing the
canonical CSV header and one row. Reruns are idempotent (the per-job CSV is
overwritten). Pass `MOOGP_SKIP_EXISTING=1` to skip cells whose CSV already
exists, which is convenient for resuming partial runs.

Useful additions:
 - `parallel ... --joblog results/jobs.log` — record exit codes and timings.
 - `parallel ... --resume --joblog results/jobs.log` — pick up where you left off.

### 2b. Run on AWS

The same `run_one.sh` works under any of the typical AWS execution patterns.
A minimal recipe with a single EC2 instance and GNU parallel:

```bash
# On a fresh EC2 instance (Ubuntu / Amazon Linux). Pick an instance type sized
# for the largest cell in the sweep (e.g. c7i.4xlarge for a 16-core sweep).
sudo apt-get update && sudo apt-get install -y git build-essential parallel python3-venv

# Bring the repo and bench environments up. Each method that needs its own
# interpreter (OILMM, PUQ) gets its own venv at the paths configured in
# ExperimentConfig (.venv-oilmm, .venv-puq).
git clone <your-fork-url> moogp-codex-play
cd moogp-codex-play
python3 -m venv .venv && .venv/bin/pip install -e . -r requirements.txt
# (repeat for .venv-oilmm and .venv-puq using their dedicated requirements)

# Generate the job list locally OR on the instance — both are deterministic.
.venv/bin/python -m experiments.run \
  --functions borehole forrester_mixed \
  --methods MOOGP MOGP LCGP OILMM PUQ \
  --ns 50 100 250 1000 2500 \
  --ps 10 20 50 \
  --reps 5 \
  --results-dir results \
  --emit-jobs results/jobs.txt

# Run the grid. -j matches vCPU count; --joblog gives a resumable log.
MOOGP_CONFIG="$PWD/results/config.json" \
MOOGP_OUTPUT_DIR="$PWD/results/jobs" \
MOOGP_SKIP_EXISTING=1 \
parallel -j "$(nproc)" --colsep ' ' --joblog results/jobs.log \
  ./experiments/run_one.sh {1} {2} {3} {4} {5} {6} :::: results/jobs.txt

# Merge and copy the artefact off the instance.
.venv/bin/python -m experiments.merge_results \
  --input-dir results/jobs \
  --output    results/results.csv
aws s3 cp results/results.csv s3://<your-bucket>/<run-prefix>/results.csv
```

For an AWS Batch array job, set the array size to the number of lines in
`jobs.txt` and have each container read its line via
`sed -n "${AWS_BATCH_JOB_ARRAY_INDEX}p" jobs.txt | xargs ./experiments/run_one.sh`.
The required env vars (`MOOGP_CONFIG`, `MOOGP_OUTPUT_DIR`) belong in the job
definition. Point `MOOGP_OUTPUT_DIR` at an EFS mount or a per-job S3-synced
directory so the merge step can collect every row at the end.

### 3. Merge per-job CSVs into one results.csv

```
python -m experiments.merge_results \
  --input-dir results/jobs \
  --output    results/results.csv
```

The merged file uses the same schema as the in-process `results.csv`, so any
existing analysis / plotting code (e.g. `plot_benchmark_boxplots.py`) works
without changes.

### Notes

 - `run_one.py` reuses `run_single_method_job`, so cells whose method needs a
   dedicated interpreter (LCGP/MOOGP via `.venv`, OILMM via `.venv-oilmm`,
   PUQ via `.venv-puq`) still dispatch into the right environment.
 - Seeds are derived from `(base_seed, function, n, p, rep)`, so an emitted
   job list is fully reproducible: rerunning one line produces the same row.
 - If a cell crashes, `run_single_method_job` records `status="error"` and the
   exception text in the per-job CSV — the worker still exits 0, so a single
   bad cell does not abort the parallel batch.
