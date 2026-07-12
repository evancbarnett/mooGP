## Setting up the benchmark environments

The benchmark uses three Python virtual environments because the methods have
incompatible TensorFlow / NumPy / Plum requirements:

| Path           | Methods                | What it has                                              |
| -------------- | ---------------------- | -------------------------------------------------------- |
| `.venv`        | MOOGP, MOGP, LCGP      | `requirements.txt` (TF 2.16 + gpflow + lcgp)             |
| `.venv-oilmm`  | OILMM                  | `oilmm`, `stheno`, `plum-dispatch`, `wbml`, TF 2.21      |
| `.venv-puq`    | PUQ                    | `PUQ` and `hetgpy` from GitHub (NumPy 2.x + SciPy 1.17)  |

All three target Python 3.11. The `ExperimentConfig` defaults
(`DEFAULT_MOOGP_PYTHON`, `DEFAULT_OILMM_PYTHON`, `DEFAULT_PUQ_PYTHON`) point at
these paths inside the repo, so once the venvs exist the runner picks the right
interpreter for each method automatically.

### `.venv` (MOOGP / MOGP / LCGP)

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install "setuptools<80"            
.venv/bin/pip install -r requirements.txt
```

The `moogp` package is imported directly from the repo, so no `pip install -e .`
is required — just run from the repo root. The `setuptools<80` pin is
deliberate: setuptools 80 removed `pkg_resources`, which `gpflow` still imports
at module load time. The benchmark code includes a runtime shim for this in
`_install_pkg_resources_shim`, so even without the pin a sweep will succeed —
but interactive `import gpflow` in the venv won't.

### `.venv-oilmm` (OILMM)

```bash
python -m venv .venv-oilmm
.venv-oilmm/bin/pip install --upgrade pip wheel
.venv-oilmm/bin/pip install \
    "oilmm==0.5.0" \
    "stheno==1.4.2" \
    "plum-dispatch==2.7.1" \
    "wbml==0.4.2" \
    "backends==1.9.0" \
    "backends-matrix==1.3.0" \
    "tensorflow==2.21.0" \
    "numpy<2"
```

The OILMM adapter installs a small `tensorflow_probability` shim and a
`plum.parametric` alias at runtime, so `tensorflow_probability` itself is **not**
required in this venv.

### `.venv-puq` (PUQ)

```bash
python -m venv .venv-puq
.venv-puq/bin/pip install --upgrade pip wheel
.venv-puq/bin/pip install \
    "git+https://github.com/davidogara/hetGPy.git" \
    "git+https://github.com/parallelUQ/PUQ.git"
```

`PUQ` pulls in NumPy 2.x and SciPy 1.17 transitively. Don't co-install MOOGP's
requirements into this venv — the NumPy/SciPy versions clash with TF 2.16.

### Verifying the setup

The benchmark adapters install runtime shims for legacy imports
(`plum.parametric`, `tensorflow_probability`, `pkg_resources`), so the
cleanest health check is to confirm the right packages are installed:

```bash
.venv/bin/pip show gpflow lcgp tensorflow > /dev/null && echo ".venv OK"
.venv-oilmm/bin/pip show oilmm stheno plum-dispatch wbml > /dev/null && echo ".venv-oilmm OK"
.venv-puq/bin/pip show PUQ hetgpy > /dev/null && echo ".venv-puq OK"
```

For an end-to-end check, run a one-cell smoke sweep — this exercises the
shimmed imports through the actual adapter code and returns
`status="error"` rows (not a crash) for any venv that's mis-installed:

```bash
.venv/bin/python -m experiments.run \
  --functions borehole \
  --methods MOOGP LCGP OILMM PUQ \
  --ns 50 --ps 4 --reps 1 --n-test 20 --maxiter 5 \
  --results-dir /tmp/moogp-smoke
```

### Thread diagnostics

Use the diagnostic script inside each venv to see which numeric libraries are
actually present. It prints the Python executable, NumPy/SciPy versions,
`np.show_config()`, active thread environment variables, `threadpoolctl` output
after a matrix multiplication, and TensorFlow thread settings when TensorFlow is
installed:

```bash
.venv/bin/python -m experiments.thread_diagnostics
.venv-oilmm/bin/python -m experiments.thread_diagnostics
.venv-puq/bin/python -m experiments.thread_diagnostics
```

If `threadpoolctl` is not installed in a venv, the script still runs but cannot
print BLAS thread-pool details. Install it into that venv when you want the
extra inspection:

```bash
<venv>/bin/pip install threadpoolctl
```

---

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

### CPU and thread settings

`run_one.sh` starts one process per benchmark cell, and `benchmark_lib.py`
dispatches that process into the method-specific venv when needed. Without
thread limits, each process can ask BLAS or TensorFlow for many CPUs, so
`parallel -j` can oversubscribe the instance badly.

Set `MOOGP_THREADS` to the per-process thread budget. When it is set,
`run_one.sh` fills any unset low-level thread variables before launching Python:
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, and
`TF_NUM_INTEROP_THREADS`. Existing explicit values are preserved, so you can
still override a specific backend variable in the shell.

The method adapters use these backends:

| Method | Adapter path | Main backend | Thread variables that matter |
| ------ | ------------ | ------------ | ---------------------------- |
| MOOGP / MOGP | `moogp.model.MOOGP`, SciPy optimizer/linalg, autograd NumPy | NumPy/SciPy BLAS/LAPACK | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`; `NUMEXPR_NUM_THREADS` is harmless if NumExpr appears |
| LCGP | `lcgp.LCGP` plus `gpflow.optimizers.Scipy` | TensorFlow/gpflow plus NumPy/SciPy | `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, plus the BLAS/OpenMP variables above |
| OILMM | `oilmm.tensorflow.OILMM`, `stheno` | TensorFlow plus NumPy | `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`, plus the BLAS/OpenMP variables above |
| PUQ | `PUQ.surrogate.emulator(method="multihetGP")`, hetGPy | NumPy/SciPy-style optimization and linear algebra | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`; no TensorFlow in this adapter |

On AWS Linux, the active BLAS control is usually `OPENBLAS_NUM_THREADS` for
PyPI NumPy/SciPy wheels or `MKL_NUM_THREADS` for conda/MKL builds.
`OMP_NUM_THREADS` is worth setting because some libraries use OpenMP internally.
`VECLIB_MAXIMUM_THREADS` is mainly for macOS Accelerate and is ignored on AWS,
but setting it is harmless.

Choose `JOBS` and `THREADS` so `JOBS * THREADS` is no more than the vCPU count
reported by `nproc`. For a `c7i.8xlarge` this is 32 vCPUs:

| Setting | When to try it |
| ------- | -------------- |
| `JOBS=8`, `THREADS=4` | Best first pass for mixed MOOGP/LCGP/OILMM/PUQ sweeps. It keeps all 32 vCPUs busy without giving any one optimizer an oversized thread pool. |
| `JOBS=16`, `THREADS=2` | Good for many smaller cells, especially MOOGP/MOGP/PUQ, where process-level parallelism usually beats larger BLAS pools. |
| `JOBS=4`, `THREADS=8` | Good for the largest LCGP/OILMM or high-`n` linear algebra cells if memory pressure or TensorFlow overhead makes too many simultaneous processes slow. |

For TensorFlow-heavy LCGP/OILMM-only sweeps, also benchmark
`TF_NUM_INTEROP_THREADS=1` with `TF_NUM_INTRAOP_THREADS=$THREADS`; it often
reduces scheduler contention when many independent jobs are already running.

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
JOBS=8
THREADS=4
MOOGP_CONFIG=results/config.json \
MOOGP_OUTPUT_DIR=results/jobs \
MOOGP_THREADS="$THREADS" \
parallel -j "$JOBS" --colsep ' ' ./experiments/run_one.sh \
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

**Important: detach the sweep from your SSH session first.** A full sweep
runs for hours to days. If you launch `parallel` directly on the SSH login
shell and the connection drops (laptop sleeps, Wi-Fi blip, you close the
terminal), every worker dies with SIGHUP and you lose the run. Always wrap
the sweep in a persistent terminal multiplexer — `tmux` is preinstalled on
Amazon Linux and one `apt-get` away on Ubuntu. The pattern:

```bash
# On the EC2 instance, before launching the sweep:
sudo apt-get install -y tmux        # Ubuntu; skip on Amazon Linux (preinstalled)
tmux new -s moogp                   # start a named session (use any name)

# ... your venv setup + the parallel command go inside this tmux session ...

# Detach without killing the session:   Ctrl-b  then  d
# Reattach later (after re-SSHing in):
tmux attach -t moogp                # or `tmux a` if there's only one session

# List sessions:                    tmux ls
# Kill a finished session:          tmux kill-session -t moogp
```

`nohup parallel ...` and `setsid` are alternatives, but tmux is strictly
better here because (a) you can re-attach and watch progress / read GNU
parallel's live stderr, (b) you can split the tmux window to monitor
`top`/`htop` and the per-job CSV directory in parallel, and (c) failed
restarts can be done in-session without re-uploading scripts.

A minimal recipe with a single EC2 instance and GNU parallel, run inside
the `tmux` session:

```bash
# On a fresh EC2 instance (Ubuntu / Amazon Linux). Pick an instance type sized
# for the largest cell in the sweep (e.g. c7i.4xlarge for a 16-core sweep).
sudo apt-get update && sudo apt-get install -y git build-essential parallel tmux python3.11 python3.11-venv

# Bring the repo and bench environments up. Each method that needs its own
# interpreter (OILMM, PUQ) gets its own venv at the paths configured in
# ExperimentConfig. See "Setting up the benchmark environments" above for the
# exact commands; in short:
git clone <your-fork-url> moogp-codex-play
cd moogp-codex-play
# .venv             — MOOGP / MOGP / LCGP
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# .venv-oilmm       — OILMM (see setup section above for the full command)
python -m venv .venv-oilmm && .venv-oilmm/bin/pip install \
    oilmm==0.5.0 stheno==1.4.2 plum-dispatch==2.7.1 wbml==0.4.2 \
    backends==1.9.0 backends-matrix==1.3.0 tensorflow==2.21.0 "numpy<2"
# .venv-puq         — PUQ (pulls hetgpy + PUQ from GitHub)
python -m venv .venv-puq && .venv-puq/bin/pip install \
    git+https://github.com/davidogara/hetGPy.git \
    git+https://github.com/parallelUQ/PUQ.git

# Generate the job list locally OR on the instance — both are deterministic.
.venv/bin/python -m experiments.run \
  --functions borehole forrester_mixed \
  --methods MOOGP MOGP LCGP OILMM PUQ \
  --ns 50 100 250 1000 2500 \
  --ps 10 20 50 \
  --reps 5 \
  --results-dir results \
  --emit-jobs results/jobs.txt

# Run the grid INSIDE tmux (see the tmux callout above) so an SSH drop does
# not kill the sweep. Keep JOBS * THREADS at or below $(nproc); --joblog
# gives a resumable log. On c7i.8xlarge, start with 8 * 4 = 32.
JOBS=8
THREADS=4
MOOGP_CONFIG="$PWD/results/config.json" \
MOOGP_OUTPUT_DIR="$PWD/results/jobs" \
MOOGP_SKIP_EXISTING=1 \
MOOGP_THREADS="$THREADS" \
parallel -j "$JOBS" --colsep ' ' --joblog results/jobs.log \
  ./experiments/run_one.sh {1} {2} {3} {4} {5} {6} :::: results/jobs.txt
# After this kicks off:  Ctrl-b d  to detach, then `tmux a` to peek back in.

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

---

## VAH heavy-ion dataset (`--functions vah_nuclear`)

The `vah_nuclear` function runs k-fold cross-validation on the VAH simulator
data shipped at `examples/nuclear_data/`:

| File                | Shape       | Notes |
| ------------------- | ----------- | ----- |
| `all_theta.csv`     | 541 × 15    | Headerless input design matrix. |
| `all_f.csv`         | 541 × 98    | Headerless simulator outputs (98 observables; `pT_fluct` already dropped). |
| `all_f_index.csv`   | metadata    | Group widths used for the diagonal noise structure. |

The runner ignores `--ns` / `--ps` for `vah_nuclear`: every fold is fit on the
full design (one held-out fold per cell), `n` and `p` in the CSV are the
dataset-wide constants `541` and `98`, and `rep` is the held-out fold index
(`1..n_folds`). Inputs and outputs are z-scored using **training-fold
statistics only**, then the same transform is applied to the held-out fold, so
RMSE / coverage / DSS are reported in standardized space and are comparable
across folds and observables.

VAH-specific flags (defaults in parentheses):

 - `--n-folds INT` (`5`) — number of CV folds. Reps cycle 1..n_folds; pass
   `--reps n_folds` for full coverage.
 - `--vah-grouping {index,none}` (`index`) — `index` reads the diagonal error
   grouping `[8, 22, 8, 8, 8, 8, 8, 8, 8, 6, 6]` from `all_f_index.csv` and
   passes it to MOOGP/MOGP/LCGP (one log-noise per observable group). `none`
   gives every output its own log-noise. OILMM and PUQ ignore this flag because
   their interfaces don't expose a per-output diagonal grouping.

Each row in `results.csv` adds two columns relative to a borehole sweep:

 - `train_rmse` — RMSE on the training fold (in standardized space; recorded
   for every function, including borehole / forrester).
 - `n_folds` — populated for `vah_nuclear` rows, blank otherwise.

The fold split is keyed by `(base_seed, "vah_nuclear", "kfold", n_folds)`, so
the held-out indices for fold `k` are stable across reruns and across methods
within the same sweep — comparing methods row-by-row is apples-to-apples.

### In-process VAH example

```
.venv/bin/python -m experiments.run \
  --functions vah_nuclear \
  --methods MOOGP MOGP LCGP OILMM PUQ \
  --reps 5 \
  --n-folds 5 \
  --q 5 \
  --maxiter 1000 \
  --base-seed 123 \
  --results-dir results-vah
```

Writes `results-vah/results.csv` and `results-vah/run_metadata.json`. Switch
`--vah-grouping none` to compare against the per-output baseline (only changes
behavior for MOOGP / MOGP / LCGP).

To mix VAH with the synthetic functions in one sweep, just pass them together —
the synthetic functions still iterate `--ns` / `--ps` / `--reps` as usual while
VAH cells iterate folds:

```
.venv/bin/python -m experiments.run \
  --functions borehole forrester_mixed vah_nuclear \
  --methods MOOGP LCGP \
  --ns 100 500 --ps 10 20 --reps 5 \
  --n-folds 5 \
  --results-dir results-mixed
```

### Parallel VAH on AWS

Same three-step pattern (emit → run → merge) as a borehole sweep; only the
emit-step flags change. On a fresh EC2 instance with the three venvs already
provisioned (see “Setting up the benchmark environments”):

```bash
# 1. Emit the job list. Each VAH (function, method, rep) becomes one cell;
#    no --ns / --ps loops. With --reps == --n-folds == 5 and 5 methods,
#    you get 25 jobs.
.venv/bin/python -m experiments.run \
  --functions vah_nuclear \
  --methods MOOGP MOGP LCGP OILMM PUQ \
  --reps 5 \
  --n-folds 5 \
  --vah-grouping index \
  --q 5 \
  --maxiter 1000 \
  --base-seed 123 \
  --results-dir results-vah \
  --emit-jobs results-vah/jobs.txt

# 2. Run the grid. Same JOBS / THREADS guidance as the borehole sweep:
#    keep JOBS * THREADS <= $(nproc). VAH cells are larger (n≈432, p=98)
#    so MOOGP / LCGP benefit from THREADS=4 or 8.
JOBS=8
THREADS=4
MOOGP_CONFIG="$PWD/results-vah/config.json" \
MOOGP_OUTPUT_DIR="$PWD/results-vah/jobs" \
MOOGP_SKIP_EXISTING=1 \
MOOGP_THREADS="$THREADS" \
parallel -j "$JOBS" --colsep ' ' --joblog results-vah/jobs.log \
  ./experiments/run_one.sh {1} {2} {3} {4} {5} {6} :::: results-vah/jobs.txt

# 3. Merge and copy off the instance. Each per-job CSV is named
#    vah_nuclear__<METHOD>__n541__p98__rep<k>.csv.
.venv/bin/python -m experiments.merge_results \
  --input-dir results-vah/jobs \
  --output    results-vah/results.csv
aws s3 cp results-vah/results.csv s3://<your-bucket>/<run-prefix>/results-vah.csv
```

For an AWS Batch array job, set the array size to the line count of
`results-vah/jobs.txt` and dispatch each line through `run_one.sh` exactly as
in the borehole recipe — there is nothing VAH-specific about the array job
itself.

### VAH notes and gotchas

 - `--ns` and `--ps` are silently ignored for `vah_nuclear`; the loop only
   iterates `(method, rep)` because the dataset has fixed dimensions. Use a
   separate sweep if you want to subsample VAH.
 - `--reps` controls how many folds you actually run, capped at `--n-folds`.
   With `--reps 3 --n-folds 5` you only fit folds 1–3 (useful for a quick
   smoke); with `--reps >= --n-folds` you get full CV. Passing `--reps` larger
   than `--n-folds` is silently clamped, since each fold is deterministic and
   re-running it would produce identical rows.
 - The MOOGP / MOGP / LCGP optimizers see *grouped* `sigma_eps`: the parameter
   vector has `len(groups) = 11` log-noise entries instead of 98, so smaller
   `--maxiter` budgets are usually fine compared to the per-output case.
 - OILMM and PUQ run with the same standardized fold splits but ignore
   `--vah-grouping` — their reported RMSE is still directly comparable to the
   grouped fits because all methods see the same standardized targets.
 - The `seed_data` column on VAH rows is the deterministic KFold seed
   (a function of `base_seed` and `n_folds`), not a per-cell seed; this is
   what makes the fold partition reproducible across reruns.

---

## Timing methodology — controlled wall-clock measurements

The parallel mode above (`JOBS=8 THREADS=4`) is tuned for **throughput** — get
the whole sweep done as fast as possible. Wall-clock recorded from a throughput
run is **not** a controlled timing measurement, because effective per-job
parallelism varies over the sweep as cells of different sizes complete.

In practice we have observed up to a **3× spread** in `fit_time_seconds` within
a single (function, method, n, p) cell on `JOBS=8 THREADS=4` runs (e.g. LCGP
n=2500 p=5 runtimes ranging from ~24k to ~75k seconds across 10 reps), with
zero job failures and zero OOM kills. The variance comes entirely from the
fluctuating system state: SMT-thread sharing when all 8 workers are busy
vs. full physical cores when smaller jobs finish and leave slots empty,
plus shared L3 / memory-bandwidth contention.

### Goal

Produce per-method wall-clock numbers that:

1. **Compare fairly across methods** — every method sees the same per-fit CPU
   budget, so any timing difference is attributable to the method, not to who
   got more BLAS threads at the time.
2. **Reproduce within ~10%** rep-to-rep — so median over 5 reps is a stable
   summary statistic.
3. **Run on the same hardware as the accuracy sweep** — so accuracy and timing
   numbers can be cited from the same experimental setup with no
   "different-machine" footnote.

The throughput sweep is kept for accuracy metrics (RMSE, coverage, DSS,
log-score); the timing sweep below produces wall-clock only.

### Setup overview ("Option D")

Same instance as the accuracy sweep (`c7i.8xlarge`, 16 physical Sapphire
Rapids cores × 2 SMT = 32 vCPUs, 64 GiB). Run **4 concurrent fits**, each
pinned via `taskset` to **4 dedicated physical cores**. SMT siblings (CPUs
16–31) are left idle so no fit ever shares a physical core with another fit.
The per-fit thread budget (`THREADS=4`) matches the accuracy sweep.

| Resource | Value | Why |
| -------- | ----- | --- |
| `JOBS` | 4 | One fit per non-overlapping 4-core block (cores 0–3, 4–7, 8–11, 12–15) |
| `THREADS` | 4 | Same per-fit budget as the accuracy sweep → directly comparable |
| Cores 16–31 | unused | SMT siblings; leaving idle eliminates intra-job hardware-thread sharing |
| `taskset -c {%}*4 …` | pinning per slot | Each GNU parallel slot is locked to its 4 physical cores; kernel cannot migrate |

This gives ~4× throughput vs serial (`JOBS=1`) at the same dollar cost, with
~97% of the timing purity of running multiple isolated smaller instances. The
remaining ~3% is L3-cache and memory-bandwidth contention between the 4
concurrent fits, which for n=2500 GP workloads is below the rep-to-rep
optimizer noise floor.

### Pre-flight: verify CPU topology

`taskset` operates on **OS CPU IDs**, not physical-core IDs. Before launching,
confirm that CPU IDs 0..15 map to distinct physical cores (i.e. SMT siblings
are CPU IDs 16..31), which is the standard layout on `c7i`:

```bash
lscpu --extended=CPU,CORE,SOCKET | head -40
```

Expected layout for `c7i.8xlarge`:

```
CPU CORE SOCKET
  0    0      0
  1    1      0
  ...
 15   15      0
 16    0      0   <-- SMT sibling of CPU 0
 17    1      0   <-- SMT sibling of CPU 1
  ...
```

If the layout is interleaved instead (CPU 0 / CPU 1 both on CORE 0), the
pinning expression needs to step by 2, e.g. `taskset -c 0,2,4,6` for slot 1.

### 1. Emit a focused timing job list

Limit the timing sweep to cells you actually want to cite. Small-n cells
(n ≤ 100) are dominated by Python / TensorFlow startup and are not
informative for a scaling story; drop them.

```bash
.venv/bin/python -m experiments.run \
  --functions borehole \
  --methods MOOGP MOGP LCGP OILMM PUQ \
  --ns 250 1000 2500 \
  --ps 5 10 20 \
  --reps 5 \
  --base-seed 123 \
  --results-dir results_timing \
  --emit-jobs results_timing/jobs.txt

wc -l results_timing/jobs.txt   # expect 5 methods × 3 ns × 3 ps × 5 reps = 225
```

Use **the same `--base-seed`** as the accuracy sweep so that
`(base_seed, function, n, p, rep)` resolves to identical method seeds — the
fit produced by the timing run is bit-identical to the corresponding row in
the accuracy sweep. This is the link that lets you cite both numbers as
coming from "the same experimental configuration."

### 2. Capture provenance

Reviewers will ask exactly which hardware and which package versions
produced the numbers. Capture once, at launch:

```bash
mkdir -p results_timing/provenance
{
  echo "=== instance ==="
  curl -s http://169.254.169.254/latest/meta-data/instance-type; echo
  echo "=== nproc ===";  nproc
  echo "=== lscpu ==="
  lscpu
  echo "=== topology ==="
  lscpu --extended=CPU,CORE,SOCKET
  echo "=== mem ===";    free -h
  echo "=== kernel ===";  uname -a
  echo "=== blas (.venv) ==="
  .venv/bin/python -c "import numpy; numpy.show_config()" || true
  .venv/bin/python -m experiments.thread_diagnostics || true
} > results_timing/provenance/env.txt 2>&1

git rev-parse HEAD        > results_timing/provenance/git_sha.txt
git status --porcelain    > results_timing/provenance/git_status.txt
.venv/bin/pip freeze       > results_timing/provenance/pip_freeze.venv.txt
.venv-oilmm/bin/pip freeze > results_timing/provenance/pip_freeze.venv-oilmm.txt
.venv-puq/bin/pip freeze   > results_timing/provenance/pip_freeze.venv-puq.txt
```

### 3. Launch the pinned sweep inside tmux

```bash
tmux new -s moogp-timing
```

Inside the tmux session, from the repo root:

```bash
JOBS=4
THREADS=4
MOOGP_CONFIG="$PWD/results_timing/config.json" \
MOOGP_OUTPUT_DIR="$PWD/results_timing/jobs" \
MOOGP_SKIP_EXISTING=1 \
MOOGP_THREADS="$THREADS" \
parallel -j "$JOBS" --colsep ' ' --joblog results_timing/jobs.log \
  'taskset -c $((({%}-1)*4))-$(( ({%}-1)*4 + 3 )) ./experiments/run_one.sh {1} {2} {3} {4} {5} {6}' \
  :::: results_timing/jobs.txt
```

What the `taskset` expression does, slot by slot:

| GNU parallel slot (`{%}`) | `taskset -c` range | Physical cores |
| --- | --- | --- |
| 1 | `0-3`   | 0, 1, 2, 3 |
| 2 | `4-7`   | 4, 5, 6, 7 |
| 3 | `8-11`  | 8, 9, 10, 11 |
| 4 | `12-15` | 12, 13, 14, 15 |

The kernel cannot migrate a job's threads off its assigned cores, and no two
jobs share a physical core. With `OMP_NUM_THREADS=4` (set by `run_one.sh`
from `MOOGP_THREADS=4`), each BLAS thread lands on its own dedicated core.

Detach with `Ctrl-b d`. Re-attach with `tmux a -t moogp-timing`.

### 4. Continuous sanity monitoring

In a separate tmux pane (`Ctrl-b "` to split horizontally):

```bash
watch -n 30 '
  echo "=== last 3 completed jobs ===";
  tail -3 results_timing/jobs.log;
  echo;
  echo "=== memory ===";
  free -h;
  echo;
  echo "=== active workers ===";
  pgrep -af method_runner;
  echo;
  echo "=== per-worker CPU affinity ===";
  for pid in $(pgrep method_runner); do
    echo "PID $pid: $(taskset -pc $pid 2>/dev/null | sed s/.*list:.//)";
  done;
  echo;
  echo "=== files written ===";
  ls results_timing/jobs 2>/dev/null | wc -l
'
```

What to watch for:

- **`pgrep -af method_runner` returns at most 4 lines.** If it ever returns
  5+, GNU parallel has launched more jobs than `JOBS=4` — investigate.
- **Each worker's affinity list is exactly 4 CPUs from a single slot's range
  (0–3, 4–7, 8–11, or 12–15).** If a worker shows the full `0-31` range,
  the `taskset` wrapper failed and the run is invalid.
- **No `Signal=9` rows in `jobs.log`** — that would indicate an OOM kill
  (extremely unlikely with `JOBS=4` on 64 GiB).
- **Memory `available` stays above ~20 GiB.** With 4 concurrent fits each
  using ~6–8 GiB RSS, this should be comfortable.

### 5. Merge and stash

```bash
.venv/bin/python -m experiments.merge_results \
  --input-dir results_timing/jobs \
  --output    results_timing/timing_results.csv

aws s3 cp results_timing/timing_results.csv \
  s3://<your-bucket>/<run-prefix>/timing_results.csv
aws s3 sync results_timing/provenance/ \
  s3://<your-bucket>/<run-prefix>/timing_provenance/
```

Keep the output filename distinct from the accuracy sweep's `results.csv` so
the two are never accidentally co-plotted.

### 6. Quality checks before publishing the numbers

```bash
# Any non-zero exits or signals?
awk 'NR>1 && ($7 != 0 || $8 != 0)' results_timing/jobs.log

# Did every requested cell complete?
wc -l results_timing/jobs.txt
ls results_timing/jobs | wc -l

# Within-cell coefficient of variation. Should be < ~10%.
.venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_csv("results_timing/timing_results.csv")
g = df.groupby(["method","n","p"])["fit_time_seconds"]
out = g.agg(["mean","std","count"]).assign(cv=lambda x: x["std"]/x["mean"])
print(out.sort_values("cv", ascending=False).head(20))
PY
```

If any (method, n, p) cell has **CV > 15%**, do not report median timing for
that cell as-is. Likely causes:

- A background process landed on the box mid-sweep (check `dmesg`/`journalctl`
  around the slow rep's timestamp).
- The first rep is much slower than the others due to JIT / import warm-up —
  trim the first rep and re-aggregate, or report median of reps 2–5.
- The method has genuine optimizer-stochasticity-driven runtime variance.
  In that case report it (e.g. with an IQR) rather than hide it.

Cross-check against the accuracy sweep's *cleanest* runs (e.g. for LCGP
n=2500 p=5 the fastest throughput-run rep was ~24k sec). The pinned timing
sweep should produce numbers in the same ballpark — within ~10% of that
floor. If it's substantially slower, the pinning may not be in effect;
re-verify with `taskset -pc <pid>` on a live worker.

### 7. Reporting language

A short methodology paragraph that pre-empts every likely reviewer question:

> Wall-clock timings are reported from a dedicated timing sweep on a single
> AWS `c7i.8xlarge` instance (16 physical Intel Sapphire Rapids cores, 32
> vCPUs with SMT, 64 GiB RAM, Ubuntu 24.04). Four fits ran concurrently,
> each pinned via `taskset` to four dedicated physical cores
> (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=`
> `TF_NUM_INTRAOP_THREADS=4`); SMT siblings were left idle so no fit shared
> a physical core with another. We report median over 5 replications per
> (method, n, p). Accuracy metrics (RMSE, coverage, DSS) are taken from a
> separate larger parallel sweep with deterministic seeds derived from
> `(base_seed, function, n, p, rep)`; the per-fit code path is unchanged
> between the two sweeps, so the timing-sweep fits are bit-identical to the
> corresponding accuracy rows.

That paragraph is true **only if** `--base-seed` matches between the two
sweeps. Verify before submission.

### When this setup is not appropriate

- **Sub-percent timing claims.** If you need to defend "method X is 2.3%
  faster than method Y," the L3 / memory-bandwidth contention between the
  4 concurrent fits is no longer negligible. Drop to `JOBS=1` (one fit at a
  time, 12 idle physical cores) or run separate `c7i.2xlarge` instances in
  parallel with `JOBS=1` on each.
- **Different per-method thread budgets.** If you want to claim "method X
  given all 16 physical cores takes T seconds," you need a different setup
  (`JOBS=1 THREADS=16` and verified per-method scaling). That's a different
  experiment with a different claim.
- **A different instance family for timing vs. accuracy.** Don't.
  Re-running on identical hardware is one of the easiest ways to preserve
  the methodological link.
