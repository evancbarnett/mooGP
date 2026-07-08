# MOOGP examples

Runnable notebooks that mirror the sections of the top-level
[`README`](../README.md). Start with **Getting started**, then dip into whichever
advanced feature you need — each notebook is self-contained and matches one
README section.

| Notebook | Covers | README section |
| --- | --- | --- |
| [`00_getting_started.ipynb`](00_getting_started.ipynb) | End-to-end workflow: generate data → fit → predict → evaluate → inspect | [Basic Usage](../README.md#basic-usage) |
| [`01_trend_specification.ipynb`](01_trend_specification.ipynb) | The trend basis `g(x)` via `terms` (intercepts, main effects, interactions) | [Specifying the trend with `terms`](../README.md#specifying-the-trend-with-terms) |
| [`02_choosing_q.ipynb`](02_choosing_q.ipynb) | Picking the number of latent GPs `q` from the singular-value spectrum of `Y` | [Choosing `q`](../README.md#choosing-q) |
| [`03_measurement_noise.ipynb`](03_measurement_noise.ipynb) | Fixed vs learned noise and grouping outputs with `diag_error_structure` | [Measurement noise and `diag_error_structure`](../README.md#measurement-noise-and-diag_error_structure) |
| [`04_standardization.ipynb`](04_standardization.ipynb) | `standardize_x` / `standardize_y` and when to disable them | [Standardization](../README.md#standardization) |
| [`05_optimizer_control.ipynb`](05_optimizer_control.ipynb) | `optimizer_opts` (`maxiter`, `ftol`, `gtol`) and reading the optimizer result | [Optimizer control and tolerance](../README.md#optimizer-control-and-tolerance) |
| [`06_kernel_and_mixing_matrix.ipynb`](06_kernel_and_mixing_matrix.ipynb) | The `orthogonal` kernel and fixing/learning the mixing matrix `Psi` | [Kernel and mixing-matrix options](../README.md#kernel-and-mixing-matrix-options) |

## Running them

```bash
pip install moogp           # or `pip install -e .` from the repo root
pip install matplotlib      # the notebooks plot with matplotlib
jupyter lab                 # open any notebook and run all cells
```

The notebooks assume `moogp` is installed and simply `import` it.
