# MOOGP — Multi-Output Orthogonal Gaussian Process

`moogp` emulates vector-valued (multi-output) functions with a **multi-output
orthogonal Gaussian process**. It models the `p` outputs as a linear combination
of `q` shared latent GPs, plus a regression trend `g(x)` that the latent
kernels are orthogonalized against. Fitting the trend separately keeps its
coefficients interpretable and improves extrapolation.

## Table of Contents

- [Installation](#installation)
  - [Test suite](#test-suite)
- [Basic Usage](#basic-usage)
- [The `MOOGP` Class](#the-moogp-class)
- [Advanced Usage](#advanced-usage)
  - [Specifying the trend with `terms`](#specifying-the-trend-with-terms)
  - [Choosing `q`](#choosing-q)
  - [Measurement noise and `diag_error_structure`](#measurement-noise-and-diag_error_structure)
  - [Standardization](#standardization)
  - [Optimizer control and tolerance](#optimizer-control-and-tolerance)
  - [Kernel and mixing-matrix options](#kernel-and-mixing-matrix-options)

## Installation

`moogp` is on PyPI and requires Python 3.11 or above:

```bash
pip install moogp
```

Its only runtime dependencies are `numpy`, `scipy`, and `autograd`.

### Test suite

A set of tests is provided to verify that `moogp` is installed correctly:

```bash
$ python
>>> import moogp
>>> moogp.__version__
<version string>
>>> moogp.test()
```

Or run `pytest` directly:

```bash
pytest src/moogp/tests
```

## Basic Usage

```python
import numpy as np
from moogp.model import MOOGP
from moogp import evaluation  # optional evaluation module

# Generate fifty 2-dimensional inputs and 4-dimensional outputs.
x = np.random.randn(50, 2)
y = np.random.randn(50, 4)

data = {"X": x, "Y": y}

# Trend g(x) = [1, x1, x2] (intercept + linear effects), with 3 latent GPs.
model = MOOGP(terms=[None, 1, 2], q=3)
model.fit(data)

# Prediction: mean and standard deviation on the original output scale.
mean, std = model.predict(x, return_std=True)

rmse = evaluation.rmse(y, mean)
coverage, _ = evaluation.intervalstats(y, mean, std ** 2)
print(f"RMSE:       {rmse}")
print(f"Coverage:   {coverage}")
```

## The `MOOGP` Class

```python
MOOGP(terms, q, Psi=None, *,
      orthogonal=True, learn_Psi=False,
      sigma_eps2=None, learn_sigma_eps=None, diag_error_structure=None,
      standardize_x="unitcube", standardize_y="zscore")
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `terms` | — | Basis for the trend `g(x)` ([section](#specifying-the-trend-with-terms)). |
| `q` | — | Number of latent GPs (`q ≤ p`) ([section](#choosing-q)). |
| `orthogonal` | `True` | Orthogonalize the kernel against `g(x)`; `False` is a standard squared exponential kernel. |
| `sigma_eps2` | `None` | Fixed per-output noise variances, shape `(p,)` ([section](#measurement-noise-and-diag_error_structure)).  |
| `learn_sigma_eps` | auto | Learn the noise; defaults to `True` when `sigma_eps2` is not given. |
| `diag_error_structure` | `None` | Group outputs that share one noise variance ([section](#measurement-noise-and-diag_error_structure)). |
| `standardize_x` | `"unitcube"` | Map inputs to `[-1, 1]` internally ([section](#standardization)). |
| `standardize_y` | `"zscore"` | Center and scale outputs internally. |
| `learn_Psi` | `False` | Learn the mixing matrix instead of deriving it from the data ([section](#kernel-and-mixing-matrix-options)). |


## Advanced Usage

### Specifying the trend with `terms`

`terms` lists the columns of the trend `g(x)`. Each entry is `None` for an
intercept, an int `j` for the main effect of input `j`, or a tuple for an
interaction:

```python
terms = [None, 1, 2, 3]        # g(x) = [1, x1, x2, x3]
terms = [None, 1, 2, (1, 2)]   # g(x) = [1, x1, x2, x1 * x2]
```

### Choosing `q`

`q` is the number of latent GPs and must satisfy `q ≤ p`. It acts as a rank: the
latent basis is taken from the top `q` principal directions of `Y`, so inspecting
the singular value spectrum of `Y` helps pick it. Start small and increase until
predictive performance stops improving; `q = p` is full rank.

### Measurement noise and `diag_error_structure`

There is one variance term per output. By default it is learned, but you
can also fix it to known values:

```python
MOOGP(terms, q)                                # learn the noise (default)
MOOGP(terms, q, sigma_eps2=[10.0, 1.0, 0.05])  # fixed, known noise
```

Variances can also be shared across outputs. For example,
take six outputs where the first two are measured precisely and the remaining four
are noisy:

```python
import numpy as np

n = 100
x = np.linspace(0.0, 1.0, n).reshape(-1, 1)
Y = np.column_stack([
    np.sin(x), np.cos(x),                                  # low-noise outputs
    np.sin(x/2), np.cos(x/2), np.sin(x/3), np.cos(x/3),   # high-noise outputs
])
Y[:, :2] += np.random.normal(0, 1e-3, size=(n, 2))
Y[:, 2:] += np.random.normal(0, 1e-1, size=(n, 4))
```

Pass `diag_error_structure` as the list of group sizes (which must sum to `p`). The
following groups the first 2 and the remaining 4 outputs, fitting two noise
variances:

```python
model = MOOGP(terms=[None, 1], q=4, diag_error_structure=[2, 4])
model.fit({"X": x, "Y": Y})
```

### Standardization

By default the model rescales data internally:

- `standardize_x="unitcube"` maps each input to `[-1, 1]`
- `standardize_y="zscore"` centers/scales each output; `"robust"` uses median/MAD.

Set either to `False` if your data is already on the right scale. Predictions are
always returned on the original output scale.

### Optimizer control and tolerance

The default optimization parameters for LBFGS-B are shown below and can be tuned via `optimizer_opts`:

```python
model.fit(data, optimizer_opts={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6})
```

Start with a relatively low value for `maxiter` and increase if better performance is needed.

### Kernel and mixing-matrix options

- `orthogonal=True` (default) orthogonalizes the latent kernels against the trend
  so the GP residual carries no signal the trend already explains; `False` gives a
  conventional multi-output GP.
- The mixing matrix `Psi` is derived from the data by default. Pass a `(p, q)`
  array to fix it, or set `learn_Psi=True` to learn it during fitting:

```python
# Default: Psi is derived from data -- much faster with large n and p
model = MOOGP(terms=[None, 1, 2], q=3)
model.fit(data)

# Fix Psi to a known (p, q) mixing matrix (here p = 4 outputs, q = 3 latent GPs).
Psi = np.random.randn(4, 3)
model = MOOGP(terms=[None, 1, 2], q=3, Psi=Psi)
model.fit(data)

# Or learn Psi during fitting.
model = MOOGP(terms=[None, 1, 2], q=3, learn_Psi=True)
model.fit(data)
```