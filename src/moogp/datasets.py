# moogp/datasets.py

import numpy as np
from numpy import pi, exp
from scipy.stats import qmc

def tstd2theta(tstd, hard=True):
        """Given standardized theta in [0, 1]^d, return non-standardized theta."""
        if tstd.ndim < 1.5:
            tstd = tstd[:, None].T
        (Treffs, Hus, LdKw, powparams) = np.split(tstd, tstd.shape[1], axis=1)

        Treff = (0.5 - 0.05) * Treffs + 0.05
        Hu = Hus * (1110 - 990) + 990
        if hard:
            Ld_Kw = LdKw * (1680 / 1500 - 1120 / 15000) + 1120 / 15000
        else:
            Ld_Kw = LdKw * (1680 / 9855 - 1120 / 12045) + 1120 / 12045

        powparam = powparams * (0.5 - (- 0.5)) + (-0.5)

        theta = np.hstack((Hu, Ld_Kw, Treff, powparam))
        return theta

def xstd2x(xstd):
    """Given standardized x in [0, 1]^2 x {0, 1}, return non-standardized x."""
    if xstd.ndim < 1.5:
        xstd = xstd[:, None].T
    (rws, Hls) = np.split(xstd, xstd.shape[1], axis=1)

    rw = rws * (np.log(0.5) - np.log(0.05)) + np.log(0.05)
    rw = np.exp(rw)
    Hl = Hls * (820 - 700) + 700

    x = np.hstack((rw, Hl))
    return x


def generate_borehole_data_nd(n,p=4, seed=67):

    def borehole_vec(x, theta):
        """Given x and theta, return vector of values."""
        (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
        (rw, Hl) = np.split(x, x.shape[1], axis=1)
        numer = 2 * np.pi * (Hu - Hl)
        denom1 = 2 * Ld_Kw / rw ** 2
        denom2 = Treff

        f = ((numer / ((denom1 + denom2))) * np.exp(powparam * rw)).reshape(-1)
        return f
    
    x_unit = qmc.LatinHypercube(d=2, scramble=True, seed=seed).random(p)
    theta_unit = qmc.LatinHypercube(d=4, scramble=True,seed=seed).random(n)
    
    
    x = xstd2x(x_unit)                  # (p,2)
    theta = tstd2theta(theta_unit)      # (n,4)

    theta_stacked = np.repeat(theta, repeats=p, axis=0)
    x_stacked = np.tile(x.astype(float), (n, 1))

    def to_m1_1(X):
        return 2.0 * (X - 0.5)
    
    theta_scaled = to_m1_1(theta_unit)

    y = borehole_vec(x_stacked, theta_stacked).reshape((n, p))
    return {"X_phys": theta, "X_scaled": theta_scaled, "Y": y, 'locations_phys': x}
        

def generate_forrester_data(n, seed=67, with_error=False, error_per_output=None, X_override=None):
    BOUNDS = np.array([[0,1]])
    rng = np.random.default_rng(seed)

    def f_1(x):
        return ((6*x-2) ** 2) * np.sin(12*x-4)
    def f_2(x):
        return 0.5 * f_1(x) + 5 * (x - 0.5) + 5
    def f_3(x):
        return -0.8 * f_1(x) - 5 * (x - 0.5) - 4

    def lhs_in_bounds(n, bounds=BOUNDS, seed=seed):
        d = bounds.shape[0]
        sampler = qmc.LatinHypercube(d=d, scramble=True, seed=seed)
        U = sampler.random(n)
        return qmc.scale(U, bounds[:, 0], bounds[:, 1])

    def to_m1_1(X, bounds=BOUNDS):
        a, b = bounds[:, 0], bounds[:, 1]
        return 2.0 * (X - a) / (b - a) - 1.0

    # allow X override
    X = lhs_in_bounds(n, seed=seed) if X_override is None else np.asarray(X_override).reshape(-1, 1)

    y = np.array([f_1(X), f_2(X), f_3(X)]).squeeze().T
    f = y.copy()

    if with_error:
        if error_per_output is None:
            raise ValueError(f"Specify error_per_output if with_error = True")
        else:
            error_per_output = np.asarray(error_per_output, dtype=float).ravel()
            if error_per_output.size == 1:
                error_per_output = np.repeat(error_per_output, y.shape[1])
            if error_per_output.size != y.shape[1]:
                raise ValueError(
                    f"error_per_output length {error_per_output.size} must match the "
                    f"number of outputs {y.shape[1]}."
                )
            y += rng.normal(0.0, np.sqrt(error_per_output), size=y.shape)

    X_scaled = to_m1_1(X)
    return {"X": X, "X_scaled": X_scaled, "y": y, "f": f}

def generate_borehold_data_1d(n, seed=67):
    # Borehole function: 8 x_j variables
    BOUNDS = np.array([
        [0.05, 0.15],
        [100.0, 5_000.0],
        [63_070., 115_600.0],
        [990.0, 1_110.0],
        [63.1, 116.0],
        [700.0, 820.0],
        [1_120.0, 1680.0],
        [9_855.0, 12_045.0],
    ])

    def borehole_y(X_phys):
        x1, x2, x3, x4, x5, x6, x7, x8 = X_phys.T
        log_term = np.log(x2 / x1)
        denom = log_term * (1.0 + (2.0 * x7 * x3) / (log_term * x1**2 * x8) + (x3 / x5))
        return (2.0 * np.pi * x3 * (x4 - x6)) / denom
    


    def lhs_in_bounds(n, bounds=BOUNDS, seed=seed):
        d = bounds.shape[0]
        sampler = qmc.LatinHypercube(d=d, scramble=True, seed=seed)
        U = sampler.random(n)                                  # in [0,1]^d
        return qmc.scale(U, bounds[:, 0], bounds[:, 1]) 
    
    def to_m1_1(X_phys, bounds=BOUNDS):
        a, b = bounds[:, 0], bounds[:, 1]
        return 2.0 * (X_phys - a) / (b - a) - 1.0

    
    X_phys = lhs_in_bounds(n, seed=seed)
    y = borehole_y(X_phys)
    X_scaled = to_m1_1(X_phys)
    out = {"X_phys": X_phys, "X_scaled": X_scaled, "y": y}
    
    return out

def borehole_vec_physical(x, theta):
    """
    Compare to ground truth at the same x, theta values
    """
    (Hu, Ld_Kw, Treff, powparam) = np.split(theta, theta.shape[1], axis=1)
    (rw, Hl) = np.split(x, x.shape[1], axis=1)
    numer = 2*np.pi*(Hu - Hl)
    denom1 = 2*Ld_Kw/(rw**2)
    denom2 = Treff
    return ((numer/(denom1 + denom2)) * np.exp(powparam * rw)).reshape(-1)


def log_lhs_1d_rescaled(n, seed, xmin=1e-3, cluster="right", include_x0=False, shuffle=False):
    """
    Log-LHS then rescale to [0,1].
    cluster="right": concentrates near x≈1 (missing early x)
    cluster="left":  concentrates near x≈0 (missing late x)

    include_x0=True: replaces one sample with exactly x=0 to improve left-edge coverage.
    shuffle=True: randomize point order (optional).
    """
    rng = np.random.default_rng(seed)

    n_gen = n - 1 if include_x0 else n

    # LHS on [0,1] for n_gen points
    u = rng.random(n_gen)
    perm = rng.permutation(n_gen)
    t = (perm + u) / n_gen  # stratified in [0,1]

    # log-uniform r in [xmin, 1]
    r = np.exp(np.log(xmin) + t * (0.0 - np.log(xmin)))  # many near xmin

    if cluster == "left":
        x_raw = r
        x = (x_raw - xmin) / (1.0 - xmin)
    elif cluster == "right":
        x_raw = 1.0 - r
        x = x_raw / (1.0 - xmin)
    else:
        raise ValueError("cluster must be 'left' or 'right'")

    x = x.reshape(-1, 1)

    if include_x0:
        x = np.vstack([x, np.array([[0.0]])])

    if shuffle:
        idx = rng.permutation(x.shape[0])
        x = x[idx]

    return x
