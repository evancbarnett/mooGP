import numpy as np
import time

from moogp.kernels import make_c_star_matrix
from moogp.model import build_cross_K, vecF, unvecF
from moogp.design import build_Gy, make_G
from scipy.linalg import solve
from moogp.forrester_illustration import fit_moogp_forrester
# 1. Paste your OLD predict function here, renamed:

def predict_old(
        self,
        Xstar,
        *,
        return_std=False,
        diag_only=True,
        include_mean_uncertainty=False,
        predict_observation=True,
    ):
        """
        Predict at new inputs Xstar (scaled in [-1,1]^d).

        Returns
        -------
        mean : (n*, p)
        std  : (n*, p) if return_std=True
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict().")

        cache = self.cache
        terms = cache["terms"]
        one_based = cache["one_based"]
        use_reml = self.use_reml

        X = cache["X"]
        Y = cache["Y"]
        Psi = cache["Psi"]
        sigma_eps2 = cache.get("sigma_eps2", None)
        lat_params = cache["lat_params"]
        Gy = cache["Gy"]
        bhat = cache["bhat"]
        A = cache["A"]
        p = Y.shape[1]
        Ky_inv_rvec = cache["Ky_inv_rvec"]

        Xs = np.asarray(Xstar)
        nstar = Xs.shape[0]

        # cross and test covariances per latent
        Cj_XsX = []
        Cj_XsXs = []
        for (sigma2_j, ell_j) in lat_params:
            Cj1 = make_c_star_matrix(Xs, X, ell=ell_j, sigma2=sigma2_j,
                                     terms=terms, one_based=one_based)
            Cj2 = make_c_star_matrix(Xs, Xs, ell=ell_j, sigma2=sigma2_j,
                                     terms=terms, one_based=one_based)
            Cj_XsX.append(Cj1)
            Cj_XsXs.append(Cj2)

        K_XsX = build_cross_K(Psi, Cj_XsX)    # (n* p × n p)
        K_XsXs = build_cross_K(Psi, Cj_XsXs)  # (n* p × n* p)

        # mean
        Gs = make_G({"X_scaled": Xs}, terms, one_based=one_based, return_names=False)
        Gs_y = build_Gy(Gs, p) if Gs.size else np.empty((nstar * p, 0))

        mean_vec = (Gs_y @ vecF(bhat)) if Gs.size else np.zeros(nstar * p)
        mean_vec += K_XsX @ Ky_inv_rvec
        mean = unvecF(mean_vec, nstar, p)

        if not return_std:
            return mean

        # variance (diag_only currently, like your code)
        V = self._solve_with_cached_Ky(K_XsX.T)  # (n p × n* p)
        diag_prior = np.einsum("ii->i", K_XsXs)
        diag_cross = np.sum(K_XsX * V.T, axis=1)
        diag = diag_prior - diag_cross

        # If predicting the observed output y(x)=f(x)+eps, add Sigma_eps ⊗ I_{n*}
        if predict_observation and (sigma_eps2 is not None):
            diag += np.repeat(np.asarray(sigma_eps2, float).ravel(), nstar)

        if include_mean_uncertainty and Gs.size and use_reml and (A is not None):
            M = Gs_y - K_XsX @ self._solve_with_cached_Ky(Gy)
            W = solve(A, M.T, assume_a="sym")
            diag += np.sum(M * W.T, axis=1)

        std = np.sqrt(np.maximum(diag, 0.0))
        std = unvecF(std, nstar, p)
        return mean, std


# 2. Setup a benchmarking function
def run_benchmark(model, X_test):
    print(f"--- Benchmarking with {X_test.shape[0]} test points ---")
    
    # Test 1: Mean Only (return_std=False)
    print("\n[TEST 1] Mean Only (return_std=False)")
    
    start = time.perf_counter()
    mean_old = predict_old(model, X_test, return_std=False)
    time_old_mean = time.perf_counter() - start
    print(f"Old Predict: {time_old_mean:.5f} seconds")
    
    start = time.perf_counter()
    mean_new = model.predict(X_test, return_std=False)
    time_new_mean = time.perf_counter() - start
    print(f"New Predict: {time_new_mean:.5f} seconds")
    
    speedup = time_old_mean / time_new_mean if time_new_mean > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x faster")
    assert np.allclose(mean_old, mean_new, rtol=1e-5, atol=1e-8), "Mismatch in Mean outputs!"

    # Test 2: Mean + Std (return_std=True)
    print("\n[TEST 2] Mean + Standard Deviation (return_std=True)")
    
    start = time.perf_counter()
    _, std_old = predict_old(model, X_test, return_std=True)
    time_old_std = time.perf_counter() - start
    print(f"Old Predict: {time_old_std:.5f} seconds")
    
    start = time.perf_counter()
    _, std_new = model.predict(X_test, return_std=True)
    time_new_std = time.perf_counter() - start
    print(f"New Predict: {time_new_std:.5f} seconds")
    
    speedup = time_old_std / time_new_std if time_new_std > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x faster")
    assert np.allclose(std_old, std_new, rtol=1e-5, atol=1e-8), "Mismatch in Std outputs!"

if __name__ == "__main__":
    start_train = time.perf_counter()
    model, X, X_scaled, Y = fit_moogp_forrester(n_train=100,
            seed=0,
            q=3,
            Psi=None,
            learn_Psi=False,
            use_fast=True)
    stop_train = time.perf_counter()

    print(f"Training time {stop_train-start_train}")

    n_test = 2000
    X_plot = np.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    X_plot_scaled = 2.0 * (X_plot - 0.5)  # same scaling as generate_forrester_data


    run_benchmark(model, X_plot_scaled)