"""Optimized Python SpaNorm benchmark."""

import numpy as np
import pandas as pd
import time
from scipy.special import gammaln, xlogy, xlog1py
from scipy.linalg import solve as sp_solve

DATA_DIR = "data"


def dnbinom_logpmf_vectorized(Y, mu, size):
    """Vectorized NB log-PMF. No gene loop.

    Parameters
    ----------
    Y : np.ndarray (ngenes, ncells)
    mu : np.ndarray (ngenes, ncells)
    size : np.ndarray (ngenes,) — 1/psi

    Returns
    -------
    np.ndarray (ngenes, ncells) log-PMF values
    """
    k = Y.astype(np.int64)
    n = size[:, np.newaxis]
    p = n / (n + mu)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return gammaln(k + n) - gammaln(k + 1) - gammaln(n) + xlogy(n, p) + xlog1py(k, -p)


def run_optimized(counts, W, idx, max_iter=30, tol=1e-4):
    """Optimized IRLS fitting."""
    ngenes, ncells = counts.shape
    Y_sub = counts[:, idx]
    W_sub = W[idx, :]
    nsub = len(idx)
    nW = W_sub.shape[1]

    # Initialize
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((ngenes, nW))
    alpha[:, 0] = 1.0

    # Dispersion (method of moments - vectorized)
    Y_mean = np.mean(Y_sub, axis=1)
    Y_var = np.var(Y_sub, axis=1, ddof=1)
    psi = np.maximum((Y_var - Y_mean) / np.maximum(Y_mean ** 2, 1e-10), 0.01)
    psi = np.maximum(psi, 1e-10)
    size = 1.0 / psi

    # Regularization (precompute)
    lam = np.full(nW - 1, 0.0001 * ncells)
    reg_diag = np.zeros(nW)
    reg_diag[1:] = lam

    # Precompute WtW components
    loglik_prev = -np.inf

    t0 = time.perf_counter()

    for iteration in range(max_iter):
        # Forward pass
        lmu = np.clip(gmean[:, None] + alpha @ W_sub.T, -50, 50)
        mu_hat = np.exp(lmu)

        # Loglik (vectorized, every iteration for convergence check)
        ll = np.sum(dnbinom_logpmf_vectorized(Y_sub, mu_hat, size))

        # Working response
        Z = lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)

        # Weights
        sig_inv = 1.0 / (psi[:, None] * np.exp(-lmu))
        sig_inv = np.clip(sig_inv, 0, 1e10)
        wt = np.minimum(np.mean(sig_inv, axis=0), np.quantile(sig_inv.mean(axis=0), 0.98))

        # Alpha update with SpaNorm constraint
        a1 = np.mean(alpha[:, 0])
        Wa1 = a1 * W_sub[:, 0:1]
        W_rest = W_sub[:, 1:]
        Z_adj = Z - gmean[:, None] - Wa1.T
        b_rest = (Z_adj * wt[None, :]) @ W_rest
        WtW_rest = W_rest.T @ (wt[:, None] * W_rest)

        # Solve with regularization
        alpha_rest = sp_solve(WtW_rest + np.diag(lam), b_rest.T, assume_a='pos').T
        alpha = np.column_stack([np.full(ngenes, a1), alpha_rest])

        # Update gmean
        Z_res = Z - alpha @ W_sub.T
        gmean = np.sum(Z_res * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        # Convergence check
        if iteration > 0 and abs(ll - loglik_prev) / max(abs(loglik_prev), 1) < tol:
            break
        loglik_prev = ll

    elapsed = time.perf_counter() - t0

    return gmean, alpha, psi, elapsed, iteration + 1


def main():
    # Load data
    counts = pd.read_csv(f'{DATA_DIR}/large_counts.csv').values.astype(np.float64)
    r_W = pd.read_csv(f'{DATA_DIR}/large_W.csv').values
    r_idx = pd.read_csv(f'{DATA_DIR}/large_idx.csv').values.ravel() - 1
    r_lc = pd.read_csv(f'{DATA_DIR}/large_r_logcounts.csv').values
    with open(f'{DATA_DIR}/large_r_time.txt') as f:
        r_time = float(f.read().strip())

    ngenes, ncells = counts.shape
    print(f"Data: {ngenes} genes x {ncells} cells")
    print(f"R time: {r_time:.3f}s")

    # Warmup
    run_optimized(counts, r_W, r_idx, max_iter=2, tol=1e10)

    # Benchmark (5 runs)
    times = []
    for i in range(5):
        gmean, alpha, psi, elapsed, iters = run_optimized(counts, r_W, r_idx, max_iter=30, tol=1e-4)
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f}s ({iters} iters)")

    avg_time = np.mean(times)
    std_time = np.std(times)

    # Normalize
    W = r_W
    n_bio = 4  # from R output
    is_bio = np.array([False] + [True]*n_bio + [False]*(W.shape[1]-1-n_bio))
    lmu_bio = np.clip(gmean[:, None] + alpha[:, is_bio] @ W[:, is_bio].T, -50, 50)
    py_lc = np.log2(np.exp(lmu_bio))

    # Correlation
    corr = np.corrcoef(r_lc.ravel(), py_lc.ravel())[0, 1]

    # Per-gene
    gene_corrs = np.zeros(ngenes)
    for g in range(ngenes):
        if np.std(r_lc[g]) > 0 and np.std(py_lc[g]) > 0:
            gene_corrs[g] = np.corrcoef(r_lc[g], py_lc[g])[0, 1]

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Correlation:       {corr:.6f}")
    print(f"Per-gene mean:     {np.mean(gene_corrs):.6f}")
    print(f"Per-gene min:      {np.min(gene_corrs):.6f}")
    print(f"\nPython avg:        {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"R time:            {r_time:.3f}s")
    print(f"Speedup:           {r_time/avg_time:.2f}x")

    if corr > 0.96:
        print(f"\n[PASS] correlation {corr:.6f} > 0.96")
    else:
        print(f"\n[FAIL] correlation {corr:.6f} <= 0.96")

    if avg_time < r_time:
        print(f"[PASS] Python {avg_time:.3f}s < R {r_time:.3f}s")
    else:
        print(f"[FAIL] Python {avg_time:.3f}s >= R {r_time:.3f}s")


if __name__ == "__main__":
    main()
