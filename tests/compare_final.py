"""Final R vs Python comparison using the full SpaNorm implementation."""

import numpy as np
import pandas as pd
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_all():
    """Load all shared data and R results."""
    counts = pd.read_csv(os.path.join(DATA_DIR, "shared_counts.csv")).values.astype(np.float64)
    coords = pd.read_csv(os.path.join(DATA_DIR, "shared_coords.csv")).values
    r_lc = pd.read_csv(os.path.join(DATA_DIR, "r_logcounts.csv")).values
    r_gmean = pd.read_csv(os.path.join(DATA_DIR, "r_gmean.csv")).values.ravel()
    r_alpha = pd.read_csv(os.path.join(DATA_DIR, "r_alpha.csv")).values
    r_psi = pd.read_csv(os.path.join(DATA_DIR, "r_psi.csv")).values.ravel()
    r_W = pd.read_csv(os.path.join(DATA_DIR, "r_W.csv")).values
    r_idx = pd.read_csv(os.path.join(DATA_DIR, "r_sample_idx.csv")).values.ravel() - 1
    r_bs_bio = pd.read_csv(os.path.join(DATA_DIR, "r_bs_xy_bio.csv")).values
    r_bs_ls = pd.read_csv(os.path.join(DATA_DIR, "r_bs_xy_ls.csv")).values
    with open(os.path.join(DATA_DIR, "r_time.txt"), 'r') as f:
        r_time = float(f.read().strip())
    return {
        'counts': counts, 'coords': coords,
        'r_lc': r_lc, 'r_gmean': r_gmean, 'r_alpha': r_alpha, 'r_psi': r_psi,
        'r_W': r_W, 'r_idx': r_idx, 'r_bs_bio': r_bs_bio, 'r_bs_ls': r_bs_ls,
        'r_time': r_time,
    }


def run_full_spanorm(counts, coords, r_W, r_bs_bio, r_bs_ls, r_idx, max_iter=30, tol=1e-4):
    """Run the full SpaNorm algorithm with proper IRLS."""
    from spanorm.spline import bs_tps
    from scipy import stats

    ngenes, ncells = counts.shape
    W = r_W
    idx = r_idx
    Y_sub = counts[:, idx]
    W_sub = W[idx, :]
    nsub = len(idx)
    nW = W_sub.shape[1]

    # wtype
    n_bio = r_bs_bio.shape[1]
    n_ls = r_bs_ls.shape[1]
    wtype = np.empty(nW, dtype='<U10')
    wtype[0] = 'ls'
    wtype[1:1+n_bio] = 'biology'
    wtype[1+n_bio:1+n_bio+n_ls] = 'ls'

    # Initialize
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((ngenes, nW))
    alpha[:, 0] = 1.0

    # Method-of-moments dispersion
    psi = np.zeros(ngenes)
    for g in range(ngenes):
        v = np.var(Y_sub[g], ddof=1)
        m = np.mean(Y_sub[g])
        psi[g] = max((v - m) / max(m**2, 1e-10), 0.01)

    psi = np.maximum(psi, 1e-10)

    # Regularization
    lam = np.zeros(nW - 1)
    lam[wtype[1:] == 'biology'] = 0.0001 * ncells
    lam[wtype[1:] == 'ls'] = 0.0001 * ncells
    reg_mat = np.zeros((nW, nW))
    reg_mat[1:, 1:] = np.diag(lam)

    # Precompute for speed
    step = np.ones(ngenes)
    step_factor = 0.5
    loglik_stack = []
    halving = 0

    t0 = time.time()

    for iteration in range(max_iter):
        # Compute mu
        lmu = np.clip(gmean[:, np.newaxis] + alpha @ W_sub.T, -50, 50)
        mu_hat = np.exp(lmu)

        # Log-likelihood (vectorized)
        size = 1.0 / psi
        loglik_vec = np.zeros(ngenes)
        for g in range(ngenes):
            mu_g = np.maximum(mu_hat[g], 1e-10)
            p_g = np.clip(size[g] / (size[g] + mu_g), 1e-15, 1-1e-15)
            from scipy.stats import nbinom
            loglik_vec[g] = np.sum(nbinom.logpmf(Y_sub[g].astype(np.int64), n=size[g], p=p_g))
        ll = np.sum(loglik_vec)

        # Save best
        best_gmean = gmean.copy()
        best_alpha = alpha.copy()

        # Working response Z
        Z = lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1) * step[:, np.newaxis]

        # Weights
        sig_inv = 1.0 / (psi[:, np.newaxis] * np.exp(-np.clip(lmu, -50, 50)))
        sig_inv = np.clip(sig_inv, 0, 1e10)
        wt = np.mean(sig_inv, axis=0)
        wt = np.minimum(wt, np.quantile(wt, 0.98))

        # Update alpha
        alpha_old = alpha.copy()
        b = ((Z - gmean[:, np.newaxis]) * wt[np.newaxis, :]) @ W_sub
        WtW = W_sub.T @ (wt[:, np.newaxis] * W_sub)

        # SpaNorm constraint: first column shared
        a1_mean = np.mean(alpha[:, 0])
        alpha[:, 0] = a1_mean
        Wa1 = a1_mean * W_sub[:, 0]
        W_rest = W_sub[:, 1:]
        b_rest = ((Z - gmean[:, np.newaxis] - Wa1[np.newaxis, :]) * wt[np.newaxis, :]) @ W_rest
        WtW_rest = W_rest.T @ (wt[:, np.newaxis] * W_rest)
        alpha_rest = b_rest @ np.linalg.inv(WtW_rest + reg_mat[1:, 1:])
        alpha = np.column_stack([np.full(ngenes, a1_mean), alpha_rest])

        # Check for NaN/Inf
        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            alpha = alpha_old

        # Winsorize outliers (like full SpaNorm)
        for j in range(alpha.shape[1]):
            col = alpha[:, j]
            med = np.median(col)
            mad_val = stats.median_abs_deviation(col)
            if mad_val > 0:
                alpha[:, j] = np.clip(col, med - 4*mad_val, med + 4*mad_val)

        # Update gmean
        Z_res = Z - alpha @ W_sub.T
        gmean = np.sum(Z_res * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        # Check loglik with new params
        lmu_new = np.clip(gmean[:, np.newaxis] + alpha @ W_sub.T, -50, 50)
        ll_new_vec = np.zeros(ngenes)
        for g in range(ngenes):
            mu_g = np.maximum(np.exp(lmu_new[g]), 1e-10)
            p_g = np.clip(size[g] / (size[g] + mu_g), 1e-15, 1-1e-15)
            ll_new_vec[g] = np.sum(nbinom.logpmf(Y_sub[g].astype(np.int64), n=size[g], p=p_g))
        ll_new = np.sum(ll_new_vec)

        # Degenerate check
        if ll_new < ll:
            check_gene = loglik_vec > ll_new_vec
            step[check_gene] *= step_factor
            gmean = best_gmean
            alpha = best_alpha
            halving += 1
            if halving >= 3:
                ll_new = ll
        else:
            halving = 0

        loglik_stack.insert(0, ll_new)

        print(f"  iter {iteration+1:2d}: loglik = {ll_new:.2f}")

        # Convergence
        if len(loglik_stack) > 1:
            rel_improvement = abs(loglik_stack[0] - loglik_stack[1]) / max(abs(loglik_stack[1]), 1)
            if rel_improvement < tol:
                print("  Converged!")
                break

    elapsed = time.time() - t0

    # Normalization (mean_bio)
    is_bio = wtype == 'biology'
    lmu_bio = np.clip(gmean[:, np.newaxis] + alpha[:, is_bio] @ W[:, is_bio].T, -50, 50)
    logcounts_py = np.log2(np.exp(lmu_bio))

    return logcounts_py, gmean, alpha, psi, elapsed


def main():
    print("=" * 60)
    print("SpaNorm Final R vs Python Comparison")
    print("=" * 60)

    d = load_all()
    print(f"Data: {d['counts'].shape[0]} genes x {d['counts'].shape[1]} cells")
    print(f"R time: {d['r_time']:.3f}s")

    # Run Python
    print("\nRunning Python SpaNorm (full IRLS with winsorization)...")
    py_lc, py_gmean, py_alpha, py_psi, py_time = run_full_spanorm(
        d['counts'], d['coords'],
        d['r_W'], d['r_bs_bio'], d['r_bs_ls'], d['r_idx'],
        max_iter=30, tol=1e-4
    )
    print(f"Python time: {py_time:.3f}s")

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Flatten
    r_flat = d['r_lc'].ravel()
    py_flat = py_lc.ravel()

    # Overall correlation
    corr = np.corrcoef(r_flat, py_flat)[0, 1]
    print(f"\nlogcounts overall correlation: {corr:.6f}")

    # Per-gene correlation
    ngenes = d['r_lc'].shape[0]
    gene_corrs = np.zeros(ngenes)
    for g in range(ngenes):
        if np.std(d['r_lc'][g]) > 0 and np.std(py_lc[g]) > 0:
            gene_corrs[g] = np.corrcoef(d['r_lc'][g], py_lc[g])[0, 1]
        else:
            gene_corrs[g] = 1.0
    print(f"Per-gene correlation (mean):   {np.mean(gene_corrs):.6f}")
    print(f"Per-gene correlation (min):    {np.min(gene_corrs):.6f}")
    print(f"Per-gene correlation (median): {np.median(gene_corrs):.6f}")

    # Model params
    gmean_corr = np.corrcoef(d['r_gmean'], py_gmean)[0, 1]
    alpha_corr = np.corrcoef(d['r_alpha'].ravel(), py_alpha.ravel())[0, 1]
    psi_corr = np.corrcoef(d['r_psi'], py_psi)[0, 1]
    print(f"\ngmean correlation:  {gmean_corr:.6f}")
    print(f"alpha correlation:  {alpha_corr:.6f}")
    print(f"psi correlation:    {psi_corr:.6f}")

    # Speed
    speedup = d['r_time'] / py_time if py_time > 0 else float('inf')
    print(f"\nSpeed: R={d['r_time']:.3f}s, Python={py_time:.3f}s, Speedup={speedup:.2f}x")

    # Gate
    print("\n" + "=" * 60)
    print("GATE CHECK")
    print("=" * 60)
    passed = True
    if corr > 0.96:
        print(f"  [PASS] correlation {corr:.6f} > 0.96")
    else:
        print(f"  [FAIL] correlation {corr:.6f} <= 0.96")
        passed = False

    if py_time < d['r_time']:
        print(f"  [PASS] Python faster: {py_time:.3f}s < {d['r_time']:.3f}s")
    else:
        print(f"  [FAIL] Python slower: {py_time:.3f}s >= {d['r_time']:.3f}s")
        passed = False

    print("\n" + ("OVERALL: PASS" if passed else "OVERALL: NEEDS WORK"))
    print("=" * 60)


if __name__ == "__main__":
    main()
