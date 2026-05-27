"""Full Visium SP1 comparison: R vs Python with proper dispersion estimation."""

import numpy as np
import pandas as pd
import time
import gzip
import zipfile
from scipy.io import mmread
from scipy.linalg import solve as sp_solve
from scipy.special import gammaln, xlogy, xlog1py, digamma, polygamma
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spanorm.spline import bs_tps

DATA_DIR = 'D:/桌面/myproject/data/GSE211956_RAW'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_visium():
    """Load Visium SP1 data."""
    with gzip.open(f'{DATA_DIR}/GSM6506110_SP1_matrix.mtx.gz', 'rt') as f:
        counts_sp = mmread(f).tocsr()
    features = pd.read_csv(f'{DATA_DIR}/GSM6506110_SP1_features.tsv.gz', sep='\t', header=None)
    barcodes = pd.read_csv(f'{DATA_DIR}/GSM6506110_SP1_barcodes.tsv.gz', sep='\t', header=None)
    with zipfile.ZipFile(f'{DATA_DIR}/GSM6506110_SP1_spatial.zip') as z:
        spatial = pd.read_csv(z.open('spatial/tissue_positions_list.csv'), header=None)
    spatial_dict = dict(zip(spatial[0].values, spatial.values))
    coords = np.zeros((len(barcodes), 2))
    for i, bc in enumerate(barcodes[0].values):
        if bc in spatial_dict:
            row = spatial_dict[bc]
            coords[i] = [float(row[4]), float(row[5])]
    valid = coords[:, 0] > 0
    counts_sp = counts_sp[:, valid]; coords = coords[valid]
    n_spots = counts_sp.shape[1]
    gene_expr = np.array((counts_sp > 0).sum(axis=1)).ravel() / n_spots
    keep = gene_expr >= 0.1
    counts_sp = counts_sp[keep]
    gene_mean = np.array(counts_sp.mean(axis=1)).ravel()
    gene_var = np.array(counts_sp.power(2).mean(axis=1)).ravel() - gene_mean**2
    top_idx = np.argsort(gene_var)[-300:]
    counts = counts_sp[top_idx].toarray().astype(np.float64)
    return counts, coords


def estimate_dispersion_edgeR(Y, mu, design=None):
    """Estimate dispersion using edgeR-like Cox-Reid approach.

    This approximates edgeR's estimateDisp by:
    1. Computing tagwise dispersion via Cox-Reid adjusted profile likelihood
    2. Shrinking toward a common dispersion (empirical Bayes)
    """
    ngenes, ncells = Y.shape

    # Step 1: Common dispersion (moment estimate)
    mu_flat = mu.ravel()
    y_flat = Y.ravel()
    var_flat = (y_flat - mu_flat)**2
    common_psi = max(np.mean(var_flat / np.maximum(mu_flat**2, 1e-10)) - 1.0/np.mean(mu_flat), 1e-4)

    # Step 2: Tagwise dispersion via Cox-Reid adjusted profile likelihood
    psi_tagwise = np.zeros(ngenes)

    for g in range(ngenes):
        y_g = Y[g]
        mu_g = mu[g]
        mean_g = np.mean(mu_g)

        # Initial estimate (method of moments)
        var_g = np.var(y_g, ddof=1)
        psi_init = max((var_g - mean_g) / max(mean_g**2, 1e-10), 1e-4)

        # Cox-Reid adjusted log-likelihood
        def cr_loglik(log_psi):
            psi = np.exp(log_psi)
            size = 1.0 / psi

            # NB log-likelihood
            p = size / (size + mu_g)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            ll = np.sum(gammaln(y_g + size) - gammaln(y_g + 1) - gammaln(size)
                       + xlogy(size, p) + xlog1py(y_g, -p))

            # Cox-Reid adjustment
            h = mu_g / (mu_g + size)
            adj = 0.5 * np.log(np.sum(h * (1 - h)) + 1e-10)

            return ll - adj

        # Optimize in log-space
        log_psi_init = np.log(psi_init)
        try:
            result = minimize_scalar(lambda x: -cr_loglik(x),
                                    bounds=(log_psi_init - 3, log_psi_init + 3),
                                    method='bounded')
            psi_tagwise[g] = np.exp(result.x)
        except Exception:
            psi_tagwise[g] = psi_init

    # Step 3: Empirical Bayes shrinkage toward common dispersion
    # Prior weight (simplified from edgeR's prior.n)
    prior_n = 10
    psi_shrunk = (prior_n * common_psi + ncells * psi_tagwise) / (prior_n + ncells)

    return np.maximum(psi_shrunk, 1e-6)


def fit_spanorm_nb(Y, W, idx, psi_init=None, max_iter=30, tol=1e-4):
    """Fit NB model with proper dispersion estimation."""
    ngenes, ncells_full = Y.shape
    Y_sub = Y[:, idx]
    W_sub = W[idx, :]
    nsub = Y_sub.shape[1]
    nW = W_sub.shape[1]

    # Initialize
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((ngenes, nW))
    alpha[:, 0] = 1.0

    # Initial dispersion
    if psi_init is None:
        mu_init = np.exp(np.clip(gmean[:, None] + alpha @ W_sub.T, -50, 50))
        psi = estimate_dispersion_edgeR(Y_sub, mu_init)
    else:
        psi = psi_init.copy()

    size = 1.0 / psi
    loglik_prev = -np.inf

    for iteration in range(max_iter):
        lmu = np.clip(gmean[:, None] + alpha @ W_sub.T, -50, 50)
        mu_hat = np.exp(lmu)

        # Log-likelihood
        k = Y_sub.astype(np.int64)
        n = size[:, None]
        p = np.clip(n / (n + mu_hat), 1e-15, 1 - 1e-15)
        ll = np.sum(gammaln(k + n) - gammaln(k + 1) - gammaln(n) + xlogy(n, p) + xlog1py(k, -p))

        # Working response
        Z = lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)

        # Weights
        sig_inv = 1.0 / (psi[:, None] * np.exp(-lmu))
        sig_inv = np.clip(sig_inv, 0, 1e10)
        wt = np.mean(sig_inv, axis=0)
        wt = np.minimum(wt, np.quantile(wt, 0.98))

        # Alpha update with SpaNorm constraint
        a1 = np.mean(alpha[:, 0])
        Wa1 = a1 * W_sub[:, 0:1]
        W_rest = W_sub[:, 1:]
        Z_adj = Z - gmean[:, None] - Wa1.T
        b_rest = (Z_adj * wt[None, :]) @ W_rest
        WtW_rest = W_rest.T @ (wt[:, None] * W_rest)

        lam = np.full(W_rest.shape[1], 0.0001 * ncells_full)
        alpha_rest = sp_solve(WtW_rest + np.diag(lam), b_rest.T, assume_a='pos').T
        alpha = np.column_stack([np.full(ngenes, a1), alpha_rest])

        # Update gmean
        Z_res = Z - alpha @ W_sub.T
        gmean = np.sum(Z_res * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        # Update dispersion
        mu_new = np.exp(np.clip(gmean[:, None] + alpha @ W_sub.T, -50, 50))
        psi = estimate_dispersion_edgeR(Y_sub, mu_new)
        size = 1.0 / psi

        print(f"  iter {iteration+1}: loglik = {ll:.2f}")

        if iteration > 0 and abs(ll - loglik_prev) / max(abs(loglik_prev), 1) < tol:
            print("  Converged!")
            break
        loglik_prev = ll

    return gmean, alpha, psi, iteration + 1


def normalize_mean_bio(gmean, alpha, W, wtype):
    """Mean-bio normalization with winsorization."""
    is_bio = wtype == 'biology'
    lmu = np.clip(gmean[:, None] + alpha[:, is_bio] @ W[:, is_bio].T, -50, 50)
    lmu_median = np.median(lmu, axis=1)
    lmu_mad = median_abs_deviation(lmu, axis=1, nan_policy='omit')
    lmu_mad = np.where(np.isnan(lmu_mad) | (lmu_mad == 0), 1.0, lmu_mad)
    lmu_max = lmu_median + 4 * lmu_mad
    lmu = np.minimum(lmu, lmu_max[:, None])
    return np.log2(np.exp(lmu))


def main():
    print("=" * 60)
    print("Full Visium SP1 R vs Python Comparison")
    print("=" * 60)

    # Load data
    print("\nLoading Visium SP1...")
    counts, coords = load_visium()
    ngenes, ncells = counts.shape
    print(f"Data: {ngenes} genes x {ncells} spots")

    # Build spline bases
    coords_scaled = np.zeros_like(coords)
    for i in range(2):
        col = coords[:, i]; rng = col.max() - col.min()
        coords_scaled[:, i] = (col - col.min()) / rng - 0.5

    bs_xy_bio, (df_bio_x, df_bio_y) = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], 3)
    df_ls_x = max(df_bio_x // 2, 1); df_ls_y = max(df_bio_y // 2, 1)
    bs_xy_ls, _ = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], max(df_ls_x, df_ls_y))

    LS = counts.sum(axis=0); LS = LS / LS.mean()
    logLS = np.log(np.maximum(1e-8, LS))
    W = np.hstack([logLS.reshape(-1, 1), bs_xy_bio, logLS.reshape(-1, 1) * bs_xy_ls])
    n_bio = bs_xy_bio.shape[1]; n_ls = bs_xy_ls.shape[1]
    wtype = np.empty(W.shape[1], dtype='<U10')
    wtype[0] = 'ls'; wtype[1:1+n_bio] = 'biology'; wtype[1+n_bio:] = 'ls'
    print(f"W: {W.shape} (bio={n_bio}, ls={n_ls})")

    # Sample
    nsub = int(round(0.25 * ncells))
    rng = np.random.default_rng(42)
    idx = rng.choice(ncells, size=nsub, replace=False)
    print(f"Sampled {nsub} spots")

    # Fit model
    print("\nFitting Python SpaNorm...")
    t0 = time.perf_counter()
    gmean, alpha, psi, iters = fit_spanorm_nb(counts, W, idx, max_iter=30, tol=1e-4)
    py_fit_time = time.perf_counter() - t0
    print(f"Python fitting time: {py_fit_time:.3f}s ({iters} iterations)")

    # Normalize
    print("Normalizing...")
    t0 = time.perf_counter()
    py_lc = normalize_mean_bio(gmean, alpha, W, wtype)
    py_norm_time = time.perf_counter() - t0
    print(f"Python normalization time: {py_norm_time*1000:.2f}ms")
    print(f"Python logcounts range: [{py_lc.min():.4f}, {py_lc.max():.4f}]")

    # Save Python results
    pd.DataFrame(py_lc).to_csv(f'{OUT_DIR}/visium_py_logcounts.csv', index=False)

    # Load R results
    print("\nLoading R reference...")
    r_lc = pd.read_csv(f'{OUT_DIR}/visium_r_logcounts.csv', header=0).values
    with open(f'{OUT_DIR}/visium_r_time.txt') as f:
        r_time = float(f.read().strip())
    print(f"R logcounts range: [{r_lc.min():.4f}, {r_lc.max():.4f}]")
    print(f"R time: {r_time:.3f}s")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    corr = np.corrcoef(r_lc.ravel(), py_lc.ravel())[0, 1]
    print(f"\nOverall correlation: {corr:.6f}")

    # Per-gene correlation
    gene_corrs = np.zeros(ngenes)
    for g in range(ngenes):
        if np.std(r_lc[g]) > 0 and np.std(py_lc[g]) > 0:
            gene_corrs[g] = np.corrcoef(r_lc[g], py_lc[g])[0, 1]
    print(f"Per-gene mean: {np.mean(gene_corrs):.6f}")
    print(f"Per-gene min: {np.min(gene_corrs):.6f}")
    print(f"Per-gene median: {np.median(gene_corrs):.6f}")

    print(f"\nSpeed:")
    print(f"  R full pipeline: {r_time:.3f}s")
    print(f"  Python fitting: {py_fit_time:.3f}s")
    print(f"  Python normalization: {py_norm_time*1000:.2f}ms")
    print(f"  Python total: {py_fit_time + py_norm_time:.3f}s")
    print(f"  Speedup: {r_time/(py_fit_time + py_norm_time):.1f}x")

    # Gate
    print(f"\n{'='*60}")
    print("GATE CHECK")
    print(f"{'='*60}")
    if corr > 0.96:
        print(f"  [PASS] Correlation {corr:.6f} > 0.96")
    else:
        print(f"  [FAIL] Correlation {corr:.6f} <= 0.96")

    total_py = py_fit_time + py_norm_time
    if total_py < r_time:
        print(f"  [PASS] Python {total_py:.3f}s < R {r_time:.3f}s")
    else:
        print(f"  [FAIL] Python {total_py:.3f}s >= R {r_time:.3f}s")


if __name__ == "__main__":
    main()
