"""Real 10x Visium SP1 data test: R vs Python comparison."""

import numpy as np
import pandas as pd
import gzip
import zipfile
import time
import sys, os
from scipy.io import mmread
from scipy import sparse
from scipy.linalg import solve as sp_solve
from scipy.special import gammaln, xlogy, xlog1py
from scipy.stats import median_abs_deviation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = 'D:/桌面/myproject/data/GSE211956_RAW'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_visium_sp1():
    """Load 10x Visium SP1 data."""
    # Count matrix
    with gzip.open(f'{DATA_DIR}/GSM6506110_SP1_matrix.mtx.gz', 'rt') as f:
        counts = mmread(f)
    counts_dense = np.asarray(counts.todense()).astype(np.float64)

    # Gene names
    features = pd.read_csv(f'{DATA_DIR}/GSM6506110_SP1_features.tsv.gz', sep='\t', header=None)
    gene_names = features[1].values

    # Barcodes
    barcodes = pd.read_csv(f'{DATA_DIR}/GSM6506110_SP1_barcodes.tsv.gz', sep='\t', header=None)
    barcode_names = barcodes[0].values

    # Spatial coordinates
    with zipfile.ZipFile(f'{DATA_DIR}/GSM6506110_SP1_spatial.zip') as z:
        spatial = pd.read_csv(z.open('spatial/tissue_positions_list.csv'), header=None)
    spatial_dict = dict(zip(spatial[0].values, spatial.values))

    coords = np.zeros((len(barcode_names), 2))
    for i, bc in enumerate(barcode_names):
        if bc in spatial_dict:
            row = spatial_dict[bc]
            coords[i] = [float(row[4]), float(row[5])]

    # Filter valid spots
    valid = coords[:, 0] > 0
    counts_filtered = counts_dense[:, valid]
    coords_filtered = coords[valid]

    # Filter lowly expressed genes
    gene_expr = np.mean(counts_filtered > 0, axis=1)
    keep_genes = gene_expr >= 0.1
    counts_final = counts_filtered[keep_genes]
    gene_names_final = gene_names[keep_genes]

    return counts_final, coords_filtered, gene_names_final


def run_python_spanorm(counts, coords, max_iter=30, tol=1e-4):
    """Run Python SpaNorm on real data."""
    from spanorm.spline import bs_tps

    ngenes, ncells = counts.shape

    # Scale coords
    coords_scaled = np.zeros_like(coords)
    for i in range(2):
        col = coords[:, i]
        rng = col.max() - col.min()
        coords_scaled[:, i] = (col - col.min()) / rng - 0.5

    # Spline bases (df_tps=6, matching R default)
    bs_xy_bio, (df_bio_x, df_bio_y) = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], 6)
    df_ls_x = max(df_bio_x // 2, 1)
    df_ls_y = max(df_bio_y // 2, 1)
    bs_xy_ls, _ = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], max(df_ls_x, df_ls_y))

    # Size factors
    LS = counts.sum(axis=0)
    LS = LS / LS.mean()
    logLS = np.log(np.maximum(1e-8, LS))

    # Build W
    W = np.hstack([logLS.reshape(-1, 1), bs_xy_bio, logLS.reshape(-1, 1) * bs_xy_ls])
    n_bio = bs_xy_bio.shape[1]
    n_ls = bs_xy_ls.shape[1]
    wtype = np.empty(W.shape[1], dtype='<U10')
    wtype[0] = 'ls'
    wtype[1:1+n_bio] = 'biology'
    wtype[1+n_bio:1+n_bio+n_ls] = 'ls'

    # Sample 25%
    nsub = int(round(0.25 * ncells))
    rng = np.random.default_rng(42)
    idx = rng.choice(ncells, size=nsub, replace=False)
    Y_sub = counts[:, idx]; W_sub = W[idx, :]; nW = W_sub.shape[1]

    # Initialize
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((ngenes, nW)); alpha[:, 0] = 1.0
    Y_mean = np.mean(Y_sub, axis=1)
    Y_var = np.var(Y_sub, axis=1, ddof=1)
    psi = np.maximum((Y_var - Y_mean) / np.maximum(Y_mean**2, 1e-10), 0.01)
    psi = np.maximum(psi, 1e-10)
    size = 1.0 / psi
    lam = np.full(nW - 1, 0.0001 * ncells)

    loglik_prev = -np.inf
    t0 = time.perf_counter()

    for iteration in range(max_iter):
        lmu = np.clip(gmean[:, None] + alpha @ W_sub.T, -50, 50)
        mu_hat = np.exp(lmu)
        k = counts[:, idx].astype(np.int64)
        n = size[:, None]
        p = np.clip(n / (n + mu_hat), 1e-15, 1-1e-15)
        ll = np.sum(gammaln(k + n) - gammaln(k + 1) - gammaln(n) + xlogy(n, p) + xlog1py(k, -p))

        Z = lmu + ((Y_sub + 0.01) / (mu_hat + 0.01) - 1)
        sig_inv = np.clip(1.0 / (psi[:, None] * np.exp(-lmu)), 0, 1e10)
        wt = np.minimum(np.mean(sig_inv, axis=0), np.quantile(sig_inv.mean(axis=0), 0.98))

        a1 = np.mean(alpha[:, 0])
        W_rest = W_sub[:, 1:]
        b_rest = ((Z - gmean[:, None] - (a1 * W_sub[:, 0:1]).T) * wt[None, :]) @ W_rest
        WtW_rest = W_rest.T @ (wt[:, None] * W_rest)
        alpha_rest = sp_solve(WtW_rest + np.diag(lam), b_rest.T, assume_a='pos').T
        alpha = np.column_stack([np.full(ngenes, a1), alpha_rest])
        gmean = np.sum((Z - alpha @ W_sub.T) * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        if iteration > 0 and abs(ll - loglik_prev) / max(abs(loglik_prev), 1) < tol:
            break
        loglik_prev = ll

    py_time = time.perf_counter() - t0

    # Normalize (mean_bio)
    is_bio = wtype == 'biology'
    lmu_bio = np.clip(gmean[:, None] + alpha[:, is_bio] @ W[:, is_bio].T, -50, 50)
    lmu_median = np.median(lmu_bio, axis=1)
    lmu_mad = median_abs_deviation(lmu_bio, axis=1, nan_policy='omit')
    lmu_mad = np.where(np.isnan(lmu_mad) | (lmu_mad == 0), 1.0, lmu_mad)
    lmu_max = lmu_median + 4 * lmu_mad
    lmu_bio = np.minimum(lmu_bio, lmu_max[:, None])
    logcounts = np.log2(np.exp(lmu_bio))

    return logcounts, py_time, iteration + 1


def main():
    print("=" * 60)
    print("Real 10x Visium SP1 Data Test")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    counts, coords, gene_names = load_visium_sp1()
    print(f"Data: {counts.shape[0]} genes x {counts.shape[1]} spots")

    # Save for R
    print("Saving for R...")
    pd.DataFrame(counts.astype(int)).to_csv(f'{OUT_DIR}/visium_counts.csv', index=False)
    pd.DataFrame(coords).to_csv(f'{OUT_DIR}/visium_coords.csv', index=False)

    # Run Python
    print("\nRunning Python SpaNorm...")
    logcounts, py_time, iters = run_python_spanorm(counts, coords)
    print(f"Python time: {py_time:.3f}s ({iters} iterations)")
    print(f"Logcounts range: [{logcounts.min():.4f}, {logcounts.max():.4f}]")
    print(f"Logcounts shape: {logcounts.shape}")

    # Save Python results
    pd.DataFrame(logcounts).to_csv(f'{OUT_DIR}/visium_py_logcounts.csv', index=False)
    print(f"Saved to {OUT_DIR}/visium_py_logcounts.csv")

    # Save summary
    with open(f'{OUT_DIR}/visium_summary.txt', 'w') as f:
        f.write(f"genes: {counts.shape[0]}\n")
        f.write(f"spots: {counts.shape[1]}\n")
        f.write(f"python_time: {py_time:.3f}\n")
        f.write(f"iterations: {iters}\n")
        f.write(f"logcounts_min: {logcounts.min():.4f}\n")
        f.write(f"logcounts_max: {logcounts.max():.4f}\n")

    print(f"\nDone! Run R comparison with:")
    print(f"  Rscript tests/run_visium_r.R")


if __name__ == "__main__":
    main()
