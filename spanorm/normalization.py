"""Main SpaNorm normalization pipeline.

Ports SpaNorm, fitSpaNorm, and the batch checking utilities.
"""

import numpy as np
from scipy import sparse, stats
import warnings

from .fit import SpaNormFit
from .spline import bs_tps
from .nb_model import (
    fit_spanorm_nb,
    get_adjustment_fun,
    calculate_mu,
)


def check_batch(batch, nobs):
    """Check and process batch design matrix.

    Parameters
    ----------
    batch : np.ndarray or None
        Batch vector or design matrix.
    nobs : int
        Number of observations.

    Returns
    -------
    np.ndarray or None
        Processed batch design matrix, or None if no batch.
    """
    if batch is None:
        return None

    batch = np.asarray(batch)

    if np.any(np.isnan(batch)):
        raise ValueError("'batch' cannot have missing values")

    if batch.ndim == 2:
        if batch.shape[0] != nobs:
            raise ValueError("Number of rows in 'batch' matrix does not match number of cells")
        if not np.issubdtype(batch.dtype, np.number):
            raise ValueError("'batch' should be a numeric matrix")

        # Check for intercept column (all ones)
        is_intercept = np.all(batch == 1, axis=0)
        if np.any(is_intercept):
            warnings.warn("'intercept' term detected and will be removed")
            batch = batch[:, ~is_intercept]
    elif batch.ndim == 1:
        if len(batch) != nobs:
            raise ValueError("Length of 'batch' vector does not match number of cells")
        # Create design matrix using one-hot encoding
        unique_vals = np.unique(batch)
        if len(unique_vals) > 1:
            # Create design matrix with intercept, then remove intercept
            design = np.zeros((nobs, len(unique_vals) - 1))
            for i, val in enumerate(unique_vals[1:]):
                design[:, i] = (batch == val).astype(float)
            batch = design
        else:
            batch = None

    return batch


def fit_spanorm(Y, coords, sample_p, gene_model, df_tps=6, lambda_a=0.0001,
                batch=None, LS=None, tol=1e-4, step_factor=0.5, maxit_nb=50,
                maxit_psi=25, maxn_psi=500, verbose=True):
    """Fit the SpaNorm model.

    Parameters
    ----------
    Y : np.ndarray or sparse matrix
        Count matrix (ngenes, ncells).
    coords : np.ndarray
        Spatial coordinates (ncells, 2).
    sample_p : float
        Proportion of cells to sample for fitting.
    gene_model : str
        Gene model ('nb').
    df_tps : int or tuple
        Degrees of freedom for thin-plate spline.
    lambda_a : float or tuple
        Smoothing parameter.
    batch : np.ndarray or None
        Batch design.
    LS : np.ndarray or None
        Size factors.
    tol : float
        Convergence tolerance.
    step_factor : float
        IRLS step factor.
    maxit_nb : int
        Maximum NB iterations.
    maxit_psi : int
        Maximum dispersion iterations.
    maxn_psi : int
        Max cells for dispersion estimation.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    SpaNormFit
        Fitted model.
    """
    if gene_model not in ("nb",):
        raise ValueError(f"'gene_model' should be one of: nb")
    if sample_p <= 0 or sample_p > 1:
        raise ValueError("'sample_p' should be in the interval (0,1]")
    lambda_a = np.atleast_1d(np.asarray(lambda_a, dtype=np.float64))
    if np.any(lambda_a < 0):
        raise ValueError("'lambda_a' should be positive")
    if len(lambda_a) not in (1, 2):
        raise ValueError("'lambda_a' should be a single value or a vector of length 2")
    if len(lambda_a) == 1:
        lambda_a = np.repeat(lambda_a, 2)

    if sparse.issparse(Y):
        Y_dense = np.asarray(Y.todense())
    else:
        Y_dense = np.asarray(Y, dtype=np.float64)

    n_genes, n_cells = Y_dense.shape

    # Scale coordinates to [-0.5, 0.5]
    coords = np.asarray(coords, dtype=np.float64)
    coords_scaled = np.zeros_like(coords)
    for i in range(2):
        col = coords[:, i]
        rng = col.max() - col.min()
        if rng > 0:
            coords_scaled[:, i] = (col - col.min()) / rng - 0.5
        else:
            coords_scaled[:, i] = 0.0

    # Get basis for thin-plate spline
    df_tps = np.atleast_1d(np.asarray(df_tps, dtype=np.int64))
    if len(df_tps) == 1:
        df_tps_bio = int(df_tps[0])
        df_tps_ls = max(int(np.ceil(df_tps_bio / 2)), 1)
    elif len(df_tps) == 2:
        df_tps_bio = int(df_tps[0])
        df_tps_ls = int(df_tps[1])
    else:
        raise ValueError("'df_tps' should be a single integer or a vector of length 2")

    # Build spline bases
    bs_xy_bio, (df_bio_x, df_bio_y) = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], df_tps_bio)
    bs_xy_ls, (df_ls_x, df_ls_y) = bs_tps(coords_scaled[:, 0], coords_scaled[:, 1], df_tps_ls)

    df_tps_actual = np.array([df_bio_x, df_bio_y, df_ls_x, df_ls_y], dtype=np.int64)

    # Calculate library size if not provided
    if LS is None:
        # Simple library size calculation (scran would be ideal but requires R)
        LS = np.asarray(Y_dense.sum(axis=0)).ravel()
        LS = LS / np.mean(LS)

    logLS = np.log(np.maximum(1e-8, LS))

    # Build covariate matrix W
    # model.matrix(~ logLS + bs.xy.bio + logLS:bs.xy.ls)[, -1]
    # = [logLS, bs_xy_bio, logLS * bs_xy_ls]
    # (intercept removed)

    # logLS column
    logLS_col = logLS.reshape(-1, 1)

    # Interaction: logLS * each column of bs_xy_ls
    logLS_bs_ls = logLS_col * bs_xy_ls

    # Assemble W (without intercept)
    W = np.hstack([logLS_col, bs_xy_bio, logLS_bs_ls])

    # Add batch design
    batch_mat = check_batch(batch, n_cells)
    if batch_mat is not None and batch_mat.shape[1] > 0:
        W = np.hstack([W, batch_mat])

    # Mark covariate types
    n_bio = df_bio_x * df_bio_y
    n_ls = df_ls_x * df_ls_y
    n_batch = batch_mat.shape[1] if batch_mat is not None else 0

    wtype = np.empty(W.shape[1], dtype="<U10")
    wtype[:] = "batch"

    # logLS column (index 0) is "ls"
    wtype[0] = "ls"

    # Biology columns (indices 1 to n_bio)
    wtype[1:1 + n_bio] = "biology"

    # LS columns (indices 1+n_bio to 1+n_bio+n_ls)
    wtype[1 + n_bio:1 + n_bio + n_ls] = "ls"

    # Create lambda vector
    lambda_a_vec = np.zeros(len(wtype))
    lambda_a_vec[wtype == "biology"] = lambda_a[0]
    lambda_a_vec[wtype == "ls"] = lambda_a[1]
    # Remove first element (for the intercept that was removed)
    lambda_a_vec = lambda_a_vec[1:]

    # Sample cells
    maxn = 3000
    nsub = int(round(sample_p * n_cells))
    if verbose:
        print(f"{nsub} cells/spots sampled to fit model")
    if nsub > maxn:
        warnings.warn(f"Consider reducing 'sample_p' to {max(0.01, maxn / n_cells):.2f}")
    elif nsub == 0:
        raise ValueError(f"'sample_p' is too small, consider using {min(1.0, maxn / n_cells):.2f}")

    rng = np.random.default_rng(42)
    idx = np.zeros(n_cells, dtype=bool)
    idx[rng.choice(n_cells, size=nsub, replace=False)] = True

    # Fit model
    fit_result = fit_spanorm_nb(
        Y_dense, W, idx,
        maxit_psi=maxit_psi, tol=tol, maxn_psi=maxn_psi,
        lambda_a=lambda_a_vec, is_spanorm=True
    )

    # Create SpaNormFit object
    fit = SpaNormFit(
        ngenes=n_genes,
        ncells=n_cells,
        gene_model=gene_model,
        df_tps=df_tps_actual,
        sample_p=sample_p,
        lambda_a=lambda_a,
        batch=batch,
        W=W,
        alpha=fit_result['alpha'],
        gmean=fit_result['gmean'],
        psi=fit_result['psi'],
        wtype=wtype,
        loglik=np.array(fit_result['loglik']),
        sampling=fit_result['sampling'],
    )

    return fit


def spanorm(adata, sample_p=0.25, gene_model="nb", adj_method="auto",
            scale_factor=1, df_tps=6, lambda_a=0.0001, batch=None,
            tol=1e-4, step_factor=0.5, maxit_nb=50, maxit_psi=25,
            maxn_psi=500, overwrite=False, verbose=True):
    """Perform SpaNorm normalization on an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data with counts in .X or .layers['counts'] and
        spatial coordinates in .obsm['spatial'].
    sample_p : float
        Proportion of cells to sample.
    gene_model : str
        Gene model ('nb').
    adj_method : str
        Adjustment method ('auto', 'logpac', 'pearson', 'medbio', 'meanbio').
    scale_factor : float
        Scaling factor.
    df_tps : int or tuple
        Degrees of freedom for thin-plate spline.
    lambda_a : float or tuple
        Smoothing parameter.
    batch : np.ndarray or None
        Batch design.
    tol : float
        Convergence tolerance.
    step_factor : float
        IRLS step factor.
    maxit_nb : int
        Maximum NB iterations.
    maxit_psi : int
        Maximum dispersion iterations.
    maxn_psi : int
        Max cells for dispersion estimation.
    overwrite : bool
        Whether to force recompute.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    anndata.AnnData
        Annotated data with normalized values in .layers['logcounts'].
    """
    import anndata as ad

    # Extract counts — AnnData stores (cells, genes), SpaNorm expects (ngenes, ncells)
    if 'counts' in adata.layers:
        counts = adata.layers['counts']
    else:
        counts = adata.X

    if sparse.issparse(counts):
        counts_dense = np.asarray(counts.todense())
    else:
        counts_dense = np.asarray(counts, dtype=np.float64)

    # Transpose to (ngenes, ncells)
    counts_dense = counts_dense.T

    # Extract spatial coordinates
    if 'spatial' not in adata.obsm:
        raise ValueError("'spatial' coordinates not found in adata.obsm")
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)

    n_genes, n_cells = counts_dense.shape

    # Check for existing fit
    fit = adata.uns.get('SpaNorm')
    if (not overwrite and fit is not None
        and fit.ngenes == n_genes
        and fit.ncells == n_cells
        and fit.gene_model == gene_model):
        if verbose:
            print("(1/2) Retrieve precomputed SpaNorm model")
    else:
        if verbose:
            print("(1/2) Fitting SpaNorm model")
        fit = fit_spanorm(
            counts_dense, coords, sample_p, gene_model,
            df_tps=df_tps, lambda_a=lambda_a, batch=batch,
            tol=tol, step_factor=step_factor, maxit_nb=maxit_nb,
            maxit_psi=maxit_psi, maxn_psi=maxn_psi, verbose=verbose
        )
        adata.uns['SpaNorm'] = fit

    # Normalize
    if not fit.has_biology():
        raise ValueError("'SpaNorm' fit should have at least one column representing 'biology'")

    if verbose:
        print("(2/2) Normalising data")

    adj_fun = get_adjustment_fun(gene_model, adj_method)
    normmat = adj_fun(counts_dense, scale_factor, fit)

    # Store as (cells, genes) in AnnData
    adata.layers['logcounts'] = normmat.T

    return adata
