"""Spatially Variable Gene (SVG) calling.

Ports SpaNormSVG, fitSpaNormTechnical, svgTest, topSVGs.
"""

import numpy as np
import warnings
from scipy import sparse, stats

from .fit import SpaNormFit
from .nb_model import calculate_mu, fit_spanorm_nb


def _bh_correction(p_values):
    """Benjamini-Hochberg FDR correction (matching R's p.adjust 'fdr')."""
    n = len(p_values)
    if n == 0:
        return np.array([])

    # Sort p-values and track original order
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Compute adjusted p-values
    adjusted = np.zeros(n)
    cumulative_min = 1.0
    for i in range(n - 1, -1, -1):
        # BH adjusted p-value: p * n / rank
        rank = i + 1
        adjusted[sorted_idx[i]] = min(cumulative_min, sorted_p[i] * n / rank)
        cumulative_min = adjusted[sorted_idx[i]]

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)
    return adjusted


def fit_spanorm_technical(Y, fit_spanorm):
    """Fit the null (technical) SpaNorm model by removing biology covariates.

    Parameters
    ----------
    Y : np.ndarray
        Count matrix (ngenes, ncells).
    fit_spanorm : SpaNormFit
        Fitted full SpaNorm model.

    Returns
    -------
    SpaNormFit
        Fitted technical model.
    """
    if sparse.issparse(Y):
        Y = np.asarray(Y.todense())

    # Select technical covariates (remove biology)
    wtype = fit_spanorm.wtype
    non_bio = wtype != "biology"
    W = fit_spanorm.W[:, non_bio]
    wtype_new = wtype[non_bio]

    # Create lambda vector
    lambda_a_vec = np.zeros(len(wtype_new))
    lambda_a_vec[wtype_new == "biology"] = fit_spanorm.lambda_a[0]
    lambda_a_vec[wtype_new == "ls"] = fit_spanorm.lambda_a[1]
    lambda_a_vec = lambda_a_vec[1:]

    # Use same sampling as the full model
    idx = fit_spanorm.sampling != "all"
    n_disp = int(np.sum(fit_spanorm.sampling == "dispersion"))

    # Fit model
    fit_result = fit_spanorm_nb(
        Y, W, idx,
        maxn_psi=n_disp,
        lambda_a=lambda_a_vec,
        is_spanorm=True,
    )

    # Create SpaNormFit object
    fit_technical = SpaNormFit(
        ngenes=fit_spanorm.ngenes,
        ncells=fit_spanorm.ncells,
        gene_model=fit_spanorm.gene_model,
        df_tps=fit_spanorm.df_tps,
        sample_p=fit_spanorm.sample_p,
        lambda_a=fit_spanorm.lambda_a,
        batch=fit_spanorm.batch,
        W=W,
        alpha=fit_result['alpha'],
        gmean=fit_result['gmean'],
        psi=fit_result['psi'],
        wtype=wtype_new,
        loglik=np.array(fit_result['loglik']),
        sampling=fit_result['sampling'],
    )

    return fit_technical


def svg_test(Y, fit_spanorm, fit_technical):
    """Perform SVG test using likelihood ratio test.

    Parameters
    ----------
    Y : np.ndarray
        Count matrix (ngenes, ncells).
    fit_spanorm : SpaNormFit
        Full SpaNorm model.
    fit_technical : SpaNormFit
        Technical (null) model.

    Returns
    -------
    dict
        Dictionary with 'svg_F', 'svg_p', 'svg_fdr' arrays.
    """
    if sparse.issparse(Y):
        Y = np.asarray(Y.todense())

    Y = Y.astype(np.float64)

    # Full model log-likelihood
    mu = calculate_mu(fit_spanorm.gmean, fit_spanorm.alpha, fit_spanorm.W)
    psi = fit_spanorm.psi.copy()
    psi_max = np.exp(np.median(np.log(psi)) + 3 * stats.median_abs_deviation(np.log(psi)))
    psi = np.minimum(psi, psi_max)

    from .nb_model import dnbinom_log
    loglik_spanorm = np.sum(dnbinom_log(Y, mu, psi), axis=1)

    # Technical model log-likelihood
    mu_tech = calculate_mu(fit_technical.gmean, fit_technical.alpha, fit_technical.W)
    psi_tech = fit_technical.psi.copy()
    psi_max_tech = np.exp(np.median(np.log(psi_tech)) + 3 * stats.median_abs_deviation(np.log(psi_tech)))
    psi_tech = np.minimum(psi_tech, psi_max_tech)
    loglik_technical = np.sum(dnbinom_log(Y, mu_tech, psi_tech), axis=1)

    # F-test (LRT approximation)
    df1 = fit_spanorm.W.shape[1] - fit_technical.W.shape[1]
    df2 = Y.shape[1] - fit_spanorm.W.shape[1]

    F_lrt = 2 * (loglik_spanorm - loglik_technical) / df1
    F_lrt = np.maximum(F_lrt, 0)

    p_val = stats.f.sf(F_lrt, df1, df2)
    fdr = _bh_correction(p_val)

    return {
        'svg_F': F_lrt,
        'svg_p': p_val,
        'svg_fdr': fdr,
    }


def spanorm_svg(adata, verbose=True):
    """Find spatially variable genes using SpaNorm.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data with SpaNorm fit in .uns['SpaNorm'].
    verbose : bool
        Whether to print progress.

    Returns
    -------
    anndata.AnnData
        Annotated data with SVG results in .var.
    """
    # Extract counts — AnnData stores (cells, genes), SpaNorm expects (ngenes, ncells)
    if 'counts' in adata.layers:
        counts = adata.layers['counts']
    else:
        counts = adata.X

    if sparse.issparse(counts):
        counts_dense = np.asarray(counts.todense())
    else:
        counts_dense = np.asarray(counts, dtype=np.float64)

    counts_dense = counts_dense.T  # (ngenes, ncells)

    # Check for existing SVG results
    if 'svg_F' in adata.var.columns:
        warnings.warn("SVG results exist in adata.var and will be overwritten")
        adata.var = adata.var.drop(columns=['svg_F', 'svg_p', 'svg_fdr'], errors='ignore')

    # Retrieve full model
    fit_spanorm = adata.uns.get('SpaNorm')
    if fit_spanorm is None:
        raise ValueError("SpaNorm fit not found in adata.uns. Run spanorm() first.")

    # Fit technical model
    fit_technical = adata.uns.get('SpaNormNull')
    if fit_technical is None:
        if verbose:
            print("(1/3) Fitting Null SpaNorm model")
        fit_technical = fit_spanorm_technical(counts_dense, fit_spanorm)
        adata.uns['SpaNormNull'] = fit_technical
    else:
        if verbose:
            print("(1/3) Retrieving Null SpaNorm model")

    # F-test
    if verbose:
        print("(2/3) Finding SVGs")
    results = svg_test(counts_dense, fit_spanorm, fit_technical)

    adata.var['svg_F'] = results['svg_F']
    adata.var['svg_p'] = results['svg_p']
    adata.var['svg_fdr'] = results['svg_fdr']

    if verbose:
        n_svgs = np.sum(results['svg_fdr'] < 0.05)
        print(f"(3/3) {n_svgs} SVGs found (FDR < 0.05)")

    return adata


def top_svgs(adata, n=10, fdr=1.0):
    """Get top SVGs from SpaNorm results.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data with SVG results.
    n : int
        Number of top SVGs to return.
    fdr : float
        FDR threshold.

    Returns
    -------
    pd.DataFrame
        Top SVGs with F-statistics, p-values, and FDR.
    """
    import pandas as pd

    if 'svg_fdr' not in adata.var.columns:
        raise ValueError("SVG results not found. Run spanorm_svg() first.")

    results = adata.var[['svg_F', 'svg_p', 'svg_fdr']].copy()
    results = results[results['svg_fdr'] <= fdr]
    results = results.sort_values('svg_fdr')
    n = min(n, len(results))
    return results.iloc[:n]
