"""SpaNorm PCA — GLM-based PCA using the SpaNorm model.

Ports SpaNormPCA, getResiduals, devianceResiduals, calculatePCA.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .fit import SpaNormFit
from .nb_model import calculate_mu, normalise_pearson, deviance_residuals


def get_residuals(emat, fit_technical, residuals_type="deviance"):
    """Compute residuals from the technical model.

    Parameters
    ----------
    emat : np.ndarray
        Count matrix (ngenes, ncells).
    fit_technical : SpaNormFit
        Technical (null) SpaNorm model.
    residuals_type : str
        Type of residuals: 'deviance' or 'pearson'.

    Returns
    -------
    np.ndarray
        Residual matrix (ngenes, ncells).
    """
    if residuals_type == "pearson":
        return normalise_pearson(emat, 1, fit_technical)
    elif residuals_type == "deviance":
        return deviance_residuals(emat, fit_technical)
    else:
        raise ValueError(f"Unknown residuals type: {residuals_type}")


def calculate_pca(emat, n_components=50):
    """Perform PCA on a matrix.

    Parameters
    ----------
    emat : np.ndarray
        Input matrix (features x samples). Will be transposed internally.
    n_components : int
        Number of components.

    Returns
    -------
    np.ndarray
        PCA coordinates (samples x components) with attributes.
    """
    from sklearn.decomposition import PCA
    from sklearn.utils.extmath import randomized_svd

    # emat is (features x samples), need (samples x features) for PCA
    X = emat.T

    # Center features
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Use randomized SVD for efficiency
    n_components = min(n_components, min(X_centered.shape) - 1)
    U, S, Vt = randomized_svd(X_centered, n_components=n_components, random_state=42)

    # PCA coordinates = U * S (equivalent to X @ V)
    pca_result = U * S

    # Compute variance explained
    var = S ** 2 / (X.shape[0] - 1)
    percent_var = var / var.sum() * 100

    # Store as attributes (matching R's output)
    pca_result = pca_result.astype(np.float64)

    return pca_result, percent_var, var, Vt


def spanorm_pca(adata, n_svgs=3000, n_components=50, svg_fdr=1.0,
                residuals_type="deviance", name="PCA"):
    """Compute GLM-based PCA using the SpaNorm model.

    Parameters
    ----------
    adata : anndata.AnnData
        Input data with SpaNorm and SVG results.
    n_svgs : int
        Number of SVGs to use for PCA.
    n_components : int
        Number of PCA components.
    svg_fdr : float
        FDR threshold for SVG selection.
    residuals_type : str
        Type of residuals ('deviance' or 'pearson').
    name : str
        Name for the PCA result in .obsm.

    Returns
    -------
    anndata.AnnData
        Annotated data with PCA results in .obsm[name].
    """
    # Get SpaNorm fit
    fit_spanorm = adata.uns.get('SpaNorm')
    fit_technical = adata.uns.get('SpaNormNull')
    if fit_spanorm is None or fit_technical is None:
        raise ValueError("SpaNorm fits not found. Run spanorm() and spanorm_svg() first.")

    # Get top SVGs
    if 'svg_fdr' not in adata.var.columns:
        raise ValueError("SVG results not found. Run spanorm_svg() first.")

    svg_mask = adata.var['svg_fdr'] <= svg_fdr
    svg_names = adata.var.index[svg_mask]

    if len(svg_names) > n_svgs:
        # Sort by FDR and take top n_svgs
        fdr_vals = adata.var.loc[svg_names, 'svg_fdr']
        top_idx = np.argsort(fdr_vals.values)[:n_svgs]
        svg_names = svg_names[top_idx]

    # Extract counts — AnnData stores (cells, genes), SpaNorm expects (ngenes, ncells)
    if 'counts' in adata.layers:
        counts = adata.layers['counts']
    else:
        counts = adata.X

    if sparse.issparse(counts):
        counts_dense = np.asarray(counts.todense())
    else:
        counts_dense = np.asarray(counts, dtype=np.float64)

    # Transpose to (ngenes, ncells) and select SVG genes
    counts_dense = counts_dense.T
    gene_idx = [list(adata.var.index).index(g) for g in svg_names if g in adata.var.index]
    emat = counts_dense[gene_idx, :]

    # Subset the technical model to match SVG genes
    from .fit import SpaNormFit
    fit_tech_sub = SpaNormFit(
        ngenes=len(gene_idx),
        ncells=fit_technical.ncells,
        gene_model=fit_technical.gene_model,
        df_tps=fit_technical.df_tps,
        sample_p=fit_technical.sample_p,
        lambda_a=fit_technical.lambda_a,
        batch=fit_technical.batch,
        W=fit_technical.W,
        alpha=fit_technical.alpha[gene_idx, :],
        gmean=fit_technical.gmean[gene_idx],
        psi=fit_technical.psi[gene_idx],
        wtype=fit_technical.wtype,
        loglik=fit_technical.loglik,
        sampling=fit_technical.sampling,
    )

    # Compute residuals
    residuals = get_residuals(emat, fit_tech_sub, residuals_type)

    # PCA
    n_components = min(n_components, min(residuals.shape) - 1, counts_dense.shape[1])
    pca_result, percent_var, var, rotation = calculate_pca(residuals, n_components)

    # Store results
    adata.obsm[name] = pca_result
    adata.uns[f'{name}_percentVar'] = percent_var
    adata.uns[f'{name}_varExplained'] = var
    adata.uns[f'{name}_rotation'] = rotation

    return adata
