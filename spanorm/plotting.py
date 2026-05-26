"""Visualization functions: plot_spatial, plot_covariate.

Ports plotSpatial and plotCovariate from R SpaNorm.
"""

import numpy as np
from scipy import sparse


def plot_spatial(adata, color=None, assay='logcounts', what='expression',
                 point_size=20, cmap='viridis', title=None, ax=None):
    """Plot spatial transcriptomics data.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial data with obsm['spatial'] coordinates.
    color : str or None
        Column in adata.obs or gene name in adata.var to color by.
    assay : str
        Layer to use for expression values.
    what : str
        What to plot: 'expression', 'annotation', or 'reduceddim'.
    point_size : float
        Size of scatter points.
    cmap : str
        Colormap for continuous values.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    coords = adata.obsm['spatial']
    x, y = coords[:, 0], coords[:, 1]

    if color is None:
        ax.scatter(x, y, s=point_size, c='lightgray')
    elif what == 'expression' or color in adata.var.index:
        # Gene expression
        gene_idx = list(adata.var.index).index(color)
        if assay in adata.layers:
            values = adata.layers[assay][:, gene_idx]
        else:
            values = adata.X[:, gene_idx]
        if sparse.issparse(values):
            values = np.asarray(values.todense()).ravel()
        sc = ax.scatter(x, y, s=point_size, c=values, cmap=cmap)
        plt.colorbar(sc, ax=ax, label=color)
    elif color in adata.obs.columns:
        # Annotation
        values = adata.obs[color]
        if values.dtype.name == 'category' or values.dtype == object:
            unique_vals = values.unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
            for i, val in enumerate(unique_vals):
                mask = values == val
                ax.scatter(x[mask], y[mask], s=point_size, c=[colors[i]], label=str(val))
            ax.legend()
        else:
            sc = ax.scatter(x, y, s=point_size, c=values.values.astype(float), cmap=cmap)
            plt.colorbar(sc, ax=ax, label=color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title or color or 'Spatial')
    ax.set_aspect('equal')

    return ax


def plot_covariate(adata, covariate='biology', gene=None, point_size=20,
                   cmap='viridis', title=None, ax=None):
    """Plot predicted expression for a covariate type.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial data with SpaNorm fit in .uns['SpaNorm'].
    covariate : str
        Covariate type: 'biology', 'ls', or 'batch'.
    gene : str or None
        Gene name to plot. If None, uses first gene.
    point_size : float
        Size of scatter points.
    cmap : str
        Colormap.
    title : str or None
        Plot title.
    ax : matplotlib.axes.Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes.
    """
    import matplotlib.pyplot as plt
    from .nb_model import calculate_mu

    fit = adata.uns.get('SpaNorm')
    if fit is None:
        raise ValueError("SpaNorm fit not found. Run spanorm() first.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    coords = adata.obsm['spatial']
    x, y = coords[:, 0], coords[:, 1]

    # Select covariate columns
    is_cov = fit.wtype == covariate
    if not np.any(is_cov):
        raise ValueError(f"No '{covariate}' covariates found in the fit")

    # Compute predicted expression
    mu = calculate_mu(fit.gmean, fit.alpha[:, is_cov], fit.W[:, is_cov])
    log_mu = np.log2(mu + 1)

    # Select gene
    if gene is not None:
        gene_idx = list(adata.var.index).index(gene)
    else:
        gene_idx = 0
        gene = adata.var.index[gene_idx]

    values = log_mu[gene_idx]

    sc = ax.scatter(x, y, s=point_size, c=values, cmap=cmap)
    plt.colorbar(sc, ax=ax, label=f'log2({gene})')
    ax.set_title(title or f'{covariate} effect: {gene}')
    ax.set_aspect('equal')

    return ax
