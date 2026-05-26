"""Preprocessing utilities: filterGenes, fastSizeFactors."""

import numpy as np
from scipy import sparse


def filter_genes(counts, prop=0.1):
    """Filter genes based on expression proportion.

    Parameters
    ----------
    counts : sparse or dense matrix
        Count matrix (genes x cells).
    prop : float
        Minimum proportion of cells where the gene must be expressed.

    Returns
    -------
    np.ndarray
        Boolean array indicating which genes to keep.
    """
    if prop < 0 or prop > 1:
        raise ValueError("'prop' must be between 0 and 1")

    if sparse.issparse(counts):
        means = np.array(counts.mean(axis=1)).ravel()
    else:
        means = np.mean(counts > 0, axis=1)

    keep = means >= prop
    return keep


def fast_size_factors(counts):
    """Compute size factors as library size divided by mean library size.

    Parameters
    ----------
    counts : sparse or dense matrix
        Count matrix (genes x cells).

    Returns
    -------
    np.ndarray
        Size factors (length = n_cells).
    """
    if sparse.issparse(counts):
        ls = np.array(counts.sum(axis=0)).ravel()
    else:
        ls = counts.sum(axis=0)

    ls = ls / np.mean(ls)
    return ls
