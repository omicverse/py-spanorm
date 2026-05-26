"""Integration tests — run the full pipeline on synthetic data."""

import numpy as np
import pytest
import anndata as ad


def create_synthetic_spatial_data(ngenes=100, ncells=500, seed=42):
    """Create synthetic spatial transcriptomics data.

    Parameters
    ----------
    ngenes : int
        Number of genes.
    ncells : int
        Number of cells/spots.
    seed : int
        Random seed.

    Returns
    -------
    anndata.AnnData
        Synthetic spatial data.
    """
    np.random.seed(seed)

    # Generate spatial coordinates (simulating a tissue section)
    coords = np.column_stack([
        np.random.uniform(0, 10, ncells),
        np.random.uniform(0, 10, ncells),
    ])

    # Generate count data with spatial structure
    # Create a spatial pattern (e.g., gradient)
    x_norm = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
    y_norm = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

    # Base expression (per gene)
    base_expr = np.random.uniform(1, 5, ngenes)

    # Generate mu matrix (ngenes x ncells)
    mu = np.outer(base_expr, np.ones(ncells))

    # Add spatial pattern to some genes
    spatial_genes = int(ngenes * 0.3)
    for g in range(spatial_genes):
        pattern = np.sin(x_norm * np.pi * (g % 3 + 1)) * np.cos(y_norm * np.pi * (g % 2 + 1))
        modulation = 1 + 2 * (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)
        mu[g, :] = mu[g, :] * modulation

    # Add library size variation
    ls = np.random.lognormal(0, 0.5, ncells)
    mu = mu * ls[np.newaxis, :]  # (ngenes, ncells)

    counts = np.zeros((ngenes, ncells), dtype=np.float64)
    for g in range(ngenes):
        psi = np.random.uniform(0.1, 0.5)
        size = 1.0 / psi
        p = size / (size + mu[g])
        counts[g] = np.random.negative_binomial(size, p)

    # Create AnnData (cells x genes convention)
    adata = ad.AnnData(X=counts.T)
    adata.obsm['spatial'] = coords
    adata.var.index = [f'gene_{i}' for i in range(ngenes)]
    adata.obs.index = [f'cell_{i}' for i in range(ncells)]
    adata.layers['counts'] = counts.T

    return adata


def test_full_pipeline():
    """Test the full SpaNorm pipeline: normalize -> SVG -> PCA."""
    from spanorm import spanorm, spanorm_svg, spanorm_pca

    adata = create_synthetic_spatial_data(ngenes=50, ncells=200, seed=42)

    # Step 1: Normalize
    adata = spanorm(adata, sample_p=0.25, df_tps=2, tol=1e-2, verbose=False)

    assert 'logcounts' in adata.layers
    assert adata.layers['logcounts'].shape == (200, 50)
    assert not np.any(np.isnan(adata.layers['logcounts']))
    assert not np.any(np.isinf(adata.layers['logcounts']))

    # Step 2: Find SVGs
    adata = spanorm_svg(adata, verbose=False)

    assert 'svg_F' in adata.var.columns
    assert 'svg_p' in adata.var.columns
    assert 'svg_fdr' in adata.var.columns
    assert not np.any(np.isnan(adata.var['svg_p']))

    # Step 3: PCA
    adata = spanorm_pca(adata, n_svgs=20, n_components=10)

    assert 'PCA' in adata.obsm
    assert adata.obsm['PCA'].shape == (200, 10)

    print("Full pipeline integration test PASSED")


def test_normalize_deterministic():
    """Test that normalization is deterministic (same seed -> same result)."""
    from spanorm import spanorm

    adata1 = create_synthetic_spatial_data(ngenes=30, ncells=100, seed=42)
    adata2 = create_synthetic_spatial_data(ngenes=30, ncells=100, seed=42)

    result1 = spanorm(adata1, sample_p=0.25, df_tps=2, tol=1e-2, verbose=False)
    result2 = spanorm(adata2, sample_p=0.25, df_tps=2, tol=1e-2, verbose=False)

    np.testing.assert_array_equal(
        result1.layers['logcounts'],
        result2.layers['logcounts'],
        err_msg="Normalization is not deterministic"
    )

    print("Deterministic normalization test PASSED")


def test_class_api():
    """Test the class-based API."""
    from spanorm import SpaNorm

    adata = create_synthetic_spatial_data(ngenes=30, ncells=100, seed=42)
    sn = SpaNorm(adata)

    sn.normalize(sample_p=0.25, df_tps=2, tol=1e-2, verbose=False)
    assert sn.fit is not None
    assert sn.fit.ngenes == 30
    assert sn.fit.ncells == 100

    sn.find_svgs(verbose=False)
    assert 'svg_fdr' in sn.adata.var.columns

    sn.pca(n_svgs=15, n_components=5)
    assert 'PCA' in sn.adata.obsm

    print("Class API test PASSED")


def test_filter_genes():
    """Test gene filtering."""
    from spanorm import filter_genes

    adata = create_synthetic_spatial_data(ngenes=30, ncells=100, seed=42)
    counts = adata.layers['counts']

    keep = filter_genes(counts.T, prop=0.1)
    assert keep.sum() > 0
    assert keep.sum() <= 30

    print("Gene filtering test PASSED")


def test_fast_size_factors():
    """Test fast size factor computation."""
    from spanorm import fast_size_factors

    adata = create_synthetic_spatial_data(ngenes=30, ncells=100, seed=42)
    counts = adata.layers['counts']

    sf = fast_size_factors(counts.T)
    assert len(sf) == 100
    assert np.all(sf > 0)
    assert np.allclose(np.mean(sf), 1.0)

    print("Fast size factors test PASSED")


def test_bs_tps_basic():
    """Test thin-plate spline basis construction."""
    from spanorm.spline import bs_tps

    np.random.seed(42)
    x = np.random.uniform(0, 10, 50)
    y = np.random.uniform(0, 10, 50)

    basis, (df_x, df_y) = bs_tps(x, y, df_tps=4)

    assert basis.shape[0] == 50
    assert basis.shape[1] == df_x * df_y
    # Check centering
    np.testing.assert_allclose(basis.mean(axis=0), 0, atol=1e-10)

    print("bs_tps basic test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
