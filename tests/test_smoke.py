"""Smoke tests — basic import and shape checks."""

import numpy as np
import pytest


def test_import():
    """Test that the package imports."""
    import spanorm
    assert hasattr(spanorm, 'SpaNorm')
    assert hasattr(spanorm, 'filter_genes')
    assert hasattr(spanorm, 'fast_size_factors')
    assert hasattr(spanorm, 'spanorm')
    assert hasattr(spanorm, 'spanorm_svg')
    assert hasattr(spanorm, 'spanorm_pca')


def test_filter_genes():
    """Test gene filtering."""
    from spanorm import filter_genes
    counts = np.array([
        [1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ])
    keep = filter_genes(counts, prop=0.5)
    assert keep[0] == True
    assert keep[1] == False
    assert keep[2] == True


def test_fast_size_factors():
    """Test size factor computation."""
    from spanorm import fast_size_factors
    counts = np.array([
        [10, 20, 30],
        [10, 20, 30],
    ])
    sf = fast_size_factors(counts)
    assert len(sf) == 3
    assert np.allclose(sf, [0.5, 1.0, 1.5])


def test_bs_tps():
    """Test thin-plate spline basis construction."""
    from spanorm.spline import bs_tps
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 100)
    y = rng.uniform(0, 10, 100)

    basis, (df_x, df_y) = bs_tps(x, y, df_tps=6)
    assert basis.shape[0] == 100
    assert basis.shape[1] == df_x * df_y
    # Check centering
    assert np.allclose(basis.mean(axis=0), 0, atol=1e-10)


def test_spa_norm_fit():
    """Test SpaNormFit dataclass."""
    from spanorm import SpaNormFit

    ngenes = 10
    ncells = 20
    n_cov = 3

    fit = SpaNormFit(
        ngenes=ngenes,
        ncells=ncells,
        gene_model="nb",
        df_tps=np.array([3, 3, 2, 2]),
        sample_p=0.25,
        lambda_a=np.array([0.0001, 0.0001]),
        batch=None,
        W=np.random.randn(ncells, n_cov),
        alpha=np.random.randn(ngenes, n_cov),
        gmean=np.random.randn(ngenes),
        psi=np.abs(np.random.randn(ngenes)) + 0.1,
        wtype=np.array(["ls", "biology", "biology"]),
        loglik=np.array([-100.0, -90.0]),
        sampling=np.array(["all"] * ncells),
    )

    assert fit.ngenes == ngenes
    assert fit.ncells == ncells
    assert fit.has_biology()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
