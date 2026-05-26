"""Class-based API for SpaNorm, wrapping the functional API."""

import numpy as np
from scipy import sparse


class SpaNorm:
    """Class-based interface for SpaNorm normalization.

    Wraps the functional API to provide method chaining (similar to Seurat/scanpy).

    Parameters
    ----------
    adata : anndata.AnnData
        Input data with counts and spatial coordinates.
    """

    def __init__(self, adata):
        self.adata = adata

    def normalize(self, sample_p=0.25, gene_model="nb", adj_method="auto",
                  scale_factor=1, df_tps=6, lambda_a=0.0001, batch=None,
                  tol=1e-4, step_factor=0.5, maxit_nb=50, maxit_psi=25,
                  maxn_psi=500, overwrite=False, verbose=True):
        """Perform SpaNorm normalization.

        Parameters
        ----------
        Same as spanorm() function.

        Returns
        -------
        SpaNorm
            Self for method chaining.
        """
        from .normalization import spanorm
        self.adata = spanorm(
            self.adata, sample_p=sample_p, gene_model=gene_model,
            adj_method=adj_method, scale_factor=scale_factor,
            df_tps=df_tps, lambda_a=lambda_a, batch=batch,
            tol=tol, step_factor=step_factor, maxit_nb=maxit_nb,
            maxit_psi=maxit_psi, maxn_psi=maxn_psi,
            overwrite=overwrite, verbose=verbose
        )
        return self

    def find_svgs(self, verbose=True):
        """Find spatially variable genes.

        Parameters
        ----------
        verbose : bool
            Whether to print progress.

        Returns
        -------
        SpaNorm
            Self for method chaining.
        """
        from .svg import spanorm_svg
        self.adata = spanorm_svg(self.adata, verbose=verbose)
        return self

    def pca(self, n_svgs=3000, n_components=50, svg_fdr=1.0,
            residuals_type="deviance", name="PCA"):
        """Compute GLM-based PCA.

        Parameters
        ----------
        Same as spanorm_pca() function.

        Returns
        -------
        SpaNorm
            Self for method chaining.
        """
        from .pca import spanorm_pca
        self.adata = spanorm_pca(
            self.adata, n_svgs=n_svgs, n_components=n_components,
            svg_fdr=svg_fdr, residuals_type=residuals_type, name=name
        )
        return self

    def top_svgs(self, n=10, fdr=1.0):
        """Get top SVGs.

        Parameters
        ----------
        n : int
            Number of top SVGs.
        fdr : float
            FDR threshold.

        Returns
        -------
        pd.DataFrame
            Top SVGs.
        """
        from .svg import top_svgs
        return top_svgs(self.adata, n=n, fdr=fdr)

    @property
    def fit(self):
        """Get the SpaNormFit object."""
        return self.adata.uns.get('SpaNorm')

    @property
    def fit_technical(self):
        """Get the technical SpaNormFit object."""
        return self.adata.uns.get('SpaNormNull')
