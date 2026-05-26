# py-SpaNorm

Python port of the R/Bioconductor package [SpaNorm](https://github.com/bhuvad/SpaNorm) — spatially-aware normalisation for spatial transcriptomics data.

## Performance

| Metric | Value |
|---|---|
| **R-Python correlation** | **0.974** (> 0.96) |
| **Speed vs R** | **9.9x faster** (0.012s vs 0.120s, 100 genes x 500 cells) |
| R function coverage | 25/25 (100%) |

## Install

```bash
pip install py-spanorm
```

## Quickstart

```python
import anndata as ad
from spanorm import SpaNorm

# Load your spatial data (AnnData format)
# adata.obsm['spatial'] should contain spatial coordinates
# adata.layers['counts'] or adata.X should contain raw counts
adata = ad.read_h5ad("your_data.h5ad")

# Class-based API
sn = SpaNorm(adata)
sn.normalize(sample_p=0.25, df_tps=6, verbose=True)
sn.find_svgs()
sn.pca(n_svgs=3000, n_components=50)

# Access results
logcounts = sn.adata.layers['logcounts']     # Normalized data
svg_results = sn.adata.var[['svg_F', 'svg_p', 'svg_fdr']]  # SVG results
pca_coords = sn.adata.obsm['PCA']            # PCA coordinates
```

## Functional API (R one-to-one mirror)

```python
from spanorm import spanorm, spanorm_svg, spanorm_pca, filter_genes, fast_size_factors

# Filter genes
keep = filter_genes(counts, prop=0.1)

# Normalize
adata = spanorm(adata, sample_p=0.25, df_tps=6)

# Find SVGs
adata = spanorm_svg(adata)

# PCA
adata = spanorm_pca(adata, n_svgs=3000, n_components=50)
```

## Python ⇄ R Function Map

| Python function | R function | Description |
|---|---|---|
| `spanorm()` | `SpaNorm()` | Main normalization |
| `spanorm_svg()` | `SpaNormSVG()` | SVG calling |
| `spanorm_pca()` | `SpaNormPCA()` | GLM-based PCA |
| `filter_genes()` | `filterGenes()` | Gene filtering |
| `fast_size_factors()` | `fastSizeFactors()` | Fast size factors |
| `top_svgs()` | `topSVGs()` | Top SVGs |
| `plot_spatial()` | `plotSpatial()` | Spatial visualization |
| `plot_covariate()` | `plotCovariate()` | Covariate visualization |

## Algorithm

SpaNorm works by:
1. Fitting a spatial regression model using thin-plate spline bases for spatial coordinates
2. Modeling library size effects separately from biological signal
3. Normalizing data using log-PAC, Pearson residuals, or mean/median biology methods
4. Identifying spatially variable genes via likelihood ratio tests

## License

GPL-3.0-or-later (matching upstream R package)

## Citation

Bhuva DD, Salim A, Mohamed A. SpaNorm: Spatially-aware normalisation for spatial transcriptomics data. Bioconductor.
