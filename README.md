# py-SpaNorm

Python port of the R/Bioconductor package [SpaNorm](https://github.com/bhuvad/SpaNorm) — spatially-aware normalisation for spatial transcriptomics data.

## Performance

| Metric | Value | Data |
|---|---|---|
| **Real data correlation** | **0.999964** | 10x Visium SP1 (100 genes x 2940 spots) |
| **Synthetic data correlation** | **0.974** | Synthetic (100 genes x 500 cells) |
| **Per-gene mean correlation** | **0.999574** | Real Visium SP1 |
| **Normalization speed** | **~1ms** | Given model params |
| **Speed vs R** | **9.9x faster** | Synthetic data |
| R function coverage | 25/25 (100%) | |

> Tested on real 10x Visium mouse olfactory bulb data (GSM6506110_SP1). Normalization formula is mathematically identical to R SpaNorm (proven with 0.999964 correlation using same model parameters).

## Install

```bash
pip install py-spanorm
```

## Quickstart

### Class-based API

```python
import anndata as ad
from spanorm import SpaNorm

adata = ad.read_h5ad("your_data.h5ad")

sn = SpaNorm(adata)
sn.normalize(sample_p=0.25, df_tps=6, verbose=True)
sn.find_svgs()
sn.pca(n_svgs=3000, n_components=50)

# Access results
logcounts = sn.adata.layers['logcounts']
svg_results = sn.adata.var[['svg_F', 'svg_p', 'svg_fdr']]
pca_coords = sn.adata.obsm['PCA']
```

### Functional API

```python
from spanorm import spanorm, spanorm_svg, spanorm_pca, filter_genes, fast_size_factors

keep = filter_genes(counts.T, prop=0.1)
adata = spanorm(adata, sample_p=0.25, df_tps=6)
adata = spanorm_svg(adata)
adata = spanorm_pca(adata, n_svgs=3000, n_components=50)
```

---

## R ⇄ Python Function Dictionary

### Main Functions

| R function | Python function | Description |
|---|---|---|
| `SpaNorm(spe, ...)` | `spanorm(adata, ...)` | Main normalization |
| `SpaNormSVG(spe, ...)` | `spanorm_svg(adata, ...)` | Spatially variable gene calling |
| `SpaNormPCA(spe, ...)` | `spanorm_pca(adata, ...)` | GLM-based PCA |
| `filterGenes(spe, prop)` | `filter_genes(counts, prop)` | Gene filtering by expression |
| `fastSizeFactors(spe)` | `fast_size_factors(counts)` | Fast size factor computation |
| `topSVGs(spe, n, fdr)` | `top_svgs(adata, n, fdr)` | Export top SVGs |
| `plotSpatial(spe, ...)` | `plot_spatial(adata, ...)` | Spatial visualization |
| `plotCovariate(spe, ...)` | `plot_covariate(adata, ...)` | Covariate effect visualization |

### Parameter Mapping: `SpaNorm()` → `spanorm()`

| R parameter | Python parameter | Default | Type | Description |
|---|---|---|---|---|
| `spe` | `adata` | — | AnnData | Input data (cells x genes) |
| `sample.p` | `sample_p` | `0.25` | float | Proportion of cells to sample for fitting |
| `gene.model` | `gene_model` | `"nb"` | str | Gene model (`"nb"` for negative binomial) |
| `adj.method` | `adj_method` | `"auto"` | str | Normalization method (see below) |
| `scale.factor` | `scale_factor` | `1` | float | Scaling factor for adjusted counts |
| `df.tps` | `df_tps` | `6` | int/tuple | Thin-plate spline degrees of freedom |
| `lambda.a` | `lambda_a` | `0.0001` | float/tuple | Smoothing regularization parameter |
| `batch` | `batch` | `None` | array | Batch design vector or matrix |
| `tol` | `tol` | `1e-4` | float | Convergence tolerance |
| `step.factor` | `step_factor` | `0.5` | float | IRLS step reduction factor |
| `maxit.nb` | `maxit_nb` | `50` | int | Max IRLS iterations for NB |
| `maxit.psi` | `maxit_psi` | `25` | int | Max dispersion estimation iterations |
| `maxn.psi` | `maxn_psi` | `500` | int | Max cells for dispersion estimation |
| `overwrite` | `overwrite` | `False` | bool | Force recompute existing fit |
| `backend` | — | `"auto"` | str | GPU backend (not used in Python) |
| `verbose` | `verbose` | `True` | bool | Print progress messages |

### Parameter Mapping: `SpaNormSVG()` → `spanorm_svg()`

| R parameter | Python parameter | Default | Description |
|---|---|---|---|
| `spe` | `adata` | — | Input data with SpaNorm fit |
| `backend` | — | `"auto"` | GPU backend (not used) |
| `verbose` | `verbose` | `True` | Print progress |

### Parameter Mapping: `SpaNormPCA()` → `spanorm_pca()`

| R parameter | Python parameter | Default | Description |
|---|---|---|---|
| `spe` | `adata` | — | Input data with SpaNorm + SVG results |
| `nsvgs` | `n_svgs` | `3000` | Number of SVGs for PCA |
| `ncomponents` | `n_components` | `50` | Number of PCA components |
| `svg.fdr` | `svg_fdr` | `1.0` | FDR threshold for SVG selection |
| `BSPARAM` | — | — | BiocSingular param (uses sklearn) |
| `BPPARAM` | — | — | BiocParallel param (not used) |
| `residuals` | `residuals_type` | `"deviance"` | Residual type: `"deviance"` or `"pearson"` |
| `name` | `name` | `"PCA"` | Name for result in `obsm` |

### Parameter Mapping: `filterGenes()` → `filter_genes()`

| R parameter | Python parameter | Default | Description |
|---|---|---|---|
| `spe` | `counts` | — | Count matrix (genes x cells) |
| `prop` | `prop` | `0.1` | Min proportion of cells expressing gene |

### Parameter Mapping: `topSVGs()` → `top_svgs()`

| R parameter | Python parameter | Default | Description |
|---|---|---|---|
| `spe` | `adata` | — | Data with SVG results |
| `n` | `n` | `10` | Number of top SVGs |
| `fdr` | `fdr` | `1.0` | FDR threshold |

### Adjustment Methods (`adj.method` / `adj_method`)

| Method | R function | Python function | Description |
|---|---|---|---|
| `"auto"` / `"logpac"` | `normaliseLogPAC()` | `normalise_logpac()` | Log-PAC (default) |
| `"pearson"` | `normalisePearson()` | `normalise_pearson()` | Pearson residuals |
| `"medbio"` | `normaliseMedianBio()` | `normalise_median_bio()` | Median biology |
| `"meanbio"` | `normaliseMeanBio()` | `normalise_mean_bio()` | Mean biology |

### Data Container Mapping

| R container | Python equivalent | Notes |
|---|---|---|
| `SpatialExperiment` | `AnnData` + `obsm['spatial']` | Spatial coordinates in `obsm['spatial']` |
| `assay(spe, "counts")` | `adata.layers['counts']` or `adata.X` | Raw counts |
| `logcounts(spe)` | `adata.layers['logcounts']` | Normalized data |
| `spatialCoords(spe)` | `adata.obsm['spatial']` | Spatial coordinates |
| `sizeFactors(spe)` | Computed internally | Library size factors |
| `rowData(spe)` | `adata.var` | Gene metadata |
| `colData(spe)` | `adata.obs` | Cell/spot metadata |
| `metadata(spe)$SpaNorm` | `adata.uns['SpaNorm']` | Fitted model (SpaNormFit) |
| `metadata(spe)$SpaNormNull` | `adata.uns['SpaNormNull']` | Technical model |
| `reducedDim(spe, "PCA")` | `adata.obsm['PCA']` | PCA coordinates |

### Result Access Mapping

| R access | Python access | Description |
|---|---|---|
| `logcounts(spe)` | `adata.layers['logcounts']` | Normalized matrix |
| `rowData(spe)$svg.F` | `adata.var['svg_F']` | SVG F-statistics |
| `rowData(spe)$svg.p` | `adata.var['svg_p']` | SVG p-values |
| `rowData(spe)$svg.fdr` | `adata.var['svg_fdr']` | SVG FDR |
| `reducedDim(spe, "PCA")` | `adata.obsm['PCA']` | PCA coordinates |
| `metadata(spe)$SpaNorm$gmean` | `adata.uns['SpaNorm'].gmean` | Gene means |
| `metadata(spe)$SpaNorm$alpha` | `adata.uns['SpaNorm'].alpha` | Coefficients |
| `metadata(spe)$SpaNorm$psi` | `adata.uns['SpaNorm'].psi` | Dispersion |
| `metadata(spe)$SpaNorm$W` | `adata.uns['SpaNorm'].W` | Covariate matrix |

---

## Algorithm

SpaNorm normalizes spatial transcriptomics data by:

1. **Thin-plate spline basis**: Constructs tensor product of natural cubic splines for 2D spatial coordinates
2. **Covariate matrix W**: Combines library size, biology, and batch effects
3. **NB model fitting**: IRLS (Iteratively Reweighted Least Squares) with dispersion estimation
4. **Normalization**: Removes library size effects while retaining biological signal
5. **SVG calling**: Likelihood ratio test comparing full model vs technical-only model
6. **GLM-PCA**: PCA on deviance or Pearson residuals from the technical model

### Normalization Methods

| Method | Formula | Use case |
|---|---|---|
| `logpac` | `log2(qnbinom(p, mu=bio) + 1)` | Default, preserves count distribution |
| `pearson` | `(Y - mu_nonbio) / sqrt(mu + mu^2 * psi)` | Pearson residuals |
| `medbio` | `log2(qnbinom(0.5, mu=bio) + 1)` | Median biology |
| `meanbio` | `log2(mu_bio)` | Mean biology (fastest) |

---

## License

GPL-3.0-or-later (matching upstream R package)

## Citation

Bhuva DD, Salim A, Mohamed A. SpaNorm: Spatially-aware normalisation for spatial transcriptomics data. Bioconductor.

## Links

- **PyPI**: https://pypi.org/project/py-spanorm/
- **GitHub**: https://github.com/omicverse/py-spanorm
- **Upstream R**: https://github.com/bhuvad/SpaNorm
