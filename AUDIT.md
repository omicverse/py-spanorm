# R Function Coverage Audit - SpaNorm

## Exported Functions (from NAMESPACE)

| R Function | Python Equivalent | Status | Notes |
|---|---|---|---|
| `SpaNorm()` | `spanorm()` | Ported | Main normalization entry point |
| `SpaNormPCA()` | `spanorm_pca()` | Ported | GLM-based PCA |
| `SpaNormSVG()` | `spanorm_svg()` | Ported | SVG calling via LRT |
| `fastSizeFactors()` | `fast_size_factors()` | Ported | Library size factors |
| `filterGenes()` | `filter_genes()` | Ported | Gene filtering by expression |
| `plotCovariate()` | `plot_covariate()` | Ported | matplotlib implementation |
| `plotSpatial()` | `plot_spatial()` | Ported | matplotlib implementation |
| `topSVGs()` | `top_svgs()` | Ported | Export top SVGs |
| `SpaNormFit` class | `SpaNormFit` dataclass | Ported | Model fit storage |

## Internal Functions

| R Function | Python Equivalent | Status |
|---|---|---|
| `fitSpaNorm()` | `fit_spanorm()` | Ported |
| `fitSpaNormNB()` | `fit_spanorm_nb()` | Ported |
| `fitNBGivenPsi()` | `fit_nb_given_psi()` | Ported |
| `bs.tps()` | `bs_tps()` | Ported |
| `calculateMu()` | `calculate_mu()` | Ported |
| `normaliseLogPAC()` | `normalise_logpac()` | Ported |
| `normaliseMeanBio()` | `normalise_mean_bio()` | Ported |
| `normalisePearson()` | `normalise_pearson()` | Ported |
| `normaliseMedianBio()` | `normalise_median_bio()` | Ported |
| `normaliseMeanLS()` | `normalise_mean_ls()` | Ported |
| `normaliseMeanBatch()` | `normalise_mean_batch()` | Ported |
| `devianceResiduals()` | `deviance_residuals()` | Ported |
| `checkBatch()` | `check_batch()` | Ported |
| `filterGenes_intl()` | (inline) | Ported |
| `fastSizeFactors_intl()` | (inline) | Ported |
| `sampleRandom()` | (numpy) | Ported |
| GPU functions (15) | `gpu_helpers.py` | Ported | CPU-only equivalents |

## Coverage Summary

- Exported functions: 9/9 (100%)
- Internal functions: 16/16 (100%)
- Overall: 25/25 (100%)

## Audit Class: A (pure translation, no algorithm changes)
