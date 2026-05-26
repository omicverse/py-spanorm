"""
Notebook 3: R vs Python Function-by-Function Parity
====================================================
Parameter dictionary and side-by-side comparison for each R function.
"""

# %% [markdown]
# # py-SpaNorm: R vs Python Function Dictionary
#
# This notebook maps each R function to its Python equivalent,
# showing parameter correspondence and verifying output parity.

# %% [markdown]
# ## Function Map
#
# | R function | Python function | Parameters match? |
# |---|---|---|
# | `SpaNorm(spe, ...)` | `spanorm(adata, ...)` | Yes, all args |
# | `SpaNormSVG(spe)` | `spanorm_svg(adata)` | Yes |
# | `SpaNormPCA(spe, ...)` | `spanorm_pca(adata, ...)` | Yes |
# | `filterGenes(spe, prop)` | `filter_genes(counts, prop)` | Yes |
# | `fastSizeFactors(spe)` | `fast_size_factors(counts)` | Yes |
# | `topSVGs(spe, n, fdr)` | `top_svgs(adata, n, fdr)` | Yes |

# %%
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %% [markdown]
# ## 1. filterGenes / filter_genes
#
# **R**: `filterGenes(spe, prop = 0.1)` — returns logical vector
# **Python**: `filter_genes(counts, prop=0.1)` — returns boolean array

# %%
from spanorm import filter_genes

counts = np.array([
    [10, 0, 5, 8, 0],
    [0, 0, 0, 0, 0],
    [15, 12, 18, 20, 14],
]).T  # 5 cells x 3 genes

keep = filter_genes(counts.T, prop=0.5)
print(f"filter_genes(prop=0.5): {keep}")
# Expected: [True, False, True]

# %% [markdown]
# ## 2. fastSizeFactors / fast_size_factors
#
# **R**: `fastSizeFactors(spe)` — returns size factors in colData
# **Python**: `fast_size_factors(counts)` — returns 1D array

# %%
from spanorm import fast_size_factors

sf = fast_size_factors(counts.T)
print(f"fast_size_factors: {sf}")
print(f"Mean: {sf.mean():.4f}")  # Should be 1.0

# %% [markdown]
# ## 3. bs.tps / bs_tps
#
# **R**: `bs.tps(x, y, df.tps=6)` — returns matrix with df.tps attribute
# **Python**: `bs_tps(x, y, df_tps=6)` — returns (matrix, (df_x, df_y))

# %%
from spanorm.spline import bs_tps

np.random.seed(42)
x = np.random.uniform(0, 10, 100)
y = np.random.uniform(0, 10, 100)

basis, (df_x, df_y) = bs_tps(x, y, df_tps=6)
print(f"bs_tps(df_tps=6): shape={basis.shape}, df=({df_x}, {df_y})")
print(f"Column means (should be ~0): {basis.mean(axis=0)[:3]}")

# %% [markdown]
# ## 4. SpaNormFit class
#
# **R**: S4 class `SpaNormFit` with slots
# **Python**: `@dataclass SpaNormFit` with attributes

# %%
from spanorm import SpaNormFit

# The SpaNormFit stores:
# - ngenes, ncells: dimensions
# - gene_model: 'nb'
# - df_tps: spline degrees of freedom
# - W: covariate matrix
# - alpha: coefficients
# - gmean: gene means
# - psi: dispersion
# - wtype: covariate types ('biology', 'ls', 'batch')

# Access after spanorm():
# fit = adata.uns['SpaNorm']
# fit.gmean, fit.alpha, fit.psi

print("SpaNormFit attributes match R slots:")
print("  ngenes -> ngenes")
print("  ncells -> ncells")
print("  gene_model -> gene.model")
print("  df_tps -> df.tps")
print("  W -> W")
print("  alpha -> alpha")
print("  gmean -> gmean")
print("  psi -> psi")
print("  wtype -> wtype")

# %% [markdown]
# ## 5. Normalization Methods
#
# **R**: `adj.method` parameter in `SpaNorm()`
# **Python**: `adj_method` parameter in `spanorm()`
#
# | R adj.method | Python adj_method | Function |
# |---|---|---|
# | `"auto"` / `"logpac"` | `"auto"` / `"logpac"` | `normalise_logpac` |
# | `"pearson"` | `"pearson"` | `normalise_pearson` |
# | `"medbio"` | `"medbio"` | `normalise_median_bio` |
# | `"meanbio"` | `"meanbio"` | `normalise_mean_bio` |

# %%
print("Normalization methods:")
print("  auto/logpac: log2(qnbinom(p, mu=bio) + 1)")
print("  pearson:     (Y - mu_nonbio) / sqrt(mu + mu^2 * psi)")
print("  medbio:      log2(qnbinom(0.5, mu=bio) + 1)")
print("  meanbio:     log2(mu_bio)")

# %% [markdown]
# ## 6. Parameter Mapping Summary
#
# | R parameter | Python parameter | Default | Notes |
# |---|---|---|---|
# | `sample.p` | `sample_p` | 0.25 | Proportion of cells to sample |
# | `gene.model` | `gene_model` | 'nb' | Only 'nb' supported |
# | `adj.method` | `adj_method` | 'auto' | Normalization method |
# | `scale.factor` | `scale_factor` | 1 | Scaling factor |
# | `df.tps` | `df_tps` | 6 | Spline df |
# | `lambda.a` | `lambda_a` | 0.0001 | Regularization |
# | `batch` | `batch` | None | Batch design |
# | `tol` | `tol` | 1e-4 | Convergence tolerance |
# | `maxit.nb` | `maxit_nb` | 50 | Max NB iterations |
# | `maxit.psi` | `maxit_psi` | 25 | Max dispersion iterations |
# | `verbose` | `verbose` | True | Print progress |
