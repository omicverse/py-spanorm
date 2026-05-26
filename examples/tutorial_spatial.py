"""
Notebook 2: py-SpaNorm Tutorial
================================
Python-only tutorial demonstrating the full SpaNorm workflow.
"""

# %% [markdown]
# # py-SpaNorm Tutorial
#
# This notebook demonstrates how to use py-SpaNorm for spatial transcriptomics
# normalization, SVG detection, and PCA.

# %%
import numpy as np
import anndata as ad
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# %% [markdown]
# ## 1. Create Synthetic Spatial Data

# %%
np.random.seed(42)
ngenes, ncells = 100, 500

coords = np.column_stack([
    np.random.uniform(0, 10, ncells),
    np.random.uniform(0, 10, ncells),
])

# Generate counts with spatial patterns
base_expr = np.random.uniform(2, 8, ngenes)
mu = np.outer(base_expr, np.ones(ncells))

x_norm = (coords[:,0] - coords[:,0].min()) / (coords[:,0].max() - coords[:,0].min())
y_norm = (coords[:,1] - coords[:,1].min()) / (coords[:,1].max() - coords[:,1].min())

for g in range(30):
    pattern = np.sin(x_norm * np.pi * (g % 3 + 1)) * np.cos(y_norm * np.pi * (g % 2 + 1))
    mu[g] *= 1 + 2 * (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-10)

mu *= np.random.lognormal(0, 0.3, ncells)[np.newaxis, :]
counts = np.zeros((ngenes, ncells), dtype=np.float64)
for g in range(ngenes):
    psi = np.random.uniform(0.1, 0.4)
    counts[g] = np.random.negative_binomial(1/psi, (1/psi) / (1/psi + mu[g]))

# Create AnnData
adata = ad.AnnData(X=counts.T)
adata.obsm['spatial'] = coords
adata.var.index = [f'gene_{i}' for i in range(ngenes)]
adata.obs.index = [f'cell_{i}' for i in range(ncells)]
adata.layers['counts'] = counts.T

print(f"Data: {adata.shape[0]} cells x {adata.shape[1]} genes")

# %% [markdown]
# ## 2. Normalize with SpaNorm

# %%
from spanorm import spanorm

adata = spanorm(adata, sample_p=0.25, df_tps=6, tol=1e-4, verbose=True)

print(f"\nNormalized data shape: {adata.layers['logcounts'].shape}")
print(f"Logcounts range: [{adata.layers['logcounts'].min():.2f}, {adata.layers['logcounts'].max():.2f}]")

# %% [markdown]
# ## 3. Find Spatially Variable Genes

# %%
from spanorm import spanorm_svg

adata = spanorm_svg(adata, verbose=True)

# Show top SVGs
top_svgs = adata.var.nsmallest(10, 'svg_fdr')[['svg_F', 'svg_p', 'svg_fdr']]
print("\nTop 10 SVGs:")
print(top_svgs)

# %% [markdown]
# ## 4. GLM-based PCA

# %%
from spanorm import spanorm_pca

adata = spanorm_pca(adata, n_svgs=50, n_components=20)

print(f"PCA shape: {adata.obsm['PCA'].shape}")
print(f"Variance explained (top 5): {adata.uns['PCA_percentVar'][:5]}")

# %% [markdown]
# ## 5. Class-based API

# %%
from spanorm import SpaNorm

adata2 = adata.copy()
sn = SpaNorm(adata2)
sn.normalize(sample_p=0.25, df_tps=6, verbose=False)
sn.find_svgs(verbose=False)
sn.pca(n_svgs=50, n_components=20)

print(f"Fit: {sn.fit.ngenes} genes, {sn.fit.ncells} cells")
print(f"Model: {sn.fit.gene_model}")
