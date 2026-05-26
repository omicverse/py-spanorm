"""SpaNorm - Spatially-aware normalisation for spatial transcriptomics data.

Python port of the R/Bioconductor package SpaNorm.
"""

from .fit import SpaNormFit
from .preprocessing import filter_genes, fast_size_factors
from .normalization import spanorm, fit_spanorm
from .svg import spanorm_svg, top_svgs
from .pca import spanorm_pca
from .core import SpaNorm
from .plotting import plot_spatial, plot_covariate
from . import gpu_helpers

__all__ = [
    "SpaNorm",
    "SpaNormFit",
    "filter_genes",
    "fast_size_factors",
    "spanorm",
    "fit_spanorm",
    "spanorm_svg",
    "top_svgs",
    "spanorm_pca",
    "plot_spatial",
    "plot_covariate",
    "gpu_helpers",
]
__version__ = "0.1.0"
