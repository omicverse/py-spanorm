"""SpaNormFit dataclass — stores the fitted SpaNorm model."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpaNormFit:
    """Stores a SpaNorm model fit, mirroring the R SpaNormFit S4 class."""

    ngenes: int
    ncells: int
    gene_model: str
    df_tps: np.ndarray  # length-4 int array [df_bio_x, df_bio_y, df_ls_x, df_ls_y]
    sample_p: float
    lambda_a: np.ndarray  # length-2 float array [lambda_bio, lambda_ls]
    batch: Optional[np.ndarray]
    W: np.ndarray  # (ncells, n_covariates) covariate matrix
    alpha: np.ndarray  # (ngenes, n_covariates) coefficient matrix
    gmean: np.ndarray  # (ngenes,) gene means
    psi: np.ndarray  # (ngenes,) over-dispersion parameters
    wtype: np.ndarray  # (n_covariates,) covariate types: "biology", "ls", "batch"
    loglik: np.ndarray  # log-likelihood values per iteration
    sampling: np.ndarray  # (ncells,) sampling labels: "all", "glm", "dispersion"

    def __post_init__(self):
        self.df_tps = np.asarray(self.df_tps, dtype=np.int64)
        self.lambda_a = np.asarray(self.lambda_a, dtype=np.float64)
        self.gmean = np.asarray(self.gmean, dtype=np.float64)
        self.psi = np.asarray(self.psi, dtype=np.float64)
        self.alpha = np.asarray(self.alpha, dtype=np.float64)
        self.W = np.asarray(self.W, dtype=np.float64)
        self.wtype = np.asarray(self.wtype, dtype="<U10")
        self.sampling = np.asarray(self.sampling, dtype="<U10")
        self.loglik = np.asarray(self.loglik, dtype=np.float64)

        if self.batch is not None:
            self.batch = np.asarray(self.batch)

        self._validate()

    def _validate(self):
        if any(self.df_tps <= 0):
            raise ValueError("'df_tps' should be greater than 0")
        if self.gene_model not in ("nb",):
            raise ValueError(f"'gene_model' should be one of: nb")
        if self.sample_p <= 0 or self.sample_p > 1:
            raise ValueError("'sample_p' should be in the interval (0,1]")
        if any(self.lambda_a < 0):
            raise ValueError("'lambda_a' should be positive")
        if len(self.loglik) > 0 and any(self.loglik > 0):
            raise ValueError("'loglik' should be less than or equal to 0")

        # Dimension checks
        if self.alpha.shape[0] != self.ngenes:
            raise ValueError("nrow of 'alpha' does not match 'ngenes'")
        if self.W.shape[0] != self.ncells:
            raise ValueError("nrow of 'W' does not match 'ncells'")
        if self.alpha.shape[1] != self.W.shape[1]:
            raise ValueError("ncol of 'alpha' and 'W' do not match")
        if len(self.gmean) != self.ngenes:
            raise ValueError("length of 'gmean' does not match 'ngenes'")
        if len(self.psi) != self.ngenes:
            raise ValueError("length of 'psi' does not match 'ngenes'")
        if len(self.wtype) != self.W.shape[1]:
            raise ValueError("length of 'wtype' does not match ncol of 'W'")
        if len(self.sampling) != self.ncells:
            raise ValueError("length of 'sampling' does not match 'ncells'")

    def has_biology(self):
        return np.any(self.wtype == "biology")
