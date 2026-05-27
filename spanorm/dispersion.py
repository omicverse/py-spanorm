"""EdgeR-like dispersion estimation using Cox-Reid adjusted profile likelihood."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import median_abs_deviation


def estimate_dispersion_coxreid(Y, mu, offset=None, trended=True):
    """Estimate gene-wise dispersion using Cox-Reid adjusted profile likelihood.

    Approximates edgeR's estimateDisp approach.

    Parameters
    ----------
    Y : np.ndarray (ngenes, ncells)
        Count matrix.
    mu : np.ndarray (ngenes, ncells)
        Fitted mean values.
    offset : np.ndarray (ngenes, ncells) or None
        Offset for NB model.
    trended : bool
        Whether to use trended dispersion (dispersion varies with mean expression).

    Returns
    -------
    np.ndarray (ngenes,)
        Estimated dispersion for each gene.
    """
    ngenes, ncells = Y.shape

    # Gene-wise mean expression
    gene_mean = np.mean(mu, axis=1)
    log_gene_mean = np.log(np.maximum(gene_mean, 1e-10))

    # Initial method-of-moments estimate
    Y_var = np.var(Y, axis=1, ddof=1)
    Y_mean = np.mean(Y, axis=1)
    psi_mom = np.maximum((Y_var - Y_mean) / np.maximum(Y_mean ** 2, 1e-10), 1e-4)

    # For each gene, optimize dispersion using Cox-Reid adjusted likelihood
    psi_cr = np.zeros(ngenes)

    for g in range(ngenes):
        y_g = Y[g]
        mu_g = mu[g]

        # Cox-Reid adjusted log-likelihood for dispersion
        def neg_cr_loglik(log_psi):
            psi = np.exp(log_psi)
            size = 1.0 / psi

            # NB log-likelihood
            from scipy.special import gammaln, xlogy, xlog1py
            p = size / (size + mu_g)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            ll = np.sum(gammaln(y_g + size) - gammaln(y_g + 1) - gammaln(size)
                       + xlogy(size, p) + xlog1py(y_g, -p))

            # Cox-Reid adjustment: -0.5 * log(sum(mu^2 / (mu + psi*mu^2)))
            # Simplified: -0.5 * log(sum(1 / (1/psi + mu)))
            adj = 0.5 * np.log(np.sum(1.0 / (1.0 / psi + mu_g)) + 1e-10)

            return -(ll - adj)

        # Optimize in log-space
        log_psi_init = np.log(psi_mom[g])
        try:
            result = minimize_scalar(neg_cr_loglik,
                                    bounds=(log_psi_init - 5, log_psi_init + 5),
                                    method='bounded')
            psi_cr[g] = np.exp(result.x)
        except Exception:
            psi_cr[g] = psi_mom[g]

    # Ensure positive
    psi_cr = np.maximum(psi_cr, 1e-6)

    if trended:
        # Fit trend: dispersion vs log(mean expression)
        # Use loess-like smoothing (simplified: polynomial fit)
        valid = np.isfinite(log_gene_mean) & np.isfinite(np.log(psi_cr))
        if np.sum(valid) > 10:
            coeffs = np.polyfit(log_gene_mean[valid], np.log(psi_cr[valid]), deg=2)
            psi_trend = np.exp(np.polyval(coeffs, log_gene_mean))

            # Shrink toward trend (empirical Bayes)
            # Prior degrees of freedom (simplified)
            prior_n = 10
            psi_shrunk = (prior_n * psi_trend + ncells * psi_cr) / (prior_n + ncells)
            return psi_shrunk

    return psi_cr


def estimate_dispersion_simple(Y, mu):
    """Simple dispersion estimation (method-of-moments with shrinkage).

    Parameters
    ----------
    Y : np.ndarray (ngenes, ncells)
        Count matrix.
    mu : np.ndarray (ngenes, ncells)
        Fitted mean values.

    Returns
    -------
    np.ndarray (ngenes,)
        Estimated dispersion.
    """
    ngenes, ncells = Y.shape

    # Method of moments
    Y_var = np.var(Y, axis=1, ddof=1)
    Y_mean = np.mean(Y, axis=1)
    psi_mom = np.maximum((Y_var - Y_mean) / np.maximum(Y_mean ** 2, 1e-10), 1e-4)

    # Shrink toward common dispersion (like edgeR's estimateGLMTagwiseDisp)
    common_psi = np.median(psi_mom)
    prior_n = 10
    psi_shrunk = (prior_n * common_psi + ncells * psi_mom) / (prior_n + ncells)

    return np.maximum(psi_shrunk, 1e-6)
