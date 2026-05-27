"""Optimized IRLS for SpaNorm NB model fitting."""

import numpy as np
from scipy.linalg import solve as sp_solve
from scipy.special import gammaln
from scipy.stats import median_abs_deviation


def dnbinom_log_fast(Y, mu, size):
    """Fast NB log-PMF. size = 1/psi, shape (ngenes,)."""
    k = Y.astype(np.float64)
    n = size[:, None]
    mu = np.maximum(mu, 1e-10)
    p = np.clip(n / (n + mu), 1e-15, 1 - 1e-15)
    return gammaln(k + n) - gammaln(k + 1) - gammaln(n) + n * np.log(p) + k * np.log1p(-p)


def poisson_log_fast(Y, lmu, mu):
    """Fast Poisson log-PMF (when psi -> 0)."""
    return Y * lmu - mu - gammaln(Y + 1)


def fit_irls_fast(Y_sub, W_sub, ncells_full, psi=None, max_iter=30, tol=1e-4):
    """Optimized IRLS fitting.

    Parameters
    ----------
    Y_sub : np.ndarray (ngenes, nsub)
        Count matrix (sampled).
    W_sub : np.ndarray (nsub, nW)
        Covariate matrix (sampled).
    ncells_full : int
        Total number of cells (for regularization).
    psi : np.ndarray (ngenes,) or None
        Initial dispersion. If None, uses Poisson approximation.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict
        Fitted model with gmean, alpha, psi, loglik.
    """
    ngenes, nsub = Y_sub.shape
    nW = W_sub.shape[1]

    # Initialize
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((ngenes, nW))
    alpha[:, 0] = 1.0

    # Dispersion
    if psi is None:
        psi = np.full(ngenes, 0.01)
    use_poisson = np.all(psi < 0.05)
    size = 1.0 / np.maximum(psi, 1e-10)

    # Pre-compute gammaln(Y+1) for Poisson
    gammaln_Y1 = None
    if use_poisson:
        gammaln_Y1 = np.sum(gammaln(Y_sub + 1))

    # Pre-compute regularization
    lam = np.full(nW - 1, 0.0001 * ncells_full)
    reg_diag = np.diag(lam)

    # Pre-compute W components
    WtW_cache = None
    loglik_prev = -np.inf

    for iteration in range(max_iter):
        # Forward pass
        lmu = np.clip(gmean[:, None] + alpha @ W_sub.T, -20, 20)
        mu = np.exp(lmu)

        # Log-likelihood (Poisson fast path when psi is small)
        if use_poisson:
            ll = np.sum(Y_sub * lmu) - np.sum(mu) - gammaln_Y1
        else:
            ll = np.sum(dnbinom_log_fast(Y_sub, mu, size))

        # Working response
        Z = lmu + ((Y_sub + 0.01) / (mu + 0.01) - 1)

        # Weights (Poisson: wt = mu; NB: wt = 1/(psi * exp(-lmu)))
        if use_poisson:
            sig_inv = mu
        else:
            sig_inv = 1.0 / (psi[:, None] * np.exp(-lmu))
            sig_inv = np.clip(sig_inv, 0, 1e6)
        wt = np.mean(sig_inv, axis=0)
        wt = np.minimum(wt, np.quantile(wt, 0.98))

        # Alpha update with SpaNorm constraint
        a1 = np.mean(alpha[:, 0])
        Wa1 = a1 * W_sub[:, 0:1]
        W_rest = W_sub[:, 1:]
        Z_adj = Z - gmean[:, None] - Wa1.T

        b_rest = (Z_adj * wt[None, :]) @ W_rest
        WtW_rest = W_rest.T @ (wt[:, None] * W_rest)

        alpha_rest = sp_solve(WtW_rest + reg_diag, b_rest.T, assume_a='pos').T
        alpha = np.column_stack([np.full(ngenes, a1), alpha_rest])

        # Update gmean
        Z_res = Z - alpha @ W_sub.T
        gmean = np.sum(Z_res * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        # Convergence check
        if iteration > 0 and abs(ll - loglik_prev) / max(abs(loglik_prev), 1) < tol:
            break
        loglik_prev = ll

    return {
        'gmean': gmean,
        'alpha': alpha,
        'psi': psi,
        'loglik': loglik_prev,
        'iterations': iteration + 1,
    }
