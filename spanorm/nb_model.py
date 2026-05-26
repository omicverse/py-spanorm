"""Negative binomial model fitting for SpaNorm.

Ports fitSpaNormNB, fitNBGivenPsi, calculateMu, and normalization methods.
"""

import numpy as np
from scipy import sparse, stats
from scipy.special import gammaln, xlogy, xlog1py
from scipy.linalg import solve as sp_solve
import warnings


def calculate_mu(gmean, alpha, W):
    """Calculate mu = exp(gmean + alpha @ W.T), winsorized.

    Parameters
    ----------
    gmean : np.ndarray
        Gene means (ngenes,).
    alpha : np.ndarray
        Coefficient matrix (ngenes, n_covariates).
    W : np.ndarray
        Covariate matrix (ncells, n_covariates).

    Returns
    -------
    np.ndarray
        mu matrix (ngenes, ncells).
    """
    # log(mu) = gmean + alpha @ W.T
    lmu = gmean[:, np.newaxis] + alpha @ W.T

    # Winsorize: cap at median + 4 * MAD per gene
    lmu_median = np.median(lmu, axis=1)
    lmu_mad = stats.median_abs_deviation(lmu, axis=1, nan_policy='omit')
    # Handle case where MAD is 0 (all same values)
    lmu_mad = np.where(np.isnan(lmu_mad) | (lmu_mad == 0), 1.0, lmu_mad)
    lmu_max = lmu_median + 4 * lmu_mad
    lmu = np.minimum(lmu, lmu_max[:, np.newaxis])

    # Clamp to prevent overflow in exp
    lmu = np.clip(lmu, -50, 50)
    mu = np.exp(lmu)
    return mu


def dnbinom_log(y, mu, psi):
    """Log density of negative binomial distribution (vectorized).

    Parameters
    ----------
    y : np.ndarray
        Count data (ngenes, ncells).
    mu : np.ndarray
        Mean parameter (ngenes, ncells).
    psi : np.ndarray
        Overdispersion parameter (ngenes,).

    Returns
    -------
    np.ndarray
        Log density (ngenes, ncells).
    """
    psi = np.maximum(psi, 1e-10)
    size = 1.0 / psi  # (ngenes,)

    k = y.astype(np.int64)
    n = size[:, np.newaxis]  # (ngenes, 1)
    mu = np.maximum(mu, 1e-10)
    p = n / (n + mu)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    return gammaln(k + n) - gammaln(k + 1) - gammaln(n) + xlogy(n, p) + xlog1py(k, -p)


def pnbinom_cdf(y, mu, psi):
    """CDF of negative binomial distribution.

    Parameters
    ----------
    y : np.ndarray
        Count data (ngenes, ncells).
    mu : np.ndarray
        Mean parameter (ngenes, ncells).
    psi : np.ndarray
        Overdispersion parameter (ngenes,).

    Returns
    -------
    np.ndarray
        CDF values (ngenes, ncells).
    """
    size = 1.0 / psi
    result = np.zeros_like(y, dtype=np.float64)
    for g in range(y.shape[0]):
        result[g] = stats.nbinom.cdf(
            y[g].astype(np.int64),
            n=size[g],
            p=size[g] / (size[g] + mu[g])
        )
    return result


def qnbinom_quantile(p, mu, psi):
    """Quantile function of negative binomial distribution.

    Parameters
    ----------
    p : np.ndarray
        Probability values (ngenes, ncells).
    mu : np.ndarray
        Mean parameter (ngenes, ncells).
    psi : np.ndarray
        Overdispersion parameter (ngenes,).

    Returns
    -------
    np.ndarray
        Quantile values (ngenes, ncells).
    """
    size = 1.0 / psi
    result = np.zeros_like(p, dtype=np.float64)
    for g in range(p.shape[0]):
        result[g] = stats.nbinom.ppf(
            p[g],
            n=size[g],
            p=size[g] / (size[g] + mu[g])
        )
    return result


def dnbinom_pmf(y, mu, psi):
    """PMF of negative binomial distribution.

    Parameters
    ----------
    y : np.ndarray
        Count data (ngenes, ncells).
    mu : np.ndarray
        Mean parameter (ngenes, ncells).
    psi : np.ndarray
        Overdispersion parameter (ngenes,).

    Returns
    -------
    np.ndarray
        PMF values (ngenes, ncells).
    """
    size = 1.0 / psi
    result = np.zeros_like(y, dtype=np.float64)
    for g in range(y.shape[0]):
        result[g] = stats.nbinom.pmf(
            y[g].astype(np.int64),
            n=size[g],
            p=size[g] / (size[g] + mu[g])
        )
    return result


def _estimate_dispersion(Y_sub_psi, offset, maxn_psi):
    """Estimate gene-wise dispersion using edgeR-like approach.

    This is a simplified version that uses scipy optimization
    to approximate edgeR's estimateDisp.

    Parameters
    ----------
    Y_sub_psi : np.ndarray
        Count matrix for dispersion estimation (ngenes, nsub_psi).
    offset : np.ndarray
        Offset matrix (ngenes, nsub_psi).
    maxn_psi : int
        Maximum number of cells for dispersion estimation.

    Returns
    -------
    np.ndarray
        Estimated dispersion for each gene (ngenes,).
    """
    ngenes = Y_sub_psi.shape[0]
    psi = np.zeros(ngenes)

    for g in range(ngenes):
        y_g = Y_sub_psi[g].astype(np.float64)
        mu_g = np.exp(offset[g])

        # Method of moments initial estimate
        var_g = np.var(y_g, ddof=1)
        mean_g = np.mean(mu_g)

        if mean_g <= 0:
            psi[g] = np.nan
            continue

        # Method of moments: Var = mu + psi * mu^2
        # psi = (var - mu) / mu^2
        psi_init = max((var_g - mean_g) / (mean_g ** 2), 1e-6)

        # Newton-Raphson refinement for NB dispersion
        # Maximize: sum_i log(NB(y_i | mu_i, psi))
        # This is equivalent to edgeR's estimateDisp approach
        psi[g] = _refine_dispersion(y_g, mu_g, psi_init)

    return psi


def _refine_dispersion(y, mu, psi_init, max_iter=20, tol=1e-4):
    """Refine dispersion estimate using Fisher scoring.

    Parameters
    ----------
    y : np.ndarray
        Count data.
    mu : np.ndarray
        Mean parameters.
    psi_init : float
        Initial dispersion estimate.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    float
        Refined dispersion estimate.
    """
    psi = psi_init
    size = 1.0 / psi

    for _ in range(max_iter):
        # NB log-likelihood gradient w.r.t. psi
        # d/dpsi log NB = sum_i [digamma(y_i + size) - digamma(size) + log(size) + 1
        #                        - log(mu_i + size) - (y_i + size)/(mu_i + size)]
        # where size = 1/psi

        # Use chain rule: d/dpsi = d/d(size) * d(size)/d(psi) = -size^2 * d/d(size)
        size = 1.0 / psi

        # Compute gradient and info w.r.t. size
        s_plus_y = size + y
        s_plus_mu = size + mu

        # Gradient of log-lik w.r.t. size
        grad_size = (np.sum(_digamma(s_plus_y) - _digamma(size)
                           + np.log(size / s_plus_mu)
                           + 1 - s_plus_y / s_plus_mu))

        # Fisher information w.r.t. size
        info_size = np.sum(_trigamma(s_plus_y) - _trigamma(size)
                          - 1 / s_plus_mu + y / (s_plus_mu ** 2))

        # Convert to psi parameterization
        grad_psi = -size ** 2 * grad_size
        info_psi = size ** 4 * info_size

        if abs(info_psi) < 1e-14:
            break

        # Fisher scoring update
        delta = grad_psi / info_psi
        psi_new = psi + delta

        # Ensure positive
        if psi_new <= 0:
            psi_new = psi / 2

        if abs(psi_new - psi) / max(abs(psi), 1e-10) < tol:
            psi = psi_new
            break

        psi = psi_new

    return max(psi, 1e-10)


def _digamma(x):
    """Digamma function (derivative of log-gamma)."""
    from scipy.special import digamma as sp_digamma
    return sp_digamma(x)


def _trigamma(x):
    """Trigamma function (second derivative of log-gamma)."""
    from scipy.special import polygamma
    return polygamma(1, x)


def fit_nb_given_psi(Y_sub, W_sub, psi, lambda_a, gmean=None, alpha=None,
                     step_factor=0.5, maxit_nb=50, tol=1e-4, loglik=None,
                     is_spanorm=False):
    """Fit NB model given dispersion estimates (IRLS).

    Parameters
    ----------
    Y_sub : np.ndarray
        Count matrix (ngenes, nsub).
    W_sub : np.ndarray
        Covariate matrix (nsub, n_covariates).
    psi : np.ndarray
        Dispersion estimates (ngenes,).
    lambda_a : np.ndarray
        Regularization parameters for each covariate (n_covariates-1,).
    gmean : np.ndarray, optional
        Initial gene means.
    alpha : np.ndarray, optional
        Initial coefficients.
    step_factor : float
        Step size reduction factor on divergence.
    maxit_nb : int
        Maximum IRLS iterations.
    tol : float
        Convergence tolerance.
    loglik : np.ndarray, optional
        Pre-computed log-likelihoods.
    is_spanorm : bool
        Whether this is a SpaNorm model (first column of alpha is shared).

    Returns
    -------
    dict
        Fitted model with keys: gmean, alpha, loglik, loglik_iter.
    """
    nsub = Y_sub.shape[1]
    nW = W_sub.shape[1]
    ngenes = Y_sub.shape[0]

    # Initialize parameters
    psi = np.maximum(psi, 1e-10)
    if gmean is None:
        gmean = np.mean(np.log(Y_sub + 1), axis=1)
    if alpha is None:
        alpha = np.zeros((ngenes, nW))
        alpha[:, 0] = 1.0

    # Initial log-likelihood
    if loglik is None:
        lmu_hat = gmean[:, np.newaxis] + alpha @ W_sub.T
        lmu_hat = np.clip(lmu_hat, -50, 50)
        sig_inv = 1.0 / (psi[:, np.newaxis] * np.exp(-lmu_hat))
        sig_inv = np.clip(sig_inv, 0, 1e10)
        loglik = np.sum(dnbinom_log(Y_sub, np.exp(lmu_hat), psi), axis=1)

    # Step sizes (per-gene for IRLS step halving)
    step = np.ones(ngenes)

    # Convergence tracking
    logl_beta = [np.sum(loglik)]
    halving = 0
    converged = False
    iteration = 1

    lambda_diag = np.diag(lambda_a) if len(lambda_a) > 1 else lambda_a[0] * np.eye(nW - 1)

    while not converged:
        # Save best estimates
        best_gmean = gmean.copy()
        best_alpha = alpha.copy()

        # Working vector Z
        lmu_hat = gmean[:, np.newaxis] + alpha @ W_sub.T
        Z = lmu_hat + ((Y_sub + 0.01) / (np.exp(lmu_hat) + 0.01) - 1) * step[:, np.newaxis]

        # Save current alpha
        alpha_old = alpha.copy()

        # Update alpha for all genes
        lmu_hat_clipped = np.clip(lmu_hat, -50, 50)
        sig_inv = 1.0 / (psi[:, np.newaxis] * np.exp(-lmu_hat_clipped))
        sig_inv = np.clip(sig_inv, 0, 1e10)
        wt_cell = np.mean(sig_inv, axis=0)
        # Prevent outlier large weights
        wt_cell = np.minimum(wt_cell, np.quantile(wt_cell, 0.98))

        # Solve weighted least squares
        # b = (Z - gmean) @ diag(wt) @ W
        # alpha = b @ inv(W.T @ diag(wt) @ W + lambda)
        Z_centered = Z - gmean[:, np.newaxis]
        b = (Z_centered * wt_cell[np.newaxis, :]) @ W_sub

        WtW = W_sub.T @ (wt_cell[:, np.newaxis] * W_sub)

        if is_spanorm and nW > 1:
            # Set first column of alpha to be the same for all genes
            a1_mean = np.mean(alpha[:, 0])
            alpha[:, 0] = a1_mean

            # Recompute without first column
            Wa1 = a1_mean * W_sub[:, 0]
            W_rest = W_sub[:, 1:]
            b_rest = ((Z_centered - Wa1[np.newaxis, :]) * wt_cell[np.newaxis, :]) @ W_rest
            WtW_rest = W_rest.T @ (wt_cell[:, np.newaxis] * W_rest)

            # Regularize and solve
            if lambda_diag.ndim == 1:
                reg = np.diag(lambda_diag)
            else:
                reg = lambda_diag

            try:
                alpha_rest = sp_solve(WtW_rest + reg, b_rest.T, assume_a='pos').T
            except np.linalg.LinAlgError:
                alpha_rest = b_rest @ np.linalg.inv(WtW_rest + reg)
            alpha = np.column_stack([alpha[:, 0:1], alpha_rest])
        else:
            # Standard update with regularization
            reg_mat = np.zeros((nW, nW))
            if nW > 1:
                if lambda_diag.ndim == 1:
                    reg_mat[1:, 1:] = np.diag(lambda_diag)
                else:
                    reg_mat[1:, 1:] = lambda_diag

            try:
                alpha = sp_solve(WtW + reg_mat, b.T, assume_a='pos').T
            except np.linalg.LinAlgError:
                alpha = b @ np.linalg.inv(WtW + reg_mat)

        # Check for NaN/Inf
        if np.any(np.isnan(alpha)) or np.any(np.isinf(alpha)):
            alpha = alpha_old

        # Reduce outliers (winsorize at median ± 4*MAD)
        for j in range(alpha.shape[1]):
            col = alpha[:, j]
            med = np.median(col)
            mad = stats.median_abs_deviation(col)
            a_max = med + 4 * mad
            a_min = med - 4 * mad
            alpha[:, j] = np.clip(col, a_min, a_max)

        # Update gmean
        Z_res = Z - alpha @ W_sub.T
        gmean = np.sum(Z_res * sig_inv, axis=1) / np.sum(sig_inv, axis=1)

        # Calculate current log-likelihood
        lmu_hat = gmean[:, np.newaxis] + alpha @ W_sub.T
        loglik_tmp = np.sum(dnbinom_log(Y_sub, np.exp(lmu_hat), psi), axis=1)

        # Check degenerate case
        degener = np.sum(loglik) > np.sum(loglik_tmp)
        if np.isnan(degener):
            degener = True

        if degener:
            # Reduce step size for genes that got worse
            check_gene = loglik > loglik_tmp
            step[check_gene] *= step_factor
            # Revert
            gmean = best_gmean
            alpha = best_alpha
            halving += 1
            if halving >= 3:
                loglik_tmp = loglik
                degener = False

        if not degener:
            loglik = loglik_tmp
            logl_beta.insert(0, np.sum(loglik))
            iteration += 1
            halving = 0

        # Check convergence
        conv_logl = False
        if len(logl_beta) > 1:
            conv_logl = (logl_beta[0] - logl_beta[1]) / abs(logl_beta[1]) < tol

        converged = conv_logl or iteration > maxit_nb

    return {
        'gmean': gmean,
        'alpha': alpha,
        'loglik': loglik,
        'loglik_iter': logl_beta,
    }


def fit_spanorm_nb(Y, W, idx, maxit_psi=25, tol=1e-4, maxn_psi=500,
                   lambda_a=None, is_spanorm=False):
    """Fit the full SpaNorm NB model with dispersion estimation loop.

    Parameters
    ----------
    Y : np.ndarray or sparse matrix
        Count matrix (ngenes, ncells).
    W : np.ndarray
        Covariate matrix (ncells, n_covariates).
    idx : np.ndarray
        Boolean mask for cells used in fitting.
    maxit_psi : int
        Maximum outer iterations for dispersion estimation.
    tol : float
        Convergence tolerance.
    maxn_psi : int
        Maximum cells for dispersion estimation.
    lambda_a : np.ndarray
        Regularization parameters.
    is_spanorm : bool
        Whether this is a SpaNorm model.

    Returns
    -------
    dict
        Fitted model.
    """
    if lambda_a is None:
        lambda_a = np.zeros(W.shape[1] - 1)

    if sparse.issparse(Y):
        Y = np.asarray(Y.todense())

    Y = Y.astype(np.float64)

    # Subset data points used for model fitting
    Y_sub = Y[:, idx]
    W_sub = W[idx, :]
    n_sub = int(np.sum(idx))
    nW = W_sub.shape[1]

    # Initial estimates
    gmean = np.mean(np.log(Y_sub + 1), axis=1)
    alpha = np.zeros((Y_sub.shape[0], nW))
    alpha[:, 0] = 1.0

    # Subset for dispersion estimation
    n_sub_psi = min(maxn_psi, n_sub)
    psi_idx_sub = np.zeros(n_sub, dtype=bool)
    rng = np.random.default_rng(42)
    psi_idx_sub[rng.choice(n_sub, size=n_sub_psi, replace=False)] = True
    Y_sub_psi = Y_sub[:, psi_idx_sub]
    psi = np.zeros(Y_sub.shape[0])

    # Convergence tracking
    logl_psi = []
    iteration = 1
    converged = False

    # Map sampling indices (full length = ncells)
    # idx_sub is the boolean mask of sampled cells
    sampling = np.where(idx, "glm", "all")
    # Among the sampled cells, mark those used for dispersion estimation
    idx_positions = np.where(idx)[0]
    sampling[idx_positions[psi_idx_sub]] = "dispersion"

    # Outer loop: estimate dispersion
    while not converged:
        # Calculate/update dispersion estimates
        Wa = alpha @ W_sub.T
        offs_psi = Wa[:, psi_idx_sub]

        psi_tmp = _estimate_dispersion(Y_sub_psi, offs_psi, n_sub_psi)
        valid_psi = ~np.isnan(psi_tmp) & ~np.isinf(psi_tmp) & (psi_tmp > 0)
        psi[valid_psi] = psi_tmp[valid_psi]
        # Ensure psi is always positive
        psi = np.maximum(psi, 1e-10)

        # Calculate log-likelihood
        lmu_hat = gmean[:, np.newaxis] + alpha @ W_sub.T
        loglik = np.sum(dnbinom_log(Y_sub, np.exp(lmu_hat), psi), axis=1)

        # Fit NB given dispersion
        fit_nb = fit_nb_given_psi(
            Y_sub, W_sub, psi, lambda_a,
            gmean=gmean, alpha=alpha, loglik=loglik,
            is_spanorm=is_spanorm, tol=tol, maxit_nb=50,
            step_factor=0.5
        )
        gmean = fit_nb['gmean']
        alpha = fit_nb['alpha']
        loglik = fit_nb['loglik']

        # Check convergence
        logl_psi.insert(0, np.sum(loglik))
        iteration += 1

        conv_logl = False
        if len(logl_psi) > 1:
            conv_logl = (logl_psi[0] - logl_psi[1]) / abs(logl_psi[1]) < tol

        converged = conv_logl or iteration > maxit_psi

    return {
        'gmean': gmean,
        'alpha': alpha,
        'psi': psi,
        'sampling': sampling,
        'loglik': logl_psi[::-1],
    }


def normalise_logpac(Y, scale_factor, fit):
    """Normalize using log-PAC method.

    Parameters
    ----------
    Y : np.ndarray
        Count matrix (ngenes, ncells).
    scale_factor : float
        Scaling factor.
    fit : SpaNormFit
        Fitted model.

    Returns
    -------
    np.ndarray
        Normalized matrix (ngenes, ncells).
    """
    if sparse.issparse(Y):
        Y = np.asarray(Y.todense())

    # Calculate mu with all effects
    mu = calculate_mu(fit.gmean, fit.alpha, fit.W)

    # Calculate mu with biology only
    is_bio = fit.wtype == "biology"
    mu_2 = calculate_mu(fit.gmean, fit.alpha[:, is_bio], fit.W[:, is_bio])

    # Winsorize dispersion
    psi = fit.psi.copy()
    psi_max = np.exp(np.median(np.log(psi)) + 3 * stats.median_abs_deviation(np.log(psi)))
    psi = np.minimum(psi, psi_max)

    # logPAC
    lb = pnbinom_cdf(Y - 1, mu, psi)
    ub = dnbinom_pmf(Y, mu, psi) + lb
    p = (lb + ub) / 2
    p = np.clip(p, 0.001, 0.999)

    # Return logPAC
    normmat = np.log2(qnbinom_quantile(p, scale_factor * mu_2, psi) + 1)

    return normmat


def normalise_mean_bio(Y, scale_factor, fit):
    """Normalize using mean biology method.

    Parameters
    ----------
    Y : np.ndarray
        Count matrix (ngenes, ncells).
    scale_factor : float
        Scaling factor.
    fit : SpaNormFit
        Fitted model.

    Returns
    -------
    np.ndarray
        Normalized matrix (ngenes, ncells).
    """
    is_bio = fit.wtype == "biology"
    normmat = np.log2(calculate_mu(fit.gmean, fit.alpha[:, is_bio], fit.W[:, is_bio]))
    return normmat


def normalise_mean_batch(Y, scale_factor, fit):
    """Normalize using mean batch method."""
    is_batch = fit.wtype == "batch"
    normmat = np.log2(calculate_mu(fit.gmean, fit.alpha[:, is_batch], fit.W[:, is_batch]))
    return normmat


def normalise_mean_ls(Y, scale_factor, fit):
    """Normalize using mean library size method."""
    is_ls = fit.wtype == "ls"
    W = fit.W[:, is_ls].copy()
    W = W / W[:, 0:1] * np.median(W[:, 0])
    normmat = np.log2(calculate_mu(fit.gmean, fit.alpha[:, is_ls], W))
    return normmat


def normalise_median_bio(Y, scale_factor, fit):
    """Normalize using median biology method."""
    is_bio = fit.wtype == "biology"
    mu_2 = calculate_mu(fit.gmean, fit.alpha[:, is_bio], fit.W[:, is_bio])

    psi = fit.psi.copy()
    psi_max = np.exp(np.median(np.log(psi)) + 3 * stats.median_abs_deviation(np.log(psi)))
    psi = np.minimum(psi, psi_max)

    normmat = np.log2(qnbinom_quantile(0.5 * np.ones_like(mu_2), scale_factor * mu_2, psi) + 1)
    return normmat


def normalise_pearson(Y, scale_factor, fit):
    """Normalize using Pearson residuals method."""
    is_bio = fit.wtype == "biology"
    mu = calculate_mu(fit.gmean, fit.alpha, fit.W)

    psi = fit.psi.copy()
    psi_max = np.exp(np.median(np.log(psi)) + 3 * stats.median_abs_deviation(np.log(psi)))
    psi = np.minimum(psi, psi_max)

    # Pearson residual: (Y - mu_nonbio) / sqrt(mu + mu^2 * psi)
    mu_nonbio = calculate_mu(fit.gmean, fit.alpha[:, ~is_bio], fit.W[:, ~is_bio])
    normmat = (Y - mu_nonbio) / np.sqrt(mu + mu ** 2 * psi[:, np.newaxis])

    return normmat


def deviance_residuals(Y, fit):
    """Compute deviance residuals.

    Parameters
    ----------
    Y : np.ndarray
        Count matrix (ngenes, ncells).
    fit : SpaNormFit
        Fitted model.

    Returns
    -------
    np.ndarray
        Deviance residuals (ngenes, ncells).
    """
    mu = calculate_mu(fit.gmean, fit.alpha, fit.W)

    psi = fit.psi.copy()
    psi_max = np.exp(np.median(np.log(psi)) + 3 * stats.median_abs_deviation(np.log(psi)))
    psi = np.minimum(psi, psi_max)

    # Deviance residuals
    Y = Y.astype(np.float64)
    dev = Y * np.log(np.where(Y > 0, Y / mu, 1.0))
    dev = dev - (Y + 1.0 / psi[:, np.newaxis]) * np.log((1 + Y * psi[:, np.newaxis]) / (1 + mu * psi[:, np.newaxis]))
    dev = np.maximum(dev, 0)
    dev = np.sign(Y - mu) * np.sqrt(2 * dev)

    return dev


def get_adjustment_fun(gene_model, adj_method):
    """Get the normalization function for the given model and method.

    Parameters
    ----------
    gene_model : str
        Gene model type ('nb').
    adj_method : str
        Adjustment method ('auto', 'logpac', 'pearson', 'medbio', 'meanbio').

    Returns
    -------
    callable
        Normalization function.
    """
    if gene_model == "nb":
        fun_map = {
            "auto": normalise_logpac,
            "logpac": normalise_logpac,
            "pearson": normalise_pearson,
            "medbio": normalise_median_bio,
            "meanbio": normalise_mean_bio,
        }
        if adj_method not in fun_map:
            raise ValueError(f"Invalid adj_method: {adj_method}")
        return fun_map[adj_method]
    else:
        raise ValueError(f"Unsupported gene model: {gene_model}")
