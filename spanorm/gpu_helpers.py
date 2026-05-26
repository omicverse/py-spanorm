"""GPU helper functions — CPU-only port.

Maps R's TensorFlow GPU functions to numpy/scipy equivalents.
All functions here are exact (E) translations — no approximation.
"""

import numpy as np
from scipy import sparse


def check_gpu():
    """Check if GPU is available. Always returns False (CPU-only port)."""
    return False


def is_tf_tensor(x):
    """Check if x is a TensorFlow tensor. Always returns False."""
    return False


def to_gpu_matrix(mat, backend='cpu'):
    """Convert to GPU matrix. No-op for CPU backend."""
    return np.asarray(mat, dtype=np.float64)


def to_gpu_vector(vec, backend='cpu'):
    """Convert to GPU vector. No-op for CPU backend."""
    return np.asarray(vec, dtype=np.float64)


def diag_mat(vec, backend='cpu'):
    """Create diagonal matrix from vector."""
    return np.diag(np.asarray(vec, dtype=np.float64))


def invert_mat(mat):
    """Invert matrix using Cholesky for positive-definite."""
    from scipy.linalg import cho_factor, cho_solve
    try:
        c, low = cho_factor(mat)
        return cho_solve((c, low), np.eye(mat.shape[0]))
    except np.linalg.LinAlgError:
        return np.linalg.inv(mat)


def tcrossprod_gpu(x, y=None):
    """Compute x @ y.T (R's tcrossprod)."""
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        return x @ x.T
    y = np.asarray(y, dtype=np.float64)
    return x @ y.T


def crossprod_gpu(x, y=None):
    """Compute x.T @ y (R's crossprod)."""
    x = np.asarray(x, dtype=np.float64)
    if y is None:
        return x.T @ x
    y = np.asarray(y, dtype=np.float64)
    return x.T @ y


def add_vec_mat_gpu(vec, mat, backend='cpu'):
    """Add vector to each column of matrix (broadcasting)."""
    return np.asarray(mat, dtype=np.float64) + np.asarray(vec, dtype=np.float64)[:, np.newaxis]


def mult_vec_mat_gpu(vec, mat, backend='cpu'):
    """Multiply each column of matrix by vector (broadcasting)."""
    return np.asarray(mat, dtype=np.float64) * np.asarray(vec, dtype=np.float64)[:, np.newaxis]


def rowmeans_gpu(mat):
    """Row means."""
    return np.mean(np.asarray(mat, dtype=np.float64), axis=1)


def colmeans_gpu(mat):
    """Column means."""
    return np.mean(np.asarray(mat, dtype=np.float64), axis=0)


def rowsums_gpu(mat):
    """Row sums."""
    return np.sum(np.asarray(mat, dtype=np.float64), axis=1)


def colsums_gpu(mat):
    """Column sums."""
    return np.sum(np.asarray(mat, dtype=np.float64), axis=0)


def dnbinom_gpu(y, mu, size, log=False):
    """NB log-PMF."""
    from .nb_model import dnbinom_log
    return dnbinom_log(np.asarray(y), np.asarray(mu), 1.0 / np.asarray(size))


def qnbinom_gpu(p, mu, size):
    """NB quantile function."""
    from scipy.stats import nbinom
    return nbinom.ppf(np.asarray(p), n=np.asarray(size), mu=np.asarray(mu))


def pnbinom_gpu(q, mu, size):
    """NB CDF."""
    from scipy.stats import nbinom
    return nbinom.cdf(np.asarray(q), n=np.asarray(size), mu=np.asarray(mu))


def copy(x):
    """Clone/copy array."""
    return np.copy(np.asarray(x))
