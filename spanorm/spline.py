"""Thin-plate spline basis construction (bs.tps equivalent)."""

import numpy as np


def _ns_basis(x, df):
    """Natural cubic spline basis matching R's splines::ns().

    R's ns(x, df) with intercept=FALSE returns df columns.
    Interior knots are placed at quantiles 1/df, 2/df, ..., (df-1)/df.

    Algorithm (matching R's source):
    1. Place df-1 interior knots at evenly spaced quantiles
    2. Build truncated power basis: [x, (x-k1)^3+, ..., (x-k_{df-1})^3+]
       That's df terms
    3. Apply natural boundary constraints (d2=0 at boundaries)
       This modifies the basis but preserves df columns

    Parameters
    ----------
    x : np.ndarray
        1D array of values.
    df : int
        Degrees of freedom (number of output columns).

    Returns
    -------
    np.ndarray
        Basis matrix (len(x), df).
    """
    n = len(x)
    x = np.asarray(x, dtype=np.float64)

    if df <= 0:
        raise ValueError("df must be positive")
    df = int(df)

    if df == 1:
        # Just the linear term
        return x.reshape(-1, 1)

    # Interior knots at evenly spaced quantiles
    n_knots = df - 1
    knot_probs = np.arange(1, n_knots + 1) / (n_knots + 1)
    knots = np.quantile(x, knot_probs)

    # Boundary knots
    x_min = x.min()
    x_max = x.max()

    # Truncated power basis: [x, (x-k1)^3+, ..., (x-k_{df-1})^3+]
    # That's df terms (1 linear + df-1 cubic)
    tp = np.zeros((n, df))
    tp[:, 0] = x
    for j in range(df - 1):
        tp[:, j + 1] = np.maximum(x - knots[j], 0.0) ** 3

    # Natural boundary constraints: d^2/dx^2 = 0 at x_min and x_max
    # d^2/dx^2 of x = 0
    # d^2/dx^2 of (x-k)^3+ = 6 * max(x-k, 0)
    # At x_min: 6 * max(x_min - k_j, 0) for each cubic term
    # At x_max: 6 * max(x_max - k_j, 0) for each cubic term

    # Constraint matrix: C @ coeff = 0 for the cubic coefficients
    # C = [[max(x_min-k1,0), ..., max(x_min-k_{df-1},0)],
    #      [max(x_max-k1,0), ..., max(x_max-k_{df-1},0)]]
    # For the linear term (column 0), d2 = 0 automatically.
    # So we only constrain columns 1..df-1.

    # R's approach: it doesn't explicitly solve constraints.
    # Instead, R constructs the B-spline basis on a specific knot sequence
    # and drops the first and last B-spline functions.

    # Let me use R's actual algorithm:
    # 1. Knot sequence: [x_min]*4 + interior_knots + [x_max]*4
    # 2. B-spline basis (order 4 = degree 3): len(knots) - 4 functions
    # 3. Drop first and last -> natural spline basis
    # 4. If intercept=FALSE, also drop the first remaining column

    # Number of interior knots = df - 1
    # Knot sequence length = 4 + (df-1) + 4 = df + 7
    # B-spline basis count = (df+7) - 4 = df + 3
    # Drop first and last -> df + 1
    # Drop intercept -> df

    all_knots = np.concatenate([
        np.repeat(x_min, 4),
        knots,
        np.repeat(x_max, 4)
    ])

    n_bspline = len(all_knots) - 4  # df + 3
    bspline_basis = np.zeros((n, n_bspline))

    # Evaluate each B-spline basis function
    for i in range(n_bspline):
        bspline_basis[:, i] = _deboor_basis(x, all_knots, i, 3)

    # Natural spline: drop first and last B-spline
    ns = bspline_basis[:, 1:-1]  # df + 1 columns

    # Drop intercept (first column) for intercept=FALSE
    ns = ns[:, 1:]  # df columns

    # Center the basis (R's ns centers internally)
    ns = ns - ns.mean(axis=0)

    return ns


def _deboor_basis(x, knots, i, degree):
    """Evaluate i-th B-spline basis function of given degree using de Boor recursion.

    Parameters
    ----------
    x : np.ndarray
        Evaluation points.
    knots : np.ndarray
        Knot vector.
    i : int
        Basis function index (0-based).
    degree : int
        B-spline degree (3 for cubic).

    Returns
    -------
    np.ndarray
        Values of the basis function.
    """
    n = len(x)
    t = knots

    # Base case: degree 0
    if degree == 0:
        result = np.zeros(n)
        # B_{i,0}(x) = 1 if t[i] <= x < t[i+1], else 0
        # Special case for the last knot interval
        mask = (x >= t[i]) & (x < t[i + 1])
        if i + 1 < len(t) - 1:
            mask = (x >= t[i]) & (x < t[i + 1])
        else:
            mask = (x >= t[i]) & (x <= t[i + 1])
        result[mask] = 1.0
        return result

    # Recursive case
    d1 = t[i + degree] - t[i]
    d2 = t[i + degree + 1] - t[i + 1]

    term1 = np.zeros(n)
    if abs(d1) > 1e-14:
        term1 = ((x - t[i]) / d1) * _deboor_basis(x, knots, i, degree - 1)

    term2 = np.zeros(n)
    if abs(d2) > 1e-14:
        term2 = ((t[i + degree + 1] - x) / d2) * _deboor_basis(x, knots, i + 1, degree - 1)

    return term1 + term2


def bs_tps(x, y, df_tps=6):
    """Construct thin-plate spline basis (equivalent to R's bs.tps).

    Constructs a tensor product of natural spline bases for 2D spatial coordinates.

    Parameters
    ----------
    x : np.ndarray
        X coordinates (length n).
    y : np.ndarray
        Y coordinates (length n).
    df_tps : int
        Maximum degrees of freedom for the thin-plate spline.

    Returns
    -------
    tuple
        (basis_matrix, (df_x, df_y)) where basis_matrix has shape (n, df_x*df_y)
        and (df_x, df_y) are the actual df along each axis.
    """
    if df_tps <= 0:
        raise ValueError("'df_tps' should be greater than 0")
    if not float(df_tps).is_integer():
        raise ValueError("'df_tps' should be an integer")
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must have the same length")

    df_tps = int(df_tps)

    # Determine df along each axis (same as R's bs.tps)
    xrng = x.max() - x.min()
    yrng = y.max() - y.min()
    gap = max(xrng, yrng) / df_tps

    if gap == 0:
        df_tps_x = 1
        df_tps_y = 1
    else:
        df_tps_x = int(np.ceil(xrng / gap))
        df_tps_y = int(np.ceil(yrng / gap))

    # Ensure at least 1
    df_tps_x = max(df_tps_x, 1)
    df_tps_y = max(df_tps_y, 1)

    # Construct natural spline basis for each axis
    bs_x = _ns_basis(x, df_tps_x)  # (n, df_tps_x)
    bs_y = _ns_basis(y, df_tps_y)  # (n, df_tps_y)

    # Tensor product: all pairwise products
    n = len(x)
    bs_xy = np.zeros((n, df_tps_x * df_tps_y))
    for i in range(df_tps_x):
        for j in range(df_tps_y):
            bs_xy[:, i * df_tps_y + j] = bs_x[:, i] * bs_y[:, j]

    # Center (subtract column means) - equivalent to R's scale(bs.xy, scale=FALSE)
    bs_xy = bs_xy - bs_xy.mean(axis=0)

    return bs_xy, (df_tps_x, df_tps_y)
