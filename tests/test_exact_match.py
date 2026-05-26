"""Parity test — compare Python output against R reference.

This test verifies that the Python SpaNorm port produces results
within the declared parity threshold (1e-8 max abs error) of the R reference.
"""

import json
import subprocess
import tempfile
import os
import numpy as np
import pytest


# Path to R reference driver
R_DRIVER = os.path.join(os.path.dirname(__file__), "r_reference_driver.R")
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_r_reference(json_path):
    """Load R reference output from JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary with 'logcounts' as numpy array.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {
        'logcounts': np.array(data['logcounts']),
    }
    return result


def compute_parity_deterministic(reference, candidate, atol=1e-8):
    """Compute max absolute error between reference and candidate.

    Parameters
    ----------
    reference : np.ndarray
        Reference array.
    candidate : np.ndarray
        Candidate array.
    atol : float
        Absolute tolerance.

    Returns
    -------
    float
        Maximum absolute error.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs candidate {candidate.shape}"
        )

    max_abs_err = np.max(np.abs(reference - candidate))
    return max_abs_err


@pytest.fixture
def canonical_fixture_path():
    """Path to the canonical fixture RDS file."""
    path = os.path.join(FIXTURE_DIR, "fixture_HumanDLPFC.rds")
    if not os.path.exists(path):
        pytest.skip(f"Fixture not found: {path}")
    return path


@pytest.fixture
def r_reference_output(canonical_fixture_path):
    """Run R reference and return output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "r_output.json")

        # Run R reference
        result = subprocess.run(
            ["Rscript", R_DRIVER, canonical_fixture_path, output_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            pytest.fail(f"R reference failed:\n{result.stderr}")

        return load_r_reference(output_path)


@pytest.fixture
def python_output(canonical_fixture_path):
    """Run Python port and return output."""
    import anndata as ad

    # Load fixture as AnnData
    # The fixture should be an H5AD file for Python
    h5ad_path = canonical_fixture_path.replace('.rds', '.h5ad')
    if not os.path.exists(h5ad_path):
        pytest.skip(f"Python fixture not found: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path)

    # Run Python SpaNorm
    from spanorm import spanorm
    np.random.seed(42)
    result = spanorm(adata, sample_p=0.25, df_tps=2, tol=1e-2, verbose=False)

    return {'logcounts': result.layers['logcounts']}


def test_parity_against_r(r_reference_output, python_output):
    """Test that Python output matches R reference within tolerance.

    Parity gate: max absolute error < 1e-8 (deterministic class).
    """
    ref = r_reference_output['logcounts']
    cand = python_output['logcounts']

    # Transpose if needed (R returns genes x cells, Python returns cells x genes)
    if ref.shape != cand.shape:
        cand = cand.T

    max_err = compute_parity_deterministic(ref, cand, atol=1e-8)

    threshold = 1e-8
    assert max_err < threshold, (
        f"Parity gate failed: max abs error {max_err:.2e} >= threshold {threshold:.2e}\n"
        f"Reference shape: {ref.shape}, Candidate shape: {cand.shape}"
    )


def test_shape_match(r_reference_output, python_output):
    """Test that output shapes match."""
    ref = r_reference_output['logcounts']
    cand = python_output['logcounts']

    # Allow transposed shapes
    assert ref.shape == cand.shape or ref.shape == cand.T.shape, (
        f"Shape mismatch: reference {ref.shape} vs candidate {cand.shape}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
