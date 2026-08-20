"""Local fixtures for the student ``Idden_code`` tests."""

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
IDDEN_CODE = REPO_ROOT / "Idden_code"
INDEPENDENT_DATA = REPO_ROOT / "IndependentData"

if str(IDDEN_CODE) not in sys.path:
    sys.path.insert(0, str(IDDEN_CODE))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the QuantumBatteries repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def constrained_basis_npz() -> Path:
    """Return the bare QuSpin constrained-basis reference NPZ."""
    path = INDEPENDENT_DATA / "minimal_pxp_constrained_basis_N10.npz"
    assert path.exists(), f"Missing reference data: {path}"
    return path


@pytest.fixture(scope="session")
def student_hamiltonian_npz() -> Path:
    """Return the QuSpin reference NPZ matching the student Hamiltonian."""
    path = INDEPENDENT_DATA / "minimal_pxp_student_hamiltonian_N10.npz"
    assert path.exists(), f"Missing reference data: {path}"
    return path
