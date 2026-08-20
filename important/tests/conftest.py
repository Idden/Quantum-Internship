"""
Shared pytest configuration for the quantum-scar test suite.

This file does three jobs:

1. Puts the right directories on ``sys.path`` so that both
   ``GitHub_QM.important.hpc.quantumScarFunctions`` (the same import path the
   HPC scripts use) and the local helpers (``data_utils``, ``plot_utils``,
   ``reference_pxp``) import no matter where pytest is invoked from.
2. Provides session-cached fixtures for the scar Hamiltonian, so a chain of
   a given size is diagonalised once per session instead of once per test.
3. Provides the ``test_subdir`` fixture that tests use to save inspectable
   artifacts under ``tests/test_output/<module>/<test_name>/``.

There is deliberately no import of any simulation code at module scope --
a broken import in the physics code should fail the tests that use it, not
abort collection of the entire suite.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent          # .../important/tests
IMPORTANT_DIR = TESTS_DIR.parent                     # .../important
REPO_ROOT = IMPORTANT_DIR.parent                     # .../GitHub_QM
PACKAGE_ROOT = REPO_ROOT.parent                      # parent of GitHub_QM

# PACKAGE_ROOT makes "GitHub_QM.important.hpc.quantumScarFunctions" importable.
# TESTS_DIR makes "data_utils" / "plot_utils" / "reference_pxp" importable.
for _path in (str(PACKAGE_ROOT), str(TESTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: test takes more than a couple of seconds"
    )


# --------------------------------------------------------------------------
# Artifact directories
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_output_dir():
    """Root directory for inspectable test artifacts, wiped once per session."""
    output_dir = TESTS_DIR / "test_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_subdir(test_output_dir, request):
    """Per-test artifact directory: ``test_output/<module>/<test_name>/``."""
    module = request.module.__name__.split(".")[-1]
    test_name = (
        request.node.name.replace("[", "_").replace("]", "").replace("/", "_")
    )
    subdir = test_output_dir / module / test_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


# --------------------------------------------------------------------------
# Physics fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def scar_functions():
    """The module under test, imported through the HPC import path."""
    from GitHub_QM.important.hpc import quantumScarFunctions

    return quantumScarFunctions


@pytest.fixture(scope="session")
def scar_system(scar_functions):
    """
    ``N -> (H0, eigenvalues, eigenstates, psi0, basisList)``, cached per size.

    Diagonalising the same chain in twenty different tests is the single
    biggest avoidable cost in this suite, so results are memoised.
    """
    cache = {}

    def _build(N):
        if N not in cache:
            cache[N] = scar_functions.get_scar_ham(N)
        return cache[N]

    return _build


@pytest.fixture(scope="session")
def basis_list(scar_system):
    """``N -> basisList`` for the periodic blockade-constrained basis."""

    def _basis(N):
        return scar_system(N)[4]

    return _basis


@pytest.fixture
def rng():
    """A seeded generator, so any randomised test is reproducible."""
    return np.random.default_rng(12345)
