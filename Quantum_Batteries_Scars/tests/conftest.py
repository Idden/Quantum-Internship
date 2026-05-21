"""
Test fixtures and pytest configuration.

Provides common fixtures for testing quantum battery simulations,
including sample systems, states, and Hamiltonians.
"""

import sys
import shutil
from pathlib import Path

import pytest
import numpy as np
from quantum_battery.core import QuantumSystem, State, Hamiltonian

# Ensure plot_utils and data_utils are importable from all test subdirectories
sys.path.insert(0, str(Path(__file__).parent))


class SimpleHamiltonian(Hamiltonian):
    """Simple test Hamiltonian - diagonal 2x2 matrix."""
    
    def __init__(self, eigenvalues: np.ndarray = None):
        """Initialize with given eigenvalues."""
        super().__init__(dim=2, time_dependent=False)
        if eigenvalues is None:
            eigenvalues = np.array([0.0, 1.0])
        self.eigenvalues = eigenvalues
    
    def get_matrix(self, t: float = 0.0) -> np.ndarray:
        """Return diagonal Hamiltonian."""
        return np.diag(self.eigenvalues)


@pytest.fixture
def simple_hamiltonian():
    """Fixture: Simple 2-level Hamiltonian."""
    return SimpleHamiltonian()


@pytest.fixture
def initial_state():
    """Fixture: Ground state |0⟩."""
    state = State(dim=2, state_type="pure")
    state.set_vector(np.array([1.0, 0.0]))
    return state


@pytest.fixture
def superposition_state():
    """Fixture: Superposition state (|0⟩ + |1⟩)/√2."""
    state = State(dim=2, state_type="pure")
    state.set_vector(np.array([1.0, 1.0]) / np.sqrt(2))
    return state


@pytest.fixture
def simple_quantum_system(simple_hamiltonian, initial_state):
    """Fixture: Simple 2-level quantum system."""
    metadata = {
        "name": "Simple Qubit",
        "description": "Two-level system for testing"
    }
    return QuantumSystem(
        hamiltonian=simple_hamiltonian,
        initial_state=initial_state,
        metadata=metadata
    )


@pytest.fixture
def temp_results_dir(tmp_path):
    """Fixture: Temporary directory for test results."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir


@pytest.fixture(scope="session")
def test_output_dir():
    """Session-scoped fixture: root directory for test artifacts."""
    output_dir = Path(__file__).parent / "test_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def test_subdir(test_output_dir, request):
    """Per-test fixture: unique subdirectory for artifacts.

    Directory structure: test_output/<module>/<test_name>/
    Parametrized tests get sanitised names (brackets → underscores).
    """
    module = request.module.__name__.split('.')[-1]
    test_name = (
        request.node.name
        .replace('[', '_').replace(']', '').replace('/', '_')
    )
    subdir = test_output_dir / module / test_name
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir
