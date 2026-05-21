"""Tests for the constrained PXP Hamiltonian and model."""

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantum_battery.models import PXPConstrainedHamiltonian, PXPChainModel
from plot_utils import (
    apply_plot_style, plot_hamiltonian_heatmap, plot_eigenvalue_spectrum,
    plot_state_bar, plot_sparsity_pattern,
)
from data_utils import save_array_data, save_metadata


def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class TestPXPConstrainedBasis:
    @pytest.mark.parametrize("L", [1, 2, 3, 4, 5, 6])
    def test_basis_dimension_matches_fibonacci(self, L, test_subdir):
        h = PXPConstrainedHamiltonian(L=L)
        # Constrained dimension is F_{L+2}
        expected_dim = fibonacci(L + 2)
        assert h.dim == expected_dim

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_basis_dimension_matches_fibonacci',
            'L': L,
            'dim': h.dim,
            'expected_dim': expected_dim,
            'fibonacci_index': L + 2,
            'artifacts': ['metadata.json'],
        })

    def test_basis_contains_all_zero_state(self, test_subdir):
        h = PXPConstrainedHamiltonian(L=5)
        zero = tuple(0 for _ in range(5))
        assert zero in h.basis.index_map

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_basis_contains_all_zero_state',
            'L': 5,
            'all_zero_config': list(zero),
            'basis_size': h.dim,
            'artifacts': ['metadata.json'],
        })


class TestPXPConstrainedHamiltonian:
    def test_sparse_matrix_construction_and_symmetry(self, test_subdir):
        h = PXPConstrainedHamiltonian(L=5, omega=1.0)
        Hs = h.build_sparse_matrix()
        # Check shape
        assert Hs.shape == (h.dim, h.dim)
        # Symmetry: H - H^T should be zero
        diff = (Hs - Hs.T).nnz
        assert diff == 0

        # --- Artifacts ---
        H_dense = Hs.toarray()
        fig = plot_hamiltonian_heatmap(H_dense, title=f'PXP H (L=5)',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        fig2 = plot_sparsity_pattern(H_dense, title='PXP Sparsity (L=5)',
                                    output_path=test_subdir / "sparsity.png")
        plt.close(fig2)
        save_array_data(test_subdir / "H.npz", hamiltonian=H_dense)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_sparse_matrix_construction_and_symmetry',
            'L': 5, 'omega': 1.0,
            'dim': h.dim,
            'nnz': int(Hs.nnz),
            'symmetry_diff_nnz': diff,
            'artifacts': ['hamiltonian.png', 'sparsity.png', 'H.npz'],
        })

    def test_dense_matrix_matches_sparse(self, test_subdir):
        h = PXPConstrainedHamiltonian(L=4, omega=0.7)
        Hs = h.get_sparse_matrix()
        Hd = h.get_matrix()
        assert np.allclose(Hs.toarray(), Hd)

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(Hd, title='PXP Dense H (L=4, $\\omega$=0.7)',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        diff = np.max(np.abs(Hs.toarray() - Hd))
        save_array_data(test_subdir / "H.npz", dense=Hd, sparse_dense=Hs.toarray())
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_dense_matrix_matches_sparse',
            'L': 4, 'omega': 0.7,
            'max_diff': float(diff),
            'artifacts': ['hamiltonian.png', 'H.npz'],
        })


class TestPXPChainModel:
    def test_model_initialization_default(self, test_subdir):
        model = PXPChainModel(L=4)
        assert model.dim == fibonacci(6)  # L=4 -> F_{6} = 8
        # Initial state is a basis vector (one-hot)
        vec = model.get_state_vector()
        assert np.isclose(np.linalg.norm(vec), 1.0)

        # --- Artifacts ---
        fig = plot_state_bar(vec, title='PXP Default Initial State (L=4)',
                            output_path=test_subdir / "state.png")
        plt.close(fig)
        save_array_data(test_subdir / "state.npz", state_vector=vec)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_model_initialization_default',
            'L': 4,
            'dim': model.dim,
            'norm': float(np.linalg.norm(vec)),
            'artifacts': ['state.png', 'state.npz'],
        })

    def test_model_initialization_with_config(self, test_subdir):
        L = 5
        config = (0, 1, 0, 1, 0)  # Valid: no adjacent ones
        model = PXPChainModel(L=L, initial_config=config)
        idx = model.basis_index(config)
        vec = model.get_state_vector()
        # Basis vector should have 1 at idx
        assert np.isclose(vec[idx], 1.0)

        # --- Artifacts ---
        fig = plot_state_bar(vec, title=f'PXP State config={config}',
                            output_path=test_subdir / "state.png")
        plt.close(fig)
        save_array_data(test_subdir / "state.npz", state_vector=vec)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_model_initialization_with_config',
            'L': L,
            'config': list(config),
            'basis_index': int(idx),
            'dim': model.dim,
            'artifacts': ['state.png', 'state.npz'],
        })

    def test_invalid_config_raises(self, test_subdir):
        L = 4
        invalid = (1, 1, 0, 0)  # Adjacent ones
        with pytest.raises(ValueError):
            PXPChainModel(L=L, initial_config=invalid)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_invalid_config_raises',
            'L': L,
            'invalid_config': list(invalid),
            'raised': 'ValueError',
            'artifacts': ['metadata.json'],
        })
