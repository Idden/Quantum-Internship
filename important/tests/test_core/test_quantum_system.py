"""Tests for core quantum system classes."""

import pytest
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantum_battery.core import QuantumSystem, State, Hamiltonian
from GitHub_QM.important.tests.plot_utils import (
    apply_plot_style, plot_hamiltonian_heatmap, plot_eigenvalue_spectrum,
    plot_state_bar, plot_density_matrix_heatmap,
)
from GitHub_QM.important.tests.data_utils import save_array_data, save_metadata


class TestState:
    """Test suite for the State class."""
    
    def test_state_initialization_pure(self, test_subdir):
        """Test initializing a pure quantum state."""
        state = State(dim=2, state_type="pure")
        assert state.dim == 2
        assert state.state_type == "pure"

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_state_initialization_pure',
            'dim': 2,
            'state_type': 'pure',
            'artifacts': ['metadata.json'],
        })
    
    def test_state_vector_setting(self, initial_state, test_subdir):
        """Test setting and retrieving state vector."""
        vector = initial_state.get_vector()
        assert len(vector) == 2
        assert np.isclose(np.linalg.norm(vector), 1.0)

        # --- Artifacts ---
        fig = plot_state_bar(vector, title='Initial State Vector',
                            output_path=test_subdir / "state_vector.png")
        plt.close(fig)
        save_array_data(test_subdir / "state.npz", vector=vector)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_state_vector_setting',
            'dim': len(vector),
            'norm': float(np.linalg.norm(vector)),
            'artifacts': ['state_vector.png', 'state.npz'],
        })
    
    def test_state_vector_normalization(self, test_subdir):
        """Test that state vectors are automatically normalized."""
        state = State(dim=2, state_type="pure")
        unnormalized = np.array([3.0, 4.0])  # Not normalized
        state.set_vector(unnormalized)
        
        vector = state.get_vector()
        norm = np.linalg.norm(vector)
        assert np.isclose(norm, 1.0)

        # --- Artifacts ---
        apply_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar([0, 1], np.abs(unnormalized), color='salmon', edgecolor='darkred')
        axes[0].set_title('Before normalization')
        axes[0].set_ylabel('$|\\psi_n|$')
        axes[1].bar([0, 1], np.abs(vector), color='steelblue', edgecolor='navy')
        axes[1].set_title('After normalization')
        axes[1].set_ylabel('$|\\psi_n|$')
        fig.suptitle('Auto-normalization test')
        fig.tight_layout()
        fig.savefig(test_subdir / "normalization.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "vectors.npz",
                       unnormalized=unnormalized, normalized=vector)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_state_vector_normalization',
            'unnorm_norm': float(np.linalg.norm(unnormalized)),
            'norm_after': float(norm),
            'artifacts': ['normalization.png', 'vectors.npz'],
        })
    
    def test_density_matrix_pure_state(self, initial_state, test_subdir):
        """Test density matrix representation of pure state."""
        rho = initial_state.get_density_matrix()
        assert rho.shape == (2, 2)
        assert np.isclose(np.trace(rho), 1.0)

        # --- Artifacts ---
        fig = plot_density_matrix_heatmap(rho, title='Pure State $\\rho$',
                                         output_path=test_subdir / "density_matrix.png")
        plt.close(fig)
        save_array_data(test_subdir / "rho.npz", rho=rho)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_density_matrix_pure_state',
            'shape': list(rho.shape),
            'trace': float(np.trace(rho).real),
            'artifacts': ['density_matrix.png', 'rho.npz'],
        })
    
    def test_state_purity(self, initial_state, superposition_state, test_subdir):
        """Test purity calculation."""
        pure_purity = initial_state.purity()
        super_purity = superposition_state.purity()
        
        # Pure states have purity = 1
        assert np.isclose(pure_purity, 1.0)
        assert np.isclose(super_purity, 1.0)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_state_purity',
            'pure_state_purity': float(pure_purity),
            'superposition_purity': float(super_purity),
            'artifacts': ['metadata.json'],
        })
    
    def test_state_normalization_check(self, initial_state, test_subdir):
        """Test normalization check method."""
        assert initial_state.is_normalized()

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_state_normalization_check',
            'is_normalized': True,
            'artifacts': ['metadata.json'],
        })
    
    def test_dimension_mismatch_error(self, test_subdir):
        """Test that setting wrong dimension raises error."""
        state = State(dim=2)
        with pytest.raises(ValueError):
            state.set_vector(np.array([1.0, 0.0, 0.0]))

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_dimension_mismatch_error',
            'state_dim': 2,
            'attempted_vector_len': 3,
            'raised': 'ValueError',
            'artifacts': ['metadata.json'],
        })


class TestHamiltonian:
    """Test suite for the Hamiltonian class."""
    
    def test_hamiltonian_initialization(self, simple_hamiltonian, test_subdir):
        """Test Hamiltonian initialization."""
        assert simple_hamiltonian.dim == 2
        assert not simple_hamiltonian.time_dependent

        # --- Artifacts ---
        H_mat = simple_hamiltonian.get_matrix()
        fig = plot_hamiltonian_heatmap(H_mat, title='Simple Hamiltonian',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        save_array_data(test_subdir / "H.npz", hamiltonian=H_mat)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_hamiltonian_initialization',
            'dim': simple_hamiltonian.dim,
            'time_dependent': simple_hamiltonian.time_dependent,
            'artifacts': ['hamiltonian.png', 'H.npz'],
        })
    
    def test_hamiltonian_hermiticity(self, simple_hamiltonian, test_subdir):
        """Test that Hamiltonian is Hermitian."""
        assert simple_hamiltonian.is_hermitian()

        # --- Artifacts ---
        H_mat = simple_hamiltonian.get_matrix()
        err = float(np.max(np.abs(H_mat - H_mat.conj().T)))
        fig = plot_hamiltonian_heatmap(H_mat, title='|H| — Hermiticity check',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        save_array_data(test_subdir / "H.npz", hamiltonian=H_mat)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_hamiltonian_hermiticity',
            'dimension': H_mat.shape[0],
            'hermiticity_error': err,
            'artifacts': ['hamiltonian.png', 'H.npz'],
        })
    
    def test_eigenvalues(self, simple_hamiltonian, test_subdir):
        """Test eigenvalue calculation."""
        eigenvals = simple_hamiltonian.get_eigenvalues()
        expected = np.array([0.0, 1.0])
        assert np.allclose(eigenvals, expected)

        # --- Artifacts ---
        fig = plot_eigenvalue_spectrum(eigenvals, title='Simple Hamiltonian Spectrum',
                                      output_path=test_subdir / "spectrum.png")
        plt.close(fig)
        save_array_data(test_subdir / "eigenvalues.npz",
                       eigenvalues=eigenvals, expected=expected)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_eigenvalues',
            'eigenvalues': eigenvals.tolist(),
            'expected': expected.tolist(),
            'max_error': float(np.max(np.abs(eigenvals - expected))),
            'artifacts': ['spectrum.png', 'eigenvalues.npz'],
        })
    
    def test_eigenstates(self, simple_hamiltonian, test_subdir):
        """Test eigenstate calculation."""
        eigenvals, eigenvecs = simple_hamiltonian.get_eigenstates()
        # Check that eigenvectors are orthonormal
        for i in range(2):
            norm = np.linalg.norm(eigenvecs[:, i])
            assert np.isclose(norm, 1.0)

        # --- Artifacts ---
        apply_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for i, ax in enumerate(axes):
            ax.bar(np.arange(len(eigenvecs[:, i])), np.abs(eigenvecs[:, i]),
                   color='steelblue', edgecolor='navy')
            ax.set_title(f'Eigenstate {i}  ($E={eigenvals[i]:.4f}$)')
            ax.set_ylabel('$|\\psi_n|$')
            ax.set_xlabel('Basis index')
        fig.tight_layout()
        fig.savefig(test_subdir / "eigenstates.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "eigen.npz",
                       eigenvalues=eigenvals, eigenvectors=eigenvecs)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_eigenstates',
            'eigenvalues': eigenvals.tolist(),
            'norms': [float(np.linalg.norm(eigenvecs[:, i])) for i in range(2)],
            'artifacts': ['eigenstates.png', 'eigen.npz'],
        })


class TestQuantumSystem:
    """Test suite for the QuantumSystem class."""
    
    def test_system_initialization(self, simple_quantum_system, test_subdir):
        """Test quantum system initialization."""
        assert simple_quantum_system.dim == 2
        assert simple_quantum_system.get_hilbert_dimension() == 2

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_system_initialization',
            'dim': simple_quantum_system.dim,
            'hilbert_dim': simple_quantum_system.get_hilbert_dimension(),
            'artifacts': ['metadata.json'],
        })
    
    def test_get_hamiltonian_matrix(self, simple_quantum_system, test_subdir):
        """Test retrieving Hamiltonian matrix."""
        H = simple_quantum_system.get_hamiltonian_matrix()
        assert H.shape == (2, 2)

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(H, title='System Hamiltonian',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        save_array_data(test_subdir / "H.npz", hamiltonian=H)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_get_hamiltonian_matrix',
            'shape': list(H.shape),
            'artifacts': ['hamiltonian.png', 'H.npz'],
        })
    
    def test_get_state_vector(self, simple_quantum_system, test_subdir):
        """Test retrieving state vector."""
        psi = simple_quantum_system.get_state_vector()
        assert len(psi) == 2

        # --- Artifacts ---
        fig = plot_state_bar(psi, title='System State Vector',
                            output_path=test_subdir / "state_vector.png")
        plt.close(fig)
        save_array_data(test_subdir / "state.npz", psi=psi)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_get_state_vector',
            'dim': len(psi),
            'norm': float(np.linalg.norm(psi)),
            'artifacts': ['state_vector.png', 'state.npz'],
        })
    
    def test_set_state(self, simple_quantum_system, superposition_state, test_subdir):
        """Test changing the quantum state."""
        original = simple_quantum_system.get_state_vector()
        simple_quantum_system.set_state(superposition_state)
        new = simple_quantum_system.get_state_vector()
        
        assert not np.allclose(original, new)

        # --- Artifacts ---
        apply_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(np.arange(len(original)), np.abs(original),
                    color='salmon', edgecolor='darkred')
        axes[0].set_title('Original state')
        axes[0].set_ylabel('$|\\psi_n|$')
        axes[1].bar(np.arange(len(new)), np.abs(new),
                    color='steelblue', edgecolor='navy')
        axes[1].set_title('New state (superposition)')
        axes[1].set_ylabel('$|\\psi_n|$')
        fig.suptitle('set_state comparison')
        fig.tight_layout()
        fig.savefig(test_subdir / "state_change.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "states.npz",
                       original=original, new=new)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_set_state',
            'original_norm': float(np.linalg.norm(original)),
            'new_norm': float(np.linalg.norm(new)),
            'artifacts': ['state_change.png', 'states.npz'],
        })
    
    def test_system_info(self, simple_quantum_system, test_subdir):
        """Test getting comprehensive system information."""
        info = simple_quantum_system.get_system_info()
        assert "dimension" in info
        assert "hamiltonian_info" in info
        assert "state_info" in info
        assert "metadata" in info

        # --- Artifacts ---
        save_metadata(test_subdir / "system_info.json", info)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_system_info',
            'info_keys': list(info.keys()),
            'artifacts': ['system_info.json', 'metadata.json'],
        })
    
    def test_dimension_mismatch_in_set_state(self, simple_quantum_system, test_subdir):
        """Test that setting incompatible state raises error."""
        wrong_state = State(dim=3, state_type="pure")
        wrong_state.set_vector(np.array([1.0, 0.0, 0.0]))
        
        with pytest.raises(ValueError):
            simple_quantum_system.set_state(wrong_state)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_dimension_mismatch_in_set_state',
            'system_dim': simple_quantum_system.dim,
            'wrong_state_dim': 3,
            'raised': 'ValueError',
            'artifacts': ['metadata.json'],
        })
