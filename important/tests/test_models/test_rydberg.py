"""
Tests for the Rydberg Hamiltonian model.

Tests cover:
- Hamiltonian construction and Hermiticity
- Dimension checks (2^N)
- Ground state properties
- Staggered Sz operator
- Energy conservation for undriven evolution
- Sparse matrix efficiency
- Resonance frequency extraction
"""

import pytest
import numpy as np
from scipy.sparse import issparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from quantum_battery.models.rydberg import (
    RydbergHamiltonian,
    compute_energy_observables,
    run_omega_scan,
    extract_resonance_frequency,
    _pauli_x_sparse,
    _pauli_z_sparse,
    _projector_sparse,
)
from GitHub_QM.important.tests.plot_utils import (
    apply_plot_style, plot_hamiltonian_heatmap, plot_eigenvalue_spectrum,
    plot_state_bar, plot_sparsity_pattern, plot_energy_evolution,
    plot_omega_scan_heatmap, PLOT_STYLE,
)
from GitHub_QM.important.tests.data_utils import save_array_data, save_metadata, save_csv_table


class TestPauliOperators:
    """Test individual Pauli operator construction."""
    
    def test_pauli_x_sparse_shape(self, test_subdir):
        """Test that Pauli X has correct shape."""
        n_qubits = 3
        for site in range(n_qubits):
            sx = _pauli_x_sparse(n_qubits, site)
            assert sx.shape == (2**n_qubits, 2**n_qubits)
            assert issparse(sx)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_pauli_x_sparse_shape',
            'n_qubits': n_qubits,
            'expected_shape': [2**n_qubits, 2**n_qubits],
            'all_sparse': True,
            'artifacts': ['metadata.json'],
        })
    
    def test_pauli_z_sparse_shape(self, test_subdir):
        """Test that Pauli Z has correct shape."""
        n_qubits = 4
        for site in range(n_qubits):
            sz = _pauli_z_sparse(n_qubits, site)
            assert sz.shape == (2**n_qubits, 2**n_qubits)
            assert issparse(sz)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_pauli_z_sparse_shape',
            'n_qubits': n_qubits,
            'expected_shape': [2**n_qubits, 2**n_qubits],
            'all_sparse': True,
            'artifacts': ['metadata.json'],
        })
    
    def test_pauli_x_squared_is_identity(self, test_subdir):
        """Test that sigma_x^2 = I."""
        n_qubits = 3
        site = 1
        sx = _pauli_x_sparse(n_qubits, site)
        sx2 = (sx @ sx).toarray()
        identity = np.eye(2**n_qubits)
        assert np.allclose(sx2, identity)

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(sx2, title=r'$\sigma_x^2$ (should be $I$)',
                                      output_path=test_subdir / "sx_squared.png")
        plt.close(fig)
        save_array_data(test_subdir / "sx2.npz", sx_squared=sx2)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_pauli_x_squared_is_identity',
            'n_qubits': n_qubits, 'site': site,
            'max_diff_from_I': float(np.max(np.abs(sx2 - identity))),
            'artifacts': ['sx_squared.png', 'sx2.npz'],
        })
    
    def test_pauli_z_squared_is_identity(self, test_subdir):
        """Test that sigma_z^2 = I."""
        n_qubits = 3
        site = 1
        sz = _pauli_z_sparse(n_qubits, site)
        sz2 = (sz @ sz).toarray()
        identity = np.eye(2**n_qubits)
        assert np.allclose(sz2, identity)

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(sz2, title=r'$\sigma_z^2$ (should be $I$)',
                                      output_path=test_subdir / "sz_squared.png")
        plt.close(fig)
        save_array_data(test_subdir / "sz2.npz", sz_squared=sz2)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_pauli_z_squared_is_identity',
            'n_qubits': n_qubits, 'site': site,
            'max_diff_from_I': float(np.max(np.abs(sz2 - identity))),
            'artifacts': ['sz_squared.png', 'sz2.npz'],
        })
    
    def test_projector_is_projector(self, test_subdir):
        """Test that P^2 = P for ground state projector."""
        n_qubits = 3
        site = 1
        P = _projector_sparse(n_qubits, site)
        P2 = (P @ P).toarray()
        assert np.allclose(P2, P.toarray())

        # --- Artifacts ---
        P_dense = P.toarray()
        fig = plot_hamiltonian_heatmap(P_dense, title=f'Projector (site={site})',
                                      output_path=test_subdir / "projector.png")
        plt.close(fig)
        save_array_data(test_subdir / "P.npz", projector=P_dense, P_squared=P2)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_projector_is_projector',
            'n_qubits': n_qubits, 'site': site,
            'max_diff_P2_P': float(np.max(np.abs(P2 - P_dense))),
            'artifacts': ['projector.png', 'P.npz'],
        })
    
    def test_projector_trace(self, test_subdir):
        """Test that projector has trace = 2^(N-1)."""
        n_qubits = 4
        site = 2
        P = _projector_sparse(n_qubits, site)
        expected_trace = 2**(n_qubits - 1)  # Half the states have |0> at site
        assert np.isclose(P.diagonal().sum(), expected_trace)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_projector_trace',
            'n_qubits': n_qubits, 'site': site,
            'trace': float(P.diagonal().sum()),
            'expected_trace': expected_trace,
            'artifacts': ['metadata.json'],
        })


class TestRydbergHamiltonian:
    """Test RydbergHamiltonian class."""
    
    @pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
    def test_dimension(self, N, test_subdir):
        """Test Hilbert space dimension is 2^N."""
        ham = RydbergHamiltonian(N=N, J=1.0)
        assert ham.dim == 2**N
        assert ham.get_sparse_matrix().shape == (2**N, 2**N)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_dimension',
            'N': N, 'dim': ham.dim,
            'expected_dim': 2**N,
            'artifacts': ['metadata.json'],
        })
    
    @pytest.mark.parametrize("N", [3, 4, 5])
    def test_hermiticity(self, N, test_subdir):
        """Test that Hamiltonian is Hermitian."""
        ham = RydbergHamiltonian(N=N, J=1.0)
        H = ham.get_matrix()
        assert np.allclose(H, H.conj().T)

        # --- Artifacts ---
        err = float(np.max(np.abs(H - H.conj().T)))
        fig = plot_hamiltonian_heatmap(H, title=f'Rydberg H (N={N})',
                                      output_path=test_subdir / "hamiltonian.png")
        plt.close(fig)
        save_array_data(test_subdir / "H.npz", hamiltonian=H)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_hermiticity',
            'N': N, 'dim': H.shape[0],
            'hermiticity_error': err,
            'artifacts': ['hamiltonian.png', 'H.npz'],
        })
    
    @pytest.mark.parametrize("periodic", [True, False])
    def test_boundary_conditions(self, periodic, test_subdir):
        """Test open vs periodic boundary conditions."""
        N = 4
        ham = RydbergHamiltonian(N=N, J=1.0, periodic=periodic)
        H = ham.get_sparse_matrix()
        
        # Both should be Hermitian
        assert np.allclose(H.toarray(), H.toarray().conj().T)
        
        # Periodic should have different spectrum
        ham_open = RydbergHamiltonian(N=N, J=1.0, periodic=False)
        ham_periodic = RydbergHamiltonian(N=N, J=1.0, periodic=True)
        
        E_open = np.linalg.eigvalsh(ham_open.get_matrix())
        E_periodic = np.linalg.eigvalsh(ham_periodic.get_matrix())
        
        # Spectra should differ (unless by coincidence)
        assert not np.allclose(E_open, E_periodic)

        # --- Artifacts ---
        apply_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
        ax.stem(np.arange(len(E_open)), E_open, linefmt='C0-', markerfmt='C0o',
                basefmt='k-', label='Open')
        ax.stem(np.arange(len(E_periodic)), E_periodic, linefmt='C1--',
                markerfmt='C1s', basefmt='k-', label='Periodic')
        ax.set_xlabel('Level index'); ax.set_ylabel('Energy')
        ax.set_title(f'Boundary comparison (N={N}, this run: periodic={periodic})')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(test_subdir / "spectrum_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "spectra.npz",
                       E_open=E_open, E_periodic=E_periodic)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_boundary_conditions',
            'N': N, 'periodic': periodic,
            'E_open_range': [float(E_open.min()), float(E_open.max())],
            'E_periodic_range': [float(E_periodic.min()), float(E_periodic.max())],
            'artifacts': ['spectrum_comparison.png', 'spectra.npz'],
        })
    
    def test_sparse_matrix_cached(self, test_subdir):
        """Test that sparse matrix is cached."""
        ham = RydbergHamiltonian(N=4, J=1.0)
        H1 = ham.get_sparse_matrix()
        H2 = ham.get_sparse_matrix()
        assert H1 is H2  # Same object

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_sparse_matrix_cached',
            'N': 4, 'same_object': True,
            'artifacts': ['metadata.json'],
        })
    
    def test_staggered_sz_hermiticity(self, test_subdir):
        """Test that staggered Sz is Hermitian."""
        ham = RydbergHamiltonian(N=5, J=1.0)
        Sz = ham.get_staggered_sz()
        assert np.allclose(Sz.toarray(), Sz.toarray().conj().T)

        # --- Artifacts ---
        Sz_dense = Sz.toarray()
        fig = plot_hamiltonian_heatmap(Sz_dense, title='Staggered $S_z$ (N=5)',
                                      output_path=test_subdir / "staggered_sz.png")
        plt.close(fig)
        save_array_data(test_subdir / "Sz.npz", staggered_sz=Sz_dense)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_staggered_sz_hermiticity',
            'N': 5, 'hermitian': True,
            'artifacts': ['staggered_sz.png', 'Sz.npz'],
        })
    
    def test_staggered_sz_diagonal(self, test_subdir):
        """Test that staggered Sz is diagonal."""
        ham = RydbergHamiltonian(N=4, J=1.0)
        Sz = ham.get_staggered_sz()
        Sz_dense = Sz.toarray()
        # Should be diagonal
        assert np.allclose(Sz_dense, np.diag(np.diag(Sz_dense)))

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(Sz_dense, title='Staggered $S_z$ (N=4, diagonal check)',
                                      output_path=test_subdir / "staggered_sz.png")
        plt.close(fig)
        save_array_data(test_subdir / "Sz.npz", staggered_sz=Sz_dense,
                       diagonal=np.diag(Sz_dense))
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_staggered_sz_diagonal',
            'N': 4,
            'is_diagonal': True,
            'diagonal_values': np.diag(Sz_dense).tolist(),
            'artifacts': ['staggered_sz.png', 'Sz.npz'],
        })
    
    def test_ground_state_normalization(self, test_subdir):
        """Test that ground state is normalized."""
        ham = RydbergHamiltonian(N=4, J=1.0)
        E0, psi0 = ham.get_ground_state()
        assert np.isclose(np.linalg.norm(psi0), 1.0)

        # --- Artifacts ---
        fig = plot_state_bar(psi0, title=f'Ground state (N=4, $E_0$={E0:.4f})',
                            output_path=test_subdir / "ground_state.png")
        plt.close(fig)
        save_array_data(test_subdir / "ground.npz", psi0=psi0)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_ground_state_normalization',
            'N': 4, 'E0': float(E0),
            'norm': float(np.linalg.norm(psi0)),
            'artifacts': ['ground_state.png', 'ground.npz'],
        })
    
    def test_ground_state_is_eigenstate(self, test_subdir):
        """Test that ground state is an eigenstate."""
        ham = RydbergHamiltonian(N=4, J=1.0)
        E0, psi0 = ham.get_ground_state()
        H = ham.get_matrix()
        
        # H|psi0> should equal E0|psi0>
        H_psi = H @ psi0
        assert np.allclose(H_psi, E0 * psi0)

        # --- Artifacts ---
        residual = np.abs(H_psi - E0 * psi0)
        apply_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].bar(np.arange(len(psi0)), np.abs(psi0), color='steelblue', edgecolor='navy')
        axes[0].set_title(f'$|\\psi_0\\rangle$ (N=4, $E_0$={E0:.4f})')
        axes[0].set_ylabel('$|\\psi_n|$')
        axes[1].bar(np.arange(len(residual)), residual, color='salmon', edgecolor='darkred')
        axes[1].set_title('$|H|\\psi_0\\rangle - E_0|\\psi_0\\rangle|$')
        axes[1].set_ylabel('Residual')
        fig.tight_layout()
        fig.savefig(test_subdir / "eigenstate_check.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "data.npz", psi0=psi0, residual=residual)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_ground_state_is_eigenstate',
            'N': 4, 'E0': float(E0),
            'max_residual': float(residual.max()),
            'artifacts': ['eigenstate_check.png', 'data.npz'],
        })
    
    def test_ground_state_is_lowest(self, test_subdir):
        """Test that ground state has lowest energy."""
        ham = RydbergHamiltonian(N=4, J=1.0)
        E0, psi0 = ham.get_ground_state()
        
        # Compare with full diagonalization
        eigenvalues = np.linalg.eigvalsh(ham.get_matrix())
        assert np.isclose(E0, eigenvalues.min())

        # --- Artifacts ---
        fig = plot_eigenvalue_spectrum(eigenvalues,
                                      title=f'Rydberg spectrum (N=4) — $E_0$={E0:.4f}',
                                      output_path=test_subdir / "spectrum.png")
        plt.close(fig)
        save_array_data(test_subdir / "eigenvalues.npz", eigenvalues=eigenvalues)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_ground_state_is_lowest',
            'N': 4, 'E0': float(E0),
            'E_min_diag': float(eigenvalues.min()),
            'n_levels': len(eigenvalues),
            'artifacts': ['spectrum.png', 'eigenvalues.npz'],
        })
    
    @pytest.mark.parametrize("J", [0.001, 0.1, 1.0, 10.0, 100.0])
    def test_different_J_values(self, J, test_subdir):
        """Test Hamiltonian construction for various J values."""
        ham = RydbergHamiltonian(N=4, J=J)
        H = ham.get_matrix()
        
        # Should still be Hermitian
        assert np.allclose(H, H.conj().T)
        
        # Check J is stored correctly
        assert ham.J == J

        # --- Artifacts ---
        eigenvalues = np.linalg.eigvalsh(H)
        fig = plot_eigenvalue_spectrum(eigenvalues,
                                      title=f'Rydberg spectrum (N=4, J={J})',
                                      output_path=test_subdir / "spectrum.png")
        plt.close(fig)
        save_array_data(test_subdir / "eigenvalues.npz", eigenvalues=eigenvalues)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_different_J_values',
            'N': 4, 'J': J,
            'hermiticity_error': float(np.max(np.abs(H - H.conj().T))),
            'E_range': [float(eigenvalues.min()), float(eigenvalues.max())],
            'artifacts': ['spectrum.png', 'eigenvalues.npz'],
        })
    
    @pytest.mark.parametrize("Omega", [0.5, 1.0, 2.0, 5.0])
    def test_different_Omega_values(self, Omega, test_subdir):
        """Test Hamiltonian construction for various Omega values."""
        ham = RydbergHamiltonian(N=4, J=1.0, Omega=Omega)
        H = ham.get_matrix()
        
        # Should still be Hermitian
        assert np.allclose(H, H.conj().T)
        
        # Check Omega is stored correctly
        assert ham.Omega == Omega
        assert ham.metadata['Omega'] == Omega

        # --- Artifacts ---
        eigenvalues = np.linalg.eigvalsh(H)
        fig = plot_eigenvalue_spectrum(eigenvalues,
                                      title=f'Rydberg spectrum (N=4, $\\Omega$={Omega})',
                                      output_path=test_subdir / "spectrum.png")
        plt.close(fig)
        save_array_data(test_subdir / "eigenvalues.npz", eigenvalues=eigenvalues)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_different_Omega_values',
            'N': 4, 'Omega': Omega,
            'hermiticity_error': float(np.max(np.abs(H - H.conj().T))),
            'E_range': [float(eigenvalues.min()), float(eigenvalues.max())],
            'artifacts': ['spectrum.png', 'eigenvalues.npz'],
        })
    
    def test_Omega_scales_transverse_field(self, test_subdir):
        """Test that Omega correctly scales the transverse field term."""
        # For N=1, H = Omega * sigma^x
        ham1 = RydbergHamiltonian(N=1, Omega=1.0)
        ham2 = RydbergHamiltonian(N=1, Omega=2.0)
        
        H1 = ham1.get_matrix()
        H2 = ham2.get_matrix()
        
        # H2 should be 2 * H1
        assert np.allclose(H2, 2.0 * H1)
        
        # Eigenvalues should scale by Omega
        eig1 = np.linalg.eigvalsh(H1)
        eig2 = np.linalg.eigvalsh(H2)
        assert np.allclose(eig2, 2.0 * eig1)

        # --- Artifacts ---
        apply_plot_style()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(np.abs(H1), cmap='viridis')
        axes[0].set_title('$H$ ($\\Omega=1$)')
        axes[1].imshow(np.abs(H2), cmap='viridis')
        axes[1].set_title('$H$ ($\\Omega=2$)')
        fig.suptitle('$\\Omega$ scaling test (N=1)')
        fig.tight_layout()
        fig.savefig(test_subdir / "omega_scaling.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_array_data(test_subdir / "data.npz", H1=H1, H2=H2,
                       eig1=eig1, eig2=eig2)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_Omega_scales_transverse_field',
            'eig_Omega1': eig1.tolist(),
            'eig_Omega2': eig2.tolist(),
            'ratio': (eig2 / eig1).tolist(),
            'artifacts': ['omega_scaling.png', 'data.npz'],
        })
    
    def test_with_J_creates_new_hamiltonian(self, test_subdir):
        """Test that with_J creates a new Hamiltonian."""
        ham1 = RydbergHamiltonian(N=4, J=1.0)
        ham2 = ham1.with_J(2.0)
        
        assert ham1.J == 1.0
        assert ham2.J == 2.0
        assert ham1 is not ham2

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_with_J_creates_new_hamiltonian',
            'J1': ham1.J, 'J2': ham2.J,
            'is_different_object': ham1 is not ham2,
            'artifacts': ['metadata.json'],
        })
    
    def test_invalid_N(self, test_subdir):
        """Test that invalid N raises error."""
        with pytest.raises(ValueError):
            RydbergHamiltonian(N=0, J=1.0)  # N must be >= 1
        
        with pytest.raises(ValueError):
            RydbergHamiltonian(N=15, J=1.0)  # N > 14 exceeds memory

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_invalid_N',
            'invalid_N_values': [0, 15],
            'raised': 'ValueError',
            'artifacts': ['metadata.json'],
        })
    
    def test_metadata(self, test_subdir):
        """Test that metadata is stored correctly."""
        ham = RydbergHamiltonian(N=4, J=2.5, periodic=True)
        
        assert ham.metadata['model'] == 'Rydberg'
        assert ham.metadata['N'] == 4
        assert ham.metadata['J'] == 2.5
        assert ham.metadata['periodic'] == True

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_metadata',
            'ham_metadata': dict(ham.metadata),
            'artifacts': ['metadata.json'],
        })


class TestQuTiPIntegration:
    """Test QuTiP integration methods."""
    
    @pytest.fixture
    def hamiltonian(self):
        return RydbergHamiltonian(N=4, J=1.0)
    
    def test_get_qutip_hamiltonian(self, hamiltonian, test_subdir):
        """Test conversion to QuTiP Qobj."""
        pytest.importorskip("qutip")
        import qutip as qt
        
        H_qobj = hamiltonian.get_qutip_hamiltonian()
        
        assert isinstance(H_qobj, qt.Qobj)
        assert H_qobj.isherm
        assert H_qobj.shape == (hamiltonian.dim, hamiltonian.dim)

        # --- Artifacts ---
        H_np = H_qobj.full()
        fig = plot_hamiltonian_heatmap(H_np, title='QuTiP Rydberg H (N=4)',
                                      output_path=test_subdir / "qutip_hamiltonian.png")
        plt.close(fig)
        save_array_data(test_subdir / "H.npz", hamiltonian=H_np)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_get_qutip_hamiltonian',
            'dim': hamiltonian.dim,
            'is_hermitian': bool(H_qobj.isherm),
            'artifacts': ['qutip_hamiltonian.png', 'H.npz'],
        })
    
    def test_get_driven_hamiltonian(self, hamiltonian, test_subdir):
        """Test driven Hamiltonian format."""
        pytest.importorskip("qutip")
        
        H_driven = hamiltonian.get_driven_hamiltonian(A=0.1, omega=1.0)
        
        assert isinstance(H_driven, list)
        assert len(H_driven) == 2  # [H0, [Sz, coeff]]
        assert len(H_driven[1]) == 2  # [Sz, coeff_func]
        assert callable(H_driven[1][1])  # Coefficient is callable

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_get_driven_hamiltonian',
            'A': 0.1, 'omega': 1.0,
            'n_terms': len(H_driven),
            'coeff_is_callable': True,
            'artifacts': ['metadata.json'],
        })
    
    def test_energy_conservation_no_drive(self, hamiltonian, test_subdir):
        """Test that energy is conserved when A=0."""
        pytest.importorskip("qutip")
        import qutip as qt
        
        N = hamiltonian.N
        
        # Get ground state
        E0, psi0 = hamiltonian.get_ground_state()
        psi0_qobj = qt.Qobj(psi0, dims=[[2]*N, [1]*N])
        
        # Evolve with A=0 (no drive)
        H_driven = hamiltonian.get_driven_hamiltonian(A=0.0, omega=1.0)
        times = np.linspace(0, 5, 51)
        opts = {"store_states": True}
        result = qt.mesolve(H_driven, psi0_qobj, times, e_ops=[], options=opts)
        
        # Compute energies - use Hamiltonian methods that have correct dims
        H0_qobj = hamiltonian.get_qutip_hamiltonian()
        # Create Sz with matching tensor dimensions
        Sz_sparse = hamiltonian.get_staggered_sz()
        Sz_qobj = qt.Qobj(Sz_sparse, dims=[[2]*N, [2]*N], isherm=True)
        
        observables = compute_energy_observables(
            result, H0_qobj, Sz_qobj, A=0.0, omega=1.0, times=times
        )
        
        # Energy should be constant (equal to E0)
        assert np.allclose(observables['E_H0'], E0, atol=1e-8)
        assert np.allclose(observables['dE_H0_dt'], 0.0, atol=1e-6)

        # --- Artifacts ---
        fig = plot_energy_evolution(times, observables['E_H0'],
                                   title=f'Energy conservation (A=0, N={N})',
                                   output_path=test_subdir / "energy.png")
        plt.close(fig)
        save_array_data(test_subdir / "energy.npz",
                       times=times, E_H0=observables['E_H0'],
                       dE_dt=observables['dE_H0_dt'])
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_energy_conservation_no_drive',
            'N': N, 'E0': float(E0),
            'E_H0_mean': float(np.mean(observables['E_H0'])),
            'E_H0_std': float(np.std(observables['E_H0'])),
            'max_dE_dt': float(np.max(np.abs(observables['dE_H0_dt']))),
            'artifacts': ['energy.png', 'energy.npz'],
        })


class TestSparseEfficiency:
    """Test that sparse matrices are used efficiently."""
    
    def test_sparse_matrix_density(self, test_subdir):
        """Test that Hamiltonian is sparse (low density)."""
        N = 8  # Larger N for better sparsity demonstration
        ham = RydbergHamiltonian(N=N, J=1.0)
        H = ham.get_sparse_matrix()
        
        total_elements = H.shape[0] * H.shape[1]
        nonzero = H.nnz
        density = nonzero / total_elements
        
        # Density should be much less than 1 (sparse)
        # For larger N, density scales as O(N/2^N)
        assert density < 0.15  # Less than 15% non-zero

        # --- Artifacts ---
        fig = plot_sparsity_pattern(H.toarray(),
                                   title=f'Rydberg H sparsity (N={N}, density={density:.4f})',
                                   output_path=test_subdir / "sparsity.png")
        plt.close(fig)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_sparse_matrix_density',
            'N': N, 'dim': H.shape[0],
            'nnz': nonzero,
            'total_elements': total_elements,
            'density': float(density),
            'artifacts': ['sparsity.png'],
        })
    
    def test_staggered_sz_is_diagonal(self, test_subdir):
        """Test that staggered Sz is diagonal (only diagonal elements)."""
        ham = RydbergHamiltonian(N=6, J=1.0)
        Sz = ham.get_staggered_sz()
        Sz_dense = Sz.toarray()
        
        # Check that off-diagonal elements are zero
        off_diag_mask = ~np.eye(ham.dim, dtype=bool)
        assert np.allclose(Sz_dense[off_diag_mask], 0.0)
        
        # Check that it's stored efficiently (at most dim non-zeros)
        # Note: sparse format may eliminate exact zeros from diagonal
        assert Sz.nnz <= ham.dim

        # --- Artifacts ---
        fig = plot_hamiltonian_heatmap(Sz_dense,
                                      title=f'Staggered $S_z$ (N=6, nnz={Sz.nnz})',
                                      output_path=test_subdir / "staggered_sz.png")
        plt.close(fig)
        save_array_data(test_subdir / "Sz.npz", staggered_sz=Sz_dense,
                       diagonal=np.diag(Sz_dense))
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_staggered_sz_is_diagonal',
            'N': 6, 'dim': ham.dim,
            'nnz': int(Sz.nnz),
            'max_off_diag': float(np.max(np.abs(Sz_dense[off_diag_mask]))),
            'artifacts': ['staggered_sz.png', 'Sz.npz'],
        })


class TestResonanceExtraction:
    """Test resonance frequency extraction from omega scans."""
    
    @pytest.fixture
    def omega_scan_results(self):
        """Run a quick omega scan for N=1 to test resonance extraction."""
        omega_values = np.linspace(1.0, 3.0, 11)
        results = run_omega_scan(
            omega_values=omega_values,
            J=1.0,
            A=0.1,
            N=1,
            t_max=50.0,
            dt=1.0,
            progress=False
        )
        return results
    
    def test_omega_scan_returns_resonance_info(self, omega_scan_results, test_subdir):
        """Test that run_omega_scan includes resonance information."""
        # Check new keys exist
        assert 'omega_max_vs_t' in omega_scan_results
        assert 'omega_max_idx_vs_t' in omega_scan_results
        assert 'E_H0_max_vs_t' in omega_scan_results
        assert 'omega_resonance' in omega_scan_results
        assert 'omega_resonance_idx' in omega_scan_results
        
        # Check shapes
        n_times = len(omega_scan_results['times'])
        assert omega_scan_results['omega_max_vs_t'].shape == (n_times,)
        assert omega_scan_results['omega_max_idx_vs_t'].shape == (n_times,)
        assert omega_scan_results['E_H0_max_vs_t'].shape == (n_times,)

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_omega_scan_returns_resonance_info',
            'n_times': n_times,
            'omega_resonance': float(omega_scan_results['omega_resonance']),
            'keys': list(omega_scan_results.keys()),
            'artifacts': ['metadata.json'],
        })
    
    def test_resonance_at_two_omega(self, omega_scan_results, test_subdir):
        """Test that resonance is found at omega = 2*Omega for N=1."""
        # For N=1 with Omega=1, resonance should be at omega = 2
        omega_res = omega_scan_results['omega_resonance']
        expected_resonance = 2.0
        
        # Allow 10% tolerance due to discrete omega sampling
        assert abs(omega_res - expected_resonance) / expected_resonance < 0.1

        # --- Artifacts ---
        E_H0 = omega_scan_results['E_H0']
        times = omega_scan_results['times']
        omega_values = omega_scan_results['omega_values']
        fig = plot_omega_scan_heatmap(omega_values, times, E_H0,
                                     title=f'$\\omega$ scan — resonance at {omega_res:.3f}',
                                     output_path=test_subdir / "omega_scan.png")
        plt.close(fig)
        save_array_data(test_subdir / "scan.npz",
                       omega_values=omega_values, times=times, E_H0=E_H0)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_resonance_at_two_omega',
            'omega_resonance': float(omega_res),
            'expected_resonance': expected_resonance,
            'relative_error': float(abs(omega_res - expected_resonance) / expected_resonance),
            'artifacts': ['omega_scan.png', 'scan.npz'],
        })
    
    def test_extract_resonance_max_energy(self, omega_scan_results, test_subdir):
        """Test extract_resonance_frequency with max_energy method."""
        res = extract_resonance_frequency(omega_scan_results, method='max_energy')
        
        assert 'omega_resonance' in res
        assert 'omega_max_vs_t' in res
        assert 'E_max_at_resonance' in res
        assert res['method'] == 'max_energy'
        
        # Resonance should be near omega = 2
        assert abs(res['omega_resonance'] - 2.0) < 0.2

        # --- Artifacts ---
        apply_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
        ax.plot(omega_scan_results['times'], res['omega_max_vs_t'],
                linewidth=2, label='$\\omega_{\\mathrm{max}}(t)$')
        ax.axhline(res['omega_resonance'], color='r', ls='--',
                   label=f'resonance = {res["omega_resonance"]:.3f}')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('$\\omega$')
        ax.set_title('max_energy resonance extraction')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(test_subdir / "resonance_max_energy.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_extract_resonance_max_energy',
            'method': 'max_energy',
            'omega_resonance': float(res['omega_resonance']),
            'E_max_at_resonance': float(res['E_max_at_resonance']),
            'artifacts': ['resonance_max_energy.png'],
        })
    
    def test_extract_resonance_time_averaged(self, omega_scan_results, test_subdir):
        """Test extract_resonance_frequency with time_averaged method."""
        res = extract_resonance_frequency(omega_scan_results, method='time_averaged')
        
        assert res['method'] == 'time_averaged'
        assert abs(res['omega_resonance'] - 2.0) < 0.2

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_extract_resonance_time_averaged',
            'method': 'time_averaged',
            'omega_resonance': float(res['omega_resonance']),
            'artifacts': ['metadata.json'],
        })
    
    def test_extract_resonance_max_at_each_time(self, omega_scan_results, test_subdir):
        """Test extract_resonance_frequency with max_at_each_time method."""
        res = extract_resonance_frequency(omega_scan_results, method='max_at_each_time')
        
        assert res['method'] == 'max_at_each_time'
        # omega_max_vs_t should have same length as times
        assert len(res['omega_max_vs_t']) == len(omega_scan_results['times'])

        # --- Artifacts ---
        apply_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
        ax.plot(omega_scan_results['times'], res['omega_max_vs_t'], 'o-', ms=3)
        ax.set_xlabel('Time $t$'); ax.set_ylabel('$\\omega_{\\mathrm{max}}$')
        ax.set_title('max_at_each_time resonance extraction')
        ax.grid(True, alpha=0.3)
        fig.savefig(test_subdir / "resonance_max_each_time.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_csv_table(test_subdir / "omega_max_vs_t.csv", {
            'time': omega_scan_results['times'],
            'omega_max': res['omega_max_vs_t'],
        }, header='max_at_each_time resonance extraction')
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_extract_resonance_max_at_each_time',
            'method': 'max_at_each_time',
            'n_times': len(res['omega_max_vs_t']),
            'artifacts': ['resonance_max_each_time.png', 'omega_max_vs_t.csv'],
        })
    
    def test_extract_resonance_invalid_method(self, omega_scan_results, test_subdir):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            extract_resonance_frequency(omega_scan_results, method='invalid_method')

        # --- Artifacts ---
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_extract_resonance_invalid_method',
            'invalid_method': 'invalid_method',
            'raised': 'ValueError',
            'artifacts': ['metadata.json'],
        })
    
    def test_omega_max_vs_t_values_in_range(self, omega_scan_results, test_subdir):
        """Test that omega_max_vs_t values are within scanned range."""
        omega_values = omega_scan_results['omega_values']
        omega_max_vs_t = omega_scan_results['omega_max_vs_t']
        
        # All values should be in the scanned omega range
        assert np.all(omega_max_vs_t >= omega_values.min())
        assert np.all(omega_max_vs_t <= omega_values.max())

        # --- Artifacts ---
        apply_plot_style()
        fig, ax = plt.subplots(figsize=PLOT_STYLE['figsize'])
        ax.plot(omega_scan_results['times'], omega_max_vs_t, 'o-', ms=3)
        ax.axhline(omega_values.min(), color='gray', ls=':', label='scan range')
        ax.axhline(omega_values.max(), color='gray', ls=':')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('$\\omega_{\\mathrm{max}}$')
        ax.set_title('$\\omega_{\\mathrm{max}}(t)$ within scan range')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(test_subdir / "omega_max_range.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        save_metadata(test_subdir / "metadata.json", {
            'test_name': 'test_omega_max_vs_t_values_in_range',
            'omega_scan_min': float(omega_values.min()),
            'omega_scan_max': float(omega_values.max()),
            'omega_max_min': float(omega_max_vs_t.min()),
            'omega_max_max': float(omega_max_vs_t.max()),
            'artifacts': ['omega_max_range.png'],
        })