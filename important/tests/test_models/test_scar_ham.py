"""
Tests for ``get_scar_ham`` -- the clean periodic PXP Hamiltonian with the
-0.026 sigma-z perturbation.

The centrepiece is the comparison against ``reference_pxp``, which builds the
same operator from scratch in the full 2**N space with Kronecker products and
then restricts it to the blockade subspace.  That reference shares no code
with ``get_scar_ham``, so agreement between the two is meaningful.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import reference_pxp as ref
from data_utils import save_array_data, save_csv_table, save_metadata
from plot_utils import (
    PLOT_STYLE,
    apply_plot_style,
    plot_eigenvalue_spectrum,
    plot_hamiltonian_heatmap,
    plot_sparsity_pattern,
)

SIZES = [4, 6, 8, 10]


def _degenerate_blocks(sorted_energies, tol=1e-9):
    """
    Group indices of a sorted eigenvalue array into degenerate blocks.

    Returns a list of index arrays, one per distinct energy.  Quantities that
    are basis dependent inside a degenerate subspace (individual overlaps,
    individual eigenvectors) are only meaningful when summed over a block.
    """
    edges = np.flatnonzero(np.diff(sorted_energies) > tol) + 1
    return np.split(np.arange(sorted_energies.size), edges)


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_hamiltonian_is_square_and_matches_basis(scar_system, N):
    H0, eigenvalues, eigenstates, psi0, basisList = scar_system(N)
    D = len(basisList)

    assert H0.shape == (D, D)
    assert len(eigenvalues) == D
    assert len(eigenstates) == D
    assert psi0.shape == (D, 1)


@pytest.mark.parametrize("N", SIZES)
def test_hamiltonian_is_hermitian(scar_system, N):
    H0 = scar_system(N)[0]
    dense = H0.full()
    assert np.allclose(dense, dense.conj().T, atol=1e-14)


@pytest.mark.parametrize("N", SIZES)
def test_hamiltonian_is_real(scar_system, N):
    """The clean PXP Hamiltonian is real symmetric.

    ``make_scar_states.py`` relies on this when it casts to float64 and calls
    ``np.linalg.eigh`` on the real array to save memory at large N.
    """
    dense = scar_system(N)[0].full()
    assert np.abs(dense.imag).max() < 1e-14


@pytest.mark.parametrize("N", SIZES)
def test_hamiltonian_has_zero_diagonal(scar_system, N):
    """Every term is an off-diagonal spin flip, so the diagonal vanishes."""
    dense = scar_system(N)[0].full()
    assert np.abs(np.diag(dense)).max() < 1e-14


@pytest.mark.parametrize("N", SIZES)
def test_hamiltonian_is_sparse(scar_system, N, test_subdir):
    """
    Each basis state couples to at most N others, so the number of nonzeros
    must grow like N*D and not D**2.  This is the property that makes the
    N=20 runs feasible at all.
    """
    H0, _, _, _, basisList = scar_system(N)
    D = len(basisList)
    dense = H0.full()
    nnz = int(np.count_nonzero(np.abs(dense) > 1e-14))

    assert nnz <= N * D
    if D > 8:
        assert nnz < D * D

    fig = plot_sparsity_pattern(dense, title=f"Scar Hamiltonian sparsity (N={N})",
                                output_path=test_subdir / "sparsity.png")
    plt.close(fig)
    save_metadata(
        test_subdir / "metadata.json",
        {"N": N, "dimension": D, "nnz": nnz, "density": nnz / (D * D)},
    )


# --------------------------------------------------------------------------
# Comparison against the independent reference
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_matches_independent_reference_elementwise(scar_system, N, test_subdir):
    """Every matrix element must match the brute-force full-space build."""
    H0, _, _, _, basisList = scar_system(N)

    produced = H0.full().real
    expected = ref.reference_hamiltonian(N, basisList)
    residual = np.abs(produced - expected)

    assert residual.max() < 1e-12, f"max elementwise error {residual.max():.3e}"

    fig = plot_hamiltonian_heatmap(produced, title=f"Scar Hamiltonian (N={N})",
                                   output_path=test_subdir / "hamiltonian.png")
    plt.close(fig)
    save_array_data(
        test_subdir / "hamiltonian_comparison.npz",
        produced=produced,
        reference=expected,
        residual=residual,
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"N": N, "dimension": produced.shape[0], "max_abs_error": float(residual.max())},
    )


@pytest.mark.parametrize("N", SIZES)
def test_spectrum_matches_independent_reference(scar_system, N, test_subdir):
    """
    Sorted eigenvalues must agree with the reference, and the Z2 spectral
    weights must agree too -- the weights are what make the scar tower
    visible, so matching energies alone would not be enough.
    """
    _, eigenvalues, eigenstates, psi0, basisList = scar_system(N)

    order = np.argsort(np.asarray(eigenvalues, dtype=float))
    produced_energy = np.asarray(eigenvalues, dtype=float)[order]

    reference_matrix = ref.reference_hamiltonian(N, basisList)
    reference_energy, reference_vectors = np.linalg.eigh(reference_matrix)

    np.testing.assert_allclose(produced_energy, reference_energy, atol=1e-10, rtol=1e-10)
    energy_difference = produced_energy - reference_energy

    # Z2 spectral weights.  psi0 is a basis vector, so the overlap with an
    # eigenstate is just that eigenvector's amplitude at the Z2 index.
    z2_vector = psi0.full().ravel()
    produced_weight = np.array(
        [abs(np.vdot(z2_vector, eigenstates[i].full().ravel())) ** 2 for i in order]
    )
    reference_weight = np.abs(reference_vectors.conj().T @ z2_vector) ** 2

    # Completeness: the Z2 state is fully resolved in either eigenbasis.
    assert produced_weight.sum() == pytest.approx(1.0, abs=1e-10)
    assert reference_weight.sum() == pytest.approx(1.0, abs=1e-10)

    # This spectrum is highly degenerate (N=10 has 123 levels in only 67
    # distinct energies).  Inside a degenerate block the eigenvectors are not
    # unique -- any rotation within the block is equally valid -- so QuTiP and
    # numpy.linalg.eigh split the Z2 weight differently between degenerate
    # partners.  That split carries no physics.
    #
    # What IS basis independent is the total weight per degenerate block,
    # because that is the trace of the Z2 projector against the block's
    # spectral projector.  Compare those.
    blocks = _degenerate_blocks(reference_energy)
    produced_block_weight = np.array([produced_weight[b].sum() for b in blocks])
    reference_block_weight = np.array([reference_weight[b].sum() for b in blocks])

    np.testing.assert_allclose(
        produced_block_weight, reference_block_weight, atol=1e-9
    )

    fig = plot_eigenvalue_spectrum(produced_energy, title=f"Scar spectrum (N={N})",
                                   output_path=test_subdir / "spectrum.png")
    plt.close(fig)

    apply_plot_style()
    fig2, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.semilogy(produced_energy, np.maximum(produced_weight, 1e-18), ".", ms=10)
    ax.set_xlabel("Energy")
    ax.set_ylabel(r"$|\langle Z_2 | E \rangle|^2$")
    ax.set_title(f"Z2 spectral weight (N={N})")
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])
    fig2.savefig(test_subdir / "z2_weight.png", dpi=PLOT_STYLE["save_dpi"],
                 bbox_inches="tight")
    plt.close(fig2)

    save_csv_table(
        test_subdir / "spectrum_comparison.csv",
        {
            "level_index": np.arange(produced_energy.size),
            "energy": produced_energy,
            "reference_energy": reference_energy,
            "energy_difference": energy_difference,
            "z2_weight": produced_weight,
        },
        header=f"N={N} scar spectrum vs independent reference",
    )
    save_metadata(
        test_subdir / "metadata.json",
        {
            "N": N,
            "max_abs_energy_difference": float(np.abs(energy_difference).max()),
            "total_z2_weight": float(produced_weight.sum()),
        },
    )


# --------------------------------------------------------------------------
# Initial state and eigenstates
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_psi0_is_the_z2_basis_vector(scar_functions, scar_system, N):
    _, _, _, psi0, basisList = scar_system(N)

    z2_index = basisList.index(scar_functions.z2_initial(N))
    vector = psi0.full().ravel()

    assert vector[z2_index] == pytest.approx(1.0)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    assert np.count_nonzero(np.abs(vector) > 1e-14) == 1


@pytest.mark.parametrize("N", SIZES)
def test_eigenstates_are_orthonormal(scar_system, N):
    _, _, eigenstates, _, basisList = scar_system(N)

    V = np.column_stack([s.full().ravel() for s in eigenstates])
    gram = V.conj().T @ V

    assert np.abs(gram - np.eye(len(basisList))).max() < 1e-10


@pytest.mark.parametrize("N", SIZES)
def test_eigenvalues_are_ascending(scar_system, N):
    eigenvalues = np.asarray(scar_system(N)[1], dtype=float)
    assert np.all(np.diff(eigenvalues) >= -1e-12)


@pytest.mark.parametrize("N", SIZES)
def test_eigenstates_solve_the_eigenproblem(scar_system, N):
    """H|E> = E|E> for a few states, spot-checked directly."""
    H0, eigenvalues, eigenstates, _, _ = scar_system(N)
    dense = H0.full()

    for i in (0, len(eigenstates) // 2, len(eigenstates) - 1):
        v = eigenstates[i].full().ravel()
        residual = dense @ v - eigenvalues[i] * v
        assert np.linalg.norm(residual) < 1e-10


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 6, 8])
def test_diagonalize_false_skips_the_spectrum(scar_functions, N):
    """
    ``diagonalize=False`` is the whole reason the N=20 runs fit in memory:
    it must return the Hamiltonian but no eigen-decomposition.
    """
    H0, eigenvalues, eigenstates, psi0, basisList = scar_functions.get_scar_ham(
        N, diagonalize=False
    )

    assert eigenvalues is None
    assert eigenstates is None
    assert H0.shape == (len(basisList), len(basisList))
    assert psi0.shape == (len(basisList), 1)


@pytest.mark.parametrize("N", [4, 6, 8])
def test_diagonalize_false_gives_the_same_hamiltonian(scar_functions, scar_system, N):
    lazy = scar_functions.get_scar_ham(N, diagonalize=False)[0]
    eager = scar_system(N)[0]
    assert np.abs(lazy.full() - eager.full()).max() < 1e-14


@pytest.mark.parametrize("N", [4, 6, 8])
def test_hamiltonian_is_linear_in_ohms(scar_functions, scar_system, N):
    """
    Both terms carry a single factor of ``ohms``, so H(ohms) = ohms * H(1).
    If the perturbation ever picked up a different power this would catch it.
    """
    base = scar_system(N)[0].full()
    scaled = scar_functions.get_scar_ham(N, ohms=2.5, diagonalize=False)[0].full()

    assert np.abs(scaled - 2.5 * base).max() < 1e-12


@pytest.mark.parametrize("N", [3, 5, 7])
def test_odd_chain_length_is_rejected(scar_functions, N):
    """The Neel state does not exist on an odd ring; the assert must fire."""
    with pytest.raises(AssertionError):
        scar_functions.get_scar_ham(N, diagonalize=False)


@pytest.mark.parametrize("N", [6, 8, 10])
def test_perturbation_is_actually_present(scar_system, N):
    """
    Guard against the -0.026 sigma-z term being dropped, zeroed, or rescaled.

    Compared against the reference built at ``pert=0`` (must differ) and at
    ``pert=-0.026`` (must match).  Note that the perturbation does NOT break
    particle-hole symmetry -- see ``test_spectrum_is_particle_hole_symmetric``
    below -- so the spectrum alone cannot detect it.  The matrix elements can.
    """
    H0, _, _, _, basisList = scar_system(N)
    produced = H0.full().real

    bare_only = ref.restrict(ref.full_space_hamiltonian(N, pert=0.0), basisList)
    with_perturbation = ref.reference_hamiltonian(N, basisList)

    assert np.abs(produced - with_perturbation).max() < 1e-12
    assert np.abs(produced - bare_only).max() > 1e-3, "the -0.026 term is missing"


@pytest.mark.parametrize("N", [6, 8, 10])
def test_spectrum_is_particle_hole_symmetric(scar_system, N):
    """
    The PXP chiral symmetry C = prod of sigma-z on one sublattice anticommutes
    with every PXP term, and the sigma-z perturbation carries only diagonal
    factors, so it inherits the same anticommutation.  The spectrum is
    therefore symmetric about E = 0, with an exact zero mode at the centre --
    which is what makes the E=0 scar in ``get_zero_scar`` well defined.
    """
    energies = np.sort(np.asarray(scar_system(N)[1], dtype=float))

    assert np.abs(energies + energies[::-1]).max() < 1e-9
    assert np.abs(energies).min() < 1e-9, "no exact zero mode in the spectrum"
