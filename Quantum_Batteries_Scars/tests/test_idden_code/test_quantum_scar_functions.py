"""Tests for the student quantum scar helper functions."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from data_utils import save_array_data, save_csv_table, save_metadata
from plot_utils import PLOT_STYLE, apply_plot_style
from quantumScarFunctions import (
    binNoConsecOnesEfficient,
    binToDeci,
    coeff,
    const,
    embed_scar_state_to_full,
    get_dis_scar_ham,
    get_scar_H1,
    get_scar_ham,
    timed_const,
    timed_drive,
    z2_initial,
)


def _periodic_blockade_basis(length: int) -> list[str]:
    basis = binNoConsecOnesEfficient(length)
    return [state for state in basis if not (state[0] == "1" and state[-1] == "1")]


def _sorted_spectrum_with_z2_weights(eigenvalues, eigenstates, psi0):
    energies = np.asarray(eigenvalues, dtype=float)
    weights = np.array([abs(psi0.overlap(state)) ** 2 for state in eigenstates], dtype=float)
    order = np.argsort(energies)
    return energies[order], weights[order]


def _sorted_reference_spectrum(eigenvalues, z2_weights):
    energies = np.asarray(eigenvalues, dtype=float)
    weights = np.asarray(z2_weights, dtype=float)
    order = np.argsort(energies)
    return energies[order], weights[order]


def _save_student_quspin_comparison(
    test_subdir,
    label: str,
    student_energy: np.ndarray,
    quspin_energy: np.ndarray,
    student_z2_weight: np.ndarray,
    quspin_z2_weight: np.ndarray,
) -> dict:
    level_index = np.arange(student_energy.size)
    energy_difference = student_energy - quspin_energy
    z2_weight_difference = student_z2_weight - quspin_z2_weight

    save_array_data(
        test_subdir / f"{label}_student_quspin_comparison.npz",
        level_index=level_index,
        student_energy=student_energy,
        quspin_energy=quspin_energy,
        energy_difference=energy_difference,
        student_z2_weight=student_z2_weight,
        quspin_z2_weight=quspin_z2_weight,
        z2_weight_difference=z2_weight_difference,
    )
    save_csv_table(
        test_subdir / f"{label}_student_quspin_comparison.csv",
        {
            "level_index": level_index,
            "student_energy": student_energy,
            "quspin_energy": quspin_energy,
            "energy_difference": energy_difference,
            "student_z2_weight": student_z2_weight,
            "quspin_z2_weight": quspin_z2_weight,
            "z2_weight_difference": z2_weight_difference,
        },
        header=f"{label} student-vs-QuSpin spectrum comparison",
    )

    apply_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    axes[0].plot(level_index, student_energy, "o", label="student")
    axes[0].plot(level_index, quspin_energy, "-", label="QuSpin")
    axes[0].set_xlabel("Sorted level index")
    axes[0].set_ylabel("Energy")
    axes[0].set_title(f"{label}: sorted spectra")
    axes[0].grid(True, alpha=PLOT_STYLE["grid_alpha"])
    axes[0].legend()

    axes[1].plot(level_index, np.abs(energy_difference), color="tab:red")
    axes[1].set_xlabel("Sorted level index")
    axes[1].set_ylabel("Absolute energy difference")
    axes[1].set_title(f"{label}: energy residual")
    axes[1].grid(True, alpha=PLOT_STYLE["grid_alpha"])

    axes[2].scatter(quspin_energy, quspin_z2_weight, label="QuSpin", s=24)
    axes[2].scatter(student_energy, student_z2_weight, label="student", marker="x", s=28)
    axes[2].set_xlabel("Energy")
    axes[2].set_ylabel("Z2 weight")
    axes[2].set_title(f"{label}: Z2 weight vs eigenenergy")
    axes[2].grid(True, alpha=PLOT_STYLE["grid_alpha"])
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(
        test_subdir / f"{label}_student_quspin_comparison.png",
        dpi=PLOT_STYLE["save_dpi"],
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "max_abs_energy_difference": float(np.max(np.abs(energy_difference))),
        "max_abs_z2_weight_difference": float(np.max(np.abs(z2_weight_difference))),
        "comparison_npz": f"{label}_student_quspin_comparison.npz",
        "comparison_csv": f"{label}_student_quspin_comparison.csv",
        "comparison_plot": f"{label}_student_quspin_comparison.png",
    }


@pytest.mark.parametrize(
    ("bitstring", "expected"),
    [("0", 0), ("1", 1), ("1010", 10), ("1111", 15), ("1000000000", 512)],
)
def test_bin_to_deci(bitstring, expected, test_subdir):
    assert binToDeci(bitstring) == expected

    save_metadata(
        test_subdir / "metadata.json",
        {"bitstring": bitstring, "expected": expected},
    )


def test_periodic_basis_matches_quspin_reference(constrained_basis_npz, test_subdir):
    length = 10
    student_basis = set(_periodic_blockade_basis(length))

    with np.load(constrained_basis_npz, allow_pickle=False) as reference:
        quspin_basis = set(reference["constrained_basis_bit_strings"].astype(str))
        reference_dimension = int(reference["constrained_basis_dimension"])

    assert student_basis == quspin_basis
    assert len(student_basis) == reference_dimension

    save_metadata(
        test_subdir / "metadata.json",
        {
            "length": length,
            "basis_dimension": len(student_basis),
            "reference_file": constrained_basis_npz.name,
        },
    )


def test_reference_npz_shapes_are_consistent(student_hamiltonian_npz, test_subdir):
    with np.load(student_hamiltonian_npz, allow_pickle=False) as reference:
        basis_dimension = int(reference["constrained_basis_dimension"])
        clean_eigenvalues = reference["student_clean_eigenvalues"]
        clean_weights = reference["student_clean_z2_weights"]
        disordered_eigenvalues = reference["student_disordered_eigenvalues"]
        disordered_weights = reference["student_disordered_z2_weights"]

    assert clean_eigenvalues.shape == (basis_dimension,)
    assert clean_weights.shape == (basis_dimension,)
    assert disordered_eigenvalues.shape == (basis_dimension,)
    assert disordered_weights.shape == (basis_dimension,)
    assert np.all(np.isfinite(clean_eigenvalues))
    assert np.all(clean_weights >= 0.0)
    assert np.isclose(np.sum(clean_weights), 1.0)

    save_array_data(
        test_subdir / "reference_shapes.npz",
        clean_eigenvalues=clean_eigenvalues,
        clean_weights=clean_weights,
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"basis_dimension": basis_dimension, "reference_file": student_hamiltonian_npz.name},
    )


def test_drive_coefficients_and_z2_state(test_subdir):
    assert z2_initial(6) == "101010"
    assert coeff(np.pi / 2, A=2.0, omega=1.0) == pytest.approx(2.0)
    assert const(3.0, A=2.0) == pytest.approx(6.0)
    assert timed_drive(0.5, A=2.0, omega=np.pi, limit=1.0) == pytest.approx(2.0)
    assert timed_drive(1.5, A=2.0, omega=np.pi, limit=1.0) == pytest.approx(0.0)
    assert timed_const(0.5, A=2.0, limit=1.0) == pytest.approx(1.0)
    assert timed_const(1.5, A=2.0, limit=1.0) == pytest.approx(0.0)

    save_metadata(test_subdir / "metadata.json", {"length": 6, "z2": z2_initial(6)})


def test_get_scar_ham_basic_properties(test_subdir):
    length = 6
    hamiltonian, eigenvalues, eigenstates, psi0, basis_list = get_scar_ham(length)
    dense_hamiltonian = hamiltonian.full()

    assert dense_hamiltonian.shape == (len(basis_list), len(basis_list))
    assert np.allclose(dense_hamiltonian, dense_hamiltonian.conj().T)
    assert len(eigenvalues) == len(basis_list)
    assert all(state.norm() == pytest.approx(1.0) for state in eigenstates)

    z2_index = basis_list.index(z2_initial(length))
    psi0_dense = psi0.full().ravel()
    assert psi0_dense[z2_index] == pytest.approx(1.0)
    assert np.linalg.norm(psi0_dense) == pytest.approx(1.0)

    save_array_data(
        test_subdir / "scar_hamiltonian.npz",
        hamiltonian=dense_hamiltonian,
        eigenvalues=np.asarray(eigenvalues, dtype=float),
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"length": length, "basis_dimension": len(basis_list), "z2_index": z2_index},
    )


def test_clean_spectrum_matches_independent_quspin(student_hamiltonian_npz, test_subdir):
    length = 10
    _, eigenvalues, eigenstates, psi0, basis_list = get_scar_ham(length)

    with np.load(student_hamiltonian_npz, allow_pickle=False) as reference:
        reference_eigenvalues = reference["student_clean_eigenvalues"]
        reference_z2_weights = reference["student_clean_z2_weights"]
        reference_basis = set(reference["constrained_basis_bit_strings"].astype(str))

    student_eigenvalues, student_z2_weights = _sorted_spectrum_with_z2_weights(
        eigenvalues,
        eigenstates,
        psi0,
    )
    quspin_eigenvalues, quspin_z2_weights = _sorted_reference_spectrum(
        reference_eigenvalues,
        reference_z2_weights,
    )

    assert set(basis_list) == reference_basis
    np.testing.assert_allclose(
        student_eigenvalues,
        quspin_eigenvalues,
        atol=1e-12,
        rtol=1e-12,
    )

    comparison = _save_student_quspin_comparison(
        test_subdir,
        "clean",
        student_eigenvalues,
        quspin_eigenvalues,
        student_z2_weights,
        quspin_z2_weights,
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"length": length, "basis_dimension": len(basis_list), **comparison},
    )


def test_disordered_spectrum_matches_independent_quspin(student_hamiltonian_npz, test_subdir):
    length = 10
    clean_hamiltonian, _, _, psi0, basis_list = get_scar_ham(length)

    with np.load(student_hamiltonian_npz, allow_pickle=False) as reference:
        ham_disorder = reference["ham_disorder"].tolist()
        reference_eigenvalues = reference["student_disordered_eigenvalues"]
        reference_z2_weights = reference["student_disordered_z2_weights"]

    _, eigenvalues, eigenstates = get_dis_scar_ham(
        clean_hamiltonian,
        length,
        basis_list,
        ham_disorder=ham_disorder,
        fixed_seed=True,
    )
    student_eigenvalues, student_z2_weights = _sorted_spectrum_with_z2_weights(
        eigenvalues,
        eigenstates,
        psi0,
    )
    quspin_eigenvalues, quspin_z2_weights = _sorted_reference_spectrum(
        reference_eigenvalues,
        reference_z2_weights,
    )

    np.testing.assert_allclose(
        student_eigenvalues,
        quspin_eigenvalues,
        atol=1e-12,
        rtol=1e-12,
    )

    comparison = _save_student_quspin_comparison(
        test_subdir,
        "disordered",
        student_eigenvalues,
        quspin_eigenvalues,
        student_z2_weights,
        quspin_z2_weights,
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"length": length, "ham_disorder": ham_disorder, **comparison},
    )


def test_scar_drive_operator_properties(test_subdir):
    length = 6
    _, _, _, _, basis_list = get_scar_ham(length)
    combined_drive, weights = get_scar_H1(length, basis_list, ds_dis=0.0)
    individual_drives, individual_weights = get_scar_H1(
        length,
        basis_list,
        ds_dis=0.0,
        indv_qubit=True,
    )

    combined_dense = combined_drive.full()
    summed_individual_dense = sum(operator.full() for operator in individual_drives)

    assert np.allclose(weights, np.ones(length))
    assert np.allclose(individual_weights, np.ones(length))
    assert np.allclose(combined_dense, combined_dense.conj().T)
    assert np.allclose(combined_dense, np.diag(np.diag(combined_dense)))
    assert np.allclose(combined_dense, summed_individual_dense)

    save_array_data(test_subdir / "drive_operator.npz", drive=combined_dense)
    save_metadata(test_subdir / "metadata.json", {"length": length})


def test_disorder_zero_and_seeded_reproducibility(test_subdir):
    length = 6
    clean_hamiltonian, _, _, _, basis_list = get_scar_ham(length)
    zero_hamiltonian, zero_eigenvalues, _ = get_dis_scar_ham(
        clean_hamiltonian,
        length,
        basis_list,
        ham_disorder=[0.0, 0.0, 0.0],
        fixed_seed=True,
    )
    disorder_a, eigenvalues_a, _ = get_dis_scar_ham(
        clean_hamiltonian,
        length,
        basis_list,
        ham_disorder=[0.1, 0.1, 0.1],
        fixed_seed=True,
    )
    disorder_b, eigenvalues_b, _ = get_dis_scar_ham(
        clean_hamiltonian,
        length,
        basis_list,
        ham_disorder=[0.1, 0.1, 0.1],
        fixed_seed=True,
    )

    assert np.allclose(zero_hamiltonian.full(), clean_hamiltonian.full())
    assert np.allclose(np.asarray(zero_eigenvalues), np.linalg.eigvalsh(clean_hamiltonian.full()))
    assert np.allclose(disorder_a.full(), disorder_a.full().conj().T)
    assert np.allclose(disorder_a.full(), disorder_b.full())
    assert np.allclose(np.asarray(eigenvalues_a), np.asarray(eigenvalues_b))

    save_array_data(
        test_subdir / "disorder_reproducibility.npz",
        eigenvalues_a=np.asarray(eigenvalues_a),
        eigenvalues_b=np.asarray(eigenvalues_b),
    )
    save_metadata(test_subdir / "metadata.json", {"length": length})


def test_embed_scar_state_to_full_preserves_z2_state(test_subdir):
    length = 4
    _, _, _, psi0, basis_list = get_scar_ham(length)
    psi_full = embed_scar_state_to_full(psi0, basis_list, length)
    full_vector = psi_full.full().ravel()
    full_z2_index = int(z2_initial(length), 2)

    assert psi_full.shape == (2**length, 1)
    assert psi_full.norm() == pytest.approx(1.0)
    assert full_vector[full_z2_index] == pytest.approx(1.0)

    save_array_data(test_subdir / "embedded_state.npz", state=full_vector)
    save_metadata(
        test_subdir / "metadata.json",
        {"length": length, "full_z2_index": full_z2_index},
    )
