"""
Tests for the drive operators: ``get_scar_H1`` and ``get_Hy``.

``get_scar_H1`` is the operator the sinusoidal drive multiplies, so it sets
the charging protocol.  ``get_Hy`` is the staggered-Y operator that
``get_zero_scar`` uses as one of the three su(2) generators.
"""

import numpy as np
import pytest

import reference_pxp as ref
from data_utils import save_array_data, save_metadata

SIZES = [4, 6, 8, 10]


# --------------------------------------------------------------------------
# get_scar_H1 -- structure
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_drive_is_diagonal_and_hermitian(scar_functions, basis_list, N, test_subdir):
    basisList = basis_list(N)
    H1, _ = scar_functions.get_scar_H1(N, basisList, ds_dis=0.0)
    dense = H1.full()

    assert dense.shape == (len(basisList), len(basisList))
    assert np.allclose(dense, dense.conj().T, atol=1e-14)
    assert np.allclose(dense, np.diag(np.diag(dense)), atol=1e-14)
    assert np.abs(dense.imag).max() < 1e-14

    save_array_data(test_subdir / "drive_operator.npz", drive=dense.real)
    save_metadata(test_subdir / "metadata.json", {"N": N, "dimension": len(basisList)})


@pytest.mark.parametrize("N", SIZES)
def test_drive_matches_independent_reference(scar_functions, basis_list, N):
    """
    With no disorder the drive is the staggered magnetisation
    ``sum_r z2_r * Z_r``, restricted to the constrained basis.  The reference
    builds that from actual Pauli matrices in the full space.
    """
    basisList = basis_list(N)
    H1, _ = scar_functions.get_scar_H1(N, basisList, ds_dis=0.0)

    expected = ref.reference_drive(N, basisList)
    assert np.abs(H1.full().real - expected).max() < 1e-12


@pytest.mark.parametrize("N", SIZES)
def test_drive_is_maximal_on_the_z2_state(scar_functions, basis_list, N):
    """
    The drive is built by projecting onto the Neel pattern, so the Neel state
    itself must sit at the top of the diagonal with eigenvalue exactly N.
    """
    basisList = basis_list(N)
    H1, _ = scar_functions.get_scar_H1(N, basisList, ds_dis=0.0)

    diagonal = np.diag(H1.full()).real
    z2_index = basisList.index(scar_functions.z2_initial(N))

    assert diagonal[z2_index] == pytest.approx(float(N))
    assert diagonal.max() == pytest.approx(float(N))


@pytest.mark.parametrize("N", SIZES)
def test_drive_weights_default_to_ones(scar_functions, basis_list, N):
    _, weights = scar_functions.get_scar_H1(N, basis_list(N), ds_dis=0.0)
    assert np.allclose(weights, np.ones(N))


# --------------------------------------------------------------------------
# get_scar_H1 -- individual-qubit mode
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_individual_drives_sum_to_the_combined_drive(scar_functions, basis_list, N):
    """
    ``indv_qubit=True`` splits the drive into one operator per site so each
    can carry its own frequency.  At equal frequencies the split must be exact,
    otherwise the frequency-disorder runs are not comparable to the clean ones.
    """
    basisList = basis_list(N)

    combined, combined_weights = scar_functions.get_scar_H1(N, basisList, ds_dis=0.0)
    individual, individual_weights = scar_functions.get_scar_H1(
        N, basisList, ds_dis=0.0, indv_qubit=True
    )

    assert len(individual) == N
    assert np.allclose(combined_weights, individual_weights)

    summed = sum(op.full() for op in individual)
    assert np.abs(combined.full() - summed).max() < 1e-12


@pytest.mark.parametrize("N", [4, 6, 8])
def test_individual_drives_are_diagonal(scar_functions, basis_list, N):
    basisList = basis_list(N)
    individual, _ = scar_functions.get_scar_H1(
        N, basisList, ds_dis=0.0, indv_qubit=True
    )

    for op in individual:
        dense = op.full()
        assert np.allclose(dense, np.diag(np.diag(dense)), atol=1e-14)


@pytest.mark.parametrize("N", [4, 6, 8])
def test_individual_drive_entries_are_plus_or_minus_one(scar_functions, basis_list, N):
    """Each site contributes ``+-1`` per basis state and nothing else."""
    basisList = basis_list(N)
    individual, _ = scar_functions.get_scar_H1(
        N, basisList, ds_dis=0.0, indv_qubit=True
    )

    for op in individual:
        diagonal = np.diag(op.full()).real
        assert np.allclose(np.abs(diagonal), 1.0)


# --------------------------------------------------------------------------
# get_scar_H1 -- drive-strength disorder
# --------------------------------------------------------------------------
def test_drive_disorder_respects_its_bound(scar_functions, basis_list):
    """Weights are ``1 + U(-ds, ds)``, so they stay inside [1-ds, 1+ds]."""
    N, ds = 8, 0.3
    _, weights = scar_functions.get_scar_H1(
        N, basis_list(N), ds_dis=ds, fixed_seed=True
    )

    assert weights.shape == (N,)
    assert np.all(weights >= 1.0 - ds - 1e-12)
    assert np.all(weights <= 1.0 + ds + 1e-12)
    assert not np.allclose(weights, np.ones(N))


def test_drive_disorder_is_reproducible_with_fixed_seed(scar_functions, basis_list):
    N = 8
    basisList = basis_list(N)

    a, wa = scar_functions.get_scar_H1(N, basisList, ds_dis=0.3, fixed_seed=True)
    b, wb = scar_functions.get_scar_H1(N, basisList, ds_dis=0.3, fixed_seed=True)

    assert np.allclose(wa, wb)
    assert np.abs(a.full() - b.full()).max() < 1e-14


def test_drive_disorder_N_dis_limits_affected_sites(scar_functions, basis_list):
    """With ``N_dis=1`` exactly one weight may differ from 1."""
    N = 8
    _, weights = scar_functions.get_scar_H1(
        N, basis_list(N), ds_dis=0.3, N_dis=1, fixed_seed=True
    )
    assert np.count_nonzero(np.abs(weights - 1.0) > 1e-12) == 1


def test_drive_disorder_still_sums_correctly(scar_functions, basis_list):
    """The combined/individual identity must survive weight disorder too."""
    N = 8
    basisList = basis_list(N)

    combined, _ = scar_functions.get_scar_H1(
        N, basisList, ds_dis=0.3, fixed_seed=True
    )
    individual, _ = scar_functions.get_scar_H1(
        N, basisList, ds_dis=0.3, fixed_seed=True, indv_qubit=True
    )

    summed = sum(op.full() for op in individual)
    assert np.abs(combined.full() - summed).max() < 1e-12


# --------------------------------------------------------------------------
# get_Hy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_Hy_is_hermitian(scar_functions, basis_list, N):
    """
    ``get_Hy`` writes the +/-i phases by hand rather than using a Pauli
    matrix, which is exactly the kind of code where a conjugation can go
    missing.  A non-Hermitian Hy would make ``get_zero_scar``'s S^2 wrong.
    """
    Hy = scar_functions.get_Hy(N, basis_list(N))
    dense = Hy.full()
    assert np.allclose(dense, dense.conj().T, atol=1e-12)


@pytest.mark.parametrize("N", SIZES)
def test_Hy_matches_independent_reference(scar_functions, basis_list, N):
    """Compare against the staggered sum of true Pauli-Y matrices."""
    basisList = basis_list(N)
    Hy = scar_functions.get_Hy(N, basisList)

    expected = ref.reference_Hy(N, basisList)
    assert np.abs(Hy.full() - expected).max() < 1e-12


@pytest.mark.parametrize("N", SIZES)
def test_Hy_is_purely_off_diagonal(scar_functions, basis_list, N):
    Hy = scar_functions.get_Hy(N, basis_list(N))
    assert np.abs(np.diag(Hy.full())).max() < 1e-14


@pytest.mark.parametrize("N", [4, 6, 8])
def test_Hy_spectrum_is_symmetric(scar_functions, basis_list, N):
    """Hy is off-diagonal in the constrained basis, so its spectrum is
    symmetric about zero -- a cheap independent check on the phases."""
    Hy = scar_functions.get_Hy(N, basis_list(N))
    energies = np.sort(np.linalg.eigvalsh(Hy.full()))
    assert np.abs(energies + energies[::-1]).max() < 1e-10
