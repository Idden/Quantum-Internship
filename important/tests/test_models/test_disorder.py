"""
Tests for ``get_dis_scar_ham`` -- the disordered scar Hamiltonian.

Disorder is the independent variable in the whole study, so the properties
that matter most here are: zero disorder changes nothing, the result stays
Hermitian, a fixed seed is reproducible, and the disorder strength actually
respects the bound that was asked for.
"""

import numpy as np
import pytest

from data_utils import save_array_data, save_metadata

N_TEST = 8
STRENGTH = 0.3


@pytest.fixture(scope="module")
def clean():
    """The clean N=8 Hamiltonian and basis, built once for this module."""
    from GitHub_QM.important.hpc.quantumScarFunctions import get_scar_ham

    H0, _, _, psi0, basisList = get_scar_ham(N_TEST, diagonalize=False)
    return H0, psi0, basisList


# --------------------------------------------------------------------------
# The no-op case
# --------------------------------------------------------------------------
def test_zero_disorder_returns_the_clean_hamiltonian(scar_functions, clean):
    """
    ``ham_disorder=[0, 0, 0]`` must be a genuine no-op.  Every clean baseline
    in the paper is produced this way, so a stray additive term here would
    contaminate the reference curve rather than the disordered one.
    """
    H0, _, basisList = clean

    H_dis, eigenvalues, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[0.0, 0.0, 0.0]
    )

    assert np.abs(H_dis.full() - H0.full()).max() < 1e-14
    np.testing.assert_allclose(
        np.sort(np.asarray(eigenvalues, dtype=float)),
        np.sort(np.linalg.eigvalsh(H0.full())),
        atol=1e-10,
    )


# --------------------------------------------------------------------------
# Structure, per disorder axis
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "disorder"),
    [
        ("z", [STRENGTH, 0.0, 0.0]),
        ("y", [0.0, STRENGTH, 0.0]),
        ("x", [0.0, 0.0, STRENGTH]),
        ("xyz", [STRENGTH, STRENGTH, STRENGTH]),
    ],
)
def test_disordered_hamiltonian_is_hermitian(scar_functions, clean, label, disorder):
    """
    Each axis is assembled by hand from explicit row/column/data triplets, so
    Hermiticity is a real risk -- particularly the y term, whose +/-i phases
    have to come out conjugate under transposition.
    """
    H0, _, basisList = clean

    H_dis, _, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=disorder, fixed_seed=True
    )
    dense = H_dis.full()

    assert np.allclose(dense, dense.conj().T, atol=1e-12), f"{label} term not Hermitian"


@pytest.mark.parametrize(
    ("label", "disorder"),
    [("z", [STRENGTH, 0, 0]), ("y", [0, STRENGTH, 0]), ("x", [0, 0, STRENGTH])],
)
def test_each_axis_actually_changes_the_hamiltonian(
    scar_functions, clean, label, disorder
):
    """A disorder axis that silently did nothing would be invisible in the
    results but would flatten the curve for that axis."""
    H0, _, basisList = clean

    H_dis, _, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=disorder, fixed_seed=True
    )

    assert np.abs(H_dis.full() - H0.full()).max() > 1e-9


def test_z_disorder_is_diagonal(scar_functions, clean):
    """
    sigma-z disorder is diagonal in the product basis, so the difference from
    the clean Hamiltonian must have no off-diagonal part.
    """
    H0, _, basisList = clean

    H_dis, _, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0.0, 0.0], fixed_seed=True
    )
    difference = H_dis.full() - H0.full()
    off_diagonal = difference - np.diag(np.diag(difference))

    assert np.abs(off_diagonal).max() < 1e-14


def test_x_and_y_disorder_are_off_diagonal(scar_functions, clean):
    """sigma-x and sigma-y flip a site, so they contribute no diagonal part."""
    H0, _, basisList = clean

    for disorder in ([0.0, STRENGTH, 0.0], [0.0, 0.0, STRENGTH]):
        H_dis, _, _ = scar_functions.get_dis_scar_ham(
            H0, N_TEST, basisList, ham_disorder=disorder, fixed_seed=True
        )
        difference = H_dis.full() - H0.full()
        assert np.abs(np.diag(difference)).max() < 1e-14


def test_z_disorder_respects_its_strength_bound(scar_functions, clean):
    """
    Site fields are drawn from U(-zd, zd), so the diagonal shift on any basis
    state cannot exceed N*zd.  This catches a strength that is applied twice
    or on the wrong scale.
    """
    H0, _, basisList = clean
    zd = STRENGTH

    H_dis, _, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[zd, 0.0, 0.0], fixed_seed=True
    )
    shift = np.diag(H_dis.full() - H0.full()).real

    assert np.abs(shift).max() <= N_TEST * zd + 1e-12


# --------------------------------------------------------------------------
# Randomness and reproducibility
# --------------------------------------------------------------------------
def test_fixed_seed_is_reproducible(scar_functions, clean, test_subdir):
    """Two ``fixed_seed=True`` calls must produce identical realizations."""
    H0, _, basisList = clean

    first, eig_a, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[0.1, 0.1, 0.1], fixed_seed=True
    )
    second, eig_b, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[0.1, 0.1, 0.1], fixed_seed=True
    )

    assert np.abs(first.full() - second.full()).max() < 1e-14
    np.testing.assert_allclose(
        np.asarray(eig_a, dtype=float), np.asarray(eig_b, dtype=float), atol=1e-12
    )

    save_array_data(
        test_subdir / "disorder_reproducibility.npz",
        eigenvalues_a=np.asarray(eig_a, dtype=float),
        eigenvalues_b=np.asarray(eig_b, dtype=float),
    )
    save_metadata(test_subdir / "metadata.json", {"N": N_TEST, "disorder": [0.1, 0.1, 0.1]})


def test_external_seed_controls_the_realization(scar_functions, clean):
    """
    The parallel drivers call ``np.random.seed(seed)`` and then
    ``get_dis_scar_ham(..., fixed_seed=False)``.  That is the mechanism by
    which realizations differ, so it has to work: same seed identical,
    different seed different.
    """
    H0, _, basisList = clean
    disorder = [STRENGTH, 0.0, 0.0]

    def realize(seed):
        np.random.seed(seed)
        return scar_functions.get_dis_scar_ham(
            H0, N_TEST, basisList, ham_disorder=disorder
        )[0].full()

    assert np.abs(realize(7) - realize(7)).max() < 1e-14
    assert np.abs(realize(7) - realize(8)).max() > 1e-9


def test_fixed_seed_overrides_the_external_seed(scar_functions, clean):
    """``fixed_seed=True`` reseeds to 0, so it must ignore the outer seed.

    This is worth pinning down because passing ``fixed_seed=True`` inside a
    parallel sweep would collapse every realization onto the same disorder.
    """
    H0, _, basisList = clean

    np.random.seed(1)
    a = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0, 0], fixed_seed=True
    )[0].full()
    np.random.seed(999)
    b = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0, 0], fixed_seed=True
    )[0].full()

    assert np.abs(a - b).max() < 1e-14


def test_N_dis_limits_the_number_of_disordered_sites(scar_functions, clean):
    """
    With ``N_dis=1`` exactly one site carries a field, so the diagonal shifts
    may take only two distinct magnitudes (+h and -h for that site).
    """
    H0, _, basisList = clean

    H_dis, _, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, N_dis=1, ham_disorder=[STRENGTH, 0.0, 0.0],
        fixed_seed=True,
    )
    shift = np.diag(H_dis.full() - H0.full()).real

    assert len(np.unique(np.round(np.abs(shift), 12))) == 1


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------
def test_diagonalize_false_skips_the_spectrum(scar_functions, clean):
    H0, _, basisList = clean

    H_dis, eigenvalues, eigenstates = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0, 0],
        fixed_seed=True, diagonalize=False,
    )

    assert eigenvalues is None
    assert eigenstates is None
    assert H_dis.shape == (len(basisList), len(basisList))


def test_diagonalize_false_gives_the_same_hamiltonian(scar_functions, clean):
    H0, _, basisList = clean

    eager = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0, 0], fixed_seed=True
    )[0]
    lazy = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, 0, 0],
        fixed_seed=True, diagonalize=False,
    )[0]

    assert np.abs(eager.full() - lazy.full()).max() < 1e-14


def test_eigenvalues_are_ascending_and_real(scar_functions, clean):
    H0, _, basisList = clean

    _, eigenvalues, _ = scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, STRENGTH, STRENGTH],
        fixed_seed=True,
    )
    energies = np.asarray(eigenvalues)

    assert np.abs(np.imag(energies)).max() < 1e-12
    assert np.all(np.diff(np.real(energies)) >= -1e-12)


def test_clean_hamiltonian_is_not_mutated(scar_functions, clean):
    """
    ``get_dis_scar_ham`` rebinds its ``H0_dis`` argument.  The caller's clean
    Hamiltonian is reused for every realization in the sweeps, so it must come
    back untouched.
    """
    H0, _, basisList = clean
    before = H0.full().copy()

    scar_functions.get_dis_scar_ham(
        H0, N_TEST, basisList, ham_disorder=[STRENGTH, STRENGTH, STRENGTH],
        fixed_seed=True,
    )

    assert np.abs(H0.full() - before).max() < 1e-14
