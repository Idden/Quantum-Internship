"""
Tests for ``get_qubit_ham`` -- the independent-qubit reference battery.

This is the baseline the scar chain is compared against in every figure, so
its normalisation matters as much as the scar side.  In particular the total
bandwidth of the N uncoupled qubits sets the denominator of ``Rtau_qubit``.
"""

import numpy as np
import pytest
import qutip as qt

N_TEST = 6
STRENGTH = 0.3


# --------------------------------------------------------------------------
# Shape and structure
# --------------------------------------------------------------------------
def test_returns_one_operator_pair_per_site(scar_functions):
    qH0, qH1, weights = scar_functions.get_qubit_ham(N_TEST)

    assert len(qH0) == N_TEST
    assert len(qH1) == N_TEST
    assert weights.shape == (N_TEST,)

    for op in list(qH0) + list(qH1):
        assert op.shape == (2, 2)


def test_operators_are_hermitian(scar_functions):
    qH0, qH1, _ = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=[STRENGTH, STRENGTH, STRENGTH], fixed_seed=True
    )

    for op in list(qH0) + list(qH1):
        dense = op.full()
        assert np.allclose(dense, dense.conj().T, atol=1e-14)


# --------------------------------------------------------------------------
# The two Hamiltonian conventions
# --------------------------------------------------------------------------
def test_default_convention_is_x_static_z_drive(scar_functions):
    """Default (``sigz_ham=False``): H0 = -wm/2 * sigma_x, H1 = w_r * sigma_z."""
    wm = 1.3
    qH0, qH1, weights = scar_functions.get_qubit_ham(N_TEST, wm=wm)

    expected_H0 = (-0.5 * wm * qt.sigmax()).full()
    for i in range(N_TEST):
        assert np.abs(qH0[i].full() - expected_H0).max() < 1e-14
        assert np.abs(qH1[i].full() - weights[i] * qt.sigmaz().full()).max() < 1e-14


def test_sigz_convention_swaps_the_roles(scar_functions):
    """``sigz_ham=True``: H0 = -wm/2 * sigma_z, H1 = w_r * sigma_x."""
    wm = 1.3
    qH0, qH1, weights = scar_functions.get_qubit_ham(N_TEST, wm=wm, sigz_ham=True)

    expected_H0 = (-0.5 * wm * qt.sigmaz()).full()
    for i in range(N_TEST):
        assert np.abs(qH0[i].full() - expected_H0).max() < 1e-14
        assert np.abs(qH1[i].full() - weights[i] * qt.sigmax().full()).max() < 1e-14


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wm", [0.5, 1.0, 2.0])
def test_clean_qubit_bandwidth_equals_wm(scar_functions, wm):
    """
    A clean qubit has eigenvalues -+wm/2, so its bandwidth is exactly wm and
    the total bandwidth of the reference battery is N*wm.  ``Rtau_qubit``
    divides by that sum, so an error here rescales every comparison figure.
    """
    qH0, _, _ = scar_functions.get_qubit_ham(N_TEST, wm=wm)

    total = 0.0
    for op in qH0:
        energies = np.linalg.eigvalsh(op.full())
        assert energies[-1] - energies[0] == pytest.approx(wm)
        total += energies[-1] - energies[0]

    assert total == pytest.approx(N_TEST * wm)


def test_drive_weights_default_to_ones(scar_functions):
    _, _, weights = scar_functions.get_qubit_ham(N_TEST, ds_dis=0.0)
    assert np.allclose(weights, np.ones(N_TEST))


def test_drive_weight_disorder_respects_its_bound(scar_functions):
    ds = 0.3
    _, qH1, weights = scar_functions.get_qubit_ham(
        N_TEST, ds_dis=ds, fixed_seed=True
    )

    assert np.all(weights >= 1.0 - ds - 1e-12)
    assert np.all(weights <= 1.0 + ds + 1e-12)
    assert not np.allclose(weights, np.ones(N_TEST))

    # the weights must actually reach the operators, not just be returned
    for i in range(N_TEST):
        assert np.abs(qH1[i].full()).max() == pytest.approx(abs(weights[i]))


# --------------------------------------------------------------------------
# Disorder
# --------------------------------------------------------------------------
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_each_disorder_axis_changes_the_static_hamiltonian(scar_functions, axis):
    disorder = [0.0, 0.0, 0.0]
    disorder[axis] = STRENGTH

    clean, _, _ = scar_functions.get_qubit_ham(N_TEST)
    noisy, _, _ = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=disorder, fixed_seed=True
    )

    difference = max(
        np.abs(noisy[i].full() - clean[i].full()).max() for i in range(N_TEST)
    )
    assert difference > 1e-9


def test_disorder_only_touches_the_static_hamiltonian(scar_functions):
    """``ham_disorder`` must not leak into the drive operators."""
    _, clean_drive, _ = scar_functions.get_qubit_ham(N_TEST)
    _, noisy_drive, _ = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=[STRENGTH, STRENGTH, STRENGTH], fixed_seed=True
    )

    for i in range(N_TEST):
        assert np.abs(noisy_drive[i].full() - clean_drive[i].full()).max() < 1e-14


def test_fixed_seed_is_reproducible(scar_functions):
    a, _, wa = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=[STRENGTH, STRENGTH, STRENGTH], ds_dis=0.2,
        fixed_seed=True,
    )
    b, _, wb = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=[STRENGTH, STRENGTH, STRENGTH], ds_dis=0.2,
        fixed_seed=True,
    )

    assert np.allclose(wa, wb)
    for i in range(N_TEST):
        assert np.abs(a[i].full() - b[i].full()).max() < 1e-14


def test_external_seed_controls_the_realization(scar_functions):
    """
    The sweeps reseed globally and then call this with ``fixed_seed=False`` so
    the qubit reference sees the same disorder realization as the scar chain.
    """
    def realize(seed):
        np.random.seed(seed)
        ops, _, _ = scar_functions.get_qubit_ham(
            N_TEST, ham_disorder=[STRENGTH, 0.0, 0.0]
        )
        return np.array([op.full() for op in ops])

    assert np.abs(realize(3) - realize(3)).max() < 1e-14
    assert np.abs(realize(3) - realize(4)).max() > 1e-9


def test_N_dis_limits_the_number_of_disordered_qubits(scar_functions):
    clean, _, _ = scar_functions.get_qubit_ham(N_TEST)
    noisy, _, _ = scar_functions.get_qubit_ham(
        N_TEST, ham_disorder=[STRENGTH, 0.0, 0.0], N_dis=1, fixed_seed=True
    )

    changed = [
        i for i in range(N_TEST)
        if np.abs(noisy[i].full() - clean[i].full()).max() > 1e-12
    ]
    assert len(changed) == 1
