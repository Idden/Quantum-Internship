import numpy as np
import pytest

import quantumScarFunctions as m

@pytest.fixture
def patched_basis_helpers(monkeypatch):
    """
    Patch external helpers so the tests are self-contained and deterministic.
    The basis is chosen to already satisfy the no-consecutive-1s rule, and
    includes one state that should be removed by the periodic boundary filter:
    '1001' has first and last bits both 1.
    """
    basis_list = [
        "0000",
        "0001",
        "0010",
        "0100",
        "0101",
        "1000",
        "1001",  # should be removed by the code
        "1010",
    ]

    monkeypatch.setattr(m, "binNoConsecOnesEfficient", lambda N: basis_list.copy())
    monkeypatch.setattr(m, "z2_initial", lambda N: "1010")

    return basis_list


def qobj_dense(q):
    """Convert a qutip Qobj to a dense numpy array."""
    return q.full()


def test_raises_if_N_is_odd():
    with pytest.raises(AssertionError, match="N must be a multiple of 2"):
        m.get_scar_ham(3)


def test_raises_if_ham_disorder_wrong_length():
    with pytest.raises(AssertionError, match="ham_disorder must have 3 values"):
        m.get_scar_ham(4, ham_disorder=[0.0, 0.0])


def test_periodic_boundary_basis_filter_removes_first_last_ones(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.0,
    )

    assert "1001" not in basis_list
    assert len(basis_list) == 7  # 8 initial - 1 removed


def test_shapes_and_return_types_non_indv_qubit(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.0,
    )

    basis_len = len(basis_list)

    assert H0.shape == (basis_len, basis_len)
    assert H1.shape == (basis_len, basis_len)
    assert len(evals) == basis_len
    assert len(estates) == basis_len
    assert psi0.shape == (basis_len, 1)


def test_shapes_and_return_types_indv_qubit(patched_basis_helpers):
    H0, H1_list, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=True,
        ohms=1.0,
        ds_dis=0.0,
    )

    basis_len = len(basis_list)

    assert H0.shape == (basis_len, basis_len)
    assert isinstance(H1_list, list)
    assert len(H1_list) == 4

    for Hr in H1_list:
        assert Hr.shape == (basis_len, basis_len)

    assert len(evals) == basis_len
    assert len(estates) == basis_len
    assert psi0.shape == (basis_len, 1)


def test_psi0_matches_z2_initial_index(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.0,
    )

    expected_index = basis_list.index("1010")
    expected = np.zeros((len(basis_list), 1), dtype=complex)
    expected[expected_index, 0] = 1.0

    np.testing.assert_allclose(qobj_dense(psi0), expected)


def test_H0_is_hermitian_without_disorder(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.0,
    )

    H0_dense = qobj_dense(H0)
    np.testing.assert_allclose(H0_dense, H0_dense.conj().T)


def test_H0_is_hermitian_with_all_disorder_channels(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.2, 0.3, 0.4],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.1,
    )

    H0_dense = qobj_dense(H0)
    np.testing.assert_allclose(H0_dense, H0_dense.conj().T, atol=1e-12)


def test_random_seed_false_is_deterministic(patched_basis_helpers):
    """
    In this implementation, random_seed=False triggers np.random.seed(0),
    so two calls should match exactly if disorder or ds_dis are nonzero.
    """
    out1 = m.get_scar_ham(
        4,
        ham_disorder=[0.2, 0.3, 0.4],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.2,
    )
    out2 = m.get_scar_ham(
        4,
        ham_disorder=[0.2, 0.3, 0.4],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.2,
    )

    H0_1, H1_1 = out1[0], out1[1]
    H0_2, H1_2 = out2[0], out2[1]

    np.testing.assert_allclose(qobj_dense(H0_1), qobj_dense(H0_2))
    np.testing.assert_allclose(qobj_dense(H1_1), qobj_dense(H1_2))


def test_random_seed_true_is_not_forced_deterministic(patched_basis_helpers):
    """
    This is a probabilistic test, but with nonzero disorder it should almost
    always differ because the code does NOT reseed when random_seed=True.
    """
    out1 = m.get_scar_ham(
        4,
        ham_disorder=[0.2, 0.3, 0.4],
        random_seed=True,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.2,
    )
    out2 = m.get_scar_ham(
        4,
        ham_disorder=[0.2, 0.3, 0.4],
        random_seed=True,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.2,
    )

    H0_1, H1_1 = out1[0], out1[1]
    H0_2, H1_2 = out2[0], out2[1]

    # Very unlikely these match exactly if randomness is really not reseeded
    assert not np.allclose(qobj_dense(H0_1), qobj_dense(H0_2)) or \
           not np.allclose(qobj_dense(H1_1), qobj_dense(H1_2))


def test_H1_is_diagonal_when_not_indv_qubit(patched_basis_helpers):
    H0, H1, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=False,
        ohms=1.0,
        ds_dis=0.2,
    )

    H1_dense = qobj_dense(H1)
    offdiag = H1_dense - np.diag(np.diag(H1_dense))
    np.testing.assert_allclose(offdiag, 0.0)


def test_each_indv_qubit_H1_is_diagonal(patched_basis_helpers):
    H0, H1_list, evals, estates, psi0, basis_list = m.get_scar_ham(
        4,
        ham_disorder=[0.0, 0.0, 0.0],
        random_seed=False,
        indv_qubit=True,
        ohms=1.0,
        ds_dis=0.2,
    )

    for Hr in H1_list:
        Hr_dense = qobj_dense(Hr)
        offdiag = Hr_dense - np.diag(np.diag(Hr_dense))
        np.testing.assert_allclose(offdiag, 0.0)
