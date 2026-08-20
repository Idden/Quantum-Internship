"""
Tests for the basis-construction helpers in ``quantumScarFunctions``.

Covers ``binToDeci``, ``binNoConsecOnesEfficient``, ``z2_initial``, the
periodic Rydberg-blockade basis built inside ``get_scar_ham``, and
``get_C_AB_matrix``.
"""

import numpy as np
import pytest

import reference_pxp as ref
from data_utils import save_metadata

# N=4 is the smallest even chain; N=12 keeps the whole file under a second.
BASIS_SIZES = [4, 6, 8, 10, 12]


# --------------------------------------------------------------------------
# binToDeci
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("bitstring", "expected"),
    [("0", 0), ("1", 1), ("1010", 10), ("1111", 15), ("1000000000", 512)],
)
def test_bin_to_deci(scar_functions, bitstring, expected):
    assert scar_functions.binToDeci(bitstring) == expected


def test_bin_to_deci_matches_int_builtin(scar_functions, rng):
    """``binToDeci`` must agree with ``int(s, 2)`` on arbitrary strings.

    This pins down the bit-ordering convention (leftmost character is the
    most significant bit) that every other test in the suite relies on.
    """
    for _ in range(200):
        length = int(rng.integers(1, 16))
        bits = "".join(rng.choice(["0", "1"], size=length))
        assert scar_functions.binToDeci(bits) == int(bits, 2)


# --------------------------------------------------------------------------
# binNoConsecOnesEfficient  (open chain, before the blockade filter)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 8, 10])
def test_no_consecutive_ones_count_is_fibonacci(scar_functions, N):
    """The OPEN chain with no two adjacent 1s has F_{N+2} configurations."""

    def fibonacci(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    assert len(scar_functions.binNoConsecOnesEfficient(N)) == fibonacci(N + 2)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 8, 10])
def test_no_consecutive_ones_are_valid_and_unique(scar_functions, N):
    strings = scar_functions.binNoConsecOnesEfficient(N)

    assert len(set(strings)) == len(strings), "duplicate configurations"
    for s in strings:
        assert len(s) == N
        assert set(s) <= {"0", "1"}
        assert "11" not in s


# --------------------------------------------------------------------------
# z2_initial
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("N", "expected"),
    [(2, "10"), (4, "1010"), (6, "101010"), (8, "10101010")],
)
def test_z2_initial(scar_functions, N, expected):
    assert scar_functions.z2_initial(N) == expected


@pytest.mark.parametrize("N", BASIS_SIZES)
def test_z2_state_is_blockade_allowed(scar_functions, N):
    """The Neel state must survive the periodic blockade, or ``get_scar_ham``
    could not use it as the initial state."""
    z2 = scar_functions.z2_initial(N)
    assert "11" not in z2
    assert not (z2[0] == "1" and z2[-1] == "1")


# --------------------------------------------------------------------------
# The periodic blockade basis built inside get_scar_ham
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", BASIS_SIZES)
def test_blockade_basis_matches_independent_enumeration(basis_list, N, test_subdir):
    """
    The basis must equal a brute-force enumeration of all 2**N strings with no
    two cyclically-adjacent 1s.

    ``reference_pxp.blockade_bitstrings`` shares no code with the recursive
    generator + filter used by ``get_scar_ham``, so this is a real check and
    not a restatement of the implementation.
    """
    produced = basis_list(N)
    expected = ref.blockade_bitstrings(N)

    assert len(produced) == len(set(produced)), "duplicate basis states"
    assert set(produced) == set(expected)

    save_metadata(
        test_subdir / "metadata.json",
        {"N": N, "dimension": len(produced), "reference": "brute-force 2**N enumeration"},
    )


@pytest.mark.parametrize("N", BASIS_SIZES)
def test_blockade_basis_dimension_is_lucas(basis_list, N):
    """
    The periodic constrained dimension is the Lucas number L_N (7, 18, 47,
    123, 322 ...), not the Fibonacci number that applies to an open chain.

    L_N = F_{N+2} - F_N is exactly the count that remains after
    ``get_scar_ham`` discards the strings starting and ending in '1'.
    """
    assert len(basis_list(N)) == ref.lucas(N)


@pytest.mark.parametrize("N", BASIS_SIZES)
def test_blockade_basis_contains_vacuum(basis_list, N):
    """The all-zero configuration is always allowed and must be present."""
    assert "0" * N in basis_list(N)


@pytest.mark.parametrize("N", BASIS_SIZES)
def test_blockade_basis_excludes_wraparound_pairs(basis_list, N):
    """No basis state may have 1s on both ends -- that is the periodic bond."""
    for s in basis_list(N):
        assert "11" not in s
        assert not (s[0] == "1" and s[-1] == "1")


def test_blockade_basis_ordering_is_stable(scar_functions):
    """Two calls must return the basis in the same order.

    Every saved ``.npz`` in ``xyz_data`` is indexed by basis position, so an
    unstable ordering would silently invalidate stored scar states.
    """
    first = scar_functions.get_scar_ham(8, diagonalize=False)[4]
    second = scar_functions.get_scar_ham(8, diagonalize=False)[4]
    assert first == second


# --------------------------------------------------------------------------
# get_C_AB_matrix  (bipartition used for the entanglement entropy)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", [4, 6, 8])
def test_C_AB_matrix_shape_and_norm(scar_functions, scar_system, N):
    """
    Reshaping a state into the A/B coefficient matrix must preserve its norm,
    since the singular values feed straight into the von Neumann entropy.
    """
    _, _, eigenstates, psi0, basisList = scar_system(N)

    for state in (psi0, eigenstates[0], eigenstates[-1]):
        C_AB = scar_functions.get_C_AB_matrix(state, basisList, N)

        assert C_AB.shape == (2 ** (N // 2), 2 ** (N - N // 2))
        assert np.linalg.norm(C_AB) == pytest.approx(state.norm(), rel=1e-12)


@pytest.mark.parametrize("N", [4, 6, 8])
def test_C_AB_entropy_of_product_state_is_zero(scar_functions, scar_system, N):
    """
    The Neel state is a product state across any cut, so its entanglement
    entropy must be zero (one nonzero singular value).
    """
    _, _, _, psi0, basisList = scar_system(N)
    C_AB = scar_functions.get_C_AB_matrix(psi0, basisList, N)

    sigma = np.linalg.svd(C_AB, compute_uv=False)
    lambdas = sigma ** 2
    lambdas = lambdas[lambdas > 1e-15]
    entropy = -np.sum(lambdas * np.log(lambdas))

    assert entropy == pytest.approx(0.0, abs=1e-12)
