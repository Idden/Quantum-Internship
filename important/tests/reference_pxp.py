"""
Independent reference construction of the periodic PXP model.

Nothing in this file imports ``quantumScarFunctions``.  Everything here is
built from scratch in the FULL 2**N Hilbert space with plain numpy Kronecker
products, then restricted to the Rydberg-blockade subspace.  That makes it a
genuine cross-check: if this file and ``quantumScarFunctions`` agree, two
independent constructions of the same physics agree.

This replaces the old ``IndependentData/minimal_pxp_*_N10.npz`` QuSpin
reference files, which are not in the repository.

Conventions, matched to ``quantumScarFunctions``:

* A configuration is a bit string of length N.  Character ``'1'`` is a
  Rydberg-excited site, ``'0'`` is the ground state.
* Site 0 is the LEFTMOST character, so the full-space index of a bit string
  is ``int(bitstring, 2)`` (site 0 is the most significant bit).
* ``sigma^z`` acts as ``+1`` on ``'1'`` and ``-1`` on ``'0'``, i.e.
  ``diag(-1, +1)`` in the ordering ``(|0>, |1>)``.  This is the sign
  convention used by the ``sigzMap`` dict in ``get_scar_ham``.
* The chain is periodic: site indices are taken mod N.
"""

from itertools import product

import numpy as np

# Single-site operators in the ordering (|0>, |1>).
_ID = np.eye(2)
_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_P = np.array([[1.0, 0.0], [0.0, 0.0]])          # projector onto |0>
_Z = np.array([[-1.0, 0.0], [0.0, 1.0]])         # +1 on |1>, -1 on |0>


def _site_op(op: np.ndarray, site: int, N: int) -> np.ndarray:
    """Embed a single-site operator at ``site`` into the full 2**N space."""
    out = np.array([[1.0]])
    for r in range(N):
        out = np.kron(out, op if r == site % N else _ID)
    return out


def blockade_bitstrings(N: int) -> list[str]:
    """
    Every length-N bit string with no two adjacent '1's, PERIODICALLY.

    Built by brute-force enumeration of all 2**N strings, so it shares no code
    path with the recursive ``binNoConsecOnesEfficient`` + filter that
    ``get_scar_ham`` uses.  Returned in ascending integer order.
    """
    allowed = []
    for bits in product("01", repeat=N):
        s = "".join(bits)
        if any(s[r] == "1" and s[(r + 1) % N] == "1" for r in range(N)):
            continue
        allowed.append(s)
    return allowed


def lucas(n: int) -> int:
    """L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}.

    The number of periodic blockade-allowed configurations on N sites is
    exactly L_N, which is the closed form the basis dimension must match.
    """
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def full_space_hamiltonian(N: int, ohms: float = 1.0, pert: float = -0.026) -> np.ndarray:
    """
    The scar Hamiltonian in the full 2**N space.

        H = (ohms/2) * sum_r P_{r-1} X_r P_{r+1}
            + pert * ohms * sum_r P_{r-1} X_r P_{r+1} (Z_{r-2} + Z_{r+2})

    Both sums run over all N sites with periodic boundaries.  The second term
    is the ``-0.026`` sigma-z perturbation built in the second loop of
    ``get_scar_ham``.
    """
    dim = 2 ** N
    bare = np.zeros((dim, dim))
    pert_term = np.zeros((dim, dim))

    for r in range(N):
        pxp = (_site_op(_P, r - 1, N)
               @ _site_op(_X, r, N)
               @ _site_op(_P, r + 1, N))
        bare += pxp
        # Z_{r-2} and Z_{r+2} commute with the PXP factor (different sites for
        # N >= 6; for N = 4 they are the same site and simply add, which is
        # what quantumScarFunctions does too).
        z_sum = _site_op(_Z, r - 2, N) + _site_op(_Z, r + 2, N)
        pert_term += pxp @ z_sum

    return (ohms / 2.0) * bare + (pert * ohms) * pert_term


def restrict(matrix: np.ndarray, basis_list: list[str]) -> np.ndarray:
    """
    Restrict a full-space operator to the subspace spanned by ``basis_list``,
    in exactly the order the strings appear in ``basis_list``.

    ``basis_list`` is only used as an ordering/labelling of the subspace; the
    matrix elements themselves come from the independent full-space
    construction above.
    """
    idx = np.array([int(s, 2) for s in basis_list])
    return matrix[np.ix_(idx, idx)]


def reference_hamiltonian(N: int, basis_list: list[str],
                          ohms: float = 1.0) -> np.ndarray:
    """The reference scar Hamiltonian, in the ordering of ``basis_list``."""
    return restrict(full_space_hamiltonian(N, ohms=ohms), basis_list)


def reference_drive(N: int, basis_list: list[str]) -> np.ndarray:
    """
    The reference drive operator, matching ``get_scar_H1`` with no disorder.

        H1 = sum_r Z_r * z2_r          (diagonal)

    where ``z2_r`` is +1/-1 for the Neel state '1010...'.  Built as a sum of
    full-space single-site Z operators, then restricted.
    """
    z2 = [1.0 if r % 2 == 0 else -1.0 for r in range(N)]
    dim = 2 ** N
    out = np.zeros((dim, dim))
    for r in range(N):
        out += z2[r] * _site_op(_Z, r, N)
    return restrict(out, basis_list)


def reference_Hy(N: int, basis_list: list[str]) -> np.ndarray:
    """
    The reference staggered-Y operator, matching ``get_Hy``.

        Hy = sum_r (-1)**r * Y_r     restricted to the blockade subspace

    ``get_Hy`` builds this by hand with an explicit +/-i phase per flipped
    site; here it comes from the actual Pauli-Y matrix.
    """
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    dim = 2 ** N
    out = np.zeros((dim, dim), dtype=complex)
    for r in range(N):
        out += ((-1.0) ** r) * _site_op_complex(Y, r, N)
    return restrict(out, basis_list)


def _site_op_complex(op: np.ndarray, site: int, N: int) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for r in range(N):
        out = np.kron(out, op if r == site % N else _ID.astype(complex))
    return out
