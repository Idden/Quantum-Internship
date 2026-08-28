"""
reference_kron.py
=================

An INDEPENDENT construction of the model, used only to prove that
scarcore.py builds the same Hamiltonians.

Nothing here imports scarcore. Every operator is assembled in the full
2^N space out of explicit 2x2 matrices and numpy Kronecker products, and
only then restricted to the blockade subspace. It is deliberately slow and
obviously correct, which is the point: it is the thing scarcore is checked
against, so it must not share any of scarcore's cleverness.

Site convention, matching quantumScarFunctions.py
------------------------------------------------
Bit '1' is the excited (Rydberg) state and carries sigma^z = +1; bit '0'
carries sigma^z = -1. In the ordering (|0>, |1>) that makes

    Z = diag(-1, +1)      P = |0><0|      X = sigma_x      Y = sigma_y

and `np.kron` with site 0 as the leading factor makes the full-space index
of a bitstring simply int(s, 2).
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[-1, 0], [0, 1]], dtype=complex)     # '0' -> -1, '1' -> +1
P0 = np.array([[1, 0], [0, 0]], dtype=complex)     # projector onto '0'


def _kron_chain(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def _place(N, assignment):
    """Full-space operator with `assignment = {site: 2x2}` and identity elsewhere."""
    return _kron_chain([assignment.get(r, I2) for r in range(N)])


def blockade_indices(N):
    """Full-space indices of the periodic-blockade basis, ascending."""
    idx, strings = [], []
    for k in range(2 ** N):
        s = format(k, f"0{N}b")
        if '11' in s or (s[0] == '1' and s[-1] == '1'):
            continue
        idx.append(k)
        strings.append(s)
    return np.array(idx, dtype=int), strings


def pxp_full(N, omega=1.0, zz_coeff=-0.026):
    """Clean deformed PXP: (omega/2) sum_r P X P  +  zz_coeff*omega * sum_r P X P (Z_{r-2}+Z_{r+2})."""
    bare = np.zeros((2 ** N, 2 ** N), dtype=complex)
    zz = np.zeros((2 ** N, 2 ** N), dtype=complex)

    for r in range(N):
        pxp = _place(N, {(r - 1) % N: P0, r: X, (r + 1) % N: P0})
        bare += pxp

        zfac = _place(N, {(r - 2) % N: Z}) + _place(N, {(r + 2) % N: Z})
        zz += pxp @ zfac

    return (omega / 2.0) * bare + (zz_coeff * omega) * zz


def field_full(N, h, op):
    """sum_r h[r] * op_r in the full space."""
    out = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for r in range(N):
        if h[r] != 0.0:
            out += h[r] * _place(N, {r: op})
    return out


def drive_full(N, drive_weights):
    """The staggered drive operator sum_r w_r Z_r z2_r."""
    z2 = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=float)
    return field_full(N, drive_weights * z2, Z)


def restrict(H_full, idx):
    """Project a full-space operator onto the blockade subspace."""
    return H_full[np.ix_(idx, idx)]
