"""
qutip_shim.py
=============

A minimal stand-in for qutip, sufficient to run the *builder* functions of
quantumScarFunctions.py (get_scar_ham, get_dis_scar_ham, get_scar_H1,
get_Hy, get_qubit_ham, get_zero_scar) without qutip installed.

This exists only so the equivalence check between scarcore.py and
quantumScarFunctions.py can run in an environment where qutip cannot be
installed. It implements nothing beyond Qobj arithmetic, basis vectors and
the three Pauli matrices -- no solvers. On the cluster, where qutip is in
the venv, do NOT use this: `validate_core.check_against_qutip()` will pick
up the real package instead.

Import it before quantumScarFunctions:

    import qutip_shim; qutip_shim.install()
    from quantumScarFunctions import get_scar_ham
"""

import sys
import types

import numpy as np
import scipy.sparse as sp


class Qobj:
    """Just enough of a Qobj: sparse storage, scalar/Qobj arithmetic, eigenstates."""

    def __init__(self, data, dims=None):
        if isinstance(data, Qobj):
            data = data.data
        if sp.issparse(data):
            self.data = data.tocsr()
        else:
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self.data = sp.csr_matrix(arr)

    # -- shape helpers -------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def full(self):
        return np.asarray(self.data.todense())

    def dag(self):
        return Qobj(self.data.conj().T)

    # -- arithmetic ----------------------------------------------------
    def __add__(self, other):
        return Qobj(self.data + (other.data if isinstance(other, Qobj) else other))

    __radd__ = __add__

    def __sub__(self, other):
        return Qobj(self.data - (other.data if isinstance(other, Qobj) else other))

    def __mul__(self, other):
        if isinstance(other, Qobj):
            return Qobj(self.data @ other.data)
        return Qobj(self.data * other)

    def __rmul__(self, other):
        return Qobj(self.data * other)

    def __truediv__(self, other):
        return Qobj(self.data / other)

    def __neg__(self):
        return Qobj(-self.data)

    def __matmul__(self, other):
        return Qobj(self.data @ other.data)

    # -- spectral ------------------------------------------------------
    def eigenstates(self, sparse=False, sort="low", eigvals=0):
        w, v = np.linalg.eigh(self.full())
        if sort == "high":
            w, v = w[::-1], v[:, ::-1]
        if eigvals:
            w, v = w[:eigvals], v[:, :eigvals]
        return w, [Qobj(v[:, k]) for k in range(v.shape[1])]

    def eigenenergies(self, sparse=False, sort="low", eigvals=0):
        return self.eigenstates(sparse=sparse, sort=sort, eigvals=eigvals)[0]

    def overlap(self, other):
        return complex(np.vdot(self.full().ravel(), other.full().ravel()))


def basis(n, i):
    v = np.zeros((n, 1), dtype=complex)
    v[i, 0] = 1.0
    return Qobj(v)


def sigmax():
    return Qobj(np.array([[0, 1], [1, 0]], dtype=complex))


def sigmay():
    return Qobj(np.array([[0, -1j], [1j, 0]], dtype=complex))


def sigmaz():
    return Qobj(np.array([[1, 0], [0, -1]], dtype=complex))


def install():
    """Register this module under the name `qutip` in sys.modules."""
    if "qutip" in sys.modules and not getattr(sys.modules["qutip"], "_is_shim", False):
        return False                      # the real qutip is present, leave it

    mod = types.ModuleType("qutip")
    mod._is_shim = True
    mod.__version__ = "shim-0"
    for name in ("Qobj", "basis", "sigmax", "sigmay", "sigmaz"):
        setattr(mod, name, globals()[name])
    sys.modules["qutip"] = mod
    return True
