"""
Tests for ``get_zero_scar`` -- the maximum-S^2 zero-energy scar state.

This is the most intricate function in the module.  The HPC version replaces
the original dense construction with shift-invert ARPACK plus a QR, so the
defining properties have to be checked directly rather than assumed:

* the returned state is normalised,
* it is annihilated by the clean Hamiltonian (it really is a zero mode),
* it lies in the maximum-S^2 sector of that null space,
* it carries the reported Z2 overlap.

A dense reference implementation is included below for the small sizes, so
the sparse path is checked against an independent computation and not just
against its own output.
"""

import numpy as np
import pytest

from data_utils import save_array_data, save_metadata

# get_zero_scar starts ARPACK at k = max(16, 0.02*D), and ARPACK needs
# k < D-1.  N=4 has D=7, so it cannot run without an explicit k0 -- see
# test_small_chain_needs_explicit_k0 at the bottom of this file.
SIZES = [6, 8, 10]


@pytest.fixture(scope="module")
def zero_scar_cache():
    from GitHub_QM.important.hpc.quantumScarFunctions import get_zero_scar

    cache = {}

    def _get(N):
        if N not in cache:
            cache[N] = get_zero_scar(N)
        return cache[N]

    return _get


# --------------------------------------------------------------------------
# Basic properties
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N", SIZES)
def test_scar_is_a_normalised_column_vector(scar_functions, basis_list, zero_scar_cache, N):
    scar, _ = zero_scar_cache(N)
    D = len(basis_list(N))

    assert scar.shape == (D, 1)
    assert scar.norm() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("N", SIZES)
def test_scar_is_annihilated_by_the_hamiltonian(scar_system, zero_scar_cache, N,
                                                test_subdir):
    """
    The defining property: H|scar> = 0.  If the ARPACK null-space filter let a
    non-zero mode through, this is what would catch it.
    """
    H0 = scar_system(N)[0]
    scar, overlap = zero_scar_cache(N)

    vector = scar.full().ravel()
    residual = np.linalg.norm(H0.full() @ vector)

    assert residual < 1e-8, f"|H|scar>| = {residual:.3e}, not a zero mode"

    save_array_data(test_subdir / "zero_scar.npz", scar=vector)
    save_metadata(
        test_subdir / "metadata.json",
        {"N": N, "residual_norm": float(residual), "z2_overlap": float(overlap)},
    )


@pytest.mark.parametrize("N", SIZES)
def test_reported_overlap_matches_the_state(scar_system, zero_scar_cache, N):
    """The returned overlap must equal |<Z2|scar>|^2 recomputed from scratch."""
    psi0 = scar_system(N)[3]
    scar, reported = zero_scar_cache(N)

    recomputed = abs(np.vdot(psi0.full().ravel(), scar.full().ravel())) ** 2

    assert float(reported) == pytest.approx(recomputed, rel=1e-10, abs=1e-12)
    assert 0.0 < float(reported) <= 1.0


@pytest.mark.parametrize("N", SIZES)
def test_overlap_decreases_with_system_size(zero_scar_cache, N):
    """
    The Z2 weight in the zero-energy scar thins out as the chain grows.  This
    is a sanity check on the trend rather than a specific value.
    """
    overlap = float(zero_scar_cache(N)[1])
    assert 0.1 < overlap < 0.5


def test_overlap_is_monotone_in_N(zero_scar_cache):
    overlaps = [float(zero_scar_cache(N)[1]) for N in SIZES]
    assert all(a > b for a, b in zip(overlaps, overlaps[1:])), overlaps


# --------------------------------------------------------------------------
# Independent dense cross-check of the sparse path
# --------------------------------------------------------------------------
def _dense_zero_scar(scar_functions, N):
    """
    Straightforward dense reconstruction of the same state.

    Builds Hx, Hy, Hz densely, normalises each so its largest eigenvalue is
    N/2, takes the null space of Hx by dense eigendecomposition, diagonalises
    S^2 = Hx^2 + Hy^2 + Hz^2 inside it, keeps the maximum-S^2 sector, and
    projects the Z2 state into it.  No ARPACK, no shift-invert, no QR.
    """
    N2 = N // 2

    H0, _, _, psi0, basisList = scar_functions.get_scar_ham(N, diagonalize=False)

    Hx = H0.full().astype(complex)
    Hy = scar_functions.get_Hy(N, basisList).full().astype(complex)
    Hz = scar_functions.get_scar_H1(N, basisList)[0].full().astype(complex)

    Hx = Hx * (N2 / np.linalg.eigvalsh(Hx).max())
    Hy = Hy * (N2 / np.linalg.eigvalsh(Hy).max())
    Hz = Hz * (N2 / np.linalg.eigvalsh(Hz).max())

    w, v = np.linalg.eigh(Hx)
    null = v[:, np.abs(w) < 1e-9]

    S2 = sum((M @ null).conj().T @ (M @ null) for M in (Hx, Hy, Hz))
    sv, ss = np.linalg.eigh(S2)
    candidates = null @ ss[:, np.abs(sv - sv[-1]) < 1e-8]

    z2 = psi0.full().ravel().astype(complex)
    scar = candidates @ (candidates.conj().T @ z2)
    norm = np.linalg.norm(scar)

    return scar / norm, float(np.abs(np.vdot(z2, scar / norm)) ** 2)


@pytest.mark.parametrize("N", [6, 8, 10])
def test_matches_dense_reference_implementation(scar_functions, zero_scar_cache, N):
    """
    The sparse ARPACK path and a plain dense computation must agree.

    Compared as a projector (outer product) rather than vector-to-vector,
    because both routines are free to return the state up to a global phase.
    """
    sparse_scar = zero_scar_cache(N)[0].full().ravel()
    dense_scar, dense_overlap = _dense_zero_scar(scar_functions, N)

    fidelity = abs(np.vdot(dense_scar, sparse_scar)) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-8), f"fidelity {fidelity:.10f}"

    assert float(zero_scar_cache(N)[1]) == pytest.approx(dense_overlap, abs=1e-8)


@pytest.mark.parametrize("N", [6, 8])
def test_scar_lies_in_the_max_S2_sector(scar_functions, zero_scar_cache, N):
    """
    S^2 |scar> = s_max |scar>, where s_max is the largest S^2 eigenvalue in
    the null space of Hx.  This is the property that distinguishes the scar
    from an arbitrary zero mode.
    """
    N2 = N // 2
    H0, _, _, _, basisList = scar_functions.get_scar_ham(N, diagonalize=False)

    Hx = H0.full().astype(complex)
    Hy = scar_functions.get_Hy(N, basisList).full().astype(complex)
    Hz = scar_functions.get_scar_H1(N, basisList)[0].full().astype(complex)

    Hx = Hx * (N2 / np.linalg.eigvalsh(Hx).max())
    Hy = Hy * (N2 / np.linalg.eigvalsh(Hy).max())
    Hz = Hz * (N2 / np.linalg.eigvalsh(Hz).max())

    S2 = Hx @ Hx + Hy @ Hy + Hz @ Hz

    w, v = np.linalg.eigh(Hx)
    null = v[:, np.abs(w) < 1e-9]
    S2_null = null.conj().T @ S2 @ null
    s_max = np.linalg.eigvalsh(S2_null).max()

    scar = zero_scar_cache(N)[0].full().ravel()
    expectation = np.real(np.vdot(scar, S2 @ scar))

    assert expectation == pytest.approx(s_max, rel=1e-6)


@pytest.mark.parametrize("N", SIZES)
def test_scar_is_deterministic(scar_functions, N):
    """Two calls must give the same state up to a global phase."""
    a = scar_functions.get_zero_scar(N)[0].full().ravel()
    b = scar_functions.get_zero_scar(N)[0].full().ravel()

    assert abs(np.vdot(a, b)) ** 2 == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------
# Known limitation
# --------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_small_chain_needs_explicit_k0(scar_functions):
    """
    Documents a real limitation of the current implementation.

    ``get_zero_scar`` starts ARPACK with ``K = max(16, int(0.02*D))``.  For
    N=4 the constrained dimension is only D=7, so K=16 violates ARPACK's
    ``k < D-1`` requirement and scipy raises.  Passing ``k0`` explicitly is
    the workaround.

    If the default is ever changed to clamp K against D, this test will start
    failing and should simply be deleted.
    """
    with pytest.raises((TypeError, ValueError)):
        scar_functions.get_zero_scar(4)

    scar, overlap = scar_functions.get_zero_scar(4, k0=2)
    assert scar.norm() == pytest.approx(1.0, abs=1e-10)
    assert 0.0 < float(overlap) <= 1.0
