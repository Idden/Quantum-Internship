"""
scarcore.py
===========

qutip-free numerical core for the scar-vs-qubit quantum battery search.

Why this file exists
--------------------
`test_main.py` did every one of these things *inside every objective
evaluation*:

    * rebuilt the constrained basis and the flip sparsity pattern with
      O(D*N) Python string operations,
    * re-drew the disorder with numpy's global RNG,
    * re-diagonalised H0_dis,
    * called qutip `sesolve` N times for the N decoupled qubits.

Only the last one is real physics work. This module removes the rest.

The key structural fact
-----------------------
The disorder is drawn as `strength x (a fixed uniform pattern)`.
`np.random.uniform(-d, d, N)` consumes exactly the same underlying uniform
deviates as `np.random.uniform(-1, 1, N)` and returns `d` times the result.
So for a FIXED seed:

    hz(z) = z * v_z        hy(y) = y * v_y        hx(x) = x * v_x
    drive_weights(dd) = 1 + dd * v_w

with v_z, v_y, v_x, v_w drawn once per seed. Therefore

    H0_dis(x, y, z) = H_clean + z*Dz + y*Ay + x*Ax
    H1(dd)          = D1_base + dd*D1_pert

is EXACTLY linear in the four disorder parameters. Every objective
evaluation becomes four sparse scalar-multiply-adds on cached matrices
instead of thousands of Python string operations plus RNG calls.

The physics is unchanged. `reference_kron.py` checks the Hamiltonians
elementwise against an independent full-2^N Kronecker construction, and
`check_against_qutip()` checks them against `quantumScarFunctions.py`
itself whenever qutip is importable (i.e. on the cluster).

Conventions carried over from quantumScarFunctions.py, do not change
-------------------------------------------------------------------
* Basis order: lexicographic ascending over length-N bitstrings with no
  two adjacent '1', then dropping any string with first == last == '1'
  (the periodic Rydberg blockade). Dimension is the Lucas number L_N.
* H0 = (Omega/2) * H_PXP  +  (-0.026 * Omega) * H_PXP_zz
* RNG draw order: hz, then hy, then hx, then the drive weights; each is
  preceded by a `np.random.choice(N, size=N, replace=False)` permutation.
  This ordering is what keeps the scar model and the qubit model on the
  same disorder realisation for a given seed (common random numbers).
"""

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigsh, splu, LinearOperator

OMEGA = 1.0          # overall PXP scale
ZZ_COEFF = -0.026    # the P(sz_{r-2} + sz_{r+2})X P deformation


# ======================================================================
# Basis
# ======================================================================

def build_basis(N):
    """
    Constrained basis, in exactly the order quantumScarFunctions.py produces.

    binNoConsecOnesEfficient recurses '0' first, then '1' when the previous
    bit was not '1', so the strings come out in ascending lexicographic
    order. The periodic blockade then removes first == last == '1'.

    Returns
    -------
    strings : list[str]
    bits    : (D, N) uint8
    """
    if N < 4 or N % 2:
        raise ValueError("N must be even and >= 4.")

    out = []

    def rec(n, prev, cur):
        if n == 0:
            out.append(cur)
            return
        rec(n - 1, '0', cur + '0')
        if prev != '1':
            rec(n - 1, '1', cur + '1')

    rec(N, None, '')

    strings = [s for s in out if not (s[0] == '1' and s[-1] == '1')]
    bits = np.array([[int(c) for c in s] for s in strings], dtype=np.uint8)

    return strings, bits


def z2_bits(N):
    """The Neel / Z2 product state |1010...>, as +-1."""
    return np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)


# ======================================================================
# Field-independent structure
# ======================================================================

def build_structure(N):
    """
    Everything about the model that does NOT depend on the five search
    parameters. Built once per N, cached to disk by build_cache.py, and
    thereafter loaded rather than recomputed.

    Returns a dict with
        D           dimension (Lucas number L_N)
        S           (D, N) int8, sigma^z eigenvalue of each site in each state
        H_clean     (D, D) csr, the clean deformed PXP Hamiltonian
        flip_rows/cols/site/phase
                    single-site spin-flip sparsity pattern that stays inside
                    the constrained basis, used for the x and y disorder
        z2_index    index of the Z2 state in the basis
        d1_base     (D,) diagonal of the undisordered drive operator
    """
    strings, bits = build_basis(N)
    D = len(strings)
    index = {s: i for i, s in enumerate(strings)}

    S = (2 * bits.astype(np.int16) - 1).astype(np.int8)

    # ---- clean PXP + the zz deformation -----------------------------
    # For each basis state i and each site r whose two neighbours are both
    # '0', flipping r gives another basis state j. The bare term contributes
    # 1; the deformation contributes sz_{r-2} + sz_{r+2} evaluated on the
    # UNFLIPPED string, matching get_scar_ham.
    rows, cols, bare, zz = [], [], [], []

    for i, s in enumerate(strings):
        for r in range(N):
            if s[(r - 1) % N] == '0' and s[(r + 1) % N] == '0':
                flipped = s[:r] + ('1' if s[r] == '0' else '0') + s[r + 1:]
                j = index.get(flipped)
                if j is None:
                    continue
                rows.append(j)
                cols.append(i)
                bare.append(1.0)
                zz.append(
                    (1 if s[(r - 2) % N] == '1' else -1)
                    + (1 if s[(r + 2) % N] == '1' else -1)
                )

    shape = (D, D)
    H_bare = sp.csr_matrix((bare, (rows, cols)), shape=shape)
    H_zz = sp.csr_matrix((np.asarray(zz, dtype=float), (rows, cols)), shape=shape)
    H_clean = (OMEGA / 2.0) * H_bare + (ZZ_COEFF * OMEGA) * H_zz

    # ---- single-site flip pattern for the x / y disorder -------------
    # Note this is NOT blockade-restricted: it is every single-site flip
    # whose image happens to still lie in the constrained basis, which is
    # what get_dis_scar_ham does.
    frows, fcols, fsite, fphase = [], [], [], []

    for i, s in enumerate(strings):
        for r in range(N):
            flipped = s[:r] + ('1' if s[r] == '0' else '0') + s[r + 1:]
            j = index.get(flipped)
            if j is not None:
                frows.append(j)
                fcols.append(i)
                fsite.append(r)
                fphase.append(1j if s[r] == '0' else -1j)

    z2 = z2_bits(N)
    z2_str = ''.join('1' if b > 0 else '0' for b in z2)

    return {
        "N": int(N),
        "D": int(D),
        "S": S,
        "H_clean": H_clean.tocsr(),
        "flip_rows": np.asarray(frows, dtype=np.int32),
        "flip_cols": np.asarray(fcols, dtype=np.int32),
        "flip_site": np.asarray(fsite, dtype=np.int32),
        "flip_phase": np.asarray(fphase, dtype=complex),
        "z2": z2.astype(np.int8),
        "z2_index": int(index[z2_str]),
        "d1_base": (S.astype(float) @ z2.astype(float)),
        "strings": strings,
    }


# ======================================================================
# Disorder: the unit fields
# ======================================================================

def draw_unit_fields(N, seed):
    """
    Draw v_z, v_y, v_x, v_w for one seed.

    This consumes the global numpy RNG in EXACTLY the order that
    get_dis_scar_ham (hz, hy, hx) followed by get_scar_H1 (drive weights)
    does, each field preceded by its site permutation. Do not reorder it:
    the common-random-numbers pairing between the scar model and the
    qubit model depends on both consuming the same stream.

    Because the DE bounds are 10**[-3, 0] for x, y, z and [0.01, 5] for dd,
    all four strengths are always non-zero, so all four draws always happen
    and the stream never branches.
    """
    np.random.seed(int(seed))

    fields = []
    for _ in range(3):                                    # hz, hy, hx
        perm = np.random.choice(N, size=N, replace=False)
        v = np.zeros(N)
        v[perm] = np.random.uniform(-1.0, 1.0, N)
        fields.append(v)

    perm = np.random.choice(N, size=N, replace=False)     # drive weights
    v_w = np.zeros(N)
    v_w[perm] = np.random.uniform(-1.0, 1.0, N)

    v_z, v_y, v_x = fields
    return v_z, v_y, v_x, v_w


def assemble_scar_H(struct, v_z, v_y, v_x, v_w, x, y, z, dd):
    """
    H0_dis and the drive operator for one parameter point, as a linear
    combination of cached pieces. No Python loops, no RNG.

    Returns
    -------
    H0 : (D, D) csr complex
    d1 : (D,) float, the diagonal of the drive operator H1
    """
    D = struct["D"]
    idx = np.arange(D, dtype=np.int32)

    H0 = struct["H_clean"].astype(complex).copy()

    # z: diagonal sigma^z field
    diag = struct["S"].astype(float) @ (z * v_z)
    H0 = H0 + sp.csr_matrix((diag.astype(complex), (idx, idx)), shape=(D, D))

    # x and y: the same flip pattern, different weights
    site = struct["flip_site"]
    data = (x * v_x)[site].astype(complex) + (y * v_y)[site] * struct["flip_phase"]
    H0 = H0 + sp.csr_matrix(
        (data, (struct["flip_rows"], struct["flip_cols"])), shape=(D, D)
    )

    # drive: H1 = diag( (S * drive_weights) @ z2 ), linear in dd
    d1 = struct["d1_base"] + dd * (
        (struct["S"].astype(float) * v_w) @ struct["z2"].astype(float)
    )

    return H0.tocsr(), d1


# ======================================================================
# Scar evolution
# ======================================================================

def evolve_scar(struct, H0, d1, amplitude, omega_d, tlist,
                rtol=1e-8, atol=1e-10):
    """
    Evolve the ground state of H0 under H(t) = H0 + A sin(omega_d t) H1.

    The dense eigendecomposition of H0 is done once here and then reused for
    three separate things, which is why it is worth its cost:

        1. the initial state (the ground state),
        2. the bandwidth used to normalise R,
        3. the level populations p_n(t) needed for the DEPHASED ergotropy.

    The old code paid for a full dense `eigh` and then threw away everything
    except items 1 and 2.

    Returns
    -------
    dict with
        E        (D,) eigenvalues of H0, ascending
        pops     (D, nt) level populations |<E_n|psi(t)>|^2
        psi      (D, nt) the state itself, for the entanglement entropy
        bandwidth
    """
    D = struct["D"]
    H0d = np.asarray(H0.todense())
    E, V = np.linalg.eigh(H0d)

    bandwidth = float(E[-1] - E[0])
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"Bad scar bandwidth: {bandwidth}")

    psi0 = V[:, 0].astype(complex)

    H0c = H0.astype(complex).tocsr()
    d1c = d1.astype(float)

    def rhs(t, psi):
        return -1j * (H0c @ psi + (amplitude * np.sin(omega_d * t)) * (d1c * psi))

    sol = solve_ivp(
        rhs,
        (float(tlist[0]), float(tlist[-1])),
        psi0,
        t_eval=tlist,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"scar integration failed: {sol.message}")

    psi = sol.y                              # (D, nt)
    c = V.conj().T @ psi                     # amplitudes in the H0 eigenbasis
    pops = np.abs(c) ** 2

    return {"E": E, "V": V, "pops": pops, "psi": psi, "bandwidth": bandwidth}


# ======================================================================
# Decoupled qubit evolution
# ======================================================================

# Standard Pauli matrices, exactly as qutip's sigmax/sigmay/sigmaz, because
# get_qubit_ham builds the comparison qubits from those.
#
# Note this is a DIFFERENT sign convention for sigma^z than the chain uses:
# the chain's sigzMap sends bit '0' -> -1 and bit '1' -> +1, which in the
# (|0>, |1>) ordering used for the basis strings is diag(-1, +1). The two
# never share a basis, so only the relative algebra inside each model
# matters, and both are internally consistent.
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def evolve_qubits(N, v_z, v_y, v_x, v_w, x, y, z, dd,
                  amplitude, omega_q, wm, tlist, rtol=1e-9, atol=1e-12):
    """
    The N decoupled comparison qubits.

        H0_i = -wm/2 * sx + z*v_z[i] sz + y*v_y[i] sy + x*v_x[i] sx
        H1_i = (1 + dd*v_w[i]) sz

    Every qubit is a 2x2 problem, so all N of them are integrated together
    as one length-2N vector. The blocks never mix; this is purely so that
    there is one integrator call instead of N.

    Returns per-qubit eigenvalues and populations, which is what the
    *local* dephased ergotropy needs -- the qubits' advantage is that their
    work is extractable with local unitaries, so their passive state must be
    computed block by block, not globally.
    """
    hz, hy, hx = z * v_z, y * v_y, x * v_x
    dw = 1.0 + dd * v_w

    H0 = np.empty((N, 2, 2), dtype=complex)
    for i in range(N):
        H0[i] = -0.5 * wm * _SX + hz[i] * _SZ + hy[i] * _SY + hx[i] * _SX

    E = np.empty((N, 2))
    V = np.empty((N, 2, 2), dtype=complex)
    for i in range(N):
        E[i], V[i] = np.linalg.eigh(H0[i])

    bandwidth = float(np.sum(E[:, 1] - E[:, 0]))
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"Bad total qubit bandwidth: {bandwidth}")

    psi0 = V[:, :, 0].reshape(-1).astype(complex)     # each qubit in its GS

    def rhs(t, y_flat):
        psi = y_flat.reshape(N, 2)
        drive = amplitude * np.sin(omega_q * t)
        out = np.einsum("nij,nj->ni", H0, psi)
        out += drive * (dw[:, None] * np.einsum("ij,nj->ni", _SZ, psi))
        return (-1j * out).reshape(-1)

    sol = solve_ivp(
        rhs,
        (float(tlist[0]), float(tlist[-1])),
        psi0,
        t_eval=tlist,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"qubit integration failed: {sol.message}")

    psi = sol.y.reshape(N, 2, -1)                     # (N, 2, nt)
    c = np.einsum("nji,njt->nit", V.conj(), psi)      # eigenbasis amplitudes
    pops = np.abs(c) ** 2                             # (N, 2, nt)

    return {"E": E, "pops": pops, "bandwidth": bandwidth}


# ======================================================================
# Metrics
# ======================================================================

def ergotropy_curves(E, pops):
    """
    Normalised energy R(t) and normalised DEPHASED ergotropy R_deph(t),
    both still to be divided by the bandwidth by the caller.

    The state is pure and the evolution unitary, so the passive state of
    the full state is the ground state and

        ergotropy(t) = <H0>(t) - E_min = R(t) * W

    i.e. R is exactly the normalised ergotropy and R(0) = 0 says the
    battery starts genuinely empty. That is the strongest thing that can
    be said for the metric and it is free.

    R_deph is the work still extractable once the coherences between H0
    eigenstates are lost: sort the populations descending onto the
    ascending energies. R_deph <= R always, and the gap between them is
    what separates coherent charging from plain heating -- at the
    coherent point in the earlier review 79% of R survived dephasing, at
    the thermal point only 31%.
    """
    energy = E @ pops                                   # (nt,)
    order = np.argsort(-pops, axis=0)                   # descending populations
    p_sorted = np.take_along_axis(pops, order, axis=0)
    passive = E @ p_sorted                              # ascending E, descending p

    return energy, passive


def scar_metrics(E, pops, bandwidth):
    """R(t) and R_deph(t) for the many-body chain (global ergotropy)."""
    energy, passive = ergotropy_curves(E, pops)
    R = (energy - energy[0]) / bandwidth
    R_deph = (energy - passive) / bandwidth
    return R, R_deph


def qubit_metrics(E, pops, bandwidth):
    """
    R(t) and R_deph(t) for the decoupled qubits.

    Both are summed over qubits BEFORE normalising, and the passive state
    is computed per qubit. That is the honest comparison: the qubits' work
    is locally extractable, so they are not allowed the global passive
    state that the chain gets.
    """
    energy = np.einsum("ni,nit->t", E, pops)
    e0 = energy[0]

    order = np.argsort(-pops, axis=1)
    p_sorted = np.take_along_axis(pops, order, axis=1)
    passive = np.einsum("ni,nit->t", E, p_sorted)

    R = (energy - e0) / bandwidth
    R_deph = (energy - passive) / bandwidth
    return R, R_deph


def first_peak(R, tlist):
    """
    Height and time of the FIRST local maximum of R(t).

    This is the coherent charging event. `max_t R` collapses the whole
    trace to one number and cannot tell it apart from a late thermal
    plateau; R_1 can.
    """
    if len(R) < 3:
        k = int(np.argmax(R))
        return float(R[k]), float(tlist[k])

    interior = np.arange(1, len(R) - 1)
    is_peak = (R[1:-1] > R[:-2]) & (R[1:-1] >= R[2:])
    peaks = interior[is_peak]

    k = int(peaks[0]) if len(peaks) else int(np.argmax(R))
    return float(R[k]), float(tlist[k])


def refined_max(R, tlist):
    """
    Maximum of R(t) with a parabolic correction through the three samples
    around the argmax.

    The earlier review measured that sampling on dt = 0.5 biases the
    maximum downward by up to 0.033, which is the same size as the effect
    being searched for, and biases the two models by different amounts
    because their frequency content differs. This removes the leading part
    of that bias for free. It is not a substitute for a fine enough grid,
    so keep --nt large enough that the peak is genuinely resolved.
    """
    k = int(np.argmax(R))
    if k == 0 or k == len(R) - 1:
        return float(R[k]), float(tlist[k])

    y0, y1, y2 = float(R[k - 1]), float(R[k]), float(R[k + 1])
    denom = y0 - 2.0 * y1 + y2

    if denom == 0.0:
        return y1, float(tlist[k])

    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))

    dt = float(tlist[1] - tlist[0])
    return y1 - 0.25 * (y0 - y2) * delta, float(tlist[k]) + delta * dt


def max_power(R, tlist):
    """
    max_t R(t)/t, the charging power.

    In the earlier review the scar chain won on power at two of the three
    test points while losing on max R, so this is probably where the real
    result lives.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tlist > 0, R / np.where(tlist > 0, tlist, 1.0), -np.inf)
    k = int(np.argmax(p))
    return float(p[k]), float(tlist[k])


def half_chain_entropy(struct, psi_col):
    """
    Half-chain von Neumann entropy of one state in the constrained basis.

    Reshapes the coefficient vector into the (2^{N/2}, 2^{N/2}) matrix
    C_AB and takes the singular values, matching get_C_AB_matrix. Compare
    against S_max = (N/2) ln 2 to read off how thermal the state is.
    """
    N = struct["N"]
    NA = N // 2
    NB = N - NA

    C = np.zeros((2 ** NA, 2 ** NB), dtype=complex)
    for k, s in enumerate(struct["strings"]):
        C[int(s[:NA], 2), int(s[NA:], 2)] = psi_col[k]

    lam = np.linalg.svd(C, compute_uv=False) ** 2
    lam = lam[lam > 1e-12]

    return 0.0 if lam.size == 0 else float(-np.sum(lam * np.log(lam)))


# ======================================================================
# Clean scar subspace (diagnostics only -- never on the DE hot path)
# ======================================================================

def _max_eig(H):
    return float(eigsh(H, k=1, which="LA", return_eigenvectors=False, tol=0)[0].real)


def build_Hy_staggered(struct):
    """The staggered sigma^y operator sum_r (-1)^r Y_r, restricted to the basis."""
    N, D = struct["N"], struct["D"]
    hy = np.array([(-1.0) ** r for r in range(N)])
    data = hy[struct["flip_site"]] * struct["flip_phase"]
    return sp.csr_matrix(
        (data, (struct["flip_rows"], struct["flip_cols"])), shape=(D, D)
    )


def get_zero_scar(struct, k0=None):
    """
    The E = 0 scar: the projection of the Z2 state onto the maximal-S^2
    part of the zero-energy subspace of the clean PXP Hamiltonian.

    Same algorithm as quantumScarFunctions.get_zero_scar, with one bug
    fixed: the ARPACK start size K is now clamped to D - 2. The original
    started at `max(16, 0.02*D)` unconditionally, so it raised at N = 4
    (D = 7) because ARPACK requires k < D - 1. That was documented in the
    test suite as "needs an explicit k0"; it is just a missing clamp.
    """
    N, D = struct["N"], struct["D"]
    N2 = N // 2

    Hx = struct["H_clean"].astype(complex).tocsr()
    Hy = build_Hy_staggered(struct).astype(complex)
    Hz = sp.diags(struct["d1_base"]).astype(complex).tocsr()

    Hx = Hx * (N2 / _max_eig(Hx))
    Hy = Hy * (N2 / _max_eig(Hy))
    Hz = Hz * (N2 / _max_eig(Hz))

    H2 = (Hx @ Hx).tocsc()

    lu = splu((H2 + 1e-9 * sp.eye(D, format="csc", dtype=complex)).tocsc())
    OPinv = LinearOperator((D, D), matvec=lu.solve, dtype=complex)

    K = k0 if k0 is not None else max(16, int(0.02 * D))
    K = max(1, min(K, D - 2))                     # <-- the fix

    while True:
        w, v = eigsh(H2, k=K, sigma=-1e-9, which="LM", OPinv=OPinv)
        nz = np.linalg.norm(Hx @ v, axis=0) < 1e-8
        if nz.sum() < K or K >= D - 2:
            break
        K = min(2 * K, D - 2)

    V, _ = np.linalg.qr(v[:, nz])

    S2 = sum((M @ V).conj().T @ (M @ V) for M in (Hx, Hy, Hz))
    sv, ss = np.linalg.eigh(S2)
    cand = V @ ss[:, np.abs(sv - sv[-1]) < 1e-10]

    z2 = np.zeros(D, dtype=complex)
    z2[struct["z2_index"]] = 1.0

    scar = cand @ (cand.conj().T @ z2)
    norm = np.linalg.norm(scar)

    if norm < 1e-14:
        raise ValueError("Z2 has no overlap with the max-S^2 zero-energy subspace")

    scar = scar / norm
    return scar, float(np.abs(np.vdot(z2, scar)) ** 2)


def build_scar_subspace(struct):
    """
    The N+1 scar tower of the clean chain: in each energy window, the
    eigenstate with the largest |<Z2|E>|^2, with the middle one replaced by
    the exact E = 0 scar.

    Selecting the middle state by `len//2` assumes the windows come out
    symmetric. They do for N = 8..16, but `argmin |E|` is free and cannot
    be wrong, so that is what is used here.
    """
    N, D = struct["N"], struct["D"]
    E, V = np.linalg.eigh(np.asarray(struct["H_clean"].todense()))

    z2 = np.zeros(D)
    z2[struct["z2_index"]] = 1.0
    overlaps = np.abs(V.conj().T @ z2) ** 2

    edges = np.linspace(E[0] - 0.5, E[-1] + 0.5, N + 2)
    picks = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = np.where((E > a) & (E < b))[0]
        if sel.size:
            picks.append(int(sel[np.argmax(overlaps[sel])]))

    states = V[:, picks].astype(complex)
    energies = E[picks]

    zero_scar, z2_overlap = get_zero_scar(struct)
    states[:, int(np.argmin(np.abs(energies)))] = zero_scar

    return {
        "scar_states": states,                 # (D, n_scar)
        "scar_indices": np.array(picks, dtype=int),
        "scar_energies": energies,
        "z2_overlap_zero_scar": z2_overlap,
        "clean_eigenvalues": E,
    }
