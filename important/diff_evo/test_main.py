import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import qutip as qt
from scipy.optimize import differential_evolution

from quantumScarFunctions import *


# ============================================================
# Global defaults
#
# wd/t_max/tlist are overwritten from the CLI in main() BEFORE the scipy
# worker pool is forked, so every worker inherits the same values.
# ============================================================

wd = 0.6366896896896898
wm = 1.0
t_max = 200.0
tlist = np.linspace(0.0, t_max, 400)

DEFAULT_OUTDIR = str(Path.cwd() / "de_results")


# ============================================================
# Per-process caches
# With scipy workers, each worker is its own Python process.
# These caches therefore run once per worker process, not once globally.
#
# _SCAR_CLEAN_CACHE and _PRECOMP_CACHE are built in main() before the pool is
# forked, so on Linux the workers share those pages copy-on-write instead of
# each rebuilding (and re-diagonalising) them.
# ============================================================

_SCAR_CLEAN_CACHE = {}
_SCAR_SUBSPACE_CACHE = {}
_PRECOMP_CACHE = {}


# ============================================================
# Differential evolution globals
# Needed because scipy workers require a picklable objective function.
# ============================================================

_DE_N = None
_DE_SEEDS = None
_DE_LOG_EVERY_EVAL = True
_DE_DENSE_EIG = False       # force dense diagonalisation (debug / tiny N)
_DE_EVAL_MEMO = {}          # in-process memo for exact repeat vectors
_DE_EVALS_DONE = 0          # used for fail-fast on the first evaluation
_DE_LOG_DIR = None          # one append-only jsonl per worker process


# ============================================================
# Utility functions
# ============================================================

def validate_N(N):
    N = int(N)
    if N < 4:
        raise ValueError("N must be >= 4.")
    if N % 2 != 0:
        raise ValueError("N must be even.")
    return N


def safe_float_name(x):
    return f"{float(x):.6f}".replace("-", "m").replace(".", "p")


def unpack_de_vector(vec):
    """
    Differential evolution searches over:

        vec[0] = log10(x)
        vec[1] = log10(y)
        vec[2] = log10(z)
        vec[3] = ds
        vec[4] = dd

    This keeps x, y, z logarithmic, like np.logspace.
    """
    logx, logy, logz, ds, dd = vec

    x = 10.0 ** float(logx)
    y = 10.0 ** float(logy)
    z = 10.0 ** float(logz)
    ds = float(ds)
    dd = float(dd)

    return x, y, z, ds, dd


def make_eval_hash(N, x, y, z, ds, dd, seeds):
    text = (
        f"N={N}|"
        f"x={x:.16e}|y={y:.16e}|z={z:.16e}|"
        f"ds={ds:.16e}|dd={dd:.16e}|"
        f"seeds={list(seeds)}|"
        f"wd={wd:.16e}|tmax={t_max:.16e}|nt={len(tlist)}"
    )
    return hashlib.md5(text.encode()).hexdigest()[:20]


def atomic_savez(path, **kwargs):
    """
    Save an npz safely.

    Important Windows/local fix:
    np.savez appends .npz if the filename does not already end with .npz.
    So the temporary file must still end in .npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # pid in the temp name: two workers must never share a temp file.
    tmp_path = path.with_name(f"{path.stem}.tmp{os.getpid()}.npz")

    np.savez(tmp_path, **kwargs)
    os.replace(tmp_path, path)


def log_eval(record):
    """
    Append one evaluation to this process's jsonl log.

    One file per worker pid, so there is no cross-process interleaving and no
    lock. Thousands of tiny .npz files on a parallel filesystem is a metadata
    problem; one line per evaluation is not, and re-ranking later is a single
    pass over these files.
    """
    if _DE_LOG_DIR is None:
        return

    # pid is resolved at call time: the workers are forked, so they must not
    # all inherit one filename.
    path = Path(_DE_LOG_DIR) / f"evals_pid{os.getpid()}.jsonl"

    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_clean_scar_cached(N):
    """
    Build the clean constrained scar Hamiltonian once per N per worker process.
    """
    N = validate_N(N)

    if N not in _SCAR_CLEAN_CACHE:
        print(f"Building clean scar Hamiltonian once for N={N}", flush=True)
        _SCAR_CLEAN_CACHE[N] = get_scar_ham(N)

    return _SCAR_CLEAN_CACHE[N]


# ============================================================
# Fast kernels
#
# These replace three things that were being redone from scratch on every
# single objective evaluation:
#
#   1. the O(D*N) python bitstring loops in get_dis_scar_ham / get_scar_H1,
#      even though the *sparsity pattern* never changes - only the random
#      field values do
#   2. the full dense diagonalisation of H0_dis, when only the two extremal
#      eigenvalues and the ground state are ever used
#   3. N separate sesolve calls for the N decoupled qubits
#
# The physics is unchanged and check_fast_kernels() asserts that at startup.
# Critically, draw_scar_fields consumes the RNG in exactly the same order as
# get_dis_scar_ham + get_scar_H1, which is what keeps the scar and qubit
# models on the same disorder realisation for a given seed.
# ============================================================

def get_precomp_cached(N):
    """
    Field-independent structure of the constrained basis. Built once per N per
    process, reused by every evaluation.

        S      (D, N) int8, the sigma^z eigenvalue of each site in each state
        rows/cols/site/phase : the single-site spin-flip sparsity pattern that
                               stays inside the constrained basis
    """
    N = validate_N(N)

    if N in _PRECOMP_CACHE:
        return _PRECOMP_CACHE[N]

    H0_clean, eigenvalues, eigenstates, psi0, basisList = get_clean_scar_cached(N)

    D = len(basisList)
    basisMap = {s: i for i, s in enumerate(basisList)}

    S = np.array([[2 * int(c) - 1 for c in s] for s in basisList], dtype=np.int8)

    rows, cols, site, phase = [], [], [], []

    for i, s in enumerate(basisList):
        for r in range(N):
            flipped = s[:r] + ('1' if s[r] == '0' else '0') + s[r + 1:]
            j = basisMap.get(flipped)

            if j is not None:
                rows.append(j)
                cols.append(i)
                site.append(r)
                phase.append(1j if s[r] == '0' else -1j)

    precomp = {
        "D": D,
        "S": S,
        "rows": np.asarray(rows, dtype=np.int32),
        "cols": np.asarray(cols, dtype=np.int32),
        "site": np.asarray(site, dtype=np.int32),
        "phase": np.asarray(phase, dtype=complex),
        "z2": 2 * np.array([int(b) for b in z2_initial(N)]) - 1,
        "H0_clean_sp": qobj_to_scipy(H0_clean).tocsr(),
        "diag_index": np.arange(D, dtype=np.int32),
    }

    _PRECOMP_CACHE[N] = precomp
    return precomp


def qobj_to_scipy(H):
    """
    Get a scipy CSR out of a Qobj without going through a dense D x D array.
    qutip 4 stores scipy sparse directly; qutip 5 wraps it.
    """
    data = getattr(H, "data", None)

    if sp.issparse(data):
        return data.tocsr()

    if hasattr(data, "as_scipy"):
        try:
            out = data.as_scipy()
            if sp.issparse(out):
                return out.tocsr()
        except Exception:
            pass

    try:
        return qobj_to_scipy(H.to("CSR"))
    except Exception:
        return sp.csr_matrix(H.full())


def draw_scar_fields(N, ham_disorder, ds_dis):
    """
    Draw the disorder fields in EXACTLY the order that
    get_dis_scar_ham (hz, then hy, then hx) and get_scar_H1 (drive weights) do.

    Do not reorder this. The whole same-seed pairing between the scar model and
    the decoupled-qubit model depends on both consuming the RNG identically.
    """
    hz = np.zeros(N)
    hy = np.zeros(N)
    hx = np.zeros(N)

    for k, h in ((0, hz), (1, hy), (2, hx)):
        if ham_disorder[k] != 0.0:
            d = ham_disorder[k]
            dis_sites = np.random.choice(N, size=N, replace=False)
            h[dis_sites] = np.random.uniform(-d, d, N)

    drive_weights = np.ones(N)

    if ds_dis != 0.0:
        dis_sites = np.random.choice(N, size=N, replace=False)
        drive_weights[dis_sites] += np.random.uniform(-ds_dis, ds_dis, N)

    return hz, hy, hx, drive_weights


def build_dis_H_fast(precomp, hz, hy, hx, has_z, has_y, has_x):
    D = precomp["D"]
    idx = precomp["diag_index"]

    H = precomp["H0_clean_sp"].astype(complex)

    if has_z:
        diag = precomp["S"] @ hz
        H = H + sp.csr_matrix((diag, (idx, idx)), shape=(D, D))

    if has_y or has_x:
        data = np.zeros(len(precomp["site"]), dtype=complex)

        if has_x:
            data = data + hx[precomp["site"]]
        if has_y:
            data = data + hy[precomp["site"]] * precomp["phase"]

        H = H + sp.csr_matrix(
            (data, (precomp["rows"], precomp["cols"])), shape=(D, D)
        )

    return H.tocsr()


def build_H1_fast(precomp, drive_weights):
    D = precomp["D"]
    idx = precomp["diag_index"]
    diag = (precomp["S"] * drive_weights) @ precomp["z2"]

    return sp.csr_matrix(
        (diag.astype(float), (idx, idx)), shape=(D, D)
    ).tocsr()


def ground_state_and_bandwidth(H_qobj, D):
    """
    simulate_scar only needs the ground state and the spectral width, never the
    full spectrum. A dense eigh costs 13 s at N=16 and minutes at N=18; two
    one-vector ARPACK solves cost 0.025 s and give the same numbers.
    """
    use_dense = _DE_DENSE_EIG or D <= 256

    if not use_dense:
        try:
            e_low, v_low = H_qobj.eigenstates(sparse=True, sort="low", eigvals=1)
            e_high = H_qobj.eigenenergies(sparse=True, sort="high", eigvals=1)
            return v_low[0], float(e_high[0] - e_low[0])
        except Exception:
            # ARPACK can fail to converge on a nasty spectrum. Fall back rather
            # than lose the evaluation.
            pass

    eigenvalues, eigenstates = H_qobj.eigenstates()
    return eigenstates[0], float(eigenvalues[-1] - eigenvalues[0])


def build_qubit_block(N, hz, hy, hx, drive_weights):
    """
    The N decoupled qubits as one block-diagonal Hamiltonian.

    The blocks never talk to each other, so evolving the normalised direct sum
    (1/sqrt(N)) (+)_i |g_i> evolves every qubit independently and

        <H0_blk>(t) = (1/N) sum_i <H0_i>(t)

    which turns N sesolve calls into one. Each qutip solver setup costs more
    than a 2x2 integration does, so this is worth 4-6x at these sizes.
    """
    sigz = qt.sigmaz().full()
    sigy = qt.sigmay().full()
    sigx = qt.sigmax().full()

    H0_blocks = []
    H1_blocks = []
    ground_states = []
    total_bandwidth = 0.0

    for i in range(N):
        H0_i = -0.5 * wm * sigx + hz[i] * sigz + hy[i] * sigy + hx[i] * sigx

        eigenvalues, eigenvectors = np.linalg.eigh(H0_i)
        bandwidth = float(eigenvalues[-1] - eigenvalues[0])

        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError(f"Bad qubit bandwidth: {bandwidth}")

        total_bandwidth += bandwidth

        H0_blocks.append(H0_i)
        H1_blocks.append(drive_weights[i] * sigz)
        ground_states.append(eigenvectors[:, 0])

    H0_blk = qt.Qobj(sp.block_diag(H0_blocks, format="csr"))
    H1_blk = qt.Qobj(sp.block_diag(H1_blocks, format="csr"))

    psi0 = np.concatenate(ground_states) / np.sqrt(N)
    psi0 = qt.Qobj(psi0.reshape(-1, 1))

    return H0_blk, H1_blk, psi0, total_bandwidth


def check_fast_kernels(N, atol=1.0e-9):
    """
    Assert that the fast path reproduces quantumScarFunctions exactly, on the
    same RNG stream.

    Run once per job, in the parent, before anything expensive. This is the
    guard against quantumScarFunctions.py drifting under main.py again: if a
    refactor changes the physics or the order of the random draws, this fails
    in the first seconds of the job instead of silently producing a different
    model for 4 hours.
    """
    N = validate_N(N)

    H0_clean, eigenvalues, eigenstates, psi0, basisList = get_clean_scar_cached(N)
    precomp = get_precomp_cached(N)

    x, y, z, dd = 0.37, 0.12, 0.44, 0.9

    np.random.seed(12345)
    H_ref, _, _ = get_dis_scar_ham(H0_clean, N, basisList, ham_disorder=[z, y, x])
    H1_ref, w_ref = get_scar_H1(N, basisList, ds_dis=dd)
    tail_ref = np.random.uniform(size=4)

    np.random.seed(12345)
    hz, hy, hx, w_fast = draw_scar_fields(N, [z, y, x], dd)
    H_fast = build_dis_H_fast(precomp, hz, hy, hx, z != 0, y != 0, x != 0)
    H1_fast = build_H1_fast(precomp, w_fast)
    tail_fast = np.random.uniform(size=4)

    dH = abs(qobj_to_scipy(H_ref) - H_fast).max() if H_fast.nnz else 0.0
    dH1 = abs(qobj_to_scipy(H1_ref) - H1_fast).max() if H1_fast.nnz else 0.0
    dw = float(np.max(np.abs(w_ref - w_fast)))
    rng_ok = bool(np.allclose(tail_ref, tail_fast))

    print(
        f"self-check N={N}: max|dH|={dH:.3e} max|dH1|={dH1:.3e} "
        f"max|dw|={dw:.3e} rng_stream_aligned={rng_ok}",
        flush=True,
    )

    if not rng_ok:
        raise RuntimeError(
            "RNG stream misaligned: draw_scar_fields no longer consumes the "
            "same random numbers as get_dis_scar_ham + get_scar_H1. The scar "
            "and qubit models would see different disorder."
        )

    if max(dH, dH1, dw) > atol:
        raise RuntimeError(
            "Fast kernels disagree with quantumScarFunctions.py "
            f"(max deviation {max(dH, dH1, dw):.3e}). quantumScarFunctions.py "
            "has probably been refactored - re-derive the fast path before "
            "trusting this run."
        )

    # extremal eigensolve against the dense answer
    ev_dense, es_dense = H_ref.eigenstates()
    gs_fast, band_fast = ground_state_and_bandwidth(H_ref, precomp["D"])
    band_dense = float(ev_dense[-1] - ev_dense[0])
    overlap = float(abs(es_dense[0].overlap(gs_fast)) ** 2)

    print(
        f"self-check N={N}: bandwidth dense={band_dense:.12f} "
        f"sparse={band_fast:.12f} |<gs_dense|gs_sparse>|^2={overlap:.12f}",
        flush=True,
    )

    if abs(band_dense - band_fast) > 1.0e-6 * max(1.0, abs(band_dense)):
        raise RuntimeError("Sparse bandwidth disagrees with the dense one.")

    if overlap < 1.0 - 1.0e-8:
        raise RuntimeError("Sparse ground state disagrees with the dense one.")


def vn_entropy_from_state(state, basisList, N):
    """
    Half-chain von Neumann entropy for a state in the constrained scar basis.

    Uses C_AB from quantumScarFunctions.py.
    lambdas = sigma^2 are Schmidt eigenvalues.
    """
    C_AB = get_C_AB_matrix(state, basisList, N)
    sigmas = np.linalg.svd(C_AB, compute_uv=False)
    lambdas = sigmas**2
    lambdas = lambdas[lambdas > 1.0e-12]

    if len(lambdas) == 0:
        return 0.0

    return float(-np.sum(lambdas * np.log(lambdas)))


def get_scar_subspace_cached(N):
    """
    Build the clean scar subspace once per N per worker process.

    This uses the same idea as your plotting helper:
    - take one high-Z2-overlap scar state from each energy section
    - replace the middle scar with get_zero_scar(N)

    Returns:
        scar_states: list of Qobj states in the constrained basis
        scar_indices: integer eigenstate indices selected from clean H0
        scar_energies: clean H0 energies at those indices
        z2_overlap_zero_scar: |<Z2|zero_scar>|^2
    """
    N = validate_N(N)

    if N in _SCAR_SUBSPACE_CACHE:
        return _SCAR_SUBSPACE_CACHE[N]

    print(f"Building clean scar subspace once for N={N}", flush=True)

    H0_clean, eigenvalues, eigenstates, psi0, basisList = get_clean_scar_cached(N)
    zero_scar, z2_overlap_zero_scar = get_zero_scar(N)

    sections = np.linspace(eigenvalues[0] - 0.5, eigenvalues[-1] + 0.5, N + 2)

    scar_indices = []

    for section_index in range(len(sections) - 1):
        eigen_section = []

        for k in range(len(eigenvalues)):
            if eigenvalues[k] > sections[section_index] and eigenvalues[k] < sections[section_index + 1]:
                eigen_section.append(k)

        if len(eigen_section) == 0:
            continue

        best_index = eigen_section[0]
        best_overlap = abs(psi0.overlap(eigenstates[best_index]))**2

        for candidate_index in eigen_section[1:]:
            overlap = abs(psi0.overlap(eigenstates[candidate_index]))**2
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = candidate_index

        scar_indices.append(best_index)

    scar_states = [eigenstates[i] for i in scar_indices]

    if len(scar_states) > 0:
        middle_index = len(scar_states) // 2
        scar_states[middle_index] = zero_scar

    scar_energies = np.array([float(eigenvalues[i]) for i in scar_indices], dtype=float)

    cached = {
        "scar_states": scar_states,
        "scar_indices": np.array(scar_indices, dtype=int),
        "scar_energies": scar_energies,
        "z2_overlap_zero_scar": float(z2_overlap_zero_scar),
    }

    _SCAR_SUBSPACE_CACHE[N] = cached
    return cached


# ============================================================
# Scar simulation
# ============================================================

def simulate_scar(N, x, y, z, ds, dd, seed, save_states=False):
    """
    Simulates the driven disordered scar system.

    If save_states=False, only Rtau_scar is returned.
    If save_states=True, the time-evolved states are also returned so final-only
    diagnostics like VN entropy and scar-subspace probabilities can be computed.
    """
    N = validate_N(N)
    np.random.seed(int(seed))

    H0_clean, eigenvalues, eigenstates, psi0, basisList = get_clean_scar_cached(N)
    precomp = get_precomp_cached(N)

    # Same fields, same RNG order as get_dis_scar_ham + get_scar_H1.
    hz, hy, hx, drive_weights = draw_scar_fields(N, [z, y, x], dd)

    H0_dis = qt.Qobj(
        build_dis_H_fast(precomp, hz, hy, hx, z != 0.0, y != 0.0, x != 0.0)
    )
    H1 = qt.Qobj(build_H1_fast(precomp, drive_weights))

    # Only the ground state and the spectral width are used, so do not pay for
    # the full spectrum here.
    initial_state, bandwidth = ground_state_and_bandwidth(H0_dis, precomp["D"])

    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError(f"Bad scar bandwidth: {bandwidth}")

    args = {"A": ds, "omega": wd}
    H = qt.QobjEvo([H0_dis, [H1, coeff]], args=args)

    if save_states:
        sol = qt.sesolve(
            H,
            initial_state,
            tlist,
            e_ops=[H0_dis],
            options={"store_states": True},
        )
    else:
        sol = qt.sesolve(
            H,
            initial_state,
            tlist,
            e_ops=[H0_dis],
        )

    energy = np.array(np.real(sol.expect[0]), dtype=float)
    Rtau_scar = (energy - energy[0]) / bandwidth

    result = {
        "Rtau_scar": Rtau_scar,
        "basisList": basisList,
    }

    if save_states:
        result["states"] = sol.states

    return result


# ============================================================
# Decoupled qubit simulation
# ============================================================

def simulate_decoupled_qubit_rtau(N, x, y, z, ds, dd, seed):
    """
    Simulates the decoupled-qubit comparison model.

    get_qubit_ham returns lists of single-qubit H0_i and H1_i.
    We evolve each qubit independently, sum absorbed energy, then divide by
    the total qubit bandwidth.

    Here wm is a global constant. The decoupled qubits use wm both as their
    bare frequency and as their drive frequency.
    """
    N = validate_N(N)
    np.random.seed(int(seed))

    # get_qubit_ham draws hz, hy, hx and then the drive weights, in that order,
    # exactly like the scar path. Drawing them here keeps that pairing while
    # letting us assemble the blocks ourselves.
    hz, hy, hx, drive_weights = draw_scar_fields(N, [z, y, x], dd)

    H0_blk, H1_blk, psi0_blk, total_bandwidth = build_qubit_block(
        N, hz, hy, hx, drive_weights
    )

    if total_bandwidth <= 0 or not np.isfinite(total_bandwidth):
        raise ValueError(f"Bad total qubit bandwidth: {total_bandwidth}")

    args = {"A": ds, "omega": wm}
    qH = qt.QobjEvo([H0_blk, [H1_blk, coeff]], args=args)
    sol = qt.sesolve(qH, psi0_blk, tlist, e_ops=[H0_blk])

    # <H0_blk> is the average over blocks, so multiply back up by N.
    energy = np.array(np.real(sol.expect[0]), dtype=float)
    total_delta_energy = N * (energy - energy[0])

    Rtau_qubit = total_delta_energy / total_bandwidth
    return Rtau_qubit


def simulate_decoupled_qubit_rtau_reference(N, x, y, z, ds, dd, seed):
    """
    The original one-sesolve-per-qubit implementation, kept so that
    check_qubit_block() can prove the block-diagonal version agrees.
    Not used in the search.
    """
    N = validate_N(N)
    np.random.seed(int(seed))

    qH0_list, qH1_list, drive_weights = get_qubit_ham(
        N,
        wm=wm,
        ham_disorder=[z, y, x],
        ds_dis=dd,
    )

    args = {"A": ds, "omega": wm}

    total_delta_energy = np.zeros(len(tlist), dtype=float)
    total_bandwidth = 0.0

    for qH0, qH1 in zip(qH0_list, qH1_list):
        eigenvalues, eigenstates = qH0.eigenstates()
        bandwidth = float(eigenvalues[-1] - eigenvalues[0])

        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError(f"Bad qubit bandwidth: {bandwidth}")

        total_bandwidth += bandwidth

        initial_state = eigenstates[0]
        qH = qt.QobjEvo([qH0, [qH1, coeff]], args=args)
        sol = qt.sesolve(qH, initial_state, tlist, e_ops=[qH0])

        energy = np.array(np.real(sol.expect[0]), dtype=float)
        total_delta_energy += energy - energy[0]

    return total_delta_energy / total_bandwidth


def check_qubit_block(N, atol=1.0e-6):
    """
    Block-diagonal qubit solve vs the original per-qubit loop, same seed.
    Also catches a future change to get_qubit_ham's return signature, which is
    exactly what broke this script last time.
    """
    N = validate_N(N)
    x, y, z, ds, dd = 0.37, 0.12, 0.44, 1.7, 0.9

    t0 = time.perf_counter()
    R_ref = simulate_decoupled_qubit_rtau_reference(N, x, y, z, ds, dd, seed=7)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    R_fast = simulate_decoupled_qubit_rtau(N, x, y, z, ds, dd, seed=7)
    t_fast = time.perf_counter() - t0

    dev = float(np.max(np.abs(R_ref - R_fast)))

    print(
        f"self-check N={N}: qubit block vs loop max|dRtau|={dev:.3e} "
        f"(max Rtau {R_ref.max():.9f} vs {R_fast.max():.9f}) "
        f"{t_ref:.2f}s -> {t_fast:.2f}s",
        flush=True,
    )

    if dev > atol:
        raise RuntimeError(
            f"Block-diagonal qubit solve disagrees with the per-qubit loop "
            f"by {dev:.3e}."
        )


# ============================================================
# Final-only scar diagnostics
# ============================================================

def compute_final_scar_diagnostics(states, basisList, N):
    """
    Computes diagnostics that are NOT used during DE optimization.

    Returns:
        vn_entropy: shape (time,)
        scar_probs: shape (num_scar_states, time)
        scar_subspace_prob: shape (time,)
    """
    N = validate_N(N)
    scar_data = get_scar_subspace_cached(N)
    scar_states = scar_data["scar_states"]

    vn_entropy = np.zeros(len(states), dtype=float)
    scar_probs = np.zeros((len(scar_states), len(states)), dtype=float)

    for t_index, state in enumerate(states):
        vn_entropy[t_index] = vn_entropy_from_state(state, basisList, N)

        for s_index, scar_state in enumerate(scar_states):
            scar_probs[s_index, t_index] = abs(scar_state.overlap(state))**2

    scar_subspace_prob = np.sum(scar_probs, axis=0)

    return {
        "vn_entropy": vn_entropy,
        "scar_probs": scar_probs,
        "scar_subspace_prob": scar_subspace_prob,
        "scar_indices": scar_data["scar_indices"],
        "scar_energies": scar_data["scar_energies"],
        "z2_overlap_zero_scar": scar_data["z2_overlap_zero_scar"],
    }


# ============================================================
# Scoring and evaluation
# ============================================================

def evaluate_parameters(N, x, y, z, ds, dd, seeds, save_curves=False, save_diagnostics=False):
    """
    Optimization score:

        score = mean_seed[ max(Rtau_scar) - max(Rtau_decoupled_qubits) ]

    score > 0 means scar beats decoupled qubits.

    Diagnostics are final-only and not used by DE unless save_diagnostics=True.
    """
    N = validate_N(N)

    seed_scores = []

    scar_curves = []
    qubit_curves = []

    vn_curves = []
    scar_prob_arrays = []
    scar_subspace_curves = []

    scar_indices = None
    scar_energies = None
    z2_overlap_zero_scar = None

    for seed in seeds:
        scar = simulate_scar(
            N=N,
            x=x,
            y=y,
            z=z,
            ds=ds,
            dd=dd,
            seed=seed,
            save_states=save_diagnostics,
        )

        Rtau_scar = scar["Rtau_scar"]

        Rtau_qubit = simulate_decoupled_qubit_rtau(
            N=N,
            x=x,
            y=y,
            z=z,
            ds=ds,
            dd=dd,
            seed=seed,
        )

        max_scar = float(np.max(Rtau_scar))
        max_qubit = float(np.max(Rtau_qubit))
        seed_score = max_scar - max_qubit

        seed_scores.append(seed_score)

        if save_curves or save_diagnostics:
            scar_curves.append(Rtau_scar)
            qubit_curves.append(Rtau_qubit)

        if save_diagnostics:
            diagnostics = compute_final_scar_diagnostics(
                states=scar["states"],
                basisList=scar["basisList"],
                N=N,
            )

            vn_curves.append(diagnostics["vn_entropy"])
            scar_prob_arrays.append(diagnostics["scar_probs"])
            scar_subspace_curves.append(diagnostics["scar_subspace_prob"])

            scar_indices = diagnostics["scar_indices"]
            scar_energies = diagnostics["scar_energies"]
            z2_overlap_zero_scar = diagnostics["z2_overlap_zero_scar"]

    seed_scores = np.array(seed_scores, dtype=float)

    result = {
        "score": float(np.mean(seed_scores)),
        "score_std": float(np.std(seed_scores)),
        "seed_scores": seed_scores,
        "max_seed_score": float(np.max(seed_scores)),
        "min_seed_score": float(np.min(seed_scores)),
    }

    if save_curves or save_diagnostics:
        scar_curves = np.array(scar_curves, dtype=float)
        qubit_curves = np.array(qubit_curves, dtype=float)

        result.update(
            {
                "Rtau_scar_mean": np.mean(scar_curves, axis=0),
                "Rtau_scar_std": np.std(scar_curves, axis=0),
                "Rtau_qubit_mean": np.mean(qubit_curves, axis=0),
                "Rtau_qubit_std": np.std(qubit_curves, axis=0),
                "Rtau_scar_all": scar_curves,
                "Rtau_qubit_all": qubit_curves,
                "max_Rtau_scar_mean": float(np.max(np.mean(scar_curves, axis=0))),
                "max_Rtau_qubit_mean": float(np.max(np.mean(qubit_curves, axis=0))),
            }
        )

    if save_diagnostics:
        vn_curves = np.array(vn_curves, dtype=float)
        scar_prob_arrays = np.array(scar_prob_arrays, dtype=float)
        scar_subspace_curves = np.array(scar_subspace_curves, dtype=float)

        # scar_prob_arrays shape:
        #     (num_seeds, num_scar_states, num_time_points)
        result.update(
            {
                "vn_entropy_mean": np.mean(vn_curves, axis=0),
                "vn_entropy_std": np.std(vn_curves, axis=0),
                "vn_entropy_all": vn_curves,
                "scar_probs_all": scar_prob_arrays,
                "scar_probs_mean": np.mean(scar_prob_arrays, axis=0),
                "scar_probs_std": np.std(scar_prob_arrays, axis=0),
                "scar_subspace_prob_mean": np.mean(scar_subspace_curves, axis=0),
                "scar_subspace_prob_std": np.std(scar_subspace_curves, axis=0),
                "scar_subspace_prob_all": scar_subspace_curves,
                "scar_indices": scar_indices,
                "scar_energies": scar_energies,
                "z2_overlap_zero_scar": float(z2_overlap_zero_scar),
            }
        )

    return result


# ============================================================
# Picklable scipy objective
# ============================================================

def differential_evolution_objective(vec):
    global _DE_N, _DE_SEEDS, _DE_LOG_EVERY_EVAL, _DE_EVALS_DONE

    N = validate_N(_DE_N)
    seeds = list(_DE_SEEDS)
    x, y, z, ds, dd = unpack_de_vector(vec)

    eval_key = make_eval_hash(N, x, y, z, ds, dd, seeds)

    # The objective is deterministic (fixed seeds), so an exact repeat can be
    # answered from memory. This is a memo, not a filesystem cache: DE almost
    # never revisits an exact point, which is why the old one-npz-per-eval
    # cache cost thousands of files and essentially never hit.
    if eval_key in _DE_EVAL_MEMO:
        return -_DE_EVAL_MEMO[eval_key]

    start_time = time.perf_counter()

    try:
        result = evaluate_parameters(
            N=N,
            x=x,
            y=y,
            z=z,
            ds=ds,
            dd=dd,
            seeds=seeds,
            save_curves=False,
            save_diagnostics=False,
        )

        score = float(result["score"])
        elapsed = time.perf_counter() - start_time

        if not np.isfinite(score):
            return 1.0e9

        _DE_EVAL_MEMO[eval_key] = score
        _DE_EVALS_DONE += 1

        log_eval(
            {
                "key": eval_key,
                "N": int(N),
                "x": x, "y": y, "z": z, "ds": ds, "dd": dd,
                "wd": float(wd), "wm": float(wm),
                "seeds": [int(s) for s in seeds],
                "score": score,
                "score_std": float(result["score_std"]),
                "seed_scores": [float(s) for s in result["seed_scores"]],
                "elapsed_seconds": float(elapsed),
                "pid": os.getpid(),
            }
        )

        if _DE_LOG_EVERY_EVAL:
            print(
                f"score={score:+.8e} "
                f"N={N} "
                f"x={x:.4e} y={y:.4e} z={z:.4e} "
                f"ds={ds:.4e} dd={dd:.4e} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

        # scipy minimizes, so return negative score.
        return -score

    except Exception as exc:
        print(
            f"FAILED N={N} x={x:.4e} y={y:.4e} z={z:.4e} ds={ds:.4e} dd={dd:.4e}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()

        # Fail fast. A returned 1e9 is meant to survive one bad integrator step,
        # not to keep a whole allocation running against a broken model. If the
        # very first evaluation in this process dies, or the failure is a
        # structural one (wrong signature, wrong types), stop the run.
        if _DE_EVALS_DONE == 0 or isinstance(exc, (TypeError, NameError, AttributeError, ImportError)):
            raise

        _DE_EVALS_DONE += 1
        return 1.0e9


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Differential evolution search for parameters where max scar Rtau beats max decoupled-qubit Rtau."
    )

    parser.add_argument("--N", type=int, default=4, help="Even system size. Must be >= 4.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)

    # Local-run controls.
    # Keep workers=1 on a laptop/desktop unless you know multiprocessing works
    # with your local Python setup.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--de-seed", type=int, default=0)

    # Seeds.
    # objective-reals should be small because it multiplies DE cost.
    parser.add_argument("--objective-reals", type=int, default=1)
    parser.add_argument("--final-reals", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)

    # Differential evolution controls.
    # Defaults are intentionally small for a test run.
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--popsize", type=int, default=3)
    parser.add_argument("--tol", type=float, default=0.01)
    parser.add_argument("--polish", action="store_true")

    # Bounds. x/y/z are log10 bounds.
    parser.add_argument("--logx-min", type=float, default=-3.0)
    parser.add_argument("--logx-max", type=float, default=0.0)
    parser.add_argument("--logy-min", type=float, default=-3.0)
    parser.add_argument("--logy-max", type=float, default=0.0)
    parser.add_argument("--logz-min", type=float, default=-3.0)
    parser.add_argument("--logz-max", type=float, default=0.0)
    parser.add_argument("--ds-min", type=float, default=0.1)
    parser.add_argument("--ds-max", type=float, default=5.0)
    parser.add_argument("--dd-min", type=float, default=0.01)
    parser.add_argument("--dd-max", type=float, default=5.0)

    parser.add_argument("--quiet-evals", action="store_true")

    # Physics knobs that used to be hard-coded module globals.
    # The measured clean scar level spacing is 0.635555 and is N-independent
    # over N = 8..16, so the default is right to 0.2% at every N you run.
    parser.add_argument("--wd", type=float, default=wd,
                        help="Scar drive frequency.")
    parser.add_argument("--t-max", type=float, default=t_max)
    parser.add_argument("--nt", type=int, default=len(tlist),
                        help="Number of time points on [0, t_max].")

    # Safety / debug.
    parser.add_argument("--skip-self-check", action="store_true",
                        help="Skip the startup check that the fast kernels still "
                             "reproduce quantumScarFunctions.py. Do not use this.")
    parser.add_argument("--dense-eig", action="store_true",
                        help="Force dense diagonalisation instead of the extremal "
                             "sparse solve. Debug only, much slower for N >= 14.")

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    global _DE_N, _DE_SEEDS, _DE_LOG_EVERY_EVAL
    global _DE_DENSE_EIG, _DE_LOG_DIR
    global wd, t_max, tlist

    args = parse_args()

    N = validate_N(args.N)

    # Set the physics globals BEFORE the worker pool is forked so every worker
    # inherits them.
    wd = float(args.wd)
    t_max = float(args.t_max)
    tlist = np.linspace(0.0, t_max, int(args.nt))
    _DE_DENSE_EIG = bool(args.dense_eig)

    # ------------------------------------------------------------
    # SLURM mode vs local mode
    # ------------------------------------------------------------
    slurm_run = "SLURM_ARRAY_TASK_ID" in os.environ

    if slurm_run:
        array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        run_label = f"island_{array_id}"
        run_mode = "slurm"
    else:
        array_id = int(args.de_seed)
        num_cpus = max(1, int(args.workers))
        run_label = f"local_seed_{array_id}"
        run_mode = "local"

    outdir = Path(args.outdir)
    run_dir = outdir / "de_scar_beats_qubit" / f"N{N}" / run_label
    evals_dir = run_dir / "evals"
    final_dir = run_dir / "final"

    evals_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Same seed rule:
    # For a given DE candidate, scar and qubits both get the same seed.
    objective_seeds = list(range(args.seed_offset, args.seed_offset + args.objective_reals))
    final_seeds = list(range(args.seed_offset, args.seed_offset + args.final_reals))

    _DE_N = N
    _DE_SEEDS = objective_seeds
    _DE_LOG_DIR = str(evals_dir)
    _DE_LOG_EVERY_EVAL = not args.quiet_evals

    bounds = [
        (args.logx_min, args.logx_max),
        (args.logy_min, args.logy_max),
        (args.logz_min, args.logz_max),
        (args.ds_min, args.ds_max),
        (args.dd_min, args.dd_max),
    ]

    print("Starting differential evolution search", flush=True)
    print(f"Run mode: {run_mode}", flush=True)
    print(f"Island / DE seed: {array_id}", flush=True)
    print(f"Workers / CPUs: {num_cpus}", flush=True)
    print(f"N: {N}", flush=True)
    print(f"Qubit count: {N}", flush=True)
    print(f"Objective seeds: {objective_seeds}", flush=True)
    print(f"Final seeds: {final_seeds}", flush=True)
    print("Same-seed pairing: scar and qubit comparison use the same seed inside each evaluation", flush=True)
    print(f"Scar drive frequency wd: {wd}", flush=True)
    print(f"Qubit bare and drive frequency wm: {wm}", flush=True)
    print(f"Bounds: {bounds}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Time grid: {len(tlist)} points on [0, {t_max}]", flush=True)

    # ------------------------------------------------------------
    # Build the caches HERE, in the parent, before differential_evolution
    # forks its worker pool. On Linux the workers then share these pages
    # copy-on-write instead of each rebuilding and re-diagonalising the clean
    # Hamiltonian.
    # ------------------------------------------------------------
    build_start = time.perf_counter()
    get_clean_scar_cached(N)
    get_precomp_cached(N)
    print(f"Clean Hamiltonian + basis structure built in "
          f"{time.perf_counter() - build_start:.2f}s", flush=True)

    if not args.skip_self_check:
        check_start = time.perf_counter()
        check_fast_kernels(N)
        check_qubit_block(N)
        print(f"Self-check passed in {time.perf_counter() - check_start:.2f}s",
              flush=True)

    start_time = time.perf_counter()

    de_result = differential_evolution(
        differential_evolution_objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        seed=array_id,
        workers=num_cpus,
        updating="deferred",
        polish=args.polish,
        disp=True,
    )

    de_elapsed = time.perf_counter() - start_time

    best_x, best_y, best_z, best_ds, best_dd = unpack_de_vector(de_result.x)
    objective_score = -float(de_result.fun)

    print("\nBest objective result", flush=True)
    print(f"objective_score = {objective_score:+.10e}", flush=True)
    print(f"x = {best_x:.16e}", flush=True)
    print(f"y = {best_y:.16e}", flush=True)
    print(f"z = {best_z:.16e}", flush=True)
    print(f"ds = {best_ds:.16e}", flush=True)
    print(f"dd = {best_dd:.16e}", flush=True)

    print("\nRunning final evaluation on best point with curves + VN + scar-subspace diagnostics...", flush=True)

    final_result = evaluate_parameters(
        N=N,
        x=best_x,
        y=best_y,
        z=best_z,
        ds=best_ds,
        dd=best_dd,
        seeds=final_seeds,
        save_curves=True,
        save_diagnostics=True,
    )

    final_score = float(final_result["score"])
    success = bool(final_score > 0.0)

    tag = (
        f"island{array_id}_N{N}"
        f"_x{safe_float_name(best_x)}"
        f"_y{safe_float_name(best_y)}"
        f"_z{safe_float_name(best_z)}"
        f"_ds{safe_float_name(best_ds)}"
        f"_dd{safe_float_name(best_dd)}"
    )

    final_npz_path = final_dir / f"{tag}_result.npz"
    summary_path = final_dir / f"{tag}_summary.json"

    atomic_savez(
        final_npz_path,
        tlist=tlist,
        N=int(N),
        qubit_count=int(N),
        wd=float(wd),
        wm=float(wm),
        t_max=float(t_max),
        x=float(best_x),
        y=float(best_y),
        z=float(best_z),
        ds=float(best_ds),
        dd=float(best_dd),
        objective_seeds=np.array(objective_seeds, dtype=int),
        final_seeds=np.array(final_seeds, dtype=int),
        objective_score=float(objective_score),
        final_score=float(final_score),
        final_score_std=float(final_result["score_std"]),
        seed_scores=final_result["seed_scores"],
        Rtau_scar_mean=final_result["Rtau_scar_mean"],
        Rtau_scar_std=final_result["Rtau_scar_std"],
        Rtau_qubit_mean=final_result["Rtau_qubit_mean"],
        Rtau_qubit_std=final_result["Rtau_qubit_std"],
        Rtau_scar_all=final_result["Rtau_scar_all"],
        Rtau_qubit_all=final_result["Rtau_qubit_all"],
        max_Rtau_scar_mean=float(final_result["max_Rtau_scar_mean"]),
        max_Rtau_qubit_mean=float(final_result["max_Rtau_qubit_mean"]),
        vn_entropy_mean=final_result["vn_entropy_mean"],
        vn_entropy_std=final_result["vn_entropy_std"],
        vn_entropy_all=final_result["vn_entropy_all"],
        scar_probs_all=final_result["scar_probs_all"],
        scar_probs_mean=final_result["scar_probs_mean"],
        scar_probs_std=final_result["scar_probs_std"],
        scar_subspace_prob_mean=final_result["scar_subspace_prob_mean"],
        scar_subspace_prob_std=final_result["scar_subspace_prob_std"],
        scar_subspace_prob_all=final_result["scar_subspace_prob_all"],
        scar_indices=final_result["scar_indices"],
        scar_energies=final_result["scar_energies"],
        z2_overlap_zero_scar=float(final_result["z2_overlap_zero_scar"]),
        success=np.array(success),
        de_elapsed_seconds=float(de_elapsed),
    )

    summary = {
        "success": success,
        "meaning": "success means final_score = mean_seed[max(Rtau_scar) - max(Rtau_decoupled_qubits)] > 0",
        "run_mode": run_mode,
        "array_id": array_id if slurm_run else None,
        "de_seed": array_id,
        "workers": num_cpus,
        "N": int(N),
        "qubit_count": int(N),
        "wd": float(wd),
        "wm": float(wm),
        "t_max": float(t_max),
        "num_t_points": int(len(tlist)),
        "x": float(best_x),
        "y": float(best_y),
        "z": float(best_z),
        "ds": float(best_ds),
        "dd": float(best_dd),
        "objective_score": float(objective_score),
        "final_score": float(final_score),
        "final_score_std": float(final_result["score_std"]),
        "max_Rtau_scar_mean": float(final_result["max_Rtau_scar_mean"]),
        "max_Rtau_qubit_mean": float(final_result["max_Rtau_qubit_mean"]),
        "objective_reals": int(args.objective_reals),
        "final_reals": int(args.final_reals),
        "seed_offset": int(args.seed_offset),
        "objective_seeds": objective_seeds,
        "final_seeds": final_seeds,
        "same_seed_pairing": True,
        "maxiter": int(args.maxiter),
        "popsize": int(args.popsize),
        "tol": float(args.tol),
        "polish": bool(args.polish),
        "de_elapsed_seconds": float(de_elapsed),
        "npz_path": str(final_npz_path),
        "scar_probs_all_shape": list(final_result["scar_probs_all"].shape),
        "scar_probs_all_shape_meaning": "num_final_seeds, num_scar_states, num_time_points",
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nFinal result", flush=True)
    print(f"success = {success}", flush=True)
    print(f"final_score = {final_score:+.10e}", flush=True)
    print(f"max scar Rtau mean = {final_result['max_Rtau_scar_mean']:.10e}", flush=True)
    print(f"max qubit Rtau mean = {final_result['max_Rtau_qubit_mean']:.10e}", flush=True)
    print(f"scar_probs_all shape = {final_result['scar_probs_all'].shape}", flush=True)
    print(f"Saved npz: {final_npz_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()