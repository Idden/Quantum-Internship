from quantumScarFunctions import *
import numpy as np
import qutip as qt
import scipy.sparse as sp

# =========================================================================
# Density-matrix helpers for the open-system (bath) section.
# Energy/battery definitions follow Campaioli et al., "Colloquium: Quantum
# batteries", Rev. Mod. Phys. 96, 031001 (2024):
#   Eq. 3  W(t)  = Tr[H0 rho(t)] - Tr[H0 rho0]            (stored energy)
#   Eq. 4  E(rho) = Tr[H0 rho] - min_U Tr[H0 U rho U^dag]  (ergotropy)
#   Eq. 5  passive state: rho's eigenvalues sorted decreasing, placed on
#          H0's eigenstates sorted increasing
#   Eq. 11 <P>_tau = W(tau) / tau                           (average power)
# =========================================================================


# -------------------------------------------------------------------------
# basic density-matrix measures
# -------------------------------------------------------------------------

def to_dm(state):
    """Ket -> density matrix. A density matrix passes through unchanged."""
    return state if state.isoper else qt.ket2dm(state)


def vn_from_rho(rho):
    """Von Neumann entropy S = -Tr[rho ln rho] of the WHOLE state.
    Zero for a pure state. Measures bath-induced mixedness, not entanglement."""
    return qt.entropy_vn(to_dm(rho))


def purity_from_rho(rho):
    """Tr[rho^2]. 1 for a pure state, 1/L for the maximally mixed state."""
    rho = to_dm(rho)
    return float(np.real((rho * rho).tr()))


def maximally_mixed(L):
    """Identity / L. Late-time reference state for pure dephasing."""
    return qt.qeye(L) / L


def check_rho(rho, tol=1e-8):
    """Sanity check: trace = 1, Hermitian, no negative eigenvalues."""
    rho = to_dm(rho)
    return {
        "trace": float(np.real(rho.tr())),
        "hermitian": bool((rho - rho.dag()).norm() < tol),
        "min_eig": float(np.min(np.real(rho.eigenenergies()))),
    }


# -------------------------------------------------------------------------
# passive state and ergotropy
# -------------------------------------------------------------------------

def calculate_pstate(rho, H, H_eig=None):
    """Passive state (Eq. 5) and its energy.

    rho's eigenvalues s_k sorted DEcreasing are placed on H's eigenstates
    sorted INcreasing (qutip returns eigenstates ascending already).
    H_eig: optional output of H.eigenstates(), so H is not rediagonalized
    at every time step.
    Returns (pstate, penergy) with penergy = Tr[H pstate].
    """
    rho = to_dm(rho)
    if H_eig is None:
        H_eig = H.eigenstates()
    evals, evecs = H_eig

    s = np.sort(np.real(rho.eigenenergies()))[::-1]

    pstate = sum(s[k] * evecs[k].proj() for k in range(len(s)))
    penergy = float(np.dot(s, evals))

    return pstate, penergy


def calculate_ergotropy(rho, H, H_evals=None):
    """Ergotropy (Eq. 4): E = Tr[H rho] - Tr[H pstate].

    Only needs eigenvalues, so it skips building the passive state.
    H_evals: optional H.eigenenergies() to avoid rediagonalizing H.
    For a pure state evolved from the ground state, E = W (your R(tau)).
    """
    rho = to_dm(rho)
    if H_evals is None:
        H_evals = H.eigenenergies()

    s = np.sort(np.real(rho.eigenenergies()))[::-1]
    energy = np.real(qt.expect(H, rho))

    return float(energy - np.dot(s, np.sort(H_evals)))


def battery_series(states, H0):
    """Battery quantities along a trajectory rho(t), for the scar chain.

    Energies are divided by the bandwidth of H0, so W matches your R(tau).
    Returns dict of arrays: W, ergotropy, S (VN entropy), purity.
    """
    H_evals = H0.eigenenergies()
    bandwidth = H_evals[-1] - H_evals[0]

    energy = np.real(qt.expect(H0, states))
    W = (energy - energy[0]) / bandwidth
    erg = np.array([calculate_ergotropy(r, H0, H_evals) for r in states]) / bandwidth
    S = np.array([vn_from_rho(r) for r in states])
    pur = np.array([purity_from_rho(r) for r in states])

    return {"W": W, "ergotropy": erg, "S": S, "purity": pur}


# -------------------------------------------------------------------------
# power
# -------------------------------------------------------------------------

def avg_power(W, tlist):
    """Average power (Eq. 11): <P>_tau = W(tau) / tau. Set to 0 at tau = 0.
    Assumes tlist starts at 0 (start of charging)."""
    W = np.asarray(W)
    tlist = np.asarray(tlist)
    P = np.zeros_like(W, dtype=float)
    P[1:] = W[1:] / tlist[1:]
    return P


def inst_power(W, tlist):
    """Instantaneous power P(t) = dW/dt (Eq. 10), by finite differences."""
    return np.gradient(np.asarray(W), np.asarray(tlist))


def max_avg_power(W, tlist):
    """Returns (max <P>, tau at which it happens)."""
    P = avg_power(W, tlist)
    k = np.argmax(P)
    return float(P[k]), float(tlist[k])


# -------------------------------------------------------------------------
# jump operators (collapse operators)
# convention: L = sqrt(gamma) * O, so pure dephasing with O = sigma^z
# makes coherences decay at rate 2*gamma_phi
# -------------------------------------------------------------------------

def get_scar_sigz_ops(N, basisList):
    """sigma^z_r in the constrained basis, r = 0..N-1. '1' -> +1, '0' -> -1
    (same convention as sigzMap in get_scar_ham). Diagonal."""
    ops = []
    for r in range(N):
        diag = [1.0 if s[r] == '1' else -1.0 for s in basisList]
        ops.append(qt.Qobj(sp.diags(diag, format="csr")))
    return ops


def get_scar_sigminus_ops(N, basisList):
    """sigma^-_r in the constrained basis: '1' -> '0' at site r.
    Removing an excitation never breaks the blockade, so this stays
    inside the PXP Hilbert space."""
    L = len(basisList)
    basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}
    ops = []
    for r in range(N):
        rows, cols = [], []
        for i, s in enumerate(basisList):
            if s[r] == '1':
                flipped = s[:r] + '0' + s[r+1:]
                rows.append(basisMap[flipped])
                cols.append(i)
        mat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(L, L))
        ops.append(qt.Qobj(mat))
    return ops


def get_scar_c_ops(N, basisList, gamma_phi=0.0, gamma_1=0.0):
    """Collapse operators for the scar chain.
    gamma_phi: dephasing rate (sigma^z_r on every site)
    gamma_1:   relaxation rate (sigma^-_r on every site)"""
    c_ops = []
    if gamma_phi > 0:
        c_ops += [np.sqrt(gamma_phi) * op for op in get_scar_sigz_ops(N, basisList)]
    if gamma_1 > 0:
        c_ops += [np.sqrt(gamma_1) * op for op in get_scar_sigminus_ops(N, basisList)]
    return c_ops


def get_qubit_c_ops(gamma_phi=0.0, gamma_1=0.0):
    """Collapse operators for ONE decoupled qubit.
    qt.sigmam() maps basis(2,0) (sigma^z = +1) -> basis(2,1) (sigma^z = -1),
    the same direction as the scar '1' -> '0' relaxation."""
    c_ops = []
    if gamma_phi > 0:
        c_ops.append(np.sqrt(gamma_phi) * qt.sigmaz())
    if gamma_1 > 0:
        c_ops.append(np.sqrt(gamma_1) * qt.sigmam())
    return c_ops


# -------------------------------------------------------------------------
# time evolution: charge (drive on) then idle (drive off), bath on for both
# -------------------------------------------------------------------------

def drive_coeff(t, args):
    """A sin(omega t). Old-style (t, args) signature, same as make_coeff."""
    return args["A"] * np.sin(args["omega"] * t)


def evolve_open(H0, H1, c_ops, rho0, args, t_charge, t_idle=0.0,
                n_charge=500, n_idle=500, options=None):
    """Lindblad evolution: charge for t_charge, then idle for t_idle.

    Two separate mesolve calls, so the drive switches off exactly at
    t_charge (the ODE solver never sees a discontinuous coefficient).

    H1: a single Qobj (global drive, args needs "A", "omega"), or a list
        of per-site Qobjs from get_scar_H1(..., indv_qubit=True) for
        frequency disorder (args needs "A" and "wd0", "wd1", ...).
    Returns (tlist, states).
    """
    opts = {"store_states": True}
    if options is not None:
        opts.update(options)

    rho0 = to_dm(rho0)

    if isinstance(H1, list):
        H = qt.QobjEvo([H0] + [[H1[r], make_coeff(r)] for r in range(len(H1))], args=args)
    else:
        H = qt.QobjEvo([H0, [H1, drive_coeff]], args=args)

    t_c = np.linspace(0, t_charge, n_charge)
    res_c = qt.mesolve(H, rho0, t_c, c_ops=c_ops, options=opts)

    tlist = list(t_c)
    states = list(res_c.states)

    if t_idle > 0:
        t_i = np.linspace(t_charge, t_charge + t_idle, n_idle)
        res_i = qt.mesolve(H0, res_c.states[-1], t_i, c_ops=c_ops, options=opts)
        tlist += list(t_i[1:])        # first idle point duplicates the last charge point
        states += list(res_i.states[1:])

    return np.array(tlist), states


def evolve_qubits_open(qH0_list, qH1_list, c_ops, args, t_charge, t_idle=0.0,
                       n_charge=500, n_idle=500, args_list=None, options=None):
    """Decoupled qubits: no interactions + local noise means rho stays a
    product state, so each qubit is evolved on its own (2x2 matrices).

    c_ops: single-qubit collapse operators from get_qubit_c_ops.
    args_list: optional per-qubit args (e.g. a different "omega" per qubit
               for frequency disorder). Otherwise every qubit uses args.
    Returns (tlist, rho_list) with rho_list[i][t] = state of qubit i.
    """
    rho_list = []
    tlist = None
    for i, (qH0, qH1) in enumerate(zip(qH0_list, qH1_list)):
        a = args_list[i] if args_list is not None else args
        ground = qH0.eigenstates()[1][0]
        tlist, states = evolve_open(qH0, qH1, c_ops, ground, a, t_charge, t_idle,
                                    n_charge, n_idle, options)
        rho_list.append(states)
    return tlist, rho_list


# -------------------------------------------------------------------------
# decoupled-qubit battery quantities (global, without building 2^N)
# -------------------------------------------------------------------------

def qubit_global_ergotropy(rhos, qH0_list):
    """GLOBAL ergotropy of rho_1 x ... x rho_N with H0 = sum_i h_i.

    Eigenvalues of a tensor product = all products of single-qubit
    eigenvalues; eigenvalues of a sum of local terms = all sums. So the
    2^N spectra are built with kron / outer-sum instead of a 2^N matrix.
    Global ergotropy >= sum of local ergotropies for mixed states.
    """
    s = np.array([1.0])
    e = np.array([0.0])
    energy = 0.0
    for rho, h in zip(rhos, qH0_list):
        s = np.kron(s, np.real(rho.eigenenergies()))
        e = np.add.outer(e, h.eigenenergies()).ravel()
        energy += np.real(qt.expect(h, rho))
    return float(energy - np.dot(np.sort(s)[::-1], np.sort(e)))


def qubit_battery_series(rho_list, qH0_list):
    """Same outputs as battery_series, for the decoupled qubits.

    Normalized by the total bandwidth sum_i (e_max,i - e_min,i), matching
    Rtau_plot_qubit. Also returns the sum of LOCAL ergotropies for
    comparison with the global one.
    """
    bandwidth = sum(h.eigenenergies()[-1] - h.eigenenergies()[0] for h in qH0_list)
    nt = len(rho_list[0])

    energy = np.zeros(nt)
    erg_global = np.zeros(nt)
    erg_local = np.zeros(nt)
    S = np.zeros(nt)
    pur = np.ones(nt)

    for t in range(nt):
        rhos_t = [rho_list[i][t] for i in range(len(qH0_list))]
        erg_global[t] = qubit_global_ergotropy(rhos_t, qH0_list)
        for rho, h in zip(rhos_t, qH0_list):
            energy[t] += np.real(qt.expect(h, rho))
            erg_local[t] += calculate_ergotropy(rho, h)
            S[t] += vn_from_rho(rho)           # entropy is additive for product states
            pur[t] *= purity_from_rho(rho)     # purity is multiplicative

    return {
        "W": (energy - energy[0]) / bandwidth,
        "ergotropy": erg_global / bandwidth,
        "ergotropy_local": erg_local / bandwidth,
        "S": S,
        "purity": pur,
    }


# -------------------------------------------------------------------------
# scar-specific diagnostics
# -------------------------------------------------------------------------

def half_chain_rho(rho, basisList, N):
    """Reduced state of the left N//2 sites, built straight from the
    constrained basis: rho_A[a, a'] = sum_b rho[(a,b), (a',b)].
    Same A/B split as get_C_AB_matrix."""
    NA = N // 2
    R = to_dm(rho).full()

    groups = {}
    for k, s in enumerate(basisList):
        groups.setdefault(s[NA:], []).append((k, int(s[:NA], 2)))

    rhoA = np.zeros((2**NA, 2**NA), dtype=complex)
    for members in groups.values():
        ks = [m[0] for m in members]
        a = [m[1] for m in members]
        rhoA[np.ix_(a, a)] += R[np.ix_(ks, ks)]

    return qt.Qobj(rhoA)


def half_chain_entropy(rho, basisList, N):
    """S(rho_A). For a pure state this equals your vn_from_states value
    (entanglement entropy). For a mixed state it ALSO counts bath-induced
    mixedness, so it is NOT an entanglement measure there."""
    return qt.entropy_vn(half_chain_rho(rho, basisList, N))


def scar_population(states, scarStates):
    """Total weight in the scar tower: sum_s <s| rho(t) |s>.
    scarStates: the list returned by scar_overlap_from_states."""
    P = sum(s.proj() for s in scarStates)
    return np.real(qt.expect(P, states))
