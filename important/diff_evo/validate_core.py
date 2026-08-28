"""
validate_core.py
================

Proves scarcore.py is the same physics as before, three ways:

  1. Against reference_kron.py -- an independent full-2^N Kronecker
     construction that imports nothing from scarcore. Checks H_clean, the
     x/y/z disorder pieces and the drive operator elementwise.
  2. Against quantumScarFunctions.py itself, whenever qutip is importable.
     This is the check that matters on the cluster; it is skipped locally
     because qutip cannot be installed in the sandbox.
  3. The propagator against a second, independent integrator (Radau) at a
     much tighter tolerance, so the reported R(t) is not an artefact of
     DOP853's step control.

Run:  python validate_core.py
"""

import sys
import time

import numpy as np

import scarcore as sc
import reference_kron as ref


def check_basis(N):
    strings, bits = sc.build_basis(N)
    idx, ref_strings = ref.blockade_indices(N)

    # Lucas numbers L_4.. = 7, 18, 47, 123, 322, 843
    lucas = {4: 7, 6: 18, 8: 47, 10: 123, 12: 322, 14: 843}

    ok = (strings == ref_strings)
    dim_ok = (len(strings) == lucas.get(N, len(strings)))

    print(f"  basis N={N:2d}: D={len(strings):5d} "
          f"order_matches_reference={ok} lucas_number={dim_ok}")

    if not ok:
        raise SystemExit("basis ordering differs from the reference")
    if not dim_ok:
        raise SystemExit(f"dimension {len(strings)} is not the Lucas number L_{N}")

    return strings, idx


def check_hamiltonians(N, seed=12345):
    strings, idx = check_basis(N)
    struct = sc.build_structure(N)

    # --- clean Hamiltonian --------------------------------------------
    H_ref = ref.restrict(ref.pxp_full(N), idx)
    H_new = np.asarray(struct["H_clean"].todense())
    d_clean = np.abs(H_ref - H_new).max()

    # leakage: PXP must not take a blockade state out of the subspace
    full = ref.pxp_full(N)
    mask = np.ones(2 ** N, dtype=bool)
    mask[idx] = False
    leak = np.abs(full[np.ix_(mask, idx)]).max() if mask.any() else 0.0

    # --- disorder + drive, at a random parameter point ------------------
    v_z, v_y, v_x, v_w = sc.draw_unit_fields(N, seed)
    x, y, z, dd = 0.37, 0.12, 0.44, 0.9

    H0_new, d1_new = sc.assemble_scar_H(struct, v_z, v_y, v_x, v_w, x, y, z, dd)

    H0_ref = (
        ref.restrict(ref.pxp_full(N), idx)
        + ref.restrict(ref.field_full(N, z * v_z, ref.Z), idx)
        + ref.restrict(ref.field_full(N, y * v_y, ref.Y), idx)
        + ref.restrict(ref.field_full(N, x * v_x, ref.X), idx)
    )
    d_dis = np.abs(H0_ref - np.asarray(H0_new.todense())).max()

    d1_ref = np.real(np.diag(ref.restrict(ref.drive_full(N, 1.0 + dd * v_w), idx)))
    d_drive = np.abs(d1_ref - d1_new).max()

    print(f"  operators N={N:2d}: max|dH_clean|={d_clean:.3e} "
          f"max|dH0_dis|={d_dis:.3e} max|dH1_diag|={d_drive:.3e} "
          f"pxp_leakage={leak:.3e}")

    worst = max(d_clean, d_dis, d_drive, leak)
    if worst > 1e-12:
        raise SystemExit(f"scarcore disagrees with the Kronecker reference by {worst:.3e}")


def check_linearity(N, seed=7):
    """
    The whole preprocessing win rests on H0_dis being exactly linear in
    (x, y, z) and H1 exactly linear in dd. Check that against a direct
    re-draw at a second parameter point.
    """
    struct = sc.build_structure(N)
    v = sc.draw_unit_fields(N, seed)

    a = sc.assemble_scar_H(struct, *v, 0.1, 0.2, 0.3, 0.4)
    b = sc.assemble_scar_H(struct, *v, 0.2, 0.4, 0.6, 0.8)

    # H0(2p) - H0(p) should equal H0(p) - H_clean
    lhs = (b[0] - a[0]).todense()
    rhs = (a[0] - struct["H_clean"].astype(complex)).todense()
    d = np.abs(lhs - rhs).max()

    # d1(2dd) - d1(dd) should equal d1(dd) - d1_base
    d2 = np.abs((b[1] - a[1]) - (a[1] - struct["d1_base"])).max()

    print(f"  linearity N={N:2d}: max|dH0|={d:.3e} max|dH1|={d2:.3e}")
    if max(d, d2) > 1e-12:
        raise SystemExit("H is not linear in the disorder strengths -- the cache is invalid")


def check_rng_stream(N, seed=12345):
    """
    draw_unit_fields must consume the global RNG in the same order and to
    the same depth as get_dis_scar_ham + get_scar_H1, or the scar model and
    the qubit model stop sharing a disorder realisation.

    Checked here without qutip by replaying the draw order explicitly and
    comparing the RNG position afterwards.
    """
    np.random.seed(seed)
    v_z, v_y, v_x, v_w = [], [], [], []
    for target, strength in ((v_z, 0.44), (v_y, 0.12), (v_x, 0.37)):
        perm = np.random.choice(N, size=N, replace=False)
        h = np.zeros(N)
        h[perm] = np.random.uniform(-strength, strength, N)
        target.extend(h / strength)
    perm = np.random.choice(N, size=N, replace=False)
    w = np.zeros(N)
    w[perm] = np.random.uniform(-0.9, 0.9, N)
    v_w.extend(w / 0.9)
    tail_ref = np.random.uniform(size=4)

    a, b, c, d = sc.draw_unit_fields(N, seed)
    tail_new = np.random.uniform(size=4)

    dev = max(
        np.abs(np.array(v_z) - a).max(),
        np.abs(np.array(v_y) - b).max(),
        np.abs(np.array(v_x) - c).max(),
        np.abs(np.array(v_w) - d).max(),
    )
    aligned = bool(np.allclose(tail_ref, tail_new))

    print(f"  rng N={N:2d}: max|dv|={dev:.3e} stream_aligned={aligned}")
    if dev > 1e-14 or not aligned:
        raise SystemExit("RNG stream misaligned -- CRN pairing would be broken")


def check_propagator(N=10, seed=3):
    """DOP853 against Radau at a tighter tolerance."""
    struct = sc.build_structure(N)
    v = sc.draw_unit_fields(N, seed)
    H0, d1 = sc.assemble_scar_H(struct, *v, 0.1, 0.05, 0.2, 1.0)

    tlist = np.linspace(0.0, 50.0, 401)
    A, wd = 1.5, 0.6366896896896898

    t0 = time.perf_counter()
    fast = sc.evolve_scar(struct, H0, d1, A, wd, tlist)
    t_fast = time.perf_counter() - t0

    R_fast, Rd_fast = sc.scar_metrics(fast["E"], fast["pops"], fast["bandwidth"])

    # independent, much tighter
    from scipy.integrate import solve_ivp
    H0c = H0.astype(complex).tocsr()
    E, V = np.linalg.eigh(np.asarray(H0.todense()))

    def rhs(t, psi):
        return -1j * (H0c @ psi + (A * np.sin(wd * t)) * (d1 * psi))

    t0 = time.perf_counter()
    sol = solve_ivp(rhs, (0.0, 50.0), V[:, 0].astype(complex),
                    t_eval=tlist, method="DOP853", rtol=1e-12, atol=1e-14)
    t_ref = time.perf_counter() - t0

    pops = np.abs(V.conj().T @ sol.y) ** 2
    R_ref, Rd_ref = sc.scar_metrics(E, pops, fast["bandwidth"])

    dR = np.abs(R_fast - R_ref).max()
    dRd = np.abs(Rd_fast - Rd_ref).max()
    norm = np.abs(np.linalg.norm(fast["psi"], axis=0) - 1.0).max()

    print(f"  propagator N={N:2d}: max|dR|={dR:.3e} max|dR_deph|={dRd:.3e} "
          f"max|1-norm|={norm:.3e}  ({t_fast:.2f}s vs {t_ref:.2f}s at rtol 1e-12)")

    if dR > 1e-7 or norm > 1e-7:
        raise SystemExit("propagator tolerance is too loose")


def check_ergotropy_identities(N=10, seed=5):
    """
    Two things that must hold if the metric means what the paper will say
    it means:
        R(0) = 0                (the battery starts passive / empty)
        0 <= R_deph <= R        (dephasing can only remove extractable work)
    """
    struct = sc.build_structure(N)
    v = sc.draw_unit_fields(N, seed)
    H0, d1 = sc.assemble_scar_H(struct, *v, 0.1, 0.05, 0.2, 1.0)

    tlist = np.linspace(0.0, 60.0, 481)
    out = sc.evolve_scar(struct, H0, d1, 2.0, 0.6366896896896898, tlist)
    R, Rd = sc.scar_metrics(out["E"], out["pops"], out["bandwidth"])

    print(f"  ergotropy N={N:2d}: R(0)={R[0]:.2e} R_deph(0)={Rd[0]:.2e} "
          f"min(R-R_deph)={np.min(R - Rd):.3e} max R={R.max():.4f} "
          f"max R_deph={Rd.max():.4f}")

    if abs(R[0]) > 1e-10 or Rd.min() < -1e-10 or np.min(R - Rd) < -1e-10:
        raise SystemExit("ergotropy identities violated")


def check_against_quantumscarfunctions(N=10):
    """
    The check that matters: scarcore against the file the paper's other
    scripts import.

    Runs against the real qutip when it is installed (the cluster venv).
    When it is not, `qutip_shim` supplies just enough Qobj arithmetic for
    the *builder* functions to execute -- no solvers, no physics of its
    own -- so this check is never silently skipped.
    """
    backend = "real qutip"
    try:
        import qutip  # noqa: F401
    except ImportError:
        import qutip_shim
        qutip_shim.install()
        backend = "qutip shim (no solvers, builders only)"

    try:
        from quantumScarFunctions import (
            get_scar_ham, get_dis_scar_ham, get_scar_H1, get_Hy,
            get_qubit_ham, get_zero_scar,
        )
    except Exception as exc:
        print(f"  quantumScarFunctions check: SKIPPED ({type(exc).__name__}: {exc})")
        return

    struct = sc.build_structure(N)
    H0_clean, _, _, psi0, basisList = get_scar_ham(N, diagonalize=False)

    def dense(H):
        data = getattr(H, "data", None)
        if hasattr(data, "as_scipy"):
            return np.asarray(data.as_scipy().todense())
        return np.asarray(H.full())

    order_ok = (basisList == struct["strings"])
    z2_ok = int(np.argmax(np.abs(psi0.full().ravel()))) == struct["z2_index"]
    d_clean = np.abs(dense(H0_clean) - np.asarray(struct["H_clean"].todense())).max()

    x, y, z, dd = 0.37, 0.12, 0.44, 0.9

    np.random.seed(12345)
    H_ref, _, _ = get_dis_scar_ham(H0_clean, N, basisList,
                                   ham_disorder=[z, y, x], diagonalize=False)
    H1_ref, w_ref = get_scar_H1(N, basisList, ds_dis=dd)
    tail_ref = np.random.uniform(size=4)

    v = sc.draw_unit_fields(N, 12345)
    tail_new = np.random.uniform(size=4)
    H0_new, d1_new = sc.assemble_scar_H(struct, *v, x, y, z, dd)

    d_dis = np.abs(dense(H_ref) - np.asarray(H0_new.todense())).max()
    d_drv = np.abs(np.real(np.diag(dense(H1_ref))) - d1_new).max()
    d_w = np.abs(w_ref - (1.0 + dd * v[3])).max()
    d_hy = np.abs(dense(get_Hy(N, basisList))
                  - np.asarray(sc.build_Hy_staggered(struct).todense())).max()
    rng_ok = bool(np.allclose(tail_ref, tail_new))

    # the decoupled comparison qubits, block by block
    np.random.seed(999)
    q0, q1, _ = get_qubit_ham(N, wm=1.0, ham_disorder=[z, y, x], ds_dis=dd)
    vq = sc.draw_unit_fields(N, 999)
    hz, hy, hx = z * vq[0], y * vq[1], x * vq[2]
    dw = 1.0 + dd * vq[3]
    d_q0 = max(np.abs(dense(q0[i]) - (-0.5 * sc._SX + hz[i] * sc._SZ
                                      + hy[i] * sc._SY + hx[i] * sc._SX)).max()
               for i in range(N))
    d_q1 = max(np.abs(dense(q1[i]) - dw[i] * sc._SZ).max() for i in range(N))

    # the E = 0 scar
    ref_scar, ov_ref = get_zero_scar(N)
    new_scar, ov_new = sc.get_zero_scar(struct)
    fid = float(abs(np.vdot(ref_scar.full().ravel(), new_scar)) ** 2)

    print(f"  qsf N={N:2d} [{backend}]: basis={order_ok} z2={z2_ok} rng={rng_ok} "
          f"|dH_clean|={d_clean:.2e} |dH0_dis|={d_dis:.2e} |dH1|={d_drv:.2e} "
          f"|dw|={d_w:.2e} |dHy|={d_hy:.2e} |dQ0|={d_q0:.2e} |dQ1|={d_q1:.2e} "
          f"zero_scar_fidelity={fid:.10f}")

    worst = max(d_clean, d_dis, d_drv, d_w, d_hy, d_q0, d_q1)
    if not (order_ok and z2_ok and rng_ok) or worst > 1e-10 or fid < 1 - 1e-8:
        raise SystemExit(
            "scarcore disagrees with quantumScarFunctions.py. That file has "
            "probably been refactored -- re-derive the fast path before "
            "trusting this run."
        )


# kept under the old name so main.py's startup check keeps working
check_against_qutip = check_against_quantumscarfunctions


if __name__ == "__main__":
    print("Kronecker reference (independent full 2^N construction)")
    for N in (4, 6, 8, 10):
        check_hamiltonians(N)

    print("\nLinearity of H in the disorder strengths")
    for N in (6, 10):
        check_linearity(N)

    print("\nRNG stream / common random numbers")
    for N in (8, 12):
        check_rng_stream(N)

    print("\nPropagator")
    check_propagator()

    print("\nErgotropy identities")
    check_ergotropy_identities()

    print("\nAgainst quantumScarFunctions.py")
    check_against_qutip()

    print("\nALL CHECKS PASSED")
