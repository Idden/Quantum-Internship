"""
End-to-end tests of the charging pipeline.

These reproduce, at a small N, exactly what ``xyz_parallel.py`` does per
realization -- build the disordered Hamiltonian, evolve the ground state
under the sinusoidal drive, and read off Rtau -- but they do it by calling
the library functions directly.  The HPC scripts run a full sweep at import
time, so they cannot be imported by a test; the point here is to cover the
same code path they exercise without touching them.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import qutip as qt

from data_utils import save_array_data, save_csv_table, save_metadata
from plot_utils import PLOT_STYLE, apply_plot_style, plot_energy_evolution

N = 6
WD = 0.6366896896896898
WM = 1.0
AMPLITUDE = 0.1
T_MAX = 20.0
N_STEPS = 60


@pytest.fixture(scope="module")
def tlist():
    return np.linspace(0.0, T_MAX, N_STEPS)


@pytest.fixture(scope="module")
def system():
    from GitHub_QM.important.hpc.quantumScarFunctions import get_scar_ham, get_scar_H1

    H0, eigenvalues, eigenstates, psi0, basisList = get_scar_ham(N)
    H1, _ = get_scar_H1(N, basisList)
    return H0, eigenvalues, eigenstates, psi0, basisList, H1


# --------------------------------------------------------------------------
# Undriven evolution: the conservation laws
# --------------------------------------------------------------------------
def test_undriven_evolution_conserves_energy(system, tlist, test_subdir):
    """
    With the drive amplitude set to zero, <H0> must be constant.  Any drift
    here is a solver-tolerance problem that would masquerade as physical
    charging in the driven runs.
    """
    from GitHub_QM.important.hpc.quantumScarFunctions import coeff

    H0, _, eigenstates, _, _, H1 = system

    # Same QobjEvo construction the sweeps use, with the amplitude set to zero.
    H = qt.QobjEvo([H0, [H1, coeff]], args={"A": 0.0, "omega": WD})
    result = qt.sesolve(H, eigenstates[0], tlist, e_ops=[H0])

    energy = np.real(result.expect[0])
    drift = np.abs(energy - energy[0]).max()

    assert drift < 1e-8, f"energy drifted by {drift:.3e} with no drive"

    fig = plot_energy_evolution(tlist, energy, title="Undriven evolution (N=6)",
                                output_path=test_subdir / "energy.png")
    plt.close(fig)
    save_metadata(test_subdir / "metadata.json", {"N": N, "max_drift": float(drift)})


def test_undriven_evolution_preserves_norm(system, tlist):
    H0, _, eigenstates, _, _, _ = system

    result = qt.sesolve(H0, eigenstates[0], tlist)

    norms = np.array([state.norm() for state in result.states])
    assert np.abs(norms - 1.0).max() < 1e-8


def test_eigenstate_is_stationary(system, tlist):
    """An eigenstate of H0 evolves only by a phase, so |<psi(0)|psi(t)>| = 1."""
    H0, _, eigenstates, _, _, _ = system

    initial = eigenstates[0]
    result = qt.sesolve(H0, initial, tlist)

    initial_vector = initial.full().ravel()
    for state in result.states:
        fidelity = abs(np.vdot(initial_vector, state.full().ravel())) ** 2
        assert fidelity == pytest.approx(1.0, abs=1e-8)


# --------------------------------------------------------------------------
# Driven evolution: Rtau
# --------------------------------------------------------------------------
def _rtau(H0, H1, psi_initial, tlist, omega, amplitude):
    """The Rtau definition used verbatim in the parallel drivers."""
    from GitHub_QM.important.hpc.quantumScarFunctions import coeff

    eigenvalues = H0.eigenenergies()
    bandwidth = eigenvalues[-1] - eigenvalues[0]

    H = qt.QobjEvo([H0, [H1, coeff]], args={"A": amplitude, "omega": omega})
    result = qt.sesolve(H, psi_initial, tlist, e_ops=[H0])

    return np.array(np.real(result.expect[0] - result.expect[0][0]) / bandwidth)


def test_rtau_shape_and_finiteness(system, tlist, test_subdir):
    H0, _, eigenstates, _, _, H1 = system

    rtau = _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)

    assert rtau.shape == tlist.shape
    assert np.all(np.isfinite(rtau))
    assert rtau[0] == pytest.approx(0.0, abs=1e-12)

    apply_plot_style()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    ax.plot(tlist, rtau)
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$R_\tau$")
    ax.set_title(f"Charging curve (N={N}, $\\omega_d$={WD:.4f})")
    ax.grid(True, alpha=PLOT_STYLE["grid_alpha"])
    fig.savefig(test_subdir / "rtau.png", dpi=PLOT_STYLE["save_dpi"],
                bbox_inches="tight")
    plt.close(fig)

    save_csv_table(
        test_subdir / "rtau.csv",
        {"time": tlist, "Rtau": rtau},
        header=f"N={N} scar charging curve",
    )
    save_metadata(
        test_subdir / "metadata.json",
        {"N": N, "wd": WD, "amplitude": AMPLITUDE, "max_Rtau": float(rtau.max())},
    )


def test_rtau_is_bounded_by_one(system, tlist):
    """
    Rtau is an energy gain divided by the full bandwidth, so it cannot exceed
    1 -- the state cannot absorb more than the spectrum allows.
    """
    H0, _, eigenstates, _, _, H1 = system

    rtau = _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)

    assert rtau.max() <= 1.0 + 1e-9
    assert rtau.min() >= -1e-9, "energy dropped below the ground state"


def test_zero_amplitude_gives_zero_rtau(system, tlist):
    H0, _, eigenstates, _, _, H1 = system

    rtau = _rtau(H0, H1, eigenstates[0], tlist, WD, 0.0)
    assert np.abs(rtau).max() < 1e-9


def test_resonant_drive_charges_more_than_far_detuned(system, tlist):
    """
    The whole point of the chosen ``wd`` is that it is near resonance.  A
    strongly detuned drive must transfer noticeably less energy over the same
    window -- if this fails, the drive is not coupling as intended.
    """
    H0, _, eigenstates, _, _, H1 = system

    resonant = _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)
    detuned = _rtau(H0, H1, eigenstates[0], tlist, 12.0 * WD, AMPLITUDE)

    assert resonant.max() > detuned.max()


# --------------------------------------------------------------------------
# The disordered pipeline, as the sweeps run it
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label,disorder", [
    ("z", [0.3, 0.0, 0.0]),
    ("y", [0.0, 0.3, 0.0]),
    ("x", [0.0, 0.0, 0.3]),
])
def test_disordered_realization_runs_end_to_end(scar_functions, system, tlist,
                                                label, disorder):
    """One full realization per disorder axis, exactly as ``run_one`` does it."""
    H0_clean, _, _, _, basisList, H1 = system

    np.random.seed(0)
    H0, eigenvalues, eigenstates = scar_functions.get_dis_scar_ham(
        H0_clean, N, basisList, ham_disorder=disorder
    )

    rtau = _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)

    assert rtau.shape == tlist.shape
    assert np.all(np.isfinite(rtau))
    assert rtau[0] == pytest.approx(0.0, abs=1e-12)


def test_realizations_are_seed_reproducible(scar_functions, system, tlist):
    """
    The sweeps average over seeds, so an identical seed must give an identical
    curve and a different seed must give a different one.
    """
    H0_clean, _, _, _, basisList, H1 = system

    def realize(seed):
        np.random.seed(seed)
        H0, _, eigenstates = scar_functions.get_dis_scar_ham(
            H0_clean, N, basisList, ham_disorder=[0.3, 0.0, 0.0]
        )
        return _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)

    np.testing.assert_allclose(realize(5), realize(5), atol=1e-12)
    assert np.abs(realize(5) - realize(6)).max() > 1e-9


# --------------------------------------------------------------------------
# The scar projector used as an e_op in xyz_parallel
# --------------------------------------------------------------------------
def test_scar_projector_is_a_valid_probability(scar_functions, system, tlist,
                                               test_subdir):
    """
    ``xyz_parallel`` measures the total weight on the scar tower with
    ``sum |<scar_k|psi(t)>|^2``.  For that to be a probability the scar states
    must be orthonormal and the sum must stay in [0, 1].
    """
    H0, eigenvalues, eigenstates, psi0, basisList, H1 = system

    # Build the scar tower the same way make_scar_states.py does: one state per
    # energy section, chosen by largest Z2 overlap, with the E=0 slot replaced
    # by the max-S^2 zero mode.
    z2 = psi0.full().ravel().real
    V = np.column_stack([s.full().ravel().real for s in eigenstates])
    energies = np.asarray(eigenvalues, dtype=float)

    sections = np.linspace(energies[0] - 0.5, energies[-1] + 0.5, N + 2)
    scar_indices = []
    for i in range(len(sections) - 1):
        in_section = [k for k in range(len(energies))
                      if sections[i] < energies[k] < sections[i + 1]]
        if not in_section:
            continue
        overlaps = [abs(z2 @ V[:, k]) ** 2 for k in in_section]
        scar_indices.append(in_section[int(np.argmax(overlaps))])

    scar_matrix = np.ascontiguousarray(V[:, scar_indices].T).astype(complex)

    mid = int(np.argmin(np.abs(energies[np.array(scar_indices)])))
    zero_scar, _ = scar_functions.get_zero_scar(N)
    scar_matrix[mid] = zero_scar.full().ravel()

    # Orthonormality is what makes the sum a projector at all.
    gram = scar_matrix.conj() @ scar_matrix.T
    gram_error = np.abs(gram - np.eye(len(gram))).max()
    assert gram_error < 1e-6, f"scar tower not orthonormal: |G - I| = {gram_error:.2e}"

    scar_conj = scar_matrix.conj()
    result = qt.sesolve(
        qt.QobjEvo([H0, [H1, scar_functions.coeff]],
                   args={"A": AMPLITUDE, "omega": WD}),
        eigenstates[0], tlist,
        e_ops=[lambda t, psi: float(np.sum(np.abs(scar_conj @ psi.full().ravel()) ** 2))],
    )
    probability = np.real(result.expect[0])

    assert np.all(probability >= -1e-9)
    assert np.all(probability <= 1.0 + 1e-9)

    save_array_data(test_subdir / "scar_probability.npz",
                    tlist=tlist, scar_probability=probability)
    save_metadata(
        test_subdir / "metadata.json",
        {
            "N": N,
            "num_scar_states": len(scar_indices),
            "gram_error": float(gram_error),
            "min_probability": float(probability.min()),
            "max_probability": float(probability.max()),
        },
    )


# --------------------------------------------------------------------------
# Output round-trip
# --------------------------------------------------------------------------
def test_npz_round_trip(system, tlist, tmp_path):
    """
    The sweeps persist results as ``.npz``.  Save and reload a realization to
    confirm nothing is lost to dtype coercion on the way to disk.
    """
    H0, _, eigenstates, _, _, H1 = system

    rtau = _rtau(H0, H1, eigenstates[0], tlist, WD, AMPLITUDE)
    path = tmp_path / "realization.npz"

    np.savez(path, tlist=tlist, scar=rtau, N=N, wd=WD)

    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) >= {"tlist", "scar", "N", "wd"}
        np.testing.assert_allclose(data["tlist"], tlist)
        np.testing.assert_allclose(data["scar"], rtau)
        assert int(data["N"]) == N
        assert float(data["wd"]) == pytest.approx(WD)
