# Tests for `important/hpc/quantumScarFunctions.py`

These tests check the quantum-scar simulation code without modifying it.

## Running them

From anywhere:

```bash
pytest important/tests
```

`conftest.py` puts the parent of `GitHub_QM/` on `sys.path`, so the tests
import your code through the same path your HPC scripts use
(`GitHub_QM.important.hpc.quantumScarFunctions`). You do not need to run
pytest from a particular directory.

Run one file, or one test:

```bash
pytest important/tests/test_models/test_zero_scar.py
pytest important/tests -k "reference"
```

Requirements: `pytest`, `numpy`, `scipy`, `qutip`, `matplotlib`.

## What is tested

| File | Covers |
|---|---|
| `test_core/test_basis.py` | `binToDeci`, `binNoConsecOnesEfficient`, `z2_initial`, the periodic blockade basis, `get_C_AB_matrix` |
| `test_core/test_drive_coeffs.py` | `coeff`, `const`, `timed_drive`, `timed_const`, `make_coeff` |
| `test_models/test_scar_ham.py` | `get_scar_ham` — structure, sparsity, spectrum, `diagonalize=`, `ohms` scaling |
| `test_models/test_disorder.py` | `get_dis_scar_ham` — all three axes, seeding, bounds, `N_dis` |
| `test_models/test_drive_operators.py` | `get_scar_H1` (combined and per-qubit), `get_Hy` |
| `test_models/test_zero_scar.py` | `get_zero_scar` — zero mode, max-S², Z2 overlap |
| `test_models/test_qubit_ham.py` | `get_qubit_ham` — the independent-qubit reference battery |
| `test_integration/test_driven_evolution.py` | End-to-end charging: energy conservation, Rtau, the scar projector |

## The independent reference

`reference_pxp.py` rebuilds the PXP Hamiltonian, the drive operator and `Hy`
**from scratch** in the full 2^N Hilbert space using plain numpy Kronecker
products, then restricts them to the blockade subspace. It imports nothing
from `quantumScarFunctions`, so agreement between the two is a real
cross-check of the physics and not a restatement of the implementation.

It currently matches your code to **0.0 elementwise** at N = 4, 6, 8, 10.

This replaces the old `IndependentData/minimal_pxp_*_N10.npz` QuSpin files,
which are not in the repository. Nothing needs to be downloaded or
regenerated — the reference is computed at test time and is fast because
2^10 = 1024 is small.

## Inspecting artifacts

Several tests write files to `tests/test_output/<module>/<test_name>/`:
spectra, sparsity patterns, charging curves, and a `metadata.json` per test.
`test_output/` is wiped at the start of each session, so run the full suite
before browsing it.

## Two known things

**`get_zero_scar` cannot run at N=4.** It starts ARPACK at
`K = max(16, int(0.02*D))`, but N=4 gives only D=7 and ARPACK requires
`k < D-1`. Pass `k0` explicitly (`get_zero_scar(4, k0=2)`) to work around it.
`test_zero_scar.py::test_small_chain_needs_explicit_k0` documents this — if
you ever clamp `K` against `D`, that test will start failing and should be
deleted.

**There are two copies of `quantumScarFunctions.py`** — one in
`important/hpc/` and one in `important/paper/helper/`. They are not
identical: the hpc copy adds the `diagonalize=` flag and a completely
rewritten sparse `get_zero_scar`. **These tests target the hpc copy only.**
If the paper copy is still the one your notebooks import, the two will drift.

## Adding a test

Prefer a small, direct test of one property. Use `test_subdir` when the test
should save something inspectable:

```python
def test_new_property(scar_system, test_subdir):
    H0, eigenvalues, eigenstates, psi0, basisList = scar_system(8)
    assert something
    save_metadata(test_subdir / "metadata.json", {"N": 8})
```

Available fixtures (defined in `conftest.py`):

- `scar_functions` — the module under test
- `scar_system(N)` — `(H0, eigenvalues, eigenstates, psi0, basisList)`, cached per N
- `basis_list(N)` — the constrained basis, cached per N
- `test_subdir` — per-test artifact directory
- `rng` — a seeded `np.random.default_rng`

Use `scar_system(N)` rather than calling `get_scar_ham(N)` directly; it is
memoised for the whole session, which is what keeps the suite quick.
