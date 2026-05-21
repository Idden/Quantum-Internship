# Tests for `Idden_code`

These tests check the student scripts in `Idden_code/` without changing those scripts.

Run them from the `QuantumBatteries` repository root:

```bash
conda run -n qdragon pytest -q tests/test_idden_code
```

The independent reference data comes from the QuSpin-page-style notebook:

- `IndependentData/minimal_PXP.ipynb`
- `IndependentData/minimal_pxp_constrained_basis_N10.npz`
- `IndependentData/minimal_pxp_student_hamiltonian_N10.npz`

Normal pytest runs should read these NPZ files. They should not regenerate notebook data.

## Inspecting Test Artifacts

Each focused test run writes inspectable files under `tests/test_output/`.

The spectrum-comparison tests create, for both the clean and disordered Hamiltonians:

- `*_student_quspin_comparison.npz` with student data, QuSpin data, and residual arrays.
- `*_student_quspin_comparison.csv` with the same values in tabular form.
- `*_student_quspin_comparison.png` showing sorted spectra, energy residuals, and Z2 weights versus eigenenergy.

The `main_local.run_one` test creates:

- `run_one_output.npz`
- `run_one_output.csv`
- `run_one_output.png`

Open `inspect_student_quspin_artifacts.ipynb` after running pytest to load these files and view summary tables and plots in one place.

## What Is Tested

- Basic helper functions such as binary conversion, Z2 state construction, and drive coefficients.
- The periodic Rydberg-blockade constrained basis against independent QuSpin data.
- The clean and disordered student scar Hamiltonian spectra against independent QuSpin data.
- Drive-operator structure and deterministic disorder behavior.
- The `main_local.run_one` output file format on a tiny N=4 job.

## Adding A New Test

Prefer a small, direct test that checks one physical or code property.

Use `test_subdir` when the test should save data or metadata:

```python
def test_new_property(test_subdir):
    result = compute_result()
    assert result is not None
    save_metadata(test_subdir / "metadata.json", {"result": str(result)})
```

Keep independent physics references in the QuSpin notebook or in inspectable NPZ files under `IndependentData/`. Avoid hidden helper layers for reference calculations.
