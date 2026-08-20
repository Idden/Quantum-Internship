# Test list — 102 functions, 246 cases

`xN` = the test runs N times (different chain lengths or parameters).

---

## `test_core/test_basis.py` — 58 cases

Basis construction: `binToDeci`, `binNoConsecOnesEfficient`, `z2_initial`, the blockade basis, `get_C_AB_matrix`.

| Test | | What it checks |
|---|---|---|
| `test_bin_to_deci` | x5 | Known bit strings convert to the right integer |
| `test_bin_to_deci_matches_int_builtin` | x1 | Matches `int(s, 2)` on 200 random strings — pins the bit ordering |
| `test_no_consecutive_ones_count_is_fibonacci` | x8 | Open chain gives F(N+2) configurations |
| `test_no_consecutive_ones_are_valid_and_unique` | x8 | No `11` anywhere, no duplicates, right length |
| `test_z2_initial` | x4 | Néel state is `1010...` |
| `test_z2_state_is_blockade_allowed` | x5 | Néel state survives the periodic blockade |
| `test_blockade_basis_matches_independent_enumeration` | x5 | Basis equals a brute-force sweep of all 2^N strings |
| `test_blockade_basis_dimension_is_lucas` | x5 | Dimension is the Lucas number L_N (7, 18, 47, 123, 322) |
| `test_blockade_basis_contains_vacuum` | x5 | All-zero state is present |
| `test_blockade_basis_excludes_wraparound_pairs` | x5 | No state has 1s on both ends |
| `test_blockade_basis_ordering_is_stable` | x1 | Two calls give the same order — your saved `.npz` files are index-based |
| `test_C_AB_matrix_shape_and_norm` | x3 | Bipartition reshape preserves the norm |
| `test_C_AB_entropy_of_product_state_is_zero` | x3 | Néel state has zero entanglement entropy |

## `test_core/test_drive_coeffs.py` — 11 cases

The scalar functions handed to `QobjEvo` as time-dependent coefficients.

| Test | | What it checks |
|---|---|---|
| `test_coeff_is_sine_drive` | x1 | `coeff = A·sin(ωt)` |
| `test_coeff_zero_at_origin_and_period` | x1 | Vanishes at t=0 and t=π/ω |
| `test_coeff_amplitude_is_bounded` | x1 | Never exceeds A |
| `test_const_is_linear_ramp` | x1 | `const = A·t` |
| `test_timed_drive_matches_coeff_before_limit` | x1 | Identical to `coeff` while the drive is on |
| `test_timed_drive_is_off_after_limit` | x1 | Exactly zero past the cutoff |
| `test_timed_drive_switches_exactly_at_limit` | x1 | Cutoff is `t < limit`, so t=limit is already off |
| `test_timed_const_matches_const_before_limit` | x1 | Same, for the linear ramp |
| `test_timed_const_is_off_after_limit` | x1 | Zero past the cutoff |
| `test_make_coeff_reads_its_own_site_frequency` | x1 | Site r reads `wd{r}` — catches late-binding bugs in frequency disorder |
| `test_make_coeff_agrees_with_coeff` | x1 | Reduces to plain `coeff` for one site |

## `test_models/test_scar_ham.py` — 62 cases

`get_scar_ham` — the clean periodic PXP Hamiltonian with the −0.026 σᶻ perturbation.

| Test | | What it checks |
|---|---|---|
| `test_hamiltonian_is_square_and_matches_basis` | x4 | Shapes agree with the basis size |
| `test_hamiltonian_is_hermitian` | x4 | H = H† |
| `test_hamiltonian_is_real` | x4 | Real symmetric — `make_scar_states.py` relies on this to save memory |
| `test_hamiltonian_has_zero_diagonal` | x4 | Every term is an off-diagonal spin flip |
| `test_hamiltonian_is_sparse` | x4 | nnz grows like N·D, not D² — this is what makes N=20 feasible |
| **`test_matches_independent_reference_elementwise`** | x4 | **Every matrix element matches the brute-force 2^N build** |
| **`test_spectrum_matches_independent_reference`** | x4 | **Eigenvalues match to 1e-10; Z2 weight matches per degenerate block** |
| `test_psi0_is_the_z2_basis_vector` | x4 | Initial state is a unit vector at the Néel index |
| `test_eigenstates_are_orthonormal` | x4 | Gram matrix is the identity |
| `test_eigenvalues_are_ascending` | x4 | Spectrum comes back sorted |
| `test_eigenstates_solve_the_eigenproblem` | x4 | H\|E⟩ = E\|E⟩, spot-checked |
| `test_diagonalize_false_skips_the_spectrum` | x3 | Returns `None` for eigenvalues/eigenstates |
| `test_diagonalize_false_gives_the_same_hamiltonian` | x3 | The lazy path doesn't change H |
| `test_hamiltonian_is_linear_in_ohms` | x3 | H(Ω) = Ω·H(1) |
| `test_odd_chain_length_is_rejected` | x3 | Odd N raises — no Néel state on an odd ring |
| `test_perturbation_is_actually_present` | x3 | Matches reference at −0.026, differs from reference at 0 |
| `test_spectrum_is_particle_hole_symmetric` | x3 | Spectrum symmetric about E=0 with an exact zero mode |

## `test_models/test_disorder.py` — 19 cases

`get_dis_scar_ham` — disorder on the z, y and x axes.

| Test | | What it checks |
|---|---|---|
| `test_zero_disorder_returns_the_clean_hamiltonian` | x1 | `[0,0,0]` is a true no-op — your clean baseline |
| `test_disordered_hamiltonian_is_hermitian` | x4 | Hermitian on z, y, x and all three together |
| `test_each_axis_actually_changes_the_hamiltonian` | x3 | No axis is silently a no-op |
| `test_z_disorder_is_diagonal` | x1 | σᶻ disorder adds only a diagonal |
| `test_x_and_y_disorder_are_off_diagonal` | x1 | σˣ/σʸ add no diagonal |
| `test_z_disorder_respects_its_strength_bound` | x1 | Diagonal shift never exceeds N·zd |
| `test_fixed_seed_is_reproducible` | x1 | Two `fixed_seed=True` calls are identical |
| `test_external_seed_controls_the_realization` | x1 | `np.random.seed(s)` then call → same s identical, different s differs |
| `test_fixed_seed_overrides_the_external_seed` | x1 | `fixed_seed=True` ignores the outer seed — would collapse a sweep |
| `test_N_dis_limits_the_number_of_disordered_sites` | x1 | `N_dis=1` disorders exactly one site |
| `test_diagonalize_false_skips_the_spectrum` | x1 | Returns `None` for eigenvalues/eigenstates |
| `test_diagonalize_false_gives_the_same_hamiltonian` | x1 | Lazy path doesn't change H |
| `test_eigenvalues_are_ascending_and_real` | x1 | Sorted and real even with all three axes on |
| `test_clean_hamiltonian_is_not_mutated` | x1 | Your reused clean H comes back untouched |

## `test_models/test_drive_operators.py` — 45 cases

`get_scar_H1` (the driven operator) and `get_Hy`.

| Test | | What it checks |
|---|---|---|
| `test_drive_is_diagonal_and_hermitian` | x4 | Diagonal, Hermitian, real |
| **`test_drive_matches_independent_reference`** | x4 | **Matches the staggered Σ z2ᵣ·Zᵣ built from real Pauli matrices** |
| `test_drive_is_maximal_on_the_z2_state` | x4 | Néel state sits at the top with eigenvalue exactly N |
| `test_drive_weights_default_to_ones` | x4 | No disorder → all weights 1 |
| `test_individual_drives_sum_to_the_combined_drive` | x4 | Per-site split is exact |
| `test_individual_drives_are_diagonal` | x3 | Each per-site operator is diagonal |
| `test_individual_drive_entries_are_plus_or_minus_one` | x3 | Each site contributes ±1 |
| `test_drive_disorder_respects_its_bound` | x1 | Weights stay in [1−ds, 1+ds] |
| `test_drive_disorder_is_reproducible_with_fixed_seed` | x1 | Same seed, same weights |
| `test_drive_disorder_N_dis_limits_affected_sites` | x1 | `N_dis=1` changes exactly one weight |
| `test_drive_disorder_still_sums_correctly` | x1 | Split stays exact under weight disorder |
| `test_Hy_is_hermitian` | x4 | The hand-written ±i phases come out conjugate |
| **`test_Hy_matches_independent_reference`** | x4 | **Matches the staggered sum of true Pauli-Y matrices** |
| `test_Hy_is_purely_off_diagonal` | x4 | No diagonal part |
| `test_Hy_spectrum_is_symmetric` | x3 | Spectrum symmetric about zero |

## `test_models/test_zero_scar.py` — 22 cases

`get_zero_scar` — the maximum-S² zero-energy scar. The most intricate function you have.

| Test | | What it checks |
|---|---|---|
| `test_scar_is_a_normalised_column_vector` | x3 | Shape (D,1), norm 1 |
| `test_scar_is_annihilated_by_the_hamiltonian` | x3 | H\|scar⟩ = 0 — the defining property |
| `test_reported_overlap_matches_the_state` | x3 | Returned overlap equals \|⟨Z2\|scar⟩\|² recomputed |
| `test_overlap_decreases_with_system_size` | x3 | Overlap stays in a physical range |
| `test_overlap_is_monotone_in_N` | x1 | Z2 weight thins out as the chain grows |
| **`test_matches_dense_reference_implementation`** | x3 | **Sparse ARPACK path matches a plain dense computation, fidelity 1.0** |
| `test_scar_lies_in_the_max_S2_sector` | x2 | ⟨S²⟩ equals the max in the null space — not just any zero mode |
| `test_scar_is_deterministic` | x3 | Two calls agree up to a global phase |
| `test_small_chain_needs_explicit_k0` | x1 | Documents the N=4 ARPACK limitation; `k0=2` works |

## `test_models/test_qubit_ham.py` — 16 cases

`get_qubit_ham` — the independent-qubit reference battery you compare against.

| Test | | What it checks |
|---|---|---|
| `test_returns_one_operator_pair_per_site` | x1 | N static + N drive operators, all 2×2 |
| `test_operators_are_hermitian` | x1 | Hermitian with disorder on |
| `test_default_convention_is_x_static_z_drive` | x1 | H0 = −wm/2·σˣ, H1 = wᵣ·σᶻ |
| `test_sigz_convention_swaps_the_roles` | x1 | `sigz_ham=True` swaps them |
| `test_clean_qubit_bandwidth_equals_wm` | x3 | Bandwidth is exactly wm — sets the `Rtau_qubit` denominator |
| `test_drive_weights_default_to_ones` | x1 | No disorder → weights 1 |
| `test_drive_weight_disorder_respects_its_bound` | x1 | In [1−ds, 1+ds], and weights reach the operators |
| `test_each_disorder_axis_changes_the_static_hamiltonian` | x3 | z, y, x each have an effect |
| `test_disorder_only_touches_the_static_hamiltonian` | x1 | Disorder doesn't leak into the drive |
| `test_fixed_seed_is_reproducible` | x1 | Same seed, same operators |
| `test_external_seed_controls_the_realization` | x1 | Global reseed drives the realization |
| `test_N_dis_limits_the_number_of_disordered_qubits` | x1 | `N_dis=1` touches one qubit |

## `test_integration/test_driven_evolution.py` — 13 cases

End-to-end charging at N=6 — the same pipeline `xyz_parallel.py` runs per realization.

| Test | | What it checks |
|---|---|---|
| `test_undriven_evolution_conserves_energy` | x1 | ⟨H0⟩ constant with A=0 — catches solver drift faking charging |
| `test_undriven_evolution_preserves_norm` | x1 | Norm stays 1 |
| `test_eigenstate_is_stationary` | x1 | An eigenstate evolves only by a phase |
| `test_rtau_shape_and_finiteness` | x1 | Rtau has the right shape, is finite, starts at 0 |
| `test_rtau_is_bounded_by_one` | x1 | Can't absorb more than the bandwidth allows |
| `test_zero_amplitude_gives_zero_rtau` | x1 | No drive → no charging |
| `test_resonant_drive_charges_more_than_far_detuned` | x1 | Your `wd` is actually near resonance |
| `test_disordered_realization_runs_end_to_end` | x3 | One full realization on each of z, y, x |
| `test_realizations_are_seed_reproducible` | x1 | Same seed same curve, different seed different curve |
| `test_scar_projector_is_a_valid_probability` | x1 | Scar tower is orthonormal and the summed probability stays in [0,1] |
| `test_npz_round_trip` | x1 | Save/reload loses nothing to dtype coercion |

---

## The four that matter most

Everything else checks structure. These four compare your code against a
construction that shares no code with it — `reference_pxp.py` builds the
operators from scratch in the full 2^N space with numpy Kronecker products:

- `test_matches_independent_reference_elementwise` — the Hamiltonian
- `test_spectrum_matches_independent_reference` — the spectrum and Z2 weights
- `test_drive_matches_independent_reference` — the drive operator
- `test_Hy_matches_independent_reference` — the staggered-Y operator

Plus `test_matches_dense_reference_implementation`, which checks the sparse
ARPACK `get_zero_scar` against a plain dense computation.

Current agreement: **0.0 elementwise** at N = 4, 6, 8, 10.
