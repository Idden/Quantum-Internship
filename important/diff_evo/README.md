# diff_evo — DE search for scar-vs-qubit battery advantage

Finds disorder + drive parameters maximising

    score = mean_seed [ max_t R_scar(t) − max_t R_qubit(t) ]

where `R(t) = (⟨H0⟩(t) − ⟨H0⟩(0)) / bandwidth`.

Because the state is pure and the evolution unitary, and the initial state is
the ground state of `H0_dis`, the passive state *is* the ground state, so
`ergotropy(t) = ⟨H0⟩(t) − E_min = R(t)·W`. **R is exactly the normalised
ergotropy**, and `R(0) = 0` says the battery starts genuinely empty. That
sentence is free and is the strongest thing available about the metric — put it
in the manuscript.

---

## Files

| File | What it is |
|---|---|
| `scarcore.py` | The numerical core. qutip-free. Basis, structure, linear-in-parameter Hamiltonian assembly, both propagators, all metrics. |
| `build_cache.py` | **Preprocessing.** Writes the per-N structure cache and the per-seed unit-disorder cache. Run once per N. |
| `main.py` | The DE driver. |
| `rerank.py` | Offline re-ranking + bootstrap CIs from the eval logs. No re-search needed. |
| `validate_core.py` | All the correctness checks. Run it before trusting a job. |
| `reference_kron.py` | Independent full-2^N Kronecker construction, used only by the checks. |
| `qutip_shim.py` | Minimal Qobj stand-in so the equivalence check can run where qutip is not installed. Not used when real qutip is present. |
| `quantumScarFunctions.py` | Unchanged physics reference, except one bug fix (see below). |
| `job_cache.sh` / `job_calib.sh` / `job_de.sh` | UIC ICC Slurm scripts, run in that order. |
| `FILES.md` | What every file does and why, plus the dependency graph. |
| `METRICS.md` | Every metric defined, with formulas and what each one catches. |

---

## Why this is faster

The old `test_main.py` did all of this *inside every objective evaluation*:

* rebuilt the constrained basis and the flip sparsity pattern with `O(D·N)`
  Python string operations,
* re-drew the disorder through numpy's global RNG,
* re-diagonalised `H0_dis` with a full dense `eigh` and then used two numbers
  from it,
* called qutip `sesolve` N separate times for the N decoupled qubits.

Only the time evolution is real work. The rest is now precomputed.

**The structural fact that makes the caching exact.** The disorder is drawn as
`strength × (a fixed uniform pattern)`, and `np.random.uniform(-d, d, N)`
consumes the same underlying deviates as `np.random.uniform(-1, 1, N)`, scaled.
So for a fixed seed:

    H0_dis(x, y, z) = H_clean + z·Dz + y·Ay + x·Ax          (exactly linear)
    H1(dd)          = D1_base + dd·D1_pert                  (exactly linear)

Assembling a Hamiltonian is now four sparse scalar-multiply-adds on cached
matrices. `validate_core.check_linearity` asserts this, and if it ever stops
holding the cache is invalid and the check fails loudly.

The dense `eigh` is still done once per evaluation, but it now pays for three
things instead of two: the initial state, the bandwidth, **and** the level
populations that the dephased ergotropy needs.

---

## The metric problem, and what is now recorded

`max_t R` can be maximised by **heating**. A candidate whose maximum occurs
late, at high half-chain entropy, is a thermalised state — the disordered
spectra are near-symmetric, so an infinite-temperature state already has
`R ≈ 0.5–0.63` with no scar physics at all.

The DE still drives on `max_t R`, so results stay comparable with earlier runs,
but **every evaluation now also logs**:

| Metric | What it catches |
|---|---|
| `score_deph` | Dephased (**incoherent**, Francica et al. PRL 125 180603) ergotropy difference. Work still extractable once coherences are lost. Separates coherent charging from heating cleanly. |
| `score_first` | First-peak height difference — the coherent charging event. |
| `score_power` | `max_t R(t)/t` difference — charging power. Likely where the real result is. |
| `S_at_t1`, `S_at_tmax` | Half-chain entropy at the peak, in **nats** (no ratio). At N=12: absolute max ln(21)=3.045, Haar-random 2.472, weak-disorder eigenstates ~2.31. Below ~0.5 nats is coherent. |
| `maxR_scar` / `maxR_qubit` separately | So you can check the score was not won by *crippling the benchmark*. |

This is already earning its keep. A calibration-run candidate scored
`+0.125` on `max R` — the best in its generation — with `S = 1.83 nats` and a
**dephased score of −0.002**. It is pure heating. The old objective would have
chased it.

Re-sort on any of these afterwards with `rerank.py`; no re-running the search.

---

## Statistics

At one realization the score is inside the disorder noise: across 8 seeds at
fixed parameters the spread exceeded the effect and only 4–6 seeds had the
right sign. DE with `--objective-reals 1` returns whichever point won the
disorder lottery.

The seeding is already **common random numbers** — the scar path and the qubit
path read the *same* `v_z, v_y, v_x, v_w` for a given seed — so the difference
is paired and far less noisy than either maximum. Since the rebuild overhead is
gone, spend the budget on realizations: `--objective-reals 8` for the search,
then `rerank.py --confirm 32` with a bootstrap CI over seeds for the number you
quote.

---

## Sampling bias

`max_t R` on a sampled grid is biased downward, and biases the two models
differently. On the old `dt = 0.5` grid that was worth up to 0.033 — the same
size as the effect. Two changes: `--nt` defaults to 1601 (`dt = 0.125`), and
`refined_max` puts a parabolic correction through the three samples around the
argmax. Output points are cheap; the integrator's own step control does not
depend on `--nt`.

---

## Bug fixed in `quantumScarFunctions.py`

`get_zero_scar` started ARPACK at `K = max(16, int(0.02*D))` with no clamp, so
at N=4 (D=7) it raised — ARPACK requires `k < D−1`. Now clamped to `D−2`, and
the doubling loop is clamped too. This is the "needs an explicit k0" behaviour
the test suite documented; it was a missing clamp, not a real limitation.

Nothing else in that file changed.

---

## Open issues, flagged not fixed

1. **Two divergent copies of `quantumScarFunctions.py`** exist in the repo
   (`important/hpc/` and `important/paper/helper/`). The paper notebooks import
   one and the HPC scripts import the other. Only the `hpc` copy is patched
   here. Consolidate.
2. **Entropy is no longer normalised at all.** It was first divided by the
   unconstrained `(N/2) ln 2 = 4.159`, which exceeds the absolute maximum
   `ln 21 = 3.045` this blockade allows. The Haar value 2.472 is a property of
   the Hilbert space, not the Hamiltonian, and the Hamiltonian's own thermal
   entropy collapses under disorder (2.31 at x=y=z=0.01 to 0.18 at 1.0 —
   localisation), so no fixed denominator is honest. `S` is reported in nats.
3. **`R` is intensive** because of the bandwidth normalisation. The standard
   many-body quantum-battery claim is *superextensive* — power scaling faster
   than N. The current normalisation makes that claim invisible by
   construction. If a collective advantage is the point, also plot unnormalised
   power vs N.
4. **Global vs local ergotropy.** The chain's ergotropy needs a global
   many-body unitary; the qubits' needs only local ones. `qubit_metrics`
   already computes the qubits' passive state *per qubit*, which is the honest
   version, but a referee will still raise the asymmetry. Address it in text.
5. **`hy` sign convention.** The chain's y-disorder operator is `+σ_y` in the
   `(|0⟩,|1⟩)` string ordering, while the qubits use qutip's `σ_y`. Both are
   reproduced exactly as `quantumScarFunctions.py` has them, and for a single
   qubit the sign of `h_y` is a symmetry, so nothing is wrong — but the two
   models are not literally reading the same `h_y` vector in a common frame.

---

## Correctness

`validate_core.py` checks, in order:

1. Basis order and dimension against an independent enumeration (Lucas numbers
   L_N = 7, 18, 47, 123, 322, 843 — **not** Fibonacci; that is the open-chain
   count).
2. `H_clean`, `H0_dis`, `H1` elementwise against `reference_kron.py`, a full
   2^N Kronecker construction that imports nothing from `scarcore`. Agreement
   is to `4e-16`, and PXP leakage out of the blockade subspace is exactly 0.
3. Exact linearity of H in the disorder strengths.
4. RNG stream alignment (the common-random-numbers guarantee).
5. The propagator against a second integrator at `rtol = 1e-12`
   (`max|ΔR| = 1e-8`).
6. `R(0) = 0` and `0 ≤ R_deph ≤ R`.
7. **Everything against `quantumScarFunctions.py` itself** — basis order, Z2
   index, RNG stream, `H_clean`, `H0_dis`, `H1`, drive weights, `Hy`, both
   qubit blocks, and the E=0 scar. Measured agreement at N = 6, 8, 10, 12:
   worst deviation `3.6e-15`, zero-scar fidelity `1.000000000000`.

Check 7 uses the real qutip when installed. Where it is not, `qutip_shim`
supplies just enough `Qobj` arithmetic for that file's *builder* functions to
run — no solvers, no physics of its own — so the check is never silently
skipped.
