# What every file does, and why it exists

Read order if you are coming back to this cold: `scarcore.py` → `build_cache.py`
→ `main.py` → `rerank.py`. The rest are support.

---

## The pipeline (4 files you actually run or read)

### `scarcore.py` — the numerical core
**Does:** builds the constrained basis, assembles `H0_dis` and the drive
operator, evolves both models, computes every metric.

**Why it exists:** the old `test_main.py` rebuilt the basis, the flip pattern,
the disorder draws and a full dense diagonalisation *inside every objective
evaluation*. Only the time evolution is real work. This file separates the
part that never changes (structure) from the part that does (four scalar
strengths), so the first part can be computed once and cached.

**Why it has no qutip:** an API drift between qutip 4 and 5 has already cost
this project a whole allocation once (`get_qubit_ham` gained a third return
value; `main.py` unpacked two; the `except: return 1e9` swallowed it and DE ran
its full budget against a constant landscape). The physics no longer depends on
that package. qutip appears only in the equivalence check.

**Depends on:** numpy, scipy. Nothing else.

---

### `build_cache.py` — the preprocessing step
**Does:** two things, and also contains the loaders `main.py` uses.

1. `cache/struct_N{N}.npz` — the constrained basis, the single-site flip
   sparsity pattern, the clean deformed PXP Hamiltonian, and the clean scar
   tower including the E=0 scar.
2. `cache/seeds_N{N}.npz` — the unit disorder fields `v_z, v_y, v_x, v_w` for
   each seed.

**Why (1):** the E=0 scar costs a sparse LU factorisation plus repeated ARPACK
shift-invert solves. The old code paid for that in every worker process of
every array task. Once per N is enough.

**Why (2):** this is the one that matters. The disorder is drawn as
`strength × (a fixed uniform pattern)`, and `np.random.uniform(-d, d, N)`
consumes the *same* underlying deviates as `np.random.uniform(-1, 1, N)`, just
scaled. So for a fixed seed:

```
H0_dis(x, y, z) = H_clean + z·Dz + y·Ay + x·Ax      (exactly linear)
H1(dd)          = D1_base + dd·D1_pert              (exactly linear)
```

Assembling a Hamiltonian is now four sparse scalar-multiply-adds instead of
thousands of Python string operations plus RNG calls.

It is also what guarantees **common random numbers**: the scar model and the
qubit model read the *same* `v` arrays for a given seed, so a comparison at
fixed seed is paired, not a disorder lottery.

**Run it once per N before anything else.** `validate_core.check_linearity`
asserts the linearity, so if this ever stops being true the cache cannot go
stale silently.

---

### `main.py` — the DE driver
**Does:** runs `scipy.optimize.differential_evolution` on
`score = mean_seed[ max_t R_scar − max_t R_qubit ]`, logs every evaluation, then
re-runs the winner on a larger seed set with full curves and diagnostics.

**Key detail — the `Objective` class.** The old code set module globals in the
parent and relied on the DE worker pool being *forked* so children would
inherit them. scipy pickles the objective through the pool's task queue, and
the start method is not guaranteed to be fork. When that assumption broke here,
every worker silently ran with `tlist = None` and the whole run died inside the
pool. The configuration now travels *with* the objective as plain primitives;
the heavy cached arrays load lazily once per process. Correct under fork and
spawn.

**Outputs per island:**
- `data/N12/island_k/evals/evals_pid*.jsonl` — one line per evaluation, all
  metrics. This is the file `rerank.py` reads.
- `data/N12/island_k/final/island{k}_N12_result.npz` — full curves for the
  winner.
- `data/N12/island_k/final/island{k}_N12_summary.json` — the headline numbers.

**Flags worth knowing:** `--search-wd` and `--search-wq` promote the two drive
frequencies to search parameters (5 → 6 → 7 dimensions). `--objective-reals`
sets disorder realizations per evaluation.

---

### `rerank.py` — offline re-ranking
**Does:** pools every `evals_pid*.jsonl` from every island, deduplicates, and
re-sorts on any recorded metric. `--confirm K` re-evaluates the shortlist at K
realizations with a bootstrap confidence interval over seeds.

**Why it exists:** DE optimises `max_t R`, which can be won by heating or by
crippling the benchmark. Rather than pick a different objective and lose
comparability with earlier runs, every evaluation records the alternative
metrics too — so you can change your mind about the metric **without re-running
the search**. That is the whole point of the JSONL log.

It is also where the statistics happen. A DE optimum found at 8 realizations is
not evidence; a mean with a bootstrap CI over 32 realizations is.

---

## Validation (3 files — you run one, it uses the other two)

### `validate_core.py` — run this before trusting a job
Seven checks, in order:

1. Basis order and dimension against an independent enumeration. Dimension is
   the **Lucas number** `L_N` = 7, 18, 47, 123, 322, 843 — *not* Fibonacci;
   that is the open-chain count.
2. `H_clean`, `H0_dis`, `H1` elementwise vs `reference_kron.py`. Agreement
   `4e-16`, PXP leakage out of the blockade subspace exactly 0.
3. Exact linearity of H in the disorder strengths (the cache's foundation).
4. RNG stream alignment (the common-random-numbers guarantee).
5. The propagator against a second integrator at `rtol = 1e-12`
   (`max|ΔR| = 1e-8`).
6. `R(0) = 0` and `0 ≤ R_deph ≤ R`.
7. **Everything against `quantumScarFunctions.py` itself** — basis order, Z2
   index, RNG stream, `H_clean`, `H0_dis`, `H1`, drive weights, `Hy`, both
   qubit blocks, and the E=0 scar. Measured worst deviation at N = 6, 8, 10,
   12: `3.6e-15`. Zero-scar fidelity `1.000000000000`.

Check 7 is the guard against `quantumScarFunctions.py` drifting under the fast
path again. `job_cache.sh` runs it.

### `reference_kron.py` — the independent yardstick
Rebuilds H, H1 and Hy in the **full 2^N space** with explicit 2×2 matrices and
numpy Kronecker products, then restricts to the blockade subspace. Deliberately
slow and obviously correct. **It imports nothing from `scarcore`** — that is the
entire point. If it shared any of `scarcore`'s cleverness it would not be a
check.

Imported at module level by `validate_core.py`. Delete it and validation
crashes.

### `qutip_shim.py` — lets check 7 run without qutip
A minimal `Qobj` (sparse storage, arithmetic, `eigenstates`) plus `basis` and
the three Paulis. Enough for `quantumScarFunctions.py`'s *builder* functions to
execute. **No solvers, no physics of its own.**

On the cluster qutip is in your venv and this file is never touched.
`validate_core` prefers the real package and only falls back to the shim.

**Why not just skip the check when qutip is missing?** Because a check that
silently skips is a check you stop trusting. This one always runs.

---

## Physics reference (1 file)

### `quantumScarFunctions.py`
Your original file, unchanged except **one bug fix**:

`get_zero_scar` started ARPACK at `K = max(16, int(0.02*D))` with no clamp. At
N=4, D=7, and ARPACK requires `k < D−1`, so it raised. Now clamped to `D−2`
(and the doubling loop with it). `get_zero_scar(4)` works, overlap 0.376126.
This is the "N=4 needs an explicit k0" behaviour the test suite documented — it
was a missing clamp, not a real limitation.

Nothing else in this file changed. `scarcore.py` is checked against it, not the
other way round: **this file is the definition of the physics.**

---

## Cluster scripts (3 files, run in order)

| File | When | What |
|---|---|---|
| `job_cache.sh` | once per N | runs `validate_core.py`, then builds both caches. Short. |
| `job_calib.sh` | once, after cache | tiny DE run to measure seconds-per-evaluation *on this cluster*, so you can size `--time`. |
| `job_de.sh` | the real run | 8 independent DE islands as a job array, at most 4 at once. |

**Why islands and not one long run:** DE is a stochastic search. One run's
optimum is not evidence. Eight independent seeds let you see whether they agree,
and `rerank.py` pools all of their evaluations into one ranking.

**Why calibrate instead of guessing:** evaluation cost varies ~20× across the
search space and correlates 0.89 with the drive amplitude `ds` — the integrator
has to resolve the drive. Size walltime off the p90, not the median. Short jobs
also start sooner.

---

## Docs

| File | Contents |
|---|---|
| `README.md` | overview, why it is faster, the metric problem, open issues |
| `METRICS.md` | every metric defined, with the exact formula and what it catches |
| `FILES.md` | this file |

---

## Dependency graph

```
scarcore.py          numpy, scipy only
reference_kron.py    numpy only
qutip_shim.py        numpy, scipy only

build_cache.py    -> scarcore
validate_core.py  -> scarcore, reference_kron, [qutip | qutip_shim],
                     quantumScarFunctions
main.py           -> scarcore, build_cache, validate_core (startup check,
                     unless --skip-self-check)
rerank.py         -> main, build_cache (only with --confirm)
```

Nothing here is optional except `qutip_shim.py`, and that only if you never run
`validate_core.py` anywhere qutip is missing. Note its import is **not**
guarded: with qutip absent *and* the shim deleted, validation raises.

**Safe to delete:** `test_main.py`, `test_job.sh` — fully superseded. Caveat:
`scarcore.py` was proved equivalent to `quantumScarFunctions.py`, not to
`test_main.py` end-to-end. If `diff_evo` is not under git, keep them until a
cluster run reproduces a result you recognise.

**Not in your folder, created at runtime:** `cache/`, `data/`, `calib/`,
`logs/`, `__pycache__/`.
