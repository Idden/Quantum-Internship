# Every metric, defined

Notation: $H_0$ is the disordered Hamiltonian (`H0_dis` for the chain, the
block-diagonal sum for the qubits), $\{E_n\}$ its eigenvalues ascending,
$|\psi(t)\rangle$ the state, $W = E_{\max} - E_{\min}$ the bandwidth.

---

## 1. `R(t)` — normalised ergotropy

$$R(t) \;=\; \frac{\langle H_0\rangle(t) - \langle H_0\rangle(0)}{W}$$

**Ergotropy** is the maximum work extractable from a state by a *unitary*:

$$\mathcal{E}(\rho) \;=\; \mathrm{Tr}(\rho H_0) \;-\; \min_{U}\,\mathrm{Tr}\!\left(U\rho U^{\dagger}H_0\right)$$

The minimiser is the **passive state** — the state you cannot extract anything
more from.

Here the evolution is unitary and the initial state is the *ground state* of
$H_0$, so $|\psi(t)\rangle$ stays **pure**. The passive state of a pure state is
the ground state (any pure state can be unitarily rotated onto it), so
$\min_U \mathrm{Tr}(U\rho U^\dagger H_0) = E_{\min}$ and

$$\mathcal{E}(t) \;=\; \langle H_0\rangle(t) - E_{\min} \;=\; R(t)\,W .$$

**So $R$ is not "absorbed energy" — it is exactly the normalised ergotropy**,
and $R(0)=0$ says the battery starts genuinely empty (passive). That sentence
belongs in the manuscript; it costs nothing and it is the strongest statement
available about the metric.

**The catch.** $R$ has a *dead value*. The disordered spectra are nearly
symmetric about $E=0$, so a state heated to infinite temperature has
$\langle H_0\rangle \approx 0$ and therefore

$$R_\infty \;=\; \frac{0 - E_{\min}}{W} \;\approx\; 0.5\text{–}0.63 .$$

That is reached by **pure heating with no scar physics at all**. $R \approx 0.55$
is not a result. This is why the metrics below exist.

Code: `scarcore.scar_metrics`, `scarcore.qubit_metrics`.

---

## 2. `R_deph(t)` — dephased ergotropy ← the important one

Let $p_n(t) = |\langle E_n|\psi(t)\rangle|^2$ be the level populations. The
**dephased state** keeps those populations and throws away every coherence:

$$\rho_{\rm deph}(t) \;=\; \sum_n p_n(t)\,|E_n\rangle\langle E_n| .$$

For a state diagonal in the energy basis, the passive state is obtained by
**pairing the largest population with the lowest energy** — sort populations
descending, energies ascending. So

$$R_{\rm deph}(t) \;=\; \frac{\displaystyle\sum_n p_n E_n \;-\; \sum_n p_n^{\downarrow}E_n^{\uparrow}}{W}$$

where $p^{\downarrow}$ is the populations sorted descending and $E^{\uparrow}$
the energies ascending.

### This metric has a standard name

$R_{\rm deph}$ is the **incoherent ergotropy** of Francica, Binder, Guarnieri,
Mitchison, Goold and Plastina, *Quantum Coherence and Ergotropy*, PRL **125**,
180603 (2020) — the ergotropy of the dephased state $\Delta(\rho)$. Their
decomposition is

$$\mathcal{E}(\rho) \;=\; \underbrace{\mathcal{E}\big(\Delta(\rho)\big)}_{\text{incoherent}} \;+\; \underbrace{\mathcal{E}(\rho) - \mathcal{E}\big(\Delta(\rho)\big)}_{\text{coherent}}$$

so in our variables $R = R_{\rm deph} + R_{\rm coh}$: the dephased score is the
incoherent part, and the gap $R - R_{\rm deph}$ is the part of the work that
exists *only* because of coherence between energy eigenstates. Cite this rather
than presenting it as an ad-hoc diagnostic.

### What it means physically

$R$ assumes you can apply *any* unitary — including one that exploits the phase
relationships between energy eigenstates. That requires knowing those phases and
acting faster than they scramble. $R_{\rm deph}$ is what survives if you cannot:
the work still extractable once the coherences are gone. For a many-body battery
that is the honest number.

Always $0 \le R_{\rm deph} \le R$ (`validate_core` asserts it).

**Nothing actually dephases.** The state stays pure for the whole simulation;
there is no decoherence in the model. $\Delta(\rho)$ is a *counterfactual* — the
state you would be left with if the energy-basis coherences were unusable — so
$R_{\rm deph}$ is a **lower bound** on extractable work under a restricted class
of operations, not a prediction about what happens. That framing is what makes
it defensible: you are not claiming decoherence occurs, you are reporting the
work that does not depend on phase control.

**It is also not "no entanglement".** $\Delta(\rho)$ is diagonal in the
*eigenbasis of $H_0$*, and each $|E_n\rangle$ is itself a highly entangled
many-body state. A mixture of mid-spectrum eigenstates typically has *more*
spatial entanglement than the pure state it came from. $R_{\rm deph}$ and $S$
measure different things in different bases and are independent diagnostics.

### Why it separates charging from heating

Take the two extremes.

**Infinite temperature**: $p_n = 1/D$ for all $n$. Then
$\sum_n p_n E_n = \bar{E} \approx 0$, and the passive rearrangement of a
*uniform* distribution is the same uniform distribution, so
$\sum_n p_n^{\downarrow}E_n^{\uparrow} = \bar E$ too. Hence

$$R_{\rm deph} = 0 \qquad\text{while}\qquad R \approx 0.55 .$$

**A fully thermalised state has zero dephased ergotropy.** All of its apparent
$R$ is the dead value.

**Coherent charging**: population concentrated in a few *high*-energy levels.
The passive rearrangement moves that weight down to the bottom of the spectrum,
which is a big change, so $R_{\rm deph}$ stays close to $R$.

Measured in this project:

| state | $R$ | $R_{\rm deph}$ | surviving |
|---|---|---|---|
| coherent peak, $t=2.15$ | 0.822 | 0.651 | **79%** |
| same run at $t=100$ | 0.503 | 0.154 | 31% |
| thermal peak, $t=101$ | 0.618 | 0.289 | 47% |

Code: `scarcore.ergotropy_curves`.

---

## 3. `S` — half-chain entanglement entropy

Yes, entanglement entropy. Cut the ring into $A$ = first $N/2$ sites and
$B$ = the rest, reshape the coefficient vector into the matrix $C_{AB}$, take
its singular values $\sigma_k$, set $\lambda_k = \sigma_k^2$ (the Schmidt
eigenvalues, i.e. eigenvalues of $\rho_A$), and

$$S \;=\; -\sum_k \lambda_k \ln \lambda_k .$$

**What it is for here:** it is the independent witness that the charge is
*scar-mediated* rather than thermal. Scar dynamics is famously low-entanglement
— that is what makes the revivals possible. A thermalised state has near-maximal
entanglement.

### Reported in nats, not as a ratio

Every candidate denominator was wrong or misleading, so there is no longer one.

| reference at $N=12$ | $S$ |
|---|---|
| $\tfrac{N}{2}\ln 2$ (naive, unconstrained) | 4.159 |
| **absolute maximum**, $\ln 21$ | **3.045** |
| Haar-random state in the constrained space | 2.472 ± 0.026 |
| mid-spectrum eigenstates, weak disorder | ~2.31 |
| mid-spectrum eigenstates, $x=y=z=1$ | ~0.18 |

Three problems with dividing:

1. $\tfrac{N}{2}\ln 2 = 4.159$ **exceeds the absolute maximum** $\ln 21 = 3.045$
   that the blockade allows. It is unreachable, not merely loose.
2. The Haar value 2.472 is a property of the Hilbert space, not of the
   Hamiltonian.
3. The Hamiltonian's own thermal entropy **collapses under disorder** — 2.31 at
   $x=y=z=0.01$ down to 0.18 at 1.0, which is localisation. A fixed denominator
   reads "coherent" for a state that is actually localised.

So `S_at_t1` and `S_at_tmax` are reported in **nats**. The reference values above
print alongside as context; nothing is divided by them.

Rule of thumb at $N=12$: $S \lesssim 0.5$ nats is coherent, $S$ near 2.3–2.5 is
thermal — but check the disorder strength before reading a low $S$ as coherence.

### When it is measured

At **two times per objective evaluation**, not at $t=0$ and $t=t_{\rm end}$:

- `S_at_t1` — at the first peak $t_1$
- `S_at_tmax` — at the time of the maximum of $R$, which is the one that matters,
  because it tells you what kind of state produced the score

That is 2 SVDs instead of 1601. The **final** evaluation (`final_evaluation` in
`main.py`) does compute the entropy at *every* time point, for the full
$S(t)$ trace saved in the npz as `vn_entropy`, shape `(n_seeds, nt)`.

Code: `scarcore.half_chain_entropy`. Recorded at two times per evaluation
(`S_at_t1`, `S_at_tmax`), not all of them — 2 SVDs instead of 1601.

---

## 4. `R_1`, `t_1` — the first peak

Height and time of the **first local maximum** of $R(t)$: the first coherent
charging event, before anything has had time to thermalise.

$\max_t R$ collapses the entire trace to one number and cannot tell a fast
coherent peak from a late thermal plateau. $R_1$ can. In the earlier review one
candidate peaked at $t=2.1$ with $S/S_{\max}=0.05$ and another at $t=101$ with
$S/S_{\max}=0.30$ — $\max_t R$ scored them as the same *kind* of thing.

Code: `scarcore.first_peak`.

---

## 5. `P_max` — charging power

$$P_{\max} \;=\; \max_t \frac{R(t)}{t}$$

The best *average* charging rate over any window starting at $t=0$. A battery
that reaches $R=0.8$ at $t=2$ is a better battery than one that reaches $R=0.85$
at $t=100$, and $\max_t R$ cannot see that.

Note: in the wider search run here the chain **loses** on power at the coherent
points ($-0.14$ to $-0.20$). Report it honestly; it is a real finding.

Code: `scarcore.max_power`.

---

## 6. `refined_max` — why the maximum is corrected

The maximum of a *sampled* oscillatory curve is biased downward, and the bias
differs between the two models because their frequency content differs. On the
old $dt = 0.5$ grid that was worth up to $0.033$ — the same size as the effect
being searched for.

Two fixes: `--nt` defaults to 1601 ($dt = 0.125$), and a parabolic correction is
fitted through the three samples around the argmax. Output points are cheap; the
integrator's own step control does not depend on `--nt`.

**It is per realization, not across them.** `refined_max` takes ONE seed's
$R(t)$ curve and corrects that curve's own sampled maximum. Seeds are combined
only afterwards, in §7. It has nothing to do with maximising over disorder
realizations.

Code: `scarcore.refined_max`.

---

## 7. How seeds are combined — read this one carefully

**Per seed $s$**, both models are run on the *same* disorder fields (common
random numbers), and a per-seed score is formed:

$$\text{score}_s \;=\; \max_t R_{\rm scar}^{(s)}(t) \;-\; \max_t R_{\rm qubit}^{(s)}(t)$$

**Then** the seeds are averaged:

$$\text{score} \;=\; \frac{1}{N_s}\sum_s \text{score}_s$$

So it is **mean of per-seed differences of per-seed maxima** — a *paired*
comparison. Pairing is what makes the difference far less noisy than either
maximum separately, and it is why the seeding scheme matters.

**Every other metric is combined the same way**: `maxR_scar` is the mean over
seeds of each seed's own maximum, `S_at_tmax` is the mean over seeds of the
entropy at *that seed's* peak time, and so on. `tmax_scar` is a mean of argmaxes
— fine as a diagnostic, but do not over-read it if the seeds peak at genuinely
different times.

### Difference from the old code — this changes reported numbers

The old npz reported

```
max_Rtau_scar_mean = max( mean_over_seeds( R(t) ) )     # max of the mean curve
```

which is **not** the same as

```
maxR_scar          = mean_over_seeds( max_t R(t) )      # mean of the maxima
```

Averaging curves first washes out peaks that occur at different times in
different realizations, so `max-of-mean ≤ mean-of-max`. The old code's *DE
objective* used mean-of-max (same as now) but its *reported* headline number was
max-of-mean. If you compare against an old npz, compare like with like.

The new npz stores the **full per-seed curves** — `R_scar` has shape
`(n_seeds, nt)` — so you can compute either from the same file.

### How many seeds

At one realization the score is dominated by disorder noise: across 8 seeds at
fixed parameters the spread exceeded the effect, and only 4–6 seeds had the
right *sign*. DE with `--objective-reals 1` returns whichever point won the
disorder lottery. Use 8 for the search, then `rerank.py --confirm 32` for the
number you quote, which also gives a bootstrap CI over seeds and the fraction of
seeds with the right sign.

---

## 8. The two composite scores in the logs

| key | definition |
|---|---|
| `score` | $\overline{\max_t R_{\rm scar} - \max_t R_{\rm qubit}}$ — what DE optimises |
| `score_deph` | same, with $R_{\rm deph}$ |
| `score_first` | same, with $R_1$ |
| `score_power` | same, with $P_{\max}$ |
| `deph_fraction_scar` | $R_{\rm deph}/R$ at the chain's peak — how much survives dephasing |

Export any ranking to a readable file:

```
python rerank.py --run data/N12 --rank-by score_deph --thermal-cut 0.15 \
                 --top 20 --report results/candidates
```

writes `candidates.csv` (every column, opens in Excel/pandas),
`candidates.md` (a paste-ready markdown table) and `candidates.json`.

---

## 9. Two ways the objective can be gamed

Both were observed in the calibration run. Check for both before believing a
score.

**Heating.** A positive `score` with $S/S_{\max} \gtrsim 0.4$ and
`score_deph` $\approx 0$ is the dead value, not a charged battery.

**Crippling the benchmark.** `score` is a *difference*, so it also goes up if
the qubits do worse. The DE's best point had `maxR_scar = 0.427` against
`maxR_qubit = 0.165` — it had turned the drive amplitude down to $d_s = 0.118$
so the qubits barely charged at all.

**Therefore: never quote `score` alone.** Always report `maxR_scar` in its own
right, `maxR_qubit` in its own right, and $S/S_{\max}$. All three are in the
JSONL for every evaluation ever run.

---

## 10. Normalisation — the open question

$R$ divides by the bandwidth $W$. Since $W$ grows roughly linearly with $N$,
$R$ is **intensive** (size-independent) by construction.

- **extensive** — grows $\propto N$ (total energy, total ergotropy)
- **intensive** — independent of $N$ (temperature, energy per site, $R$)
- **superextensive** — grows *faster* than $N$

The normalisation is doing real work: the chain and the $N$ qubits have
different bandwidths, so comparing raw ergotropies is comparing two batteries of
different capacity. Dividing by $W$ is what makes "fraction of capacity charged"
a fair comparison.

The cost is that any *collective* advantage that shows up as scaling in $N$ is
invisible in $R$ by construction. This is a genuine trade-off, not a mistake —
see the discussion note in the chat thread rather than assuming either choice is
simply better.
