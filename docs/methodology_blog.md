# Why We Pair Frequentist Gates with Bayesian ROPE in Checkout A/B Testing

Shipping a checkout redesign on a single p-value leaves product teams exposed to two opposite failure modes: trusting corrupted randomization, or shipping a statistically significant lift that is too small to matter in production. In our evaluation of a redesigned checkout flow across **39,880 validated users** (`19,850` Control, `20,030` Treatment), combining Frequentist validity gates with Bayesian decision metrics resolved both problems cleanly.

## 1. Validate Telemetry Before Looking at Lift

Raw clickstream logs rarely arrive clean. Before estimating conversion rates, the pipeline (`src/data_pipeline.py`) enforces strict Pydantic v2 row contracts (`UserEventRecord`, `extra="forbid"`, `allow_inf_nan=False`) and removes `720` corrupted records out of `40,600` raw events (`98.23%` retention):
- **241** mixed-group user events (users exposed to both arms)
- **239** duplicate events (retaining earliest timestamp)
- **120** invalid/negative revenue rows
- **120** non-monotonic funnel rows (for example, `clicked_cta=True` when `viewed_landing=False`)

Next, a chi-square Sample Ratio Mismatch (SRM) test at `alpha = 0.001` checks whether traffic split matches the intended `50/50` allocation (`19,850` vs. `20,030`; `chi2 = 0.8124`, `p = 0.3674`). In `src/report.py`, any SRM failure forces an automatic **`HOLD`** decision regardless of observed conversion lift.

## 2. Why Two Statistical Lenses Beat One

| Question Stakeholders Ask | Frequentist Tool | Bayesian Tool |
|---|---|---|
| Was the sample large enough beforehand? | `NormalIndPower` (`10,323`/arm required for 12% MDE at 80% power; **97.34%** achieved) | Prior-to-posterior precision |
| Could a `1.84 pp` lift be random noise? | Two-proportion z-test (`z = 5.98`, `p = 2.24e-09`) + Newcombe-Wilson 95% CI `[+1.24, +2.45] pp` | `P(Treatment > Control) = 100.00%` |
| Is the lift large enough to justify rollout cost? | Minimum Detectable Effect (`+19.03%` rel. vs. 12% target) | `P(> ROPE)` (`100.00%` above `+/-0.50 pp` practical margin) + `0.0000%` expected CR loss |
| Are subgroup wins real or multiple-testing noise? | Benjamini-Hochberg FDR (`fdr_bh`) across 7 device/channel slices | — |

Using Newcombe-Wilson intervals avoids normal-approximation distortion near `0` or `1`, while pairing Welch's t-test (`p = 3.06e-08`) with Mann-Whitney U (`p = 1.44e-09`) handles right-skewed revenue per user (`$5.99 -> $7.21`, `+$1.22` RPU). On the Bayesian side, checking three priors—Weak Uniform `Beta(1, 1)`, Informative `Beta(10, 90)`, and Skeptical `Beta(20, 180)`—confirms the relative lift posterior (`+18.88%` to `+19.06%`) is driven by the data rather than prior assumptions.

## 3. Subgroup Heterogeneity and Trade-Offs

Applying Benjamini-Hochberg FDR control across all seven subgroups prevents cherry-picking:
- **Significant gains:** Email (`+27.50%`, BH `p = 0.0004`), Mobile (`+25.02%`, BH `p = 1.43e-05`), Organic (`+19.34%`, BH `p = 0.0003`), Desktop (`+18.36%`, BH `p = 0.0001`), and Paid Search (`+16.16%`, BH `p = 0.0082`).
- **No detected lift:** Tablet (`+2.75%`, BH `p = 0.7681`) and Social (`+13.77%`, BH `p = 0.1439`).

**Trade-offs & Limitations:**
1. **Fixed-horizon assumption:** Classical two-proportion z-tests assume no unadjusted interim peeking; continuous monitoring would require group-sequential alpha spending or always-valid mSPRT.
2. **Short observation window:** A 14-day window cannot capture novelty decay or repeat-purchase behavior; post-launch monitoring across two weekly cycles remains necessary.
3. **Deterministic simulation:** The dataset is synthetically generated (`seed=42`) to model realistic checkout funnel dynamics and telemetry defects reproducibly.

## 4. Decision Outcome

With SRM passing, `97.34%` statistical power, `p = 2.24e-09`, and `100.00%` posterior probability above the practical ROPE threshold, the recommendation is **`SHIP`**, projected at **`+1,660` incremental conversions/month** and **`+$109,603`/month** at `3,000` daily users.
