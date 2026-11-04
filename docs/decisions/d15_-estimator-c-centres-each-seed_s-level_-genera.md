---
id: doc-u6h
title: 'D15: Estimator C centres each seed''s level; generators keep one input kind
  per seed'
type: decision
scope: internal
created_at: '2026-11-03T20:10:37.883662Z'
updated_at: '2026-11-03T21:07:05.412909Z'
---

- **Date:** 2026-11-03. **Status:** accepted (bead dreamcounter-0ea5b7).
- **Problem:** correct programs written the ordinary way were refused as `TooNoisy` by estimator C. The sorted-order and subsequence-count generators drew the kind of input (and the pair-sums threshold) per `(n, seed)`. A program whose cost depends on the kind therefore had step counts whose level changed between sizes. Measured with 5 seeds on 3 November 2026, with `n²` misfit and noise margin before the fix: pair_sums 0.028 and 0.082, close-steps quadratic 0.045 and 0.11, zigzag quadratic 0.054 and 0.134.
- **Decision:**
  1. `TaskRng` exposes the seed (`seed()`) and a stream from the seed alone (`seed_stream()`).
  2. Sorted-order and subsequence-count take kind `seed mod 3` at every size. Every other choice that sets a seed's cost level is drawn once per seed: the sign of a one-sign input, the offset of close steps and zigzag, and the first sign of the runs of sign changes. Only elements, which elements are 0, and run lengths are drawn at each size.
     - Pair sums: seed `k` takes plan `(k / 3) mod 6` from its kind's row of a table. A plan is exact (a pair sum) or random, placed at the bottom, the top or a quantile drawn from the seed stream. Each row holds all six plans, so the plan does not follow the kind. The default five seeds give: tied exact at a quantile (several pairs sum to it), spread exact at a quantile, wide `-2 top` (answer 0), tied random, spread `2 top` (every pair). They have no wide exact threshold and no wide answer strictly between 0 and every pair.
     - Sign changes: bound `m = 3 seed mod 5`. Over seeds 0 to 14 every kind gets every bound. The default five give uniform inputs 0 and 4, half-zero inputs 3 and 2, and runs of signs 1.
     - With five seeds, a kind has at most two seeds, so fixing choices per seed narrows the mix of regimes the default sweep sees. Before, 30 inputs mixed them freely.
  3. `Sweep::in_seed_order` records that step `k` at every size comes from seed `k`. The harness builds its sweeps this way.
  4. On such sweeps, C subtracts each seed's mean `ln` level over the window before it takes medians and estimates noise. The noise variance at each size is multiplied by `N/(N−1)`, so it is unbiased when the seeds' noise is independent. C's noise is the largest square root over the sizes, and centring makes the sizes' variances vary less. So with few seeds the centred noise is slightly below the per-size noise: measured over 3000 sweeps, about 0.98 of it at two seeds, 0.99 at five, and close to 1 at forty. Adding a constant to every median leaves every misfit unchanged, so a seed's level offset says nothing about the shape of growth.
  5. Centring depends on the sweep, not the noise model: under `NoiseModel::Exact` and `NoiseModel::Known` a sweep in seed order is also fitted on centred medians.
  6. The family tests' check on each seed alone uses the noise of one seed measured after the same centring. Otherwise seeds that differ only in level would make that check lenient.
- **Result:** pair_sums and zigzag quadratic natural solutions are now named `n²` (misfit 0.0045 and 0.0032). pair_sums needs both halves: without centring its misfit is 0.011, and with kinds and plans drawn per size it is 0.029. Zigzag quadratic needs the generators only. Close-steps quadratic has misfit 0.0112 with margin 0.033, just above the tolerance; drawing its offset once per seed did not change that, so its test only prints the verdict. Insertion sort is still refused: its noise is within each input.
- **Rejected:**
  - Mixing every kind inside one input: this loses the coverage of least gaps and thresholds that the kinds exist for.
  - More seeds: about 350 are needed for a seed spread of 0.05.
  - Widening the tolerance by the noise: this loses the separation of `n log n` from `n^1.135`.
  - Centring every sweep: a sweep whose counts are in any order, for example sorted, would hide real noise.
- **Follow-up:** the private held-out candidates need the same generator change (bead filed 2026-11-03).
