---
id: doc-m4a
title: Results log
type: reference
scope: internal
created_at: '2026-10-27T17:11:14.071510Z'
updated_at: '2026-10-27T22:51:49.958762Z'
---

Dated entries for every experiment run, including failed and null results. Each
entry gives the command that reproduces it, the config and seed, and the raw numbers
before interpretation.

No experiments have been run yet. Phase 1 produced documents only (now in DOCA).

---

## 2026-10-27: Prior-work gate (literature check, not an experiment)

- **Question:** has cost-aware library learning already been done (2024 to 2026)?
- **Method:** about fifteen web searches across arXiv, ACL Anthology, PLDI/EGRAPHS
  2026 and ICLR 2026, plus reading the closest hits in full.
- **Result:** no work found that learns a library with program execution cost in the
  selection objective. Nearest neighbour: AlgoSkill (arXiv 2606.29999), which uses
  complexity in an RL reward over a *fixed, hand-designed* skill library.
- **Limits:** keyword search, not a systematic review. A forward-citation crawl from
  LILO and Stitch would raise confidence.
- **Details:** the literature notes §1.


---

## 2026-10-27: First evaluator throughput baseline

This baseline is the reference that later evaluator changes are measured against (spec §11).

- **Command:** `cargo build --release -p dc-bench && target/release/dc-bench` (defaults: 16 threads, 5 repetitions, 0.5 s batch, 0.3 s warm-up).
- **Commit:** `340b9ce` on branch `p2.14-bench`, on top of the unmerged prelude branch `p2.11-prelude`. Bead dreamcounter-2fc11c.
- **Build profile:** `release` (opt-level 3, `debug = 1`), default `RUSTFLAGS`.
- **Machine:** AMD Ryzen 7 2700 Eight-Core Processor, 8 cores, 16 threads, 31 GiB RAM.
- **Load:** the machine was shared with other agents. Load average before the three runs: 7.30, 11.73, 15.46. After: 11.08, 15.46, 15.61. Peak RSS of one run: 118 MB.
- **Inputs:** fixed. `sum` and `max_subarray`: SplitMix64 seed 1, integers in [-1000, 1000]. `msort`: seed 2, integers in [0, 10⁶). `bsearch`: the array `[0; 3; 6; ...]`, target at index 7n/10, one search per evaluation.
- **Checks:** inside the timed loop, every evaluation's result is compared with a Rust reference, and its step count with the first run's.
- **Method:** each rate is the median of 5 timed batches. The 16-thread rate is all threads' evaluations over the wall time from the first thread's start to the last thread's finish. Each cell is the median of three runs of the benchmark.

| program | n | steps/eval | 1T evals/s | 1T Msteps/s | 16T evals/s | 16T Msteps/s | 16T/1T |
|---|---|---|---|---|---|---|---|
| sum | 10000 | 40001 | 125.7 | 5.03 | 908.2 | 36.33 | 7.2 |
| max_subarray | 10000 | 150003 | 22.8 | 3.43 | 190.9 | 28.63 | 8.4 |
| msort | 1000 | 290817 | 8.6 | 2.50 | 65.1 | 18.93 | 7.6 |
| fib_naive | 20 | 142292 | 24.7 | 3.51 | 185.1 | 26.33 | 7.5 |
| fib_memo | 60 | 661 | 4577.5 | 3.03 | 37691.8 | 24.91 | 8.2 |
| bsearch | 100000 | 248 | 9823.5 | 2.44 | 79405.4 | 19.69 | 8.1 |

Spread across the three runs, in evals/s (min to max, 1T then 16T):

- `sum`: 122.8 to 127.5; 828.9 to 921.1.
- `max_subarray`: 22.3 to 23.5; 177.9 to 200.5.
- `msort`: 6.5 to 9.8; 56.7 to 71.6.
- `fib_naive`: 23.3 to 26.3; 183.5 to 186.3.
- `fib_memo`: 4050.2 to 4974.2; 33531.6 to 38529.1.
- `bsearch`: 9822.9 to 10420.5; 73831.3 to 80552.3.



---

## 2026-10-27: Reference programs of known complexity through the growth estimators

- **Command:** `cargo test -p dc-tasks --test reference_growth -- --nocapture` (lines tagged `RESULT`). Per-seed check on the default sweep: `cargo test --release -p dc-tasks --test reference_growth -- --ignored`.
- **Commit:** `20f01fe` on branch `p3.3-reference-programs`. Bead dreamcounter-ed8988; follow-up dreamcounter-db440d.
- **Config:** harness defaults (5 seeds, seeds 0..4, fuel 10x reference, cap 5e7). Fit window: upper half. Sweeps: `ref-sum` 16..1024 (doubling), `ref-sort` 16..512 (doubling), `zero-subsets` n = 4, 6, ..., 14. Insertion sort is measured under merge sort's fuel plan.
- **Commit of the merged version:** `aca401f` on main (PR 16). The numbers below were measured at `20f01fe`; the review fix changed only how ideals are computed, and the fitted values are unchanged.
- **Classification rule:** estimator A's slope is compared with the closed-form least-squares slope of each exact series (n, n log n, n², 2ⁿ) over the same fit window ("ideal"), computed in the test independently of A. A separate check requires A to recover each closed-form slope on the exact series within 1e-3. Pass: nearest ideal is the right class and the distance is at most a quarter of the gap to the nearest other class. B: right γ, α within 0.1 of the class exponent. "Drift" is A's slope over the last three sizes minus over the first three.

| program | expected | A α̂ (R²) | A ideal / gap | B γ, α | B γ̂, stability vs seed spread | B γ per seed | drift |
|---|---|---|---|---|---|---|---|
| sum_ref | n | 0.9992 (1.000000) | 1.0000 / 0.171 | 0, 0.9992 | -0.026, 0.0037 vs 0 | 0,0,0,0,0 | 0.008 |
| msort | n log n | 1.1538 (0.999989) | 1.1813 / 0.181 | 1, 0.9726 | 0.597, 0.00038 vs 0.0097 | 1,1,0,0,1 | -0.073 |
| isort | n² | 1.9647 (0.999949) | 2.0000 / 0.819 | 0, 1.9647 | -2.134, 0.0104 vs 0.119 | 0,0,2,0,0 | 0.209 |
| zero_subsets | 2ⁿ | 8.2220 (0.997673) | 8.2210 / 6.221 | 0, 8.2220 (no 2ⁿ class) | -50.5, 0.059 vs 0 | 0,0,0,0,0 | 4.227 |

Default sort sweep 16..1024 (release, ignored test): msort A 1.1456, B γ 1 (γ̂ 0.784, stability 0.0020), per seed 1,1,1,1,1. isort A 1.9768, B γ 0 (γ̂ -1.412, stability 0.0135), per seed 0,1,0,0,0. Merge sort to 4096: per seed γ 1 throughout (γ̂ 0.71 to 0.80).

- **Result:** on the medians, every program classifies correctly under both estimators. A separates all four classes on these sweeps.
- **Weaknesses found:** (1) B's γ is not robust to the spread between seeds. Single seeds flip γ for both sorts, and B's own stability radius is below the seed spread, so the right answer on medians is not guaranteed. (2) No estimator names 2ⁿ: A reports α̂ ≈ 8.2, which on one window is indistinguishable from n⁸. Only the drift across the sweep (+4.2, against |drift| ≤ 0.21 for the polynomials) shows super-polynomial growth. Spec §9's estimator C is not built. Both are in bead dreamcounter-db440d.
