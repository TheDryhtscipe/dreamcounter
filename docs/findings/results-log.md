---
id: doc-m4a
title: Results log
type: reference
scope: internal
created_at: '2026-10-27T17:11:14.071510Z'
updated_at: '2026-11-15T21:40:54.849113Z'
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



---

## 2026-10-28: Forward-citation crawl from LILO and Stitch (literature check, not an experiment)

- **Question:** does any paper that cites LILO or Stitch learn a library with the runtime cost or asymptotic growth of the programs in its objective? Bead dreamcounter-c69fcd.
- **Method:**
  1. Semantic Scholar Graph API, `paper/arXiv:<id>/citations` (limit 1000, one page each) for LILO (arXiv 2310.19791) and Stitch (arXiv 2211.16605).
  2. OpenAlex `cites:` filter on both works, to catch papers Semantic Scholar misses.
  3. Every citing paper triaged by title and abstract. A regular-expression pass over all abstracts for runtime terms (asymptotic, running time, runtime, execution time, time complexity, program cost, efficient programs) checked the manual triage.
  4. Full text retrieved for the papers that touch cost, efficiency, runtime or complexity, and the sections on cost read.
  5. One hop: Semantic Scholar citations of each close hit, triaged by title.
- **Not used:** Google Scholar "cited by" (it blocks scripted access). Semantic Scholar gives LILO 61 citations; Google Scholar may list more.
- **Counts:**
  - Semantic Scholar: 61 papers cite LILO, 93 cite Stitch, 21 cite both, 133 unique.
  - OpenAlex: 33 cite Stitch and 2 cite LILO. They add 5 papers not in Semantic Scholar, plus the EACL 2026 version of a paper already counted.
  - Triaged: all 138 by title; 122 also by abstract. 16 had no abstract to hand: 11 in Semantic Scholar and the 5 OpenAlex-only papers, whose titles are outside the topic except one. That one, a 2026 AGI conference chapter titled "Resource-Bounded Incremental Induction: A Principled Framework for Continual Learning" (doi 10.1007/978-3-032-33195-3_3), could not be read: it is behind a bot check and has no abstract in Crossref, OpenAlex or Semantic Scholar.
  - Full text read on the cost sections: 10 papers. These are Cobbler, AMaze, Achille and Soatto, Zhou et al., ReaComp and Token Sugar (all below), plus AbstractBeam arXiv 2405.17514, TroVE arXiv 2401.12869, DreamProver arXiv 2604.26311 and Zhao et al. arXiv 2603.23244, none of which mention program runtime).
  - One hop: 20 citing papers of the close hits, triaged by title. None is close.
- **Close hits, and how each differs from the claim:**
  - **Cobbler** (Lubin et al., "Equivalence by Canonicalization for Synthesis-Backed Refactoring", PLDI 2024, doi 10.1145/3656453). Rewrites direct-style code into calls to a *fixed* library (Elm combinators, NumPy). Speed-up is measured after the fact (median 1.95x at input size 10⁸ for NumPy). The library is not learned, and cost is not in the objective; the objective is equivalence.
  - **AMaze** (Ye et al., "Accelerating Syntax-Guided Program Synthesis by Optimizing Domain-Specific Languages", PACMPL 2026, doi 10.1145/3776679). Changes the DSL to minimise *synthesiser* time. The paper sets itself against library learning, which it describes as minimising program size. The cost is search time, not the runtime of the synthesised programs.
  - **Achille and Soatto**, "AI Agents as Universal Task Solvers: It's All About Time" (arXiv 2510.12066). Theory: learning should minimise the time to solve new tasks, and compression by Occam's razor is the wrong target. It supports our motivation, but its "time" is the solver's inference time, and it proposes no library-learning algorithm.
  - **Berlot-Attwell et al.**, "Is This LLM Library Learning? Evaluation Must Account For Compute and Behaviour" (EACL 2026, doi 10.18653/v1/2026.eacl-long.163; earlier as "LLM Library Learning Fails: A LEGO-Prover Case Study"). Cost means the compute of the learning system, used to make evaluation fair. Program runtime is not considered.
  - **Zhou et al.**, "Path-dependent program induction under resource constraints explains human sequence learning" (arXiv 2606.20623). A library learner with a memory cost (description length) and a computational cost. The computational cost is a search budget, not program runtime.
  - **Efficiency of the learner or the LLM, not of the programs:** ReaComp (arXiv 2605.05485, LLM token cost), Token Sugar (arXiv 2512.08266, token count of mined code patterns), babble (arXiv 2212.04596, compression speed), Stevie (arXiv 2308.08334, learning time), Aulile (Rahman et al., TMLR 2024, auxiliary distance-to-goal function).
- **Result:** no paper in the citation neighbourhood of LILO and Stitch learns a library with program execution cost or asymptotic growth in the selection objective. The claim that cost-aware library learning is open still stands. The nearest work optimises synthesiser time (AMaze), rewrites into a fixed fast library (Cobbler), or argues for time over compression without an algorithm (Achille and Soatto).
- **Limits:** Semantic Scholar and OpenAlex coverage only. One paper (the resource-bounded induction chapter) is unread; from its title it most likely bounds the learner's resources, but this is not checked.



---

## 2026-10-28: Estimator C (two-term class model) on the reference programs

- **Command:** `cargo test -p dc-tasks --release --test reference_growth -- --include-ignored --nocapture` (lines tagged `RESULT`).
- **Commit:** `4a5368d` on branch `db440d-class-select` (PR 19). Bead dreamcounter-db440d.
- **Config:** `ClassSelect::default()`: whole sweep, tolerance 0.01 (absolute, `ln` units), lower-order ratio ρ = 1, noise margin 3 × the largest standard error of a median, noise from the seeds. Each seed alone is fitted with `NoiseModel::Known(s)`, where `s` is the largest standard deviation of one seed's `ln` steps at a size, measured on the full sweep. Harness defaults otherwise (5 seeds, seeds 0..4). Sweeps: `ref-sum` 16..1024, `ref-sort` 16..512 and the default 16..1024, `zero-subsets` n = 4, 6, ..., 14.
- **Model:** each class `f` is fitted as `c · (f(n) + r · g(n))`, with `g` its lower-order term (`1` for `log n` and `n`; `n` for `n log n`, `n²` and `2ⁿ`; `n²` for `n³`). `r` is bounded so that `−½ f ≤ r g ≤ ρ f` at every size.

| program | sweep | verdict on medians | misfit (right class) | noise margin | per seed |
|---|---|---|---|---|---|
| sum_ref | 16..1024 | n | 0.0000 | 0.0000 | 5 of 5 named n, misfit 0.0000 |
| msort | 16..512 | n log n | 0.0015 | 0.0122 | 5 of 5 named n log n, misfits 0.0048, 0.0058, 0.0019, 0.0028, 0.0074 |
| msort | 16..1024 | n log n | 0.0017 | 0.0122 | 5 of 5 named n log n, misfits 0.0052, 0.0058, 0.0022, 0.0028, 0.0074 |
| isort | 16..512 | TooNoisy | 0.0388 (n²) | 0.2690 | 5 of 5 TooNoisy, misfits 0.0945, 0.0719, 0.0739, 0.0687, 0.0410 |
| isort | 16..1024 | TooNoisy | 0.0388 (n²) | 0.2690 | 5 of 5 TooNoisy, misfits 0.1010, 0.0719, 0.0739, 0.0687, 0.0410 |
| zero_subsets | 4..14 | 2ⁿ | 0.0015 | 0.0000 | 5 of 5 named 2ⁿ, misfit 0.0015 |

The next-closest class to each median series: msort `n` (0.112 on 16..512, 0.151 on 16..1024); isort `n log n` (0.924, 1.203); sum `n log n` (0.287); zero_subsets `n³` (1.379).

- **Result:** C names the right class or refuses on every median series and on every seed. It never names another class. This meets the acceptance criterion agreed for the bead.
- **Insertion sort is refused.** Its seeds vary by about 12 % in steps, so its medians are 0.039 from `n²`, four times the tolerance. Under the earlier one-term model (upper-half window, tolerance 0.04) it was named `n²` on the default sweep at 0.026.
- **zero_subsets** counts exactly `21 · 2ⁿ − 8`. With `n³` as the lower-order term of `2ⁿ`, its misfit is 0.0071. With `n` it is 0.0015, and with `1` it is 0.0000. `n` was chosen: it also fits `2ⁿ − n − 2` (from `T(n) = 2T(n−1) + n`) at 0.0089, where `1` gives 0.0173. Against `1.9ⁿ` and `2.1ⁿ` over 4..14 it gives 0.115 and 0.122, where `n³` gives 0.042 and 0.055.
- **Exact power laws, default sweep 16..1024** (minimax by a 20001-point scan in Python): `n^a` is named `n` for `a` in 0.99..1.01, `n²` in 1.99..2.01 and `n³` in 2.99..3.01. No `n^a` is named `n log n`; the closest is `n^1.135`, at 0.0194 (on 16..512: `n^1.14`, at 0.0145).




---

## 2026-10-29: Collapse family: estimator C on random and fixed-schedule inputs

- **Question:** does the naive collapse reference (find the leftmost redex, rewrite it, start again) grow as `n²` on small-alphabet inputs, so that estimator C can classify it? This is the spike step of bead dreamcounter-b307d3.
- **Command:** `cargo test --release -p dc-tasks --test collapse -- --nocapture` (lines tagged `RESULT` and `PLAN`).
- **Commits:** the random generators were measured on an uncommitted tree off `d985235`. The fixed-schedule generator was measured at `eea2988` on branch `p3.7-collapse` (PR 33).
- **Config:** harness defaults (5 seeds, fuel 10x the naive reference, cap 5e7). Sweep 16..1024 (doubling). `ClassSelect::default()`.
- **Random inputs.** Merge: values uniform in `0..=1` or `0..=2`. Cancel: a first half of non-zero values, then the negation of the current stack top with odds 3 in 4. Absorb: a first half that walks by -1, 0 or +1, then values uniform in `0..=9`.

| reference | n² misfit | n log n misfit | noise margin | verdict |
|---|---|---|---|---|
| merge_naive | 0.1392 | 1.1244 | 0.5076 | TooNoisy |
| cancel_naive | 0.1139 | 1.1573 | 0.4118 | TooNoisy |
| absorb_naive | 0.0369 | 1.3466 | 0.4109 | TooNoisy |

- **Fixed-schedule inputs.** The first `n/2` elements have no redex. The rest are a cycle of 8 elements, made of units that are each rewritten on top of the stack: a pair, a nested pair and an overlapping pair (for cancel: pair, nested, pair). Every seed makes the same rewrites at the same depths. Only the values differ.

| reference | verdict | misfit | noise margin | steps at 1024 |
|---|---|---|---|---|
| merge_naive | n² | 0.0020 | 0.0000 | 2,804,814 |
| cancel_naive | n² | 0.0022 | 0.0000 | 2,378,637 |
| absorb_naive, absorb_count_naive | n² | 0.0015 | 0.0000 | 3,526,989 |
| every `_good` reference | n | 0.0000 | 0.0000 | 46,279 (absorb_good) |

- **Result:** the naive reference grows as `n²` under both generators. Under random inputs, C refuses it as TooNoisy. The noise margin of 0.41 to 0.51 comes from seed spread at `n` = 16 and 32. Under the fixed schedule, C names `n²` for every naive reference and `n` for every good one, on the medians and on each seed alone.
- **Limit:** under the fixed schedule, the sweep meets only the schedule's three redex shapes. A stack fold limited to 3 rewrites per element passed both the examples and the sweep at `eea2988`. Examples with chains of 4 rewrites, added at `545fa87`, catch it. No check covers chains longer than 4.



---

## 2026-11-03: Estimator C on each seed alone, across the development families

- **Question:** on how many seeds alone does C name the class of each family reference and natural solution? Until bead dreamcounter-b941dd, the shared `check_class` test helper accepted a refusal on a seed alone, so the family tests did not check this.
- **Command:** `cargo test -p dc-tasks --release --locked --no-fail-fast --test sorted_order --test subsequence_count --test collapse --test natural_solutions -- --nocapture` (lines tagged `SEEDS` list the refused seeds).
- **Commit:** `6f3b673` on branch `fix/b941dd-check-class`, off `7df10f0`.
- **Config:** harness defaults (5 seeds, seeds 0..4). Each seed alone is fitted with `NoiseModel::Known(s)`, where `s` is `single_seed_noise`: the seed-centred noise of one seed's `ln` steps (D15).
- **Result:** C never names another class on a seed alone. It names the class on all 5 seeds for 37 of the 43 programs checked. The exceptions are all refusals:

| program | on the medians | seeds refused | verdict on those seeds | noise margin |
|---|---|---|---|---|
| inversions_good (reference) | n log n | 3 | TooNoisy | 0.0351 |
| pair_sums_natural | n² | 0, 3 | TooNoisy | 0.0554 |
| zigzag_quadratic_natural | n² | 1, 2 | TooNoisy | 0.1998 |
| close_steps_quadratic_natural | refused | 2, 3, 4 | TooNoisy | 0.0585 |
| lower_median_insertion | refused | 0 to 4 | TooNoisy | 0.3114 |
| sign_changes_quadratic (reference) | refused (NoFit) | 0 to 4 | NoFit | 0.0000 |

- **Check:** each test now states how many seeds C may refuse. Lowering each nonzero allowance by one fails the four tests that have one. Multiplying the per-seed noise by 4 fails `sorted_order_distinct` and `sorted_order_min_gap`. It does not fail the collapse family, whose per-seed noise is 0.



---

## 2026-11-04: Corpus run: every development reference against every oracle, and cost curves

- **Question:** do all references of the 14 development variants agree with their oracles on more generated inputs than a sweep uses? Do their cost curves have the expected shape? The variants are collapse (4), sorted order (5) and subsequence count (5). Bead dreamcounter-114909 (P3.10).
- **Commands:**
  - Correctness, every reference at every size: `cargo test --release -p dc-tasks --test corpus -- --ignored every_reference_at_every_size`.
  - Cost table: `cargo test --release -p dc-tasks --test corpus -- --ignored corpus_cost_report --nocapture` (lines tagged `COST`).
- **Commit:** `00ef7b7` on branch `p3.10-corpus-run` (PR 42), off `20c1b44`.
- **Config:** correctness at seeds 0..31, with a fuel cap of 5e7 per run. The `2ⁿ` subsequence-count references run at n = 8 and 16 only. The cost table uses seeds 0..4, the sweep's own seeds. C is `ClassSelect::default()` on the medians, in seed order.
- **Correctness result:** every reference agrees with its oracle on every generated input at every size it runs at, for all 32 seeds. At the largest size, the 32 seeds give 32 different inputs for every variant. The run took 145 s wall and 13.7 CPU-minutes in release, on a Ryzen 7 2700 with 16 threads.
- **Cost table** (median steps at the smallest and largest sizes; C's verdict on the medians):

| variant | reference | smallest size: steps | largest size: steps | C |
|---|---|---|---|---|
| collapse/merge | merge_naive | 32: 4,890 | 1024: 3,798,670 | n² |
| collapse/merge | merge_good | 32: 1,444 | 1024: 45,960 | n |
| collapse/cancel | cancel_naive | 16: 891 | 1024: 2,378,637 | n² |
| collapse/cancel | cancel_good | 16: 603 | 1024: 38,151 | n |
| collapse/absorb | absorb_naive | 32: 5,105 | 1024: 4,225,197 | n² |
| collapse/absorb | absorb_good | 32: 1,522 | 1024: 48,487 | n |
| collapse/absorb-count | absorb_count_naive | 32: 5,105 | 1024: 4,225,197 | n² |
| collapse/absorb-count | absorb_count_good | 32: 1,522 | 1024: 48,487 | n |
| sorted-order/inversions | inversions_naive | 16: 992 | 512: 920,328 | n² |
| sorted-order/inversions | inversions_good | 16: 2,590 | 512: 158,339 | n log n |
| sorted-order/pair-sums | pair_sums_naive | 16: 1,113 | 512: 1,051,145 | n² |
| sorted-order/pair-sums | pair_sums_good | 16: 2,470 | 512: 144,197 | n log n |
| sorted-order/min-gap | min_gap_naive | 16: 1,608 | 512: 1,574,920 | n² |
| sorted-order/min-gap | min_gap_good | 16: 2,406 | 512: 142,449 | n log n |
| sorted-order/distinct | distinct_naive | 16: 1,024 | 512: 921,352 | n² |
| sorted-order/distinct | distinct_good | 16: 2,376 | 512: 141,427 | n log n |
| sorted-order/lower-median | lower_median_naive | 16: 3,756 | 512: 3,675,396 | n² |
| sorted-order/lower-median | lower_median_good | 16: 2,329 | 512: 139,396 | n log n |
| subsequence-count/sum-mod-7 | sum_mod_7_naive | 8: 5,367 | 16: 1,376,247 | not fitted |
| subsequence-count/sum-mod-7 | sum_mod_7_quadratic | 8: 2,660 | 256: 1,957,024 | n² |
| subsequence-count/sum-mod-7 | sum_mod_7_good | 8: 782 | 256: 24,094 | n |
| subsequence-count/close-steps | close_steps_naive | 8: 8,682 | 16: 2,228,202 | not fitted |
| subsequence-count/close-steps | close_steps_quadratic | 8: 582 | 256: 558,210 | n² |
| subsequence-count/close-steps | close_steps_good | 8: 567 | 256: 17,183 | n |
| subsequence-count/no-three-in-a-row | no_three_in_a_row_naive | 8: 6,024 | 16: 1,526,828 | not fitted |
| subsequence-count/no-three-in-a-row | no_three_in_a_row_quadratic | 8: 261 | 256: 166,793 | n² |
| subsequence-count/no-three-in-a-row | no_three_in_a_row_good | 8: 151 | 256: 4,367 | n |
| subsequence-count/zigzag | zigzag_naive | 8: 8,969 | 16: 2,293,769 | not fitted |
| subsequence-count/zigzag | zigzag_quadratic | 8: 1,114 | 256: 1,051,394 | n² |
| subsequence-count/zigzag | zigzag_good | 8: 2,177 | 256: 68,889 | n |
| subsequence-count/sign-changes | sign_changes_naive | 8: 11,233 | 16: 2,883,553 | not fitted |
| subsequence-count/sign-changes | sign_changes_quadratic | 8: 18,372 | 256: 5,244,848 | NoFit |
| subsequence-count/sign-changes | sign_changes_good | 8: 13,773 | 256: 431,405 | n |

- **Result:** C names the expected class for every reference fitted on the whole sweep, except `sign_changes_quadratic`. C refuses that one with NoFit. This refusal is known: see the entry of 2026-11-03, and `tests/subsequence_count.rs`, which allows it. The `2ⁿ` references are not fitted, because two sizes are too few for a fit. From n = 8 to 16 their steps grow by a factor of 253 to 257, close to `2⁸` = 256. Absorb and absorb-count have the same step counts at every size.
- **What CI runs:** `cargo test` runs the good references at every size and the other references at the lower half of each sweep, at the same 32 seeds. That took about 4.8 CPU-minutes at the test profile (opt-level 1). The whole run is an ignored test.



---

## 2026-11-04: Evaluator throughput with a randomly keyed hasher for memo maps

- **Question:** how much single-thread throughput does the evaluator lose when two memo maps use the standard library's SipHash with random keys instead of FxHash? The maps are the memo table's entries and the node map of a large memo key. Bead dreamcounter-36be80 made this change, so that a program cannot choose integers whose hashes all collide.
- **Builds:** base `b7f73a5` (main). New `1a138fa` on branch `fix-36be80-hash-flooding`. Both `release` with default `RUSTFLAGS`. The later commits on the branch change only tests and comments.
- **Probe:** a scratch example program in `dc-bench`, not committed. It compiles each benchmark program and runs it at its full `n` on one thread: a tenth of the count as warm-up, then a fixed count (`sum` 300, `fib_memo` 30,000). It reports steps per second of the thread's CPU time, read from `/proc/thread-self/schedstat`. `sum` does not use `memorec`, so it is the control.
- **Method:** 8 rounds, each running the base build and then the new one. The table gives the median of the 8 runs.
- **Machine:** AMD Ryzen 7 2700, 16 threads. Load average 3.72 at the start and 3.49 at the end, from desktop and agent processes. `perf` was not available (`perf_event_paranoid`).

| program | base Msteps/CPU-s | new Msteps/CPU-s | change |
|---|---|---|---|
| sum | 5.355 | 5.382 | +0.5 % |
| fib_memo | 3.448 | 3.364 | −2.4 % |

- **Raw `fib_memo` runs, base:** 3.503, 3.396, 3.391, 3.454, 3.441, 3.285, 3.485, 3.479. **New:** 3.417, 3.167, 3.385, 3.360, 3.367, 3.388, 3.293, 3.354. The new build was slower in 7 of the 8 rounds.
- **Result:** the memo-heavy benchmark program loses about 2.4 % single-thread throughput, and the control does not change.
- **Discarded run:** a first attempt with `dc-bench` itself ran under a load average of 17 to 38. It showed differences of up to 29 % on `sum`, which the change does not touch, so its numbers are not used.
- **Collision check:** in a debug test build with FxHash, a memo key of 16,000 integers chosen to collide took 416 ms, against 6.7 ms for 2,000. That is 62 times the time for 8 times the integers. With the new hasher, the test `a_large_key_of_colliding_ints_builds_in_linear_time` gave ratios of 3.3 to 9.5 under load.



---

## 2026-11-04: Interning throughput with foldhash for the arena index

- **Question:** how much slower is interning when `Arena`'s index from terms to ids uses foldhash with a seed for each arena instead of FxHash? Bead dreamcounter-e3548c made this change, so that a program's literals, variable indices and choice of child pairs cannot make interning quadratic. No enumerator exists yet, so interning stands in for enumeration throughput.
- **Builds:** base `f0c5f02` (main) with the index hasher named but still FxHash. New `93c5365` on branch `fix-e3548c-arena-int-hash`, with `foldhash::fast::RandomState` (foldhash 0.2.0). Both `release` with default `RUSTFLAGS`. The later commits on the branch change only tests and comments.
- **Probe:** a scratch example program in `dc-core`, not committed. It interns 2,000,000 terms on one thread, chosen by an xorshift generator from a fixed seed. Half are applications and the rest are lambdas, small integers, `if`s and `let`s over earlier ids. Some terms are new and some repeat. It reports CPU nanoseconds per `intern` call, read from `/proc/thread-self/schedstat`, as the best of 5 runs.
  - **Large arena:** new ids join the pool of children until it holds 300,000. The arena ends with 1,667,451 nodes.
  - **Small arena:** the pool stops at 2,000 ids, and the second child comes from the first 64. The arena ends with 349,891 nodes, so the hash function's own cost weighs more.
- **Method:** 8 rounds, each running the base build and then the new one. Rounds 1 to 6 ran at a load average of about 3.6. During rounds 7 and 8 the load rose and both builds slowed by about 60 %, so the table uses rounds 1 to 6.
- **Machine:** AMD Ryzen 7 2700, 16 threads.

| arena | base ns/intern | new ns/intern | change |
|---|---|---|---|
| large | 159.8 | 170.5 | +6.7 % |
| small | 63.5 | 80.3 | +26 % |

- **Raw runs, rounds 1 to 8, ns/intern.** Large base: 161.0, 159.0, 160.5, 186.5, 159.0, 151.1, 256.1, 263.5. Large new: 170.9, 170.0, 172.0, 165.0, 165.0, 175.5, 259.9, 252.0. Small base: 63.5, 63.5, 61.5, 82.5, 61.0, 67.5, 118.3, 110.5. Small new: 82.5, 78.0, 118.0, 75.0, 75.5, 104.0, 119.6, 124.5.
- **Result:** interning costs about 7 % more on a large arena and about 23 to 26 % more on a small one. The fastest small-arena runs are 61.0 ns and 75.0 ns.
- **Rejected alternative:** FxHash over the term followed by one seeded foldhash step on its 64-bit result. On integers crafted to collide under FxHash, it put 4,096 terms in 689 of 4,096 buckets, against about 2,589 expected for a random hash. The crafted FxHash values differ in about 12 bits, and the single foldhash step does not spread them. The test `the_index_spreads_ints_chosen_to_collide` fails on it. Its speed was not measured.
- **Collision check:** in a debug test build with FxHash, interning 16,000 integers chosen to collide took 1.15 s, against 16.5 ms for 2,000: 70 times the time for 8 times the integers. With foldhash, the test `interning_ints_chosen_to_collide_takes_linear_time` gave ratios of 2.5 to 8.9 over 30 runs.




---

## 2026-11-04: Memory of memo tables that only their own entries reach

- **Question:** how much memory does a long evaluation keep when each iteration makes a memo table whose entries hold the table's own `self`? Before bead dreamcounter-3fd21a, such a table stayed alive until the evaluation ended. The bead adds a reachability walk that releases tables the program can no longer reach. Does the walk change step counts or time?
- **Builds:** base `origin/main` at the time of the run. New: branch `fix/dreamcounter-3fd21a-memo-reachability` (PR 47, head `ad20d66`). The RSS runs were at earlier commits on the branch with the same release logic. Both builds are `release` with default `RUSTFLAGS`, each in its own target directory.
- **Probe:** a scratch example program `memo_rss` in `dc-eval`, not committed. It runs a loop of 300,000 iterations. Each iteration calls `memorec` with a curried function, so the table's entries hold its own `self`. A control runs the same loop with a plain `memorec` whose entries do not hold `self`. Peak RSS is from `/usr/bin/time -f '%M'`.
- **Machine:** AMD Ryzen 7 2700, 16 threads. The load average was 14 to 19 during the timing runs, so the times are rough.

| case | base peak RSS | new peak RSS | base time | new time | steps |
|---|---|---|---|---|---|
| table reached only by its own entries, 300,000 iterations | 417 MB | 2.2 MB | 1.417 s (median) | 0.68 to 0.73 s | 2,100,000 on both |
| control: plain `memorec` | not measured | not measured | 0.692 s | 0.668 s | equal on both |
| review case: tables beside an array that many values share, 3000 × 3000 | 418 MB | 5.9 MB | not measured | not measured | not measured |

- **deep.rs test binary:** peak RSS with the tests run in parallel fell from 1.19 GB to 422 MB. An independent re-run gave 464 MB. The largest single test now peaks at 123 MB. The array test nests 20,000 levels, and the list test drops a list of 100,000 elements. Each still overflows a small stack when the drop of arrays, lists or trees is made recursive.
- **Result:** tables that only their own entries reach are released, and peak memory no longer grows with the iteration count. Step counts do not change. The new build was faster on the bead case. Under this load, that difference does not measure the walk's own cost.
- **Cost bound:** the test `reachability_walks_cost_at_most_a_constant_times_the_steps` checks that walk work stays within a constant times the steps taken. A walk that would visit more parts than the steps since the last finished walk gives up. The next walk then waits twice as long.
- **Limits:** while the reachable memory is larger than the steps since the last finished walk, walks give up. Unreachable tables then stay until a walk finishes. Some frames count as roots even where they may not be the only path to a table: `unfold`'s `g`, `tabulate`'s `f`, and memo stores. This can keep a table longer; it never releases a reachable one.




---

## 2026-11-12: Seed-centred per-seed noise on the reference programs

- **Question:** does the check on each seed alone in `reference_growth.rs` give different verdicts when its per-seed noise takes out each seed's level, as estimator C does (D15)? Until bead dreamcounter-eb1ace, that file had its own uncentred copy of `single_seed_noise`. The family tests have used the centred helper since PR 40.
- **Command:** `cargo test --release -p dc-tasks --test reference_growth -- --include-ignored --nocapture --test-threads=1` (lines tagged `RESULT`).
- **Builds:** base `c354a9d` (main). New `04364f5` on branch `eb1ace-centre-seed-noise` (PR 50). The later commit on the branch changes only doc comments.
- **Config:** harness defaults (5 seeds, seeds 0..4). Every sweep comes from `measure`, so it is in seed order. Sweeps as in the entry of 2026-10-28.

| program | sweep | per-seed noise, base | per-seed noise, new | noise margin, base | noise margin, new | per seed, both builds |
|---|---|---|---|---|---|---|
| sum_ref | 16..1024 | 9.9e-16 | 0.0 | 0.0000 | 0.0000 | 5 of 5 named n |
| msort | 16..512 | 0.007258 | 0.007410 | 0.0218 | 0.0222 | 5 of 5 named n log n |
| msort | 16..1024 | 0.007258 | 0.007401 | 0.0218 | 0.0222 | 5 of 5 named n log n |
| isort | 16..512 | 0.159984 | 0.156343 | 0.4800 | 0.4690 | 5 of 5 TooNoisy |
| isort | 16..1024 | 0.159984 | 0.156877 | 0.4800 | 0.4706 | 5 of 5 TooNoisy |
| zero_subsets | 4..14 | 0.0 | 0.0 | 0.0000 | 0.0000 | 5 of 5 named 2ⁿ |

- **Result:** no verdict on a seed alone changes. The per-seed noise moves by about 2 % or less, because the seeds of these reference programs barely differ in level. Merge sort's noise rises slightly: the centred variance is multiplied by `N / (N − 1)` over `N` sizes.
- **Check:** the test `per_seed_noise_is_centred_on_a_sweep_in_seed_order` fails on the base build (0.1687 against C's 0.0175). An independent review made it fail on the new build in two ways: with centring turned off, and with the `N / (N − 1)` factor removed (0.0162 against 0.0175).




---

## 2026-11-13: Expansion throughput with foldhash for the Expander's maps

- **Question:** how much slower is library expansion when the `Expander`'s three maps use foldhash with a seed for each map instead of FxHash? The maps are the task memo, the renaming index and the memo of `restrict`. Bead dreamcounter-994cf2 made this change for the reason given in dreamcounter-e3548c: a program's shape chooses which combinations of ids and depths become keys, and FxHash is affine in each field.
- **Builds:** base `11728c6` on branch `fix-994cf2-expander-hash`, which adds only the new tests to `bdcd963` (main), so its maps still use FxHash. New `df20da9` on the same branch, with `foldhash::fast::RandomState` (foldhash 0.2). Both builds are `release` with default `RUSTFLAGS`, each in its own target directory. The runs were made at `9905e29` and `2961fc4`, which differ from these commits only in their author address. The later commit on the branch changes only comments.
- **Probe:** a scratch example program `expand_probe` in `dc-core`, not committed. An xorshift generator with seed `0x9E3779B97F4A7C15` builds 24 libraries of arity 1 to 3. Their bodies have about 12 nodes and may use earlier libraries. The generator then builds random programs of lambdas, `if`s and saturated library uses, with variables, small integers and `true` as leaves. One `Expander` expands every program in turn into a clone of the arena. The probe reports CPU nanoseconds per program, read from `/proc/thread-self/schedstat`, as the best of 5 passes.
  - **Large programs:** 5,000 programs of about 200 generated nodes (`expand_probe 200 5000`).
  - **Small programs:** 30,000 programs of about 30 generated nodes (`expand_probe 30 30000`), so per-program costs weigh more.
- **Method:** 8 rounds at both sizes. Each round runs the base build and then the new one. The table gives the median of the 8 runs.
- **Machine:** AMD Ryzen 7 2700, 16 threads. The load average was 8.9 to 10.5 throughout, from other agents.

| programs | base ns/program | new ns/program | change |
|---|---|---|---|
| large | 46,598 | 47,295 | +1.5 % |
| small | 12,281 | 11,703 | −4.7 % |

- **Raw runs, rounds 1 to 8, ns/program.** Large base: 51,282, 56,184, 44,799, 47,799, 43,943, 45,398, 41,667, 48,550. Large new: 51,998, 48,390, 50,191, 51,914, 43,997, 44,905, 39,721, 46,200. Small base: 11,974, 12,469, 12,330, 12,666, 11,994, 12,232, 10,800, 12,916. Small new: 12,071, 13,432, 11,334, 11,000, 12,499, 11,065, 11,288, 12,667.
- **Result:** no change in expansion time is measurable at this load. The two builds differ by less than the spread of either build between rounds, which is about ±10 %. The new build was faster in 3 of the 8 large rounds and 5 of the 8 small rounds.
- **Collision check:** three tests each hash 4,096 keys whose FxHash values agree in their low 50 bits before FxHash's final rotation. The tests are `the_renaming_index_spreads_keys_chosen_to_collide`, `the_restrict_memo_spreads_keys_chosen_to_collide` and `the_task_memo_spreads_keys_chosen_to_collide`. With FxHash, all 4,096 keys fall in 1 bucket and 1 tag in each map. With foldhash, each test needs more than 2,000 of 4,096 buckets and more than 100 of 128 tags, and passes. Expansion time on such keys was not measured, because no program is known that produces them.



---

## 2026-11-15: Evaluator profile, and throughput when a primitive with all its arguments builds no partial application

- **Question:** where does the evaluator spend its time on one thread? How much faster is it when a primitive that the term applies to all its arguments runs without a function value for each argument? Bead dreamcounter-a0627f.
- **Machine:** AMD Ryzen 7 2700, 16 threads. `perf` was not available (`perf_event_paranoid` is 4), and `valgrind` is not installed.

### Profile

- **Build:** `d96299c` (main at the time of the local checkout), `release` with `debug = 1`, default `RUSTFLAGS`.
- **Probe:** a scratch Cargo package outside the repository, not committed, with a path dependency on `dc-bench`. It compiles one benchmark program and runs it at its full `n` on one thread for 12 s.
- **Method:** `gdb` ran the probe as its child, with `handle SIGINT stop print nopass`. A shell loop sent `SIGINT` to the probe every 10 ms. A Python loop in `gdb` recorded the function names of the stack at each stop. Each sample is put in the first group it matches: a `Vec` growing anywhere in the stack; then a value drop anywhere in the stack; then an allocator function in the innermost six frames; then `memcpy` as the innermost frame.
- **Load average:** 2.79 after the first set of runs and 2.92 after the second. A first set of runs recorded no samples, because `noprint` implies `nostop`.

| program | samples | interpreter | dropping values, with their frees | other allocator calls | `Vec` growth |
|---|---|---|---|---|---|
| sum | 927 | 37.2 % | 32.9 % | 17.4 % | 12.5 % |
| msort | 946 | 36.3 % | 35.7 % | 19.0 % | 9.0 % |
| fib_memo | 931 | 39.7 % | 23.2 % | 24.3 % | 12.8 % |
| max_subarray | 962 | 46.2 % | 25.6 % | 19.9 % | 8.1 % |

- **`Vec` growth from `Machine::apply`:** 11.2 %, 6.6 %, 8.8 %, and 6.9 % of the samples of `sum`, `msort`, `fib_memo`, and `max_subarray`. That is the `Fun::Partial` arm, which copied the argument vector and pushed onto it for each argument.
- **Throughput with and without sampling:** `sum` 4.79 and 5.25 Msteps/s; `msort` 2.68 and 2.88; `fib_memo` 2.89 and 3.22; `max_subarray` 3.17 and 3.53. These are wall-clock rates.
- **Result:** 54 to 63 % of the samples are in allocation, freeing, and dropping values. The `max_subarray` row has 0.3 % of its samples in `memcpy`, not shown.

### Throughput before and after

- **Builds:** base `9071b02` (main). New `1369d6a` on branch `a0627f-saturated-prims`. Both `release` with `debug = 1` and default `RUSTFLAGS`, each in its own target directory.
- **Probe:** a second scratch package, not committed. For each `dc-bench` program at its full `n`, on one thread, it warms up for 0.2 s and then runs ten times the warm-up's count of evaluations. It checks every step count against the first run's. It reports steps per second of the thread's CPU time, read from `/proc/thread-self/schedstat`.
- **Method:** 6 rounds, each running the base build and then the new one. The table gives the median of the 6 runs.
- **Load average:** 2.89 at the start and 2.87 at the end.

| program | steps/eval, both builds | base Msteps/CPU-s | new Msteps/CPU-s | new / base |
|---|---|---|---|---|
| sum | 40,001 | 5.378 | 9.974 | 1.85 |
| max_subarray | 150,003 | 3.752 | 5.998 | 1.60 |
| msort | 290,817 | 3.040 | 5.627 | 1.85 |
| fib_naive | 142,292 | 3.816 | 7.557 | 1.98 |
| fib_memo | 661 | 3.293 | 5.986 | 1.82 |
| bsearch | 248 | 2.745 | 5.572 | 2.03 |

- **Raw runs, rounds 1 to 6, Msteps/CPU-s.**
  - `sum` base: 5.550, 5.334, 5.359, 5.570, 5.398, 4.926. New: 10.105, 9.937, 9.861, 10.010, 10.166, 9.881.
  - `max_subarray` base: 3.787, 3.676, 3.762, 3.741, 3.795, 3.621. New: 5.806, 6.065, 6.086, 5.986, 6.010, 5.917.
  - `msort` base: 3.014, 3.102, 3.081, 3.067, 3.005, 2.978. New: 5.623, 5.718, 5.777, 5.631, 5.551, 5.527.
  - `fib_naive` base: 3.827, 3.838, 3.918, 3.805, 3.725, 3.741. New: 7.547, 7.495, 7.584, 7.597, 7.329, 7.566.
  - `fib_memo` base: 3.367, 3.301, 3.259, 3.320, 3.285, 3.254. New: 5.828, 5.955, 6.083, 6.056, 6.016, 5.801.
  - `bsearch` base: 2.793, 2.816, 2.728, 2.718, 2.762, 2.595. New: 5.621, 5.495, 5.645, 5.478, 5.581, 5.563.
- **Result:** the new build is 1.6 to 2.0 times as fast on one thread, and it was faster in all 6 rounds for every program. The step counts are the same on both builds.
- **Not measured:** throughput on 16 threads, the task corpus, and a profile of the new build.
