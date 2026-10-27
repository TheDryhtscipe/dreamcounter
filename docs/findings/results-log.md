---
id: doc-m4a
title: Results log
type: reference
scope: internal
created_at: '2026-10-27T17:11:14.071510Z'
updated_at: '2026-10-27T21:23:09.453834Z'
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
