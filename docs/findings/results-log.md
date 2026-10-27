---
id: doc-m4a
title: Results log
type: reference
scope: internal
created_at: '2026-10-27T17:11:14.071510Z'
updated_at: '2026-10-27T17:11:28.346646Z'
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
