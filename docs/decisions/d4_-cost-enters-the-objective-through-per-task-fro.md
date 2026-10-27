---
id: doc-tmq
title: 'D4: Cost enters the objective through per-task frontier selection'
type: decision
scope: internal
created_at: '2026-10-27T17:11:07.726395Z'
updated_at: '2026-10-27T17:11:18.268986Z'
---

- **Date:** 2026-10-27. **Status:** needs the project lead (it reframes how criterion (b) works).
- **Decision:** keep up to K verified solutions per task. Score a library with
  `J(L) = DL(L) + Σ_t min_{p ∈ F_t} [DL(p | L) + λ·C(p)]`. Log component (b) as the
  change in `Σ_t C(selected_t)`. Integrate with plain Stitch by alternating
  selection and compression.
- **Why:** under D1, and in fact under any sensible semantics, refactoring a fixed
  solution set cannot change its cost. Cost-awareness has to act on *which*
  solutions the library is compressed against, and its real payoff is in the next
  wake phase.
- **Alternatives:** cost as a hard constraint (filter frontiers to near-best growth,
  then pure MDL); a cost term inside Stitch's per-node utility (rejected: it would
  reward abstracting expensive *syntax*, not expensive *behaviour*).
- **See:** language spec §8.
