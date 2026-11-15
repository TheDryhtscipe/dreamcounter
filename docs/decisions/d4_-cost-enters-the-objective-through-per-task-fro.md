---
id: doc-tmq
title: 'D4: Cost enters the objective through per-task frontier selection'
type: decision
scope: internal
created_at: '2026-10-27T17:11:07.726395Z'
updated_at: '2026-11-12T02:43:05.793931Z'
---

- **Date:** 2026-10-27. **Status:** accepted by the project lead on 2026-10-27.
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
- **Additions (2026-11-12, accepted by the project lead):**
  - **Members with no fitted growth (Q10, bead dreamcounter-1e9af3).** A frontier member can lack `C(p)`. Its value then depends on the cause:
    - A member that timed out inside the fit window gets a large fixed penalty. A timeout is evidence of high cost.
    - A member that estimator C refused (`TooNoisy` or `NoFit`) gets the largest fitted `C` in its task plus 1. It ranks after every fitted member of that task, because a refusal says only that the measurement failed.
    - The cause is logged with each such member.
    - Rejected: one rule for both causes; leaving unfitted members out of `F_t`, which rewards libraries that help only programs whose cost cannot be measured; a fallback slope without the noise check.
    - The size of the timeout penalty is set in P5 (bead dreamcounter-183fbb).
  - **Weighting (Q4, bead dreamcounter-0fb72b).** P5 builds both forms: the λ-weighted term above, and the constraint form (filter frontiers to near-best growth, then pure MDL). Each run logs description length, the change in cost growth, and reuse across task families as separate numbers. P5 runs λ = 0, the plain-compression control, and two or three other values of λ on the same corpus. No weighting is chosen before P4 data exists.
  - **Reuse (Q4).** Reuse across task families is an evaluation metric (D8), not a term in `J(L)`. More reuse under the cost-aware objective is then a finding that the objective did not build in. A filter that keeps only abstractions used in at least k families may become a later ablation.
- **See:** language spec §8.

