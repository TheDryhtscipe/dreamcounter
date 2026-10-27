---
id: doc-nnn
title: 'D5: Growth estimation: log-log slope by default, swappable'
type: decision
scope: internal
created_at: '2026-10-27T17:11:08.596091Z'
updated_at: '2026-10-27T17:11:19.107258Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decision:** estimator A (OLS slope on the upper half of a doubling sweep, median
  of 5 seeds per size) as the default. Estimator B (c·n^α·(log n)^γ) implemented
  alongside it. Both behind a `GrowthEstimator` trait.
- **Why:** A is what the handoff asked for and is simple to interpret. B covers the
  case A cannot handle well (n log n against n^1.17).
