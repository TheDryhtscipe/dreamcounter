---
id: doc-tas
title: 'D3: Cost model specifics'
type: decision
scope: internal
created_at: '2026-10-27T17:11:06.866289Z'
updated_at: '2026-10-27T18:08:01.545366Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decisions:**
  - One step per β-reduction, primitive application, combinator iteration and branch
    decision. Variable lookup, `let`, literals and closure creation are free.
  - Lists cache their length; `length` costs 1.
  - Arrays are persistent; `array_set` costs ⌈log₂(n+1)⌉ + 1, matching the real
    implementation.
  - Structural `==` costs the number of nodes compared.
  - The fuel checks in `bfix` and `memorec` are free (D9 replaced the ghost measures
    of D2).
  - Checked 64-bit integer arithmetic; overflow is an error.
  - No `sort`, hash map or mutation primitives.
- **Alternatives:** O(n) `length` as in ML (more conventional, but makes measures
  expensive and penalises a harmless idiom); O(1) array update (unrealistic for a
  pure language and the implementation would not match); wall-clock time (ruled out
  by the handoff).
- **Why:** each cost should match what the implementation actually does, so that
  costs stay explainable and the implementation stays honest.

