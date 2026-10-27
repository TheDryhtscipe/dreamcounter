---
id: doc-26m
title: 'D2: Recursion through measure-checked `wfrec` / `memorec`, not a fuel argument'
type: decision
scope: internal
created_at: '2026-10-27T17:11:06.019928Z'
updated_at: '2026-10-27T17:11:16.493960Z'
---

- **Date:** 2026-10-27. **Status:** needs the project lead (it interprets "bounded fixpoint with
  fuel").
- **Decision:** `wfrec m F x` requires each recursive call to strictly decrease the
  non-negative integer measure `m`, checked at runtime. `memorec` is the same plus a
  memo table. Global fuel remains as a safety net.
- **Alternatives:** (a) `bfix fuel F x` with an explicit per-call depth bound; (b) `fix`
  bounded only by global fuel.
- **Why:** (a) puts an arbitrary number into every program, which the compressor
  then has to abstract over, and the LLM has to guess. (b) is general recursion in
  all but name, which the handoff rules out. Measures are total by construction, add
  meaningful shared syntax (`length` for D&C, `hi - lo` for binary search), and cost
  little at runtime (ghost, capped at 64 steps).
- **See:** language spec §6.
