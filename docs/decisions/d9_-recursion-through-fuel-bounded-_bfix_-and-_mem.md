---
id: doc-sh3
title: 'D9: Recursion through fuel-bounded `bfix` and `memorec`'
type: decision
scope: internal
created_at: '2026-10-27T18:08:10.774601Z'
updated_at: '2026-10-27T18:08:10.774601Z'
---

- **Date:** 2026-10-27. **Status:** accepted. The project lead chose `bfix` over `wfrec` (Q2). The fuel details below are Claude's design and open to change.
- **Decision:** recursion goes through `bfix n F x`: open recursion with depth fuel `n`. Calling the recursive argument at fuel 0 raises `FuelExhausted`. `memorec n F x` adds a memo table, and its fuel bounds table misses across the whole table. `wfrec` is dropped. Fuel is an ordinary `Int` expression, so a program can tie it to its input, for example `bfix (length xs + 1)`.
- **Alternatives:**
  - (a) `wfrec`: well-founded recursion with a runtime-checked measure (D2);
  - (b) `bfix` that returns a default value when the fuel runs out;
  - (c) `bfix` with one total-call budget shared by all calls;
  - (d) per-depth fuel for `memorec`.
- **Why:**
  - (a) rejects recursion that halts but has no simple decreasing integer, such as iteration to a fixed point, Collatz-style loops and search whose progress is not monotone. The LLM would have to fake a measure with an iteration counter, which is `bfix` written by hand.
  - (b) would let a truncated program pass its examples by accident.
  - Depth fuel, not (c), matches the classical bounded fixpoint, `bfix n F = F (bfix (n-1) F)`. It also keeps a call's result independent of work done in sibling calls.
  - (d) would make a cached result depend on how much fuel was left when it was first computed. The same call could then succeed or fail depending on evaluation history. A shared miss count is deterministic, because evaluation is strict and left to right.
- **Costs of the choice:** programs carry a fuel expression. A literal fuel is the arbitrary number that D2 tried to avoid. Fuel expressions in terms of the input reduce that problem, and give the compressor shared structure where it exists. A fuel too small for large inputs shows up as errors in the growth sweeps.
- **Supersedes:** D2. **See:** language spec §6.
