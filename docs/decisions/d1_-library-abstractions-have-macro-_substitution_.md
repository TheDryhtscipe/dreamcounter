---
id: doc-6jf
title: 'D1: Library abstractions have macro (substitution) semantics'
type: decision
scope: internal
created_at: '2026-10-27T17:10:50.913090Z'
updated_at: '2026-10-27T17:11:15.610386Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decision:** `L a0 ... ak` means `body[#i := ai]`. Abstractions are expanded
  before evaluation. Refactoring is then exactly value- and cost-preserving.
- **Alternatives:** (a) abstractions as ordinary call-by-value functions; (b)
  call-by-need parameters.
- **Why:** (a) makes refactoring perform hidden CSE (cost drops when a parameter was
  duplicated) and turns lazy positions strict (cost rises or a new error appears, e.g.
  a guarded `div` inside an `if`). Either would let syntactic compression change
  meaning and cost. (b) fixes semantics but still lets sharing reduce cost, and adds
  thunk machinery to the evaluator. Macro semantics is the only option under which
  "refactor" is a pure description-length operation.
- **Cost of the choice:** expanded terms can blow up when parameters are duplicated
  through nested abstractions. Hash-consing keeps them as DAGs; to be monitored.
- **See:** language spec §8.
