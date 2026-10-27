---
id: doc-9n9
title: 'D6: CEK machine with explicit continuation stack'
type: decision
scope: internal
created_at: '2026-10-27T17:11:09.460165Z'
updated_at: '2026-10-27T17:11:19.918184Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decision:** the evaluator is an explicit-stack environment machine, not a
  recursive tree-walker.
- **Why:** dc programs recurse deeply (folds over 10⁴-element lists, D&C). Host
  recursion would overflow the Rust stack or need large thread stacks. Step counting
  and fuel checks also end up in one place.
