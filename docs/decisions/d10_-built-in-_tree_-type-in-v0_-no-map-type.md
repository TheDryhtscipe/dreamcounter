---
id: doc-j6x
title: 'D10: Built-in `Tree` type in v0, no map type'
type: decision
scope: internal
created_at: '2026-10-27T18:08:14.823123Z'
updated_at: '2026-10-27T18:08:14.823123Z'
---

- **Date:** 2026-10-27. **Status:** accepted by the project lead (Q3).
- **Decision:** v0 has a built-in binary tree type `Tree τ` with `leaf`, `node`, `tree_case` and `tree_fold`. v0 has no map type. The rest of the base set proposed in language spec §2, §5 and §6 stands: no sort primitive and no mutation.
- **Alternatives:**
  - trees encoded in arrays by heap indexing;
  - user-defined algebraic data types;
  - a built-in map type.
- **Why:**
  - Heaps, binary search trees and segment trees are natural to write with a tree type. Several target primitives, such as monotone structures and priority-driven search, depend on them.
  - With heap indexing, the LLM writes index arithmetic, and the compressor sees arithmetic instead of tree structure.
  - User-defined types are more general. They cost more in every component and give the LLM more to get wrong.
  - Maps: `memorec` covers memo tables, and persistent arrays cover indexed state. A map type gets reconsidered if a task family needs one that `memorec` cannot replace.
- **See:** language spec §2, §5 and §10.
