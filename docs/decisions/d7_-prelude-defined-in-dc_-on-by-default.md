---
id: doc-mh6
title: 'D7: Prelude defined in dc, on by default'
type: decision
scope: internal
created_at: '2026-10-27T17:11:10.324188Z'
updated_at: '2026-10-27T17:11:20.849324Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decision:** `map`, `filter`, `append`, etc. are written in dc and treated as
  opaque primitives by the compressor. An ablation turns them off.
- **Why:** the LLM will use them constantly, so leaving them out would mostly cost
  proposal validity. Defining them in dc keeps their cost honest.
