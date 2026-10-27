---
id: doc-kb9
title: 'D8: Evaluation plan gains a compute-matched baseline and reuse measurement'
type: decision
scope: internal
created_at: '2026-10-27T17:11:11.238933Z'
updated_at: '2026-10-27T17:11:21.682678Z'
---

- **Date:** 2026-10-27. **Status:** proposed.
- **Decision:** every headline comparison includes an LLM-only baseline given the
  same LLM-call and evaluation budget, and we report per-abstraction reuse rates in
  new solutions.
- **Why:** Berlot-Attwell et al. (EACL 2026) found three published LLM library
  learning systems did not beat a compute-matched prompting baseline, and found no
  evidence of lemma reuse in LEGO-Prover. See the literature notes §1.
