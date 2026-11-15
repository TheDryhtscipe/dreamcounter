---
id: doc-d9s
title: 'D11: Proposals come from Claude sessions pulling work over MCP; API calls
  are an optional second backend'
type: decision
scope: internal
created_at: '2026-10-27T21:23:29.915628Z'
updated_at: '2026-11-15T19:38:34.116144Z'
---

- **Date:** 2026-10-27. **Status:** accepted. The project lead chose the pull-based proposer to keep API spending low (Q5). The protocol details below are Claude's design and open to change.
- **Decision:** DreamCounter runs as an MCP server (`dc-mcp`) that owns the run state: tasks, library, frontiers, and iteration. Claude sessions are the proposer. Each proposer agent calls `dc_next_batch` to get a prompt payload, writes programs, and calls `dc_submit`. The server evaluates proposals, selects frontiers, and runs Stitch compression when an iteration's quota is in. Naming new abstractions (AutoDoc) is another pulled job of the same kind. A workflow in a session fans proposer agents out for each iteration. The P4 LLM backend has two implementations: pull over MCP (the default) and API calls (optional, for example a cheap-model or local-model ablation).
- **Alternatives:**
  - (a) API calls only, as the original plan assumed;
  - (b) an MCP wrapper used only by agents doing development work, with the loop still on the API.
- **Why:**
  - (a) costs about 24M input and 4M output tokens per run before caching (Q5 sizing). The current budget does not cover several runs plus ablations.
  - (b) keeps the cost of (a).
  - The server holds all state between iterations, so a run can be spread over many sessions and resume under usage limits.
- **Requirements this places on the design:**
  1. **No contamination.** A proposer sees only the server's payload. Proposer agents run from an empty directory, are briefed to call the two tools and read nothing else, and never see the repository, the hypothesis, or held-out families (Q7). The server logs every payload it hands out and every submission. Global user rules still load into proposer sessions; record that in any write-up.
  2. **A blind arm.** In the LILO-faithful arm, `dc_submit` acknowledges a submission without returning results, so the proposer gets no evaluation feedback. An arm that lets the proposer test programs first is a separate, deliberate variant with its own decision.
  3. **Sample independence.** Samples written in one context are correlated, unlike independent API samples at a fixed temperature. Where frontier diversity matters, each sample comes from a separate proposer agent; otherwise the run records that samples share a context.
  4. **Accounting.** Model version and temperature cannot be fixed. The server records the model identity where the session reports it, and counts proposals and approximate tokens. The compute-matched baseline (D8) is matched on proposals and tokens, not on money.
  5. **The project lead starts runs.** Launching proposer sessions is a side-effectful process under the user's rules, so a run starts only on the project lead's instruction.
- **Additions (2026-11-12, accepted by the project lead):**
  - **Second backend (Q5, bead dreamcounter-2170f2).** The optional backend's first use is a local open-weights model of about 7B parameters, quantised, as the weaker-model ablation. There is no paid API arm for now. That choice is revisited after P5 results show whether the limits on sampling control matter.
  - **Where the model runs.** This machine may not have the resources for the model. A Windows machine that a friend owns probably does. So the backend calls a model served over HTTP at a configured endpoint, and dc stays on this Linux machine. dc does not need to build on Windows. Each run records the served model's name, quantisation, and sampling settings. Each run needs the project lead's go (requirement 5).
  - **Feedback arm (Q9, bead dreamcounter-d8f7a4).** Every P4 arm is blind (requirement 2). A feedback arm comes later, as a separate, labelled ablation, after the blind arm has first results. In it, the proposer may call an evaluation tool (value, steps, and example results) before it submits. Its compute-matched LLM-only baseline gets the same tool and the same call budget. Evaluation calls are a third budget, beside proposals and tokens (bead dreamcounter-07fa2f).
- **Costs of the choice:** less control over sampling than an API gives; usage limits set throughput; a contamination check is needed before results can be trusted.
- **See:** D8 (compute-matched baseline), Q5 (budget), Q7 (held-out family), language spec §11, D21 (doc-amd: how proposer sessions are started and locked down, and the contamination check).
