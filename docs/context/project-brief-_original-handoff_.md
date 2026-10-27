---
id: doc-j6s
title: Project brief (original handoff)
type: context
scope: internal
created_at: '2026-10-27T17:12:07.792457Z'
updated_at: '2026-10-27T17:12:09.162984Z'
---

This is the project lead's original brief to the first (cloud) session, copied verbatim. Only the heading markers were restored. The brief asks for `DECISIONS.md` and `RESULTS.md`; in this repository those live in DOCA as the decision docs (D1 to D8) and the results log.

---

Handoff: LLM-guided library learning for algorithmic primitives

## What this project is

A DreamCoder-style system that learns a library of reusable abstractions from solved programming tasks. It differs from existing work in its target. It looks for algorithmic primitives: structural ideas such as divide-and-conquer, sliding windows, memoised recursion over a lattice, monotone stacks, and whatever else turns out to pay for itself. Domain helpers are not the goal.

The loop has three parts:

1. Wake: an LLM, conditioned on the current library, proposes candidate programs for tasks in a small purpose-built language. Candidates are checked by fast evaluation against examples.
2. Sleep (abstraction): a compressor refactors the corpus of solutions and proposes abstractions. These are scored on description length and on measured computational cost.
3. Naming: the LLM names and documents accepted abstractions so that later proposals can use them.

The main research bet is the cost-aware objective. Pure MDL compression is indifferent between an O(n²) solution and an O(n log n) one when both pass the tests, and it will often keep the shorter, worse program. If the language reports exact, deterministic step counts, an abstraction can be rewarded for shortening programs and also for reducing their cost growth across input sizes. I expect this to be what separates algorithmic ideas from shorthand. Treat it as a hypothesis to test, not an assumption.

## Prior work to read first (before writing code)

* DreamCoder (Ellis et al., 2021): the original wake/sleep/dream architecture and its limits.
* Stitch (Bowers et al., 2023): fast corpus-guided top-down abstraction learning, written in Rust. Likely the compression backend or its starting point.
* babble (Cao et al., 2023): library learning modulo equational theories with e-graphs (egg). Relevant because algorithmic schemas share structure rather than syntax.
* LILO (Grand et al., 2023): LLM synthesis plus Stitch plus automatic documentation of abstractions. This is the baseline. Reproducing a LILO-like loop in our language is phase 1, not the contribution.
* FunSearch / AlphaEvolve: LLM-driven program evolution with evaluators. Useful for the proposal and evaluation loop design, though they don't learn libraries.

Write a short notes file summarising what each one does, its known limits, and which parts we reuse, adapt or avoid. Check whether newer work (2024 to 2026) has already done cost-aware library learning. If it has, stop and report back before building, because that changes the project's framing.

## The language

Design goals, in priority order: fast evaluation, analysable structure, exact cost accounting, LLM-writability.

* Small, pure, strict, typed lambda calculus. Hindley-Milner or simpler; polymorphism over a few base types (int, bool, list, tuple, maybe array/map).
* De Bruijn indices internally, hash-consed terms so that equality and subterm sharing are O(1).
* No unrestricted general recursion. Recursion goes only through a small set of combinators: fold, unfold, bounded fixpoint with fuel, and a memoised recursion primitive. This keeps evaluation total under fuel and gives the compressor recognisable structure. If this turns out to be too restrictive for some algorithm class, document the case and raise it with me instead of quietly adding `fix`.
* Evaluation reports a deterministic step count, never wall-clock time. Define the cost model explicitly (what one step is, what each primitive costs) and write it down. Cost growth is estimated by evaluating at several input sizes and fitting the result; start with comparing fitted log-log slopes, and keep the fitting method swappable.
* Implementation in Rust. Interpreter first, with a closure-converting interpreter or compact bytecode once profiling shows it is needed. Millions of evaluations are expected, so measure throughput (evals/sec on a reference benchmark) from the first working version and track it.
* Surface syntax close to OCaml/Haskell so the LLM writes it reasonably well. A fallback worth keeping in mind: the LLM writes a restricted Python subset that is translated into the core language. Don't build the translator unless the direct approach clearly fails; measure LLM proposal validity rate first.

## Abstraction learning

* Start by integrating or adapting Stitch for syntactic compression. Do not reimplement it.
* Score candidate abstractions on: (a) corpus description-length reduction, (b) change in cost growth of the refactored solutions, (c) reuse across distinct task families. The weighting is an open question; make it configurable and log all three components separately.
* Second stage, after the baseline works: compression modulo rewrites (babble / egg) to catch structurally shared schemas whose instances differ syntactically. Choosing the rewrite rules is itself a design problem; start with a small hand-picked set and record which ones matter.
* Optionally, let the LLM propose abstractions directly ("these solutions share structure X; here is a candidate"). Every LLM-proposed abstraction must be validated by actually refactoring the corpus with it and measuring the gain. Nothing enters the library on the LLM's say-so.

## The task corpus

This is probably a third of the work and the main determinant of what can be discovered.

* Task families where the same algorithmic idea pays off repeatedly, each with input generators that support input-size sweeps so cost differences show up.
* Held-out families for transfer evaluation, decided before any training run and not looked at during development.
* Include some families that are deliberately non-textbook (unusual combinatorial structures, made-up but well-defined problems). The LLM already knows standard algorithms. On textbook tasks, "discovery" is mostly recall, and we need families where it can't be.
* Each task: type signature, input generator, reference oracle (may be slow), and a natural-language description that can be withheld for ablations.

## LLM usage

* Proposal model: probably via API. Local hardware is a Ryzen 7 2700 (8 cores / 16 threads), 32 GB RAM, and a GPU with 8 GB VRAM, which is fine for evaluation and compression but limited for local LLM inference. Keep the LLM interface behind a trait so local and API backends are swappable, and log every prompt/response with token counts.
* Prompts include the current library with LLM-written documentation, the task signature and examples, and a few solved programs as exemplars.
* Track proposal validity rate (parses, type-checks, terminates within fuel) and solve rate per iteration. These are the first numbers to look at when something goes wrong.

## Evaluation and ablations

A library that looks plausible is not a result. The questions are:

1. Does solving held-out tasks get cheaper as the library grows, in both search effort (LLM calls / evaluations to first solution) and solution cost?
2. Do abstractions learned on one family transfer to different families?
3. Does the cost-aware objective find abstractions that pure MDL misses? Run both objectives on identical corpora and compare.
4. How much is the LLM prior doing? Ablations: no task descriptions, obfuscated primitive names, a weaker proposal model, and enumerative search with the learned library but no LLM.

Fix random seeds, log configs, and make every reported number reproducible from a single command.

## Phased plan

Track each phase as an epic, with beads small enough for one session each.

1. Literature notes and language spec. Exit: notes file, written language spec including cost model, list of open questions for me.
2. Core language. Parser, type checker, evaluator with step counting, hash-consing. Exit: property tests for the evaluator (e.g. fold laws, determinism of step counts), throughput benchmark recorded.
3. Task corpus v0. Three to five families with generators and oracles, plus one held-out family. Exit: oracle and generator tests pass; size sweeps produce sensible cost curves for hand-written reference solutions of known complexity.
4. Baseline loop. LLM proposals plus Stitch compression plus naming. Roughly LILO in our language. Exit: library grows over iterations and held-out solve rate is measured, whatever it turns out to be.
5. Cost-aware objective. Exit: head-to-head comparison against pure MDL, reported honestly whether or not it helps.
6. Structural abstraction (e-graph rewrites, LLM-proposed abstractions). Only after 5 is solid.

## Working standards

* Tests must pass because the code is correct. Never weaken, skip or special-case a test to make it pass. If a test looks wrong, explain why and ask.
* Report results as they are. Negative and null results are useful and should be written up with the same care as positive ones. Do not describe something as working or finished unless it has been run and checked.
* Keep a running `DECISIONS.md` recording design choices, the alternatives considered, and why. Keep `RESULTS.md` with dated entries for every experiment run, including failed ones.
* When a decision is genuinely mine to make (objective weighting, scope changes, adding general recursion, spending significant API budget), stop and ask rather than choosing silently.

## Open questions to raise with me early

* Which base types and primitives go into the initial language? Propose a minimal set with justification.
* How to weight compression against cost improvement, and whether cost should be a constraint or a term in the objective.
* API budget per run and which proposal model to use.
* Whether to vendor Stitch, depend on it as a crate, or fork it.
