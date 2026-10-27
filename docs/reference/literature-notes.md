---
id: doc-yty
title: Literature notes
type: reference
scope: internal
created_at: '2026-10-27T17:11:13.061930Z'
updated_at: '2026-10-27T17:11:25.841537Z'
---

Status: phase 1, written 2026-10-27. Each entry says how closely it was read:
**read** (paper or full HTML version read for this note), **abstract** (abstract or
summary page only), **recalled** (from prior knowledge, not re-read; check before
relying on details).

## 1. Gate check: has cost-aware library learning already been done?

**Answer: I found no work that learns a library of abstractions with the
computational cost of programs as part of the selection objective.** The project's
framing stands. The search covered 2024 to October 2026 across arXiv, ACL
Anthology, PLDI/EGRAPHS 2026 and ICLR 2026, using about fifteen queries (library
learning + runtime/complexity/efficiency/cost, algorithmic schemas, e-graph
follow-ups, LLM skill libraries, evolutionary program search). It is not
exhaustive. A citation crawl forward from LILO and Stitch would be the next step
if you want more certainty.

The nearest neighbours, and why each falls short of the project's idea:

| Work | What it has | What it lacks |
|---|---|---|
| **AlgoSkill** (Zhao, Song, Cai; arXiv 2606.29999, June 2026) **read** | Algorithm design for competitive programming. Complexity enters the RL reward (R_eff penalises time/memory bound violations and empirical runtime). Empirical growth fit of the form c·n^α·(log n)^γ, cross-checked against a symbolic estimate. | The library is 20 **hand-designed** skills ("Monotonicity Detection", "Caching", "Data-Structure Substitution", ...). Only the scheduler over skills is learned. No abstraction discovery. |
| **Librarian / MiniCode** (Kovačič, Chiu, Lee, Zhao, Ellis; arXiv 2506.11058, v3 Oct 2025) **read** | LLM sample-and-rerank refactoring of real codebases into a shared library. Tests as a hard constraint, MDL under a reference LM as the ranking score. | No runtime term. They also tried token count, cyclomatic complexity and maintainability index; none are execution cost. |
| **Prospective compression** (Hernandez Cano et al.; arXiv 2605.09985, 2026) **read (summary)** | Library objective that anticipates future tasks. | Pure description length (AST nodes saved). Steps appear only as a behavioural metric. |
| **ReuseRL / "Skill Reuse as Compression in Agentic RL"** (Xu et al.; arXiv 2605.31509, v2 Sept 2026) **abstract** | MDL skill dictionary as an RL regulariser. | Explicitly treats an execution-length penalty as a degenerate baseline, not part of the objective. |
| **EvoLib** (Xu et al., Microsoft; arXiv 2605.14477, May 2026) **read (summary)** | Test-time skill library with information-gain credit assignment. | Score is solution quality; efficiency means tokens, not program cost. |
| **Metaopt / "Learning efficient logic programs"** (Cropper & Muggleton, *Machine Learning* 108, 2019) **recalled + confirmed it exists** | ILP that prefers programs with lower resource complexity, not just smaller ones. The closest prior art for "cost, not size, as the selection criterion". | Learns single programs, not a reusable library. Logic programs, no LLM. |
| **Synthesis with asymptotic resource bounds** (Knoth, Wang, Polikarpova, Hoffmann; arXiv 2103.04188, 2021) **abstract** | Synthesis constrained by asymptotic resource bounds via resource-annotated types. Useful for the cost model. | No library learning. |

My reading of the gap is that the field has (a) syntactic or LM-likelihood
compression with no runtime term, (b) efficiency objectives for single programs, and
(c) complexity-aware search over a fixed hand-written skill set. Learning the
skill set itself under a cost-aware objective is open as far as I could find.

Two papers change how the project should be run even though they do not do what we do:

- **"Is This LLM Library Learning? Evaluation Must Account For Compute and
  Behaviour"** (Berlot-Attwell, Sesterhenn, Rudzicz, Si; EACL 2026) **abstract.**
  Three published in-context library learning systems failed to beat a plain
  prompting baseline once compute was matched. For LEGO-Prover they found no
  evidence that learned lemmas were reused. Consequence for us: every comparison
  needs a **compute-matched LLM-only baseline**, and we should measure actual
  reuse (how often each library entry appears in new solutions), not only solve
  rate. I've added both to the evaluation plan.
- **AlgoSkill**: its 20 hand-written skills are a ready-made list of
  what "algorithmic primitive" might mean. It's useful as a checklist when we read our
  learned library, and as an honest description of the alternative to learning.

## 2. Core papers named in the handoff

### DreamCoder (Ellis et al., PLDI 2021) **recalled**

- **What it does.** Wake: enumerative search over a typed lambda calculus, guided
  by a neural recognition model that predicts a distribution over library
  primitives per task. Abstraction sleep: refactors the frontier of solutions using
  version spaces and adds abstractions that maximise a Bayesian MDL objective
  (prior on library × likelihood of best programs). Dream sleep: trains the
  recognition model on replayed solutions and "fantasies" sampled from the library.
- **Known limits.** Enumeration is exponential in program size, so it only finds
  short programs. Version-space refactoring is very expensive in memory and time
  (LILO quotes over two CPU-months per domain). Objective is pure description
  length. Domains are small DSLs.
- **Reuse / adapt / avoid.** Reuse the overall loop and the idea of keeping a
  *frontier* of several solutions per task (important for us, see language spec §8).
  Avoid version spaces. Dream sleep is out of scope for now; the LLM plays the role
  of the recognition model.

### Stitch (Bowers et al., POPL 2023) **read**

- **What it does.** Corpus-guided top-down search for one abstraction at a time.
  Partial abstractions with holes track the corpus subtrees they match, which gives
  an upper bound on utility; branch-and-bound with dominance pruning (redundant
  argument elimination, argument capture). Optimal bottom-up rewriting by dynamic
  programming. Utility = cost(P) − cost(Rewrite(P, A)) − cost(A), with per-node
  costs for app, lam, var, abstraction-var and primitive.
- **Performance.** Comparable or better compression than DreamCoder at 3 to 4
  orders of magnitude less time and about 2 orders less memory.
- **Known limits.** One abstraction per run (libraries are built by iterating).
  No higher-order abstractions directly. No reasoning modulo equations
  (commutativity etc.). Argument capture does not apply when an argument contains a
  free variable. Utility is exact only with at most one rewrite site per program
  under the version-space extension.
- **Practicalities.** MIT licence. Rust, run as `cargo run --release --bin=compress`.
  Input is JSON programs in a Lisp syntax with de Bruijn `$i` and abstraction
  variables `#i`; any unreserved token is a primitive, so no DSL declaration is
  needed. Cost is configurable per node type (`--cost-app`, `--cost-lam`, ...).
  Python bindings live in a separate repo. I did not confirm whether the core is
  published as a library crate; see the `question` beads.
- **Reuse / adapt / avoid.** Reuse as the phase 4 compressor. Our core language
  must round-trip to Stitch's input syntax losslessly. The utility function is the
  natural place where a cost term would go, but see language spec §8: syntactic
  refactoring cannot change program cost under our semantics, so cost enters through
  frontier selection rather than inside Stitch's utility.

### babble (Cao et al., POPL 2023) **recalled + repo checked**

- **What it does.** Library learning modulo equational theories. Builds an
  e-graph of the corpus (egg), saturates with domain rewrites, anti-unifies e-class
  pairs to generate candidate abstractions, then selects with a beam-search
  extraction.
- **Known limits.** Anti-unification is quadratic in corpus size (stated by the
  E-Stitch authors). Results depend on the rewrite rules supplied. Repo is MIT,
  Rust, marked "experimental", POPL artefact on a separate branch.
- **Reuse / adapt / avoid.** Phase 6 only. Consider **E-Stitch** instead (below).

### LILO (Grand et al., ICLR 2024) **read**

- **What it does.** Dual search: Codex proposes programs from few-shot prompts of
  solved tasks, plus enumerative search over a task-conditioned grammar. Stitch
  compresses solutions; the library is re-derived from the base DSL each iteration.
  AutoDoc has an LLM name and document each abstraction.
- **Results.** Solve rates over DreamCoder: REGEX 77.1% vs 43.9%, LOGO 49.0% vs
  28.5%, CLEVR 96.8% vs 94.5%. AutoDoc helped REGEX and CLEVR but not LOGO.
- **Known limits (authors').** AutoDoc names are sometimes wrong or uninformative.
  High run-to-run variance. The LLM-only baseline also improves over iterations, so
  the library's share of the gain is a matter of degree. Context-length scaling of
  in-context libraries is unclear. Codex via a free beta harms reproducibility.
- **Reuse / adapt / avoid.** Phase 4 is a LILO reproduction in our language.
  Adapt: re-deriving the library from scratch each iteration is a sensible default
  (no path dependence). Avoid: any dependence on a model that may disappear; log
  everything and pin model versions.

### FunSearch (Romera-Paredes et al., Nature 2024) / AlphaEvolve (DeepMind, 2025) **recalled**

- **What they do.** LLM-driven program evolution. A program database (islands in
  FunSearch; a richer archive in AlphaEvolve) supplies parent programs to the
  prompt; an automatic evaluator scores children; good children are kept.
  AlphaEvolve evolves whole files with diff-based edits and multi-objective scores.
- **Known limits.** No library: knowledge lives in the population of programs and in
  the prompt, not in reusable named abstractions. Need a cheap, trustworthy
  evaluator; reward hacking against the evaluator is a known failure.
- **Reuse / adapt / avoid.** Adapt the evaluator discipline (deterministic, sandboxed,
  cheap first-pass filters before expensive checks) and the idea of sampling parents
  from a diverse archive for prompts. Open-source reimplementations exist
  (OpenEvolve; ShinkaEvolve, arXiv 2509.19349, which targets sample efficiency); I
  have not read them in detail.

## 3. Other relevant work found

- **E-Stitch** (Gupta, Bowers, Solar-Lezama; EGRAPHS @ PLDI 2026) **abstract.**
  Stitch's top-down search lifted to e-graphs, so it handles equivalences without
  babble's quadratic anti-unification. Preliminary results beat babble on speed and
  compression on babble's benchmarks. No code link on the workshop page. **Likely
  the better phase 6 backend if code becomes available**; worth emailing the authors.
- **ReGAL** (Stengel-Eskin, Prasad, Bansal; 2024) **recalled.** LLM refactors
  programs into helper functions, verified by execution. An LLM-side alternative to
  Stitch; relevant to the optional "LLM proposes abstractions" path.
- **TroVE** (Wang, Neubig, Fried; 2024) **recalled.** Induces a toolbox of functions,
  growing and trimming it by usage. Same relevance as ReGAL.
- **AbstractBeam** (2024) **abstract.** Library learning combined with LambdaBeam
  bottom-up search. Relevant if we build the enumerative-search ablation.
- **Leroy** (2024) **abstract.** Library learning for imperative languages built on
  Stitch. Relevant only if we fall back to a Python subset.
- **TheoryCoder-2** (Ahmed, Irie, Tenenbaum, Bates, Gershman; arXiv 2602.00929, Jan
  2026) **read (summary).** LLM-synthesised PDDL abstractions for planning; no
  explicit selection objective. The authors note they have no quantitative way to
  evaluate abstraction quality, which is the problem our cost metric partly answers.
- **Library learning with e-graphs on jazz harmony** (Ren et al., arXiv 2605.04622,
  2026) **title only.** A further babble-style application.
- **Smith's KIDS / algorithm design tactics** (1980s to 90s) **recalled.** Hand-written
  schemas for divide-and-conquer, global search, etc., instantiated by deduction. The
  classical version of "algorithmic primitives as abstractions". Useful for writing
  the reference schemas we hope the system rediscovers.

## 4. Summary: reuse, adapt, avoid

- **Reuse:** Stitch (phase 4 compressor). The LILO loop shape. Compute-matched
  baselines and reuse measurement from Berlot-Attwell et al.
- **Adapt:** DreamCoder's per-task frontier (made cost-aware). AlgoSkill's growth fit
  c·n^α·(log n)^γ as a second estimator alongside the log-log slope. FunSearch-style
  evaluator discipline.
- **Avoid:** Version-space refactoring. babble's anti-unification at scale (prefer
  E-Stitch if available). Any evaluation without a compute-matched baseline.

## Sources

- [AlgoSkill, arXiv 2606.29999](https://arxiv.org/abs/2606.29999)
- [Refactoring Codebases through Library Design, arXiv 2506.11058](https://arxiv.org/html/2506.11058v3)
- [Prospective Compression in Human Abstraction Learning, arXiv 2605.09985](https://arxiv.org/html/2605.09985v1)
- [Skill Reuse as Compression in Agentic RL, arXiv 2605.31509](https://www.alphaxiv.org/abs/2605.31509)
- [EvoLib, arXiv 2605.14477](https://arxiv.org/html/2605.14477v1)
- [Learning efficient logic programs, Machine Learning 2019](https://link.springer.com/doi/10.1007/s10994-018-5712-6)
- [Synthesis with Asymptotic Resource Bounds, arXiv 2103.04188](https://arxiv.org/abs/2103.04188)
- [Is This LLM Library Learning?, EACL 2026](https://aclanthology.org/2026.eacl-long.163/)
- [Stitch, arXiv 2211.16605](https://arxiv.org/pdf/2211.16605) and [repo](https://github.com/mlb2251/stitch)
- [babble, POPL 2023](https://cseweb.ucsd.edu/~npolikarpova/publications/popl23.pdf) and [repo](https://github.com/dcao/babble)
- [LILO, arXiv 2310.19791](https://arxiv.org/html/2310.19791v4)
- [E-Stitch, EGRAPHS 2026](https://pldi26.sigplan.org/details/egraphs-2026-papers/7/E-Stitch-Top-Down-Library-Learning-for-E-Graphs)
- [ReGAL, arXiv 2401.16467](https://arxiv.org/abs/2401.16467)
- [AbstractBeam, arXiv 2405.17514](https://arxiv.org/abs/2405.17514)
- [Leroy, arXiv 2410.06438](https://arxiv.org/pdf/2410.06438)
- [TheoryCoder-2, arXiv 2602.00929](https://arxiv.org/html/2602.00929)
- [Library learning with e-graphs on jazz harmony, arXiv 2605.04622](https://arxiv.org/html/2605.04622)
- [ShinkaEvolve, arXiv 2509.19349](https://arxiv.org/pdf/2509.19349)
- [AlphaEvolve white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
