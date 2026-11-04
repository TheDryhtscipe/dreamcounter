---
id: doc-yty
title: Literature notes
type: reference
scope: internal
created_at: '2026-10-27T17:11:13.061930Z'
updated_at: '2026-10-28T12:00:09.024238Z'
---

Status: phase 1, written 2026-10-27. Each entry says how closely it was read:
**read** (paper or full HTML version read for this note), **abstract** (abstract or
summary page only), **recalled** (from prior knowledge, not re-read; check before
relying on details), **verified 2026-10-28** (claims re-checked against the paper
itself for bead dreamcounter-42ade7, or for bead dreamcounter-c69fcd where the entry
says so).

Citation check, 2026-10-28: every arXiv id, DOI and venue in this note was checked
against arxiv.org, Crossref or the ACL Anthology. Where an entry was wrong, the
corrected details are given in place. An entry still marked **recalled** has a
verified citation, but its claims were not re-read.

## 1. Gate check: has cost-aware library learning already been done?

**Answer: I found no work that learns a library of abstractions with the
computational cost of programs as part of the selection objective.** The project's
framing stands. The search covered 2024 to October 2026 across arXiv, ACL
Anthology, PLDI/EGRAPHS 2026 and ICLR 2026, using about fifteen queries (library
learning + runtime/complexity/efficiency/cost, algorithmic schemas, e-graph
follow-ups, LLM skill libraries, evolutionary program search). It is not
exhaustive. A forward-citation crawl from LILO and Stitch followed on 2026-10-28
(bead dreamcounter-c69fcd; Results log, doc-m4a, entry "Forward-citation crawl from
LILO and Stitch"). It found no paper that puts the runtime or growth of programs in
a library-learning objective. Its nearest neighbours are AMaze, Cobbler, Achille
and Soatto, and Zhou et al. (in the table below) and Berlot-Attwell et al. (after
the table).

The nearest neighbours, and why each falls short of the project's idea:

| Work | What it has | What it lacks |
|---|---|---|
| **AlgoSkill** (Song, Cai, Zhao; arXiv 2606.29999, June 2026) **read; verified 2026-10-28** | Algorithm design for competitive programming. Complexity enters the RL reward (R_eff penalises time/memory bound violations and empirical runtime). Empirical growth fit of the form c·n^α·(log n)^γ, cross-checked against a symbolic estimate. | The library is 20 **hand-designed** skills ("Monotonicity Detection", "Caching", "Data-Structure Substitution", ...). Only the scheduler over skills is learned. No abstraction discovery. |
| **Librarian / MiniCode** (Kovačič, Chiu, Lee, Zhao, Ellis; arXiv 2506.11058, v1 May 2025, v3 5 Oct 2025) **read; citation verified 2026-10-28** | LLM sample-and-rerank refactoring of real codebases into a shared library. Tests as a hard constraint, MDL under a reference LM as the ranking score. | No runtime term. They also tried token count, cyclomatic complexity and maintainability index; none are execution cost. |
| **Prospective compression** (Hernandez Cano et al.; arXiv 2605.09985, May 2026) **read (summary); citation verified 2026-10-28** | Library objective that anticipates future tasks. | Pure description length (AST nodes saved). Steps appear only as a behavioural metric. |
| **ReuseRL / "Skill Reuse as Compression in Agentic RL"** (Xu et al.; arXiv 2605.31509, v2 Sept 2026; accepted at EMNLP 2026 main conference per the arXiv comment) **verified 2026-10-28** | MDL skill dictionary as an RL regulariser. | Explicitly treats a pure round-length penalty (a uniform cost per agent step) as a degenerate fixed-code instance of its objective, and uses it as a baseline. It is not part of the proposed objective. |
| **EvoLib** ("Test-Time Learning with an Evolving Library", Xu, Sordoni, Singh, Gero, Galley, Yuan, Gao; arXiv 2605.14477, v1 May 2026, v2 July 2026; the Microsoft affiliation was not checked) **read (summary); citation verified 2026-10-28** | Test-time skill library with information-gain credit assignment. | Score is solution quality; efficiency means tokens, not program cost. |
| **Metaopt / "Learning efficient logic programs"** (Cropper & Muggleton, *Machine Learning* 108, 2019; online April 2018, DOI 10.1007/s10994-018-5712-6) **recalled; citation verified 2026-10-28, claims not re-read** | ILP that prefers programs with lower resource complexity, not just smaller ones. The closest prior art for "cost, not size, as the selection criterion". | Learns single programs, not a reusable library. Logic programs, no LLM. |
| **Synthesis with asymptotic resource bounds** (Hu, Cyphert, D'Antoni, Reps; CAV 2021, arXiv 2103.04188) **abstract; verified 2026-10-28** | Synthesises recursive functions that meet a functional specification and a big-O resource bound. A type system assigns asymptotic complexity to terms and tracks recurrence relations, justified by the Master Theorem and Akra–Bazzi. Tool: SynPlexity. Useful for the cost model. Correction: an earlier version of this note gave the authors as Knoth, Wang, Polikarpova and Hoffmann. They wrote a different, related paper: "Resource-guided program synthesis" (PLDI 2019), which uses resource-annotated types with concrete bounds. | No library learning. |
| **AMaze** (Ye, Ji, Xiong, Zhang; "Accelerating Syntax-Guided Program Synthesis by Optimizing Domain-Specific Languages", PACMPL 10 (POPL), January 2026, pp. 1066–1093, DOI 10.1145/3776679) **abstract; citation verified 2026-10-28 against Crossref. Found by the forward-citation crawl.** | Changes the DSL of a syntax-guided synthesiser so that synthesis runs faster. A machine-learned model estimates synthesis time from the enumeration ranks of "feature components", so the synthesiser need not be run on each candidate DSL. | The cost is the synthesiser's search time, not the runtime of the synthesised programs. The crawl entry says the paper sets itself against library learning, which it describes as minimising program size; this note has not re-checked that, because the full text was not reachable. |
| **Cobbler** (Lubin, Ferguson, Ye, Yim, Chasins; "Equivalence by Canonicalization for Synthesis-Backed Refactoring", PACMPL 8 (PLDI), article 223, June 2024, DOI 10.1145/3656453) **read (problem statement, evaluation and related work); verified 2026-10-28 for bead dreamcounter-c69fcd** | Enumerative synthesis that rewrites direct-style code into calls to a given component library (Elm combinators, NumPy). Equivalence is checked by canonicalising both programs. NumPy rewrites are timed afterwards: at input size 10⁸ the median speed-up is 1.95× (synthesis time included), and 42 of 56 programs speed up. | The library is fixed, not learned. The paper calls library learning the converse of its problem (§10.2.3). Runtime is measured after synthesis and is not in the search objective. The objective is equivalence, and the cost metrics it suggests count components or holes. |
| **Achille and Soatto**, "AI Agents as Universal Task Solvers" (arXiv 2510.12066, v1 October 2025, v2 February 2026) **read (sections 1 to 6 of the HTML version); verified 2026-10-28 for bead dreamcounter-c69fcd** | Theory. Learning should find structure that makes future inference faster, and time is the quantity to optimise when scaling reasoning models. The Occam's-razor preference for compression is the wrong target for this. Proposes time-weighted sampling, a time-penalised reward and Gittins-index stopping rules. | "Time" is the solver's own inference time (the length of a reasoning trace, weighted by its probability), not the runtime of the programs it outputs. It proposes no library-learning algorithm. It supports our motivation, not our method. The Results log title adds "It's All About Time"; arXiv's record for both versions does not. |
| **Zhou et al.** (Zhou, Nagy, Dayan, Wu; "Path-dependent program induction under resource constraints explains human sequence learning", arXiv 2606.20623, v1 May 2026) **read (the cost model and discussion of the HTML version); verified 2026-10-28 for bead dreamcounter-c69fcd** | A cognitive model of sequence learning: a hierarchical adaptor grammar with a local and a global library, learned under a memory cost (description length) and a computational cost. | The computational cost is an inference budget: a search budget (how many candidate subprograms the learner considers) and a backtracking budget. The execution cost of the induced programs is not modelled. |

My reading of the gap is that the field has (a) syntactic or LM-likelihood
compression with no runtime term, (b) efficiency objectives for single programs, and
(c) complexity-aware search over a fixed hand-written skill set. Learning the
skill set itself under a cost-aware objective is open as far as I could find.

Two papers change how the project should be run even though they do not do what we do:

- **"Is This LLM Library Learning? Evaluation Must Account For Compute and
  Behaviour"** (Berlot-Attwell, Sesterhenn, Rudzicz, Si; EACL 2026, Rabat, March
  2026, pp. 3534–3568) **verified 2026-10-28.** They studied three published
  in-context library learning systems: LEGO-Prover, TroVE and AgentOptimizer. None
  consistently outperformed a plain prompting baseline. Accuracy gains often
  vanished or reversed once compute was accounted for. They measure compute as
  total LLM tokens (input plus output). For LEGO-Prover they found no evidence of
  direct reuse of learned lemmas, and evidence against soft reuse (reuse by
  modifying related examples). Consequence for us: every comparison
  needs a **compute-matched LLM-only baseline**, and we should measure actual
  reuse (how often each library entry appears in new solutions), not only solve
  rate. I've added both to the evaluation plan. DOI
  10.18653/v1/2026.eacl-long.163 (checked against Crossref 2026-10-28). The
  forward-citation crawl (bead dreamcounter-c69fcd; Results log, doc-m4a,
  2026-10-28 entry) lists it as a close hit. How it differs from our claim: its cost
  is the compute of the learning system, used to make evaluation fair; the runtime
  of the learned programs is not part of it.
- **AlgoSkill**: its 20 hand-written skills are a ready-made list of
  what "algorithmic primitive" might mean. It's useful as a checklist when we read our
  learned library, and as an honest description of the alternative to learning.

## 2. Core papers named in the handoff

### DreamCoder (Ellis et al., PLDI 2021) **verified 2026-10-28**

Checked against the main text of arXiv 2006.08381 ("DreamCoder: Growing
generalizable, interpretable knowledge with wake-sleep Bayesian program learning").
The PLDI 2021 version has a different title: "DreamCoder: bootstrapping inductive
program synthesis with wake-sleep library learning" (DOI 10.1145/3453483.3454080).
The supplement was not re-read.

- **What it does.** Wake: enumerative search over a polymorphically typed lambda
  calculus, guided by a neural recognition model. The model predicts, from a task's
  examples, an approximate posterior over programs. (Its exact parametrisation is in
  the supplement, which was not checked.) Each task keeps a beam of the k = 5
  programs with the highest posterior. Abstraction sleep: refactors those programs
  using version spaces (combined with ideas from e-graphs; refactorings bounded to
  3 evaluation steps) and adds abstractions that maximise a Bayesian MDL objective
  (description length of the library plus description lengths of the refactored
  programs). Dream sleep: trains the recognition model on a 50/50 mix of replayed
  solutions and "fantasies" sampled from the library.
- **Known limits.** Enumeration is exponential in program size, so it only finds
  short programs. Version-space refactoring is very expensive in memory and time
  (LILO quotes over two CPU-months per domain, citing DreamCoder's appendix J; the
  DreamCoder paper says learning its "origami" functional-programming library took
  about a year of total CPU time). Objective is pure description length. Domains
  are small DSLs.
- **Reuse / adapt / avoid.** Reuse the overall loop and the idea of keeping a
  *frontier* of several solutions per task (important for us, see language spec §8).
  Avoid version spaces. Dream sleep is out of scope for now; the LLM plays the role
  of the recognition model.

### Stitch (Bowers et al., POPL 2023) **read; verified 2026-10-28**

"Top-Down Synthesis for Library Learning", Bowers, Olausson, Wong, Grand,
Tenenbaum, Ellis, Solar-Lezama; PACMPL 7 (POPL), article 41, January 2023.

- **What it does.** Corpus-guided top-down search for one abstraction at a time.
  Partial abstractions with holes track the corpus subtrees they match, which gives
  an upper bound on utility; branch-and-bound with dominance pruning (redundant
  argument elimination, argument capture). Optimal bottom-up rewriting by dynamic
  programming. Utility = cost(P) − cost(Rewrite(P, A)) − cost(A), with per-node
  costs for app, lam, var, abstraction-var and primitive.
- **Performance.** Comparable or better compression than DreamCoder at 3 to 4
  orders of magnitude less time and about 2 orders less memory.
- **Known limits.** One abstraction per run (libraries are built by iterating).
  No higher-order abstractions directly. No native reasoning modulo equations
  (commutativity etc.); §5 of the paper shows Stitch can instead search over a
  version space produced by a rewrite system, and §6.5 uses this to recover
  higher-order functions. Argument capture does not apply when an argument contains a
  free variable. Utility is exact only with at most one rewrite site per program
  under the version-space extension.
- **Practicalities.** MIT licence. Rust, run as `cargo run --release --bin=compress`.
  Input is JSON programs in a Lisp syntax with de Bruijn `$i` and abstraction
  variables `#i`; any unreserved token is a primitive, so no DSL declaration is
  needed. Cost is configurable per node type (`--cost-app`, `--cost-lam`, ...).
  Python bindings live in a separate repo. The core is a library crate:
  `Cargo.toml` names the package `stitch_core` (MIT) and the repo has
  `src/lib.rs` (checked 2026-10-28). crates.io has only `stitch_core` 0.1.0, from
  November 2022. Current `main` depends on the `lambdas` crate by git revision, so
  it cannot be published to crates.io as it stands; we would depend on Stitch by
  git revision.
- **Reuse / adapt / avoid.** Reuse as the phase 4 compressor. Our core language
  must round-trip to Stitch's input syntax losslessly. The utility function is the
  natural place where a cost term would go, but see language spec §8: syntactic
  refactoring cannot change program cost under our semantics, so cost enters through
  frontier selection rather than inside Stitch's utility.

### babble (Cao et al., POPL 2023) **verified 2026-10-28**

"babble: Learning Better Abstractions with E-Graphs and Anti-unification", Cao,
Kunkel, Nandi, Willsey, Tatlock, Polikarpova; PACMPL 7 (POPL), article 14, January
2023. Re-read the paper and checked the repo.

- **What it does.** Library learning modulo equational theories. Builds an
  e-graph of the corpus (egg), saturates with domain rewrites, anti-unifies e-class
  pairs to generate candidate abstractions, then selects a library with a custom
  extraction (the authors found ILP extraction did not scale). A beam-search
  approximation trades precision for scalability.
- **Known limits.** Anti-unification is quadratic in corpus size: the E-Stitch
  authors state this, and the babble paper notes that there are quadratically many
  e-class pairs to consider. Results depend on the rewrite rules supplied. Repo is MIT,
  Rust, marked "experimental", POPL artefact on a separate branch.
- **Reuse / adapt / avoid.** Phase 6 only. Consider **E-Stitch** instead (below).

### LILO (Grand et al., ICLR 2024) **read; verified 2026-10-28**

"LILO: Learning Interpretable Libraries by Compressing and Documenting Code",
Grand, Wong, Bowers, Olausson, Liu, Tenenbaum, Andreas; arXiv 2310.19791 v4 (ICLR
2024 camera-ready).

- **What it does.** Dual search: Codex proposes programs from few-shot prompts of
  solved tasks, plus enumerative search over a task-conditioned grammar. Stitch
  compresses solutions; the library is re-derived from the base DSL each iteration.
  Stitch is the compressor for every library-learning condition in the paper,
  including the DreamCoder baseline. AutoDoc has an LLM name and document each
  abstraction.
- **Results.** Solve rates over DreamCoder: REGEX 77.1% vs 43.9%, LOGO 49.0% vs
  28.5%, CLEVR 96.8% vs 94.5%. These are mean test solve rates over 3 seeds in
  online synthesis (Table 1). The LLM Solver (+ Search) baseline reached 76.6% on
  REGEX and 96.4% on CLEVR, so LILO's clear margin over it is on LOGO. AutoDoc
  helped REGEX (+9.73) and CLEVR (+2.27) but not LOGO (+0.00).
- **Known limits (authors').** AutoDoc names are sometimes wrong or uninformative.
  High run-to-run variance (LOGO standard deviation 22 points for LILO). The
  LLM-only baseline also improves over iterations, so the library's share of the
  gain is a matter of degree. Context-length scaling of in-context libraries is
  unclear. Correction: the authors do not list the Codex free beta as a
  limitation. They used OpenAI's free Codex beta for researchers and released all
  LLM generations to preserve reproducibility. The reproducibility risk of a
  withdrawn model is our concern, not a claim of the paper.
- **Reuse / adapt / avoid.** Phase 4 is a LILO reproduction in our language.
  Adapt: re-deriving the library from scratch each iteration is a sensible default
  (no path dependence). Avoid: any dependence on a model that may disappear; log
  everything and pin model versions.

### FunSearch (Romera-Paredes et al., Nature 2024) / AlphaEvolve (Novikov et al., 2025) **verified 2026-10-28**

FunSearch: "Mathematical discoveries from program search with large language
models", *Nature* 625 (2024), published online 14 December 2023, DOI
10.1038/s41586-023-06924-6. AlphaEvolve: "AlphaEvolve: A coding agent for
scientific and algorithmic discovery", Novikov et al. (Google DeepMind), arXiv
2506.13131, June 2025. Both re-read for the claims below.

- **What they do.** LLM-driven program evolution. A program database supplies
  parent programs to the prompt; an automatic evaluator scores children; good
  children are kept. FunSearch evolves a single function (for example a `priority`
  function) inside a fixed skeleton that the user writes. Its database is an
  island model: a prompt takes k programs from one island, favouring higher-scoring
  and shorter programs ("best-shot prompting"), and the worst half of the islands
  is periodically reseeded from the best. AlphaEvolve evolves whole code files with
  diff-based edits, can optimise several metrics at once, and its database combines
  MAP-Elites with island models. It also supports an evaluation cascade: harder,
  more expensive tests run only on candidates that pass cheaper ones.
- **Known limits.** No library: knowledge lives in the population of programs and in
  the prompt, not in reusable named abstractions. Both need an automated evaluator;
  AlphaEvolve's authors name this as its main limitation, and FunSearch needed fast
  evaluation (at most 20 minutes on 1 CPU, per the AlphaEvolve paper's comparison).
  Reward hacking against the evaluator is our concern; neither paper reports it
  as a failure they observed.
- **Reuse / adapt / avoid.** Adapt the evaluator discipline (deterministic, sandboxed,
  cheap first-pass filters before expensive checks) and the idea of sampling parents
  from a diverse archive for prompts. Open-source reimplementations exist
  (OpenEvolve; ShinkaEvolve, Lange, Imajuku, Cetin, arXiv 2509.19349, which
  targets sample efficiency; citation verified 2026-10-28); I have not read them
  in detail. The OpenEvolve reference was not checked.

## 3. Other relevant work found

- **E-Stitch** (Gupta, Bowers, Solar-Lezama; EGRAPHS @ PLDI 2026) **abstract.**
  Stitch's top-down search lifted to e-graphs, so it handles equivalences without
  babble's quadratic anti-unification. Preliminary results beat babble on speed and
  compression on babble's benchmarks. Authors and venue verified 2026-10-28 on the
  workshop page, which lists a preprint PDF but no code link. **Likely
  the better phase 6 backend if code becomes available**; worth emailing the authors.
- **ReGAL** (Stengel-Eskin, Prasad, Bansal; ICML 2024, arXiv 2401.16467)
  **recalled; citation verified 2026-10-28.** LLM refactors
  programs into helper functions, verified by execution. An LLM-side alternative to
  Stitch; relevant to the optional "LLM proposes abstractions" path.
- **TroVE** (Wang, Fried, Neubig; arXiv 2401.12869, 2024) **abstract; verified
  2026-10-28.** Induces a toolbox of functions by using, growing and periodically
  trimming it. Same relevance as ReGAL. TroVE is one of the three systems that
  Berlot-Attwell et al. found did not consistently beat a compute-matched
  prompting baseline.
- **AbstractBeam** (Zenkner, Dierkes, Sesterhenn, Bartelt; arXiv 2405.17514, 2024)
  **abstract; citation verified 2026-10-28.** Library learning combined with LambdaBeam
  bottom-up search. Relevant if we build the enumerative-search ablation.
- **Leroy** (Bellur, Alghamdi, Workneh, Izraelevitz; HATRA 2024 workshop, arXiv
  2410.06438) **abstract; citation verified 2026-10-28.** Library learning for
  imperative languages built on Stitch. Relevant only if we fall back to a Python
  subset.
- **TheoryCoder-2** ("Learning Abstractions for Hierarchical Planning in
  Program-Synthesis Agents"; Ahmed, Irie, Tenenbaum, Bates, Gershman; arXiv
  2602.00929, Jan 2026) **read (summary); citation verified 2026-10-28, claims not
  re-read.** LLM-synthesised PDDL abstractions for planning; no
  explicit selection objective. The authors note they have no quantitative way to
  evaluate abstraction quality, which is the problem our cost metric partly answers.
- **Library learning with e-graphs on jazz harmony** (Ren, Bowers, Guan, Rohrmeier; arXiv
  2605.04622, May 2026) **title only; citation verified 2026-10-28.** A further babble-style application.
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
- [Synthesis with Asymptotic Resource Bounds, arXiv 2103.04188](https://arxiv.org/abs/2103.04188) and [CAV 2021](https://doi.org/10.1007/978-3-030-81685-8_37)
- [Resource-guided program synthesis, PLDI 2019](https://doi.org/10.1145/3314221.3314602)
- [AMaze, PACMPL 10 (POPL) 2026](https://doi.org/10.1145/3776679)
- [Cobbler, PACMPL 8 (PLDI) 2024](https://doi.org/10.1145/3656453)
- [AI Agents as Universal Task Solvers, arXiv 2510.12066](https://arxiv.org/abs/2510.12066)
- [Path-dependent program induction under resource constraints, arXiv 2606.20623](https://arxiv.org/abs/2606.20623)
- [DreamCoder, arXiv 2006.08381](https://arxiv.org/abs/2006.08381) and [PLDI 2021](https://doi.org/10.1145/3453483.3454080)
- [Is This LLM Library Learning?, EACL 2026](https://aclanthology.org/2026.eacl-long.163/)
- [Stitch, arXiv 2211.16605](https://arxiv.org/pdf/2211.16605) and [repo](https://github.com/mlb2251/stitch)
- [babble, POPL 2023](https://cseweb.ucsd.edu/~npolikarpova/publications/popl23.pdf) ([DOI](https://doi.org/10.1145/3571207)) and [repo](https://github.com/dcao/babble)
- [LILO, arXiv 2310.19791](https://arxiv.org/html/2310.19791v4)
- [E-Stitch, EGRAPHS 2026](https://pldi26.sigplan.org/details/egraphs-2026-papers/7/E-Stitch-Top-Down-Library-Learning-for-E-Graphs)
- [ReGAL, arXiv 2401.16467](https://arxiv.org/abs/2401.16467)
- [TroVE, arXiv 2401.12869](https://arxiv.org/abs/2401.12869)
- [AbstractBeam, arXiv 2405.17514](https://arxiv.org/abs/2405.17514)
- [Leroy, arXiv 2410.06438](https://arxiv.org/pdf/2410.06438)
- [TheoryCoder-2, arXiv 2602.00929](https://arxiv.org/html/2602.00929)
- [Library learning with e-graphs on jazz harmony, arXiv 2605.04622](https://arxiv.org/html/2605.04622)
- [ShinkaEvolve, arXiv 2509.19349](https://arxiv.org/pdf/2509.19349)
- [FunSearch, Nature 625](https://doi.org/10.1038/s41586-023-06924-6)
- [AlphaEvolve, arXiv 2506.13131](https://arxiv.org/abs/2506.13131)
- [AlphaEvolve white paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
