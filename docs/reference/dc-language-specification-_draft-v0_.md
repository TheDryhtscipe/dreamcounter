---
id: doc-c7w
title: dc language specification (draft v0)
type: reference
scope: internal
created_at: '2026-10-27T17:11:12.135054Z'
updated_at: '2026-10-27T18:07:45.032272Z'
---

Status: phase 1 draft, 2026-10-27. Nothing here is implemented yet. Items marked
**[ask]** are listed in the `question` beads (Q1 to Q8) and need a decision from the project lead before
phase 2 relies on them. Design choices and rejected alternatives are logged in
the decision docs (D1 to D10). Q1 to Q3 were answered on 2026-10-27 (decisions D4,
D9 and D10).

Working name for the language: **dc**.

## 1. Goals, in priority order

1. Fast evaluation (millions of evaluations per run).
2. Analysable structure (the compressor must see recursion schemes as syntax).
3. Exact, deterministic cost accounting.
4. LLM-writability.

## 2. Types

```
τ ::= Int | Bool | Unit
    | (τ, τ)            -- pairs; n-tuples nest to the right
    | List τ
    | Option τ
    | Array τ           -- persistent, indexed
    | Tree τ            -- built-in binary tree
    | τ -> τ
    | α                 -- type variable
```

- Inference: Hindley-Milner with let-polymorphism. Top-level definitions and
  library abstractions are generalised. Lambda-bound variables are monomorphic.
- No user-defined algebraic data types in v0. A built-in binary tree type `Tree τ`
  (§5) supports heaps, binary search trees and segment trees (decision D10).
- No map type in v0. `memorec` (§6) covers memo tables, and persistent arrays cover
  indexed state.
- `Int` is 64-bit signed with **checked** arithmetic: overflow is a runtime error,
  not wraparound. This stops solutions that accidentally depend on wrapping.

## 3. Core terms

```
e ::= x_i                    -- de Bruijn index
    | λ. e
    | e e
    | let e in e             -- binds index 0 in the body
    | c                      -- literal: integer, true, false, unit
    | p                      -- primitive or combinator (table in §5, §6)
    | L                      -- reference to a library abstraction (§8)
    | if e then e else e     -- special form, lazy in branches
    | e && e | e || e        -- special forms, short-circuit
```

- Terms live in a hash-consed arena: each distinct node gets a `TermId(u32)`, and
  structurally equal terms share the same id. Equality and subterm sharing are O(1).
- De Bruijn indices are used throughout the core. Names exist only in the surface
  syntax and the pretty-printer.
- Pattern matching, tuple destructuring and multi-argument `fun` are surface sugar
  (§10) and desugar to the eliminators in §5.

## 4. Evaluation

- Strict, call-by-value, left-to-right.
- Implemented as an environment machine with an explicit continuation stack (CEK
  style), not by host recursion. Deep recursion in dc must not overflow the Rust
  stack, and an explicit machine makes step counting and fuel checks one place.
- Every evaluation returns exactly one of:
  - `Value(v, steps)`
  - `Error(kind, steps)`: division by zero, overflow, array index out of bounds,
    fuel exhausted inside `bfix` or `memorec` (§6), equality on functions.
  - `Timeout(steps)`: fuel exhausted.
- **Fuel** is a global step budget per evaluation. It is a safety net: every
  combinator except `unfold` terminates without it.

## 5. Primitives and their costs

A **step** is the unit of cost. §7 states the full cost model; this table gives
each primitive's cost when applied to evaluated arguments. "+ calls" means the steps
taken inside function arguments are added on top.

| Primitive | Type | Cost |
|---|---|---|
| `+ - * div mod` | `Int -> Int -> Int` | 1 (error on overflow / zero divisor) |
| `= < <=` | `Int -> Int -> Bool` | 1 |
| `not` | `Bool -> Bool` | 1 |
| `==` | `a -> a -> Bool` (structural) | number of value nodes compared; error on closures |
| `pair` | `a -> b -> (a, b)` | 1 |
| `fst`, `snd` | | 1 |
| `none`, `some` | | 1 |
| `option_case` | `Option a -> (Unit -> b) -> (a -> b) -> b` | 1 + calls |
| `nil`, `cons` | | 1 |
| `list_case` | `List a -> (Unit -> b) -> (a -> List a -> b) -> b` | 1 + calls |
| `length` | `List a -> Int` | 1 (lists cache their length) |
| `array_tabulate` | `Int -> (Int -> a) -> Array a` | n + calls |
| `array_get` | `Array a -> Int -> a` | 1 |
| `array_set` | `Array a -> Int -> a -> Array a` | 1 + ⌈log₂(n+1)⌉ |
| `array_size` | | 1 |
| `array_of_list`, `array_to_list` | | n + 1 |
| `leaf` | `Tree a` | 1 |
| `node` | `Tree a -> a -> Tree a -> Tree a` | 1 |
| `tree_case` | `Tree a -> (Unit -> b) -> (Tree a -> a -> Tree a -> b) -> b` | 1 + calls |
| `tree_fold` | `b -> (b -> a -> b -> b) -> Tree a -> b` | 1 per node + calls |

Notes:

- `option_case`/`list_case`/`tree_case` take the empty-case branch as a thunk, so under strict
  evaluation it runs only when taken. Calling the thunk costs one β step (§7), so the
  laziness is charged for. The surface `match` produces this form (§10).
- **No `sort` primitive.** O(n log n) sorting is something the system should have to
  build. A sort primitive can be added for a specific ablation.
- **No hash maps or mutable state.** `memorec` (§6) covers memo tables; persistent
  arrays cover indexed state at a log-factor cost.
- `array_set` is charged ⌈log₂(n+1)⌉ + 1 because the implementation will be a
  persistent vector (`im` or `rpds` crate) whose updates really are O(log n). The
  cost model tracks what the machine does.

## 6. Recursion

There is no `fix`. Recursion goes only through these combinators.

### `foldl`, `foldr`

```
foldl : (b -> a -> b) -> b -> List a -> b
foldr : (a -> b -> b) -> b -> List a -> b
```

Cost: 1 per element + calls. Both strict.

### `loop`

```
loop : Int -> b -> (Int -> b -> b) -> b      -- loop n z f = f (n-1) (... (f 0 z))
```

Counting loop with no intermediate list. Cost: 1 per iteration + calls. `n ≤ 0`
returns `z`.

### `unfold`

```
unfold : (s -> Option (a, s)) -> s -> List a
```

Cost: 1 per produced element + calls. **The one combinator whose termination is not
structural**: it halts only by returning `none` or by running out of fuel. Kept
because "generate until" is a recognisable shape the compressor should see.

### `bfix`: bounded fixpoint

```
bfix : Int -> ((a -> b) -> a -> b) -> a -> b
```

`bfix n F x` evaluates `F self x`, where `self` is `bfix (n-1) F`. If `self` is
called when the fuel `n` is 0, evaluation stops with `FuelExhausted`. The fuel
bounds the depth of the recursion: each call passes one less fuel to its own
recursive calls, and sibling calls each receive the same amount (decision D9).

- Any recursion pattern is allowed if it finishes within the fuel. This includes
  iteration to a fixed point, Collatz-style loops and search whose progress is not
  monotone in any simple integer.
- Running out of fuel is an error, not a default value. A default would let a
  truncated program pass its examples by accident.
- The fuel is an ordinary `Int` expression, not only a literal. A program can tie it
  to the input, for example `bfix (length xs + 1)` for list recursion. Such
  expressions are visible syntax that the compressor can share across programs.
- Depth fuel bounds depth, not total work: a tree recursion of depth n can make up
  to 2ⁿ calls. The global fuel (§4) bounds total work.
- Cost: 1 per recursive call + calls inside `F`. The fuel check is free.
- Mutual recursion needs an encoding: a tag in the argument. If this becomes common
  in practice, it will be written up and raised with the project lead rather than
  added to the language quietly.

### `memorec`: memoised bounded recursion

```
memorec : Int -> ((k -> v) -> k -> v) -> k -> v
```

`memorec n F x` is open recursion like `bfix`, plus a memo table keyed on the
argument.

- The fuel `n` bounds the number of table misses across the whole table, not the
  depth. Exceeding it raises `FuelExhausted`. A per-depth bound would make a cached
  result depend on how much fuel was left when it was first computed, so the same
  call could succeed or fail depending on evaluation history. Evaluation is strict
  and left to right, so the miss count is deterministic.
- Key types restricted to `Int`, `Bool`, `Unit` and tuples of those, so hashing is
  O(1) in practice. Subset DP uses an `Int` bitmask. Memoised recursion over a
  lattice is `memorec` over keys that encode the lattice elements.
- The table is created at the outermost `memorec` call and dropped when it returns.
  Nothing is shared between evaluations.
- Cost: hit = 1 + key size; miss = 1 + key size + calls inside `F`.
- Hash function: a fixed, unseeded hash (e.g. FxHash). Iteration order of the
  table is never observable, so costs stay deterministic.

## 7. Cost model

**Definition.** The cost of evaluating closed term `p` on input `x` is the number of
steps taken by the abstract strict machine of §4, where a step is:

- one β-reduction (applying a closure to one argument: curried application of an
  n-argument function costs n steps);
- one primitive application, at the cost given in §5;
- one combinator iteration or recursive call, as given in §6;
- one `if` / `&&` / `||` decision.

Free: variable lookup, `let` binding (the bound expression's own evaluation is
charged as usual), literals, closure creation.

**Invariants** (each becomes a property test in phase 2):

1. **Determinism.** `steps(p, x)` is a function of `p` and `x` only. Not of
   thread count, run, hash seed, or implementation optimisations such as
   hash-consing, caching of evaluated subterms or a later bytecode backend.
2. **Refactoring invariance.** If `p'` is `p` rewritten with library abstractions,
   then `eval(p', x) = eval(p, x)`, steps included (follows from §8).
3. **Monotone fuel.** If `eval(p, x)` finishes within fuel `f`, it gives the same
   result at every fuel `f' ≥ f`.

**What the model deliberately ignores:** memory, allocation, cache behaviour, and
wall-clock time. Constant factors in the step count are not comparable across
languages, only across dc programs.

## 8. Library abstractions

A library abstraction `L = λ#0 ... #k. body` (closed apart from its parameters, in
Stitch's form) has **macro semantics**: an application `L a0 ... ak` means exactly
the term `body[#i := ai]`, with de Bruijn shifting for parameters substituted under
binders. The implementation expands abstractions before evaluation. Expansion is
cheap because terms are hash-consed, and the expanded term is a DAG, not a tree.

Unsaturated uses (the LLM writes `L a0` where `L` takes two parameters) are
η-expanded at elaboration to `λy. L a0 y`. That adds one β step, so invariant 2
holds only for saturated uses. Stitch's rewriter always produces saturated uses.

**Why macro semantics and not ordinary function calls.** Under call-by-value, turning
an abstraction into a real function changes behaviour in two ways that matter here:

- a parameter that appeared several times in the original is now evaluated once,
  so refactoring silently performs common-subexpression elimination and *reduces*
  cost;
- a parameter that sat in a lazy position (an `if` branch, under a lambda) is now
  evaluated eagerly. That can *increase* cost, or raise an error where the original
  did not (e.g. a guarded `div`).

Either effect would let syntactic compression change both cost and meaning. Macro
semantics rules both out.

### Consequence for the research bet (important)

Under macro semantics, refactoring a fixed set of solutions with any abstraction
leaves every solution's cost unchanged. So criterion (b) in the handoff, "change in
cost growth of the refactored solutions", is **identically zero** if each task has one
solution. This is not specific to my choice of semantics: under function-call
semantics, the only cost changes would be the CSE/strictness artefacts above, which
are not what we want to reward.

Cost can still enter the objective, but through **which solution is chosen for each
task**, not through refactoring itself. Proposal (details in the decision docs (D1 to D8) D4):

- Keep a frontier `F_t` of up to K verified solutions per task, chosen for cost
  diversity, as DreamCoder kept frontiers.
- Score a library `L` with

  ```
  J(L) = DL(L) + Σ_t  min_{p ∈ F_t} [ DL(p | L) + λ · C(p) ]
  ```

  where `C(p)` is a cost penalty derived from the fitted growth of `p` (§9).
- `λ = 0` is pure MDL: each task contributes its shortest solution. With `λ > 0`, an
  abstraction earns credit by shortening the solutions that are cheap. It can also
  flip a task's selected solution from a short, slow one to a fast one that the
  abstraction has made short. Logged component (b) becomes the change in
  `Σ_t C(selected_t)`. It is nonzero exactly when selections flip.
- A constraint version is the alternative: drop from `F_t` every solution whose
  growth exponent exceeds the best known for `t` by more than ε, then do pure MDL.
- The effect we are actually betting on is downstream. A library built from
  fragments of efficient solutions should make the next wake phase propose efficient
  programs more often. That has to be measured in the next iteration's proposals,
  not in the compression score.

Implementation note: plain Stitch compresses one program per task. The simplest
integration is alternation: select per task under the current `L`, compress the
selected programs, reselect, repeat to a fixed point or a few rounds. Whether Stitch
can take multiple programs per task with a per-task minimum is something to check in
its source.

## 9. Cost growth estimation

Each task family provides `generate(n, seed)`. For a program `p`:

1. For each size `n` in the family's sweep (default `16, 32, ..., 1024`; families
   with super-polynomial reference solutions declare smaller sweeps), evaluate on 5
   seeded inputs and take the median step count.
2. Skip the sweep entirely unless `p` has already passed the task's examples.
3. Per-size fuel: 10× the reference solution's steps at that size, capped by a global
   maximum. A timeout at size `n` is recorded, not treated as a failure of `p` on
   the task.

Estimators, behind a `GrowthEstimator` trait so they can be swapped:

- **A (default): log-log slope.** Ordinary least squares of `log steps` on `log n`
  over the upper half of the sweep, which drops small-n transients. Reports slope
  `α̂` and R².
- **B: c · n^α · (log n)^γ** with `γ ∈ {0, 1, 2}` on a grid and `α` continuous; pick
  the best by residual. The same form AlgoSkill uses.
- **C (later): class selection** among {1, log n, n, n log n, n², n³, 2ⁿ} by best fit.

Known weakness of A: over `n ∈ [128, 1024]`, `n log n` gives a local slope of
roughly 1 + 1/ln n ≈ 1.17. That separates it from `n` (step counts are exact) but
not from `n^1.17`. B exists for that case. Comparisons between programs should use
the same estimator and sweep.

Default `C(p)` for §8: `α̂_p` from estimator A. This is a placeholder and the
weighting is open (see the `question` beads).

## 10. Surface syntax

OCaml/Haskell-like, because that is what the LLM writes best among small functional
syntaxes. Comments are `--`.

```
let sum xs = foldl (fun acc x -> acc + x) 0 xs

let max_subarray xs =
  fst (foldl (fun (best, cur) x ->
         let cur2 = max x (cur + x) in (max best cur2, cur2))
       (min_int, 0) xs)

let merge ab =
  bfix (length (fst ab) + length (snd ab) + 1)
    (fun self (a, b) ->
      match a with
      | [] -> b
      | x :: xs ->
        match b with
        | [] -> a
        | y :: ys -> if x <= y then x :: self (xs, b) else y :: self (a, ys))
    ab

let msort xs =
  bfix (length xs + 1)
    (fun self xs ->
      if length xs <= 1 then xs
      else
        let (l, r) = split_at (length xs / 2) xs in
        merge (self l, self r))
    xs

let fib n =
  memorec (n + 1)
    (fun self k -> if k < 2 then k else self (k - 1) + self (k - 2))
    n
```

Desugaring:

- `fun a b -> e` becomes nested lambdas. `fun (a, b) -> e` binds a fresh variable
  and uses `fst`/`snd`.
- `match xs with [] -> e1 | h :: t -> e2` becomes
  `list_case xs (fun _ -> e1) (fun h t -> e2)`. Likewise `match o with None -> e1 |
  Some x -> e2` becomes `option_case o (fun _ -> e1) (fun x -> e2)`, and
  `match t with Leaf -> e1 | Node (l, x, r) -> e2` becomes
  `tree_case t (fun _ -> e1) (fun l x r -> e2)`.
- Patterns are one level deep. Nested patterns are a parse error with a message the
  LLM can act on. This keeps desugaring trivial and error messages clear.
- Infix `::`, `+`, `<=` etc. map to primitives; `/` is `div`.

**Prelude.** `map, filter, append, reverse, take, drop, split_at, range, zip, sum,
max, min, min_int, max_int, nth, concat, any, all` are defined *in dc* on top of the combinators, so
their cost is the honest cost of their definitions. Default: prelude on, because the
LLM will reach for `map` constantly. The compressor treats prelude names as opaque
primitives. An ablation turns the prelude off and checks whether the compressor
rediscovers `map`/`filter` from the base combinators, as a sanity check on the
compressor.

**Stitch serialisation.** Core terms print to Stitch's input format: `(lam body)`,
`$i` for de Bruijn variables, application as `(f a b)`, primitives and library
abstractions as tokens. Integer literals print as tokens such as `int_3`, so Stitch
treats each literal as a distinct primitive and can abstract over it with `#i`.
Round-tripping core → Stitch → core must be the identity. That's a phase 2 property
test.

## 11. Implementation plan (Rust)

Cargo workspace, crates:

- `dc-core`: term arena, hash-consing, types, HM inference.
- `dc-surface`: lexer, parser, desugarer, pretty-printer.
- `dc-eval`: CEK machine, cost accounting, fuel, errors.
- `dc-stitch`: serialisation and the bridge to Stitch (vendored, depended on or
  forked, **[ask]**).
- `dc-tasks`: task families, generators, oracles (phase 3).
- `dc-bench`: throughput benchmark.

Values: `Int(i64) | Bool | Unit | Pair(Rc<..>) | List(Rc<Cons{head, tail, len}>) |
Option | Array(im::Vector<Value>) | Closure(Rc<{TermId, Env}>) | PrimPartial(..)`.
Environments are persistent linked frames behind `Rc`.

**Throughput benchmark** (recorded from the first working evaluator, tracked in
the results log): a fixed set of programs and inputs. `sum` and `max_subarray` on
n = 10⁴, `msort` on n = 10³, naive `fib 20`, memo `fib 60`, and a binary search
over a 10⁵ array. Report steps/second and evaluations/second, single-threaded and
across 16 threads. I have no figure to predict here; the first measurement sets the
baseline. The bytecode/closure-conversion work waits until profiling says it is
needed, as the handoff specifies.

## 12. Known gaps and where restrictions may bite

- **No mutable arrays.** Union-find, in-place partitioning and similar get an extra
  log factor from persistent arrays, and amortised analyses (path compression) do
  not carry over. Graph families should expect this.
- **Mutual recursion** needs an encoding (§6).
- **Fuel is chosen by the program.** A fuel expression that is too small for large
  inputs makes a correct algorithm fail at those sizes. The growth sweeps (§9) record
  this as errors at large n.
- **Single size parameter per family** in v0. Graph tasks with independent V and E
  will need either a fixed density or a two-parameter sweep later.
- **`unfold` relies on fuel.** If proposals often time out inside `unfold`, give it a
  fuel argument like `bfix`.

