---
id: doc-c7w
title: dc language specification (draft v0)
type: reference
scope: internal
created_at: '2026-10-27T17:11:12.135054Z'
updated_at: '2026-10-27T22:24:39.932620Z'
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
    fuel exhausted inside `bfix` or `memorec` (§6), equality on functions, an
    invalid `memorec` key, or a `memorec` `self` called in a later evaluation
    (`MemoSelfEscaped`, §6).
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
| `==` | `a -> a -> Bool` (structural) | number of value-node pairs compared (§7); error on closures |
| `pair` | `a -> b -> (a, b)` | 1 |
| `fst`, `snd` | `(a, b) -> a`, `(a, b) -> b` | 1 |
| `none`, `some` | `Option a`, `a -> Option a` | 1 |
| `option_case` | `Option a -> (Unit -> b) -> (a -> b) -> b` | 1 + calls |
| `nil`, `cons` | `List a`, `a -> List a -> List a` | 1 |
| `list_case` | `List a -> (Unit -> b) -> (a -> List a -> b) -> b` | 1 + calls |
| `length` | `List a -> Int` | 1 (lists cache their length) |
| `array_tabulate` | `Int -> (Int -> a) -> Array a` | max(n, 0) + calls |
| `array_get` | `Array a -> Int -> a` | 1 |
| `array_set` | `Array a -> Int -> a -> Array a` | 1 + ⌈log₂(n+1)⌉ |
| `array_size` | `Array a -> Int` | 1 |
| `array_of_list`, `array_to_list` | `List a -> Array a`, `Array a -> List a` | n + 1 |
| `leaf` | `Tree a` | 1 |
| `node` | `Tree a -> a -> Tree a -> Tree a` | 1 |
| `tree_case` | `Tree a -> (Unit -> b) -> (Tree a -> a -> Tree a -> b) -> b` | 1 + calls |
| `tree_fold` | `b -> (b -> a -> b -> b) -> Tree a -> b` | 1 per internal node + calls; 0 per `leaf` |

Notes:

- **Types.** The Type column is the scheme that `dc_core::types::prim_scheme` gives
  each primitive, up to renaming of type variables.
- **Operator names.** The Primitive column uses surface operators where they exist.
  The canonical core names (used in core syntax and in Stitch serialisation) are:

  | Surface | Core |
  |---|---|
  | `+`, `-`, `*`, `/`, `mod` | `add`, `sub`, `mul`, `div`, `mod` |
  | `=` (on `Int`) | `int_eq` |
  | `<`, `<=` | `lt`, `le` |
  | `==` (structural) | `eq` |
  | `::` | `cons` |
  | `&&`, `\|\|` | the special forms `and`, `or` (§3), not primitives |

  `a > b` desugars to `lt b a` and `a >= b` to `le b a`. When `a` is not a variable
  or literal, a `let` binds `a` first, so `a` is still evaluated before `b`. The
  operator sections `( > )`, `( >= )`, `( && )` and `( || )` are rejected.
- **`div` and `mod` overflow.** `div i64::MIN -1` and `mod i64::MIN -1` both raise
  `Overflow`. For `mod` the mathematical result, 0, is representable, but the
  evaluator uses checked remainder, which fails on these operands.
- **`tree_fold` charges internal nodes only.** A `leaf` makes no call of `f`, so it
  costs 0, and `tree_fold z f leaf` costs only the evaluation of its arguments.
  `eq` still counts each `leaf` as a node (§7).
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

`bfix n F x` evaluates `F self x`, where `self` is `bfix (n-1) F`. Any `bfix`
call with fuel `n ≤ 0` raises `FuelExhausted`, and this includes the first call
`bfix n F x` itself. So `bfix 0 F x` fails after 0 steps, and `bfix n` allows at
most `n` nested calls. The fuel
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
- Cost: 1 per call + calls inside `F`. The first call `bfix n F x` is charged 1
  like every recursive call. The fuel check is free: a call that raises
  `FuelExhausted` is not charged.
- Mutual recursion needs an encoding: a tag in the argument. If this becomes common
  in practice, it will be written up and raised with the project lead rather than
  added to the language quietly.

### `memorec`: memoised bounded recursion

```
memorec : Key k => Int -> ((k -> v) -> k -> v) -> k -> v
```

`memorec n F x` is open recursion like `bfix`, plus a memo table keyed on the
argument.

- The fuel `n` bounds the number of table misses across the whole table, not the
  depth. Exceeding it raises `FuelExhausted`. A per-depth bound would make a cached
  result depend on how much fuel was left when it was first computed, so the same
  call could succeed or fail depending on evaluation history. Evaluation is strict
  and left to right, so the miss count is deterministic. With `n ≤ 0` the first
  call is a miss with no misses left, so it raises `FuelExhausted` after 0 steps.
- Key types restricted to `Int`, `Bool`, `Unit` and tuples of those, so hashing is
  O(1) in practice. Subset DP uses an `Int` bitmask. Memoised recursion over a
  lattice is `memorec` over keys that encode the lattice elements.
- The type checker enforces the key restriction with a `Key` constraint on the
  key type variable. Binding a `Key` variable to `Int`, `Bool` or `Unit`
  discharges it, binding it to a pair puts the constraint on both components, and
  binding it to any other type is a `BadMemoKey` type error. The constraint is
  kept when a `let` generalises the variable, so it also holds through let-bound
  helpers and library abstractions. A closed term whose key type is still a
  variable gets a constrained scheme, for example
  `Key a => ((a -> b) -> a -> b) -> a -> b`. The evaluator also checks each key
  and raises `InvalidMemoKey`, charged 0 steps, but a well-typed program never
  reaches that error.
- Each application of the `memorec` primitive creates a new table. Calls through
  its `self` use that table. Tables are never shared between `memorec`
  applications or between evaluations.
- A table lives until the evaluation ends, not until its `memorec` returns. A
  `memorec` whose result is a function (a curried `F`) returns a closure that calls
  `self` later, and that `self` keeps working with its own table and its remaining
  misses. Releasing a table that nothing can reach changes no step count.
- A `self` returned by one evaluation and called in a later one (only host code
  can do this) raises `MemoSelfEscaped`, charged 0 steps.
- Cost: hit = 1 + key size; miss = 1 + key size + calls inside `F`. Key size is
  the number of value nodes in the key: 1 for each `Int`, `Bool`, `Unit` and pair.
  So an `Int` key costs 2 per call.
- The first call `memorec n F x` is a memo call on `x`, and with the empty table it
  is a miss. A result is stored only when its call returns, so a call on a key that
  is still being computed is another miss and uses another unit of miss fuel.
- A miss with no misses left raises `FuelExhausted` and is not charged.
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

**Readings.** These state exactly what the evaluator charges where §5 and §6 leave
a choice open. `crates/dc-eval/src/cost.rs` holds the same table, and the machine
charges only through it.

- **Primitives as values.** Evaluating a primitive of arity 1 or more to a function
  value costs 0, and so does applying it to fewer arguments than its arity. A
  nullary primitive (`nil`, `none`, `leaf`) costs 1 each time it is evaluated.
- **Charge timing.** A primitive's cost is charged when it is applied to its last
  argument, before it computes its result. A per-element or per-call cost of a
  combinator is charged just before the call of its function argument for that
  element. `unfold` is the exception: it charges an element when `g` returns
  `some`, so the final call that returns `none` costs only its own steps.
  `tree_fold` charges a node when it combines the node's two subtree results, so in
  post-order.
- **Failing primitives are charged in full.** A primitive that raises an error
  (`div` or `mod` by zero, overflow, an array index out of bounds) is charged its
  full cost first, so the step count of the error includes the failing
  application. An `eq` that reaches a function is charged the node pairs compared
  up to and including the pair with the function.
- **Failing recursive calls are not charged.** A `bfix` call with fuel ≤ 0 and a
  `memorec` miss with no misses left raise `FuelExhausted` without a charge (§6).
- **Empty iterations cost 0.** `foldl` and `foldr` on an empty list, `loop` with
  `n ≤ 0` and `array_tabulate` with `n ≤ 0` charge nothing beyond their arguments.
- **`&&` and `||` decide once**, on their left operand. When the right operand is
  evaluated, its own steps are added, but no second decision is charged.
- **`eq` node count.** Each scalar, pair, option, array, list cell, `nil`, tree
  node and `leaf` counts as one node. The comparison runs in pre-order, left to
  right, and stops at the first difference. Arrays of different sizes differ at
  their own node.
- **Timeout steps.** The machine stops with `Timeout(s)` at the first charge that
  takes the step count above the fuel. So `s > fuel`, and `s = fuel + 1` when that
  charge is a single step.
- **`eq` near the fuel limit.** An `eq` that would compare more node pairs than the
  fuel left stops counting one pair above it, so it ends with
  `Timeout(fuel + 1)`. Sharing can make a value exponentially larger than the steps
  that built it, so the full count is not computed. A `memorec` key larger than the
  fuel left is sized without being built, so its timeout count is exact.
- **Escaped memo selves.** A `memorec` `self` called after its `memorec` returned,
  within the same evaluation, costs what any `memorec` call costs (§6). A `self`
  called in a later evaluation raises `MemoSelfEscaped`, charged 0 steps.
- **Applying a program to inputs.** The evaluator evaluates the program term (free
  for a `lam`) and applies it to each input in turn. Each application of a closure
  is one β step. A program that is a primitive is charged as a primitive.

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
η-expanded at elaboration to `λy. L a0 y`. In general a use with `n` of `k`
arguments, a bare `L` included, gets `k - n` η binders, and applying the result
costs one β step per binder: one β step per missing argument. So invariant 2
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

Prelude conventions (source: `crates/dc-surface/src/prelude.dc`):

- `range lo hi` is `[lo; lo + 1; ...; hi - 1]`, and empty when `hi <= lo`. A range
  of more than `max_int` elements raises `Overflow`.
- `nth xs i` takes the list first. It returns the element at index `i`, counting
  from 0. An index out of range, negative included, raises `IndexOutOfBounds`.
- A prelude reference is a function, not a macro. It is replaced by the closed term
  of its definition (a `lam` or a literal), and applying it is ordinary
  application, charged one β step per argument. So a prelude function costs exactly
  what the same definition costs as a top-level `let` in the program. Learned
  abstractions (§8) keep macro semantics.
- Prelude references are expanded first: in the program and in each learned
  abstraction's closed term, before learned abstractions are expanded. Prelude
  definitions are closed and contain no library references, so the order does not
  change the expanded term.

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

