---
id: doc-ue2
title: 'D12: The locate walk caches failures as facts about types, matched by instance'
type: decision
scope: internal
created_at: '2026-10-28T18:03:21.359838Z'
updated_at: '2026-10-28T22:54:09.060668Z'
---

- **Date:** 2026-10-28. **Status:** superseded by D13 (doc-mnj), 2026-10-28. It was never merged (PR 28). It is kept as the record of why no failure-cache key could end the doubling. Bead `dreamcounter-bb28d2`.
- **Context:** The type checker walks an ill-typed term a second time, the locate walk, so that the error it reports is the one the tree walk would report (`TypeChecker::run` in `crates/dc-core/src/types/infer.rs`). Each use of a shared subterm is tried from a checkpoint. If the use fails, the walk goes back to the checkpoint and walks the subterm in place. A cache of failed keys stops the walk from trying a use that is known to fail. Without the cache, a chain of `n` shared subterms over an ill-typed leaf costs `2^n`.
- **Problem:** A key whose scope has a polymorphic binding with free type variables (an open scope) can fail in one state of the store and work in another. Two validity rules were tried, and each adversarial review found a new exponential variant:
  1. The failure is valid until the walk goes back past the trail position it was noted at. Round 1 found that unrelated trail writes, including no-op path compression, invalidate every entry (shape G6).
  2. Rule 1, or else the store reachable from the slots is unchanged. Round 2 found that a condition at each level that refines a node made by the level above invalidates every entry.

  Both rules tie a failure to the store's history (trail positions and node identities). Going back to a checkpoint changes that history even when the types involved are unchanged, or are made more specific.

## Decision

Record each failure as a fact about types that does not mention the store. The fact holds in every state, so nothing ever invalidates it.

1. **The fact.** When fresh inference of a use with key `(t, scope)` fails, the checker is back at the use's checkpoint. It writes the scope's current types into a *failure pattern*:
   - each monomorphic slot as `Mono`;
   - each polymorphic slot whose type has no free variables as `Closed(c)`, where `c` is the canonical id of that type;
   - each other polymorphic slot as `Open(shape, images)`. `shape` is the canonical id of the type the binding had when it was bound, with its `i`th free node, in order of first occurrence, written as open variable `i` (`TypeChecker::shape_of_class`, cached per canonical id). It fixes the binding's quantified variables and where its free nodes occur, but not which nodes they are. `images` are the canonical ids of what those free nodes resolve to now. Shape and images together give the binding's type now, up to renaming its unbound variables. Links are followed. Unbound variables that are not quantified are numbered in order of first occurrence across the whole scope, with their `Key` flag (a new `Canon::Open { index, key }`). Rigid variables keep their node, because rigid nodes are made before the walk and never change. The shape and free nodes of each canonical id are found once and cached, so a pattern costs time in what the free variables have become, not in the size of the bindings' types.

   The pattern names no node that can change, so it means the same in every state of the store. The fact is: "subterm `t`, inferred afresh with these slot types, fails."
2. **The lookup.** A later use of `t` is walked in place at once if its current pattern is an *instance* of a recorded pattern for `t`. That holds when some substitution of the recorded pattern's open variables makes it equal to the current pattern. Each slot must have the same kind in both patterns, the same canonical id if closed, and the same shape if open, so the bindings have the same quantified variables and their free nodes occur in the same places. The substitution must keep rigid variables, must send `Key` variables to key-safe types, and must not send an open variable to a type that contains a quantified variable of a binding. Such a substitution would capture the variable. The round-3 review found that a failure under a helper of type `a -> x` matched a helper of type `a -> a`, and the well-typed chain under the second helper was walked as a tree (`2^n`). The free nodes of a binding are not generalised while a use of the binding is in scope, so an image never contains a quantified variable. The capture check is kept as a guard, and a debug assertion lets the differential tests check this. A key-safe type is `Int`, `Bool`, `Unit`, a `Key`-flagged variable, a `Key`-flagged rigid variable, or a pair of key-safe types. An exact match is the common case and is checked first. Matching follows each pattern node once, through a map from pattern ids to current ids, so it costs time linear in the pattern's DAG.
3. **What goes.** The failure log, its trail positions, `slot_state`, and all rollback handling of failures. A recorded fact stays until the call ends. The skip of no-op path-compression writes stays: it is cheap and keeps the trail short.
4. **Charging.** Writing out a pattern and matching it are charged per node and per table probe. Comparing a pattern with those already recorded is charged per pattern and per image. All of this is charged under the same cfg(test) read and probe judges as the rest of the checker. A use computes its pattern only if `t` already has a recorded failure, so uses of subterms that never failed cost what they cost before.

## Why the facts are sound

Inferring a subterm afresh reads the store only through its slots' types and nodes that it makes itself. Monomorphic slots are replaced by fresh variables. The memo's reuse agrees with the tree walk (`TypeChecker::walk`). The levels of the slots' free variables do not affect the result. They are all at or below the use's level, so no `let` inside the subterm generalises them, whatever their exact values.

So fresh inference fails in a state exactly when no substitution for the slots' variables types the subterm. Algorithm J is complete (Damas and Milner 1982), so this holds whatever order J unifies in. The substitution must keep rigid variables and send `Key` variables to key types. This is a property of the pattern alone, up to renaming of its open variables.

Suppose the current pattern `C` is `σ(P)` for a recorded pattern `P` that fails. If some `S` typed the subterm under `C`, then `S ∘ σ` would type it under `P`. Rigid variables are kept by both. A `Key` variable of `P` goes under `σ` to a key-safe type, and `S` keeps that type a key type. That contradicts the failure of `P`. So the use fails in the current state too.

A cached failure only decides that a use is walked in place, and a use walked in place is walked as the tree walk walks it (`TypeChecker::run`). So the cache changes cost only, never the error reported. This was already true of the rules this decision replaces, and their reviews confirmed it.

## Why the instance match is needed, and why it is enough for the known family

Going back to a checkpoint restores its state exactly. Walking the use in place then makes, up to renaming, the same unifications as its failed fresh inference made, with each monomorphic parameter replaced by the type actually bound. So each type the in-place walk meets is an instance of the type the failed attempt met at the same point. Every inner use that failed in the attempt fails again in place, and its recorded pattern matches by instance.

This holds of types, not of store nodes. A helper bound inside the subterm over one of its monomorphic bindings is built over the fresh variable that stands for that binding in the fresh attempt, and over the binding's own node in place. Its two types have one shape, but their canonical ids name different free nodes. The round-4 review found that keying an open binding by its canonical id meant such a helper never matched in place. Every level was then tried fresh again under each level walked in place, which is `2^n` (failures noted `2^n - 1`; 103.7M units at `n = 14`). Keying it by shape and images makes the in-place pattern an instance of the recorded one again.

The first visit to a point is in its most general mode: every use around it is tried fresh before it is walked in place. So every later visit meets an instance of the state first met there. A failed fresh try is therefore never repeated at the same point.

An exact match is not enough. If a level's condition unifies a part of the helper's free variable with a monomorphic binding that is `Int`, that part becomes `Int` in place but stays a variable when tried fresh. The states at the leaf are then the subsets of the levels walked in place, so there are `2^n` of them. Each is an instance of the first, which is the most general.

The chains reported so far all fit this argument: s2, s7, s8, G3, G6, g9, the round-2 deepening chain and its s8 analogue, the monomorphic-parameter variant above, and the round-4 chain whose levels each bind a helper over a monomorphic variable (`failures_under_helpers_made_inside_the_subterm_are_located_in_polynomial_steps`). Each gets a step-bound test. Three more step-bound tests check that a failure under one helper does not match a different helper:
- In `a_failure_under_one_helper_does_not_match_another`, the helpers are `a -> a` and `a -> x`.
- In `a_failure_under_one_open_helper_does_not_match_another_of_another_shape`, they are `a -> (x, a)` and `a -> (a, x)`, which differ only in shape. That test fails if the same-shape requirement is removed.
- In `a_failure_under_one_helper_does_not_match_another_of_the_same_shape`, they are `a -> (x, a)` and `a -> (y, a)`, with `x` fixed to `Int` and `y` to `Bool`. They differ only in their images. That test fails if the images are left out of the pattern.

## Options considered

- **(a) A key with the canonical current types of the slots' free variables.** This is part 1 of the decision. Exact match alone fails on the monomorphic-parameter variant, so the decision adds the instance match.
- **(b) Re-validate under the budget when the slots have changed** (infer the subterm again, perhaps with its free variables made fresh). Rejected. Re-inference is the work the cache exists to avoid. It recurses into the locate machinery, and a re-check that fails is another failed try. The instance match gives the same guarantee for every state that is an instance of a failed one, at the cost of a linear match. The round-2 suggestion of re-inferring with fresh free variables gives a single most-general fact. The instance match already gets that wherever the first failure happens in the most general state, which the previous section shows is the usual case.
- **(c) Is a general polynomial bound achievable?** Not in the size of the term. Typability in ML with `let` is DEXPTIME-complete (Mairson 1990; Kfoury, Tiuryn, and Urzyczyn 1990), and a DAG term is at least as compact as the `let` form of the same term. So even the first walk has no polynomial bound, which is why the work budget (`DEFAULT_WORK_BUDGET`, D-budget PR 20) exists. The meaningful target is that the locate walk does not add an exponential factor to the first walk. The argument above shows that no failed fresh try repeats at the same point, which removes the doubling that all the reported shapes had. It does not bound the number of points the walk visits, or re-walks of successful open-scope uses whose memo entries were dropped by going back. The stated bound in general remains the work budget. The module doc is changed to say so, not to claim the locate walk is polynomial.

## Costs

- A use of a subterm that has failed before writes out what its open slots' free variables resolve to now, which is linear in that DAG, and scans the recorded patterns for that subterm. The first version wrote out each open binding's whole type at every such use. The round-3 review measured 17.6M units for a helper of 300 nodes under a chain of 100. Writing out only the images brought a similar test shape (`locating_under_a_large_helper_does_not_rewrite_its_type_at_every_use`) from 4.58M to 0.53M units. Each recorded pattern was a failed fresh try, which cost more than the scan.
- The `canon` table holds the pattern nodes until the call ends, and is released with the other tables.
