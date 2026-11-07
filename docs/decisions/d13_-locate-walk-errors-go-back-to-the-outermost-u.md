---
id: doc-mnj
title: 'D13: Locate-walk errors go back to the outermost use being tried'
type: decision
scope: internal
created_at: '2026-10-28T22:53:39.678329Z'
updated_at: '2026-11-05T00:42:03.408771Z'
relates_to:
- doc-ue2
---

- **Date:** 2026-10-28. **Status:** accepted. It supersedes D12 (doc-ue2). Bead `dreamcounter-bb28d2`. Branch `fix-bb28d2-outermost-unwinding`.
- **Context:** The locate walk (`TypeChecker::run` in `crates/dc-core/src/types/infer.rs`) tries each use of a shared subterm from a checkpoint. Until this decision, an error went back to the innermost checkpoint. On a chain `T_k = (if c T_(k-1) T_(k-1))` over an ill-typed leaf, that costs `2^n`. A cache of failed keys hid the doubling for closed scopes. Five rounds of review then tried to extend the cache to open scopes (D12 and the four rules before it). Each round found a new exponential shape.

## Decision

1. On an error with checkpoints open, the walk takes the whole checkpoint stack. It notes as failed each popped closed key that is not in the memo, then goes back to the oldest checkpoint and walks that use in place (`TypeChecker::unwind`).
2. The closed-key failed set stays. The open-scope failure cache of PR 28 (failure patterns, `Canon::Open`, shape and instance matching, the capture guard) is not adopted. PR 28 is closed unmerged.
3. Test-only counters record rollbacks, first-walk units, shared subterms and failed keys. Judges in the tests assert:
   - errors equal the tree walk's;
   - rollbacks are at most the number of shared subterms (a `debug_assert` as well);
   - locate units are at most `4 × (rollbacks + 1) × first-walk units + 100`.

## Why the innermost rule doubled, and why no cache key could fix it

A use walked in place after an innermost rollback is still inside the enclosing uses being tried. If the error is inside it, the next rollback throws that in-place work away and rebuilds it under the enclosing use. So `T(k) = 2 T(k-1)`. A failure cache can only skip tries that would fail. It cannot stop the in-place work from being thrown away.

Every key the five rounds tried tied a failure to how the store represents a state: trail position, slot state, canonical id, shape. Fresh inference fails at every instance of a failing state. The in-place walk meets an instance of the state its failed try met, made by substituting the actual types of the monomorphic bindings for the try's fresh parameters. A substitution can refine a variable, merge two variables, or deepen one. For any key finer than the type-instance order, some choice of a monomorphic binding's type gives an instance that the key does not match. So any such rule loses to some shape.

## Why going back to the outermost use works

- **Progress.** After going back to the oldest checkpoint, no use is being tried. The use walked in place is never thrown away.
- **Chain.** A use whose try failed also fails in place, because the in-place state is an instance of the try's state. So the final error lies inside every use walked in place. Each later rollback goes at least one level deeper into nested shared uses. Rollbacks are bounded by that nesting depth and by the number of shared subterms.
- **Errors.** Which checkpoint the walk goes back to changes cost only. The final error is raised only when no use is being tried, and the work done up to it is the tree walk's unifications up to renaming. The trail reaches back to the oldest checkpoint, so going back to it is exact.

The locate walk costs about one memoised walk per rollback, plus one. That is a bound relative to the first walk. ML typability with `let` is DEXPTIME-complete (Mairson 1990; Kfoury, Tiuryn and Urzyczyn 1990), so no bound in term size is possible. The work budget (`DEFAULT_WORK_BUDGET`) remains the general bound. If the locate walk runs out, the first walk's error is returned. Whether a term is accepted never changes.

## Measured

These are release measurements of infer mode, from the branch's tests and probes.

| Shape | Units at n=40 | Units at n=80 | Rollbacks |
|---|---|---|---|
| s2 (ill-typed leaf under an open helper) | 13,560 | 44,380 | n |
| partition (round-5 shape) | 1,765,346 | 12,415,386 | n |
| p2 | 833,749 | 5,328,229 | n |
| fastfail | 7,728 | 14,808 | 2 |

PR 28 (head 9bf07fc) reaches a 200M-unit cap on partition at n=20. On fastfail it costs 223,872 units at n=80, quadratic in n, although its cache records every failure.

On the repo's proptest generators the worst cost ratio, locate units over `(rollbacks + 1) × first-walk units`, is 1.569 (20,000 cases each, infer and check). The growth fuzzer over 20,000 random chains found 0 differences from the tree walk, and its worst ratio is 1.424.

## Costs and limits

- Partition-like chains cost about `n` times the first walk. They reach the default budget near depth 90, and past that they report the first walk's error.
- A correct type-level failure cache on top of this rule could win back up to a factor of the nesting depth. The worst measured loss against PR 28 is 1.6× (mono-ill-outer). Leaving the cache out is a choice about cost, not a theorem. Under this rule a cache that misses costs one more try and never doubles.
- Open-scope memo entries are dropped when the walk goes back, so successful subterms are inferred again. This is a separate follow-up. D16 (doc-s4m, 2026-11-04) keeps those whose nodes were not changed before the entry was kept, checked against the trail when the walk goes back.
- Freeze-doubling on closed DAGs was a pre-existing exponential (`dreamcounter-715365`). This decision did not change it. It was fixed in PR 30, merged on 2026-10-29. A second change in the same PR makes `freeze` share nodes an earlier freeze made, since a kept type over two closed shared subterms holds two distinct frozen copies, and copying both doubled the store on interleaved families. In `Frame::Memo`, the first use of a closed shared subterm got an instance of the kept type before `freeze` copied it, and later uses got instances of the copy. A level over two uses then held two distinct copies, and freezing it doubled the store. The instance is now taken from the type as it goes into the memo. On `(pair P_n (add true 1))` at n=16 the locate walk fell from 1,378,309 units to 3,310, against 4,270 for the first walk.

## Options considered

- **A type-level instance match under innermost rollback** (two prototypes in the council of 28 Oct 2026). It is correct as a cache, and failures fall to `n`. Rejected: it keeps the discarded in-place work. It costs about 87M units on partition at n=80, and its failure mode is a doubling.
- **Landing PR 28 as a partial improvement.** Rejected by the project lead in session 10.
- **Reporting a coarser location for large DAGs.** Not needed: the budget fallback already does this.

## When to reopen

Reopen `dreamcounter-bb28d2` only for one of these:
- a located error that differs from the tree walk's;
- rollbacks above the nesting depth of shared uses;
- locate units above `4 × (rollbacks + 1) × first-walk units`, where no single try costs more than the first walk and the tree walk of the failing path costs no more than the locate walk spends on it.

The round-6 review found that the third rule cannot hold for every term. A use walked in place sees the actual types of its monomorphic bindings, as the tree walk does, where the first walk saw fresh variables. A chain that rebinds a variable whose type is a pair tree 13 deep costs 134 times the first walk at n=400, with 2 rollbacks. `origin/main` costs more on the same shape, so this is a limit of the rule, not a regression.

Route slow first walks to `dreamcounter-9bd0d8`. Single expensive tries were the subject of `dreamcounter-715365`, now fixed; file a new bead for a new one.