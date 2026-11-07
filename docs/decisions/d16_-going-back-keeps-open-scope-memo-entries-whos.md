---
id: doc-s4m
title: 'D16: Going back keeps open-scope memo entries whose nodes were not changed
  before they were kept'
type: decision
scope: internal
created_at: '2026-11-04T23:04:37.930456Z'
updated_at: '2026-11-05T14:05:41.509548Z'
---

- **Date:** 2026-11-04, reworked 2026-11-05. **Status:** accepted; merged 2026-11-05 in PR 48 as `d96299c`. It refines D13 (doc-mnj). Bead `dreamcounter-772e12`.
- **Context:** Under D13, when the locate walk (`TypeChecker::run` in `crates/dc-core/src/types/infer.rs`) goes back to the oldest checkpoint, `back_to` removed every memo key logged since that checkpoint. A key is logged when its scope is open: a polymorphic binding in its scope has a type with free variables. Removing the key threw away subterms that had been inferred successfully, so the walk inferred them afresh again.

## Decision

1. When a key with an open scope goes into the memo while uses are being tried, `keep_past_checkpoints` logs it with its kept type and the length of the trail at that moment. Nothing is walked or recorded then.
2. `back_to` undoes the trail from the newest change to the oldest. As it does so, it maps each node to its first change since the checkpoint, as a trail index. A change from one link to another is not counted, because a compressed path ends at the same node. A change between two variables with the same level and `Key` flag is not counted, because the rank only orders unions. The map is built only if a key was logged since the checkpoint.
3. Then `keep_logged` finds, for each logged key, the earliest first change among the nodes its kept type depends on. These are the nodes reached from the kept type, and from the free variables of the `Canon` of each polymorphic binding in the scope. Links are followed without being compressed. The key is kept if that earliest change is at or past the trail length at which the key was kept. Otherwise it is removed.
4. A variable that is not quantified, or a rigid node, made since the checkpoint counts as changed before every key: nothing reachable after going back refers to such a node, so no key that depends on it is kept.
5. The walk memoises its answer for each node and each `Canon` id, so one walk serves every key examined at a rollback. Each node, `Canon`, and counted trail change costs a unit.

## Why this is sound

Inferring a shared subterm reads only the types of the bindings in its scope. It changes only those types and nodes it makes itself. Monomorphic bindings are replaced by fresh parameters. The expected type of the position is checked after the memo frame, not inside it. The `Canon` id in the key fixes the shape of each polymorphic binding's type. The rest of what inference reads is what the store holds at that shape's free variables.

Every change to a node made before the oldest checkpoint is on the trail. Once the trail is undone, such a node reads as it did at the checkpoint. It read the same when the key was kept if, and only if, its first change on the trail is at or past the trail length then. So, if every node the kept type depends on passes this test, the inference changed none of them, and read them as they are now. Inferring the subterm afresh in the restored store would give the same kept type, up to renaming, and would change nothing outside it.

A node made since the checkpoint is not on the trail unless an inner checkpoint was made after it. Such a node that was a link or a constructor when the key was kept still is: nothing undoes a binding except the trail, and the walk follows the link to the nodes it leads to, which are checked in turn. Such a node that was a variable when the key was kept and is bound now was bound after the key was kept, inside the same use being tried. If it was a free variable of the scope, its binding changes the `Canon` of that binding, so the key can never match again, and keeping it is harmless. If the kept type made it, it stayed unquantified only because it was unified into the type of a variable at a lower level, which is on the trail or is such a free variable; so the key is removed through that variable, or can never match.

The walk reads each node as it is after going back. A node that a kept type reached through a link that is now undone is reached as a variable, and its change is before the key was kept, so the key is removed. A node reached now but not when the key was kept only adds a check. So the walk is exact for nodes made before the checkpoint, and conservative otherwise.

Once a kept key survives `back_to`, no checkpoint is open, and it has the same standing as a key made with no checkpoint open.

## Judges

- `going_back_drops_memo_entries_whose_scope_it_changes` (tests.rs): a shared subterm binds the scope's free variable `x` to `Int` inside a try that then fails. If its key were kept, the walk in place would accept `not x`. It fails, by the `both walks accept the same terms` assertion, when `keep_logged` never removes a key. So does `shared_subterms_under_open_helpers_agree_with_walking_the_tree`.
- The growth fuzzer (`locating_errors_in_random_chains_goes_back_once_a_level`, 20,000 seeds, release) passes with the change. With `keep_logged` never removing a key, 25 of the first 1,100 seeds report an error that differs from the tree walk's, the first at seed 15.
- `going_back_keeps_memo_entries_it_does_not_change` (tests.rs): on `partition_chain(a, 80, true)`, a test-only counter (`repeat_tries`) counts keys the second walk infers afresh after it has already done so. The test requires fewer repeats than rollbacks. Without the change it fails with 3,474 repeats and 80 rollbacks.
- `going_back_drops_memo_entries_whose_scope_lost_a_key_constraint` (tests.rs): a shared subterm uses `x` as a `memorec` key inside a try that then fails. It fails if `differs` does not compare the `Key` flag.
- `keeping_memo_entries_costs_nothing_until_a_use_fails` (tests.rs): many shared subterms `s_i = (add i g)` have a helper over a large type in scope, but take their type from the memo entry of `g`, and no use fails before the error at the tail. The second walk must cost at most 1.1 times the first. Recording each key's nodes as it was kept, as the first version of this decision did, cost 1.76 times the first walk at `m` = 100 and 2.56 times at `m` = 4,000 with a cap on the recording, and 13.6 times and out of budget without one.
- The existing judges: the errors equal the tree walk's, there are at most as many rollbacks as shared subterms, and `locate_cost` holds.

## Measured

Release measurements of infer mode on `partition_chain(a, 80, true)`, against `605d966` (`main`) and `9ff6600` (the first version, which recorded each key's nodes as it was kept, capped).

| | `605d966` | `9ff6600` | `e22c63d` |
|---|---|---|---|
| Fresh tries, second walk | 10,033 | 6,638 | 6,638 |
| Keys inferred afresh again | 3,474 | 79 | 79 |
| Units, first walk | 190,552 | 190,552 | 190,552 |
| Units, second walk | 12,224,834 | 11,788,720 | 9,478,048 |
| Of which, recording and checking keys | 0 | 2,386,105 | 75,433 |
| Keys logged, examined at a rollback, removed | | 3,398, 3,317, 0 | 3,398, 3,317, 0 |
| Rollbacks | 80 | 80 | 80 |

The second walk costs 22% less than on `main`. On `9ff6600` the recording and checking took 20% of the second walk and used up 85% of what keeping the keys saved.

On `subterms_over_a_large_helper` with a pair-tree library type 11 deep, units of the second walk (the first walk is the same in all three):

| `m` | First walk | `605d966` | `9ff6600` | `e22c63d` |
|---|---|---|---|---|
| 100 | 33,179 | 33,583 | 58,303 | 33,583 |
| 4,000 | 438,841 | 454,874 | 1,123,094 | 454,874 |

No key is examined at a rollback on this shape, so the check costs nothing, and the second walk costs what it did on `main`.

The growth fuzzer's worst cost ratio over 20,000 seeds is 1.537, as on `main`; on `9ff6600` it was 1.850.

The bead's figure of 88,810 fresh tries did not reproduce on `605d966`; the measured figure was 10,033.

## Costs and limits

- Most of the remaining fresh tries, 6,559 of 6,638, are keys the second walk had not met before. Each level walked in place fixes a part of `x`, so the helpers built over `x` get new `Canon` ids, and so new keys. A coarser key could reuse those, but that is a separate design question.
- The 79 remaining repeats are uses that were being tried when an error went back, about one a rollback. Their fresh inference had not finished, so there was nothing to keep.
- The check is charged to the work budget, so a rollback that examines keys costs at most the size of the store those keys reach, plus the trail since the checkpoint. On partition at depth 80 it is 0.8% of the second walk.
- A key whose type reaches an unquantified variable made since the checkpoint is always removed. On partition at depth 80, no key was removed for this reason, and no key was removed at all: every change the tried uses made was after the keys that depended on it were kept.
- The comparison of levels in `differs`, and the removal of keys that reach a new unquantified variable, fail no judge when taken out. Both are kept because they cost little and need no argument.

## Options considered

- **Recording each key's nodes with their contents as the key is kept, and comparing them after going back** (the first version of this decision, commits `cc2eb4b` and `9ff6600`, reviewed twice). Not adopted: the recording ran for every key logged while a use was tried, whether or not a rollback ever examined it. On partition at depth 80 it cost 2.39M units to save 2.82M; on the large-helper shape every key was promoted without a rollback, and the recording was the whole of a 1.8 to 2.6 times cost increase. A cap on the recording bounded the increase but could not remove it.
- **The HONING rule on the bead** (old nodes off the trail, new nodes only constructors and quantified variables), checked as each key is kept. Not adopted: it misses nodes made since the checkpoint that a nested try changed and the trail changes back.
- **Building a set of trailed nodes at each rollback** and rejecting any dependency on it. Not adopted: it also rejects keys whose nodes changed only after they were kept, which going back undoes anyway. The adopted rule is this one with the time of each change kept, so that those keys are told apart.
- **Copying each kept type out of reach, as `freeze` does for closed scopes.** Not adopted: a copy still shares the scope's free variables, so it needs the same check, and it costs a copy at every key.
