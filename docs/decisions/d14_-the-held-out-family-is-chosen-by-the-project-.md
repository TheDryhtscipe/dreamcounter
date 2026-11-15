---
id: doc-aq8
title: 'D14: The held-out family is chosen by the project lead from four candidates,
  and loaded from a private crate'
type: decision
scope: internal
created_at: '2026-10-29T18:39:53.431269Z'
updated_at: '2026-11-12T12:07:55.054911Z'
---

Status: accepted 2026-10-29 (answers Q7, dreamcounter-ee11eb). Work: P3.9 (dreamcounter-4eda41).

## Decision

Claude proposes four candidate held-out families (below). All four are built and validated in a private crate outside the repository. The project lead then picks one and does not say which. The choice is written to a file that no agent reads. Only the final evaluation command reads it.

The project lead chose this protocol on 2026-10-29, from the suggestion in Q7.

## Why

A held-out family measures transfer: whether a library learned on the development families helps on a family that nobody tuned the system for. If an agent knows which family is held out, its development choices (language, prelude, objective weights, proposer prompts) can drift towards that family, even without intent. Hiding the choice among four built candidates keeps every agent equally ignorant of it, including the agents that build the candidates.

## The candidates

Each candidate follows the corpus contract (spec §9, `TaskFamily` in `crates/dc-tasks/src/task.rs`): three variants that share one idea, a naive and a good dc reference per variant of different growth classes, a seeded generator, a Rust oracle, and examples. The variant rules are invented for this corpus where possible, as in the development families.

The candidates differ in how they relate to the development corpus. Three reuse a technique that a development family teaches, in a new setting. One uses a technique that no development family teaches. So the choice also decides what kind of transfer is measured.

### H1: Shadows (a stack over a list)

Input: a list of `n` heights, `0..=99`.

| variant | answer |
|---|---|
| next taller | the sum over `i` of the distance to the next strictly taller element to the right, 0 if there is none |
| mutual sight | the number of pairs `i < j` where every element strictly between is lower than both |
| window minimums | the sum of the minimum of every contiguous window, mod 1 000 003 |

Naive: `n²`. Good: `n`, with a stack of unresolved elements. Shares its technique with the collapse family (a stack fold).

### H2: Overlaps (sort, then sweep)

Input: a list of `n` intervals `(start, length)`, with `start` in `0..10⁶` and `length` in `1..=10⁴`.

| variant | answer |
|---|---|
| deepest point | the largest number of intervals that share a point |
| fewest stabs | the fewest points such that every interval contains one |
| touching pairs | the number of pairs of intervals that share a point |

Naive: `n²` (or `n³` for fewest stabs by trial). Good: `n log n`, by sorting endpoints and one sweep. The program must build its own sort, as there is no sort primitive. Shares its technique with the sorted-order family.

### H3: Windows (two pointers)

Input: a list of `n` values in `0..=9` and a bound `k`.

| variant | answer |
|---|---|
| light windows | the number of contiguous windows whose sum is at most `k` |
| few kinds | the length of the longest window with at most `k` distinct values |
| full palette | the length of the shortest window that contains every value present in the list |

Naive: `n²`. Good: `n`, with two indices that only move right and a 10-slot count array. No development family teaches this technique, so H3 measures transfer to an unseen technique.

### H4: Bit patterns (counting below a bound)

Input: a list of `n` bits, read as a binary number `N` (most significant bit first).

| variant | answer, mod 1 000 003 |
|---|---|
| no adjacent ones | the number of `x` in `0..=N` with no two adjacent 1 bits |
| ones mod 3 | the number of `x` in `0..=N` whose count of 1 bits is a multiple of 3 |
| no triple | the number of `x` in `0..=N` with no three equal bits in a row (as `n`-bit strings) |

Naive: `2ⁿ` (enumerate every `x`), so it declares a small sweep for the naive reference as subsequence counting does. Good: `n`, a left-to-right pass that carries a small state and a "still equal to `N`" flag. Shares the small-state counting technique with the subsequence-counting family, and adds the bound.

## Protocol

1. **Build.** All four candidates are built in a private crate, `dc-heldout`, at `~/.local/share/dreamcounter/heldout/`. It is a local git repository with no remote, by design. It depends on this repository's crates by path. This repository never refers to it, so the public build and CI do not need it.
2. **Validate.** Each candidate must pass `validate_family`, and estimator C must name the expected classes for its naive and good references, on every seed. The project lead picks only after all four pass.
3. **Choose.** The project lead writes the chosen candidate's name (`H1`, `H2`, `H3` or `H4`) to `~/.local/share/dreamcounter/heldout/CHOICE`. No agent reads that file, lists its contents, or runs a command that prints it.
4. **Evaluate.** `dc-heldout` provides the evaluation command. It reads `CHOICE`, runs the learned library on the chosen family, and reports aggregate results only. It runs once per completed run, after training, never during development.
5. **Keep the others out.** The three candidates that are not chosen are not added to the development corpus, so that the choice cannot be worked out by elimination. The project lead can later make them extra held-out families.

## As built

The four candidates were built on 2026-10-29 (beads dreamcounter-4c48e2, ca97cb, fb8728, b3a3c8). All four pass the step 2 validation. H3 was rebuilt with untimed checks (below) after a review found that a wrong program could pass its generated inputs. H3 is ready: its bead, dreamcounter-fb8728, closed on 2026-11-04 after a fifth independent review found no blocker. All four candidates are ready, and the project lead can pick. The candidates differ from the descriptions above in these ways:

- **Sweeps.** H1 uses the default sweep `16..1024`. H2, H3 and H4 stop at 512, because their naive or plan references need more than the default fuel cap of 50 million steps at 1024 under a plan of ten times the reference's steps.
- **H2 intervals.** An interval `(s, l)` covers the integer points `s..=s+l-1`, a half-open interval `[s, s+l)`. Naive fewest stabs is `n²` (`n` rounds of a greedy over the unsorted list), not `n³` by trial, which would not finish at 512.
- **H3 full palette** takes no bound `k`. Its signature is `List Int -> Int`, and an empty list gives 0.
- **H4 references.** Each variant has three references: the naive `2ⁿ` reference, checked on its own short sweep `6..16`; an `n²` reference, which makes the fuel plan; and the good `n` reference.
- **Inputs built for natural solutions.** In each candidate, every input mixes all of its kinds, so a natural correct solution costs about the same on every seed of a size and estimator C can name its class. Each candidate's module documentation lists what its sweep holds.
- **Choices that hold for a seed.** Since 2026-11-12 (dreamcounter-40ca4e), H1, H3 and H4 take each input's whole-input choices from the seed alone, as the development families do. Estimator C takes out each seed's level on a sweep in seed order, so such a choice adds no noise. The choices are:
  - H1: the order of the four blocks.
  - H3: the alphabet (its size by the seed mod 7), the late value, its copies (by the seed mod 3), and the kinds of the middle's blocks.
  - H4: whether `N` is left so, broken, or set to the modulus edge, by the seed mod 3.

  H2 still draws its choices per input. Per-seed choices there cut the sweep inputs that catch one wrong deepest-point program from 4 to 1. Its natural solutions are named `n log n` either way.
- **Untimed checks.** A variant can also have checks (`HeldoutVariant::checks`). A check is an input that a program must answer as the oracle does, within the default fuel cap, but that is not timed. The evaluation counts a variant solved only if the program passes its checks as well as the sweep. It treats a variant whose good reference fails a check as broken. Only H3 has checks. They hold the regimes that make natural solutions too noisy for C when the sweep holds them: bounds of 0, bounds that few or most elements exceed, long answer windows, and full palette windows away from the end of the list. Each H3 variant has 204 checks, of sizes up to 512, the sweep's largest. A larger check could time out, under the default fuel cap, a correct program that the sweep's fuel plan allows. Stopping at 512 makes such a timeout unlikely but does not rule it out, because a check's cost depends on its input. For example, an `n²` program that stops early costs about 8 times as much on a check as on the sweep.
- **H3 few kinds' bound.** The sweep's bound `k` is always 2, so a natural solution's cost per element is the same on every seed. The checks use bounds from 0 to 10.
- **Natural solutions and C.** C names the good class for most natural solutions written in the tests, but not all of them. The tests require it for 5 natural solutions in H1, 3 in H2 (`n log n`), 13 in H3, and 3 in H4. One more natural few-kinds solution in H3, which rescans the last positions each time the window breaks the bound, is measured but refused as too noisy. Within each seed its cost grows slower than `n`, because the middle's blocks are `n / 16` long and the window breaks the bound less often per element at larger sizes. Taking out each seed's level cannot fix that. Estimator C refusing natural correct solutions affects the evaluation's `good_class` count for every candidate (dreamcounter-0ea5b7).

## Limits

- The agents that build the candidates see all four. Everyone knows the held-out family is one of them. The protocol hides which one, not the set.
- Since 2026-11-12 (dreamcounter-54585a), a Claude Code hook enforces the rule in step 3 on this machine. The hook is `~/.claude/hooks/guard-heldout-choice.py`, and its tests are in `~/.claude/hooks/tests/guard-heldout-choice.test.sh`. It denies these calls:
  - file tool calls on `CHOICE`;
  - Grep calls that would search it;
  - shell commands that name it, match it with a glob, read the held-out crate's root recursively, list that root into another command, or delete or move it.

  The evaluation command does not name `CHOICE`, so the hook lets it run. Glob calls are allowed, because they return names only.
- The hook guards against accidental reads. It is not a sandbox. It does not stop a command that builds the path at run time, for example from a variable or a script file. It does not stop a recursive reader that follows a symlink into the held-out crate. The hook's docstring lists its other limits.

## Rejected alternatives

- **The project lead writes the held-out family alone.** That hides the family completely, but it puts all the implementation work on the project lead.
- **Build only the chosen family, in secret.** The agent that builds it would know the choice.
