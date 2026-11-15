---
id: doc-gqz
title: 'D17: The Stitch bridge sets the cut-through flags and drops steps that still
  do not decode'
type: decision
scope: internal
created_at: '2026-11-14T00:26:18.056645Z'
updated_at: '2026-11-14T16:36:56.685389Z'
---

- **Date:** 2026-11-14. **Status:** accepted. Bead `dreamcounter-f22be4`. Branch `f22be4-stitch-bridge`.
- **Context:** Stitch can put an abstraction or a parameter through a special form or through the `lam_0` binder that encodes a let. The result has no core term. P2.13 left two options open: drop such abstractions in the bridge, or constrain Stitch with its flags. Q6 (`dreamcounter-f018fb`) chose `stitch_core` as a git dependency at rev `350804b7`. It named three flags that appeared to rule the shapes out, but did not run Stitch.

## Decision

1. A new crate, `dc-compress`, holds the bridge. It keeps Stitch's 73 dependencies out of `dc-stitch`, which `dc-surface` and `dc-eval` use.
2. `stitch_config` sets `--fused-lambda-tags 0`, `--no-curried-bodies` and `--no-curried-metavars`. It also sets one thread, no shuffling, no truncation, silent output and `eta_long` off.
3. `accept_steps` takes the library and checks each Stitch step. It decodes the abstraction's body, then every rewritten program. It then expands each rewritten program under the library and the abstractions kept so far. The result must be the term that the input program expands to under the library. At the first step that fails a check, it keeps the steps before it and drops that step and every later one. Later steps can use the dropped abstraction, so they cannot be kept. `Compression::rejected` records the step, its name, its body and the reason. `Compression::stitch_steps` counts all steps Stitch returned, so a caller can log how often steps are dropped.
4. Library `l` is named `fn_{l}`. The bridge passes the library's length to Stitch as `previous_abstractions`, so Stitch names its `i`-th new abstraction `fn_{len + i}`, which is the id `LibId(len + i)`. The bridge checks each name against this rule and drops a step whose name differs.
5. `decode_abstraction` refuses an arity whose parameters do not all occur in the body (`DecodeError::UnusedParameter`). Each parameter of a Stitch abstraction is a metavariable of its pattern and occurs in the body, so the rule rejects nothing Stitch produces. It bounds the arity by the size of the body, so a huge arity no longer builds that many lambdas. The bridge also drops a step whose arity is above `CompressConfig::max_arity`.
6. `compress` refuses a program whose Stitch expression has more than `MAX_STITCH_DEPTH` levels (`CompressError::TooDeep`). The limit is 4,096. A leaf is one level, and a lambda or an application is one more than its deepest child. Stitch recurses once for each level, and a stack overflow aborts the process. `compress` runs Stitch, and builds Stitch's configuration, on a separate thread with a 256 MiB stack. That is 64 KiB for each level at the limit.
7. `accept_steps` checks the steps in a scratch arena, into which it copies the library and the input programs. The caller's arena gains only the abstractions that are kept and the rewritten programs of the last step that is kept. Each run builds one name table. A step's name is added to the table, and removed if the step is dropped (`LibNames::remove`).
8. `stitch_config` sets `--no-mismatch-check`. Stitch otherwise asserts that the rewritten programs cost what it estimated, and panics when the estimate is too low. The bridge checks the rewritten programs by expansion (point 3), so it does not need the assertion.
9. The library and each input program must expand. `compress` checks this before Stitch runs, and `accept_steps` checks it when there are no steps. A failure gives `CompressError::Library` or `CompressError::Expand`.
10. `compress_isolated` runs Stitch in a child process, the worker, with limits (bead `dreamcounter-061990`, 2026-11-14). The module `dc_compress::worker` and the binary `dc-stitch-worker` hold it. `compress` still runs Stitch in the caller's process, with no limits. No caller uses `compress_isolated` yet. The iteration driver in P4.4 (`dreamcounter-86c25d`) is to use it.
    - The caller writes the encoded programs as JSON on the worker's standard input, and reads Stitch's steps as JSON from its standard output. It checks the steps with `accept_steps`, as for a run in its own process.
    - The caller kills the worker after `Limits::wall` (default 10 minutes), or when its cancel flag is set. It keeps at most `Limits::output_bytes` of the output (default 256 MiB).
    - The worker sets `RLIMIT_AS` to `Limits::memory_bytes` (default 4 GiB) and `RLIMIT_CPU` to the wall limit plus 5 seconds. It lowers each to a hard limit it inherits, and says so on its standard error. It keeps its soft CPU limit below the hard limit, so a CPU limit ends the run with `SIGXCPU`. Its hard CPU limit is at most `u64::MAX` nanoseconds in whole seconds (about 584 years), because a larger one wraps to a small one in the kernel. It sets its core file size to 0 and, on Linux, marks itself not dumpable, so a crash writes no core file and sends none to a crash handler.
    - On Linux the worker asks for `SIGKILL` when its parent dies, and exits if its parent has already gone. The CPU limit stops a worker whose caller has gone on any Unix.
    - `WorkerError` says why a run gave no steps: a timeout, a cancellation, a failed allocation (`OutOfMemory`), a Stitch thread that cannot start (`NoThread`), the CPU limit (`CpuLimit`), another failed exit such as a panic (`Failed`), too much output, or output that does not parse. Below about 260 MiB of address space, the thread's 256 MiB stack does not fit.
    - The limits need Unix. dc runs on Linux.

## Evidence from real Stitch runs

These runs used the pinned rev, in the tests in `crates/dc-compress/tests/bridge.rs`.

- With Stitch's default flags, a corpus of lets with varying bound values and bodies gave the abstraction `(mul int_3 (add int_1 (let_ #1 #0)))`. Its parameter `#0` sits on the let's binder. A corpus of paired ifs gave the arity-0 body `(if_ (lt int_1 int_2))`, a partly applied special form. Neither decodes.
- With the bridge's flags, the same corpora gave `(mul int_3 (add int_1 #0))` and `(if_ (lt int_1 int_2) #1 #0)`. Every rewritten program expands back to its original.
- A corpus that applies `and`, `or`, `eq` and `pair` to the same operands needs `--no-curried-metavars`. Without it, Stitch put a parameter in head position, and a rewritten program passed the bare token `and_`, which does not decode.
- An adversarial review ran 1,313 Stitch steps on random corpora with the flags on. The corpora held lets, nested lets, ifs, `and` and `or`, lambdas, folds and library references, over up to three rounds. No step was rejected, every rewritten program expanded to its original, and Stitch did not panic. With each flag off in turn, the same harness found rejections. Two separate processes gave the same output.
- Before the flags were added to `stitch_config`, the bridge's round-trip tests failed with exactly these two rejections.
- Three runs on the same programs gave the same abstractions and rewritten programs.
- A second round, given the first round's abstraction as the library, named its new abstraction `fn_1`. The first round's `fn_0` passed through as an opaque token.

### Evidence for points 3, 6, and 7

A code review of PR 53 led to the expansion check, the depth limit, and the scratch arena. These measurements used test builds (`opt-level = 1`) and a test thread with a 2 MiB stack. The adversarial review above ran before these changes and does not cover them. Two later reviews do: see the next section.

- Before the expansion check, `accept_steps` kept a step whose second rewritten program expanded to a different term from its input program.
- Before the depth limit, a program of 20,000 nested applications of `fst` aborted the test process with a stack overflow.
- Programs of 5,000, 10,000, and 14,000 nested applications did not overflow in 45 seconds. Stitch was still running when each run was stopped.
- A program of 2,000 nested applications did not finish in 120 seconds. Two list literals of 200 elements took 7 seconds.
- With Stitch's configuration built on the caller's thread, a caller with a 64 KiB stack overflowed. With the configuration built on Stitch's thread, the same caller got a result for three programs of 300 nested applications.

### Evidence from two adversarial reviews of the changes

Each reviewer had the bead, one commit, and the build steps, and worked in a scratch worktree with test builds.

The first review covered points 3, 6, and 7.

- It gave `accept_steps` 18 hostile sets of steps on a 64 KiB stack. None panicked. In each case the names matched the abstractions kept, the arena did not grow when nothing was kept, and the programs returned expanded to the inputs.
- It ran 5,000 random corpora for two rounds each. Of 9,189 runs with steps, none was rejected and each result expanded to its input. Two runs hit Stitch's cost assertion, which led to point 8.
- With no steps, an invalid library or an input program that does not expand returned `Ok`. This led to point 9.
- A chain of 1,000 nested applications overflowed a Stitch stack of 64 KiB and of 128 KiB, and ran on 256 KiB.
- A chain of 4,095 nested applications ran in 131.9 seconds on the 256 MiB stack, with a peak of 1.79 GB and no overflow.
- With the limit raised and the 256 MiB stack, the chain of 20,000 ran for 90 seconds with no abort and was stopped.
- Two programs of 4,095 nested lambdas, and two list literals of 2,040 elements, did not finish in 15 minutes. Each used about 1.8 GB.

The second review covered points 8 and 9.

- At the pinned rev, Stitch reads `no_mismatch_check` in two places: the assertion in `rewriting.rs`, and an assertion that applies only with `eta_long` on. The search, the choice of abstraction, the rewriting, and the naming do not use it.
- It ran 10,000 random corpora, 30,000 calls to `compress`. None panicked, no step was rejected, and each result expanded to its input.
- With the assertion back on, 17 of 17,972 runs panicked. In each, the rewritten programs cost more than Stitch estimated, and the abstraction used a parameter more than once. With the assertion off, each of the 17 gave a step that was kept.
- In two of those cases Stitch estimated a gain of 14,748 and of 606. The real gains were 10,304 and 505.
- No test checks that `compress` checks the inputs before Stitch runs. The tests check that it refuses them.

### Evidence for point 10

These runs used the worker built from bead `dreamcounter-061990`. The tests are in `crates/dc-compress/tests/worker.rs`.

- Two copies of a list literal of 1,000 elements took 2.97 seconds and 217 MB (release build). The worker gave the same steps as Stitch in the caller's process, over two rounds.
- Two copies of 2,000 elements ran until the 5 second wall limit and gave `TimedOut`.
- Two copies of 10,000 elements under a 1 GiB memory limit gave `OutOfMemory` in about 17.5 seconds. The worker's peak resident memory was 772 MB.
- A caller killed with `SIGKILL` left no worker running.
- In the tests, a run over the wall limit, a cancelled run, a run over the memory limit, a run over the CPU limit, and a panic in Stitch each give their own `WorkerError`. The same run of 1,000 elements that stops under 384 MiB finishes under 1 GiB.
- A test reads the worker's limits from `/proc`. Under inherited hard limits, the worker set its core file size to 0, its CPU limit to 99 seconds soft and 100 hard, and its address space to 2 GiB, and was not dumpable.
- Before each fix, its test failed. Each check was run in a scratch worktree on a stub of the fix: 12 of 14 tests failed on a worker that ran Stitch in its own process, and then 4, 4, 3, 1, and 2 tests failed in the later rounds.

Four adversarial reviews covered the worker. Each reviewer had the bead, the diff, and the build steps.

- The first found that a small memory limit was reported as a plain failure, the output cap shared the memory limit, standard error kept only its start, a run that ended at the deadline could be reported as a timeout, there was no way to cancel, and inherited hard limits made the worker fail. Each was fixed.
- The second found no way past the time, memory, or output limit. It found that a CPU limit with soft equal to hard ended the run with `SIGKILL`, that a lowered limit was not reported, that a Stitch thread that could not start was reported as out of memory, and that out of memory was judged from the exit signal alone. Each was fixed, and the worker sets its core file size to 0.
- The third found that a CPU limit too large for the kernel overflowed, that a crash handler still received piped core files, and that one test passed without its check. Each was fixed.
- The fourth found that a CPU limit of exactly 584 years still set a hard limit one second past the kernel's range. A larger limit set none, so an inherited hard CPU limit stayed as both soft and hard, and the run ended with `SIGKILL`. It also found that a test's stand-in shell left a crash report. The hard CPU limit is now lowered to the kernel's range. It stopped two copies of a list literal of 10,000 elements under 1 GiB with `OutOfMemory` in 15.5 seconds.
- The fifth found that the first fix for the crash report did not work. apport writes a report for a program from a package that aborts, even with a core limit of 0. The test now runs its own binary as the stand-in that aborts, and apport ignores a program from no package. It also checked the CPU limits the worker sets at and above the kernel's range, under inherited limits of 1:2, 1:1, and 0:0. Each run that used CPU ended with `SIGXCPU`, except under 0:0, where the kernel killed the worker at once.

## Costs and limits

- With `--no-mismatch-check`, Stitch ranks abstractions by an estimate that can be too high. A step can compress less than Stitch reported, and Stitch can choose an abstraction that is not its best. The bridge does not check that a step compresses. In the second review, no step that was kept left the programs no cheaper.

- `--no-curried-metavars` also forbids a parameter in head position, such as `(pair (let_ int_5 (lam_0 (#1 $0 int_2))) ...)` with `#1` bound to a primitive. That shape decodes. With the flag, Stitch found no abstraction on such a corpus. Dropping the flag would rely on the decode guard alone. This is a `HONING:` note on the bead, to be measured in P5.
- `compress` refuses a program with free variables (`CompressError::OpenProgram`) before Stitch runs. Without that check, an open program came back as a dropped Stitch step.
- Stitch panics on input it cannot parse (about 45 `unwrap` sites). The bridge gives Stitch only text from `encode`. `compress` does not catch panics. `compress_isolated` reports them as `WorkerError::Failed` (point 10).
- Stitch's time and memory grow quickly with the size of the programs. In the review of the bridge, two copies of a list literal of 1,000 elements took 41.3 seconds and 230 MB (release build). In the bead `dreamcounter-061990`, the same shape with elements 0, 2, 4, and so on took 2.97 seconds and 217 MB (release build). The element pattern of the first run is not recorded. At 10,000 elements a run reached 5.4 GB in 4 minutes and was stopped. `MAX_ENCODE_NODES` is far above this. `compress_isolated` bounds a run (point 10). P4.4 (`dreamcounter-86c25d`) chooses the limits for real runs, and what to do with an iteration whose run is stopped.
- The worker's limits have costs.
    - `RLIMIT_AS` counts reserved address space, not only memory in use. Near the 256 MiB stack, runs slow down: two lists of 100 elements took 0.02 seconds under 400 MiB and 0.78 seconds under 280 MiB (release build).
    - A wrapper program must replace itself with the worker (`exec`). If it starts the worker as its own child, the parent check fails and every run fails.
    - Both sides refuse unknown JSON fields, so a worker built from other source fails if the format changed. A worker built from other source with the same format is not detected. dc-mcp can run its own binary as the worker to rule this out.
    - Each run starts a process, so a run costs the process start-up as well as Stitch's time.
- The depth limit stops the stack overflow, not the run time. A list literal of 1,000 elements is under the limit. A list literal of 10,000 elements is over it and is refused. At the limit, the stack size is measured for one shape only: the chain of 4,095 nested applications. The other two shapes tried at that depth did not finish. No test fixes the stack size, because a program deep enough to need it takes Stitch minutes. The overflow appeared between 14,000 and 20,000 levels on a 2 MiB stack.
- The scratch arena costs one copy of the library and the input programs for each call, and a second decode of what is kept. `accept_steps` builds a new `Expander` for each step, because `dc-core` cannot add a library to an existing one.
- `RejectReason::Library` and `RejectReason::Expand` have no test. A body or a program that decodes cannot reach them.
- The bridge does not check types. A refactored program can fail to type while its expansion types (P2.12 note on P4.4). The driver must check each rewritten program.
- `CompressConfig::default()` uses Stitch's own defaults: 3 iterations and arity at most 2. P4.4 chooses the values for real runs.

## Options considered

- **Drop undecodable steps only, without the flags.** Rejected: on the let corpus, Stitch's first and best abstraction cuts through the binder, so the run would learn nothing.
- **Patch Stitch.** Rejected: the flags already rule out the shapes, and Q6 decided against a fork until P5 needs one.
- **Skip only the undecodable step and keep later ones.** Rejected: later steps and rewritten programs can refer to the dropped abstraction.

- **Print Stitch's results without recursion, with no depth limit.** Rejected: Stitch itself recurses on each level, so the overflow would remain.
- **Add a rollback to the `dc-core` arena, with no scratch arena.** Rejected: it changes the arena's index of terms, which is a larger change than the copy.

For point 10:

- **Cap the size of each program before compression.** Rejected as the bound: it limits neither time nor memory. Stitch's cost depends on the shape of the programs as well as their size. A pre-filter can still be added in P4.4, to waste less time.
- **Drop long programs from the frontier.** Not decided here: it changes what the learner sees, which is a question for the objective in P4.4.
- **Run Stitch on a thread with a timeout.** Rejected: Stitch cannot be cancelled, a thread cannot be killed, and a panic, an abort, or memory growth would still reach the caller.
- **Use a cgroup for the memory limit.** Rejected: it needs privileges or systemd, and more set-up. `RLIMIT_AS` needs neither.
