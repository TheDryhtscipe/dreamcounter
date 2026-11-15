---
id: doc-2f4
title: 'D22: The operator ends an iteration with dc_finish_iteration; the dc-mcp binary
  is its own Stitch worker; the run stores K, the Stitch limits, and the prelude''s
  fingerprint'
type: decision
scope: internal
created_at: '2026-11-15T22:29:02.835185Z'
updated_at: '2026-11-15T23:15:03.436076Z'
---

- **Date:** 2026-11-15. **Status:** accepted by Claude as technical design (autonomous lane). Open to change by the project lead. Bead dreamcounter-967d1d (P4.4b). Follows D11 (doc-d9s), D18 (doc-g97), D19 (doc-6nh), and D20 (doc-hw6).
- **Decision:**
  1. **An operator tool ends the iteration.** `dc_finish_iteration` on the operator's server runs the step of D19 and commits its result. Proposers' servers do not run it. The tool refuses unless every task has its programs for the iteration and all of them are checked (`quota_in`).
  2. **Read, step, commit.** The server copies the library, the frontiers, and the verified programs under the run's lock. It runs `dc_driver::run_iteration` without the lock. It then commits one record: a `FrontierSet` for each task, a `LibraryExtended` if abstractions join the library, a log record of kind `compression` that holds the `IterationReport` as JSON, and an `IterationFinished` with the number of verified programs the step read.
  3. **Another server ends the iteration first.** The commit is written only to the journal the step read, and only if the run is still in the iteration the step read. Otherwise the call fails and writes nothing.
  4. **A submission comes in during the step.** A batch whose lease ended can still be submitted, and an iteration cannot end while a submission waits to be checked. The server checks what waits and tries the commit again, up to eight times. It does not run the step again, because the step's outcome depends only on what it read. Programs verified during the step stay for the next step.
  5. **The binary is its own worker.** `dc-mcp --stitch-worker` calls `dc_compress::worker::serve`. The server starts the running binary again with that argument, as its direct child. On Linux it starts the worker from `/proc/self/exe`. That is the file the server runs, even after another file has taken its path, so the worker is the same build as the server. On other systems it starts the file at the binary's path.
  6. **The run stores its Stitch settings.** `RunConfig` gains `frontier_size` (K, with no default), and `stitch`: the number of abstractions for each run of Stitch, the largest arity, the time limit, and the memory limit. The defaults are 3, 2, 600 seconds, and 4 GiB. The memory limit is at least 512 MiB.
  7. **The run stores the prelude's fingerprint.** `RunConfig.prelude` holds `Prelude::fingerprint` of the build that made the run. `dc_next_batch`, the checks of submissions, and `dc_finish_iteration` refuse a run whose fingerprint is not the build's.
  8. **Journal format 3.** The new settings have no defaults, so the format version is 3. This code does not read version 2.
  9. **Cancel and shutdown.** Each call has a flag that stops its Stitch worker. The server sets the flag when the client cancels the request (`notifications/cancelled`), when the call's future is dropped, and when the client of the server is gone. The call then fails as interrupted and writes nothing. The step reads the flag while Stitch runs, and under the run's lock at each attempt to commit.
- **Alternatives:**
  - (a) End the iteration from the proposer's server when the quota is in. Rejected: a run has many proposer servers, and each would start a Stitch run of up to 4 GiB. A proposer's server would also hold a path that changes the library. The project lead starts every run (D11, requirement 5), so an operator's call fits.
  - (b) Keep the Stitch limits in the server's settings. Rejected: a Stitch run stopped at a limit ends the iteration with no new abstractions, so the limits change the run's result. They belong with the run.
  - (c) Read version 2 journals with defaults for the new settings. Rejected: K has no default (D19), and a fingerprint cannot be made up for a run after it starts. A version 2 run could not end an iteration.
  - (d) Use the separate `dc-stitch-worker` binary. Rejected: the server would have to find it, and a worker from another build is not detected when the format is the same.
  - (e) Take a lock file for the step, so that two servers do not both run Stitch. Left out: the second commit already fails cleanly, and one operator runs the step.
  - (f) Let the operator end an iteration that lacks programs. Left out: it changes what the compute-matched baseline counts (P4.7), so it is the project lead's call.
- **Why:** one record for the step means a restart finds the whole step or none of it. The step runs without the lock because it can take as long as the Stitch time limit, and proposers' servers must go on meanwhile.
- **Limits:**
  - Two servers can run the step at the same time. The second to commit has run Stitch for nothing.
  - A cancel that comes after the committing attempt has read the flag does not stop the commit. A cancelled call gets no reply, so the client must read the run's status.
  - The step does not read the flag while it measures programs before Stitch. A cancel in that phase takes effect when Stitch starts.
  - When the client's connection ends, rmcp waits about 5 seconds for running calls before the server stops them. A step that ends in that time commits.
  - On Linux the worker's command line is `/proc/self/exe --stitch-worker`, so a search for processes named `dc-mcp` does not find it.
  - No test covers two cases: the commit attempt after the eighth check, and the second prelude comparison in `check_pending` when a run is replaced during the call.
  - No test starts the `dc-mcp` binary, closes its connection during a Stitch run, and looks for the worker. The tests cover the flag, the cancel notification, and the dropped call inside the test process.
  - The server checks what waits up to eight times, and tries the commit after each. If a check is still pending then, the call fails and the step must run again.
  - If two servers run the step, their steps may have read different programs: one may have been checked between the two reads. The first commit stands.
  - The fingerprint covers the prelude's definitions. It does not cover what the primitives do.
  - `dc_create_run` now needs `frontier_size`.
