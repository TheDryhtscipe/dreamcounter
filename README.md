# dreamcounter

dreamcounter is a research project in DreamCoder-style library learning. It aims at algorithmic primitives, such as divide and conquer, sliding windows, memoised recursion, and monotone stacks, rather than domain helpers. Given it is aimed at research within mathematics explicitly using LLMs, it is also coded in partnership with Claude.

The research bet is this: an objective that rewards an abstraction for reducing the computational cost growth of solutions, not only their description length, will separate algorithmic ideas from shorthand.

## Status

- Phase 1 (literature and language specification) is done.
- Phase 2 (the core language) is done: parser, type inference, evaluator with exact step counts, library expansion, prelude, property tests, and a throughput baseline.
- Phase 3 (task corpus) is done. Four development families are built: collapse games, sorted-order queries, constrained subsequence counting, and dominance counting. A corpus run checked every reference program against every oracle. Four candidate held-out families are built outside this repository, and the project lead picks one without saying which (decision D14).
- Phase 4 (the baseline loop) runs end to end. `dc-propose` starts locked-down proposer sessions. `dc-mcp` hands them batches, checks the programs they submit, and ends each iteration with Stitch compression. Proposer sessions then name the new library functions. When an iteration ends, `dc-mcp` measures the cost of each frontier program. A run can turn library learning off, to give the LLM-only baseline.
- The first runs were on 18 November 2026. A pilot run of two iterations solved all 14 tasks of the first three families in both iterations. A probe of the dominance family solved every task. The results log has the details. Harder development tasks are in progress (decision D28).
- Phase 5 (the cost-aware objective) has started. `dc-driver` computes `J(L)` from description length and measured cost (decisions D4 and D27). The loop still selects frontier programs by description length alone.
- Phase 6 has not started.

## Phases

1. Literature and language specification.
2. Core language: the `dc` language, a typed lambda calculus with an exact step-count cost model, written in Rust.
3. Task corpus with input generators, oracles, and growth estimation.
4. Baseline loop modelled on LILO: LLM proposals and Stitch compression. The loop runs as an MCP server that Claude sessions pull proposal jobs from (decision D11).
5. Cost-aware library objective, compared with pure description length.
6. Structural abstraction with e-graphs.

## Crates

The code is a Cargo workspace under `crates/`:

| Crate | Contents |
|---|---|
| `dc-core` | Hash-consed term arena, primitive table, types, Hindley–Milner type inference, and library-abstraction expansion |
| `dc-surface` | Surface syntax (lexer, parser, desugarer, and pretty-printer) and the prelude, written in `dc` |
| `dc-eval` | CEK evaluator with exact step counting and fuel |
| `dc-stitch` | Serialisation of `dc` terms to and from the Stitch format |
| `dc-compress` | Library learning: runs Stitch on `dc` programs and decodes the abstractions it finds |
| `dc-driver` | One iteration of the loop (frontier update, Stitch compression, and library update), and the cost-aware objective `J(L)` |
| `dc-mcp` | MCP server that keeps the state of each run on disk, hands out proposal batches and naming jobs, checks submitted programs, ends iterations, and measures frontier costs |
| `dc-propose` | Starts locked-down Claude Code sessions that take batches or naming jobs from a `dc-mcp` run |
| `dc-tasks` | Task format, seeded input generators, oracles, growth estimators, and the task families |
| `dc-bench` | Evaluator throughput benchmark |

## Running the loop

The project lead starts every proposer launch (decision D11). To run the loop:

1. Build the binaries with `cargo build --release -p dc-mcp -p dc-propose`.
2. Connect an MCP client to `dc-mcp --state-dir <dir>`. This is the operator's server.
3. Create a run with the operator tool `dc_create_run`.
4. Start proposer sessions:

   ```bash
   dc-propose --state-dir <dir> --run <run> --model <model id> --log-dir <dir> --sessions <n> --parallel <n>
   ```

   `dc-propose` uses the `dc-mcp` beside its own executable. To use another, give `--dc-mcp` an absolute path. Each session runs in its own temporary directory, so a relative path does not start the server.
5. When the batches are in, end the iteration with `dc_finish_iteration`.
6. Run `dc-propose` again for the next iteration. If the run has library functions waiting for names, its first sessions name them.

`dc_run_status`, `dc_frontiers`, and `dc_proposal_stats` report on a run. `dc-propose` writes `summary.json` to its log directory.

## Building and testing

The toolchain is pinned in `rust-toolchain.toml`.

To run the format, lint, and test checks that CI runs, run:

```bash
scripts/gate.sh
```

CI also builds the API documentation with `cargo doc --workspace --no-deps`.

To run the throughput benchmark, run:

```bash
cargo run --release -p dc-bench
```

The benchmark takes the options `--threads`, `--reps`, `--batch`, and `--warmup`.

## Documentation

The `docs/` directory holds the project documentation:

- `docs/context/`: the original project brief.
- `docs/reference/`: the `dc` language specification and the literature notes.
- `docs/decisions/`: design decisions, with the alternatives considered.
- `docs/findings/`: the results log, a dated record of every experiment.

## Licence

dreamcounter is licensed under either of these, at your option:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT licence ([LICENSE-MIT](LICENSE-MIT))
