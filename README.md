# dreamcounter

dreamcounter is a research project in DreamCoder-style library learning. It aims at algorithmic primitives, such as divide and conquer, sliding windows, memoised recursion, and monotone stacks, rather than domain helpers. Given it is aimed at research within mathematics explicitly using LLMs, it is also coded in partnership with Claude.

The research bet is this: an objective that rewards an abstraction for reducing the computational cost growth of solutions, not only their description length, will separate algorithmic ideas from shorthand.

## Status

- Phase 1 (literature and language specification) is done.
- Phase 2 (the core language) is built: parser, type inference, evaluator with exact step counts, library expansion, prelude, property tests, and a throughput baseline. Robustness fixes are still open.
- Phase 3 (task corpus) is in progress. Three task families are built: collapse games, sorted-order queries, and constrained subsequence counting. The held-out family and the corpus-wide test run are still to do.
- Phases 4 to 6 have not started.

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
| `dc-tasks` | Task format, seeded input generators, oracles, growth estimators, and the task families |
| `dc-bench` | Evaluator throughput benchmark |

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
