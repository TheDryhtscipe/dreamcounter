# dreamcounter

dreamcounter is a research project in DreamCoder-style library learning. It aims at algorithmic primitives, such as divide and conquer, sliding windows, memoised recursion, and monotone stacks, rather than domain helpers.

The research bet is this. An objective that rewards an abstraction for reducing the computational cost growth of solutions, not only their description length, will separate algorithmic ideas from shorthand.

## Status

Phase 1 (literature and language specification) is done apart from citation checks. Phase 2 (the core language) has started.

## Phases

1. Literature and language specification.
2. Core language: the `dc` language, a typed lambda calculus with an exact step-count cost model, written in Rust.
3. Task corpus with input generators, oracles, and growth estimation.
4. Baseline loop modelled on LILO: LLM proposals and Stitch compression.
5. Cost-aware library objective, compared with pure description length.
6. Structural abstraction with e-graphs.

## Building

The code is a Cargo workspace under `crates/`. To run the same checks as CI (format, lint and tests), run `scripts/gate.sh`.

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
