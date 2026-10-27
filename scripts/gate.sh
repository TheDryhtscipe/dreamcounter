#!/usr/bin/env bash
# Local gate: the same checks CI runs.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUSTFLAGS="-D warnings"
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked
cargo test --workspace --locked
