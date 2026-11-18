#!/usr/bin/env bash
# Runs one shard of `cargo test --workspace --locked`, so that CI can run the
# slow tests in parallel jobs.
#
#   scripts/test-shard.sh --list    prints the shard names as a JSON array
#   scripts/test-shard.sh <shard>   runs that shard's tests
#
# Each named shard runs the tests whose names contain one of its patterns.
# The shard `rest` skips every test that any named shard runs. So the shards
# together run every test, and a new test runs in `rest` unless its name
# matches a pattern. A test that matches patterns in two shards runs twice.
# CI reads the shard names from `--list`, so a shard added here runs there.
set -euo pipefail
cd "$(dirname "$0")/.."

# One line per shard: the name, then its patterns. Each pattern names one
# test that takes minutes of CPU (crates/dc-tasks/tests/chains.rs and
# corpus.rs, about 4 to 8 minutes each on a CI runner).
SHARDS="
gate-1 gate_count gate_longest
gate-2 gate_heaviest the_gate_turns_away_a_lean_quadratic_program the_gate_turns_away_a_program_that_is_quadratic_on_an_antichain
gate-3 programs_a_proposer_wrote_pass_the_gate the_gate_lets_through_a_less_careful_n_log2_n_program the_gate_turns_away_a_program_that_is_quadratic_only_on_sorted_xs
corpus the_corpus_holds_every_variant_once good_references_at_every_size_others_at_the_lower_half
"

names=()
all_patterns=()
while read -r name patterns; do
    [[ -z "$name" ]] && continue
    names+=("$name")
    read -r -a ps <<<"$patterns"
    all_patterns+=("${ps[@]}")
done <<<"$SHARDS"

shard="${1:-}"
if [[ "$shard" == "--list" ]]; then
    printf '['
    for n in "${names[@]}"; do printf '"%s",' "$n"; done
    printf '"rest"]\n'
    exit 0
fi

if [[ "$shard" == "rest" ]]; then
    args=()
    for p in "${all_patterns[@]}"; do args+=(--skip "$p"); done
    exec cargo test --workspace --locked -- "${args[@]}"
fi

while read -r name patterns; do
    if [[ "$name" == "$shard" ]]; then
        read -r -a ps <<<"$patterns"
        exec cargo test --workspace --locked -- "${ps[@]}"
    fi
done <<<"$SHARDS"

echo "unknown shard: '$shard'; the shards are $("$0" --list)" >&2
exit 2
