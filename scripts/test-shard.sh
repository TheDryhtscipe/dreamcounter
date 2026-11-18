#!/usr/bin/env bash
# Runs one shard of `cargo test --workspace --locked`, so that CI can run the
# slow tests in parallel jobs.
#
#   scripts/test-shard.sh --list    prints the shard names as a JSON array
#   scripts/test-shard.sh <shard>   runs that shard's tests
#
# Each named shard runs the tests with the names it lists, matched whole
# (libtest's --exact). The shard `rest` skips each of those names. So the
# shards together run every test, and a new test runs in `rest` unless a
# shard lists its name. A named shard fails if one of its names matches no
# test, so that a renamed test cannot leave its shard empty without notice.
# CI reads the shard names from `--list`, so a shard added here runs there.
set -euo pipefail
cd "$(dirname "$0")/.."

# One line per shard: the name, then the test names. Each test takes
# minutes of CPU (crates/dc-tasks/tests/chains.rs, corpus.rs). A CI runner
# has 4 CPUs, and the slowest of these tests took about 8 minutes on one in
# November 2026, so each shard holds at most four.
SHARDS="
gate-1 gate_count gate_longest gate_heaviest gate_longest_count
gate-2 the_gate_lets_through_a_less_careful_longest_program the_gate_lets_through_a_less_careful_longest_count_program the_gate_turns_away_a_program_that_is_quadratic_on_an_antichain the_gate_turns_away_a_lean_quadratic_program
gate-3 a_proposers_longest_program_passes_the_gate a_proposers_heaviest_program_passes_the_gate the_gate_turns_away_a_program_that_is_quadratic_only_on_sorted_xs dominance_variants_good_at_every_size_others_at_the_lower_half
corpus-chains chain_variants_good_at_every_size_others_at_the_lower_half
"

names=()
all_tests=()
while read -r name tests; do
    [[ -z "$name" ]] && continue
    names+=("$name")
    read -r -a ts <<<"$tests"
    all_tests+=("${ts[@]}")
done <<<"$SHARDS"

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "usage: $0 --list | <shard>; the shards are ${names[*]} rest" >&2
    exit 2
fi
shard="$1"

if [[ "$shard" == "--list" ]]; then
    printf '['
    for n in "${names[@]}"; do printf '"%s",' "$n"; done
    printf '"rest"]\n'
    exit 0
fi

if [[ "$shard" == "rest" ]]; then
    args=()
    for t in "${all_tests[@]}"; do args+=(--skip "$t"); done
    exec cargo test --workspace --locked -- --exact "${args[@]}"
fi

ts=()
while read -r name tests; do
    if [[ -n "$name" && "$name" == "$shard" ]]; then
        read -r -a ts <<<"$tests"
    fi
done <<<"$SHARDS"
if [[ ${#ts[@]} -eq 0 ]]; then
    echo "unknown shard: '$shard'; the shards are ${names[*]} rest" >&2
    exit 2
fi

listed=$(cargo test --workspace --locked -- --exact --list "${ts[@]}")
for t in "${ts[@]}"; do
    if ! grep -qxF "$t: test" <<<"$listed"; then
        echo "shard $shard: no test is named $t" >&2
        exit 1
    fi
done
exec cargo test --workspace --locked -- --exact "${ts[@]}"
