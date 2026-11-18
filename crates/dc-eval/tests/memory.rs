//! The memory limit of one evaluation (`Limits::memory`, bead
//! dreamcounter-a3db93).
//!
//! Each test measures the evaluation's peak heap with the counting allocator
//! (`dc_eval::memory`). Each runaway program is given enough fuel to hold
//! several times the limit, so with the check removed it would end in a
//! timeout far above the limit, and these tests would fail.

mod common;

use common::parse;
use dc_eval::{apply_within, memory, Limits, Outcome, Value, UNLIMITED};

/// The limit the runaway programs get.
const LIMIT: u64 = 12 << 20;

/// How far above the limit the measured peak may be. One step of these
/// programs allocates a few kilobytes at most, and a collection step checks
/// every 1024 elements.
const SLACK: u64 = 64 << 10;

/// Runs `src` on `inputs` within `limits`, and returns its outcome and the
/// peak heap it held, in bytes. The inputs are built before the evaluation
/// starts, so they are not part of the peak.
fn measure(src: &str, inputs: &[Value], limits: Limits) -> (Outcome, u64) {
    let (arena, t) = parse(src);
    let base = memory::live_bytes();
    memory::reset_peak();
    let outcome = apply_within(&arena, t, inputs, limits);
    let peak = u64::try_from(memory::peak_bytes() - base).unwrap_or(0);
    (outcome, peak)
}

#[track_caller]
fn assert_stopped_near_the_limit(src: &str, inputs: &[Value], fuel: u64, limit: u64) {
    let (outcome, peak) = measure(
        src,
        inputs,
        Limits {
            fuel,
            memory: limit,
        },
    );
    assert!(
        matches!(outcome, Outcome::MemoryLimit(s) if s <= fuel),
        "{src}: expected MemoryLimit, got {}",
        outcome.display_within(4)
    );
    assert!(
        peak <= limit + SLACK,
        "{src}: peak {peak} B is above the limit {limit} B by more than {SLACK} B"
    );
}

/// A runaway `array_tabulate`: about 32 bytes per step, so 4M steps would
/// hold about 128 MB.
const TABULATE: &str = "(array_tabulate 1000000000000 (lam $0))";

/// A runaway `unfold`: about 16 bytes per step, so 8M steps would hold about
/// 128 MB.
const UNFOLD: &str = "(unfold (lam (some (pair $0 (add $0 1)))) 0)";

/// A loop that keeps every `array_set` version of an 8192-element array:
/// about 300 bytes per step, so 500k steps would hold about 150 MB.
const SET_VERSIONS: &str = "(loop 1000000000 (pair (array_tabulate 8192 (lam 0)) nil) \
     (lam (lam (pair (array_set (fst $0) (mod $1 8192) $1) (cons (fst $0) (snd $0))))))";

#[test]
fn a_runaway_array_tabulate_stops_at_the_limit() {
    assert_stopped_near_the_limit(TABULATE, &[], 4_000_000, LIMIT);
}

#[test]
fn a_runaway_unfold_stops_at_the_limit() {
    assert_stopped_near_the_limit(UNFOLD, &[], 8_000_000, LIMIT);
}

#[test]
fn a_loop_that_keeps_every_array_set_version_stops_at_the_limit() {
    assert_stopped_near_the_limit(SET_VERSIONS, &[], 500_000, LIMIT);
}

/// A list of the ints `0..n`, built outside the evaluation.
fn range(n: i64) -> Value {
    Value::int_list(&(0..n).collect::<Vec<i64>>())
}

#[test]
fn array_of_list_stops_inside_its_one_step() {
    // A million elements take about 16 MB as an array; the step stops when
    // the array passes 8 MB.
    assert_stopped_near_the_limit(
        "(lam (array_of_list $0))",
        &[range(1_000_000)],
        10_000_000,
        8 << 20,
    );
}

#[test]
fn array_to_list_stops_inside_its_one_step() {
    // The copy of a million elements (16 MB) fits in 24 MB; the list built
    // from it does not, and the step stops while building it.
    let array = Value::array((0..1_000_000).map(Value::Int));
    assert_stopped_near_the_limit("(lam (array_to_list $0))", &[array], 10_000_000, 24 << 20);
}

#[test]
fn foldr_stops_before_it_copies_a_list_too_long_for_the_limit() {
    assert_stopped_near_the_limit(
        "(lam (foldr (lam (lam (add $0 $1))) 0 $0))",
        &[range(1_000_000)],
        10_000_000,
        8 << 20,
    );
}

#[test]
fn deep_recursion_stops_when_its_continuation_stack_would_pass_the_limit() {
    // A non-tail fold over a long input list: each element leaves a frame.
    let src =
        "(lam (bfix 100000000 (lam (lam (list_case $0 (lam 0) (lam (lam (add $1 ($3 $0))))))) $0))";
    assert_stopped_near_the_limit(src, &[range(2_000_000)], 100_000_000, 4 << 20);
}

#[test]
fn memory_freed_during_the_run_does_not_count() {
    // 2000 arrays of 8192 elements, each dropped before the next: about
    // 500 MB allocated in all, but only one array held at a time.
    let src = "(loop 2000 0 (lam (lam (add $0 (array_size (array_tabulate 8192 (lam 0)))))))";
    let (outcome, peak) = measure(
        src,
        &[],
        Limits {
            fuel: UNLIMITED,
            memory: 4 << 20,
        },
    );
    assert!(
        matches!(outcome, Outcome::Value(Value::Int(16_384_000), _)),
        "{}",
        outcome.display_within(4)
    );
    assert!(peak < 4 << 20, "peak {peak}");
}

#[test]
fn a_run_within_the_limit_has_the_outcome_it_has_with_no_limit() {
    for src in [
        "(foldl (lam (lam (add $0 $1))) 0 (unfold (lam (if (lt $0 100000) (some (pair $0 (add $0 1))) none)) 0))",
        "(array_to_list (array_tabulate 50000 (lam (mul $0 $0))))",
    ] {
        let (free, _) = measure(src, &[], Limits::fuel(UNLIMITED));
        let (limited, peak) = measure(
            src,
            &[],
            Limits {
                fuel: UNLIMITED,
                memory: 64 << 20,
            },
        );
        assert!(peak < 64 << 20, "{src}: peak {peak}");
        assert_eq!(free.to_string(), limited.to_string(), "{src}");
        assert!(matches!(limited, Outcome::Value(..)), "{src}");
    }
}

#[test]
fn the_fuel_still_wins_when_it_runs_out_first() {
    let (outcome, _) = measure(
        TABULATE,
        &[],
        Limits {
            fuel: 1000,
            memory: LIMIT,
        },
    );
    assert!(matches!(outcome, Outcome::Timeout(1001)), "{outcome}");
}
