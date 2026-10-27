//! Helpers shared by the integration tests.
#![allow(dead_code)]

use dc_core::{sexpr, Arena, TermId};
use dc_eval::{apply, ErrorKind, Outcome, Tree, Value, UNLIMITED};

/// Parses `src` into a fresh arena.
pub fn parse(src: &str) -> (Arena, TermId) {
    let mut arena = Arena::new();
    let t = sexpr::parse(&mut arena, src).unwrap_or_else(|e| panic!("{src}: {e}"));
    (arena, t)
}

/// Evaluates `src` applied to `inputs` with `fuel`.
pub fn run_fuel(src: &str, inputs: &[Value], fuel: u64) -> Outcome {
    let (arena, t) = parse(src);
    apply(&arena, t, inputs, fuel)
}

/// Evaluates `src` applied to `inputs` with no fuel limit.
pub fn run_on(src: &str, inputs: &[Value]) -> Outcome {
    run_fuel(src, inputs, UNLIMITED)
}

/// Evaluates the closed term `src` with no fuel limit.
pub fn run(src: &str) -> Outcome {
    run_on(src, &[])
}

/// Asserts that `src` on `inputs` returns `expected` in exactly `steps` steps.
#[track_caller]
pub fn assert_value_on(src: &str, inputs: &[Value], expected: Value, steps: u64) {
    match run_on(src, inputs) {
        Outcome::Value(v, s) => {
            assert_eq!(v, expected, "value of {src}");
            assert_eq!(s, steps, "steps of {src}");
        }
        other => panic!("{src}: expected a value, got {other}"),
    }
}

/// Asserts that the closed term `src` returns `expected` in exactly `steps` steps.
#[track_caller]
pub fn assert_value(src: &str, expected: Value, steps: u64) {
    assert_value_on(src, &[], expected, steps);
}

/// Asserts that `src` on `inputs` stops with error `kind` after exactly `steps`.
#[track_caller]
pub fn assert_error_on(src: &str, inputs: &[Value], kind: ErrorKind, steps: u64) {
    match run_on(src, inputs) {
        Outcome::Error(k, s) => {
            assert_eq!(k, kind, "error kind of {src}");
            assert_eq!(s, steps, "steps of {src}");
        }
        other => panic!("{src}: expected {kind:?}, got {other}"),
    }
}

/// Asserts that the closed term `src` stops with error `kind` after exactly `steps`.
#[track_caller]
pub fn assert_error(src: &str, kind: ErrorKind, steps: u64) {
    assert_error_on(src, &[], kind, steps);
}

/// The steps of an evaluation that must finish with a value.
#[track_caller]
pub fn value_steps(o: &Outcome) -> u64 {
    match o {
        Outcome::Value(_, s) => *s,
        other => panic!("expected a value, got {other}"),
    }
}

pub fn int(n: i64) -> Value {
    Value::Int(n)
}

pub fn ints(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

/// `[1, 2, ..., n]`.
pub fn range1(n: i64) -> Value {
    Value::list((1..=n).map(Value::Int))
}

/// A balanced tree holding `lo..hi` in order.
pub fn balanced(lo: i64, hi: i64) -> Tree {
    if lo >= hi {
        return Tree::leaf();
    }
    let mid = lo + (hi - lo) / 2;
    Tree::node(balanced(lo, mid), Value::Int(mid), balanced(mid + 1, hi))
}

/// Runs `f` on a thread with a small stack, so a host recursion over a deep
/// structure overflows instead of passing by luck.
pub fn on_small_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(f)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}
