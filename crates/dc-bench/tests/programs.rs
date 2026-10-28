//! The benchmark programs (spec §11) at small n: each gives its Rust reference
//! result, its step count is deterministic, and a wrong result is rejected.

use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use dc_bench::{programs, BenchError, Instance, Program};
use dc_eval::{Tree, Value};

/// The spec §11 set, with the input size each is benchmarked at.
const SPEC_SET: [(&str, i64); 6] = [
    ("sum", 10_000),
    ("max_subarray", 10_000),
    ("msort", 1_000),
    ("fib_naive", 20),
    ("fib_memo", 60),
    ("bsearch", 100_000),
];

fn program(name: &str) -> Program {
    programs()
        .into_iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no benchmark program named {name}"))
}

#[test]
fn the_benchmark_set_is_the_spec_set() {
    let got: Vec<(&str, i64)> = programs().iter().map(|p| (p.name, p.full_n)).collect();
    assert_eq!(got, SPEC_SET);
}

#[test]
fn the_rust_references_match_known_values() {
    assert_eq!(program("fib_naive").expected(20), Value::Int(6_765));
    assert_eq!(
        program("fib_memo").expected(60),
        Value::Int(1_548_008_755_920)
    );
    assert_eq!(program("sum").expected(0), Value::Int(0));
    let sorted = program("msort").expected(50);
    let Value::List(l) = &sorted else {
        panic!("msort reference is not a list: {sorted}")
    };
    let xs: Vec<i64> = l.iter().map(|v| v.as_int().unwrap()).collect();
    assert_eq!(xs.len(), 50);
    assert!(xs.windows(2).all(|w| w[0] <= w[1]), "{xs:?}");
}

#[test]
fn every_program_gives_its_reference_result_at_small_n() {
    for (name, _) in SPEC_SET {
        let p = program(name);
        let c = p.compile().unwrap_or_else(|e| panic!("{name}: {e}"));
        let inst = p.instance(p.small_n);
        let steps = c.run(&inst).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(steps > 0, "{name}");
    }
}

#[test]
fn step_counts_are_deterministic() {
    for (name, _) in SPEC_SET {
        let p = program(name);
        let a = p.compile().unwrap();
        let b = p.compile().unwrap();
        let s1 = a.run(&p.instance(p.small_n)).unwrap();
        let inst = p.instance(p.small_n);
        let s2 = b.run(&inst).unwrap();
        let s3 = b.run(&inst).unwrap();
        assert_eq!((s1, s1), (s2, s3), "{name}");
    }
}

#[test]
fn step_counts_grow_with_n() {
    for (name, _) in SPEC_SET {
        let p = program(name);
        let c = p.compile().unwrap();
        let small = c.run(&p.instance(p.small_n)).unwrap();
        // Binary search is logarithmic: doubling n need not add a halving step.
        let factor = if name == "bsearch" { 16 } else { 2 };
        let larger = c.run(&p.instance(p.small_n * factor)).unwrap();
        assert!(larger > small, "{name}: {small} then {larger}");
    }
}

#[test]
fn a_wrong_result_is_rejected() {
    for (name, _) in SPEC_SET {
        let p = program(name);
        let c = p.compile().unwrap();
        let n = p.small_n;
        let wrong = Instance::new(p.inputs(n), Value::Bool(true));
        match c.run(&wrong) {
            Err(BenchError::WrongResult { .. }) => {}
            other => panic!("{name}: expected WrongResult, got {other:?}"),
        }
    }
}

#[test]
fn a_wrong_result_is_shown_shortened_even_when_shared() {
    // The expected value is a tree of 2^64 - 1 nodes built from 64 shared ones;
    // the error must show a prefix of it without walking the whole tree.
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let p = program("sum");
        let c = p.compile().unwrap();
        let mut t = Tree::leaf();
        for k in 0..64 {
            t = Tree::node(t.clone(), Value::Int(k), t);
        }
        let wrong = Instance::new(p.inputs(p.small_n), Value::tree(t));
        let shown = match c.run(&wrong) {
            Err(e @ BenchError::WrongResult { .. }) => e.to_string(),
            other => format!("expected WrongResult, got {other:?}"),
        };
        let _ = tx.send(shown);
    });
    match rx.recv_timeout(Duration::from_secs(20)) {
        Ok(s) => {
            assert!(s.contains("expected Node (Node ("), "{s}");
            assert!(s.ends_with("..."), "{s}");
            assert!(s.len() < 1_000, "{} bytes", s.len());
        }
        Err(RecvTimeoutError::Timeout) => panic!("the error was not shown within 20 s"),
        Err(RecvTimeoutError::Disconnected) => panic!("the test thread panicked"),
    }
}
