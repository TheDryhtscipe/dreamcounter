//! Spec §7 invariants: monotone fuel (3) and determinism (1).
//!
//! Every case carries its expected outcome, worked out by hand, so the
//! determinism tests also catch a change in any step count, not only a
//! difference between runs or threads.

mod common;

use common::*;
use dc_core::{sexpr, Arena, TermId};
use dc_eval::{apply, Outcome, Value, UNLIMITED};
use proptest::prelude::*;

/// A program, a builder for its inputs, and its expected outcome. Inputs are
/// built per thread, because values hold `Rc` and are not `Send`.
struct Case {
    src: &'static str,
    inputs: fn() -> Vec<Value>,
    /// The outcome with no fuel limit, as `Outcome`'s `Display` writes it.
    expected: &'static str,
    /// Whether every charge in this program is a single step, so a timeout must
    /// report exactly `fuel + 1`.
    unit_charges: bool,
}

fn no_inputs() -> Vec<Value> {
    vec![]
}

const CASES: &[Case] = &[
    // β 1 + 40 × 4.
    Case {
        src: "(lam (foldl (lam (lam (add $1 $0))) 0 $0))",
        inputs: || vec![range1(40)],
        expected: "Value(820, 161)",
        unit_charges: true,
    },
    // β 1 + nil 1 + 8 × 2.
    Case {
        src: "(lam (foldr cons nil $0))",
        inputs: || vec![range1(8)],
        expected: "Value([1; 2; 3; 4; 5; 6; 7; 8], 18)",
        unit_charges: true,
    },
    // Naive fib 12: β 1 + add 1 + C(12) = 3021, with C(x) = 8 + C(x-1) + C(x-2).
    Case {
        src: "(lam (bfix (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
        inputs: || vec![int(12)],
        expected: "Value(144, 3023)",
        unit_charges: true,
    },
    // Memo fib 40: 11n + 1. Memo calls charge 2 (key size 1).
    Case {
        src: "(lam (memorec (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
        inputs: || vec![int(40)],
        expected: "Value(102334155, 441)",
        unit_charges: false,
    },
    // 6 × 7 + 4.
    Case {
        src: "(unfold (lam (if (lt 0 $0) (some (pair $0 (sub $0 1))) none)) 6)",
        inputs: no_inputs,
        expected: "Value([6; 5; 4; 3; 2; 1], 46)",
        unit_charges: true,
    },
    // β 1 + length 1 + tabulate 6 × 3 + array_set (1 + ⌈log₂ 7⌉ = 4) + to_list 7.
    Case {
        src: "(lam (array_to_list (array_set (array_tabulate (length $0) (lam (mul $0 $0))) 3 0)))",
        inputs: || vec![range1(6)],
        expected: "Value([0; 1; 4; 0; 16; 25], 31)",
        unit_charges: false,
    },
    // β 1 + 31 nodes × 6; 0 + 1 + ... + 30.
    Case {
        src: "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))",
        inputs: || vec![Value::Tree(balanced(0, 31))],
        expected: "Value(465, 187)",
        unit_charges: true,
    },
    // β 2 + nodes (20 cells, 20 heads, nil) 41.
    Case {
        src: "(lam (lam (eq $1 $0)))",
        inputs: || vec![range1(20), range1(20)],
        expected: "Value(true, 43)",
        unit_charges: false,
    },
    // Errors: the outcome must be the same error at every sufficient fuel.
    // β 1 + 10 elements × (1 + β 2 + sub 1 + div 1); the 10th divides by zero.
    Case {
        src: "(lam (foldl (lam (lam (div $1 (sub 10 $0)))) 1000 $0))",
        inputs: || vec![range1(20)],
        expected: "Error(DivisionByZero, 51)",
        unit_charges: true,
    },
    // β 1 + 8 calls × 6 before the 9th call finds no fuel.
    Case {
        src: "(lam (bfix $0 (lam (lam (if (le $0 0) 1 (mul $0 ($1 (sub $0 1)))))) $0))",
        inputs: || vec![int(8)],
        expected: "Error(FuelExhausted, 49)",
        unit_charges: true,
    },
    // β 1 + pair 2 + nodes (pair, 10 cells, 10 heads, nil, fun) 23.
    Case {
        src: "(lam (eq (pair $0 (lam $0)) (pair $0 (lam $0))))",
        inputs: || vec![range1(10)],
        expected: "Error(EqOnFunction, 26)",
        unit_charges: false,
    },
    // β 1 + tabulate 4 × 2 + first iteration (1 + β 2 + sub 1 + array_set 4).
    Case {
        src: "(lam (loop 30 (array_tabulate 4 (lam $0)) (lam (lam (array_set $0 (sub $1 20) 1)))))",
        inputs: || vec![int(0)],
        expected: "Error(IndexOutOfBounds, 17)",
        unit_charges: false,
    },
];

fn build(src: &str) -> (Arena, TermId) {
    let mut arena = Arena::new();
    let t = sexpr::parse(&mut arena, src).expect("parse");
    (arena, t)
}

/// An outcome as comparable, `Send` text: kind, value or error, and steps.
fn summary(o: &Outcome) -> String {
    o.to_string()
}

fn run_case(case: &Case, fuel: u64) -> Outcome {
    let (arena, t) = build(case.src);
    apply(&arena, t, &(case.inputs)(), fuel)
}

#[test]
fn every_case_gives_its_expected_outcome() {
    for case in CASES {
        assert_eq!(
            summary(&run_case(case, UNLIMITED)),
            case.expected,
            "{}",
            case.src
        );
    }
    let errors = CASES
        .iter()
        .filter(|c| c.expected.starts_with("Error"))
        .count();
    // The error cases exercise the fuel invariant on errors as well as values.
    assert_eq!(errors, 4);
}

proptest! {
    #[test]
    fn fuel_is_monotone(idx in 0..CASES.len(), seed in any::<u64>(), extra in 0u64..1000) {
        let case = &CASES[idx];
        let full = run_case(case, UNLIMITED);
        let total = full.steps();
        let fuel = seed % (total + 3);
        let o = run_case(case, fuel);
        if fuel >= total {
            prop_assert_eq!(summary(&o), summary(&full));
        } else {
            match &o {
                Outcome::Timeout(s) => {
                    let s = *s;
                    prop_assert!(s > fuel, "timeout at {} with fuel {}", s, fuel);
                    prop_assert!(s <= total);
                    if case.unit_charges {
                        prop_assert_eq!(s, fuel + 1);
                    }
                }
                other => prop_assert!(false, "fuel {} < {}: {}", fuel, total, other),
            }
        }
        // Any outcome reached within fuel f is unchanged at f + extra.
        if !matches!(o, Outcome::Timeout(_)) {
            prop_assert_eq!(summary(&run_case(case, fuel + extra)), summary(&o));
        }
    }
}

#[test]
fn repeated_runs_give_the_expected_outcome() {
    for case in CASES {
        for _ in 0..3 {
            assert_eq!(
                summary(&run_case(case, UNLIMITED)),
                case.expected,
                "{}",
                case.src
            );
        }
    }
}

#[test]
fn threads_give_the_expected_outcome() {
    // One immutable arena shared by every thread; values are built inside each.
    let mut arena = Arena::new();
    let terms: Vec<TermId> = CASES
        .iter()
        .map(|c| sexpr::parse(&mut arena, c.src).expect("parse"))
        .collect();
    let arena = &arena;
    let terms = &terms;
    let expected: Vec<&str> = CASES.iter().map(|c| c.expected).collect();
    let results: Vec<Vec<String>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|k| {
                s.spawn(move || {
                    // Different threads walk the cases in different orders.
                    let mut out = vec![String::new(); CASES.len()];
                    for i in 0..CASES.len() {
                        let j = (i + k) % CASES.len();
                        let o = apply(arena, terms[j], &(CASES[j].inputs)(), UNLIMITED);
                        out[j] = summary(&o);
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect()
    });
    for r in results {
        assert_eq!(r, expected);
    }
}

#[test]
fn hash_consing_does_not_change_cost() {
    // The same program parsed into a fresh arena and into one that already holds
    // other terms (so different ids) costs the same.
    let case = &CASES[3];
    let (a1, t1) = build(case.src);
    let mut a2 = Arena::new();
    sexpr::parse(&mut a2, "(add 1 (mul 2 3))").expect("parse");
    for c in CASES {
        sexpr::parse(&mut a2, c.src).expect("parse");
    }
    let t2 = sexpr::parse(&mut a2, case.src).expect("parse");
    assert_ne!(
        t1, t2,
        "the ids should differ for this test to mean anything"
    );
    assert_eq!(
        summary(&apply(&a1, t1, &(case.inputs)(), UNLIMITED)),
        case.expected
    );
    assert_eq!(
        summary(&apply(&a2, t2, &(case.inputs)(), UNLIMITED)),
        case.expected
    );
}
