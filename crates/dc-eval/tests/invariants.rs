//! Spec §7 invariants: monotone fuel (3) and determinism (1).

mod common;

use common::*;
use dc_core::{sexpr, Arena, TermId};
use dc_eval::{apply, Outcome, Value, UNLIMITED};
use proptest::prelude::*;

/// A program and a builder for its inputs. Inputs are built per thread, because
/// values hold `Rc` and are not `Send`.
struct Case {
    src: &'static str,
    inputs: fn() -> Vec<Value>,
}

fn no_inputs() -> Vec<Value> {
    vec![]
}

const CASES: &[Case] = &[
    Case {
        src: "(lam (foldl (lam (lam (add $1 $0))) 0 $0))",
        inputs: || vec![range1(40)],
    },
    Case {
        src: "(lam (foldr cons nil $0))",
        inputs: || vec![range1(30)],
    },
    Case {
        src: "(lam (bfix (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
        inputs: || vec![int(12)],
    },
    Case {
        src: "(lam (memorec (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
        inputs: || vec![int(40)],
    },
    Case {
        src: "(unfold (lam (if (lt 0 $0) (some (pair $0 (sub $0 1))) none)) 20)",
        inputs: no_inputs,
    },
    Case {
        src: "(lam (array_to_list (array_set (array_tabulate (length $0) (lam (mul $0 $0))) 3 0)))",
        inputs: || vec![range1(25)],
    },
    Case {
        src: "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))",
        inputs: || vec![Value::Tree(balanced(0, 31))],
    },
    Case {
        src: "(lam (lam (eq $1 $0)))",
        inputs: || vec![range1(20), range1(20)],
    },
    // Errors: the outcome must be the same error at every sufficient fuel.
    Case {
        src: "(lam (foldl (lam (lam (div $1 (sub 10 $0)))) 1000 $0))",
        inputs: || vec![range1(20)],
    },
    Case {
        src: "(lam (bfix $0 (lam (lam (if (le $0 0) 1 (mul $0 ($1 (sub $0 1)))))) $0))",
        inputs: || vec![int(8)],
    },
    Case {
        src: "(lam (eq (pair $0 (lam $0)) (pair $0 (lam $0))))",
        inputs: || vec![range1(10)],
    },
    Case {
        src: "(lam (loop 30 (array_tabulate 4 (lam $0)) (lam (lam (array_set $0 (sub $1 20) 1)))))",
        inputs: || vec![int(0)],
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
fn every_case_finishes_without_a_fuel_limit() {
    let mut errors = 0;
    for case in CASES {
        match run_case(case, UNLIMITED) {
            Outcome::Timeout(_) => panic!("{} timed out", case.src),
            Outcome::Error(..) => errors += 1,
            Outcome::Value(..) => {}
        }
    }
    // The error cases exercise the invariant on errors as well as values.
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
fn evaluation_is_deterministic_across_runs() {
    for case in CASES {
        let first = summary(&run_case(case, UNLIMITED));
        for _ in 0..3 {
            assert_eq!(summary(&run_case(case, UNLIMITED)), first, "{}", case.src);
        }
    }
}

#[test]
fn evaluation_is_deterministic_across_threads() {
    // One immutable arena shared by every thread; values are built inside each.
    let mut arena = Arena::new();
    let terms: Vec<TermId> = CASES
        .iter()
        .map(|c| sexpr::parse(&mut arena, c.src).expect("parse"))
        .collect();
    let arena = &arena;
    let terms = &terms;
    let reference: Vec<String> = CASES
        .iter()
        .zip(terms)
        .map(|(c, &t)| summary(&apply(arena, t, &(c.inputs)(), UNLIMITED)))
        .collect();
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
        assert_eq!(r, reference);
    }
}

#[test]
fn hash_consing_does_not_change_cost() {
    // The same program parsed into a fresh arena and into one that already holds
    // other terms (so different ids) costs the same.
    let src = CASES[3].src;
    let (a1, t1) = build(src);
    let mut a2 = Arena::new();
    sexpr::parse(&mut a2, "(add 1 (mul 2 3))").expect("parse");
    for c in CASES {
        sexpr::parse(&mut a2, c.src).expect("parse");
    }
    let t2 = sexpr::parse(&mut a2, src).expect("parse");
    assert_ne!(
        t1, t2,
        "the ids should differ for this test to mean anything"
    );
    let o1 = apply(&a1, t1, &[int(40)], UNLIMITED);
    let o2 = apply(&a2, t2, &[int(40)], UNLIMITED);
    assert_eq!(summary(&o1), summary(&o2));
}
