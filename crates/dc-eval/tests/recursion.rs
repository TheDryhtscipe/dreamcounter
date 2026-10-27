//! bfix and memorec (P2.8, spec §6, decision D9).

mod common;

use common::*;
use dc_eval::{ErrorKind, Fun, Outcome, Value};

/// Factorial with depth fuel `n + 1`.
const FACT: &str =
    "(lam (bfix (add $0 1) (lam (lam (if (le $0 0) 1 (mul $0 ($1 (sub $0 1)))))) $0))";
/// Factorial with depth fuel `n`: one call short.
const FACT_SHORT: &str = "(lam (bfix $0 (lam (lam (if (le $0 0) 1 (mul $0 ($1 (sub $0 1)))))) $0))";

#[test]
fn bfix_factorial_costs_seven_per_call() {
    // β 1 + add 1, then per call x > 0: call 1 + β 2 + le 1 + if 1 + sub 1 + mul 1 = 7,
    // and the base call: call 1 + β 2 + le 1 + if 1 = 5. Total 2 + 7n + 5 = 7n + 7.
    let mut fact = 1i64;
    for n in 0..=12i64 {
        if n > 0 {
            fact *= n;
        }
        assert_value_on(FACT, &[int(n)], int(fact), 7 * n as u64 + 7);
    }
}

#[test]
fn bfix_fuel_bounds_the_depth_and_exhaustion_is_an_error() {
    // n calls each charge call 1 + β 2 + le 1 + if 1 + sub 1 = 6 before calling
    // `self` again; the (n+1)th call has fuel 0. Total β 1 + 6n.
    for n in 0..=6i64 {
        assert_error_on(
            FACT_SHORT,
            &[int(n)],
            ErrorKind::FuelExhausted,
            1 + 6 * n as u64,
        );
    }
}

#[test]
fn bfix_depth_fuel_is_exact() {
    // Countdown recursion from x makes x + 1 nested calls, so it needs fuel x + 1.
    let p = "(lam (lam (bfix $0 (lam (lam (if (le $0 0) 0 (add 1 ($1 (sub $0 1)))))) $1)))";
    for x in 0..8i64 {
        for fuel in 0..10i64 {
            let o = run_on(p, &[int(x), int(fuel)]);
            if fuel > x {
                assert!(
                    matches!(&o, Outcome::Value(v, _) if *v == int(x)),
                    "x={x} fuel={fuel}: {o}"
                );
            } else {
                assert!(
                    matches!(o, Outcome::Error(ErrorKind::FuelExhausted, _)),
                    "x={x} fuel={fuel}: {o}"
                );
            }
        }
    }
}

#[test]
fn bfix_sibling_calls_get_equal_fuel() {
    // A full binary recursion of depth d makes 2^(d+1) - 1 calls but needs only
    // depth d + 1. If siblings shared one budget, the second sibling would starve.
    let tree = |fuel: &str| {
        format!("(lam (bfix {fuel} (lam (lam (if (le $0 0) 1 (add ($1 (sub $0 1)) ($1 (sub $0 1)))))) $0))")
    };
    let enough = tree("(add $0 1)");
    let short = tree("$0");
    for d in 0..=10i64 {
        // Internal call: call 1 + β 2 + le 1 + if 1 + sub 2 + add 1 = 8, for 2^d - 1
        // calls; leaf call: 5, for 2^d calls; plus β 1 + add 1.
        let leaves = 1u64 << d;
        let steps = 2 + 8 * (leaves - 1) + 5 * leaves;
        assert_value_on(&enough, &[int(d)], int(1 << d), steps);
        assert!(matches!(
            run_on(&short, &[int(d)]),
            Outcome::Error(ErrorKind::FuelExhausted, _)
        ));
    }
}

#[test]
fn bfix_fuel_can_be_any_int_expression() {
    // Fuel `length xs + 1` for a list recursion that sums the list.
    let p = "(lam (bfix (add (length $0) 1) (lam (lam (list_case $0 (lam 0) (lam (lam (add $1 ($3 $0))))))) $0))";
    let xs = range1(20);
    match run_on(p, &[xs]) {
        Outcome::Value(v, _) => assert_eq!(v, int(210)),
        other => panic!("{other}"),
    }
}

const FIB_F: &str = "(lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2))))))";

fn naive_fib(n: i64) -> Outcome {
    run_on(&format!("(lam (bfix (add $0 1) {FIB_F} $0))"), &[int(n)])
}

fn memo_fib(n: i64) -> Outcome {
    run_on(&format!("(lam (memorec (add $0 1) {FIB_F} $0))"), &[int(n)])
}

fn host_fib(n: i64) -> i64 {
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

#[test]
fn naive_bfix_fib_is_exponential() {
    // C(x) = 5 for x < 2 (call 1 + β 2 + lt 1 + if 1); otherwise
    // C(x) = 8 + C(x-1) + C(x-2) (call 1 + β 2 + lt 1 + if 1 + sub 2 + add 1).
    // Total: β 1 + add 1 + C(n).
    let mut c = vec![5u64, 5];
    for x in 2..=20 {
        c.push(8 + c[x - 1] + c[x - 2]);
    }
    for n in 0..=20i64 {
        assert_value_on(
            &format!("(lam (bfix (add $0 1) {FIB_F} $0))"),
            &[int(n)],
            int(host_fib(n)),
            2 + c[n as usize],
        );
    }
    // The growth ratio tends to the golden ratio.
    let ratio = value_steps(&naive_fib(20)) as f64 / value_steps(&naive_fib(19)) as f64;
    assert!((ratio - 1.618).abs() < 0.01, "ratio {ratio}");
}

#[test]
fn memo_fib_is_linear() {
    // Misses: every x in 0..=n once. Miss x ≥ 2: 1 + key 1 + β 2 + lt 1 + if 1 +
    // sub 2 + add 1 = 9; miss x < 2: 1 + 1 + β 2 + lt 1 + if 1 = 6. Hits: self (x-2)
    // for x in 3..=n, 1 + key 1 = 2 each. Total β 1 + add 1 + 9(n-1) + 12 + 2(n-2)
    // = 11n + 1 for n ≥ 2.
    for n in 2..=60i64 {
        assert_value_on(
            &format!("(lam (memorec (add $0 1) {FIB_F} $0))"),
            &[int(n)],
            int(host_fib(n)),
            11 * n as u64 + 1,
        );
    }
    // n < 2: β 1 + add 1 + one miss of 6.
    assert!(matches!(memo_fib(0), Outcome::Value(_, 8)));
    assert!(matches!(memo_fib(1), Outcome::Value(_, 8)));
    assert_eq!(value_steps(&memo_fib(30)) - value_steps(&memo_fib(29)), 11);
}

#[test]
fn memorec_fuel_counts_misses_across_the_table() {
    // fib n misses exactly n + 1 keys, at any depth. Fuel n fails on the last miss,
    // fib 0 inside fib 2: β 1 + (n - 1) × 7 charged before each self (x-1)
    // (miss 2 + β 2 + lt 1 + if 1 + sub 1) + fib 1's miss 6 + fib 2's sub 1 = 7n + 1.
    let short = format!("(lam (memorec $0 {FIB_F} $0))");
    for n in 2..=20i64 {
        assert_error_on(
            &short,
            &[int(n)],
            ErrorKind::FuelExhausted,
            7 * n as u64 + 1,
        );
    }
}

#[test]
fn memorec_hits_do_not_use_fuel() {
    // call 1 (miss): 2 + β 2 + int_eq 1 + if 1 = 6; call 0 (miss): 6; call 0 (hit): 2;
    // add 1. Total 15, with 2 misses.
    let p = |fuel: i64| {
        format!("(memorec {fuel} (lam (lam (if (int_eq $0 0) 0 (add ($1 0) ($1 0))))) 1)")
    };
    assert_value(&p(2), int(0), 15);
    assert_value(&p(100), int(0), 15);
    // With 1 miss allowed, the miss on 0 fails after the first call's 6 steps.
    assert_error(&p(1), ErrorKind::FuelExhausted, 6);
}

#[test]
fn memorec_charges_key_size() {
    // pair 1 + miss (1 + key 3) + β 2 = 7.
    assert_value("(memorec 5 (lam (lam 0)) (pair 1 2))", int(0), 7);
    // pair 1; miss (1,2): 4; β 2; fst 1; int_eq 1; if 1; pair 1; miss (0,2): 4; β 2;
    // fst 1; int_eq 1; if 1; pair 1; hit (0,2): 4; add 1. Total 26.
    let p = "(memorec 5 (lam (lam (if (int_eq (fst $0) 0) 0 (add ($1 (pair 0 2)) ($1 (pair 0 2)))))) (pair 1 2))";
    assert_value(p, int(0), 26);
}

#[test]
fn memorec_recursive_call_with_invalid_key() {
    // miss 2 + β 2 + nil 1, then the self call's key is a list.
    let p = "(memorec 5 (lam (lam ($1 nil))) 3)";
    assert_error(p, ErrorKind::InvalidMemoKey, 5);
}

#[test]
fn memorec_cycle_runs_out_of_misses() {
    // A call on the key being computed is a miss again (nothing is stored yet).
    // Each miss: 2 + β 2 = 4; the fourth miss fails after 3 × 4 = 12.
    assert_error(
        "(memorec 3 (lam (lam ($1 $0))) 7)",
        ErrorKind::FuelExhausted,
        12,
    );
}

#[test]
fn memo_tables_are_not_shared_between_evaluations() {
    // Two evaluations of the same program give the same, all-miss cost.
    let a = value_steps(&memo_fib(25));
    let b = value_steps(&memo_fib(25));
    assert_eq!(a, b);
    assert_eq!(a, 11 * 25 + 1);
}

#[test]
fn each_memorec_gets_its_own_table() {
    // Two memorec evaluations on the same key, inside an outer memorec, each with
    // one miss allowed. Separate tables: each is a miss (2 + β 2 = 4). A shared table
    // would make the second a hit (2), and shared misses would make it fail.
    // outer miss 2 + β 2 = 4; inner 4 + 4; add 1. Total 13.
    let inner = "(memorec 1 (lam (lam 7)) 5)";
    let p = format!("(memorec 1 (lam (lam (add {inner} {inner}))) 0)");
    assert_value(&p, int(14), 13);
}

/// The memo table that an escaped `self` refers to, if it is still alive.
fn escaped_table_alive(v: &Value) -> bool {
    match v {
        Value::Fun(f) => match &**f {
            Fun::MemoSelf { table, .. } => table.upgrade().is_some(),
            other => panic!("expected a memorec self, got {other:?}"),
        },
        other => panic!("expected a function, got {other}"),
    }
}

#[test]
fn memo_table_is_dropped_when_memorec_returns() {
    // F returns (self, 0), which is also stored in the table under key 1, so the
    // table holds an entry that holds `self`. If `self` owned the table, this cycle
    // would keep it alive after the evaluation.
    match &run("(memorec 5 (lam (lam (pair $1 0))) 1)") {
        Outcome::Value(Value::Pair(p), 5) => assert!(!escaped_table_alive(&p.0)),
        other => panic!("{other}"),
    }
    // A bare escaped self: miss 2 + β 2.
    match run("(memorec 5 (lam (lam $1)) 1)") {
        Outcome::Value(v, 4) => assert!(!escaped_table_alive(&v)),
        other => panic!("{other}"),
    }
}

#[test]
fn calling_an_escaped_memo_self_is_an_error() {
    // miss 2 + β 2 = 4; the escaped call is charged 0.
    assert_error(
        "((memorec 5 (lam (lam $1)) 1) 2)",
        ErrorKind::MemoSelfEscaped,
        4,
    );
    // Through a closure that captured self: miss 2 + β 2 + β 1 (the closure) = 5.
    assert_error(
        "((memorec 5 (lam (lam (lam ($2 $0)))) 1) 3)",
        ErrorKind::MemoSelfEscaped,
        5,
    );
}
