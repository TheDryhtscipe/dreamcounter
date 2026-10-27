//! Every error kind, with the step count at the point of failure (P2.6).

mod common;

use common::*;
use dc_core::LibId;
use dc_core::Prim;
use dc_eval::{ErrorKind, Outcome, Stuck};

#[test]
fn division_by_zero_is_charged_then_raised() {
    // div 1.
    assert_error("(div 1 0)", ErrorKind::DivisionByZero, 1);
    assert_error("(mod 1 0)", ErrorKind::DivisionByZero, 1);
    // add 1 + β 1 + div 1 = 3.
    assert_error("((lam (div $0 0)) (add 1 2))", ErrorKind::DivisionByZero, 3);
}

#[test]
fn checked_arithmetic_overflows() {
    assert_error("(add 9223372036854775807 1)", ErrorKind::Overflow, 1);
    assert_error("(sub -9223372036854775808 1)", ErrorKind::Overflow, 1);
    assert_error("(mul 4611686018427387904 2)", ErrorKind::Overflow, 1);
    assert_error("(div -9223372036854775808 -1)", ErrorKind::Overflow, 1);
    assert_error("(mod -9223372036854775808 -1)", ErrorKind::Overflow, 1);
    // No overflow at the boundary itself.
    assert_value("(add 9223372036854775806 1)", int(i64::MAX), 1);
}

#[test]
fn array_index_out_of_bounds() {
    // tabulate 2 × (1 + β 1) = 4, array_get 1 = 5.
    let get = "(array_get (array_tabulate 2 (lam $0)) 2)";
    assert_error(get, ErrorKind::IndexOutOfBounds, 5);
    let get_neg = "(array_get (array_tabulate 2 (lam $0)) -1)";
    assert_error(get_neg, ErrorKind::IndexOutOfBounds, 5);
    // tabulate 4 + array_set (1 + ⌈log₂ 3⌉ = 3) = 7.
    let set = "(array_set (array_tabulate 2 (lam $0)) 5 0)";
    assert_error(set, ErrorKind::IndexOutOfBounds, 7);
}

#[test]
fn eq_on_functions() {
    // nodes (fun) 1.
    assert_error("(eq (lam $0) (lam $0))", ErrorKind::EqOnFunction, 1);
    // pair 2 + nodes (pair, 1, fun) 3 = 5.
    assert_error(
        "(eq (pair 1 (lam $0)) (pair 1 (lam $0)))",
        ErrorKind::EqOnFunction,
        5,
    );
    // The comparison stops at 1 vs 2 before it reaches the functions:
    // pair 2 + nodes (pair, 1 vs 2) 2 = 4.
    assert_value(
        "(eq (pair 1 (lam $0)) (pair 2 (lam $0)))",
        dc_eval::Value::Bool(false),
        4,
    );
}

#[test]
fn fuel_exhausted_from_bfix_and_memorec() {
    // bfix 0: the first call already has no fuel; nothing is charged.
    assert_error("(bfix 0 (lam (lam $0)) 1)", ErrorKind::FuelExhausted, 0);
    assert_error("(memorec 0 (lam (lam $0)) 1)", ErrorKind::FuelExhausted, 0);
}

#[test]
fn invalid_memo_keys() {
    let f = "(lam (lam 0))";
    // nil 1.
    assert_error(
        &format!("(memorec 5 {f} nil)"),
        ErrorKind::InvalidMemoKey,
        1,
    );
    assert_error(
        &format!("(memorec 5 {f} (lam $0))"),
        ErrorKind::InvalidMemoKey,
        0,
    );
    // some 1.
    assert_error(
        &format!("(memorec 5 {f} (some 1))"),
        ErrorKind::InvalidMemoKey,
        1,
    );
    // nil 1 + pair 1: a pair is a key only if its parts are.
    assert_error(
        &format!("(memorec 5 {f} (pair 1 nil))"),
        ErrorKind::InvalidMemoKey,
        2,
    );
    // Valid keys: Int, Bool, Unit and nested pairs of those.
    assert_value(
        &format!("(memorec 5 {f} (pair true (pair unit 3)))"),
        int(0),
        2 + 6 + 2,
    );
}

#[test]
fn timeout_is_the_first_count_above_the_fuel() {
    // 3000 iterations × (1 + β 2 + add 1) = 12000 steps.
    let p = "(loop 3000 0 (lam (lam (add $1 $0))))";
    assert!(matches!(run_fuel(p, &[], 10), Outcome::Timeout(11)));
    assert!(matches!(run_fuel(p, &[], 11999), Outcome::Timeout(12000)));
    assert!(matches!(run_fuel(p, &[], 12000), Outcome::Value(_, 12000)));
    assert!(matches!(run_fuel(p, &[], 0), Outcome::Timeout(1)));
    // A multi-step charge can jump past the fuel: array_of_list on 1000 elements
    // costs 1001 at once.
    let q = "(lam (array_of_list $0))";
    assert!(matches!(
        run_fuel(q, &[range1(1000)], 5),
        Outcome::Timeout(1002)
    ));
}

/// Runs `src` with `fuel` on another thread and fails if it takes more than
/// `secs` seconds, so host work that the fuel does not bound shows up as a failure.
fn run_fuel_within(src: &'static str, fuel: u64, secs: u64) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run_fuel(src, &[], fuel).to_string());
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("{src} with fuel {fuel} ran longer than {secs} s"))
}

#[test]
fn eq_on_a_shared_structure_stops_at_the_fuel() {
    // A tree with 2^64 - 1 nodes, built in 257 steps by sharing each subtree:
    // leaf 1 + 64 × (1 + β 2 + node 1). eq counts every node of both sides, so it
    // goes over the fuel; the comparison stops at the first node above it.
    let p = "(let (loop 64 leaf (lam (lam (node $0 0 $0)))) (eq $0 $0))";
    assert_eq!(run_fuel_within(p, 10_000, 20), "Timeout(10001)");
    // The same with pairs: 64 × (1 + β 2 + pair 1).
    let q = "(let (loop 64 0 (lam (lam (pair $0 $0)))) (eq $0 $0))";
    assert_eq!(run_fuel_within(q, 10_000, 20), "Timeout(10001)");
}

#[test]
fn memo_key_on_a_shared_structure_is_sized_before_it_is_built() {
    // A key of 2^41 - 1 nodes built in 40 × (1 + β 2 + pair 1) = 160 steps. The
    // call is charged 1 + key size, so the timeout count is 160 + 2^41.
    let p = "(memorec 1 (lam (lam 0)) (loop 40 0 (lam (lam (pair $0 $0)))))";
    assert_eq!(
        run_fuel_within(p, 10_000, 20),
        format!("Timeout({})", 160 + (1u64 << 41))
    );
    // An invalid key is found without charging, whatever its size: here the
    // function comes after the 2^41 - 1 valid nodes. pair 1 more.
    let q = "(memorec 1 (lam (lam 0)) (pair (loop 40 0 (lam (lam (pair $0 $0)))) (lam 0)))";
    assert_eq!(run_fuel_within(q, 10_000, 20), "Error(InvalidMemoKey, 161)");
    // A miss with no misses left is not charged, so it raises FuelExhausted
    // however large the key.
    let r = "(memorec 0 (lam (lam 0)) (loop 40 0 (lam (lam (pair $0 $0)))))";
    assert_eq!(run_fuel_within(r, 10_000, 20), "Error(FuelExhausted, 160)");
}

#[test]
fn unfold_without_end_times_out() {
    // Each iteration: β 1 + pair 1 + some 1 + produced element 1 = 4.
    let p = "(unfold (lam (some (pair 1 $0))) 0)";
    assert!(matches!(run_fuel(p, &[], 1000), Outcome::Timeout(1001)));
}

#[test]
fn stuck_terms() {
    let stuck = |s| ErrorKind::Stuck(s);
    assert_error("(1 2)", stuck(Stuck::NotAFunction), 0);
    assert_error("$0", stuck(Stuck::UnboundVar(0)), 0);
    // β 1, then index 3 in a one-entry environment.
    assert_error("((lam $3) 1)", stuck(Stuck::UnboundVar(3)), 1);
    assert_error("lib_0", stuck(Stuck::UnexpandedLib(LibId(0))), 0);
    // lib refs are stuck even in a lazy position once reached: if 1.
    assert_error(
        "(if true lib_2 0)",
        stuck(Stuck::UnexpandedLib(LibId(2))),
        1,
    );
    assert_error("(if 1 2 3)", stuck(Stuck::NotABool), 0);
    assert_error("(add true 1)", stuck(Stuck::BadArgument(Prim::Add)), 0);
    assert_error("(fst 1)", stuck(Stuck::BadArgument(Prim::Fst)), 0);
    // β 1, then the generator returned a non-option.
    assert_error(
        "(unfold (lam 1) 0)",
        stuck(Stuck::BadArgument(Prim::Unfold)),
        1,
    );
}
