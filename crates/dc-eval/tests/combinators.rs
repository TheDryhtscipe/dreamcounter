//! foldl, foldr, loop and unfold (P2.7): exact costs and fold laws.

mod common;

use common::*;
use dc_eval::{Outcome, Value};
use proptest::prelude::*;

const SUM_FOLDL: &str = "(lam (foldl (lam (lam (add $1 $0))) 0 $0))";

#[test]
fn foldl_sum_costs_one_plus_four_per_element() {
    // β 1 (program) + per element: iteration 1 + β 2 + add 1 = 4.
    for n in [0, 1, 2, 5, 100] {
        assert_value_on(
            SUM_FOLDL,
            &[range1(n)],
            int(n * (n + 1) / 2),
            1 + 4 * n as u64,
        );
    }
}

#[test]
fn foldl_and_foldr_associate_in_opposite_directions() {
    // 1 + 4 × 3 = 13 each.
    let l = "(lam (foldl (lam (lam (sub $1 $0))) 0 $0))";
    let r = "(lam (foldr (lam (lam (sub $1 $0))) 0 $0))";
    // ((0 - 1) - 2) - 3.
    assert_value_on(l, &[ints(&[1, 2, 3])], int(-6), 13);
    // 1 - (2 - (3 - 0)).
    assert_value_on(r, &[ints(&[1, 2, 3])], int(2), 13);
}

#[test]
fn folds_with_a_primitive_function() {
    // β 1 + per element: iteration 1 + add 1 (partial application free) = 2.
    assert_value_on("(lam (foldl add 0 $0))", &[range1(10)], int(55), 21);
    // β 1 + nil 1 + per element: iteration 1 + cons 1 = 2.
    assert_value_on("(lam (foldr cons nil $0))", &[range1(10)], range1(10), 22);
}

#[test]
fn loop_counts_up_from_zero() {
    // 3 × (iteration 1 + β 2 + add 1) = 12; 0 + 1 + 2.
    assert_value("(loop 3 0 (lam (lam (add $1 $0))))", int(3), 12);
    // f i acc = cons i acc, applied for i = 0, 1, 2: nil 1 + 3 × (1 + cons 1) = 7.
    assert_value("(loop 3 nil cons)", ints(&[2, 1, 0]), 7);
    // n ≤ 0 returns z for free.
    assert_value("(loop 0 5 (lam (lam (add $1 $0))))", int(5), 0);
    assert_value("(loop -2 5 (lam (lam (add $1 $0))))", int(5), 0);
}

#[test]
fn unfold_charges_produced_elements_only() {
    // Productive call: β 1 + lt 1 + if 1 + sub 1 + pair 1 + some 1 + element 1 = 7.
    // Final call: β 1 + lt 1 + if 1 + none 1 = 4. Total 3 × 7 + 4 = 25.
    let p = "(unfold (lam (if (lt 0 $0) (some (pair $0 (sub $0 1))) none)) 3)";
    assert_value(p, ints(&[3, 2, 1]), 25);
    // β 1 + none 1.
    assert_value("(unfold (lam none) 0)", ints(&[]), 2);
}

#[test]
fn foldr_on_a_shared_tail_list_keeps_its_input() {
    // The input is shared with the result's tail: the fold must not consume it.
    let xs = range1(5);
    let p = "(lam (foldr cons (cons 0 nil) $0))";
    // β 1 + nil 1 + cons 1 + 5 × 2 = 13.
    assert_value_on(p, std::slice::from_ref(&xs), ints(&[1, 2, 3, 4, 5, 0]), 13);
    assert_eq!(xs, range1(5));
}

fn list(xs: &[i64]) -> Value {
    ints(xs)
}

fn outcome_value(o: Outcome) -> (Value, u64) {
    match o {
        Outcome::Value(v, s) => (v, s),
        other => panic!("expected a value, got {other}"),
    }
}

// F is not commutative or associative, so the laws below check the order of
// application, not only the multiset of elements.
const F: &str = "(lam (lam (sub $0 (mul 2 $1))))";

proptest! {
    #[test]
    fn foldr_cons_nil_is_identity(xs in prop::collection::vec(-1000i64..1000, 0..60)) {
        let n = xs.len() as u64;
        let (v, s) = outcome_value(run_on("(lam (foldr cons nil $0))", &[list(&xs)]));
        prop_assert_eq!(v, list(&xs));
        // β 1 + nil 1 + 2 per element.
        prop_assert_eq!(s, 2 + 2 * n);
    }

    #[test]
    fn foldl_over_append_is_foldl_of_foldl(
        xs in prop::collection::vec(-50i64..50, 0..20),
        ys in prop::collection::vec(-50i64..50, 0..20),
    ) {
        let (n, m) = (xs.len() as u64, ys.len() as u64);
        // append xs ys = foldr cons ys xs.
        let lhs = format!("(lam (lam (foldl {F} 0 (foldr cons $0 $1))))");
        let rhs = format!("(lam (lam (foldl {F} (foldl {F} 0 $1) $0)))");
        let inputs = [list(&xs), list(&ys)];
        let (lv, ls) = outcome_value(run_on(&lhs, &inputs));
        let (rv, rs) = outcome_value(run_on(&rhs, &inputs));
        prop_assert_eq!(lv, rv);
        // F per element: iteration 1 + β 2 + mul 1 + sub 1 = 5.
        // rhs: β 2 + 5 (n + m). lhs adds the append: 2 per element of xs.
        prop_assert_eq!(rs, 2 + 5 * (n + m));
        prop_assert_eq!(ls, rs + 2 * n);
    }

    #[test]
    fn foldr_over_append_is_foldr_of_foldr(
        xs in prop::collection::vec(-50i64..50, 0..20),
        ys in prop::collection::vec(-50i64..50, 0..20),
    ) {
        let lhs = format!("(lam (lam (foldr {F} 0 (foldr cons $0 $1))))");
        let rhs = format!("(lam (lam (foldr {F} (foldr {F} 0 $0) $1)))");
        let inputs = [list(&xs), list(&ys)];
        let (n, m) = (xs.len() as u64, ys.len() as u64);
        let (lv, ls) = outcome_value(run_on(&lhs, &inputs));
        let (rv, rs) = outcome_value(run_on(&rhs, &inputs));
        prop_assert_eq!(lv, rv);
        // As for foldl: rhs β 2 + 5 (n + m); lhs adds the append, 2 per element of xs.
        prop_assert_eq!(rs, 2 + 5 * (n + m));
        prop_assert_eq!(ls, rs + 2 * n);
    }

    #[test]
    fn foldl_matches_a_host_fold(xs in prop::collection::vec(-50i64..50, 0..40)) {
        let expected = xs.iter().fold(0i64, |acc, &x| x - 2 * acc);
        let (v, s) = outcome_value(run_on(&format!("(lam (foldl {F} 0 $0))"), &[list(&xs)]));
        prop_assert_eq!(v, int(expected));
        prop_assert_eq!(s, 1 + 5 * xs.len() as u64);
    }

    #[test]
    fn foldr_matches_a_host_fold(xs in prop::collection::vec(-50i64..50, 0..60)) {
        let expected = xs.iter().rev().fold(0i64, |acc, &x| acc - 2 * x);
        let (v, s) = outcome_value(run_on(&format!("(lam (foldr {F} 0 $0))"), &[list(&xs)]));
        prop_assert_eq!(v, int(expected));
        // β 1 + 5 per element, as for foldl.
        prop_assert_eq!(s, 1 + 5 * xs.len() as u64);
    }

    #[test]
    fn foldl_is_foldr_of_the_flipped_function_over_the_reverse(
        xs in prop::collection::vec(-50i64..50, 0..40),
    ) {
        let n = xs.len() as u64;
        // reverse xs = foldl (λacc. λx. x :: acc) nil xs; flip F = λx. λacc. F acc x.
        let lhs = format!("(lam (foldl {F} 7 $0))");
        let rhs = format!(
            "(lam (foldr (lam (lam ({F} $0 $1))) 7 (foldl (lam (lam (cons $0 $1))) nil $0)))"
        );
        let (lv, ls) = outcome_value(run_on(&lhs, &[list(&xs)]));
        let (rv, rs) = outcome_value(run_on(&rhs, &[list(&xs)]));
        prop_assert_eq!(lv, rv);
        prop_assert_eq!(ls, 1 + 5 * n);
        // β 1 + nil 1 + reverse 4 per element (iteration, β 2, cons) + the flipped
        // fold 7 per element (iteration, β 2 into flip, β 2 into F, mul, sub).
        prop_assert_eq!(rs, 2 + 4 * n + 7 * n);
    }

    #[test]
    fn loop_matches_foldl_over_a_range(n in -3i64..60) {
        // loop n z f = f (n-1) (... (f 0 z)), so it equals foldl (flip f) z [0..n).
        let host = (0..n.max(0)).fold(0i64, |acc, i| acc - 2 * i);
        let p = "(lam (loop $0 0 (lam (lam (sub $0 (mul 2 $1))))))";
        let (v, s) = outcome_value(run_on(p, &[int(n)]));
        prop_assert_eq!(v, int(host));
        prop_assert_eq!(s, 1 + 5 * n.max(0) as u64);
    }
}
