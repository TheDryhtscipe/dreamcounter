//! Deep recursion and deep values must not use the host stack (spec §4, D6).
//! Every test runs on a 256 KiB thread, so a host recursion proportional to the
//! depth would overflow.

mod common;

use common::*;
use dc_eval::{List, Outcome, Tree, Value};

const N: i64 = 100_000;

#[test]
fn foldr_over_a_hundred_thousand_elements() {
    on_small_stack(|| {
        // β 1 + 4 per element.
        let p = "(lam (foldr (lam (lam (add $1 $0))) 0 $0))";
        assert_value_on(p, &[range1(N)], int(N * (N + 1) / 2), 1 + 4 * N as u64);
    });
}

#[test]
fn bfix_recursion_a_hundred_thousand_deep() {
    on_small_stack(|| {
        // Not a tail call: each level adds 1 after its recursive call returns.
        // Per call x > 0: call 1 + β 2 + le 1 + if 1 + sub 1 + add 1 = 7; base 5;
        // plus β 1 + add 1 for the fuel. Total 7n + 7.
        let p = "(lam (bfix (add $0 1) (lam (lam (if (le $0 0) 0 (add 1 ($1 (sub $0 1)))))) $0))";
        assert_value_on(p, &[int(N)], int(N), 7 * N as u64 + 7);
    });
}

#[test]
fn memorec_recursion_a_hundred_thousand_deep() {
    on_small_stack(|| {
        // Every key misses once: miss 2 + β 2 + le 1 + if 1 + sub 1 + add 1 = 8 for
        // x > 0, 6 for x = 0; plus 2. Total 8n + 8.
        let p =
            "(lam (memorec (add $0 1) (lam (lam (if (le $0 0) 0 (add 1 ($1 (sub $0 1)))))) $0))";
        assert_value_on(p, &[int(N)], int(N), 8 * N as u64 + 8);
    });
}

#[test]
fn a_hundred_thousand_composed_closures() {
    on_small_stack(|| {
        // foldl builds acc' = λy. acc (y + x): a chain of n closures, then applies
        // it to 0. β 1 + n × (1 + β 2) + n × (β 1 + add 1) + β 1 (the identity).
        let p = "(lam (foldl (lam (lam (lam ($2 (add $0 $1))))) (lam $0) $0 0))";
        assert_value_on(p, &[range1(N)], int(N * (N + 1) / 2), 5 * N as u64 + 2);
    });
}

#[test]
fn dropping_a_million_element_list() {
    on_small_stack(|| {
        let xs: Vec<i64> = (0..1_000_000).collect();
        let v = Value::int_list(&xs);
        drop(v);
        // A bare list and a bare degenerate tree, outside any `Value`.
        let l = List::from_values(xs.iter().map(|&n| Value::Int(n)));
        assert_eq!(l.len(), 1_000_000);
        drop(l);
        let mut t = Tree::leaf();
        for n in 0..1_000_000 {
            t = Tree::node(t, Value::Int(n), Tree::leaf());
        }
        drop(t);
        // A list built by the machine, returned and then dropped.
        let o = run_on("(lam (foldr cons nil $0))", &[range1(1_000_000)]);
        assert_eq!(o.steps(), 2 + 2_000_000);
        drop(o);
    });
}

#[test]
fn deeply_nested_options_pairs_and_trees() {
    on_small_stack(|| {
        // none 1 + n × (1 + β 2 + some 1).
        let o = run("(loop 100000 none (lam (lam (some $0))))");
        assert!(matches!(o, Outcome::Value(_, s) if s == 4 * N as u64 + 1));
        drop(o);
        // eq walks n options and the final none: β 1 + (4n + 1) + (n + 1) = 5n + 3.
        let e = run_on(
            "(lam (let (loop $0 none (lam (lam (some $0)))) (eq $0 $0)))",
            &[int(N)],
        );
        assert!(matches!(e, Outcome::Value(Value::Bool(true), s) if s == 5 * N as u64 + 3));
        // unit 0 + n × (1 + β 2 + pair 1).
        let p = run("(loop 100000 unit (lam (lam (pair $1 $0))))");
        assert!(matches!(p, Outcome::Value(_, s) if s == 4 * N as u64));
        drop(p);
        // A left spine of depth n, folded: β 1 + loop (leaf 1 + 5n) + tree_fold
        // n × (1 + β 3 + add 1) = 10n + 2.
        let t = "(lam (tree_fold 0 (lam (lam (lam (add 1 $2)))) (loop $0 leaf (lam (lam (node $0 $1 leaf))))))";
        assert_value_on(t, &[int(N)], int(N), 10 * N as u64 + 2);
    });
}
