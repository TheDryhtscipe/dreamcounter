//! Persistent arrays (P2.9) and the built-in tree type (P2.15, evaluator half).

mod common;

use common::*;
use dc_eval::{Tree, Value};
use proptest::prelude::*;

fn array(xs: &[i64]) -> Value {
    Value::array(xs.iter().map(|&n| Value::Int(n)))
}

#[test]
fn tabulate_charges_one_per_element_plus_calls() {
    // 4 × (element 1 + β 1 + mul 1) = 12, array_get 1 = 13.
    assert_value(
        "(array_get (array_tabulate 4 (lam (mul $0 $0))) 3)",
        int(9),
        13,
    );
    // Elements in index order: 3 × (1 + β 1) = 6, array_to_list 3 + 1 = 4.
    assert_value(
        "(array_to_list (array_tabulate 3 (lam $0)))",
        ints(&[0, 1, 2]),
        10,
    );
    // n ≤ 0 is an empty array for free; array_size 1.
    assert_value("(array_size (array_tabulate 0 (lam $0)))", int(0), 1);
    assert_value("(array_size (array_tabulate -3 (lam $0)))", int(0), 1);
}

#[test]
fn array_set_costs_one_plus_ceil_log2_of_size_plus_one() {
    // β 1 + array_set (1 + ⌈log₂(n+1)⌉).
    for (n, log) in [(1, 1), (2, 2), (3, 2), (4, 3), (7, 3), (8, 4), (1000, 10)] {
        let xs: Vec<i64> = (0..n).collect();
        let mut expected = xs.clone();
        expected[0] = 7;
        assert_value_on(
            "(lam (array_set $0 0 7))",
            &[array(&xs)],
            array(&expected),
            2 + log,
        );
    }
}

#[test]
fn array_set_is_persistent() {
    // β 1 + array_set (1 + 3) + get 1 + get 1 + pair 1 = 8; the old array is unchanged.
    let p = "(lam (let (array_set $0 0 7) (pair (array_get $1 0) (array_get $0 0))))";
    assert_value_on(p, &[array(&[1, 2, 3, 4])], Value::pair(int(1), int(7)), 8);
}

#[test]
fn array_list_conversions_cost_n_plus_one() {
    // nil 1 + cons 2 + array_of_list 3 = 6.
    assert_value("(array_of_list (cons 1 (cons 2 nil)))", array(&[1, 2]), 6);
    // β 1 + (n + 1) + (n + 1).
    for n in [0, 1, 50] {
        assert_value_on(
            "(lam (array_to_list (array_of_list $0)))",
            &[range1(n)],
            range1(n),
            2 * n as u64 + 3,
        );
    }
}

#[test]
fn eq_on_arrays() {
    // tabulate 2 + 4, nodes (array of different size) 1 = 7.
    let p = "(eq (array_tabulate 1 (lam $0)) (array_tabulate 2 (lam $0)))";
    assert_value(p, Value::Bool(false), 7);
    // β 2 + nodes (array, 1, 2, 3) 4 = 6.
    let q = "(lam (lam (eq $1 $0)))";
    assert_value_on(
        q,
        &[array(&[1, 2, 3]), array(&[1, 2, 3])],
        Value::Bool(true),
        6,
    );
}

#[test]
fn binary_search_over_an_array() {
    // Lower bound of 777 in [0, 2, 4, ...]: a bfix recursion on (lo, hi).
    let p = "(lam (bfix 64 (lam (lam (let (fst $0) (let (snd $1) \
             (if (le $0 $1) $1 (let (div (add $1 $0) 2) \
             (if (lt (array_get $5 $0) 777) ($4 (pair (add $0 1) $1)) ($4 (pair $2 $0))))))))) \
             (pair 0 (array_size $0))))";
    let xs: Vec<i64> = (0..100_000).map(|i| 2 * i).collect();
    match run_on(p, &[array(&xs)]) {
        dc_eval::Outcome::Value(v, _) => assert_eq!(v, int(389)),
        other => panic!("{other}"),
    }
}

#[test]
fn tree_constructors_and_case() {
    // leaf 2 + node 1 = 3.
    assert_value(
        "(node leaf 5 leaf)",
        Value::Tree(Tree::node(Tree::leaf(), int(5), Tree::leaf())),
        3,
    );
    // leaf 1 + tree_case 1 + β 1 = 3.
    assert_value("(tree_case leaf (lam 7) (lam (lam (lam $1))))", int(7), 3);
    // β 1 + tree_case 1 + β 3 = 5; the node branch receives left, value, right.
    let t = Tree::node(
        Tree::leaf(),
        int(4),
        Tree::node(Tree::leaf(), int(9), Tree::leaf()),
    );
    assert_value_on(
        "(lam (tree_case $0 (lam 7) (lam (lam (lam $1)))))",
        &[Value::Tree(t.clone())],
        int(4),
        5,
    );
    assert_value_on(
        "(lam (tree_case $0 (lam leaf) (lam (lam (lam $0)))))",
        &[Value::Tree(t)],
        Value::Tree(Tree::node(Tree::leaf(), int(9), Tree::leaf())),
        5,
    );
}

#[test]
fn tree_fold_charges_internal_nodes_only() {
    // leaf 1; no node, so no iteration and no call.
    assert_value("(tree_fold 9 (lam (lam (lam 0))) leaf)", int(9), 1);
    // β 1 + per node: 1 + β 3 + add 2 = 6.
    let sum = "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))";
    for k in [0i64, 1, 2, 7, 100] {
        assert_value_on(
            sum,
            &[Value::Tree(balanced(1, k + 1))],
            int(k * (k + 1) / 2),
            1 + 6 * k as u64,
        );
    }
    // eq on trees: construction 3 + 3, nodes (node, leaf, 1, leaf) 4 = 10.
    assert_value(
        "(eq (node leaf 1 leaf) (node leaf 1 leaf))",
        Value::Bool(true),
        10,
    );
}

/// In-order traversal by tree_fold: f l x r = append l (x :: r).
const INORDER: &str = "(lam (tree_fold nil (lam (lam (lam (foldr cons (cons $1 $0) $2)))) $0))";

fn host_inorder(t: &Tree, out: &mut Vec<i64>) {
    if let Some(n) = &t.0 {
        host_inorder(&n.left, out);
        out.push(n.value.as_int().unwrap());
        host_inorder(&n.right, out);
    }
}

fn arb_tree() -> impl Strategy<Value = Tree> {
    let leaf = Just(()).prop_map(|_| Tree::leaf());
    leaf.prop_recursive(6, 64, 2, |inner| {
        (inner.clone(), -100i64..100, inner).prop_map(|(l, x, r)| Tree::node(l, Value::Int(x), r))
    })
}

proptest! {
    #[test]
    fn tree_fold_passes_left_value_right(t in arb_tree()) {
        let mut expected = Vec::new();
        host_inorder(&t, &mut expected);
        let o = run_on(INORDER, &[Value::Tree(t)]);
        prop_assert_eq!(o.value().cloned(), Some(ints(&expected)));
    }

    #[test]
    fn tree_fold_size_is_one_per_node(t in arb_tree()) {
        // β 1 + per node: 1 + β 3 + add 2 = 6.
        let k = t.size();
        let p = "(lam (tree_fold 0 (lam (lam (lam (add (add $2 1) $0)))) $0))";
        let o = run_on(p, &[Value::Tree(t)]);
        prop_assert_eq!(o.value().cloned(), Some(int(k as i64)));
        prop_assert_eq!(o.steps(), 1 + 6 * k);
    }
}

// ---------------------------------------------------------------- tree_fold laws

/// A combiner that is not symmetric in its subtrees and uses the value:
/// f l x r = 3l + x - 2r. One call costs β 3 + mul 1 + add 1 + mul 1 + sub 1 = 7.
const SKEW: &str = "(lam (lam (lam (sub (add (mul $2 3) $1) (mul 2 $0)))))";

/// `fun t -> tree_fold 0 SKEW t`.
fn fold_skew() -> String {
    format!("(lam (tree_fold 0 {SKEW} $0))")
}

fn host_skew(t: &Tree) -> i64 {
    match &t.0 {
        None => 0,
        Some(n) => 3 * host_skew(&n.left) + n.value.as_int().unwrap() - 2 * host_skew(&n.right),
    }
}

#[test]
fn tree_fold_of_leaf_is_the_initial_value_without_calling_f() {
    // f divides by zero, so any call of it would be an error.
    let boom = "(lam (lam (lam (div 1 0))))";
    // β 1 + leaf 1 = 2; tree_fold on a leaf charges nothing and calls nothing.
    assert_value_on(
        &format!("(lam (tree_fold $0 {boom} leaf))"),
        &[int(42)],
        int(42),
        2,
    );
    // One node: the call happens and fails after β 1 + leaf 2 + node 1 + iteration 1
    // + β 3 + div 1.
    assert_error_on(
        &format!("(lam (tree_fold $0 {boom} (node leaf 1 leaf)))"),
        &[int(42)],
        dc_eval::ErrorKind::DivisionByZero,
        9,
    );
}

proptest! {
    #[test]
    fn tree_fold_node_law(l in arb_tree(), x in -100i64..100, r in arb_tree()) {
        // tree_fold z f (node l x r) = f (tree_fold z f l) x (tree_fold z f r).
        let whole = Tree::node(l.clone(), int(x), r.clone());
        let lhs = run_on(&fold_skew(), &[Value::Tree(whole.clone())]);
        let rhs_src = format!(
            "(lam (lam (lam ({SKEW} (tree_fold 0 {SKEW} $2) $1 (tree_fold 0 {SKEW} $0)))))"
        );
        let rhs = run_on(&rhs_src, &[Value::Tree(l.clone()), int(x), Value::Tree(r.clone())]);
        prop_assert_eq!(lhs.value().cloned(), rhs.value().cloned());
        prop_assert_eq!(lhs.value().cloned(), Some(int(host_skew(&whole))));

        // Cost is additive over the node: the two subtree folds (less their outer β),
        // plus one iteration and one call of f for the root, plus the outer β.
        let sl = value_steps(&run_on(&fold_skew(), &[Value::Tree(l)]));
        let sr = value_steps(&run_on(&fold_skew(), &[Value::Tree(r)]));
        prop_assert_eq!(value_steps(&lhs), (sl - 1) + (sr - 1) + (1 + 7) + 1);
        // And so linear in the internal nodes: β 1 + 8 per node.
        prop_assert_eq!(value_steps(&lhs), 1 + 8 * whole.size());
    }

    #[test]
    fn tree_fold_with_node_is_the_identity(t in arb_tree()) {
        // tree_fold leaf node t = t. β 1 + leaf 1 + per node: 1 + β 3 + node 1.
        let p = "(lam (tree_fold leaf (lam (lam (lam (node $2 $1 $0)))) $0))";
        let k = t.size();
        let o = run_on(p, &[Value::Tree(t.clone())]);
        prop_assert_eq!(o.value().cloned(), Some(Value::Tree(t)));
        prop_assert_eq!(o.steps(), 2 + 5 * k);
    }

    #[test]
    fn tree_case_then_fold_agrees_with_fold(t in arb_tree()) {
        // Unfolding one level by tree_case and folding the subtrees gives the fold.
        let one_level = format!(
            "(lam (tree_case $0 (lam 0) (lam (lam (lam ({SKEW} (tree_fold 0 {SKEW} $2) $1 (tree_fold 0 {SKEW} $0)))))))"
        );
        let a = run_on(&fold_skew(), &[Value::Tree(t.clone())]);
        let b = run_on(&one_level, &[Value::Tree(t)]);
        prop_assert_eq!(a.value().cloned(), b.value().cloned());
    }
}
