//! The built-in tree type end to end (P2.15): surface `Leaf`/`Node` syntax and
//! `match` desugar to the tree primitives, the result type-checks, evaluates at the
//! spec §5 costs, and round-trips through the Stitch format.

use dc_core::types::infer;
use dc_core::{sexpr, Arena, Scheme, TermId};
use dc_eval::{apply, Outcome, Tree, Value, UNLIMITED};
use dc_stitch::LibNames;
use dc_surface::TopLevel;
use proptest::prelude::*;

const NO_LIBS: &[Scheme] = &[];

const PROGRAMS: &str = r#"
let tree_sum t = tree_fold 0 (fun l x r -> l + x + r) t

let size t =
  bfix 64 (fun self t -> match t with Leaf -> 0 | Node (l, _, r) -> self l + 1 + self r) t

let insert v t =
  bfix 64 (fun self t ->
    match t with
    | Leaf -> Node (Leaf, v, Leaf)
    | Node (l, x, r) -> if v < x then Node (self l, x, r) else Node (l, x, self r)) t

let inorder t = tree_fold [] (fun l x r -> foldr (fun y acc -> y :: acc) (x :: r) l) t

let bst_sort xs = inorder (foldl (fun t x -> insert x t) Leaf xs)

let mirror t = match t with Leaf -> Leaf | Node (l, x, r) -> Node (r, x, l)
"#;

fn close(name: &str) -> (Arena, TermId) {
    let mut env = TopLevel::new();
    env.add_source("trees", PROGRAMS)
        .unwrap_or_else(|e| panic!("{e}"));
    let mut arena = Arena::new();
    let id = env
        .close_definition(&mut arena, name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    (arena, id)
}

fn core(name: &str) -> String {
    let (arena, id) = close(name);
    sexpr::print(&arena, id).unwrap()
}

fn type_of(name: &str) -> Scheme {
    let (arena, id) = close(name);
    infer(&arena, id, NO_LIBS).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn run(name: &str, inputs: &[Value]) -> (Value, u64) {
    let (arena, id) = close(name);
    match apply(&arena, id, inputs, UNLIMITED) {
        Outcome::Value(v, s) => (v, s),
        other => panic!("{name}: expected a value, got {other}"),
    }
}

fn sch(s: &str) -> Scheme {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn int_tree(t: &HostTree) -> Tree {
    match t {
        HostTree::Leaf => Tree::leaf(),
        HostTree::Node(l, x, r) => Tree::node(int_tree(l), Value::Int(*x), int_tree(r)),
    }
}

#[derive(Clone, Debug)]
enum HostTree {
    Leaf,
    Node(Box<HostTree>, i64, Box<HostTree>),
}

impl HostTree {
    fn nodes(&self) -> u64 {
        match self {
            HostTree::Leaf => 0,
            HostTree::Node(l, _, r) => l.nodes() + 1 + r.nodes(),
        }
    }
    fn sum(&self) -> i64 {
        match self {
            HostTree::Leaf => 0,
            HostTree::Node(l, x, r) => l.sum() + x + r.sum(),
        }
    }
}

fn arb_tree() -> impl Strategy<Value = HostTree> {
    Just(HostTree::Leaf).prop_recursive(6, 64, 2, |inner| {
        (inner.clone(), -100i64..100, inner)
            .prop_map(|(l, x, r)| HostTree::Node(Box::new(l), x, Box::new(r)))
    })
}

#[test]
fn tree_programs_desugar_to_the_tree_primitives() {
    assert_eq!(
        core("tree_sum"),
        "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))"
    );
    // match t with Leaf -> e1 | Node (l, x, r) -> e2
    //   ==> tree_case t (fun _ -> e1) (fun l x r -> e2)
    assert_eq!(
        core("size"),
        "(lam (bfix 64 (lam (lam (tree_case $0 (lam 0) (lam (lam (lam (add (add ($4 $2) 1) ($4 $0)))))))) $0))"
    );
    assert_eq!(
        core("mirror"),
        "(lam (tree_case $0 (lam leaf) (lam (lam (lam (node $0 $1 $2))))))"
    );
}

#[test]
fn tree_programs_get_their_types() {
    assert_eq!(type_of("tree_sum"), sch("Tree Int -> Int"));
    assert_eq!(type_of("size"), sch("Tree a -> Int"));
    assert_eq!(type_of("insert"), sch("Int -> Tree Int -> Tree Int"));
    assert_eq!(type_of("inorder"), sch("Tree a -> List a"));
    assert_eq!(type_of("bst_sort"), sch("List Int -> List Int"));
    assert_eq!(type_of("mirror"), sch("Tree a -> Tree a"));
}

#[test]
fn bst_sort_sorts() {
    let xs = [5i64, -3, 8, 0, 5, 12, -7, 1];
    let (v, _) = run("bst_sort", &[Value::int_list(&xs)]);
    let mut sorted = xs.to_vec();
    sorted.sort();
    assert_eq!(v, Value::int_list(&sorted));
}

#[test]
fn mirror_costs_one_case_and_one_node() {
    // β 1 + tree_case 1 + β 3 + node 1 = 6; on a leaf: β 1 + tree_case 1 + β 1 + leaf 1 = 4.
    let t = Tree::node(
        Tree::leaf(),
        Value::Int(1),
        Tree::node(Tree::leaf(), Value::Int(2), Tree::leaf()),
    );
    let (v, s) = run("mirror", &[Value::Tree(t)]);
    let want = Tree::node(
        Tree::node(Tree::leaf(), Value::Int(2), Tree::leaf()),
        Value::Int(1),
        Tree::leaf(),
    );
    assert_eq!((v, s), (Value::Tree(want), 6));
    assert_eq!(
        run("mirror", &[Value::Tree(Tree::leaf())]),
        (Value::Tree(Tree::leaf()), 4)
    );
}

#[test]
fn tree_programs_round_trip_through_stitch() {
    let names = LibNames::new();
    for name in [
        "tree_sum", "size", "insert", "inorder", "bst_sort", "mirror",
    ] {
        let (mut arena, id) = close(name);
        let text = dc_stitch::encode(&arena, id, &names).unwrap_or_else(|e| panic!("{e}"));
        for tok in ["tree_case", "tree_fold", "node", "leaf"] {
            if sexpr::print(&arena, id).unwrap().contains(tok) {
                assert!(text.contains(tok), "{name}: {tok} missing from {text}");
            }
        }
        let back = dc_stitch::decode(&mut arena, &text, &names).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(back, id, "{name}: {text}");
    }
}

proptest! {
    #[test]
    fn tree_sum_costs_six_per_node(t in arb_tree()) {
        // β 1 + per node: iteration 1 + β 3 + add 2.
        let (v, s) = run("tree_sum", &[Value::Tree(int_tree(&t))]);
        prop_assert_eq!(v, Value::Int(t.sum()));
        prop_assert_eq!(s, 1 + 6 * t.nodes());
    }

    #[test]
    fn match_recursion_cost_is_linear_in_nodes(t in arb_tree()) {
        // Each node adds one Node-branch visit and one more leaf, so the cost is
        // the cost on a leaf plus a fixed amount per node.
        let one = Tree::node(Tree::leaf(), Value::Int(0), Tree::leaf());
        let (_, s0) = run("size", &[Value::Tree(Tree::leaf())]);
        let (_, s1) = run("size", &[Value::Tree(one)]);
        let (v, s) = run("size", &[Value::Tree(int_tree(&t))]);
        prop_assert_eq!(v, Value::Int(t.nodes() as i64));
        prop_assert_eq!(s, s0 + (s1 - s0) * t.nodes());
    }
}
