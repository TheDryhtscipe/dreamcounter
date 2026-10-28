//! Values that share parts can be exponentially larger as trees than in memory:
//! `loop 64 (node $0 i $0)` is built in a few hundred steps and has 2^64 nodes.
//! Equality and `Debug` on such values must finish in time linear in the memory
//! they use, and equality must stay exact. Each test runs on its own thread and
//! fails if that thread has not finished within [`LIMIT`].

use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use dc_eval::{eval, List, Outcome, Tree, Value, UNLIMITED};

/// Far longer than any of these tests takes when it does not walk the full tree.
const LIMIT: Duration = Duration::from_secs(20);

/// Runs `f` on a 256 KiB thread and fails if it does not finish within [`LIMIT`].
/// A thread that runs over is left behind; the test process ends it on exit.
fn finishes<F: FnOnce() + Send + 'static>(f: F) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            f();
            let _ = tx.send(());
        })
        .expect("spawn test thread");
    match rx.recv_timeout(LIMIT) {
        Ok(()) => handle.join().expect("test thread"),
        Err(RecvTimeoutError::Disconnected) => match handle.join() {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(()) => panic!("test thread ended without reporting"),
        },
        Err(RecvTimeoutError::Timeout) => panic!("did not finish within {LIMIT:?}"),
    }
}

/// `node t k t` for k in 0..levels, from `leaf`: 2^levels - 1 nodes as a tree.
fn shared_tree(levels: i64) -> Tree {
    let mut t = Tree::leaf();
    for k in 0..levels {
        t = Tree::node(t.clone(), Value::Int(k), t);
    }
    t
}

/// `(v, v)` nested `levels` deep from `base`.
fn shared_pairs(levels: u32, base: Value) -> Value {
    let mut v = base;
    for _ in 0..levels {
        v = Value::pair(v.clone(), v);
    }
    v
}

/// `[v; v]` nested `levels` deep from `()`.
fn shared_lists(levels: u32) -> Value {
    let mut v = Value::Unit;
    for _ in 0..levels {
        v = Value::list([v.clone(), v]);
    }
    v
}

/// An array of `width` copies of the previous level, `levels` deep from `base`.
fn shared_arrays(levels: u32, width: usize, base: Value) -> Value {
    let mut v = base;
    for _ in 0..levels {
        v = Value::array(vec![v; width]);
    }
    v
}

/// A function value.
fn function() -> Value {
    let mut arena = dc_core::Arena::new();
    let t = dc_core::sexpr::parse(&mut arena, "(lam $0)").expect("parse");
    match eval(&arena, t, UNLIMITED) {
        Outcome::Value(v, _) => v,
        other => panic!("{other}"),
    }
}

#[test]
fn equal_shared_trees_built_apart_are_equal() {
    finishes(|| {
        let a = Value::tree(shared_tree(64));
        let b = Value::tree(shared_tree(64));
        assert!(a == b);
        assert!(a == a.clone());
    });
}

#[test]
fn a_difference_after_a_shared_subtree_is_found() {
    finishes(|| {
        // The trees differ only in the root value of the right subtree, which
        // pre-order reaches after the whole left subtree.
        let t = shared_tree(63);
        let Tree(Some(n)) = &t else {
            panic!("not a node")
        };
        let other = Tree::node(n.left.clone(), Value::Int(99), n.right.clone());
        let x = Value::tree(Tree::node(t.clone(), Value::Int(0), t.clone()));
        let y = Value::tree(Tree::node(t.clone(), Value::Int(0), other));
        assert!(x != y);
        assert!(x == x.clone());
    });
}

#[test]
fn shared_pairs_lists_and_arrays_compare_exactly() {
    finishes(|| {
        let p = shared_pairs(64, Value::Int(1));
        assert!(p == shared_pairs(64, Value::Int(1)));
        let q = Value::pair(
            shared_pairs(63, Value::Int(1)),
            shared_pairs(63, Value::Int(2)),
        );
        assert!(p != q);

        let l = shared_lists(64);
        assert!(l == shared_lists(64));
        let m = Value::list([shared_lists(63), Value::list([shared_lists(62)])]);
        assert!(l != m);

        // 64^12 = 2^72 elements as a tree.
        let a = shared_arrays(12, 64, Value::Int(1));
        assert!(a == shared_arrays(12, 64, Value::Int(1)));
        let b = Value::array(
            (0..64)
                .map(|i| shared_arrays(11, 64, Value::Int(if i == 63 { 2 } else { 1 })))
                .collect::<Vec<_>>(),
        );
        assert!(a != b);
    });
}

#[test]
fn a_function_behind_a_shared_part_is_never_equal() {
    finishes(|| {
        // Pre-order reaches the function only after 2^63 nodes on the left.
        let v = Value::pair(shared_pairs(63, Value::Int(1)), shared_pairs(1, function()));
        assert!(v != v.clone());
    });
}

#[test]
fn debug_of_a_shared_value_is_bounded() {
    finishes(|| {
        let s = format!("{:?}", Value::tree(shared_tree(64)));
        assert!(s.len() < 100_000, "{} bytes", s.len());
        assert!(s.starts_with("Node (Node (Node ("), "{s}");
        assert!(s.contains("..."), "{s}");
        let s = format!("{:?}", shared_lists(64));
        assert!(s.len() < 100_000, "{} bytes", s.len());
        let s = format!("{:?}", shared_arrays(12, 64, Value::Int(1)));
        assert!(s.len() < 100_000, "{} bytes", s.len());
    });
}

#[test]
fn debug_of_a_list_uses_the_surface_syntax() {
    let l = List::from_values([Value::Int(1), Value::Int(2)]);
    assert_eq!(format!("{l:?}"), "[1; 2]");
    assert_eq!(format!("{l:?}"), Value::List(l).to_string());
}

#[test]
fn display_within_elides_past_the_limit() {
    let xs = Value::int_list(&[1, 2, 3]);
    assert_eq!(xs.display_within(0).to_string(), "...");
    // The list and its first two elements, then the rest as one `...`.
    assert_eq!(xs.display_within(3).to_string(), "[1; 2; ...]");
    assert_eq!(xs.display_within(4).to_string(), xs.to_string());
    let p = Value::pair(Value::Int(1), Value::pair(Value::Int(2), Value::Int(3)));
    assert_eq!(p.display_within(2).to_string(), "(1, ...)");
    assert_eq!(p.display_within(5).to_string(), "(1, (2, 3))");
    let a = Value::array([Value::Int(1), Value::Int(2), Value::Int(3)]);
    assert_eq!(a.display_within(2).to_string(), "[|1; ...|]");
    let t = Value::tree(Tree::node(Tree::leaf(), Value::Int(1), Tree::leaf()));
    assert_eq!(t.display_within(1).to_string(), "Node (..., ..., ...)");
    assert_eq!(t.display_within(4).to_string(), "Node (Leaf, 1, Leaf)");
    let o = Outcome::Value(xs, 5);
    assert_eq!(o.display_within(2).to_string(), "Value([1; ...], 5)");
    assert_eq!(o.display_within(u64::MAX).to_string(), o.to_string());
}

#[test]
fn display_within_a_shared_value_is_bounded() {
    finishes(|| {
        let t = Value::tree(shared_tree(64));
        let s = t.display_within(1_000).to_string();
        assert!(s.len() < 100_000, "{} bytes", s.len());
        // The written prefix is the start of the full tree: the leftmost path.
        assert!(s.starts_with(&"Node (".repeat(64)), "{s}");
        assert!(s.contains("Leaf, 0, Leaf"), "{s}");
        assert!(s.ends_with(")"), "{s}");
    });
}

#[test]
fn large_unshared_values_compare_exactly() {
    // Past the size where equality starts remembering the pairs it compared.
    let n = 100_000;
    let xs: Vec<i64> = (0..n).collect();
    let mut ys = xs.clone();
    ys[(n - 1) as usize] = -1;
    assert!(Value::int_list(&xs) == Value::int_list(&xs));
    assert!(Value::int_list(&xs) != Value::int_list(&ys));
    assert!(Value::int_list(&xs) != Value::int_list(&xs[1..]));
    let arr = |v: &[i64]| Value::array(v.iter().map(|&n| Value::Int(n)));
    assert!(arr(&xs) == arr(&xs));
    assert!(arr(&xs) != arr(&ys));
    let tree = |v: &[i64]| {
        Value::tree(v.iter().fold(Tree::leaf(), |t, &n| {
            Tree::node(Tree::leaf(), Value::Int(n), t)
        }))
    };
    assert!(tree(&xs) == tree(&xs));
    assert!(tree(&xs) != tree(&ys));
}
