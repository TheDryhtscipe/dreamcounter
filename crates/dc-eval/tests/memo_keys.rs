//! Memo keys (spec §6): two values give equal keys exactly when they are equal as
//! trees, however their parts are shared in memory, for keys on both sides of
//! the size at which keys switch from the flat to the shared form.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use dc_eval::{MemoKey, Value};
use proptest::prelude::*;

/// A key value as a tree, independent of how it is laid out in memory.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Shape {
    Int(i64),
    Bool(bool),
    Unit,
    Pair(Box<Shape>, Box<Shape>),
}

impl Shape {
    fn nodes(&self) -> u64 {
        match self {
            Shape::Pair(a, b) => 1 + a.nodes() + b.nodes(),
            _ => 1,
        }
    }
}

/// Builds `s` as a value. Where `share` says so for a node, a pair equal to one
/// built before is reused, so the same tree comes out with different sharing.
fn build(s: &Shape, share: &mut impl FnMut() -> bool, seen: &mut HashMap<Shape, Value>) -> Value {
    if let Some(v) = seen.get(s) {
        if share() {
            return v.clone();
        }
    }
    let v = match s {
        Shape::Int(n) => Value::Int(*n),
        Shape::Bool(b) => Value::Bool(*b),
        Shape::Unit => Value::Unit,
        Shape::Pair(a, b) => {
            let a = build(a, share, seen);
            let b = build(b, share, seen);
            Value::Pair(Rc::new((a, b)))
        }
    };
    seen.insert(s.clone(), v.clone());
    v
}

fn build_with(s: &Shape, bits: &[bool]) -> Value {
    let mut i = 0;
    let mut share = || {
        i += 1;
        bits.get(i % bits.len().max(1)).copied().unwrap_or(false)
    };
    build(s, &mut share, &mut HashMap::new())
}

fn hash_of(k: &MemoKey) -> u64 {
    let mut h = DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

/// The most nodes a generated shape has.
const MAX_NODES: u64 = 3000;

/// Shapes of up to [`MAX_NODES`] nodes, many of them above the largest flat key.
/// Each is the largest node of a random DAG whose pairs point at earlier nodes, so
/// equal subtrees, and so sharing, are common.
fn shape() -> impl Strategy<Value = Shape> {
    let node = prop_oneof![
        1 => (0i64..3).prop_map(|n| (0u8, n, 0usize, 0usize)),
        1 => any::<bool>().prop_map(|b| (1u8, i64::from(b), 0, 0)),
        4 => (any::<usize>(), any::<usize>()).prop_map(|(a, b)| (2u8, 0, a, b)),
    ];
    prop::collection::vec(node, 1..60).prop_map(|nodes| {
        let mut shapes: Vec<Shape> = Vec::new();
        for (kind, n, a, b) in nodes {
            let s = match (kind, shapes.len()) {
                (0, _) => Shape::Int(n),
                (1, _) => Shape::Bool(n == 1),
                (_, 0) => Shape::Unit,
                (_, len) => {
                    // Mostly one of the last few nodes, so shapes compound.
                    let pick = |i: usize| match i % 8 {
                        0 => i / 8 % len,
                        k => len - 1 - k % len.min(4),
                    };
                    let (a, b) = (&shapes[pick(a)], &shapes[pick(b)]);
                    if 1 + a.nodes() + b.nodes() > MAX_NODES {
                        continue;
                    }
                    Shape::Pair(Box::new(a.clone()), Box::new(b.clone()))
                }
            };
            shapes.push(s);
        }
        shapes
            .into_iter()
            .max_by_key(Shape::nodes)
            .unwrap_or(Shape::Unit)
    })
}

/// A shape and a nearby one: the same, or with one leaf changed.
fn shape_pair() -> impl Strategy<Value = (Shape, Shape)> {
    (shape(), any::<bool>(), any::<u64>()).prop_map(|(s, same, at)| {
        if same {
            return (s.clone(), s);
        }
        let mut t = s.clone();
        change_leaf(&mut t, at);
        (s, t)
    })
}

/// Changes one leaf of `s`, picked by walking left or right by the bits of `at`.
fn change_leaf(s: &mut Shape, mut at: u64) {
    let mut s = s;
    loop {
        match s {
            Shape::Pair(a, b) => {
                s = if at & 1 == 0 { a } else { b };
                at >>= 1;
            }
            Shape::Int(n) => {
                *n += 1;
                return;
            }
            Shape::Bool(b) => {
                *b = !*b;
                return;
            }
            Shape::Unit => {
                *s = Shape::Int(0);
                return;
            }
        }
    }
}

proptest! {
    #[test]
    fn keys_are_equal_exactly_when_the_trees_are(
        (s, t) in shape_pair(),
        bits_s in prop::collection::vec(any::<bool>(), 1..8),
        bits_t in prop::collection::vec(any::<bool>(), 1..8),
    ) {
        let ks = MemoKey::from_value(&build_with(&s, &bits_s)).expect("valid key");
        let kt = MemoKey::from_value(&build_with(&t, &bits_t)).expect("valid key");
        prop_assert_eq!(ks.size(), s.nodes());
        prop_assert_eq!(kt.size(), t.nodes());
        prop_assert_eq!(ks == kt, s == t);
        if s == t {
            prop_assert_eq!(hash_of(&ks), hash_of(&kt));
        }
    }
}

/// The doubling key `pair k k` built n times from 0, laid out fully shared (n
/// pairs in memory) and fully unshared (2^(n+1) - 1 nodes).
#[test]
fn a_doubling_key_is_the_same_key_shared_or_not() {
    for n in [5u32, 6, 7, 12] {
        let mut shared = Value::Int(0);
        for _ in 0..n {
            shared = Value::Pair(Rc::new((shared.clone(), shared)));
        }
        let mut shape = Shape::Int(0);
        for _ in 0..n {
            shape = Shape::Pair(Box::new(shape.clone()), Box::new(shape));
        }
        let unshared = build_with(&shape, &[false]);
        let a = MemoKey::from_value(&shared).expect("valid key");
        let b = MemoKey::from_value(&unshared).expect("valid key");
        assert_eq!(a.size(), (1u64 << (n + 1)) - 1);
        assert_eq!(a, b, "n = {n}");
        assert_eq!(hash_of(&a), hash_of(&b));
    }
}

#[test]
fn keys_one_node_apart_at_the_form_boundary_differ() {
    // Left-nested chains of n pairs have 2n + 1 nodes: 63, 65 and 67 nodes
    // straddle the largest flat key (64 nodes).
    let chain = |n: usize, last: i64| {
        let mut v = Value::Int(last);
        for _ in 0..n {
            v = Value::Pair(Rc::new((v, Value::Unit)));
        }
        MemoKey::from_value(&v).expect("valid key")
    };
    for n in [31, 32, 33] {
        assert_eq!(chain(n, 1).size(), 2 * n as u64 + 1);
        assert_eq!(chain(n, 1), chain(n, 1));
        assert_ne!(chain(n, 1), chain(n, 2));
        assert_ne!(chain(n, 1), chain(n + 1, 1));
    }
}

#[test]
fn a_value_holding_a_non_key_part_is_not_a_key_small_or_large() {
    // An option inside a pair, in a small key and below 2^41 - 1 shared nodes.
    let none = Value::Option(None);
    let small = Value::Pair(Rc::new((Value::Int(1), none.clone())));
    assert!(MemoKey::from_value(&small).is_none());
    let mut large = small;
    for _ in 0..40 {
        large = Value::Pair(Rc::new((large.clone(), large)));
    }
    assert!(MemoKey::from_value(&large).is_none());
}

#[test]
fn a_pair_and_its_swap_inside_one_key_are_different_parts() {
    // With A and B distinct doubling keys and P = (A, B), the key (P, (B, A))
    // must differ from (P, P), small or large: in the shared form, (B, A) must
    // not be taken for the node of (A, B).
    let doubling = |n: u32, base: i64| {
        let mut v = Value::Int(base);
        for _ in 0..n {
            v = Value::Pair(Rc::new((v.clone(), v)));
        }
        v
    };
    for n in [1u32, 2, 3, 4, 5] {
        let (a, b) = (doubling(n, 0), doubling(n, 1));
        let p = Value::Pair(Rc::new((a.clone(), b.clone())));
        let swapped = Value::Pair(Rc::new((p.clone(), Value::Pair(Rc::new((b, a))))));
        let same = Value::Pair(Rc::new((p.clone(), p)));
        let ks = MemoKey::from_value(&swapped).expect("valid key");
        let kp = MemoKey::from_value(&same).expect("valid key");
        assert_eq!(ks.size(), (1u64 << (n + 3)) - 1);
        assert_ne!(ks, kp, "n = {n}");
    }
}
