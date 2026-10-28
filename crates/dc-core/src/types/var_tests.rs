//! Collecting and numbering type variables: the work grows linearly with the number
//! of variables, and the results match a plain list-based reference.
//!
//! The work is the number of equality comparisons on type variables and on parsed
//! variable names, which [`probes`] counts (see there for what it does not count).
//! A lookup by scanning a list makes one comparison per element passed, so a scan in
//! place of a hash map makes these counts grow quadratically.

use proptest::prelude::*;

use super::*;

/// A balanced tree of pairs over `n` distinct variables, numbered out of order so
/// that normalising renames every one of them.
fn balanced_pairs(n: u32) -> Type {
    fn build(lo: u32, hi: u32, n: u32) -> Type {
        if hi - lo == 1 {
            Type::var(3 * (n - lo))
        } else {
            let mid = lo + (hi - lo) / 2;
            Type::pair(build(lo, mid, n), build(mid, hi, n))
        }
    }
    build(0, n, n)
}

/// `Key` text constraining every variable of `t`, followed by `t`.
fn all_keyed_text(t: &Type) -> String {
    let keys: Vec<String> = t.vars().iter().map(|v| format!("Key {v}")).collect();
    format!("{} => {t}", keys.join(", "))
}

/// The variable comparisons made by `f`.
fn probes_of(f: impl FnOnce()) -> u64 {
    probes::take();
    f();
    probes::take()
}

/// Asserts that `work(4n)` is at most five times `work(n)`: linear work gives four,
/// work quadratic in `n` gives about sixteen.
fn assert_linear(what: &str, n: u32, work: impl Fn(u32) -> u64) {
    let small = work(n);
    let large = work(4 * n);
    assert!(small > 0, "{what}: no comparisons counted at {n} variables");
    assert!(
        large <= 5 * small,
        "{what}: {small} comparisons at {n} variables but {large} at {}; the work is not linear",
        4 * n
    );
}

#[test]
fn building_a_scheme_takes_linear_work_in_its_variables() {
    assert_linear("Scheme::with_keys", 1024, |n| {
        let t = balanced_pairs(n);
        let keys = t.vars();
        probes_of(|| {
            let s = Scheme::with_keys(t, &keys);
            assert_eq!(s.keys().len(), n as usize);
        })
    });
}

#[test]
fn listing_variables_takes_linear_work_with_repeats() {
    assert_linear("Type::vars", 1024, |n| {
        let t = balanced_pairs(n);
        let t = Type::arrow(t.clone(), t);
        probes_of(|| assert_eq!(t.vars().len(), n as usize))
    });
}

#[test]
fn parsing_a_type_takes_linear_work_in_its_variables() {
    assert_linear("Type::from_str", 1024, |n| {
        let text = balanced_pairs(n).to_string();
        probes_of(|| {
            let t: Type = text.parse().expect("printed types parse");
            assert_eq!(t.vars().len(), n as usize);
        })
    });
}

#[test]
fn parsing_a_keyed_scheme_takes_linear_work_in_its_variables() {
    assert_linear("Scheme::from_str", 1024, |n| {
        let text = all_keyed_text(&balanced_pairs(n));
        probes_of(|| {
            let s: Scheme = text.parse().expect("printed schemes parse");
            assert_eq!(s.keys().len(), n as usize);
        })
    });
}

#[test]
fn checking_against_a_keyed_scheme_takes_linear_work_in_its_variables() {
    assert_linear("check with a keyed library scheme", 1024, |n| {
        let s = Scheme::with_keys(balanced_pairs(n), &balanced_pairs(n).vars());
        let libs = [s.clone()];
        let mut arena = crate::term::Arena::new();
        let term = crate::sexpr::parse(&mut arena, "lib_0").expect("a library use parses");
        probes_of(|| {
            check(&arena, term, &libs[..], &s).expect("a library use has its own scheme");
        })
    });
}

// ---------------------------------------------------------------- differential

/// The variables of `t` in order of first occurrence, by linear search.
fn ref_vars(t: &Type) -> Vec<TyVar> {
    fn go(t: &Type, out: &mut Vec<TyVar>) {
        match t {
            Type::Int | Type::Bool | Type::Unit => {}
            Type::Var(v) => {
                if !out.contains(v) {
                    out.push(*v);
                }
            }
            Type::List(a) | Type::Option(a) | Type::Array(a) | Type::Tree(a) => go(a, out),
            Type::Pair(a, b) | Type::Arrow(a, b) => {
                go(a, out);
                go(b, out);
            }
        }
    }
    let mut out = vec![];
    go(t, &mut out);
    out
}

/// `t` renumbered in order of first occurrence and the sorted, deduplicated keys
/// that occur in it, by linear search.
fn ref_with_keys(t: &Type, keys: &[TyVar]) -> (Type, Vec<TyVar>) {
    let order = ref_vars(t);
    let rename = |v: TyVar| {
        let i = order.iter().position(|&w| w == v).expect("v occurs in t");
        TyVar(u32::try_from(i).expect("few variables"))
    };
    let ty = t.map_vars(&mut |v| rename(v));
    let mut keys: Vec<TyVar> = keys
        .iter()
        .filter(|v| order.contains(v))
        .map(|&v| rename(v))
        .collect();
    keys.sort();
    keys.dedup();
    (ty, keys)
}

/// Types over a few dozen variables, so that variables repeat.
fn arb_type() -> impl Strategy<Value = Type> {
    let leaf = prop_oneof![
        Just(Type::Int),
        Just(Type::Bool),
        Just(Type::Unit),
        (0u32..60).prop_map(Type::var),
    ];
    leaf.prop_recursive(6, 80, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Type::pair(a, b)),
            (inner.clone(), inner.clone()).prop_map(|(a, b)| Type::arrow(a, b)),
            inner.clone().prop_map(Type::list),
            inner.clone().prop_map(Type::option),
            inner.clone().prop_map(Type::array),
            inner.prop_map(Type::tree),
        ]
    })
}

/// Keys, some of them absent from the type and some repeated.
fn arb_keys() -> impl Strategy<Value = Vec<TyVar>> {
    proptest::collection::vec((0u32..70).prop_map(TyVar), 0..12)
}

proptest! {
    #[test]
    fn vars_match_the_reference(t in arb_type()) {
        prop_assert_eq!(t.vars(), ref_vars(&t));
    }

    #[test]
    fn with_keys_matches_the_reference(t in arb_type(), keys in arb_keys()) {
        let (ty, ks) = ref_with_keys(&t, &keys);
        let s = Scheme::with_keys(t, &keys);
        prop_assert_eq!(s.ty(), &ty);
        prop_assert_eq!(s.keys(), &ks[..]);
    }

    #[test]
    fn parsed_types_match_the_reference(t in arb_type()) {
        let text = t.to_string();
        let back: Type = text.parse().map_err(|e: TypeParseError| TestCaseError::fail(format!("{text}: {e}")))?;
        prop_assert_eq!(back, ref_with_keys(&t, &[]).0);
    }

    #[test]
    fn parsed_schemes_match_the_reference(t in arb_type(), keys in arb_keys()) {
        let context: Vec<String> = keys.iter().map(|k| format!("Key {k}")).collect();
        let text = if keys.is_empty() { t.to_string() } else { format!("{} => {t}", context.join(", ")) };
        let back: Scheme = text.parse().map_err(|e: TypeParseError| TestCaseError::fail(format!("{text}: {e}")))?;
        let (ty, ks) = ref_with_keys(&t, &keys);
        prop_assert_eq!(back.ty(), &ty);
        prop_assert_eq!(back.keys(), &ks[..]);
    }
}
