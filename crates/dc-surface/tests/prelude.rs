//! The prelude (spec §10, decision D7): results against Rust references, exact step
//! counts as closed formulas, type schemes, the off-switch, the Stitch view, and
//! the spec §10 sample programs.
//!
//! # How the step formulas are derived
//!
//! Every formula counts the steps of a test program `fun a b -> f a b` applied to
//! its inputs with `dc_eval::apply`, from the definition in `src/prelude.dc` and the
//! cost table in `dc-eval/src/cost.rs`:
//!
//! - β step (applying a closure to one argument): 1. The test program's own
//!   `fun` costs one β per input, and applying the prelude function one β per
//!   argument. A prelude reference expands to a closed `lam` (wrapped in free
//!   `let`s for the definitions it uses), and evaluating a `lam` or `let` is free.
//! - `nil`, `cons`, `pair`, `add`, `sub`, `le`, `lt`, `int_eq`, `mod`, `length`: 1
//!   each when saturated; a primitive passed as a value (`(::)`, `(+)`) costs
//!   nothing until it is applied.
//! - `if`, `&&`, `||`: 1 per decision.
//! - `list_case`: 1, plus the branch call: the empty branch is called with `unit`
//!   (1 β), the cons branch with head and tail (2 β).
//! - `foldl`, `foldr`, `loop`: 1 per element or iteration, plus the call of the
//!   function argument (2 β for a two-argument `fun`, 1 for a saturated primitive).
//! - `bfix n F x` and each call of its `self`: 1, plus `F self x` (2 β). The
//!   curried definitions (`take`, `drop`, `zip`) then apply the result to their
//!   last argument: 1 more β. Their fuel `length xs + 1` costs 2.
//!
//! The derivation of each formula is next to its test.

use std::sync::OnceLock;

use dc_core::types::infer;
use dc_core::{Arena, LibId, Scheme, Term, TermId};
use dc_eval::{apply, ErrorKind, Outcome, Value};
use dc_stitch::LibNames;
use dc_surface::prelude::{Prelude, PreludeMode, PRELUDE_LIB_BASE};
use dc_surface::TopLevel;
use proptest::prelude::*;

/// A generous step budget: every test program here takes well under it.
const FUEL: u64 = 50_000_000;

fn prelude() -> &'static Prelude {
    static P: OnceLock<Prelude> = OnceLock::new();
    P.get_or_init(|| Prelude::load().unwrap_or_else(|e| panic!("{e}")))
}

/// Compiles `src` against `top`, type-checks it against the prelude, expands the
/// prelude references and applies the result to `inputs`.
fn run_with(top: &TopLevel, src: &str, inputs: &[Value]) -> Outcome {
    let p = prelude();
    let mut arena = Arena::new();
    let t = top
        .close_expression(&mut arena, "test", src)
        .unwrap_or_else(|e| panic!("{src}: {e}"));
    infer(&arena, t, p).unwrap_or_else(|e| panic!("{src}: {e}"));
    let t = p.expand(&mut arena, t);
    apply(&arena, t, inputs, FUEL)
}

fn run(src: &str, inputs: &[Value]) -> Outcome {
    run_with(&prelude().top_level(PreludeMode::On), src, inputs)
}

#[track_caller]
fn value(src: &str, inputs: &[Value]) -> (Value, u64) {
    match run(src, inputs) {
        Outcome::Value(v, s) => (v, s),
        other => panic!("{src}: expected a value, got {other}"),
    }
}

#[track_caller]
fn error(src: &str, inputs: &[Value]) -> (ErrorKind, u64) {
    match run(src, inputs) {
        Outcome::Error(k, s) => (k, s),
        other => panic!("{src}: expected an error, got {other}"),
    }
}

fn ints(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

fn int(n: i64) -> Value {
    Value::Int(n)
}

fn len(xs: &[i64]) -> u64 {
    xs.len() as u64
}

fn small_list() -> impl Strategy<Value = Vec<i64>> {
    prop::collection::vec(-1000i64..1000, 0..40)
}

// ---- loading, schemes, names ----

#[test]
fn the_prelude_defines_exactly_the_spec_names_with_the_expected_schemes() {
    let expected = [
        ("min_int", "Int"),
        ("max_int", "Int"),
        ("max", "Int -> Int -> Int"),
        ("min", "Int -> Int -> Int"),
        ("map", "(a -> b) -> List a -> List b"),
        ("filter", "(a -> Bool) -> List a -> List a"),
        ("append", "List a -> List a -> List a"),
        ("reverse", "List a -> List a"),
        ("concat", "List (List a) -> List a"),
        ("sum", "List Int -> Int"),
        ("range", "Int -> Int -> List Int"),
        ("take", "Int -> List a -> List a"),
        ("drop", "Int -> List a -> List a"),
        ("split_at", "Int -> List a -> (List a, List a)"),
        ("zip", "List a -> List b -> List (a, b)"),
        ("nth", "List a -> Int -> a"),
        ("any", "(a -> Bool) -> List a -> Bool"),
        ("all", "(a -> Bool) -> List a -> Bool"),
    ];
    let p = prelude();
    let got: Vec<(&str, String)> = p
        .entries()
        .iter()
        .map(|e| (e.name(), e.scheme().to_string()))
        .collect();
    let want: Vec<(&str, String)> = expected
        .iter()
        .map(|(n, s)| (*n, s.parse::<Scheme>().unwrap().to_string()))
        .collect();
    assert_eq!(got, want);
    // `length` is a primitive, so the prelude does not define it.
    assert!(p.get("length").is_none());
}

#[test]
fn prelude_ids_are_a_reserved_block_and_name_their_entries() {
    let p = prelude();
    for (i, e) in p.entries().iter().enumerate() {
        assert_eq!(e.lib(), LibId(PRELUDE_LIB_BASE + i as u32));
        assert_eq!(p.name(e.lib()), Some(e.name()));
    }
    assert_eq!(p.name(LibId(0)), None);
    assert_eq!(p.name(LibId(PRELUDE_LIB_BASE + 1000)), None);
}

#[test]
fn a_program_refers_to_prelude_names_as_library_references() {
    let p = prelude();
    let mut arena = Arena::new();
    let t = p
        .top_level(PreludeMode::On)
        .close_expression(&mut arena, "test", "map")
        .unwrap();
    assert_eq!(arena.get(t), Term::Lib(p.get("map").unwrap().lib()));
    // Expansion replaces it with the closed definition: a `lam`.
    let x = p.expand(&mut arena, t);
    assert!(matches!(arena.get(x), Term::Lam(_)), "{x:?}");
    assert!(arena.is_closed(x));
}

#[test]
fn constants_expand_to_literals_and_cost_nothing() {
    assert_eq!(value("min_int", &[]), (int(i64::MIN), 0));
    assert_eq!(value("max_int", &[]), (int(i64::MAX), 0));
}

#[test]
fn a_type_error_in_a_prelude_source_is_reported() {
    let e = Prelude::from_source("bad", "let f x = x + true").unwrap_err();
    assert!(e.to_string().contains("'f'"), "{e}");
}

#[test]
fn expansion_leaves_other_library_references_alone() {
    let p = prelude();
    let mut arena = Arena::new();
    let l = arena.lib(LibId(0));
    let map = arena.lib(p.get("map").unwrap().lib());
    let t = arena.app(l, map);
    let x = p.expand(&mut arena, t);
    let Term::App(f, a) = arena.get(x) else {
        panic!("{x:?}")
    };
    assert_eq!(f, l);
    assert!(matches!(arena.get(a), Term::Lam(_)));
}

// ---- the off-switch ----

#[test]
fn prelude_off_compiles_with_only_the_primitives() {
    let p = prelude();
    let off = p.top_level(PreludeMode::Off);
    let mut arena = Arena::new();
    for name in p.entries().iter().map(|e| e.name()) {
        let err = off
            .close_expression(&mut arena, "test", name)
            .expect_err(name);
        assert_eq!(
            err.kind,
            dc_surface::ErrorKind::UnknownName,
            "{name}: {err}"
        );
    }
    // The combinators are still there, so the prelude's functions can be written
    // by hand; `map` written out costs what the prelude's `map` costs.
    let own = "let map f xs = foldr (fun x acc -> f x :: acc) [] xs";
    let mut top = p.top_level(PreludeMode::Off);
    top.add_source("program", own).unwrap();
    let src = "fun xs -> map (fun x -> x + 1) xs";
    let xs = ints(&[3, 1, 2]);
    let Outcome::Value(by_hand, by_hand_steps) = run_with(&top, src, std::slice::from_ref(&xs))
    else {
        panic!("map written by hand failed")
    };
    assert_eq!((by_hand, by_hand_steps), value(src, &[xs]));
    assert_eq!(PreludeMode::default(), PreludeMode::On);
}

#[test]
fn a_program_definition_shadows_a_prelude_name() {
    let mut top = prelude().top_level(PreludeMode::On);
    top.add_source("program", "let sum xs = 7").unwrap();
    match run_with(&top, "fun xs -> sum xs", &[ints(&[1, 2])]) {
        Outcome::Value(v, _) => assert_eq!(v, int(7)),
        other => panic!("{other}"),
    }
}

// ---- the Stitch view ----

fn stitch_names(p: &Prelude) -> LibNames {
    let mut names = LibNames::new();
    for e in p.entries() {
        names.insert(e.lib(), e.name()).unwrap();
    }
    names
}

#[test]
fn stitch_sees_prelude_names_as_opaque_tokens() {
    let p = prelude();
    let names = stitch_names(p);
    let mut arena = Arena::new();
    let src = "fun xs -> sum (map (fun x -> x * x) (filter (fun x -> x mod 2 = 0) xs))";
    let t = p
        .top_level(PreludeMode::On)
        .close_expression(&mut arena, "test", src)
        .unwrap();
    let text = dc_stitch::encode(&arena, t, &names).unwrap();
    assert_eq!(
        text,
        "(lam (sum (map (lam (mul $0 $0)) (filter (lam (int_eq (mod $0 int_2) int_0)) $0))))"
    );
    let back: TermId = dc_stitch::decode(&mut arena, &text, &names).unwrap();
    assert_eq!(back, t);
}

// ---- functions: results against Rust, exact steps ----

#[test]
fn max_and_min() {
    // 2 (program β) + 2 (β into max) + 1 (le) + 1 (if) = 6.
    for (a, b) in [(1, 2), (2, 1), (5, 5), (i64::MIN, i64::MAX)] {
        assert_eq!(
            value("fun a b -> max a b", &[int(a), int(b)]),
            (int(a.max(b)), 6)
        );
        assert_eq!(
            value("fun a b -> min a b", &[int(a), int(b)]),
            (int(a.min(b)), 6)
        );
    }
}

/// map with f = (fun x -> x + 1), which costs 2 per call (1 β + 1 add):
/// 1 (program β) + 2 (β into map) + 1 (nil) + per element [1 (foldr) + 2 (β into
/// the step function) + 2 (f) + 1 (cons)] = 4 + 6n.
fn map_steps(n: u64) -> u64 {
    4 + 6 * n
}

/// filter with p = (fun x -> x mod 2 = 0), which costs 3 per call (β, mod, int_eq):
/// 1 + 2 + 1 (nil) + per element [1 (foldr) + 2 (β) + 3 (p) + 1 (if)], + 1 (cons)
/// per kept element = 4 + 7n + kept.
fn filter_steps(n: u64, kept: u64) -> u64 {
    4 + 7 * n + kept
}

/// append xs ys = foldr (::) ys xs: 2 (program β) + 2 (β) + per element of xs
/// [1 (foldr) + 1 (cons)] = 4 + 2·|xs|.
fn append_steps(n: u64) -> u64 {
    4 + 2 * n
}

/// reverse: 1 + 1 (β) + 1 (nil) + per element [1 (foldl) + 2 (β) + 1 (cons)] = 3 + 4n.
fn reverse_steps(n: u64) -> u64 {
    3 + 4 * n
}

/// concat xss = foldr append [] xss: 1 + 1 (β) + 1 (nil) + per inner list of
/// length m [1 (foldr) + 2 (β into append) + 2m (append's foldr)] = 3 + 3k + 2M,
/// with k inner lists of M elements in all.
fn concat_steps(k: u64, total: u64) -> u64 {
    3 + 3 * k + 2 * total
}

/// sum = foldl (+) 0: 1 + 1 (β) + per element [1 (foldl) + 1 (add)] = 2 + 2n.
fn sum_steps(n: u64) -> u64 {
    2 + 2 * n
}

/// range lo hi: 2 (program β) + 2 (β) + 1 (sub for top) + 1 (sub for
/// hi - lo) + 1 (nil) + per element [1 (loop) + 2 (β) + 1 (sub) + 1 (cons)] = 7 + 5n,
/// n = max(hi - lo, 0).
fn range_steps(n: u64) -> u64 {
    7 + 5 * n
}

/// take k xs, n = |xs|. Setup: 2 (program β) + 2 (β into take) + 2 (fuel: length,
/// add) + 4 (first call: bfix 1 + F self k 2 β + applying to ys 1 β) = 10.
/// Each element taken: le 1 + if 1 + list_case 1 + 2 β + sub 1 + cons 1, plus the
/// next call's 4 = 11. The last call: k ≤ 0 gives le + if + nil = 3; an
/// exhausted list gives le + if + list_case + 1 β + nil = 5.
fn take_steps(k: i64, n: u64) -> u64 {
    if k < 0 {
        13
    } else if (k as u64) <= n {
        13 + 11 * k as u64
    } else {
        15 + 11 * n
    }
}

/// drop k xs: as take, but each element dropped costs le + if + list_case +
/// 2 β + sub + the next call's 4 = 10, and a last call with k ≤ 0 costs le + if = 2.
fn drop_steps(k: i64, n: u64) -> u64 {
    if k < 0 {
        12
    } else if (k as u64) <= n {
        12 + 10 * k as u64
    } else {
        15 + 10 * n
    }
}

/// split_at k xs = (take k xs, drop k xs): 2 (program β) + 2 (β) + take (its
/// formula less its 2 program β) + drop (likewise) + 1 (pair).
fn split_at_steps(k: i64, n: u64) -> u64 {
    2 + 2 + (take_steps(k, n) - 2) + (drop_steps(k, n) - 2) + 1
}

/// zip xs ys, m = min(|xs|, |ys|). Setup 2 + 2 + 2 (fuel) + 4 (first call) = 10.
/// Each pair: list_case 1 + 2 β + list_case 1 + 2 β + pair 1 + cons 1, plus the
/// next call's 4 = 12. Last call: xs empty gives list_case + 1 β + nil = 3; ys
/// empty first gives list_case + 2 β + list_case + 1 β + nil = 6.
fn zip_steps(na: u64, nb: u64) -> u64 {
    if na <= nb {
        13 + 12 * na
    } else {
        16 + 12 * nb
    }
}

/// any p xs with p = (fun x -> x < 0), 2 per call (β, lt). Setup: 1 (program
/// β) + 2 (β into any) + 2 (fuel) + 3 (first call: bfix 1 + F self ys 2 β) = 8.
/// Each element that does not decide: list_case 1 + 2 β + p 2 + `||` 1, plus the
/// next call's 3 = 9. The deciding element: 1 + 2 + 2 + 1 = 6. The end of the list:
/// list_case 1 + 1 β = 2. So 14 + 9i if element i decides, else 10 + 9n.
/// all is the same with `&&`.
fn any_steps(decides_at: Option<u64>, n: u64) -> u64 {
    match decides_at {
        Some(i) => 14 + 9 * i,
        None => 10 + 9 * n,
    }
}

/// nth xs i. 2 (program β) + 2 (β) + lt 1 + if 1 = 6, then:
/// - i < 0: nil 1 + array_of_list 1 + array_get 1 (charged, then fails) = 9 in all;
/// - 0 ≤ i: drop i xs (drop_steps less its 2 program β) + list_case 1, then the
///   cons branch (2 β) for i < n, or the empty branch (1 β) + nil + array_of_list
///   + array_get = 4 for i ≥ n.
fn nth_steps(i: i64, n: u64) -> u64 {
    if i < 0 {
        9
    } else if (i as u64) < n {
        6 + (drop_steps(i, n) - 2) + 1 + 2
    } else {
        6 + (drop_steps(i, n) - 2) + 1 + 4
    }
}

#[test]
fn cost_formulas_at_several_sizes() {
    for n in [0u64, 1, 2, 5, 16, 100] {
        let xs: Vec<i64> = (0..n as i64).collect();
        let l = ints(&xs);
        let s = |src: &str, inputs: &[Value]| value(src, inputs).1;
        assert_eq!(
            s(
                "fun xs -> map (fun x -> x + 1) xs",
                std::slice::from_ref(&l)
            ),
            map_steps(n)
        );
        // Half of 0..n is even, rounding up.
        let kept = n.div_ceil(2);
        assert_eq!(
            s(
                "fun xs -> filter (fun x -> x mod 2 = 0) xs",
                std::slice::from_ref(&l)
            ),
            filter_steps(n, kept)
        );
        assert_eq!(
            s("fun xs ys -> append xs ys", &[l.clone(), l.clone()]),
            append_steps(n)
        );
        assert_eq!(
            s("fun xs -> reverse xs", std::slice::from_ref(&l)),
            reverse_steps(n)
        );
        assert_eq!(
            s("fun xs -> sum xs", std::slice::from_ref(&l)),
            sum_steps(n)
        );
        assert_eq!(
            s("fun a b -> range a b", &[int(-3), int(n as i64 - 3)]),
            range_steps(n)
        );
        for k in [-1, 0, 1, n as i64 / 2, n as i64, n as i64 + 2] {
            assert_eq!(
                s("fun k xs -> take k xs", &[int(k), l.clone()]),
                take_steps(k, n)
            );
            assert_eq!(
                s("fun k xs -> drop k xs", &[int(k), l.clone()]),
                drop_steps(k, n)
            );
            assert_eq!(
                s("fun k xs -> split_at k xs", &[int(k), l.clone()]),
                split_at_steps(k, n)
            );
        }
        assert_eq!(
            s("fun a b -> zip a b", &[l.clone(), l.clone()]),
            zip_steps(n, n)
        );
        assert_eq!(
            s(
                "fun xs -> any (fun x -> x < 0) xs",
                std::slice::from_ref(&l)
            ),
            any_steps(None, n)
        );
        assert_eq!(
            s(
                "fun xs -> all (fun x -> 0 <= x) xs",
                std::slice::from_ref(&l)
            ),
            any_steps(None, n)
        );
        let xss = Value::list((0..n).map(|i| ints(&xs[..i as usize])));
        assert_eq!(
            s("fun xss -> concat xss", &[xss]),
            concat_steps(n, n * n.saturating_sub(1) / 2)
        );
        if n > 0 {
            assert_eq!(
                s("fun xs i -> nth xs i", &[l.clone(), int(n as i64 - 1)]),
                nth_steps(n as i64 - 1, n)
            );
        }
    }
}

#[test]
fn growth_is_linear_where_it_should_be() {
    // Doubling n doubles the per-element part of each count: the formulas are
    // affine in n with the slopes above, so reverse and append are linear and
    // take is linear in its count, not in the list length.
    let long: Vec<i64> = (0..1000).collect();
    let l = ints(&long);
    let t = |k: i64| value("fun k xs -> take k xs", &[int(k), l.clone()]).1;
    assert_eq!(t(20) - t(10), 110);
    assert_eq!(t(10), take_steps(10, 1000));
}

#[test]
fn nth_out_of_range_is_an_index_error() {
    let l = ints(&[4, 5, 6]);
    for i in [-2i64, 3, 4, 10] {
        assert_eq!(
            error("fun xs i -> nth xs i", &[l.clone(), int(i)]),
            (ErrorKind::IndexOutOfBounds, nth_steps(i, 3)),
            "i = {i}"
        );
    }
}

fn rust_take(k: i64, xs: &[i64]) -> Vec<i64> {
    xs.iter().take(k.max(0) as usize).copied().collect()
}

fn rust_drop(k: i64, xs: &[i64]) -> Vec<i64> {
    xs.iter().skip(k.max(0) as usize).copied().collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn map_matches_rust(xs in small_list()) {
        let want: Vec<i64> = xs.iter().map(|x| x + 1).collect();
        prop_assert_eq!(
            value("fun xs -> map (fun x -> x + 1) xs", &[ints(&xs)]),
            (ints(&want), map_steps(len(&xs)))
        );
    }

    #[test]
    fn filter_matches_rust(xs in small_list()) {
        let want: Vec<i64> = xs.iter().copied().filter(|x| x % 2 == 0).collect();
        prop_assert_eq!(
            value("fun xs -> filter (fun x -> x mod 2 = 0) xs", &[ints(&xs)]),
            (ints(&want), filter_steps(len(&xs), len(&want)))
        );
    }

    #[test]
    fn append_and_reverse_match_rust(xs in small_list(), ys in small_list()) {
        let both: Vec<i64> = xs.iter().chain(&ys).copied().collect();
        prop_assert_eq!(
            value("fun xs ys -> append xs ys", &[ints(&xs), ints(&ys)]),
            (ints(&both), append_steps(len(&xs)))
        );
        let rev: Vec<i64> = xs.iter().rev().copied().collect();
        prop_assert_eq!(
            value("fun xs -> reverse xs", &[ints(&xs)]),
            (ints(&rev), reverse_steps(len(&xs)))
        );
    }

    #[test]
    fn concat_matches_rust(xss in prop::collection::vec(small_list(), 0..8)) {
        let want: Vec<i64> = xss.concat();
        let input = Value::list(xss.iter().map(|xs| ints(xs)));
        prop_assert_eq!(
            value("fun xss -> concat xss", &[input]),
            (ints(&want), concat_steps(xss.len() as u64, len(&want)))
        );
    }

    #[test]
    fn sum_matches_rust(xs in small_list()) {
        prop_assert_eq!(
            value("fun xs -> sum xs", &[ints(&xs)]),
            (int(xs.iter().sum()), sum_steps(len(&xs)))
        );
    }

    #[test]
    fn range_matches_rust(lo in -50i64..50, hi in -50i64..50) {
        let want: Vec<i64> = (lo..hi).collect();
        prop_assert_eq!(
            value("fun a b -> range a b", &[int(lo), int(hi)]),
            (ints(&want), range_steps(len(&want)))
        );
    }

    #[test]
    fn take_drop_split_at_match_rust(xs in small_list(), k in -5i64..50) {
        let n = len(&xs);
        let (t, d) = (rust_take(k, &xs), rust_drop(k, &xs));
        prop_assert_eq!(
            value("fun k xs -> take k xs", &[int(k), ints(&xs)]),
            (ints(&t), take_steps(k, n))
        );
        prop_assert_eq!(
            value("fun k xs -> drop k xs", &[int(k), ints(&xs)]),
            (ints(&d), drop_steps(k, n))
        );
        prop_assert_eq!(
            value("fun k xs -> split_at k xs", &[int(k), ints(&xs)]),
            (Value::pair(ints(&t), ints(&d)), split_at_steps(k, n))
        );
    }

    #[test]
    fn zip_matches_rust(xs in small_list(), ys in small_list()) {
        let want = Value::list(xs.iter().zip(&ys).map(|(&x, &y)| Value::pair(int(x), int(y))));
        prop_assert_eq!(
            value("fun a b -> zip a b", &[ints(&xs), ints(&ys)]),
            (want, zip_steps(len(&xs), len(&ys)))
        );
    }

    #[test]
    fn any_and_all_match_rust(xs in small_list()) {
        let first_neg = xs.iter().position(|&x| x < 0).map(|i| i as u64);
        prop_assert_eq!(
            value("fun xs -> any (fun x -> x < 0) xs", &[ints(&xs)]),
            (Value::Bool(first_neg.is_some()), any_steps(first_neg, len(&xs)))
        );
        // all (0 <= x) is decided by the same element, as false.
        prop_assert_eq!(
            value("fun xs -> all (fun x -> 0 <= x) xs", &[ints(&xs)]),
            (Value::Bool(first_neg.is_none()), any_steps(first_neg, len(&xs)))
        );
    }

    #[test]
    fn nth_matches_rust(xs in small_list(), i in -3i64..45) {
        let src = "fun xs i -> nth xs i";
        let steps = nth_steps(i, len(&xs));
        match usize::try_from(i).ok().and_then(|i| xs.get(i)) {
            Some(&x) => prop_assert_eq!(value(src, &[ints(&xs), int(i)]), (int(x), steps)),
            None => prop_assert_eq!(
                error(src, &[ints(&xs), int(i)]),
                (ErrorKind::IndexOutOfBounds, steps)
            ),
        }
    }

    #[test]
    fn max_and_min_match_rust(a in any::<i64>(), b in any::<i64>()) {
        prop_assert_eq!(value("fun a b -> max a b", &[int(a), int(b)]), (int(a.max(b)), 6));
        prop_assert_eq!(value("fun a b -> min a b", &[int(a), int(b)]), (int(a.min(b)), 6));
    }
}

// ---- the spec §10 sample programs ----

/// The sample programs of spec §10, verbatim. They are also the throughput
/// benchmark programs (spec §11, bead P2.14).
const SPEC_SAMPLES: &str = r"
let sum xs = foldl (fun acc x -> acc + x) 0 xs

let max_subarray xs =
  fst (foldl (fun (best, cur) x ->
         let cur2 = max x (cur + x) in (max best cur2, cur2))
       (min_int, 0) xs)

let merge ab =
  bfix (length (fst ab) + length (snd ab) + 1)
    (fun self (a, b) ->
      match a with
      | [] -> b
      | x :: xs ->
        match b with
        | [] -> a
        | y :: ys -> if x <= y then x :: self (xs, b) else y :: self (a, ys))
    ab

let msort xs =
  bfix (length xs + 1)
    (fun self xs ->
      if length xs <= 1 then xs
      else
        let (l, r) = split_at (length xs / 2) xs in
        merge (self l, self r))
    xs

let fib n =
  memorec (n + 1)
    (fun self k -> if k < 2 then k else self (k - 1) + self (k - 2))
    n
";

fn samples() -> TopLevel {
    let mut top = prelude().top_level(PreludeMode::On);
    top.add_source("spec-samples", SPEC_SAMPLES)
        .unwrap_or_else(|e| panic!("{e}"));
    top
}

fn sample(top: &TopLevel, entry: &str, inputs: &[Value]) -> Value {
    let p = prelude();
    let mut arena = Arena::new();
    let t = top.close_definition(&mut arena, entry).unwrap();
    infer(&arena, t, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let t = p.expand(&mut arena, t);
    match apply(&arena, t, inputs, FUEL) {
        Outcome::Value(v, _) => v,
        other => panic!("{entry}: {other}"),
    }
}

#[test]
fn spec_samples_have_the_expected_types() {
    let p = prelude();
    let top = samples();
    for (entry, ty) in [
        ("sum", "List Int -> Int"),
        ("max_subarray", "List Int -> Int"),
        ("merge", "(List Int, List Int) -> List Int"),
        ("msort", "List Int -> List Int"),
        ("fib", "Int -> Int"),
    ] {
        let mut arena = Arena::new();
        let t = top.close_definition(&mut arena, entry).unwrap();
        assert_eq!(
            infer(&arena, t, p).unwrap(),
            ty.parse::<Scheme>().unwrap(),
            "{entry}"
        );
    }
}

#[test]
fn spec_sample_fib_by_memorec() {
    let top = samples();
    let mut fib = vec![0i64, 1];
    for i in 2..=60 {
        fib.push(fib[i - 1] + fib[i - 2]);
    }
    for n in [0, 1, 2, 10, 20, 60] {
        assert_eq!(
            sample(&top, "fib", &[int(n)]),
            int(fib[n as usize]),
            "fib {n}"
        );
    }
}

fn rust_max_subarray(xs: &[i64]) -> i64 {
    let (mut best, mut cur) = (i64::MIN, 0i64);
    for &x in xs {
        cur = x.max(cur + x);
        best = best.max(cur);
    }
    best
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn spec_samples_match_rust(xs in small_list(), ys in small_list()) {
        let top = samples();
        prop_assert_eq!(sample(&top, "sum", &[ints(&xs)]), int(xs.iter().sum()));
        prop_assert_eq!(sample(&top, "max_subarray", &[ints(&xs)]), int(rust_max_subarray(&xs)));
        let mut sorted = xs.clone();
        sorted.sort();
        prop_assert_eq!(sample(&top, "msort", &[ints(&xs)]), ints(&sorted));
        let (mut a, mut b) = (xs.clone(), ys.clone());
        a.sort();
        b.sort();
        let mut merged: Vec<i64> = a.iter().chain(&b).copied().collect();
        merged.sort();
        prop_assert_eq!(
            sample(&top, "merge", &[Value::pair(ints(&a), ints(&b))]),
            ints(&merged)
        );
    }
}
