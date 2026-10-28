use std::collections::HashMap;

use proptest::prelude::*;

use super::*;
use crate::prim::Prim;
use crate::sexpr;
use crate::term::{Arena, LibId, TermId};

const NO_LIBS: &[Scheme] = &[];

fn sch(s: &str) -> Scheme {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn ty(s: &str) -> Type {
    s.parse().unwrap_or_else(|e| panic!("{s}: {e}"))
}

fn parse_term(arena: &mut Arena, src: &str) -> TermId {
    sexpr::parse(arena, src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn infer_with(src: &str, libs: &[Scheme]) -> Result<Scheme, TypeError> {
    let mut a = Arena::new();
    let id = parse_term(&mut a, src);
    infer(&a, id, libs)
}

fn infer_src(src: &str) -> Result<Scheme, TypeError> {
    infer_with(src, NO_LIBS)
}

fn assert_infers(src: &str, expected: &str) {
    match infer_src(src) {
        Ok(s) => assert_eq!(s, sch(expected), "{src}: got `{s}`, want `{expected}`"),
        Err(e) => panic!("{src}: expected `{expected}`, got error\n{e}"),
    }
}

fn expect_err(src: &str) -> TypeError {
    match infer_src(src) {
        Ok(s) => panic!("{src}: expected a type error, got `{s}`"),
        Err(e) => e,
    }
}

fn check_src(src: &str, expected: &str) -> Result<(), TypeError> {
    let mut a = Arena::new();
    let id = parse_term(&mut a, src);
    check(&a, id, NO_LIBS, &sch(expected))
}

// ---------------------------------------------------------------- Display / parse

#[test]
fn types_display_readably() {
    let a = Type::var(0);
    let t = Type::fun(
        [Type::Int, Type::list(a.clone())],
        Type::pair(a.clone(), Type::Int),
    );
    assert_eq!(t.to_string(), "Int -> List a -> (a, Int)");
    let hof = Type::arrow(
        Type::arrow(a.clone(), Type::var(1)),
        Type::list(Type::list(a.clone())),
    );
    assert_eq!(hof.to_string(), "(a -> b) -> List (List a)");
    assert_eq!(
        Type::list(Type::arrow(Type::Int, Type::Bool)).to_string(),
        "List (Int -> Bool)"
    );
    assert_eq!(
        Type::tree(Type::option(Type::Unit)).to_string(),
        "Tree (Option Unit)"
    );
    let triple = Type::pair(Type::Int, Type::pair(Type::Bool, Type::Unit));
    assert_eq!(triple.to_string(), "(Int, Bool, Unit)");
    let left_nested = Type::pair(Type::pair(Type::Int, Type::Bool), Type::Unit);
    assert_eq!(left_nested.to_string(), "((Int, Bool), Unit)");
    assert_eq!(
        Type::pair(Type::arrow(a.clone(), a), Type::Int).to_string(),
        "(a -> a, Int)"
    );
    assert_eq!(TyVar(25).to_string(), "z");
    assert_eq!(TyVar(26).to_string(), "a1");
    assert_eq!(TyVar(53).to_string(), "b2");
}

#[test]
fn parse_handles_precedence_and_tuples() {
    assert_eq!(
        ty("(a -> b) -> List a -> List b"),
        Type::fun(
            [
                Type::arrow(Type::var(0), Type::var(1)),
                Type::list(Type::var(0))
            ],
            Type::list(Type::var(1))
        )
    );
    assert_eq!(
        ty("(Int, Bool, Unit)"),
        Type::pair(Type::Int, Type::pair(Type::Bool, Type::Unit))
    );
    assert_eq!(ty("a -> b -> c"), ty("a -> (b -> c)"));
    assert_ne!(ty("a -> b -> c"), ty("(a -> b) -> c"));
    for bad in [
        "",
        "List",
        "Int ->",
        "(Int",
        "Int Int",
        "Foo",
        "Key a",
        "Key Int => a",
        "a $",
    ] {
        assert!(bad.parse::<Scheme>().is_err(), "{bad:?} parsed");
    }
}

/// Runs `f` on a thread with a 256 KiB stack, so host recursion over deep type text
/// overflows (and aborts the test binary) instead of passing by luck.
fn on_small_stack<F: FnOnce() + Send + 'static>(f: F) {
    on_stack(256 << 10, f);
}

/// Runs `f` on a thread with a stack of `bytes`.
fn on_stack<F: FnOnce() + Send + 'static>(bytes: usize, f: F) {
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(f)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

/// `n` copies of `prefix(`, then `inner`, then `n` closing brackets.
fn wrapped(prefix: &str, n: usize, inner: &str) -> String {
    format!("{}{inner}{}", format!("{prefix}(").repeat(n), ")".repeat(n))
}

/// `(Int, Int, ..., Int)` with `n` items: a pair chain `n` levels deep.
fn tuple_of(n: usize) -> String {
    format!("({})", vec!["Int"; n].join(", "))
}

/// Asserts that `src` is refused as too deep, both as a type and as a scheme.
fn assert_too_deep(src: &str) {
    let e = src.parse::<Type>().expect_err("deep type text parsed");
    assert!(e.message.contains("deeper than"), "{e}");
    let e = src.parse::<Scheme>().expect_err("deep scheme text parsed");
    assert!(e.message.contains("deeper than"), "{e}");
}

#[test]
fn deeply_bracketed_type_text_is_refused() {
    // Brackets with no constructor add no depth to the type, only to the text.
    on_small_stack(|| assert_too_deep(&wrapped("", 200_000, "Int")));
}

#[test]
fn deeply_nested_constructor_text_is_refused() {
    on_small_stack(|| assert_too_deep(&wrapped("List ", 100_000, "Int")));
}

#[test]
fn long_arrow_chains_are_refused() {
    // `->` nests to the right with no brackets, so each arrow is one level of type.
    on_small_stack(|| assert_too_deep(&format!("{}Int", "Int -> ".repeat(200_000))));
}

#[test]
fn long_tuples_are_refused() {
    // A flat tuple has one bracket but nests to the right as pairs, so the type would
    // be too deep to drop, print or check on a small stack.
    on_small_stack(|| assert_too_deep(&tuple_of(100_000)));
}

#[test]
fn deep_scheme_text_is_refused() {
    on_small_stack(|| {
        for body in [
            wrapped("", 200_000, "a"),
            wrapped("Option ", 100_000, "a"),
            format!("{}a", "a -> ".repeat(200_000)),
            tuple_of(100_000).replace("Int", "a"),
        ] {
            let e = format!("Key a => {body}")
                .parse::<Scheme>()
                .expect_err("deep scheme text parsed");
            assert!(e.message.contains("deeper than"), "{e}");
        }
    });
}

#[test]
fn type_text_at_the_depth_limit_parses() {
    // The parser needs little stack at any depth, but printing a type at the limit
    // recurses `MAX_TYPE_DEPTH` levels: about 400 KiB in a debug build.
    on_stack(1 << 20, || {
        let d = MAX_TYPE_DEPTH;
        // Each form at exactly `MAX_TYPE_DEPTH` levels, then one level deeper. For
        // bare brackets the levels are the text's: the root plus one per bracket.
        type Form = fn(usize) -> String;
        let forms: [(Form, usize); 4] = [
            (|n| wrapped("List ", n, "Int"), d - 1),
            (|n| format!("{}Int", "Int -> ".repeat(n)), d - 1),
            (tuple_of, d),
            (|n| wrapped("", n, "Int"), d - 1),
        ];
        for (form, n) in forms {
            let src = form(n);
            let t: Type = src.parse().unwrap_or_else(|e| panic!("at the limit: {e}"));
            // An at-limit type prints as text that parses back to it.
            assert_eq!(t.to_string().parse::<Type>(), Ok(t.clone()));
            assert_eq!(src.parse::<Scheme>().map(|s| s.ty().clone()), Ok(t));
            assert_too_deep(&form(n + 1));
        }
    });
}

#[test]
fn schemes_compare_up_to_renaming() {
    assert_eq!(sch("x -> y -> x"), sch("a -> b -> a"));
    assert_ne!(sch("a -> b -> a"), sch("a -> b -> b"));
    assert_eq!(sch("Key k => k -> v"), sch("Key a => a -> b"));
    assert_ne!(sch("Key k => k -> v"), sch("k -> v"));
    assert_eq!(
        sch("Key b, Key a => (a, b)").to_string(),
        "Key a, Key b => (a, b)"
    );
}

fn arb_type() -> impl Strategy<Value = Type> {
    let leaf = prop_oneof![
        Just(Type::Int),
        Just(Type::Bool),
        Just(Type::Unit),
        (0u32..30).prop_map(Type::var),
    ];
    leaf.prop_recursive(5, 40, 2, |inner| {
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

proptest! {
    #[test]
    fn scheme_display_parse_round_trips(t in arb_type(), key_mask in any::<u32>()) {
        let keys: Vec<TyVar> = t.vars().into_iter().filter(|v| key_mask & (1 << (v.0 % 32)) != 0).collect();
        let s = Scheme::with_keys(t, &keys);
        let text = s.to_string();
        let back: Scheme = text.parse().map_err(|e: TypeParseError| TestCaseError::fail(format!("{text}: {e}")))?;
        prop_assert_eq!(back, s);
    }

    #[test]
    fn type_display_parse_round_trips_up_to_renaming(t in arb_type()) {
        let text = t.to_string();
        let back: Type = text.parse().map_err(|e: TypeParseError| TestCaseError::fail(format!("{text}: {e}")))?;
        prop_assert_eq!(Scheme::new(back), Scheme::new(t));
    }
}

// ---------------------------------------------------------------- primitive schemes

/// Expected schemes, transcribed from the spec tables (§5, §6), independently of the
/// implementation.
fn spec_scheme(p: Prim) -> &'static str {
    match p {
        Prim::Add | Prim::Sub | Prim::Mul | Prim::Div | Prim::Mod => "Int -> Int -> Int",
        Prim::IntEq | Prim::Lt | Prim::Le => "Int -> Int -> Bool",
        Prim::Not => "Bool -> Bool",
        Prim::Eq => "a -> a -> Bool",
        Prim::Pair => "a -> b -> (a, b)",
        Prim::Fst => "(a, b) -> a",
        Prim::Snd => "(a, b) -> b",
        Prim::None => "Option a",
        Prim::Some => "a -> Option a",
        Prim::OptionCase => "Option a -> (Unit -> b) -> (a -> b) -> b",
        Prim::Nil => "List a",
        Prim::Cons => "a -> List a -> List a",
        Prim::ListCase => "List a -> (Unit -> b) -> (a -> List a -> b) -> b",
        Prim::Length => "List a -> Int",
        Prim::ArrayTabulate => "Int -> (Int -> a) -> Array a",
        Prim::ArrayGet => "Array a -> Int -> a",
        Prim::ArraySet => "Array a -> Int -> a -> Array a",
        Prim::ArraySize => "Array a -> Int",
        Prim::ArrayOfList => "List a -> Array a",
        Prim::ArrayToList => "Array a -> List a",
        Prim::Leaf => "Tree a",
        Prim::Node => "Tree a -> a -> Tree a -> Tree a",
        Prim::TreeCase => "Tree a -> (Unit -> b) -> (Tree a -> a -> Tree a -> b) -> b",
        Prim::TreeFold => "b -> (b -> a -> b -> b) -> Tree a -> b",
        Prim::Foldl => "(b -> a -> b) -> b -> List a -> b",
        Prim::Foldr => "(a -> b -> b) -> b -> List a -> b",
        Prim::Loop => "Int -> b -> (Int -> b -> b) -> b",
        Prim::Unfold => "(s -> Option (a, s)) -> s -> List a",
        Prim::Bfix => "Int -> ((a -> b) -> a -> b) -> a -> b",
        Prim::Memorec => "Key k => Int -> ((k -> v) -> k -> v) -> k -> v",
    }
}

#[test]
fn every_prim_has_its_spec_scheme() {
    for &p in Prim::ALL {
        assert_eq!(prim_scheme(p), sch(spec_scheme(p)), "{p}");
    }
}

#[test]
fn prim_schemes_agree_with_prim_arities() {
    for &p in Prim::ALL {
        assert_eq!(
            prim_scheme(p).ty().arrow_count(),
            p.arity(),
            "{p}: {}",
            prim_scheme(p)
        );
    }
}

#[test]
fn inferring_a_bare_prim_gives_its_scheme_and_it_round_trips() {
    for &p in Prim::ALL {
        let s = infer_src(p.name()).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert_eq!(s, prim_scheme(p), "{p}");
        assert_eq!(s.to_string().parse::<Scheme>().unwrap(), s, "{p}");
    }
}

// ---------------------------------------------------------------- good terms

#[test]
fn literals_and_special_forms() {
    assert_infers("42", "Int");
    assert_infers("true", "Bool");
    assert_infers("unit", "Unit");
    assert_infers("(if (lt 1 2) 3 4)", "Int");
    assert_infers("(lam (lam (and $1 (or $0 false))))", "Bool -> Bool -> Bool");
    assert_infers("(lam (if $0 nil (cons 1 nil)))", "Bool -> List Int");
}

#[test]
fn identity_and_compose_get_principal_types() {
    assert_infers("(lam $0)", "a -> a");
    assert_infers("(lam (lam $1))", "a -> b -> a");
    assert_infers(
        "(lam (lam (lam ($2 ($1 $0)))))",
        "(b -> c) -> (a -> b) -> a -> c",
    );
}

#[test]
fn foldl_sum_is_list_int_to_int() {
    assert_infers(
        "(lam (foldl (lam (lam (add $1 $0))) 0 $0))",
        "List Int -> Int",
    );
}

#[test]
fn let_bound_identity_is_polymorphic() {
    assert_infers("(let (lam $0) (pair ($0 1) ($0 true)))", "(Int, Bool)");
    // Generalisation happens only at let: the body keeps the free variable shared.
    assert_infers("(lam (let $0 (pair $0 $1)))", "a -> (a, a)");
}

#[test]
fn let_does_not_generalise_variables_of_the_enclosing_lambda() {
    // $1 in the bound expression is the lambda's variable; the let must not make it
    // polymorphic, so using the result at Int fixes the lambda's argument to Int.
    assert_infers("(lam (let (lam $1) (add ($0 unit) 1)))", "Int -> Int");
    let e = expect_err("(lam (let (lam $1) (pair (add ($0 unit) 1) (not ($0 unit)))))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
}

#[test]
fn let_does_not_generalise_variables_linked_into_the_enclosing_lambda() {
    // Inside the bound expression, list_case creates a fresh element type and unifies
    // the lambda's variable with `List elem`. That element type now belongs to the
    // lambda, so the let must leave it monomorphic.
    let bound = "(list_case $0 (lam nil) (lam (lam $0)))";
    assert_infers(
        &format!("(lam (let {bound} (cons 1 $0)))"),
        "List Int -> List Int",
    );
    let e = expect_err(&format!(
        "(lam (let {bound} (pair (cons 1 $0) (cons true $0))))"
    ));
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
}

#[test]
fn bfix_list_length() {
    assert_infers(
        "(lam (bfix (add (length $0) 1) (lam (lam (list_case $0 (lam 0) (lam (lam (add 1 ($3 $0))))))) $0))",
        "List a -> Int",
    );
}

#[test]
fn tree_programs_get_principal_types() {
    // tree_fold sum: the element type is fixed to Int by add.
    assert_infers(
        "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))",
        "Tree Int -> Int",
    );
    // tree_fold size: the element type stays free.
    assert_infers(
        "(lam (tree_fold 0 (lam (lam (lam (add (add $2 1) $0)))) $0))",
        "Tree a -> Int",
    );
    // tree_fold with node as the combiner rebuilds a tree of the same type.
    assert_infers(
        "(lam (tree_fold leaf (lam (lam (lam (node $2 $1 $0)))) $0))",
        "Tree a -> Tree a",
    );
    // tree_case: the node branch receives left subtree, value, right subtree.
    assert_infers(
        "(lam (lam (tree_case $0 (lam $2) (lam (lam (lam $1))))))",
        "a -> Tree a -> a",
    );
    assert_infers(
        "(lam (tree_case $0 (lam leaf) (lam (lam (lam $2)))))",
        "Tree a -> Tree a",
    );
    // Recursion over a tree through bfix: the height.
    assert_infers(
        "(lam (bfix 64 (lam (lam (tree_case $0 (lam 0) (lam (lam (lam (add 1 (add ($4 $2) ($4 $0))))))))) $0))",
        "Tree a -> Int",
    );
    assert_infers("(node leaf (pair 1 true) leaf)", "Tree (Int, Bool)");
}

#[test]
fn tree_element_types_must_agree() {
    // A node whose subtree holds a different element type.
    let e = expect_err("(node leaf 1 (node leaf true leaf))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    // tree_case branches must return the same type.
    let e = expect_err("(tree_case leaf (lam 0) (lam (lam (lam true))))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    // A tree is not a list.
    let e = expect_err("(length (node leaf 1 leaf))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    // tree_fold passes the element as the combiner's middle argument.
    assert_infers(
        "(tree_fold 0 (lam (lam (lam (add $2 $0)))) (node leaf true leaf))",
        "Int",
    );
    let e = expect_err("(tree_fold 0 (lam (lam (lam (add $1 $0)))) (node leaf true leaf))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
}

#[test]
fn memorec_fib() {
    assert_infers(
        "(lam (memorec (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
        "Int -> Int",
    );
}

#[test]
fn memorec_tuple_keys_are_accepted() {
    assert_infers(
        "(memorec 10 (lam (lam 0)) (pair 1 (pair true unit)))",
        "Int",
    );
}

#[test]
fn memorec_with_undetermined_key_carries_a_key_constraint() {
    assert_infers(
        "(lam (lam (memorec 10 $1 $0)))",
        "Key k => ((k -> v) -> k -> v) -> k -> v",
    );
    // A pair of two undetermined keys: both components get the constraint.
    assert_infers(
        "(lam (lam (lam (memorec 10 $2 (pair $1 $0)))))",
        "Key a, Key b => (((a, b) -> c) -> (a, b) -> c) -> a -> b -> c",
    );
}

#[test]
fn library_schemes_are_instantiated_per_use() {
    let libs = [sch("a -> a"), sch("List a -> Int")];
    assert_eq!(
        infer_with("(pair (lib_0 1) (lib_0 (lib_1 nil)))", &libs).unwrap(),
        sch("(Int, Int)")
    );
    assert_eq!(
        infer_with("(pair (lib_0 1) (lib_0 true))", &libs).unwrap(),
        sch("(Int, Bool)")
    );
    let mut map = HashMap::new();
    map.insert(LibId(7), sch("a -> List a"));
    let mut a = Arena::new();
    let id = parse_term(&mut a, "(lib_7 3)");
    assert_eq!(infer(&a, id, &map).unwrap(), sch("List Int"));
}

#[test]
fn checker_is_reusable_across_terms() {
    let mut a = Arena::new();
    let mut tc = TypeChecker::new();
    let bad = parse_term(&mut a, "(1 2)");
    let good = parse_term(&mut a, "(lam (cons $0 nil))");
    assert!(tc.infer(&a, bad, NO_LIBS).is_err());
    assert_eq!(tc.infer(&a, good, NO_LIBS).unwrap(), sch("a -> List a"));
    assert_eq!(tc.infer(&a, good, NO_LIBS).unwrap(), sch("a -> List a"));
}

// ---------------------------------------------------------------- bad terms

#[test]
fn applying_an_int_is_not_a_function() {
    let e = expect_err("(1 2)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::NotAFunction {
            found: Type::Int,
            applied_to: 1,
            accepted: 0
        }
    );
    assert_eq!(e.term_text, "(1 2)");
    let e = expect_err("(add 1 2 3)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::NotAFunction {
            found: Type::Int,
            applied_to: 3,
            accepted: 2
        }
    );
    assert!(e.to_string().contains("takes only 2"), "{e}");
}

#[test]
fn if_branches_must_agree() {
    let e = expect_err("(if true 1 false)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: Type::Int,
            found: Type::Bool,
            conflict: None
        }
    );
    assert_eq!(e.term_text, "false");
    assert!(e.context.contains("else branch"), "{e}");
}

#[test]
fn if_condition_must_be_bool() {
    let e = expect_err("(if 1 2 3)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: Type::Bool,
            found: Type::Int,
            conflict: None
        }
    );
    assert_eq!(e.term_text, "1");
    assert!(e.context.contains("condition"), "{e}");
}

#[test]
fn and_or_operands_must_be_bool() {
    let e = expect_err("(and true 0)");
    assert!(
        matches!(
            *e.kind,
            TypeErrorKind::Mismatch {
                expected: Type::Bool,
                found: Type::Int,
                ..
            }
        ),
        "{e}"
    );
    assert_eq!(e.term_text, "0");
    let e = expect_err("(or unit true)");
    assert!(
        matches!(
            *e.kind,
            TypeErrorKind::Mismatch {
                expected: Type::Bool,
                found: Type::Unit,
                ..
            }
        ),
        "{e}"
    );
    assert_eq!(e.term_text, "unit");
}

#[test]
fn self_application_fails_the_occurs_check() {
    let e = expect_err("(lam ($0 $0))");
    assert!(matches!(*e.kind, TypeErrorKind::InfiniteType { .. }), "{e}");
    assert!(e.to_string().contains("infinite type"), "{e}");
}

#[test]
fn unknown_library_reference() {
    let e = expect_err("(lib_3 1)");
    assert_eq!(*e.kind, TypeErrorKind::UnknownLib(LibId(3)));
    assert_eq!(e.term_text, "lib_3");
    let libs = [sch("a -> a")];
    assert!(infer_with("(lib_0 1)", &libs).is_ok());
    assert_eq!(
        infer_with("(lib_1 1)", &libs).unwrap_err().kind,
        Box::new(TypeErrorKind::UnknownLib(LibId(1)))
    );
}

#[test]
fn lambda_bound_variables_are_monomorphic() {
    let e = expect_err("(lam (pair ($0 1) ($0 true)))");
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: Type::Int,
            found: Type::Bool,
            conflict: None
        }
    );
    assert_eq!(e.term_text, "true");
    assert_eq!(e.context, "argument 1 of `$0`");
}

#[test]
fn free_variables_are_errors() {
    let e = expect_err("(lam $1)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::UnboundVariable {
            index: 1,
            in_scope: 1
        }
    );
    let e = expect_err("$0");
    assert_eq!(
        *e.kind,
        TypeErrorKind::UnboundVariable {
            index: 0,
            in_scope: 0
        }
    );
    // A let's bound expression does not see the let's own variable.
    let e = expect_err("(let $0 1)");
    assert_eq!(
        *e.kind,
        TypeErrorKind::UnboundVariable {
            index: 0,
            in_scope: 0
        }
    );
}

#[test]
fn bad_memorec_keys_are_rejected() {
    let e = expect_err("(memorec 10 (lam (lam 0)) nil)");
    assert!(
        matches!(&*e.kind, TypeErrorKind::BadMemoKey { found } if matches!(found, Type::List(_))),
        "{e}"
    );
    let e = expect_err("(memorec 10 (lam (lam 0)) (pair 1 (lam $0)))");
    assert!(
        matches!(
            &*e.kind,
            TypeErrorKind::BadMemoKey {
                found: Type::Arrow(..)
            }
        ),
        "{e}"
    );
    // The key type is fixed by the recursive function's use of its argument.
    let e = expect_err("(lam (memorec 10 (lam (lam (length $0))) $0))");
    assert!(matches!(&*e.kind, TypeErrorKind::BadMemoKey { .. }), "{e}");
}

#[test]
fn key_constraints_survive_let_polymorphism() {
    // A let-bound memoising helper used at Int is fine ...
    assert_infers(
        "(let (lam (lam (memorec 10 $1 $0))) ($0 (lam (lam 0)) 5))",
        "Int",
    );
    // ... and used at a list key it is rejected, even though memorec is out of sight.
    let e = expect_err("(let (lam (lam (memorec 10 $1 $0))) ($0 (lam (lam 0)) nil))");
    assert!(matches!(&*e.kind, TypeErrorKind::BadMemoKey { .. }), "{e}");
}

#[test]
fn key_constraints_in_library_schemes_are_enforced() {
    let libs = [sch("Key a => a -> Int")];
    assert!(infer_with("(lib_0 (pair 1 true))", &libs).is_ok());
    let e = infer_with("(lib_0 nil)", &libs).unwrap_err();
    assert!(matches!(&*e.kind, TypeErrorKind::BadMemoKey { .. }), "{e}");
}

// ---------------------------------------------------------------- messages

#[test]
fn messages_name_the_argument_types_and_subterm() {
    let e = expect_err("(lam (foldl (lam (lam (add $1 $0))) true $0))");
    let msg = e.to_string();
    assert!(msg.contains("argument 2 of `foldl`"), "{msg}");
    assert!(msg.contains("expected `Int`"), "{msg}");
    assert!(msg.contains("found `Bool`"), "{msg}");
    assert!(msg.contains("term: true"), "{msg}");
}

#[test]
fn messages_rename_type_variables_from_a() {
    // Lots of variables are created before the error; the message must still use
    // a, b, ... with one letter per variable across expected and found.
    let e = expect_err("(lam (lam (lam (cons $0 (pair $0 $0)))))");
    let msg = e.to_string();
    assert!(msg.contains("expected `List a`"), "{msg}");
    assert!(msg.contains("found `(a, a)`"), "{msg}");
}

#[test]
fn messages_report_the_inner_conflict() {
    let e = expect_err("(cons 1 (cons true nil))");
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: ty("List Int"),
            found: ty("List Bool"),
            conflict: Some((Type::Int, Type::Bool)),
        }
    );
    assert!(
        e.to_string()
            .contains("conflict: `Int` does not match `Bool`"),
        "{e}"
    );
}

#[test]
fn long_subterms_are_shortened() {
    let mut src = String::from("0");
    for _ in 0..200 {
        src = format!("(add 1 {src})");
    }
    let src = format!("(if {src} 1 2)");
    let e = expect_err(&src);
    assert!(e.term_text.len() < 300, "{}", e.term_text.len());
    assert!(e.term_text.ends_with("..."), "{}", e.term_text);
}

// ---------------------------------------------------------------- check

#[test]
fn check_accepts_terms_at_least_as_general_as_expected() {
    assert!(check_src("(lam $0)", "Int -> Int").is_ok());
    assert!(check_src("(lam $0)", "a -> a").is_ok());
    assert!(check_src("(lam (lam $1))", "Int -> Bool -> Int").is_ok());
    assert!(check_src(
        "(lam (foldl (lam (lam (add $1 $0))) 0 $0))",
        "List Int -> Int"
    )
    .is_ok());
}

#[test]
fn check_rejects_terms_less_general_than_expected() {
    let e = check_src("(lam (add $0 1))", "a -> a").unwrap_err();
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    assert!(e.note.as_deref().unwrap_or("").contains("rigid"), "{e}");
    // Two distinct expected variables cannot be unified with each other.
    let e = check_src("(lam (lam $0))", "a -> b -> a").unwrap_err();
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
}

#[test]
fn check_rejects_wrong_types() {
    // The expected type is pushed through the lambda, so the clash is reported at
    // its body against the rest of the expected type.
    let e = check_src("(lam 1)", "Int -> Int -> Int").unwrap_err();
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: ty("Int -> Int"),
            found: Type::Int,
            conflict: None,
        }
    );
    assert_eq!(e.term_text, "1", "{e}");
    assert!(e.context.contains("the body of `lam`"), "{e}");
    assert!(check_src("(lam $0)", "Int -> Bool").is_err());
}

#[test]
fn check_enforces_key_constraints_on_expected_variables() {
    let helper = "(lam (lam (memorec 10 $1 $0)))";
    let e = check_src(helper, "((a -> b) -> a -> b) -> a -> b").unwrap_err();
    assert!(matches!(&*e.kind, TypeErrorKind::BadMemoKey { .. }), "{e}");
    assert!(e.note.as_deref().unwrap_or("").contains("Key a"), "{e}");
    assert!(check_src(helper, "Key a => ((a -> b) -> a -> b) -> a -> b").is_ok());
    assert!(check_src(helper, "((Int -> b) -> Int -> b) -> Int -> b").is_ok());
}

// ---------------------------------------------------------------- laws on random terms

fn arb_src() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        (0u32..3).prop_map(|i| format!("${i}")),
        (-2i64..3).prop_map(|n| n.to_string()),
        Just("true".to_string()),
        Just("unit".to_string()),
        proptest::sample::select(Prim::ALL).prop_map(|p| p.name().to_string()),
    ];
    leaf.prop_recursive(5, 40, 3, |inner| {
        prop_oneof![
            3 => inner.clone().prop_map(|b| format!("(lam {b})")),
            3 => (inner.clone(), inner.clone()).prop_map(|(f, a)| format!("({f} {a})")),
            1 => (inner.clone(), inner.clone()).prop_map(|(x, b)| format!("(let {x} {b})")),
            1 => (inner.clone(), inner.clone(), inner).prop_map(|(c, t, e)| format!("(if {c} {t} {e})")),
        ]
    })
}

#[test]
fn laws_on_random_terms() {
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(2000));
    let typed = std::cell::Cell::new(0u32);
    runner
        .run(&arb_src(), |src| {
            let mut a = Arena::new();
            let id = sexpr::parse(&mut a, &src).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut tc = TypeChecker::new();
            let first = tc.infer(&a, id, NO_LIBS);
            // Re-interning the same text in a fresh arena gives the same answer.
            let mut b = Arena::new();
            let id_b =
                sexpr::parse(&mut b, &src).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let fresh = infer(&b, id_b, NO_LIBS);
            prop_assert_eq!(
                first.as_ref().map_err(|e| e.to_string()),
                fresh.as_ref().map_err(|e| e.to_string())
            );
            if let Ok(s) = &first {
                typed.set(typed.get() + 1);
                // A term checks against its own principal scheme.
                prop_assert!(
                    tc.check(&a, id, NO_LIBS, s).is_ok(),
                    "{} does not check against {}",
                    src,
                    s
                );
                // Binding a closed term with let and returning it changes nothing.
                if a.is_closed(id) {
                    let v = a.var(0);
                    let wrapped = a.let_(id, v);
                    let w = tc.infer(&a, wrapped, NO_LIBS);
                    prop_assert_eq!(w.as_ref().ok(), Some(s), "{}", src);
                }
            }
            Ok(())
        })
        .unwrap();
    // The laws above are only exercised by well-typed terms; make sure there are some.
    assert!(
        typed.get() > 100,
        "only {} of 2000 random terms were well typed",
        typed.get()
    );
}

// ---------------------------------------------------------------- depth and sharing

/// `(cons 1 (cons 1 ... (cons last nil)))` with `n` elements, built without the
/// (recursive) s-expression parser.
fn deep_list(a: &mut Arena, n: usize, last: TermId) -> TermId {
    let cons = a.prim(Prim::Cons);
    let one = a.int(1);
    let nil = a.prim(Prim::Nil);
    let mut t = a.apps(cons, &[last, nil]);
    for _ in 1..n {
        t = a.apps(cons, &[one, t]);
    }
    t
}

#[test]
fn deep_terms_are_checked_without_host_recursion() {
    // Far deeper than the host stack of a test thread allows for a recursive walk.
    const N: usize = 200_000;
    let mut a = Arena::new();
    let one = a.int(1);
    let list = deep_list(&mut a, N, one);
    assert_eq!(infer(&a, list, NO_LIBS), Ok(sch("List Int")));
    assert_eq!(check(&a, list, NO_LIBS, &sch("List Int")), Ok(()));

    // An error at the bottom is still found, and its subterm text stays short.
    let t = a.bool(true);
    let bad = deep_list(&mut a, N, t);
    let e = infer(&a, bad, NO_LIBS).unwrap_err();
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    assert!(e.term_text.len() < 300, "{}", e.term_text.len());
    let e = check(&a, bad, NO_LIBS, &sch("List Int")).unwrap_err();
    assert!(e.term_text.len() < 300, "{}", e.term_text.len());

    // Deep lets and ifs.
    let mut t = a.int(7);
    for _ in 0..N {
        let c = a.bool(false);
        let other = a.int(8);
        let body = a.if_(c, t, other);
        let v = a.var(0);
        let bound = a.lam(v);
        t = a.let_(bound, body);
    }
    assert_eq!(infer(&a, t, NO_LIBS), Ok(sch("Int")));
    assert_eq!(check(&a, t, NO_LIBS, &sch("Int")), Ok(()));
}

#[test]
fn deep_lambda_nests_report_a_too_large_type() {
    // N nested lambdas have an arrow type N levels deep. That is reported as an
    // error rather than built (and later printed or dropped) recursively.
    let mut a = Arena::new();
    let mut t = a.var(0);
    for _ in 0..100_000 {
        t = a.lam(t);
    }
    let e = infer(&a, t, NO_LIBS).unwrap_err();
    assert!(matches!(*e.kind, TypeErrorKind::TypeTooLarge { .. }), "{e}");
    assert!(e.to_string().contains("too large"), "{e}");
}

/// The doubler `$0 = (lam (pair $0 $0))` applied `n` times to `1`: its type is a
/// DAG with `n` nodes but a tree with `2^n` leaves.
fn doubled(n: usize) -> String {
    let mut x = String::from("1");
    for _ in 0..n {
        x = format!("($0 {x})");
    }
    x
}

#[test]
fn shared_type_structure_is_walked_once() {
    // Before the store walks were memoised these took time exponential in n
    // (about a minute at n = 30 in a release build).
    let start = std::time::Instant::now();
    let d = doubled(60);
    assert_infers(
        &format!("(let (lam (pair $0 $0)) (length (cons {d} nil)))"),
        "Int",
    );
    // The same through a let: generalisation and instantiation of the shared type.
    assert_infers(
        &format!("(let (lam (pair $0 $0)) (let {d} (length (cons $0 (cons $0 nil)))))"),
        "Int",
    );
    // A principal type that is exponentially large as a tree is an error, not an
    // attempt to build it.
    let e = expect_err(&format!("(let (lam (pair $0 $0)) {d})"));
    assert!(matches!(*e.kind, TypeErrorKind::TypeTooLarge { .. }), "{e}");
    // A clash inside such a type is reported without printing it in full.
    let e = expect_err(&format!("(let (lam (pair $0 $0)) (add 1 {d}))"));
    assert!(e.to_string().len() < 2000, "{}", e.to_string().len());
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
}

/// A checker whose term walk panics after `steps` steps, so a test of how much
/// work inference does fails fast, without a timer or a thread left running.
fn bounded(steps: u64) -> TypeChecker {
    let mut tc = TypeChecker::new();
    tc.step_limit = Some(steps);
    tc
}

/// `(if c t t)` nested `n` deep around `leaf`: a DAG of `n` nodes that is a tree of
/// `2^n` nodes. Every level's two branches are the same `TermId`.
fn shared_ifs(a: &mut Arena, n: usize, c: TermId, leaf: TermId) -> TermId {
    let mut t = leaf;
    for _ in 0..n {
        t = a.if_(c, t, t);
    }
    t
}

#[test]
fn shared_subterms_are_inferred_once() {
    // Walking the term as a tree takes 2^30 steps at n = 30; the budget is linear.
    const N: usize = 30;
    const STEPS: u64 = 100 * N as u64;
    // Closed.
    let mut a = Arena::new();
    let c = a.bool(true);
    let one = a.int(1);
    let t = shared_ifs(&mut a, N, c, one);
    assert_eq!(bounded(STEPS).infer(&a, t, NO_LIBS), Ok(sch("Int")));
    assert_eq!(bounded(STEPS).check(&a, t, NO_LIBS, &sch("Int")), Ok(()));
    assert!(bounded(STEPS).check(&a, t, NO_LIBS, &sch("Bool")).is_err());
    // Open: the shared subterms use the lambda's variable.
    let v = a.var(0);
    let body = shared_ifs(&mut a, N, v, v);
    let f = a.lam(body);
    let want = sch("Bool -> Bool");
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(want.clone()));
    assert_eq!(bounded(STEPS).check(&a, f, NO_LIBS, &want), Ok(()));
    // Under a let, so the shared subterms see a polymorphic binding.
    let id = a.lam(v);
    let inner = a.app(v, one);
    let body = shared_ifs(&mut a, N, c, inner);
    let l = a.let_(id, body);
    assert_eq!(bounded(STEPS).infer(&a, l, NO_LIBS), Ok(sch("Int")));
}

/// `t' = (pair (fst t) (lam (snd t)))` nested `n` deep around `(pair $0 $0)`, under
/// a lambda: the two uses of `t` see different binders for `$0`.
fn pairs_across_scopes(a: &mut Arena, n: usize) -> TermId {
    let v = a.var(0);
    let (pair, fst, snd) = (a.prim(Prim::Pair), a.prim(Prim::Fst), a.prim(Prim::Snd));
    let mut t = a.apps(pair, &[v, v]);
    for _ in 0..n {
        let l = a.app(fst, t);
        let r = a.app(snd, t);
        let r = a.lam(r);
        t = a.apps(pair, &[l, r]);
    }
    a.lam(t)
}

#[test]
fn open_shared_subterms_are_reused_across_scopes() {
    // Each level uses the open subterm below it twice, in two different scopes, so
    // the tree has 2^n nodes. Keying the reuse on the scope alone never hits.
    const N: usize = 40;
    const STEPS: u64 = 100 * N as u64;
    let mut a = Arena::new();
    let v = a.var(0);
    // t' = (let t t): in the scope of the let, and under the let's binder.
    let mut t = v;
    for _ in 0..N {
        t = a.let_(t, t);
    }
    let f = a.lam(t);
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(sch("a -> a")));
    // t' = ((lam t) t): under a new lambda binder, and outside it.
    let mut t = v;
    for _ in 0..N {
        let l = a.lam(t);
        t = a.app(l, t);
    }
    let f = a.lam(t);
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(sch("a -> a")));
    let want = sch("Int -> Int");
    assert_eq!(bounded(STEPS).check(&a, f, NO_LIBS, &want), Ok(()));
    // t' = (if true t (let (lam $0) t)) around ($0 1), under (let (lam $0) ...): the
    // uses see different polymorphic bindings of the same type.
    let (yes, one) = (a.bool(true), a.int(1));
    let id = a.lam(v);
    let mut t = a.app(v, one);
    for _ in 0..N {
        let l = a.let_(id, t);
        t = a.if_(yes, t, l);
    }
    let f = a.let_(id, t);
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(sch("Int")));
    // Uses at different types, compared with the tree walk at a size it can do.
    let small = pairs_across_scopes(&mut a, 8);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, small, NO_LIBS);
    assert!(want.is_ok(), "{want:?}");
    assert_eq!(bounded(STEPS).infer(&a, small, NO_LIBS), want);
    let big = pairs_across_scopes(&mut a, N);
    let got = bounded(STEPS).infer(&a, big, NO_LIBS);
    assert!(got.is_ok(), "{got:?}");
}

#[test]
fn shared_spine_prefixes_do_not_hide_shared_arguments() {
    // t' = (if (lt $0 0) (add t 1) (add t 2)): the two applications share the
    // prefix `(add t)`, which the walk never infers on its own, so `t` must be
    // reused as an argument of both spines.
    const N: usize = 30;
    const STEPS: u64 = 100 * N as u64;
    let mut a = Arena::new();
    let v = a.var(0);
    let (zero, one, two) = (a.int(0), a.int(1), a.int(2));
    let (add, lt) = (a.prim(Prim::Add), a.prim(Prim::Lt));
    let cond = a.apps(lt, &[v, zero]);
    let mut t = v;
    for _ in 0..N {
        let l = a.apps(add, &[t, one]);
        let r = a.apps(add, &[t, two]);
        t = a.if_(cond, l, r);
    }
    let f = a.lam(t);
    let want = sch("Int -> Int");
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(want.clone()));
    assert_eq!(bounded(STEPS).check(&a, f, NO_LIBS, &want), Ok(()));
    // The prefix is a whole argument of `fst`: ((pair t) 1) and ((pair t) 2).
    let yes = a.bool(true);
    let (pair, fst) = (a.prim(Prim::Pair), a.prim(Prim::Fst));
    let mut t = v;
    for _ in 0..N {
        let l = a.apps(pair, &[t, one]);
        let l = a.app(fst, l);
        let r = a.apps(pair, &[t, two]);
        let r = a.app(fst, r);
        t = a.if_(yes, l, r);
    }
    let f = a.lam(t);
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), Ok(sch("a -> a")));
}

#[test]
fn shared_subterms_far_from_their_binders_are_reused_across_scopes() {
    // As in `open_shared_subterms_are_reused_across_scopes`, but the variable is
    // bound more than 64 binders out, as a reference to an early definition in a
    // long chain of lets is.
    const N: usize = 30;
    const STEPS: u64 = 400 * N as u64;
    const OUTER: u32 = 70;
    let mut a = Arena::new();
    let v = a.var(OUTER - 1);
    let wrap = |a: &mut Arena, mut t: TermId| {
        for _ in 0..OUTER {
            t = a.lam(t);
        }
        t
    };
    let mut t = v;
    for _ in 0..N {
        t = a.let_(t, t);
    }
    let f = wrap(&mut a, t);
    let got = bounded(STEPS).infer(&a, f, NO_LIBS);
    assert!(got.is_ok(), "{got:?}");
    let v = a.var(OUTER - 1);
    let mut t = v;
    for _ in 0..N {
        let l = a.lam(t);
        t = a.app(l, t);
    }
    let f = wrap(&mut a, t);
    let got = bounded(STEPS).infer(&a, f, NO_LIBS);
    assert!(got.is_ok(), "{got:?}");
}

#[test]
fn errors_inside_shared_subterms_are_located_as_in_the_tree() {
    // A duplicate of an ill-typed subterm does not change where the error is
    // reported: it names the variable and its argument position, as it would
    // without the duplicate.
    let e = expect_err("(lam (if $0 (add $0 1) (add $0 1)))");
    assert_eq!(e.term_text, "$0", "{e}");
    assert!(e.context.contains("argument 1 of `add`"), "{e}");
    let cases = [
        ("(lam (if $0 (add $0 1) (add $0 1)))", None),
        (
            "(lam (lam (pair (add $0 1) (pair (cons $1 (cons $0 nil)) \
             (cons $1 (cons $0 nil))))))",
            Some("a -> a"),
        ),
        (
            "(lam (lam (pair (add $0 1) (pair (cons $1 (cons $0 nil)) \
             (cons $1 (cons $0 nil))))))",
            Some("a -> b -> (Int, (List a, List a))"),
        ),
        (
            "(lam (pair (fst $0) (and (fst $0) (add (fst $0) 1))))",
            None,
        ),
    ];
    for (src, expected) in cases {
        let mut a = Arena::new();
        let id = parse_term(&mut a, src);
        let mut memo = TypeChecker::new();
        let mut tree = TypeChecker::new();
        tree.tree_walk = true;
        let (got, want) = match expected {
            None => (
                memo.infer(&a, id, NO_LIBS).map(|_| ()),
                tree.infer(&a, id, NO_LIBS).map(|_| ()),
            ),
            Some(s) => (
                memo.check(&a, id, NO_LIBS, &sch(s)),
                tree.check(&a, id, NO_LIBS, &sch(s)),
            ),
        };
        assert!(want.is_err(), "{src}");
        assert_eq!(got, want, "{src}");
    }
}

/// `(lam (pair t (let true t)))` with `t` = `(if true u u)` nested `n` deep around
/// `u` = `(add $0 1)`: the second `t` sees `$0` bound to a Bool.
fn clash_under_let(a: &mut Arena, n: usize) -> TermId {
    let (v, one, yes) = (a.var(0), a.int(1), a.bool(true));
    let (add, pair) = (a.prim(Prim::Add), a.prim(Prim::Pair));
    let leaf = a.apps(add, &[v, one]);
    let t = shared_ifs(a, n, yes, leaf);
    let l = a.let_(yes, t);
    let body = a.apps(pair, &[t, l]);
    a.lam(body)
}

#[test]
fn errors_in_large_shared_terms_are_located_as_in_the_tree() {
    // At n = 18 the tree walk takes millions of steps; locating the error must
    // still name the subterm the tree walk names, in a polynomial number of steps.
    const N: usize = 18;
    const STEPS: u64 = 100 * (N * N) as u64;
    let mut a = Arena::new();
    let f = clash_under_let(&mut a, N);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, f, NO_LIBS);
    let e = want.clone().expect_err("ill typed");
    assert!(e.context.contains("argument 1 of `add`"), "{e}");
    assert_eq!(TypeChecker::new().infer(&a, f, NO_LIBS), want);
    assert_eq!(bounded(STEPS).infer(&a, f, NO_LIBS), want);
    let s = sch("Int -> (Int, Int)");
    let want = tree.check(&a, f, NO_LIBS, &s);
    assert!(want.is_err());
    assert_eq!(bounded(STEPS).check(&a, f, NO_LIBS, &s), want);
}

/// `t_n`, with `t_(k+1) = (if true t_k t_k)` and `t_0 = (add true 1)` if `open` is
/// false, else `lam x. t_n` with `t_0 = (add x true)`: a leaf that is ill typed on
/// its own, shared down a chain.
fn ill_typed_leaf_chain(a: &mut Arena, n: usize, open: bool) -> TermId {
    let (add, yes, one) = (a.prim(Prim::Add), a.bool(true), a.int(1));
    let mut t = if open {
        let v0 = a.var(0);
        a.apps(add, &[v0, yes])
    } else {
        a.apps(add, &[yes, one])
    };
    for _ in 0..n {
        t = a.if_(yes, t, t);
    }
    if open {
        a.lam(t)
    } else {
        t
    }
}

#[test]
fn ill_typed_shared_leaves_are_located_in_polynomial_steps() {
    // Each use of a subterm whose fresh inference failed is walked in place at
    // once, not tried again from a checkpoint: otherwise every level doubles the
    // work of locating the error.
    for open in [false, true] {
        let mut a = Arena::new();
        let small = ill_typed_leaf_chain(&mut a, 10, open);
        let mut tree = TypeChecker::new();
        tree.tree_walk = true;
        let want = tree.infer(&a, small, NO_LIBS);
        assert!(want.is_err());
        assert_eq!(TypeChecker::new().infer(&a, small, NO_LIBS), want);

        const N: usize = 40;
        const STEPS: u64 = 100 * (N * N) as u64;
        let big = ill_typed_leaf_chain(&mut a, N, open);
        let e = bounded(STEPS)
            .infer(&a, big, NO_LIBS)
            .expect_err("ill typed");
        assert_eq!(e.context, want.clone().unwrap_err().context, "open {open}");
        let s = if open { sch("Int -> Int") } else { sch("Int") };
        let e = bounded(STEPS)
            .check(&a, big, NO_LIBS, &s)
            .expect_err("ill typed");
        let want = tree.check(&a, small, NO_LIBS, &s).unwrap_err();
        assert_eq!(e.context, want.context, "open {open}");
    }
}

#[test]
fn only_failed_fresh_inference_marks_a_key_failed() {
    // `(add $0 1)` infers afresh as `Int -> Int`; only fitting it to `$0 : Bool`
    // fails, which depends on the binder, so the key must not be marked failed.
    let mut a = Arena::new();
    let (add, one, v0) = (a.prim(Prim::Add), a.int(1), a.var(0));
    let t = a.apps(add, &[v0, one]);
    let body = a.if_(v0, t, t);
    let f = a.lam(body);
    let mut tc = TypeChecker::new();
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    assert_eq!(tc.infer(&a, f, NO_LIBS), tree.infer(&a, f, NO_LIBS));
    assert_eq!(tc.failed_keys(), 0);
    // A subterm ill typed on its own is marked, so the count is not always 0.
    let f = parse_term(&mut a, "(lam (if $0 (add true 1) (add true 1)))");
    assert_eq!(tc.infer(&a, f, NO_LIBS), tree.infer(&a, f, NO_LIBS));
    assert_eq!(tc.failed_keys(), 1);
}

/// `lam x. let (lam (pair x $0)) in t_n`, with `t_0 = ($0 1)` and `t_(k+1) =
/// (if true t_k (let (lam (pair (fst ($1 0)) $0)) t_k))`: every level binds a new
/// polymorphic helper whose type has the free variable `x`.
fn helpers_over_a_lambda_variable(a: &mut Arena, n: usize) -> TermId {
    let leaf = parse_term(a, "($0 1)");
    helpers_over_a_lambda_variable_from(a, n, leaf)
}

/// [`helpers_over_a_lambda_variable`] with `t_0 = leaf`.
fn helpers_over_a_lambda_variable_from(a: &mut Arena, n: usize, leaf: TermId) -> TermId {
    let (v0, v1, zero, yes) = (a.var(0), a.var(1), a.int(0), a.bool(true));
    let (pair, fst) = (a.prim(Prim::Pair), a.prim(Prim::Fst));
    let b0 = a.apps(pair, &[v1, v0]);
    let b0 = a.lam(b0);
    let prev = a.app(v1, zero);
    let x = a.app(fst, prev);
    let b = a.apps(pair, &[x, v0]);
    let b = a.lam(b);
    let mut t = leaf;
    for _ in 0..n {
        let l = a.let_(b, t);
        t = a.if_(yes, t, l);
    }
    let body = a.let_(b0, t);
    a.lam(body)
}

#[test]
fn shared_subterms_under_new_helpers_with_free_type_variables_are_reused() {
    const N: usize = 30;
    const STEPS: u64 = 100 * N as u64;
    let mut a = Arena::new();
    let small = helpers_over_a_lambda_variable(&mut a, 6);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, small, NO_LIBS);
    assert_eq!(want, Ok(sch("a -> (a, Int)")));
    assert_eq!(bounded(STEPS).infer(&a, small, NO_LIBS), want);
    let big = helpers_over_a_lambda_variable(&mut a, N);
    assert_eq!(bounded(STEPS).infer(&a, big, NO_LIBS), want);
}

/// `lam x. let h = (lam (pair x $0)) in body`, where `body` is `t_n`, or
/// `(pair first t_n)` if `first` is given, with `t_(k+1) = (if cond t_k t_k)`
/// and `t_0 = leaf`, all read with `h` as `$0` and `x` as `$1`: every shared
/// subterm uses the polymorphic helper `h`, whose type has the free variable
/// `x`.
fn chain_under_an_open_helper(
    a: &mut Arena,
    n: usize,
    cond: &str,
    leaf: &str,
    first: Option<&str>,
) -> TermId {
    let cond = parse_term(a, cond);
    let mut t = parse_term(a, leaf);
    for _ in 0..n {
        t = a.if_(cond, t, t);
    }
    if let Some(first) = first {
        let (pair, first) = (a.prim(Prim::Pair), parse_term(a, first));
        t = a.apps(pair, &[first, t]);
    }
    let h = parse_term(a, "(lam (pair $1 $0))");
    let l = a.let_(h, t);
    a.lam(l)
}

/// Shapes whose error is in shared subterms that use a polymorphic helper
/// with a free type variable, so the scope of their keys is open.
fn open_scope_failures(a: &mut Arena, n: usize) -> Vec<(&'static str, TermId)> {
    let s7_leaf = parse_term(a, "(add ($0 1) 1)");
    vec![
        (
            "s2: ill typed whatever the helper's free variable is",
            chain_under_an_open_helper(a, n, "true", S2_LEAF, None),
        ),
        (
            "s7: a new helper at every level",
            helpers_over_a_lambda_variable_from(a, n, s7_leaf),
        ),
        (
            "s8: ill typed only once the helper's free variable is Int",
            chain_under_an_open_helper(a, n, "true", S8_LEAF, Some(S8_FIRST)),
        ),
        (
            "g3: s8 with a use of the helper that fixes its free variable at every level",
            chain_under_an_open_helper(a, n, FIX_X, S8_LEAF, Some(S8_FIRST)),
        ),
        (
            "g6: s2 with a use of the helper that fixes its free variable at every level",
            chain_under_an_open_helper(a, n, FIX_X, S2_LEAF, None),
        ),
        (
            "g9: s2 with the free variable of a helper of its own fixed at every level",
            chain_fixing_helpers_of_its_own(a, n),
        ),
    ]
}

/// The leaf of s2: ill typed whatever the helper's free variable is.
const S2_LEAF: &str = "(add (snd ($0 1)) true)";
/// The leaf of s8, ill typed once the helper's free variable is `Int`, and the
/// first part of the pair around its chain, which makes it `Int`.
const S8_LEAF: &str = "(if (fst ($0 1)) true true)";
const S8_FIRST: &str = "(add (fst ($0 1)) 1)";
/// A condition that fixes the helper's free variable to `Int` again: once it
/// is, unifying it with `Int` changes nothing.
const FIX_X: &str = "(fst (pair true (add (fst ($0 1)) 1)))";

/// `lam x. lam z_1 ... lam z_n. let g_n = (lam (pair z_n $0)) in ... let g_1 =
/// (lam (pair z_1 $0)) in let h = (lam (pair x $0)) in t_n`, with `t_0` the
/// leaf of s2 and `t_k = (if (fst (pair true (add (fst (g_k 1)) 1))) t_(k-1)
/// t_(k-1))`. Each level fixes the free variable `z_k` of a helper of its
/// own, a node made before any use is tried, which the keys below it do not
/// depend on: their scopes reach out to `g_(k-1)` only.
fn chain_fixing_helpers_of_its_own(a: &mut Arena, n: usize) -> TermId {
    let width = u32::try_from(n).expect("a small depth");
    let mut t = parse_term(a, S2_LEAF);
    for k in 1..=width {
        // Under the `let`s, `h` is `$0` and `g_k` is `$k`.
        let cond = parse_term(a, &format!("(fst (pair true (add (fst (${k} 1)) 1)))"));
        t = a.if_(cond, t, t);
    }
    // Under the `let`s of `g_1 ... g_n` and the argument, `x` is `$(2n + 1)`.
    let h = parse_term(a, &format!("(lam (pair ${} $0))", 2 * width + 1));
    let mut body = a.let_(h, t);
    for k in 1..=width {
        // Under the `let`s of `g_(k+1) ... g_n` and the argument, `z_k` is
        // `$(2n - 2k + 1)`.
        let g = parse_term(a, &format!("(lam (pair ${} $0))", 2 * (width - k) + 1));
        body = a.let_(g, body);
    }
    for _ in 0..=n {
        body = a.lam(body);
    }
    body
}

/// The type [`assert_open_scope_failure_located`] checks the shape of
/// [`open_scope_failures`] with depth `n` against.
fn open_scope_expected(name: &str, n: usize) -> Scheme {
    if name.starts_with("g9") {
        sch(&format!("{}Bool", "Int -> ".repeat(n + 1)))
    } else {
        sch("Int -> (Int, Bool)")
    }
}

/// Checks that the error of shape `which` of [`open_scope_failures`] is the
/// tree walk's, and is found in a number of steps linear in the depth of
/// sharing. The keys of its shared subterms have a polymorphic slot whose
/// type has a free variable, so a failure seen once is not known to hold in
/// every state of the store.
fn assert_open_scope_failure_located(which: usize) {
    const SMALL: usize = 10;
    const N: usize = 40;
    let mut a = Arena::new();
    let (name, small) = open_scope_failures(&mut a, SMALL)[which];
    // About 50 steps a level are taken; the tree walk takes 2^N. In g9 each
    // failed level is walked in place down to the leaf, through a condition
    // of its own at every level below, which takes about 15 N^2 steps in all.
    let steps: u64 = if name.starts_with("g9") {
        20 * (N * N) as u64
    } else {
        100 * N as u64
    };
    let (_, big) = open_scope_failures(&mut a, N)[which];
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, small, NO_LIBS);
    assert!(want.is_err(), "{name}");
    assert_eq!(TypeChecker::new().infer(&a, small, NO_LIBS), want, "{name}");
    // The error is in the leaf, so it is the same at every depth.
    assert_eq!(bounded(steps).infer(&a, big, NO_LIBS), want, "{name}");
    let expected = open_scope_expected(name, SMALL);
    let want = tree.check(&a, small, NO_LIBS, &expected);
    assert!(want.is_err(), "{name}");
    assert_eq!(
        TypeChecker::new().check(&a, small, NO_LIBS, &expected),
        want,
        "{name}"
    );
    let expected = open_scope_expected(name, N);
    let want = tree.check(&a, big, NO_LIBS, &expected);
    assert_eq!(
        bounded(steps).check(&a, big, NO_LIBS, &expected),
        want,
        "{name}"
    );
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_s2() {
    assert_open_scope_failure_located(0);
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_s7() {
    assert_open_scope_failure_located(1);
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_s8() {
    assert_open_scope_failure_located(2);
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_g3() {
    assert_open_scope_failure_located(3);
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_g6() {
    assert_open_scope_failure_located(4);
}

#[test]
fn open_scope_failures_are_located_in_polynomial_steps_g9() {
    assert_open_scope_failure_located(5);
}

/// `snd` applied `j` times to `e`.
fn snds(j: usize, e: &str) -> String {
    format!("{}{e}{}", "(snd ".repeat(j), ")".repeat(j))
}

/// `lam x_1 ... lam x_p. let h = helper in body`, where `body` is `t_n`, or
/// `(pair first t_n)` if `first` is given, with `t_k = (if (cond (n - k + 1))
/// t_(k-1) t_(k-1))` and `t_0 = leaf`: [`chain_under_an_open_helper`] with a
/// condition of its own at each level, numbered from 1 at the top.
fn chain_with_conds(
    a: &mut Arena,
    n: usize,
    params: usize,
    helper: &str,
    cond: impl Fn(usize) -> String,
    leaf: &str,
    first: Option<&str>,
) -> TermId {
    let mut t = parse_term(a, leaf);
    for k in 1..=n {
        let c = parse_term(a, &cond(n - k + 1));
        t = a.if_(c, t, t);
    }
    if let Some(first) = first {
        let (pair, first) = (a.prim(Prim::Pair), parse_term(a, first));
        t = a.apps(pair, &[first, t]);
    }
    let h = parse_term(a, helper);
    let mut body = a.let_(h, t);
    for _ in 0..params {
        body = a.lam(body);
    }
    body
}

/// A condition that makes the part `snd^j` of the helper's free variable `x`
/// a pair of one variable: each level refines a node of `x` that the level
/// above made.
fn deepening(j: usize) -> String {
    let part = snds(j, "(fst ($1 1))");
    format!("(fst (pair true (lam (if true {part} (pair $0 $0)))))")
}

/// A condition that unifies part `fst (snd^j x)` of the helper's free variable
/// with the second parameter `y`, which is `Int`. Tried afresh, `y` is a
/// fresh variable, so the part stays a variable; walked in place, it becomes
/// `Int`.
fn through_a_parameter(j: usize) -> String {
    let part = snds(j, "(fst ($0 1))");
    format!("(fst (pair true (if true $1 (fst {part}))))")
}

/// Shapes whose keys have an open scope and whose levels refine the scope's
/// free variable in a different way at each level, so no two levels see the
/// same store. The conditions grow with the depth, so these are kept apart
/// from [`open_scope_failures`], which the read judge runs at a large depth.
fn refining_scope_failures(a: &mut Arena, n: usize) -> Vec<(&'static str, TermId)> {
    vec![
        (
            "r2: s2 with each level refining the helper's free variable further",
            chain_with_conds(a, n, 1, "(lam (pair $1 $0))", deepening, S2_LEAF, None),
        ),
        (
            "r2s8: s8 on the first part of the free variable, refining the second",
            chain_with_conds(
                a,
                n,
                1,
                "(lam (pair $1 $0))",
                deepening,
                "(if (fst (fst ($0 1))) true true)",
                Some("(add (fst (fst ($0 1))) 1)"),
            ),
        ),
        (
            "m: s2 with each level fixing a part of the free variable only in place",
            chain_with_conds(
                a,
                n,
                2,
                "(lam (pair $2 $0))",
                through_a_parameter,
                S2_LEAF,
                Some("(add $1 1)"),
            ),
        ),
    ]
}

/// Checks that the error of shape `which` of [`refining_scope_failures`] is
/// the tree walk's, for inference and for checking against `a -> b -> c`
/// (the shapes have at most two parameters), within `steps(N)` steps at
/// depth `N`.
fn assert_refining_failure_located(which: usize, steps: impl Fn(u64) -> u64) {
    const SMALL: usize = 8;
    const N: usize = 40;
    let mut a = Arena::new();
    let (name, small) = refining_scope_failures(&mut a, SMALL)[which];
    let (_, big) = refining_scope_failures(&mut a, N)[which];
    let steps = steps(N as u64);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, small, NO_LIBS);
    assert!(want.is_err(), "{name}");
    assert_eq!(TypeChecker::new().infer(&a, small, NO_LIBS), want, "{name}");
    // The tree walk fails in the leftmost leaf, so it is cheap at any depth.
    let want = tree.infer(&a, big, NO_LIBS);
    assert_eq!(bounded(steps).infer(&a, big, NO_LIBS), want, "{name}");
    let expected = sch("a -> (a, Bool)");
    for t in [small, big] {
        let want = tree.check(&a, t, NO_LIBS, &expected);
        assert!(want.is_err(), "{name}");
        assert_eq!(
            bounded(steps).check(&a, t, NO_LIBS, &expected),
            want,
            "{name}"
        );
    }
}

#[test]
fn refining_scope_failures_are_located_in_polynomial_steps_r2() {
    assert_refining_failure_located(0, |n| 100 * n * n);
}

#[test]
fn refining_scope_failures_are_located_in_polynomial_steps_r2s8() {
    assert_refining_failure_located(1, |n| 100 * n * n);
}

#[test]
fn refining_scope_failures_are_located_in_polynomial_steps_m() {
    assert_refining_failure_located(2, |n| 100 * n * n);
}

/// `lam x. (pair U U)`, with `U = (pair (let first t_n) (let second t_n))`,
/// `t_(k+1) = (if true t_k t_k)` and `t_0 = leaf`: a chain shared under two
/// helpers, well typed under the first and ill typed under the second. The
/// second `U` is walked in place once the first has failed, so a failure
/// under the second helper that wrongly matched the first would walk the
/// first's chain as a tree.
fn chain_under_two_helpers_from(
    a: &mut Arena,
    n: usize,
    leaf: &str,
    first: &str,
    second: &str,
) -> TermId {
    let mut t = parse_term(a, leaf);
    let yes = a.bool(true);
    for _ in 0..n {
        t = a.if_(yes, t, t);
    }
    let (first, second) = (parse_term(a, first), parse_term(a, second));
    let (first, second) = (a.let_(first, t), a.let_(second, t));
    let pair = a.prim(Prim::Pair);
    let u = a.apps(pair, &[first, second]);
    let body = a.apps(pair, &[u, u]);
    a.lam(body)
}

/// Builds [`chain_under_two_helpers_from`] at `n = 8` and `n = N`, and checks
/// that inferring and checking both report the error at `leaves` (the
/// leaf's failing part when inferring, then when checking, where `x` is
/// rigid) in at most `20 N^2` steps. That is g9's bound: walking each failing
/// level in place tries the level below it afresh, so the walk is quadratic
/// in `n` (9,156 steps at `N = 40` for the first shape below), while a
/// failure that wrongly matched the well-typed helper makes it `2^n`. Walked
/// as a tree, the big chain is `2^N` too, so both chains are compared with
/// the small one's tree walk: terms are shared in the arena, so the error is
/// at the same leaf.
fn assert_two_helpers_located(leaf: &str, first: &str, second: &str, leaves: [&str; 2]) {
    const N: usize = 40;
    let limit = 20 * (N * N) as u64;
    let mut a = Arena::new();
    let small = chain_under_two_helpers_from(&mut a, 8, leaf, first, second);
    let big = chain_under_two_helpers_from(&mut a, N, leaf, first, second);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    // The type the term would have if the chain were well typed under both
    // helpers, so that checking fails at a leaf too, not at the body.
    let expected = sch("a -> (((Int, Int), (Int, Int)), ((Int, Int), (Int, Int)))");
    let want_infer = tree.infer(&a, small, NO_LIBS);
    let want_check = tree.check(&a, small, NO_LIBS, &expected);
    let got = [want_infer.as_ref().err(), want_check.as_ref().err()];
    for (e, leaf) in got.into_iter().zip(leaves) {
        let e = e.expect("ill typed under the second helper");
        assert_eq!(e.term_text, leaf, "{e}");
    }
    for t in [small, big] {
        assert_eq!(bounded(limit).infer(&a, t, NO_LIBS), want_infer);
        assert_eq!(bounded(limit).check(&a, t, NO_LIBS, &expected), want_check);
    }
}

/// The helpers are `a -> a`, closed, and `a -> x`. A failure under the
/// second says nothing about the first, whose type quantifies the variable
/// that the second leaves free. Inferring, `x` becomes `Int` at `(add ($0 1)
/// 1)`, so `($0 true)` fails; checking, `x` is rigid, so `($0 1)` does.
#[test]
fn a_failure_under_one_helper_does_not_match_another() {
    assert_two_helpers_located(
        "(pair (add ($0 1) 1) (if ($0 true) 1 1))",
        "(lam $0)",
        "(lam $1)",
        ["($0 true)", "($0 1)"],
    );
}

/// The helpers are `a -> (x, a)` and `a -> (a, x)`: both open, over the same
/// free variable, so their patterns write out the same types for it and
/// differ only in the binding's own type. Inferring, `x` becomes `Int` at
/// the `add`, so the condition of `if` fails; checking, `x` is rigid, so
/// the argument of `add` does.
#[test]
fn a_failure_under_one_open_helper_does_not_match_another_of_another_shape() {
    assert_two_helpers_located(
        "(pair (add (snd ($0 1)) 1) (if (snd ($0 true)) 1 1))",
        "(lam (pair $1 $0))",
        "(lam (pair $0 $1))",
        ["(snd ($0 true))", "(snd ($0 1))"],
    );
}

/// `lam x. let h = (lam (let (pair x x) (let (pair $0 $0) ... (pair $0
/// arg)))) in t_n`, with `m` `let`s in the helper and `t_n` the chain of s2:
/// a helper whose type is a DAG of `m` nodes over one free variable.
fn chain_under_a_large_helper(a: &mut Arena, n: usize, m: usize) -> TermId {
    let width = u32::try_from(m).expect("a small helper");
    let mut src = format!("(pair $0 ${width})");
    for k in (1..=m).rev() {
        let bound = if k == 1 {
            "(pair $1 $1)"
        } else {
            "(pair $0 $0)"
        };
        src = format!("(let {bound} {src})");
    }
    let helper = format!("(lam {src})");
    chain_with_conds(a, n, 1, &helper, |_| "true".into(), S2_LEAF, None)
}

#[test]
fn locating_under_a_large_helper_does_not_rewrite_its_type_at_every_use() {
    // Each level walked in place looks up every level below it, so a lookup
    // that wrote out the helper's whole type would cost n^2 m.
    const N: usize = 60;
    const M: usize = 200;
    let mut a = Arena::new();
    let small = chain_under_a_large_helper(&mut a, 6, 4);
    let big = chain_under_a_large_helper(&mut a, N, M);
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, small, NO_LIBS);
    assert!(want.is_err());
    assert_eq!(TypeChecker::new().infer(&a, small, NO_LIBS), want);
    let mut tc = TypeChecker::with_work_budget(u64::MAX);
    let want = tree.infer(&a, big, NO_LIBS);
    assert_eq!(tc.infer(&a, big, NO_LIBS), want);
    assert!(tc.spent() <= LARGE_HELPER_UNITS, "{} units", tc.spent());
}

/// About twice the units measured with patterns that cost what the free
/// variables have become (526,527); writing out the whole helper took 4,581,403.
const LARGE_HELPER_UNITS: u64 = 1_000_000;

#[test]
fn shared_subterms_with_many_free_variables_agree_with_the_tree() {
    // More free variables than the memo records one by one: the scope is all the
    // bindings out to the outermost one used.
    const VARS: u32 = 70;
    let mut a = Arena::new();
    let mut t = a.int(1);
    let pair = a.prim(Prim::Pair);
    for i in 0..VARS {
        let v = a.var(i);
        t = a.apps(pair, &[v, t]);
    }
    // In one scope twice, and in a scope one binder further in.
    let l = a.lam(t);
    let both = a.apps(pair, &[t, t]);
    let mut f = a.apps(pair, &[both, l]);
    for _ in 0..VARS {
        f = a.lam(f);
    }
    let mut tree = TypeChecker::new();
    tree.tree_walk = true;
    let want = tree.infer(&a, f, NO_LIBS);
    assert!(want.is_ok(), "{want:?}");
    assert_eq!(bounded(10_000).infer(&a, f, NO_LIBS), want);
}

#[test]
fn shared_subterms_stay_polymorphic_and_constrain_their_scope() {
    // The two `(lam $0)` are one TermId, used at Int and at Bool.
    assert_infers(
        "((lam (lam (pair ($1 1) ($0 true)))) (lam $0) (lam $0))",
        "(Int, Bool)",
    );
    // An open shared subterm: `(snd (pair $0 nil))` is one TermId, in the same scope,
    // used as a List Int and as a List Bool.
    assert_infers(
        "(lam ((lam (lam (pair $1 $0))) (cons 1 (snd (pair $0 nil))) \
         (cons true (snd (pair $0 nil)))))",
        "a -> (List Int, List Bool)",
    );
    // A shared subterm still constrains the variables it uses, at each use.
    assert_infers("(lam (pair (add $0 1) (add $0 1)))", "Int -> (Int, Int)");
    assert_infers("(lam (pair (fst $0) (fst $0)))", "(a, b) -> (a, a)");
    let e = expect_err("(lam (pair (fst $0) (and (fst $0) (add (fst $0) 1))))");
    assert!(matches!(*e.kind, TypeErrorKind::Mismatch { .. }), "{e}");
    // The same subterm under different scopes is not confused.
    assert_infers(
        "(lam (pair (fst $0) (lam (fst $0))))",
        "(a, b) -> (a, (c, d) -> c)",
    );
}

/// Random terms with many shared subterms: the same subterm twice in one form,
/// under the same binders or in different scopes.
fn arb_shared_src() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        (0u32..3).prop_map(|i| format!("${i}")),
        (-1i64..2).prop_map(|n| n.to_string()),
        Just("true".to_string()),
        Just("nil".to_string()),
        proptest::sample::select(Prim::ALL).prop_map(|p| p.name().to_string()),
    ];
    leaf.prop_recursive(6, 64, 3, |inner| {
        prop_oneof![
            3 => inner.clone().prop_map(|b| format!("(lam {b})")),
            3 => (inner.clone(), inner.clone()).prop_map(|(f, a)| format!("({f} {a})")),
            1 => (inner.clone(), inner.clone()).prop_map(|(x, b)| format!("(let {x} {b})")),
            2 => (inner.clone(), inner.clone()).prop_map(|(c, t)| format!("(if {c} {t} {t})")),
            2 => inner.clone().prop_map(|t| format!("((lam (lam (pair $1 $0))) {t} {t})")),
            1 => inner.clone().prop_map(|t| format!("(pair {t} (lam {t}))")),
            1 => inner.clone().prop_map(|t| format!("((lam {t}) {t})")),
            1 => inner.clone().prop_map(|t| format!("(if true {t} (let (lam $0) {t}))")),
            // A helper whose type has the free variable of the lambda around it.
            1 => inner.clone().prop_map(|t| {
                format!("(lam (if true {t} (let (lam (pair $1 $0)) {t})))")
            }),
            1 => (inner.clone(), inner.clone()).prop_map(|(t, u)| format!("(let {t} ({u} {t}))")),
            // Two spines sharing the prefix `({f} {t})`.
            1 => (inner.clone(), inner).prop_map(|(f, t)| {
                format!("(if true ({f} {t} 1) ({f} {t} true))")
            }),
        ]
    })
}

#[test]
fn reusing_shared_subterms_agrees_with_walking_the_tree() {
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(4000));
    let (typed, untyped) = (std::cell::Cell::new(0u32), std::cell::Cell::new(0u32));
    runner
        .run(&arb_shared_src(), |src| {
            let mut a = Arena::new();
            let id = sexpr::parse(&mut a, &src).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut memo = TypeChecker::new();
            let mut tree = TypeChecker::new();
            tree.tree_walk = true;
            let got = memo.infer(&a, id, NO_LIBS);
            let want = tree.infer(&a, id, NO_LIBS);
            // Errors too: an error is located by walking the tree again.
            prop_assert_eq!(&got, &want, "{}", src);
            if let Ok(s) = &want {
                typed.set(typed.get() + 1);
                prop_assert!(memo.check(&a, id, NO_LIBS, s).is_ok(), "{}", src);
                // Checking against a less general type agrees too.
                for t in ["Int", "Bool -> Bool", "List Int"] {
                    let s = sch(t);
                    prop_assert_eq!(
                        memo.check(&a, id, NO_LIBS, &s),
                        tree.check(&a, id, NO_LIBS, &s),
                        "{} against {}",
                        src,
                        t
                    );
                }
            } else {
                untyped.set(untyped.get() + 1);
            }
            Ok(())
        })
        .unwrap();
    // Both outcomes must be exercised for the comparison to mean anything.
    assert!(typed.get() > 400, "only {} well typed", typed.get());
    assert!(untyped.get() > 400, "only {} ill typed", untyped.get());
}

/// Random terms under `lam x. let h = (lam (pair x $0)) in ...`, whose
/// shared subterms use polymorphic helpers whose types have the free variable
/// `x`, and constrain `x` in different ways, so that whether a shared subterm
/// is typable depends on what `x` has been found to be.
fn arb_open_helper_src() -> impl Strategy<Value = String> {
    let leaf = prop_oneof![
        (0u32..3).prop_map(|i| format!("${i}")),
        (-1i64..2).prop_map(|n| n.to_string()),
        Just("true".to_string()),
        Just("(fst ($0 1))".to_string()),
        Just("(snd ($0 true))".to_string()),
        Just("(add (fst ($0 1)) 1)".to_string()),
        Just("(if (fst ($0 1)) true true)".to_string()),
        Just("(add (snd ($0 1)) true)".to_string()),
        Just("($0 (fst ($0 1)))".to_string()),
        proptest::sample::select(Prim::ALL).prop_map(|p| p.name().to_string()),
    ];
    let body = leaf.prop_recursive(6, 64, 3, |inner| {
        prop_oneof![
            2 => inner.clone().prop_map(|b| format!("(lam {b})")),
            2 => (inner.clone(), inner.clone()).prop_map(|(f, a)| format!("({f} {a})")),
            3 => (inner.clone(), inner.clone()).prop_map(|(c, t)| format!("(if {c} {t} {t})")),
            2 => (inner.clone(), inner.clone()).prop_map(|(u, t)| format!("(pair {u} {t})")),
            2 => inner.clone().prop_map(|t| format!("(if true {t} (let (lam (pair (fst ($0 0)) $0)) {t}))")),
            1 => inner.clone().prop_map(|t| format!("(let (lam (pair $1 $0)) {t})")),
            1 => (inner.clone(), inner).prop_map(|(x, b)| format!("(let {x} (pair {b} {b}))")),
        ]
    });
    body.prop_map(|b| format!("(lam (let (lam (pair $1 $0)) {b}))"))
}

/// Compares the errors of the memoised walk and the tree walk. Which keys are
/// marked failed changes only the work done, not the error (see
/// `TypeChecker::note_failure`), so this test does not notice if keys stop
/// being marked or are dropped too soon: the step bounds of
/// `open_scope_failures_are_located_in_polynomial_steps_*` guard that.
#[test]
fn shared_subterms_under_open_helpers_agree_with_walking_the_tree() {
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(4000));
    let (typed, untyped) = (std::cell::Cell::new(0u32), std::cell::Cell::new(0u32));
    runner
        .run(&arb_open_helper_src(), |src| {
            let mut a = Arena::new();
            let id = sexpr::parse(&mut a, &src).map_err(|e| TestCaseError::fail(e.to_string()))?;
            let mut memo = TypeChecker::new();
            let mut tree = TypeChecker::new();
            tree.tree_walk = true;
            let want = tree.infer(&a, id, NO_LIBS);
            prop_assert_eq!(&memo.infer(&a, id, NO_LIBS), &want, "{}", src);
            if want.is_ok() {
                typed.set(typed.get() + 1);
            } else {
                untyped.set(untyped.get() + 1);
            }
            for t in ["Int -> (Int, Bool)", "Bool -> Bool", "a -> a"] {
                let s = sch(t);
                prop_assert_eq!(
                    memo.check(&a, id, NO_LIBS, &s),
                    tree.check(&a, id, NO_LIBS, &s),
                    "{} against {}",
                    src,
                    t
                );
            }
            Ok(())
        })
        .unwrap();
    // Both outcomes must be exercised for the comparison to mean anything.
    assert!(typed.get() > 400, "only {} well typed", typed.get());
    assert!(untyped.get() > 400, "only {} ill typed", untyped.get());
}

// ---------------------------------------------------------------- check locates errors

#[test]
fn check_reports_the_subterm_where_the_expected_type_clashes() {
    // The expected parameter type reaches the memorec key argument.
    let e = check_src("(lam (memorec 5 (lam (lam 0)) $0))", "List Int -> Int").unwrap_err();
    assert!(matches!(&*e.kind, TypeErrorKind::BadMemoKey { .. }), "{e}");
    assert_eq!(e.term_text, "$0", "{e}");
    assert!(e.context.contains("argument 3 of `memorec`"), "{e}");

    // The expected result type reaches the branches of an if.
    let e = check_src("(lam (lam (if $1 1 true)))", "Bool -> a -> Int").unwrap_err();
    assert_eq!(e.term_text, "true", "{e}");
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: Type::Int,
            found: Type::Bool,
            conflict: None,
        }
    );

    // ... and the result of an application, before its arguments are looked at.
    let e = check_src("(lam (let 3 (add $1 $0)))", "Int -> Bool").unwrap_err();
    assert_eq!(e.term_text, "(add $1 $0)", "{e}");
    assert!(e.context.contains("expected type"), "{e}");

    // A lambda where the expected type is not a function is reported at the lambda.
    let e = check_src("(if true (lam $0) 1)", "Int").unwrap_err();
    assert_eq!(e.term_text, "(lam $0)", "{e}");
    assert!(e.context.contains("then branch"), "{e}");

    // An over-applied head is still reported as such, not as a result mismatch.
    let e = check_src("(1 2)", "Int").unwrap_err();
    assert!(
        matches!(*e.kind, TypeErrorKind::NotAFunction { accepted: 0, .. }),
        "{e}"
    );

    // Operands of and/or keep their own positions.
    let e = check_src("(lam (and $0 1))", "Bool -> Bool").unwrap_err();
    assert_eq!(e.term_text, "1", "{e}");
    assert!(e.context.contains("second operand of `and`"), "{e}");
}

#[test]
fn bounded_printing_agrees_with_sexpr_print() {
    let mut a = Arena::new();
    for src in [
        "(lam (foldl (lam (lam (add $1 $0))) 0 $0))",
        "(let (lam $0) (if (and true (or false true)) ($0 unit) lib_3))",
        "((lam $0) -7 (pair 1 2))",
    ] {
        let id = parse_term(&mut a, src);
        assert_eq!(super::infer::short(&a, id), src);
    }
}

proptest! {
    #[test]
    fn bounded_printing_is_a_prefix_of_sexpr_print(src in arb_src()) {
        let mut a = Arena::new();
        let id = parse_term(&mut a, &src);
        let full = sexpr::print(&a, id).unwrap();
        let short = super::infer::short(&a, id);
        if full.len() <= 200 {
            prop_assert_eq!(short, full);
        } else {
            let kept = short.strip_suffix(" ...").expect("long text ends with ...");
            prop_assert!(full.starts_with(kept));
            prop_assert_eq!(kept.len(), 200);
        }
    }
}

/// A well-typed closed term of type `Int` whose shared subterms see exponentially
/// many different scopes: `t_0 = (let (pair $0 (pair $1 ... unit)) 0)` uses all
/// `n` bindings, and `t_{k+1} = (let (pair (let (lam $0) t_k) (let (lam (lam $1))
/// t_k)) 0)`. Each path to `t_0` binds every variable to one of two closed
/// polymorphic helpers of different types, so `t_k` is inferred for up to
/// `2^(n-k)` scopes. Returns `t_n`.
fn helper_choices(a: &mut Arena, n: usize) -> TermId {
    let mut pairs = String::from("unit");
    for i in (0..n).rev() {
        pairs = format!("(pair ${i} {pairs})");
    }
    let mut t = parse_term(a, &format!("(let {pairs} 0)"));
    let pair = a.prim(Prim::Pair);
    let id = parse_term(a, "(lam $0)");
    let konst = parse_term(a, "(lam (lam $1))");
    let zero = a.int(0);
    for _ in 0..n {
        let l = a.let_(id, t);
        let r = a.let_(konst, t);
        let p = a.apps(pair, &[l, r]);
        t = a.let_(p, zero);
    }
    t
}

fn too_much_work(limit: u64) -> TypeErrorKind {
    TypeErrorKind::TooMuchWork { limit }
}

#[test]
fn exponential_work_is_refused_quickly() {
    // 151 distinct nodes. Without a budget this takes about 8 s in a release
    // build and minutes in a debug one; the default budget stops it in about
    // 0.2 s in a release build (debug builds are about 18 times slower).
    let mut a = Arena::new();
    let t = helper_choices(&mut a, 18);
    let start = std::time::Instant::now();
    let e = infer(&a, t, NO_LIBS).unwrap_err();
    let elapsed = start.elapsed();
    assert_eq!(*e.kind, too_much_work(DEFAULT_WORK_BUDGET), "{e}");
    assert!(e.to_string().contains("units of work"), "{e}");
    let limit = if cfg!(debug_assertions) { 15 } else { 1 };
    assert!(
        elapsed < std::time::Duration::from_secs(limit),
        "{elapsed:?}"
    );
}

#[test]
fn default_budget_admits_normal_programs() {
    let mut a = Arena::new();
    // Two to the eight scopes for the innermost shared subterm: real work, far
    // below the budget.
    let t = helper_choices(&mut a, 8);
    let mut tc = TypeChecker::new();
    assert_eq!(tc.work_budget(), DEFAULT_WORK_BUDGET);
    assert_eq!(tc.infer(&a, t, NO_LIBS), Ok(sch("Int")));
    assert_eq!(tc.check(&a, t, NO_LIBS, &sch("Int")), Ok(()));
    let src = "(lam (foldl (lam (lam (add $1 (length (array_to_list (array_of_list $0)))))) \
               0 $0))";
    assert_infers(src, "List (List a) -> Int");
}

#[test]
fn a_smaller_budget_refuses_what_the_default_admits() {
    let mut a = Arena::new();
    let t = helper_choices(&mut a, 8);
    assert_eq!(infer(&a, t, NO_LIBS), Ok(sch("Int")));
    let mut tc = TypeChecker::with_work_budget(1_000);
    assert_eq!(tc.work_budget(), 1_000);
    let e = tc.infer(&a, t, NO_LIBS).unwrap_err();
    assert_eq!(*e.kind, too_much_work(1_000), "{e}");
    let e = tc.check(&a, t, NO_LIBS, &sch("Int")).unwrap_err();
    assert_eq!(*e.kind, too_much_work(1_000), "{e}");
    // A small term fits in the small budget, and the budget applies per call.
    let one = parse_term(&mut a, "(add 1 2)");
    assert_eq!(tc.infer(&a, one, NO_LIBS), Ok(sch("Int")));
    assert_eq!(tc.infer(&a, one, NO_LIBS), Ok(sch("Int")));
    // Raising the budget again admits the term on the same checker.
    tc.set_work_budget(DEFAULT_WORK_BUDGET);
    assert_eq!(tc.infer(&a, t, NO_LIBS), Ok(sch("Int")));
}

/// `(pair p_1 (pair p_2 ... (pair p_l unit)))` where `p_0 = 0` and `p_i = (p_{i-1}
/// 0)`: every prefix of the longest application spine is also a child elsewhere,
/// so the spines of the distinct nodes have about `l^2 / 2` children in all.
fn spine_prefixes(a: &mut Arena, l: usize) -> TermId {
    let zero = a.int(0);
    let mut prefixes = Vec::with_capacity(l);
    let mut p = zero;
    for _ in 0..l {
        p = a.app(p, zero);
        prefixes.push(p);
    }
    let pair = a.prim(Prim::Pair);
    let mut t = a.unit();
    for &q in prefixes.iter().rev() {
        t = a.apps(pair, &[q, t]);
    }
    t
}

#[test]
fn finding_shared_subterms_is_charged_to_the_budget() {
    // Children of all the spines: about 125 000, against a budget of 10 000. The
    // search for shared subterms runs before the first step of the term walk, so
    // it must refuse the term itself: the walk would stop at its sixth step.
    let mut a = Arena::new();
    let t = spine_prefixes(&mut a, 500);
    let mut tc = TypeChecker::with_work_budget(10_000);
    tc.step_limit = Some(5);
    let e = tc.infer(&a, t, NO_LIBS).unwrap_err();
    assert_eq!(*e.kind, too_much_work(10_000), "{e}");
    // With enough budget the term is checked, and the search is paid for.
    let mut tc = TypeChecker::new();
    assert!(
        tc.infer(&a, t, NO_LIBS).is_err(),
        "applying an Int is an error"
    );
    assert!(tc.spent() >= 500 * 499 / 2, "{}", tc.spent());
}

/// `d` lets of `(lam $0)` around `c_u`, where `c_0 = 0` and `c_{k+1} = (let (pair
/// s c_k) 0)`. `s = (pair $0 (pair $1 ... (pair $65 $(d-1))))` has more than
/// [`MAX_FREE_VARS`] free variables, so its scope has a slot for each of the `d`
/// bindings, and it is used `u` times.
fn wide_scope_uses(a: &mut Arena, d: u32, u: usize) -> TermId {
    let pair = a.prim(Prim::Pair);
    let id = parse_term(a, "(lam $0)");
    let mut s = a.var(d - 1);
    for i in (0..66u32).rev() {
        let v = a.var(i);
        s = a.apps(pair, &[v, s]);
    }
    let zero = a.int(0);
    let mut c = zero;
    for _ in 0..u {
        let p = a.apps(pair, &[s, c]);
        c = a.let_(p, zero);
    }
    let mut t = c;
    for _ in 0..d {
        t = a.let_(id, t);
    }
    t
}

#[test]
fn each_use_of_a_shared_subterm_pays_for_its_scope() {
    // Each use builds, hashes and compares a key of 20 000 slots. Without
    // charging for that, the call spends about 470 000 units, under 20 000 a use.
    let (d, u) = (20_000, 50);
    let mut a = Arena::new();
    let t = wide_scope_uses(&mut a, d, u);
    let mut tc = TypeChecker::new();
    assert_eq!(tc.infer(&a, t, NO_LIBS), Ok(sch("Int")));
    assert!(tc.spent() >= u64::from(d) * u as u64, "{}", tc.spent());
}

#[test]
fn running_out_while_locating_returns_the_first_error() {
    // The first walk reports the clash at the whole shared subterm; the second
    // walk names `$0`. The smallest budget that does not refuse the term is one
    // the first walk fits in and the second does not, so it gives the first
    // walk's error.
    let mut a = Arena::new();
    let t = parse_term(&mut a, "(lam (if $0 (add $0 1) (add $0 1)))");
    let located = TypeChecker::new().infer(&a, t, NO_LIBS).unwrap_err();
    assert_eq!(located.term_text, "$0", "{located}");
    let result = |budget: u64| TypeChecker::with_work_budget(budget).infer(&a, t, NO_LIBS);
    let (mut lo, mut hi) = (1u64, 1_000_000u64);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match result(mid) {
            Err(e) if *e.kind == too_much_work(mid) => lo = mid + 1,
            _ => hi = mid,
        }
    }
    let first = result(lo).unwrap_err();
    assert_eq!(first.term_text, "(add $0 1)", "{first}");
    assert!(first.context.contains("shared subterm"), "{first}");
}

/// Reads of the term arena, the type store and schemes, table slots visited
/// and hash-table probes on this thread so far.
fn reads() -> u64 {
    crate::term::GETS.with(std::cell::Cell::get)
        + super::infer::STORE_READS.with(std::cell::Cell::get)
        + super::infer::SCHEME_READS.with(std::cell::Cell::get)
        + super::infer::TABLE_SLOTS.with(std::cell::Cell::get)
        + super::infer::PROBES.with(std::cell::Cell::get)
}

/// Infers `t` with `tc` against [`adversarial_libs`] and returns the reads it
/// took and the units it was charged.
fn reads_and_units(tc: &mut TypeChecker, a: &Arena, t: TermId) -> (u64, u64) {
    let before = reads();
    let _ = tc.infer(a, t, &adversarial_libs());
    (reads() - before, tc.spent())
}

/// Checks `t` with `tc` against `expected` and returns the reads it took and
/// the units it was charged.
fn check_reads_and_units(
    tc: &mut TypeChecker,
    a: &Arena,
    t: TermId,
    expected: &Scheme,
) -> (u64, u64) {
    let libs = adversarial_libs();
    let before = reads();
    let _ = tc.check(a, t, &libs, expected);
    (reads() - before, tc.spent())
}

/// A balanced tree of pairs over the variables `lo..hi`.
fn pair_tree(lo: u32, hi: u32) -> Type {
    if hi - lo == 1 {
        Type::Var(TyVar(lo))
    } else {
        let mid = lo + (hi - lo) / 2;
        Type::pair(pair_tree(lo, mid), pair_tree(mid, hi))
    }
}

/// The number of variables of [`large_key_scheme`]: its type has `2v - 1`
/// nodes, under [`MAX_TYPE_NODES`].
const LARGE_SCHEME_VARS: u32 = 4_096;

/// A scheme of [`LARGE_SCHEME_VARS`] variables, each with a `Key` constraint,
/// in a balanced tree of pairs.
fn large_key_scheme() -> Scheme {
    let keys: Vec<TyVar> = (0..LARGE_SCHEME_VARS).map(TyVar).collect();
    Scheme::with_keys(pair_tree(0, LARGE_SCHEME_VARS), &keys)
}

/// The library the adversarial shapes may use: `lib0` has
/// [`large_key_scheme`].
fn adversarial_libs() -> Vec<Scheme> {
    vec![large_key_scheme()]
}

/// `c_n`, where `c_0 = unit` and `c_{k+1} = (snd (pair lib0 c_k))`: `n` uses of
/// a library with a large scheme. Type `Unit`.
fn large_scheme_uses(a: &mut Arena, n: usize) -> TermId {
    let lib = a.lib(LibId(0));
    let pair = a.prim(Prim::Pair);
    let snd = a.prim(Prim::Snd);
    let mut c = a.unit();
    for _ in 0..n {
        let p = a.apps(pair, &[lib, c]);
        c = a.app(snd, p);
    }
    c
}

/// `((lam ($0 0 ... 0)) k)` with `l` arguments, where `k = (lam (lam ... 0))` is
/// `l` deep: the head of the spine has a variable type, so each argument
/// unifies it with an arrow. Type `Int`.
fn variable_head_spine(a: &mut Arena, l: usize) -> TermId {
    let zero = a.int(0);
    let v = a.var(0);
    let body = a.apps(v, &vec![zero; l]);
    let f = a.lam(body);
    let mut k = zero;
    for _ in 0..l {
        k = a.lam(k);
    }
    a.app(f, k)
}

/// `c_m`, where `c_0 = 0` and `c_{k+1} = (h c_k)` with `h = (lam (i i ... i))`, `l`
/// arguments, and `i = (lam $0)`: a head whose text has a long spine, applied
/// `m` times.
fn long_head_chain(a: &mut Arena, l: usize, m: usize) -> TermId {
    let i = parse_term(a, "(lam $0)");
    let body = a.apps(i, &vec![i; l]);
    let h = a.lam(body);
    let mut c = a.int(0);
    for _ in 0..m {
        c = a.app(h, c);
    }
    c
}

/// `c_d`, where `c_0 = (0 unit ... unit)` with `l` arguments and `c_{k+1} = (if
/// true c_k c_k)`: ill typed at the bottom of `d` levels of sharing.
fn shared_bad_spine(a: &mut Arena, l: usize, d: usize) -> TermId {
    let zero = a.int(0);
    let unit = a.unit();
    let mut c = a.apps(zero, &vec![unit; l]);
    let t = a.bool(true);
    for _ in 0..d {
        c = a.if_(t, c, c);
    }
    c
}

/// `c_d`, where `c_0 = 0` and `c_{k+1} = (if true c_k c_k)`.
fn if_nest(a: &mut Arena, d: usize) -> TermId {
    let mut c = a.int(0);
    let t = a.bool(true);
    for _ in 0..d {
        c = a.if_(t, c, c);
    }
    c
}

/// `(lam c_d)`, where `c_0 = (add $0 1)` and `c_{k+1} = (if true c_k c_k)`:
/// each shared subterm has a free variable.
fn open_if_nest(a: &mut Arena, d: usize) -> TermId {
    let add = a.prim(Prim::Add);
    let (v0, one, t) = (a.var(0), a.int(1), a.bool(true));
    let mut c = a.apps(add, &[v0, one]);
    for _ in 0..d {
        c = a.if_(t, c, c);
    }
    a.lam(c)
}

/// `c_{n-1}`, where `c_0 = 0` and `c_i = (if true c_{i-1} c_r)` for an `r < i`
/// drawn from a fixed pseudo-random sequence: a DAG whose shared nodes are
/// reached in no particular order.
fn random_if_dag(a: &mut Arena, n: usize) -> TermId {
    let t = a.bool(true);
    let mut cs = vec![a.int(0)];
    let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
    for i in 1..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let r = (x % i as u64) as usize;
        let c = a.if_(t, cs[i - 1], cs[r]);
        cs.push(c);
    }
    cs[n - 1]
}

/// Under `n` lambdas, `(if (eq $0 $1) (if (eq $1 $2) ... (if $(n-1) c_d 0) ...
/// 0) 0)`, where `c_0 = (add $0 1)` and `c_{k+1} = (if true c_k c_k)`: the
/// conditions link the variables of the lambdas in a chain, the last fixes
/// them to `Bool`, and `c_d` is ill typed at the bottom of `d` levels of
/// sharing, so the second walk goes back over the chain at each level.
fn chain_rollback(a: &mut Arena, n: u32, d: usize) -> TermId {
    let add = a.prim(Prim::Add);
    let eq = a.prim(Prim::Eq);
    let (zero, one, t) = (a.int(0), a.int(1), a.bool(true));
    let v0 = a.var(0);
    let mut c = a.apps(add, &[v0, one]);
    for _ in 0..d {
        c = a.if_(t, c, c);
    }
    let last = a.var(n - 1);
    let mut body = a.if_(last, c, zero);
    for i in (0..n - 1).rev() {
        let (x, y) = (a.var(i), a.var(i + 1));
        let cond = a.apps(eq, &[x, y]);
        body = a.if_(cond, body, zero);
    }
    for _ in 0..n {
        body = a.lam(body);
    }
    body
}

/// The most reads of the arena and the store a unit of work may stand for.
const READS_PER_UNIT: u64 = 4;

/// The most units a call may spend past twice a small budget. The step that
/// runs past the budget is paid in full: a unification can visit every node of
/// the store, which is no larger than the units spent so far, and an error can
/// write out a few types of at most [`MAX_TYPE_NODES`] nodes.
const OVERRUN: u64 = 4 * MAX_TYPE_NODES as u64 + 1_000;

/// Terms whose checking once did work that was not charged to the budget.
fn adversarial_shapes(a: &mut Arena) -> Vec<(&'static str, TermId)> {
    vec![
        ("variable head spine", variable_head_spine(a, 2_000)),
        ("long head chain", long_head_chain(a, 2_000, 2_000)),
        ("shared bad spine", shared_bad_spine(a, 20_000, 100)),
        ("spine prefixes", spine_prefixes(a, 2_000)),
        ("wide scope uses", wide_scope_uses(a, 2_000, 20)),
        ("helper choices", helper_choices(a, 10)),
        ("if nest", if_nest(a, 100_000)),
        ("large scheme uses", large_scheme_uses(a, 20)),
        ("chain rollback", chain_rollback(a, 2_000, 2_000)),
    ]
    .into_iter()
    .chain(open_scope_failures(a, 2_000))
    .chain(refining_scope_failures(a, 100))
    .collect()
}

#[test]
fn every_read_of_the_term_or_the_store_is_charged() {
    // A judge for passes the budget does not see: each unit charged must
    // stand for a bounded number of reads of the term arena and the type
    // store, whatever the term.
    let mut a = Arena::new();
    let shapes = adversarial_shapes(&mut a);
    let normal = parse_term(
        &mut a,
        "(lam (foldl (lam (lam (add $1 (length (array_to_list (array_of_list $0)))))) 0 $0))",
    );
    for (name, t) in shapes.into_iter().chain([("normal", normal)]) {
        let mut tc = TypeChecker::new();
        let (reads, units) = reads_and_units(&mut tc, &a, t);
        assert!(
            reads <= READS_PER_UNIT * units + 1_000,
            "{name}: {reads} reads for {units} units"
        );
    }
}

#[test]
fn a_small_budget_bounds_the_work_on_any_term() {
    // The work of a call that runs out is bounded by its budget, not by the
    // size of the term: the search for shared subterms, the collection of a
    // spine and the error text all stop.
    let budget = 1_000;
    let mut a = Arena::new();
    let mut shapes = adversarial_shapes(&mut a);
    let zero = a.int(0);
    shapes.push(("long spine", a.apps(zero, &vec![zero; 100_000])));
    // The tree walk skips the search, so it shows the term walk's own checks.
    for tree_walk in [false, true] {
        for &(name, t) in &shapes {
            let mut tc = TypeChecker::with_work_budget(budget);
            tc.tree_walk = tree_walk;
            let (reads, units) = reads_and_units(&mut tc, &a, t);
            assert!(
                units <= 2 * budget + OVERRUN,
                "{name} (tree walk {tree_walk}): {units} units against a budget of {budget}"
            );
            assert!(
                reads <= READS_PER_UNIT * units + 1_000,
                "{name} (tree walk {tree_walk}): {reads} reads for {units} units"
            );
        }
    }
}

#[test]
fn the_search_and_free_variables_are_charged_at_any_budget() {
    // The search for shared subterms and the finding of the free variables of
    // a shared subterm each probe hash tables a few times per node. A budget
    // that stops the call soon after them leaves their reads to be paid by
    // their own units, not by those of the term walk after them. Both stop
    // between nodes, so the reads and units a larger budget adds are those of
    // the passes it reaches, and each pass must pay for its own reads. Table
    // and stack slots freed at the end are left out of what is added: a table
    // doubles at once, and is paid for by all the units that filled it.
    let slots = || super::infer::TABLE_SLOTS.with(std::cell::Cell::get);
    let mut a = Arena::new();
    let shapes = [
        ("if nest", if_nest(&mut a, 20_000)),
        ("open if nest", open_if_nest(&mut a, 20_000)),
        ("random if DAG", random_if_dag(&mut a, 20_000)),
    ];
    for (name, t) in shapes {
        let mut tc = TypeChecker::new();
        let _ = reads_and_units(&mut tc, &a, t);
        let full = tc.spent();
        let mut last = (0, 0);
        for i in 1..=60 {
            let budget = full * i / 60;
            let mut tc = TypeChecker::with_work_budget(budget);
            let before = slots();
            let (reads, units) = reads_and_units(&mut tc, &a, t);
            assert!(
                reads <= READS_PER_UNIT * units + 1_000,
                "{name}, budget {budget}: {reads} reads for {units} units"
            );
            let reads = reads - (slots() - before);
            let (more_reads, more_units) = (reads.saturating_sub(last.0), units - last.1);
            assert!(
                more_reads <= READS_PER_UNIT * more_units + 1_000,
                "{name}, budget {budget}: {more_reads} more reads for {more_units} more units"
            );
            last = (reads, units);
        }
    }
}

#[test]
fn each_hash_table_probe_is_charged() {
    // Hash-table probes miss the cache on large terms, so they cost more than
    // the other reads and the time of a unit follows them. Within the slack of
    // READS_PER_UNIT a pass could probe several times per unit unseen; this
    // holds probes alone to one per unit.
    let probes = || super::infer::PROBES.with(std::cell::Cell::get);
    let mut a = Arena::new();
    let mut shapes = adversarial_shapes(&mut a);
    shapes.push(("open if nest", open_if_nest(&mut a, 20_000)));
    shapes.push(("random if DAG", random_if_dag(&mut a, 20_000)));
    for (name, t) in shapes {
        let mut tc = TypeChecker::new();
        let before = probes();
        let _ = reads_and_units(&mut tc, &a, t);
        let (probed, units) = (probes() - before, tc.spent());
        assert!(
            probed <= units + 1_000,
            "{name}: {probed} probes for {units} units"
        );
    }
}

#[test]
fn checking_against_a_large_key_scheme_is_charged() {
    // Each variable of the expected type and of a library's scheme is made
    // once per walk or use, and whether it has a `Key` constraint is looked
    // up: the lookup must cost a bounded number of reads per unit charged,
    // whatever the number of keys.
    let mut a = Arena::new();
    let t = a.lib(LibId(0));
    let expected = large_key_scheme();
    for budget in [1_000, DEFAULT_WORK_BUDGET] {
        let mut tc = TypeChecker::with_work_budget(budget);
        let (reads, units) = check_reads_and_units(&mut tc, &a, t, &expected);
        assert!(
            reads <= READS_PER_UNIT * units + 1_000,
            "budget {budget}: {reads} reads for {units} units"
        );
        assert!(
            units <= 2 * budget + OVERRUN,
            "{units} units against a budget of {budget}"
        );
    }
    let libs = adversarial_libs();
    assert_eq!(TypeChecker::new().check(&a, t, &libs, &expected), Ok(()));
}

/// A balanced tree of `n` pairs of `Int`, whose last leaf is `last`.
fn closed_pair_tree(n: u32, last: &Type) -> Type {
    fn go(lo: u32, hi: u32, n: u32, last: &Type) -> Type {
        if hi - lo == 1 {
            if hi == n {
                last.clone()
            } else {
                Type::Int
            }
        } else {
            let mid = lo + (hi - lo) / 2;
            Type::pair(go(lo, mid, n, last), go(mid, hi, n, last))
        }
    }
    go(0, n, n, last)
}

#[test]
fn a_step_that_errors_past_the_budget_reports_too_much_work() {
    // The unification of the branches of the `if` is one step that visits
    // both large types and fails at their last leaves. Under a budget it runs
    // past, the call is out of work, not ill typed.
    let libs = vec![
        Scheme::new(closed_pair_tree(4_096, &Type::Int)),
        Scheme::new(closed_pair_tree(4_096, &Type::Bool)),
    ];
    let mut a = Arena::new();
    let t = parse_term(&mut a, "(if true lib_0 lib_1)");
    let mut tc = TypeChecker::new();
    let e = tc.infer(&a, t, &libs).unwrap_err();
    assert!(!matches!(*e.kind, TypeErrorKind::TooMuchWork { .. }), "{e}");
    let full = tc.spent();
    let mut past = 0;
    for budget in (1..full).step_by(101) {
        let mut tc = TypeChecker::with_work_budget(budget);
        let e = tc.infer(&a, t, &libs).unwrap_err();
        if tc.spent() > budget {
            past += 1;
            assert_eq!(*e.kind, too_much_work(budget), "budget {budget}");
        }
    }
    assert!(past > 0);
}

/// `c_k`, where `c_0 = base` and `c_{j+1} = (let c_j (pair $0 $0))`: a term
/// whose type is a pair tree `k` deep with one store node per level.
fn doubling(a: &mut Arena, base: TermId, k: usize) -> TermId {
    let body = parse_term(a, "(pair $0 $0)");
    let mut c = base;
    for _ in 0..k {
        c = a.let_(c, body);
    }
    c
}

/// A balanced tree of pairs `k` deep whose `2^k` leaves are `leaf(i)`.
fn distinct_tree(a: &mut Arena, k: u32, leaf: &mut dyn FnMut(&mut Arena, i64) -> TermId) -> TermId {
    fn go(
        a: &mut Arena,
        lo: i64,
        hi: i64,
        leaf: &mut dyn FnMut(&mut Arena, i64) -> TermId,
    ) -> TermId {
        if hi - lo == 1 {
            leaf(a, lo)
        } else {
            let mid = lo + (hi - lo) / 2;
            let l = go(a, lo, mid, leaf);
            let r = go(a, mid, hi, leaf);
            let pair = a.prim(Prim::Pair);
            a.apps(pair, &[l, r])
        }
    }
    go(a, 0, 1 << k, leaf)
}

/// `(if true A B)` where `A` is a tree `k` deep of distinct nodes whose leaves
/// are distinct doublings `k` deep, and `B` is `k` doublings of a tree `k`
/// deep of distinct nodes. Both types are the full pair tree `2k` deep over
/// `Int`, but the sharing differs: below depth `k`, each of `A`'s `2^k` chains
/// meets every node of `B`'s tree, so unifying them visits `4^k` pairs of
/// store nodes, from types of about `k 2^k` nodes each. The `if` is bound by a
/// `let` whose body is `0`, as its type is too large to write out. Type `Int`.
fn crossed_sharing(a: &mut Arena, k: u32) -> TermId {
    let zero = a.int(0);
    crossed_sharing_then(a, k, zero)
}

/// [`crossed_sharing`] with `body` in place of `0` under the `let`.
fn crossed_sharing_then(a: &mut Arena, k: u32, body: TermId) -> TermId {
    let depth = k as usize;
    let left = distinct_tree(a, k, &mut |a, i| {
        let base = a.int(i);
        doubling(a, base, depth)
    });
    let tree = distinct_tree(a, k, &mut |a, i| a.int(i));
    let right = doubling(a, tree, depth);
    let t = a.bool(true);
    let both = a.if_(t, left, right);
    a.let_(both, body)
}

/// `(cons 1 (cons 1 ... nil))`, `m` deep: `m` small unifications of lists.
fn cons_chain(a: &mut Arena, m: usize) -> TermId {
    let cons = a.prim(Prim::Cons);
    let one = a.int(1);
    let mut c = a.prim(Prim::Nil);
    for _ in 0..m {
        c = a.apps(cons, &[one, c]);
    }
    c
}

#[test]
fn a_large_unification_does_not_slow_later_ones() {
    // Emptying a hash table costs time in its capacity, which a large
    // unification raises. The unifications after it, in the same call or the
    // next call on the same checker, must not pay for that each time.
    let mut a = Arena::new();
    let chain = cons_chain(&mut a, 20_000);
    let t = crossed_sharing_then(&mut a, 9, chain);
    let mut tc = TypeChecker::new();
    let (reads, units) = reads_and_units(&mut tc, &a, t);
    assert!(
        reads <= READS_PER_UNIT * units + 1_000,
        "in one call: {reads} reads for {units} units"
    );
    let (reads, units) = reads_and_units(&mut tc, &a, chain);
    assert!(
        reads <= READS_PER_UNIT * units + 1_000,
        "in the next call: {reads} reads for {units} units"
    );
}

/// `(cons (pair x_1 x_1) (cons ... nil))` with `x_i = (cons i nil)`: `n`
/// distinct shared subterms, which fill the checker's per-walk tables.
fn many_shared(a: &mut Arena, n: usize) -> TermId {
    let (cons, pair, nil) = (a.prim(Prim::Cons), a.prim(Prim::Pair), a.prim(Prim::Nil));
    let mut t = nil;
    for i in 0..n {
        let k = a.int(i as i64);
        let x = a.apps(cons, &[k, nil]);
        let p = a.apps(pair, &[x, x]);
        t = a.apps(cons, &[p, t]);
    }
    t
}

/// `(let (lam (pair $0 (pair $0 ... (pair $0 $0)))) 0)` with `k` pairs: a
/// polymorphic binding whose type has `k` different subtrees, each of which
/// the checker gives a [`Canon`](super::infer) id.
fn deep_polymorphic_let(a: &mut Arena, k: usize) -> TermId {
    let pair = a.prim(Prim::Pair);
    let v0 = a.var(0);
    let mut t = v0;
    for _ in 0..k {
        t = a.apps(pair, &[v0, t]);
    }
    let f = a.lam(t);
    let zero = a.int(0);
    a.let_(f, zero)
}

#[test]
fn a_large_call_does_not_slow_later_calls() {
    // A call fills the tables that the next call empties. Clearing a full
    // table costs time in its capacity, and so does freeing a large empty
    // one, so the small calls after a large one must not inherit its tables.
    // Each large call fills a different table: the memo and the shared
    // subterms, the ids of the types of polymorphic bindings, and the keys
    // whose fresh inference failed. A call that runs out deep in the walk
    // leaves its stacks full instead: the pending shared subterms, the frames
    // and the bindings, whose elements the next call would drop.
    let mut a = Arena::new();
    let bigs = [
        (
            "many shared",
            many_shared(&mut a, 20_000),
            DEFAULT_WORK_BUDGET,
        ),
        (
            "deep polymorphic let",
            deep_polymorphic_let(&mut a, 1_000),
            DEFAULT_WORK_BUDGET,
        ),
        (
            "failed keys",
            ill_typed_leaf_chain(&mut a, 1_000, false),
            DEFAULT_WORK_BUDGET,
        ),
        (
            "deep open if nest",
            open_if_nest(&mut a, 100_000),
            1_000_000,
        ),
    ];
    let (add, one, two) = (a.prim(Prim::Add), a.int(1), a.int(2));
    let small = a.apps(add, &[one, two]);
    for (name, big, budget) in bigs {
        let mut tc = TypeChecker::with_work_budget(budget);
        let (reads, units) = reads_and_units(&mut tc, &a, big);
        assert!(
            reads <= READS_PER_UNIT * units + 1_000,
            "{name}, large call: {reads} reads for {units} units"
        );
        tc.set_work_budget(100);
        for call in 1..=2 {
            let (reads, units) = reads_and_units(&mut tc, &a, small);
            assert!(
                reads <= READS_PER_UNIT * units + 1_000,
                "{name}, small call {call}: {reads} reads for {units} units"
            );
        }
    }
}

#[test]
fn one_unification_is_bounded_by_the_budget() {
    // A unification of two types that share differently visits pairs of
    // nodes, as many as the product of their sizes. The budget must stop it
    // there, not after it.
    let mut a = Arena::new();
    let t = crossed_sharing(&mut a, 9);
    let mut tc = TypeChecker::new();
    assert!(tc.infer(&a, t, NO_LIBS).is_ok());
    let full = tc.spent();
    let mut tried = 0;
    for budget in (1_000..full).step_by((full / 40) as usize) {
        let mut tc = TypeChecker::with_work_budget(budget);
        let e = tc.infer(&a, t, NO_LIBS).unwrap_err();
        assert_eq!(*e.kind, too_much_work(budget), "budget {budget}");
        assert!(
            tc.spent() <= 2 * budget + OVERRUN,
            "{} units against a budget of {budget}",
            tc.spent()
        );
        tried += 1;
    }
    assert!(tried >= 20);
}

#[test]
fn running_out_in_the_first_walk_does_not_start_the_second() {
    // The second walk only names the clash the first one found; if the first
    // ran out of work there is no clash to name.
    let mut a = Arena::new();
    let t = parse_term(&mut a, "(lam (if $0 (add $0 1) (add $0 1)))");
    let mut tc = TypeChecker::new();
    assert!(tc.infer(&a, t, NO_LIBS).is_err());
    assert_eq!(tc.walks, 2);
    let full = tc.spent();
    for budget in 1..=full {
        let mut tc = TypeChecker::with_work_budget(budget);
        let e = tc.infer(&a, t, NO_LIBS).unwrap_err();
        if *e.kind == too_much_work(budget) {
            assert_eq!(tc.walks, 1, "budget {budget}");
        }
    }
}

#[test]
fn bounded_printing_of_a_long_spine_stops_early() {
    // The head of a spine is at its far end. Printing stops once the
    // arguments met are enough to pass the limit, so it never walks to it.
    let mut a = Arena::new();
    let zero = a.int(0);
    let one = a.int(1);
    let spine = a.apps(zero, &vec![one; 100_000]);
    let t = a.lam(spine);
    let before = crate::term::GETS.with(std::cell::Cell::get);
    let text = super::infer::short(&a, t);
    let gets = crate::term::GETS.with(std::cell::Cell::get) - before;
    assert!(gets <= 1_000, "{gets} reads");
    let kept = text.strip_suffix(" ...").expect("long text ends with ...");
    assert!(sexpr::print(&a, t).unwrap().starts_with(kept), "{text}");
    // Under the limit, a spine prints whole.
    let short_spine = a.apps(zero, &[one; 50]);
    let full = sexpr::print(&a, short_spine).unwrap();
    assert_eq!(super::infer::short(&a, short_spine), full);
}

/// Under 64 lambdas, `(pair c_n c_n)`, where `c_0 = (pair $f(0) (pair $f(1) ...
/// (pair $f(62) $f(63))))` and `c_{k+1} = (pair c_k unit)`.
fn chain_over_variables(a: &mut Arena, n: u64, f: impl Fn(u32) -> u32) -> TermId {
    let pair = a.prim(Prim::Pair);
    let mut c = a.var(f(63));
    for i in (0..63u32).rev() {
        let x = a.var(f(i));
        c = a.apps(pair, &[x, c]);
    }
    let unit = a.unit();
    for _ in 0..n {
        c = a.apps(pair, &[c, unit]);
    }
    let mut t = a.apps(pair, &[c, c]);
    for _ in 0..64 {
        t = a.lam(t);
    }
    t
}

#[test]
fn finding_free_variables_is_charged_to_the_budget() {
    // Two terms of the same shape. In the first every node of the chain has 64
    // free variables, so finding the free variables of the shared `c_n`
    // gathers 64 indices for each of the two application nodes of each level;
    // in the second each has one.
    let n = 300u64;
    let mut a = Arena::new();
    let wide = chain_over_variables(&mut a, n, |i| i);
    let narrow = chain_over_variables(&mut a, n, |_| 0);
    let spent = |t: TermId| {
        let mut tc = TypeChecker::new();
        assert!(tc.infer(&a, t, NO_LIBS).is_ok());
        tc.spent()
    };
    let (wide, narrow) = (spent(wide), spent(narrow));
    assert!(wide >= narrow + 120 * n, "{wide} against {narrow}");
}

#[test]
fn finding_free_variables_stops_at_the_budget() {
    // Each node of the chain has 64 free variables, so finding them costs
    // about twenty times what the search for shared subterms does. A budget
    // the search just fits in must stop the finding too.
    let n = 3_000;
    let mut a = Arena::new();
    let t = chain_over_variables(&mut a, n, |i| i);
    // The search takes about 18 000 units.
    for budget in (1..=60).map(|i| i * 1_000) {
        let mut tc = TypeChecker::with_work_budget(budget);
        let _ = tc.infer(&a, t, NO_LIBS);
        assert!(
            tc.spent() <= 2 * budget + OVERRUN,
            "budget {budget}: {} units",
            tc.spent()
        );
    }
}

#[test]
fn a_long_chain_of_links_keeps_the_error_located() {
    // The conditions link the variables of the lambdas one to the next. If
    // each new link made the chain longer, the second walk would follow the
    // whole chain again after each failed use, and run out of work long before
    // naming the clash; linking by rank keeps each path short.
    let n = 2_000;
    let mut a = Arena::new();
    let t = chain_rollback(&mut a, n, n as usize);
    let mut tc = TypeChecker::with_work_budget(2_000_000);
    let e = tc.infer(&a, t, NO_LIBS).unwrap_err();
    assert_eq!(tc.walks, 2);
    assert_eq!(e.term_text, "$0", "{e}");
    assert!(tc.spent() <= 300 * u64::from(n), "{} units", tc.spent());
}
