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
}

/// `lam x. let (lam (pair x $0)) in t_n`, with `t_0 = ($0 1)` and `t_(k+1) =
/// (if true t_k (let (lam (pair (fst ($1 0)) $0)) t_k))`: every level binds a new
/// polymorphic helper whose type has the free variable `x`.
fn helpers_over_a_lambda_variable(a: &mut Arena, n: usize) -> TermId {
    let (v0, v1, zero, one, yes) = (a.var(0), a.var(1), a.int(0), a.int(1), a.bool(true));
    let (pair, fst) = (a.prim(Prim::Pair), a.prim(Prim::Fst));
    let b0 = a.apps(pair, &[v1, v0]);
    let b0 = a.lam(b0);
    let prev = a.app(v1, zero);
    let x = a.app(fst, prev);
    let b = a.apps(pair, &[x, v0]);
    let b = a.lam(b);
    let mut t = a.app(v0, one);
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
        let full = sexpr::print(&a, id);
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
