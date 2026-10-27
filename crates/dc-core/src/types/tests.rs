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
    let e = check_src("(lam 1)", "Int -> Int -> Int").unwrap_err();
    assert_eq!(
        *e.kind,
        TypeErrorKind::Mismatch {
            expected: ty("Int -> Int -> Int"),
            // Unification binds the argument type to Int before it reaches the clash,
            // and errors show types as they stand at the point of failure.
            found: ty("Int -> Int"),
            conflict: Some((ty("Int -> Int"), Type::Int)),
        }
    );
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
