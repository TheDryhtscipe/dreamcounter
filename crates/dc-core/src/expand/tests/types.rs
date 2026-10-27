//! Expansion preserves types: a refactored program, typed with library schemes
//! inferred from the library definitions, has the type of its expansion.
//!
//! "Has the type" is exact for the examples here. In general the expansion's type is
//! at least as general as the refactored program's: a library parameter is
//! monomorphic in the body, while the substituted argument is typed afresh at every
//! occurrence (see `a_duplicated_polymorphic_argument_generalises_the_expansion`).
//! The property test checks that direction for every case and counts how often the
//! types are equal.

use std::cell::Cell;

use proptest::test_runner::{Config, TestRunner};

use super::*;
use crate::types::{check as check_type, infer, Scheme};

/// The schemes of `libs`, each inferred from its definition with the earlier ones.
fn lib_schemes(a: &Arena, libs: &[LibDef]) -> Vec<Scheme> {
    let mut out: Vec<Scheme> = Vec::with_capacity(libs.len());
    for (i, def) in libs.iter().enumerate() {
        let s = infer(a, def.term, &out)
            .unwrap_or_else(|e| panic!("lib_{i} {}: {e}", sexpr::print(a, def.term)));
        out.push(s);
    }
    out
}

/// The type of `prog` with library uses, and the type of its expansion.
fn both_types(libs: &[(&str, u32)], prog: &str) -> (Scheme, Scheme) {
    let mut a = Arena::new();
    let libs = defs(&mut a, libs);
    let schemes = lib_schemes(&a, &libs);
    let p = parse(&mut a, prog);
    let refactored = infer(&a, p, &schemes).unwrap_or_else(|e| panic!("{prog}: {e}"));
    let e = expand(&mut a, &libs, p).unwrap_or_else(|e| panic!("{prog}: {e}"));
    let no_libs: &[Scheme] = &[];
    let expanded =
        infer(&a, e, no_libs).unwrap_or_else(|err| panic!("{}: {err}", sexpr::print(&a, e)));
    (refactored, expanded)
}

#[track_caller]
fn same_type(libs: &[(&str, u32)], prog: &str, expected: &str) {
    let (refactored, expanded) = both_types(libs, prog);
    let want: Scheme = expected
        .parse()
        .unwrap_or_else(|e| panic!("{expected}: {e}"));
    assert_eq!(refactored, want, "type of {prog}");
    assert_eq!(expanded, want, "type of the expansion of {prog}");
}

/// λf. λg. λx. f (g x).
const COMPOSE: (&str, u32) = ("(lam (lam (lam ($2 ($1 $0)))))", 3);

#[test]
fn saturated_uses_keep_their_type() {
    same_type(
        &[COMPOSE],
        "(lib_0 (lam (pair $0 $0)) (add 1) 2)",
        "(Int, Int)",
    );
    // Open arguments: the program's own binders are typed the same way.
    same_type(
        &[COMPOSE],
        "(lam (lam (lib_0 $1 $0 3)))",
        "(a -> b) -> (Int -> a) -> b",
    );
}

#[test]
fn unsaturated_uses_keep_their_type() {
    same_type(&[COMPOSE], "lib_0", "(a -> b) -> (c -> a) -> c -> b");
    same_type(&[COMPOSE], "(lib_0 (add 1))", "(a -> Int) -> a -> Int");
    same_type(
        &[COMPOSE],
        "(lam (lib_0 $0 (lam (pair $0 true))))",
        "((a, Bool) -> b) -> a -> b",
    );
    // A bare use passed to a higher-order primitive.
    same_type(
        &[("(lam (lam (add $1 $0)))", 2)],
        "(foldl lib_0 0)",
        "List Int -> Int",
    );
}

#[test]
fn oversaturated_uses_keep_their_type() {
    // λ#0. λy. (#0, y): arity 1, so the second argument goes to the expansion.
    same_type(
        &[("(lam (lam (pair $1 $0)))", 1)],
        "(lib_0 1 true)",
        "(Int, Bool)",
    );
    // A use whose result is a function, applied to more arguments.
    same_type(
        &[("(lam $0)", 1)],
        "(lib_0 (lam (lam (pair $0 $1))) 1 nil)",
        "(List a, Int)",
    );
}

#[test]
fn let_polymorphic_libraries_keep_their_type() {
    // The identity, used at two types in one program, saturated and bare.
    let id = ("(lam $0)", 1);
    same_type(&[id], "(pair (lib_0 1) (lib_0 true))", "(Int, Bool)");
    same_type(&[id], "(let lib_0 (pair ($0 1) ($0 true)))", "(Int, Bool)");
    // A body with an inner polymorphic let: λx. let i = λy. y in (i x, i true).
    same_type(
        &[("(lam (let (lam $0) (pair ($0 $1) ($0 true))))", 1)],
        "(pair (lib_0 1) (lib_0 nil))",
        "((Int, Bool), (List a, Bool))",
    );
    // A later library uses an earlier polymorphic one at two types.
    same_type(
        &[id, ("(lam (pair (lib_0 $0) (lib_0 0)))", 1)],
        "(lib_1 unit)",
        "(Unit, Int)",
    );
}

#[test]
fn key_constraints_survive_libraries() {
    // λf. λn. memorec n f: the key type of its uses is still checked.
    let lib = ("(lam (lam (memorec 10 $1 $0)))", 2);
    same_type(
        &[lib],
        "(lam (lib_0 (lam (lam (if $0 0 1))) $0))",
        "Bool -> Int",
    );
    same_type(&[lib], "lib_0", "Key a => ((a -> b) -> a -> b) -> a -> b");
}

#[test]
fn a_duplicated_polymorphic_argument_generalises_the_expansion() {
    // λx. (x, x): the parameter is monomorphic, so `lib_0 nil` is (List a, List a),
    // while `(nil, nil)` is (List a, List b).
    let (refactored, expanded) = both_types(&[("(lam (pair $0 $0))", 1)], "(lib_0 nil)");
    assert_eq!(refactored.to_string(), "(List a, List a)");
    assert_eq!(expanded.to_string(), "(List a, List b)");
    // The refactored type is an instance of the expansion's.
    let mut a = Arena::new();
    let e = parse(&mut a, "(pair nil nil)");
    let no_libs: &[Scheme] = &[];
    assert_eq!(check_type(&a, e, no_libs, &refactored), Ok(()));
}

/// Rewrites the library use at preorder position `pos` of `rewritten`, `L a0 ...
/// a(k-1)`, to `let f = L a0 ... a(n-1) in f a(n) ... a(k-1)`: an unsaturated use.
fn unsaturate(a: &mut Arena, rewritten: TermId, pos: usize, n: usize) -> TermId {
    let mut nodes = vec![];
    preorder(a, rewritten, &mut nodes);
    let (head, args) = crate::expand::spine(a, nodes[pos]);
    let bound = a.apps(head, &args[..n]);
    let rest: Vec<TermId> = args[n..].iter().map(|&x| shift(a, x, 1)).collect();
    let f = a.var(0);
    let body = a.apps(f, &rest);
    let use_site = a.let_(bound, body);
    replace_at(a, rewritten, pos, use_site)
}

/// Whether `t` has a use of library 0 with more than `arity` arguments.
fn has_oversaturated_use(a: &Arena, t: TermId, arity: u32) -> bool {
    let mut nodes = vec![];
    preorder(a, t, &mut nodes);
    nodes.into_iter().any(|n| {
        let (head, args) = crate::expand::spine(a, n);
        a.get(head) == Term::Lib(LibId(0)) && args.len() > arity as usize
    })
}

/// For random programs that type, with a subterm abstracted the way Stitch does and
/// the library's scheme inferred from its definition: if the refactored program
/// types, its expansion has a type of which the refactored type is an instance.
/// Saturated, unsaturated and over-saturated uses are all exercised, and in most
/// cases the two types are equal.
#[test]
fn refactored_programs_type_as_their_expansion() {
    let mut runner = TestRunner::new(Config {
        cases: 4096,
        failure_persistence: None,
        ..Config::default()
    });
    let typed = Cell::new(0usize);
    let equal = Cell::new(0usize);
    let unsaturated = Cell::new(0usize);
    let oversaturated = Cell::new(0usize);
    let strategy = (
        gen_term(),
        any::<usize>(),
        proptest::option::of(any::<usize>()),
        proptest::option::of(any::<usize>()),
        any::<bool>(),
    );
    let result = runner.run(&strategy, |(g, pos, extra, split, head)| {
        let mut a = Arena::new();
        // Close the program: generated variables are below 4.
        let mut prog = build(&mut a, &g);
        for _ in 0..4 {
            prog = a.lam(prog);
        }
        let no_libs: &[Scheme] = &[];
        if infer(&a, prog, no_libs).is_err() {
            return Ok(());
        }
        let mut nodes = vec![];
        preorder(&a, prog, &mut nodes);
        // Half the time, abstract the function of an application, so that the use
        // is over-saturated.
        let heads: Vec<usize> = (0..nodes.len())
            .filter(|&i| matches!(a.get(nodes[i]), Term::App(..)))
            .map(|i| i + 1)
            .collect();
        let pos = if head && !heads.is_empty() {
            heads[pos % heads.len()]
        } else {
            pos % nodes.len()
        };
        let (lib, mut rewritten) = abstract_at(&mut a, prog, pos, extra);
        let Ok(scheme) = infer(&a, lib.term, no_libs) else {
            return Ok(());
        };
        let mut is_unsaturated = false;
        if let Some(split) = split {
            if lib.arity > 0 {
                let n = split % lib.arity as usize;
                rewritten = unsaturate(&mut a, rewritten, pos, n);
                is_unsaturated = true;
            }
        }
        let schemes = vec![scheme];
        let Ok(refactored) = infer(&a, rewritten, &schemes) else {
            return Ok(());
        };
        let expanded = expand(&mut a, &[lib], rewritten).expect("expand");
        let expanded_type = infer(&a, expanded, no_libs);
        prop_assert!(
            expanded_type.is_ok(),
            "{} has type {refactored} but its expansion {} does not type: {:?}",
            sexpr::print(&a, rewritten),
            sexpr::print(&a, expanded),
            expanded_type
        );
        prop_assert_eq!(
            check_type(&a, expanded, no_libs, &refactored),
            Ok(()),
            "{} has type {} but its expansion {} does not have it",
            sexpr::print(&a, rewritten),
            refactored,
            sexpr::print(&a, expanded)
        );
        typed.set(typed.get() + 1);
        if expanded_type.as_ref() == Ok(&refactored) {
            equal.set(equal.get() + 1);
        }
        if is_unsaturated {
            unsaturated.set(unsaturated.get() + 1);
        }
        if has_oversaturated_use(&a, rewritten, lib.arity) {
            oversaturated.set(oversaturated.get() + 1);
        }
        Ok(())
    });
    if let Err(e) = result {
        panic!("{e}");
    }
    let (typed, equal) = (typed.get(), equal.get());
    let (unsaturated, oversaturated) = (unsaturated.get(), oversaturated.get());
    // The test is not vacuous: enough cases type, and each kind of use occurs.
    assert!(typed >= 300, "only {typed} cases typed");
    assert!(
        unsaturated >= 100,
        "only {unsaturated} unsaturated cases typed"
    );
    assert!(
        oversaturated >= 30,
        "only {oversaturated} over-saturated cases typed"
    );
    assert!(
        equal * 10 >= typed * 9,
        "only {equal} of {typed} expansions have the refactored type exactly"
    );
}
