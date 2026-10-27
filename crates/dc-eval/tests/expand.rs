//! Library abstractions through the evaluator (spec §8, decision D1): a refactored
//! program, once expanded, evaluates to the same value in the same number of steps
//! as the program it came from (spec §7, invariant 2). Unsaturated uses cost one
//! extra β step per η binder.

mod common;

use common::*;
use dc_core::term::children;
use dc_core::{expand, sexpr, Arena, LibDef, LibId, Term, TermId};
use dc_eval::{apply, Outcome, Value, UNLIMITED};
use proptest::prelude::*;

fn parse_in(arena: &mut Arena, src: &str) -> TermId {
    sexpr::parse(arena, src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

/// Expands `src` under `libs` and runs it on `inputs`.
fn run_with_libs(libs: &[(&str, u32)], src: &str, inputs: &[Value]) -> Outcome {
    let mut arena = Arena::new();
    let defs: Vec<LibDef> = libs
        .iter()
        .map(|&(l, arity)| LibDef {
            arity,
            term: parse_in(&mut arena, l),
        })
        .collect();
    let t = parse_in(&mut arena, src);
    let t = expand(&mut arena, &defs, t).unwrap_or_else(|e| panic!("{src}: {e}"));
    apply(&arena, t, inputs, UNLIMITED)
}

/// Asserts that `refactored` under `libs` and `original` both give `expected` (an
/// outcome as `Outcome`'s `Display` writes it, steps included).
#[track_caller]
fn same(libs: &[(&str, u32)], refactored: &str, original: &str, inputs: &[Value], expected: &str) {
    assert_eq!(
        run_on(original, inputs).to_string(),
        expected,
        "original {original}"
    );
    assert_eq!(
        run_with_libs(libs, refactored, inputs).to_string(),
        expected,
        "refactored {refactored}"
    );
}

#[test]
fn a_duplicated_parameter_is_evaluated_at_each_occurrence() {
    // λ#0. #0 + #0. As a function call this would evaluate `mul` once (2 steps
    // fewer than the original); as a macro it costs what the original costs.
    same(
        &[("(lam (add $0 $0))", 1)],
        "(lib_0 (mul 3 4))",
        "(add (mul 3 4) (mul 3 4))",
        &[],
        "Value(24, 3)",
    );
}

#[test]
fn a_parameter_in_a_lazy_position_stays_lazy() {
    // The guarded `div` of D1: λ#0 #1. if #0 = 0 then 0 else #1.
    let guard = ("(lam (lam (if (eq $1 0) 0 $0)))", 2);
    let refactored = "(lam (lib_0 $0 (div 100 $0)))";
    let original = "(lam (if (eq $0 0) 0 (div 100 $0)))";
    // β 1 + eq 1 + if 1.
    same(&[guard], refactored, original, &[int(0)], "Value(0, 3)");
    // ... + div 1.
    same(&[guard], refactored, original, &[int(5)], "Value(20, 4)");
    // Under a lambda: λ#0. λ. #0 is a closure, not an error.
    let delay = ("(lam (lam $1))", 1);
    same(
        &[delay],
        "(let (lib_0 (div 1 0)) 5)",
        "(let (lam (div 1 0)) 5)",
        &[],
        "Value(5, 0)",
    );
    // Over-saturated: the closure is applied, then the division fails.
    same(
        &[delay],
        "(lib_0 (div 1 0) 7)",
        "((lam (div 1 0)) 7)",
        &[],
        "Error(DivisionByZero, 2)",
    );
}

#[test]
fn open_arguments_under_binders_keep_their_meaning() {
    // λ#0. λ. let 1 in #0 + $0, with an argument that uses both program inputs.
    same(
        &[("(lam (lam (let 1 (add $2 $0))))", 1)],
        "(lam (lam (lib_0 (sub $1 $0) 100)))",
        "(lam (lam ((lam (let 1 (add (sub $3 $2) $0))) 100)))",
        &[int(10), int(3)],
        // β 2 + β 1 + sub 1 + add 1.
        "Value(8, 5)",
    );
}

#[test]
fn unsaturated_uses_cost_one_beta_step_per_eta_binder() {
    let sub = ("(lam (lam (sub $1 $0)))", 2);
    // Saturated: sub 1.
    same(&[sub], "(lib_0 50 8)", "(sub 50 8)", &[], "Value(42, 1)");
    // One argument short: λy. sub 50 y, applied: + 1 β.
    assert_eq!(
        run_with_libs(&[sub], "(let (lib_0 50) ($0 8))", &[]).to_string(),
        "Value(42, 2)"
    );
    // Bare: λx. λy. sub x y, applied to both: + 2 β.
    assert_eq!(
        run_with_libs(&[sub], "(let lib_0 ($0 50 8))", &[]).to_string(),
        "Value(42, 3)"
    );
    // Bare, as a function argument: exactly the hand-written lambda.
    same(
        &[sub],
        "(lam (foldl lib_0 100 $0))",
        "(lam (foldl (lam (lam (sub $1 $0))) 100 $0))",
        &[range1(4)],
        // β 1 + 4 × (1 + β 2 + sub 1).
        "Value(90, 17)",
    );
}

#[test]
fn libraries_may_build_on_earlier_ones() {
    let double = ("(lam (add $0 $0))", 1);
    let quad = ("(lam (lib_0 (lib_0 $0)))", 1);
    same(
        &[double, quad],
        "(lam (lib_1 $0))",
        "(lam (add (add $0 $0) (add $0 $0)))",
        &[int(3)],
        // β 1 + add 3.
        "Value(12, 4)",
    );
    // A use as another use's argument, and a library used unsaturated in a body.
    let flip = ("(lam (lam (sub $0 $1)))", 2);
    let apply_to_7 = ("(lam ($0 7))", 1);
    same(
        &[flip, apply_to_7],
        "(lib_1 (lib_0 10))",
        "((lam (sub $0 10)) 7)",
        &[],
        // η β 1 + sub 1.
        "Value(-3, 2)",
    );
}

// ---------------------------------------------------------------------------
// Property: abstract a random subterm of a real program into a library, expand,
// and evaluate. The outcome, steps included, is the original's.
// ---------------------------------------------------------------------------

/// Programs and inputs, from the invariant tests.
fn programs() -> Vec<(&'static str, Vec<Value>)> {
    vec![
        ("(lam (foldl (lam (lam (add $1 $0))) 0 $0))", vec![range1(10)]),
        ("(lam (foldr cons nil $0))", vec![range1(5)]),
        (
            "(lam (bfix (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
            vec![int(8)],
        ),
        (
            "(lam (memorec (add $0 1) (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))",
            vec![int(20)],
        ),
        (
            "(lam (array_to_list (array_set (array_tabulate (length $0) (lam (mul $0 $0))) 3 0)))",
            vec![range1(6)],
        ),
        (
            "(lam (tree_fold 0 (lam (lam (lam (add (add $2 $1) $0)))) $0))",
            vec![Value::Tree(balanced(0, 15))],
        ),
        (
            "(lam (foldl (lam (lam (div $1 (sub 10 $0)))) 1000 $0))",
            vec![range1(12)],
        ),
        (
            "(lam (let (add $0 1) (if (lt $0 3) (and (lt 0 $1) (lt $1 $0)) (or (eq $0 4) false))))",
            vec![int(2)],
        ),
        (
            "(lam (lam (let (mul $1 $0) (if (lt $0 10) (div $0 (sub $2 $2)) (add $0 $0)))))",
            vec![int(4), int(5)],
        ),
    ]
}

fn preorder(a: &Arena, t: TermId, out: &mut Vec<TermId>) {
    out.push(t);
    for c in children(&a.get(t)) {
        preorder(a, c, out);
    }
}

fn rebuild(node: Term, kids: &[TermId]) -> Term {
    match node {
        Term::Lam(_) => Term::Lam(kids[0]),
        Term::Let(..) => Term::Let(kids[0], kids[1]),
        Term::App(..) => Term::App(kids[0], kids[1]),
        Term::And(..) => Term::And(kids[0], kids[1]),
        Term::Or(..) => Term::Or(kids[0], kids[1]),
        Term::If(..) => Term::If(kids[0], kids[1], kids[2]),
        leaf => leaf,
    }
}

/// Rebuilds `t`, letting `f` replace a node (given its binder depth) first.
fn map_term(
    a: &mut Arena,
    t: TermId,
    d: u32,
    f: &mut dyn FnMut(&mut Arena, TermId, u32) -> Option<TermId>,
) -> TermId {
    if let Some(r) = f(a, t, d) {
        return r;
    }
    let node = a.get(t);
    let kids: Vec<TermId> = match node {
        Term::Lam(b) => vec![map_term(a, b, d + 1, f)],
        Term::Let(x, b) => {
            let x = map_term(a, x, d, f);
            vec![x, map_term(a, b, d + 1, f)]
        }
        other => children(&other)
            .into_iter()
            .map(|c| map_term(a, c, d, f))
            .collect(),
    };
    a.intern(rebuild(node, &kids))
}

fn replace_at(a: &mut Arena, t: TermId, pos: &mut usize, with: TermId) -> TermId {
    if *pos == 0 {
        *pos = usize::MAX;
        return with;
    }
    if *pos == usize::MAX {
        return t;
    }
    *pos -= 1;
    let node = a.get(t);
    let kids: Vec<TermId> = children(&node)
        .into_iter()
        .map(|c| replace_at(a, c, pos, with))
        .collect();
    a.intern(rebuild(node, &kids))
}

/// Abstracts the subterm at preorder position `pos` into library 0, as Stitch would:
/// its free variables become parameters, and every occurrence of the closed
/// subterm `extra` (if it names one) becomes one more parameter.
fn abstract_at(a: &mut Arena, prog: TermId, pos: usize, extra: Option<usize>) -> (LibDef, TermId) {
    let mut nodes = vec![];
    preorder(a, prog, &mut nodes);
    let s = nodes[pos];
    let n = a.free_depth(s);
    let mut inner = vec![];
    preorder(a, s, &mut inner);
    let closed = extra
        .map(|e| inner[e % inner.len()])
        .filter(|&c| a.is_closed(c));
    let k = n + u32::from(closed.is_some());
    let body = map_term(a, s, 0, &mut |a, t, d| {
        if Some(t) == closed {
            return Some(a.var(d + k - 1 - n));
        }
        match a.get(t) {
            Term::Var(j) if j >= d => Some(a.var(j + k - n)),
            _ => None,
        }
    });
    let mut term = body;
    for _ in 0..k {
        term = a.lam(term);
    }
    let lib = a.lib(LibId(0));
    let mut args: Vec<TermId> = (0..n).map(|i| a.var(n - 1 - i)).collect();
    args.extend(closed);
    let use_site = a.apps(lib, &args);
    let mut p = pos;
    let rewritten = replace_at(a, prog, &mut p, use_site);
    (LibDef { arity: k, term }, rewritten)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn refactoring_preserves_value_and_steps(
        which in any::<usize>(),
        pos in any::<usize>(),
        extra in proptest::option::of(any::<usize>()),
    ) {
        let progs = programs();
        let (src, inputs) = &progs[which % progs.len()];
        let mut a = Arena::new();
        let prog = parse_in(&mut a, src);
        let mut nodes = vec![];
        preorder(&a, prog, &mut nodes);
        let (lib, refactored) = abstract_at(&mut a, prog, pos % nodes.len(), extra);
        prop_assert_ne!(refactored, prog);
        let expanded = expand(&mut a, &[lib], refactored).expect("expand");
        let want = apply(&a, prog, inputs, UNLIMITED).to_string();
        let got = apply(&a, expanded, inputs, UNLIMITED).to_string();
        prop_assert_eq!(got, want, "{}", sexpr::print(&a, refactored));
    }
}
