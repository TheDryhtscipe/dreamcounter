//! Stitch output goes straight into expansion: abstractions decoded with
//! `decode_abstraction` are `LibDef`s, and a Stitch-rewritten program expands to the
//! program Stitch rewrote.

use std::cell::Cell;

use dc_core::prim::Prim;
use dc_core::{expand, sexpr, Arena, LibDef, TermId};
use dc_stitch::{decode, decode_abstraction, expr, to_expr, DecodeError, Expr, LibNames};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

fn names() -> LibNames {
    LibNames::sequential("fn_", 4).expect("names")
}

/// Decodes Stitch's abstractions, given as `(body, arity)` in library order.
fn libs(a: &mut Arena, abstractions: &[(&str, u32)], n: &LibNames) -> Vec<LibDef> {
    abstractions
        .iter()
        .map(|&(body, arity)| LibDef {
            arity,
            term: decode_abstraction(a, body, arity, n).unwrap_or_else(|e| panic!("{body}: {e}")),
        })
        .collect()
}

/// Expands the Stitch-rewritten program `rewritten` under `abstractions` and checks
/// that it is the core term `original`.
#[track_caller]
fn round_trip(abstractions: &[(&str, u32)], rewritten: &str, original: &str) {
    let mut a = Arena::new();
    let n = names();
    let libs = libs(&mut a, abstractions, &n);
    let p = decode(&mut a, rewritten, &n).unwrap_or_else(|e| panic!("{rewritten}: {e}"));
    let want = sexpr::parse(&mut a, original).unwrap_or_else(|e| panic!("{original}: {e}"));
    let got = expand(&mut a, &libs, p).unwrap_or_else(|e| panic!("{rewritten}: {e}"));
    assert_eq!(sexpr::print(&a, got), sexpr::print(&a, want), "{rewritten}");
    assert_eq!(got, want);
}

#[test]
fn parameters_under_binders_take_shifted_arguments() {
    // #0 sits under the two lambdas of the folded function; #1 does not.
    round_trip(
        &[("(foldl (lam (lam (add $1 (mul $0 #0)))) int_0 #1)", 2)],
        "(lam (lam (pair (fn_0 $1 $0) (fn_0 int_2 $1))))",
        "(lam (lam (pair (foldl (lam (lam (add $1 (mul $0 $3)))) 0 $0) \
         (foldl (lam (lam (add $1 (mul $0 2)))) 0 $1))))",
    );
}

#[test]
fn lets_and_lazy_forms_in_abstraction_bodies() {
    // The let's body is under `lam_0`, so #1 there takes its argument shifted by 1.
    round_trip(
        &[(
            "(let_ #0 (lam_0 (if_ (lt $0 int_10) (mul $0 #1) int_0)))",
            2,
        )],
        "(lam (fn_0 (add $0 int_1) $0))",
        "(lam (let (add $0 1) (if (lt $0 10) (mul $0 $1) 0)))",
    );
    round_trip(
        &[("(and_ #0 (or_ #1 #0))", 2)],
        "(lam (fn_0 (lt $0 int_1) false))",
        "(lam (and (lt $0 1) (or false (lt $0 1))))",
    );
}

#[test]
fn arguments_that_bind_and_nested_abstractions() {
    // An argument with its own binders, placed under one more: its bound variables
    // stay, its free one shifts.
    round_trip(
        &[("(lam (foldl #0 $0 #1))", 2)],
        "(lam (fn_0 (lam (lam (add $1 $2))) $0))",
        "(lam (lam (foldl (lam (lam (add $1 $3))) $0 $1)))",
    );
    // A later abstraction uses an earlier one under a binder.
    round_trip(
        &[("(add #0 int_1)", 1), ("(lam (fn_0 (mul #0 $0)))", 1)],
        "(lam (fn_1 $0))",
        "(lam (lam (add (mul $1 $0) 1)))",
    );
    // A nullary abstraction.
    round_trip(
        &[("(cons int_1 nil)", 0)],
        "(pair fn_0 fn_0)",
        "(pair (cons 1 nil) (cons 1 nil))",
    );
}

// ---------------------------------------------------------------------------
// Random programs, rewritten the way Stitch does.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Gen {
    Var(u32),
    Int(i64),
    Bool(bool),
    Prim(Prim),
    Lam(Box<Gen>),
    App(Box<Gen>, Box<Gen>),
    Let(Box<Gen>, Box<Gen>),
    If(Box<Gen>, Box<Gen>, Box<Gen>),
    Or(Box<Gen>, Box<Gen>),
}

fn gen_term() -> impl Strategy<Value = Gen> {
    let leaf = prop_oneof![
        3 => (0u32..3).prop_map(Gen::Var),
        1 => (0i64..3).prop_map(Gen::Int),
        1 => any::<bool>().prop_map(Gen::Bool),
        1 => proptest::sample::select(vec![Prim::Add, Prim::Cons, Prim::Nil, Prim::Pair])
            .prop_map(Gen::Prim),
    ];
    leaf.prop_recursive(6, 48, 3, |inner| {
        prop_oneof![
            1 => inner.clone().prop_map(|b| Gen::Lam(Box::new(b))),
            2 => (inner.clone(), inner.clone())
                .prop_map(|(f, x)| Gen::App(Box::new(f), Box::new(x))),
            1 => (inner.clone(), inner.clone()).prop_map(|(x, b)| Gen::Let(Box::new(x), Box::new(b))),
            1 => (inner.clone(), inner.clone(), inner.clone())
                .prop_map(|(c, t, e)| Gen::If(Box::new(c), Box::new(t), Box::new(e))),
            1 => (inner.clone(), inner).prop_map(|(x, y)| Gen::Or(Box::new(x), Box::new(y))),
        ]
    })
}

fn build(a: &mut Arena, g: &Gen) -> TermId {
    match g {
        Gen::Var(i) => a.var(*i),
        Gen::Int(n) => a.int(*n),
        Gen::Bool(b) => a.bool(*b),
        Gen::Prim(p) => a.prim(*p),
        Gen::Lam(b) => {
            let b = build(a, b);
            a.lam(b)
        }
        Gen::App(f, x) => {
            let f = build(a, f);
            let x = build(a, x);
            a.app(f, x)
        }
        Gen::Let(x, b) => {
            let x = build(a, x);
            let b = build(a, b);
            a.let_(x, b)
        }
        Gen::If(c, t, e) => {
            let c = build(a, c);
            let t = build(a, t);
            let e = build(a, e);
            a.if_(c, t, e)
        }
        Gen::Or(x, y) => {
            let x = build(a, x);
            let y = build(a, y);
            a.or(x, y)
        }
    }
}

/// The number of nodes of `e`.
fn size(e: &Expr) -> usize {
    match e {
        Expr::Lam(b) | Expr::TaggedLam(_, b) => 1 + size(b),
        Expr::App(f, x) => 1 + size(f) + size(x),
        _ => 1,
    }
}

/// The node at preorder position `pos` of `e`, with the number of binders above it.
fn node_at(e: &Expr, pos: usize) -> (&Expr, u32) {
    fn go<'e>(e: &'e Expr, pos: &mut usize, d: u32) -> Option<(&'e Expr, u32)> {
        if *pos == 0 {
            return Some((e, d));
        }
        *pos -= 1;
        match e {
            Expr::Lam(b) | Expr::TaggedLam(_, b) => go(b, pos, d + 1),
            Expr::App(f, x) => go(f, pos, d).or_else(|| go(x, pos, d)),
            _ => None,
        }
    }
    let mut pos = pos;
    go(e, &mut pos, 0).expect("position in range")
}

/// `e` with the node at preorder position `pos` replaced by `with`.
fn replace_at(e: &Expr, pos: usize, with: &Expr) -> Expr {
    fn go(e: &Expr, pos: &mut Option<usize>, with: &Expr) -> Expr {
        match pos {
            Some(0) => {
                *pos = None;
                return with.clone();
            }
            Some(p) => *p -= 1,
            None => return e.clone(),
        }
        match e {
            Expr::Lam(b) => Expr::lam(go(b, pos, with)),
            Expr::TaggedLam(t, b) => Expr::tagged_lam(*t, go(b, pos, with)),
            Expr::App(f, x) => {
                let f = go(f, pos, with);
                Expr::app(f, go(x, pos, with))
            }
            other => other.clone(),
        }
    }
    let mut pos = Some(pos);
    go(e, &mut pos, with)
}

/// Stitch's abstraction of `e`: each distinct free variable becomes a parameter,
/// numbered in order of first occurrence. Returns the body and, for each parameter,
/// the free variable's index at `e`'s root.
fn abstract_free(e: &Expr) -> (Expr, Vec<u32>) {
    fn go(e: &Expr, d: u32, params: &mut Vec<u32>) -> Expr {
        match e {
            Expr::Var(i) if *i >= d => {
                let j = i - d;
                let p = params.iter().position(|&q| q == j).unwrap_or_else(|| {
                    params.push(j);
                    params.len() - 1
                });
                Expr::IVar(u32::try_from(p).expect("few parameters"))
            }
            Expr::Lam(b) => Expr::lam(go(b, d + 1, params)),
            Expr::TaggedLam(t, b) => Expr::tagged_lam(*t, go(b, d + 1, params)),
            Expr::App(f, x) => {
                let f = go(f, d, params);
                Expr::app(f, go(x, d, params))
            }
            other => other.clone(),
        }
    }
    let mut params = vec![];
    let body = go(e, 0, &mut params);
    (body, params)
}

/// For random programs, Stitch's encoding with one subterm abstracted as `fn_0` and
/// the use applied to the subterm's free variables decodes, through
/// `decode_abstraction` and `decode`, to a library and a program that expand to the
/// original term.
#[test]
fn stitch_rewritten_programs_expand_to_the_original() {
    let n = names();
    let mut runner = TestRunner::new(Config {
        cases: 2048,
        failure_persistence: None,
        ..Config::default()
    });
    let accepted = Cell::new(0usize);
    let open = Cell::new(0usize);
    let result = runner.run(&(gen_term(), any::<usize>()), |(g, pos)| {
        let mut a = Arena::new();
        let mut prog = build(&mut a, &g);
        for _ in 0..3 {
            prog = a.lam(prog);
        }
        let e = to_expr(&a, prog, &n).map_err(|e| TestCaseError::fail(e.to_string()))?;
        let pos = pos % size(&e);
        let (node, _) = node_at(&e, pos);
        // `--fused-lambda-tags 0`: no abstraction is rooted at a let's binder.
        if matches!(node, Expr::TaggedLam(..)) {
            return Ok(());
        }
        let (body, params) = abstract_free(node);
        let arity = u32::try_from(params.len()).expect("few parameters");
        let mut use_site = Expr::sym("fn_0");
        for &j in &params {
            use_site = Expr::app(use_site, Expr::Var(j));
        }
        let rewritten = expr::print(&replace_at(&e, pos, &use_site));
        let body = expr::print(&body);
        // Cuts through a special form have no core term; Stitch's flags rule them
        // out (crate docs).
        let lib = match decode_abstraction(&mut a, &body, arity, &n) {
            Ok(t) => t,
            Err(DecodeError::MalformedSpecialForm { .. }) => return Ok(()),
            Err(e) => return Err(TestCaseError::fail(format!("{body}: {e}"))),
        };
        let p = match decode(&mut a, &rewritten, &n) {
            Ok(t) => t,
            Err(DecodeError::MalformedSpecialForm { .. }) => return Ok(()),
            Err(e) => return Err(TestCaseError::fail(format!("{rewritten}: {e}"))),
        };
        let got = expand(&mut a, &[LibDef { arity, term: lib }], p)
            .map_err(|e| TestCaseError::fail(format!("{rewritten}: {e}")))?;
        prop_assert_eq!(
            sexpr::print(&a, got),
            sexpr::print(&a, prog),
            "abstraction {} / {}, program {}",
            body,
            arity,
            rewritten
        );
        prop_assert_eq!(got, prog);
        accepted.set(accepted.get() + 1);
        if arity > 0 {
            open.set(open.get() + 1);
        }
        Ok(())
    });
    if let Err(e) = result {
        panic!("{e}");
    }
    let (accepted, open) = (accepted.get(), open.get());
    assert!(accepted >= 1000, "only {accepted} cases decoded");
    assert!(open >= 300, "only {open} cases had parameters");
}
