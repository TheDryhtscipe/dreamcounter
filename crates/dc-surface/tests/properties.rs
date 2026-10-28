//! Property tests: operator precedence, and pretty-printing as an inverse of
//! compilation.

use dc_core::{sexpr, Arena, LibId, Prim, Term, TermId};
use dc_surface::pretty::pretty_unchecked;
use dc_surface::{compile_expression, pretty, TopLevel};
use proptest::prelude::*;

/// An arithmetic and comparison expression tree, independent of the crate's AST.
#[derive(Clone, Debug)]
enum Arith {
    Lit(i64),
    Bin(Op, Box<Arith>, Box<Arith>),
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Cons,
}

impl Op {
    /// (text, precedence, right associative)
    fn info(self) -> (&'static str, u8, bool) {
        match self {
            Op::Cons => ("::", 4, true),
            Op::Add => ("+", 5, false),
            Op::Sub => ("-", 5, false),
            Op::Mul => ("*", 6, false),
            Op::Div => ("/", 6, false),
            Op::Mod => ("mod", 6, false),
        }
    }

    fn prim(self) -> Prim {
        match self {
            Op::Add => Prim::Add,
            Op::Sub => Prim::Sub,
            Op::Mul => Prim::Mul,
            Op::Div => Prim::Div,
            Op::Mod => Prim::Mod,
            Op::Cons => Prim::Cons,
        }
    }
}

/// Prints with the fewest parentheses OCaml precedence allows.
fn print_min(e: &Arith, needed: u8) -> String {
    match e {
        Arith::Lit(n) if *n < 0 => format!("({n})"),
        Arith::Lit(n) => n.to_string(),
        Arith::Bin(op, a, b) => {
            let (text, p, right) = op.info();
            let (pl, pr) = if right { (p + 1, p) } else { (p, p + 1) };
            let s = format!("{} {text} {}", print_min(a, pl), print_min(b, pr));
            if p < needed {
                format!("({s})")
            } else {
                s
            }
        }
    }
}

fn build(arena: &mut Arena, e: &Arith) -> TermId {
    match e {
        Arith::Lit(n) => arena.int(*n),
        Arith::Bin(op, a, b) => {
            let f = arena.prim(op.prim());
            let ta = build(arena, a);
            let tb = build(arena, b);
            arena.apps(f, &[ta, tb])
        }
    }
}

fn arb_arith() -> impl Strategy<Value = Arith> {
    let leaf = (-50i64..50).prop_map(Arith::Lit);
    leaf.prop_recursive(5, 40, 2, |inner| {
        (
            prop_oneof![
                Just(Op::Add),
                Just(Op::Sub),
                Just(Op::Mul),
                Just(Op::Div),
                Just(Op::Mod),
                Just(Op::Cons)
            ],
            inner.clone(),
            inner,
        )
            .prop_map(|(op, a, b)| Arith::Bin(op, Box::new(a), Box::new(b)))
    })
}

proptest! {
    #[test]
    fn minimal_parentheses_parse_with_ocaml_precedence(e in arb_arith()) {
        let text = print_min(&e, 0);
        let mut arena = Arena::new();
        let expected = build(&mut arena, &e);
        let got = compile_expression(&mut arena, &text)
            .map_err(|err| TestCaseError::fail(format!("{text}: {err}")))?;
        prop_assert_eq!(
            sexpr::print(&arena, got).unwrap(),
            sexpr::print(&arena, expected).unwrap(),
            "text: {}", text
        );
    }
}

/// Random closed core terms, built with binders so that variables are in scope.
fn arb_core(depth: u32) -> BoxedStrategy<CoreGen> {
    let leaf = prop_oneof![
        (0u32..6).prop_map(CoreGen::Var),
        (-3i64..20).prop_map(CoreGen::Int),
        any::<bool>().prop_map(CoreGen::Bool),
        Just(CoreGen::Unit),
        proptest::sample::select(Prim::ALL).prop_map(CoreGen::Prim),
        (0u32..3).prop_map(CoreGen::Lib),
    ];
    if depth == 0 {
        return leaf.boxed();
    }
    let inner = arb_core(depth - 1);
    prop_oneof![
        2 => leaf,
        2 => inner.clone().prop_map(|b| CoreGen::Lam(Box::new(b))),
        3 => (inner.clone(), proptest::collection::vec(inner.clone(), 1..4))
            .prop_map(|(f, xs)| CoreGen::App(Box::new(f), xs)),
        2 => (inner.clone(), inner.clone())
            .prop_map(|(a, b)| CoreGen::Let(Box::new(a), Box::new(b))),
        1 => (inner.clone(), inner.clone(), inner.clone())
            .prop_map(|(a, b, c)| CoreGen::If(Box::new(a), Box::new(b), Box::new(c))),
        1 => (inner.clone(), inner.clone()).prop_map(|(a, b)| CoreGen::And(Box::new(a), Box::new(b))),
        1 => (inner.clone(), inner.clone()).prop_map(|(a, b)| CoreGen::Or(Box::new(a), Box::new(b))),
        // Shapes the printer's sugar targets, built the way the desugarer builds
        // them, including near misses where a hidden variable is used.
        2 => (0usize..3, inner.clone(), inner.clone(), inner.clone())
            .prop_map(|(f, s, e, c)| CoreGen::Case(f, Box::new(s), Box::new(e), Box::new(c))),
        2 => (inner.clone(), inner.clone(), any::<bool>())
            .prop_map(|(a, b, fst)| CoreGen::Destructure(Box::new(a), Box::new(b), fst)),
        1 => (inner.clone(), inner, any::<bool>())
            .prop_map(|(a, b, lt)| CoreGen::Compare(Box::new(a), Box::new(b), lt)),
    ]
    .boxed()
}

#[derive(Clone, Debug)]
enum CoreGen {
    Var(u32),
    Int(i64),
    Bool(bool),
    Unit,
    Prim(Prim),
    Lib(u32),
    Lam(Box<CoreGen>),
    App(Box<CoreGen>, Vec<CoreGen>),
    Let(Box<CoreGen>, Box<CoreGen>),
    If(Box<CoreGen>, Box<CoreGen>, Box<CoreGen>),
    And(Box<CoreGen>, Box<CoreGen>),
    Or(Box<CoreGen>, Box<CoreGen>),
    /// `(case s (lam e) (lam^k c))` for list, option or tree.
    Case(usize, Box<CoreGen>, Box<CoreGen>, Box<CoreGen>),
    /// `(lam (let (fst $0) (let (snd $1) b)))`, or only the `snd` let.
    Destructure(Box<CoreGen>, Box<CoreGen>, bool),
    /// `(let a (lt b $0))` / `le`.
    Compare(Box<CoreGen>, Box<CoreGen>, bool),
}

/// Builds the term; variables are reduced modulo the binders in scope (or become
/// a literal when there are none), so the result is closed.
fn realise(arena: &mut Arena, g: &CoreGen, depth: u32) -> TermId {
    match g {
        CoreGen::Var(i) if depth == 0 => arena.int(i64::from(*i)),
        CoreGen::Var(i) => arena.var(i % depth),
        CoreGen::Int(n) => arena.int(*n),
        CoreGen::Bool(b) => arena.bool(*b),
        CoreGen::Unit => arena.unit(),
        CoreGen::Prim(p) => arena.prim(*p),
        CoreGen::Lib(l) => arena.lib(LibId(*l)),
        CoreGen::Lam(b) => {
            let b = realise(arena, b, depth + 1);
            arena.lam(b)
        }
        CoreGen::App(f, xs) => {
            let f = realise(arena, f, depth);
            let xs: Vec<TermId> = xs.iter().map(|x| realise(arena, x, depth)).collect();
            arena.apps(f, &xs)
        }
        CoreGen::Let(a, b) => {
            let a = realise(arena, a, depth);
            let b = realise(arena, b, depth + 1);
            arena.let_(a, b)
        }
        CoreGen::If(a, b, c) => {
            let a = realise(arena, a, depth);
            let b = realise(arena, b, depth);
            let c = realise(arena, c, depth);
            arena.if_(a, b, c)
        }
        CoreGen::And(a, b) => {
            let a = realise(arena, a, depth);
            let b = realise(arena, b, depth);
            arena.and(a, b)
        }
        CoreGen::Or(a, b) => {
            let a = realise(arena, a, depth);
            let b = realise(arena, b, depth);
            arena.or(a, b)
        }
        CoreGen::Case(f, s, e, c) => {
            let (prim, k) = [
                (Prim::ListCase, 2),
                (Prim::OptionCase, 1),
                (Prim::TreeCase, 3),
            ][*f];
            let p = arena.prim(prim);
            let s = realise(arena, s, depth);
            let e = realise(arena, e, depth + 1);
            let e = arena.lam(e);
            let mut c = realise(arena, c, depth + k);
            for _ in 0..k {
                c = arena.lam(c);
            }
            arena.apps(p, &[s, e, c])
        }
        CoreGen::Destructure(a, b, with_fst) => {
            let a = realise(arena, a, depth + 1);
            let snd = arena.prim(Prim::Snd);
            let fst = arena.prim(Prim::Fst);
            let body = if *with_fst {
                let b = realise(arena, b, depth + 3);
                let v1 = arena.var(1);
                let pb = arena.app(snd, v1);
                let inner = arena.let_(pb, b);
                let v0 = arena.var(0);
                let pa = arena.app(fst, v0);
                arena.let_(pa, inner)
            } else {
                let b = realise(arena, b, depth + 2);
                let v0 = arena.var(0);
                let pb = arena.app(snd, v0);
                arena.let_(pb, b)
            };
            let l = arena.lam(body);
            arena.app(l, a)
        }
        CoreGen::Compare(a, b, lt) => {
            let p = arena.prim(if *lt { Prim::Lt } else { Prim::Le });
            let a = realise(arena, a, depth);
            let b = realise(arena, b, depth + 1);
            let v = arena.var(0);
            let body = arena.apps(p, &[b, v]);
            arena.let_(a, body)
        }
    }
}

fn lib_name(l: LibId) -> String {
    format!("lib{}", l.0)
}

fn recompile(text: &str) -> Result<String, String> {
    let mut arena = Arena::new();
    let mut env = TopLevel::new();
    for l in 0..3 {
        let t = arena.lib(LibId(l));
        env.define_inline(&arena, &lib_name(LibId(l)), t)
            .map_err(|e| e.to_string())?;
    }
    let id = env
        .close_expression(&mut arena, "pretty", text)
        .map_err(|e| format!("{text}\n{e}"))?;
    Ok(sexpr::print(&arena, id).unwrap())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn pretty_then_compile_is_the_identity(g in arb_core(4)) {
        let mut arena = Arena::new();
        let id = realise(&mut arena, &g, 0);
        prop_assume!(arena.is_closed(id));
        let expected = sexpr::print(&arena, id).unwrap();
        let text = pretty(&arena, id, lib_name).unwrap();
        let back = recompile(&text).map_err(TestCaseError::fail)?;
        prop_assert_eq!(&back, &expected, "printed as: {}", text);
        // pretty() falls back to the plain printer, so that one must round-trip
        // on its own; and the sugared printer should not need the fallback.
        for sugar in [false, true] {
            let text = pretty_unchecked(&arena, id, lib_name, sugar).unwrap();
            let back = recompile(&text).map_err(TestCaseError::fail)?;
            prop_assert_eq!(&back, &expected, "sugar {}: printed as: {}", sugar, text);
        }
    }
}

/// The spec §10 examples print back as readable surface syntax that compiles to
/// the same term. Printing is checked against the expected text, not only
/// round-tripped, so that the sugar (match, tuple binders, infix) is really used.
#[test]
fn pretty_prints_spec_examples_readably() {
    let cases = [
        (
            "let sum xs = foldl (fun acc x -> acc + x) 0 xs",
            "sum",
            "fun x -> foldl (fun x1 x2 -> x1 + x2) 0 x",
        ),
        (
            "let merge ab =
  bfix (length (fst ab) + length (snd ab) + 1)
    (fun self (a, b) ->
      match a with
      | [] -> b
      | x :: xs ->
        match b with
        | [] -> a
        | y :: ys -> if x <= y then x :: self (xs, b) else y :: self (a, ys))
    ab",
            "merge",
            "fun x -> bfix (length (fst x) + length (snd x) + 1) \
             (fun x1 (x2, x3) -> match x2 with | [] -> x3 | x4 :: x5 -> \
             match x3 with | [] -> x2 | x6 :: x7 -> \
             if x4 <= x6 then x4 :: x1 (x5, x3) else x6 :: x1 (x2, x7)) x",
        ),
        ("let g f x = f x > x + 1", "g", "fun x x1 -> x x1 > x1 + 1"),
        (
            "let f o = match o with None -> 0 | Some (a, s) -> a + s",
            "f",
            "fun x -> match x with | None -> 0 | Some (x1, x2) -> x1 + x2",
        ),
        (
            "let f t = match t with Leaf -> [1; 2] | Node (l, _, r) -> [Node (l, 0, r)] == []",
            "f",
            "fun x -> match x with | Leaf -> [1; 2] | Node (x1, _, x2) -> [Node (x1, 0, x2)] == []",
        ),
        (
            "let f = let k = (1, 2, 3) in let (a, b) = k in fun _ -> a - b",
            "f",
            "let v = (1, 2, 3) in let (v1, v2) = v in fun _ -> v1 - v2",
        ),
    ];
    for (src, name, expected) in cases {
        let mut env = TopLevel::new();
        env.add_source("p", src).unwrap();
        let mut arena = Arena::new();
        let id = env.close_definition(&mut arena, name).unwrap();
        let text = pretty(&arena, id, lib_name).unwrap();
        assert_eq!(text, expected, "{name}");
        assert_eq!(
            recompile(&text).unwrap(),
            sexpr::print(&arena, id).unwrap(),
            "{name} did not round-trip"
        );
    }
}

#[test]
fn pretty_names_free_variables_and_never_reuses_a_prim_name() {
    let mut arena = Arena::new();
    let v = arena.var(0);
    let len = arena.prim(Prim::Length);
    let body = arena.app(len, v);
    let applied = arena_app(&mut arena, body, 1);
    let open = arena.lam(applied);
    let text = pretty(&arena, open, lib_name).unwrap();
    assert!(matches!(arena.get(open), Term::Lam(_)));
    assert_eq!(text, "fun x -> length x free0");
}

fn arena_app(arena: &mut Arena, f: TermId, free: u32) -> TermId {
    let v = arena.var(free);
    arena.app(f, v)
}
