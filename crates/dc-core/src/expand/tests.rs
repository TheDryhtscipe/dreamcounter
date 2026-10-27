use proptest::prelude::*;

use super::*;
use crate::prim::Prim;
use crate::sexpr;
use crate::term::Term;

fn parse(arena: &mut Arena, src: &str) -> TermId {
    sexpr::parse(arena, src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn defs(arena: &mut Arena, libs: &[(&str, u32)]) -> Vec<LibDef> {
    libs.iter()
        .map(|&(src, arity)| LibDef {
            arity,
            term: parse(arena, src),
        })
        .collect()
}

/// Expands `prog` under `libs` and checks that the result is the term `expected`.
#[track_caller]
fn check(libs: &[(&str, u32)], prog: &str, expected: &str) {
    let mut a = Arena::new();
    let libs = defs(&mut a, libs);
    let p = parse(&mut a, prog);
    let want = parse(&mut a, expected);
    let got = expand(&mut a, &libs, p).unwrap_or_else(|e| panic!("{prog}: {e}"));
    assert_eq!(
        sexpr::print(&a, got),
        sexpr::print(&a, want),
        "expansion of {prog}"
    );
    assert_eq!(got, want);
}

#[track_caller]
fn check_err(libs: &[(&str, u32)], prog: &str, expected: ExpandError) {
    let mut a = Arena::new();
    let libs = defs(&mut a, libs);
    let p = parse(&mut a, prog);
    assert_eq!(expand(&mut a, &libs, p), Err(expected), "{prog}");
}

#[test]
fn saturated_use_substitutes_parameters_in_order() {
    // #0 - #1.
    let sub = ("(lam (lam (sub $1 $0)))", 2);
    check(&[sub], "(lib_0 5 3)", "(sub 5 3)");
    check(
        &[sub],
        "(add (lib_0 5 3) (lib_0 3 5))",
        "(add (sub 5 3) (sub 3 5))",
    );
}

#[test]
fn arguments_are_shifted_under_binders_of_the_body() {
    // λ#0. λ. let 1 in #0 + $0: #0 sits under a lambda and a let.
    let lib = ("(lam (lam (let 1 (add $2 $0))))", 1);
    // The argument refers to the program's two binders.
    check(
        &[lib],
        "(lam (lam (lib_0 (add $0 $1))))",
        "(lam (lam (lam (let 1 (add (add $2 $3) $0)))))",
    );
    // A let's bound expression is outside its binder.
    // λ#0. let #0 in $0.
    check(
        &[("(lam (let $0 $0))", 1)],
        "(lam (lib_0 $0))",
        "(lam (let $0 $0))",
    );
    // The argument itself binds: its bound variables are not shifted.
    check(
        &[lib],
        "(lam (lib_0 (lam (add $0 $1))))",
        "(lam (lam (let 1 (add (lam (add $0 $3)) $0))))",
    );
}

#[test]
fn duplicated_and_lazy_parameters_are_substituted_not_shared() {
    // #0 + #0.
    check(
        &[("(lam (add $0 $0))", 1)],
        "(lib_0 (mul 3 4))",
        "(add (mul 3 4) (mul 3 4))",
    );
    // if #0 = 0 then 0 else #1, and λ. #0.
    check(
        &[("(lam (lam (if (eq $1 0) 0 $0)))", 2)],
        "(lib_0 0 (div 1 0))",
        "(if (eq 0 0) 0 (div 1 0))",
    );
    check(
        &[("(lam (lam $1))", 1)],
        "(lib_0 (div 1 0))",
        "(lam (div 1 0))",
    );
}

#[test]
fn unsaturated_uses_are_eta_expanded() {
    let sub = ("(lam (lam (sub $1 $0)))", 2);
    check(&[sub], "(lib_0 5)", "(lam (sub 5 $0))");
    check(&[sub], "lib_0", "(lam (lam (sub $1 $0)))");
    // Supplied arguments move under the η binders.
    check(&[sub], "(lam (lib_0 $0))", "(lam (lam (sub $1 $0)))");
    check(
        &[sub],
        "(foldl lib_0 0)",
        "(foldl (lam (lam (sub $1 $0))) 0)",
    );
}

#[test]
fn oversaturated_uses_apply_the_expansion_to_the_rest() {
    // λ#0. λ. #0 + $0.
    let lib = ("(lam (lam (add $1 $0)))", 1);
    check(&[lib], "(lib_0 3 4)", "((lam (add 3 $0)) 4)");
    check(
        &[("(lam (lam (lam (add $2 $0))))", 1)],
        "(lib_0 3 4 5)",
        "((lam (lam (add 3 $0))) 4 5)",
    );
}

#[test]
fn nullary_libraries_expand_to_their_body() {
    check(
        &[("(mul 2 3)", 0)],
        "(add lib_0 lib_0)",
        "(add (mul 2 3) (mul 2 3))",
    );
    check(
        &[("(lam (add $0 1))", 0)],
        "(lib_0 4)",
        "((lam (add $0 1)) 4)",
    );
}

#[test]
fn library_bodies_may_use_earlier_libraries() {
    let double = ("(lam (add $0 $0))", 1);
    // λ#0. double (double #0), and a partial use of lib_0 inside lib_2.
    let quad = ("(lam (lib_0 (lib_0 $0)))", 1);
    let adder = ("(lam (lam (lam (add $2 $1))))", 2);
    check(&[double, quad], "(lib_1 7)", "(add (add 7 7) (add 7 7))");
    check(
        &[double, quad, adder],
        "(lib_2 1 (lib_1 2))",
        "(lam (add 1 (add (add 2 2) (add 2 2))))",
    );
    // A library argument that is itself a use of a library.
    check(
        &[double, quad],
        "(lib_0 (lib_1 1))",
        "(add (add (add 1 1) (add 1 1)) (add (add 1 1) (add 1 1)))",
    );
    // A library body using an earlier one unsaturated, under a binder.
    check(
        &[
            ("(lam (lam (sub $1 $0)))", 2),
            ("(lam (lam (foldl (lib_0 $1) $0)))", 1),
        ],
        "(lam (lib_1 $0))",
        "(lam (lam (foldl (lam (sub $2 $0)) $0)))",
    );
}

#[test]
fn errors() {
    let sub = ("(lam (lam (sub $1 $0)))", 2);
    check_err(&[sub], "(lib_1 1 2)", ExpandError::UnknownLib(LibId(1)));
    check_err(
        &[("(lam (lib_0 $0))", 1)],
        "1",
        ExpandError::ForwardReference {
            lib: LibId(0),
            refers_to: LibId(0),
        },
    );
    check_err(
        &[("(lam (lib_1 $0))", 1), sub],
        "1",
        ExpandError::ForwardReference {
            lib: LibId(0),
            refers_to: LibId(1),
        },
    );
    check_err(
        &[("(lam (lib_5 $0))", 1)],
        "1",
        ExpandError::UnknownLib(LibId(5)),
    );
    check_err(
        &[("(lam (add $0 $1))", 1)],
        "1",
        ExpandError::NotClosed {
            lib: LibId(0),
            free_depth: 1,
        },
    );
    check_err(
        &[("(lam (lam (add $1 $0)))", 3)],
        "1",
        ExpandError::TooFewLambdas {
            lib: LibId(0),
            arity: 3,
            lambdas: 2,
        },
    );
    // A lambda-headed body counts its own lambdas as well.
    check_err(
        &[("(lam (add $0 1))", 2)],
        "1",
        ExpandError::TooFewLambdas {
            lib: LibId(0),
            arity: 2,
            lambdas: 1,
        },
    );
}

#[test]
fn programs_without_libraries_are_unchanged() {
    check(
        &[],
        "(lam (let (add $0 1) (if (lt $0 2) $1 $0)))",
        "(lam (let (add $0 1) (if (lt $0 2) $1 $0)))",
    );
}

/// Runs `f` on a thread with a small stack, so host recursion over a deep term
/// overflows instead of passing by luck.
fn on_small_stack<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(f)
        .expect("spawn test thread")
        .join()
        .expect("test thread panicked");
}

/// `λ. L (L (... (L $0)))` with `n` uses of library 0.
fn nested_open_uses(a: &mut Arena, n: usize) -> TermId {
    let lib = a.lib(LibId(0));
    let mut prog = a.var(0);
    for _ in 0..n {
        prog = a.app(lib, prog);
    }
    a.lam(prog)
}

#[test]
fn deep_terms_expand_without_host_recursion() {
    on_small_stack(|| {
        let mut a = Arena::new();
        // λ#0. λ. #0 + $0: every level puts the open argument under one more binder.
        let libs = defs(&mut a, &[("(lam (lam (add $1 $0)))", 1)]);
        let n = 200_000;
        let prog = nested_open_uses(&mut a, n);
        // By hand: the program's $0 ends up under n body binders, so it is $n, and
        // level by level the expansion is λ. add w $0.
        let add = a.prim(Prim::Add);
        let mut want = a.var(u32::try_from(n).unwrap());
        for _ in 0..n {
            let v0 = a.var(0);
            let body = a.apps(add, &[want, v0]);
            want = a.lam(body);
        }
        let want = a.lam(want);
        let before = a.len();
        let got = expand(&mut a, &libs, prog).expect("expand");
        assert_eq!(got, want);
        // Linear: no shifted copies of intermediate levels are built.
        assert!(
            a.len() - before <= 10,
            "expansion added {} nodes",
            a.len() - before
        );
    });
}

#[test]
fn shared_duplication_stays_a_dag() {
    on_small_stack(|| {
        let mut a = Arena::new();
        // λ#0. pair #0 #0: each level doubles the tree size of the expansion.
        let libs = defs(&mut a, &[("(lam (pair $0 $0))", 1)]);
        let n = 5_000;
        let lib = a.lib(LibId(0));
        let mut prog = a.int(1);
        for _ in 0..n {
            prog = a.app(lib, prog);
        }
        let before = a.len();
        let got = expand(&mut a, &libs, prog).expect("expand");
        // One `pair x` and one `pair x x` per level.
        assert!(
            a.len() - before <= 2 * n + 10,
            "expansion added {} nodes",
            a.len() - before
        );
        let pair = a.prim(Prim::Pair);
        let mut want = a.int(1);
        for _ in 0..n {
            want = a.apps(pair, &[want, want]);
        }
        assert_eq!(got, want);
    });
}

#[test]
fn shared_duplication_under_binders_stays_linear() {
    on_small_stack(|| {
        let mut a = Arena::new();
        // λ#0. λ. pair #0 (pair #0 $0): an open argument, duplicated, under a binder.
        let libs = defs(&mut a, &[("(lam (lam (pair $1 (pair $1 $0))))", 1)]);
        let n = 100_000;
        let prog = nested_open_uses(&mut a, n);
        let before = a.len();
        let got = expand(&mut a, &libs, prog).expect("expand");
        let added = a.len() - before;
        // Three nodes per level for `λ. pair w (pair w $0)` plus the applications.
        assert!(added <= 6 * n + 10, "expansion added {added} nodes");
        let pair = a.prim(Prim::Pair);
        let mut want = a.var(u32::try_from(n).unwrap());
        for _ in 0..n {
            let v0 = a.var(0);
            let inner = a.apps(pair, &[want, v0]);
            let body = a.apps(pair, &[want, inner]);
            want = a.lam(body);
        }
        let want = a.lam(want);
        assert_eq!(got, want);
    });
}

// ---------------------------------------------------------------------------
// Property: abstracting a subterm into a library and expanding gives the same
// term back, id for id.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Gen {
    Var(u32),
    Int(i64),
    Bool(bool),
    Unit,
    Add,
    Lam(Box<Gen>),
    App(Box<Gen>, Box<Gen>),
    Let(Box<Gen>, Box<Gen>),
    If(Box<Gen>, Box<Gen>, Box<Gen>),
    And(Box<Gen>, Box<Gen>),
}

fn gen_term() -> impl Strategy<Value = Gen> {
    let leaf = prop_oneof![
        (0u32..4).prop_map(Gen::Var),
        (0i64..3).prop_map(Gen::Int),
        any::<bool>().prop_map(Gen::Bool),
        Just(Gen::Unit),
        Just(Gen::Add),
    ];
    leaf.prop_recursive(6, 48, 3, |inner| {
        prop_oneof![
            inner.clone().prop_map(|b| Gen::Lam(Box::new(b))),
            (inner.clone(), inner.clone()).prop_map(|(f, x)| Gen::App(Box::new(f), Box::new(x))),
            (inner.clone(), inner.clone()).prop_map(|(x, b)| Gen::Let(Box::new(x), Box::new(b))),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(|(c, t, e)| Gen::If(
                Box::new(c),
                Box::new(t),
                Box::new(e)
            )),
            (inner.clone(), inner).prop_map(|(x, y)| Gen::And(Box::new(x), Box::new(y))),
        ]
    })
}

fn build(a: &mut Arena, g: &Gen) -> TermId {
    match g {
        Gen::Var(i) => a.var(*i),
        Gen::Int(n) => a.int(*n),
        Gen::Bool(b) => a.bool(*b),
        Gen::Unit => a.unit(),
        Gen::Add => a.prim(Prim::Add),
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
        Gen::And(x, y) => {
            let x = build(a, x);
            let y = build(a, y);
            a.and(x, y)
        }
    }
}

/// The preorder positions of a term as a tree, with their nodes.
fn preorder(a: &Arena, t: TermId, out: &mut Vec<TermId>) {
    out.push(t);
    for c in crate::term::children(&a.get(t)) {
        preorder(a, c, out);
    }
}

/// Rebuilds `t` with `f` applied to each node bottom-up, passing the binder depth.
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
    let new = match node {
        Term::Lam(b) => Term::Lam(map_term(a, b, d + 1, f)),
        Term::Let(x, b) => {
            let x = map_term(a, x, d, f);
            Term::Let(x, map_term(a, b, d + 1, f))
        }
        Term::App(x, y) => {
            let x = map_term(a, x, d, f);
            Term::App(x, map_term(a, y, d, f))
        }
        Term::And(x, y) => {
            let x = map_term(a, x, d, f);
            Term::And(x, map_term(a, y, d, f))
        }
        Term::Or(x, y) => {
            let x = map_term(a, x, d, f);
            Term::Or(x, map_term(a, y, d, f))
        }
        Term::If(x, y, z) => {
            let x = map_term(a, x, d, f);
            let y = map_term(a, y, d, f);
            Term::If(x, y, map_term(a, z, d, f))
        }
        leaf => leaf,
    };
    a.intern(new)
}

/// Replaces the subterm at preorder position `pos` of `t` with `with`.
fn replace_at(a: &mut Arena, t: TermId, pos: usize, with: TermId) -> TermId {
    fn go(a: &mut Arena, t: TermId, pos: &mut usize, with: TermId) -> TermId {
        if *pos == 0 {
            *pos = usize::MAX;
            return with;
        }
        if *pos == usize::MAX {
            return t;
        }
        *pos -= 1;
        let node = a.get(t);
        let mut kids = vec![];
        for c in crate::term::children(&node) {
            kids.push(go(a, c, pos, with));
        }
        let new = match node {
            Term::Lam(_) => Term::Lam(kids[0]),
            Term::Let(..) => Term::Let(kids[0], kids[1]),
            Term::App(..) => Term::App(kids[0], kids[1]),
            Term::And(..) => Term::And(kids[0], kids[1]),
            Term::Or(..) => Term::Or(kids[0], kids[1]),
            Term::If(..) => Term::If(kids[0], kids[1], kids[2]),
            leaf => leaf,
        };
        a.intern(new)
    }
    let mut pos = pos;
    go(a, t, &mut pos, with)
}

/// Abstracts the subterm at `pos` of `prog` into library 0, the way Stitch does: the
/// subterm's free variables become parameters, and, if `extra` names a closed
/// subterm of it, every occurrence of that subterm becomes one more parameter.
/// Returns the library and the rewritten program.
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
    // In the closed form λ^k. body, parameter #i under d binders is $(d + k - 1 - i).
    // A free $(d + p) of s is parameter #(n - 1 - p); the extra one is #n.
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
    let rewritten = replace_at(a, prog, pos, use_site);
    (LibDef { arity: k, term }, rewritten)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn abstracting_then_expanding_gives_the_same_term(
        g in gen_term(),
        pos in any::<usize>(),
        extra in proptest::option::of(any::<usize>()),
    ) {
        let mut a = Arena::new();
        let prog = build(&mut a, &g);
        let mut nodes = vec![];
        preorder(&a, prog, &mut nodes);
        let pos = pos % nodes.len();
        let (lib, rewritten) = abstract_at(&mut a, prog, pos, extra);
        prop_assert!(a.is_closed(lib.term));
        let got = expand(&mut a, &[lib], rewritten).expect("expand");
        prop_assert_eq!(sexpr::print(&a, got), sexpr::print(&a, prog));
        prop_assert_eq!(got, prog);
    }

    /// Abstracting twice, the second library using the first, also round-trips.
    #[test]
    fn nested_abstraction_round_trips(
        g in gen_term(),
        pos in any::<usize>(),
        pos2 in any::<usize>(),
    ) {
        let mut a = Arena::new();
        let prog = build(&mut a, &g);
        let mut nodes = vec![];
        preorder(&a, prog, &mut nodes);
        let (lib0, once) = abstract_at(&mut a, prog, pos % nodes.len(), None);
        // Abstract again in the rewritten program, renaming the new library to 1.
        let mut nodes = vec![];
        preorder(&a, once, &mut nodes);
        // Cutting a proper prefix of a `lib_0` spine out (bare `lib_0`, or `lib_0` with
        // too few arguments) would make that use unsaturated, which η-expansion
        // charges for (spec §8); Stitch never does that.
        let at = pos2 % nodes.len();
        let (head, args) = spine(&a, nodes[at]);
        prop_assume!(a.get(head) != Term::Lib(LibId(0)) || args.len() >= lib0.arity as usize);
        let (lib1, twice) = abstract_at(&mut a, once, at, None);
        // `abstract_at` names every new library 0, so rename the second one's use
        // site to 1. Its arguments are variables, so its head is the only library.
        let mut nodes = vec![];
        preorder(&a, twice, &mut nodes);
        let renamed = map_term(&mut a, nodes[at], 0, &mut |a, t, _| match a.get(t) {
            Term::Lib(LibId(0)) => Some(a.lib(LibId(1))),
            _ => None,
        });
        let twice = replace_at(&mut a, twice, at, renamed);
        let got = expand(&mut a, &[lib0, lib1], twice).expect("expand");
        prop_assert_eq!(sexpr::print(&a, got), sexpr::print(&a, prog));
        prop_assert_eq!(got, prog);
    }
}
