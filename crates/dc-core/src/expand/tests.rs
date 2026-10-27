use proptest::prelude::*;

use super::*;
use crate::prim::Prim;
use crate::sexpr;
use crate::term::Term;

mod types;

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
    // A let in the argument: its body sees the let's binder, not a shifted one.
    check(
        &[("(lam (lam $1))", 1)],
        "(lam (lib_0 (let 1 (add $0 $1))))",
        "(lam (lam (let 1 (add $0 $2))))",
    );
}

#[test]
fn nested_open_arguments_keep_the_shift_of_their_own_binder() {
    // The inner argument is renamed by a renaming with breakpoints at 0 and 1, and
    // its child `$0` is free only to depth 1: cutting the renaming there must keep
    // the breakpoint at 0.
    check(
        &[("(lam (lam $1))", 1)],
        "(lam (lib_0 (lam (lib_0 (add $0 $1)))))",
        "(lam (lam (lam (lam (add $1 $3)))))",
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
    /// A library, by its index modulo the number of libraries in scope.
    Lib(u8),
    Lam(Box<Gen>),
    App(Box<Gen>, Box<Gen>),
    Let(Box<Gen>, Box<Gen>),
    If(Box<Gen>, Box<Gen>, Box<Gen>),
    And(Box<Gen>, Box<Gen>),
    /// A library applied to arguments, by index as for `Lib`.
    Use(u8, Vec<Gen>),
}

fn gen_leaf() -> impl Strategy<Value = Gen> {
    prop_oneof![
        (0u32..4).prop_map(Gen::Var),
        (0i64..3).prop_map(Gen::Int),
        any::<bool>().prop_map(Gen::Bool),
        Just(Gen::Unit),
        Just(Gen::Add),
    ]
}

fn gen_node(inner: BoxedStrategy<Gen>) -> impl Strategy<Value = Gen> {
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
}

/// Terms without library references.
fn gen_term() -> impl Strategy<Value = Gen> {
    gen_leaf().prop_recursive(6, 48, 3, |inner| gen_node(inner.boxed()))
}

/// Terms with library references and uses with any number of arguments.
fn gen_term_with_libs() -> impl Strategy<Value = Gen> {
    // Mostly low variables, so that arguments are open, parameters are used and
    // nested uses build renamings with several breakpoints.
    let leaf = prop_oneof![
        6 => (0u32..3).prop_map(Gen::Var),
        1 => (3u32..5).prop_map(Gen::Var),
        3 => gen_leaf(),
        1 => any::<u8>().prop_map(Gen::Lib),
    ];
    leaf.prop_recursive(10, 96, 4, |inner| {
        let inner = inner.boxed();
        prop_oneof![
            3 => gen_node(inner.clone()),
            2 => (any::<u8>(), proptest::collection::vec(inner, 0..4))
                .prop_map(|(l, args)| Gen::Use(l, args)),
        ]
    })
}

fn build(a: &mut Arena, g: &Gen) -> TermId {
    build_in(a, g, 0)
}

/// Builds `g` where `libs` libraries are in scope. With none, a library is `()`.
fn build_in(a: &mut Arena, g: &Gen, libs: u32) -> TermId {
    let lib = |a: &mut Arena, l: u8| {
        if libs == 0 {
            a.unit()
        } else {
            a.lib(LibId(u32::from(l) % libs))
        }
    };
    match g {
        Gen::Var(i) => a.var(*i),
        Gen::Int(n) => a.int(*n),
        Gen::Bool(b) => a.bool(*b),
        Gen::Unit => a.unit(),
        Gen::Add => a.prim(Prim::Add),
        Gen::Lib(l) => lib(a, *l),
        Gen::Lam(b) => {
            let b = build_in(a, b, libs);
            a.lam(b)
        }
        Gen::App(f, x) => {
            let f = build_in(a, f, libs);
            let x = build_in(a, x, libs);
            a.app(f, x)
        }
        Gen::Let(x, b) => {
            let x = build_in(a, x, libs);
            let b = build_in(a, b, libs);
            a.let_(x, b)
        }
        Gen::If(c, t, e) => {
            let c = build_in(a, c, libs);
            let t = build_in(a, t, libs);
            let e = build_in(a, e, libs);
            a.if_(c, t, e)
        }
        Gen::And(x, y) => {
            let x = build_in(a, x, libs);
            let y = build_in(a, y, libs);
            a.and(x, y)
        }
        Gen::Use(l, args) => {
            let head = lib(a, *l);
            let args: Vec<TermId> = args.iter().map(|x| build_in(a, x, libs)).collect();
            a.apps(head, &args)
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

/// The preorder positions of a term as a tree, with their binder depths.
fn preorder_depths(a: &Arena, t: TermId, d: u32, out: &mut Vec<(TermId, u32)>) {
    out.push((t, d));
    match a.get(t) {
        Term::Lam(b) => preorder_depths(a, b, d + 1, out),
        Term::Let(x, b) => {
            preorder_depths(a, x, d, out);
            preorder_depths(a, b, d + 1, out);
        }
        node => {
            for c in crate::term::children(&node) {
                preorder_depths(a, c, d, out);
            }
        }
    }
}

/// The lowest free de Bruijn index of `t` under `d` binders, if any.
fn lowest_free(a: &Arena, t: TermId, d: u32) -> Option<u32> {
    match a.get(t) {
        Term::Var(j) => j.checked_sub(d),
        Term::Lam(b) => lowest_free(a, b, d + 1),
        Term::Let(x, b) => {
            let x = lowest_free(a, x, d);
            let b = lowest_free(a, b, d + 1);
            x.into_iter().chain(b).min()
        }
        node => crate::term::children(&node)
            .into_iter()
            .filter_map(|c| lowest_free(a, c, d))
            .min(),
    }
}

/// Shifts the free variables of `t` by `by`, which may be negative.
fn shift(a: &mut Arena, t: TermId, by: i64) -> TermId {
    map_term(a, t, 0, &mut |a, t, d| match a.get(t) {
        Term::Var(j) if j >= d => {
            let j = u32::try_from(i64::from(j) + by).expect("shifted below 0");
            Some(a.var(j))
        }
        _ => None,
    })
}

/// Abstracts the subterm at `pos` of `prog` into library 0, the way Stitch does: the
/// subterm's free variables become parameters, and, if `extra` is given, one of its
/// subterms that uses none of its binders becomes one more parameter. Every
/// occurrence of that subterm at the same depth is replaced, and the use passes the
/// subterm, which may bind variables and have free ones.
/// Returns the library and the rewritten program.
fn abstract_at(a: &mut Arena, prog: TermId, pos: usize, extra: Option<usize>) -> (LibDef, TermId) {
    let mut nodes = vec![];
    preorder(a, prog, &mut nodes);
    let s = nodes[pos];
    let n = a.free_depth(s);
    let mut inner = vec![];
    preorder_depths(a, s, 0, &mut inner);
    // `s` itself is always a candidate.
    let candidates: Vec<(TermId, u32)> = inner
        .into_iter()
        .filter(|&(e, d)| lowest_free(a, e, 0).is_none_or(|m| m >= d))
        .collect();
    let chosen = extra.map(|e| candidates[e % candidates.len()]);
    let k = n + u32::from(chosen.is_some());
    // In the closed form λ^k. body, parameter #i under d binders is $(d + k - 1 - i).
    // A free $(d + p) of s is parameter #(n - 1 - p); the extra one is #n.
    let body = map_term(a, s, 0, &mut |a, t, d| {
        if chosen == Some((t, d)) {
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
    if let Some((e, d)) = chosen {
        args.push(shift(a, e, -i64::from(d)));
    }
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

// ---------------------------------------------------------------------------
// Cost and state of the expander, measured by its own tables.
// ---------------------------------------------------------------------------

/// `λ. L (λ. L (λ. ... (λ. $n)))` with `n` uses of library 0: each use's argument
/// binds a variable and refers to the outermost binder.
fn lambda_nested_open_uses(a: &mut Arena, n: u32) -> TermId {
    let lib = a.lib(LibId(0));
    let mut t = a.var(n);
    for _ in 0..n {
        let l = a.lam(t);
        t = a.app(lib, l);
    }
    a.lam(t)
}

/// `λ^(2n+1). $(2n)`: the expansion of [`lambda_nested_open_uses`] under
/// `λ#0. λ. #0`, where every level adds the library's binder and the argument's.
fn lambda_nested_expected(a: &mut Arena, n: u32) -> TermId {
    let mut want = a.var(2 * n);
    for _ in 0..=2 * n {
        want = a.lam(want);
    }
    want
}

#[test]
fn renamings_stay_linear_for_open_arguments_that_bind() {
    let mut a = Arena::new();
    let libs = defs(&mut a, &[("(lam (lam $1))", 1)]);
    let n = 2_000;
    let prog = lambda_nested_open_uses(&mut a, n);
    let mut ex = Expander::new(&mut a, &libs).expect("libraries");
    // `run` keeps the per-program tables, so they can be measured.
    let got = ex
        .run(&mut a, Task::Expand(prog, IDENTITY), None)
        .expect("expand");
    let points = ex.renamings.stored_points();
    assert_eq!(got, lambda_nested_expected(&mut a, n));
    let n = n as usize;
    assert!(
        points <= 4 * n + 10,
        "{points} renaming points stored for {n} levels"
    );
}

#[test]
fn deep_open_arguments_that_bind_expand_without_host_recursion() {
    on_small_stack(|| {
        let mut a = Arena::new();
        let libs = defs(&mut a, &[("(lam (lam $1))", 1)]);
        let n = 200_000;
        let prog = lambda_nested_open_uses(&mut a, n);
        let got = expand(&mut a, &libs, prog).expect("expand");
        assert_eq!(got, lambda_nested_expected(&mut a, n));
    });
}

#[test]
fn an_expander_keeps_no_state_between_programs() {
    let mut a = Arena::new();
    let libs = defs(&mut a, &[("(lam (lam (pair $1 $0)))", 1)]);
    let mut ex = Expander::new(&mut a, &libs).expect("libraries");
    let state = |ex: &Expander| (ex.memo.len(), ex.sites.len(), ex.renamings.stored_points());
    assert_eq!(state(&ex), (0, 0, 0), "after preparing the libraries");
    for i in 0..20 {
        let p = parse(&mut a, &format!("(lam (lam (lib_0 (add $1 {i}))))"));
        ex.expand(&mut a, p).expect("expand");
        assert_eq!(state(&ex), (0, 0, 0), "after program {i}");
    }
    let bad = parse(&mut a, "(lam (lib_0 (lib_1 $0)))");
    assert!(ex.expand(&mut a, bad).is_err());
    assert_eq!(state(&ex), (0, 0, 0), "after a failed program");
}

#[test]
fn a_shared_subterm_is_expanded_once_under_different_renamings() {
    let mut a = Arena::new();
    // #0 placed under 0 .. m - 1 binders: m different renamings of the argument.
    let m = 40;
    // λ#0. pair #0 (λ. pair #0 (λ. ... (pair #0 ()))).
    let mut body = a.unit();
    for d in (0..m).rev() {
        let param = a.var(d);
        let pair = a.prim(Prim::Pair);
        body = a.apps(pair, &[param, body]);
        if d > 0 {
            body = a.lam(body);
        }
    }
    let term = a.lam(body);
    let libs = [LibDef { arity: 1, term }];
    // The argument: an open part, a closed part and a part that uses only its own
    // binder, each of size about `size`.
    let size = 200;
    let mut closed = a.int(0);
    let mut own = a.var(0);
    for i in 0..size {
        let one = a.int(i);
        let add = a.prim(Prim::Add);
        closed = a.apps(add, &[closed, one]);
        own = a.apps(add, &[own, one]);
    }
    let own = a.lam(own);
    let open = a.var(0);
    let pair = a.prim(Prim::Pair);
    let arg = a.apps(pair, &[open, closed, own]);
    let lib = a.lib(LibId(0));
    let use_site = a.app(lib, arg);
    let prog = a.lam(use_site);
    let mut ex = Expander::new(&mut a, &libs).expect("libraries");
    let got = ex
        .run(&mut a, Task::Expand(prog, IDENTITY), None)
        .expect("expand");
    let tasks = ex.memo.len();
    assert_eq!(got, naive_expand(&mut a, &libs, prog));
    // Each renaming of the argument costs a few tasks; its closed and self-bound
    // parts are expanded once. Without sharing it would be about `m * size`.
    let (m, size) = (m as usize, size as usize);
    assert!(
        tasks <= 10 * m + 6 * size,
        "{tasks} tasks for {m} renamings of an argument of size {size}"
    );
}

/// `λ. and p1 (and p2 (... (and pn ())))` where `pi` is `head 1 2 ... i`: one long
/// application spine whose every prefix is used separately.
fn shared_spine_prefixes(a: &mut Arena, head: TermId, n: i64) -> TermId {
    let mut prefixes = vec![];
    let mut p = head;
    for i in 1..=n {
        let arg = a.int(i);
        p = a.app(p, arg);
        prefixes.push(p);
    }
    let mut t = a.unit();
    for &p in prefixes.iter().rev() {
        t = a.and(p, t);
    }
    a.lam(t)
}

#[test]
fn shared_prefixes_of_a_spine_are_planned_in_linear_work() {
    // Small enough for the recursive reference, large enough that quadratic work
    // (about n^2 / 2 subtasks) is far above the bound.
    let n = 500;
    let mut a = Arena::new();
    // λ#0. λ. #0 + $0, applied to every prefix of a long spine: its first use is
    // over-saturated by all the later arguments.
    let libs = defs(&mut a, &[("(lam (lam (add $1 $0)))", 1)]);
    let v0 = a.var(0);
    let lib = a.lib(LibId(0));
    for head in [v0, lib] {
        let prog = shared_spine_prefixes(&mut a, head, n);
        let mut ex = Expander::new(&mut a, &libs).expect("libraries");
        let got = ex
            .run(&mut a, Task::Expand(prog, IDENTITY), None)
            .expect("expand");
        assert_eq!(got, naive_expand(&mut a, &libs, prog));
        let n = n as usize;
        assert!(
            ex.planned <= 10 * n,
            "{} subtasks planned for {n} prefixes",
            ex.planned
        );
    }
}

// ---------------------------------------------------------------------------
// Property: expansion agrees with a naive reference that expands arguments first,
// η-expands unsaturated uses by hand and substitutes with explicit shifts.
// ---------------------------------------------------------------------------

/// `body[#i := args[i]]` for a body with `k` parameters, where `#i` under `d`
/// binders is `$(d + k - 1 - i)`. Each argument is shifted by the binders it is
/// placed under.
fn naive_subst(a: &mut Arena, b: TermId, d: u32, k: u32, args: &[TermId]) -> TermId {
    let node = a.get(b);
    let new = match node {
        Term::Var(j) if j >= d => {
            let arg = args[(k - 1 - (j - d)) as usize];
            return shift(a, arg, i64::from(d));
        }
        Term::Lam(c) => Term::Lam(naive_subst(a, c, d + 1, k, args)),
        Term::Let(x, c) => {
            let x = naive_subst(a, x, d, k, args);
            Term::Let(x, naive_subst(a, c, d + 1, k, args))
        }
        Term::App(x, y) => {
            let x = naive_subst(a, x, d, k, args);
            Term::App(x, naive_subst(a, y, d, k, args))
        }
        Term::And(x, y) => {
            let x = naive_subst(a, x, d, k, args);
            Term::And(x, naive_subst(a, y, d, k, args))
        }
        Term::Or(x, y) => {
            let x = naive_subst(a, x, d, k, args);
            Term::Or(x, naive_subst(a, y, d, k, args))
        }
        Term::If(x, y, z) => {
            let x = naive_subst(a, x, d, k, args);
            let y = naive_subst(a, y, d, k, args);
            Term::If(x, y, naive_subst(a, z, d, k, args))
        }
        leaf => leaf,
    };
    a.intern(new)
}

/// The reference expansion of `t`.
fn naive_expand(a: &mut Arena, libs: &[LibDef], t: TermId) -> TermId {
    let (head, args) = spine(a, t);
    if let Term::Lib(LibId(l)) = a.get(head) {
        let def = libs[l as usize];
        let k = def.arity as usize;
        if args.len() < k {
            // λ^e. L (args ↑ e) $(e-1) ... $0, then expanded.
            let e = u32::try_from(k - args.len()).unwrap();
            let mut full: Vec<TermId> = args.iter().map(|&x| shift(a, x, i64::from(e))).collect();
            full.extend((0..e).rev().map(|i| a.var(i)));
            let saturated = a.apps(head, &full);
            let mut out = naive_expand(a, libs, saturated);
            for _ in 0..e {
                out = a.lam(out);
            }
            return out;
        }
        let mut body = naive_expand(a, &libs[..l as usize], def.term);
        for _ in 0..k {
            match a.get(body) {
                Term::Lam(b) => body = b,
                other => panic!("library lost a lambda: {other:?}"),
            }
        }
        let args: Vec<TermId> = args.iter().map(|&x| naive_expand(a, libs, x)).collect();
        let instance = naive_subst(a, body, 0, def.arity, &args[..k]);
        return a.apps(instance, &args[k..]);
    }
    if !args.is_empty() {
        let head = naive_expand(a, libs, head);
        let args: Vec<TermId> = args.iter().map(|&x| naive_expand(a, libs, x)).collect();
        return a.apps(head, &args);
    }
    map_term(a, t, 0, &mut |a, c, _| {
        (c != t).then(|| naive_expand(a, libs, c))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn expansion_agrees_with_naive_substitution(
        lib_gens in proptest::collection::vec((gen_term_with_libs(), 0u32..2), 1..4),
        prog in gen_term_with_libs(),
    ) {
        let mut a = Arena::new();
        let mut libs = vec![];
        for (i, (g, extra)) in lib_gens.iter().enumerate() {
            // Library i may use libraries 0 .. i; its free variables are parameters.
            let body = build_in(&mut a, g, u32::try_from(i).unwrap());
            let arity = a.free_depth(body) + extra;
            let mut term = body;
            for _ in 0..arity {
                term = a.lam(term);
            }
            libs.push(LibDef { arity, term });
        }
        let prog = build_in(&mut a, &prog, u32::try_from(libs.len()).unwrap());
        let want = naive_expand(&mut a, &libs, prog);
        let got = expand(&mut a, &libs, prog).expect("expand");
        prop_assert_eq!(
            sexpr::print(&a, got),
            sexpr::print(&a, want),
            "{}",
            sexpr::print(&a, prog)
        );
        prop_assert_eq!(got, want);
    }
}

// ---------------------------------------------------------------------------
// Renamings against a reference model: a renaming as a plain list of
// (position, increase) breakpoints.
// ---------------------------------------------------------------------------

/// Breakpoints as strictly increasing positions with positive increases, from a
/// list of (gap, delta) with gap 0 allowed only first.
fn breakpoints(raw: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut at = 0;
    raw.iter()
        .enumerate()
        .map(|(i, &(gap, delta))| {
            at += if i == 0 { gap } else { gap + 1 };
            (at, delta + 1)
        })
        .collect()
}

/// Interns the renaming with these breakpoints.
fn interned(rs: &mut Renamings, points: &[(u32, u32)]) -> RenId {
    let mut r = IDENTITY;
    for (i, &(at, delta)) in points.iter().enumerate().rev() {
        let prev = if i == 0 { 0 } else { points[i - 1].0 };
        r = rs.cons(at - prev, delta, r);
    }
    r
}

fn reference_apply(points: &[(u32, u32)], j: u32) -> u32 {
    j + points
        .iter()
        .filter(|&&(at, _)| at <= j)
        .map(|&(_, d)| d)
        .sum::<u32>()
}

#[test]
fn restrict_keeps_breakpoints_below_the_depth_only() {
    let mut rs = Renamings::new();
    // Breakpoints at 0 and 2: cutting at 1 keeps the one at 0.
    let r = interned(&mut rs, &[(0, 1), (2, 1)]);
    let want = interned(&mut rs, &[(0, 1)]);
    assert_eq!(rs.restrict(r, 1), want);
    let cut = rs.restrict(r, 1);
    assert_eq!(rs.apply(cut, 0), 1);
    // Breakpoints at 1 and 3: cutting at 3 keeps only the one at 1, so a term free
    // to depth 3 gets the same renaming as under a single breakpoint at 1.
    let r = interned(&mut rs, &[(1, 2), (3, 1)]);
    let want = interned(&mut rs, &[(1, 2)]);
    assert_eq!(rs.restrict(r, 3), want);
    assert_eq!(rs.restrict(r, 4), r);
    assert_eq!(rs.restrict(r, 1), IDENTITY);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// `restrict` keeps exactly the breakpoints below the depth, so equal tasks get
    /// equal renamings, and `lift` and `after_shift` agree with the model.
    #[test]
    fn renamings_agree_with_a_breakpoint_list(
        raw in proptest::collection::vec((0u32..3, 0u32..3), 0..6),
        others in proptest::collection::vec(
            proptest::collection::vec((0u32..3, 0u32..3), 0..6),
            0..4,
        ),
        depths in proptest::collection::vec(0u32..14, 1..6),
        by in 0u32..3,
    ) {
        let mut rs = Renamings::new();
        // Other renamings first, so that cells and memo entries are shared.
        for o in &others {
            let pts = breakpoints(o);
            let r = interned(&mut rs, &pts);
            for &d in &depths {
                rs.restrict(r, d);
            }
        }
        let pts = breakpoints(&raw);
        let r = interned(&mut rs, &pts);
        for &d in &depths {
            let kept: Vec<(u32, u32)> = pts.iter().copied().filter(|&(at, _)| at < d).collect();
            let want = interned(&mut rs, &kept);
            prop_assert_eq!(rs.restrict(r, d), want, "depth {} of {:?}", d, pts);
        }
        for j in 0..14 {
            prop_assert_eq!(rs.apply(r, j), reference_apply(&pts, j));
        }
        let lifted: Vec<(u32, u32)> = pts.iter().map(|&(at, d)| (at + 1, d)).collect();
        prop_assert_eq!(rs.lift(r), interned(&mut rs, &lifted));
        let mut shifted = pts.clone();
        if by > 0 {
            match shifted.first_mut() {
                Some((0, d)) => *d += by,
                _ => shifted.insert(0, (0, by)),
            }
        }
        prop_assert_eq!(rs.after_shift(r, by), interned(&mut rs, &shifted));
    }
}
