//! Robustness tests: compile time, nesting depth and arena ownership.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use dc_core::{sexpr, Arena, LibId, Prim, TermId};
use dc_surface::pretty::{
    pretty_unchecked, PrettyError, INLINE_DEPTH, MAX_PRETTY_DEPTH, MAX_PRETTY_NODES,
};
use dc_surface::{compile_expression, pretty, ErrorKind, SurfaceError, TopLevel, MAX_DEPTH};

/// Compiles `src` on another thread and fails if it takes longer than `limit`.
fn compile_within(src: String, limit: Duration) -> Result<String, SurfaceError> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut arena = Arena::new();
        let r = compile_expression(&mut arena, &src).map(|t| sexpr::print(&arena, t).unwrap());
        let _ = tx.send(r);
    });
    rx.recv_timeout(limit)
        .unwrap_or_else(|_| panic!("compiling took longer than {limit:?}"))
}

#[test]
fn nested_greater_than_compiles_each_operand_once() {
    // Each level puts the previous level inside the right operand of '>'. When a
    // right operand is compiled twice per level, the time doubles per level and 22
    // levels take minutes; compiled once, it takes milliseconds.
    let levels = 22;
    let mut e = "x".to_owned();
    for _ in 0..levels {
        e = format!("((x + 1) > (if {e} then 1 else 0))");
    }
    let core = compile_within(format!("fun x -> {e}"), Duration::from_secs(10)).unwrap();
    assert_eq!(core.matches("(lt ").count(), levels);
    // Neither operand is a variable or literal, so every level keeps the let.
    assert_eq!(core.matches("(let (add ").count(), levels);
}

#[test]
fn greater_than_with_a_trivial_right_operand_swaps_without_a_let() {
    // The right operand is compiled under the hidden binder and found trivial; its
    // index must be shifted back out.
    let mut arena = Arena::new();
    let t = compile_expression(&mut arena, "fun y -> fun x -> (x + 1) > y").unwrap();
    assert_eq!(
        sexpr::print(&arena, t).unwrap(),
        "(lam (lam (lt $1 (add $0 1))))"
    );
    let t = compile_expression(&mut arena, "fun x -> (x + 1) >= 3").unwrap();
    assert_eq!(sexpr::print(&arena, t).unwrap(), "(lam (le 3 (add $0 1)))");
}

/// Builds an input nested `n` deep, of one of several shapes.
fn nested(shape: &str, n: usize) -> String {
    match shape {
        "parens" => format!("{}1{}", "(".repeat(n), ")".repeat(n)),
        "cons" => format!("{}[]", "1 :: ".repeat(n)),
        "plus" => format!("0{}", " + 1".repeat(n)),
        "let" => format!("{}x", "let x = 1 in ".repeat(n)),
        "neg" => format!("{}x", "- ".repeat(n)).replace("x", "(fun x -> x) 1"),
        "fun" => format!("{}1", "fun x -> ".repeat(n)),
        "if" => format!("{}1{}", "if true then ".repeat(n), " else 0".repeat(n)),
        "list" => format!("[{}]", vec!["1"; n].join("; ")),
        "tuple" => format!("({})", vec!["1"; n].join(", ")),
        "match" => {
            format!(
                "fun xs -> {}0",
                "match xs with [] -> 0 | h :: t -> (".repeat(n)
            ) + &")".repeat(n)
        }
        _ => unreachable!(),
    }
}

const SHAPES: [&str; 10] = [
    "parens", "cons", "plus", "let", "neg", "fun", "if", "list", "tuple", "match",
];

#[test]
fn deep_input_is_an_error_not_a_stack_overflow() {
    for shape in SHAPES {
        let src = nested(shape, 20_000);
        let mut arena = Arena::new();
        let e = match compile_expression(&mut arena, &src) {
            Ok(_) => panic!("{shape}: expected a nesting error"),
            Err(e) => e,
        };
        assert_eq!(e.kind, ErrorKind::TooDeep, "{shape}: {e}");
        assert!(e.line >= 1 && e.column >= 1, "{shape}: {e}");
        assert!(
            e.message.contains(&MAX_DEPTH.to_string()),
            "{shape}: the message should state the limit: {e}"
        );
    }
}

#[test]
fn deep_program_definition_is_an_error_not_a_stack_overflow() {
    let src = format!("let main =\n  {}", nested("parens", 20_000));
    let mut env = TopLevel::new();
    let e = env.add_source("prog", &src).unwrap_err();
    assert_eq!(e.kind, ErrorKind::TooDeep, "{e}");
    assert_eq!(e.line, 2, "{e}");
}

#[test]
fn input_just_inside_the_limit_compiles_and_pretty_prints() {
    // A limit is only useful if everything under it works, on a test thread's
    // default stack, in a debug build. Shapes weigh differently per level, so find
    // each shape's deepest accepted input.
    for shape in SHAPES {
        let accepts = |n: usize| compile_expression(&mut Arena::new(), &nested(shape, n)).is_ok();
        let (mut lo, mut hi) = (1, MAX_DEPTH + 1);
        assert!(accepts(lo), "{shape} at depth 1");
        assert!(!accepts(hi), "{shape} past the limit");
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if accepts(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // The heaviest shape (a match per level) costs five levels each.
        assert!(lo >= MAX_DEPTH / 5, "{shape}: only {lo} levels accepted");
        let mut arena = Arena::new();
        let t = compile_expression(&mut arena, &nested(shape, lo)).unwrap();
        let printed = pretty(&arena, t, |l: LibId| format!("lib_{}", l.0)).unwrap();
        assert!(!printed.is_empty(), "{shape}");
        let s = sexpr::print(&arena, t).unwrap();
        assert_eq!(sexpr::parse(&mut arena, &s).ok(), Some(t), "{shape}");
    }
}

#[test]
fn inline_names_work_with_a_different_arena() {
    let mut a1 = Arena::new();
    let lib = a1.lib(LibId(4));
    let v0 = a1.var(0);
    let body = a1.app(lib, v0);
    let k = a1.lam(body);
    let mut env = TopLevel::new();
    env.define_inline(&a1, "k", k).unwrap();
    env.add_source("p", "let f x = k x").unwrap();

    // a2 holds other terms first, so the same TermId means something else there.
    let mut a2 = Arena::new();
    for i in 0..20 {
        a2.int(1000 + i);
    }
    let e = env.close_expression(&mut a2, "e", "k 1").unwrap();
    assert_eq!(sexpr::print(&a2, e).unwrap(), "((lam (lib_4 $0)) 1)");
    let f = env.close_definition(&mut a2, "f").unwrap();
    assert_eq!(sexpr::print(&a2, f).unwrap(), "(lam ((lam (lib_4 $0)) $0))");
    let k2 = env.close_definition(&mut a2, "k").unwrap();
    assert_eq!(sexpr::print(&a2, k2).unwrap(), "(lam (lib_4 $0))");
}

/// A core term with `n` levels of one shape around `1`, built directly (the
/// surface front end does not accept anything this deep).
fn core_nested(a: &mut Arena, shape: &str, n: usize) -> TermId {
    let one = a.int(1);
    let v0 = a.var(0);
    let v1 = a.var(1);
    let tt = a.bool(true);
    let lt = a.prim(Prim::Lt);
    let not = a.prim(Prim::Not);
    let fst = a.prim(Prim::Fst);
    let snd = a.prim(Prim::Snd);
    let cons = a.prim(Prim::Cons);
    let nil = a.prim(Prim::Nil);
    let pair = a.prim(Prim::Pair);
    let list_case = a.prim(Prim::ListCase);
    let some = a.prim(Prim::Some);
    let mut t = one;
    for _ in 0..n {
        t = match shape {
            // fun x -> fun x1 -> ... (each binder used below)
            "lam" => {
                let body = a.apps(pair, &[v0, t]);
                a.lam(body)
            }
            // let v = 1 in v + ...
            "let" => {
                let add = a.prim(Prim::Add);
                let body = a.apps(add, &[v0, t]);
                a.let_(one, body)
            }
            // let f x = x in f (...)
            "let_fun" => {
                let id = a.lam(v0);
                let body = a.app(v0, t);
                a.let_(id, body)
            }
            "if" => a.if_(tt, t, one),
            "and_or" => {
                let o = a.or(tt, t);
                a.and(o, tt)
            }
            "arg" => a.app(not, t),
            "head" => a.app(t, one),
            "some" => a.app(some, t),
            "cons" => a.apps(cons, &[t, nil]),
            "tuple" => a.apps(pair, &[t, one]),
            // match xs with [] -> 1 | h :: tl -> (...)
            "match" => {
                let empty = a.lam(one);
                let full_body = a.lam(t);
                let full = a.lam(full_body);
                a.apps(list_case, &[v0, empty, full])
            }
            // fun (a, b) -> ... via fst/snd lets
            "tuple_binder" => {
                let f = a.app(fst, v0);
                let s = a.app(snd, v1);
                let inner = a.apps(pair, &[v0, t]);
                let l2 = a.let_(s, inner);
                let l1 = a.let_(f, l2);
                a.lam(l1)
            }
            // (t + 1) > (...) as a let around lt
            "greater" => {
                let add = a.prim(Prim::Add);
                let lhs = a.apps(add, &[v0, one]);
                let cmp = a.apps(lt, &[t, v0]);
                let l = a.let_(lhs, cmp);
                a.lam(l)
            }
            _ => unreachable!("{shape}"),
        };
    }
    t
}

const CORE_SHAPES: [&str; 13] = [
    "lam",
    "let",
    "let_fun",
    "if",
    "and_or",
    "arg",
    "head",
    "some",
    "cons",
    "tuple",
    "match",
    "tuple_binder",
    "greater",
];

fn lib_text(l: LibId) -> String {
    format!("lib_{}", l.0)
}

#[test]
fn deep_core_terms_are_a_pretty_error_not_a_stack_overflow() {
    for shape in CORE_SHAPES {
        let mut arena = Arena::new();
        let t = core_nested(&mut arena, shape, 100_000);
        assert_eq!(
            pretty(&arena, t, lib_text),
            Err(PrettyError::TooDeep),
            "{shape}"
        );
        for sugar in [true, false] {
            assert_eq!(
                pretty_unchecked(&arena, t, lib_text, sugar),
                Err(PrettyError::TooDeep),
                "{shape}"
            );
        }
    }
}

#[test]
fn core_terms_up_to_the_pretty_limit_print() {
    // Everything within the limit must print on a test thread's default stack in a
    // debug build. Find each shape's deepest accepted term and print it.
    for shape in CORE_SHAPES {
        let accepts = |n: usize| {
            let mut arena = Arena::new();
            let t = core_nested(&mut arena, shape, n);
            pretty(&arena, t, lib_text).is_ok()
        };
        let (mut lo, mut hi) = (0, MAX_PRETTY_DEPTH + 1);
        assert!(!accepts(hi), "{shape} past the limit");
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if accepts(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // The heaviest shape costs six levels each.
        assert!(
            lo >= MAX_PRETTY_DEPTH / 6,
            "{shape}: only {lo} levels accepted"
        );
        let mut arena = Arena::new();
        let t = core_nested(&mut arena, shape, lo);
        for sugar in [true, false] {
            let text = pretty_unchecked(&arena, t, lib_text, sugar).unwrap();
            assert!(!text.is_empty(), "{shape}");
        }
    }
}

#[test]
fn pretty_depth_counts_the_deepest_path_through_shared_nodes() {
    // `deep` is reached both at the top and under a long chain; the second path is
    // the one over the limit, and it is found second.
    let mut arena = Arena::new();
    let deep = core_nested(&mut arena, "arg", MAX_PRETTY_DEPTH - 10);
    let mut chain = deep;
    for _ in 0..20 {
        chain = arena.lam(chain);
    }
    let pair = arena.prim(Prim::Pair);
    let both = arena.apps(pair, &[deep, chain]);
    assert_eq!(pretty(&arena, both, lib_text), Err(PrettyError::TooDeep));
    let both = arena.apps(pair, &[chain, deep]);
    assert_eq!(pretty(&arena, both, lib_text), Err(PrettyError::TooDeep));
    assert!(pretty(&arena, deep, lib_text).is_ok());
    assert!(PrettyError::TooDeep
        .to_string()
        .contains(&MAX_PRETTY_DEPTH.to_string()));
}

/// The deepest term of `shape` that [`pretty`] accepts.
fn deepest_printable(shape: &str) -> (Arena, TermId) {
    let accepts = |n: usize| {
        let mut arena = Arena::new();
        let t = core_nested(&mut arena, shape, n);
        pretty(&arena, t, lib_text).is_ok()
    };
    let (mut lo, mut hi) = (0, MAX_PRETTY_DEPTH + 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if accepts(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mut arena = Arena::new();
    let t = core_nested(&mut arena, shape, lo);
    (arena, t)
}

/// Set for the child process that [`pretty_needs_little_of_the_callers_stack`]
/// starts.
const STACK_PROBE: &str = "DC_PRETTY_STACK_PROBE";

#[test]
fn pretty_needs_little_of_the_callers_stack() {
    // A stack overflow aborts the process, so the probe runs in a child process.
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "pretty_stack_probe",
            "--test-threads",
            "1",
        ])
        .env(STACK_PROBE, "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "the probe failed: {}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "the probe did not run: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
#[ignore = "run in a child process by pretty_needs_little_of_the_callers_stack"]
fn pretty_stack_probe() {
    if std::env::var_os(STACK_PROBE).is_none() {
        return;
    }
    // A caller with only 256 KiB of stack left prints the deepest term of every
    // shape, and every term shallow enough to be printed on the caller's thread.
    thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            for shape in CORE_SHAPES {
                for n in 1..=INLINE_DEPTH {
                    let mut arena = Arena::new();
                    let t = core_nested(&mut arena, shape, n);
                    assert!(pretty(&arena, t, lib_text).is_ok(), "{shape} {n}");
                    for sugar in [true, false] {
                        assert!(pretty_unchecked(&arena, t, lib_text, sugar).is_ok());
                    }
                }
                let (arena, t) = deepest_printable(shape);
                assert!(pretty(&arena, t, lib_text).is_ok(), "{shape}");
                for sugar in [true, false] {
                    assert!(pretty_unchecked(&arena, t, lib_text, sugar).is_ok());
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn pretty_refuses_terms_too_large_written_out() {
    // `(if c t t)` nested n deep has n + 2 distinct nodes and about 2^(n+1) as a
    // tree.
    let shared_ifs = |arena: &mut Arena, n: usize| {
        let c = arena.bool(true);
        let mut t = arena.int(1);
        for _ in 0..n {
            t = arena.if_(c, t, t);
        }
        t
    };
    let mut arena = Arena::new();
    let small = shared_ifs(&mut arena, 10);
    assert!(pretty(&arena, small, lib_text).is_ok());
    // Past the node limit, though far within the depth limit.
    let n = (MAX_PRETTY_NODES as f64).log2().ceil() as usize;
    let big = shared_ifs(&mut arena, n);
    assert_eq!(
        pretty(&arena, big, lib_text).err(),
        Some(PrettyError::TooLarge)
    );
    for sugar in [true, false] {
        assert_eq!(
            pretty_unchecked(&arena, big, lib_text, sugar).err(),
            Some(PrettyError::TooLarge)
        );
    }
    // The largest: would print 2^60 subterms.
    let huge = shared_ifs(&mut arena, 60);
    assert_eq!(
        pretty(&arena, huge, lib_text).err(),
        Some(PrettyError::TooLarge)
    );
    assert!(PrettyError::TooLarge
        .to_string()
        .contains(&MAX_PRETTY_NODES.to_string()));
}
