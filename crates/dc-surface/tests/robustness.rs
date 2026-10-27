//! Robustness tests: compile time, nesting depth and arena ownership.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use dc_core::{sexpr, Arena, LibId};
use dc_surface::{compile_expression, pretty, ErrorKind, SurfaceError, TopLevel, MAX_DEPTH};

/// Compiles `src` on another thread and fails if it takes longer than `limit`.
fn compile_within(src: String, limit: Duration) -> Result<String, SurfaceError> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut arena = Arena::new();
        let r = compile_expression(&mut arena, &src).map(|t| sexpr::print(&arena, t));
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
    assert_eq!(sexpr::print(&arena, t), "(lam (lam (lt $1 (add $0 1))))");
    let t = compile_expression(&mut arena, "fun x -> (x + 1) >= 3").unwrap();
    assert_eq!(sexpr::print(&arena, t), "(lam (le 3 (add $0 1)))");
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
        let printed = pretty(&arena, t, |l: LibId| format!("lib_{}", l.0));
        assert!(!printed.is_empty(), "{shape}");
        let s = sexpr::print(&arena, t);
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
    assert_eq!(sexpr::print(&a2, e), "((lam (lib_4 $0)) 1)");
    let f = env.close_definition(&mut a2, "f").unwrap();
    assert_eq!(sexpr::print(&a2, f), "(lam ((lam (lib_4 $0)) $0))");
    let k2 = env.close_definition(&mut a2, "k").unwrap();
    assert_eq!(sexpr::print(&a2, k2), "(lam (lib_4 $0))");
}
