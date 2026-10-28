//! The front end's stack promise: anything within [`MAX_DEPTH`] parses, desugars
//! and pretty-prints on a 1 MiB stack in a debug build, and anything deeper is a
//! [`ErrorKind::TooDeep`] error on that stack, for every construct that nests.
//! The probe uses half that, so that the promise keeps some headroom as frame
//! sizes change between compiler versions.
//!
//! A stack overflow aborts the process, so each shape runs in a child process
//! (this test binary, running the ignored probe), and the parent checks that it
//! exited cleanly.

use std::process::Command;
use std::thread;

use dc_core::{Arena, LibId};
use dc_surface::{compile_expression, pretty, ErrorKind, TopLevel, MAX_DEPTH};

/// Half the stack the promise is made for.
const STACK: usize = 512 << 10;

/// Names the shape the child process probes.
const PROBE: &str = "DC_SURFACE_STACK_PROBE";

/// Builds an expression with `n` levels of `shape`. Every shape nests at least
/// one level per `n`, so `n = MAX_DEPTH + 1` is always too deep.
fn nested(shape: &str, n: usize) -> String {
    let rep = |s: &str| s.repeat(n);
    match shape {
        // Expressions.
        "parens" => format!("{}1{}", rep("("), rep(")")),
        "arg" => format!("{}x{}", rep("not ("), rep(")")),
        "head" => format!("{}f{}", rep("("), rep(" 1)")),
        "neg" => format!("{}1", rep("- (")) + &rep(")"),
        "neg_bare" => format!("{}x", rep("- ")),
        "or" => format!("{}false", rep("true || ")),
        "and" => format!("{}true", rep("true && ")),
        "cmp" => format!("{}1{}", rep("1 < ("), rep(")")),
        "greater" => format!("{}1{}", rep("(x + 1) > ("), rep(")")),
        "cons" => format!("{}[]", rep("1 :: ")),
        "plus" => format!("0{}", rep(" + 1")),
        "times" => format!("{}1{}", rep("2 * ("), rep(")")),
        "list" => format!("[{}]", vec!["1"; n].join("; ")),
        "list_nested" => format!("{}1{}", rep("["), rep("]")),
        "tuple" => format!("({})", vec!["1"; n].join(", ")),
        "tuple_nested" => format!("{}1{}", rep("(1, "), rep(")")),
        "args" => format!("f{}", rep(" 1")),
        "let_body" => format!("{}x", rep("let x = 1 in ")),
        "let_rhs" => format!("{}1{}", rep("let x = "), rep(" in x")),
        "fun" => format!("{}1", rep("fun x -> ")),
        "params" => format!("fun{} -> 1", rep(" x")),
        "if_then" => format!("{}1{}", rep("if true then "), rep(" else 0")),
        "if_cond" => format!("{}true{}", rep("if "), rep(" then true else false")),
        "match_arm" => format!("{}0", rep("match xs with [] -> 0 | h :: t -> (")) + &rep(")"),
        "match_scrutinee" => format!("{}[]{}", rep("match "), rep(" with _ -> []")),
        "node" => format!("{}Leaf{}", rep("Node (Leaf, 1, "), rep(")")),
        // Patterns.
        "pat_tuple" => {
            let opens: String = (0..n).map(|i| format!("(a{i}, ")).collect();
            format!("fun {opens}b{} -> 1", rep(")"))
        }
        "pat_parens" => format!("fun {}a{} -> 1", rep("("), rep(")")),
        "pat_cons" => format!("match xs with {}t -> 1 | _ -> 0", rep("h :: ")),
        "pat_list" => format!("match xs with {}{} -> 1 | _ -> 0", rep("["), rep("]")),
        _ => unreachable!("{shape}"),
    }
}

const SHAPES: [&str; 30] = [
    "parens",
    "arg",
    "head",
    "neg",
    "neg_bare",
    "or",
    "and",
    "cmp",
    "greater",
    "cons",
    "plus",
    "times",
    "list",
    "list_nested",
    "tuple",
    "tuple_nested",
    "args",
    "let_body",
    "let_rhs",
    "fun",
    "params",
    "if_then",
    "if_cond",
    "match_arm",
    "match_scrutinee",
    "node",
    "pat_tuple",
    "pat_parens",
    "pat_cons",
    "pat_list",
];

/// The source as an expression with its free names bound, so that shallow
/// instances compile.
fn closed(shape: &str, n: usize) -> String {
    format!("fun f x xs -> {}", nested(shape, n))
}

/// Compiles and prints `shape` at every depth up to just past the limit, and far
/// past it, as an expression and as a top-level definition.
fn probe(shape: &str) {
    let lib = |l: LibId| format!("lib_{}", l.0);
    // The shallowest instance is a valid program, so every stage runs.
    let mut arena = Arena::new();
    let t = compile_expression(&mut arena, &closed(shape, 1))
        .unwrap_or_else(|e| panic!("{shape} at depth 1: {e}"));
    assert!(pretty(&arena, t, lib).is_ok(), "{shape} at depth 1");
    for n in (1..=MAX_DEPTH + 1).chain([MAX_DEPTH * 10, 20_000]) {
        let src = closed(shape, n);
        let mut arena = Arena::new();
        match compile_expression(&mut arena, &src) {
            Ok(t) => {
                assert!(n <= MAX_DEPTH, "{shape} at depth {n} was accepted");
                // A deep term is printed on a thread of its own; a shallow one
                // here.
                let _ = pretty(&arena, t, lib);
            }
            Err(e) if n > MAX_DEPTH => assert_eq!(e.kind, ErrorKind::TooDeep, "{shape} {n}"),
            // Shallower nested patterns, for example, are rejected by the
            // desugarer; only reaching it matters here.
            Err(_) => {}
        }
        let mut env = TopLevel::new();
        let r = env.add_source(
            "prog",
            &format!("let main f x xs =\n  {}", nested(shape, n)),
        );
        if n > MAX_DEPTH {
            assert_eq!(r.unwrap_err().kind, ErrorKind::TooDeep, "{shape} {n}");
        }
    }
}

#[test]
fn every_nesting_shape_fits_in_half_a_mib_of_stack() {
    let mut failed = vec![];
    for shape in SHAPES {
        let out = Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "stack_probe", "--test-threads", "1"])
            .env(PROBE, shape)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() || !stdout.contains("1 passed") {
            failed.push(format!(
                "{shape}: {}\n{stdout}\n{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
#[ignore = "run in a child process by every_nesting_shape_fits_in_half_a_mib_of_stack"]
fn stack_probe() {
    let Some(shape) = std::env::var_os(PROBE) else {
        return;
    };
    let shape = shape.into_string().unwrap();
    thread::Builder::new()
        .stack_size(STACK)
        .spawn(move || probe(&shape))
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn every_shape_is_valid_when_shallow() {
    // The probe relies on this, and it runs here on the default stack too.
    for shape in SHAPES {
        let mut arena = Arena::new();
        if let Err(e) = compile_expression(&mut arena, &closed(shape, 1)) {
            panic!("{shape}: {e}");
        }
    }
}
