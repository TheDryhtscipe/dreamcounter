//! The deepest programs a proposer can submit, shared by the tests that check
//! them.
#![allow(dead_code)]

use std::collections::HashSet;

use dc_core::Prim;
use dc_mcp::proposer::MAX_PROGRAM_LEN;
use dc_mcp::state::Check;
use dc_surface::prelude::Prelude;

/// The error of a program nested more deeply than the surface syntax allows.
pub const TOO_DEEP: &str = "nested more than 100 levels";

/// A program, the stage its check stops at, and text its error contains.
#[derive(Clone, Debug)]
pub struct Deep {
    /// What the program does to be deep.
    pub shape: &'static str,
    /// The program.
    pub text: String,
    /// The stage its check stops at, as [`stage`] names it.
    pub stage: &'static str,
    /// Text the check's error contains, if it fails.
    pub error: Option<&'static str>,
}

/// The name of the stage `check` stopped at.
pub fn stage(check: &Check) -> &'static str {
    match check {
        Check::Compile { .. } => "compile",
        Check::Type { .. } => "type",
        Check::Run { .. } => "run",
        Check::Passed { .. } => "passed",
        Check::Extra => "extra",
    }
}

/// The error of `check`, if it failed.
pub fn error(check: &Check) -> Option<&str> {
    match check {
        Check::Compile { error } | Check::Type { error } | Check::Run { error } => Some(error),
        Check::Passed { .. } | Check::Extra => None,
    }
}

/// Panics unless `checks` stop where `programs` say, with the errors they say.
pub fn assert_checks(programs: &[Deep], checks: &[Check]) {
    assert_eq!(programs.len(), checks.len());
    for (p, c) in programs.iter().zip(checks) {
        assert_eq!(stage(c), p.stage, "{}: {c:?}", p.shape);
        if let Some(want) = p.error {
            let got = error(c).unwrap_or_default();
            assert!(got.contains(want), "{}: {c:?}", p.shape);
        }
    }
}

/// `prefix`, then `open` and `close` around `middle` as many times as fit in
/// a program, then `suffix`.
fn nested(prefix: &str, open: &str, middle: &str, close: &str, suffix: &str) -> String {
    let room = MAX_PROGRAM_LEN - prefix.len() - middle.len() - suffix.len();
    let n = room / (open.len() + close.len());
    format!(
        "{prefix}{}{middle}{}{suffix}",
        open.repeat(n),
        close.repeat(n)
    )
}

/// `prefix`, then `unit` as many times as fit in a program, then `suffix`.
fn repeated(prefix: &str, unit: &str, suffix: &str) -> String {
    let n = (MAX_PROGRAM_LEN - prefix.len() - suffix.len()) / unit.len();
    format!("{prefix}{}{suffix}", unit.repeat(n))
}

/// Short names, shortest first, that `taken` does not hold.
fn names(taken: HashSet<String>) -> impl Iterator<Item = String> {
    const FIRST: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    const REST: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    (1..)
        .flat_map(|len: u32| {
            let count = FIRST.len() * REST.len().pow(len - 1);
            (0..count).map(move |mut i| {
                let mut name = vec![FIRST[i % FIRST.len()]];
                i /= FIRST.len();
                for _ in 1..len {
                    name.push(REST[i % REST.len()]);
                    i /= REST.len();
                }
                String::from_utf8(name).unwrap()
            })
        })
        .filter(move |n| !taken.contains(n))
}

/// `source`, then as many definitions as fit in a program, each made by
/// `define(name, previous name)` and using the one before it, then `solve`
/// as the last. The first is `first(name)`. Names are as short as they can
/// be, and none is a keyword, a primitive, a prelude function, a name in
/// `source`, or `x` or `xs`, which `first` and `define` may use for
/// parameters.
fn chain(
    source: &str,
    first: impl Fn(&str) -> String,
    define: impl Fn(&str, &str) -> String,
) -> String {
    let prelude = Prelude::load().unwrap();
    let mut taken: HashSet<String> = source
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '\''))
        .map(str::to_owned)
        .collect();
    for keyword in [
        "let", "rec", "in", "fun", "if", "then", "else", "match", "with", "true", "false", "mod",
        // The entry, and the parameter names `define` may use.
        "solve", "x", "xs",
    ] {
        taken.insert(keyword.to_owned());
    }
    let mut names =
        names(taken).filter(|n| prelude.get(n).is_none() && Prim::from_name(n).is_none());
    let mut previous = names.next().unwrap();
    let mut text = format!("{source}\n{}\n", first(&previous));
    for name in names {
        let line = define(&name, &previous);
        if text.len() + line.len() + "let solve=".len() + name.len() + 1 > MAX_PROGRAM_LEN {
            break;
        }
        text.push_str(&line);
        previous = name;
    }
    text.push_str(&format!("let solve={previous}\n"));
    assert!(text.len() <= MAX_PROGRAM_LEN);
    text
}

/// Programs as deep as a submission can make them, each with where its check
/// stops.
pub fn deepest_programs() -> Vec<Deep> {
    let merge = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == "collapse/merge")
        .unwrap();
    let solution = merge.references.last().unwrap().entry;
    let too_deep = |shape, text| Deep {
        shape,
        text,
        stage: "compile",
        error: Some(TOO_DEEP),
    };
    vec![
        // The surface syntax refuses more than 100 levels in one definition.
        too_deep("brackets", nested("let solve xs = ", "(", "xs", ")", "\n")),
        too_deep(
            "lambdas",
            repeated("let solve = ", "fun a -> ", "fun xs -> xs\n"),
        ),
        too_deep("lets", repeated("let solve xs = ", "let a = 1 in ", "xs\n")),
        too_deep(
            "ifs",
            repeated("let solve xs = ", "if true then xs else ", "xs\n"),
        ),
        too_deep("conses", repeated("let solve xs = ", "1 :: ", "xs\n")),
        too_deep(
            "nested lists",
            nested("let solve xs = let a = ", "[", "1", "]", " in xs\n"),
        ),
        too_deep(
            "a long list",
            repeated("let solve xs = let a = [1", ";1", "] in xs\n"),
        ),
        too_deep(
            "a sum",
            repeated("let solve xs = let a = 1", " + 1", " in xs\n"),
        ),
        too_deep(
            "arguments",
            repeated("let solve xs = (fun a -> xs)", " 1", "\n"),
        ),
        Deep {
            shape: "brackets at the limit",
            text: format!("let solve xs = {}xs{}\n", "(".repeat(99), ")".repeat(99)),
            stage: "run",
            error: Some("example"),
        },
        // Definitions nest in one another, one `let` each in core, with no
        // limit but the program's length: a core term thousands of levels
        // deep.
        Deep {
            shape: "a chain of definitions that passes",
            text: chain(
                merge.source,
                |n| format!("let {n} = {solution}"),
                |n, p| format!("let {n}={p} "),
            ),
            stage: "passed",
            error: None,
        },
        Deep {
            shape: "a chain of definitions whose type grows",
            text: chain(
                "",
                |n| format!("let {n} xs = xs"),
                |n, p| format!("let {n} x=[{p} x] "),
            ),
            stage: "type",
            error: Some("units of work"),
        },
        Deep {
            shape: "a chain of definitions whose run doubles",
            text: chain(
                "",
                |n| format!("let {n} xs = xs"),
                |n, p| format!("let {n} x={p} ({p} x) "),
            ),
            stage: "run",
            error: Some("no result within"),
        },
    ]
}
