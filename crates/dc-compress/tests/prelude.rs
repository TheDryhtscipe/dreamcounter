//! The bridge with programs that use the prelude's functions: Stitch sees each
//! one as a token with its name, in the caller's process and in the worker, and
//! a step is checked with the prelude expanded first.

use std::sync::atomic::AtomicBool;

use dc_compress::worker::Worker;
use dc_compress::{
    accept_steps_with, compress_isolated_with, compress_with, expand_prelude, library_names,
    CompressConfig, CompressError, Compression, FrontierProgram, IsolatedRun, StitchStep,
};
use dc_core::{expand, sexpr, Arena, LibDef, LibId, Prim, Term, TermId};
use dc_stitch::to_expr;
use dc_stitch::{EncodeError, NameError};
use dc_surface::prelude::{Prelude, PreludeMode, PRELUDE_LIB_BASE};

const WORKER: &str = env!("CARGO_BIN_EXE_dc-stitch-worker");

/// A cancel flag that is never set.
static NEVER: AtomicBool = AtomicBool::new(false);

fn compile(prelude: &Prelude, arena: &mut Arena, src: &str) -> TermId {
    prelude
        .top_level(PreludeMode::On)
        .close_expression(arena, "test", src)
        .unwrap_or_else(|e| panic!("{src}: {e}"))
}

/// Four programs that share the shape `map (+ k) (filter (> j) xs)`.
fn shared(prelude: &Prelude, arena: &mut Arena) -> Vec<FrontierProgram> {
    [(1, 2), (3, 4), (5, 6), (7, 8)]
        .iter()
        .enumerate()
        .map(|(i, (k, j))| {
            let src = format!("fun xs -> map (fun x -> x + {k}) (filter (fun x -> x > {j}) xs)");
            FrontierProgram {
                task: format!("task_{i}"),
                program: compile(prelude, arena, &src),
            }
        })
        .collect()
}

/// Whether `id` refers to a prelude function, and whether it holds `foldr`, the
/// combinator `map` and `filter` are defined with.
fn scan(arena: &Arena, id: TermId) -> (bool, bool) {
    let (mut prelude_ref, mut foldr) = (false, false);
    let mut stack = vec![id];
    while let Some(t) = stack.pop() {
        let node = arena.get(t);
        match node {
            Term::Lib(LibId(l)) if l >= PRELUDE_LIB_BASE => prelude_ref = true,
            Term::Prim(Prim::Foldr) => foldr = true,
            _ => {}
        }
        stack.extend(dc_core::term::children(&node));
    }
    (prelude_ref, foldr)
}

/// `id` expanded as the bridge expands it: the prelude, then `libs`.
fn full(prelude: &Prelude, arena: &mut Arena, libs: &[LibDef], id: TermId) -> TermId {
    let libs: Vec<LibDef> = libs
        .iter()
        .map(|d| LibDef {
            arity: d.arity,
            term: prelude.expand(arena, d.term),
        })
        .collect();
    let bare = prelude.expand(arena, id);
    expand(arena, &libs, bare).expect("expands")
}

/// Checks that `c` keeps prelude references, holds no prelude body, and that
/// each rewritten program expands to its original.
fn keeps_references(
    prelude: &Prelude,
    a: &mut Arena,
    programs: &[FrontierProgram],
    c: &Compression,
) {
    assert!(!c.new_libs.is_empty(), "Stitch found no abstraction");
    let terms: Vec<TermId> = c
        .new_libs
        .iter()
        .map(|d| d.term)
        .chain(c.rewritten.iter().copied())
        .collect();
    let scans: Vec<(bool, bool)> = terms.iter().map(|&t| scan(a, t)).collect();
    assert!(scans.iter().any(|s| s.0), "no output uses the prelude");
    assert!(
        scans.iter().all(|s| !s.1),
        "a prelude body reached the output"
    );
    for (p, &r) in programs.iter().zip(&c.rewritten) {
        let want = full(prelude, a, &[], p.program);
        let got = full(prelude, a, &c.new_libs, r);
        assert_eq!(got, want, "a rewriting changed a program");
    }
}

#[test]
fn compress_with_a_prelude_keeps_its_references() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let programs = shared(&prelude, &mut a);
    for p in &programs {
        assert!(!scan(&a, p.program).1, "the input already holds foldr");
    }
    let cfg = CompressConfig::default();
    let c = compress_with(&mut a, Some(&prelude), &[], &programs, &cfg).expect("compresses");
    keeps_references(&prelude, &mut a, &programs, &c);
    // The names table holds the prelude's names, so Stitch saw `map` as a token.
    let map = prelude.get("map").unwrap().lib();
    assert_eq!(c.names.name(map), Some("map"));
}

#[test]
fn without_the_prelude_a_prelude_reference_is_refused() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let programs = shared(&prelude, &mut a);
    let cfg = CompressConfig::default();
    let e = compress_with(&mut a, None, &[], &programs, &cfg).unwrap_err();
    assert!(
        matches!(
            e,
            CompressError::Encode {
                error: EncodeError::UnnamedLib(LibId(l)),
                ..
            } if l >= PRELUDE_LIB_BASE
        ),
        "{e:?}"
    );
}

#[test]
fn the_worker_gives_the_same_result_with_a_prelude() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let programs = shared(&prelude, &mut a);
    let cfg = CompressConfig::default();
    let in_process =
        compress_with(&mut a, Some(&prelude), &[], &programs, &cfg).expect("in process");
    let worker = Worker::new(WORKER);
    let run = IsolatedRun {
        worker: &worker,
        cancel: &NEVER,
    };
    let isolated =
        compress_isolated_with(&mut a, Some(&prelude), &[], &programs, &cfg, run).expect("worker");
    assert_eq!(isolated, in_process);
    keeps_references(&prelude, &mut a, &programs, &isolated);
}

/// `id` as Stitch text, with the prelude's names.
fn stitch_text(prelude: &Prelude, a: &Arena, id: TermId) -> String {
    let names = library_names(0, Some(prelude)).unwrap();
    to_expr(a, id, &names).unwrap().to_string()
}

/// A step that abstracts the prelude's `map` applied to one function, written
/// by hand: `fn_0 = (map #0)`, and each program rewritten to use it.
fn map_step(prelude: &Prelude, a: &mut Arena) -> (Vec<TermId>, StitchStep) {
    let srcs = [
        "fun xs -> map (fun x -> x + 1) xs",
        "fun xs -> map (fun x -> x * 2) xs",
    ];
    let programs: Vec<TermId> = srcs.iter().map(|s| compile(prelude, a, s)).collect();
    let rewritten: Vec<String> = programs
        .iter()
        .map(|&p| stitch_text(prelude, a, p))
        .map(|t| t.replacen("(map ", "(fn_0 ", 1))
        .collect();
    for r in &rewritten {
        assert!(
            r.contains("(fn_0 "),
            "the step does not use its abstraction: {r}"
        );
    }
    let step = StitchStep {
        name: "fn_0".to_owned(),
        arity: 1,
        body: "(map #0)".to_owned(),
        rewritten,
    };
    (programs, step)
}

#[test]
fn accept_steps_with_a_prelude_keeps_a_step_that_uses_a_prelude_token() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let (programs, step) = map_step(&prelude, &mut a);
    let c = accept_steps_with(&mut a, Some(&prelude), &[], &programs, &[step], 2).expect("accepts");
    assert_eq!(c.rejected, None);
    assert_eq!(c.new_libs.len(), 1);
    let map = prelude.get("map").unwrap().lib();
    let body = sexpr::print(&a, c.new_libs[0].term).unwrap();
    assert!(body.contains(&format!("lib_{}", map.0)), "{body}");
    for (&p, &r) in programs.iter().zip(&c.rewritten) {
        let want = full(&prelude, &mut a, &[], p);
        let got = full(&prelude, &mut a, &c.new_libs, r);
        assert_eq!(got, want);
    }
}

#[test]
fn accept_steps_without_the_prelude_refuses_its_references() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let (programs, step) = map_step(&prelude, &mut a);
    let e = accept_steps_with(&mut a, None, &[], &programs, &[step], 2).unwrap_err();
    assert!(matches!(e, CompressError::Expand { .. }), "{e:?}");
}

#[test]
fn expand_prelude_with_none_returns_the_term() {
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let t = compile(&prelude, &mut a, "map (fun x -> x) [1]");
    assert_eq!(expand_prelude(None, &mut a, t), t);
    let e = expand_prelude(Some(&prelude), &mut a, t);
    assert_ne!(e, t);
    assert!(scan(&a, e).1, "map was not expanded to its foldr body");
}

#[test]
fn a_prelude_name_that_looks_like_an_abstraction_is_refused() {
    // Stitch names new abstractions fn_0, fn_1, ...: a prelude function under
    // such a name would clash with one of them later in the run.
    for src in ["let fn_0 x = x + 1", "let fn_7 x = x", "let fn_x x = x"] {
        let prelude = Prelude::from_source("p", src).unwrap();
        let e = library_names(0, Some(&prelude)).unwrap_err();
        assert!(
            matches!(e, CompressError::Names(NameError::Illegal { .. })),
            "{src}: {e:?}"
        );
        let mut a = Arena::new();
        let t = compile(&prelude, &mut a, "fun x -> x");
        let step = StitchStep {
            name: "fn_0".to_owned(),
            arity: 0,
            body: "1".to_owned(),
            rewritten: vec!["(lam $0)".to_owned()],
        };
        let e = accept_steps_with(&mut a, Some(&prelude), &[], &[t], &[step], 2).unwrap_err();
        assert!(matches!(e, CompressError::Names(_)), "{src}: {e:?}");
    }
}

#[test]
fn a_rewritten_program_keeps_a_prelude_reference_outside_the_abstraction() {
    // The abstraction is the list [1; 2]; `map` and `filter` stay in the
    // rewritten programs, so the check must expand them there too.
    let prelude = Prelude::load().unwrap();
    let mut a = Arena::new();
    let srcs = ["map (fun x -> x) [1; 2]", "filter (fun x -> true) [1; 2]"];
    let programs: Vec<TermId> = srcs.iter().map(|s| compile(&prelude, &mut a, s)).collect();
    let step = StitchStep {
        name: "fn_0".to_owned(),
        arity: 0,
        body: "(cons int_1 (cons int_2 nil))".to_owned(),
        rewritten: vec![
            "(map (lam $0) fn_0)".to_owned(),
            "(filter (lam true) fn_0)".to_owned(),
        ],
    };
    let c = accept_steps_with(&mut a, Some(&prelude), &[], &programs, &[step], 2).expect("accepts");
    assert_eq!(c.rejected, None);
    assert_eq!(c.new_libs.len(), 1);
    for (&p, &r) in programs.iter().zip(&c.rewritten) {
        assert!(scan(&a, r).0, "the rewriting lost its prelude reference");
        let want = full(&prelude, &mut a, &[], p);
        let got = full(&prelude, &mut a, &c.new_libs, r);
        assert_eq!(got, want);
    }
}
