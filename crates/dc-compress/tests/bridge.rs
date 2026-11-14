//! The bridge against real Stitch runs: the abstractions it keeps decode, and the
//! rewritten programs expand back to the programs Stitch was given.

use dc_compress::{
    compress, stitch_config, CompressConfig, CompressError, Compression, FrontierProgram,
    MAX_STITCH_DEPTH,
};
use dc_core::{expand, sexpr, Arena, LibDef, LibId, TermId};
use dc_stitch::{decode, decode_abstraction, encode, DecodeError, LibNames, AND, IF, LET};
use stitch_core::{multistep_compression, MultistepCompressionConfig};

/// A program that is the same apart from a let whose bound value and body vary.
/// Without `--fused-lambda-tags 0`, Stitch puts a parameter on the let's binder.
const LETS: [&str; 5] = [
    "(mul 3 (add 1 (let 5 (mul $0 $0))))",
    "(mul 3 (add 1 (let 6 (sub $0 2))))",
    "(mul 3 (add 1 (let 7 (add $0 $0))))",
    "(mul 3 (add 1 (let 8 (div $0 4))))",
    "(mul 3 (add 1 (let 9 (mod $0 3))))",
];

/// Programs that differ in the function applied to the same two operands.
/// Without `--no-curried-metavars`, Stitch puts a parameter in head position and
/// passes a bare reserved token such as `and_` as its argument.
const HEADS: [&str; 5] = [
    "(pair (and (lt 1 2) (lt 3 4)) (cons 5 nil))",
    "(pair (or (lt 1 2) (lt 3 4)) (cons 5 nil))",
    "(pair (eq (lt 1 2) (lt 3 4)) (cons 5 nil))",
    "(pair (pair (lt 1 2) (lt 3 4)) (cons 5 nil))",
    "(pair (and (lt 1 2) (lt 3 4)) (cons 6 nil))",
];

/// Pairs of ifs with the same condition. Without `--no-curried-bodies`, Stitch
/// learns the partly applied `(if_ (lt int_1 int_2))`.
const IFS: [&str; 3] = [
    "(pair (if (lt 1 2) 3 4) (if (lt 1 2) 5 6))",
    "(pair (if (lt 1 2) 7 8) (if (lt 1 2) 9 10))",
    "(pair (if (lt 1 2) 11 12) (if (lt 1 2) 13 14))",
];

fn parse(a: &mut Arena, src: &str) -> TermId {
    sexpr::parse(a, src).unwrap_or_else(|e| panic!("{src}: {e}"))
}

fn frontier(a: &mut Arena, srcs: &[&str]) -> Vec<FrontierProgram> {
    srcs.iter()
        .enumerate()
        .map(|(i, s)| FrontierProgram {
            task: format!("task_{i}"),
            program: parse(a, s),
        })
        .collect()
}

/// One step of a raw Stitch run: the body, the arity and the rewritten programs.
type RawStep = (String, usize, Vec<String>);

/// Runs Stitch with its default flags (apart from the thread and silence
/// settings) and returns its steps.
fn raw_stitch(a: &mut Arena, srcs: &[&str]) -> Vec<RawStep> {
    let names = LibNames::new();
    let text: Vec<String> = srcs
        .iter()
        .map(|s| {
            let id = parse(a, s);
            encode(a, id, &names).expect("encode")
        })
        .collect();
    let tasks = (0..text.len()).map(|i| format!("task_{i}")).collect();
    let mut cfg = MultistepCompressionConfig::new();
    cfg.silent = true;
    cfg.step.threads = 1;
    let (steps, _) = multistep_compression(&text, Some(tasks), None, None, None, &cfg);
    steps
        .iter()
        .map(|s| {
            (
                s.inv.body.to_string(),
                s.inv.arity,
                s.rewritten.iter().map(|p| p.to_string()).collect(),
            )
        })
        .collect()
}

/// The reserved token of the first special form that Stitch, run without the
/// bridge's flags on `srcs`, cuts through: in an abstraction's body, or else in a
/// rewritten program. Checks that the corpus needs the flags.
fn raw_stitch_cuts_through(srcs: &[&str]) -> &'static str {
    let mut a = Arena::new();
    let steps = raw_stitch(&mut a, srcs);
    let mut names = LibNames::new();
    for (i, (body, arity, rewritten)) in steps.iter().enumerate() {
        let decoded = decode_abstraction(&mut a, body, *arity as u32, &names);
        if let Err(DecodeError::MalformedSpecialForm { token, .. }) = decoded {
            return token;
        }
        names
            .insert(LibId(i as u32), format!("fn_{i}"))
            .expect("fresh name");
        for p in rewritten {
            if let Err(DecodeError::MalformedSpecialForm { token, .. }) = decode(&mut a, p, &names)
            {
                return token;
            }
        }
    }
    panic!("no step cut through a special form: {steps:?}");
}

/// Checks that `c` kept every Stitch step, and that each rewritten program
/// expands under `libs` and the new abstractions to what `programs` expands to
/// under `libs`.
#[track_caller]
fn expands_back(a: &mut Arena, libs: &[LibDef], programs: &[FrontierProgram], c: &Compression) {
    assert_eq!(c.rejected, None);
    assert!(!c.new_libs.is_empty(), "Stitch found no abstraction");
    assert_eq!(c.new_libs.len(), c.stitch_steps);
    assert_eq!(c.rewritten.len(), programs.len());
    let mut all = libs.to_vec();
    all.extend_from_slice(&c.new_libs);
    for (p, r) in programs.iter().zip(&c.rewritten) {
        let want = expand(a, libs, p.program).expect("expand the original");
        let got = expand(a, &all, *r).expect("expand the rewritten program");
        assert_eq!(
            sexpr::print(a, got).expect("print"),
            sexpr::print(a, want).expect("print")
        );
        assert_eq!(got, want);
    }
}

#[test]
fn without_the_flags_stitch_puts_a_parameter_on_a_let_binder() {
    assert_eq!(raw_stitch_cuts_through(&LETS), LET);
}

#[test]
fn without_the_flags_stitch_abstracts_a_partly_applied_if() {
    assert_eq!(raw_stitch_cuts_through(&IFS), IF);
}

#[test]
fn without_the_flags_stitch_passes_a_bare_and_as_an_argument() {
    assert_eq!(raw_stitch_cuts_through(&HEADS), AND);
}

#[test]
fn the_bridge_keeps_every_step_on_a_corpus_of_lets() {
    let mut a = Arena::new();
    let programs = frontier(&mut a, &LETS);
    let c = compress(&mut a, &[], &programs, &CompressConfig::default()).expect("compress");
    expands_back(&mut a, &[], &programs, &c);
}

#[test]
fn the_bridge_keeps_every_step_on_a_corpus_of_heads() {
    let mut a = Arena::new();
    let programs = frontier(&mut a, &HEADS);
    let c = compress(&mut a, &[], &programs, &CompressConfig::default()).expect("compress");
    expands_back(&mut a, &[], &programs, &c);
}

#[test]
fn the_bridge_keeps_every_step_on_a_corpus_of_ifs() {
    let mut a = Arena::new();
    let programs = frontier(&mut a, &IFS);
    let c = compress(&mut a, &[], &programs, &CompressConfig::default()).expect("compress");
    expands_back(&mut a, &[], &programs, &c);
}

#[test]
fn new_abstractions_take_the_ids_after_the_library() {
    let mut a = Arena::new();
    let libs = vec![
        LibDef {
            arity: 1,
            term: parse(&mut a, "(lam (add $0 1))"),
        },
        LibDef {
            arity: 2,
            term: parse(&mut a, "(lam (lam (mul $1 $0)))"),
        },
    ];
    let programs = frontier(
        &mut a,
        &[
            "(lib_1 (lib_0 (sub 7 3)) (lib_0 (sub 9 3)))",
            "(lib_1 (lib_0 (sub 2 3)) (lib_0 (sub 4 3)))",
            "(lib_1 (lib_0 (sub 5 3)) (lib_0 (sub 6 3)))",
        ],
    );
    let c = compress(&mut a, &libs, &programs, &CompressConfig::default()).expect("compress");
    expands_back(&mut a, &libs, &programs, &c);
    assert_eq!(c.names.name(LibId(2)), Some("fn_2"));
    assert_eq!(c.names.len(), libs.len() + c.new_libs.len());
    // The rewritten programs use the first new abstraction under its new id.
    let printed: Vec<String> = c
        .rewritten
        .iter()
        .map(|r| sexpr::print(&a, *r).expect("print"))
        .collect();
    assert!(printed.iter().all(|p| p.contains("lib_2")), "{printed:?}");
}

#[test]
fn a_second_round_continues_the_numbering() {
    // Compress once, then compress the rewritten programs again with the grown
    // library: ids and names carry on from where the first round stopped.
    let mut a = Arena::new();
    // An inner shape (add (mul 3 4) _) in many contexts, and an outer shape
    // (pair _ (cons _ nil)) around different operands.
    let srcs = [
        "(pair (add (mul 3 4) 5) (cons (add (mul 3 4) 6) nil))",
        "(pair (sub 1 2) (cons (sub 7 8) nil))",
        "(pair (lt 1 9) (cons (lt 2 9) nil))",
        "(pair (add (mul 3 4) 1) (cons (fst (add (mul 3 4) 2)) nil))",
        "(snd (add (mul 3 4) 7))",
        "(fst (add (mul 3 4) 8))",
    ];
    let programs = frontier(&mut a, &srcs);
    let one = CompressConfig {
        iterations: 1,
        max_arity: 2,
    };
    let first = compress(&mut a, &[], &programs, &one).expect("first round");
    expands_back(&mut a, &[], &programs, &first);
    let library = first.new_libs.clone();
    let again: Vec<FrontierProgram> = programs
        .iter()
        .zip(&first.rewritten)
        .map(|(p, r)| FrontierProgram {
            task: p.task.clone(),
            program: *r,
        })
        .collect();
    let second = compress(&mut a, &library, &again, &one).expect("second round");
    expands_back(&mut a, &library, &again, &second);
    assert_eq!(second.names.name(LibId(1)), Some("fn_1"));
    let first_program = sexpr::print(&a, second.rewritten[0]).expect("print");
    assert!(
        first_program.contains("lib_0") && first_program.contains("lib_1"),
        "{first_program}"
    );
}

/// Checks that Stitch gives the same result for the same programs within one
/// process. `the_config_sets_the_bridge_flags` checks the one-thread and
/// no-shuffle settings, which are also Stitch's defaults.
#[test]
fn the_same_programs_give_the_same_result() {
    let run = || {
        let mut a = Arena::new();
        let programs = frontier(&mut a, &LETS);
        let c = compress(&mut a, &[], &programs, &CompressConfig::default()).expect("compress");
        let libs: Vec<String> = c
            .new_libs
            .iter()
            .map(|l| format!("{} {}", l.arity, sexpr::print(&a, l.term).expect("print")))
            .collect();
        let rewritten: Vec<String> = c
            .rewritten
            .iter()
            .map(|r| sexpr::print(&a, *r).expect("print"))
            .collect();
        (libs, rewritten)
    };
    let first = run();
    assert!(!first.0.is_empty());
    for _ in 0..3 {
        assert_eq!(run(), first);
    }
}

#[test]
fn no_programs_give_no_abstractions() {
    let mut a = Arena::new();
    let c = compress(&mut a, &[], &[], &CompressConfig::default()).expect("compress");
    assert!(c.new_libs.is_empty());
    assert!(c.rewritten.is_empty());
    assert_eq!(c.stitch_steps, 0);
    assert_eq!(c.rejected, None);
}

#[test]
fn the_config_sets_the_bridge_flags() {
    let s = stitch_config(
        &CompressConfig {
            iterations: 7,
            max_arity: 3,
        },
        4,
    );
    assert_eq!(s.iterations, 7);
    assert_eq!(s.step.max_arity, 3);
    assert_eq!(s.previous_abstractions, 4);
    assert_eq!(s.abstraction_prefix, "fn_");
    assert_eq!(s.step.threads, 1);
    assert!(!s.shuffle);
    assert!(s.step.no_curried_bodies);
    assert!(s.step.no_curried_metavars);
    assert!(!s.step.eta_long);
    // The tag set is private; its Debug form shows it.
    assert_eq!(
        format!("{:?}", s.step.fused_lambda_tags),
        "FusedLambdaTags { tags: Some({0}) }"
    );
}

#[test]
fn an_open_program_is_refused_before_stitch_runs() {
    let mut a = Arena::new();
    let programs = frontier(
        &mut a,
        &[
            "(add 1 (mul 2 (sub 3 4)))",
            "(add $0 (mul 2 (sub 3 5)))",
            "(add 1 (mul 2 (sub 3 6)))",
        ],
    );
    assert_eq!(
        compress(&mut a, &[], &programs, &CompressConfig::default()),
        Err(CompressError::OpenProgram { index: 1 })
    );
}

/// `(fst (fst ... (fst k)))`, `n` applications deep.
fn chain(a: &mut Arena, n: usize, k: i64) -> TermId {
    let f = parse(a, "fst");
    let mut t = a.int(k);
    for _ in 0..n {
        t = a.app(f, t);
    }
    t
}

#[test]
fn a_program_too_deep_for_stitch_is_refused() {
    // Without the limit, Stitch overflows the stack on this program and the
    // process aborts.
    let mut a = Arena::new();
    let programs = vec![
        FrontierProgram {
            task: "task_0".to_string(),
            program: chain(&mut a, 3, 0),
        },
        FrontierProgram {
            task: "task_1".to_string(),
            program: chain(&mut a, 20_000, 1),
        },
    ];
    assert_eq!(
        compress(&mut a, &[], &programs, &CompressConfig::default()),
        Err(CompressError::TooDeep { index: 1 })
    );
    // The limit counts the levels of the Stitch expression: an integer under
    // `MAX_STITCH_DEPTH - 1` applications is the deepest program accepted.
    let one_over = FrontierProgram {
        task: "task_0".to_string(),
        program: chain(&mut a, MAX_STITCH_DEPTH, 0),
    };
    assert_eq!(
        compress(&mut a, &[], &[one_over], &CompressConfig::default()),
        Err(CompressError::TooDeep { index: 0 })
    );
}

#[test]
fn stitch_runs_off_the_callers_stack() {
    // A caller with a small stack still gets a result: Stitch runs on its own
    // thread. Building Stitch's configuration alone overflows a 64 KiB stack.
    let run = std::thread::Builder::new()
        .stack_size(64 * 1024)
        .spawn(|| {
            let mut a = Arena::new();
            let programs: Vec<FrontierProgram> = (0..3)
                .map(|i| FrontierProgram {
                    task: format!("task_{i}"),
                    program: chain(&mut a, 300, i),
                })
                .collect();
            let c = compress(&mut a, &[], &programs, &CompressConfig::default()).expect("compress");
            expands_back(&mut a, &[], &programs, &c);
        })
        .expect("spawn");
    run.join().expect("compress on a small stack");
}
