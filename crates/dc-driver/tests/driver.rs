//! The iteration step against real Stitch runs and against stand-in Stitch
//! runners that return chosen results.

use std::collections::BTreeMap;
use std::time::Duration;

use dc_compress::worker::WorkerError;
use dc_compress::{CompressConfig, CompressError, Compression, FrontierProgram};
use dc_core::types::check;
use dc_core::{expand, sexpr, Arena, LibDef, Scheme, TermId};
use dc_driver::{
    run_iteration, stitch_cost, Candidate, CompressionReport, DriverConfig, DriverError, Fate,
    InProcess, IterationInput, IterationOutcome, LibraryEntry, Source, Stitch, TaskSpec,
};
use dc_stitch::LibNames;

fn scheme(text: &str) -> Scheme {
    text.parse().unwrap_or_else(|e| panic!("{text}: {e:?}"))
}

fn task(name: &str, signature: &str) -> TaskSpec {
    TaskSpec {
        name: name.to_owned(),
        signature: scheme(signature),
    }
}

fn candidate(task: &str, program: &str) -> Candidate {
    Candidate {
        task: task.to_owned(),
        program: program.to_owned(),
    }
}

fn config(k: usize) -> DriverConfig {
    DriverConfig {
        frontier_size: k,
        compress: CompressConfig::default(),
    }
}

fn frontiers(entries: &[(&str, &[&str])]) -> BTreeMap<String, Vec<String>> {
    entries
        .iter()
        .map(|(t, ps)| (t.to_string(), ps.iter().map(|p| p.to_string()).collect()))
        .collect()
}

/// The cost Stitch gives `program`, under a library of `library_len` entries.
fn cost(program: &str, library_len: u32) -> u64 {
    let mut a = Arena::new();
    let id = sexpr::parse(&mut a, program).unwrap();
    let names = LibNames::sequential(dc_compress::ABSTRACTION_PREFIX, library_len).unwrap();
    stitch_cost(&a, id, &names).unwrap()
}

/// A Stitch runner that finds nothing: no abstractions, each program as given.
#[derive(Default)]
struct Nothing {
    /// The task labels of each call's programs.
    calls: Vec<Vec<String>>,
}

impl Stitch for Nothing {
    fn run(
        &mut self,
        _arena: &mut Arena,
        _libs: &[LibDef],
        programs: &[FrontierProgram],
        _cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        self.calls
            .push(programs.iter().map(|p| p.task.clone()).collect());
        Ok(Compression {
            new_libs: Vec::new(),
            rewritten: programs.iter().map(|p| p.program).collect(),
            names: LibNames::new(),
            stitch_steps: 0,
            rejected: None,
        })
    }
}

/// A Stitch runner that returns the abstractions and rewritten programs it was
/// given, as text, whatever the input. Only for inputs where the rewriting is
/// right.
struct Fixed {
    libs: Vec<(u32, &'static str)>,
    rewritten: Vec<&'static str>,
}

impl Stitch for Fixed {
    fn run(
        &mut self,
        arena: &mut Arena,
        _libs: &[LibDef],
        programs: &[FrontierProgram],
        _cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        assert_eq!(programs.len(), self.rewritten.len(), "the test's input");
        Ok(Compression {
            new_libs: self
                .libs
                .iter()
                .map(|&(arity, body)| LibDef {
                    arity,
                    term: sexpr::parse(arena, body).unwrap(),
                })
                .collect(),
            rewritten: self
                .rewritten
                .iter()
                .map(|p| sexpr::parse(arena, p).unwrap())
                .collect(),
            names: LibNames::new(),
            stitch_steps: self.libs.len(),
            rejected: None,
        })
    }
}

/// A Stitch runner that fails each call with the next error in its list, and
/// then finds nothing.
struct Failing {
    errors: Vec<CompressError>,
    inner: Nothing,
}

impl Stitch for Failing {
    fn run(
        &mut self,
        arena: &mut Arena,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        if self.errors.is_empty() {
            return self.inner.run(arena, libs, programs, cfg);
        }
        self.inner
            .calls
            .push(programs.iter().map(|p| p.task.clone()).collect());
        Err(self.errors.remove(0))
    }
}

fn run(
    tasks: &[TaskSpec],
    library: &[LibraryEntry],
    frontiers: &BTreeMap<String, Vec<String>>,
    candidates: &[Candidate],
    cfg: &DriverConfig,
    stitch: &mut dyn Stitch,
) -> Result<IterationOutcome, DriverError> {
    run_iteration(
        IterationInput {
            tasks,
            library,
            frontiers,
            candidates,
        },
        cfg,
        stitch,
    )
}

fn fate_of(outcome: &IterationOutcome, source: Source) -> &Fate {
    &outcome
        .report
        .programs
        .iter()
        .find(|p| p.source == source)
        .unwrap_or_else(|| panic!("no fate for {source:?}"))
        .fate
}

#[test]
fn stitch_cost_uses_stitchs_default_costs() {
    // lam 1, app 1, and 100 for a variable, a parameter or a symbol.
    assert_eq!(cost("1", 0), 100);
    assert_eq!(cost("(lam $0)", 0), 101);
    // (add 1 2) is two applications and three symbols.
    assert_eq!(cost("(add 1 2)", 0), 302);
    assert_eq!(cost("(lib_0 1)", 1), 201);
}

#[test]
fn each_task_keeps_its_k_shortest_programs_in_a_stable_order() {
    let tasks = [task("a", "Int"), task("b", "Int")];
    let short = "(add 1 2)";
    let shorter = "7";
    let long = "(add (add 1 2) (add 3 4))";
    let tie = "(add 2 1)";
    let fronts = frontiers(&[("a", &[long, short])]);
    let cands = [
        candidate("a", tie),
        candidate("b", short),
        candidate("a", shorter),
    ];
    let mut stitch = Nothing::default();
    let out = run(&tasks, &[], &fronts, &cands, &config(3), &mut stitch).unwrap();

    // `short` and `tie` cost the same: the member comes before the candidate.
    assert_eq!(out.frontiers["a"], vec![shorter, short, tie]);
    assert_eq!(out.frontiers["b"], vec![short]);
    assert_eq!(
        *fate_of(
            &out,
            Source::Member {
                task: "a".into(),
                index: 0
            }
        ),
        Fate::Trimmed
    );
    assert_eq!(
        *fate_of(
            &out,
            Source::Member {
                task: "a".into(),
                index: 1
            }
        ),
        Fate::Kept
    );
    for index in 0..3 {
        assert_eq!(*fate_of(&out, Source::Candidate { index }), Fate::Kept);
    }
    assert!(out.new_library.is_empty());
    // Stitch got every kept program, in task order, with its task label.
    assert_eq!(stitch.calls, vec![vec!["a", "a", "a", "b"]]);
}

#[test]
fn a_program_that_expands_to_a_kept_one_is_a_duplicate() {
    let tasks = [task("a", "Int")];
    let library = [LibraryEntry {
        arity: 1,
        term: "(lam (add $0 $0))".into(),
    }];
    let fronts = frontiers(&[("a", &["(lib_0 3)"])]);
    let cands = [
        candidate("a", "(add 3 3)"),
        candidate("a", "(lib_0 3)"),
        candidate("a", "(add 4 4)"),
    ];
    let out = run(
        &tasks,
        &library,
        &fronts,
        &cands,
        &config(5),
        &mut Nothing::default(),
    )
    .unwrap();
    let member = Source::Member {
        task: "a".into(),
        index: 0,
    };
    assert_eq!(
        *fate_of(&out, Source::Candidate { index: 0 }),
        Fate::Duplicate { of: member.clone() }
    );
    assert_eq!(
        *fate_of(&out, Source::Candidate { index: 1 }),
        Fate::Duplicate { of: member }
    );
    assert_eq!(*fate_of(&out, Source::Candidate { index: 2 }), Fate::Kept);
    assert_eq!(out.frontiers["a"], vec!["(lib_0 3)", "(add 4 4)"]);
}

#[test]
fn invalid_candidates_are_reported_and_left_out() {
    let tasks = [task("a", "Int -> Int")];
    let cands = [
        candidate("nope", "(lam $0)"),
        candidate("a", "(lam"),
        candidate("a", "(lam $1)"),
        candidate("a", "(lib_0 1)"),
        candidate("a", "(lam true)"),
        candidate("a", "(lam (add $0 1))"),
    ];
    let mut stitch = Nothing::default();
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    for index in 0..5 {
        match fate_of(&out, Source::Candidate { index }) {
            Fate::Invalid { reason } => assert!(!reason.is_empty()),
            other => panic!("candidate {index}: {other:?}"),
        }
    }
    assert_eq!(*fate_of(&out, Source::Candidate { index: 5 }), Fate::Kept);
    assert_eq!(out.frontiers["a"], vec!["(lam (add $0 1))"]);
}

#[test]
fn a_type_variable_in_the_signature_is_rigid() {
    // `lam (add $0 1)` has type Int -> Int, not a -> a for every a.
    let tasks = [task("id", "a -> a")];
    let cands = [
        candidate("id", "(lam (add $0 1))"),
        candidate("id", "(lam $0)"),
    ];
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut Nothing::default(),
    )
    .unwrap();
    assert!(matches!(
        fate_of(&out, Source::Candidate { index: 0 }),
        Fate::Invalid { .. }
    ));
    assert_eq!(out.frontiers["id"], vec!["(lam $0)"]);
}

/// Programs over several tasks that share a large common part, so that Stitch
/// finds an abstraction.
const SHARED: [(&str, &str); 4] = [
    ("t0", "(lam (add (mul (add $0 1) (add $0 2)) 10))"),
    ("t1", "(lam (add (mul (add $0 1) (add $0 2)) 20))"),
    ("t2", "(lam (add (mul (add $0 1) (add $0 2)) 30))"),
    ("t3", "(lam (add (mul (add $0 1) (add $0 2)) 40))"),
];

fn shared_tasks() -> Vec<TaskSpec> {
    SHARED.iter().map(|(t, _)| task(t, "Int -> Int")).collect()
}

/// The programs of [`SHARED`] as frontiers.
fn shared_input() -> BTreeMap<String, Vec<String>> {
    SHARED
        .iter()
        .map(|(t, p)| (t.to_string(), vec![p.to_string()]))
        .collect()
}

fn shared_candidates() -> Vec<Candidate> {
    SHARED.iter().map(|(t, p)| candidate(t, p)).collect()
}

/// Checks that `out` is a sound update of `library` and `input`: the new
/// library expands and types, and each task's frontier holds programs that
/// type against its signature and expand to the input programs.
fn check_sound(
    tasks: &[TaskSpec],
    library: &[LibraryEntry],
    input: &BTreeMap<String, Vec<String>>,
    out: &IterationOutcome,
) {
    let mut a = Arena::new();
    let libs: Vec<LibDef> = library
        .iter()
        .chain(&out.new_library)
        .map(|e| LibDef {
            arity: e.arity,
            term: sexpr::parse(&mut a, &e.term).unwrap(),
        })
        .collect();
    let mut schemes: Vec<Scheme> = Vec::new();
    for def in &libs {
        let s = dc_core::types::infer(&a, def.term, &schemes).expect("the library types");
        schemes.push(s);
    }
    for t in tasks {
        let expanded = |a: &mut Arena, p: &str, libs: &[LibDef]| -> TermId {
            let id = sexpr::parse(a, p).unwrap();
            expand(a, libs, id).unwrap()
        };
        let mut want: Vec<TermId> = input
            .get(&t.name)
            .map(|ps| ps.iter().map(|p| expanded(&mut a, p, &libs)).collect())
            .unwrap_or_default();
        let mut got: Vec<TermId> = Vec::new();
        for p in &out.frontiers[&t.name] {
            let id = sexpr::parse(&mut a, p).unwrap();
            check(&a, id, &schemes, &t.signature)
                .unwrap_or_else(|e| panic!("{}: {p}: {e}", t.name));
            got.push(expand(&mut a, &libs, id).unwrap());
        }
        want.sort();
        got.sort();
        assert_eq!(got, want, "task {}", t.name);
    }
}

#[test]
fn a_real_stitch_run_extends_the_library_and_rewrites_the_frontiers() {
    let tasks = shared_tasks();
    let cands = shared_candidates();
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut InProcess,
    )
    .unwrap();

    assert!(!out.new_library.is_empty(), "{:?}", out.report);
    let CompressionReport::Ran {
        kept, fallbacks, ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!(*kept, out.new_library.len());
    assert!(fallbacks.is_empty(), "{fallbacks:?}");
    // Every program now uses the new abstraction and is shorter.
    for (t, p) in SHARED {
        let rewritten = &out.frontiers[t];
        assert_eq!(rewritten.len(), 1);
        assert!(rewritten[0].contains("lib_0"), "{t}: {}", rewritten[0]);
        assert!(cost(&rewritten[0], out.new_library.len() as u32) < cost(p, 0));
    }
    check_sound(&tasks, &[], &shared_input(), &out);
}

#[test]
fn the_same_input_gives_the_same_outcome() {
    let tasks = shared_tasks();
    let cands = shared_candidates();
    let first = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut InProcess,
    )
    .unwrap();
    let second = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut InProcess,
    )
    .unwrap();
    // Not two empty results: Stitch found something to repeat.
    assert!(!first.new_library.is_empty(), "{:?}", first.report);
    assert_eq!(first, second);
    // The report reads back from its JSON.
    let json = serde_json::to_string(&first.report).unwrap();
    assert_eq!(
        serde_json::from_str::<dc_driver::IterationReport>(&json).unwrap(),
        first.report
    );
}

#[test]
fn a_second_iteration_uses_the_library_it_is_given() {
    let tasks = shared_tasks();
    let cands = shared_candidates();
    let first = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut InProcess,
    )
    .unwrap();
    // A new candidate in the old form for t0: it duplicates the kept one.
    let again = [candidate("t0", SHARED[0].1)];
    let second = run(
        &tasks,
        &first.new_library,
        &first.frontiers,
        &again,
        &config(5),
        &mut InProcess,
    )
    .unwrap();
    assert!(matches!(
        fate_of(&second, Source::Candidate { index: 0 }),
        Fate::Duplicate { .. }
    ));
    check_sound(&tasks, &first.new_library, &first.frontiers, &second);
    for t in &tasks {
        assert_eq!(second.frontiers[&t.name].len(), 1);
    }
}

/// The program from P2.12 whose refactoring with `λ#0. (#0, #0)` does not type,
/// though its expansion does.
const TWO_TYPES: &str = "(let (lam $0) ((lam (pair (fst $0 1) (snd $0 true))) (pair $0 $0)))";
const TWO_TYPES_REWRITTEN: &str =
    "(let (lam $0) ((lam (pair (fst $0 1) (snd $0 true))) (lib_0 $0)))";

#[test]
fn a_rewritten_program_that_does_not_type_falls_back_to_its_original() {
    let tasks = [task("p", "(Int, Bool)"), task("q", "(Int, Int)")];
    let cands = [candidate("p", TWO_TYPES), candidate("q", "(pair 5 5)")];
    let mut stitch = Fixed {
        libs: vec![(1, "(lam (pair $0 $0))")],
        rewritten: vec![TWO_TYPES_REWRITTEN, "(lib_0 5)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(out.frontiers["p"], vec![TWO_TYPES]);
    assert_eq!(out.frontiers["q"], vec!["(lib_0 5)"]);
    assert_eq!(out.new_library.len(), 1);
    let CompressionReport::Ran {
        fallbacks, kept, ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!(*kept, 1);
    assert_eq!(fallbacks.len(), 1);
    assert_eq!(fallbacks[0].task, "p");
    assert_eq!(fallbacks[0].program, TWO_TYPES);
}

#[test]
fn a_rewriting_that_changes_the_program_falls_back() {
    // The runner claims (lib_0 3) for (add 3 3) with lib_0 = λ#0. (mul #0 #0):
    // the right type, the wrong program. And (lib_0 5) for (mul 5 5), which is
    // right.
    let tasks = [task("a", "Int"), task("b", "Int")];
    let cands = [candidate("a", "(add 3 3)"), candidate("b", "(mul 5 5)")];
    let mut stitch = Fixed {
        libs: vec![(1, "(lam (mul $0 $0))")],
        rewritten: vec!["(lib_0 3)", "(lib_0 5)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(out.frontiers["a"], vec!["(add 3 3)"]);
    assert_eq!(out.frontiers["b"], vec!["(lib_0 5)"]);
    let CompressionReport::Ran { fallbacks, .. } = &out.report.compression else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!(fallbacks.len(), 1);
    assert_eq!(fallbacks[0].task, "a");
    assert_eq!(fallbacks[0].rewritten, "(lib_0 3)");
}

#[test]
fn an_abstraction_no_kept_program_uses_is_cut() {
    // Only the program for p used the abstraction, and it falls back.
    let tasks = [task("p", "(Int, Bool)")];
    let cands = [candidate("p", TWO_TYPES)];
    let mut stitch = Fixed {
        libs: vec![(1, "(lam (pair $0 $0))")],
        rewritten: vec![TWO_TYPES_REWRITTEN],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert!(out.new_library.is_empty());
    assert_eq!(out.frontiers["p"], vec![TWO_TYPES]);
    let CompressionReport::Ran {
        found,
        kept,
        unused,
        ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!((*found, *kept, *unused), (1, 0, 1));
}

#[test]
fn only_trailing_unused_abstractions_are_cut() {
    // lib_0 is used only through lib_1, and lib_2 is not used at all.
    let tasks = [task("a", "Int"), task("b", "Int")];
    let cands = [
        candidate("a", "(add (add 1 1) 2)"),
        candidate("b", "(mul 3 3)"),
    ];
    let mut stitch = Fixed {
        libs: vec![
            (1, "(lam (add $0 $0))"),
            (1, "(lam (add (lib_0 1) $0))"),
            (1, "(lam (mul $0 $0))"),
        ],
        rewritten: vec!["(lib_1 2)", "(mul 3 3)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(
        out.new_library,
        vec![
            LibraryEntry {
                arity: 1,
                term: "(lam (add $0 $0))".into()
            },
            LibraryEntry {
                arity: 1,
                term: "(lam (add (lib_0 1) $0))".into()
            },
        ]
    );
    assert_eq!(out.frontiers["a"], vec!["(lib_1 2)"]);
    let CompressionReport::Ran {
        found,
        kept,
        unused,
        ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!((*found, *kept, *unused), (3, 2, 1));
}

#[test]
fn the_library_is_cut_at_the_first_abstraction_that_does_not_type() {
    // lib_1 applies its parameter to an Int and to a Bool: no monomorphic type.
    // lib_2 comes after it and is cut with it, though it types.
    let tasks = [task("a", "Int"), task("b", "(Int, Bool)"), task("c", "Int")];
    let cands = [
        candidate("a", "(add (add 1 1) 2)"),
        candidate("b", "(pair ((lam $0) 1) ((lam $0) true))"),
        candidate("c", "(mul 4 4)"),
    ];
    let mut stitch = Fixed {
        libs: vec![
            (1, "(lam (add $0 $0))"),
            (1, "(lam (pair ($0 1) ($0 true)))"),
            (1, "(lam (mul $0 $0))"),
        ],
        rewritten: vec!["(add (lib_0 1) 2)", "(lib_1 (lam $0))", "(lib_2 4)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(out.new_library.len(), 1);
    assert_eq!(out.frontiers["a"], vec!["(add (lib_0 1) 2)"]);
    assert_eq!(
        out.frontiers["b"],
        vec!["(pair ((lam $0) 1) ((lam $0) true))"]
    );
    assert_eq!(out.frontiers["c"], vec!["(mul 4 4)"]);
    let CompressionReport::Ran {
        found,
        kept,
        untyped,
        fallbacks,
        ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!((*found, *kept), (3, 1));
    assert_eq!(untyped.as_ref().map(|u| u.abstraction), Some(1));
    let fell_back: Vec<&str> = fallbacks.iter().map(|f| f.task.as_str()).collect();
    assert_eq!(fell_back, vec!["b", "c"]);
    // c's rewriting types on its own, but uses lib_2, which comes after the cut.
    assert!(
        fallbacks[1].reason.contains("one after it"),
        "{}",
        fallbacks[1].reason
    );
}

#[test]
fn a_failed_stitch_run_still_updates_the_frontiers() {
    let tasks = shared_tasks();
    let cands = shared_candidates();
    let mut stitch = Failing {
        errors: vec![CompressError::Worker(WorkerError::TimedOut {
            wall: Duration::from_secs(1),
        })],
        inner: Nothing::default(),
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert!(out.new_library.is_empty());
    for (t, p) in SHARED {
        assert_eq!(out.frontiers[t], vec![p]);
    }
    match &out.report.compression {
        CompressionReport::Failed { error, .. } => assert!(!error.is_empty()),
        other => panic!("{other:?}"),
    }
    // A failed run is not retried.
    assert_eq!(stitch.inner.calls.len(), 1);
}

#[test]
fn a_program_the_bridge_refuses_is_left_out_and_the_run_retried() {
    let tasks = shared_tasks();
    let cands = shared_candidates();
    let mut stitch = Failing {
        errors: vec![CompressError::TooDeep { index: 2 }],
        inner: Nothing::default(),
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(
        stitch.inner.calls,
        vec![vec!["t0", "t1", "t2", "t3"], vec!["t0", "t1", "t3"]]
    );
    let CompressionReport::Ran { left_out, .. } = &out.report.compression else {
        panic!("{:?}", out.report.compression);
    };
    assert_eq!(left_out.len(), 1);
    assert_eq!(left_out[0].task, "t2");
    // The program left out stays in its frontier.
    assert_eq!(out.frontiers["t2"], vec![SHARED[2].1]);
}

#[test]
fn with_no_programs_stitch_does_not_run() {
    let tasks = [task("a", "Int")];
    let mut stitch = Nothing::default();
    let out = run(&tasks, &[], &BTreeMap::new(), &[], &config(5), &mut stitch).unwrap();
    assert!(stitch.calls.is_empty());
    assert_eq!(
        out.report.compression,
        CompressionReport::NoPrograms { left_out: vec![] }
    );
    assert_eq!(out.frontiers["a"], Vec::<String>::new());
}

/// One case of bad input: the tasks, the library, the frontiers, the frontier
/// size, and a check on the error.
struct BadCase {
    tasks: Vec<TaskSpec>,
    library: Vec<LibraryEntry>,
    frontiers: BTreeMap<String, Vec<String>>,
    k: usize,
    expected: fn(&DriverError) -> bool,
}

fn lib(arity: u32, term: &str) -> LibraryEntry {
    LibraryEntry {
        arity,
        term: term.to_owned(),
    }
}

#[test]
fn bad_input_state_is_an_error() {
    let one = || vec![task("a", "Int")];
    let case = |library: Vec<LibraryEntry>, fronts: &[(&str, &[&str])], expected| BadCase {
        tasks: one(),
        library,
        frontiers: frontiers(fronts),
        k: 1,
        expected,
    };
    let member =
        |e: &DriverError| matches!(e, DriverError::Member { task, index: 0, .. } if task == "a");
    let cases = [
        BadCase {
            k: 0,
            ..case(vec![], &[], |e| *e == DriverError::NoFrontier)
        },
        BadCase {
            tasks: vec![task("a", "Int"), task("a", "Int")],
            ..case(vec![], &[], |e| {
                *e == DriverError::DuplicateTask("a".into())
            })
        },
        // A library entry that does not parse, has too few lambdas, or does
        // not type.
        case(vec![lib(0, "(")], &[], |e| {
            matches!(e, DriverError::Library { index: 0, .. })
        }),
        case(vec![lib(1, "(lam $0)"), lib(2, "(lam $0)")], &[], |e| {
            matches!(e, DriverError::Library { index: 1, .. })
        }),
        case(
            vec![lib(1, "(lam $0)"), lib(1, "(lam ($0 $0))")],
            &[],
            |e| matches!(e, DriverError::Library { index: 1, .. }),
        ),
        // A frontier for a task the run does not have.
        case(vec![], &[("b", &["1"])], |e| {
            *e == DriverError::UnknownTask("b".into())
        }),
        // A frontier member that does not parse, is open, uses an unknown
        // library, or has the wrong type.
        case(vec![], &[("a", &["("])], member),
        case(vec![], &[("a", &["$0"])], member),
        case(vec![lib(1, "(lam $0)")], &[("a", &["(lib_1 1)"])], member),
        case(vec![], &[("a", &["true"])], member),
    ];
    for c in cases {
        let mut stitch = Nothing::default();
        let result = run(
            &c.tasks,
            &c.library,
            &c.frontiers,
            &[],
            &config(c.k),
            &mut stitch,
        );
        match result {
            Err(e) => assert!((c.expected)(&e), "{:?} {:?}: {e:?}", c.library, c.frontiers),
            Ok(out) => panic!("{:?} {:?}: {out:?}", c.library, c.frontiers),
        }
        assert!(stitch.calls.is_empty());
    }
}

#[test]
fn new_abstractions_are_numbered_after_the_given_library() {
    // With one library entry already, the first new abstraction is lib_1.
    let tasks = shared_tasks();
    let library = [lib(1, "(lam (sub $0 1))")];
    let cands = shared_candidates();
    let out = run(
        &tasks,
        &library,
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut InProcess,
    )
    .unwrap();
    assert!(!out.new_library.is_empty(), "{:?}", out.report);
    for (t, _) in SHARED {
        let p = &out.frontiers[t][0];
        assert!(p.contains("lib_1") && !p.contains("lib_0"), "{p}");
    }
    check_sound(&tasks, &library, &shared_input(), &out);
}

#[test]
fn a_candidate_too_large_to_print_is_invalid_and_the_step_goes_on() {
    // Each level of (add ... 1) is four nodes written out, so this is over
    // dc_core::sexpr::MAX_PRINT_NODES.
    let levels = (dc_core::sexpr::MAX_PRINT_NODES / 4) + 1;
    let huge = format!("{}1{}", "(add ".repeat(levels), " 1)".repeat(levels));
    let tasks = [task("a", "Int"), task("b", "Int")];
    let cands = [candidate("a", &huge), candidate("b", "(add 1 2)")];
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut Nothing::default(),
    )
    .unwrap();
    match fate_of(&out, Source::Candidate { index: 0 }) {
        Fate::Invalid { reason } => assert!(reason.contains("nodes written out"), "{reason}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(out.frontiers["a"], Vec::<String>::new());
    assert_eq!(out.frontiers["b"], vec!["(add 1 2)"]);
}

#[test]
fn an_interrupted_stitch_run_is_an_error_not_a_finished_iteration() {
    for error in [
        WorkerError::Cancelled,
        WorkerError::Spawn(std::io::ErrorKind::NotFound),
        WorkerError::Io(std::io::ErrorKind::BrokenPipe),
    ] {
        let mut stitch = Failing {
            errors: vec![CompressError::Worker(error.clone())],
            inner: Nothing::default(),
        };
        let result = run(
            &shared_tasks(),
            &[],
            &BTreeMap::new(),
            &shared_candidates(),
            &config(5),
            &mut stitch,
        );
        assert_eq!(
            result,
            Err(DriverError::Interrupted(CompressError::Worker(error)))
        );
    }
}

#[test]
fn each_frontier_is_sorted_again_under_the_new_library() {
    // Before: (add 1 2) costs 302, the mul program far more. After the
    // rewriting, (lib_0 5) costs 201 and comes first.
    let tasks = [task("a", "Int")];
    let long = "(mul (add 5 6) (add 5 6))";
    let cands = [candidate("a", long), candidate("a", "(add 1 2)")];
    let mut stitch = Fixed {
        libs: vec![(1, "(lam (mul (add $0 6) (add $0 6)))")],
        rewritten: vec!["(add 1 2)", "(lib_0 5)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(out.frontiers["a"], vec!["(lib_0 5)", "(add 1 2)"]);
}

#[test]
fn the_report_lists_programs_by_task_then_strays() {
    let tasks = [task("a", "Int"), task("b", "Int")];
    let fronts = frontiers(&[("a", &["1"]), ("b", &["2"])]);
    let cands = [
        candidate("b", "3"),
        candidate("zz", "4"),
        candidate("a", "5"),
    ];
    let out = run(
        &tasks,
        &[],
        &fronts,
        &cands,
        &config(5),
        &mut Nothing::default(),
    )
    .unwrap();
    let order: Vec<Source> = out
        .report
        .programs
        .iter()
        .map(|p| p.source.clone())
        .collect();
    assert_eq!(
        order,
        vec![
            Source::Member {
                task: "a".into(),
                index: 0
            },
            Source::Candidate { index: 2 },
            Source::Member {
                task: "b".into(),
                index: 0
            },
            Source::Candidate { index: 0 },
            Source::Candidate { index: 1 },
        ]
    );
}

#[test]
fn refusals_in_a_row_count_from_the_shorter_list() {
    let mut stitch = Failing {
        errors: vec![
            CompressError::TooDeep { index: 3 },
            CompressError::TooDeep { index: 0 },
        ],
        inner: Nothing::default(),
    };
    let out = run(
        &shared_tasks(),
        &[],
        &BTreeMap::new(),
        &shared_candidates(),
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert_eq!(
        stitch.inner.calls,
        vec![
            vec!["t0", "t1", "t2", "t3"],
            vec!["t0", "t1", "t2"],
            vec!["t1", "t2"]
        ]
    );
    let CompressionReport::Ran { left_out, .. } = &out.report.compression else {
        panic!("{:?}", out.report.compression);
    };
    let tasks: Vec<&str> = left_out.iter().map(|l| l.task.as_str()).collect();
    assert_eq!(tasks, vec!["t3", "t0"]);
}

#[test]
fn a_refusal_that_names_no_program_fails_the_run() {
    let mut stitch = Failing {
        errors: vec![CompressError::TooDeep { index: 4 }],
        inner: Nothing::default(),
    };
    let out = run(
        &shared_tasks(),
        &[],
        &BTreeMap::new(),
        &shared_candidates(),
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert!(matches!(
        out.report.compression,
        CompressionReport::Failed { .. }
    ));
    assert_eq!(stitch.inner.calls.len(), 1);
}

#[test]
fn a_runner_that_returns_the_wrong_number_of_programs_fails_the_run() {
    let tasks = [task("a", "Int"), task("b", "Int")];
    let cands = [candidate("a", "1"), candidate("b", "2")];
    /// Returns one rewritten program, whatever it is given.
    struct Short;
    impl Stitch for Short {
        fn run(
            &mut self,
            _arena: &mut Arena,
            _libs: &[LibDef],
            programs: &[FrontierProgram],
            _cfg: &CompressConfig,
        ) -> Result<Compression, CompressError> {
            Ok(Compression {
                new_libs: Vec::new(),
                rewritten: vec![programs[0].program],
                names: LibNames::new(),
                stitch_steps: 0,
                rejected: None,
            })
        }
    }
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut Short,
    )
    .unwrap();
    assert!(matches!(
        out.report.compression,
        CompressionReport::Failed { .. }
    ));
    assert_eq!(out.frontiers["b"], vec!["2"]);
}

#[test]
fn new_abstractions_that_do_not_expand_fail_the_run() {
    // Arity 2 with one lambda types, as λx. x, but does not expand.
    let tasks = [task("a", "Int")];
    let cands = [candidate("a", "(add 1 1)")];
    let mut stitch = Fixed {
        libs: vec![(2, "(lam $0)")],
        rewritten: vec!["(add 1 1)"],
    };
    let out = run(
        &tasks,
        &[],
        &BTreeMap::new(),
        &cands,
        &config(5),
        &mut stitch,
    )
    .unwrap();
    assert!(out.new_library.is_empty());
    match &out.report.compression {
        CompressionReport::Failed { error, .. } => {
            assert!(error.contains("do not expand"), "{error}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_frontier_member_repeated_is_a_duplicate() {
    let tasks = [task("a", "Int")];
    let fronts = frontiers(&[("a", &["(add 3 3)", "(add 3 3)"])]);
    let out = run(
        &tasks,
        &[],
        &fronts,
        &[],
        &config(5),
        &mut Nothing::default(),
    )
    .unwrap();
    assert_eq!(
        *fate_of(
            &out,
            Source::Member {
                task: "a".into(),
                index: 1
            }
        ),
        Fate::Duplicate {
            of: Source::Member {
                task: "a".into(),
                index: 0
            }
        }
    );
    assert_eq!(out.frontiers["a"], vec!["(add 3 3)"]);
}

#[test]
fn a_library_entry_that_uses_a_missing_library_is_named() {
    // The entry is 0, though it names lib_7.
    let result = run(
        &[task("a", "Int")],
        &[lib(0, "lib_7")],
        &BTreeMap::new(),
        &[],
        &config(1),
        &mut Nothing::default(),
    );
    assert!(
        matches!(result, Err(DriverError::Library { index: 0, .. })),
        "{result:?}"
    );
}
