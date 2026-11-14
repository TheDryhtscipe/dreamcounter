//! The iteration step with programs that use the prelude's functions, as the
//! programs of proposers do (`dc_surface::prelude`).

use std::collections::BTreeMap;

use dc_compress::CompressConfig;
use dc_core::{expand, sexpr, Arena, LibDef, LibId, Scheme};
use dc_driver::{
    run_iteration, Candidate, CompressionReport, DriverConfig, DriverError, Fate, InProcess,
    IterationInput, IterationOutcome, LibraryEntry, Source, TaskSpec,
};
use dc_surface::prelude::{Prelude, PreludeMode};

fn scheme(text: &str) -> Scheme {
    text.parse().unwrap_or_else(|e| panic!("{text}: {e:?}"))
}

fn task(name: &str, signature: &str) -> TaskSpec {
    TaskSpec {
        name: name.to_owned(),
        signature: scheme(signature),
    }
}

/// `src` compiled with the prelude on, as core text.
fn compile(prelude: &Prelude, src: &str) -> String {
    let mut arena = Arena::new();
    let t = prelude
        .top_level(PreludeMode::On)
        .close_expression(&mut arena, "test", src)
        .unwrap_or_else(|e| panic!("{src}: {e}"));
    sexpr::print(&arena, t).unwrap()
}

fn candidate(task: &str, program: String) -> Candidate {
    Candidate {
        task: task.to_owned(),
        program,
    }
}

fn config() -> DriverConfig {
    DriverConfig {
        frontier_size: 5,
        compress: CompressConfig::default(),
    }
}

fn run(
    prelude: Option<&Prelude>,
    tasks: &[TaskSpec],
    library: &[LibraryEntry],
    frontiers: &BTreeMap<String, Vec<String>>,
    candidates: &[Candidate],
) -> Result<IterationOutcome, DriverError> {
    run_iteration(
        IterationInput {
            tasks,
            library,
            frontiers,
            candidates,
            prelude,
        },
        &config(),
        &mut InProcess,
    )
}

/// The text of `LibId`, as `sexpr` prints a library reference.
fn lib_text(id: LibId) -> String {
    format!("lib_{}", id.0)
}

#[test]
fn a_candidate_that_uses_map_is_kept_with_the_prelude_and_invalid_without() {
    let prelude = Prelude::load().unwrap();
    let tasks = [task("inc", "List Int -> List Int")];
    let program = compile(&prelude, "fun xs -> map (fun x -> x + 1) xs");
    let map = lib_text(prelude.get("map").unwrap().lib());
    assert!(program.contains(&map), "{program}");
    let cands = [candidate("inc", program.clone())];

    let out = run(Some(&prelude), &tasks, &[], &BTreeMap::new(), &cands).unwrap();
    assert_eq!(
        out.report.programs[0].fate,
        Fate::Kept,
        "{:?}",
        out.report.programs
    );
    assert_eq!(out.frontiers["inc"], std::slice::from_ref(&program));

    let out = run(None, &tasks, &[], &BTreeMap::new(), &cands).unwrap();
    assert!(
        matches!(out.report.programs[0].fate, Fate::Invalid { .. }),
        "{:?}",
        out.report.programs
    );
    assert!(out.frontiers["inc"].is_empty());
}

#[test]
fn a_frontier_member_that_uses_the_prelude_loads_only_with_the_prelude() {
    let prelude = Prelude::load().unwrap();
    let tasks = [task("inc", "List Int -> List Int")];
    let member = compile(&prelude, "fun xs -> map (fun x -> x + 1) xs");
    let fronts = BTreeMap::from([("inc".to_owned(), vec![member.clone()])]);
    let out = run(Some(&prelude), &tasks, &[], &fronts, &[]).unwrap();
    assert_eq!(out.frontiers["inc"], [member]);
    assert!(matches!(
        run(None, &tasks, &[], &fronts, &[]),
        Err(DriverError::Member { .. })
    ));
}

#[test]
fn a_library_entry_may_use_the_prelude_and_a_wrong_type_is_still_caught() {
    let prelude = Prelude::load().unwrap();
    let map = lib_text(prelude.get("map").unwrap().lib());
    let tasks = [task("inc", "List Int -> List Int")];
    // `lib_0 xs` is `map (+1) xs`.
    let entry = LibraryEntry {
        arity: 1,
        term: format!("(lam ({map} (lam (add $0 1)) $0))"),
    };
    let cands = [candidate("inc", "(lam (lib_0 $0))".to_owned())];
    let out = run(
        Some(&prelude),
        &tasks,
        std::slice::from_ref(&entry),
        &BTreeMap::new(),
        &cands,
    )
    .unwrap();
    assert_eq!(out.report.programs[0].fate, Fate::Kept, "{:?}", out.report);

    // Without the prelude the entry is an unknown library.
    assert!(matches!(
        run(
            None,
            &tasks,
            std::slice::from_ref(&entry),
            &BTreeMap::new(),
            &cands
        ),
        Err(DriverError::Library { index: 0, .. })
    ));

    // A task the entry does not solve: the candidate does not type.
    let bad = [task("flag", "Bool")];
    let out = run(
        Some(&prelude),
        &bad,
        std::slice::from_ref(&entry),
        &BTreeMap::new(),
        &[candidate("flag", "(lib_0 nil)".to_owned())],
    )
    .unwrap();
    assert!(matches!(out.report.programs[0].fate, Fate::Invalid { .. }));
}

/// Four tasks whose programs share the shape `map (+ k) (filter (> j) xs)`.
fn shared() -> (Prelude, Vec<TaskSpec>, Vec<Candidate>) {
    let prelude = Prelude::load().unwrap();
    let mut tasks = Vec::new();
    let mut cands = Vec::new();
    for (i, (k, j)) in [(1, 0), (2, 5), (3, 7), (4, 9)].into_iter().enumerate() {
        let name = format!("t{i}");
        let src = format!("fun xs -> map (fun x -> x + {k}) (filter (fun x -> x > {j}) xs)");
        cands.push(candidate(&name, compile(&prelude, &src)));
        tasks.push(task(&name, "List Int -> List Int"));
    }
    (prelude, tasks, cands)
}

/// `text` expanded under `prelude` and then `defs`.
fn expanded(prelude: &Prelude, arena: &mut Arena, defs: &[LibDef], text: &str) -> String {
    let t = sexpr::parse(arena, text).unwrap();
    let t = prelude.expand(arena, t);
    let t = expand(arena, defs, t).unwrap();
    sexpr::print(arena, t).unwrap()
}

#[test]
fn stitch_sees_prelude_names_and_rewritten_programs_keep_the_references() {
    let (prelude, tasks, cands) = shared();
    let out = run(Some(&prelude), &tasks, &[], &BTreeMap::new(), &cands).unwrap();
    for p in &out.report.programs {
        assert_eq!(p.fate, Fate::Kept, "{:?}", out.report);
    }
    let CompressionReport::Ran {
        kept, fallbacks, ..
    } = &out.report.compression
    else {
        panic!("{:?}", out.report.compression);
    };
    assert!(*kept > 0, "{:?}", out.report.compression);
    assert!(fallbacks.is_empty(), "{fallbacks:?}");

    let map = lib_text(prelude.get("map").unwrap().lib());
    let filter = lib_text(prelude.get("filter").unwrap().lib());
    // The prelude references are in the new library or in the programs, never
    // replaced by a definition: no program or entry holds a `foldr` body.
    let all: Vec<&String> = out
        .new_library
        .iter()
        .map(|e| &e.term)
        .chain(out.frontiers.values().flatten())
        .collect();
    assert!(all.iter().any(|t| t.contains(&map)), "{all:?}");
    assert!(all.iter().any(|t| t.contains(&filter)), "{all:?}");
    assert!(all.iter().all(|t| !t.contains("foldr")), "{all:?}");
    assert!(
        out.new_library.iter().any(|e| e.term.contains("lib_4294")),
        "an abstraction should use a prelude function: {:?}",
        out.new_library
    );

    // The rewriting keeps what each program means.
    let mut arena = Arena::new();
    let defs: Vec<LibDef> = out
        .new_library
        .iter()
        .map(|e| LibDef {
            arity: e.arity,
            term: {
                let t = sexpr::parse(&mut arena, &e.term).unwrap();
                prelude.expand(&mut arena, t)
            },
        })
        .collect();
    for c in &cands {
        let want = expanded(&prelude, &mut arena, &[], &c.program);
        let got = expanded(&prelude, &mut arena, &defs, &out.frontiers[&c.task][0]);
        assert_eq!(got, want, "{}", c.task);
        assert_ne!(out.frontiers[&c.task][0], c.program, "{}", c.task);
    }
}

#[test]
fn the_new_library_feeds_the_next_iteration_with_the_prelude() {
    let (prelude, tasks, cands) = shared();
    let first = run(Some(&prelude), &tasks, &[], &BTreeMap::new(), &cands).unwrap();
    assert!(!first.new_library.is_empty());
    // Run again from the first iteration's library and frontiers: they load.
    let again = run(
        Some(&prelude),
        &tasks,
        &first.new_library,
        &first.frontiers,
        &[],
    )
    .unwrap();
    assert_eq!(again.frontiers, first.frontiers);
    let source = |t: &str| Source::Member {
        task: t.to_owned(),
        index: 0,
    };
    assert!(again
        .report
        .programs
        .iter()
        .any(|p| p.source == source("t0") && p.fate == Fate::Kept));
}
