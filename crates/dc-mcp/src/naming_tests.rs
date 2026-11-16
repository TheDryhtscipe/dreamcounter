//! Tests of naming jobs (decision D23).

use super::*;
use crate::proposer::{next_batch, NextBatch};
use crate::state::{Arm, Check, Event, FrontierMember, RunConfig, StitchSettings};
use crate::verify::task;
use tempfile::TempDir;

const MERGE: &str = "collapse/merge";
const CANCEL: &str = "collapse/cancel";

fn config(autodoc: bool) -> RunConfig {
    RunConfig {
        autodoc,
        arm: Arm::Blind,
        seed: 3,
        proposals_per_task: 1,
        programs_per_batch: 1,
        lease_ms: 1000,
        max_unchecked_per_batch: 3,
        frontier_size: 5,
        stitch: StitchSettings::default(),
        prelude: shipped_prelude(),
    }
}

fn setup(autodoc: bool, library: usize) -> (TempDir, Store, RunId, VerifierCache) {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    store
        .create_run(&run, config(autodoc), vec![MERGE.into(), CANCEL.into()])
        .unwrap();
    if library > 0 {
        let entries = (0..library)
            .map(|_| LibraryEntry {
                arity: 1,
                term: "(lam $0)".into(),
                ..LibraryEntry::default()
            })
            .collect();
        store
            .append(&run, vec![Event::LibraryExtended { entries }])
            .unwrap();
    }
    (dir, store, run, VerifierCache::default())
}

fn job(store: &Store, run: &RunId, v: &VerifierCache, worker: &str, now: u64) -> NextNamingJob {
    next_naming_job(store, run, worker, now, v).unwrap()
}

fn batch(store: &Store, run: &RunId, v: &VerifierCache, worker: &str) -> NextBatch {
    next_batch(store, run, worker, 0, v).unwrap()
}

/// That `result` is a refusal whose text has `why`.
fn assert_rejected(result: Result<(), crate::store::StoreError>, why: &str) {
    match result {
        Ok(()) => panic!("accepted; expected a refusal with {why:?}"),
        Err(e) => assert!(e.to_string().contains(why), "{e}"),
    }
}

fn lib(store: &Store, run: &RunId) -> Vec<LibraryEntry> {
    store.state(run).unwrap().library().to_vec()
}

/// A program that solves collapse/merge, with `wrap` applied to its entry.
fn solves_merge_through(wrap: &str) -> String {
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == MERGE)
        .unwrap();
    format!(
        "{}\nlet solve = {wrap} ({})\n",
        v.source,
        v.references.last().unwrap().entry
    )
}

#[test]
fn names_are_checked_against_the_language_the_prelude_and_the_library() {
    let mut library = vec![LibraryEntry::default(), LibraryEntry::default()];
    library[1].name = Some("taken".into());
    assert_eq!(check_name("twice_map", 0, &library), Ok(()));
    assert_eq!(check_name("_go'", 0, &library), Ok(()));
    // An entry may keep its own name.
    assert_eq!(check_name("taken", 1, &library), Ok(()));
    let long = "a".repeat(MAX_ENTRY_NAME_LEN + 1);
    for bad in [
        "", "x", "xs", "Twice", "let", "mod", "solve", "fn_3", "fn_12", "map", "foldr", "taken",
        "a b", "x-y", &long,
    ] {
        assert!(
            check_name(bad, 0, &library).is_err(),
            "{bad:?} was accepted"
        );
    }
    // Only fn_ with digits alone is a generated name.
    assert_eq!(check_name("fn_x", 0, &library), Ok(()));
    assert_eq!(check_name("fn_", 0, &library), Ok(()));
}

#[test]
fn docs_are_one_bounded_line() {
    assert_eq!(check_doc("Applies f twice."), Ok(()));
    let long = "a".repeat(MAX_DOC_LEN + 1);
    for bad in [
        "",
        "   ",
        "two\nlines",
        "tab\there",
        "line\u{2028}separator",
        "bidi\u{202e}override",
        "zero\u{200b}width",
        &long,
    ] {
        assert!(check_doc(bad).is_err(), "{bad:?} was accepted");
    }
}

#[test]
fn uses_finds_whole_library_references_only() {
    assert!(uses("(lam (lib_1 $0))", 1));
    assert!(uses("lib_1", 1));
    assert!(!uses("(lam (lib_10 $0))", 1));
    assert!(!uses("(lam (xlib_1 $0))", 1));
    assert!(!uses("(lam (lib_2 $0))", 1));
}

#[test]
fn batches_wait_until_each_new_function_is_named_then_show_its_name_and_doc() {
    let (_dir, store, run, v) = setup(true, 1);
    let waiting = batch(&store, &run, &v, "a");
    assert!(waiting.batch.is_none());
    let wait = waiting.wait.unwrap();
    assert!(
        wait.reason.contains("dc_next_naming_job"),
        "the wait must say how to get naming jobs"
    );
    assert!(
        wait.names_pending,
        "a launcher reads this flag, not the reason"
    );

    let got = job(&store, &run, &v, "n", 0).job.unwrap();
    assert_eq!(got.job, 0);
    assert!(got.prompt.contains("let fn_0 = "), "{}", got.prompt);
    assert!(got
        .prompt
        .contains(&format!("job 0, token \"{}\"", got.token)));

    let receipt = submit_naming(
        &store,
        &run,
        "n",
        0,
        &got.token,
        "ident",
        "Returns its argument.",
    )
    .unwrap();
    assert_eq!(receipt.name, "ident");
    assert!(receipt.refused.is_empty());
    let entry = &lib(&store, &run)[0];
    assert_eq!(entry.name.as_deref(), Some("ident"));
    assert_eq!(entry.doc.as_deref(), Some("Returns its argument."));

    let p = batch(&store, &run, &v, "a").batch.unwrap().prompt;
    assert!(p.contains("-- Returns its argument.\nident : "), "{p}");
    assert!(p.contains("let ident = "), "{p}");
    assert!(!p.contains("fn_0"), "{p}");
    // No job is left.
    assert!(job(&store, &run, &v, "n", 0).job.is_none());
}

#[test]
fn a_bad_name_ends_the_job_and_the_function_keeps_its_generated_name() {
    let (_dir, store, run, v) = setup(true, 1);
    let got = job(&store, &run, &v, "n", 0).job.unwrap();
    let receipt = submit_naming(&store, &run, "n", 0, &got.token, "map", "Copies.").unwrap();
    assert_eq!(receipt.name, "fn_0");
    assert_eq!(receipt.doc.as_deref(), Some("Copies."));
    assert_eq!(receipt.refused.len(), 1, "{:?}", receipt.refused);
    assert!(receipt.refused[0].contains("prelude"));
    let state = store.state(&run).unwrap();
    assert!(state.is_named(0));
    assert_eq!(state.library()[0].name, None);
    // The run goes on, under the generated name.
    let p = batch(&store, &run, &v, "a").batch.unwrap().prompt;
    assert!(p.contains("-- Copies.\nfn_0 : "), "{p}");
}

#[test]
fn a_bad_doc_is_dropped_and_the_name_kept() {
    let (_dir, store, run, v) = setup(true, 1);
    let got = job(&store, &run, &v, "n", 0).job.unwrap();
    let receipt = submit_naming(&store, &run, "n", 0, &got.token, "ident", "a\nb").unwrap();
    assert_eq!(receipt.name, "ident");
    assert_eq!(receipt.doc, None);
    assert_eq!(receipt.refused.len(), 1);
    assert_eq!(lib(&store, &run)[0].doc, None);
}

#[test]
fn a_wrong_token_or_worker_or_a_second_answer_is_refused_and_writes_nothing() {
    let (_dir, store, run, v) = setup(true, 1);
    let got = job(&store, &run, &v, "n", 0).job.unwrap();
    let before = store.state(&run).unwrap();
    for (worker, token, job) in [
        ("n", "wrong", 0),
        ("m", got.token.as_str(), 0),
        ("n", got.token.as_str(), 7),
    ] {
        assert!(matches!(
            submit_naming(&store, &run, worker, job, token, "ident", "Doc."),
            Err(Refused::Request(_))
        ));
    }
    assert_eq!(store.state(&run).unwrap(), before);
    submit_naming(&store, &run, "n", 0, &got.token, "ident", "Doc.").unwrap();
    let after = store.state(&run).unwrap();
    assert!(matches!(
        submit_naming(&store, &run, "n", 0, &got.token, "other", "Doc."),
        Err(Refused::Request(_))
    ));
    assert_eq!(store.state(&run).unwrap(), after);
}

#[test]
fn a_job_is_held_for_its_lease_then_handed_out_again() {
    let (_dir, store, run, v) = setup(true, 1);
    let first = job(&store, &run, &v, "a", 0).job.unwrap();
    // A worker holds one job; the only entry is held, so b waits.
    assert!(job(&store, &run, &v, "a", 10)
        .wait
        .unwrap()
        .reason
        .contains("you hold"));
    let held = job(&store, &run, &v, "b", 10).wait.unwrap();
    assert!(
        held.reason.contains("held by another job"),
        "{}",
        held.reason
    );
    assert!(held.names_pending);
    // After the lease, b gets a new job for the same entry.
    // At its deadline the job is still held, as a batch is.
    assert!(job(&store, &run, &v, "b", 1000).job.is_none());
    let second = job(&store, &run, &v, "b", 1001).job.unwrap();
    assert_eq!(second.job, 1);
    // The first answer in wins; the later one is refused.
    submit_naming(&store, &run, "a", 0, &first.token, "ident", "Doc.").unwrap();
    assert!(submit_naming(&store, &run, "b", 1, &second.token, "other", "Doc.").is_err());
    assert_eq!(lib(&store, &run)[0].name.as_deref(), Some("ident"));
}

#[test]
fn two_functions_cannot_take_the_same_name() {
    let (_dir, store, run, v) = setup(true, 2);
    let a = job(&store, &run, &v, "a", 0).job.unwrap();
    let b = job(&store, &run, &v, "b", 0).job.unwrap();
    submit_naming(&store, &run, "a", a.job, &a.token, "ident", "Doc.").unwrap();
    let r = submit_naming(&store, &run, "b", b.job, &b.token, "ident", "Doc.").unwrap();
    assert_eq!(r.name, "fn_1");
    let names: Vec<_> = lib(&store, &run).into_iter().map(|e| e.name).collect();
    assert_eq!(names, [Some("ident".to_owned()), None]);
}

#[test]
fn with_autodoc_off_there_are_no_jobs_and_batches_do_not_wait() {
    let (_dir, store, run, v) = setup(false, 1);
    assert!(matches!(
        next_naming_job(&store, &run, "n", 0, &v),
        Err(Refused::Request(_))
    ));
    assert!(batch(&store, &run, &v, "a").batch.is_some());
}

#[test]
fn a_named_function_is_in_scope_by_its_name_and_by_fn_i() {
    let named = LibraryEntry {
        arity: 1,
        term: "(lam $0)".into(),
        name: Some("ident".into()),
        doc: None,
    };
    let family = task(MERGE).unwrap();
    for wrap in ["ident", "fn_0"] {
        let mut verifier = Verifier::new(std::slice::from_ref(&named)).unwrap();
        let check = verifier.check(&*family, &solves_merge_through(wrap));
        assert!(matches!(check, Check::Passed { .. }), "{wrap}: {check:?}");
    }
    // Without the name, `ident` is unknown.
    let unnamed = LibraryEntry {
        name: None,
        ..named
    };
    let mut verifier = Verifier::new(std::slice::from_ref(&unnamed)).unwrap();
    let check = verifier.check(&*family, &solves_merge_through("ident"));
    assert!(matches!(check, Check::Compile { .. }), "{check:?}");
}

#[test]
fn the_job_shows_frontier_programs_that_use_the_function() {
    let (_dir, store, run, v) = setup(true, 1);
    store
        .append(
            &run,
            vec![Event::FrontierSet {
                task: MERGE.into(),
                members: vec![FrontierMember {
                    program: "(lam (lib_0 $0))".into(),
                }],
            }],
        )
        .unwrap();
    let got = job(&store, &run, &v, "n", 0).job.unwrap();
    assert!(
        got.prompt.contains("## Programs that use it"),
        "{}",
        got.prompt
    );
    assert!(got.prompt.contains("fun x -> fn_0 x\n"), "{}", got.prompt);
    // Payloads never name a task.
    for task in [MERGE, CANCEL] {
        assert!(!got.prompt.contains(task), "{}", got.prompt);
    }
}

#[test]
fn the_state_refuses_names_that_do_not_come_from_a_checked_job() {
    let (_dir, store, run, v) = setup(true, 1);
    // A library extension cannot carry a name.
    let named = Event::LibraryExtended {
        entries: vec![LibraryEntry {
            arity: 1,
            term: "(lam $0)".into(),
            name: Some("ident".into()),
            doc: None,
        }],
    };
    assert_rejected(store.append(&run, vec![named]), "no name or doc");
    let done = |job: u64, name: &str| {
        Event::NamingDone(NamingDone {
            job,
            entry: 0,
            name: Some(name.into()),
            doc: None,
            refused: vec![],
        })
    };
    // A result for a job never issued.
    assert_rejected(
        store.append(&run, vec![done(0, "ident")]),
        "no naming job 0",
    );
    job(&store, &run, &v, "n", 0).job.unwrap();
    // A name whose form the check refuses.
    assert_rejected(store.append(&run, vec![done(0, "solve")]), "entry point");
    assert_rejected(store.append(&run, vec![done(0, "ab")]), "3 to 40 bytes");
    assert_rejected(
        store.append(&run, vec![done(0, "Ident")]),
        "not a variable name",
    );
    store.append(&run, vec![done(0, "ident")]).unwrap();
    // The entry is named once.
    assert_rejected(store.append(&run, vec![done(0, "other")]), "already named");
}

#[test]
fn replay_does_not_check_names_against_the_builds_primitives_and_prelude() {
    // A journal stays readable if a later build gains a primitive or prelude
    // function with a name the run used; the server alone refuses those.
    let (_dir, store, run, v) = setup(true, 1);
    job(&store, &run, &v, "n", 0).job.unwrap();
    let done = Event::NamingDone(NamingDone {
        job: 0,
        entry: 0,
        name: Some("map".into()),
        doc: None,
        refused: vec![],
    });
    store.append(&run, vec![done]).unwrap();
    assert_eq!(lib(&store, &run)[0].name.as_deref(), Some("map"));
}

#[test]
fn entries_and_settings_written_before_naming_read_back_the_same() {
    let old = r#"{"arity":1,"term":"(lam $0)"}"#;
    let entry: LibraryEntry = serde_json::from_str(old).unwrap();
    assert_eq!(entry.name, None);
    assert_eq!(serde_json::to_string(&entry).unwrap(), old);
    let mut json = serde_json::to_value(config(true)).unwrap();
    json.as_object_mut().unwrap().remove("autodoc");
    let read: RunConfig = serde_json::from_value(json).unwrap();
    assert!(!read.autodoc, "a run made before the setting has it off");
}

#[test]
fn the_state_issues_one_live_job_per_entry_and_per_worker() {
    let (_dir, store, run, _v) = setup(true, 2);
    let issued = |job: u64, entry: u64, worker: &str, at: u64| {
        Event::NamingIssued(NamingIssued {
            job,
            entry,
            worker: worker.into(),
            token: "t".into(),
            issued_ms: at,
            deadline_ms: at + 1000,
            prompt: String::new(),
        })
    };
    store.append(&run, vec![issued(0, 0, "a", 0)]).unwrap();
    // Entry 0 is held, and so is worker a, until 1000.
    assert_rejected(
        store.append(&run, vec![issued(1, 0, "b", 10)]),
        "still live",
    );
    assert_rejected(
        store.append(&run, vec![issued(1, 1, "a", 10)]),
        "still live",
    );
    store.append(&run, vec![issued(1, 1, "b", 10)]).unwrap();
    // After the lease, both can be issued again.
    store.append(&run, vec![issued(2, 0, "a", 1001)]).unwrap();
    // Job numbers run in order, and the entry must exist.
    assert_rejected(
        store.append(&run, vec![issued(9, 0, "c", 5000)]),
        "next job is 3",
    );
    assert_rejected(
        store.append(&run, vec![issued(3, 7, "c", 5000)]),
        "no entry 7",
    );
}

#[test]
fn the_job_states_every_rule_the_checks_apply() {
    let (_dir, store, run, v) = setup(true, 1);
    let prompt = job(&store, &run, &v, "n", 0).job.unwrap().prompt;
    for rule in [
        format!("{MIN_ENTRY_NAME_LEN} to {MAX_ENTRY_NAME_LEN} bytes"),
        "not a keyword".to_owned(),
        format!("not `{ENTRY}`"),
        "not fn_ followed by digits".to_owned(),
        "not a primitive or\nprelude function".replace("\n", " "),
        "not a name listed below".to_owned(),
        "printable ASCII".to_owned(),
        format!("at most {MAX_DOC_LEN} bytes"),
    ] {
        assert!(prompt.contains(&rule), "the prompt does not say {rule:?}");
    }
}

#[test]
fn a_wait_says_whether_names_are_pending() {
    let (_dir, store, run, v) = setup(true, 1);
    let got = job(&store, &run, &v, "a", 0).job.unwrap();
    // The one entry is held by a's job: names are still pending.
    assert!(job(&store, &run, &v, "b", 0).wait.unwrap().names_pending);
    submit_naming(&store, &run, "a", got.job, &got.token, "ident", "Doc.").unwrap();
    // Once it is named, nothing waits: a naming job's wait says so.
    let none = job(&store, &run, &v, "b", 0).wait.unwrap();
    assert!(none.reason.contains("no function waits"), "{}", none.reason);
    assert!(!none.names_pending);
    // A batch's wait with no names pending says so too.
    let first = batch(&store, &run, &v, "c");
    assert!(first.batch.is_some());
    let busy = batch(&store, &run, &v, "c").wait.unwrap();
    assert!(!busy.names_pending, "{}", busy.reason);
}
