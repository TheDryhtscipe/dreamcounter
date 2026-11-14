use std::fs;
use std::path::Path;

use tempfile::TempDir;

use super::*;
use crate::state::{Arm, FrontierMember, LibraryEntry, LogKind, Rejected};

fn config() -> RunConfig {
    RunConfig {
        arm: Arm::Blind,
        seed: 11,
        proposals_per_task: 3,
    }
}

fn id(text: &str) -> RunId {
    RunId::new(text).unwrap()
}

fn tasks() -> Vec<String> {
    vec!["sum".into(), "sort".into()]
}

fn note(iteration: u64, text: &str) -> Event {
    Event::Logged(LogRecord {
        kind: LogKind::Note,
        iteration,
        task: None,
        text: text.into(),
    })
}

fn frontier(task: &str, program: &str) -> Event {
    Event::FrontierSet {
        task: task.into(),
        members: vec![FrontierMember {
            program: program.into(),
        }],
    }
}

fn journal(root: &Path, run: &str) -> std::path::PathBuf {
    root.join(run).join("journal")
}

/// A run that is part of the way through its second iteration.
fn mid_iteration(store: &Store, run: &RunId) {
    store.create_run(run, config(), tasks()).unwrap();
    store.commit(run, |_| vec![note(0, "first")]).unwrap();
    store
        .commit(run, |_| {
            vec![
                frontier("sum", "(lam 0)"),
                Event::LibraryExtended {
                    entries: vec![LibraryEntry {
                        body: "(lam (lam 1))".into(),
                    }],
                },
                Event::IterationFinished { iteration: 0 },
            ]
        })
        .unwrap();
    store
        .commit(run, |_| {
            vec![Event::Logged(LogRecord {
                kind: LogKind::Payload,
                iteration: 1,
                task: Some("sort".into()),
                text: "payload text".into(),
            })]
        })
        .unwrap();
}

#[test]
fn a_new_store_on_the_same_directory_sees_the_same_run() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let first = Store::open(dir.path()).unwrap();
    mid_iteration(&first, &run);
    let before = first.state(&run).unwrap();
    assert_eq!(before.iteration(), 1);
    assert_eq!(before.library().len(), 1);
    assert_eq!(before.frontier("sum").unwrap().len(), 1);
    assert_eq!(before.log_count(LogKind::Payload), 1);
    let log_before = first.log(&run).unwrap();
    assert_eq!(log_before.len(), 2);
    assert_eq!(log_before[1].text, "payload text");
    drop(first);

    let second = Store::open(dir.path()).unwrap();
    assert_eq!(second.state(&run).unwrap(), before);
    assert_eq!(second.log(&run).unwrap(), log_before);
    assert_eq!(second.runs().unwrap(), vec![run]);
}

#[test]
fn create_refuses_an_existing_run_and_leaves_it_alone() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let bytes = fs::read(journal(dir.path(), "r1")).unwrap();
    assert!(matches!(
        store.create_run(&run, config(), tasks()),
        Err(StoreError::RunExists(_))
    ));
    // Another store, which has nothing cached, refuses too.
    let other = Store::open(dir.path()).unwrap();
    assert!(matches!(
        other.create_run(&run, config(), tasks()),
        Err(StoreError::RunExists(_))
    ));
    assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);
}

#[test]
fn create_refuses_a_bad_task_set_and_makes_no_run() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    assert!(matches!(
        store.create_run(&id("r1"), config(), vec![]),
        Err(StoreError::Rejected(Rejected::NoTasks))
    ));
    assert_eq!(store.runs().unwrap(), vec![]);
    assert!(matches!(
        store.state(&id("r1")),
        Err(StoreError::NoSuchRun(_))
    ));
}

#[test]
fn a_run_that_was_never_created_is_reported_and_not_made() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("ghost");
    assert!(matches!(store.state(&run), Err(StoreError::NoSuchRun(_))));
    assert!(matches!(
        store.commit(&run, |_| vec![note(0, "x")]),
        Err(StoreError::NoSuchRun(_))
    ));
    assert!(matches!(store.log(&run), Err(StoreError::NoSuchRun(_))));
    assert!(!dir.path().join("ghost").exists());
}

#[test]
fn a_rejected_commit_writes_nothing() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let state = store.state(&run).unwrap();
    let bytes = fs::read(journal(dir.path(), "r1")).unwrap();

    // The first event alone is fine. The second is not, so neither is kept.
    let result = store.commit(&run, |_| vec![note(1, "kept?"), note(7, "wrong iteration")]);
    assert!(matches!(
        result,
        Err(StoreError::Rejected(Rejected::WrongIteration {
            current: 1,
            given: 7
        }))
    ));
    assert_eq!(store.state(&run).unwrap(), state);
    assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);

    // A commit with no events writes nothing either.
    store.commit(&run, |_| vec![]).unwrap();
    assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);
}

#[test]
fn commit_shows_the_closure_the_current_state() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    store
        .commit(&run, |state| {
            assert_eq!(state.iteration(), 1);
            vec![Event::IterationFinished {
                iteration: state.iteration(),
            }]
        })
        .unwrap();
    assert_eq!(store.state(&run).unwrap().iteration(), 2);
}

#[test]
fn a_torn_last_record_is_dropped_and_the_run_goes_on() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let writer = Store::open(dir.path()).unwrap();
    mid_iteration(&writer, &run);
    let before = writer.state(&run).unwrap();
    let path = journal(dir.path(), "r1");
    let kept = fs::read(&path).unwrap();
    writer
        .commit(&run, |_| vec![note(1, "torn"), frontier("sort", "(lam 0)")])
        .unwrap();
    let full = fs::read(&path).unwrap();
    assert_ne!(writer.state(&run).unwrap(), before);
    drop(writer);

    for cut in kept.len() + 1..full.len() {
        fs::write(&path, &full[..cut]).unwrap();
        let store = Store::open(dir.path()).unwrap();
        assert_eq!(store.state(&run).unwrap(), before, "cut at {cut}");
        // Reading does not need the tail gone, but the next commit must not
        // write after it.
        store.commit(&run, |_| vec![note(1, "after")]).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert_eq!(&bytes[..kept.len()], &kept[..], "cut at {cut}");
        // Nothing of the torn record is left after the new one.
        let record = &bytes[kept.len()..];
        assert_eq!(
            record.iter().filter(|b| **b == b'\n').count(),
            1,
            "cut at {cut}"
        );
        assert_eq!(record.last(), Some(&b'\n'), "cut at {cut}");
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(
            reopened.state(&run).unwrap(),
            store.state(&run).unwrap(),
            "cut at {cut}"
        );
        assert_eq!(reopened.log(&run).unwrap().last().unwrap().text, "after");
        assert_eq!(reopened.state(&run).unwrap().log_count(LogKind::Note), 2);
    }
}

#[test]
fn a_damaged_journal_is_an_error_and_is_not_changed() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    mid_iteration(&Store::open(dir.path()).unwrap(), &run);
    let path = journal(dir.path(), "r1");
    let good = fs::read(&path).unwrap();
    let last_line = good[..good.len() - 1]
        .iter()
        .rposition(|b| *b == b'\n')
        .unwrap()
        + 1;

    // One byte in the header, in a middle record, and in the last record.
    for at in [3, good.len() / 2, last_line + 4] {
        let mut bad = good.clone();
        bad[at] ^= 0x01;
        fs::write(&path, &bad).unwrap();
        let store = Store::open(dir.path()).unwrap();
        assert!(
            matches!(store.state(&run), Err(StoreError::Corrupt { .. })),
            "byte {at}"
        );
        assert!(
            matches!(
                store.commit(&run, |_| vec![note(1, "x")]),
                Err(StoreError::Corrupt { .. })
            ),
            "byte {at}"
        );
        assert!(
            matches!(store.log(&run), Err(StoreError::Corrupt { .. })),
            "byte {at}"
        );
        assert_eq!(fs::read(&path).unwrap(), bad, "byte {at}");
    }
}

#[test]
fn a_journal_that_shrank_under_a_store_is_an_error() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let store = Store::open(dir.path()).unwrap();
    store.create_run(&run, config(), tasks()).unwrap();
    let path = journal(dir.path(), "r1");
    let short = fs::read(&path).unwrap();
    store.commit(&run, |_| vec![note(0, "a")]).unwrap();
    fs::write(&path, &short).unwrap();
    assert!(matches!(store.state(&run), Err(StoreError::Corrupt { .. })));
}

#[test]
fn two_stores_on_one_run_see_each_other_s_commits() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let a = Store::open(dir.path()).unwrap();
    let b = Store::open(dir.path()).unwrap();
    a.create_run(&run, config(), tasks()).unwrap();
    // Each store has the run cached before the other writes.
    assert_eq!(b.state(&run).unwrap().iteration(), 0);
    for i in 0..6_u64 {
        let store = if i % 2 == 0 { &a } else { &b };
        store
            .commit(&run, |state| {
                assert_eq!(state.iteration(), i);
                vec![
                    note(i, &format!("n{i}")),
                    Event::IterationFinished { iteration: i },
                ]
            })
            .unwrap();
    }
    assert_eq!(a.state(&run).unwrap(), b.state(&run).unwrap());
    assert_eq!(a.state(&run).unwrap().iteration(), 6);
    let texts: Vec<String> = b.log(&run).unwrap().into_iter().map(|r| r.text).collect();
    assert_eq!(texts, ["n0", "n1", "n2", "n3", "n4", "n5"]);
    // A store with nothing cached reads the same from the file alone.
    let c = Store::open(dir.path()).unwrap();
    assert_eq!(c.state(&run).unwrap(), a.state(&run).unwrap());
}

#[test]
fn threads_and_stores_do_not_lose_commits() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    Store::open(dir.path())
        .unwrap()
        .create_run(&run, config(), tasks())
        .unwrap();
    let shared = std::sync::Arc::new(Store::open(dir.path()).unwrap());
    std::thread::scope(|scope| {
        for t in 0..4 {
            let run = run.clone();
            let root = dir.path().to_owned();
            let shared = std::sync::Arc::clone(&shared);
            scope.spawn(move || {
                // Two threads share one store, and two have their own.
                let own = Store::open(&root).unwrap();
                let store = if t < 2 { &*shared } else { &own };
                for i in 0..25 {
                    store
                        .commit(&run, |_| vec![note(0, &format!("{t}-{i}"))])
                        .unwrap();
                }
            });
        }
    });
    let store = Store::open(dir.path()).unwrap();
    assert_eq!(store.state(&run).unwrap().log_count(LogKind::Note), 100);
    let mut texts: Vec<String> = store
        .log(&run)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    texts.sort();
    texts.dedup();
    assert_eq!(texts.len(), 100);
}

#[test]
fn runs_lists_created_runs_in_order_and_skips_other_entries() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    for name in ["b", "a", "c"] {
        store.create_run(&id(name), config(), tasks()).unwrap();
    }
    fs::create_dir(dir.path().join("not-a-run")).unwrap();
    fs::create_dir(dir.path().join("Not A Run Id")).unwrap();
    fs::write(dir.path().join("file"), "x").unwrap();
    assert_eq!(store.runs().unwrap(), vec![id("a"), id("b"), id("c")]);
}
