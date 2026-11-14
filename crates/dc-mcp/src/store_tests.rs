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
    store.append(run, vec![note(0, "first")]).unwrap();
    store
        .append(run, {
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
        .append(run, {
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
        store.append(&run, vec![note(0, "x")]),
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
    let result = store.append(&run, vec![note(1, "kept?"), note(7, "wrong iteration")]);
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
    store.append(&run, vec![]).unwrap();
    assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);
}

#[test]
fn commit_shows_the_closure_the_current_state() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let seen: Result<u64, StoreError> = store.commit(&run, |state| {
        let iteration = state.iteration();
        Ok((iteration, vec![Event::IterationFinished { iteration }]))
    });
    assert_eq!(seen.unwrap(), 1);
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
        .append(&run, vec![note(1, "torn"), frontier("sort", "(lam 0)")])
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
        store.append(&run, vec![note(1, "after")]).unwrap();
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

    // One byte in the header, and one in a record that has a record after it.
    for at in [3, good.len() / 2] {
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
                store.append(&run, vec![note(1, "x")]),
                Err(StoreError::Corrupt { .. })
            ),
            "byte {at}"
        );
        assert!(
            matches!(store.log(&run), Err(StoreError::Corrupt { .. })),
            "byte {at}"
        );
        assert_eq!(fs::read(&path).unwrap(), bad, "byte {at}");
        assert_eq!(fs::read_dir(dir.path().join("r1")).unwrap().count(), 1);
    }
}

#[test]
fn a_record_that_cannot_be_applied_is_reported_where_it_starts() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let store = Store::open(dir.path()).unwrap();
    store.create_run(&run, config(), tasks()).unwrap();
    store.append(&run, vec![note(0, "a")]).unwrap();
    let path = journal(dir.path(), "r1");
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.len() as u64;
    // A record that reads back, for an iteration the run is not in.
    bytes.extend_from_slice(crate::journal::encode(2, &[note(5, "b")]).as_bytes());
    fs::write(&path, &bytes).unwrap();
    for store in [store, Store::open(dir.path()).unwrap()] {
        match store.state(&run) {
            Err(StoreError::Corrupt { error, .. }) => assert_eq!(error.offset, at),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_journal_of_another_format_version_is_not_called_damaged() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    Store::open(dir.path())
        .unwrap()
        .create_run(&run, config(), tasks())
        .unwrap();
    let path = journal(dir.path(), "r1");
    let text = fs::read_to_string(&path).unwrap();
    fs::write(&path, text.replacen("journal 1\n", "journal 2\n", 1)).unwrap();
    match Store::open(dir.path()).unwrap().state(&run) {
        Err(StoreError::UnsupportedFormat { header, .. }) => assert_eq!(header, "dc-mcp journal 2"),
        other => panic!("{other:?}"),
    }
    fs::write(&path, "").unwrap();
    assert!(matches!(
        Store::open(dir.path()).unwrap().state(&run),
        Err(StoreError::Corrupt { .. })
    ));
    fs::write(&path, crate::journal::HEADER).unwrap();
    assert!(matches!(
        Store::open(dir.path()).unwrap().state(&run),
        Err(StoreError::Corrupt { .. })
    ));
}

#[test]
fn a_damaged_tail_is_set_aside_by_the_next_commit_and_not_deleted() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let writer = Store::open(dir.path()).unwrap();
    mid_iteration(&writer, &run);
    let before = writer.state(&run).unwrap();
    let path = journal(dir.path(), "r1");
    let kept = fs::read(&path).unwrap();
    writer.append(&run, vec![note(1, "last")]).unwrap();
    drop(writer);
    let full = fs::read(&path).unwrap();

    // A byte of the last record, its line break, and bytes after it.
    let mut in_record = full.clone();
    in_record[kept.len() + 5] ^= 0x01;
    let mut line_break = full.clone();
    *line_break.last_mut().unwrap() ^= 0x01;
    let mut zeros = kept.clone();
    zeros.extend_from_slice(&[0; 100]);
    for (case, damaged) in [in_record, line_break, zeros].into_iter().enumerate() {
        let run_dir = dir.path().join("r1");
        for entry in fs::read_dir(&run_dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_name() != "journal" {
                fs::remove_file(entry.path()).unwrap();
            }
        }
        fs::write(&path, &damaged).unwrap();
        let store = Store::open(dir.path()).unwrap();
        // Reading changes nothing.
        assert_eq!(store.state(&run).unwrap(), before, "case {case}");
        assert_eq!(fs::read(&path).unwrap(), damaged, "case {case}");
        assert_eq!(fs::read_dir(&run_dir).unwrap().count(), 1, "case {case}");
        // A rejected commit changes nothing either.
        assert!(store.append(&run, vec![note(9, "no")]).is_err());
        assert_eq!(fs::read(&path).unwrap(), damaged, "case {case}");
        assert_eq!(fs::read_dir(&run_dir).unwrap().count(), 1, "case {case}");

        store.append(&run, vec![note(1, "after")]).unwrap();
        let aside = run_dir.join(format!("torn-{}-0", kept.len()));
        assert_eq!(
            fs::read(&aside).unwrap(),
            &damaged[kept.len()..],
            "case {case}"
        );
        assert_eq!(
            &fs::read(&path).unwrap()[..kept.len()],
            &kept[..],
            "case {case}"
        );

        // A second tail at the same place gets its own file.
        let mut again = fs::read(&path).unwrap();
        again.truncate(kept.len());
        again.extend_from_slice(b"second");
        fs::write(&path, &again).unwrap();
        let store = Store::open(dir.path()).unwrap();
        store.append(&run, vec![note(1, "after")]).unwrap();
        assert_eq!(
            fs::read(&aside).unwrap(),
            &damaged[kept.len()..],
            "case {case}"
        );
        let second = run_dir.join(format!("torn-{}-1", kept.len()));
        assert_eq!(fs::read(second).unwrap(), b"second", "case {case}");
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
    store.append(&run, vec![note(0, "a")]).unwrap();
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
        let seen: Result<u64, StoreError> = store.commit(&run, |state| {
            let events = vec![
                note(i, &format!("n{i}")),
                Event::IterationFinished { iteration: i },
            ];
            Ok((state.iteration(), events))
        });
        assert_eq!(seen.unwrap(), i);
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
                        .append(&run, vec![note(0, &format!("{t}-{i}"))])
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

#[test]
fn commit_returns_the_closure_s_value_or_its_error_and_writes_nothing_on_error() {
    #[derive(Debug, PartialEq)]
    enum Refused {
        Full(u64),
        Store(String),
    }
    impl From<StoreError> for Refused {
        fn from(error: StoreError) -> Refused {
            Refused::Store(error.to_string())
        }
    }
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let bytes = fs::read(journal(dir.path(), "r1")).unwrap();
    let state = store.state(&run).unwrap();

    let refused: Result<(), Refused> =
        store.commit(&run, |state| Err(Refused::Full(state.iteration())));
    assert_eq!(refused, Err(Refused::Full(1)));
    let rejected: Result<(), Refused> = store.commit(&run, |_| Ok(((), vec![note(4, "x")])));
    assert!(matches!(rejected, Err(Refused::Store(text)) if text.contains("iteration 4")));
    let missing: Result<(), Refused> = store.commit(&id("none"), |_| Ok(((), vec![])));
    assert!(matches!(missing, Err(Refused::Store(text)) if text.contains("no run")));
    assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);
    assert_eq!(store.state(&run).unwrap(), state);

    let value: Result<&str, Refused> = store.commit(&run, |_| Ok(("picked", vec![note(1, "x")])));
    assert_eq!(value, Ok("picked"));
    assert_eq!(store.read(&run, |s| s.log_count(LogKind::Note)).unwrap(), 2);
}

#[test]
fn a_run_locked_elsewhere_is_busy_and_other_runs_still_work() {
    let dir = TempDir::new().unwrap();
    let store =
        Store::open_with_lock_wait(dir.path(), std::time::Duration::from_millis(50)).unwrap();
    let (held, free) = (id("held"), id("free"));
    store.create_run(&held, config(), tasks()).unwrap();
    store.create_run(&free, config(), tasks()).unwrap();
    let bytes = fs::read(journal(dir.path(), "held")).unwrap();

    let lock = fs::File::open(dir.path().join("held")).unwrap();
    lock.lock().unwrap();
    let started = std::time::Instant::now();
    assert!(matches!(store.state(&held), Err(StoreError::Busy(_))));
    assert!(started.elapsed() >= std::time::Duration::from_millis(50));
    assert!(matches!(
        store.append(&held, vec![note(0, "x")]),
        Err(StoreError::Busy(_))
    ));
    assert!(matches!(store.log(&held), Err(StoreError::Busy(_))));
    assert!(matches!(
        store.create_run(&held, config(), tasks()),
        Err(StoreError::Busy(_))
    ));
    assert_eq!(fs::read(journal(dir.path(), "held")).unwrap(), bytes);
    store.append(&free, vec![note(0, "x")]).unwrap();
    assert_eq!(store.runs().unwrap(), vec![free, held.clone()]);

    drop(lock);
    store.append(&held, vec![note(0, "x")]).unwrap();
    assert_eq!(store.state(&held).unwrap().log_count(LogKind::Note), 1);
}

#[test]
fn a_run_directory_holds_only_the_journal() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let names: Vec<_> = fs::read_dir(dir.path().join("r1"))
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names, ["journal"]);
}

/// A file whose operations are recorded, and fail when told to.
#[derive(Default)]
struct FakeFile {
    bytes: Vec<u8>,
    calls: Vec<&'static str>,
    /// The calls that fail, counted from 0 over all operations.
    fail: Vec<usize>,
    /// The content when `sync` last returned `Ok`.
    synced: Vec<u8>,
}

impl FakeFile {
    fn step(&mut self, name: &'static str) -> std::io::Result<()> {
        self.calls.push(name);
        if self.fail.contains(&(self.calls.len() - 1)) {
            Err(std::io::Error::other(name))
        } else {
            Ok(())
        }
    }
}

impl Durable for FakeFile {
    fn truncate(&mut self, len: u64) -> std::io::Result<()> {
        self.step("truncate")?;
        self.bytes.truncate(len as usize);
        Ok(())
    }

    fn write_at(&mut self, at: u64, bytes: &[u8]) -> std::io::Result<()> {
        assert_eq!(at as usize, self.bytes.len());
        // A failed write leaves half of the bytes.
        let failed = self.step("write");
        let kept = if failed.is_ok() {
            bytes.len()
        } else {
            bytes.len() / 2
        };
        self.bytes.extend_from_slice(&bytes[..kept]);
        failed
    }

    fn sync(&mut self) -> std::io::Result<()> {
        // A failed sync leaves the bytes in the file.
        self.step("sync")?;
        self.synced = self.bytes.clone();
        Ok(())
    }
}

fn fake(fail: &[usize]) -> FakeFile {
    FakeFile {
        bytes: b"goodTORN".to_vec(),
        fail: fail.to_vec(),
        ..FakeFile::default()
    }
}

#[test]
fn an_append_is_synced_after_the_write_and_before_it_returns() {
    let mut file = fake(&[]);
    append(&mut file, 4, b"record\n").unwrap();
    assert_eq!(file.calls, ["truncate", "write", "sync"]);
    assert_eq!(file.synced, b"goodrecord\n");
}

#[test]
fn a_failed_append_takes_the_record_out_again() {
    // The first truncate, the write, and the sync fail in turn.
    for failing in 0..3 {
        let mut file = fake(&[failing]);
        let result = append(&mut file, 4, b"record\n");
        assert!(
            matches!(result, Err(AppendError::NotWritten(_))),
            "step {failing}"
        );
        assert_eq!(file.bytes, b"good", "step {failing}");
        assert_eq!(file.synced, b"good", "step {failing}");
        assert_eq!(&file.calls[file.calls.len() - 2..], ["truncate", "sync"]);
    }
}

#[test]
fn an_append_that_cannot_be_taken_out_has_an_unknown_outcome() {
    // The sync fails with the whole record written, then the truncate fails
    // or its sync does.
    for failing in [[2, 3], [2, 4]] {
        let mut file = fake(&failing);
        let result = append(&mut file, 4, b"record\n");
        assert!(
            matches!(result, Err(AppendError::Unknown(_))),
            "{failing:?}"
        );
    }
}
