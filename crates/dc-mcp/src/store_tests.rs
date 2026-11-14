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
    fs::write(&path, text.replacen("journal 1 ", "journal 2 ", 1)).unwrap();
    match Store::open(dir.path()).unwrap().state(&run) {
        Err(StoreError::UnsupportedFormat { header, .. }) => {
            assert!(header.starts_with("dc-mcp journal 2 "), "{header}");
        }
        other => panic!("{other:?}"),
    }
    fs::write(&path, "").unwrap();
    assert!(matches!(
        Store::open(dir.path()).unwrap().state(&run),
        Err(StoreError::Corrupt { .. })
    ));
    fs::write(&path, crate::journal::header(5)).unwrap();
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

/// A journal file whose operations fail when told to. See `Store::faults`.
pub(crate) struct Faulty<'a> {
    file: &'a mut fs::File,
    calls: usize,
    fail: Vec<usize>,
}

impl<'a> Faulty<'a> {
    pub(crate) fn new(file: &'a mut fs::File, fail: Vec<usize>) -> Faulty<'a> {
        Faulty {
            file,
            calls: 0,
            fail,
        }
    }

    fn step(&mut self) -> std::io::Result<()> {
        self.calls += 1;
        if self.fail.contains(&(self.calls - 1)) {
            Err(std::io::Error::other("injected"))
        } else {
            Ok(())
        }
    }
}

impl Durable for Faulty<'_> {
    fn truncate(&mut self, len: u64) -> std::io::Result<()> {
        self.step()?;
        self.file.truncate(len)
    }

    fn write_at(&mut self, at: u64, bytes: &[u8]) -> std::io::Result<()> {
        self.step()?;
        self.file.write_at(at, bytes)
    }

    fn sync(&mut self) -> std::io::Result<()> {
        // A failed sync leaves the bytes in the file.
        self.step()?;
        self.file.sync()
    }
}

#[test]
fn a_commit_whose_write_fails_is_not_in_the_run() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let state = store.state(&run).unwrap();
    let bytes = fs::read(journal(dir.path(), "r1")).unwrap();

    // The steps of an append are truncate, write, sync. Each fails in turn.
    for step in 0..3 {
        *store.faults.lock().unwrap() = vec![step];
        let failed = store.append(&run, vec![note(1, "lost")]);
        assert!(
            matches!(failed, Err(StoreError::Io { .. })),
            "step {step}: {failed:?}"
        );
        assert_eq!(
            fs::read(journal(dir.path(), "r1")).unwrap(),
            bytes,
            "step {step}"
        );
        assert_eq!(store.state(&run).unwrap(), state, "step {step}");
        assert_eq!(Store::open(dir.path()).unwrap().state(&run).unwrap(), state);
    }
    store.append(&run, vec![note(1, "kept")]).unwrap();
    let texts: Vec<String> = store
        .log(&run)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    assert_eq!(texts, ["first", "payload text", "kept"]);
}

#[test]
fn a_commit_with_an_unknown_outcome_is_read_from_the_file_afterwards() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    // The sync fails with the record written, and so does the truncate that
    // would take it out. The record stays in the file.
    *store.faults.lock().unwrap() = vec![2, 3];
    let failed = store.append(&run, vec![note(1, "maybe")]);
    assert!(
        matches!(failed, Err(StoreError::OutcomeUnknown { .. })),
        "{failed:?}"
    );
    // The store does not guess: it says what the file says, as a new store does.
    let fresh = Store::open(dir.path()).unwrap().state(&run).unwrap();
    assert_eq!(fresh.log_count(LogKind::Note), 2);
    assert_eq!(store.state(&run).unwrap(), fresh);
    store.append(&run, vec![note(1, "next")]).unwrap();
    assert_eq!(
        Store::open(dir.path())
            .unwrap()
            .state(&run)
            .unwrap()
            .log_count(LogKind::Note),
        3
    );
}

#[test]
fn commit_then_sees_the_state_after_the_events_with_the_run_still_locked() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let other =
        Store::open_with_lock_wait(dir.path(), std::time::Duration::from_millis(20)).unwrap();
    let run = id("r1");
    mid_iteration(&store, &run);
    let seen: Result<(u64, u64, bool), StoreError> = store.commit_then(
        &run,
        |state| Ok((state.log_count(LogKind::Note), vec![note(1, "x")])),
        |before, state| {
            let busy = matches!(other.state(&run), Err(StoreError::Busy(_)));
            (before, state.log_count(LogKind::Note), busy)
        },
    );
    assert_eq!(seen.unwrap(), (1, 2, true));
    // With no events, `then` sees the state as it is.
    let seen: Result<u64, StoreError> = store.commit_then(
        &run,
        |_| Ok(((), vec![])),
        |(), state| state.log_count(LogKind::Note),
    );
    assert_eq!(seen.unwrap(), 2);
}

#[test]
fn a_run_removed_and_made_again_is_read_from_its_start() {
    // The second run's journal is longer, the same length, or shorter.
    for (seed, notes) in [(999, 2), (12, 1), (12, 0)] {
        let dir = TempDir::new().unwrap();
        let run = id("r1");
        let a = Store::open(dir.path()).unwrap();
        a.create_run(&run, config(), tasks()).unwrap();
        a.append(&run, vec![note(0, "from the first run")]).unwrap();

        fs::remove_dir_all(dir.path().join("r1")).unwrap();
        let b = Store::open(dir.path()).unwrap();
        let second = RunConfig { seed, ..config() };
        b.create_run(&run, second.clone(), tasks()).unwrap();
        for i in 0..notes {
            b.append(&run, vec![note(0, &format!("second run {i}"))])
                .unwrap();
        }
        let bytes = fs::read(journal(dir.path(), "r1")).unwrap();

        // The first store sees the new run, not the one it remembers.
        let seen = a.state(&run).unwrap();
        assert_eq!(seen.config(), &second, "seed {seed}");
        assert_eq!(seen.log_count(LogKind::Note), notes, "seed {seed}");
        a.append(&run, vec![note(0, "from a")]).unwrap();
        let after = fs::read(journal(dir.path(), "r1")).unwrap();
        assert_eq!(&after[..bytes.len()], &bytes[..], "seed {seed}");
        assert_eq!(fs::read_dir(dir.path().join("r1")).unwrap().count(), 1);
        let fresh = Store::open(dir.path()).unwrap();
        assert_eq!(
            fresh.state(&run).unwrap().log_count(LogKind::Note),
            notes + 1
        );
        assert_eq!(fresh.state(&run).unwrap().log_count(LogKind::Recovery), 0);
        assert_eq!(
            b.state(&run).unwrap(),
            fresh.state(&run).unwrap(),
            "seed {seed}"
        );
    }
}

#[test]
fn a_last_record_that_changed_under_a_store_is_an_error_once() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let store = Store::open(dir.path()).unwrap();
    store.create_run(&run, config(), tasks()).unwrap();
    let path = journal(dir.path(), "r1");
    let start = fs::read(&path).unwrap();
    store.append(&run, vec![note(0, "aaaa")]).unwrap();
    // Another record of the same length in its place, and one after it.
    let mut swapped = start.clone();
    swapped.extend_from_slice(crate::journal::encode(1, &[note(0, "bbbb")]).as_bytes());
    swapped.extend_from_slice(crate::journal::encode(2, &[note(0, "cccc")]).as_bytes());
    fs::write(&path, &swapped).unwrap();

    assert!(matches!(store.state(&run), Err(StoreError::Corrupt { .. })));
    assert_eq!(fs::read(&path).unwrap(), swapped);
    let texts: Vec<String> = store
        .log(&run)
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    assert_eq!(texts, ["bbbb", "cccc"]);
    assert_eq!(store.state(&run).unwrap().log_count(LogKind::Note), 2);
}

#[test]
fn a_set_aside_tail_is_reported_first_in_the_commit_that_replaces_it() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    mid_iteration(&Store::open(dir.path()).unwrap(), &run);
    let path = journal(dir.path(), "r1");
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.len();
    bytes.extend_from_slice(b"7 half a rec");
    fs::write(&path, &bytes).unwrap();

    let store = Store::open(dir.path()).unwrap();
    assert_eq!(store.state(&run).unwrap().log_count(LogKind::Recovery), 0);
    // The commit ends the iteration. The report is still in the iteration
    // the run was in.
    let seen: Result<u64, StoreError> = store.commit_then(
        &run,
        |_| {
            Ok((
                (),
                vec![Event::IterationFinished { iteration: 1 }, note(2, "after")],
            ))
        },
        |(), state| state.log_count(LogKind::Recovery),
    );
    assert_eq!(seen.unwrap(), 1);
    let log = Store::open(dir.path()).unwrap().log(&run).unwrap();
    let report = &log[log.len() - 2];
    assert_eq!((report.kind, report.iteration), (LogKind::Recovery, 1));
    assert!(
        report.text.contains(&format!("torn-{at}-0")),
        "{}",
        report.text
    );
    assert!(report.text.starts_with("12 bytes"), "{}", report.text);
    assert_eq!(log[log.len() - 1].text, "after");
    // One record holds the report and the commit's events.
    let lines = fs::read(&path).unwrap()[at..]
        .iter()
        .filter(|b| **b == b'\n')
        .count();
    assert_eq!(lines, 1);
    // A later commit has nothing to report.
    store.append(&run, vec![note(2, "more")]).unwrap();
    assert_eq!(store.state(&run).unwrap().log_count(LogKind::Recovery), 1);
}

#[test]
fn the_longest_lock_wait_does_not_overflow() {
    let dir = TempDir::new().unwrap();
    let store = Store::open_with_lock_wait(dir.path(), std::time::Duration::MAX).unwrap();
    let run = id("r1");
    store.create_run(&run, config(), tasks()).unwrap();
    assert_eq!(store.state(&run).unwrap().iteration(), 0);
}

#[test]
fn a_lock_on_a_removed_directory_does_not_license_a_write_to_its_successor() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    let store = Store::open_with_lock_wait(dir.path(), std::time::Duration::from_secs(2)).unwrap();
    store.create_run(&run, config(), tasks()).unwrap();
    let old_lock = fs::File::open(dir.path().join("r1")).unwrap();
    old_lock.lock().unwrap();

    std::thread::scope(|scope| {
        // This waits for the lock on the directory that is about to go.
        let waiting = scope.spawn(|| store.append(&run, vec![note(0, "late")]));
        std::thread::sleep(std::time::Duration::from_millis(200));
        fs::remove_dir_all(dir.path().join("r1")).unwrap();
        Store::open(dir.path())
            .unwrap()
            .create_run(&run, config(), tasks())
            .unwrap();
        let bytes = fs::read(journal(dir.path(), "r1")).unwrap();
        // Another server holds the new run. The old lock is released.
        let new_lock = fs::File::open(dir.path().join("r1")).unwrap();
        new_lock.lock().unwrap();
        drop(old_lock);
        let result = waiting.join().unwrap();
        assert!(matches!(result, Err(StoreError::Replaced(_))), "{result:?}");
        assert_eq!(fs::read(journal(dir.path(), "r1")).unwrap(), bytes);
    });
}

#[test]
fn create_does_not_write_to_the_successor_of_a_removed_directory() {
    // The call must be waiting for the lock when the directory goes. If the
    // machine is busy it may not be there yet. It then finds the new
    // directory locked, which shows nothing, so the test waits longer and
    // tries again.
    for pause in [200, 800, 3200] {
        let dir = TempDir::new().unwrap();
        let run = id("r1");
        let store =
            Store::open_with_lock_wait(dir.path(), std::time::Duration::from_secs(2)).unwrap();
        fs::create_dir(dir.path().join("r1")).unwrap();
        let old_lock = fs::File::open(dir.path().join("r1")).unwrap();
        old_lock.lock().unwrap();

        let result = std::thread::scope(|scope| {
            let waiting = scope.spawn(|| store.create_run(&run, config(), tasks()));
            std::thread::sleep(std::time::Duration::from_millis(pause));
            fs::remove_dir(dir.path().join("r1")).unwrap();
            fs::create_dir(dir.path().join("r1")).unwrap();
            // Another server holds the new directory. The old lock is released.
            let new_lock = fs::File::open(dir.path().join("r1")).unwrap();
            new_lock.lock().unwrap();
            drop(old_lock);
            waiting.join().unwrap()
        });
        assert_eq!(fs::read_dir(dir.path().join("r1")).unwrap().count(), 0);
        if matches!(result, Err(StoreError::Busy(_))) {
            continue;
        }
        assert!(matches!(result, Err(StoreError::Replaced(_))), "{result:?}");
        return;
    }
    panic!("the call never waited on the directory that was removed");
}

/// The names and contents of the files set aside in a run's directory.
fn set_aside_files(root: &Path, run: &str) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = fs::read_dir(root.join(run))
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("torn-"))
        .map(|entry| {
            (
                entry.file_name().to_string_lossy().into_owned(),
                fs::read(entry.path()).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn a_failed_commit_over_a_torn_tail_puts_the_tail_back_and_a_later_one_reports_it() {
    let dir = TempDir::new().unwrap();
    let run = id("r1");
    mid_iteration(&Store::open(dir.path()).unwrap(), &run);
    let path = journal(dir.path(), "r1");
    let mut bytes = fs::read(&path).unwrap();
    let at = bytes.len();
    bytes.extend_from_slice(b"7 half a rec");
    fs::write(&path, &bytes).unwrap();

    let store = Store::open(dir.path()).unwrap();
    // The steps of an append are truncate, write, sync. Each fails in turn.
    for step in 0..3 {
        *store.faults.lock().unwrap() = vec![step];
        let failed = store.append(&run, vec![note(1, "lost")]);
        assert!(
            matches!(failed, Err(StoreError::Io { .. })),
            "step {step}: {failed:?}"
        );
        assert_eq!(fs::read(&path).unwrap(), bytes, "step {step}");
    }
    // Three failed commits set the same bytes aside once.
    let aside = vec![(format!("torn-{at}-0"), b"7 half a rec".to_vec())];
    assert_eq!(set_aside_files(dir.path(), "r1"), aside);

    store.append(&run, vec![note(1, "kept")]).unwrap();
    assert_eq!(store.state(&run).unwrap().log_count(LogKind::Recovery), 1);
    assert_eq!(set_aside_files(dir.path(), "r1"), aside);
    let fresh = Store::open(dir.path()).unwrap().state(&run).unwrap();
    assert_eq!(fresh.log_count(LogKind::Recovery), 1);
}

#[test]
fn a_file_that_could_not_be_filled_is_not_left_behind() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("torn-1-0");
    let failed = write_new(&path, |file| {
        file.write_all(b"part")?;
        Err(std::io::Error::other("injected"))
    });
    assert!(failed.is_err());
    assert!(!path.exists());

    write_new(&path, |file| file.write_all(b"whole")).unwrap();
    // A file that is there is not written over, and not removed.
    let again = write_new(&path, |file| file.write_all(b"other"));
    assert_eq!(again.unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
    assert_eq!(fs::read(&path).unwrap(), b"whole");
}

#[test]
fn a_tail_is_put_back_whole_or_not_at_all() {
    let after_failed_append = |fail: &[usize]| FakeFile {
        bytes: b"good".to_vec(),
        fail: fail.to_vec(),
        ..FakeFile::default()
    };
    let mut file = after_failed_append(&[]);
    put_back(&mut file, 4, b"TORN");
    assert_eq!(file.calls, ["write", "sync"]);
    assert_eq!(file.synced, b"goodTORN");

    // The write fails with half of the tail written, or the sync fails.
    for failing in 0..2 {
        let mut file = after_failed_append(&[failing]);
        put_back(&mut file, 4, b"TORN");
        assert_eq!(file.bytes, b"good", "step {failing}");
        assert_eq!(file.synced, b"good", "step {failing}");
    }
}

#[test]
fn an_entry_that_is_not_a_plain_file_does_not_stop_a_tail_being_set_aside() {
    let kinds = [
        "directory",
        "dangling link",
        "link to the same bytes",
        "pipe",
        "longer file",
    ];
    for kind in kinds {
        let dir = TempDir::new().unwrap();
        let run = id("r1");
        mid_iteration(&Store::open(dir.path()).unwrap(), &run);
        let path = journal(dir.path(), "r1");
        let mut bytes = fs::read(&path).unwrap();
        let at = bytes.len();
        bytes.extend_from_slice(b"7 half a rec");
        fs::write(&path, &bytes).unwrap();
        let taken = dir.path().join("r1").join(format!("torn-{at}-0"));
        match kind {
            "directory" => fs::create_dir(&taken).unwrap(),
            "dangling link" => std::os::unix::fs::symlink("nothing-here", &taken).unwrap(),
            "link to the same bytes" => {
                // The link's own length is the tail's, and so is its target's.
                fs::write(dir.path().join("r1").join("same-bytes-x"), b"7 half a rec").unwrap();
                std::os::unix::fs::symlink("same-bytes-x", &taken).unwrap();
            }
            "pipe" => {
                let made = std::process::Command::new("mkfifo").arg(&taken).status();
                assert!(made.unwrap().success());
            }
            _ => fs::write(&taken, b"7 half a record of another time").unwrap(),
        }

        let store = Store::open(dir.path()).unwrap();
        store.append(&run, vec![note(1, "kept")]).unwrap();
        assert_eq!(
            store.state(&run).unwrap().log_count(LogKind::Recovery),
            1,
            "{kind}"
        );
        let next = dir.path().join("r1").join(format!("torn-{at}-1"));
        assert_eq!(fs::read(next).unwrap(), b"7 half a rec", "{kind}");
        // What was there is still there.
        assert!(fs::symlink_metadata(&taken).is_ok(), "{kind}");
    }
}
