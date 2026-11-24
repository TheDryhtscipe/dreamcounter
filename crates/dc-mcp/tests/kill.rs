//! A process that is killed while it commits to a run in a loop loses no
//! commit it was told is written, and the run goes on from there.
//!
//! The test runs this test binary again as a child. The child commits to a run
//! in a loop and prints a line after each commit returns. The parent kills it
//! with SIGKILL and reads the run.
//!
//! A kill can tear a write: Linux checks for a fatal signal between the folios
//! of a write, and a record here is longer than a page, so on a file system
//! that writes it in more than one folio a kill can tear it. The next child
//! then sets the torn tail aside and logs a [`LogKind::Recovery`]. So that
//! this is tested on every run, not only when a kill happens to land in a
//! write, the test leaves a torn tail of its own before one round.
//!
//! What this does not show: that a commit survives a power cut, which needs
//! the `fsync` (see `store_tests.rs` for the order of the write and the sync).

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use dc_mcp::state::{Arm, Event, FrontierMember, LogKind, LogRecord, RunConfig, RunId, RunState};
use dc_mcp::store::{Store, StoreError};

const ROOT_ENV: &str = "DC_MCP_KILL_TEST_ROOT";
const RUN: &str = "killed";
/// The child ends an iteration with every fifth commit.
const COMMITS_PER_ITERATION: u64 = 5;
/// Before this round the test leaves a torn tail in the journal.
const TORN_ROUND: usize = 3;
/// The torn tail the test leaves: the start of a record.
const TORN_TAIL: &[u8] = b"9999 00000000 half a rec";

fn config() -> RunConfig {
    RunConfig {
        max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
        autodoc: false,
        arm: Arm::Blind,
        library_learning: true,
        seed: 3,
        proposals_per_task: 1,
        programs_per_batch: 1,
        lease_ms: 60_000,
        max_unchecked_per_batch: 3,
        frontier_size: 5,
        stitch: dc_mcp::state::StitchSettings::default(),
        prelude: 0,
        compared_with: None,
        approach_hints: false,
        show_received: false,
        refuse_bucketing: false,
        lambda: dc_driver::Lambda::ZERO,
        seed_entries: Vec::new(),
    }
}

fn tasks() -> Vec<String> {
    vec!["t0".into(), "t1".into()]
}

/// The events of the child's commit number `i`, counting from 0.
fn batch(i: u64) -> Vec<Event> {
    let iteration = i / COMMITS_PER_ITERATION;
    let mut events = vec![Event::Logged(LogRecord {
        kind: LogKind::Note,
        iteration,
        task: None,
        // Long enough that a write is not trivially small.
        text: format!("commit {i} {}", "x".repeat(4096)),
    })];
    if i % COMMITS_PER_ITERATION == COMMITS_PER_ITERATION - 1 {
        events.push(Event::FrontierSet {
            task: format!("t{}", iteration % 2),
            members: vec![FrontierMember {
                program: format!("(program {i})"),
                cost: None,
            }],
        });
        events.push(Event::IterationFinished { iteration, used: 0 });
    }
    events
}

/// The state the run should have, given its log: the child's commits in order,
/// with each [`LogKind::Recovery`] record where the store wrote it, first in
/// the commit after a torn tail. Each note in the log must be the next
/// commit's. The recovery records are taken as the store wrote them.
fn expected(log: &[LogRecord]) -> RunState {
    let mut state = RunState::create(config(), tasks()).unwrap();
    let mut commits = 0;
    for record in log {
        match record.kind {
            LogKind::Recovery => state.apply(&Event::Logged(record.clone())).unwrap(),
            LogKind::Note => {
                let events = batch(commits);
                assert_eq!(events[0], Event::Logged(record.clone()), "note {commits}");
                for event in &events {
                    state.apply(event).unwrap();
                }
                commits += 1;
            }
            other => panic!("the child logs no {other:?}"),
        }
    }
    state
}

/// The child: commits for ever, from wherever the run is. It is not a test of
/// its own, so it does nothing unless the parent set the variable.
#[test]
fn writer_child() {
    let Some(root) = std::env::var_os(ROOT_ENV) else {
        return;
    };
    let store = Store::open(Path::new(&root)).unwrap();
    let run = RunId::new(RUN).unwrap();
    let mut out = std::io::stdout();
    loop {
        let done: Result<u64, StoreError> = store.commit(&run, |state| {
            let done = state.log_count(LogKind::Note);
            Ok((done, batch(done)))
        });
        let done = done.unwrap();
        writeln!(out, "\nack {done}").unwrap();
        out.flush().unwrap();
    }
}

#[test]
fn a_killed_writer_loses_no_acknowledged_commit() {
    let dir = tempfile::TempDir::new().unwrap();
    let run = RunId::new(RUN).unwrap();
    Store::open(dir.path())
        .unwrap()
        .create_run(&run, config(), tasks())
        .unwrap();

    let mut total_acked = 0;
    // Each round kills the child after a different number of commits, so the
    // kills fall at different points of an iteration.
    for (round, acks_before_kill) in [1, 3, 7, 12, 2, 9].into_iter().enumerate() {
        if round == TORN_ROUND {
            // What a kill in the middle of a write leaves: part of a record.
            let journal = dir.path().join(RUN).join("journal");
            let mut file = OpenOptions::new().append(true).open(journal).unwrap();
            file.write_all(TORN_TAIL).unwrap();
        }
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "writer_child", "--nocapture", "--test-threads=1"])
            .env(ROOT_ENV, dir.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let mut last_ack = None;
        let mut seen = 0;
        while seen < acks_before_kill {
            let line = lines.next().expect("the child stopped by itself").unwrap();
            if let Some(n) = line.strip_prefix("ack ") {
                last_ack = Some(n.parse::<u64>().unwrap());
                seen += 1;
            }
        }
        child.kill().unwrap();
        child.wait().unwrap();
        let last_ack = last_ack.unwrap();
        // The child went on from where the last one was killed.
        assert!(last_ack >= total_acked, "{last_ack} < {total_acked}");
        total_acked = last_ack + 1;

        let store = Store::open(dir.path()).unwrap();
        let state = store.state(&run).unwrap();
        let commits = state.log_count(LogKind::Note);
        assert!(
            commits >= total_acked,
            "{total_acked} commits were acknowledged and {commits} are in the run"
        );
        let log = store.log(&run).unwrap();
        assert_eq!(state, expected(&log));
        // Each recovery set aside a torn tail in a file of its own. A kill in
        // the middle of a write can tear one, so there may be more than the
        // one the test made, but each child recovers at most one, first.
        let recoveries: Vec<&LogRecord> = log
            .iter()
            .filter(|record| record.kind == LogKind::Recovery)
            .collect();
        let torn_files = std::fs::read_dir(dir.path().join(RUN))
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with("torn-")
            })
            .count();
        assert_eq!(recoveries.len(), torn_files, "{recoveries:?}");
        if round >= TORN_ROUND {
            // A natural tear just before merges with the test's tail, so the
            // set-aside tail can be longer, but it ends with it.
            let made = recoveries.iter().any(|record| {
                let aside = record.text.rsplit(' ').next().unwrap();
                std::fs::read(dir.path().join(RUN).join(aside))
                    .unwrap()
                    .ends_with(TORN_TAIL)
            });
            assert!(
                made,
                "the test's torn tail was not recovered: {recoveries:?}"
            );
        }
        assert!(recoveries.len() <= round, "{recoveries:?}");
    }
    assert!(total_acked >= 34);
    assert!(
        Store::open(dir.path())
            .unwrap()
            .state(&run)
            .unwrap()
            .iteration()
            >= 6
    );
}
