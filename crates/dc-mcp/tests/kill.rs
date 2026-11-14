//! A process that is killed while it commits to a run in a loop loses no
//! commit it was told is written, and the run goes on from there.
//!
//! The test runs this test binary again as a child. The child commits to a run
//! in a loop and prints a line after each commit returns. The parent kills it
//! with SIGKILL and reads the run.
//!
//! What this does not show: that a commit survives a power cut, which needs
//! the `fsync` (see `store_tests.rs` for the order of the write and the sync),
//! or what a torn tail does (SIGKILL does not tear a write in practice; see
//! the torn-tail tests in `store_tests.rs`).

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use dc_mcp::state::{Arm, Event, FrontierMember, LogKind, LogRecord, RunConfig, RunId, RunState};
use dc_mcp::store::{Store, StoreError};

const ROOT_ENV: &str = "DC_MCP_KILL_TEST_ROOT";
const RUN: &str = "killed";
/// The child ends an iteration with every fifth commit.
const COMMITS_PER_ITERATION: u64 = 5;

fn config() -> RunConfig {
    RunConfig {
        arm: Arm::Blind,
        seed: 3,
        proposals_per_task: 1,
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
            }],
        });
        events.push(Event::IterationFinished { iteration });
    }
    events
}

/// The state after the child's first `commits` commits.
fn expected(commits: u64) -> RunState {
    let mut state = RunState::create(config(), tasks()).unwrap();
    for event in (0..commits).flat_map(batch) {
        state.apply(&event).unwrap();
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
    for acks_before_kill in [1, 3, 7, 12, 2, 9] {
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
        assert_eq!(state, expected(commits));
        assert_eq!(store.log(&run).unwrap().len() as u64, commits);
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
