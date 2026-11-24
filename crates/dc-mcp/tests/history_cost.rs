//! What the proposer's calls cost on a run with a long history.
//!
//! The test builds runs with 10, 1 000 and 10 000 finished iterations. Each
//! finished iteration has four batches: three submitted and checked, one
//! never submitted. In the current iteration every slot is leased. It then
//! times the calls a proposer and the operator make all the time, on each
//! run, and prints the mean time of each call.
//!
//! It is ignored, because it takes a while and its numbers depend on the
//! machine. Run it with
//! `cargo test --release -p dc-mcp --test history_cost -- --ignored --nocapture`.
//! The bound that a call touches only the current iteration's batches is
//! checked by the unit tests in `state.rs`; this test shows the time.

use std::path::PathBuf;
use std::time::Instant;

use dc_mcp::proposer::next_batch;
use dc_mcp::state::{
    Arm, BatchIssued, Check, Checked, Event, IssuedPayload, LogKind, LogRecord, RunConfig, RunId,
    Standing, Submitted,
};
use dc_mcp::store::Store;
use dc_mcp::verify::VerifierCache;
use tempfile::TempDir;

const MERGE: &str = "collapse/merge";
const CANCEL: &str = "collapse/cancel";
/// Iterations written in one commit while the run is built.
const ITERATIONS_PER_COMMIT: u64 = 100;
/// Calls timed for each kind of call.
const CALLS: u32 = 200;
/// The time of the calls, in milliseconds since the Unix epoch.
const NOW_MS: u64 = 1_000_000;

fn issued(batch: u64, iteration: u64, task: &str, deadline_ms: u64) -> Event {
    Event::BatchIssued(BatchIssued {
        batch,
        iteration,
        task: task.to_owned(),
        wanted: 2,
        worker: format!("w{}", batch % 4),
        token: format!("t{batch}"),
        issued_ms: 0,
        deadline_ms,
        payload: IssuedPayload::Whole("payload".to_owned()),
    })
}

/// The events of finished iteration `iteration`, whose batches start at
/// `first`.
fn finished_iteration(iteration: u64, first: u64) -> Vec<Event> {
    let mut events = Vec::new();
    for (k, task) in [MERGE, MERGE, CANCEL, CANCEL].into_iter().enumerate() {
        events.push(issued(first + k as u64, iteration, task, 1000));
    }
    for batch in first..first + 3 {
        events.push(Event::Submitted(Submitted {
            batch,
            worker: format!("w{}", batch % 4),
            token: format!("t{batch}"),
            iteration,
            received_ms: 10,
            model: None,
            programs: vec!["let solve x = x".to_owned(); 2],
            standing: Standing::OnTime,
        }));
        events.push(Event::Checked(Checked {
            batch,
            library_len: 0,
            checks: vec![
                Check::Compile {
                    error: "e".to_owned()
                };
                2
            ],
        }));
    }
    events.push(Event::IterationFinished { iteration, used: 0 });
    events
}

/// A store on a file system without a disk if there is one, so the time of
/// a sync does not hide the time of the work.
fn temp_dir() -> TempDir {
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        TempDir::new_in(shm).unwrap()
    } else {
        TempDir::new().unwrap()
    }
}

/// A run with `past` finished iterations, and every slot of the current one
/// leased until long after [`NOW_MS`].
fn build(past: u64) -> (TempDir, Store, RunId) {
    let dir = temp_dir();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("long").unwrap();
    store
        .create_run(
            &run,
            RunConfig {
                max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
                autodoc: false,
                arm: Arm::Blind,
                library_learning: true,
                seed: 1,
                proposals_per_task: 4,
                programs_per_batch: 2,
                lease_ms: 1000,
                max_unchecked_per_batch: 3,
                frontier_size: 5,
                stitch: dc_mcp::state::StitchSettings::default(),
                prelude: dc_mcp::verify::shipped_prelude(),
                compared_with: None,
                approach_hints: false,
                show_received: false,
                refuse_bucketing: false,
                lambda: dc_driver::Lambda::ZERO,
                seed_entries: Vec::new(),
            },
            vec![MERGE.to_owned(), CANCEL.to_owned()],
        )
        .unwrap();
    let mut iteration = 0;
    while iteration < past {
        let end = (iteration + ITERATIONS_PER_COMMIT).min(past);
        let events: Vec<Event> = (iteration..end)
            .flat_map(|i| finished_iteration(i, i * 4))
            .collect();
        store.append(&run, events).unwrap();
        iteration = end;
    }
    let first = past * 4;
    let current: Vec<Event> = [MERGE, MERGE, CANCEL, CANCEL]
        .into_iter()
        .enumerate()
        .map(|(k, task)| issued(first + k as u64, past, task, u64::MAX / 2))
        .collect();
    store.append(&run, current).unwrap();
    (dir, store, run)
}

/// The mean time of `call`, in microseconds.
fn mean_us(mut call: impl FnMut()) -> f64 {
    call();
    let started = Instant::now();
    for _ in 0..CALLS {
        call();
    }
    started.elapsed().as_secs_f64() * 1e6 / f64::from(CALLS)
}

#[test]
#[ignore = "a timing run: see the module docs"]
fn hot_calls_on_long_runs() {
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12}",
        "past iter", "held us", "wait us", "read us", "commit us"
    );
    for past in [10_u64, 1_000, 10_000] {
        let (_dir, store, run) = build(past);
        let verifiers = VerifierCache::default();
        let held = mean_us(|| {
            let reply = next_batch(&store, &run, "w0", NOW_MS, &verifiers).unwrap();
            assert!(reply.batch.is_none());
        });
        let wait = mean_us(|| {
            let reply = next_batch(&store, &run, "someone else", NOW_MS, &verifiers).unwrap();
            assert!(reply.wait.is_some());
        });
        let read = mean_us(|| {
            let seen = store
                .read(&run, |s| {
                    let it = s.iteration();
                    (
                        s.pending_checks().len(),
                        s.quota_in(),
                        s.next_task(NOW_MS).is_some(),
                        s.task_counts(it, MERGE).batches + s.task_counts(it, CANCEL).batches,
                    )
                })
                .unwrap();
            assert_eq!(seen, (0, false, false, 4));
        });
        let commit = mean_us(|| {
            store
                .append(
                    &run,
                    vec![Event::Logged(LogRecord {
                        kind: LogKind::Note,
                        iteration: past,
                        task: None,
                        text: "n".to_owned(),
                    })],
                )
                .unwrap();
        });
        println!("{past:>10} {held:>12.1} {wait:>12.1} {read:>12.1} {commit:>12.1}");
    }
}
