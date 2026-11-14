//! Reading a run that a store has not read before holds about one record of
//! its journal in memory at a time, not the whole journal.
//!
//! The test writes a journal of many large records and then runs this test
//! binary again as a child, which opens a new store and reads the run once.
//! The child reports how much its resident memory grew, from its resident set
//! just before the read to its peak (`VmRSS` and `VmHWM` in
//! `/proc/self/status`). A child is used so that what the parent allocated to
//! write the journal does not count.
//!
//! The bound is a few records. A store that reads the whole journal, or keeps
//! every record's events until it has read them all, grows by the journal's
//! size or more.

use std::ops::ControlFlow;
use std::path::Path;
use std::process::Command;

use dc_mcp::state::{Arm, Event, LogKind, LogRecord, RunConfig, RunId};
use dc_mcp::store::Store;

const ROOT_ENV: &str = "DC_MCP_MEMORY_TEST_ROOT";
const MODE_ENV: &str = "DC_MCP_MEMORY_TEST_MODE";
const RUN: &str = "big";

/// The bound on how much memory grows, in records.
const BOUND_RECORDS: u64 = 8;

/// A field of `/proc/self/status`, in kibibytes.
fn status_kib(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status
        .lines()
        .find_map(|line| line.strip_prefix(field))
        .unwrap_or_else(|| panic!("no {field} in /proc/self/status"));
    line.trim().trim_end_matches("kB").trim().parse().unwrap()
}

/// The child: reads the run once, from a new store, and prints what it read
/// and how much its memory grew. It is not a test of its own, so it does
/// nothing unless the parent set the variable.
#[test]
fn cold_reader_child() {
    let Some(root) = std::env::var_os(ROOT_ENV) else {
        return;
    };
    let mode = std::env::var(MODE_ENV).unwrap();
    let store = Store::open(Path::new(&root)).unwrap();
    let run = RunId::new(RUN).unwrap();
    let before = status_kib("VmRSS:");
    let started = std::time::Instant::now();
    let (records, bytes) = match mode.as_str() {
        "state" => {
            let notes = store
                .read(&run, |state| state.log_count(LogKind::Note))
                .unwrap();
            (notes, 0)
        }
        "log" => {
            let (mut records, mut bytes) = (0_u64, 0_u64);
            store
                .for_each_log(&run, |record| {
                    records += 1;
                    bytes += record.text.len() as u64;
                    ControlFlow::Continue(())
                })
                .unwrap();
            (records, bytes)
        }
        other => panic!("unknown mode {other}"),
    };
    let millis = started.elapsed().as_millis();
    let grew = status_kib("VmHWM:").saturating_sub(before);
    println!("\nresult {records} {bytes} {grew} {millis}");
}

/// Runs the child in `mode` on the store at `root`: the records and bytes it
/// read, how much its memory grew in kibibytes, and how long it took in
/// milliseconds.
fn cold_read(root: &Path, mode: &str) -> (u64, u64, u64, u64) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "cold_reader_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROOT_ENV, root)
        .env(MODE_ENV, mode)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("result "))
        .unwrap_or_else(|| panic!("no result from the child: {stdout}"));
    let fields: Vec<u64> = line.split(' ').map(|n| n.parse().unwrap()).collect();
    (fields[0], fields[1], fields[2], fields[3])
}

/// Writes a journal of `notes` notes of `note_bytes` bytes each, reads its
/// state and its log from new stores, and checks that memory grew by at most
/// [`BOUND_RECORDS`] notes each time.
fn check_cold_reads(notes: u64, note_bytes: usize) {
    let dir = tempfile::TempDir::new().unwrap();
    let run = RunId::new(RUN).unwrap();
    let store = Store::open(dir.path()).unwrap();
    let config = RunConfig {
        arm: Arm::Blind,
        seed: 5,
        proposals_per_task: 1,
    };
    store.create_run(&run, config, vec!["t".into()]).unwrap();
    for i in 0..notes {
        let mut text = format!("note {i} ");
        text.push_str(&"x".repeat(note_bytes - text.len()));
        let note = Event::Logged(LogRecord {
            kind: LogKind::Note,
            iteration: 0,
            task: None,
            text,
        });
        store.append(&run, vec![note]).unwrap();
    }
    let journal = std::fs::metadata(dir.path().join(RUN).join("journal"))
        .unwrap()
        .len();
    assert!(journal >= notes * note_bytes as u64);
    drop(store);

    // The line being read, the events parsed from it, and the allocator's
    // slack.
    let bound_kib = BOUND_RECORDS * note_bytes as u64 / 1024;
    let journal_kib = journal / 1024;

    let (read, _, grew, millis) = cold_read(dir.path(), "state");
    println!("state: {journal_kib} KiB journal, memory grew {grew} KiB, {millis} ms");
    assert_eq!(read, notes);
    assert!(
        grew <= bound_kib,
        "reading the state of a {journal_kib} KiB journal grew memory by {grew} KiB; the bound \
         is {bound_kib} KiB"
    );

    let (records, bytes, grew, millis) = cold_read(dir.path(), "log");
    println!("log: {journal_kib} KiB journal, memory grew {grew} KiB, {millis} ms");
    assert_eq!(records, notes);
    assert_eq!(bytes, notes * note_bytes as u64);
    assert!(
        grew <= bound_kib,
        "reading the log of a {journal_kib} KiB journal grew memory by {grew} KiB; the bound is \
         {bound_kib} KiB"
    );
}

/// 64 notes of 1 MiB: the bound is an eighth of the journal.
#[test]
fn a_cold_read_holds_a_few_records_not_the_journal() {
    check_cold_reads(64, 1 << 20);
}

/// The size the bead measured: 50 notes of 4 MiB, a 200 MiB journal. It
/// writes 200 MiB to the temporary directory, so it runs only when asked:
/// `cargo test --release -p dc-mcp --test memory -- --ignored --nocapture`.
#[test]
#[ignore = "writes a 200 MiB journal"]
fn a_cold_read_of_a_200_mib_journal_holds_a_few_records() {
    check_cold_reads(50, 4 << 20);
}
