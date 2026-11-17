//! A claim on a check, held by a process that is killed, does not keep the
//! submission from being checked.
//!
//! The test runs this test binary again as a child. The child claims the
//! check of the run's pending submission, prints a line, and waits. While it
//! holds the claim, the parent's checks leave the submission alone. The
//! parent then kills the child with SIGKILL and checks the submission.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use dc_mcp::proposer::{check_pending, next_batch, submit};
use dc_mcp::state::{Arm, RunConfig, RunId};
use dc_mcp::store::Store;
use dc_mcp::verify::VerifierCache;

const ROOT_ENV: &str = "DC_MCP_CLAIM_TEST_ROOT";
const RUN: &str = "claimed";

/// The child: claims the check of batch 0 and holds it until it is killed.
/// It is not a test of its own, so it does nothing unless the parent set the
/// variable.
#[test]
fn claimant_child() {
    let Some(root) = std::env::var_os(ROOT_ENV) else {
        return;
    };
    let store = Store::open(Path::new(&root)).unwrap();
    let run = RunId::new(RUN).unwrap();
    let claim = store.try_claim_check(&run, 0).unwrap().unwrap();
    let mut out = std::io::stdout();
    writeln!(out, "claimed {}", claim.batch()).unwrap();
    out.flush().unwrap();
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}

#[test]
fn a_claim_held_by_a_killed_process_does_not_strand_its_submission() {
    let dir = tempfile::TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new(RUN).unwrap();
    store
        .create_run(
            &run,
            RunConfig {
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
                prelude: dc_mcp::verify::shipped_prelude(),
            },
            vec!["collapse/merge".into()],
        )
        .unwrap();
    let verifiers = VerifierCache::default();
    let sent = next_batch(&store, &run, "a", 0, &verifiers)
        .unwrap()
        .batch
        .unwrap();
    assert_eq!(sent.batch, 0);
    let token = store.state(&run).unwrap().batches()[0].token.clone();
    submit(
        &store,
        &run,
        0,
        "a",
        &token,
        vec!["let solve xs = xs".into()],
        None,
        1,
    )
    .unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "claimant_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ROOT_ENV, dir.path())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    loop {
        let line = lines.next().expect("the child stopped by itself").unwrap();
        // The test harness may put the test's name first on the line.
        if line.ends_with("claimed 0") {
            break;
        }
    }

    // While the child holds the claim, the submission is left to it.
    let left = check_pending(&store, &run, &verifiers);
    let pending = store.state(&run).unwrap().pending_checks();
    child.kill().unwrap();
    child.wait().unwrap();
    let left = left.unwrap();
    assert_eq!(left.checked, 0, "a claimed submission was checked");
    assert_eq!(left.claimed_elsewhere, [0]);
    assert_eq!(pending, [0]);

    // The claim died with the child; its file is still there.
    assert_eq!(store.check_claims(&run).unwrap(), [0]);
    let done = check_pending(&store, &run, &verifiers).unwrap();
    assert_eq!((done.checked, done.claimed_elsewhere.len()), (1, 0));
    assert!(store.state(&run).unwrap().pending_checks().is_empty());
    assert!(
        store.check_claims(&run).unwrap().is_empty(),
        "the claim file is removed once the submission is checked"
    );
}
