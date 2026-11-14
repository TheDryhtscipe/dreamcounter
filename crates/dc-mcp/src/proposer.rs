//! The proposer's side of a run: handing out batches and taking in
//! submissions (decision D11).
//!
//! [`next_batch`] picks the task with the most open slots (see
//! [`RunState::next_task`]), builds its payload, and commits a
//! [`BatchIssued`] event that holds the payload as it is sent.
//!
//! [`submit`] commits every submission for a batch the run has issued, as a
//! [`Submitted`] event with the programs as they were sent, as soon as it
//! arrives. So the first submission for a batch is the first to arrive. The
//! submission's programs count towards the quota at once.
//!
//! [`check_pending`] then checks the programs of each submission that is
//! the first for its batch from the batch's worker and in the batch's
//! iteration (see [`crate::verify`]), and commits a [`Checked`] event with the
//! results. An iteration cannot end while a check is pending.
//!
//! **The blind arm.** The reply to a submission is the batch and how many
//! programs came in, whatever the programs are. It comes before the programs
//! are checked, so its timing does not depend on them. Checks run in the
//! background on the same machine, so a long check can slow the replies to
//! later calls; that is the one path from what programs do to how fast a
//! reply comes.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::prompt::{payload, Payload};
use crate::state::{BatchIssued, Check, Checked, Event, LibraryEntry, RunId, RunState, Submitted};
use crate::store::{Store, StoreError};
use crate::verify::{task, Verifier};

/// The longest worker id or model name, in bytes.
pub const MAX_NAME_LEN: usize = 200;
/// The longest program, in bytes.
pub const MAX_PROGRAM_LEN: usize = 64 * 1024;
/// The most programs one submission may hold.
pub const MAX_PROGRAMS: usize = 32;

/// The time, for leases.
pub trait Clock: Send + Sync + fmt::Debug {
    /// Milliseconds since the Unix epoch.
    fn now_ms(&self) -> u64;
}

/// The system's clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

/// Why a proposer's request was refused. Nothing was written.
#[derive(Debug)]
pub enum Refused {
    /// The store failed.
    Store(StoreError),
    /// The request is not one the server takes.
    Request(String),
    /// The run's state is wrong, for example its library does not load.
    State(String),
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Store(e) => write!(f, "{e}"),
            Refused::Request(why) => f.write_str(why),
            Refused::State(why) => write!(f, "the run cannot take work: {why}"),
        }
    }
}

impl std::error::Error for Refused {}

impl From<StoreError> for Refused {
    fn from(e: StoreError) -> Refused {
        Refused::Store(e)
    }
}

/// Why there is no batch now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Wait {
    /// In words.
    pub reason: String,
    /// When to ask again, in seconds.
    pub retry_after_seconds: u64,
}

/// What `dc_next_batch` returns: a batch, or why there is none now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NextBatch {
    /// The batch, if there is one.
    pub batch: Option<Payload>,
    /// Why there is no batch, if there is none.
    pub wait: Option<Wait>,
}

/// How long to wait when every task has its programs.
const QUOTA_IN_RETRY_SECONDS: u64 = 60;

fn check_name(what: &str, name: &str) -> Result<(), Refused> {
    if name.is_empty() || name.len() > MAX_NAME_LEN {
        return Err(Refused::Request(format!(
            "the {what} must have 1 to {MAX_NAME_LEN} bytes"
        )));
    }
    Ok(())
}

/// Why no batch can be issued at `now_ms`.
fn wait(state: &RunState, now_ms: u64) -> Wait {
    if state.quota_in() {
        return Wait {
            reason: "every task has its programs for this iteration; the next iteration has \
                     not started"
                .to_owned(),
            retry_after_seconds: QUOTA_IN_RETRY_SECONDS,
        };
    }
    let first_free = state
        .batches()
        .iter()
        .filter(|b| b.iteration == state.iteration() && !b.submitted && now_ms <= b.deadline_ms)
        .map(|b| b.deadline_ms)
        .min()
        .unwrap_or(now_ms);
    Wait {
        reason: "every program still needed is held by another batch".to_owned(),
        retry_after_seconds: (first_free - now_ms).div_ceil(1000).max(1),
    }
}

/// Hands `worker` the next batch of `run` at `now_ms`, or says why there is
/// none.
pub fn next_batch(
    store: &Store,
    run: &RunId,
    worker: &str,
    now_ms: u64,
) -> Result<NextBatch, Refused> {
    check_name("worker id", worker)?;
    store.commit(run, |state| {
        let Some((name, wanted)) = state.next_task(now_ms) else {
            let reply = NextBatch {
                batch: None,
                wait: Some(wait(state, now_ms)),
            };
            return Ok((reply, Vec::new()));
        };
        let family =
            task(name).ok_or_else(|| Refused::State(format!("there is no task {name:?}")))?;
        let verifier = Verifier::new(state.library()).map_err(|e| Refused::State(e.to_string()))?;
        let batch = state.batches().len() as u64;
        let sent = payload(state, &verifier, &*family, batch, wanted);
        let text = serde_json::to_string(&sent)
            .map_err(|e| Refused::State(format!("the payload does not serialise: {e}")))?;
        let event = Event::BatchIssued(BatchIssued {
            batch,
            iteration: state.iteration(),
            task: name.to_owned(),
            wanted,
            worker: worker.to_owned(),
            issued_ms: now_ms,
            deadline_ms: now_ms.saturating_add(state.config().lease_ms),
            payload: text,
        });
        let reply = NextBatch {
            batch: Some(sent),
            wait: None,
        };
        Ok((reply, vec![event]))
    })
}

/// What `dc_submit` returns. In the blind arm it says only what came in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Receipt {
    /// The batch.
    pub batch: u64,
    /// How many programs came in.
    pub received: usize,
}

/// Takes in `programs` for `batch` of `run` from `worker`, received at
/// `now_ms`, with the model the proposer named, if any. The submission is
/// committed as it arrives; its programs are checked later, by
/// [`check_pending`]. So the reply does not wait for the checks, and does not
/// take longer for one program than for another.
pub fn submit(
    store: &Store,
    run: &RunId,
    batch: u64,
    worker: &str,
    programs: Vec<String>,
    model: Option<String>,
    now_ms: u64,
) -> Result<Receipt, Refused> {
    check_name("worker id", worker)?;
    if let Some(model) = &model {
        check_name("model name", model)?;
    }
    if programs.len() > MAX_PROGRAMS {
        return Err(Refused::Request(format!(
            "a submission holds at most {MAX_PROGRAMS} programs"
        )));
    }
    if programs.iter().any(|p| p.len() > MAX_PROGRAM_LEN) {
        return Err(Refused::Request(format!(
            "a program has at most {MAX_PROGRAM_LEN} bytes"
        )));
    }
    let receipt = Receipt {
        batch,
        received: programs.len(),
    };
    store.commit(run, |state| {
        let standing = state
            .standing(batch, worker, now_ms)
            .ok_or_else(|| Refused::Request(format!("batch {batch} was not issued in this run")))?;
        let event = Event::Submitted(Submitted {
            batch,
            worker: worker.to_owned(),
            iteration: state.iteration(),
            received_ms: now_ms,
            model,
            programs,
            standing,
        });
        Ok((receipt, vec![event]))
    })
}

/// A submission that waits to be checked, and the library to check it with.
struct Pending {
    batch: u64,
    task: String,
    wanted: usize,
    programs: Vec<String>,
    library: Vec<LibraryEntry>,
}

fn first_pending(state: &RunState) -> Option<Pending> {
    let batch = *state.pending_checks().first()?;
    let b = &state.batches()[batch as usize];
    Some(Pending {
        batch,
        task: b.task.clone(),
        wanted: b.wanted as usize,
        programs: b.awaiting_check.clone().unwrap_or_default(),
        library: state.library().to_vec(),
    })
}

/// Checks `programs` for a batch of `task` that asked for `wanted`.
fn checks(
    library: &[LibraryEntry],
    task_name: &str,
    wanted: usize,
    programs: &[String],
) -> Result<Vec<Check>, Refused> {
    let family =
        task(task_name).ok_or_else(|| Refused::State(format!("there is no task {task_name:?}")))?;
    let mut verifier = Verifier::new(library).map_err(|e| Refused::State(e.to_string()))?;
    Ok(programs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i < wanted {
                verifier.check(&*family, p)
            } else {
                Check::Extra
            }
        })
        .collect())
}

/// Checks every submission of `run` that waits to be checked, oldest first,
/// and commits a [`Checked`] event for each. Returns how many it checked.
///
/// Checking runs the programs, so it runs without the run's lock, on a copy.
/// If the library changed in the meantime, the commit is refused and the
/// submission is checked again. If another server checked it first, nothing
/// is written. An error leaves the rest waiting: a library that does not load
/// stops every check until it is put right.
pub fn check_pending(store: &Store, run: &RunId) -> Result<usize, Refused> {
    let mut done = 0;
    while let Some(p) = store.read(run, first_pending)? {
        let checks = checks(&p.library, &p.task, p.wanted, &p.programs)?;
        let event = Event::Checked(Checked {
            batch: p.batch,
            library_len: p.library.len() as u64,
            checks,
        });
        let wrote: bool = store.commit(run, |state| -> Result<_, Refused> {
            let still = state.pending_checks().contains(&p.batch)
                && state.library().len() == p.library.len();
            Ok(if still {
                (true, vec![event])
            } else {
                (false, Vec::new())
            })
        })?;
        done += usize::from(wrote);
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use super::*;
    use crate::state::{Arm, RunConfig, Standing};
    use tempfile::TempDir;

    const MERGE: &str = "collapse/merge";
    const CANCEL: &str = "collapse/cancel";

    fn setup(quota: u32, per_batch: u32) -> (TempDir, Store, RunId) {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let run = RunId::new("r").unwrap();
        store
            .create_run(
                &run,
                RunConfig {
                    arm: Arm::Blind,
                    seed: 3,
                    proposals_per_task: quota,
                    programs_per_batch: per_batch,
                    lease_ms: 1000,
                },
                vec![MERGE.into(), CANCEL.into()],
            )
            .unwrap();
        (dir, store, run)
    }

    /// A program that solves collapse/merge: its good reference.
    fn solves_merge() -> String {
        let v = dc_tasks::families::development()
            .into_iter()
            .find(|v| v.family.name() == MERGE)
            .unwrap();
        format!(
            "{}\nlet solve = {}\n",
            v.source,
            v.references.last().unwrap().entry
        )
    }

    fn events(store: &Store, run: &RunId) -> Vec<Event> {
        let mut all = Vec::new();
        store
            .for_each_event(run, |e| {
                all.push(e);
                ControlFlow::Continue(())
            })
            .unwrap();
        all
    }

    fn issued(store: &Store, run: &RunId) -> Vec<BatchIssued> {
        events(store, run)
            .into_iter()
            .filter_map(|e| match e {
                Event::BatchIssued(b) => Some(b),
                _ => None,
            })
            .collect()
    }

    fn submissions(store: &Store, run: &RunId) -> Vec<Submitted> {
        events(store, run)
            .into_iter()
            .filter_map(|e| match e {
                Event::Submitted(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    fn checked(store: &Store, run: &RunId) -> Vec<Checked> {
        events(store, run)
            .into_iter()
            .filter_map(|e| match e {
                Event::Checked(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_batch_is_kept_with_its_payload_as_sent() {
        let (_dir, store, run) = setup(1, 1);
        let got = next_batch(&store, &run, "agent-1", 5000).unwrap();
        let payload = got.batch.unwrap();
        assert_eq!(payload.batch, 0);
        let kept = issued(&store, &run);
        assert_eq!(kept.len(), 1);
        let b = &kept[0];
        assert_eq!((b.task.as_str(), b.worker.as_str()), (MERGE, "agent-1"));
        assert_eq!((b.issued_ms, b.deadline_ms), (5000, 6000));
        assert_eq!(crate::prompt::read_payload(b).unwrap(), payload);
    }

    #[test]
    fn work_waits_while_every_slot_is_leased_and_comes_back_when_a_lease_ends() {
        let (_dir, store, run) = setup(1, 1);
        assert!(next_batch(&store, &run, "a", 0).unwrap().batch.is_some());
        assert!(next_batch(&store, &run, "b", 0).unwrap().batch.is_some());
        let w = next_batch(&store, &run, "c", 400).unwrap();
        assert!(w.batch.is_none());
        let wait = w.wait.unwrap();
        assert_eq!(wait.retry_after_seconds, 1);
        assert!(wait.reason.contains("held"));
        // Nothing is written for a wait.
        assert_eq!(issued(&store, &run).len(), 2);

        // Batch 0's lease ends at 1000: its task is handed out again.
        let again = next_batch(&store, &run, "c", 1001).unwrap().batch.unwrap();
        assert_eq!(again.batch, 2);
        assert_eq!(issued(&store, &run)[2].task, MERGE);
    }

    #[test]
    fn a_submission_is_kept_on_arrival_and_checked_afterwards() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        next_batch(&store, &run, "b", 0).unwrap();
        let good = submit(&store, &run, 0, "a", vec![solves_merge()], None, 10).unwrap();
        let bad = submit(&store, &run, 1, "b", vec!["let solve = (".into()], None, 10).unwrap();
        assert_eq!(
            good,
            Receipt {
                batch: 0,
                received: 1
            }
        );
        assert_eq!(
            bad,
            Receipt {
                batch: 1,
                received: 1
            }
        );
        // Nothing is checked yet, and the iteration is not done.
        let state = store.state(&run).unwrap();
        assert_eq!(state.pending_checks(), [0, 1]);
        assert_eq!(state.task_counts(0, MERGE).received, 1);
        assert_eq!(state.task_counts(0, MERGE).programs, 0);
        assert!(!state.quota_in());
        assert!(checked(&store, &run).is_empty());
        let w = next_batch(&store, &run, "c", 20).unwrap().wait.unwrap();
        assert!(w.reason.contains("held"), "{}", w.reason);

        assert_eq!(check_pending(&store, &run).unwrap(), 2);
        assert_eq!(check_pending(&store, &run).unwrap(), 0);
        let state = store.state(&run).unwrap();
        assert!(state.pending_checks().is_empty());
        assert_eq!(state.task_counts(0, MERGE).passed, 1);
        assert_eq!(state.task_counts(0, CANCEL).compiled, 0);
        assert_eq!(state.verified().len(), 1);
        assert_eq!(state.verified()[0].task, MERGE);
        assert!(state.quota_in());
        let w = next_batch(&store, &run, "c", 20).unwrap().wait.unwrap();
        assert!(w.reason.contains("every task has its programs"));
    }

    #[test]
    fn every_submission_is_kept_as_it_was_sent() {
        let (_dir, store, run) = setup(2, 2);
        next_batch(&store, &run, "a", 0).unwrap();
        next_batch(&store, &run, "b", 0).unwrap();
        let programs = vec![
            "let solve xs = xs".to_owned(),
            solves_merge(),
            "-- an extra one\nlet solve xs = []".to_owned(),
        ];
        // Another worker's submission for batch 0 is kept but does not take
        // the batch.
        submit(&store, &run, 0, "b", vec!["not mine".into()], None, 9).unwrap();
        submit(
            &store,
            &run,
            0,
            "a",
            programs.clone(),
            Some("model-x".into()),
            10,
        )
        .unwrap();
        // The first submission to arrive takes the batch; a second is a
        // repeat, whether or not the first has been checked.
        submit(&store, &run, 0, "a", vec!["again".into()], None, 11).unwrap();
        check_pending(&store, &run).unwrap();
        // After the iteration ends, batch 1's submission is late.
        store
            .append(&run, vec![Event::IterationFinished { iteration: 0 }])
            .unwrap();
        submit(&store, &run, 1, "b", vec!["late".into()], None, 12).unwrap();

        let subs = submissions(&store, &run);
        assert_eq!(subs.len(), 4);
        assert_eq!(
            (subs[0].standing, subs[0].worker.as_str()),
            (Standing::Foreign, "b")
        );
        assert_eq!(subs[1].programs, programs);
        assert_eq!(subs[1].model.as_deref(), Some("model-x"));
        assert_eq!(subs[1].standing, Standing::OnTime);
        assert_eq!(subs[2].standing, Standing::Repeat);
        assert_eq!(subs[2].programs, ["again"]);
        assert_eq!((subs[3].standing, subs[3].iteration), (Standing::Late, 1));

        let checks = checked(&store, &run);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].batch, 0);
        assert!(matches!(checks[0].checks[0], Check::Run { .. }));
        assert!(matches!(checks[0].checks[1], Check::Passed { .. }));
        assert_eq!(checks[0].checks[2], Check::Extra);

        let state = store.state(&run).unwrap();
        let merge = state.task_counts(0, MERGE);
        assert_eq!((merge.programs, merge.passed), (2, 1));
        let ic = state.iteration_counts(0);
        assert_eq!((ic.submissions, ic.repeats, ic.foreign), (3, 1, 1));
        assert_eq!(state.iteration_counts(1).late, 1);
        assert_eq!(state.task_counts(1, CANCEL).programs, 0);
    }

    #[test]
    fn a_submission_after_its_lease_is_still_checked_and_counted() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        submit(&store, &run, 0, "a", vec![solves_merge()], None, 5000).unwrap();
        check_pending(&store, &run).unwrap();
        assert_eq!(submissions(&store, &run)[0].standing, Standing::Expired);
        assert_eq!(store.state(&run).unwrap().task_counts(0, MERGE).passed, 1);
    }

    #[test]
    fn checks_use_the_library_as_it_is_when_they_run() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        submit(
            &store,
            &run,
            0,
            "a",
            vec!["let solve xs = fn_0 xs".into()],
            None,
            1,
        )
        .unwrap();
        store
            .append(
                &run,
                vec![Event::LibraryExtended {
                    entries: vec![LibraryEntry {
                        arity: 1,
                        term: "(lam $0)".into(),
                    }],
                }],
            )
            .unwrap();
        check_pending(&store, &run).unwrap();
        let c = &checked(&store, &run)[0];
        assert_eq!(c.library_len, 1);
        assert!(
            !matches!(c.checks[0], Check::Compile { .. }),
            "{:?}",
            c.checks
        );
    }

    #[test]
    fn a_library_that_does_not_load_leaves_the_submissions_kept_and_waiting() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        store
            .append(
                &run,
                vec![Event::LibraryExtended {
                    entries: vec![LibraryEntry {
                        arity: 1,
                        term: "(lam (add $0 true))".into(),
                    }],
                }],
            )
            .unwrap();
        submit(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        assert!(matches!(
            check_pending(&store, &run),
            Err(Refused::State(_))
        ));
        assert_eq!(submissions(&store, &run).len(), 1);
        assert_eq!(store.state(&run).unwrap().pending_checks(), [0]);
    }

    #[test]
    fn bad_requests_are_refused_and_write_nothing() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        let before = events(&store, &run);
        let long = "x".repeat(MAX_PROGRAM_LEN + 1);
        let cases: Vec<Result<Receipt, Refused>> = vec![
            submit(&store, &run, 9, "a", vec![], None, 1),
            submit(&store, &run, 0, "a", vec![long], None, 1),
            submit(
                &store,
                &run,
                0,
                "a",
                vec![String::new(); MAX_PROGRAMS + 1],
                None,
                1,
            ),
            submit(&store, &run, 0, "a", vec![], Some(String::new()), 1),
            submit(
                &store,
                &run,
                0,
                "a",
                vec![],
                Some("m".repeat(MAX_NAME_LEN + 1)),
                1,
            ),
            submit(&store, &run, 0, "", vec![], None, 1),
        ];
        for case in cases {
            assert!(matches!(case, Err(Refused::Request(_))), "{case:?}");
        }
        for worker in ["", &"w".repeat(MAX_NAME_LEN + 1)] {
            assert!(matches!(
                next_batch(&store, &run, worker, 1),
                Err(Refused::Request(_))
            ));
        }
        assert_eq!(events(&store, &run), before);
    }
}
