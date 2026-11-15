//! The proposer's side of a run: handing out batches and taking in
//! submissions (decision D11).
//!
//! [`next_batch`] picks the task with the most open slots (see
//! [`RunState::next_task`]), builds its payload, and commits a
//! [`BatchIssued`] event that holds the payload as it is sent. A worker that
//! holds a batch whose lease has not ended gets no other until it submits.
//! One that lost the payload waits for the lease to end: the payload is not
//! sent again, because another session using the same id would get the
//! token.
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
//! are checked, so its timing does not depend on them. Two paths remain from
//! what programs do to what a proposer sees. Checks run in the background on
//! the same machine, so a long check can slow the replies to later calls.
//! And while the last programs an iteration needs are being checked,
//! [`next_batch`] says so, so a proposer that polls learns how long the
//! checks took, though not what they found.
//!
//! **Tokens.** Each batch has a random token, sent only in its payload. A
//! submission must give the batch's token and worker id; one that does not
//! is kept as foreign and not checked, so a proposer cannot take a batch it
//! was not sent, whatever worker id it gives.

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::prompt::{payload, Payload};
use crate::state::{
    BatchIssued, Check, Checked, Event, LibraryEntry, RunId, RunState, Standing, Submitted,
};
use crate::store::{Change, CheckClaim, RefusalKind, Store, StoreError};
use crate::verify::{task, Verifier, VerifierCache};

/// The longest worker id or model name, in bytes.
pub const MAX_NAME_LEN: usize = 200;
/// The longest program, in bytes.
pub const MAX_PROGRAM_LEN: usize = 64 * 1024;
/// The most programs one submission may hold.
pub const MAX_PROGRAMS: usize = crate::state::MAX_PROGRAMS_PER_BATCH as usize;

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
    let first_free = state.leased(now_ms).map(|(_, b)| b.deadline_ms).min();
    let Some(first_free) = first_free else {
        return Wait {
            reason: "the programs still needed have come in and are being checked".to_owned(),
            retry_after_seconds: 1,
        };
    };
    Wait {
        reason: "every program still needed is held by another batch".to_owned(),
        retry_after_seconds: (first_free - now_ms).div_ceil(1000).max(1),
    }
}

/// What one attempt of [`next_batch`] under the run's lock came to.
enum Attempt {
    /// The reply. The events committed with it are the commit's.
    Done(NextBatch),
    /// A batch is due, but the cache has no verifier for the run's library,
    /// which is this copy. It is built without the lock and the attempt made
    /// again.
    Build(Vec<LibraryEntry>),
}

/// Hands `worker` the next batch of `run` at `now_ms`, or says why there is
/// none. The payload is built with the verifier `verifiers` keeps for the
/// run's library.
///
/// When `verifiers` has no verifier for the run's library, the verifier is
/// built without the run's lock and the request made again under it. If the
/// library grew in the meantime, the second attempt builds the verifier
/// under the lock rather than try again.
pub fn next_batch(
    store: &Store,
    run: &RunId,
    worker: &str,
    now_ms: u64,
    verifiers: &VerifierCache,
) -> Result<NextBatch, Refused> {
    next_batch_with(store, run, worker, now_ms, verifiers, || {})
}

/// [`next_batch`], which runs `built` after it builds a verifier without the
/// lock. Tests use it to change the run in that window.
fn next_batch_with(
    store: &Store,
    run: &RunId,
    worker: &str,
    now_ms: u64,
    verifiers: &VerifierCache,
    mut built: impl FnMut(),
) -> Result<NextBatch, Refused> {
    check_name("worker id", worker)?;
    let mut built_outside = false;
    loop {
        let attempt = store.commit(run, |state| {
            issue(state, worker, now_ms, verifiers, built_outside)
        })?;
        match attempt {
            Attempt::Done(reply) => return Ok(reply),
            Attempt::Build(library) => {
                verifiers
                    .get_or_build(&library)
                    .map_err(|e| Refused::State(e.to_string()))?;
                built_outside = true;
                built();
            }
        }
    }
}

/// [`next_batch`] under the run's lock. Builds a verifier only if
/// `build_here`; otherwise asks for one to be built when `verifiers` has
/// none for the run's library.
fn issue(
    state: &RunState,
    worker: &str,
    now_ms: u64,
    verifiers: &VerifierCache,
    build_here: bool,
) -> Result<(Attempt, Vec<Event>), Refused> {
    // One live batch at a time for each worker id, so that a proposer
    // that keeps asking under one id cannot hold every slot. Ids are what
    // the caller says, so one that uses many ids still can.
    let held = state.leased(now_ms).find(|(_, b)| b.worker == worker);
    if let Some((held, b)) = held {
        let left = (b.deadline_ms - now_ms).div_ceil(1000).max(1);
        let reply = NextBatch {
            batch: None,
            wait: Some(Wait {
                reason: format!(
                    "you hold batch {held}; submit it with dc_submit, with its token, \
                     before you ask for another. If you no longer have it, it is released \
                     when its lease ends, in {left} seconds"
                ),
                retry_after_seconds: left,
            }),
        };
        return Ok((Attempt::Done(reply), Vec::new()));
    }
    let Some((name, wanted)) = state.next_task(now_ms) else {
        let reply = NextBatch {
            batch: None,
            wait: Some(wait(state, now_ms)),
        };
        return Ok((Attempt::Done(reply), Vec::new()));
    };
    let family = task(name).ok_or_else(|| Refused::State(format!("there is no task {name:?}")))?;
    // Only a verifier built for exactly the library under the lock is used.
    let verifier: Arc<Verifier> = match verifiers.get(state.library()) {
        Some(kept) => kept,
        None if build_here => verifiers
            .get_or_build(state.library())
            .map_err(|e| Refused::State(e.to_string()))?,
        None => return Ok((Attempt::Build(state.library().to_vec()), Vec::new())),
    };
    let batch = state.batch_count();
    let token = new_token()?;
    let sent = payload(state, &verifier, &*family, batch, &token, wanted);
    let text = serde_json::to_string(&sent)
        .map_err(|e| Refused::State(format!("the payload does not serialise: {e}")))?;
    let event = Event::BatchIssued(BatchIssued {
        batch,
        iteration: state.iteration(),
        task: name.to_owned(),
        wanted,
        worker: worker.to_owned(),
        token,
        issued_ms: now_ms,
        deadline_ms: now_ms.saturating_add(state.config().lease_ms),
        payload: text,
    });
    let reply = NextBatch {
        batch: Some(sent),
        wait: None,
    };
    Ok((Attempt::Done(reply), vec![event]))
}

/// What `dc_submit` returns. In the blind arm it says only what came in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Receipt {
    /// The batch.
    pub batch: u64,
    /// How many programs came in.
    pub received: usize,
}

/// A new batch token: 128 bits from the operating system, in hex.
fn new_token() -> Result<String, Refused> {
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes)
        .map_err(|e| Refused::State(format!("no random bytes for a batch token: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// Takes in `programs` for `batch` of `run` from `worker`, with the batch's
/// `token`, received at `now_ms`, with the model the proposer named, if any.
/// The submission is committed as it arrives; its programs are checked later,
/// by [`check_pending`]. So the reply does not wait for the checks, and does
/// not take longer for one program than for another.
///
/// A submission with the wrong token or worker id is committed as
/// [`Standing::Foreign`] and the reply is an error that says so. It says
/// nothing about the programs.
///
/// A submission that is not checked (foreign, late or repeat) is kept only
/// while its batch has fewer than the run's
/// [`max_unchecked_per_batch`](crate::state::RunConfig::max_unchecked_per_batch)
/// of its kind (see [`RunState::keeps`]). Past that it is counted in the
/// run's refusals, not kept, and the reply is an error that says so. So a
/// proposer that keeps submitting for a batch does not grow the journal.
#[allow(clippy::too_many_arguments)]
pub fn submit(
    store: &Store,
    run: &RunId,
    batch: u64,
    worker: &str,
    token: &str,
    programs: Vec<String>,
    model: Option<String>,
    now_ms: u64,
) -> Result<Receipt, Refused> {
    check_name("worker id", worker)?;
    check_name("token", token)?;
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
    let (standing, kept) = store.commit_or_refuse(run, |state| -> Result<_, Refused> {
        let standing = state
            .standing(batch, worker, token, now_ms)
            .ok_or_else(|| Refused::Request(format!("batch {batch} was not issued in this run")))?;
        if !state.keeps(batch, standing) {
            let kind = if standing == Standing::Foreign {
                RefusalKind::Foreign
            } else {
                RefusalKind::Repeat
            };
            let refuse = Change::Refuse {
                iteration: state.iteration(),
                kind,
            };
            return Ok(((standing, false), refuse));
        }
        let event = Event::Submitted(Submitted {
            batch,
            worker: worker.to_owned(),
            token: token.to_owned(),
            iteration: state.iteration(),
            received_ms: now_ms,
            model,
            programs,
            standing,
        });
        Ok(((standing, true), Change::Events(vec![event])))
    })?;
    match (standing, kept) {
        (Standing::Foreign, true) => Err(Refused::Request(format!(
            "batch {batch} was not sent to worker {worker:?} with that token; the submission \
             is kept but will not be checked"
        ))),
        (Standing::Foreign, false) => Err(Refused::Request(format!(
            "batch {batch} was not sent to worker {worker:?} with that token; the run keeps no \
             more such submissions for this batch, so this one is counted but not kept"
        ))),
        (_, false) => Err(Refused::Request(format!(
            "batch {batch} already has its submission, and the run keeps no more for it, so \
             this one is counted but not kept. Ask dc_next_batch for new work"
        ))),
        (_, true) => Ok(receipt),
    }
}

/// The library a pending submission is checked with.
enum Library {
    /// The verifier the cache keeps for it.
    Kept(Arc<Verifier>),
    /// A copy of it, when the cache keeps no verifier for it.
    Copied(Vec<LibraryEntry>),
}

/// A submission that waits to be checked, and the library to check it with.
struct Pending {
    batch: u64,
    task: String,
    wanted: usize,
    programs: Vec<String>,
    library_len: usize,
    library: Library,
}

/// The submission of `batch` if it waits to be checked, and the library to
/// check it with.
fn pending(state: &RunState, batch: u64, verifiers: &VerifierCache) -> Option<Pending> {
    let b = state.batch(batch)?;
    let programs = b.awaiting_check.clone()?;
    // The library is copied only when the cache has no verifier for it.
    let library = match verifiers.get(state.library()) {
        Some(kept) => Library::Kept(kept),
        None => Library::Copied(state.library().to_vec()),
    };
    Some(Pending {
        batch,
        task: b.task.clone(),
        wanted: b.wanted as usize,
        programs,
        library_len: state.library().len(),
        library,
    })
}

/// Whether `batch`'s submission was checked: it came in and waits no more.
/// Such a batch is never checked again.
fn settled(state: &RunState, batch: u64) -> bool {
    state
        .batch(batch)
        .is_some_and(|b| b.submitted && b.awaiting_check.is_none())
}

/// A claim on the first of `batches` that no other checker holds.
fn claim_first(store: &Store, run: &RunId, batches: &[u64]) -> Result<Option<CheckClaim>, Refused> {
    for &batch in batches {
        if let Some(claim) = store.try_claim_check(run, batch)? {
            return Ok(Some(claim));
        }
    }
    Ok(None)
}

/// Checks `programs` for a batch of `task` that asked for `wanted`, with a
/// clone of `verifier`, so the checks do not add to the kept one.
fn checks(
    verifier: &Verifier,
    task_name: &str,
    wanted: usize,
    programs: &[String],
) -> Result<Vec<Check>, Refused> {
    let family =
        task(task_name).ok_or_else(|| Refused::State(format!("there is no task {task_name:?}")))?;
    let mut verifier = verifier.clone();
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

/// What [`check_pending`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checks {
    /// How many submissions it checked.
    pub checked: usize,
    /// The batches whose submissions still waited at the end, each claimed
    /// by another checker, in order.
    pub claimed_elsewhere: Vec<u64>,
}

/// Checks every submission of `run` that waits to be checked, oldest first,
/// and commits a [`Checked`] event for each. The checks use the verifier
/// `verifiers` keeps for the run's library; if it keeps none, one is built
/// and kept.
///
/// Each check is claimed first (see [`Store::try_claim_check`]), so several
/// servers check different submissions and each submission once. A
/// submission another checker holds is left to it, and this call goes on to
/// the next; so it can return while submissions still wait. The claim files
/// of submissions that were checked are removed.
///
/// Checking runs the programs, so it runs without the run's lock, and so
/// does a build of the verifier. If the library changed in the meantime, the
/// commit is refused and the submission is checked again. If another server
/// checked it first, or the run was removed and made again, nothing is
/// written. An error leaves the rest waiting: a library that does not load
/// stops every check until it is put right.
pub fn check_pending(
    store: &Store,
    run: &RunId,
    verifiers: &VerifierCache,
) -> Result<Checks, Refused> {
    check_pending_with(store, run, verifiers, &mut |verifier, p| {
        checks(verifier, &p.task, p.wanted, &p.programs)
    })
}

/// How [`check_pending_with`] checks a pending submission's programs.
type CheckWith<'a> = dyn FnMut(&Verifier, &Pending) -> Result<Vec<Check>, Refused> + 'a;

/// [`check_pending`], with the programs of each pending submission checked
/// by `check`. Tests use it to see and slow the checks.
fn check_pending_with(
    store: &Store,
    run: &RunId,
    verifiers: &VerifierCache,
    check: &mut CheckWith<'_>,
) -> Result<Checks, Refused> {
    // Claim files left by checkers that stopped, listed before the run is
    // read, so that a batch read as settled was settled when it was listed.
    let left = store.check_claims(run)?;
    let mut sweep = true;
    let mut done = 0;
    let claimed_elsewhere: Vec<u64> = loop {
        let (waiting, stale) = store.read(run, |state| {
            let stale: Vec<u64> = if sweep {
                left.iter()
                    .copied()
                    .filter(|&b| settled(state, b))
                    .collect()
            } else {
                Vec::new()
            };
            (state.pending_checks(), stale)
        })?;
        if sweep {
            store.remove_settled_claims(run, &stale)?;
            sweep = false;
        }
        // Batches another checker holds are left to it.
        let Some(claim) = claim_first(store, run, &waiting)? else {
            break waiting;
        };
        // Another checker may have checked it since the run was read.
        let (p, journal) =
            store.read_journal(run, |state| pending(state, claim.batch(), verifiers))?;
        let Some(p) = p else {
            claim.release();
            continue;
        };
        let verifier = match &p.library {
            Library::Kept(kept) => Arc::clone(kept),
            Library::Copied(library) => verifiers
                .get_or_build(library)
                .map_err(|e| Refused::State(e.to_string()))?,
        };
        let checks = check(&verifier, &p)?;
        let event = Event::Checked(Checked {
            batch: p.batch,
            library_len: p.library_len as u64,
            checks,
        });
        // Written only to the journal the programs were read from: if the
        // run was removed and made again, its batch with this number is
        // another batch. And only if the batch still waits with the programs
        // that were checked.
        let wrote: Option<bool> = store.commit_to(run, journal, |state| -> Result<_, Refused> {
            let same = state
                .batch(p.batch)
                .and_then(|b| b.awaiting_check.as_ref())
                .is_some_and(|programs| *programs == p.programs);
            let still = same && state.library().len() == p.library_len;
            Ok(if still {
                (true, vec![event])
            } else {
                (false, Vec::new())
            })
        })?;
        // A refused commit leaves the batch waiting; it is claimed again.
        claim.release();
        done += usize::from(wrote == Some(true));
    };
    Ok(Checks {
        checked: done,
        claimed_elsewhere,
    })
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use std::time::Duration;

    use super::*;
    use crate::state::{Arm, RunConfig};
    use tempfile::TempDir;

    const MERGE: &str = "collapse/merge";
    const CANCEL: &str = "collapse/cancel";

    thread_local! {
        /// The verifiers of this test's run, as one server keeps them.
        static VERIFIERS: VerifierCache = VerifierCache::default();
    }

    fn next_batch(
        store: &Store,
        run: &RunId,
        worker: &str,
        now_ms: u64,
    ) -> Result<NextBatch, Refused> {
        VERIFIERS.with(|v| super::next_batch(store, run, worker, now_ms, v))
    }

    /// How many submissions [`super::check_pending`] checked.
    fn check_pending(store: &Store, run: &RunId) -> Result<usize, Refused> {
        VERIFIERS
            .with(|v| super::check_pending(store, run, v))
            .map(|c| c.checked)
    }

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
                    max_unchecked_per_batch: 3,
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

    /// The token of `batch`, or a made-up one if the run has no such batch.
    fn token(store: &Store, run: &RunId, batch: u64) -> String {
        let state = store.state(run).unwrap();
        state
            .batches()
            .get(batch as usize)
            .map_or_else(|| "none".to_owned(), |b| b.token.clone())
    }

    /// Submits with the batch's own token.
    fn sub(
        store: &Store,
        run: &RunId,
        batch: u64,
        worker: &str,
        programs: Vec<String>,
        model: Option<String>,
        now_ms: u64,
    ) -> Result<Receipt, Refused> {
        let t = token(store, run, batch);
        submit(store, run, batch, worker, &t, programs, model, now_ms)
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
        let good = sub(&store, &run, 0, "a", vec![solves_merge()], None, 10).unwrap();
        let bad = sub(&store, &run, 1, "b", vec!["let solve = (".into()], None, 10).unwrap();
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
        assert!(w.reason.contains("being checked"), "{}", w.reason);

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
        let refused = sub(&store, &run, 0, "b", vec!["not mine".into()], None, 9);
        assert!(matches!(refused, Err(Refused::Request(_))), "{refused:?}");
        sub(
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
        sub(&store, &run, 0, "a", vec!["again".into()], None, 11).unwrap();
        check_pending(&store, &run).unwrap();
        // After the iteration ends, batch 1's submission is late.
        store
            .append(
                &run,
                vec![Event::IterationFinished {
                    iteration: 0,
                    used: 0,
                }],
            )
            .unwrap();
        sub(&store, &run, 1, "b", vec!["late".into()], None, 12).unwrap();

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
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 5000).unwrap();
        check_pending(&store, &run).unwrap();
        assert_eq!(submissions(&store, &run)[0].standing, Standing::Expired);
        assert_eq!(store.state(&run).unwrap().task_counts(0, MERGE).passed, 1);
    }

    #[test]
    fn checks_use_the_library_as_it_is_when_they_run() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        sub(
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
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        assert!(matches!(
            check_pending(&store, &run),
            Err(Refused::State(_))
        ));
        assert_eq!(submissions(&store, &run).len(), 1);
        assert_eq!(store.state(&run).unwrap().pending_checks(), [0]);
    }

    #[test]
    fn a_worker_holding_a_live_batch_gets_no_other() {
        let (_dir, store, run) = setup(2, 1);
        assert_eq!(
            next_batch(&store, &run, "a", 0)
                .unwrap()
                .batch
                .unwrap()
                .batch,
            0
        );
        let w = next_batch(&store, &run, "a", 10).unwrap().wait.unwrap();
        assert!(w.reason.contains("you hold batch 0"), "{}", w.reason);
        // Its lease ends at 1000.
        assert_eq!(w.retry_after_seconds, 1);
        let w = next_batch(&store, &run, "a", 10).unwrap().wait.unwrap();
        assert!(w.reason.contains("in 1 seconds"), "{}", w.reason);
        assert_eq!(issued(&store, &run).len(), 1);
        // Another worker still gets work.
        assert!(next_batch(&store, &run, "b", 10).unwrap().batch.is_some());
        // Once the batch is submitted, or its lease ends, the worker gets more.
        sub(&store, &run, 0, "a", vec!["x".into()], None, 20).unwrap();
        assert!(next_batch(&store, &run, "a", 30).unwrap().batch.is_some());
        assert!(next_batch(&store, &run, "b", 500).unwrap().wait.is_some());
        assert!(next_batch(&store, &run, "b", 1011).unwrap().batch.is_some());
    }

    #[test]
    fn a_wait_for_checks_says_so() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        next_batch(&store, &run, "b", 0).unwrap();
        sub(&store, &run, 0, "a", vec!["x".into()], None, 1).unwrap();
        sub(&store, &run, 1, "b", vec!["y".into()], None, 1).unwrap();
        let w = next_batch(&store, &run, "c", 2).unwrap().wait.unwrap();
        assert!(w.reason.contains("being checked"), "{}", w.reason);
    }

    #[test]
    fn a_submission_without_the_batch_token_is_kept_but_takes_nothing() {
        let (_dir, store, run) = setup(1, 1);
        let sent = next_batch(&store, &run, "a", 0).unwrap().batch.unwrap();
        let real = token(&store, &run, 0);
        assert_eq!(sent.token, real);
        assert_eq!(real.len(), 32);
        assert!(real.bytes().all(|b| b.is_ascii_hexdigit()));
        // Another batch gets another token.
        next_batch(&store, &run, "b", 0).unwrap();
        assert_ne!(token(&store, &run, 1), real);

        // The right worker id with a wrong token, as a proposer that shares
        // the id could send, is refused and kept as foreign.
        for wrong in [token(&store, &run, 1), "0".repeat(32), "x".to_owned()] {
            let got = submit(&store, &run, 0, "a", &wrong, vec![], None, 1);
            let Err(Refused::Request(why)) = got else {
                panic!("{got:?}");
            };
            assert!(why.contains("token"), "{why}");
            assert!(!why.contains(&real), "{why}");
        }
        let subs = submissions(&store, &run);
        assert_eq!(subs.len(), 3);
        assert!(subs.iter().all(|s| s.standing == Standing::Foreign));
        // The batch is still open for the proposer it was sent to.
        let got = submit(&store, &run, 0, "a", &real, vec![solves_merge()], None, 2);
        assert_eq!(got.unwrap().received, 1);
        assert_eq!(submissions(&store, &run)[3].standing, Standing::OnTime);
        check_pending(&store, &run).unwrap();
        assert_eq!(store.state(&run).unwrap().task_counts(0, MERGE).passed, 1);
    }

    /// A library entry that loads: the identity.
    fn identity() -> LibraryEntry {
        LibraryEntry {
            arity: 1,
            term: "(lam $0)".into(),
        }
    }

    fn extend(store: &Store, run: &RunId) {
        store
            .append(
                run,
                vec![Event::LibraryExtended {
                    entries: vec![identity()],
                }],
            )
            .unwrap();
    }

    /// Two checkers of one run, as two proposer servers have, with their own
    /// verifiers. While one checks the oldest pending submission, the other
    /// checks the next one, without waiting, and each is checked once.
    #[test]
    fn a_submission_another_checker_is_checking_is_left_to_it() {
        use std::sync::mpsc;
        use std::sync::Mutex;

        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        next_batch(&store, &run, "b", 0).unwrap();
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        sub(
            &store,
            &run,
            1,
            "b",
            vec!["let solve xs = xs".into()],
            None,
            1,
        )
        .unwrap();

        // Each check, as (checker, batch).
        let seen: Mutex<Vec<(char, u64)>> = Mutex::new(Vec::new());
        let (entered, on_entry) = mpsc::channel::<()>();
        let (release, on_release) = mpsc::channel::<()>();
        let (store, run, seen) = (&store, &run, &seen);
        let first = std::thread::scope(|scope| {
            let first = scope.spawn(move || {
                let verifiers = VerifierCache::default();
                let mut released = true;
                let done = check_pending_with(store, run, &verifiers, &mut |v, p| {
                    seen.lock().unwrap().push(('A', p.batch));
                    if p.batch == 0 {
                        entered.send(()).unwrap();
                        // A slow check, until the other checker is done.
                        released = on_release.recv_timeout(Duration::from_secs(60)).is_ok();
                    }
                    checks(v, &p.task, p.wanted, &p.programs)
                })
                .unwrap();
                (done.checked, released)
            });
            on_entry.recv_timeout(Duration::from_secs(60)).unwrap();
            let verifiers = VerifierCache::default();
            let second = check_pending_with(store, run, &verifiers, &mut |v, p| {
                seen.lock().unwrap().push(('B', p.batch));
                checks(v, &p.task, p.wanted, &p.programs)
            })
            .unwrap();
            let by_second: Vec<u64> = seen
                .lock()
                .unwrap()
                .iter()
                .filter(|(who, _)| *who == 'B')
                .map(|&(_, batch)| batch)
                .collect();
            // Released before asserting, so a failure does not hang the test.
            release.send(()).unwrap();
            assert_eq!(by_second, [1], "the second checker checks only batch 1");
            assert_eq!(
                second,
                Checks {
                    checked: 1,
                    claimed_elsewhere: vec![0]
                }
            );
            first.join().unwrap()
        });
        assert_eq!(first, (1, true), "the first checker checks batch 0 only");
        let mut batches: Vec<u64> = seen.lock().unwrap().iter().map(|&(_, b)| b).collect();
        batches.sort_unstable();
        assert_eq!(batches, [0, 1], "each submission is checked once");
        assert!(store.state(run).unwrap().pending_checks().is_empty());
        let mut written: Vec<u64> = checked(store, run).iter().map(|c| c.batch).collect();
        written.sort_unstable();
        assert_eq!(written, [0, 1]);
    }

    #[test]
    fn a_claim_is_held_by_one_checker_and_its_file_goes_with_its_release() {
        let (_dir, store, run) = setup(1, 1);
        let held = store.try_claim_check(&run, 0).unwrap().unwrap();
        assert!(store.try_claim_check(&run, 0).unwrap().is_none());
        assert!(store.try_claim_check(&run, 1).unwrap().is_some());
        assert_eq!(store.check_claims(&run).unwrap(), [0, 1]);
        held.release();
        assert_eq!(store.check_claims(&run).unwrap(), [1]);
        // A claim that is dropped, as by a holder that stopped, can be taken.
        drop(store.try_claim_check(&run, 1).unwrap().unwrap());
        assert!(store.try_claim_check(&run, 1).unwrap().is_some());
    }

    /// The config of `setup(1, 1)`, for a run made again.
    fn config() -> RunConfig {
        RunConfig {
            arm: Arm::Blind,
            seed: 3,
            proposals_per_task: 1,
            programs_per_batch: 1,
            lease_ms: 1000,
            max_unchecked_per_batch: 3,
        }
    }

    /// While checkers each with a store of their own take and release one
    /// claim, never two hold it at once.
    #[test]
    fn a_claim_is_held_by_one_checker_at_a_time() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let (dir, _store, run) = setup(1, 1);
        let inside = AtomicUsize::new(0);
        let both = AtomicUsize::new(0);
        let taken = AtomicUsize::new(0);
        // Until 2000 claims are taken, or for a second at most.
        let until = std::time::Instant::now() + Duration::from_secs(1);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (inside, both, taken, run) = (&inside, &both, &taken, &run);
                let root = dir.path().to_path_buf();
                scope.spawn(move || {
                    let store = Store::open(&root).unwrap();
                    while taken.load(SeqCst) < 2000 && std::time::Instant::now() < until {
                        if let Some(claim) = store.try_claim_check(run, 0).unwrap() {
                            taken.fetch_add(1, SeqCst);
                            if inside.fetch_add(1, SeqCst) != 0 {
                                both.fetch_add(1, SeqCst);
                            }
                            std::thread::sleep(Duration::from_micros(200));
                            inside.fetch_sub(1, SeqCst);
                            claim.release();
                        }
                    }
                });
            }
        });
        let taken = taken.into_inner();
        assert!(taken > 100, "{taken} claims taken");
        assert_eq!(both.into_inner(), 0, "two checkers held one claim");
    }

    /// A claimant of a removed run does not remove the claim file of the run
    /// made in its place, which another checker holds.
    #[test]
    fn a_claim_of_a_removed_run_leaves_its_successor_s_claim_alone() {
        let (dir, store, run) = setup(1, 1);
        let old = store.try_claim_check(&run, 0).unwrap().unwrap();
        std::fs::remove_dir_all(dir.path().join("r")).unwrap();
        store
            .create_run(&run, config(), vec![MERGE.into(), CANCEL.into()])
            .unwrap();
        let held = store.try_claim_check(&run, 0).unwrap().unwrap();
        assert!(store.try_claim_check(&run, 0).unwrap().is_none());
        old.release();
        assert!(
            store.try_claim_check(&run, 0).unwrap().is_none(),
            "a third checker claimed batch 0 while another holds it"
        );
        assert_eq!(store.check_claims(&run).unwrap(), [0]);
        held.release();
    }

    /// A checker of a run that is removed and made again while it checks
    /// writes nothing to the new run.
    #[test]
    fn checks_of_a_removed_run_do_not_land_in_its_successor() {
        use std::sync::mpsc;
        let (dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        assert_eq!(store.state(&run).unwrap().batches()[0].task, MERGE);
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        let (entered, on_entry) = mpsc::channel::<()>();
        let (release, on_release) = mpsc::channel::<()>();
        let root = dir.path().to_path_buf();
        let (store, run) = (&store, &run);
        let first = std::thread::scope(|scope| {
            let first = scope.spawn(move || {
                let store = Store::open(&root).unwrap();
                let verifiers = VerifierCache::default();
                let mut slow = true;
                check_pending_with(&store, run, &verifiers, &mut |v, p| {
                    let checks = checks(v, &p.task, p.wanted, &p.programs);
                    // The first check, of the old run's program, is slow.
                    if std::mem::take(&mut slow) {
                        entered.send(()).unwrap();
                        on_release.recv_timeout(Duration::from_secs(60)).unwrap();
                    }
                    checks
                })
            });
            on_entry.recv_timeout(Duration::from_secs(60)).unwrap();
            std::fs::remove_dir_all(dir.path().join("r")).unwrap();
            store
                .create_run(run, config(), vec![MERGE.into(), CANCEL.into()])
                .unwrap();
            next_batch(store, run, "z", 0).unwrap();
            sub(store, run, 0, "z", vec!["let solve = 0".into()], None, 1).unwrap();
            release.send(()).unwrap();
            first.join().unwrap()
        });
        // The checker went on to check the new run's own submission.
        assert_eq!(first.unwrap().checked, 1);
        let state = store.state(run).unwrap();
        assert!(state.pending_checks().is_empty());
        assert_eq!(checked(store, run).len(), 1);
        assert!(
            state.verified().is_empty(),
            "the old run's checks were written to the new run"
        );
        assert_eq!(state.task_counts(0, MERGE).passed, 0);
    }

    #[test]
    fn a_claim_file_left_for_a_checked_submission_is_removed() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        next_batch(&store, &run, "b", 0).unwrap();
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        assert_eq!(check_pending(&store, &run).unwrap(), 1);
        assert!(store.check_claims(&run).unwrap().is_empty());
        // A holder stopped after its commit and before it removed the file.
        drop(store.try_claim_check(&run, 0).unwrap().unwrap());
        // Batch 1 has no submission: its file is not removed.
        drop(store.try_claim_check(&run, 1).unwrap().unwrap());
        assert_eq!(store.check_claims(&run).unwrap(), [0, 1]);
        assert_eq!(check_pending(&store, &run).unwrap(), 0);
        assert_eq!(store.check_claims(&run).unwrap(), [1]);
    }

    #[test]
    fn a_verifier_is_built_once_for_each_library_length() {
        let (_dir, store, run) = setup(4, 1);
        let start = crate::verify::builds();
        let built = || crate::verify::builds() - start;
        for (i, worker) in ["a", "b", "c"].into_iter().enumerate() {
            let sent = next_batch(&store, &run, worker, 0).unwrap().batch.unwrap();
            assert_eq!(sent.batch, i as u64);
        }
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        sub(
            &store,
            &run,
            1,
            "b",
            vec!["let solve xs = xs".into()],
            None,
            1,
        )
        .unwrap();
        assert_eq!(check_pending(&store, &run).unwrap(), 2);
        assert!(next_batch(&store, &run, "d", 2).unwrap().batch.is_some());
        assert_eq!(built(), 1, "one build for the empty library");

        // The library grows: the next batch and the next check use a new
        // verifier, built once.
        extend(&store, &run);
        let sent = next_batch(&store, &run, "e", 3).unwrap().batch.unwrap();
        assert!(sent.prompt.contains("fn_0"), "{}", sent.prompt);
        sub(
            &store,
            &run,
            2,
            "c",
            vec!["let solve xs = fn_0 xs".into()],
            None,
            4,
        )
        .unwrap();
        assert_eq!(check_pending(&store, &run).unwrap(), 1);
        assert!(next_batch(&store, &run, "f", 5).unwrap().batch.is_some());
        assert_eq!(built(), 2, "one more build for the library of length 1");
        let c = checked(&store, &run);
        assert_eq!(c.last().unwrap().library_len, 1);
        assert!(
            !matches!(c.last().unwrap().checks[0], Check::Compile { .. }),
            "{:?}",
            c.last().unwrap().checks
        );
    }

    #[test]
    fn a_verifier_built_for_a_library_that_grew_meanwhile_is_not_used() {
        let (_dir, store, run) = setup(1, 1);
        let verifiers = VerifierCache::default();
        let start = crate::verify::builds();
        // The library grows after the verifier for the empty library is
        // built without the lock, before the batch is issued under it.
        let mut grown = false;
        let sent = next_batch_with(&store, &run, "a", 0, &verifiers, || {
            if !grown {
                extend(&store, &run);
                grown = true;
            }
        })
        .unwrap()
        .batch
        .unwrap();
        assert!(grown);
        assert!(sent.prompt.contains("fn_0"), "{}", sent.prompt);
        assert_eq!(crate::verify::builds() - start, 2);
        // The verifier kept is the one for the grown library.
        let library = store.read(&run, |s| s.library().to_vec()).unwrap();
        assert_eq!(library.len(), 1);
        assert!(verifiers.get(&library).is_some());
        assert!(verifiers.get(&[]).is_none());
    }

    /// The length of the run's journal in bytes.
    fn journal_len(dir: &TempDir, run: &RunId) -> u64 {
        std::fs::metadata(dir.path().join(run.as_str()).join("journal"))
            .unwrap()
            .len()
    }

    /// Whether each length is more than the one before.
    fn grows(lens: &[u64]) -> bool {
        lens.windows(2).all(|w| w[0] < w[1])
    }

    /// Whether every length is the same.
    fn flat(lens: &[u64]) -> bool {
        lens.windows(2).all(|w| w[0] == w[1])
    }

    #[test]
    fn a_looping_proposer_cannot_grow_the_journal_past_the_cap() {
        let (dir, store, run) = setup(2, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        sub(&store, &run, 0, "a", vec![solves_merge()], None, 1).unwrap();
        let big = vec!["x".repeat(MAX_PROGRAM_LEN); 4];
        let cap: usize = 3;
        // Repeats from the batch's worker: the first `cap` are kept, and
        // after them the journal does not grow.
        let mut lens = vec![journal_len(&dir, &run)];
        let mut replies = Vec::new();
        for i in 0..10 {
            replies.push(sub(&store, &run, 0, "a", big.clone(), None, 2 + i));
            lens.push(journal_len(&dir, &run));
        }
        assert!(grows(&lens[..=cap]), "{lens:?}");
        assert!(flat(&lens[cap..]), "{lens:?}");
        for (i, got) in replies.iter().enumerate() {
            assert_eq!(got.is_ok(), i < cap, "repeat {i}: {got:?}");
        }
        // Foreign submissions for the same batch have a cap of their own.
        let mut lens = vec![journal_len(&dir, &run)];
        let mut foreign = Ok(Receipt {
            batch: 0,
            received: 0,
        });
        for i in 0..10 {
            foreign = submit(&store, &run, 0, "b", "bad", big.clone(), None, 20 + i);
            assert!(matches!(foreign, Err(Refused::Request(_))), "{foreign:?}");
            lens.push(journal_len(&dir, &run));
        }
        assert!(grows(&lens[..=cap]), "{lens:?}");
        assert!(flat(&lens[cap..]), "{lens:?}");
        assert_eq!(submissions(&store, &run).len(), 1 + 2 * cap);
        let state = store.state(&run).unwrap();
        let ic = state.iteration_counts(0);
        assert_eq!((ic.repeats, ic.foreign), (cap as u64, cap as u64));
        // The refused ones are counted, and the proposer is told so.
        let refused = 10 - cap as u64;
        let counts = store.refusals(&run).unwrap().iteration(0);
        assert_eq!((counts.repeats, counts.foreign), (refused, refused));
        let Err(Refused::Request(why)) = &replies[cap] else {
            panic!("{:?}", replies[cap]);
        };
        assert!(why.contains("counted but not kept"), "{why}");
        let Err(Refused::Request(why)) = &foreign else {
            panic!("{foreign:?}");
        };
        assert!(why.contains("counted but not kept"), "{why}");
        // The count is on disk: a server started afresh reads the same run
        // and the same count.
        let len = journal_len(&dir, &run);
        let restarted = Store::open(dir.path()).unwrap();
        assert_eq!(restarted.state(&run).unwrap(), state);
        assert_eq!(restarted.refusals(&run).unwrap().iteration(0), counts);
        assert_eq!(restarted.refusals(&run).unwrap().total(), counts);
        assert_eq!(journal_len(&dir, &run), len);
        // The cap is for each batch: another batch's submission is kept and
        // checked as before.
        next_batch(&store, &run, "a", 40).unwrap();
        let got = sub(&store, &run, 1, "a", vec![solves_merge()], None, 41);
        assert_eq!(got.unwrap().received, 1);
        let last = submissions(&store, &run).pop().unwrap();
        assert_eq!((last.batch, last.standing), (1, Standing::OnTime));
    }

    #[test]
    fn bad_requests_are_refused_and_write_nothing() {
        let (_dir, store, run) = setup(1, 1);
        next_batch(&store, &run, "a", 0).unwrap();
        let before = events(&store, &run);
        let long = "x".repeat(MAX_PROGRAM_LEN + 1);
        let cases: Vec<Result<Receipt, Refused>> = vec![
            sub(&store, &run, 9, "a", vec![], None, 1),
            sub(&store, &run, 0, "a", vec![long], None, 1),
            sub(
                &store,
                &run,
                0,
                "a",
                vec![String::new(); MAX_PROGRAMS + 1],
                None,
                1,
            ),
            sub(&store, &run, 0, "a", vec![], Some(String::new()), 1),
            sub(
                &store,
                &run,
                0,
                "a",
                vec![],
                Some("m".repeat(MAX_NAME_LEN + 1)),
                1,
            ),
            sub(&store, &run, 0, "", vec![], None, 1),
            submit(&store, &run, 0, "a", "", vec![], None, 1),
            submit(
                &store,
                &run,
                0,
                "a",
                &"t".repeat(MAX_NAME_LEN + 1),
                vec![],
                None,
                1,
            ),
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
