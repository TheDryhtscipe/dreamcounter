//! Ending an iteration: the step of `dc_driver` run on a run's state, and its
//! result committed (decisions D18 and D19).
//!
//! [`finish_iteration`] does three things:
//!
//! 1. **Read.** Under the run's lock it copies what the step needs: the
//!    library, each task's frontier, and the verified programs no step has
//!    taken. It refuses unless every task has its programs for the iteration
//!    and all of them are checked ([`RunState::quota_in`]), and unless the
//!    run's programs were compiled with this build's prelude
//!    ([`crate::verify::check_prelude`]).
//! 2. **Step.** Without the lock, it runs [`dc_driver::run_iteration`] with
//!    the Stitch runner it is given. This can take as long as the run's
//!    Stitch limits allow. Proposers' servers go on meanwhile: a submission
//!    for a batch whose lease ended can still come in and be checked.
//! 3. **Commit.** In one record, so that a restart finds all of it or none:
//!    a `FrontierSet` for each task, a `LibraryExtended` if abstractions
//!    join the library, a log record of kind [`LogKind::Compression`] whose
//!    text is the step's [`IterationReport`] as JSON, and an
//!    `IterationFinished` that takes the verified programs the step read.
//!    Programs verified during the step stay for the next one.
//!
//! # More than one server
//!
//! The commit is written only if the run is still in the iteration the step
//! read, in the journal it read. If another server ended the iteration
//! first, the result is [`FinishError::AlreadyFinished`] and nothing is
//! written: the other server's outcome stands. The two steps may have read
//! different programs, if one was checked between the two reads. Nothing
//! stops two servers from running the step at the same time; the second to
//! commit has run Stitch for nothing.
//!
//! An iteration cannot end while a submission waits to be checked. If one
//! came in during the step, this server checks it
//! ([`crate::proposer::check_pending`]) and tries the commit again, without
//! running the step again: the step's outcome depends only on what it read.
//! It checks up to [`COMMIT_TRIES`] times, and tries the commit after each.
//!
//! # A step that is interrupted
//!
//! If the Stitch run is cancelled, or its worker cannot be started or read,
//! the result is [`FinishError::Interrupted`] and nothing is written. Call
//! again. A Stitch run that ends at its time or memory limit is not an
//! interruption: the iteration ends without new abstractions, and the log
//! record says why (D19, item 6).
//!
//! The cancel flag of [`finish_iteration_in_worker`] is read while Stitch
//! runs, and under the run's lock at each attempt to commit. A flag set
//! after the attempt that commits has read it does not stop the commit. It
//! is not read while the step measures programs before Stitch.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use dc_compress::worker::{Limits, Worker};
use dc_compress::CompressConfig;
use dc_driver::{
    run_iteration, Candidate, DriverConfig, DriverError, Isolated, IterationInput, IterationReport,
    Stitch, TaskSpec,
};
use dc_surface::prelude::Prelude;

use crate::proposer::check_pending;
use crate::state::{
    Event, FrontierMember, LibraryEntry, LogKind, LogRecord, RunId, RunState, StitchSettings,
};
use crate::store::{Store, StoreError};
use crate::verify::{check_prelude, shipped_prelude, task, PreludeChanged, VerifierCache};

/// How many times [`finish_iteration`] checks submissions that came in
/// during the step. It tries its commit before the first check and after
/// each.
pub const COMMIT_TRIES: usize = 8;

/// How long [`finish_iteration`] waits before it tries its commit again when
/// another server holds the check that is pending.
const CHECK_WAIT: Duration = Duration::from_millis(250);

/// What [`finish_iteration`] did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Finished {
    /// The iteration that ended.
    pub iteration: u64,
    /// How many verified programs the step took.
    pub used: u64,
    /// How many abstractions joined the library.
    pub new_abstractions: usize,
    /// What happened to each program, as it is in the run's log.
    pub report: IterationReport,
}

/// Why [`finish_iteration`] did not end the iteration. Nothing was written,
/// unless the store's error says the outcome is not known.
#[derive(Debug)]
pub enum FinishError {
    /// The store failed.
    Store(StoreError),
    /// Some task has not received all its programs for the iteration, or a
    /// submission waits to be checked.
    QuotaNotIn {
        /// The iteration.
        iteration: u64,
        /// The tasks that have not received all their programs.
        short: Vec<String>,
        /// How many submissions wait to be checked.
        pending_checks: usize,
    },
    /// The run's programs were compiled with another prelude.
    Prelude(PreludeChanged),
    /// The run has a task this build does not know.
    UnknownTask(String),
    /// The step refused the run's state: a library entry or a frontier
    /// member is wrong.
    State(DriverError),
    /// The Stitch run was cancelled, or its worker could not be started or
    /// read. Call again.
    Interrupted(String),
    /// Another server ended the iteration first, or the run was removed and
    /// made again.
    AlreadyFinished {
        /// The iteration the step read.
        iteration: u64,
    },
    /// Submissions that came in during the step still wait after
    /// [`COMMIT_TRIES`] checks, or checking them failed.
    ChecksPending(String),
}

impl fmt::Display for FinishError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FinishError::Store(e) => write!(f, "{e}"),
            FinishError::QuotaNotIn {
                iteration,
                short,
                pending_checks,
            } => {
                write!(f, "iteration {iteration} cannot end yet: ")?;
                if !short.is_empty() {
                    write!(
                        f,
                        "{} of its tasks have not received all their programs (the first is \
                         {:?})",
                        short.len(),
                        short[0]
                    )?;
                }
                if !short.is_empty() && *pending_checks > 0 {
                    write!(f, ", and ")?;
                }
                if *pending_checks > 0 {
                    write!(f, "{pending_checks} submissions wait to be checked")?;
                }
                write!(f, ". Nothing was changed")
            }
            FinishError::Prelude(e) => write!(f, "{e}"),
            FinishError::UnknownTask(t) => {
                write!(
                    f,
                    "the run has the task {t:?}, which this server does not know"
                )
            }
            FinishError::State(e) => write!(f, "the run's state is wrong: {e}"),
            FinishError::Interrupted(e) => write!(
                f,
                "the Stitch run was interrupted: {e}. Nothing was changed. Call again"
            ),
            FinishError::AlreadyFinished { iteration } => write!(
                f,
                "iteration {iteration} was ended by another server, or the run was replaced, \
                 while this step ran. This step changed nothing"
            ),
            FinishError::ChecksPending(why) => write!(
                f,
                "the step ran, but the iteration cannot end while submissions wait to be \
                 checked: {why}. Nothing was changed. Run dc_check_pending and call again"
            ),
        }
    }
}

impl std::error::Error for FinishError {}

impl From<StoreError> for FinishError {
    fn from(e: StoreError) -> FinishError {
        FinishError::Store(e)
    }
}

/// The settings of the step for a run with the settings `stitch` and the
/// frontier size `frontier_size`.
fn driver_config(frontier_size: u32, stitch: &StitchSettings) -> DriverConfig {
    DriverConfig {
        frontier_size: frontier_size as usize,
        compress: CompressConfig {
            iterations: stitch.iterations as usize,
            max_arity: stitch.max_arity,
        },
    }
}

/// What the step reads of a run.
struct Snapshot {
    iteration: u64,
    tasks: Vec<String>,
    library: Vec<dc_driver::LibraryEntry>,
    frontiers: BTreeMap<String, Vec<String>>,
    candidates: Vec<Candidate>,
    config: DriverConfig,
    stitch: StitchSettings,
}

/// The step's input, or why the iteration cannot end now.
fn snapshot(state: &RunState) -> Result<Snapshot, FinishError> {
    check_prelude(state.config(), shipped_prelude()).map_err(FinishError::Prelude)?;
    if !state.quota_in() {
        let quota = u64::from(state.config().proposals_per_task);
        return Err(FinishError::QuotaNotIn {
            iteration: state.iteration(),
            short: state
                .tasks()
                .iter()
                .filter(|t| state.task_counts(state.iteration(), t).received < quota)
                .cloned()
                .collect(),
            pending_checks: state.pending_checks().len(),
        });
    }
    Ok(Snapshot {
        iteration: state.iteration(),
        tasks: state.tasks().to_vec(),
        library: state
            .library()
            .iter()
            .map(|e| dc_driver::LibraryEntry {
                arity: e.arity,
                term: e.term.clone(),
            })
            .collect(),
        frontiers: state
            .tasks()
            .iter()
            .map(|t| {
                let members = state.frontier(t).unwrap_or_default();
                (
                    t.clone(),
                    members.iter().map(|m| m.program.clone()).collect(),
                )
            })
            .collect(),
        candidates: state
            .verified()
            .iter()
            .map(|v| Candidate {
                task: v.task.clone(),
                program: v.program.clone(),
            })
            .collect(),
        config: driver_config(state.config().frontier_size, &state.config().stitch),
        stitch: state.config().stitch,
    })
}

/// The argument that makes the `dc-mcp` binary a Stitch worker: started with
/// it alone, the binary calls `dc_compress::worker::serve`.
pub const WORKER_ARG: &str = "--stitch-worker";

/// The file a Linux process runs, by a name that does not change when the
/// file's path is given to another file. A child is started from it before
/// the child becomes another program, so it names the parent's file.
const RUNNING_BINARY: &str = "/proc/self/exe";

/// How to start the Stitch worker process of a step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerStart {
    /// The program. It must be the worker itself, not a wrapper that starts
    /// it as a child (`dc_compress::worker`).
    pub program: PathBuf,
    /// Its arguments.
    pub args: Vec<OsString>,
}

impl WorkerStart {
    /// The running binary, started again with [`WORKER_ARG`]. For the
    /// `dc-mcp` binary, whose `main` serves the worker then.
    ///
    /// Where `/proc/self/exe` exists (Linux), the worker is started from
    /// it: that is the file this process runs, even after another file has
    /// taken its path, so the worker is the build the server is. Elsewhere
    /// the worker is started from the path of the binary, and is whatever
    /// file is at that path when the step runs.
    pub fn this_binary() -> std::io::Result<WorkerStart> {
        let running = PathBuf::from(RUNNING_BINARY);
        let program = if running.exists() {
            running
        } else {
            std::env::current_exe()?
        };
        Ok(WorkerStart {
            program,
            args: vec![OsString::from(WORKER_ARG)],
        })
    }
}

/// What runs Stitch for a step.
enum Runner<'a> {
    /// A runner the caller made.
    Given(&'a mut dyn Stitch),
    /// A worker process, with the limits of the run's settings.
    Worker {
        start: &'a WorkerStart,
        cancel: &'a AtomicBool,
    },
}

/// Ends the current iteration of `run`: runs the iteration step with
/// `stitch` and commits its result in one record. See the module docs.
///
/// The run's Stitch limits do not apply to `stitch`. For tests and tools;
/// a server uses [`finish_iteration_in_worker`].
pub fn finish_iteration(
    store: &Store,
    run: &RunId,
    stitch: &mut dyn Stitch,
) -> Result<Finished, FinishError> {
    finish(store, run, Runner::Given(stitch))
}

/// As [`finish_iteration`], with Stitch run in a worker process started as
/// `start` says, under the time and memory limits of the run's settings
/// ([`StitchSettings`]). Setting `cancel` from another thread stops the
/// worker within about 50 milliseconds; the result is then
/// [`FinishError::Interrupted`]. The result is the same if `cancel` is set
/// when an attempt to commit reads it, under the run's lock.
pub fn finish_iteration_in_worker(
    store: &Store,
    run: &RunId,
    start: &WorkerStart,
    cancel: &AtomicBool,
) -> Result<Finished, FinishError> {
    finish(store, run, Runner::Worker { start, cancel })
}

fn finish(store: &Store, run: &RunId, runner: Runner<'_>) -> Result<Finished, FinishError> {
    let (snap, journal) = store.read_journal(run, snapshot)?;
    let snap = snap?;
    let mut tasks = Vec::with_capacity(snap.tasks.len());
    for name in &snap.tasks {
        let family = task(name).ok_or_else(|| FinishError::UnknownTask(name.clone()))?;
        tasks.push(TaskSpec {
            name: name.clone(),
            signature: family.signature(),
        });
    }
    // The fingerprint was compared under the lock, so this is the prelude
    // the run's programs were compiled with.
    let prelude = Prelude::load().expect("the shipped prelude loads");
    let input = IterationInput {
        tasks: &tasks,
        library: &snap.library,
        frontiers: &snap.frontiers,
        candidates: &snap.candidates,
        prelude: Some(&prelude),
    };
    let mut cancelled: Option<&AtomicBool> = None;
    let stepped = match runner {
        Runner::Given(stitch) => run_iteration(input, &snap.config, stitch),
        Runner::Worker { start, cancel } => {
            cancelled = Some(cancel);
            let worker = Worker {
                program: start.program.clone(),
                args: start.args.clone(),
                limits: Limits {
                    wall: Duration::from_millis(snap.stitch.wall_ms),
                    memory_bytes: snap.stitch.memory_bytes,
                    ..Limits::default()
                },
            };
            let mut isolated = Isolated {
                worker: &worker,
                cancel,
            };
            run_iteration(input, &snap.config, &mut isolated)
        }
    };
    let outcome = stepped.map_err(|e| match e {
        DriverError::Interrupted(why) => FinishError::Interrupted(why.to_string()),
        other => FinishError::State(other),
    })?;

    let used = snap.candidates.len() as u64;
    let mut events = Vec::with_capacity(snap.tasks.len() + 3);
    for name in &snap.tasks {
        let members = outcome.frontiers.get(name).cloned().unwrap_or_default();
        events.push(Event::FrontierSet {
            task: name.clone(),
            members: members
                .into_iter()
                .map(|program| FrontierMember { program })
                .collect(),
        });
    }
    if !outcome.new_library.is_empty() {
        events.push(Event::LibraryExtended {
            entries: outcome
                .new_library
                .iter()
                .map(|e| LibraryEntry {
                    arity: e.arity,
                    term: e.term.clone(),
                })
                .collect(),
        });
    }
    events.push(Event::Logged(LogRecord {
        kind: LogKind::Compression,
        iteration: snap.iteration,
        task: None,
        text: serde_json::to_string(&outcome.report)
            .expect("a report of strings and numbers serialises"),
    }));
    events.push(Event::IterationFinished {
        iteration: snap.iteration,
        used,
    });

    let beaten = || FinishError::AlreadyFinished {
        iteration: snap.iteration,
    };
    let mut waiting = String::new();
    for check in 0..=COMMIT_TRIES {
        // Written only to the journal the step read, and only if the run
        // is where the step read it. The frontiers and the library change
        // only when an iteration ends, so they are as the step read them.
        let wrote = store.commit_to(run, journal, |state| -> Result<_, FinishError> {
            // Read under the run's lock, at each attempt: a step with no
            // programs for Stitch starts no worker, and a cancel may come
            // after the worker is done or while the call waits for the lock.
            if cancelled.is_some_and(|c| c.load(Ordering::SeqCst)) {
                return Err(FinishError::Interrupted(
                    "the call was cancelled".to_owned(),
                ));
            }
            if state.iteration() != snap.iteration || state.library().len() != snap.library.len() {
                return Err(beaten());
            }
            Ok(match state.first_pending_check() {
                Some(batch) => (Some(batch), Vec::new()),
                None => (None, events.clone()),
            })
        })?;
        let batch = match wrote {
            None => return Err(beaten()),
            Some(None) => {
                return Ok(Finished {
                    iteration: snap.iteration,
                    used,
                    new_abstractions: outcome.new_library.len(),
                    report: outcome.report,
                })
            }
            Some(Some(batch)) => batch,
        };
        if check == COMMIT_TRIES {
            break;
        }
        // A submission came in during the step. It is of this iteration, so
        // it is checked with the library the step read.
        let checks = check_pending(store, run, &VerifierCache::default())
            .map_err(|e| FinishError::ChecksPending(e.to_string()))?;
        waiting = format!("batch {batch}'s submission waits");
        if !checks.claimed_elsewhere.is_empty() {
            waiting = format!(
                "another server is checking batch {}",
                checks.claimed_elsewhere[0]
            );
            std::thread::sleep(CHECK_WAIT);
        }
    }
    Err(FinishError::ChecksPending(waiting))
}

#[cfg(test)]
#[path = "iterate_tests.rs"]
mod tests;
