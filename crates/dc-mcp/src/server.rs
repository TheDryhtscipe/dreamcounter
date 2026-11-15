//! The MCP tools.
//!
//! There are two servers. [`DcServer`] is the operator's: it creates runs and
//! reports on them. [`ProposerServer`] is a proposer's: it serves one run and
//! has only `dc_next_batch` and `dc_submit` (see [`crate::proposer`]). The
//! `dc-mcp` binary serves one or the other.
//!
//! Every tool reads or writes through the [`Store`], which brings its copy of
//! the run up to the journal on disk first. A request the server cannot carry
//! out is a tool error with a message. It changes nothing in a run's state,
//! unless the message says the outcome is not known.
//!
//! A tool call runs on a thread of its own, so a call that waits for a run's
//! lock does not hold up other calls or the protocol.

use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::Duration;

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{schemars, tool, tool_router};
use serde::{Deserialize, Serialize};

use crate::proposer::{check_pending, next_batch, submit, Clock, NextBatch, Receipt, Refused};
use crate::state::{Arm, Event, LogKind, LogRecord, RunConfig, RunId, RunState};
use crate::store::{Store, StoreError};

/// The arguments of `dc_create_run`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateRun {
    /// The new run's id: 1 to 64 characters from a-z, 0-9, '-' and '_', starting
    /// with a letter or a digit.
    pub run: String,
    /// The seed for every random choice the run makes.
    pub seed: u64,
    /// How many proposals each task gets in one iteration. At least 1.
    pub proposals_per_task: u32,
    /// The names of the run's tasks. If absent, every task of the development
    /// corpus.
    pub tasks: Option<Vec<String>>,
    /// The most programs one batch asks for: 1 to 32. If absent, 1: each
    /// program then comes from its own batch.
    pub programs_per_batch: Option<u32>,
    /// How long a proposer has to submit a batch, in seconds, before its
    /// programs can be handed out again. At least 1. If absent, 1800.
    pub lease_seconds: Option<u64>,
}

/// The most programs one batch asks for when `dc_create_run` does not say.
pub const DEFAULT_PROGRAMS_PER_BATCH: u32 = 1;
/// The lease of a batch, in seconds, when `dc_create_run` does not say.
pub const DEFAULT_LEASE_SECONDS: u64 = 1800;

/// The arguments of `dc_run_status`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunArg {
    /// The run's id.
    pub run: String,
}

/// The arguments of `dc_add_note`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AddNote {
    /// The run's id.
    pub run: String,
    /// The remark.
    pub text: String,
}

/// The arguments of `dc_list_runs`: none.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NoArgs {}

/// One task of a run.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TaskStatus {
    /// The task's name.
    pub name: String,
    /// How many programs its frontier has.
    pub frontier_size: usize,
}

/// How many log records of each kind a run has.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LogCounts {
    /// Compression results.
    pub compression: u64,
    /// Remarks.
    pub note: u64,
    /// Reports that the store set aside bytes of the journal that were not a
    /// record. More than 0 can mean lost changes: look at the `torn-*` files
    /// in the run's directory. Such a file with no report means the same.
    pub recovery: u64,
}

/// Where a run is.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RunStatus {
    /// The run's id.
    pub run: String,
    /// The arm of the experiment. `blind`: the proposer gets no evaluation
    /// results.
    pub arm: String,
    /// The run's seed.
    pub seed: u64,
    /// How many proposals each task gets in one iteration.
    pub proposals_per_task: u32,
    /// The current iteration. The first is 0.
    pub iteration: u64,
    /// The run's tasks, in the order given at creation.
    pub tasks: Vec<TaskStatus>,
    /// How many definitions the library has.
    pub library_size: usize,
    /// How many log records of each kind the run has.
    pub log_counts: LogCounts,
    /// How many batches the run has issued, in all iterations.
    pub batches_issued: usize,
    /// What came in during the current iteration.
    pub current: IterationStats,
}

impl RunStatus {
    fn of(run: &RunId, state: &RunState) -> RunStatus {
        let config = state.config();
        RunStatus {
            run: run.as_str().to_owned(),
            arm: match config.arm {
                Arm::Blind => "blind".to_owned(),
            },
            seed: config.seed,
            proposals_per_task: config.proposals_per_task,
            iteration: state.iteration(),
            tasks: state
                .tasks()
                .iter()
                .map(|name| TaskStatus {
                    name: name.clone(),
                    frontier_size: state.frontier(name).map_or(0, <[_]>::len),
                })
                .collect(),
            library_size: state.library().len(),
            log_counts: LogCounts {
                compression: state.log_count(LogKind::Compression),
                note: state.log_count(LogKind::Note),
                recovery: state.log_count(LogKind::Recovery),
            },
            batches_issued: state.batches().len(),
            current: IterationStats::of(state, state.iteration()),
        }
    }
}

/// What came in for one task in one iteration.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TaskStats {
    /// The task's name.
    pub name: String,
    /// Batches issued for it.
    pub batches: u64,
    /// Programs received that its batches asked for, checked or not, from
    /// submissions that came in during the batch's iteration.
    pub received: u64,
    /// Programs checked: those its batches asked for, from submissions that
    /// came in during the batch's iteration.
    pub programs: u64,
    /// Programs that compiled.
    pub compiled: u64,
    /// Programs that compiled and had the task's type.
    pub typed: u64,
    /// Programs that passed every example.
    pub passed: u64,
    /// Whether at least one program passed.
    pub solved: bool,
}

/// What came in during one iteration.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct IterationStats {
    /// The iteration.
    pub iteration: u64,
    /// Whether every task has its programs, all checked. For the current
    /// iteration it says the iteration can be finished.
    pub quota_in: bool,
    /// Submissions that wait to be checked. Only the current iteration has
    /// any: an iteration cannot finish while checks are pending.
    pub pending_checks: usize,
    /// How many tasks have a program that passed.
    pub tasks_solved: usize,
    /// Submissions received, of any standing.
    pub submissions: u64,
    /// Submissions for a batch of an earlier iteration: kept, not checked.
    pub late: u64,
    /// Submissions for a batch that already had one: kept, not checked.
    pub repeats: u64,
    /// Submissions from a worker that does not hold the batch: kept, not
    /// checked.
    pub foreign: u64,
    /// Bytes of payload handed out. About four bytes make a token.
    pub payload_bytes: u64,
    /// Bytes of program text received.
    pub submission_bytes: u64,
    /// Each task, in the run's order.
    pub tasks: Vec<TaskStats>,
}

impl IterationStats {
    fn of(state: &RunState, iteration: u64) -> IterationStats {
        let tasks: Vec<TaskStats> = state
            .tasks()
            .iter()
            .map(|name| {
                let c = state.task_counts(iteration, name);
                TaskStats {
                    name: name.clone(),
                    batches: c.batches,
                    received: c.received,
                    programs: c.programs,
                    compiled: c.compiled,
                    typed: c.typed,
                    passed: c.passed,
                    solved: c.passed > 0,
                }
            })
            .collect();
        let counts = state.iteration_counts(iteration);
        let quota = u64::from(state.config().proposals_per_task);
        let current = iteration == state.iteration();
        IterationStats {
            iteration,
            quota_in: if current {
                state.quota_in()
            } else {
                tasks.iter().all(|t| t.programs >= quota)
            },
            pending_checks: if current {
                state.pending_checks().len()
            } else {
                0
            },
            tasks_solved: tasks.iter().filter(|t| t.solved).count(),
            submissions: counts.submissions,
            late: counts.late,
            repeats: counts.repeats,
            foreign: counts.foreign,
            payload_bytes: counts.payload_bytes,
            submission_bytes: counts.submission_bytes,
            tasks,
        }
    }
}

/// The result of `dc_check_pending`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct CheckReport {
    /// The run's id.
    pub run: String,
    /// How many submissions this call checked.
    pub checked: usize,
    /// How many submissions still wait to be checked. More than 0 means
    /// another submission came in meanwhile.
    pub pending_checks: usize,
}

/// The result of `dc_proposal_stats`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct ProposalStats {
    /// The run's id.
    pub run: String,
    /// Each iteration from 0 to the current one.
    pub iterations: Vec<IterationStats>,
}

/// The result of `dc_list_runs`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct RunList {
    /// The ids of the runs, in order.
    pub runs: Vec<String>,
}

/// The names of the tasks a run can have: the development corpus, in its
/// order. The held-out family is not among them (decision D14).
pub fn known_tasks() -> Vec<String> {
    dc_tasks::families::development()
        .iter()
        .map(|variant| variant.family.name().to_owned())
        .collect()
}

/// The MCP server for the runs of one [`Store`].
#[derive(Clone, Debug)]
pub struct DcServer {
    store: Arc<Store>,
}

impl DcServer {
    /// A server for the runs in `store`.
    pub fn new(store: Store) -> DcServer {
        DcServer {
            store: Arc::new(store),
        }
    }
}

/// The longest text of a refused argument that an error message repeats.
const MAX_ECHO: usize = 80;

fn run_id(text: &str) -> Result<RunId, String> {
    RunId::new(text).map_err(|e| {
        let shown: String = text.chars().take(MAX_ECHO).collect();
        let more = if shown.len() < text.len() { "..." } else { "" };
        format!("{shown:?}{more} is not a run id: {e}")
    })
}

impl DcServer {
    /// Runs `work` on a thread that may block, and waits for it.
    async fn blocking<T, F>(&self, work: F) -> Result<Json<T>, String>
    where
        T: Send + 'static,
        F: FnOnce(&Store) -> Result<T, String> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        tokio::task::spawn_blocking(move || work(&store))
            .await
            .map_err(|e| format!("the tool call did not finish: {e}"))?
            .map(Json)
    }
}

#[tool_router(server_handler)]
impl DcServer {
    /// Makes a run.
    #[tool(
        name = "dc_create_run",
        description = "Create a DreamCounter run at iteration 0, with an empty library and an \
                       empty frontier for each task. Fails if a run of that id exists. Returns \
                       the new run's status."
    )]
    async fn create_run(
        &self,
        Parameters(arg): Parameters<CreateRun>,
    ) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        let known = known_tasks();
        let tasks = match arg.tasks {
            None => known,
            Some(tasks) => {
                if let Some(unknown) = tasks.iter().find(|t| !known.contains(t)) {
                    let shown: String = unknown.chars().take(MAX_ECHO).collect();
                    return Err(format!(
                        "there is no task {shown:?}. The tasks are: {}",
                        known.join(", ")
                    ));
                }
                tasks
            }
        };
        let config = RunConfig {
            arm: Arm::Blind,
            seed: arg.seed,
            proposals_per_task: arg.proposals_per_task,
            programs_per_batch: arg.programs_per_batch.unwrap_or(DEFAULT_PROGRAMS_PER_BATCH),
            lease_ms: arg
                .lease_seconds
                .unwrap_or(DEFAULT_LEASE_SECONDS)
                .saturating_mul(1000),
        };
        self.blocking(move |store| {
            let state = store
                .create_run(&run, config, tasks)
                .map_err(|e| e.to_string())?;
            Ok(RunStatus::of(&run, &state))
        })
        .await
    }

    /// Reports where a run is.
    #[tool(
        name = "dc_run_status",
        description = "Report where a DreamCounter run is: its settings, its iteration, the size \
                       of each task's frontier, the size of the library, and how many log \
                       records it has."
    )]
    async fn run_status(
        &self,
        Parameters(arg): Parameters<RunArg>,
    ) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            store
                .read(&run, |state| RunStatus::of(&run, state))
                .map_err(|e| e.to_string())
        })
        .await
    }

    /// Adds an operator's remark to a run's log.
    #[tool(
        name = "dc_add_note",
        description = "Add a remark to a DreamCounter run's log, in the run's current iteration. \
                       Returns the run's status once the remark is on disk."
    )]
    async fn add_note(
        &self,
        Parameters(arg): Parameters<AddNote>,
    ) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            let noted: Result<RunStatus, StoreError> = store.commit_then(
                &run,
                |state| {
                    let note = Event::Logged(LogRecord {
                        kind: LogKind::Note,
                        iteration: state.iteration(),
                        task: None,
                        text: arg.text,
                    });
                    Ok(((), vec![note]))
                },
                |(), state| RunStatus::of(&run, state),
            );
            noted.map_err(|e| e.to_string())
        })
        .await
    }

    /// Reports what proposers sent in each iteration.
    #[tool(
        name = "dc_proposal_stats",
        description = "Report, for each iteration of a DreamCounter run, what proposers sent: \
                       for each task the batches issued, the programs checked, how many \
                       compiled, typed and passed their examples, and whether the task was \
                       solved; and the bytes of payload and program text."
    )]
    async fn proposal_stats(
        &self,
        Parameters(arg): Parameters<RunArg>,
    ) -> Result<Json<ProposalStats>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            store
                .read(&run, |state| ProposalStats {
                    run: run.as_str().to_owned(),
                    iterations: (0..=state.iteration())
                        .map(|i| IterationStats::of(state, i))
                        .collect(),
                })
                .map_err(|e| e.to_string())
        })
        .await
    }

    /// Checks the submissions of a run that wait to be checked.
    #[tool(
        name = "dc_check_pending",
        description = "Check every submission of a DreamCounter run that waits to be checked, \
                       and wait until they are done. Proposer servers do this in the \
                       background; use this when one stopped before it finished, or to see why \
                       checks fail. Returns how many it checked and how many still wait."
    )]
    async fn check_pending(
        &self,
        Parameters(arg): Parameters<RunArg>,
    ) -> Result<Json<CheckReport>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            let checked = check_pending(store, &run).map_err(|e| e.to_string())?;
            let pending_checks = store
                .read(&run, |state| state.pending_checks().len())
                .map_err(|e| e.to_string())?;
            Ok(CheckReport {
                run: run.as_str().to_owned(),
                checked,
                pending_checks,
            })
        })
        .await
    }

    /// Lists the runs.
    #[tool(
        name = "dc_list_runs",
        description = "List the ids of the DreamCounter runs in the server's state directory."
    )]
    async fn list_runs(
        &self,
        Parameters(NoArgs {}): Parameters<NoArgs>,
    ) -> Result<Json<RunList>, String> {
        self.blocking(|store| {
            let runs = store.runs().map_err(|e| e.to_string())?;
            Ok(RunList {
                runs: runs.iter().map(|r| r.as_str().to_owned()).collect(),
            })
        })
        .await
    }
}

/// The arguments of `dc_next_batch`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NextBatchArgs {
    /// Your id, as your brief gives it: 1 to 200 bytes.
    pub worker: String,
}

/// The arguments of `dc_submit`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SubmitArgs {
    /// Your id: the one you gave `dc_next_batch` for this batch.
    pub worker: String,
    /// The batch's number, from `dc_next_batch`.
    pub batch: u64,
    /// The batch's token, from `dc_next_batch`.
    pub token: String,
    /// The programs, one string for each: the full text of a dc program that
    /// defines `solve`. At most 32, each at most 64 KiB.
    pub programs: Vec<String>,
    /// The model you are, if you know it: 1 to 200 bytes.
    pub model: Option<String>,
}

/// The MCP server a proposer connects to (decision D11). It serves one run
/// and has two tools, `dc_next_batch` and `dc_submit`, so a proposer cannot
/// list runs or see a run's library, frontiers or results.
#[derive(Clone, Debug)]
pub struct ProposerServer {
    store: Arc<Store>,
    run: RunId,
    clock: Arc<dyn Clock>,
    checker: Arc<Checker>,
}

/// Runs the checks of pending submissions, one at a time in each process.
#[derive(Debug, Default)]
struct Checker {
    /// Held while checks run.
    running: Mutex<()>,
    /// Set when there may be new work.
    dirty: AtomicBool,
}

impl Checker {
    /// Runs `check` until no new work has been signalled, unless another
    /// thread is running it already. `idle` runs when this thread has found
    /// no new work but still holds `running`; tests use it to signal work in
    /// that window.
    fn drain(&self, mut check: impl FnMut(), mut idle: impl FnMut()) {
        loop {
            // With the fence after `drop` below, either this thread sees the
            // lock free or the holder sees `dirty` set.
            fence(Ordering::SeqCst);
            let running = match self.running.try_lock() {
                Ok(running) => running,
                // The lock guards no data, so a check that panicked while it
                // held the lock left nothing half done.
                Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                // The thread that holds it looks at `dirty` after it lets go.
                Err(TryLockError::WouldBlock) => return,
            };
            if self.dirty.swap(false, Ordering::SeqCst) {
                check();
                continue;
            }
            idle();
            drop(running);
            fence(Ordering::SeqCst);
            // Work signalled while this thread held the lock was refused the
            // lock: take it up.
            if !self.dirty.load(Ordering::SeqCst) {
                return;
            }
        }
    }
}

/// How long the background checks wait after an error before they try
/// again, at first and at most.
const CHECK_RETRY_FIRST: Duration = Duration::from_secs(5);
const CHECK_RETRY_MAX: Duration = Duration::from_secs(300);

impl ProposerServer {
    /// A server for proposers of `run` in `store`, with leases timed by
    /// `clock`.
    pub fn new(store: Store, run: RunId, clock: Arc<dyn Clock>) -> ProposerServer {
        ProposerServer {
            store: Arc::new(store),
            run,
            clock,
            checker: Arc::default(),
        }
    }

    /// Checks the run's pending submissions on a thread of its own. Call it
    /// after a submission, and once when the server starts, so a submission
    /// left unchecked by a server that stopped is checked. Must be called
    /// within a Tokio runtime.
    pub fn check_in_background(&self) {
        self.checker.dirty.store(true, Ordering::SeqCst);
        let store = Arc::clone(&self.store);
        let run = self.run.clone();
        let checker = Arc::clone(&self.checker);
        tokio::task::spawn_blocking(move || {
            let mut retry = CHECK_RETRY_FIRST;
            checker.drain(
                || match catch_check(|| {
                    check_pending(&store, &run)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                }) {
                    Ok(_) => retry = CHECK_RETRY_FIRST,
                    Err(e) => {
                        // Standard error may be closed.
                        let _ = writeln!(
                            std::io::stderr(),
                            "dc-mcp: the submissions of run {run} could not be checked: {e}; \
                             trying again in {} s",
                            retry.as_secs()
                        );
                        std::thread::sleep(retry);
                        retry = (retry * 2).min(CHECK_RETRY_MAX);
                        checker.dirty.store(true, Ordering::SeqCst);
                    }
                },
                || {},
            );
        });
    }

    /// Runs `work` on a thread that may block, and waits for it.
    async fn blocking<T, F>(&self, work: F) -> Result<Json<T>, String>
    where
        T: Send + 'static,
        F: FnOnce(&Store, &RunId, u64) -> Result<T, Refused> + Send + 'static,
    {
        let store = Arc::clone(&self.store);
        let run = self.run.clone();
        let clock = Arc::clone(&self.clock);
        let done = tokio::task::spawn_blocking(move || work(&store, &run, clock.now_ms()))
            .await
            .map_err(|e| {
                let _ = writeln!(
                    std::io::stderr(),
                    "dc-mcp: run {}: a tool call did not finish: {e}",
                    self.run
                );
                "the server could not do this; tell the operator".to_owned()
            })?;
        done.map(Json)
            .map_err(|refused| proposer_reply(&self.run, refused))
    }
}

/// Runs `check`, and makes a panic in it an error, so the background checks
/// log it and try again rather than stop.
fn catch_check(check: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(check)).unwrap_or_else(|panic| {
        let what = panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "no message".to_owned());
        Err(format!("the check panicked: {what}"))
    })
}

/// What a proposer of `run` is told when its call is refused. Only a refused
/// request, which is the proposer's to fix, is told in full; anything else
/// goes to standard error for the operator, as it may show the state
/// directory or the run's state.
fn proposer_reply(run: &RunId, refused: Refused) -> String {
    match refused {
        // The details are for the operator, not the proposer.
        Refused::State(why) => {
            let _ = writeln!(
                std::io::stderr(),
                "dc-mcp: run {run} cannot take work: {why}"
            );
            "the run cannot take work now; tell the operator".to_owned()
        }
        Refused::Request(why) => why,
        Refused::Store(StoreError::Busy(_)) => {
            "the run is busy; ask again in a few seconds".to_owned()
        }
        Refused::Store(e) => {
            let _ = writeln!(std::io::stderr(), "dc-mcp: run {run}: {e}");
            if matches!(e, StoreError::OutcomeUnknown { .. }) {
                "the server cannot tell whether this was recorded; do not send it again, \
                 and tell the operator"
                    .to_owned()
            } else {
                "the server could not do this; tell the operator".to_owned()
            }
        }
    }
}

#[tool_router(server_handler)]
impl ProposerServer {
    /// Hands out a batch of work.
    #[tool(
        name = "dc_next_batch",
        description = "Get a batch of work: a task, and how many programs to write for it. The \
                       prompt says how to write them. If there is no work now, `batch` is \
                       null and `wait` says why and when to ask again."
    )]
    async fn next_batch(
        &self,
        Parameters(arg): Parameters<NextBatchArgs>,
    ) -> Result<Json<NextBatch>, String> {
        self.blocking(move |store, run, now| next_batch(store, run, &arg.worker, now))
            .await
    }

    /// Takes in the programs for a batch.
    #[tool(
        name = "dc_submit",
        description = "Submit the programs for a batch, with the batch's number and token from \
                       dc_next_batch. The reply says how many programs were received. It does \
                       not say whether they are right."
    )]
    async fn submit(
        &self,
        Parameters(arg): Parameters<SubmitArgs>,
    ) -> Result<Json<Receipt>, String> {
        let receipt = self
            .blocking(move |store, run, now| {
                submit(
                    store,
                    run,
                    arg.batch,
                    &arg.worker,
                    &arg.token,
                    arg.programs,
                    arg.model,
                    now,
                )
            })
            .await;
        // A submission may be kept even when the reply is an error, if the
        // store cannot tell whether its write went through.
        self.check_in_background();
        receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_signalled_while_the_checker_lets_go_is_not_lost() {
        let checker = Checker::default();
        checker.dirty.store(true, Ordering::SeqCst);
        let mut checks = 0;
        let mut signalled = false;
        checker.drain(
            || checks += 1,
            || {
                // Another thread signals work and finds the lock held.
                if !signalled {
                    signalled = true;
                    checker.dirty.store(true, Ordering::SeqCst);
                    assert!(checker.running.try_lock().is_err());
                }
            },
        );
        assert_eq!(checks, 2);
        assert!(!checker.dirty.load(Ordering::SeqCst));
    }

    #[test]
    fn work_signalled_during_a_check_is_checked() {
        let checker = Checker::default();
        checker.dirty.store(true, Ordering::SeqCst);
        let mut checks = 0;
        checker.drain(
            || {
                checks += 1;
                if checks < 3 {
                    checker.dirty.store(true, Ordering::SeqCst);
                }
            },
            || {},
        );
        assert_eq!(checks, 3);
    }

    #[test]
    fn a_checker_whose_lock_was_poisoned_still_checks() {
        let checker = Arc::new(Checker::default());
        let poisoner = Arc::clone(&checker);
        let panicked = std::thread::spawn(move || {
            let _held = poisoner.running.lock().unwrap();
            panic!("a check panicked");
        })
        .join();
        assert!(panicked.is_err());
        assert!(checker.running.is_poisoned());
        checker.dirty.store(true, Ordering::SeqCst);
        let mut checks = 0;
        checker.drain(|| checks += 1, || {});
        assert_eq!(checks, 1);
        assert!(!checker.dirty.load(Ordering::SeqCst));
    }

    #[test]
    fn a_panicking_check_is_an_error_with_its_message() {
        let got = catch_check(|| panic!("the verifier broke"));
        let Err(why) = got else {
            panic!("a panic was taken for success: {got:?}");
        };
        assert!(why.contains("the verifier broke"), "{why}");
        assert_eq!(catch_check(|| Ok(())), Ok(()));
        assert_eq!(catch_check(|| Err("no".to_owned())), Err("no".to_owned()));
    }

    #[test]
    fn a_proposer_is_not_told_what_the_store_says() {
        let run = RunId::new("r1").unwrap();
        let secret = "/home/operator/state-dir/r1/journal";
        let refusals = [
            Refused::Store(StoreError::Io {
                path: secret.into(),
                error: std::io::Error::other("disk full"),
            }),
            Refused::Store(StoreError::Rejected(crate::state::Rejected::UnknownTask(
                secret.to_owned(),
            ))),
            Refused::State(secret.to_owned()),
        ];
        for refused in refusals {
            let told = proposer_reply(&run, refused);
            assert!(!told.contains(secret), "{told}");
            assert!(!told.contains("disk full"), "{told}");
            assert!(told.contains("operator"), "{told}");
        }
        // When the store cannot tell whether a submission was kept, sending
        // it again would make it a repeat.
        let unknown = proposer_reply(
            &run,
            Refused::Store(StoreError::OutcomeUnknown {
                path: secret.into(),
                error: std::io::Error::other("disk full"),
            }),
        );
        assert!(!unknown.contains(secret), "{unknown}");
        assert!(unknown.contains("not send it again"), "{unknown}");
        let busy = proposer_reply(&run, Refused::Store(StoreError::Busy(run.clone())));
        assert!(busy.contains("again"), "{busy}");
        // A refused request is the proposer's to fix, so it is told why.
        let why = "batch 9 was not issued in this run";
        assert_eq!(proposer_reply(&run, Refused::Request(why.to_owned())), why);
    }

    #[test]
    fn a_checker_with_no_work_does_nothing() {
        let checker = Checker::default();
        let mut checks = 0;
        checker.drain(|| checks += 1, || {});
        assert_eq!(checks, 0);
    }
}
