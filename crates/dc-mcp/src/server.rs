//! The MCP tools.
//!
//! There are two servers. [`DcServer`] is the operator's: it creates runs and
//! reports on them. [`ProposerServer`] is a proposer's: it serves one run in
//! one [`ProposerMode`], and has only that mode's two tools: `dc_next_batch`
//! and `dc_submit` (see [`crate::proposer`]), or `dc_next_naming_job` and
//! `dc_submit_naming` (see [`crate::naming`]). The `dc-mcp` binary serves one
//! of these.
//!
//! Every tool reads or writes through the [`Store`], which brings its copy of
//! the run up to the journal on disk first. A request the server cannot carry
//! out is a tool error with a message. It changes nothing in a run's state,
//! unless the message says the outcome is not known.
//!
//! A tool call runs on a thread of its own, so a call that waits for a run's
//! lock does not hold up other calls or the protocol.

use std::future::Future;
use std::io::Write;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, TryLockError};
use std::time::{Duration, Instant};

use dc_driver::CompressionReport;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::service::RequestContext;
use rmcp::{schemars, tool, tool_handler, tool_router, RoleServer, ServerHandler};
use serde::{Deserialize, Serialize};

use crate::diversity::{
    passed_programs, report as diversity_report, DiversityReport, GrowthLimits,
};
use crate::iterate::{finish_iteration_in_worker, WorkerStart};
use crate::naming::{next_naming_job, submit_naming, NamingReceipt, NextNamingJob};
use crate::proposer::{
    available_memory, check_pending, check_pending_parallel, default_check_workers, next_batch,
    submit, Clock, NextBatch, Receipt, Refused, MAX_CHECK_WORKERS,
};
use crate::state::{
    Arm, Event, FrontierMember, LibraryUptake, LogKind, LogRecord, RunConfig, RunId, RunState,
    StitchSettings, DEFAULT_MAX_UNCHECKED_PER_BATCH, MAX_EVALUATION_MEMORY,
    MIN_STITCH_MEMORY_BYTES,
};
use crate::store::{Refusals, Store, StoreError};
use crate::verify::{shipped_prelude, VerifierCache};
use dc_tasks::harness::DEFAULT_MAX_MEMORY;

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
    /// The most submissions kept for one batch that are not checked, of
    /// each kind: foreign ones, and late or repeat ones from the batch's
    /// worker. Past it a submission is counted but not kept. At least 1. If
    /// absent, 3.
    pub max_unchecked_per_batch: Option<u32>,
    /// The most programs a task's frontier keeps when an iteration ends. At
    /// least 1.
    pub frontier_size: u32,
    /// The most abstractions the Stitch run that ends an iteration finds.
    /// At least 1. If absent, 3.
    pub stitch_iterations: Option<u32>,
    /// The largest arity of an abstraction. If absent, 2.
    pub stitch_max_arity: Option<u32>,
    /// How long the Stitch run that ends an iteration may take, in seconds.
    /// A run stopped at this limit ends the iteration with no new
    /// abstractions. At least 1. If absent, 600.
    pub stitch_wall_seconds: Option<u64>,
    /// The memory of the Stitch run's process, in MiB, counted as address
    /// space. A run stopped at this limit ends the iteration with no new
    /// abstractions. At least 512. If absent, 4096.
    pub stitch_memory_mib: Option<u64>,
    /// Whether proposers name and document the abstractions each iteration
    /// adds (decision D23). While a new abstraction has no name, no batch is
    /// issued, so turn it on only when proposers will pull naming jobs. If
    /// absent, false.
    pub autodoc: Option<bool>,
    /// Whether the run learns a library. If false, the run is the LLM-only
    /// baseline (decision D24): each iteration selects frontiers as usual,
    /// but no Stitch run ends it and the library stays empty, so autodoc
    /// must be off, and it takes no Stitch setting. To match it with the
    /// run it is compared with on proposals, name that run in
    /// compared_with, end as many iterations in both, then compare the
    /// batches and payload bytes that `dc_proposal_stats` reports for the
    /// two runs. Tokens are not matched. If absent, true.
    pub library_learning: Option<bool>,
    /// The most memory one evaluation of a program may hold, in MiB: each
    /// run on an example, a large input or a size of the cost sweep. A run
    /// that would hold more fails with a memory limit, like a run out of
    /// fuel. Each check of a program holds up to this much at once, and the
    /// server runs several checks at a time, so the server can hold a
    /// multiple of it. At least 1 and at most 1048576 (1 TiB). If absent,
    /// the compared run's, or 512.
    pub max_memory_mib: Option<u64>,
    /// The id of an existing run that this run is compared with, such as
    /// the main run of an LLM-only baseline. Either run may learn a
    /// library. If given, tasks, programs_per_batch, lease_seconds,
    /// max_unchecked_per_batch, max_memory_mib, approach_hints and
    /// show_received, when absent, are that run's, and the run is not made
    /// unless its seed, proposals_per_task, frontier_size, tasks (in order),
    /// programs_per_batch, lease_seconds, max_unchecked_per_batch,
    /// max_memory_mib, approach_hints and show_received are that run's too,
    /// and that run uses this server's prelude. The error names each
    /// setting that differs.
    /// `dc_run_status` reports the pairing.
    pub compared_with: Option<String>,
    /// Whether each payload names an approach to try, drawn by the run's
    /// seed from a fixed list that does not depend on the task (decision
    /// D32). A task's batches in one iteration that hold a live lease or
    /// were submitted get different approaches until the list runs out; a
    /// batch whose lease ended unsubmitted gives its approach to the next.
    /// If absent, the compared run's, or false.
    pub approach_hints: Option<bool>,
    /// Whether each payload shows the programs already received for its
    /// task in this iteration, without results, and asks for a different
    /// approach (decision D32). If absent, the compared run's, or false.
    pub show_received: Option<bool>,
    /// Whether the step that ends each iteration refuses the bucketing
    /// abstractions Stitch finds (decision D36). An abstraction is
    /// bucketing if it puts a value onto the list in an array slot, as
    /// `array_set a k (cons v (array_get a k))` or with the list named
    /// first by a `let`, or if it uses a library entry or an earlier new
    /// abstraction that is bucketing. A refused abstraction does not join
    /// the library: each use of it is written out, and the compression log
    /// lists it in `written_out` with `bucketing` true. It needs
    /// library_learning. It is not taken from the compared run, and the two
    /// runs may differ in it. If absent, false.
    pub refuse_bucketing: Option<bool>,
    /// Entries that join the library when iteration 0 ends, after the
    /// entries Stitch finds then, each with its name and doc and with no
    /// naming job (decision D36). At most 16. In a term, `lib_k` is entry
    /// `k` of this list, and an entry may use only the prelude and the
    /// entries before it. Each must type. They need library_learning. They
    /// are not taken from the compared run, and the two runs may differ in
    /// them. If absent, none.
    pub seed_entries: Option<Vec<SeedEntry>>,
}

/// One entry of `dc_create_run`'s seed_entries.
#[derive(Clone, Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SeedEntry {
    /// The closed term `λ^arity. body`, as an s-expression, at most 4096
    /// bytes.
    pub term: String,
    /// The number of parameters: the leading lambdas of the term that a use
    /// fills in by substitution.
    pub arity: u32,
    /// Its name, under the rules for a name a proposer gives an entry.
    pub name: String,
    /// What it does, in one line of printable ASCII, at most 400 bytes.
    pub doc: String,
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

/// The arguments of `dc_check_pending`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CheckPendingArg {
    /// The run's id.
    pub run: String,
    /// How many submissions to check at once, each on a thread of its own,
    /// from 1 to 64. Each check may hold up to the run's max_memory_mib. If
    /// absent, one for each core, but no more than the memory the system
    /// has available when the call starts holds at the run's
    /// max_memory_mib each. Other calls running at once, and memory
    /// limits of a cgroup, are not counted.
    pub workers: Option<usize>,
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
    /// The most programs one batch asks for.
    pub programs_per_batch: u32,
    /// How long a proposer has to submit a batch, in seconds, rounded up.
    pub lease_seconds: u64,
    /// The most submissions kept for one batch that are not checked, of
    /// each kind.
    pub max_unchecked_per_batch: u32,
    /// The most programs a task's frontier keeps when an iteration ends.
    pub frontier_size: u32,
    /// The settings of the Stitch run that ends an iteration. Null if the
    /// run does not learn a library, since no Stitch run ends its
    /// iterations.
    pub stitch: Option<StitchStatus>,
    /// The current iteration. The first is 0.
    pub iteration: u64,
    /// The run's tasks, in the order given at creation.
    pub tasks: Vec<TaskStatus>,
    /// How many definitions the library has.
    pub library_size: usize,
    /// Whether proposers name new definitions (decision D23).
    pub autodoc: bool,
    /// Whether the run learns a library. False for the LLM-only baseline
    /// (decision D24), whose library stays empty.
    pub library_learning: bool,
    /// The most memory one evaluation may hold, in MiB, rounded down.
    pub max_memory_mib: u64,
    /// The run this one is compared with, if it was made with one.
    pub compared_with: Option<Comparison>,
    /// Whether each payload names an approach to try (decision D32).
    pub approach_hints: bool,
    /// Whether each payload shows the programs already received for its
    /// task in this iteration (decision D32).
    pub show_received: bool,
    /// Whether the run refuses the bucketing abstractions Stitch finds
    /// (decision D36).
    pub refuse_bucketing: bool,
    /// The names of the run's seed entries, in order (decision D36). They
    /// are in the library from the end of iteration 0.
    pub seed_entries: Vec<String>,
    /// How many definitions wait for a naming result. While any does, no
    /// batch is issued.
    pub unnamed: u64,
    /// How many log records of each kind the run has.
    pub log_counts: LogCounts,
    /// How many batches the run has issued, in all iterations.
    pub batches_issued: usize,
    /// What came in during the current iteration.
    pub current: IterationStats,
}

/// The run that a run is compared with.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Comparison {
    /// That run's id.
    pub run: String,
    /// That run's current iteration, to set beside this run's. Null if
    /// that run cannot be read.
    pub iteration: Option<u64>,
}

/// The settings of the Stitch run that ends an iteration.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct StitchStatus {
    /// The most abstractions one run finds.
    pub iterations: u32,
    /// The largest arity of an abstraction.
    pub max_arity: u32,
    /// How long a run may take, in seconds, rounded up.
    pub wall_seconds: u64,
    /// The memory of a run's process, in MiB, rounded down.
    pub memory_mib: u64,
}

impl RunStatus {
    fn of(run: &RunId, state: &RunState, refusals: &Refusals) -> RunStatus {
        let config = state.config();
        RunStatus {
            run: run.as_str().to_owned(),
            arm: match config.arm {
                Arm::Blind => "blind".to_owned(),
            },
            seed: config.seed,
            proposals_per_task: config.proposals_per_task,
            programs_per_batch: config.programs_per_batch,
            lease_seconds: config.lease_ms.div_ceil(1000),
            max_unchecked_per_batch: config.max_unchecked_per_batch,
            frontier_size: config.frontier_size,
            stitch: config.library_learning.then(|| StitchStatus {
                iterations: config.stitch.iterations,
                max_arity: config.stitch.max_arity,
                wall_seconds: config.stitch.wall_ms.div_ceil(1000),
                memory_mib: config.stitch.memory_bytes >> 20,
            }),
            iteration: state.iteration(),
            autodoc: config.autodoc,
            library_learning: config.library_learning,
            max_memory_mib: config.max_memory >> 20,
            compared_with: config.compared_with.as_ref().map(|other| Comparison {
                run: other.clone(),
                iteration: None,
            }),
            approach_hints: config.approach_hints,
            show_received: config.show_received,
            refuse_bucketing: config.refuse_bucketing,
            seed_entries: config
                .seed_entries
                .iter()
                .filter_map(|e| e.name.clone())
                .collect(),
            unnamed: state.names_pending(),
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
            batches_issued: usize::try_from(state.batch_count()).unwrap_or(usize::MAX),
            current: IterationStats::of(state, refusals, state.iteration()),
        }
    }

    /// Fills in the iteration of the run this one is compared with. It
    /// reads that run after this one, so the two runs are never locked at
    /// once.
    fn compared(mut self, store: &Store) -> RunStatus {
        if let Some(comparison) = &mut self.compared_with {
            comparison.iteration = RunId::new(&comparison.run)
                .ok()
                .and_then(|other| store.read(&other, RunState::iteration).ok());
        }
        self
    }
}

/// `bytes` in MiB, with any part of a MiB in bytes, so that limits that
/// differ by less than a MiB are told apart.
fn mib(bytes: u64) -> String {
    match bytes % (1 << 20) {
        0 => format!("{} MiB", bytes >> 20),
        part => format!("{} MiB and {part} bytes", bytes >> 20),
    }
}

/// The settings that two runs compared with each other must share, as
/// `dc_create_run` names them, where `config` and `tasks` differ from
/// `other` and `other_tasks`. Each is said with both values. They are the
/// settings that fix the proposals a run hands out (decisions D24 and D32), and the
/// memory limit, which decides which programs pass their checks and what
/// they cost.
fn differences(
    config: &RunConfig,
    tasks: &[String],
    other: &RunConfig,
    other_tasks: &[String],
) -> Vec<String> {
    let mut found = Vec::new();
    let mut check = |name: &str, mine: String, theirs: String| {
        if mine != theirs {
            found.push(format!("{name} is {mine}, not {theirs}"));
        }
    };
    check("seed", config.seed.to_string(), other.seed.to_string());
    check(
        "proposals_per_task",
        config.proposals_per_task.to_string(),
        other.proposals_per_task.to_string(),
    );
    check(
        "frontier_size",
        config.frontier_size.to_string(),
        other.frontier_size.to_string(),
    );
    check(
        "programs_per_batch",
        config.programs_per_batch.to_string(),
        other.programs_per_batch.to_string(),
    );
    // Both in ms, so a lease that differs by less than a second is found
    // and said exactly.
    let seconds = |ms: u64| match ms % 1000 {
        0 => (ms / 1000).to_string(),
        part => format!("{}.{part:03}", ms / 1000),
    };
    check(
        "lease_seconds",
        seconds(config.lease_ms),
        seconds(other.lease_ms),
    );
    // It decides which late or repeat submissions are kept.
    check(
        "max_unchecked_per_batch",
        config.max_unchecked_per_batch.to_string(),
        other.max_unchecked_per_batch.to_string(),
    );
    // They change the payload (decision D32).
    check(
        "approach_hints",
        config.approach_hints.to_string(),
        other.approach_hints.to_string(),
    );
    check(
        "show_received",
        config.show_received.to_string(),
        other.show_received.to_string(),
    );
    // It decides which programs pass their checks, and what they cost.
    check(
        "max_memory_mib",
        mib(config.max_memory),
        mib(other.max_memory),
    );
    if tasks != other_tasks {
        let shown = |t: &[String]| {
            let all = t.join(", ");
            let cut: String = all.chars().take(MAX_ECHO).collect();
            if cut.len() < all.len() {
                format!("[{cut}...]")
            } else {
                format!("[{cut}]")
            }
        };
        found.push(format!(
            "tasks are {}, not {}",
            shown(tasks),
            shown(other_tasks)
        ));
    }
    if config.prelude != other.prelude {
        found.push(
            "prelude differs: the run compared with was made with another prelude than \
             this server's"
                .to_owned(),
        );
    }
    found
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
    /// Programs that gave, on the first example they failed, the value of
    /// the task's known mistake (decision D36). For a chain task that is
    /// the answer without the order of positions. A program with the
    /// mistake that fails an earlier example for another reason is not
    /// counted, and a program that gives that value for another reason is.
    /// Absent for a task with no example that tells its known mistake from
    /// the right answer, such as a task that names none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub known_mistake: Option<u64>,
    /// Whether at least one program passed.
    pub solved: bool,
}

impl TaskStats {
    /// The stats of task `name` from its counts `c` in one iteration.
    fn of(name: &str, c: crate::state::TaskCounts) -> TaskStats {
        TaskStats {
            name: name.to_owned(),
            batches: c.batches,
            received: c.received,
            programs: c.programs,
            compiled: c.compiled,
            typed: c.typed,
            passed: c.passed,
            known_mistake: crate::verify::shows_known_mistake(name).then_some(c.known_mistake),
            solved: c.passed > 0,
        }
    }
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
    /// Late or repeat submissions refused because their batch had as many
    /// kept as the run keeps: counted, not kept.
    pub refused_repeats: u64,
    /// Foreign submissions refused because their batch had as many kept as
    /// the run keeps: counted, not kept.
    pub refused_foreign: u64,
    /// Bytes of payload handed out. About four bytes make a token.
    pub payload_bytes: u64,
    /// Bytes of program text received.
    pub submission_bytes: u64,
    /// What came in from each model the proposers named, in the models'
    /// order, with those that named none first (decision D24).
    pub models: Vec<ModelStats>,
    /// Which learned library entries the programs that passed refer to.
    pub library_uptake: UptakeStats,
    /// Each task, in the run's order.
    pub tasks: Vec<TaskStats>,
}

/// Which learned library entries the programs that passed in one iteration
/// refer to. A seed entry of the run counts as a learned entry here. Only programs that passed every example are counted, as only
/// they are kept compiled. A program counts an entry once however often it
/// refers to it. Every reference in the compiled program counts, one in a
/// binding that is never used too, so this counts uptake, not calls made
/// when the program runs. Only a direct reference counts: one inside
/// another entry's body does not. Prelude functions are not learned
/// entries.
#[derive(Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct UptakeStats {
    /// Programs that passed every example.
    pub passed: u64,
    /// Of those, the programs that refer to at least one learned entry.
    pub using_library: u64,
    /// Each entry referred to, by index, in order.
    pub entries: Vec<EntryUse>,
}

/// The passed programs of one iteration that refer to one library entry.
#[derive(Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct EntryUse {
    /// The entry's index in the library.
    pub entry: u64,
    /// The entry's name now, if it has one; programs that call it by no
    /// name call it `fn_<entry>`.
    pub name: Option<String>,
    /// How many programs refer to it.
    pub programs: u64,
    /// The tasks of those programs, in order of name.
    pub tasks: Vec<String>,
}

impl UptakeStats {
    fn of(state: &RunState, uptake: LibraryUptake) -> UptakeStats {
        UptakeStats {
            passed: uptake.passed,
            using_library: uptake.using_library,
            entries: uptake
                .entries
                .into_iter()
                .map(|(entry, used)| EntryUse {
                    entry,
                    name: usize::try_from(entry)
                        .ok()
                        .and_then(|i| state.library().get(i))
                        .and_then(|e| e.name.clone()),
                    programs: used.programs,
                    tasks: used.tasks.into_iter().collect(),
                })
                .collect(),
        }
    }
}

/// What came in during one iteration from proposers that named one model.
#[derive(Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct ModelStats {
    /// The model the proposers named, or none if they named none.
    pub model: Option<String>,
    /// Submissions received, of any standing.
    pub submissions: u64,
    /// Programs in those submissions.
    pub programs: u64,
}

impl IterationStats {
    fn of(state: &RunState, refusals: &Refusals, iteration: u64) -> IterationStats {
        let tasks: Vec<TaskStats> = state
            .tasks()
            .iter()
            .map(|name| TaskStats::of(name, state.task_counts(iteration, name)))
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
            refused_repeats: refusals.iteration(iteration).repeats,
            refused_foreign: refusals.iteration(iteration).foreign,
            payload_bytes: counts.payload_bytes,
            submission_bytes: counts.submission_bytes,
            models: state
                .model_counts(iteration)
                .into_iter()
                .map(|(model, c)| ModelStats {
                    model,
                    submissions: c.submissions,
                    programs: c.programs,
                })
                .collect(),
            library_uptake: UptakeStats::of(state, state.library_uptake(iteration)),
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
    /// another submission came in meanwhile, or another server is checking
    /// one.
    pub pending_checks: usize,
    /// The batches this call left to another server, which held their
    /// claims. A batch that stays here while no check finishes may be held
    /// by a process that hangs: stop that process, and its claim goes with
    /// it.
    pub claimed_elsewhere: Vec<u64>,
    /// The most submissions this call would check at once: the workers
    /// asked for, or the default. It starts no more workers than
    /// submissions wait when it begins.
    pub workers: usize,
}

/// The result of `dc_finish_iteration`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct FinishReport {
    /// The iteration that ended.
    pub finished_iteration: u64,
    /// How many verified programs the step took.
    pub programs_used: u64,
    /// How many abstractions joined the library.
    pub new_abstractions: usize,
    /// What the Stitch run did. `ran`: it finished. `failed`: it was
    /// stopped at a limit or went wrong, so the library is unchanged.
    /// `no_programs`: there was no program to compress, so it did not run.
    /// `off`: the run does not learn a library, so it did not run.
    pub compression: String,
    /// Why the Stitch run failed, if it did.
    pub compression_error: Option<String>,
    /// The run after the step, in its next iteration.
    pub status: RunStatus,
}

/// The arguments of `dc_frontiers`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrontiersArg {
    /// The run's id.
    pub run: String,
    /// The iteration whose frontiers to report. Every finished iteration if
    /// it is left out.
    pub iteration: Option<u64>,
}

/// The result of `dc_frontiers`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct FrontiersReport {
    /// The run's id.
    pub run: String,
    /// Each finished iteration asked for, in order.
    pub iterations: Vec<IterationFrontiers>,
}

/// The frontiers the step that ended an iteration kept.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct IterationFrontiers {
    /// The iteration.
    pub iteration: u64,
    /// Each task's frontier, in the run's task order.
    pub tasks: Vec<TaskFrontier>,
}

/// One task's frontier.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct TaskFrontier {
    /// The task.
    pub task: String,
    /// The kept programs, best first, each with what it costs on the task's
    /// sweep. `cost` is left out for a run made before costs were measured.
    pub members: Vec<FrontierMember>,
}

impl FrontiersReport {
    /// The frontiers of `iteration`, or of every finished iteration.
    fn of(run: &RunId, state: &RunState, iteration: Option<u64>) -> Result<Self, String> {
        let wanted: Vec<u64> = match iteration {
            Some(i) if i >= state.iteration() => {
                return Err(format!(
                    "iteration {i} has not finished; the run is in iteration {}",
                    state.iteration()
                ))
            }
            Some(i) => vec![i],
            None => (0..state.iteration()).collect(),
        };
        let iterations = wanted
            .into_iter()
            .map(|i| {
                let kept = state.kept_frontiers(i);
                IterationFrontiers {
                    iteration: i,
                    tasks: state
                        .tasks()
                        .iter()
                        .map(|t| TaskFrontier {
                            task: t.clone(),
                            members: kept.and_then(|k| k.get(t)).cloned().unwrap_or_default(),
                        })
                        .collect(),
                }
            })
            .collect();
        Ok(FrontiersReport {
            run: run.as_str().to_owned(),
            iterations,
        })
    }
}

/// The arguments of `dc_library`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LibraryArg {
    /// The run's id.
    pub run: String,
    /// The iteration whose library to report: the entries its programs
    /// could call, added by the iterations before it. The current
    /// iteration if it is left out, which is the whole library.
    pub iteration: Option<u64>,
}

/// The arguments of `dc_diversity`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DiversityArg {
    /// The run's id.
    pub run: String,
    /// The iteration to report. Every iteration up to the current one if it
    /// is left out.
    pub iteration: Option<u64>,
    /// Whether to run each distinct program on its task's growth sweep and
    /// name its growth class. Off if it is left out.
    pub growth: Option<bool>,
    /// With growth on, no program's run starts after this many seconds.
    /// 600 if it is left out; at most 3600.
    pub deadline_seconds: Option<u64>,
}

/// The default and the most time `dc_diversity` gives to measuring growth,
/// in seconds.
const DIVERSITY_DEADLINE_SECONDS: u64 = 600;
const MAX_DIVERSITY_DEADLINE_SECONDS: u64 = 3600;

/// The result of `dc_library`.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LibraryReport {
    /// The run's id.
    pub run: String,
    /// The iteration whose library this is.
    pub iteration: u64,
    /// Each entry, by index.
    pub entries: Vec<LibraryItem>,
}

/// One library entry.
#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct LibraryItem {
    /// The entry's index. A program calls it by its name, or by
    /// `fn_<index>` if it has none.
    pub index: u64,
    /// The number of parameters.
    pub arity: u32,
    /// The closed term, as an s-expression of core syntax. `lib_<i>` in it
    /// is entry `i`, or a prelude function if `i` is 4294901760 or more.
    pub term: String,
    /// The name a proposer gave it, if any.
    pub name: Option<String>,
    /// What it does, in one line, as a proposer wrote it, if any.
    pub doc: Option<String>,
    /// The iteration whose end added it. Programs can call it from the next
    /// iteration on.
    pub iteration_added: u64,
}

impl LibraryReport {
    /// The library of `iteration`, or of the current iteration.
    fn of(run: &RunId, state: &RunState, iteration: Option<u64>) -> Result<Self, String> {
        let current = state.iteration();
        let iteration = match iteration {
            Some(i) if i > current => {
                return Err(format!(
                    "iteration {i} has not started; the run is in iteration {current}"
                ))
            }
            Some(i) => i,
            None => current,
        };
        let entries = state
            .library()
            .iter()
            .zip(0u64..)
            .filter_map(|(entry, index)| {
                let added = state.added_in(index)?;
                (added < iteration).then(|| LibraryItem {
                    index,
                    arity: entry.arity,
                    term: entry.term.clone(),
                    name: entry.name.clone(),
                    doc: entry.doc.clone(),
                    iteration_added: added,
                })
            })
            .collect();
        Ok(LibraryReport {
            run: run.as_str().to_owned(),
            iteration,
            entries,
        })
    }
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
    /// How `dc_finish_iteration` starts its Stitch worker.
    worker: Arc<WorkerStart>,
    /// What stops the long calls of this server: Stitch runs and measuring.
    stops: Arc<Stops>,
}

/// The cancel flags of the `dc_finish_iteration` and `dc_diversity` calls
/// that are running.
#[derive(Debug, Default)]
struct Stops {
    /// Whether the server shut down, and the flag of each running call.
    /// Both under one lock, so that a call that starts while the server
    /// shuts down is stopped.
    calls: Mutex<(bool, Vec<Arc<AtomicBool>>)>,
}

impl Stops {
    fn calls(&self) -> std::sync::MutexGuard<'_, (bool, Vec<Arc<AtomicBool>>)> {
        // The lock guards plain values: one that a panic left is still good.
        self.calls.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The cancel flag of a new call. It is set already if the server shut
    /// down.
    fn call(self: &Arc<Stops>) -> CallStop {
        let mut calls = self.calls();
        let flag = Arc::new(AtomicBool::new(calls.0));
        calls.1.push(Arc::clone(&flag));
        CallStop {
            flag,
            stops: Arc::clone(self),
        }
    }

    /// Sets the flag of every call, running or still to come.
    fn stop_all(&self) {
        let mut calls = self.calls();
        calls.0 = true;
        for flag in &calls.1 {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

/// The cancel flag of one `dc_finish_iteration` call. Dropping it sets the
/// flag: a call whose future is dropped stops its Stitch run.
#[derive(Debug)]
struct CallStop {
    flag: Arc<AtomicBool>,
    stops: Arc<Stops>,
}

impl CallStop {
    fn stop(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
}

impl Drop for CallStop {
    fn drop(&mut self) {
        self.stop();
        self.stops
            .calls()
            .1
            .retain(|flag| !Arc::ptr_eq(flag, &self.flag));
    }
}

impl DcServer {
    /// A server for the runs in `store`, whose `dc_finish_iteration` starts
    /// its Stitch worker as `worker` says.
    pub fn new(store: Store, worker: WorkerStart) -> DcServer {
        DcServer {
            store: Arc::new(store),
            worker: Arc::new(worker),
            stops: Arc::default(),
        }
    }

    /// Stops every `dc_finish_iteration` call of this server and its
    /// clones, running or still to come: its Stitch run stops, and the call
    /// fails as interrupted and writes nothing, unless it had already read
    /// its cancel flag for the commit (`crate::iterate`). Call it when the
    /// client is gone, so that a call does not go on for as long as its
    /// Stitch run may take.
    pub fn shut_down(&self) {
        self.stops.stop_all();
    }
}

#[cfg(test)]
impl DcServer {
    /// How many `dc_finish_iteration` and `dc_diversity` calls are running.
    pub(crate) fn finish_calls_running(&self) -> usize {
        self.stops.calls().1.len()
    }

    /// `dc_finish_iteration`, for tests in other modules.
    pub(crate) async fn finish_iteration_for_test(
        &self,
        arg: RunArg,
        cancelled: impl Future<Output = ()>,
    ) -> Result<FinishReport, String> {
        self.finish(arg, cancelled).await.map(|Json(report)| report)
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
        crate::blocking::spawn(move || work(&store))
            .await
            .map_err(|e| format!("the tool call did not finish: {e}"))?
            .map(Json)
    }

    /// `dc_finish_iteration`. When `cancelled` is done, or this future is
    /// dropped, the call's cancel flag is set: its Stitch run stops, and
    /// the call writes nothing unless its commit had already read the flag
    /// (`crate::iterate`).
    async fn finish(
        &self,
        arg: RunArg,
        cancelled: impl Future<Output = ()>,
    ) -> Result<Json<FinishReport>, String> {
        let run = run_id(&arg.run)?;
        let worker = Arc::clone(&self.worker);
        let stop = self.stops.call();
        let stopping = Arc::clone(&stop.flag);
        let work = self.blocking(move |store| {
            let done = finish_iteration_in_worker(store, &run, &worker, &stopping)
                .map_err(|e| e.to_string())?;
            let (compression, compression_error) = match &done.report.compression {
                CompressionReport::Ran { .. } => ("ran", None),
                CompressionReport::Failed { error, .. } => ("failed", Some(error.clone())),
                CompressionReport::NoPrograms { .. } => ("no_programs", None),
                CompressionReport::Off => ("off", None),
            };
            // The step is committed. If this read fails, say so, not that
            // the step failed.
            let status = store
                .read_with_refusals(&run, |state, refusals| RunStatus::of(&run, state, refusals))
                .map(|status| status.compared(store))
                .map_err(|e| {
                    format!(
                        "iteration {} ended, but the run's status could not be read: {e}",
                        done.iteration
                    )
                })?;
            Ok(FinishReport {
                finished_iteration: done.iteration,
                programs_used: done.used,
                new_abstractions: done.new_abstractions,
                compression: compression.to_owned(),
                compression_error,
                status,
            })
        });
        tokio::pin!(work);
        tokio::pin!(cancelled);
        tokio::select! {
            done = &mut work => done,
            () = &mut cancelled => {
                stop.stop();
                // The step ends as interrupted, or it was past stopping.
                work.await
            }
        }
    }
}

#[tool_router(server_handler)]
impl DcServer {
    /// Makes a run.
    #[tool(
        name = "dc_create_run",
        description = "Create a DreamCounter run at iteration 0, with an empty library and an \
                       empty frontier for each task. Fails if a run of that id exists, or if \
                       the run named in compared_with has other settings that fix proposals. \
                       Returns the new run's status."
    )]
    async fn create_run(
        &self,
        Parameters(arg): Parameters<CreateRun>,
    ) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        let compared_with = match &arg.compared_with {
            None => None,
            Some(text) => {
                let other = run_id(text).map_err(|e| format!("compared_with: {e}"))?;
                if other == run {
                    return Err("a run cannot be compared with itself".to_owned());
                }
                Some(other)
            }
        };
        let library_learning = arg.library_learning.unwrap_or(true);
        // Refused here, in the tool's own names and units.
        if !library_learning {
            for (name, given) in [
                ("stitch_iterations", arg.stitch_iterations.is_some()),
                ("stitch_max_arity", arg.stitch_max_arity.is_some()),
                ("stitch_wall_seconds", arg.stitch_wall_seconds.is_some()),
                ("stitch_memory_mib", arg.stitch_memory_mib.is_some()),
            ] {
                if given {
                    return Err(format!(
                        "{name} is a Stitch setting, and a run with library_learning off \
                         runs no Stitch"
                    ));
                }
            }
        }
        let refuse_bucketing = arg.refuse_bucketing.unwrap_or(false);
        let seed_entries: Vec<crate::state::LibraryEntry> = arg
            .seed_entries
            .unwrap_or_default()
            .into_iter()
            .map(|e| crate::state::LibraryEntry {
                arity: e.arity,
                term: e.term,
                name: Some(e.name),
                doc: Some(e.doc),
            })
            .collect();
        if !library_learning {
            for (name, given) in [
                ("refuse_bucketing", refuse_bucketing),
                ("seed_entries", !seed_entries.is_empty()),
            ] {
                if given {
                    return Err(format!(
                        "{name} changes the library, and a run with library_learning off \
                         has none"
                    ));
                }
            }
        }
        if arg.max_memory_mib == Some(0) {
            return Err("max_memory_mib must be at least 1".to_owned());
        }
        let most_mib = MAX_EVALUATION_MEMORY >> 20;
        if arg.max_memory_mib.is_some_and(|m| m > most_mib) {
            return Err(format!("max_memory_mib must be at most {most_mib}"));
        }
        if arg.stitch_iterations == Some(0) {
            return Err("stitch_iterations must be at least 1".to_owned());
        }
        if arg.stitch_wall_seconds == Some(0) {
            return Err("stitch_wall_seconds must be at least 1".to_owned());
        }
        let least_mib = MIN_STITCH_MEMORY_BYTES >> 20;
        if arg.stitch_memory_mib.is_some_and(|m| m < least_mib) {
            return Err(format!("stitch_memory_mib must be at least {least_mib}"));
        }
        self.blocking(move |store| {
            let other = match &compared_with {
                None => None,
                Some(other) => Some(
                    store
                        .read(other, |s| (s.config().clone(), s.tasks().to_vec()))
                        .map_err(|e| format!("compared_with: {e}"))?,
                ),
            };
            let known = known_tasks();
            let tasks = match arg.tasks {
                None => other
                    .as_ref()
                    .map_or(known, |(_, other_tasks)| other_tasks.clone()),
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
            let defaults = StitchSettings::default();
            let config = RunConfig {
                autodoc: arg.autodoc.unwrap_or(false),
                library_learning,
                max_memory: arg.max_memory_mib.map_or_else(
                    || {
                        other
                            .as_ref()
                            .map_or(DEFAULT_MAX_MEMORY, |(c, _)| c.max_memory)
                    },
                    |m| m.saturating_mul(1 << 20),
                ),
                arm: Arm::Blind,
                seed: arg.seed,
                proposals_per_task: arg.proposals_per_task,
                programs_per_batch: arg
                    .programs_per_batch
                    .or(other.as_ref().map(|(c, _)| c.programs_per_batch))
                    .unwrap_or(DEFAULT_PROGRAMS_PER_BATCH),
                lease_ms: arg.lease_seconds.map_or_else(
                    || {
                        other
                            .as_ref()
                            .map_or(DEFAULT_LEASE_SECONDS.saturating_mul(1000), |(c, _)| {
                                c.lease_ms
                            })
                    },
                    |s| s.saturating_mul(1000),
                ),
                max_unchecked_per_batch: arg
                    .max_unchecked_per_batch
                    .or(other.as_ref().map(|(c, _)| c.max_unchecked_per_batch))
                    .unwrap_or(DEFAULT_MAX_UNCHECKED_PER_BATCH),
                frontier_size: arg.frontier_size,
                stitch: StitchSettings {
                    iterations: arg.stitch_iterations.unwrap_or(defaults.iterations),
                    max_arity: arg.stitch_max_arity.unwrap_or(defaults.max_arity),
                    wall_ms: arg
                        .stitch_wall_seconds
                        .map_or(defaults.wall_ms, |s| s.saturating_mul(1000)),
                    memory_bytes: arg
                        .stitch_memory_mib
                        .map_or(defaults.memory_bytes, |m| m.saturating_mul(1 << 20)),
                },
                prelude: shipped_prelude(),
                compared_with: compared_with.as_ref().map(|r| r.as_str().to_owned()),
                approach_hints: arg
                    .approach_hints
                    .or(other.as_ref().map(|(c, _)| c.approach_hints))
                    .unwrap_or(false),
                show_received: arg
                    .show_received
                    .or(other.as_ref().map(|(c, _)| c.show_received))
                    .unwrap_or(false),
                refuse_bucketing,
                seed_entries,
            };
            if let (Some(name), Some((other_config, other_tasks))) = (&compared_with, &other) {
                let found = differences(&config, &tasks, other_config, other_tasks);
                if !found.is_empty() {
                    return Err(format!(
                        "the run must share these settings with run {:?}, which it is \
                         compared with: {}",
                        name.as_str(),
                        found.join("; ")
                    ));
                }
            }
            let state = store
                .create_run(&run, config, tasks)
                .map_err(|e| e.to_string())?;
            Ok(RunStatus::of(&run, &state, &Refusals::default()).compared(store))
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
                .read_with_refusals(&run, |state, refusals| RunStatus::of(&run, state, refusals))
                .map(|status| status.compared(store))
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
            let noted: Result<RunStatus, StoreError> = store.commit_then_with_refusals(
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
                |(), state, refusals| RunStatus::of(&run, state, &refusals.read()),
            );
            noted
                .map(|status| status.compared(store))
                .map_err(|e| e.to_string())
        })
        .await
    }

    /// Reports what proposers sent in each iteration.
    #[tool(
        name = "dc_proposal_stats",
        description = "Report, for each iteration of a DreamCounter run, what proposers sent: \
                       for each task the batches issued, the programs checked, how many \
                       compiled, typed and passed their examples, and whether the task was \
                       solved; the bytes of payload and program text; and the submissions \
                       and programs from each model the proposers named; and which learned \
                       library entries the programs that passed refer to, so the library's uptake \
                       can be counted."
    )]
    async fn proposal_stats(
        &self,
        Parameters(arg): Parameters<RunArg>,
    ) -> Result<Json<ProposalStats>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            store
                .read_with_refusals(&run, |state, refusals| ProposalStats {
                    run: run.as_str().to_owned(),
                    iterations: (0..=state.iteration())
                        .map(|i| IterationStats::of(state, refusals, i))
                        .collect(),
                })
                .map_err(|e| e.to_string())
        })
        .await
    }

    /// Reports each finished iteration's frontiers and what they cost.
    #[tool(
        name = "dc_frontiers",
        description = "Report the frontiers of a DreamCounter run: for each finished iteration, \
                       or the one given, each task's kept programs, best first, as core text, \
                       each with its cost on the task's growth sweep. The cost is measured when \
                       the iteration ends: the step counts of each seed at each size, their \
                       median, and estimator C's growth class (or why it names none). A program \
                       is incorrect if it fails a generated input, and not measured if the \
                       time for measuring ran out. Read only."
    )]
    async fn frontiers(
        &self,
        Parameters(arg): Parameters<FrontiersArg>,
    ) -> Result<Json<FrontiersReport>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            store
                .read(&run, |state| {
                    FrontiersReport::of(&run, state, arg.iteration)
                })
                .map_err(|e| e.to_string())?
        })
        .await
    }

    /// Reports a run's library.
    #[tool(
        name = "dc_library",
        description = "Report the library of a DreamCounter run: for the current iteration, or \
                       the one given, each abstraction its programs could call, with its \
                       index, arity, term as core text, the name and one-line doc a proposer \
                       gave it (if any), and the iteration whose end added it. \
                       dc_proposal_stats says which entries each iteration's passed programs \
                       call. Read only."
    )]
    async fn library(
        &self,
        Parameters(arg): Parameters<LibraryArg>,
    ) -> Result<Json<LibraryReport>, String> {
        let run = run_id(&arg.run)?;
        self.blocking(move |store| {
            store
                .read(&run, |state| LibraryReport::of(&run, state, arg.iteration))
                .map_err(|e| e.to_string())?
        })
        .await
    }

    /// Reports how varied each task's passed programs are.
    #[tool(
        name = "dc_diversity",
        description = "Report how varied the programs that passed each task of a DreamCounter \
                       run are, for each iteration up to the current one, or the one given: \
                       programs passed, distinct core terms (names of variables do not \
                       count), distinct terms once library and prelude calls are written out, \
                       the mean Jaccard distance between the compound subterms of each pair, \
                       and each distinct program with how many passed programs it stands for. \
                       With growth true it also runs each distinct program on its task's growth \
                       sweep and gives estimator C's verdict and the number of distinct \
                       classes, reusing the cost of a kept frontier program with the same \
                       expansion; no program's run starts after deadline_seconds (default 600, \
                       at most 3600). Read only."
    )]
    async fn diversity(
        &self,
        Parameters(arg): Parameters<DiversityArg>,
        context: RequestContext<RoleServer>,
    ) -> Result<Json<DiversityReport>, String> {
        let run = run_id(&arg.run)?;
        let seconds = arg.deadline_seconds.unwrap_or(DIVERSITY_DEADLINE_SECONDS);
        if seconds > MAX_DIVERSITY_DEADLINE_SECONDS {
            return Err(format!(
                "deadline_seconds is {seconds}; it must be at most {MAX_DIVERSITY_DEADLINE_SECONDS}"
            ));
        }
        let growth = arg.growth.unwrap_or(false);
        let stop = self.stops.call();
        let stopping = Arc::clone(&stop.flag);
        let work = self.blocking(move |store| {
            // The state first, then the journal: the journal holds at least
            // what the state does. Measuring runs on the copy, without the
            // run's lock.
            let state = store.state(&run).map_err(|e| e.to_string())?;
            let passed = passed_programs(store, &run).map_err(|e| e.to_string())?;
            let limits = growth.then(|| GrowthLimits {
                deadline: Instant::now() + Duration::from_secs(seconds),
                cancel: &stopping,
            });
            diversity_report(&run, &state, &passed, arg.iteration, limits)
        });
        let cancelled = context.ct.cancelled_owned();
        tokio::pin!(work);
        tokio::pin!(cancelled);
        tokio::select! {
            done = &mut work => done,
            () = &mut cancelled => {
                stop.stop();
                work.await
            }
        }
    }

    /// Checks the submissions of a run that wait to be checked.
    #[tool(
        name = "dc_check_pending",
        description = "Check every submission of a DreamCounter run that waits to be checked, \
                       and wait until they are done. Several are checked at once, each on a \
                       thread of its own: workers of them (default: one for each core, as far \
                       as the available memory holds them at the run's max_memory_mib each). \
                       A submission that another server is checking is left to it. Proposer \
                       servers check in the background, one at a time; use this when one \
                       stopped before it finished, or to see why checks fail. Returns how \
                       many it checked, how many still wait, the batches another server is \
                       checking, and the most it would check at once. The default counts \
                       only the memory free when the call starts: a caller that runs \
                       several calls at once should pass workers."
    )]
    async fn check_pending(
        &self,
        Parameters(arg): Parameters<CheckPendingArg>,
    ) -> Result<Json<CheckReport>, String> {
        let run = run_id(&arg.run)?;
        if let Some(workers) = arg.workers {
            if !(1..=MAX_CHECK_WORKERS).contains(&workers) {
                return Err(format!(
                    "workers is {workers}; it must be from 1 to {MAX_CHECK_WORKERS}"
                ));
            }
        }
        self.blocking(move |store| {
            let workers = match arg.workers {
                Some(workers) => workers,
                None => {
                    let max_memory = store
                        .read(&run, |state| state.config().max_memory)
                        .map_err(|e| e.to_string())?;
                    let cores = std::thread::available_parallelism().map_or(1, usize::from);
                    default_check_workers(cores, available_memory(), max_memory)
                }
            };
            // This server serves many runs, so it keeps no verifier between
            // calls; one call builds one for each library length it meets.
            let verifiers = VerifierCache::default();
            let done = check_pending_parallel(store, &run, &verifiers, workers)
                .map_err(|e| e.to_string())?;
            let pending_checks = store
                .read(&run, |state| state.pending_checks().len())
                .map_err(|e| e.to_string())?;
            Ok(CheckReport {
                run: run.as_str().to_owned(),
                checked: done.checked,
                pending_checks,
                claimed_elsewhere: done.claimed_elsewhere,
                workers,
            })
        })
        .await
    }

    /// Ends a run's current iteration.
    #[tool(
        name = "dc_finish_iteration",
        description = "End the current iteration of a DreamCounter run, once every task has its \
                       programs and all are checked (quota_in in dc_run_status). It keeps each \
                       task's best programs as its frontier, runs Stitch on them to find new \
                       abstractions for the library, measures each kept program on its task's \
                       growth sweep (dc_frontiers reports it), and starts the next iteration, \
                       all in one change to the run. The call first measures every program's \
                       size, then waits for Stitch for at most the run's stitch.wall_seconds, \
                       then measures the kept programs' cost, starting no program's run after \
                       10 minutes (a program not reached is recorded as not measured), then \
                       checks what \
                       proposers submitted meanwhile, up to 8 times. If Stitch is stopped at a \
                       limit the iteration still ends, with the library unchanged. If you \
                       cancel the call before its change is written, Stitch stops and the run \
                       is not changed. A cancelled call gets no reply, and a cancel may come \
                       too late, so read dc_run_status afterwards. Returns what the step did and the run's status; the full \
                       report is in the run's log."
    )]
    async fn finish_iteration(
        &self,
        Parameters(arg): Parameters<RunArg>,
        context: RequestContext<RoleServer>,
    ) -> Result<Json<FinishReport>, String> {
        self.finish(arg, context.ct.cancelled_owned()).await
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

/// The arguments of `dc_submit_naming`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SubmitNamingArgs {
    /// Your id: the one you gave `dc_next_naming_job` for this job.
    pub worker: String,
    /// The job's number, from `dc_next_naming_job`.
    pub job: u64,
    /// The job's token, from `dc_next_naming_job`.
    pub token: String,
    /// The function's new name: a lower-case letter or '_', then letters,
    /// digits, '_' or '\''; 3 to 40 bytes; not a keyword, `solve`, `fn_`
    /// and digits, a primitive or prelude function, or another library name.
    pub name: String,
    /// One line of printable ASCII, at most 400 bytes, that says what the
    /// function does.
    pub doc: String,
}

/// Which of a proposer's two jobs a [`ProposerServer`] serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposerMode {
    /// Batches: `dc_next_batch` and `dc_submit` (decision D11).
    Batch,
    /// Naming new abstractions: `dc_next_naming_job` and `dc_submit_naming`
    /// (decision D23).
    Naming,
}

/// The MCP server a proposer connects to (decision D11). It serves one run
/// in one [`ProposerMode`], and lists and takes only that mode's two tools.
/// A call to any other tool, the other mode's included, is refused as a
/// tool that is not there. A proposer cannot list runs or see a run's
/// results.
///
/// Either mode checks the run's pending submissions in the background.
#[derive(Clone, Debug)]
pub struct ProposerServer {
    store: Arc<Store>,
    run: RunId,
    /// The tools of the server's mode, the only ones it lists and takes.
    tools: ToolRouter<ProposerServer>,
    clock: Arc<dyn Clock>,
    checker: Arc<Checker>,
    /// The verifier of the run's latest library, for batches and checks.
    verifiers: Arc<VerifierCache>,
}

/// Runs the checks of pending submissions, one at a time in each process.
#[derive(Debug, Default)]
struct Checker {
    /// Held while checks run.
    running: Mutex<()>,
    /// Set when there may be new work.
    dirty: AtomicBool,
    /// Taken by [`Checker::signal`] and [`Checker::back_off`] around
    /// `woken`, so a signal cannot fall between a waiter's look at `dirty`
    /// and its wait.
    wake: Mutex<()>,
    /// Notified when work is signalled, to end a wait after an error.
    woken: Condvar,
    /// How many checks have failed, for tests.
    #[cfg(test)]
    failures: std::sync::atomic::AtomicUsize,
}

impl Checker {
    /// Signals that there may be new work, and wakes a thread that waits
    /// after an error to try again.
    fn signal(&self) {
        self.dirty.store(true, Ordering::SeqCst);
        // Taking the lock waits for a waiter that has seen `dirty` clear to
        // start its wait, so the notification reaches it.
        drop(self.wake.lock().unwrap_or_else(PoisonError::into_inner));
        self.woken.notify_all();
    }

    /// Signals work only if no thread is running checks, or waiting after
    /// an error to try again, and says whether it did. It sets `dirty` while
    /// it holds `running`, so a thread that takes `running` later clears
    /// `dirty` before it checks, and a wait after that check is not ended
    /// by this signal.
    fn signal_if_idle(&self) -> bool {
        let running = match self.running.try_lock() {
            Ok(running) => running,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        self.dirty.store(true, Ordering::SeqCst);
        drop(running);
        true
    }

    /// Waits `wait`, or less if work is signalled meanwhile, also before
    /// this call. Returns whether a signal ended the wait. It holds no lock
    /// of the run while it waits.
    fn back_off(&self, wait: Duration) -> bool {
        let guard = self.wake.lock().unwrap_or_else(PoisonError::into_inner);
        let (_guard, waited) = self
            .woken
            .wait_timeout_while(guard, wait, |()| !self.dirty.load(Ordering::SeqCst))
            .unwrap_or_else(PoisonError::into_inner);
        !waited.timed_out()
    }

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
    /// A server for proposers of `run` in `store`, in `mode`, with leases
    /// timed by `clock`.
    pub fn new(
        store: Store,
        run: RunId,
        mode: ProposerMode,
        clock: Arc<dyn Clock>,
    ) -> ProposerServer {
        ProposerServer {
            store: Arc::new(store),
            run,
            tools: match mode {
                ProposerMode::Batch => Self::batch_tools(),
                ProposerMode::Naming => Self::naming_tools(),
            },
            clock,
            checker: Arc::default(),
            verifiers: Arc::default(),
        }
    }

    /// Checks the run's pending submissions on a thread of its own. Call it
    /// after a submission, and once when the server starts, so a submission
    /// left unchecked by a server that stopped is checked. Must be called
    /// within a Tokio runtime.
    ///
    /// After an error the checks wait, 5 s at first and twice as long after
    /// each wait that ran out, up to 300 s, and then try again. A call here
    /// ends such a wait at once. It does not shorten the next wait: while the
    /// cause lasts, each call costs one more try. Submissions are few, so
    /// these tries are too; see [`ProposerServer::check_if_idle`] for calls
    /// that come as often as proposers poll.
    pub fn check_in_background(&self) {
        self.checker.signal();
        self.drain_in_background();
    }

    /// Checks the run's pending submissions on a thread of its own, unless
    /// this server's checks are running or waiting after an error to try
    /// again. Call it when a proposer is told to wait, so a check whose
    /// claimant stopped is taken up. It does not end a wait after an error,
    /// so proposers that poll do not make a lasting error cost a try each
    /// time. If the checks are just finishing, it does nothing; the
    /// proposer's next poll starts them. Must be called within a Tokio
    /// runtime.
    pub fn check_if_idle(&self) {
        if self.checker.signal_if_idle() {
            self.drain_in_background();
        }
    }

    /// Runs the checker's drain on a thread of its own.
    fn drain_in_background(&self) {
        let store = Arc::clone(&self.store);
        let run = self.run.clone();
        let checker = Arc::clone(&self.checker);
        let verifiers = Arc::clone(&self.verifiers);
        crate::blocking::spawn(move || {
            let mut retry = CHECK_RETRY_FIRST;
            checker.drain(
                || match catch_check(|| {
                    crate::blocking::releasing(|| check_pending(&store, &run, &verifiers))
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                }) {
                    Ok(_) => retry = CHECK_RETRY_FIRST,
                    Err(e) => {
                        #[cfg(test)]
                        checker.failures.fetch_add(1, Ordering::SeqCst);
                        // Standard error may be closed.
                        let _ = writeln!(
                            std::io::stderr(),
                            "dc-mcp: the submissions of run {run} could not be checked: {e}; \
                             trying again in {} s, or when a submission comes in",
                            retry.as_secs()
                        );
                        // The thread holds `running`, so no other check
                        // starts, but a new submission ends the wait.
                        if !checker.back_off(retry) {
                            retry = (retry * 2).min(CHECK_RETRY_MAX);
                        }
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
        let done = crate::blocking::spawn(move || work(&store, &run, clock.now_ms()))
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

/// The tools a server in its mode lists and takes. Only these are listed,
/// and a call to any other is refused as a tool that is not there.
#[tool_handler(router = self.tools)]
impl ServerHandler for ProposerServer {}

#[tool_router(router = batch_tools)]
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
        let verifiers = Arc::clone(&self.verifiers);
        let reply = self
            .blocking(move |store, run, now| next_batch(store, run, &arg.worker, now, &verifiers))
            .await;
        // A proposer that waits may wait for a check whose claimant stopped
        // (see `check_pending`); a check here takes it up. A claim that is
        // held is skipped, so this does not wait for other servers' checks.
        if reply.as_ref().is_ok_and(|r| r.0.wait.is_some()) {
            self.check_if_idle();
        }
        reply
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

#[tool_router(router = naming_tools)]
impl ProposerServer {
    /// Hands out a naming job (decision D23).
    #[tool(
        name = "dc_next_naming_job",
        description = "Get a function of the run's library to name and document. The prompt \
                       shows the function and says how to answer. If there is none now, \
                       `job` is null and `wait` says why and when to ask again."
    )]
    async fn next_naming_job(
        &self,
        Parameters(arg): Parameters<NextBatchArgs>,
    ) -> Result<Json<NextNamingJob>, String> {
        let verifiers = Arc::clone(&self.verifiers);
        self.blocking(move |store, run, now| {
            next_naming_job(store, run, &arg.worker, now, &verifiers)
        })
        .await
    }

    /// Takes in a name and doc for a naming job (decision D23).
    #[tool(
        name = "dc_submit_naming",
        description = "Name and document the function of a naming job, with the job's number \
                       and token from dc_next_naming_job. The reply gives the name the function \
                       now has: yours, or its old one if yours broke a rule."
    )]
    async fn submit_naming(
        &self,
        Parameters(arg): Parameters<SubmitNamingArgs>,
    ) -> Result<Json<NamingReceipt>, String> {
        self.blocking(move |store, run, _now| {
            submit_naming(
                store,
                run,
                &arg.worker,
                arg.job,
                &arg.token,
                &arg.name,
                &arg.doc,
            )
        })
        .await
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

    #[test]
    fn a_wait_after_an_error_ends_on_a_signal_and_not_otherwise() {
        let checker = Arc::new(Checker::default());
        let started = std::time::Instant::now();
        assert!(!checker.back_off(Duration::from_millis(50)));
        assert!(started.elapsed() >= Duration::from_millis(50));

        // A signal that came before the wait ends it at once. If it did not,
        // the wait runs out and the test fails.
        checker.signal();
        assert!(checker.back_off(Duration::from_secs(20)));

        // A signal during the wait ends it.
        checker.dirty.store(false, Ordering::SeqCst);
        let signaller = Arc::clone(&checker);
        let signalled = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            signaller.signal();
        });
        assert!(checker.back_off(Duration::from_secs(20)));
        signalled.join().unwrap();
    }

    /// Every setting that fixes the proposals is compared, including the
    /// two that `dc_create_run` cannot make differ by itself: the prelude,
    /// and a lease that differs by less than a second.
    #[test]
    fn task_stats_hold_the_known_mistake_only_for_a_task_that_shows_one() {
        let counts = crate::state::TaskCounts {
            programs: 5,
            known_mistake: 3,
            ..Default::default()
        };
        let json = |name: &str| serde_json::to_value(TaskStats::of(name, counts)).unwrap();
        assert_eq!(json("chains/heaviest")["known_mistake"], 3);
        let zero = serde_json::to_value(TaskStats::of("chains/heaviest", Default::default()));
        assert_eq!(zero.unwrap()["known_mistake"], 0);
        // A task that names no mistake has no count, not a count of 0.
        let merge = json("collapse/merge");
        assert_eq!(merge["programs"], 5);
        assert!(merge.get("known_mistake").is_none(), "{merge}");
    }

    #[test]
    fn differences_name_every_setting_that_fixes_proposals() {
        let base = RunConfig {
            max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
            autodoc: false,
            arm: Arm::Blind,
            library_learning: true,
            seed: 3,
            proposals_per_task: 2,
            programs_per_batch: 2,
            lease_ms: 60_000,
            max_unchecked_per_batch: 3,
            frontier_size: 5,
            stitch: crate::state::StitchSettings::default(),
            prelude: 9,
            compared_with: None,
            approach_hints: false,
            show_received: false,
            refuse_bucketing: false,
            seed_entries: Vec::new(),
        };
        let tasks = vec!["a".to_owned(), "b".to_owned()];
        // What does not fix the proposals may differ.
        let free = RunConfig {
            library_learning: false,
            stitch: crate::state::StitchSettings {
                iterations: 9,
                ..crate::state::StitchSettings::default()
            },
            compared_with: Some("main".into()),
            ..base.clone()
        };
        assert_eq!(
            differences(&free, &tasks, &base, &tasks),
            Vec::<String>::new()
        );
        for (changed, said) in [
            (
                RunConfig {
                    seed: 4,
                    ..base.clone()
                },
                "seed is 4, not 3",
            ),
            (
                RunConfig {
                    proposals_per_task: 1,
                    ..base.clone()
                },
                "proposals_per_task is 1, not 2",
            ),
            (
                RunConfig {
                    frontier_size: 6,
                    ..base.clone()
                },
                "frontier_size is 6, not 5",
            ),
            (
                RunConfig {
                    programs_per_batch: 1,
                    ..base.clone()
                },
                "programs_per_batch is 1, not 2",
            ),
            (
                RunConfig {
                    lease_ms: 59_500,
                    ..base.clone()
                },
                "lease_seconds is 59.500, not 60",
            ),
            (
                RunConfig {
                    approach_hints: true,
                    ..base.clone()
                },
                "approach_hints is true, not false",
            ),
            (
                RunConfig {
                    show_received: true,
                    refuse_bucketing: false,
                    seed_entries: Vec::new(),
                    ..base.clone()
                },
                "show_received is true, not false",
            ),
            (
                RunConfig {
                    prelude: 8,
                    ..base.clone()
                },
                "prelude differs: the run compared with was made with another prelude than \
                 this server's",
            ),
        ] {
            assert_eq!(differences(&changed, &tasks, &base, &tasks), vec![said]);
        }
        let kept = RunConfig {
            max_unchecked_per_batch: 7,
            ..base.clone()
        };
        assert_eq!(
            differences(&kept, &tasks, &base, &tasks),
            vec!["max_unchecked_per_batch is 7, not 3"]
        );
        let reversed = vec!["b".to_owned(), "a".to_owned()];
        assert_eq!(
            differences(&base, &reversed, &base, &tasks),
            vec!["tasks are [b, a], not [a, b]"]
        );
    }

    /// `dc_proposal_stats` reports, for each iteration, what came in from
    /// each model the proposers named (D24 accounting).
    #[test]
    fn proposal_stats_count_submissions_by_model() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let run = RunId::new("r").unwrap();
        store
            .create_run(
                &run,
                RunConfig {
                    max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
                    autodoc: false,
                    arm: Arm::Blind,
                    library_learning: true,
                    seed: 3,
                    proposals_per_task: 8,
                    programs_per_batch: 2,
                    lease_ms: 1_000_000,
                    max_unchecked_per_batch: 3,
                    frontier_size: 5,
                    stitch: crate::state::StitchSettings::default(),
                    prelude: crate::verify::shipped_prelude(),
                    compared_with: None,
                    approach_hints: false,
                    show_received: false,
                    refuse_bucketing: false,
                    seed_entries: Vec::new(),
                },
                vec!["collapse/merge".into()],
            )
            .unwrap();
        let verifiers = VerifierCache::default();
        let models = [Some("model-b"), None, Some("model-a"), Some("model-b")];
        for (i, model) in models.iter().enumerate() {
            let worker = format!("w{i}");
            let sent = next_batch(&store, &run, &worker, 0, &verifiers)
                .unwrap()
                .batch
                .unwrap();
            let programs = vec!["let solve = (".to_owned(); i % 2 + 1];
            submit(
                &store,
                &run,
                sent.batch,
                &worker,
                &sent.token,
                programs,
                model.map(str::to_owned),
                1,
            )
            .unwrap();
        }
        let stats = store
            .read_with_refusals(&run, |state, refusals| {
                IterationStats::of(state, refusals, 0)
            })
            .unwrap();
        let row = |model: Option<&str>, submissions, programs| ModelStats {
            model: model.map(str::to_owned),
            submissions,
            programs,
        };
        assert_eq!(
            stats.models,
            vec![
                row(None, 1, 2),
                row(Some("model-a"), 1, 1),
                row(Some("model-b"), 2, 3),
            ]
        );
        assert_eq!(stats.submissions, 4);
    }

    /// A proposer server of a run with one submission waiting to be checked.
    struct Fixture {
        dir: tempfile::TempDir,
        server: ProposerServer,
        /// Another store on the run, which waits as long as a server does
        /// for the run's lock, for the test's own reads.
        reader: Store,
        /// Taken out and shut down first when the fixture is dropped, so a
        /// background check that waits does not hold up the test, and the
        /// directory is not removed under it.
        runtime: Option<tokio::runtime::Runtime>,
    }

    /// How long the server's store in a [`Fixture`] waits for the run's
    /// lock: long enough to win it from the test's reads, short enough
    /// to fail soon when the test holds it.
    const FIXTURE_LOCK_WAIT: Duration = Duration::from_secs(1);

    impl Fixture {
        fn new() -> Fixture {
            let dir = tempfile::TempDir::new().unwrap();
            let store = Store::open_with_lock_wait(dir.path(), FIXTURE_LOCK_WAIT).unwrap();
            let run = RunId::new("r").unwrap();
            store
                .create_run(
                    &run,
                    RunConfig {
                        max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
                        autodoc: false,
                        arm: Arm::Blind,
                        library_learning: true,
                        seed: 3,
                        proposals_per_task: 1,
                        programs_per_batch: 1,
                        lease_ms: 1_000_000,
                        max_unchecked_per_batch: 3,
                        frontier_size: 5,
                        stitch: crate::state::StitchSettings::default(),
                        prelude: crate::verify::shipped_prelude(),
                        compared_with: None,
                        approach_hints: false,
                        show_received: false,
                        refuse_bucketing: false,
                        seed_entries: Vec::new(),
                    },
                    vec!["collapse/merge".into()],
                )
                .unwrap();
            let verifiers = VerifierCache::default();
            let sent = next_batch(&store, &run, "a", 0, &verifiers)
                .unwrap()
                .batch
                .unwrap();
            submit(
                &store,
                &run,
                0,
                "a",
                &sent.token,
                vec!["let solve = (".into()],
                None,
                1,
            )
            .unwrap();
            let server = ProposerServer::new(
                store,
                run,
                ProposerMode::Batch,
                Arc::new(crate::proposer::SystemClock),
            );
            let reader = Store::open(dir.path()).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            Fixture {
                dir,
                server,
                reader,
                runtime: Some(runtime),
            }
        }

        /// Lets the test start background checks until the guard drops.
        fn enter(&self) -> tokio::runtime::EnterGuard<'_> {
            self.runtime.as_ref().unwrap().enter()
        }

        fn pending_checks(&self) -> usize {
            self.reader
                .read(&self.server.run, |state| state.pending_checks().len())
                .unwrap()
        }

        /// Holds the run's lock, as another process does, until dropped.
        fn hold_the_run(&self) -> std::fs::File {
            let lock = std::fs::File::open(self.dir.path().join("r")).unwrap();
            lock.lock().unwrap();
            lock
        }

        fn failures(&self) -> usize {
            self.server.checker.failures.load(Ordering::SeqCst)
        }

        /// Waits until a background check has failed.
        fn wait_for_a_failure(&self) {
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            while self.failures() == 0 {
                assert!(std::time::Instant::now() < deadline, "no check failed");
                std::thread::sleep(Duration::from_millis(10));
            }
        }

        /// Waits until the run has no pending check, at most `limit`, and
        /// says whether it got there.
        fn checked_within(&self, limit: Duration) -> bool {
            let deadline = std::time::Instant::now() + limit;
            loop {
                if self.pending_checks() == 0 {
                    return true;
                }
                if std::time::Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(runtime) = self.runtime.take() {
                runtime.shutdown_background();
            }
        }
    }

    #[test]
    fn a_check_that_failed_is_tried_again_with_no_new_submission() {
        let f = Fixture::new();
        let _entered = f.enter();
        let lock = f.hold_the_run();
        f.server.check_in_background();
        f.wait_for_a_failure();
        drop(lock);
        // No new signal: the retry alone gets the check done, after the
        // first wait.
        assert!(
            f.checked_within(CHECK_RETRY_FIRST * 4),
            "the failed check was not tried again"
        );
    }

    #[test]
    fn a_new_submission_cuts_short_the_wait_after_a_failed_check() {
        let f = Fixture::new();
        let _entered = f.enter();
        let lock = f.hold_the_run();
        f.server.check_in_background();
        f.wait_for_a_failure();
        drop(lock);
        // A submission comes in while the checker waits to try again.
        let signalled = std::time::Instant::now();
        f.server.check_in_background();
        assert!(
            f.checked_within(CHECK_RETRY_FIRST / 2),
            "the check waited {:?} after the signal",
            signalled.elapsed()
        );
    }

    #[test]
    fn proposers_that_wait_do_not_cut_short_the_wait_after_a_failed_check() {
        let f = Fixture::new();
        let _entered = f.enter();
        // The cause lasts the whole test.
        let _lock = f.hold_the_run();
        f.server.check_in_background();
        // Proposers poll `dc_next_batch` and are told to wait.
        let polled = std::time::Instant::now();
        while polled.elapsed() < CHECK_RETRY_FIRST + Duration::from_secs(2) {
            f.server.check_if_idle();
            std::thread::sleep(Duration::from_millis(100));
        }
        // A try takes FIXTURE_LOCK_WAIT to fail. The first wait is 5 s, so
        // a third try cannot start before 1 + 5 + 1 + 10 s.
        let tries = f.failures();
        assert!((1..=2).contains(&tries), "{tries} tries");
    }

    #[test]
    fn a_proposer_that_waits_starts_a_check_if_none_runs() {
        let f = Fixture::new();
        let _entered = f.enter();
        f.server.check_if_idle();
        assert!(
            f.checked_within(Duration::from_secs(30)),
            "the pending submission was not checked"
        );
    }
}
