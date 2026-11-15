//! The state of one run, and the events that change it.
//!
//! A [`RunState`] is the result of applying a run's events in order, starting
//! from [`RunState::create`]. Nothing else changes it, so the journal of events
//! on disk (see [`crate::journal`]) holds the whole state.
//!
//! Programs and library bodies are text here. The state does not parse them:
//! the code that evaluates and compresses them does.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

/// The longest run id, in bytes.
pub const MAX_RUN_ID_LEN: usize = 64;

/// The name of a run, which is also the name of its directory.
///
/// It has 1 to [`MAX_RUN_ID_LEN`] characters from `a-z`, `0-9`, `-` and `_`,
/// and starts with a letter or a digit. So it is one path component and cannot
/// name a parent directory.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(String);

impl RunId {
    /// Checks `text` and makes it a run id.
    pub fn new(text: &str) -> Result<RunId, InvalidRunId> {
        let bytes = text.as_bytes();
        let first_ok = bytes
            .first()
            .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
        let rest_ok = bytes
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'-' || *b == b'_');
        if first_ok && rest_ok && bytes.len() <= MAX_RUN_ID_LEN {
            Ok(RunId(text.to_owned()))
        } else {
            Err(InvalidRunId)
        }
    }

    /// The id as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The text is not a run id. See [`RunId`] for the rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidRunId;

impl fmt::Display for InvalidRunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a run id has 1 to {MAX_RUN_ID_LEN} characters from a-z, 0-9, '-' and '_', \
             and starts with a letter or a digit"
        )
    }
}

impl std::error::Error for InvalidRunId {}

/// What the proposer learns about its submissions (decision D11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// The proposer gets no evaluation results.
    Blind,
}

/// The settings of a run, fixed when it is created.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// The arm of the experiment.
    pub arm: Arm,
    /// The seed for every random choice the run makes.
    pub seed: u64,
    /// How many programs each task gets in one iteration. It is positive.
    pub proposals_per_task: u32,
    /// The most programs one batch asks for: 1 to [`MAX_PROGRAMS_PER_BATCH`].
    pub programs_per_batch: u32,
    /// How long a proposer has to submit a batch, in milliseconds. After
    /// that the batch's programs can be handed out again. It is positive.
    pub lease_ms: u64,
    /// The most submissions kept for one batch that are not checked, of
    /// each of two kinds: foreign ones, and late or repeat ones from the
    /// batch's worker. A submission past it is refused and counted, not
    /// kept (see `Store::refuse`). It is positive. A run made before the
    /// setting has [`DEFAULT_MAX_UNCHECKED_PER_BATCH`].
    #[serde(default = "default_max_unchecked_per_batch")]
    pub max_unchecked_per_batch: u32,
}

/// The most submissions kept for one batch that are not checked, of each
/// kind, when a run does not say (see [`RunConfig::max_unchecked_per_batch`]).
/// An honest proposer sends at most one late or repeat submission in the
/// usual case, a retry after a lost reply; three leave room for a second
/// retry and bound what one batch can add to the journal at six
/// submissions.
pub const DEFAULT_MAX_UNCHECKED_PER_BATCH: u32 = 3;

fn default_max_unchecked_per_batch() -> u32 {
    DEFAULT_MAX_UNCHECKED_PER_BATCH
}

/// One definition in the library.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    /// The number of parameters.
    pub arity: u32,
    /// The closed term `λ^arity. body`, as an s-expression (`dc_core::sexpr`).
    pub term: String,
}

/// One program in a task's frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontierMember {
    /// The program, a core term as an s-expression (`dc_core::sexpr`).
    pub program: String,
}

/// What a log record holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogKind {
    /// The result of a compression run.
    Compression,
    /// A remark by the operator or the server.
    Note,
    /// A report by the store that it set aside bytes of the journal that were
    /// not a record (see `crate::store`).
    Recovery,
}

/// One entry in the run's log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogRecord {
    /// What the record holds.
    pub kind: LogKind,
    /// The iteration the record was written in.
    pub iteration: u64,
    /// The task the record is about, if it is about one.
    pub task: Option<String>,
    /// The content. The code that writes a kind of record sets its format.
    pub text: String,
}

/// A batch of work handed to a proposer: one task, and how many programs to
/// write for it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchIssued {
    /// The batch's number. Batches are numbered 0, 1, ... over the whole run.
    pub batch: u64,
    /// The iteration the batch is for: the current one when it is issued.
    pub iteration: u64,
    /// The task.
    pub task: String,
    /// How many programs the batch asks for. It is positive.
    pub wanted: u32,
    /// The id the proposer gave for itself.
    pub worker: String,
    /// A random secret, sent only in the payload. A submission must give it,
    /// so a proposer cannot submit for a batch it was not sent.
    pub token: String,
    /// When the batch was issued, in milliseconds since the Unix epoch.
    pub issued_ms: u64,
    /// When the lease ends, in milliseconds since the Unix epoch.
    pub deadline_ms: u64,
    /// The payload handed to the proposer, exactly as it was sent.
    pub payload: String,
}

/// How a submission stands to its batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Standing {
    /// The first submission for the batch to arrive from its worker, in its
    /// iteration, before its lease ended. Its programs are checked, by a
    /// [`Checked`] event, and counted.
    OnTime,
    /// As `OnTime`, after the lease ended. Its programs are checked and
    /// counted too: they are real samples, even if the batch's slots were
    /// handed out again.
    Expired,
    /// The first submission for the batch to arrive from its worker, after
    /// the batch's iteration ended. It is kept but not checked: it was written
    /// for another library.
    Late,
    /// A later submission for the batch from its worker. It is kept but not
    /// checked, up to the cap (see [`RunState::keeps`]).
    Repeat,
    /// A submission that gives the wrong token for the batch, or names a
    /// worker the batch was not issued to. It is kept but not checked, and
    /// the batch stays open for the proposer it was sent to. Past the cap
    /// (see [`RunState::keeps`]) it is counted, not kept.
    Foreign,
}

impl Standing {
    /// Whether the submission's programs are checked and counted.
    pub fn evaluated(self) -> bool {
        matches!(self, Standing::OnTime | Standing::Expired)
    }
}

/// The result of checking one submitted program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Check {
    /// The program did not compile: a syntax error, an unknown name, or no
    /// definition of the entry.
    Compile {
        /// The error.
        error: String,
    },
    /// The program compiled but does not have the task's type.
    Type {
        /// The error.
        error: String,
    },
    /// The program typed but did not give the expected output on an example.
    Run {
        /// What went wrong.
        error: String,
    },
    /// The program passed every example.
    Passed {
        /// The compiled program, a closed core term as an s-expression, with
        /// its prelude and library references.
        program: String,
    },
    /// The program came after the number the batch asked for, and was not
    /// evaluated.
    Extra,
}

/// What a proposer submitted for a batch, as it was sent. It is committed
/// when it arrives, before its programs are checked, so the order of
/// submissions in the journal is the order they arrived in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Submitted {
    /// The batch.
    pub batch: u64,
    /// The id the submitting proposer gave for itself.
    pub worker: String,
    /// The token the submitting proposer gave, as it was sent.
    pub token: String,
    /// The run's iteration when the submission came in.
    pub iteration: u64,
    /// When it came in, in milliseconds since the Unix epoch.
    pub received_ms: u64,
    /// The model the proposer said it is, if it said.
    pub model: Option<String>,
    /// The programs, exactly as they were sent.
    pub programs: Vec<String>,
    /// How the submission stands to its batch.
    pub standing: Standing,
}

/// What the server found when it checked the programs of a batch's
/// submission (one whose standing is [`Standing::evaluated`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Checked {
    /// The batch.
    pub batch: u64,
    /// How many library entries the programs were checked with. It must be
    /// the library's size when the event is applied.
    pub library_len: u64,
    /// One check for each program of the submission.
    pub checks: Vec<Check>,
}

/// One change to a run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    /// The run starts. It is the first event of every run and occurs once.
    Created {
        /// The run's settings.
        config: RunConfig,
        /// The names of the run's tasks.
        tasks: Vec<String>,
    },
    /// A record joins the log.
    Logged(LogRecord),
    /// A batch is handed to a proposer.
    BatchIssued(BatchIssued),
    /// A proposer submits programs for a batch.
    Submitted(Submitted),
    /// The server has checked a submission's programs.
    Checked(Checked),
    /// A task gets a new frontier, which replaces the one it had.
    FrontierSet {
        /// The task.
        task: String,
        /// The new frontier, which may be empty.
        members: Vec<FrontierMember>,
    },
    /// Definitions join the end of the library.
    LibraryExtended {
        /// The new definitions, at least one.
        entries: Vec<LibraryEntry>,
    },
    /// The current iteration ends and the next one starts.
    IterationFinished {
        /// The iteration that ends.
        iteration: u64,
        /// How many of the verified programs, from the first, the iteration
        /// step took. Programs checked after the step read them stay
        /// verified for the next iteration's step.
        used: u64,
    },
}

/// Whether `given` is `secret`, taking a time that does not depend on where
/// they first differ.
fn same_secret(secret: &str, given: &str) -> bool {
    secret.len() == given.len()
        && secret
            .bytes()
            .zip(given.bytes())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

mod history {
    //! The parts of a run's state that grow with every iteration: its
    //! batches, and its counts for each iteration. Each is behind methods
    //! that count each entry they hand out. The containers are private to
    //! this module, so no code in [`RunState`](super::RunState) can read an
    //! entry without it being counted. The tests use the count to check
    //! that the calls a proposer or the operator make all the time do not
    //! read the history of finished iterations.

    use std::collections::BTreeMap;

    use super::Batch;

    #[cfg(test)]
    thread_local! {
        /// How many entries this thread has read through [`Batches`] and
        /// [`ByIteration`].
        pub(super) static READ: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// Notes that `n` entries were read.
    fn count(n: usize) {
        #[cfg(test)]
        READ.with(|read| read.set(read.get() + n as u64));
        #[cfg(not(test))]
        let _ = n;
    }

    /// Every batch of a run: batch `i` is entry `i`.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct Batches(Vec<Batch>);

    impl Batches {
        /// How many batches there are. It reads none.
        pub(super) fn len(&self) -> usize {
            self.0.len()
        }

        /// Adds the next batch. It reads none.
        pub(super) fn push(&mut self, batch: Batch) {
            self.0.push(batch);
        }

        /// Batch `number`, if there is one. It reads one.
        pub(super) fn get(&self, number: u64) -> Option<&Batch> {
            count(1);
            self.0.get(usize::try_from(number).ok()?)
        }

        /// Batch `number`, to change, if there is one. It reads one.
        pub(super) fn get_mut(&mut self, number: u64) -> Option<&mut Batch> {
            count(1);
            self.0.get_mut(usize::try_from(number).ok()?)
        }

        /// Every batch, as a slice. It counts as reading all of them.
        pub(super) fn as_slice(&self) -> &[Batch] {
            count(self.0.len());
            &self.0
        }
    }

    /// A value for each iteration that has one.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(super) struct ByIteration<V>(BTreeMap<u64, V>);

    impl<V: Default> ByIteration<V> {
        /// The value of `iteration`, if it has one. It reads one.
        pub(super) fn get(&self, iteration: u64) -> Option<&V> {
            count(1);
            self.0.get(&iteration)
        }

        /// The value of `iteration`, to change, made if it has none. It
        /// reads one.
        pub(super) fn entry(&mut self, iteration: u64) -> &mut V {
            count(1);
            self.0.entry(iteration).or_default()
        }

        /// Every iteration's value, in order. It reads each one it hands
        /// out.
        #[cfg_attr(not(test), allow(dead_code))]
        pub(super) fn iter(&self) -> impl Iterator<Item = (u64, &V)> + '_ {
            self.0.iter().inspect(|_| count(1)).map(|(&i, v)| (i, v))
        }
    }
}

use history::{Batches, ByIteration};

/// The most programs a batch may ask for. A submission may hold this many
/// programs, so every program a batch asks for fits in one submission.
pub const MAX_PROGRAMS_PER_BATCH: u32 = 32;

/// Why an event cannot be applied to a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejected {
    /// A run needs at least one task.
    NoTasks,
    /// The task is named twice in the task set.
    DuplicateTask(String),
    /// `proposals_per_task`, `programs_per_batch`, `lease_ms` or
    /// `max_unchecked_per_batch` is zero.
    ZeroSetting(&'static str),
    /// `programs_per_batch` is over [`MAX_PROGRAMS_PER_BATCH`].
    TooManyPerBatch(u32),
    /// The first event of a run is not `Created`.
    NotCreated,
    /// A `Created` event came after the first event.
    CreatedTwice,
    /// The run has no task of this name.
    UnknownTask(String),
    /// The event names an iteration that is not the current one.
    WrongIteration {
        /// The run's current iteration.
        current: u64,
        /// The iteration the event names.
        given: u64,
    },
    /// A `LibraryExtended` event has no entries.
    NoEntries,
    /// The iteration number would overflow.
    IterationOverflow,
    /// A `BatchIssued` event does not have the next batch number.
    WrongBatchNumber {
        /// The next batch number.
        next: u64,
        /// The number the event gives.
        given: u64,
    },
    /// A `BatchIssued` event asks for no programs, or its lease ends before
    /// it starts.
    BadBatch,
    /// The run has issued no batch of this number.
    UnknownBatch(u64),
    /// A `Submitted` event's standing, or a `Checked` event's checks, do not
    /// fit the batch.
    BadSubmission(u64),
    /// A `Checked` event names a batch with no submission waiting to be
    /// checked.
    NotAwaitingCheck(u64),
    /// A `Checked` event was made with a library of another size.
    StaleLibrary {
        /// The batch.
        batch: u64,
        /// The library's size.
        library: u64,
        /// The size the checks were made with.
        checked_with: u64,
    },
    /// The iteration cannot end while a submission of it waits to be checked.
    ChecksPending(u64),
    /// An `IterationFinished` event says the step took more programs than
    /// are verified.
    UnknownVerified {
        /// How many the event says the step took.
        used: u64,
        /// How many are verified.
        verified: u64,
    },
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rejected::NoTasks => write!(f, "a run needs at least one task"),
            Rejected::DuplicateTask(t) => write!(f, "task {t:?} is named twice"),
            Rejected::ZeroSetting(name) => write!(f, "{name} must be at least 1"),
            Rejected::TooManyPerBatch(n) => write!(
                f,
                "programs_per_batch is {n}; it must be at most {MAX_PROGRAMS_PER_BATCH}"
            ),
            Rejected::NotCreated => write!(f, "the first event of a run must create it"),
            Rejected::CreatedTwice => write!(f, "the run is already created"),
            Rejected::UnknownTask(t) => write!(f, "the run has no task {t:?}"),
            Rejected::WrongIteration { current, given } => write!(
                f,
                "the event names iteration {given}, but the run is in iteration {current}"
            ),
            Rejected::NoEntries => write!(f, "a library extension needs at least one entry"),
            Rejected::IterationOverflow => write!(f, "the iteration number would overflow"),
            Rejected::WrongBatchNumber { next, given } => write!(
                f,
                "the event issues batch {given}, but the next batch is {next}"
            ),
            Rejected::BadBatch => write!(
                f,
                "a batch must ask for at least one program and end after it starts"
            ),
            Rejected::UnknownBatch(b) => write!(f, "the run has no batch {b}"),
            Rejected::BadSubmission(b) => write!(
                f,
                "the submission's standing or checks do not fit batch {b}"
            ),
            Rejected::NotAwaitingCheck(b) => {
                write!(f, "batch {b} has no submission waiting to be checked")
            }
            Rejected::StaleLibrary {
                batch,
                library,
                checked_with,
            } => write!(
                f,
                "batch {batch} was checked with {checked_with} library entries, \
                 but the library has {library}"
            ),
            Rejected::ChecksPending(b) => write!(
                f,
                "the iteration cannot end: batch {b}'s submission waits to be checked"
            ),
            Rejected::UnknownVerified { used, verified } => write!(
                f,
                "the iteration step took {used} verified programs, but there are {verified}"
            ),
        }
    }
}

impl std::error::Error for Rejected {}

/// A batch the run has issued.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    /// The iteration it is for.
    pub iteration: u64,
    /// The task.
    pub task: String,
    /// How many programs it asks for.
    pub wanted: u32,
    /// The proposer's id.
    pub worker: String,
    /// The batch's token.
    pub token: String,
    /// When its lease ends, in milliseconds since the Unix epoch.
    pub deadline_ms: u64,
    /// Whether a submission for it has come in from its worker.
    pub submitted: bool,
    /// The programs of its submission, while they wait to be checked.
    pub awaiting_check: Option<Vec<String>>,
    /// How many foreign submissions for it are kept.
    pub foreign_kept: u32,
    /// How many late or repeat submissions from its worker are kept.
    pub unchecked_kept: u32,
}

/// What came in for one task in one iteration. Only evaluated submissions
/// count (see [`Standing::evaluated`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaskCounts {
    /// Batches issued for the task.
    pub batches: u64,
    /// Programs received in submissions that are checked, up to the number
    /// each batch asked for. They count towards the quota when they arrive.
    pub received: u64,
    /// Programs checked: those a batch asked for, not the extra ones.
    pub programs: u64,
    /// Programs that compiled.
    pub compiled: u64,
    /// Programs that compiled and typed.
    pub typed: u64,
    /// Programs that passed every example.
    pub passed: u64,
}

/// What came in during one iteration, over all tasks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IterationCounts {
    /// Bytes of payload handed out.
    pub payload_bytes: u64,
    /// Bytes of program text received, in every submission.
    pub submission_bytes: u64,
    /// Submissions received, of any standing.
    pub submissions: u64,
    /// Submissions that came after their batch's iteration ended.
    pub late: u64,
    /// Submissions for a batch that already had one.
    pub repeats: u64,
    /// Submissions from a worker the batch was not issued to.
    pub foreign: u64,
}

/// A program that passed its task's examples and that no iteration step has
/// taken yet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    /// The task.
    pub task: String,
    /// The compiled program (see [`Check::Passed`]).
    pub program: String,
}

/// Everything the server knows about a run, except the text of its log,
/// payloads and submissions.
///
/// That text is in the journal. The state keeps counts.
///
/// # Indexes
///
/// `batches` holds every batch since the run began, so a search through it
/// costs more the longer the run. The calls a proposer or the operator makes
/// all the time search two indexes instead, which hold only batch numbers:
///
/// * `live`: the batches of the current iteration that no submission from
///   their worker has come in for. A lease is live while its batch is in
///   `live` and its deadline has not passed (see [`RunState::leased`]). A
///   batch leaves when it is submitted, and the set empties when the
///   iteration ends. A batch whose lease ended without a submission stays
///   until then, so the set is at most the current iteration's batches.
/// * `awaiting`: the batches whose submission waits to be checked (their
///   [`Batch::awaiting_check`] is `Some`). An iteration cannot end while one
///   waits, so these are of the current iteration too.
///
/// `task_counts` is keyed by iteration and then by task, so a lookup by a
/// task's name borrows the name and does not copy it.
///
/// The indexes are derived: [`RunState::apply`] keeps them, so a replay of
/// the journal builds them again, and nothing about them is written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    config: RunConfig,
    tasks: Vec<String>,
    library: Vec<LibraryEntry>,
    frontiers: BTreeMap<String, Vec<FrontierMember>>,
    iteration: u64,
    log_counts: BTreeMap<LogKind, u64>,
    batches: Batches,
    live: BTreeSet<u64>,
    awaiting: BTreeSet<u64>,
    task_counts: ByIteration<BTreeMap<String, TaskCounts>>,
    iteration_counts: ByIteration<IterationCounts>,
    verified: Vec<Verified>,
}

/// Changes the counts of `task` in `iteration` by `change`. The task's name
/// is copied only the first time it has counts in the iteration.
fn change_task_counts(
    counts: &mut ByIteration<BTreeMap<String, TaskCounts>>,
    iteration: u64,
    task: &str,
    change: impl FnOnce(&mut TaskCounts),
) {
    let by_task = counts.entry(iteration);
    match by_task.get_mut(task) {
        Some(c) => change(c),
        None => {
            let mut c = TaskCounts::default();
            change(&mut c);
            by_task.insert(task.to_owned(), c);
        }
    }
}

impl RunState {
    /// The state of a new run: iteration 0, an empty library, and an empty
    /// frontier for each task.
    pub fn create(config: RunConfig, tasks: Vec<String>) -> Result<RunState, Rejected> {
        if tasks.is_empty() {
            return Err(Rejected::NoTasks);
        }
        if config.programs_per_batch > MAX_PROGRAMS_PER_BATCH {
            return Err(Rejected::TooManyPerBatch(config.programs_per_batch));
        }
        for (name, value) in [
            ("proposals_per_task", u64::from(config.proposals_per_task)),
            ("programs_per_batch", u64::from(config.programs_per_batch)),
            ("lease_ms", config.lease_ms),
            (
                "max_unchecked_per_batch",
                u64::from(config.max_unchecked_per_batch),
            ),
        ] {
            if value == 0 {
                return Err(Rejected::ZeroSetting(name));
            }
        }
        let mut frontiers = BTreeMap::new();
        for task in &tasks {
            if frontiers.insert(task.clone(), Vec::new()).is_some() {
                return Err(Rejected::DuplicateTask(task.clone()));
            }
        }
        Ok(RunState {
            config,
            tasks,
            library: Vec::new(),
            frontiers,
            iteration: 0,
            log_counts: BTreeMap::new(),
            batches: Batches::default(),
            live: BTreeSet::new(),
            awaiting: BTreeSet::new(),
            task_counts: ByIteration::default(),
            iteration_counts: ByIteration::default(),
            verified: Vec::new(),
        })
    }

    /// The state after `events`, where the first one creates the run.
    pub fn replay<'a>(events: impl IntoIterator<Item = &'a Event>) -> Result<RunState, Rejected> {
        let mut events = events.into_iter();
        let Some(Event::Created { config, tasks }) = events.next() else {
            return Err(Rejected::NotCreated);
        };
        let mut state = RunState::create(config.clone(), tasks.clone())?;
        for event in events {
            state.apply(event)?;
        }
        Ok(state)
    }

    /// Changes the state by `event`. On an error the state is unchanged:
    /// every check comes before the first change. `Store::commit` relies on
    /// this to apply events without a copy of the state.
    ///
    /// In tests, every rejection checks this: the state is copied before the
    /// event and compared with the copy after it.
    pub fn apply(&mut self, event: &Event) -> Result<(), Rejected> {
        #[cfg(test)]
        let before = self.clone();
        let result = self.apply_checked(event);
        #[cfg(test)]
        if result.is_err() {
            assert!(
                *self == before,
                "a rejected event changed the state: {event:?}, {result:?}"
            );
        }
        result
    }

    /// Changes the state by `event`, with every check before the first
    /// change. See [`RunState::apply`].
    fn apply_checked(&mut self, event: &Event) -> Result<(), Rejected> {
        match event {
            Event::Created { .. } => return Err(Rejected::CreatedTwice),
            Event::Logged(record) => {
                self.check_iteration(record.iteration)?;
                if let Some(task) = &record.task {
                    self.check_task(task)?;
                }
                *self.log_counts.entry(record.kind).or_insert(0) += 1;
            }
            Event::BatchIssued(issued) => self.issue(issued)?,
            Event::Submitted(submitted) => self.submit(submitted)?,
            Event::Checked(checked) => self.check(checked)?,
            Event::FrontierSet { task, members } => {
                self.check_task(task)?;
                self.frontiers.insert(task.clone(), members.clone());
            }
            Event::LibraryExtended { entries } => {
                if entries.is_empty() {
                    return Err(Rejected::NoEntries);
                }
                self.library.extend(entries.iter().cloned());
            }
            Event::IterationFinished { iteration, used } => {
                self.check_iteration(*iteration)?;
                if let Some(b) = self.awaiting.first() {
                    return Err(Rejected::ChecksPending(*b));
                }
                let verified = self.verified.len() as u64;
                let taken = usize::try_from(*used)
                    .ok()
                    .filter(|&n| n <= self.verified.len())
                    .ok_or(Rejected::UnknownVerified {
                        used: *used,
                        verified,
                    })?;
                self.iteration = self
                    .iteration
                    .checked_add(1)
                    .ok_or(Rejected::IterationOverflow)?;
                // The iteration step reads the verified programs and commits
                // this event later, without the run's lock in between; a
                // program checked in that time waits for the next step.
                self.verified.drain(..taken);
                // No batch of the ended iteration holds a live lease now.
                self.live.clear();
            }
        }
        Ok(())
    }

    fn issue(&mut self, issued: &BatchIssued) -> Result<(), Rejected> {
        self.check_iteration(issued.iteration)?;
        self.check_task(&issued.task)?;
        let next = self.batches.len() as u64;
        if issued.batch != next {
            return Err(Rejected::WrongBatchNumber {
                next,
                given: issued.batch,
            });
        }
        if issued.wanted == 0 || issued.deadline_ms <= issued.issued_ms || issued.token.is_empty() {
            return Err(Rejected::BadBatch);
        }
        self.batches.push(Batch {
            iteration: issued.iteration,
            task: issued.task.clone(),
            wanted: issued.wanted,
            worker: issued.worker.clone(),
            token: issued.token.clone(),
            deadline_ms: issued.deadline_ms,
            submitted: false,
            awaiting_check: None,
            foreign_kept: 0,
            unchecked_kept: 0,
        });
        self.live.insert(issued.batch);
        change_task_counts(&mut self.task_counts, issued.iteration, &issued.task, |c| {
            c.batches += 1
        });
        let counts = self.iteration_counts.entry(issued.iteration);
        counts.payload_bytes = counts
            .payload_bytes
            .saturating_add(issued.payload.len() as u64);
        Ok(())
    }

    /// The standing a submission for `batch` from `worker`, giving `token`,
    /// has if it comes in at `received_ms`, or `None` if the run has no such
    /// batch.
    pub fn standing(
        &self,
        batch: u64,
        worker: &str,
        token: &str,
        received_ms: u64,
    ) -> Option<Standing> {
        let b = self.batches.get(batch)?;
        Some(if b.worker != worker || !same_secret(&b.token, token) {
            Standing::Foreign
        } else if b.submitted {
            Standing::Repeat
        } else if b.iteration != self.iteration {
            Standing::Late
        } else if received_ms <= b.deadline_ms {
            Standing::OnTime
        } else {
            Standing::Expired
        })
    }

    /// Whether a submission of `standing` for `batch` is kept: one that is
    /// checked always is; one that is not is kept while the batch has fewer
    /// than [`RunConfig::max_unchecked_per_batch`] of its kind (foreign, or
    /// late and repeat). `false` if the run has no such batch.
    pub fn keeps(&self, batch: u64, standing: Standing) -> bool {
        let Some(b) = self.batches.get(batch) else {
            return false;
        };
        let cap = self.config.max_unchecked_per_batch;
        match standing {
            Standing::OnTime | Standing::Expired => true,
            Standing::Foreign => b.foreign_kept < cap,
            Standing::Late | Standing::Repeat => b.unchecked_kept < cap,
        }
    }

    fn submit(&mut self, s: &Submitted) -> Result<(), Rejected> {
        self.check_iteration(s.iteration)?;
        let standing = self
            .standing(s.batch, &s.worker, &s.token, s.received_ms)
            .ok_or(Rejected::UnknownBatch(s.batch))?;
        if standing != s.standing {
            return Err(Rejected::BadSubmission(s.batch));
        }
        // `standing` found the batch, so this finds it too.
        let Some(batch) = self.batches.get_mut(s.batch) else {
            return Err(Rejected::UnknownBatch(s.batch));
        };
        if standing != Standing::Foreign {
            batch.submitted = true;
            self.live.remove(&s.batch);
        }
        // A journal from before the cap can hold more than it allows, so
        // the cap is the server's to keep (see `RunState::keeps`), not a
        // reason to reject the event.
        match standing {
            Standing::Foreign => batch.foreign_kept = batch.foreign_kept.saturating_add(1),
            Standing::Late | Standing::Repeat => {
                batch.unchecked_kept = batch.unchecked_kept.saturating_add(1);
            }
            Standing::OnTime | Standing::Expired => {}
        }
        if standing.evaluated() {
            batch.awaiting_check = Some(s.programs.clone());
            self.awaiting.insert(s.batch);
            let received = (s.programs.len() as u64).min(u64::from(batch.wanted));
            change_task_counts(&mut self.task_counts, batch.iteration, &batch.task, |c| {
                c.received += received;
            });
        }
        let counts = self.iteration_counts.entry(s.iteration);
        counts.submissions += 1;
        let bytes: u64 = s.programs.iter().map(|p| p.len() as u64).sum();
        counts.submission_bytes = counts.submission_bytes.saturating_add(bytes);
        match standing {
            Standing::Late => counts.late += 1,
            Standing::Repeat => counts.repeats += 1,
            Standing::Foreign => counts.foreign += 1,
            Standing::OnTime | Standing::Expired => {}
        }
        Ok(())
    }

    fn check(&mut self, c: &Checked) -> Result<(), Rejected> {
        let library = self.library.len() as u64;
        // Every check below comes before the first change to the batch.
        let batch = self
            .batches
            .get_mut(c.batch)
            .ok_or(Rejected::UnknownBatch(c.batch))?;
        let Some(programs) = &batch.awaiting_check else {
            return Err(Rejected::NotAwaitingCheck(c.batch));
        };
        if c.library_len != library {
            return Err(Rejected::StaleLibrary {
                batch: c.batch,
                library,
                checked_with: c.library_len,
            });
        }
        let wanted = batch.wanted as usize;
        let fits = c.checks.len() == programs.len()
            && c.checks
                .iter()
                .enumerate()
                .all(|(i, check)| matches!(check, Check::Extra) == (i >= wanted));
        if !fits {
            return Err(Rejected::BadSubmission(c.batch));
        }
        batch.awaiting_check = None;
        self.awaiting.remove(&c.batch);
        change_task_counts(&mut self.task_counts, batch.iteration, &batch.task, |tc| {
            for check in &c.checks {
                let (compiled, typed, passed) = match check {
                    Check::Extra => continue,
                    Check::Compile { .. } => (false, false, false),
                    Check::Type { .. } => (true, false, false),
                    Check::Run { .. } => (true, true, false),
                    Check::Passed { .. } => (true, true, true),
                };
                tc.programs += 1;
                tc.compiled += u64::from(compiled);
                tc.typed += u64::from(typed);
                tc.passed += u64::from(passed);
            }
        });
        // An iteration cannot end with a check pending, so the batch is of
        // the current iteration.
        for check in &c.checks {
            if let Check::Passed { program } = check {
                self.verified.push(Verified {
                    task: batch.task.clone(),
                    program: program.clone(),
                });
            }
        }
        Ok(())
    }

    /// The batches whose submission waits to be checked, in order.
    pub fn pending_checks(&self) -> Vec<u64> {
        self.awaiting.iter().copied().collect()
    }

    /// The first batch whose submission waits to be checked, if any.
    pub fn first_pending_check(&self) -> Option<u64> {
        self.awaiting.first().copied()
    }

    /// Whether `batch`'s submission waits to be checked.
    pub fn awaits_check(&self, batch: u64) -> bool {
        self.awaiting.contains(&batch)
    }

    fn check_iteration(&self, given: u64) -> Result<(), Rejected> {
        if given == self.iteration {
            Ok(())
        } else {
            Err(Rejected::WrongIteration {
                current: self.iteration,
                given,
            })
        }
    }

    fn check_task(&self, task: &str) -> Result<(), Rejected> {
        if self.frontiers.contains_key(task) {
            Ok(())
        } else {
            Err(Rejected::UnknownTask(task.to_owned()))
        }
    }

    /// The run's settings.
    pub fn config(&self) -> &RunConfig {
        &self.config
    }

    /// The names of the run's tasks, in the order given at creation.
    pub fn tasks(&self) -> &[String] {
        &self.tasks
    }

    /// The library. Entry `i` is library definition `i`.
    pub fn library(&self) -> &[LibraryEntry] {
        &self.library
    }

    /// The frontier of `task`, or `None` if the run has no such task.
    pub fn frontier(&self, task: &str) -> Option<&[FrontierMember]> {
        self.frontiers.get(task).map(Vec::as_slice)
    }

    /// The current iteration. The first is 0.
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// How many log records of `kind` the run has.
    pub fn log_count(&self, kind: LogKind) -> u64 {
        self.log_counts.get(&kind).copied().unwrap_or(0)
    }

    /// Every batch issued, in order: batch `i` is entry `i`. It is as long
    /// as the run, so a search through it costs more the longer the run:
    /// use [`RunState::batch`], [`RunState::leased`] or
    /// [`RunState::first_pending_check`] in a call made all the time.
    pub fn batches(&self) -> &[Batch] {
        self.batches.as_slice()
    }

    /// Batch `number`, if the run has issued it.
    pub fn batch(&self, number: u64) -> Option<&Batch> {
        self.batches.get(number)
    }

    /// How many batches the run has issued: the next batch's number.
    pub fn batch_count(&self) -> u64 {
        self.batches.len() as u64
    }

    /// What came in for `task` in `iteration`.
    pub fn task_counts(&self, iteration: u64, task: &str) -> TaskCounts {
        self.task_counts
            .get(iteration)
            .and_then(|by_task| by_task.get(task))
            .copied()
            .unwrap_or_default()
    }

    /// What came in during `iteration`.
    pub fn iteration_counts(&self, iteration: u64) -> IterationCounts {
        self.iteration_counts
            .get(iteration)
            .copied()
            .unwrap_or_default()
    }

    /// The programs that passed their examples and that no iteration step has
    /// taken yet, in the order they were checked. They are of the current
    /// iteration, and of earlier ones if they were checked after the step
    /// that ended it read them.
    pub fn verified(&self) -> &[Verified] {
        &self.verified
    }

    /// How many more programs `task` needs in the current iteration, given the
    /// programs received and the live leases at `now_ms`. A lease is live until
    /// its batch is submitted or its deadline has passed.
    pub fn open_slots(&self, task: &str, now_ms: u64) -> u64 {
        let received = self.task_counts(self.iteration, task).received;
        let leased: u64 = self
            .leased(now_ms)
            .filter(|(_, b)| b.task == task)
            .map(|(_, b)| u64::from(b.wanted))
            .sum();
        u64::from(self.config.proposals_per_task).saturating_sub(received.saturating_add(leased))
    }

    /// The batches of the current iteration whose lease is live at `now_ms`:
    /// not submitted, and with a deadline that has not passed. With each
    /// batch comes its number. They are in order of their numbers.
    ///
    /// It reads only the current iteration's unsubmitted batches, not every
    /// batch of the run: see the indexes in [`RunState`].
    pub fn leased(&self, now_ms: u64) -> impl Iterator<Item = (u64, &Batch)> + '_ {
        self.live
            .iter()
            .filter_map(|&i| Some((i, self.batches.get(i)?)))
            .filter(move |(_, b)| now_ms <= b.deadline_ms)
    }

    /// Whether every task has received its programs for the current
    /// iteration and every one of them has been checked.
    pub fn quota_in(&self) -> bool {
        let quota = u64::from(self.config.proposals_per_task);
        self.awaiting.is_empty()
            && self
                .tasks
                .iter()
                .all(|t| self.task_counts(self.iteration, t).received >= quota)
    }

    /// The task the next batch should be for at `now_ms`, and how many
    /// programs it should ask for: the task with the most open slots, the
    /// earliest in the run's order on a tie. `None` if no task has an open
    /// slot.
    pub fn next_task(&self, now_ms: u64) -> Option<(&str, u32)> {
        let mut best: Option<(&str, u64)> = None;
        for task in &self.tasks {
            let open = self.open_slots(task, now_ms);
            if open > 0 && best.is_none_or(|(_, b)| open > b) {
                best = Some((task, open));
            }
        }
        best.map(|(task, open)| {
            let wanted = open.min(u64::from(self.config.programs_per_batch));
            (task, wanted as u32)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RunConfig {
        RunConfig {
            arm: Arm::Blind,
            seed: 7,
            proposals_per_task: 4,
            programs_per_batch: 2,
            lease_ms: 1000,
            max_unchecked_per_batch: 3,
        }
    }

    fn new_state() -> RunState {
        RunState::create(config(), vec!["a".into(), "b".into()]).unwrap()
    }

    #[test]
    fn run_ids_are_single_safe_path_components() {
        for ok in [
            "r",
            "run-1",
            "2026_11_14-blind",
            &"a".repeat(MAX_RUN_ID_LEN),
        ] {
            assert_eq!(RunId::new(ok).unwrap().as_str(), ok);
        }
        let too_long = "a".repeat(MAX_RUN_ID_LEN + 1);
        for bad in [
            "", ".", "..", "a/b", "../a", "a.b", "-a", "_a", "A", "a b", "a\n", "é", "a\0",
            &too_long,
        ] {
            assert_eq!(RunId::new(bad), Err(InvalidRunId), "{bad:?}");
        }
    }

    #[test]
    fn create_refuses_bad_task_sets_and_quotas() {
        assert_eq!(RunState::create(config(), vec![]), Err(Rejected::NoTasks));
        assert_eq!(
            RunState::create(config(), vec!["a".into(), "a".into()]),
            Err(Rejected::DuplicateTask("a".into()))
        );
        let zero = RunConfig {
            proposals_per_task: 0,
            ..config()
        };
        assert_eq!(
            RunState::create(zero, vec!["a".into()]),
            Err(Rejected::ZeroSetting("proposals_per_task"))
        );
        let zero = RunConfig {
            max_unchecked_per_batch: 0,
            ..config()
        };
        assert_eq!(
            RunState::create(zero, vec!["a".into()]),
            Err(Rejected::ZeroSetting("max_unchecked_per_batch"))
        );
        for (n, ok) in [
            (MAX_PROGRAMS_PER_BATCH, true),
            (MAX_PROGRAMS_PER_BATCH + 1, false),
        ] {
            let c = RunConfig {
                programs_per_batch: n,
                ..config()
            };
            let got = RunState::create(c, vec!["a".into()]);
            assert_eq!(got.is_ok(), ok, "{n}: {got:?}");
        }
    }

    #[test]
    fn events_change_each_part_of_the_state() {
        let mut s = new_state();
        assert_eq!(s.iteration(), 0);
        assert_eq!(s.frontier("a"), Some(&[][..]));
        assert_eq!(s.frontier("c"), None);

        let member = FrontierMember {
            program: "(lam 0)".into(),
        };
        s.apply(&Event::FrontierSet {
            task: "a".into(),
            members: vec![member.clone()],
        })
        .unwrap();
        assert_eq!(s.frontier("a"), Some(&[member][..]));
        assert_eq!(s.frontier("b"), Some(&[][..]));

        let entry = LibraryEntry {
            arity: 1,
            term: "(lam 0)".into(),
        };
        s.apply(&Event::LibraryExtended {
            entries: vec![entry.clone()],
        })
        .unwrap();
        assert_eq!(s.library(), &[entry]);

        s.apply(&Event::Logged(LogRecord {
            kind: LogKind::Note,
            iteration: 0,
            task: Some("b".into()),
            text: "p".into(),
        }))
        .unwrap();
        assert_eq!(s.log_count(LogKind::Note), 1);
        assert_eq!(s.log_count(LogKind::Compression), 0);

        s.apply(&Event::IterationFinished {
            iteration: 0,
            used: 0,
        })
        .unwrap();
        assert_eq!(s.iteration(), 1);
    }

    #[test]
    fn a_rejected_event_leaves_the_state_unchanged() {
        let s = new_state();
        let log = |iteration, task: Option<&str>| {
            Event::Logged(LogRecord {
                kind: LogKind::Note,
                iteration,
                task: task.map(str::to_owned),
                text: String::new(),
            })
        };
        let cases = [
            (
                Event::Created {
                    config: config(),
                    tasks: vec!["a".into()],
                },
                Rejected::CreatedTwice,
            ),
            (
                Event::FrontierSet {
                    task: "c".into(),
                    members: vec![],
                },
                Rejected::UnknownTask("c".into()),
            ),
            (
                Event::LibraryExtended { entries: vec![] },
                Rejected::NoEntries,
            ),
            (
                Event::IterationFinished {
                    iteration: 1,
                    used: 0,
                },
                Rejected::WrongIteration {
                    current: 0,
                    given: 1,
                },
            ),
            (
                log(3, None),
                Rejected::WrongIteration {
                    current: 0,
                    given: 3,
                },
            ),
            (log(0, Some("c")), Rejected::UnknownTask("c".into())),
        ];
        for (event, why) in cases {
            let mut t = s.clone();
            assert_eq!(t.apply(&event), Err(why), "{event:?}");
            assert_eq!(t, s, "{event:?}");
        }
    }

    #[test]
    fn the_iteration_number_does_not_wrap() {
        let mut s = new_state();
        s.iteration = u64::MAX;
        assert_eq!(
            s.apply(&Event::IterationFinished {
                iteration: u64::MAX,
                used: 0,
            }),
            Err(Rejected::IterationOverflow)
        );
        assert_eq!(s.iteration(), u64::MAX);
    }

    #[test]
    fn replay_needs_a_created_event_first() {
        let created = Event::Created {
            config: config(),
            tasks: vec!["a".into(), "b".into()],
        };
        let finished = Event::IterationFinished {
            iteration: 0,
            used: 0,
        };
        assert_eq!(RunState::replay([]), Err(Rejected::NotCreated));
        assert_eq!(RunState::replay([&finished]), Err(Rejected::NotCreated));
        assert_eq!(RunState::replay([&created]), Ok(new_state()));
        assert_eq!(
            RunState::replay([&created, &finished]).unwrap().iteration(),
            1
        );
        assert_eq!(
            RunState::replay([&created, &created]),
            Err(Rejected::CreatedTwice)
        );
    }

    fn issued(batch: u64, task: &str, wanted: u32, issued_ms: u64) -> Event {
        Event::BatchIssued(BatchIssued {
            batch,
            iteration: 0,
            task: task.into(),
            wanted,
            worker: "w".into(),
            token: format!("t{batch}"),
            issued_ms,
            deadline_ms: issued_ms + 1000,
            payload: "pay".into(),
        })
    }

    fn sub(
        batch: u64,
        worker: &str,
        iteration: u64,
        received_ms: u64,
        standing: Standing,
        programs: usize,
    ) -> Event {
        sub_token(
            batch,
            worker,
            &format!("t{batch}"),
            iteration,
            received_ms,
            standing,
            programs,
        )
    }

    fn sub_token(
        batch: u64,
        worker: &str,
        token: &str,
        iteration: u64,
        received_ms: u64,
        standing: Standing,
        programs: usize,
    ) -> Event {
        Event::Submitted(Submitted {
            batch,
            worker: worker.into(),
            token: token.into(),
            iteration,
            received_ms,
            model: None,
            programs: vec!["let solve x = x".to_owned(); programs],
            standing,
        })
    }

    fn submitted(batch: u64, received_ms: u64, standing: Standing, programs: usize) -> Event {
        sub(batch, "w", 0, received_ms, standing, programs)
    }

    fn checked(batch: u64, checks: Vec<Check>) -> Event {
        Event::Checked(Checked {
            batch,
            library_len: 0,
            checks,
        })
    }

    fn passed(p: &str) -> Check {
        Check::Passed { program: p.into() }
    }

    #[test]
    fn batches_go_to_the_task_with_the_most_open_slots() {
        // Quota 4, at most 2 programs a batch, tasks a and b.
        let mut s = new_state();
        assert_eq!(s.next_task(0), Some(("a", 2)));
        s.apply(&issued(0, "a", 2, 0)).unwrap();
        assert_eq!(s.open_slots("a", 0), 2);
        assert_eq!(s.next_task(0), Some(("b", 2)));
        s.apply(&issued(1, "b", 2, 0)).unwrap();
        assert_eq!(s.next_task(0), Some(("a", 2)));
        s.apply(&issued(2, "a", 2, 0)).unwrap();
        s.apply(&issued(3, "b", 2, 0)).unwrap();
        assert_eq!(s.next_task(0), None);
        assert!(!s.quota_in());

        // A lease that has ended frees its slots; one still live does not.
        assert_eq!(s.next_task(1000), None);
        assert_eq!(s.open_slots("a", 1001), 4);
        assert_eq!(s.next_task(1001), Some(("a", 2)));

        // A submission with one program of two counts one when it arrives,
        // before it is checked.
        s.apply(&submitted(0, 10, Standing::OnTime, 1)).unwrap();
        assert_eq!(s.open_slots("a", 1001), 3);
        assert_eq!(s.task_counts(0, "a").batches, 2);
        assert_eq!(s.task_counts(0, "a").received, 1);
        assert_eq!(s.task_counts(0, "a").programs, 0);
    }

    #[test]
    fn the_last_batch_of_a_task_asks_only_for_what_is_left() {
        let mut s = new_state();
        s.apply(&issued(0, "a", 2, 0)).unwrap();
        s.apply(&submitted(0, 5, Standing::OnTime, 2)).unwrap();
        s.apply(&issued(1, "a", 1, 0)).unwrap();
        // a has 4 - 2 received - 1 leased = 1 open slot; b has 4.
        assert_eq!(s.open_slots("a", 0), 1);
        s.apply(&issued(2, "b", 2, 0)).unwrap();
        s.apply(&issued(3, "b", 2, 0)).unwrap();
        assert_eq!(s.next_task(0), Some(("a", 1)));
    }

    #[test]
    fn checks_are_counted_by_stage_and_submissions_by_standing() {
        let mut s = new_state();
        for b in 0..4 {
            s.apply(&issued(b, "a", 2, 0)).unwrap();
        }
        s.apply(&submitted(0, 10, Standing::OnTime, 3)).unwrap();
        s.apply(&submitted(1, 2000, Standing::Expired, 2)).unwrap();
        // A later submission from the batch's worker is a repeat; one from
        // another worker is foreign and leaves the batch open.
        s.apply(&submitted(0, 20, Standing::Repeat, 1)).unwrap();
        s.apply(&sub(2, "intruder", 0, 20, Standing::Foreign, 1))
            .unwrap();
        assert_eq!(s.standing(2, "w", "t2", 30), Some(Standing::OnTime));
        // The wrong token, or one of another length, makes a submission
        // foreign.
        for token in ["t3", "t", "t20", ""] {
            assert_eq!(s.standing(2, "w", token, 30), Some(Standing::Foreign));
        }
        assert_eq!(s.pending_checks(), [0, 1]);
        assert!(!s.quota_in());

        // `Extra` is only right after the programs the batch asked for, and
        // there is one check for each program.
        for bad in [
            vec![
                Check::Compile { error: "c".into() },
                Check::Extra,
                Check::Type { error: "t".into() },
            ],
            vec![Check::Compile { error: "c".into() }],
        ] {
            let mut t = s.clone();
            assert_eq!(t.apply(&checked(0, bad)), Err(Rejected::BadSubmission(0)));
            assert_eq!(t, s);
        }
        let stale = Event::Checked(Checked {
            batch: 0,
            library_len: 1,
            checks: vec![],
        });
        let mut t = s.clone();
        assert!(matches!(
            t.apply(&stale),
            Err(Rejected::StaleLibrary { batch: 0, .. })
        ));
        assert_eq!(t, s);
        let mut t = s.clone();
        assert_eq!(
            t.apply(&checked(2, vec![])),
            Err(Rejected::NotAwaitingCheck(2))
        );
        assert_eq!(t, s);

        // The iteration cannot end while checks are pending.
        assert_eq!(
            s.clone().apply(&Event::IterationFinished {
                iteration: 0,
                used: 0
            }),
            Err(Rejected::ChecksPending(0))
        );

        s.apply(&checked(
            0,
            vec![
                Check::Compile { error: "c".into() },
                Check::Type { error: "t".into() },
                Check::Extra,
            ],
        ))
        .unwrap();
        s.apply(&checked(
            1,
            vec![Check::Run { error: "r".into() }, passed("p1")],
        ))
        .unwrap();
        assert_eq!(
            s.clone().apply(&checked(1, vec![passed("x"), passed("y")])),
            Err(Rejected::NotAwaitingCheck(1))
        );
        assert_eq!(
            s.task_counts(0, "a"),
            TaskCounts {
                batches: 4,
                received: 4,
                programs: 4,
                compiled: 3,
                typed: 2,
                passed: 1,
            }
        );
        let ic = s.iteration_counts(0);
        assert_eq!(
            (ic.submissions, ic.late, ic.repeats, ic.foreign),
            (4, 0, 1, 1)
        );
        assert_eq!(ic.payload_bytes, 4 * 3);
        assert_eq!(
            s.verified(),
            &[Verified {
                task: "a".into(),
                program: "p1".into()
            }]
        );

        // The iteration step cannot take more programs than are verified.
        assert_eq!(
            s.clone().apply(&Event::IterationFinished {
                iteration: 0,
                used: 2,
            }),
            Err(Rejected::UnknownVerified {
                used: 2,
                verified: 1
            })
        );
        // A program checked after the step read the verified programs is
        // kept for the next step.
        let mut later = s.clone();
        later
            .apply(&Event::IterationFinished {
                iteration: 0,
                used: 0,
            })
            .unwrap();
        assert_eq!(later.verified(), s.verified());

        // The iteration ends: a submission for its batch 2 is late, kept, not
        // counted, and the verified programs the step took are gone.
        s.apply(&Event::IterationFinished {
            iteration: 0,
            used: 1,
        })
        .unwrap();
        assert!(s.verified().is_empty());
        assert_eq!(s.standing(2, "w", "t2", 10), Some(Standing::Late));
        s.apply(&sub(2, "w", 1, 10, Standing::Late, 1)).unwrap();
        assert!(s.pending_checks().is_empty());
        assert_eq!(s.task_counts(1, "a"), TaskCounts::default());
        assert_eq!(s.iteration_counts(1).late, 1);
        assert_eq!(s.standing(2, "w", "t2", 10), Some(Standing::Repeat));
    }

    #[test]
    fn a_submission_must_have_the_standing_the_state_gives() {
        let mut s = new_state();
        s.apply(&issued(0, "a", 1, 0)).unwrap();
        for event in [
            submitted(0, 10, Standing::Expired, 1),
            submitted(0, 5000, Standing::OnTime, 1),
            submitted(0, 10, Standing::Late, 1),
            submitted(0, 10, Standing::Foreign, 1),
            sub(0, "other", 0, 10, Standing::OnTime, 1),
            sub_token(0, "w", "t1", 0, 10, Standing::OnTime, 1),
        ] {
            let mut t = s.clone();
            assert_eq!(
                t.apply(&event),
                Err(Rejected::BadSubmission(0)),
                "{event:?}"
            );
            assert_eq!(t, s);
        }
        let mut t = s.clone();
        assert_eq!(
            t.apply(&submitted(7, 10, Standing::OnTime, 1)),
            Err(Rejected::UnknownBatch(7))
        );
        assert_eq!(
            t.apply(&sub(0, "w", 1, 10, Standing::OnTime, 1)),
            Err(Rejected::WrongIteration {
                current: 0,
                given: 1
            })
        );
        assert_eq!(t, s);
    }

    #[test]
    fn a_batch_must_have_the_next_number_and_a_lease() {
        let mut s = new_state();
        assert_eq!(
            s.apply(&issued(1, "a", 1, 0)),
            Err(Rejected::WrongBatchNumber { next: 0, given: 1 })
        );
        assert_eq!(s.apply(&issued(0, "a", 0, 0)), Err(Rejected::BadBatch));
        assert_eq!(
            s.apply(&issued(0, "c", 1, 0)),
            Err(Rejected::UnknownTask("c".into()))
        );
        let mut no_lease = issued(0, "a", 1, 0);
        if let Event::BatchIssued(b) = &mut no_lease {
            b.deadline_ms = b.issued_ms;
        }
        assert_eq!(s.apply(&no_lease), Err(Rejected::BadBatch));
        let mut no_token = issued(0, "a", 1, 0);
        if let Event::BatchIssued(b) = &mut no_token {
            b.token.clear();
        }
        assert_eq!(s.apply(&no_token), Err(Rejected::BadBatch));
        assert_eq!(s, new_state());
    }

    #[test]
    fn the_quota_is_in_when_every_task_has_its_programs_checked() {
        let mut s = RunState::create(
            RunConfig {
                proposals_per_task: 1,
                ..config()
            },
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        s.apply(&issued(0, "a", 1, 0)).unwrap();
        s.apply(&issued(1, "b", 1, 0)).unwrap();
        s.apply(&submitted(0, 1, Standing::OnTime, 1)).unwrap();
        s.apply(&submitted(1, 1, Standing::OnTime, 1)).unwrap();
        assert!(!s.quota_in(), "checks are pending");
        assert_eq!(s.next_task(0), None);
        s.apply(&checked(0, vec![Check::Compile { error: "x".into() }]))
            .unwrap();
        assert!(!s.quota_in());
        s.apply(&checked(1, vec![passed("p")])).unwrap();
        assert!(s.quota_in());
    }

    /// A run with `past` finished iterations, each with four batches: three
    /// submitted and checked, and one never submitted. The current iteration
    /// has four batches: one submitted and waiting to be checked, one
    /// submitted by another worker (so still leased), and two leased.
    ///
    /// It applies its events with `apply_checked`, without the copy that
    /// `apply` makes in tests: every event applies, and the copy of a
    /// growing state at each of its thousands of events would make the
    /// test take seconds.
    fn long_run(past: u64) -> RunState {
        let mut s = new_state();
        let issue = |s: &mut RunState, batch: u64, task: &str, deadline_ms: u64| {
            s.apply_checked(&Event::BatchIssued(BatchIssued {
                batch,
                iteration: s.iteration(),
                task: task.into(),
                wanted: 2,
                worker: "w".into(),
                token: format!("t{batch}"),
                issued_ms: 0,
                deadline_ms,
                payload: String::new(),
            }))
            .unwrap();
        };
        for i in 0..past {
            let first = i * 4;
            for (k, task) in ["a", "a", "b", "b"].into_iter().enumerate() {
                issue(&mut s, first + k as u64, task, 1000);
            }
            for batch in first..first + 3 {
                s.apply_checked(&sub(batch, "w", i, 10, Standing::OnTime, 2))
                    .unwrap();
                s.apply_checked(&checked(
                    batch,
                    vec![Check::Compile { error: "e".into() }; 2],
                ))
                .unwrap();
            }
            s.apply_checked(&Event::IterationFinished {
                iteration: i,
                used: 0,
            })
            .unwrap();
        }
        let first = past * 4;
        for (k, task) in ["a", "a", "b", "b"].into_iter().enumerate() {
            issue(&mut s, first + k as u64, task, 1_000_000);
        }
        s.apply_checked(&sub(first, "w", past, 10, Standing::OnTime, 2))
            .unwrap();
        s.apply_checked(&sub(first + 1, "intruder", past, 10, Standing::Foreign, 2))
            .unwrap();
        s
    }

    /// What the calls of [`hot_calls`] found.
    #[derive(Debug, PartialEq)]
    struct Found<'a> {
        pending: Vec<u64>,
        first_pending: Option<u64>,
        awaits: bool,
        quota_in: bool,
        /// `quota_in` once the pending check is done, so that it reads the
        /// task counts.
        quota_in_once_checked: bool,
        next_task: Option<(&'a str, u32)>,
        open_slots: u64,
        leased: Vec<u64>,
        standing: Option<Standing>,
        wanted: Option<u32>,
        finish: Result<(), Rejected>,
    }

    /// Makes each call a proposer or the operator makes all the time on `s`,
    /// whose first batch of the current iteration is `first`, and returns
    /// what they found and how many batches they read.
    fn hot_calls(s: &RunState, first: u64) -> (Found<'_>, u64) {
        let now = 500;
        let mut copy = s.clone();
        let finish = Event::IterationFinished {
            iteration: s.iteration(),
            used: 0,
        };
        let mut checked_copy = s.clone();
        checked_copy
            .apply(&Event::Checked(Checked {
                batch: first,
                library_len: s.library().len() as u64,
                checks: vec![Check::Compile { error: "e".into() }; 2],
            }))
            .unwrap();
        history::READ.with(|n| n.set(0));
        let found = Found {
            pending: s.pending_checks(),
            first_pending: s.first_pending_check(),
            awaits: s.awaits_check(first),
            quota_in: s.quota_in(),
            quota_in_once_checked: checked_copy.quota_in(),
            next_task: s.next_task(now),
            open_slots: s.open_slots("a", now) + s.open_slots("b", now),
            leased: s.leased(now).map(|(i, _)| i).collect(),
            standing: s.standing(first + 2, "w", &format!("t{}", first + 2), now),
            wanted: s.batch(first).map(|b| b.wanted),
            finish: copy.apply(&finish),
        };
        (found, history::READ.with(std::cell::Cell::get))
    }

    /// The indexes hold what a search of every batch finds.
    fn assert_indexes_hold(s: &RunState) {
        let all = s.batches();
        let live: BTreeSet<u64> = (0..all.len() as u64)
            .filter(|&i| {
                let b = &all[i as usize];
                b.iteration == s.iteration && !b.submitted
            })
            .collect();
        let awaiting: BTreeSet<u64> = (0..all.len() as u64)
            .filter(|&i| all[i as usize].awaiting_check.is_some())
            .collect();
        assert_eq!(s.live, live);
        assert_eq!(s.awaiting, awaiting);
        // And the task counts count every batch once.
        let counted: u64 = s
            .task_counts
            .iter()
            .flat_map(|(_, by_task)| by_task.values())
            .map(|c| c.batches)
            .sum();
        assert_eq!(counted, s.batch_count());
    }

    #[test]
    fn searches_read_only_the_batches_of_the_current_iteration() {
        let (few, many) = (long_run(10), long_run(1000));
        assert_indexes_hold(&few);
        assert_indexes_hold(&many);
        let (short, short_read) = hot_calls(&few, 40);
        let (long, long_read) = hot_calls(&many, 4000);
        // The calls find the same batches, by number, after 10 and after
        // 1000 finished iterations.
        let expect = |first: u64| Found {
            pending: vec![first],
            first_pending: Some(first),
            awaits: true,
            quota_in: false,
            quota_in_once_checked: false,
            next_task: None,
            open_slots: 0,
            leased: vec![first + 1, first + 2, first + 3],
            standing: Some(Standing::OnTime),
            wanted: Some(2),
            finish: Err(Rejected::ChecksPending(first)),
        };
        assert_eq!(short, expect(40));
        assert_eq!(long, expect(4000));
        // And they read no more of the run's history after 1000 than after
        // 10. Five of the calls search the three live leases (next_task
        // searches once for each task), two read one batch by its number,
        // and six read one task's counts in the current iteration (each
        // open_slots, and quota_in once checked for each task).
        assert_eq!(long_read, short_read);
        assert!(short_read <= 5 * 3 + 2 + 6, "{short_read}");
    }

    #[test]
    fn the_indexes_follow_submissions_checks_and_the_end_of_an_iteration() {
        let mut s = new_state();
        for b in 0..4 {
            s.apply(&issued(b, "a", 1, 0)).unwrap();
        }
        assert_indexes_hold(&s);
        // Foreign leaves the batch live; on time and expired take it out
        // and make it wait for a check; a repeat changes neither.
        s.apply(&sub(0, "intruder", 0, 5, Standing::Foreign, 1))
            .unwrap();
        s.apply(&submitted(1, 5, Standing::OnTime, 1)).unwrap();
        s.apply(&submitted(2, 5000, Standing::Expired, 1)).unwrap();
        s.apply(&submitted(1, 6, Standing::Repeat, 1)).unwrap();
        assert_indexes_hold(&s);
        assert_eq!(s.leased(0).map(|(i, _)| i).collect::<Vec<_>>(), [0, 3]);
        assert_eq!(s.pending_checks(), [1, 2]);
        assert!(s.awaits_check(2) && !s.awaits_check(0));
        s.apply(&checked(1, vec![passed("p")])).unwrap();
        s.apply(&checked(2, vec![passed("q")])).unwrap();
        assert_indexes_hold(&s);
        s.apply(&Event::IterationFinished {
            iteration: 0,
            used: 2,
        })
        .unwrap();
        // Batches 0 and 3 were never submitted; they are not live in the new
        // iteration, and a late submission does not bring them back.
        assert_indexes_hold(&s);
        assert_eq!(s.leased(0).count(), 0);
        s.apply(&sub(3, "w", 1, 10, Standing::Late, 1)).unwrap();
        assert_indexes_hold(&s);
        assert!(s.pending_checks().is_empty());
    }
}
