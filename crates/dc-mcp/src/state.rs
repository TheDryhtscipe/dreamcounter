//! The state of one run, and the events that change it.
//!
//! A [`RunState`] is the result of applying a run's events in order, starting
//! from [`RunState::create`]. Nothing else changes it, so the journal of events
//! on disk (see [`crate::journal`]) holds the whole state.
//!
//! Programs and library bodies are text here. The state does not parse them:
//! the code that evaluates and compresses them does.

use std::collections::BTreeMap;
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
    /// checked.
    Repeat,
    /// A submission that gives the wrong token for the batch, or names a
    /// worker the batch was not issued to. It is kept but not checked, and
    /// the batch stays open for the proposer it was sent to.
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
    /// `proposals_per_task`, `programs_per_batch` or `lease_ms` is zero.
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

/// A program of the current iteration that passed its task's examples.
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    config: RunConfig,
    tasks: Vec<String>,
    library: Vec<LibraryEntry>,
    frontiers: BTreeMap<String, Vec<FrontierMember>>,
    iteration: u64,
    log_counts: BTreeMap<LogKind, u64>,
    batches: Vec<Batch>,
    task_counts: BTreeMap<(u64, String), TaskCounts>,
    iteration_counts: BTreeMap<u64, IterationCounts>,
    verified: Vec<Verified>,
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
            batches: Vec::new(),
            task_counts: BTreeMap::new(),
            iteration_counts: BTreeMap::new(),
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

    /// Changes the state by `event`. On an error the state is unchanged.
    pub fn apply(&mut self, event: &Event) -> Result<(), Rejected> {
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
                if let Some(b) = self.pending_checks().first() {
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
        });
        self.task_counts
            .entry((issued.iteration, issued.task.clone()))
            .or_default()
            .batches += 1;
        let counts = self.iteration_counts.entry(issued.iteration).or_default();
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
        let b = self.batches.get(usize::try_from(batch).ok()?)?;
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

    fn submit(&mut self, s: &Submitted) -> Result<(), Rejected> {
        self.check_iteration(s.iteration)?;
        let standing = self
            .standing(s.batch, &s.worker, &s.token, s.received_ms)
            .ok_or(Rejected::UnknownBatch(s.batch))?;
        if standing != s.standing {
            return Err(Rejected::BadSubmission(s.batch));
        }
        let batch = &mut self.batches[s.batch as usize];
        if standing != Standing::Foreign {
            batch.submitted = true;
        }
        let (iteration, task, wanted) = (batch.iteration, batch.task.clone(), batch.wanted);
        if standing.evaluated() {
            batch.awaiting_check = Some(s.programs.clone());
            let received = (s.programs.len() as u64).min(u64::from(wanted));
            self.task_counts
                .entry((iteration, task))
                .or_default()
                .received += received;
        }
        let counts = self.iteration_counts.entry(s.iteration).or_default();
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
        let index = usize::try_from(c.batch)
            .ok()
            .filter(|&i| i < self.batches.len())
            .ok_or(Rejected::UnknownBatch(c.batch))?;
        let library = self.library.len() as u64;
        let batch = &self.batches[index];
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
        let (iteration, task) = (batch.iteration, batch.task.clone());
        self.batches[index].awaiting_check = None;
        let tc = self
            .task_counts
            .entry((iteration, task.clone()))
            .or_default();
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
        // An iteration cannot end with a check pending, so the batch is of
        // the current iteration.
        for check in &c.checks {
            if let Check::Passed { program } = check {
                self.verified.push(Verified {
                    task: task.clone(),
                    program: program.clone(),
                });
            }
        }
        Ok(())
    }

    /// The batches whose submission waits to be checked, in order.
    pub fn pending_checks(&self) -> Vec<u64> {
        self.batches
            .iter()
            .enumerate()
            .filter(|(_, b)| b.awaiting_check.is_some())
            .map(|(i, _)| i as u64)
            .collect()
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

    /// Every batch issued, in order: batch `i` is entry `i`.
    pub fn batches(&self) -> &[Batch] {
        &self.batches
    }

    /// What came in for `task` in `iteration`.
    pub fn task_counts(&self, iteration: u64, task: &str) -> TaskCounts {
        self.task_counts
            .get(&(iteration, task.to_owned()))
            .copied()
            .unwrap_or_default()
    }

    /// What came in during `iteration`.
    pub fn iteration_counts(&self, iteration: u64) -> IterationCounts {
        self.iteration_counts
            .get(&iteration)
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
            .batches
            .iter()
            .filter(|b| {
                b.iteration == self.iteration
                    && b.task == task
                    && !b.submitted
                    && now_ms <= b.deadline_ms
            })
            .map(|b| u64::from(b.wanted))
            .sum();
        u64::from(self.config.proposals_per_task).saturating_sub(received.saturating_add(leased))
    }

    /// Whether every task has received its programs for the current
    /// iteration and every one of them has been checked.
    pub fn quota_in(&self) -> bool {
        let quota = u64::from(self.config.proposals_per_task);
        self.pending_checks().is_empty()
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
            assert_eq!(
                s.clone().apply(&checked(0, bad)),
                Err(Rejected::BadSubmission(0))
            );
        }
        let stale = Event::Checked(Checked {
            batch: 0,
            library_len: 1,
            checks: vec![],
        });
        assert!(matches!(
            s.clone().apply(&stale),
            Err(Rejected::StaleLibrary { batch: 0, .. })
        ));
        assert_eq!(
            s.clone().apply(&checked(2, vec![])),
            Err(Rejected::NotAwaitingCheck(2))
        );

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
}
