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

use dc_core::sexpr;
use dc_core::term::{children, Arena, LibId, Term};
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
    /// kept (see [`Store::commit_or_refuse`](crate::store::Store::commit_or_refuse)). It is positive. A run made before the
    /// setting has [`DEFAULT_MAX_UNCHECKED_PER_BATCH`].
    #[serde(default = "default_max_unchecked_per_batch")]
    pub max_unchecked_per_batch: u32,
    /// The most programs a task's frontier keeps when an iteration ends
    /// (K of decision D19). It is positive.
    pub frontier_size: u32,
    /// The settings of the Stitch run that ends each iteration.
    pub stitch: StitchSettings,
    /// The fingerprint (`dc_surface::prelude::Prelude::fingerprint`) of the
    /// prelude the run's programs are compiled with. A stored program names
    /// a prelude function by its position, so it keeps its meaning only
    /// under that prelude: see [`crate::verify::check_prelude`].
    pub prelude: u64,
    /// Whether proposers name and document new abstractions (LILO's
    /// AutoDoc, decision D23). While it is on, no batch is issued until
    /// every library entry has a naming result. A run made before the
    /// setting has it off.
    #[serde(default)]
    pub autodoc: bool,
    /// Whether the run learns a library: each iteration's step runs Stitch
    /// and adds what it finds. With it off the step only selects frontiers
    /// and the library stays empty, which makes the run the LLM-only
    /// baseline (decisions D8 and D24). A run made before the setting has
    /// it on.
    #[serde(default = "learns_library")]
    pub library_learning: bool,
    /// The most memory one evaluation may hold, in bytes
    /// (`dc_tasks::harness::HarnessConfig::max_memory`): each run of a
    /// program on an example, a gate input or a size of its sweep. A run
    /// that would hold more stops with a memory limit failure. It is
    /// positive. A run made before the setting has
    /// `dc_tasks::harness::DEFAULT_MAX_MEMORY`.
    #[serde(default = "default_max_memory")]
    pub max_memory: u64,
    /// The id of the run this one is compared with, such as the main run
    /// of an LLM-only baseline (decision D24). `dc_create_run` checks that
    /// the two runs share every setting that fixes the proposals they hand
    /// out. A run made before the setting has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compared_with: Option<String>,
}

/// The [`RunConfig::library_learning`] of a run made before the setting.
fn learns_library() -> bool {
    true
}

/// The [`RunConfig::max_memory`] of a run made before the setting.
fn default_max_memory() -> u64 {
    dc_tasks::harness::DEFAULT_MAX_MEMORY
}

/// The settings of the Stitch run that ends each iteration: what Stitch
/// looks for, and the limits on its worker process
/// (`dc_compress::worker::Limits`). A run that ends at a limit ends the
/// iteration without new abstractions (decision D19), so the limits are the
/// run's, not the server's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StitchSettings {
    /// The most abstractions one run finds. It is positive.
    pub iterations: u32,
    /// The largest arity of an abstraction.
    pub max_arity: u32,
    /// The longest the server waits for the worker, in milliseconds. It is
    /// positive.
    pub wall_ms: u64,
    /// The worker's address space, in bytes. Stitch's thread reserves a
    /// stack of 256 MiB, so this is at least [`MIN_STITCH_MEMORY_BYTES`].
    pub memory_bytes: u64,
}

/// The least address space a run may give its Stitch worker: below about
/// 260 MiB Stitch's thread cannot start, and a little above it Stitch runs
/// many times slower (`dc_compress::worker::Limits::memory_bytes`).
pub const MIN_STITCH_MEMORY_BYTES: u64 = 512 << 20;

impl Default for StitchSettings {
    /// Stitch's own defaults, 3 abstractions of arity at most 2, and the
    /// worker's default limits: ten minutes and 4 GiB.
    fn default() -> StitchSettings {
        StitchSettings {
            iterations: 3,
            max_arity: 2,
            wall_ms: 600_000,
            memory_bytes: 4 << 30,
        }
    }
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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    /// The number of parameters.
    pub arity: u32,
    /// The closed term `λ^arity. body`, as an s-expression (`dc_core::sexpr`).
    pub term: String,
    /// The name a proposer gave it (decision D23), if any. Without one it is
    /// `fn_i`. A [`Event::LibraryExtended`] entry has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// What it does, in one line, as a proposer wrote it, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc: Option<String>,
}

/// One program in a task's frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FrontierMember {
    /// The program, a core term as an s-expression (`dc_core::sexpr`).
    pub program: String,
    /// What the program costs on its task's sweep, measured when the
    /// iteration that kept it ended (`crate::cost`). `None` in a run made
    /// before costs were measured. It plays no part in which programs are
    /// kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ProgramCost>,
}

/// What a frontier program costs on its task's sweep (`crate::cost`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ProgramCost {
    /// The program ran at every size of the sweep, and every run that
    /// finished gave the right output.
    Measured {
        /// Each size of the sweep, in order.
        sizes: Vec<SizeCost>,
        /// Estimator C's verdict, with its default settings.
        verdict: CostVerdict,
    },
    /// The program gave a wrong output or a runtime error on a generated
    /// input. It passed its task's examples, but it is not a solution, so it
    /// gets no verdict.
    Incorrect {
        /// Each size of the sweep, in order.
        sizes: Vec<SizeCost>,
        /// The smallest size with such a failure.
        n: u64,
        /// The seed that failed there.
        seed: u64,
    },
    /// The program could not be run on the sweep: it did not expand, or it
    /// failed its task's type or examples, or the task has no fuel plan, or
    /// measuring it failed. The next iteration that keeps it measures it
    /// again.
    Refused {
        /// Why.
        reason: String,
    },
    /// The program was not measured: the time for measuring ran out, or the
    /// call was cancelled. The next iteration that keeps it measures it.
    NotMeasured {
        /// Why.
        reason: String,
    },
}

/// One size of a frontier program's sweep.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SizeCost {
    /// The input size.
    pub n: u64,
    /// The fuel of each run, from the task's fuel plan: ten times the
    /// reference's steps at this size, within the harness's cap, or the cap
    /// if the reference did not finish within it.
    pub fuel: u64,
    /// The step count of each seed, in seed order, if every seed finished
    /// with the right output; otherwise empty.
    pub steps: Vec<u64>,
    /// The median of `steps`, if there are any.
    pub median: Option<u64>,
    /// The first seed that failed at this size, and how. Later seeds were
    /// not run.
    pub failed: Option<SizeFailure>,
}

/// The first failed run at one size of a sweep.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SizeFailure {
    /// The seed.
    pub seed: u64,
    /// How it failed.
    pub kind: FailureKind,
}

/// How a run on the sweep failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// It used all its fuel. This is a result, not an error.
    Timeout,
    /// It held more memory than the harness's limit
    /// (`dc_tasks::harness::HarnessConfig::max_memory`). Like a timeout, this
    /// is a result, not an error.
    MemoryLimit,
    /// It stopped with a runtime error.
    Error,
    /// It finished with the wrong output.
    WrongOutput,
}

/// Estimator C's verdict on a sweep (`dc_tasks::growth::ClassVerdict`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CostVerdict {
    /// C names this class.
    Class(CostClass),
    /// The medians are too noisy to name a class. `best` has the smallest
    /// misfit.
    TooNoisy {
        /// The class with the smallest misfit.
        best: CostClass,
    },
    /// Some size has one seed, so the noise is unknown.
    NoiseUnknown {
        /// The class with the smallest misfit.
        best: CostClass,
    },
    /// No class fits: the growth is between classes or outside them.
    NoFit {
        /// The class with the smallest misfit.
        best: CostClass,
    },
    /// C did not fit the sweep, for example because sizes timed out.
    NotFitted {
        /// C's reason.
        reason: String,
    },
}

/// A growth class of estimator C (`dc_tasks::growth::GrowthClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum CostClass {
    /// `1`.
    #[serde(rename = "1")]
    Constant,
    /// `log n`.
    #[serde(rename = "log n")]
    Log,
    /// `n`.
    #[serde(rename = "n")]
    Linear,
    /// `n log n`.
    #[serde(rename = "n log n")]
    NLogN,
    /// `n²`.
    #[serde(rename = "n^2")]
    Quadratic,
    /// `n³`.
    #[serde(rename = "n^3")]
    Cubic,
    /// `2ⁿ`.
    #[serde(rename = "2^n")]
    Exponential,
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
    /// The payload handed to the proposer. `crate::prompt::PromptTable`
    /// gives it back exactly as it was sent.
    pub payload: IssuedPayload,
}

/// How a [`BatchIssued`] event holds its payload (dreamcounter-20c6e5).
///
/// In the journal the first form is a JSON string and the second a JSON
/// object, so each reads back as the form it was written in.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IssuedPayload {
    /// The payload as it was sent: the JSON text of a
    /// [`Payload`](crate::prompt::Payload), whole. Runs written before
    /// prompts were stored once hold this form. It still reads, and means
    /// what it meant; the server no longer writes it.
    Whole(String),
    /// The payload with its prompt in parts. The batch number, token and
    /// programs wanted are the event's own fields.
    Parts(PayloadParts),
}

/// A payload whose prompt is made of parts, some of them stored once in the
/// run by [`Event::PromptStored`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadParts {
    /// The lease the payload gave, in seconds.
    pub lease_seconds: u64,
    /// The prompt: its parts joined in order, with nothing between them.
    pub prompt: Vec<PromptPart>,
}

/// A part of a prompt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum PromptPart {
    /// Text that is written out in the event.
    Text(String),
    /// The text of the stored prompt with this id.
    Stored(u64),
}

/// A piece of prompt text, stored once in the run so that the payloads that
/// use it can name it (dreamcounter-20c6e5).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptStored {
    /// The prompt's id. Prompts are numbered 0, 1, ... over the whole run.
    pub id: u64,
    /// The text. No other stored prompt of the run has the same text.
    pub text: String,
}

/// What the state keeps of the run's stored prompts: a digest of each text
/// and the length of the text as a JSON string, but not the text itself.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct StoredPrompts {
    /// The id of each stored prompt, by the SHA-256 digest of its text.
    ids: BTreeMap<[u8; 32], u64>,
    /// The length in bytes of each stored prompt's text when it is written
    /// inside a JSON string, by id.
    json_lens: Vec<u64>,
}

/// The SHA-256 digest of `text`.
fn digest(text: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(text.as_bytes()).into()
}

/// The length in bytes of `text` when it is written inside a JSON string,
/// without the quotes.
fn json_len(text: &str) -> u64 {
    serde_json::to_string(text).map_or(0, |s| s.len().saturating_sub(2)) as u64
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

/// A naming job is handed to a proposer (decision D23).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamingIssued {
    /// The job's number. Jobs are numbered 0, 1, ... over the whole run.
    pub job: u64,
    /// The library entry to name.
    pub entry: u64,
    /// The worker id the proposer gave.
    pub worker: String,
    /// The secret the proposer must give back with its answer.
    pub token: String,
    /// When the job was issued, in milliseconds since the Unix epoch.
    pub issued_ms: u64,
    /// When its lease ends. After that the entry can get another job.
    pub deadline_ms: u64,
    /// The prompt the proposer was given.
    pub prompt: String,
}

/// A naming job ends (decision D23).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NamingDone {
    /// The job.
    pub job: u64,
    /// The entry it named, which is the job's.
    pub entry: u64,
    /// The name the entry gets, or `None` if it keeps `fn_i`.
    pub name: Option<String>,
    /// The doc the entry gets, if any.
    pub doc: Option<String>,
    /// What the proposer gave that was refused, and why, in words.
    pub refused: Vec<String>,
}

/// What the state keeps of a naming job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamingJob {
    /// The library entry to name.
    pub entry: u64,
    /// The worker id the proposer gave.
    pub worker: String,
    /// The job's secret.
    pub token: String,
    /// When its lease ends.
    pub deadline_ms: u64,
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
    /// A piece of prompt text is stored, for payloads to name.
    PromptStored(PromptStored),
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
    /// A naming job for a library entry is handed to a proposer (D23).
    NamingIssued(NamingIssued),
    /// A naming job ends: the entry gets its name and doc, or keeps `fn_i`.
    NamingDone(NamingDone),
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
pub(crate) fn same_secret(secret: &str, given: &str) -> bool {
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
    /// A setting that must be positive is zero: `proposals_per_task`,
    /// `programs_per_batch`, `lease_ms`, `max_unchecked_per_batch`,
    /// `frontier_size`, or the Stitch run's `iterations` or `wall_ms`.
    ZeroSetting(&'static str),
    /// The Stitch worker's memory is under [`MIN_STITCH_MEMORY_BYTES`].
    TooLittleStitchMemory(u64),
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
    /// A naming event does not fit the run, in words (decision D23).
    Naming(String),
    /// A `LibraryExtended` event has no entries.
    NoEntries,
    /// A `LibraryExtended` event came in a run that does not learn a
    /// library ([`RunConfig::library_learning`] is off).
    NoLearning,
    /// The settings turn autodoc on in a run that does not learn a library.
    AutodocWithoutLearning,
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
    /// A `PromptStored` event does not have the next prompt number.
    WrongPromptNumber {
        /// The next prompt number.
        next: u64,
        /// The number the event gives.
        given: u64,
    },
    /// A `PromptStored` event's text is already stored, under this id.
    DuplicatePrompt(u64),
    /// A `BatchIssued` event names a prompt the run has not stored.
    UnknownPrompt(u64),
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
            Rejected::TooLittleStitchMemory(bytes) => write!(
                f,
                "the Stitch worker's memory is {bytes} bytes; it must be at least \
                 {MIN_STITCH_MEMORY_BYTES}"
            ),
            Rejected::NotCreated => write!(f, "the first event of a run must create it"),
            Rejected::CreatedTwice => write!(f, "the run is already created"),
            Rejected::UnknownTask(t) => write!(f, "the run has no task {t:?}"),
            Rejected::WrongIteration { current, given } => write!(
                f,
                "the event names iteration {given}, but the run is in iteration {current}"
            ),
            Rejected::NoEntries => write!(f, "a library extension needs at least one entry"),
            Rejected::NoLearning => {
                write!(
                    f,
                    "the run does not learn a library (library_learning is off)"
                )
            }
            Rejected::AutodocWithoutLearning => write!(
                f,
                "autodoc needs library learning: a run that does not learn a library has \
                 nothing to name"
            ),
            Rejected::Naming(why) => f.write_str(why),
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
            Rejected::WrongPromptNumber { next, given } => write!(
                f,
                "the event stores prompt {given}, but the next prompt is {next}"
            ),
            Rejected::DuplicatePrompt(id) => {
                write!(f, "the prompt's text is already stored as prompt {id}")
            }
            Rejected::UnknownPrompt(id) => write!(f, "the run has stored no prompt {id}"),
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

/// What came in during one iteration from proposers that named one model, or
/// from those that named none (decision D24).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelCounts {
    /// Submissions received, of any standing.
    pub submissions: u64,
    /// Programs in those submissions.
    pub programs: u64,
}

/// Which learned library entries the programs that passed in one iteration
/// refer to. A program counts an entry once however often it refers to it.
/// Every reference in the compiled program counts, one in a binding that is
/// never used too. Only a direct reference counts: one inside another
/// entry's body does not. Prelude functions are not learned entries and are
/// not counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LibraryUptake {
    /// Programs that passed every example.
    pub passed: u64,
    /// Of those, the programs that refer to at least one learned entry.
    pub using_library: u64,
    /// For each entry referred to, by its index, the programs that refer to it.
    pub entries: BTreeMap<u64, EntryUptake>,
}

/// The passed programs of one iteration that refer to one library entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EntryUptake {
    /// How many programs refer to it.
    pub programs: u64,
    /// The tasks of those programs.
    pub tasks: BTreeSet<String>,
}

/// The learned library entries that `program`, a core term as an
/// s-expression, refers to directly: its references below `library_len`. The
/// prelude's references are far above any learned index, so they are left
/// out. A program that does not parse refers to none; a passed program is
/// printed by the server, so it always parses.
fn learned_refs(program: &str, library_len: u64) -> BTreeSet<u64> {
    let mut arena = Arena::new();
    let Ok(root) = sexpr::parse(&mut arena, program) else {
        return BTreeSet::new();
    };
    // The arena shares equal subterms, so each node is visited once.
    let mut seen = BTreeSet::new();
    let mut stack = vec![root];
    let mut refs = BTreeSet::new();
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let term = arena.get(t);
        if let Term::Lib(LibId(l)) = term {
            if u64::from(l) < library_len {
                refs.insert(u64::from(l));
            }
        }
        stack.extend(children(&term));
    }
    refs
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
/// That text is in the journal. The state keeps counts. Of each stored
/// prompt ([`Event::PromptStored`]) it keeps a SHA-256 digest of the text and
/// its length, so that a new payload can name a prompt that is already
/// stored (see [`RunState::stored_prompt`]).
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
    kept: ByIteration<BTreeMap<String, Vec<FrontierMember>>>,
    iteration: u64,
    log_counts: BTreeMap<LogKind, u64>,
    batches: Batches,
    prompts: StoredPrompts,
    live: BTreeSet<u64>,
    awaiting: BTreeSet<u64>,
    task_counts: ByIteration<BTreeMap<String, TaskCounts>>,
    iteration_counts: ByIteration<IterationCounts>,
    model_counts: ByIteration<BTreeMap<Option<String>, ModelCounts>>,
    verified: Vec<Verified>,
    naming_jobs: Vec<NamingJob>,
    named: Vec<bool>,
    /// The iteration whose end added each library entry, by index.
    added: Vec<u64>,
    uptake: ByIteration<LibraryUptake>,
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
            ("frontier_size", u64::from(config.frontier_size)),
            ("stitch.iterations", u64::from(config.stitch.iterations)),
            ("stitch.wall_ms", config.stitch.wall_ms),
            ("max_memory", config.max_memory),
        ] {
            if value == 0 {
                return Err(Rejected::ZeroSetting(name));
            }
        }
        if config.stitch.memory_bytes < MIN_STITCH_MEMORY_BYTES {
            return Err(Rejected::TooLittleStitchMemory(config.stitch.memory_bytes));
        }
        if config.autodoc && !config.library_learning {
            return Err(Rejected::AutodocWithoutLearning);
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
            kept: ByIteration::default(),
            iteration: 0,
            log_counts: BTreeMap::new(),
            batches: Batches::default(),
            prompts: StoredPrompts::default(),
            live: BTreeSet::new(),
            awaiting: BTreeSet::new(),
            task_counts: ByIteration::default(),
            iteration_counts: ByIteration::default(),
            model_counts: ByIteration::default(),
            verified: Vec::new(),
            naming_jobs: Vec::new(),
            named: Vec::new(),
            added: Vec::new(),
            uptake: ByIteration::default(),
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
            Event::PromptStored(stored) => self.store_prompt(stored)?,
            Event::BatchIssued(issued) => self.issue(issued)?,
            Event::Submitted(submitted) => self.submit(submitted)?,
            Event::Checked(checked) => self.check(checked)?,
            Event::FrontierSet { task, members } => {
                self.check_task(task)?;
                self.frontiers.insert(task.clone(), members.clone());
                self.kept
                    .entry(self.iteration)
                    .insert(task.clone(), members.clone());
            }
            Event::LibraryExtended { entries } => {
                if !self.config.library_learning {
                    return Err(Rejected::NoLearning);
                }
                if entries.is_empty() {
                    return Err(Rejected::NoEntries);
                }
                if entries.iter().any(|e| e.name.is_some() || e.doc.is_some()) {
                    return Err(Rejected::Naming(
                        "a library extension's entries have no name or doc; they are \
                         named by naming jobs"
                            .to_owned(),
                    ));
                }
                self.library.extend(entries.iter().cloned());
                self.named.resize(self.library.len(), false);
                self.added.resize(self.library.len(), self.iteration);
            }
            Event::NamingIssued(issued) => self.issue_naming(issued)?,
            Event::NamingDone(done) => self.finish_naming(done)?,
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
        let payload_bytes = self.payload_len(issued)?;
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
        counts.payload_bytes = counts.payload_bytes.saturating_add(payload_bytes);
        Ok(())
    }

    fn store_prompt(&mut self, stored: &PromptStored) -> Result<(), Rejected> {
        let next = self.prompts.json_lens.len() as u64;
        if stored.id != next {
            return Err(Rejected::WrongPromptNumber {
                next,
                given: stored.id,
            });
        }
        let key = digest(&stored.text);
        if let Some(&id) = self.prompts.ids.get(&key) {
            return Err(Rejected::DuplicatePrompt(id));
        }
        self.prompts.ids.insert(key, stored.id);
        self.prompts.json_lens.push(json_len(&stored.text));
        Ok(())
    }

    /// The length in bytes of the JSON text of `issued`'s payload, or why
    /// the event is rejected: it names a prompt the run has not stored.
    fn payload_len(&self, issued: &BatchIssued) -> Result<u64, Rejected> {
        let parts = match &issued.payload {
            IssuedPayload::Whole(text) => return Ok(text.len() as u64),
            IssuedPayload::Parts(parts) => parts,
        };
        // The payload with an empty prompt, and then the prompt's parts as
        // they are written inside its JSON string: JSON escapes each
        // character on its own, so the parts' lengths add up.
        let frame = crate::prompt::Payload {
            batch: issued.batch,
            token: issued.token.clone(),
            programs_wanted: issued.wanted,
            lease_seconds: parts.lease_seconds,
            prompt: String::new(),
        };
        let mut len = serde_json::to_string(&frame).map_or(0, |s| s.len() as u64);
        for part in &parts.prompt {
            let part_len = match part {
                PromptPart::Text(text) => json_len(text),
                PromptPart::Stored(id) => usize::try_from(*id)
                    .ok()
                    .and_then(|i| self.prompts.json_lens.get(i))
                    .copied()
                    .ok_or(Rejected::UnknownPrompt(*id))?,
            };
            len = len.saturating_add(part_len);
        }
        Ok(len)
    }

    /// The id of the stored prompt whose text is `text`, if the run has
    /// stored one.
    pub fn stored_prompt(&self, text: &str) -> Option<u64> {
        self.prompts.ids.get(&digest(text)).copied()
    }

    /// How many prompts the run has stored: the next prompt's id.
    pub fn prompt_count(&self) -> u64 {
        self.prompts.json_lens.len() as u64
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
        let by_model = self
            .model_counts
            .entry(s.iteration)
            .entry(s.model.clone())
            .or_default();
        by_model.submissions = by_model.submissions.saturating_add(1);
        by_model.programs = by_model.programs.saturating_add(s.programs.len() as u64);
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
        let uptake = self.uptake.entry(batch.iteration);
        for check in &c.checks {
            if let Check::Passed { program } = check {
                uptake.passed += 1;
                let refs = learned_refs(program, library);
                if !refs.is_empty() {
                    uptake.using_library += 1;
                }
                for entry in refs {
                    let used = uptake.entries.entry(entry).or_default();
                    used.programs += 1;
                    used.tasks.insert(batch.task.clone());
                }
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

    /// Applies a [`NamingIssued`] event, with every check first.
    fn issue_naming(&mut self, issued: &NamingIssued) -> Result<(), Rejected> {
        let bad = |why: String| Err(Rejected::Naming(why));
        if !self.config.autodoc {
            return bad("the run does not name abstractions (autodoc is off)".to_owned());
        }
        let next = self.naming_jobs.len() as u64;
        if issued.job != next {
            return bad(format!(
                "the event issues naming job {}, but the next job is {next}",
                issued.job
            ));
        }
        if issued.deadline_ms <= issued.issued_ms {
            return bad("a naming job must end after it starts".to_owned());
        }
        match self.named.get(issued.entry as usize) {
            None => return bad(format!("the library has no entry {}", issued.entry)),
            Some(true) => return bad(format!("entry {} is already named", issued.entry)),
            Some(false) => {}
        }
        if let Some((job, _)) = self
            .live_naming(issued.issued_ms)
            .find(|(_, j)| j.entry == issued.entry || j.worker == issued.worker)
        {
            return bad(format!(
                "naming job {job} is still live for that entry or worker"
            ));
        }
        self.naming_jobs.push(NamingJob {
            entry: issued.entry,
            worker: issued.worker.clone(),
            token: issued.token.clone(),
            deadline_ms: issued.deadline_ms,
        });
        Ok(())
    }

    /// Applies a [`NamingDone`] event, with every check first.
    fn finish_naming(&mut self, done: &NamingDone) -> Result<(), Rejected> {
        let bad = |why: String| Err(Rejected::Naming(why));
        let Some(job) = self.naming_jobs.get(done.job as usize) else {
            return bad(format!("the run has no naming job {}", done.job));
        };
        if job.entry != done.entry {
            return bad(format!(
                "naming job {} is for entry {}, not {}",
                done.job, job.entry, done.entry
            ));
        }
        let index = done.entry as usize;
        if self.named.get(index) != Some(&false) {
            return bad(format!("entry {} is already named", done.entry));
        }
        if let Some(name) = &done.name {
            crate::naming::check_name_form(name, index, &self.library).map_err(Rejected::Naming)?;
        }
        if let Some(doc) = &done.doc {
            crate::naming::check_doc(doc).map_err(Rejected::Naming)?;
        }
        let entry = &mut self.library[index];
        entry.name.clone_from(&done.name);
        entry.doc.clone_from(&done.doc);
        self.named[index] = true;
        Ok(())
    }

    /// The naming jobs whose lease has not ended at `now_ms` and whose
    /// entry has no naming result, with their numbers.
    pub fn live_naming(&self, now_ms: u64) -> impl Iterator<Item = (u64, &NamingJob)> + '_ {
        self.naming_jobs
            .iter()
            .enumerate()
            .filter(move |(_, j)| now_ms <= j.deadline_ms && !self.named[j.entry as usize])
            .map(|(i, j)| (i as u64, j))
    }

    /// Naming job `job`, if the run has issued it.
    pub fn naming_job(&self, job: u64) -> Option<&NamingJob> {
        self.naming_jobs.get(usize::try_from(job).ok()?)
    }

    /// How many naming jobs the run has issued.
    pub fn naming_job_count(&self) -> u64 {
        self.naming_jobs.len() as u64
    }

    /// Whether library entry `entry` has a naming result.
    pub fn is_named(&self, entry: u64) -> bool {
        usize::try_from(entry)
            .ok()
            .and_then(|i| self.named.get(i))
            .copied()
            .unwrap_or(false)
    }

    /// Whether batches wait for naming: autodoc is on and some library
    /// entry has no naming result (decision D23, item 7).
    pub fn naming_pending(&self) -> bool {
        self.names_pending() > 0
    }

    /// How many library entries have no naming result, if autodoc is on;
    /// 0 if it is off.
    pub fn names_pending(&self) -> u64 {
        if !self.config.autodoc {
            return 0;
        }
        self.named.iter().filter(|n| !**n).count() as u64
    }

    /// The first library entry with no naming result and no live job at
    /// `now_ms`, if autodoc is on.
    pub fn next_unnamed(&self, now_ms: u64) -> Option<u64> {
        if !self.config.autodoc {
            return None;
        }
        let held: BTreeSet<u64> = self.live_naming(now_ms).map(|(_, j)| j.entry).collect();
        (0..self.named.len() as u64).find(|&e| !self.named[e as usize] && !held.contains(&e))
    }

    /// The library. Entry `i` is library definition `i`.
    pub fn library(&self) -> &[LibraryEntry] {
        &self.library
    }

    /// The iteration whose end added library entry `entry`, or `None` if
    /// the library has no such entry. Programs can call it from the next
    /// iteration on.
    pub fn added_in(&self, entry: u64) -> Option<u64> {
        usize::try_from(entry)
            .ok()
            .and_then(|i| self.added.get(i))
            .copied()
    }

    /// Which learned library entries the programs that passed in
    /// `iteration` refer to.
    pub fn library_uptake(&self, iteration: u64) -> LibraryUptake {
        self.uptake.get(iteration).cloned().unwrap_or_default()
    }

    /// The frontier of `task`, or `None` if the run has no such task.
    pub fn frontier(&self, task: &str) -> Option<&[FrontierMember]> {
        self.frontiers.get(task).map(Vec::as_slice)
    }

    /// The frontiers set while `iteration` was the current iteration: the
    /// programs the step that ended it kept, by task, with their costs.
    /// `None` if no frontier was set then, as before the iteration ends.
    pub fn kept_frontiers(&self, iteration: u64) -> Option<&BTreeMap<String, Vec<FrontierMember>>> {
        self.kept.get(iteration)
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

    /// What came in during `iteration` for each model the proposers named,
    /// with `None` for those that named none, in the models' order.
    pub fn model_counts(&self, iteration: u64) -> Vec<(Option<String>, ModelCounts)> {
        self.model_counts
            .get(iteration)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
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
            max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
            autodoc: false,
            arm: Arm::Blind,
            library_learning: true,
            seed: 7,
            proposals_per_task: 4,
            programs_per_batch: 2,
            lease_ms: 1000,
            max_unchecked_per_batch: 3,
            frontier_size: 5,
            stitch: crate::state::StitchSettings::default(),
            prelude: 0,
            compared_with: None,
        }
    }

    /// A run with library learning off (the LLM-only baseline, D24) cannot
    /// name abstractions, and its library takes no entries.
    #[test]
    fn a_run_without_library_learning_has_no_naming_and_no_library() {
        let baseline = RunConfig {
            library_learning: false,
            ..config()
        };
        let named = RunConfig {
            autodoc: true,
            ..baseline.clone()
        };
        assert_eq!(
            RunState::create(named, vec!["a".into()]),
            Err(Rejected::AutodocWithoutLearning)
        );
        let mut state = RunState::create(baseline, vec!["a".into()]).unwrap();
        let entry = LibraryEntry {
            name: None,
            doc: None,
            arity: 1,
            term: "(lam $0)".into(),
        };
        assert_eq!(
            state.apply(&Event::LibraryExtended {
                entries: vec![entry.clone()]
            }),
            Err(Rejected::NoLearning)
        );
        assert!(state.library().is_empty());
        // The same entry joins a run that learns a library.
        let mut learning = new_state();
        learning
            .apply(&Event::LibraryExtended {
                entries: vec![entry],
            })
            .unwrap();
        assert_eq!(learning.library().len(), 1);
    }

    /// A run written before the setting learns a library.
    #[test]
    fn settings_written_before_library_learning_read_back_with_it_on() {
        let mut json = serde_json::to_value(RunConfig {
            library_learning: false,
            ..config()
        })
        .unwrap();
        json.as_object_mut().unwrap().remove("library_learning");
        let read: RunConfig = serde_json::from_value(json).unwrap();
        assert!(read.library_learning);
    }

    /// A run written before the setting has the default memory limit.
    #[test]
    fn settings_written_before_max_memory_read_back_with_the_default() {
        let mut json = serde_json::to_value(RunConfig {
            max_memory: 1 << 20,
            ..config()
        })
        .unwrap();
        assert_eq!(json["max_memory"], 1 << 20);
        json.as_object_mut().unwrap().remove("max_memory");
        let read: RunConfig = serde_json::from_value(json).unwrap();
        assert_eq!(read.max_memory, dc_tasks::harness::DEFAULT_MAX_MEMORY);
    }

    /// A run written before the setting is compared with no run, and a run
    /// compared with none writes no field, so its journal reads as before.
    #[test]
    fn settings_written_before_compared_with_read_back_with_none() {
        let paired = RunConfig {
            compared_with: Some("main".into()),
            ..config()
        };
        let json = serde_json::to_value(&paired).unwrap();
        assert_eq!(json["compared_with"], "main");
        let read: RunConfig = serde_json::from_value(json).unwrap();
        assert_eq!(read, paired);
        let json = serde_json::to_value(config()).unwrap();
        assert!(json.get("compared_with").is_none(), "{json}");
        let read: RunConfig = serde_json::from_value(json).unwrap();
        assert_eq!(read.compared_with, None);
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
            autodoc: false,
            proposals_per_task: 0,
            ..config()
        };
        assert_eq!(
            RunState::create(zero, vec!["a".into()]),
            Err(Rejected::ZeroSetting("proposals_per_task"))
        );
        let zero = RunConfig {
            autodoc: false,
            max_unchecked_per_batch: 0,
            frontier_size: 5,
            stitch: crate::state::StitchSettings::default(),
            prelude: 0,
            compared_with: None,
            ..config()
        };
        assert_eq!(
            RunState::create(zero, vec!["a".into()]),
            Err(Rejected::ZeroSetting("max_unchecked_per_batch"))
        );
        for (name, zero) in [
            (
                "frontier_size",
                RunConfig {
                    autodoc: false,
                    frontier_size: 0,
                    ..config()
                },
            ),
            (
                "stitch.iterations",
                RunConfig {
                    autodoc: false,
                    stitch: StitchSettings {
                        iterations: 0,
                        ..StitchSettings::default()
                    },
                    ..config()
                },
            ),
            (
                "stitch.wall_ms",
                RunConfig {
                    autodoc: false,
                    stitch: StitchSettings {
                        wall_ms: 0,
                        ..StitchSettings::default()
                    },
                    ..config()
                },
            ),
            (
                "max_memory",
                RunConfig {
                    max_memory: 0,
                    ..config()
                },
            ),
        ] {
            assert_eq!(
                RunState::create(zero, vec!["a".into()]),
                Err(Rejected::ZeroSetting(name))
            );
        }
        for (bytes, ok) in [
            (MIN_STITCH_MEMORY_BYTES, true),
            (MIN_STITCH_MEMORY_BYTES - 1, false),
        ] {
            let c = RunConfig {
                autodoc: false,
                stitch: StitchSettings {
                    memory_bytes: bytes,
                    ..StitchSettings::default()
                },
                ..config()
            };
            let got = RunState::create(c, vec!["a".into()]);
            assert_eq!(got.is_ok(), ok, "{bytes}: {got:?}");
        }
        for (n, ok) in [
            (MAX_PROGRAMS_PER_BATCH, true),
            (MAX_PROGRAMS_PER_BATCH + 1, false),
        ] {
            let c = RunConfig {
                autodoc: false,
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
            cost: None,
        };
        s.apply(&Event::FrontierSet {
            task: "a".into(),
            members: vec![member.clone()],
        })
        .unwrap();
        assert_eq!(s.frontier("a"), Some(&[member][..]));
        assert_eq!(s.frontier("b"), Some(&[][..]));

        let entry = LibraryEntry {
            name: None,
            doc: None,
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

    /// Each library entry keeps the iteration whose end added it, and each
    /// iteration counts the learned entries its passed programs refer to.
    #[test]
    fn the_library_records_when_each_entry_came_and_which_programs_use_it() {
        let mut s = new_state();
        let entry = |term: &str| LibraryEntry {
            name: None,
            doc: None,
            arity: 1,
            term: term.into(),
        };
        s.apply(&Event::LibraryExtended {
            entries: vec![entry("(lam $0)"), entry("(lam (lib_0 $0))")],
        })
        .unwrap();
        s.apply(&Event::IterationFinished {
            iteration: 0,
            used: 0,
        })
        .unwrap();
        s.apply(&Event::LibraryExtended {
            entries: vec![entry("(lam (lib_1 $0))")],
        })
        .unwrap();
        assert_eq!(s.added_in(0), Some(0));
        assert_eq!(s.added_in(1), Some(0));
        assert_eq!(s.added_in(2), Some(1));
        assert_eq!(s.added_in(3), None);

        for (batch, task) in [(0, "a"), (1, "b")] {
            s.apply(&Event::BatchIssued(BatchIssued {
                batch,
                iteration: 1,
                task: task.into(),
                wanted: 2,
                worker: "w".into(),
                token: format!("t{batch}"),
                issued_ms: 0,
                deadline_ms: 1000,
                payload: IssuedPayload::Whole("pay".into()),
            }))
            .unwrap();
            s.apply(&sub(batch, "w", 1, 0, Standing::OnTime, 2))
                .unwrap();
        }
        let prelude = dc_surface::prelude::PRELUDE_LIB_BASE;
        // Entry 0 twice in one program counts once; a prelude function is
        // not a learned entry; a failed program uses nothing.
        s.apply(&Event::Checked(Checked {
            batch: 0,
            library_len: 3,
            checks: vec![
                passed(&format!(
                    "(lam (lib_0 (lib_0 (lib_1 $0)) (lib_{prelude} $0)))"
                )),
                passed(&format!("(lam (lib_{prelude} $0))")),
            ],
        }))
        .unwrap();
        s.apply(&Event::Checked(Checked {
            batch: 1,
            library_len: 3,
            checks: vec![
                passed("(lam (lib_1 $0))"),
                Check::Run {
                    error: "lib_2".into(),
                },
            ],
        }))
        .unwrap();

        assert_eq!(s.library_uptake(0), LibraryUptake::default());
        let uptake = s.library_uptake(1);
        assert_eq!(uptake.passed, 3);
        assert_eq!(uptake.using_library, 2);
        let used: Vec<(u64, u64, Vec<&str>)> = uptake
            .entries
            .iter()
            .map(|(e, u)| (*e, u.programs, u.tasks.iter().map(String::as_str).collect()))
            .collect();
        assert_eq!(used, vec![(0, 1, vec!["a"]), (1, 2, vec!["a", "b"])]);
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
            payload: IssuedPayload::Whole("pay".into()),
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

    fn stored(id: u64, text: &str) -> Event {
        Event::PromptStored(PromptStored {
            id,
            text: text.into(),
        })
    }

    /// `issued`, with its prompt in parts.
    fn issued_in_parts(batch: u64, prompt: Vec<PromptPart>) -> Event {
        let mut event = issued(batch, "a", 1, 0);
        if let Event::BatchIssued(b) = &mut event {
            b.payload = IssuedPayload::Parts(PayloadParts {
                lease_seconds: 1,
                prompt,
            });
        }
        event
    }

    #[test]
    fn a_prompt_is_stored_once_with_the_next_number() {
        let mut s = new_state();
        s.apply(&stored(0, "guide")).unwrap();
        let before = s.clone();
        assert_eq!(
            s.apply(&stored(2, "task")),
            Err(Rejected::WrongPromptNumber { next: 1, given: 2 })
        );
        assert_eq!(
            s.apply(&stored(1, "guide")),
            Err(Rejected::DuplicatePrompt(0))
        );
        assert_eq!(s, before);
        assert_eq!(s.stored_prompt("guide"), Some(0));
        assert_eq!(s.stored_prompt("task"), None);
        s.apply(&stored(1, "task")).unwrap();
        assert_eq!((s.stored_prompt("task"), s.prompt_count()), (Some(1), 2));
    }

    #[test]
    fn a_batch_that_names_a_prompt_not_stored_is_rejected_and_changes_nothing() {
        let mut s = new_state();
        s.apply(&stored(0, "guide")).unwrap();
        let before = s.clone();
        // The unknown prompt comes after a known one and some text, so a
        // check made part by part as the state changed would leave a trace.
        let bad = issued_in_parts(
            0,
            vec![
                PromptPart::Text("x".into()),
                PromptPart::Stored(0),
                PromptPart::Stored(1),
            ],
        );
        assert_eq!(s.apply(&bad), Err(Rejected::UnknownPrompt(1)));
        assert_eq!(s, before);
        assert_eq!(s.batch_count(), 0);
        assert_eq!(s.iteration_counts(0), IterationCounts::default());
        // A prompt stored in the same commit, before the batch, is known.
        s.apply(&stored(1, "task \"one\"\n")).unwrap();
        s.apply(&bad).unwrap();
        assert_eq!(s.batch_count(), 1);
    }

    /// A payload in parts counts the bytes of its JSON text as it was sent,
    /// with the parts escaped as they are inside the JSON string.
    #[test]
    fn a_payload_in_parts_counts_the_bytes_of_its_json_text() {
        let mut s = new_state();
        s.apply(&stored(0, "a \"quoted\"\nline\t\u{1}é")).unwrap();
        let event = issued_in_parts(
            0,
            vec![
                PromptPart::Text("\\head\n".into()),
                PromptPart::Stored(0),
                PromptPart::Text("tail".into()),
            ],
        );
        s.apply(&event).unwrap();
        let Event::BatchIssued(b) = &event else {
            unreachable!()
        };
        let sent = crate::prompt::Payload {
            batch: 0,
            token: b.token.clone(),
            programs_wanted: 1,
            lease_seconds: 1,
            prompt: "\\head\na \"quoted\"\nline\t\u{1}étail".into(),
        };
        let text = serde_json::to_string(&sent).unwrap();
        assert_eq!(s.iteration_counts(0).payload_bytes, text.len() as u64);
    }

    /// An event in the old form holds the payload as a JSON string, and
    /// one in the new form as an object; each reads back as written.
    #[test]
    fn each_form_of_payload_reads_back_as_it_was_written() {
        let whole = issued(0, "a", 1, 0);
        let json = serde_json::to_string(&whole).unwrap();
        assert!(json.contains("\"payload\":\"pay\""), "{json}");
        assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), whole);
        let parts = issued_in_parts(0, vec![PromptPart::Stored(3), PromptPart::Text("t".into())]);
        let json = serde_json::to_string(&parts).unwrap();
        assert!(
            json.contains(
                "\"payload\":{\"lease_seconds\":1,\"prompt\":[{\"stored\":3},{\"text\":\"t\"}]}"
            ),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Event>(&json).unwrap(), parts);
    }

    #[test]
    fn the_quota_is_in_when_every_task_has_its_programs_checked() {
        let mut s = RunState::create(
            RunConfig {
                autodoc: false,
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
                payload: IssuedPayload::Whole(String::new()),
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
