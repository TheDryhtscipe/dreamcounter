//! A run's records: the saved results of dc-mcp's operator tools, and the
//! summaries dc-propose writes.
//!
//! The operator saves each tool's structured result as JSON in the run's
//! records directory:
//!
//! | file                  | tool                                   | needed               |
//! |-----------------------|----------------------------------------|----------------------|
//! | `status.json`         | `dc_run_status`                        | always               |
//! | `proposal_stats.json` | `dc_proposal_stats`                    | always               |
//! | `library.json`        | `dc_library`, at the run's end         | if present           |
//! | `diversity.json`      | `dc_diversity` with `growth` true      | for the cheap rate   |
//!
//! dc-propose writes a `summary.json` in each launch's log directory. Every
//! `summary.json` under a run's log directories whose `run` is the run's id
//! counts towards its tokens and cost.
//!
//! dc-mcp's report types are only `Serialize`, so the types here mirror the
//! fields this crate reads. Fields it does not read are ignored, so the
//! records may carry more. A test serializes dc-mcp's own types and reads
//! them back with these, so a renamed field fails it.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use dc_mcp::state::{CostClass, CostVerdict};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{Map, Value};

/// Why a run's records cannot be read.
#[derive(Debug)]
pub struct RecordError(pub String);

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RecordError {}

/// The fields of `dc_run_status` this crate reads.
#[derive(Clone, Debug, Deserialize)]
pub struct RunStatus {
    /// The run's id.
    pub run: String,
    /// The run's seed.
    pub seed: u64,
    /// The most programs one batch asks for.
    pub programs_per_batch: u32,
    /// The current iteration.
    pub iteration: u64,
    /// The run's tasks, in order.
    pub tasks: Vec<TaskStatus>,
    /// Whether the run learns a library.
    pub library_learning: bool,
}

/// A task of `dc_run_status`.
#[derive(Clone, Debug, Deserialize)]
pub struct TaskStatus {
    /// The task's name.
    pub name: String,
}

/// The result of `dc_proposal_stats`.
#[derive(Clone, Debug, Deserialize)]
pub struct ProposalStats {
    /// The run's id.
    pub run: String,
    /// Each iteration from 0 to the current one.
    pub iterations: Vec<IterationStats>,
}

/// One iteration of `dc_proposal_stats`.
#[derive(Clone, Debug, Deserialize)]
pub struct IterationStats {
    /// The iteration.
    pub iteration: u64,
    /// Whether every task has its programs, all checked.
    pub quota_in: bool,
    /// Submissions that wait to be checked.
    pub pending_checks: u64,
    /// Submissions for a batch of an earlier iteration.
    pub late: u64,
    /// Submissions for a batch that already had one.
    pub repeats: u64,
    /// Submissions from a worker that does not hold the batch: kept, not
    /// checked, so left out of every rate (D33's exclusions).
    pub foreign: u64,
    /// Foreign submissions refused and only counted.
    pub refused_foreign: u64,
    /// Which learned entries the passed programs refer to.
    pub library_uptake: UptakeStats,
    /// Each task, in the run's order.
    pub tasks: Vec<TaskStats>,
}

/// Library uptake in one iteration.
#[derive(Clone, Debug, Deserialize)]
pub struct UptakeStats {
    /// Programs that passed.
    pub passed: u64,
    /// Of those, the programs that refer to a learned entry.
    pub using_library: u64,
    /// Each entry referred to.
    pub entries: Vec<EntryUse>,
}

/// The passed programs of one iteration that refer to one entry.
#[derive(Clone, Debug, Deserialize)]
pub struct EntryUse {
    /// The entry's index.
    pub entry: u64,
    /// Its name, if it has one.
    pub name: Option<String>,
    /// How many programs refer to it.
    pub programs: u64,
    /// The tasks of those programs.
    pub tasks: Vec<String>,
}

/// One task in one iteration of `dc_proposal_stats`.
#[derive(Clone, Debug, Deserialize)]
pub struct TaskStats {
    /// The task's name.
    pub name: String,
    /// Batches issued for it.
    pub batches: u64,
    /// Programs checked.
    pub programs: u64,
    /// Programs that compiled.
    pub compiled: u64,
    /// Programs that compiled and typed.
    pub typed: u64,
    /// Programs that passed every example.
    pub passed: u64,
    /// Whether a program passed.
    pub solved: bool,
}

/// The result of `dc_library`.
#[derive(Clone, Debug, Deserialize)]
pub struct LibraryReport {
    /// The iteration whose library it is.
    pub iteration: u64,
    /// Each entry.
    pub entries: Vec<LibraryItem>,
}

/// One entry of `dc_library`.
#[derive(Clone, Debug, Deserialize)]
pub struct LibraryItem {
    /// The entry's index.
    pub index: u64,
    /// Its number of parameters.
    pub arity: u32,
    /// Its name, if a proposer gave one.
    pub name: Option<String>,
    /// Its one-line doc, if a proposer gave one.
    pub doc: Option<String>,
    /// The iteration whose end added it.
    pub iteration_added: u64,
}

/// The result of `dc_diversity`.
#[derive(Clone, Debug, Deserialize)]
pub struct DiversityReport {
    /// The run's id.
    pub run: String,
    /// Each iteration.
    pub iterations: Vec<IterationDiversity>,
}

/// One iteration of `dc_diversity`.
#[derive(Clone, Debug, Deserialize)]
pub struct IterationDiversity {
    /// The iteration.
    pub iteration: u64,
    /// Each task.
    pub tasks: Vec<TaskDiversity>,
}

/// One task in one iteration of `dc_diversity`.
#[derive(Clone, Debug, Deserialize)]
pub struct TaskDiversity {
    /// The task.
    pub task: String,
    /// Programs that passed.
    pub passed: u64,
    /// The distinct expanded programs.
    pub groups: Vec<ProgramGroup>,
}

/// One distinct expanded program of `dc_diversity`.
#[derive(Clone, Debug, Deserialize)]
pub struct ProgramGroup {
    /// How many passed programs expand to it.
    pub count: u64,
    /// Its cost on the growth sweep, if growth was asked for.
    pub growth: Option<GrowthResult>,
}

/// What a distinct program costs, as `dc_diversity` gives it.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrowthResult {
    /// Estimator C's verdict.
    Verdict(CostVerdict),
    /// A wrong output or an error on a generated input.
    Incorrect,
    /// It could not be run on the sweep.
    Refused {
        /// Why.
        reason: String,
    },
    /// The time for measuring ran out.
    NotMeasured {
        /// Why.
        reason: String,
    },
}

impl GrowthResult {
    /// The class estimator C named, if it named one.
    pub fn class(&self) -> Option<CostClass> {
        match self {
            GrowthResult::Verdict(CostVerdict::Class(c)) => Some(*c),
            _ => None,
        }
    }
}

/// A dc-propose launch summary.
#[derive(Clone, Debug, Deserialize)]
pub struct LaunchSummary {
    /// The launch's id.
    pub launch: String,
    /// The run it worked on.
    pub run: String,
    /// Whether the launch was interrupted.
    #[serde(default)]
    pub interrupted: bool,
    /// Its sessions.
    pub sessions: Vec<SessionSummary>,
}

/// One session of a dc-propose launch. `usage` and `total_cost_usd` come
/// from the session's final `result` event, as Claude Code gave them; a
/// session that ended without one has neither.
#[derive(Clone, Debug, Deserialize)]
pub struct SessionSummary {
    /// `batch` or `naming`.
    pub kind: String,
    /// How it ended.
    pub outcome: String,
    /// Its token use.
    #[serde(default)]
    pub usage: Option<Usage>,
    /// Its cost at API prices, as Claude Code reports it.
    #[serde(default)]
    pub total_cost_usd: Option<f64>,
}

/// A session's token use.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
pub struct Usage {
    /// Input tokens not read from or written to the cache.
    #[serde(default)]
    pub input_tokens: u64,
    /// Input tokens written to the cache.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Input tokens read from the cache.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    /// Output tokens.
    #[serde(default)]
    pub output_tokens: u64,
}

/// Everything recorded about one run.
#[derive(Clone, Debug)]
pub struct RunRecords {
    /// Where the records were read from.
    pub dir: PathBuf,
    /// `dc_run_status`.
    pub status: RunStatus,
    /// `dc_run_status` as JSON, for comparing settings.
    pub status_json: Map<String, Value>,
    /// `dc_proposal_stats`.
    pub stats: ProposalStats,
    /// `dc_library`, if saved.
    pub library: Option<LibraryReport>,
    /// `dc_diversity` with growth, if saved.
    pub diversity: Option<DiversityReport>,
    /// The launches that worked on this run.
    pub launches: Vec<LaunchSummary>,
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, RecordError> {
    let text = fs::read_to_string(path)
        .map_err(|e| RecordError(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| RecordError(format!("cannot parse {}: {e}", path.display())))
}

fn read_optional<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, RecordError> {
    if path.exists() {
        read_json(path).map(Some)
    } else {
        Ok(None)
    }
}

/// Every `summary.json` under `dir`, in path order.
fn summaries(dir: &Path, found: &mut Vec<PathBuf>) -> Result<(), RecordError> {
    let entries = fs::read_dir(dir)
        .map_err(|e| RecordError(format!("cannot read {}: {e}", dir.display())))?;
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()
        .map_err(|e| RecordError(format!("cannot read {}: {e}", dir.display())))?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            summaries(&path, found)?;
        } else if path.file_name().is_some_and(|n| n == "summary.json") {
            found.push(path);
        }
    }
    Ok(())
}

impl RunRecords {
    /// Reads the records in `dir`, and the launch summaries for the run
    /// under each of `logs`.
    ///
    /// # Errors
    ///
    /// If a needed file is missing or does not parse, if the files are of
    /// different runs, or if a log directory cannot be read.
    pub fn read(dir: &Path, logs: &[PathBuf]) -> Result<RunRecords, RecordError> {
        let status_json: Map<String, Value> = read_json(&dir.join("status.json"))?;
        let status: RunStatus = serde_json::from_value(Value::Object(status_json.clone()))
            .map_err(|e| RecordError(format!("cannot parse {}/status.json: {e}", dir.display())))?;
        let stats: ProposalStats = read_json(&dir.join("proposal_stats.json"))?;
        let library: Option<LibraryReport> = read_optional(&dir.join("library.json"))?;
        let diversity: Option<DiversityReport> = read_optional(&dir.join("diversity.json"))?;
        if stats.run != status.run {
            return Err(RecordError(format!(
                "{}: proposal_stats.json is of run {}, status.json of run {}",
                dir.display(),
                stats.run,
                status.run
            )));
        }
        if let Some(d) = &diversity {
            if d.run != status.run {
                return Err(RecordError(format!(
                    "{}: diversity.json is of run {}, status.json of run {}",
                    dir.display(),
                    d.run,
                    status.run
                )));
            }
        }
        let mut paths = Vec::new();
        for log in logs {
            summaries(log, &mut paths)?;
        }
        // A directory listed twice, or inside another listed one, or a copy
        // of a summary, must not count a launch twice: each launch once.
        let mut seen = std::collections::BTreeSet::new();
        let mut launches = Vec::new();
        for path in paths {
            let summary: LaunchSummary = read_json(&path)?;
            if summary.run == status.run && seen.insert(summary.launch.clone()) {
                launches.push(summary);
            }
        }
        Ok(RunRecords {
            dir: dir.to_owned(),
            status,
            status_json,
            stats,
            library,
            diversity,
            launches,
        })
    }

    /// The iteration's stats, if the run has reached it.
    pub fn iteration(&self, i: u64) -> Option<&IterationStats> {
        self.stats.iterations.iter().find(|s| s.iteration == i)
    }

    /// The iteration's diversity, if recorded.
    pub fn diversity_of(&self, i: u64) -> Option<&IterationDiversity> {
        self.diversity
            .as_ref()?
            .iterations
            .iter()
            .find(|d| d.iteration == i)
    }

    /// The run's settings: `dc_run_status` without the fields that say
    /// where the run is rather than how it was made.
    pub fn settings(&self) -> BTreeMap<String, Value> {
        const PROGRESS: [&str; 10] = [
            "run",
            "seed",
            "iteration",
            "tasks",
            "library_size",
            "unnamed",
            "log_counts",
            "batches_issued",
            "current",
            "compared_with",
        ];
        self.status_json
            .iter()
            .filter(|(k, _)| !PROGRESS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
#[path = "records_tests.rs"]
mod tests;
