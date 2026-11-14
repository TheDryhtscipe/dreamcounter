//! The MCP tools.
//!
//! Every tool reads or writes through the [`Store`], so a tool's answer is the
//! state on disk. A request the server cannot carry out is a tool error with a
//! message, and changes nothing.

use std::sync::Arc;

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{schemars, tool, tool_router};
use serde::{Deserialize, Serialize};

use crate::state::{Arm, LogKind, RunConfig, RunId, RunState};
use crate::store::Store;

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
}

/// The arguments of `dc_run_status`.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunArg {
    /// The run's id.
    pub run: String,
}

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
    /// Prompt payloads handed to proposers.
    pub payload: u64,
    /// Submissions received.
    pub submission: u64,
    /// Evaluation results.
    pub evaluation: u64,
    /// Compression results.
    pub compression: u64,
    /// Remarks.
    pub note: u64,
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
                payload: state.log_count(LogKind::Payload),
                submission: state.log_count(LogKind::Submission),
                evaluation: state.log_count(LogKind::Evaluation),
                compression: state.log_count(LogKind::Compression),
                note: state.log_count(LogKind::Note),
            },
        }
    }
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

fn run_id(text: &str) -> Result<RunId, String> {
    RunId::new(text).map_err(|e| format!("{text:?} is not a run id: {e}"))
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
    fn create_run(
        &self,
        Parameters(arg): Parameters<CreateRun>,
    ) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        let known = known_tasks();
        let tasks = match arg.tasks {
            None => known,
            Some(tasks) => {
                if let Some(unknown) = tasks.iter().find(|t| !known.contains(t)) {
                    return Err(format!(
                        "there is no task {unknown:?}. The tasks are: {}",
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
        };
        let state = self
            .store
            .create_run(&run, config, tasks)
            .map_err(|e| e.to_string())?;
        Ok(Json(RunStatus::of(&run, &state)))
    }

    /// Reports where a run is.
    #[tool(
        name = "dc_run_status",
        description = "Report where a DreamCounter run is: its settings, its iteration, the size \
                       of each task's frontier, the size of the library, and how many log \
                       records it has."
    )]
    fn run_status(&self, Parameters(arg): Parameters<RunArg>) -> Result<Json<RunStatus>, String> {
        let run = run_id(&arg.run)?;
        let state = self.store.state(&run).map_err(|e| e.to_string())?;
        Ok(Json(RunStatus::of(&run, &state)))
    }

    /// Lists the runs.
    #[tool(
        name = "dc_list_runs",
        description = "List the ids of the DreamCounter runs in the server's state directory."
    )]
    fn list_runs(&self) -> Result<Json<RunList>, String> {
        let runs = self.store.runs().map_err(|e| e.to_string())?;
        Ok(Json(RunList {
            runs: runs.iter().map(|r| r.as_str().to_owned()).collect(),
        }))
    }
}
