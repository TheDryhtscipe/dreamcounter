//! The MCP tools.
//!
//! Every tool reads or writes through the [`Store`], which brings its copy of
//! the run up to the journal on disk first. A request the server cannot carry
//! out is a tool error with a message. It changes nothing in a run's state,
//! unless the message says the outcome is not known.
//!
//! A tool call runs on a thread of its own, so a call that waits for a run's
//! lock does not hold up other calls or the protocol.

use std::sync::Arc;

use rmcp::handler::server::wrapper::{Json, Parameters};
use rmcp::{schemars, tool, tool_router};
use serde::{Deserialize, Serialize};

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
}

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
    /// Reports that the store set aside bytes of the journal that were not a
    /// record. More than 0 can mean lost changes: look at the `torn-*` files
    /// in the run's directory.
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
                recovery: state.log_count(LogKind::Recovery),
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
