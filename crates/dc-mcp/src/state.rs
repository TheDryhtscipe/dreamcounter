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
    /// How many proposals each task gets in one iteration. It is positive.
    pub proposals_per_task: u32,
}

/// One definition in the library.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    /// The body, a core term as an s-expression (`dc_core::sexpr`).
    pub body: String,
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
    /// A prompt payload handed to a proposer.
    Payload,
    /// What a proposer submitted.
    Submission,
    /// The result of evaluating a submission.
    Evaluation,
    /// The result of a compression run.
    Compression,
    /// A remark by the operator or the server.
    Note,
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
    },
}

/// Why an event cannot be applied to a state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejected {
    /// A run needs at least one task.
    NoTasks,
    /// The task is named twice in the task set.
    DuplicateTask(String),
    /// `proposals_per_task` is zero.
    NoProposals,
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
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rejected::NoTasks => write!(f, "a run needs at least one task"),
            Rejected::DuplicateTask(t) => write!(f, "task {t:?} is named twice"),
            Rejected::NoProposals => write!(f, "proposals_per_task must be at least 1"),
            Rejected::NotCreated => write!(f, "the first event of a run must create it"),
            Rejected::CreatedTwice => write!(f, "the run is already created"),
            Rejected::UnknownTask(t) => write!(f, "the run has no task {t:?}"),
            Rejected::WrongIteration { current, given } => write!(
                f,
                "the event names iteration {given}, but the run is in iteration {current}"
            ),
            Rejected::NoEntries => write!(f, "a library extension needs at least one entry"),
            Rejected::IterationOverflow => write!(f, "the iteration number would overflow"),
        }
    }
}

impl std::error::Error for Rejected {}

/// Everything the server knows about a run, except the text of its log.
///
/// The log's records are in the journal. The state keeps a count of each kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    config: RunConfig,
    tasks: Vec<String>,
    library: Vec<LibraryEntry>,
    frontiers: BTreeMap<String, Vec<FrontierMember>>,
    iteration: u64,
    log_counts: BTreeMap<LogKind, u64>,
}

impl RunState {
    /// The state of a new run: iteration 0, an empty library, and an empty
    /// frontier for each task.
    pub fn create(config: RunConfig, tasks: Vec<String>) -> Result<RunState, Rejected> {
        if tasks.is_empty() {
            return Err(Rejected::NoTasks);
        }
        if config.proposals_per_task == 0 {
            return Err(Rejected::NoProposals);
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
            Event::IterationFinished { iteration } => {
                self.check_iteration(*iteration)?;
                self.iteration = self
                    .iteration
                    .checked_add(1)
                    .ok_or(Rejected::IterationOverflow)?;
            }
        }
        Ok(())
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RunConfig {
        RunConfig {
            arm: Arm::Blind,
            seed: 7,
            proposals_per_task: 4,
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
            Err(Rejected::NoProposals)
        );
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
            body: "(lam 0)".into(),
        };
        s.apply(&Event::LibraryExtended {
            entries: vec![entry.clone()],
        })
        .unwrap();
        assert_eq!(s.library(), &[entry]);

        s.apply(&Event::Logged(LogRecord {
            kind: LogKind::Payload,
            iteration: 0,
            task: Some("b".into()),
            text: "p".into(),
        }))
        .unwrap();
        assert_eq!(s.log_count(LogKind::Payload), 1);
        assert_eq!(s.log_count(LogKind::Submission), 0);

        s.apply(&Event::IterationFinished { iteration: 0 }).unwrap();
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
                Event::IterationFinished { iteration: 1 },
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
                iteration: u64::MAX
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
        let finished = Event::IterationFinished { iteration: 0 };
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
}
