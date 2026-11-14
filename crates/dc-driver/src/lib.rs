//! One iteration of the DreamCounter loop, run when an iteration's proposals
//! are in (decisions D4 and D11).
//!
//! [`run_iteration`] takes the run's tasks, its library, the frontier of each
//! task, and the verified programs proposed in the iteration. It returns the
//! new frontiers, the abstractions that join the library, and an
//! [`IterationReport`] that says what happened to each program. It does this
//! in four steps:
//!
//! 1. **Check.** Each library entry must parse, expand and type. Each frontier
//!    member must parse, be small enough to print again, be closed, expand
//!    under the library, and type against its task's signature. A failure is a
//!    [`DriverError`]: the run's state is wrong. A candidate that fails the same
//!    checks is reported as [`Fate::Invalid`] and left out.
//! 2. **Frontiers.** Each task's pool is its frontier, then its candidates in
//!    the order given. A program whose expansion equals that of an earlier one
//!    in the pool is a [`Fate::Duplicate`]. The pool is sorted by
//!    [`stitch_cost`], stably, and the first [`DriverConfig::frontier_size`]
//!    programs are kept. This is selection at λ = 0, pure description length
//!    (D4): Stitch is given every kept program with its task's label, and it
//!    counts only the cheapest rewrite of each task.
//! 3. **Compression.** The kept programs of every task, in task order, go to a
//!    [`Stitch`] runner. A program the bridge refuses before Stitch runs (it
//!    cannot be encoded, or is too deep) is left out and stays in its frontier
//!    as it is. If the run is cancelled, or its worker cannot be started, the
//!    step returns [`DriverError::Interrupted`]. Any other error, such as a
//!    worker stopped at its time limit, skips compression for the iteration:
//!    the frontiers are still updated, and the library is unchanged.
//! 4. **Library update.** Each new abstraction gets a type scheme inferred from
//!    its body with the earlier ones, and the new abstractions are cut at the
//!    first that does not type. A rewritten program is accepted if it types
//!    against its task's signature with the library's schemes and expands to
//!    what its original expands to. Otherwise the program stays as it was:
//!    the rewriting keeps the expansion but not always the type (see the
//!    dc-core test `a_parameter_used_at_two_types_can_make_the_refactored_program_untypable`).
//!    The bridge returns only the programs rewritten with every abstraction it
//!    accepted. So a program that uses a cut abstraction stays as it was, even
//!    when a rewriting with the earlier abstractions alone would type. New
//!    abstractions after the last one an accepted rewriting uses are cut too.
//!    Each frontier is sorted again by cost under the new library.
//!
//! # Determinism
//!
//! Programs are handled in a fixed order, ties keep the earlier program, and
//! the bridge runs Stitch on one thread without shuffling (dc-compress). So
//! when Stitch finishes, the same input gives the same outcome. A run in a
//! worker process can also end at its time or memory limit, and whether it does
//! depends on the machine. To replay an iteration exactly, read its outcome
//! from the run's log; do not run the step again.
//!
//! # What the caller checks
//!
//! The candidates must be verified: [`run_iteration`] checks their syntax,
//! expansion and type, not their behaviour on the task's examples. How fast a
//! program's cost grows with its input plays no part yet: that is the
//! cost-aware objective of P5.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::AtomicBool;

use dc_compress::worker::{Worker, WorkerError};
use dc_compress::{
    compress, compress_isolated, CompressConfig, CompressError, Compression, FrontierProgram,
    ABSTRACTION_PREFIX,
};
use dc_core::sexpr::MAX_PRINT_NODES;
use dc_core::term::children;
use dc_core::types::{check, infer};
use dc_core::{sexpr, Arena, ExpandError, Expander, LibDef, LibId, Scheme, Term, TermId};
use dc_stitch::{to_expr, EncodeError, Expr, LibNames};
use serde::{Deserialize, Serialize};

/// One library definition, as text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryEntry {
    /// The number of parameters.
    pub arity: u32,
    /// The closed term `λ^arity. body`, as an s-expression (`dc_core::sexpr`).
    pub term: String,
}

/// A task of the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskSpec {
    /// The task's name, as the frontiers and candidates name it.
    pub name: String,
    /// The type every solution must have. Its type variables are rigid.
    pub signature: Scheme,
}

/// A verified program proposed in this iteration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The task it solves.
    pub task: String,
    /// The program, a closed core term as an s-expression, which may use the
    /// library.
    pub program: String,
}

/// The settings of the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DriverConfig {
    /// The most programs a task's frontier keeps. It is positive.
    pub frontier_size: usize,
    /// The settings of the Stitch run.
    pub compress: CompressConfig,
}

/// The input of one step. Every name in `frontiers` and every program in it is
/// checked: see the crate docs.
#[derive(Clone, Copy, Debug)]
pub struct IterationInput<'a> {
    /// The run's tasks, in the run's order.
    pub tasks: &'a [TaskSpec],
    /// The library. Entry `i` is `lib_i`.
    pub library: &'a [LibraryEntry],
    /// The frontier of each task, best first. A task with no entry has an
    /// empty frontier.
    pub frontiers: &'a BTreeMap<String, Vec<String>>,
    /// The verified programs of this iteration, in the order they came in.
    pub candidates: &'a [Candidate],
}

/// Runs Stitch on the selected programs.
pub trait Stitch {
    /// Compresses `programs`, which may use `libs`, as [`dc_compress::compress`]
    /// does.
    fn run(
        &mut self,
        arena: &mut Arena,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError>;
}

/// Runs Stitch in this process, with no limits ([`dc_compress::compress`]).
/// For tests and tools. A Stitch panic carries on in the caller.
#[derive(Clone, Copy, Debug, Default)]
pub struct InProcess;

impl Stitch for InProcess {
    fn run(
        &mut self,
        arena: &mut Arena,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        compress(arena, libs, programs, cfg)
    }
}

/// Runs Stitch in a worker process with the worker's limits
/// ([`dc_compress::compress_isolated`]). Setting `cancel` stops the run.
#[derive(Clone, Copy, Debug)]
pub struct Isolated<'a> {
    /// How to start the worker, and its limits.
    pub worker: &'a Worker,
    /// Set from another thread to stop the run.
    pub cancel: &'a AtomicBool,
}

impl Stitch for Isolated<'_> {
    fn run(
        &mut self,
        arena: &mut Arena,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        compress_isolated(arena, libs, programs, cfg, self.worker, self.cancel)
    }
}

/// Where a program in a task's pool came from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    /// Member `index` of the task's frontier.
    Member {
        /// The task.
        task: String,
        /// The member's place in the frontier given.
        index: usize,
    },
    /// Candidate `index` of the input.
    Candidate {
        /// The candidate's place in the input.
        index: usize,
    },
}

/// What happened to a program in the step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Fate {
    /// It is in its task's new frontier, perhaps rewritten.
    Kept,
    /// It expands to the same term as an earlier program of its task.
    Duplicate {
        /// The earlier program.
        of: Source,
    },
    /// Its task kept [`DriverConfig::frontier_size`] programs that cost less,
    /// or that cost the same and came earlier.
    Trimmed,
    /// The candidate names no task of the run, or does not parse, is too large
    /// to print, is not closed, does not expand under the library, or does not
    /// type against its task's signature.
    Invalid {
        /// Why.
        reason: String,
    },
}

/// The fate of one program.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramFate {
    /// Where the program came from.
    pub source: Source,
    /// What happened to it.
    pub fate: Fate,
}

/// A program left out of the Stitch run, because the bridge refused it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeftOut {
    /// Its task.
    pub task: String,
    /// The program.
    pub program: String,
    /// Why the bridge refused it.
    pub reason: String,
}

/// A new abstraction that does not type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Untyped {
    /// Its place among the new abstractions: it would have been
    /// `lib_{library.len() + abstraction}`.
    pub abstraction: usize,
    /// Its body.
    pub term: String,
    /// The type error.
    pub error: String,
}

/// A program whose rewriting was refused, so it stays as it was.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fallback {
    /// Its task.
    pub task: String,
    /// The program, as kept.
    pub program: String,
    /// The rewriting that was refused.
    pub rewritten: String,
    /// Why.
    pub reason: String,
}

/// What the compression step did.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum CompressionReport {
    /// No program was left to compress, so Stitch did not run.
    NoPrograms {
        /// The programs left out, if any were left out.
        left_out: Vec<LeftOut>,
    },
    /// The run failed, so the library is unchanged and the programs are as
    /// they were.
    Failed {
        /// The programs left out before the run that failed.
        left_out: Vec<LeftOut>,
        /// The error.
        error: String,
    },
    /// The run finished.
    Ran {
        /// The programs left out of the run.
        left_out: Vec<LeftOut>,
        /// The number of programs Stitch was given.
        programs: usize,
        /// The number of steps Stitch returned.
        stitch_steps: usize,
        /// The first step the bridge dropped, and why.
        rejected: Option<String>,
        /// The number of abstractions the bridge accepted.
        found: usize,
        /// The first accepted abstraction that does not type.
        untyped: Option<Untyped>,
        /// The number of typed abstractions cut because no accepted rewriting
        /// used them.
        unused: usize,
        /// The number of abstractions that join the library.
        kept: usize,
        /// The programs whose rewriting was refused.
        fallbacks: Vec<Fallback>,
    },
}

/// What happened to each program in one step, for the run's log. The new
/// library and frontiers are in [`IterationOutcome`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IterationReport {
    /// The fate of every frontier member and every candidate: for each task in
    /// order, its members and then its candidates, and last the candidates
    /// that name no task of the run.
    pub programs: Vec<ProgramFate>,
    /// The compression step.
    pub compression: CompressionReport,
}

/// The result of one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationOutcome {
    /// The abstractions that join the end of the library.
    pub new_library: Vec<LibraryEntry>,
    /// The new frontier of every task, best first.
    pub frontiers: BTreeMap<String, Vec<String>>,
    /// What happened.
    pub report: IterationReport,
}

/// Why the step gave no outcome. Most variants mean the input state is wrong;
/// [`DriverError::Interrupted`] means the step should be run again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DriverError {
    /// [`DriverConfig::frontier_size`] is zero.
    NoFrontier,
    /// A task is named twice.
    DuplicateTask(String),
    /// A frontier is given for a task the run does not have.
    UnknownTask(String),
    /// A library entry does not parse, expand or type.
    Library {
        /// The entry's index.
        index: usize,
        /// Why.
        reason: String,
    },
    /// A frontier member does not parse, is not closed, does not expand, or
    /// does not type.
    Member {
        /// The task.
        task: String,
        /// The member's index.
        index: usize,
        /// Why.
        reason: String,
    },
    /// A term is too large to print as text. Every program is checked for
    /// this before it joins a pool, so this is not expected.
    TooLarge,
    /// The Stitch run was cancelled, or its worker or thread could not be
    /// started or read. This says nothing about the input: do not record the
    /// iteration as finished, and run the step again.
    Interrupted(CompressError),
    /// The library has more entries than a library id can number.
    LibraryTooLarge,
}

impl fmt::Display for DriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DriverError::NoFrontier => f.write_str("the frontier size must be at least 1"),
            DriverError::DuplicateTask(t) => write!(f, "task {t:?} is named twice"),
            DriverError::UnknownTask(t) => write!(f, "a frontier is given for unknown task {t:?}"),
            DriverError::Library { index, reason } => write!(f, "library entry {index}: {reason}"),
            DriverError::Member {
                task,
                index,
                reason,
            } => write!(f, "member {index} of task {task:?}: {reason}"),
            DriverError::TooLarge => f.write_str("a term is too large to print"),
            DriverError::LibraryTooLarge => f.write_str("the library has too many entries"),
            DriverError::Interrupted(e) => write!(f, "the Stitch run was interrupted: {e}"),
        }
    }
}

impl std::error::Error for DriverError {}

/// The cost Stitch gives `program` with its default costs: 1 for a lambda or
/// an application, and 100 for a variable, a parameter, a primitive, a literal
/// or a library use. `names` names the library.
pub fn stitch_cost(arena: &Arena, program: TermId, names: &LibNames) -> Result<u64, EncodeError> {
    let e = to_expr(arena, program, names)?;
    Ok(expr_cost(&e))
}

/// [`stitch_cost`] of an expression, found with a work list.
fn expr_cost(e: &Expr) -> u64 {
    let mut total: u64 = 0;
    let mut stack = vec![e];
    while let Some(e) = stack.pop() {
        match e {
            Expr::Lam(b) | Expr::TaggedLam(_, b) => {
                total = total.saturating_add(1);
                stack.push(b);
            }
            Expr::App(f, x) => {
                total = total.saturating_add(1);
                stack.push(f);
                stack.push(x);
            }
            Expr::Var(_) | Expr::IVar(_) | Expr::Sym(_) => total = total.saturating_add(100),
        }
    }
    total
}

/// The names of a library of `len` entries, as the bridge names them.
fn names(len: usize) -> Result<LibNames, DriverError> {
    let len = u32::try_from(len).map_err(|_| DriverError::LibraryTooLarge)?;
    LibNames::sequential(ABSTRACTION_PREFIX, len).map_err(|_| DriverError::LibraryTooLarge)
}

/// `id` as text. Every program in the pool passed the same check, and an
/// abstraction's body is no larger than the programs it was found in.
fn print(arena: &Arena, id: TermId) -> Result<String, DriverError> {
    sexpr::print(arena, id).map_err(|_| DriverError::TooLarge)
}

/// The largest library id `id` uses, found with a work list over the shared
/// term graph.
fn max_lib(arena: &Arena, id: TermId) -> Option<usize> {
    let mut seen = HashSet::new();
    let mut stack = vec![id];
    let mut max = None;
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let term = arena.get(t);
        if let Term::Lib(LibId(l)) = term {
            let l = l as usize;
            max = Some(max.map_or(l, |m: usize| m.max(l)));
        }
        stack.extend(children(&term));
    }
    max
}

/// The checked library: its definitions, an expander, and their schemes.
struct Library {
    defs: Vec<LibDef>,
    expander: Expander,
    schemes: Vec<Scheme>,
}

impl Library {
    fn load(arena: &mut Arena, entries: &[LibraryEntry]) -> Result<Library, DriverError> {
        let mut defs = Vec::with_capacity(entries.len());
        for (index, e) in entries.iter().enumerate() {
            let bad = |reason: String| DriverError::Library { index, reason };
            let term = sexpr::parse(arena, &e.term).map_err(|err| bad(err.to_string()))?;
            // The expander names the entry that uses a library, except when the
            // library does not exist. So check that here, for each entry.
            if let Some(l) = max_lib(arena, term).filter(|&l| l >= index) {
                return Err(bad(format!(
                    "it uses lib_{l}, which is not an earlier entry"
                )));
            }
            defs.push(LibDef {
                arity: e.arity,
                term,
            });
        }
        let expander = Expander::new(arena, &defs).map_err(|err| DriverError::Library {
            index: match err {
                ExpandError::UnknownLib(LibId(l))
                | ExpandError::ForwardReference { lib: LibId(l), .. }
                | ExpandError::NotClosed { lib: LibId(l), .. }
                | ExpandError::TooFewLambdas { lib: LibId(l), .. } => l as usize,
            },
            reason: err.to_string(),
        })?;
        let mut schemes: Vec<Scheme> = Vec::with_capacity(defs.len());
        for (index, def) in defs.iter().enumerate() {
            let s = infer(arena, def.term, &schemes).map_err(|err| DriverError::Library {
                index,
                reason: err.to_string(),
            })?;
            schemes.push(s);
        }
        Ok(Library {
            defs,
            expander,
            schemes,
        })
    }

    /// Parses `text` and checks that it can be printed again, is closed,
    /// expands, and types against `signature`. Returns the program and its
    /// expansion.
    fn check(
        &mut self,
        arena: &mut Arena,
        text: &str,
        signature: &Scheme,
    ) -> Result<(TermId, TermId), String> {
        let program = sexpr::parse(arena, text).map_err(|e| e.to_string())?;
        // The new frontier is text, so a program must print.
        if arena.tree_size(program, MAX_PRINT_NODES).is_none() {
            return Err(format!(
                "the program has more than {MAX_PRINT_NODES} nodes written out"
            ));
        }
        if !arena.is_closed(program) {
            return Err("the program is not closed".to_owned());
        }
        let expanded = self
            .expander
            .expand(arena, program)
            .map_err(|e| e.to_string())?;
        check(arena, program, &self.schemes, signature).map_err(|e| e.to_string())?;
        Ok((program, expanded))
    }
}

/// A program in a task's pool that passed its checks.
#[derive(Clone, Debug)]
struct Pooled {
    source: Source,
    program: TermId,
    /// The program's expansion. A rewriting must keep it.
    expanded: TermId,
    /// [`stitch_cost`] under the library, or why Stitch cannot be given the
    /// program.
    cost: Result<u64, String>,
}

impl Pooled {
    /// The cost to sort by. A program Stitch cannot be given sorts last.
    fn sort_key(&self) -> u64 {
        *self.cost.as_ref().unwrap_or(&u64::MAX)
    }
}

/// Builds each task's new frontier from its members and candidates (step 2 of
/// the crate docs). Returns the kept programs of each task, in task order, and
/// the fate of every program.
fn select(
    arena: &mut Arena,
    library: &mut Library,
    names: &LibNames,
    input: IterationInput<'_>,
    frontier_size: usize,
) -> Result<(Vec<Vec<Pooled>>, Vec<ProgramFate>), DriverError> {
    let mut by_task: HashMap<&str, usize> = HashMap::new();
    for (i, t) in input.tasks.iter().enumerate() {
        if by_task.insert(t.name.as_str(), i).is_some() {
            return Err(DriverError::DuplicateTask(t.name.clone()));
        }
    }
    if let Some(t) = input
        .frontiers
        .keys()
        .find(|t| !by_task.contains_key(t.as_str()))
    {
        return Err(DriverError::UnknownTask(t.clone()));
    }
    let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); input.tasks.len()];
    let mut stray = Vec::new();
    for (index, c) in input.candidates.iter().enumerate() {
        match by_task.get(c.task.as_str()) {
            Some(&t) => candidates[t].push(index),
            None => stray.push(ProgramFate {
                source: Source::Candidate { index },
                fate: Fate::Invalid {
                    reason: format!("the run has no task {:?}", c.task),
                },
            }),
        }
    }

    let mut kept = Vec::with_capacity(input.tasks.len());
    let mut fates = Vec::new();
    for (t, task) in input.tasks.iter().enumerate() {
        let members = input
            .frontiers
            .get(&task.name)
            .map(Vec::as_slice)
            .unwrap_or_default();
        // Each program's source, and its fate if it is not in the pool.
        let mut entries: Vec<(Source, Option<Fate>)> = Vec::new();
        let mut pool: Vec<Pooled> = Vec::new();
        let mut seen: HashMap<TermId, Source> = HashMap::new();
        let mut add = |source: Source,
                       checked: Result<(TermId, TermId), String>,
                       arena: &Arena|
         -> Option<Fate> {
            let (program, expanded) = match checked {
                Ok(c) => c,
                Err(reason) => return Some(Fate::Invalid { reason }),
            };
            if let Some(of) = seen.get(&expanded) {
                return Some(Fate::Duplicate { of: of.clone() });
            }
            seen.insert(expanded, source.clone());
            pool.push(Pooled {
                source,
                program,
                expanded,
                cost: stitch_cost(arena, program, names).map_err(|e| e.to_string()),
            });
            None
        };
        for (index, text) in members.iter().enumerate() {
            let source = Source::Member {
                task: task.name.clone(),
                index,
            };
            let checked = library
                .check(arena, text, &task.signature)
                .map_err(|reason| DriverError::Member {
                    task: task.name.clone(),
                    index,
                    reason,
                })?;
            let fate = add(source.clone(), Ok(checked), arena);
            entries.push((source, fate));
        }
        for &index in &candidates[t] {
            let source = Source::Candidate { index };
            let checked = library.check(arena, &input.candidates[index].program, &task.signature);
            let fate = add(source.clone(), checked, arena);
            entries.push((source, fate));
        }
        // A stable sort: among programs of equal cost, the earlier stays first.
        pool.sort_by_key(Pooled::sort_key);
        let trimmed: HashSet<Source> = pool
            .iter()
            .skip(frontier_size)
            .map(|p| p.source.clone())
            .collect();
        pool.truncate(frontier_size);
        for (source, fate) in entries {
            let fate = fate.unwrap_or(if trimmed.contains(&source) {
                Fate::Trimmed
            } else {
                Fate::Kept
            });
            fates.push(ProgramFate { source, fate });
        }
        kept.push(pool);
    }
    fates.extend(stray);
    Ok((kept, fates))
}

/// How the Stitch run of step 3 ended.
enum StitchRun {
    /// No program was left to give Stitch.
    NoPrograms,
    /// The run failed for a reason that the same input would give again.
    Failed(String),
    /// The run finished. `slots` holds the place of each program Stitch was
    /// given, as (task, position in the task's kept programs).
    Ran {
        slots: Vec<(usize, usize)>,
        compression: Box<Compression>,
    },
}

/// Whether `e` says nothing about the input: the run was cancelled, or the
/// worker or Stitch's thread could not be started or read.
fn is_interruption(e: &CompressError) -> bool {
    matches!(
        e,
        CompressError::StitchThread(_)
            | CompressError::Worker(
                WorkerError::Cancelled | WorkerError::Spawn(_) | WorkerError::Io(_)
            )
    )
}

/// Runs Stitch on the kept programs, leaving out each program the bridge
/// refuses before Stitch runs (step 3 of the crate docs). Programs that cannot
/// be encoded are left out first; the bridge's other refusals are found one
/// run at a time. Adds the programs left out to `left_out`.
fn run_stitch(
    arena: &mut Arena,
    libs: &[LibDef],
    tasks: &[TaskSpec],
    kept: &[Vec<Pooled>],
    cfg: &CompressConfig,
    stitch: &mut dyn Stitch,
    left_out: &mut Vec<LeftOut>,
) -> Result<StitchRun, DriverError> {
    let mut leave_out = |arena: &Arena, t: usize, p: usize, reason: String| {
        print(arena, kept[t][p].program).map(|program| {
            left_out.push(LeftOut {
                task: tasks[t].name.clone(),
                program,
                reason,
            })
        })
    };
    let mut slots = Vec::new();
    for (t, pool) in kept.iter().enumerate() {
        for (p, pooled) in pool.iter().enumerate() {
            match &pooled.cost {
                Ok(_) => slots.push((t, p)),
                Err(e) => leave_out(arena, t, p, format!("it cannot be encoded: {e}"))?,
            }
        }
    }
    loop {
        if slots.is_empty() {
            return Ok(StitchRun::NoPrograms);
        }
        let programs: Vec<FrontierProgram> = slots
            .iter()
            .map(|&(t, p)| FrontierProgram {
                task: tasks[t].name.clone(),
                program: kept[t][p].program,
            })
            .collect();
        match stitch.run(arena, libs, &programs, cfg) {
            Ok(c) if c.rewritten.len() == slots.len() => {
                return Ok(StitchRun::Ran {
                    slots,
                    compression: Box::new(c),
                })
            }
            Ok(c) => {
                return Ok(StitchRun::Failed(format!(
                    "the runner returned {} rewritten programs for {} inputs",
                    c.rewritten.len(),
                    slots.len()
                )))
            }
            Err(
                e @ (CompressError::OpenProgram { index }
                | CompressError::Encode { index, .. }
                | CompressError::TooDeep { index }),
            ) if index < slots.len() => {
                let (t, p) = slots.remove(index);
                leave_out(arena, t, p, e.to_string())?;
            }
            Err(e) if is_interruption(&e) => return Err(DriverError::Interrupted(e)),
            Err(e) => return Ok(StitchRun::Failed(e.to_string())),
        }
    }
}

/// Step 4 of the crate docs: types the new abstractions, accepts each
/// rewriting that types and keeps its program's expansion, and cuts the
/// abstractions that are not used. Changes the accepted programs in `kept`
/// and returns the new library entries and the report.
fn update_library(
    arena: &mut Arena,
    library: &Library,
    tasks: &[TaskSpec],
    kept: &mut [Vec<Pooled>],
    slots: &[(usize, usize)],
    c: &Compression,
    left_out: Vec<LeftOut>,
) -> Result<(Vec<LibraryEntry>, CompressionReport), DriverError> {
    let old_len = library.defs.len();
    let mut schemes = library.schemes.clone();
    let mut untyped = None;
    for (i, def) in c.new_libs.iter().enumerate() {
        match infer(arena, def.term, &schemes) {
            Ok(s) => schemes.push(s),
            Err(e) => {
                untyped = Some(Untyped {
                    abstraction: i,
                    term: print(arena, def.term)?,
                    error: e.to_string(),
                });
                break;
            }
        }
    }
    let typed = schemes.len() - old_len;
    // The bridge checks that each rewriting keeps the expansion. It is checked
    // again here, against the typed abstractions, so that a runner that skips
    // the check cannot change a program.
    let typed_libs: Vec<LibDef> = library
        .defs
        .iter()
        .chain(&c.new_libs[..typed])
        .copied()
        .collect();
    let mut expander = match Expander::new(arena, &typed_libs) {
        Ok(x) => x,
        Err(e) => {
            let error = format!("the new abstractions do not expand: {e}");
            return Ok((Vec::new(), CompressionReport::Failed { left_out, error }));
        }
    };
    let mut fallbacks = Vec::new();
    // The largest library id an accepted rewriting uses.
    let mut used: Option<usize> = None;
    for (&(t, p), &rewritten) in slots.iter().zip(&c.rewritten) {
        let task = &tasks[t];
        let uses = max_lib(arena, rewritten);
        let text = sexpr::print(arena, rewritten);
        let refused = if text.is_err() {
            Some(format!(
                "it has more than {MAX_PRINT_NODES} nodes written out"
            ))
        } else if uses.is_some_and(|l| l >= schemes.len()) {
            Some("it uses a new abstraction that does not type, or one after it".to_owned())
        } else if let Err(e) = check(arena, rewritten, &schemes, &task.signature) {
            Some(e.to_string())
        } else {
            match expander.expand(arena, rewritten) {
                Err(e) => Some(format!("it does not expand: {e}")),
                Ok(e) if e != kept[t][p].expanded => {
                    Some("it does not expand to its original".to_owned())
                }
                Ok(_) => None,
            }
        };
        match refused {
            Some(reason) => fallbacks.push(Fallback {
                task: task.name.clone(),
                program: print(arena, kept[t][p].program)?,
                rewritten: text.unwrap_or_default(),
                reason,
            }),
            None => {
                kept[t][p].program = rewritten;
                used = used.max(uses);
            }
        }
    }
    // Cut the new abstractions after the last one used. A kept one uses only
    // earlier ones, so those before it stay.
    let kept_new = used.map_or(0, |l| (l + 1).saturating_sub(old_len));
    let new_library = c.new_libs[..kept_new]
        .iter()
        .map(|def| {
            Ok(LibraryEntry {
                arity: def.arity,
                term: print(arena, def.term)?,
            })
        })
        .collect::<Result<Vec<LibraryEntry>, DriverError>>()?;
    let report = CompressionReport::Ran {
        left_out,
        programs: slots.len(),
        stitch_steps: c.stitch_steps,
        rejected: c
            .rejected
            .as_ref()
            .map(|r| format!("step {} ({}): {}", r.step, r.name, r.reason)),
        found: c.new_libs.len(),
        untyped,
        unused: typed - kept_new,
        kept: kept_new,
        fallbacks,
    };
    Ok((new_library, report))
}

/// Runs one step. See the crate docs.
pub fn run_iteration(
    input: IterationInput<'_>,
    cfg: &DriverConfig,
    stitch: &mut dyn Stitch,
) -> Result<IterationOutcome, DriverError> {
    if cfg.frontier_size == 0 {
        return Err(DriverError::NoFrontier);
    }
    let mut arena = Arena::new();
    let mut library = Library::load(&mut arena, input.library)?;
    let old_len = library.defs.len();
    let old_names = names(old_len)?;
    let (mut kept, programs) = select(
        &mut arena,
        &mut library,
        &old_names,
        input,
        cfg.frontier_size,
    )?;

    let mut left_out = Vec::new();
    let run = run_stitch(
        &mut arena,
        &library.defs,
        input.tasks,
        &kept,
        &cfg.compress,
        stitch,
        &mut left_out,
    )?;
    let (new_library, compression) = match run {
        StitchRun::NoPrograms => (Vec::new(), CompressionReport::NoPrograms { left_out }),
        StitchRun::Failed(error) => (Vec::new(), CompressionReport::Failed { left_out, error }),
        StitchRun::Ran { slots, compression } => update_library(
            &mut arena,
            &library,
            input.tasks,
            &mut kept,
            &slots,
            &compression,
            left_out,
        )?,
    };

    // Sort each frontier again under the new library.
    let new_names = names(old_len + new_library.len())?;
    let mut frontiers = BTreeMap::new();
    for (task, pool) in input.tasks.iter().zip(&mut kept) {
        for p in pool.iter_mut() {
            p.cost = stitch_cost(&arena, p.program, &new_names).map_err(|e| e.to_string());
        }
        pool.sort_by_key(Pooled::sort_key);
        let texts = pool
            .iter()
            .map(|p| print(&arena, p.program))
            .collect::<Result<Vec<String>, DriverError>>()?;
        frontiers.insert(task.name.clone(), texts);
    }
    Ok(IterationOutcome {
        new_library,
        frontiers,
        report: IterationReport {
            programs,
            compression,
        },
    })
}
