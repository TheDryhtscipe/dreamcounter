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
//!    the order given. Of the programs with one expansion, the pool keeps the
//!    cheapest, and the earliest of those on a tie; the others are each a
//!    [`Fate::Duplicate`] of it. The pool is sorted, and the first
//!    [`DriverConfig::frontier_size`] programs are kept. At λ = 0
//!    ([`DriverConfig::lambda`]) the sort is by [`stitch_cost`], stably: pure
//!    description length. At λ > 0 it is by `DL(p | L) + λ · C(p)` of
//!    `J(L)` (decisions D4 and D27, [`objective::rank`]), with each `C(p)`
//!    from the program's [`Growth`] ([`IterationInput::growth`]) taken over
//!    the pool, the earlier on a tie, and incorrect programs last.
//! 3. **Compression.** The kept programs of every task, in task order, go to a
//!    [`Stitch`] runner, which counts only the cheapest rewrite of each task.
//!    At λ = 0 every kept program goes. At λ > 0 a task's correct programs go
//!    whose `C(p)`, taken over the kept frontier, is at most that of the
//!    member `J(L)` selects: Stitch then chooses abstractions for the
//!    selected member and for members that grow no faster, never for a
//!    member that grows faster (D4: select, compress, select again). A
//!    program the bridge refuses before Stitch runs (it
//!    cannot be encoded, or is too deep) is left out and stays in its frontier
//!    as it is. If the run is cancelled, or its worker cannot be started, the
//!    step returns [`DriverError::Interrupted`]. Any other error, such as a
//!    worker stopped at its time limit, skips compression for the iteration:
//!    the frontiers are still updated, and the library is unchanged.
//! 4. **Library update.** First, each new abstraction with no parameters whose
//!    body is not a lambda, a literal, a primitive or a library reference is
//!    written out in place: in the later abstractions and in the rewritten
//!    programs, its uses are replaced by its body, and the abstractions after
//!    it are numbered down. A use of a library entry means its definition
//!    written out (D1), so such an entry is computed again at each use, though
//!    it reads as a constant. The report lists each one
//!    ([`CompressionReport::Ran`], `written_out`).
//!    With [`DriverConfig::refuse_bucketing`], each new abstraction left that
//!    is bucketing is written out as well (decision D36). An abstraction is
//!    bucketing if its body holds a term that puts a value onto the list in an
//!    array slot (the two forms are in `bucketing.rs`), or a use of a library
//!    entry or an earlier new abstraction that is bucketing. A use of it is
//!    replaced by its body with the arguments put in for the parameters, as
//!    expansion does (D1), so a use with too few arguments becomes a function
//!    of the others. The report marks each one (`bucketing`), except one with
//!    no parameters that does work, which the first rule has already taken.
//!    Then each new abstraction gets a type scheme inferred from
//!    its body with the earlier ones, and the new abstractions are cut at the
//!    first that does not type or is too large to print. A rewritten program
//!    is accepted if it types against its task's signature with the library's
//!    schemes and expands to what its original expands to. Otherwise the program stays as it was:
//!    the rewriting keeps the expansion but not always the type (see the
//!    dc-core test `a_parameter_used_at_two_types_can_make_the_refactored_program_untypable`).
//!    The bridge returns only the programs rewritten with every abstraction it
//!    accepted. So a program that uses a cut abstraction stays as it was, even
//!    when a rewriting with the earlier abstractions alone would type. New
//!    abstractions after the last one an accepted rewriting uses are cut too.
//!    Each frontier is sorted again under the new library, as in step 2 but
//!    with each `C(p)` taken over the new frontier, so its first member is
//!    the one `J(L)` selects. [`IterationReport::selection`] gives each
//!    member's description length, `C(p)` and the cause if it has no fitted
//!    class, the selected member, and the member λ = 0 would select from the
//!    same frontier.
//!
//! # The prelude
//!
//! Programs that proposers write are compiled with the prelude, so they use its
//! functions as `Term::Lib(LibId(PRELUDE_LIB_BASE + i))` (`dc_surface::prelude`).
//! [`IterationInput::prelude`] says which prelude they were compiled against:
//! `None` is the prelude-off ablation, and a prelude reference is then an unknown
//! library. With a prelude, every check above types a reference with the
//! function's scheme over the learned schemes, and expands the prelude's
//! references first and the learned abstractions second. Stitch is given each
//! prelude function as an opaque token with its own name, never its body, and an
//! abstraction or a rewritten program may use it. Frontier members, library
//! entries and new abstractions keep the references, so they are not replaced by
//! definitions in the run's state. [`stitch_cost`] charges a reference as one
//! library use (100), the same as Stitch does.
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
//! expansion and type, not their behaviour on the task's examples. The step
//! does not measure how a program's cost grows: the caller gives each
//! program's [`Growth`]. A rewritten program keeps its original's expansion,
//! so a cost measured on the expansion holds for the program in its new
//! frontier, found through [`IterationOutcome::sources`].

mod bucketing;
pub mod objective;

pub use objective::{Growth, Lambda};

use objective::{member_costs, Member, Unfitted};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::atomic::AtomicBool;

use dc_compress::worker::{Worker, WorkerError};
use dc_compress::{
    compress_isolated_with, compress_with, expand_prelude, library_names, CompressConfig,
    CompressError, Compression, FrontierProgram, IsolatedRun,
};
use dc_core::sexpr::MAX_PRINT_NODES;
use dc_core::term::children;
use dc_core::types::{check, infer};
use dc_core::{
    sexpr, Arena, ExpandError, Expander, LibDef, LibId, Scheme, Term, TermId, TypeError,
};
use dc_stitch::{to_expr, EncodeError, Expr, LibNames};
use dc_surface::prelude::Prelude;
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
    /// The settings of the Stitch run, or `None` for a step with no
    /// compression: it selects the frontiers and leaves the library as it is,
    /// as an LLM-only baseline needs (decision D24).
    pub compress: Option<CompressConfig>,
    /// Whether each bucketing abstraction Stitch finds is refused: written
    /// out where it is used instead of joining the library (decision D36;
    /// step 4 of the crate docs).
    pub refuse_bucketing: bool,
    /// λ of the cost-aware objective `J(L)` (decision D4): how much one step
    /// of `C(p)` weighs against description length when frontiers are
    /// selected. At [`Lambda::ZERO`] the step selects by description length
    /// alone, as it did before λ was a setting.
    pub lambda: Lambda,
}

/// What is known about the cost growth of each program the step reads, for
/// selection under `J(L)` ([`objective`]). A program with no entry is
/// [`Growth::NotMeasured`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Growths {
    /// Each task's frontier members, in the frontier's order.
    pub members: BTreeMap<String, Vec<Growth>>,
    /// Each candidate, in the order of [`IterationInput::candidates`].
    pub candidates: Vec<Growth>,
}

impl Growths {
    fn of(&self, source: &Source) -> Growth {
        let found = match source {
            Source::Member { task, index } => self.members.get(task).and_then(|m| m.get(*index)),
            Source::Candidate { index } => self.candidates.get(*index),
        };
        found.copied().unwrap_or(Growth::NotMeasured)
    }
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
    /// The prelude that programs were compiled against, or `None` when they
    /// were compiled without it (`PreludeMode::Off`). A program, a frontier
    /// member or a library entry may use its functions as
    /// `Term::Lib(LibId(PRELUDE_LIB_BASE + i))`; with `None`, such a reference
    /// is an unknown library.
    pub prelude: Option<&'a Prelude>,
    /// The cost growth of each program, or `None` if none is known: every
    /// program is then [`Growth::NotMeasured`].
    pub growth: Option<&'a Growths>,
}

/// Runs Stitch on the selected programs.
pub trait Stitch {
    /// Compresses `programs`, which may use `libs` and the functions of
    /// `prelude`, as [`dc_compress::compress_with`] does.
    fn run(
        &mut self,
        arena: &mut Arena,
        prelude: Option<&Prelude>,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError>;
}

/// Runs Stitch in this process, with no limits ([`dc_compress::compress_with`]).
/// For tests and tools. A Stitch panic carries on in the caller.
#[derive(Clone, Copy, Debug, Default)]
pub struct InProcess;

impl Stitch for InProcess {
    fn run(
        &mut self,
        arena: &mut Arena,
        prelude: Option<&Prelude>,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        compress_with(arena, prelude, libs, programs, cfg)
    }
}

/// Runs Stitch in a worker process with the worker's limits
/// ([`dc_compress::compress_isolated_with`]). Setting `cancel` stops the run.
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
        prelude: Option<&Prelude>,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        let run = IsolatedRun {
            worker: self.worker,
            cancel: self.cancel,
        };
        compress_isolated_with(arena, prelude, libs, programs, cfg, run)
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
    /// It expands to the same term as another program of its task, which is
    /// cheaper, or costs the same and came earlier.
    Duplicate {
        /// The other program. Its own fate is [`Fate::Kept`] or
        /// [`Fate::Trimmed`], so it is not always in the new frontier.
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

/// The first new abstraction that is cut because it does not type or is too
/// large to print.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Untyped {
    /// Its place among the new abstractions left after those written out
    /// ([`WrittenOut`]): it would have been `lib_{library.len() + abstraction}`.
    pub abstraction: usize,
    /// Its term, or nothing if it is too large to print.
    pub term: String,
    /// The type error, or why it cannot be printed.
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
    /// Compression is off for the run ([`DriverConfig::compress`] is
    /// `None`), so Stitch did not run.
    Off,
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
        /// The first accepted abstraction that does not type or is too large
        /// to print. It and the abstractions after it are cut.
        untyped: Option<Untyped>,
        /// The number of typed abstractions cut because no accepted rewriting
        /// used them.
        unused: usize,
        /// The number of abstractions that join the library.
        kept: usize,
        /// The programs whose rewriting was refused.
        fallbacks: Vec<Fallback>,
        /// The abstractions written out in place instead of joining the
        /// library, in the order Stitch found them (step 4 of the crate
        /// docs). Absent in a report from before they were.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        written_out: Vec<WrittenOut>,
    },
}

/// An abstraction written out where it is used instead of joining the
/// library: one with no parameters that does work, or a bucketing one that
/// the step refuses ([`DriverConfig::refuse_bucketing`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WrittenOut {
    /// Its index among the abstractions Stitch found.
    pub abstraction: usize,
    /// Its body, as an s-expression, as Stitch found it, or nothing if it is
    /// too large to print.
    pub term: String,
    /// Whether it was written out because it is bucketing. Absent, and false,
    /// for one with no parameters that does work.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bucketing: bool,
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
    /// How each task's new frontier scores under `J(L)`. `None` in a report
    /// written before the step logged it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<SelectionReport>,
}

/// How each task's new frontier scores under `J(L)` (decisions D4 and
/// D27), with the selection λ = 0 would make from the same frontier
/// (decision D33). Each member's `C(p)` is
/// [`member_costs`] of its task's new frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReport {
    /// The run's λ.
    pub lambda: Lambda,
    /// Each task of the run.
    pub tasks: BTreeMap<String, TaskSelection>,
    /// `Σ_t C(selected_t)`, component (b) of D4.
    pub total_cost: u64,
    /// `Σ_t C` of the member λ = 0 selects in each task.
    pub baseline_total_cost: u64,
    /// How many tasks select another member than λ = 0 does.
    pub changed: usize,
}

/// One task's new frontier under `J(L)`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskSelection {
    /// The position of the selected member, or `None` if no member is
    /// correct.
    pub selected: Option<usize>,
    /// The position of the member λ = 0 would select.
    pub baseline: Option<usize>,
    /// Each member, in the frontier's order.
    pub members: Vec<MemberScore>,
}

/// One member of a new frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberScore {
    /// `DL(p | L)` under the new library, or `None` if Stitch cannot be
    /// given the program; it then ranks as the largest description length.
    pub dl: Option<u64>,
    /// `C(p)`, or `None` for an incorrect member, which is not a solution.
    pub c: Option<u32>,
    /// Why the member has no fitted class, if it has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unfitted: Option<Unfitted>,
}

/// The result of one step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IterationOutcome {
    /// The abstractions that join the end of the library.
    pub new_library: Vec<LibraryEntry>,
    /// The new frontier of every task, best first.
    pub frontiers: BTreeMap<String, Vec<String>>,
    /// Where each member of each new frontier came from, in the same order.
    pub sources: BTreeMap<String, Vec<Source>>,
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
    /// A term is too large to print as text. Every program and abstraction is
    /// checked for this first, so this means a bug.
    TooLarge,
    /// The Stitch run was cancelled, or its worker or thread could not be
    /// started or read. This says nothing about the input: do not record the
    /// iteration as finished, and run the step again.
    Interrupted(CompressError),
    /// The library has more entries than a library id can number.
    LibraryTooLarge,
    /// A prelude function's name cannot be given to Stitch, as
    /// [`dc_compress::library_names`] says. The shipped prelude has no such name.
    PreludeName(String),
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
            DriverError::PreludeName(e) => write!(f, "a prelude function cannot be named: {e}"),
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

/// The names of a library of `len` entries and of the prelude's functions, as
/// the bridge gives them to Stitch ([`dc_compress::library_names`]).
fn names(len: usize, prelude: Option<&Prelude>) -> Result<LibNames, DriverError> {
    let len = u32::try_from(len).map_err(|_| DriverError::LibraryTooLarge)?;
    // The learned names fail only for a library that reaches PRELUDE_LIB_BASE
    // entries, so an error is about a prelude name.
    library_names(len, prelude).map_err(|err| DriverError::PreludeName(err.to_string()))
}

/// `id` as text. Every program in the pool and every abstraction kept is
/// checked first, so this fails only on a bug.
fn print(arena: &Arena, id: TermId) -> Result<String, DriverError> {
    sexpr::print(arena, id).map_err(|_| DriverError::TooLarge)
}

/// The largest learned library id `id` uses, found with a work list over the
/// shared term graph. The prelude's references are not learned ids and are
/// skipped.
fn max_lib(arena: &Arena, prelude: Option<&Prelude>, id: TermId) -> Option<usize> {
    let mut seen = HashSet::new();
    let mut stack = vec![id];
    let mut max = None;
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let term = arena.get(t);
        if let Term::Lib(l) = term {
            if prelude.is_some_and(|p| p.name(l).is_some()) {
                continue;
            }
            let l = l.0 as usize;
            max = Some(max.map_or(l, |m: usize| m.max(l)));
        }
        stack.extend(children(&term));
    }
    max
}

/// The checked library: its definitions, an expander, and their schemes.
struct Library<'a> {
    /// The prelude the programs may use.
    prelude: Option<&'a Prelude>,
    /// The definitions as given, which may use the prelude.
    defs: Vec<LibDef>,
    /// Expands the learned abstractions, over terms with the prelude already
    /// expanded.
    expander: Expander,
    schemes: Vec<Scheme>,
}

impl<'a> Library<'a> {
    fn load(
        arena: &mut Arena,
        prelude: Option<&'a Prelude>,
        entries: &[LibraryEntry],
    ) -> Result<Library<'a>, DriverError> {
        let mut defs = Vec::with_capacity(entries.len());
        for (index, e) in entries.iter().enumerate() {
            let bad = |reason: String| DriverError::Library { index, reason };
            let term = sexpr::parse(arena, &e.term).map_err(|err| bad(err.to_string()))?;
            // The expander names the entry that uses a library, except when the
            // library does not exist. So check that here, for each entry.
            if let Some(l) = max_lib(arena, prelude, term).filter(|&l| l >= index) {
                return Err(bad(format!(
                    "it uses lib_{l}, which is not an earlier entry"
                )));
            }
            defs.push(LibDef {
                arity: e.arity,
                term,
            });
        }
        let expandable: Vec<LibDef> = defs
            .iter()
            .map(|d| LibDef {
                arity: d.arity,
                term: expand_prelude(prelude, arena, d.term),
            })
            .collect();
        let expander = Expander::new(arena, &expandable).map_err(|err| DriverError::Library {
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
            let s = infer_in(arena, prelude, def.term, &schemes).map_err(|err| {
                DriverError::Library {
                    index,
                    reason: err.to_string(),
                }
            })?;
            schemes.push(s);
        }
        Ok(Library {
            prelude,
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
        let bare = expand_prelude(self.prelude, arena, program);
        let expanded = self
            .expander
            .expand(arena, bare)
            .map_err(|e| e.to_string())?;
        check_in(arena, self.prelude, program, &self.schemes, signature)
            .map_err(|e| e.to_string())?;
        Ok((program, expanded))
    }
}

/// [`infer`] with the prelude's schemes added to `schemes`.
fn infer_in(
    arena: &Arena,
    prelude: Option<&Prelude>,
    term: TermId,
    schemes: &[Scheme],
) -> Result<Scheme, TypeError> {
    match prelude {
        Some(p) => infer(arena, term, &p.with_libs(schemes)),
        None => infer(arena, term, schemes),
    }
}

/// [`check`] with the prelude's schemes added to `schemes`.
fn check_in(
    arena: &Arena,
    prelude: Option<&Prelude>,
    term: TermId,
    schemes: &[Scheme],
    signature: &Scheme,
) -> Result<(), TypeError> {
    match prelude {
        Some(p) => check(arena, term, &p.with_libs(schemes), signature),
        None => check(arena, term, schemes, signature),
    }
}

/// A program and its expansion, or why the program is invalid.
type Checked = Result<(TermId, TermId), String>;

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
    /// What is known about its cost growth.
    growth: Growth,
}

impl Pooled {
    /// The cost to sort by. A program Stitch cannot be given sorts last.
    fn sort_key(&self) -> u64 {
        *self.cost.as_ref().unwrap_or(&u64::MAX)
    }

    /// The program as a member of `J(L)`: its sort key as `DL(p | L)`.
    fn member(&self) -> Member {
        Member {
            dl: self.sort_key(),
            growth: self.growth,
        }
    }
}

/// `pool`'s members for [`objective`].
fn members(pool: &[Pooled]) -> Vec<Member> {
    pool.iter().map(Pooled::member).collect()
}

/// The position [`objective::select`] gives in `pool` at `lambda`.
fn selected(pool: &[Pooled], lambda: f64) -> Option<usize> {
    objective::select(&members(pool), lambda)
        .expect("a Lambda is valid")
        .map(|s| s.index)
}

/// Sorts `pool` best first. At λ = 0 this is a stable sort by
/// [`Pooled::sort_key`], as the step did before λ was a setting. Otherwise
/// it is [`objective::rank`] with each `C(p)` taken over `pool`: by
/// `DL(p | L) + λ · C(p)`, the earlier on a tie, incorrect programs last.
fn order(pool: &mut Vec<Pooled>, lambda: Lambda) {
    if lambda.is_zero() {
        pool.sort_by_key(Pooled::sort_key);
        return;
    }
    let ranked = objective::rank(&members(pool), lambda.get()).expect("a Lambda is valid");
    let mut taken: Vec<Option<Pooled>> = pool.drain(..).map(Some).collect();
    pool.extend(
        ranked
            .into_iter()
            .map(|i| taken[i].take().expect("rank gives each position once")),
    );
}

/// Which members of `pool`, a task's frontier, Stitch is given. At λ = 0,
/// all of them. Otherwise the correct members whose `C(p)` (over `pool`) is
/// at most that of the selected member: none if no member is correct.
///
/// The selected member is then the shortest of those given, so what
/// Stitch minimises, `DL(L)` plus each task's shortest given member, is
/// `J(L)` less `λ · Σ_t C(selected_t)` under the old library. A member that
/// grows faster than the selected one is not given, so no abstraction is
/// chosen for it; it stays in the frontier as it is.
fn given_to_stitch(pool: &[Pooled], lambda: Lambda) -> Vec<bool> {
    if lambda.is_zero() {
        return vec![true; pool.len()];
    }
    let costs = member_costs(&members(pool));
    let Some(limit) = selected(pool, lambda.get()).and_then(|i| costs[i]) else {
        return vec![false; pool.len()];
    };
    costs
        .iter()
        .map(|c| c.is_some_and(|c| c.c <= limit.c))
        .collect()
}

/// The selection report of the new frontiers `kept`, each task's in
/// `tasks`'s order.
fn selection_report(tasks: &[TaskSpec], kept: &[Vec<Pooled>], lambda: Lambda) -> SelectionReport {
    let mut report = SelectionReport {
        lambda,
        tasks: BTreeMap::new(),
        total_cost: 0,
        baseline_total_cost: 0,
        changed: 0,
    };
    for (task, pool) in tasks.iter().zip(kept) {
        let costs = member_costs(&members(pool));
        let selected = selected(pool, lambda.get());
        let baseline = selected_at_zero(pool);
        let c =
            |i: Option<usize>| -> u64 { i.and_then(|i| costs[i]).map_or(0, |m| u64::from(m.c)) };
        report.total_cost += c(selected);
        report.baseline_total_cost += c(baseline);
        if selected != baseline {
            report.changed += 1;
        }
        let members = pool
            .iter()
            .zip(&costs)
            .map(|(p, m)| MemberScore {
                dl: p.cost.as_ref().ok().copied(),
                c: m.map(|m| m.c),
                unfitted: m.and_then(|m| m.unfitted),
            })
            .collect();
        report.tasks.insert(
            task.name.clone(),
            TaskSelection {
                selected,
                baseline,
                members,
            },
        );
    }
    report
}

/// The position λ = 0 selects in `pool`.
fn selected_at_zero(pool: &[Pooled]) -> Option<usize> {
    selected(pool, 0.0)
}

/// Builds each task's new frontier from its members and candidates (step 2 of
/// the crate docs). Returns the kept programs of each task, in task order, and
/// the fate of every program.
fn select(
    arena: &mut Arena,
    library: &mut Library<'_>,
    names: &LibNames,
    input: IterationInput<'_>,
    frontier_size: usize,
    lambda: Lambda,
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
        // Each program in order: its source, and its program and expansion or
        // why it is invalid.
        let mut checked: Vec<(Source, Checked)> = Vec::new();
        for (index, text) in members.iter().enumerate() {
            let source = Source::Member {
                task: task.name.clone(),
                index,
            };
            let c = library
                .check(arena, text, &task.signature)
                .map_err(|reason| DriverError::Member {
                    task: task.name.clone(),
                    index,
                    reason,
                })?;
            checked.push((source, Ok(c)));
        }
        for &index in &candidates[t] {
            let c = library.check(arena, &input.candidates[index].program, &task.signature);
            checked.push((Source::Candidate { index }, c));
        }
        // Of the programs with one expansion, the cheapest is pooled, and the
        // earliest of those if several cost the same. The others are its
        // duplicates.
        let costs: Vec<Option<Result<u64, String>>> = checked
            .iter()
            .map(|(_, c)| {
                c.as_ref().ok().map(|&(program, _)| {
                    stitch_cost(arena, program, names).map_err(|e| e.to_string())
                })
            })
            .collect();
        let key = |i: usize| match &costs[i] {
            Some(Ok(cost)) => *cost,
            _ => u64::MAX,
        };
        let mut best: HashMap<TermId, usize> = HashMap::new();
        for (i, (_, c)) in checked.iter().enumerate() {
            if let Ok((_, expanded)) = c {
                let b = best.entry(*expanded).or_insert(i);
                if key(i) < key(*b) {
                    *b = i;
                }
            }
        }
        let mut entries: Vec<(Source, Option<Fate>)> = Vec::new();
        let mut pool: Vec<Pooled> = Vec::new();
        for (i, ((source, c), cost)) in checked.iter().zip(costs).enumerate() {
            let fate = match (c, cost) {
                (Err(reason), _) => Some(Fate::Invalid {
                    reason: reason.clone(),
                }),
                (Ok((_, expanded)), _) if best[expanded] != i => Some(Fate::Duplicate {
                    of: checked[best[expanded]].0.clone(),
                }),
                (Ok((program, expanded)), cost) => {
                    pool.push(Pooled {
                        source: source.clone(),
                        program: *program,
                        expanded: *expanded,
                        cost: cost.unwrap_or_else(|| Err(String::new())),
                        growth: input.growth.map_or(Growth::NotMeasured, |g| g.of(source)),
                    });
                    None
                }
            };
            entries.push((source.clone(), fate));
        }
        // Among programs of equal score, the earlier stays first.
        order(&mut pool, lambda);
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
    /// Compression is off, so Stitch was not asked.
    Off,
    /// No program was left to give Stitch.
    NoPrograms,
    /// The run failed, but not because it was interrupted: see
    /// [`is_interruption`].
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

/// Runs Stitch on the kept programs that [`given_to_stitch`] picks under
/// `lambda`, leaving out each program the bridge
/// refuses before Stitch runs (step 3 of the crate docs). Programs that cannot
/// be encoded are left out first; the bridge's other refusals are found one
/// run at a time. Adds the programs left out to `left_out`.
#[allow(clippy::too_many_arguments)]
fn run_stitch(
    arena: &mut Arena,
    library: &Library<'_>,
    tasks: &[TaskSpec],
    kept: &[Vec<Pooled>],
    lambda: Lambda,
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
        let given = given_to_stitch(pool, lambda);
        for (p, pooled) in pool.iter().enumerate() {
            if !given[p] {
                continue;
            }
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
        match stitch.run(arena, library.prelude, &library.defs, &programs, cfg) {
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

/// Whether an abstraction with no parameters and this body is a value: a
/// lambda, a literal, a primitive or a library reference. Any other body is
/// evaluated again at each use, because a use means its definition written
/// out (D1). That is a little work for a partial application such as
/// `(add 1)`, and can be a great deal for one that builds a table; the rule
/// does not tell them apart.
fn is_value(arena: &Arena, body: TermId) -> bool {
    matches!(
        arena.get(body),
        Term::Lam(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) | Term::Lib(_)
    )
}

/// `root` with each reference to a new abstraction `old_len + i` replaced by
/// `subst[i]`. The replacements are closed, so no index is shifted. Walks the
/// shared term graph with a work list, each node once.
fn substitute_libs(
    arena: &mut Arena,
    root: TermId,
    old_len: usize,
    subst: &[TermId],
    memo: &mut HashMap<TermId, TermId>,
) -> TermId {
    let mut stack = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        let term = arena.get(id);
        if !ready {
            stack.push((id, true));
            for c in children(&term) {
                if !memo.contains_key(&c) {
                    stack.push((c, false));
                }
            }
            continue;
        }
        let m = |c: TermId| memo[&c];
        let out = match term {
            Term::Lib(LibId(l)) => {
                let l = l as usize;
                match l.checked_sub(old_len).and_then(|i| subst.get(i)) {
                    Some(&t) => t,
                    None => id,
                }
            }
            Term::Lam(b) => {
                let b = m(b);
                arena.intern(Term::Lam(b))
            }
            Term::App(f, a) => {
                let t = Term::App(m(f), m(a));
                arena.intern(t)
            }
            Term::Let(b, e) => {
                let t = Term::Let(m(b), m(e));
                arena.intern(t)
            }
            Term::If(c, t, e) => {
                let t = Term::If(m(c), m(t), m(e));
                arena.intern(t)
            }
            Term::And(a, b) => {
                let t = Term::And(m(a), m(b));
                arena.intern(t)
            }
            Term::Or(a, b) => {
                let t = Term::Or(m(a), m(b));
                arena.intern(t)
            }
            Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) => id,
        };
        memo.insert(id, out);
    }
    memo[&root]
}

/// The first part of step 4 of the crate docs: each new abstraction with no
/// parameters whose body is not a value ([`is_value`]) is written out where
/// the later abstractions and the rewritten programs use it, and the
/// abstractions after it are numbered down. Returns the compression changed
/// so, and the abstractions written out.
fn write_out_work(
    arena: &mut Arena,
    old_len: usize,
    c: &Compression,
) -> Result<(Compression, Vec<WrittenOut>, Vec<usize>), DriverError> {
    let mut subst: Vec<TermId> = Vec::with_capacity(c.new_libs.len());
    let mut new_libs = Vec::with_capacity(c.new_libs.len());
    let mut written_out = Vec::new();
    let mut origin = Vec::with_capacity(c.new_libs.len());
    for (i, def) in c.new_libs.iter().enumerate() {
        // The body refers only to earlier abstractions, which `subst` covers.
        let mut memo = HashMap::new();
        let body = substitute_libs(arena, def.term, old_len, &subst, &mut memo);
        if def.arity == 0 && !is_value(arena, body) {
            // A runner that skips the bridge's checks could give a body too
            // large to print; the report then has no term, as for `Untyped`.
            written_out.push(WrittenOut {
                abstraction: i,
                term: sexpr::print(arena, def.term).unwrap_or_default(),
                bucketing: false,
            });
            subst.push(body);
        } else {
            let id = u32::try_from(old_len + new_libs.len()).map_err(|_| DriverError::TooLarge)?;
            subst.push(arena.intern(Term::Lib(LibId(id))));
            origin.push(i);
            new_libs.push(LibDef {
                arity: def.arity,
                term: body,
            });
        }
    }
    if written_out.is_empty() {
        return Ok((c.clone(), written_out, origin));
    }
    let mut memo = HashMap::new();
    let rewritten = c
        .rewritten
        .iter()
        .map(|&p| substitute_libs(arena, p, old_len, &subst, &mut memo))
        .collect();
    Ok((
        Compression {
            new_libs,
            rewritten,
            names: c.names.clone(),
            stitch_steps: c.stitch_steps,
            rejected: c.rejected.clone(),
        },
        written_out,
        origin,
    ))
}

/// The integer literals and the library references in the terms `roots`.
fn ints_and_libs(arena: &Arena, roots: &[TermId]) -> (HashSet<i64>, Vec<(TermId, LibId)>) {
    let mut ints = HashSet::new();
    let mut libs = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = roots.to_vec();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let term = arena.get(id);
        match term {
            Term::Int(n) => {
                ints.insert(n);
            }
            Term::Lib(l) => libs.push((id, l)),
            _ => stack.extend(children(&term)),
        }
    }
    libs.sort_by_key(|&(_, l)| l);
    (ints, libs)
}

/// `root` with each node that is a key of `map` replaced by its value. The
/// values are closed, so no index is shifted. Walks the shared term graph
/// with a work list, each node once. `memo` holds the nodes already done, and
/// can be given again for another root with the same `map`.
pub fn replace_nodes(
    arena: &mut Arena,
    root: TermId,
    map: &HashMap<TermId, TermId>,
    memo: &mut HashMap<TermId, TermId>,
) -> TermId {
    let mut stack = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        if let Some(&to) = map.get(&id) {
            memo.insert(id, to);
            continue;
        }
        let term = arena.get(id);
        if !ready {
            stack.push((id, true));
            for c in children(&term) {
                if !memo.contains_key(&c) {
                    stack.push((c, false));
                }
            }
            continue;
        }
        let m = |c: TermId| memo[&c];
        let out = match term {
            Term::Lam(b) => Term::Lam(m(b)),
            Term::App(f, a) => Term::App(m(f), m(a)),
            Term::Let(b, e) => Term::Let(m(b), m(e)),
            Term::If(c, t, e) => Term::If(m(c), m(t), m(e)),
            Term::And(a, b) => Term::And(m(a), m(b)),
            Term::Or(a, b) => Term::Or(m(a), m(b)),
            Term::Var(_)
            | Term::Int(_)
            | Term::Bool(_)
            | Term::Unit
            | Term::Prim(_)
            | Term::Lib(_) => term,
        };
        let out = arena.intern(out);
        memo.insert(id, out);
    }
    memo[&root]
}

/// The second part of step 4 of the crate docs, for a step that refuses
/// bucketing abstractions: each new abstraction of `c` that is bucketing is
/// written out where the other abstractions and the rewritten programs use
/// it, and the abstractions after it are numbered down. `old` is the library,
/// and `origin[j]` is the index Stitch gave `c`'s abstraction `j`, which
/// `found` holds as Stitch found it. Returns the compression changed so, and
/// the abstractions written out; or why the refused abstractions do not
/// expand.
///
/// A use is written out by expansion (D1), and the expander replaces every
/// library reference it meets. So each reference that stays is first put
/// aside as an integer literal that occurs nowhere in the terms, and put
/// back after.
fn write_out_bucketing(
    arena: &mut Arena,
    old: &[LibDef],
    c: &Compression,
    origin: &[usize],
    found: &[LibDef],
) -> Result<Result<(Compression, Vec<WrittenOut>), String>, DriverError> {
    let old_len = old.len();
    let mut flags: Vec<bool> = Vec::with_capacity(old_len + c.new_libs.len());
    for def in old.iter().chain(&c.new_libs) {
        // An entry refers only to earlier ones, which `flags` covers. A
        // prelude function has an id past every entry, and is not bucketing.
        let flag = bucketing::is_bucketing(arena, def.term, |l| {
            flags.get(l as usize).copied().unwrap_or(false)
        });
        flags.push(flag);
    }
    let refused = &flags[old_len..];
    if !refused.contains(&true) {
        return Ok(Ok((c.clone(), Vec::new())));
    }

    let roots: Vec<TermId> = c
        .new_libs
        .iter()
        .map(|d| d.term)
        .chain(c.rewritten.iter().copied())
        .collect();
    let (ints, libs) = ints_and_libs(arena, &roots);
    // `aside` sends a reference that stays to its literal, and a reference to
    // a refused abstraction to that abstraction's place among the refused.
    // `back` sends each literal to the reference under the new numbering.
    let mut aside = HashMap::new();
    let mut back = HashMap::new();
    let mut fresh = (i64::MIN..).filter(|n| !ints.contains(n));
    for (node, LibId(l)) in libs {
        let new = (l as usize)
            .checked_sub(old_len)
            .filter(|&j| j < refused.len());
        let (to, lib) = match new {
            Some(j) if refused[j] => {
                let rank = refused[..j].iter().filter(|&&r| r).count();
                let rank = u32::try_from(rank).map_err(|_| DriverError::LibraryTooLarge)?;
                (arena.intern(Term::Lib(LibId(rank))), None)
            }
            Some(j) => {
                let before = refused[..j].iter().filter(|&&r| r).count();
                let id = u32::try_from(old_len + j - before)
                    .map_err(|_| DriverError::LibraryTooLarge)?;
                let n = fresh.next().ok_or(DriverError::LibraryTooLarge)?;
                (arena.int(n), Some(LibId(id)))
            }
            None => {
                let n = fresh.next().ok_or(DriverError::LibraryTooLarge)?;
                (arena.int(n), Some(LibId(l)))
            }
        };
        aside.insert(node, to);
        if let Some(lib) = lib {
            let lib = arena.intern(Term::Lib(lib));
            back.insert(to, lib);
        }
    }

    let mut aside_memo = HashMap::new();
    let mut put = |arena: &mut Arena, t: TermId| replace_nodes(arena, t, &aside, &mut aside_memo);
    let refused_defs: Vec<LibDef> = c
        .new_libs
        .iter()
        .zip(refused)
        .filter(|&(_, &r)| r)
        .map(|(def, _)| LibDef {
            arity: def.arity,
            term: put(arena, def.term),
        })
        .collect();
    let mut expander = match Expander::new(arena, &refused_defs) {
        Ok(x) => x,
        Err(e) => return Ok(Err(format!("the refused abstractions do not expand: {e}"))),
    };
    let mut back_memo = HashMap::new();
    let mut write_out = |arena: &mut Arena, t: TermId| -> Result<TermId, String> {
        let t = put(arena, t);
        let t = expander
            .expand(arena, t)
            .map_err(|e| format!("a use of a refused abstraction does not expand: {e}"))?;
        Ok(replace_nodes(arena, t, &back, &mut back_memo))
    };

    let mut new_libs = Vec::new();
    let mut written_out = Vec::new();
    for ((def, &r), &i) in c.new_libs.iter().zip(refused).zip(origin) {
        if r {
            written_out.push(WrittenOut {
                abstraction: i,
                term: sexpr::print(arena, found[i].term).unwrap_or_default(),
                bucketing: true,
            });
        } else {
            match write_out(arena, def.term) {
                Ok(term) => new_libs.push(LibDef {
                    arity: def.arity,
                    term,
                }),
                Err(e) => return Ok(Err(e)),
            }
        }
    }
    let mut rewritten = Vec::with_capacity(c.rewritten.len());
    for &p in &c.rewritten {
        match write_out(arena, p) {
            Ok(t) => rewritten.push(t),
            Err(e) => return Ok(Err(e)),
        }
    }
    Ok(Ok((
        Compression {
            new_libs,
            rewritten,
            names: c.names.clone(),
            stitch_steps: c.stitch_steps,
            rejected: c.rejected.clone(),
        },
        written_out,
    )))
}

/// What a Stitch run found.
struct Found<'a> {
    /// For each program given to Stitch, its task and its place in the
    /// task's kept programs.
    slots: &'a [(usize, usize)],
    /// The abstractions and the rewritten programs, one for each slot.
    c: &'a Compression,
}

/// Step 4 of the crate docs: types the new abstractions, accepts each
/// rewriting that types and keeps its program's expansion, and cuts the
/// abstractions that are not used. Changes the accepted programs in `kept`
/// and returns the new library entries and the report.
fn update_library(
    arena: &mut Arena,
    library: &Library<'_>,
    tasks: &[TaskSpec],
    kept: &mut [Vec<Pooled>],
    found: Found<'_>,
    left_out: Vec<LeftOut>,
    refuse_bucketing: bool,
) -> Result<(Vec<LibraryEntry>, CompressionReport), DriverError> {
    let Found { slots, c } = found;
    let old_len = library.defs.len();
    let found = c.new_libs.len();
    let (mut written, mut written_out, origin) = write_out_work(arena, old_len, c)?;
    if refuse_bucketing {
        match write_out_bucketing(arena, &library.defs, &written, &origin, &c.new_libs)? {
            Ok((without, refused)) => {
                written = without;
                written_out.extend(refused);
                written_out.sort_by_key(|w| w.abstraction);
            }
            Err(error) => return Ok((Vec::new(), CompressionReport::Failed { left_out, error })),
        }
    }
    let c = &written;
    let mut schemes = library.schemes.clone();
    let mut untyped = None;
    for (i, def) in c.new_libs.iter().enumerate() {
        // An abstraction can be a few lambdas larger than the programs it was
        // found in, so it may not print.
        if arena.tree_size(def.term, MAX_PRINT_NODES).is_none() {
            untyped = Some(Untyped {
                abstraction: i,
                term: String::new(),
                error: format!("it has more than {MAX_PRINT_NODES} nodes written out"),
            });
            break;
        }
        match infer_in(arena, library.prelude, def.term, &schemes) {
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
    let prelude = library.prelude;
    let typed_libs: Vec<LibDef> = library
        .defs
        .iter()
        .chain(&c.new_libs[..typed])
        .map(|d| LibDef {
            arity: d.arity,
            term: expand_prelude(prelude, arena, d.term),
        })
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
        let uses = max_lib(arena, prelude, rewritten);
        let text = sexpr::print(arena, rewritten);
        let refused = if text.is_err() {
            Some(format!(
                "it has more than {MAX_PRINT_NODES} nodes written out"
            ))
        } else if uses.is_some_and(|l| l >= schemes.len()) {
            Some("it uses a new abstraction that does not type, or one after it".to_owned())
        } else if let Err(e) = check_in(arena, prelude, rewritten, &schemes, &task.signature) {
            Some(e.to_string())
        } else {
            let bare = expand_prelude(prelude, arena, rewritten);
            match expander.expand(arena, bare) {
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
        found,
        untyped,
        unused: typed - kept_new,
        kept: kept_new,
        fallbacks,
        written_out,
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
    let mut library = Library::load(&mut arena, input.prelude, input.library)?;
    let old_len = library.defs.len();
    let old_names = names(old_len, input.prelude)?;
    let (mut kept, programs) = select(
        &mut arena,
        &mut library,
        &old_names,
        input,
        cfg.frontier_size,
        cfg.lambda,
    )?;

    let mut left_out = Vec::new();
    let run = match &cfg.compress {
        None => StitchRun::Off,
        Some(compress) => run_stitch(
            &mut arena,
            &library,
            input.tasks,
            &kept,
            cfg.lambda,
            compress,
            stitch,
            &mut left_out,
        )?,
    };
    let (new_library, compression) = match run {
        StitchRun::Off => (Vec::new(), CompressionReport::Off),
        StitchRun::NoPrograms => (Vec::new(), CompressionReport::NoPrograms { left_out }),
        StitchRun::Failed(error) => (Vec::new(), CompressionReport::Failed { left_out, error }),
        StitchRun::Ran { slots, compression } => update_library(
            &mut arena,
            &library,
            input.tasks,
            &mut kept,
            Found {
                slots: &slots,
                c: &compression,
            },
            left_out,
            cfg.refuse_bucketing,
        )?,
    };

    // Sort each frontier again under the new library.
    let new_names = names(old_len + new_library.len(), input.prelude)?;
    let mut frontiers = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for (task, pool) in input.tasks.iter().zip(&mut kept) {
        for p in pool.iter_mut() {
            p.cost = stitch_cost(&arena, p.program, &new_names).map_err(|e| e.to_string());
        }
        order(pool, cfg.lambda);
        let texts = pool
            .iter()
            .map(|p| print(&arena, p.program))
            .collect::<Result<Vec<String>, DriverError>>()?;
        frontiers.insert(task.name.clone(), texts);
        sources.insert(
            task.name.clone(),
            pool.iter().map(|p| p.source.clone()).collect(),
        );
    }
    let selection = selection_report(input.tasks, &kept, cfg.lambda);
    Ok(IterationOutcome {
        new_library,
        frontiers,
        sources,
        report: IterationReport {
            programs,
            compression,
            selection: Some(selection),
        },
    })
}
