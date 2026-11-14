//! Library learning for dc: runs [Stitch](https://github.com/mlb2251/stitch) on
//! dc programs and decodes the abstractions it finds into library definitions.
//!
//! [`compress`] encodes each program with `dc-stitch`, runs Stitch's
//! multistep compression, and decodes the result with [`accept_steps`]: each
//! abstraction becomes a [`LibDef`], and the rewritten programs become core terms
//! that refer to it.
//!
//! # Library ids and names
//!
//! A library is a slice of [`LibDef`]s indexed by [`LibId`], as
//! [`dc_core::expand()`] takes it. The bridge names library `l` as
//! `fn_{l}` ([`ABSTRACTION_PREFIX`]), and tells Stitch how many abstractions the
//! library already has. Stitch names its `i`-th new abstraction
//! `fn_{len + i}` (`previous_abstractions + i`, Stitch's `compression.rs`), so
//! each new abstraction gets the id `LibId(len + i)` and keeps it in later
//! iterations, when it is part of the library passed in. [`accept_steps`] checks
//! each name Stitch reports against that rule.
//!
//! # Stitch flags
//!
//! Stitch can cut an abstraction or a parameter through a special form or the
//! binder of a let, which leaves a body or a rewritten program with no core term
//! (see the `dc-stitch` crate docs). [`stitch_config`] sets the three flags meant
//! to rule this out: `--fused-lambda-tags 0`, `--no-curried-bodies` and
//! `--no-curried-metavars`. It runs Stitch on one thread, without shuffling, so
//! that the same programs give the same result.
//!
//! It also sets `--no-mismatch-check`. Stitch otherwise asserts that the
//! rewritten programs cost what it estimated, and panics when the estimate is
//! too low. That was seen when an abstraction uses a parameter more than once
//! and is used inside its own argument. So a step can compress less than Stitch
//! reported. The rewritten programs are still checked:
//! see the next section.
//!
//! # Steps that are dropped
//!
//! Each Stitch step adds one abstraction and rewrites the programs with it, and
//! later steps can use earlier abstractions. [`accept_steps`] keeps a step when
//! its abstraction and its rewritten programs decode, and each rewritten program
//! expands to the term its input program expands to. At the first step that
//! fails, it keeps the steps before it, drops it and every later step, and
//! reports why in [`Compression::rejected`].
//!
//! # Running Stitch in a worker process
//!
//! Stitch's time and memory are not bounded: they grow quickly with the size of
//! the programs, and two copies of a list literal of 1,000 elements already take
//! about 40 seconds. Stitch also panics on input it cannot handle, and an
//! allocation failure or a stack overflow in it aborts the process.
//! [`compress_isolated`] runs Stitch in a child process with a time and a memory
//! limit (see the [`worker`] module), so none of this reaches the caller.
//! [`compress`] runs Stitch in the caller's process, with no limits.

pub mod worker;

use std::collections::HashMap;
use std::fmt;

use dc_core::term::children;
use dc_core::{Arena, ExpandError, Expander, LibDef, LibId, Term, TermId};
use dc_stitch::{
    decode, decode_abstraction, expr, to_expr, DecodeError, EncodeError, Expr, LibNames, NameError,
    LET_LAMBDA_TAG,
};
use serde::{Deserialize, Serialize};
use stitch_core::{multistep_compression, MultistepCompressionConfig};
use worker::{StitchRequest, Worker, WorkerError};

/// The prefix of every library name the bridge uses: library `l` is `fn_{l}`.
pub const ABSTRACTION_PREFIX: &str = "fn_";

/// The most levels a program's Stitch expression may have, where a leaf is one
/// level and a lambda or an application is one more than its deepest child.
///
/// Stitch recurses once for each level, so a deeper program could overflow its
/// stack, which aborts the process. [`compress`] refuses such a program.
pub const MAX_STITCH_DEPTH: usize = 4096;

/// The stack of the thread Stitch runs on. Divided by [`MAX_STITCH_DEPTH`], it
/// is 64 KiB for each level. No test fixes this size: a program deep enough to
/// need it takes Stitch minutes.
const STITCH_STACK_BYTES: usize = 256 << 20;

/// A program to compress, with the task it solves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontierProgram {
    /// The task label. For each task, Stitch counts only the cheapest of its
    /// programs after rewriting.
    pub task: String,
    /// A closed core term, which may refer to the library.
    pub program: TermId,
}

/// The settings of one compression run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressConfig {
    /// The largest number of abstractions to find. Stitch stops early when no
    /// abstraction compresses the programs.
    pub iterations: usize,
    /// The largest arity of an abstraction.
    pub max_arity: u32,
}

impl Default for CompressConfig {
    /// Stitch's own defaults: 3 iterations and arity at most 2.
    fn default() -> CompressConfig {
        CompressConfig {
            iterations: 3,
            max_arity: 2,
        }
    }
}

/// One step of Stitch's output, as text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StitchStep {
    /// The abstraction's name.
    pub name: String,
    /// The abstraction's arity.
    pub arity: usize,
    /// The abstraction's body, with parameters `#0 ... #(arity - 1)`.
    pub body: String,
    /// Every program, rewritten with this abstraction and all earlier ones.
    pub rewritten: Vec<String>,
}

/// The result of a compression run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compression {
    /// The new abstractions, in order: the `i`-th has the id `LibId(len + i)`,
    /// where `len` is the length of the library passed in.
    pub new_libs: Vec<LibDef>,
    /// The programs, in input order, rewritten with the new abstractions.
    pub rewritten: Vec<TermId>,
    /// The names of the old and the new abstractions.
    pub names: LibNames,
    /// The number of steps Stitch returned, accepted or not.
    pub stitch_steps: usize,
    /// The first step that was dropped, if any. The steps after it were dropped
    /// too.
    pub rejected: Option<Rejection>,
}

/// A Stitch step that was dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    /// The step's index in Stitch's output.
    pub step: usize,
    /// The name Stitch gave the abstraction.
    pub name: String,
    /// The abstraction's body, as Stitch wrote it.
    pub body: String,
    /// Why the step was dropped.
    pub reason: RejectReason,
}

/// Why a Stitch step was dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Stitch named the abstraction differently from the bridge's rule.
    UnexpectedName {
        /// The name the bridge expected, or `None` if the step's id would not fit
        /// in a [`LibId`].
        expected: Option<String>,
    },
    /// The arity is above [`CompressConfig::max_arity`].
    ArityTooLarge {
        /// The abstraction's arity.
        arity: usize,
    },
    /// The abstraction's body does not decode.
    Abstraction(DecodeError),
    /// The step does not rewrite exactly the input programs.
    ProgramCount {
        /// The number of rewritten programs in the step.
        got: usize,
        /// The number of input programs.
        expected: usize,
    },
    /// A rewritten program does not decode.
    Program {
        /// The program's index in the input.
        index: usize,
        /// The decoding error.
        error: DecodeError,
    },
    /// The library does not expand with the step's abstraction added. A body
    /// that decodes is closed and refers only to earlier abstractions, so this
    /// is not expected.
    Library(ExpandError),
    /// A rewritten program does not expand. A program that decodes refers only
    /// to named abstractions, so this is not expected.
    Expand {
        /// The program's index in the input.
        index: usize,
        /// The expansion error.
        error: ExpandError,
    },
    /// A rewritten program does not expand to what its input program expands to.
    Changed {
        /// The program's index in the input.
        index: usize,
    },
}

impl fmt::Display for RejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RejectReason::UnexpectedName {
                expected: Some(name),
            } => write!(f, "expected the name {name}"),
            RejectReason::UnexpectedName { expected: None } => {
                f.write_str("the abstraction's id does not fit in a library id")
            }
            RejectReason::ArityTooLarge { arity } => write!(f, "arity {arity} is too large"),
            RejectReason::Abstraction(e) => write!(f, "the body does not decode: {e}"),
            RejectReason::ProgramCount { got, expected } => {
                write!(f, "{got} rewritten programs for {expected} inputs")
            }
            RejectReason::Program { index, error } => {
                write!(f, "rewritten program {index} does not decode: {error}")
            }
            RejectReason::Library(e) => {
                write!(f, "the library does not expand with the abstraction: {e}")
            }
            RejectReason::Expand { index, error } => {
                write!(f, "rewritten program {index} does not expand: {error}")
            }
            RejectReason::Changed { index } => write!(
                f,
                "rewritten program {index} does not expand to its input program"
            ),
        }
    }
}

/// Why a compression run could not start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompressError {
    /// The library has more entries than a [`LibId`] can number.
    LibraryTooLarge,
    /// The library names could not be built.
    Names(NameError),
    /// A program has free variables.
    OpenProgram {
        /// The program's index in the input.
        index: usize,
    },
    /// A program could not be encoded.
    Encode {
        /// The program's index in the input.
        index: usize,
        /// The encoding error.
        error: EncodeError,
    },
    /// A program's Stitch expression has more than [`MAX_STITCH_DEPTH`] levels.
    TooDeep {
        /// The program's index in the input.
        index: usize,
    },
    /// The thread that runs Stitch could not be started.
    StitchThread(std::io::ErrorKind),
    /// The Stitch worker process gave no steps: it was stopped at a limit, it
    /// failed, or its output does not parse.
    Worker(WorkerError),
    /// The library does not expand.
    Library(ExpandError),
    /// An input program does not expand under the library.
    Expand {
        /// The program's index in the input.
        index: usize,
        /// The expansion error.
        error: ExpandError,
    },
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressError::LibraryTooLarge => f.write_str("the library has too many entries"),
            CompressError::Names(e) => write!(f, "library names: {e}"),
            CompressError::OpenProgram { index } => write!(f, "program {index} is not closed"),
            CompressError::Encode { index, error } => {
                write!(f, "program {index} cannot be encoded: {error}")
            }
            CompressError::TooDeep { index } => write!(
                f,
                "program {index} has more than {MAX_STITCH_DEPTH} levels, too deep for Stitch"
            ),
            CompressError::StitchThread(kind) => {
                write!(f, "the Stitch thread could not be started: {kind}")
            }
            CompressError::Worker(e) => e.fmt(f),
            CompressError::Library(e) => write!(f, "the library does not expand: {e}"),
            CompressError::Expand { index, error } => {
                write!(f, "program {index} does not expand: {error}")
            }
        }
    }
}

impl std::error::Error for CompressError {}

impl std::error::Error for RejectReason {}

/// The Stitch configuration for `cfg`, given a library of `library_len` entries.
pub fn stitch_config(cfg: &CompressConfig, library_len: usize) -> MultistepCompressionConfig {
    let mut s = MultistepCompressionConfig::new();
    s.iterations = cfg.iterations;
    s.abstraction_prefix = ABSTRACTION_PREFIX.to_string();
    s.previous_abstractions = library_len;
    s.shuffle = false;
    s.truncate = None;
    s.silent = true;
    s.step.max_arity = cfg.max_arity as usize;
    s.step.threads = 1;
    s.step.fused_lambda_tags = LET_LAMBDA_TAG
        .to_string()
        .parse()
        .expect("Stitch parses a single tag");
    s.step.no_curried_bodies = true;
    s.step.no_curried_metavars = true;
    s.step.eta_long = false;
    // Stitch asserts that the rewritten programs cost what it estimated. Its
    // estimate can be too low when an abstraction uses a parameter more than
    // once and is used inside its own argument, and the assertion then panics. The rewriting is still right, and
    // `accept_steps` checks it by expansion.
    s.step.no_mismatch_check = true;
    s
}

/// The names of a library of `len` entries.
fn library_names(len: u32) -> Result<LibNames, CompressError> {
    LibNames::sequential(ABSTRACTION_PREFIX, len).map_err(CompressError::Names)
}

/// The number of levels of `e`: 1 for a leaf, and one more than its deepest
/// child for a lambda or an application. Found with a work list.
fn depth(e: &Expr) -> usize {
    let mut deepest = 0;
    let mut stack = vec![(e, 1usize)];
    while let Some((e, d)) = stack.pop() {
        deepest = deepest.max(d);
        match e {
            Expr::Lam(b) | Expr::TaggedLam(_, b) => stack.push((b, d + 1)),
            Expr::App(f, x) => {
                stack.push((f, d + 1));
                stack.push((x, d + 1));
            }
            Expr::Var(_) | Expr::IVar(_) | Expr::Sym(_) => {}
        }
    }
    deepest
}

/// Runs Stitch on `text` on its own thread, with a stack of
/// [`STITCH_STACK_BYTES`], and returns its steps as text. A panic in Stitch
/// carries on in the caller.
fn run_stitch(
    text: &[String],
    tasks: Vec<String>,
    cfg: &CompressConfig,
    library_len: usize,
) -> Result<Vec<StitchStep>, CompressError> {
    std::thread::scope(|scope| {
        let stitch = std::thread::Builder::new()
            .name("stitch".to_string())
            .stack_size(STITCH_STACK_BYTES)
            .spawn_scoped(scope, move || {
                // Stitch builds its configuration with an argument parser, which
                // also needs more stack than a small caller has.
                let cfg = stitch_config(cfg, library_len);
                let (results, _json) =
                    multistep_compression(text, Some(tasks), None, None, None, &cfg);
                results
                    .iter()
                    .map(|r| StitchStep {
                        name: r.inv.name.clone(),
                        arity: r.inv.arity,
                        body: r.inv.body.to_string(),
                        rewritten: r.rewritten.iter().map(|p| p.to_string()).collect(),
                    })
                    .collect::<Vec<StitchStep>>()
            })
            .map_err(|e| CompressError::StitchThread(e.kind()))?;
        match stitch.join() {
            Ok(steps) => Ok(steps),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// Runs Stitch on `programs`, which may refer to `libs`, in the caller's
/// process, and decodes the result.
///
/// The new abstractions extend `libs`; see [`Compression`]. Stitch is given no
/// weights. A program with free variables gives [`CompressError::OpenProgram`],
/// and one whose Stitch expression has more than [`MAX_STITCH_DEPTH`] levels
/// gives [`CompressError::TooDeep`].
///
/// Stitch runs on its own thread, so it does not use the caller's stack. Its
/// time and memory are not bounded, and grow quickly with the length of a long
/// list literal. Stitch panics on input it cannot handle, and this function
/// does not catch the panic. [`compress_isolated`] bounds and isolates it.
///
/// The arena gains the new abstractions and the rewritten programs only: the
/// steps are checked in a scratch arena, as [`accept_steps`] describes.
pub fn compress(
    arena: &mut Arena,
    libs: &[LibDef],
    programs: &[FrontierProgram],
    cfg: &CompressConfig,
) -> Result<Compression, CompressError> {
    compress_via(arena, libs, programs, cfg, |text, tasks| {
        run_stitch(&text, tasks, cfg, libs.len())
    })
}

/// [`compress`], with Stitch run by `worker` in a child process with the
/// worker's [`Limits`](worker::Limits).
///
/// A run stopped at a limit, a panic or abort in Stitch, and output that does
/// not parse give [`CompressError::Worker`]. The worker's output is checked
/// with [`accept_steps`], as in [`compress`]. The same programs give the same
/// result as [`compress`].
pub fn compress_isolated(
    arena: &mut Arena,
    libs: &[LibDef],
    programs: &[FrontierProgram],
    cfg: &CompressConfig,
    worker: &Worker,
) -> Result<Compression, CompressError> {
    compress_via(arena, libs, programs, cfg, |text, tasks| {
        let request = StitchRequest {
            programs: text,
            tasks,
            iterations: cfg.iterations,
            max_arity: cfg.max_arity,
            library_len: libs.len(),
        };
        worker.run(&request).map_err(CompressError::Worker)
    })
}

/// [`compress`], with Stitch run by `stitch` on the encoded programs and their
/// task labels.
fn compress_via(
    arena: &mut Arena,
    libs: &[LibDef],
    programs: &[FrontierProgram],
    cfg: &CompressConfig,
    stitch: impl FnOnce(Vec<String>, Vec<String>) -> Result<Vec<StitchStep>, CompressError>,
) -> Result<Compression, CompressError> {
    let len = u32::try_from(libs.len()).map_err(|_| CompressError::LibraryTooLarge)?;
    let names = library_names(len)?;
    let mut text = Vec::with_capacity(programs.len());
    for (index, p) in programs.iter().enumerate() {
        if !arena.is_closed(p.program) {
            return Err(CompressError::OpenProgram { index });
        }
        let e = to_expr(arena, p.program, &names)
            .map_err(|error| CompressError::Encode { index, error })?;
        if depth(&e) > MAX_STITCH_DEPTH {
            return Err(CompressError::TooDeep { index });
        }
        text.push(expr::print(&e));
    }
    let ids: Vec<TermId> = programs.iter().map(|p| p.program).collect();
    // Check the library and the programs before Stitch runs, which can take long.
    let inputs = check_inputs(arena, libs, &ids)?;
    let steps = if programs.is_empty() {
        Vec::new()
    } else {
        let tasks: Vec<String> = programs.iter().map(|p| p.task.clone()).collect();
        stitch(text, tasks)?
    };
    accept(arena, inputs, names, &ids, &steps, cfg.max_arity)
}

/// Decodes Stitch's `steps` for `programs` over the library `libs`. Keeps the
/// steps before the first that fails a check, and drops that step and the rest.
///
/// The library and each input program must expand, whether or not there are
/// steps: if not, the result is [`CompressError::Library`] or
/// [`CompressError::Expand`].
///
/// A step is kept when its abstraction and its rewritten programs decode, and
/// each rewritten program expands, under `libs` and the abstractions kept so
/// far, to the term its input program expands to under `libs`.
///
/// The steps are checked in a scratch arena. `arena` gains only the
/// abstractions that are kept and the programs of the last step that is kept.
///
/// [`compress`] calls this on Stitch's output. It is public so that recorded
/// Stitch output can be decoded again.
pub fn accept_steps(
    arena: &mut Arena,
    libs: &[LibDef],
    programs: &[TermId],
    steps: &[StitchStep],
    max_arity: u32,
) -> Result<Compression, CompressError> {
    let len = u32::try_from(libs.len()).map_err(|_| CompressError::LibraryTooLarge)?;
    let names = library_names(len)?;
    let inputs = check_inputs(arena, libs, programs)?;
    accept(arena, inputs, names, programs, steps, max_arity)
}

/// The library and the input programs of a run, copied into a scratch arena and
/// checked.
struct Inputs {
    /// The arena the steps are checked in, so that the terms of a dropped step
    /// and of the steps before the last do not stay in the caller's arena.
    scratch: Arena,
    /// The library, in `scratch`.
    libs: Vec<LibDef>,
    /// What each input program expands to under the library, in `scratch`.
    want: Vec<TermId>,
}

/// Copies `libs` and `programs` into a scratch arena, and checks that the
/// library and each program expand.
fn check_inputs(
    arena: &Arena,
    libs: &[LibDef],
    programs: &[TermId],
) -> Result<Inputs, CompressError> {
    let mut scratch = Arena::new();
    let mut copied = HashMap::new();
    let libs: Vec<LibDef> = libs
        .iter()
        .map(|def| LibDef {
            arity: def.arity,
            term: copy_term(arena, &mut scratch, def.term, &mut copied),
        })
        .collect();
    let mut expander = Expander::new(&mut scratch, &libs).map_err(CompressError::Library)?;
    let mut want = Vec::with_capacity(programs.len());
    for (index, p) in programs.iter().enumerate() {
        let p = copy_term(arena, &mut scratch, *p, &mut copied);
        let expanded = expander
            .expand(&mut scratch, p)
            .map_err(|error| CompressError::Expand { index, error })?;
        want.push(expanded);
    }
    Ok(Inputs {
        scratch,
        libs,
        want,
    })
}

/// [`accept_steps`], given the checked inputs and the names of the library.
fn accept(
    arena: &mut Arena,
    inputs: Inputs,
    mut names: LibNames,
    programs: &[TermId],
    steps: &[StitchStep],
    max_arity: u32,
) -> Result<Compression, CompressError> {
    let Inputs {
        mut scratch,
        libs: mut all,
        want,
    } = inputs;
    let library_len = all.len();
    let mut accepted = 0;
    let mut rejected = None;
    for (i, step) in steps.iter().enumerate() {
        let next = library_len
            .checked_add(i)
            .and_then(|l| u32::try_from(l).ok())
            .map(LibId);
        match check_step(
            &mut scratch,
            next,
            step,
            &mut names,
            max_arity,
            &mut all,
            &want,
        ) {
            Ok(()) => accepted = i + 1,
            Err(reason) => {
                rejected = Some(Rejection {
                    step: i,
                    name: step.name.clone(),
                    body: step.body.clone(),
                    reason,
                });
                break;
            }
        }
    }
    // Decode what is kept into the caller's arena. It decoded in the scratch
    // arena against the same names.
    let new_libs = steps[..accepted]
        .iter()
        .zip(&all[library_len..])
        .map(|(step, def)| LibDef {
            arity: def.arity,
            term: decode_abstraction(arena, &step.body, def.arity, &names)
                .expect("the body decoded in the scratch arena"),
        })
        .collect();
    let rewritten = match accepted.checked_sub(1) {
        None => programs.to_vec(),
        Some(last) => steps[last]
            .rewritten
            .iter()
            .map(|p| decode(arena, p, &names).expect("the program decoded in the scratch arena"))
            .collect(),
    };
    Ok(Compression {
        new_libs,
        rewritten,
        names,
        stitch_steps: steps.len(),
        rejected,
    })
}

/// Copies the term `id` of `from` into `to`, with a work list. `copied` maps the
/// terms of `from` already copied to their copies.
fn copy_term(
    from: &Arena,
    to: &mut Arena,
    id: TermId,
    copied: &mut HashMap<TermId, TermId>,
) -> TermId {
    let mut stack = vec![id];
    while let Some(&t) = stack.last() {
        if copied.contains_key(&t) {
            stack.pop();
            continue;
        }
        let node = from.get(t);
        let pending: Vec<TermId> = children(&node)
            .into_iter()
            .filter(|c| !copied.contains_key(c))
            .collect();
        if !pending.is_empty() {
            stack.extend(pending);
            continue;
        }
        let c = |child: TermId| copied[&child];
        let copy = match node {
            Term::Lam(b) => Term::Lam(c(b)),
            Term::App(f, x) => Term::App(c(f), c(x)),
            Term::Let(x, b) => Term::Let(c(x), c(b)),
            Term::If(i, t, e) => Term::If(c(i), c(t), c(e)),
            Term::And(x, y) => Term::And(c(x), c(y)),
            Term::Or(x, y) => Term::Or(c(x), c(y)),
            leaf @ (Term::Var(_)
            | Term::Int(_)
            | Term::Bool(_)
            | Term::Unit
            | Term::Prim(_)
            | Term::Lib(_)) => leaf,
        };
        let new = to.intern(copy);
        copied.insert(t, new);
        stack.pop();
    }
    copied[&id]
}

/// Checks one step in the scratch arena. Its abstraction should get the id `id`.
/// `libs` is the library with the abstractions kept so far, and `want` holds
/// what each input program expands to.
///
/// If the step passes, its abstraction is added to `libs` and its name to
/// `names`. If not, both are left as they were.
fn check_step(
    scratch: &mut Arena,
    id: Option<LibId>,
    step: &StitchStep,
    names: &mut LibNames,
    max_arity: u32,
    libs: &mut Vec<LibDef>,
    want: &[TermId],
) -> Result<(), RejectReason> {
    let Some(id) = id else {
        return Err(RejectReason::UnexpectedName { expected: None });
    };
    let name = format!("{ABSTRACTION_PREFIX}{}", id.0);
    if step.name != name {
        return Err(RejectReason::UnexpectedName {
            expected: Some(name),
        });
    }
    let arity = match u32::try_from(step.arity) {
        Ok(a) if a <= max_arity => a,
        _ => return Err(RejectReason::ArityTooLarge { arity: step.arity }),
    };
    // The body may refer to earlier abstractions, but not to itself.
    let term =
        decode_abstraction(scratch, &step.body, arity, names).map_err(RejectReason::Abstraction)?;
    if step.rewritten.len() != want.len() {
        return Err(RejectReason::ProgramCount {
            got: step.rewritten.len(),
            expected: want.len(),
        });
    }
    names
        .insert(id, name)
        .expect("the bridge's names are legal, and the id and name are fresh");
    libs.push(LibDef { arity, term });
    let checked = check_programs(scratch, step, names, libs, want);
    if checked.is_err() {
        names.remove(id);
        libs.pop();
    }
    checked
}

/// Checks that each rewritten program of `step` decodes, and expands under
/// `libs` to the term in `want`.
fn check_programs(
    scratch: &mut Arena,
    step: &StitchStep,
    names: &LibNames,
    libs: &[LibDef],
    want: &[TermId],
) -> Result<(), RejectReason> {
    let mut programs = Vec::with_capacity(want.len());
    for (index, p) in step.rewritten.iter().enumerate() {
        let decoded =
            decode(scratch, p, names).map_err(|error| RejectReason::Program { index, error })?;
        programs.push(decoded);
    }
    let mut expander = Expander::new(scratch, libs).map_err(RejectReason::Library)?;
    for (index, (p, want)) in programs.iter().zip(want).enumerate() {
        let got = expander
            .expand(scratch, *p)
            .map_err(|error| RejectReason::Expand { index, error })?;
        if got != *want {
            return Err(RejectReason::Changed { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dc_core::sexpr;

    fn step(name: &str, arity: usize, body: &str, rewritten: &[&str]) -> StitchStep {
        StitchStep {
            name: name.to_string(),
            arity,
            body: body.to_string(),
            rewritten: rewritten.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn core(a: &mut Arena, src: &str) -> TermId {
        sexpr::parse(a, src).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    /// A library of one entry.
    fn library(a: &mut Arena) -> Vec<LibDef> {
        vec![LibDef {
            arity: 1,
            term: core(a, "(lam (sub $0 1))"),
        }]
    }

    /// Two programs over [`library`], and a first step that decodes.
    fn setup(a: &mut Arena) -> (Vec<TermId>, StitchStep) {
        let programs = vec![
            core(a, "(add (mul 2 3) (lib_0 1))"),
            core(a, "(add (mul 2 3) 4)"),
        ];
        let good = step(
            "fn_1",
            1,
            "(add (mul int_2 int_3) #0)",
            &["(fn_1 (fn_0 int_1))", "(fn_1 int_4)"],
        );
        (programs, good)
    }

    #[test]
    fn steps_that_decode_are_kept_in_order() {
        let mut a = Arena::new();
        let (programs, good) = setup(&mut a);
        // The second step's body uses the first new abstraction.
        let next = step("fn_2", 0, "(fn_1 int_4)", &["(fn_1 (fn_0 int_1))", "fn_2"]);
        let libs = library(&mut a);
        let c = accept_steps(&mut a, &libs, &programs, &[good, next], 2).expect("accept");
        assert_eq!(c.rejected, None);
        assert_eq!(c.stitch_steps, 2);
        assert_eq!(
            c.new_libs,
            vec![
                LibDef {
                    arity: 1,
                    term: core(&mut a, "(lam (add (mul 2 3) $0))")
                },
                LibDef {
                    arity: 0,
                    term: core(&mut a, "(lib_1 4)")
                },
            ]
        );
        assert_eq!(
            c.rewritten,
            vec![core(&mut a, "(lib_1 (lib_0 1))"), core(&mut a, "lib_2")]
        );
        assert_eq!(c.names.name(LibId(2)), Some("fn_2"));
    }

    /// Accepts `[good, bad, later]`, where `later` would decode, and checks that
    /// only `good` is kept and `bad` is reported with `reason`.
    #[track_caller]
    fn drops_from(bad: StitchStep, reason: RejectReason) {
        let mut a = Arena::new();
        let (programs, good) = setup(&mut a);
        let later = step("fn_3", 1, "(mul #0 #0)", &["(fn_3 int_1)", "(fn_3 int_2)"]);
        let libs = library(&mut a);
        let c =
            accept_steps(&mut a, &libs, &programs, &[good, bad.clone(), later], 2).expect("accept");
        assert_eq!(
            c.rejected,
            Some(Rejection {
                step: 1,
                name: bad.name,
                body: bad.body,
                reason
            })
        );
        assert_eq!(c.stitch_steps, 3);
        assert_eq!(c.new_libs.len(), 1);
        assert_eq!(
            c.rewritten,
            vec![core(&mut a, "(lib_1 (lib_0 1))"), core(&mut a, "(lib_1 4)")]
        );
        assert_eq!(c.names.len(), 2);
        assert_eq!(c.names.name(LibId(2)), None);
    }

    #[test]
    fn a_body_cut_through_a_let_is_dropped_with_the_later_steps() {
        drops_from(
            step(
                "fn_2",
                2,
                "(let_ #1 #0)",
                &["(fn_1 (fn_0 int_1))", "(fn_1 int_4)"],
            ),
            RejectReason::Abstraction(DecodeError::MalformedSpecialForm {
                token: "let_",
                message: "the body must be a (lam_0 ...), got #0".to_string(),
            }),
        );
    }

    #[test]
    fn a_rewritten_program_cut_through_an_if_is_dropped() {
        // A curried argument `(if_ true)` has no core term.
        let bad = step(
            "fn_2",
            1,
            "(#0 int_2 int_3)",
            &["(fn_1 (fn_0 int_1))", "(fn_2 (if_ true))"],
        );
        let mut a = Arena::new();
        let names = LibNames::sequential(ABSTRACTION_PREFIX, 3).expect("names");
        let error = decode(&mut a, "(fn_2 (if_ true))", &names).expect_err("malformed");
        drops_from(bad, RejectReason::Program { index: 1, error });
    }

    #[test]
    fn a_name_off_the_numbering_is_dropped() {
        drops_from(
            step("fn_7", 0, "int_1", &["fn_7", "fn_7"]),
            RejectReason::UnexpectedName {
                expected: Some("fn_2".to_string()),
            },
        );
    }

    #[test]
    fn an_abstraction_above_the_arity_bound_is_dropped() {
        drops_from(
            step("fn_2", 3, "(add #0 (add #1 #2))", &["fn_2", "fn_2"]),
            RejectReason::ArityTooLarge { arity: 3 },
        );
    }

    #[test]
    fn a_step_with_the_wrong_number_of_programs_is_dropped() {
        drops_from(
            step("fn_2", 0, "int_1", &["fn_2"]),
            RejectReason::ProgramCount {
                got: 1,
                expected: 2,
            },
        );
    }

    #[test]
    fn an_abstraction_cannot_refer_to_itself() {
        drops_from(
            step("fn_2", 0, "(add fn_2 int_1)", &["fn_2", "fn_2"]),
            RejectReason::Abstraction(DecodeError::UnknownToken("fn_2".to_string())),
        );
    }

    #[test]
    fn a_step_whose_id_would_overflow_is_refused() {
        // `accept_steps` passes `None` when the library length plus the step
        // index does not fit in a `u32`. That path is defensive: it needs a name
        // table of about 2^32 entries, so this test calls `check_step` directly.
        let mut a = Arena::new();
        let s = step("fn_0", 0, "int_1", &[]);
        assert_eq!(
            check_step(
                &mut a,
                None,
                &s,
                &mut LibNames::new(),
                2,
                &mut Vec::new(),
                &[]
            ),
            Err(RejectReason::UnexpectedName { expected: None })
        );
    }

    #[test]
    fn a_rewritten_program_that_expands_to_another_term_is_dropped() {
        // The step decodes, but its second program is `1`, not `(add (mul 2 3) 4)`.
        drops_from(
            step("fn_2", 0, "int_1", &["(fn_1 (fn_0 int_1))", "fn_2"]),
            RejectReason::Changed { index: 1 },
        );
    }

    #[test]
    fn an_input_program_that_does_not_expand_is_an_error() {
        let mut a = Arena::new();
        let libs = library(&mut a);
        let (mut programs, good) = setup(&mut a);
        programs[1] = core(&mut a, "(lib_5 1)");
        assert_eq!(
            accept_steps(&mut a, &libs, &programs, &[good], 2),
            Err(CompressError::Expand {
                index: 1,
                error: ExpandError::UnknownLib(LibId(5)),
            })
        );
    }

    #[test]
    fn the_inputs_are_checked_when_there_are_no_steps() {
        let mut a = Arena::new();
        let libs = library(&mut a);
        let unknown = core(&mut a, "(lib_9 1)");
        assert_eq!(
            accept_steps(&mut a, &libs, &[unknown], &[], 2),
            Err(CompressError::Expand {
                index: 0,
                error: ExpandError::UnknownLib(LibId(9)),
            })
        );
        let open = vec![LibDef {
            arity: 1,
            term: core(&mut a, "(lam (add $0 $5))"),
        }];
        assert_eq!(
            accept_steps(&mut a, &open, &[], &[], 2),
            Err(CompressError::Library(ExpandError::NotClosed {
                lib: LibId(0),
                free_depth: 5,
            }))
        );
    }

    #[test]
    fn the_arena_gains_only_what_is_kept() {
        // Steps that are dropped, and the programs of a step that a later step
        // replaced, leave nothing in the arena.
        let mut a = Arena::new();
        let libs = library(&mut a);
        let (programs, good) = setup(&mut a);
        let next = step("fn_2", 0, "(fn_1 int_4)", &["(fn_1 (fn_0 int_1))", "fn_2"]);
        let bad = step(
            "fn_3",
            1,
            "(pair #0 int_77)",
            &["(fn_3 int_78)", "(fn_3 int_79)"],
        );
        let c = accept_steps(&mut a, &libs, &programs, &[good, next, bad], 2).expect("accept");
        assert_eq!(c.rejected.as_ref().map(|r| r.step), Some(2));
        assert_eq!(c.new_libs.len(), 2);
        // The same library, programs and result, written out in a new arena.
        let mut b = Arena::new();
        library(&mut b);
        setup(&mut b);
        for src in [
            "(lam (add (mul 2 3) $0))",
            "(lib_1 4)",
            "(lib_1 (lib_0 1))",
            "lib_2",
        ] {
            core(&mut b, src);
        }
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn depth_counts_the_levels_of_an_expression() {
        let d = |src: &str| depth(&expr::parse(src).expect("parse"));
        assert_eq!(d("int_1"), 1);
        assert_eq!(d("(fst int_1)"), 2);
        assert_eq!(d("(lam (lam $0))"), 3);
        // `(add a b)` is `((add a) b)`.
        assert_eq!(d("(add int_1 int_2)"), 3);
        assert_eq!(d("(add int_1 (fst (fst int_2)))"), 4);
        assert_eq!(d("(let_ int_1 (lam_0 (fst $0)))"), 4);
    }
}
