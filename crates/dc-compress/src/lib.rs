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
//! (see the `dc-stitch` crate docs). [`stitch_config`] sets the three flags that
//! rule this out: `--fused-lambda-tags 0`, `--no-curried-bodies` and
//! `--no-curried-metavars`. It runs Stitch on one thread, without shuffling, so
//! that the same programs give the same result.
//!
//! # Steps that do not decode
//!
//! Each Stitch step adds one abstraction and rewrites the programs with it, and
//! later steps can use earlier abstractions. If a step's abstraction or one of
//! its rewritten programs does not decode, [`accept_steps`] keeps the steps
//! before it, drops it and every later step, and reports why in
//! [`Compression::rejected`].

use std::fmt;

use dc_core::{Arena, LibDef, LibId, TermId};
use dc_stitch::{
    decode, decode_abstraction, encode, DecodeError, EncodeError, LibNames, NameError,
    LET_LAMBDA_TAG,
};
use stitch_core::{multistep_compression, MultistepCompressionConfig};

/// The prefix of every library name the bridge uses: library `l` is `fn_{l}`.
pub const ABSTRACTION_PREFIX: &str = "fn_";

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
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// A program could not be encoded.
    Encode {
        /// The program's index in the input.
        index: usize,
        /// The encoding error.
        error: EncodeError,
    },
}

impl fmt::Display for CompressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressError::LibraryTooLarge => f.write_str("the library has too many entries"),
            CompressError::Names(e) => write!(f, "library names: {e}"),
            CompressError::Encode { index, error } => {
                write!(f, "program {index} cannot be encoded: {error}")
            }
        }
    }
}

impl std::error::Error for CompressError {}

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
    s
}

/// The names of a library of `len` entries.
fn library_names(len: u32) -> Result<LibNames, CompressError> {
    LibNames::sequential(ABSTRACTION_PREFIX, len).map_err(CompressError::Names)
}

/// Runs Stitch on `programs`, which may refer to `libs`, and decodes the result.
///
/// The new abstractions extend `libs`; see [`Compression`]. Stitch is given no
/// weights.
pub fn compress(
    arena: &mut Arena,
    libs: &[LibDef],
    programs: &[FrontierProgram],
    cfg: &CompressConfig,
) -> Result<Compression, CompressError> {
    let len = u32::try_from(libs.len()).map_err(|_| CompressError::LibraryTooLarge)?;
    let names = library_names(len)?;
    let mut text = Vec::with_capacity(programs.len());
    for (index, p) in programs.iter().enumerate() {
        let encoded = encode(arena, p.program, &names)
            .map_err(|error| CompressError::Encode { index, error })?;
        text.push(encoded);
    }
    let ids: Vec<TermId> = programs.iter().map(|p| p.program).collect();
    if programs.is_empty() {
        return accept_steps(arena, len, &ids, &[], cfg.max_arity);
    }
    let tasks: Vec<String> = programs.iter().map(|p| p.task.clone()).collect();
    let (results, _json) = multistep_compression(
        &text,
        Some(tasks),
        None,
        None,
        None,
        &stitch_config(cfg, libs.len()),
    );
    let steps: Vec<StitchStep> = results
        .iter()
        .map(|r| StitchStep {
            name: r.inv.name.clone(),
            arity: r.inv.arity,
            body: r.inv.body.to_string(),
            rewritten: r.rewritten.iter().map(|p| p.to_string()).collect(),
        })
        .collect();
    accept_steps(arena, len, &ids, &steps, cfg.max_arity)
}

/// Decodes Stitch's `steps` for `programs` over a library of `library_len`
/// entries. Keeps the steps before the first that does not decode, and drops
/// that step and the rest.
pub fn accept_steps(
    arena: &mut Arena,
    library_len: u32,
    programs: &[TermId],
    steps: &[StitchStep],
    max_arity: u32,
) -> Result<Compression, CompressError> {
    let mut names = library_names(library_len)?;
    let mut new_libs = Vec::new();
    let mut rewritten = programs.to_vec();
    let mut rejected = None;
    for (i, step) in steps.iter().enumerate() {
        let next = u32::try_from(i)
            .ok()
            .and_then(|i| library_len.checked_add(i))
            .map(LibId);
        match accept_step(arena, next, step, &names, max_arity, programs.len()) {
            Ok(accepted) => {
                new_libs.push(accepted.def);
                names = accepted.names;
                rewritten = accepted.programs;
            }
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
    Ok(Compression {
        new_libs,
        rewritten,
        names,
        stitch_steps: steps.len(),
        rejected,
    })
}

/// A decoded step.
struct Accepted {
    def: LibDef,
    /// The name table with the step's abstraction added.
    names: LibNames,
    programs: Vec<TermId>,
}

/// Decodes one step, whose abstraction should get the id `id`.
fn accept_step(
    arena: &mut Arena,
    id: Option<LibId>,
    step: &StitchStep,
    names: &LibNames,
    max_arity: u32,
    program_count: usize,
) -> Result<Accepted, RejectReason> {
    let expected = id.map(|LibId(l)| format!("{ABSTRACTION_PREFIX}{l}"));
    let (id, name) = match (id, expected) {
        (Some(id), Some(name)) if step.name == name => (id, name),
        (_, expected) => return Err(RejectReason::UnexpectedName { expected }),
    };
    let arity = match u32::try_from(step.arity) {
        Ok(a) if a <= max_arity => a,
        _ => return Err(RejectReason::ArityTooLarge { arity: step.arity }),
    };
    // The body may refer to earlier abstractions, but not to itself.
    let term =
        decode_abstraction(arena, &step.body, arity, names).map_err(RejectReason::Abstraction)?;
    if step.rewritten.len() != program_count {
        return Err(RejectReason::ProgramCount {
            got: step.rewritten.len(),
            expected: program_count,
        });
    }
    let mut with_new = names.clone();
    with_new
        .insert(id, name)
        .expect("the bridge's names are legal, and the id and name are fresh");
    let mut programs = Vec::with_capacity(program_count);
    for (index, p) in step.rewritten.iter().enumerate() {
        let decoded =
            decode(arena, p, &with_new).map_err(|error| RejectReason::Program { index, error })?;
        programs.push(decoded);
    }
    Ok(Accepted {
        def: LibDef { arity, term },
        names: with_new,
        programs,
    })
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

    /// Two programs over a library of one entry, and a first step that decodes.
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
        let c = accept_steps(&mut a, 1, &programs, &[good, next], 2).expect("accept");
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
        let c = accept_steps(&mut a, 1, &programs, &[good, bad.clone(), later], 2).expect("accept");
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
        // A curried argument `(if_ int_1)` has no core term.
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
        // index does not fit in a `u32`.
        let mut a = Arena::new();
        let s = step("fn_0", 0, "int_1", &[]);
        assert_eq!(
            accept_step(&mut a, None, &s, &LibNames::new(), 2, 0).err(),
            Some(RejectReason::UnexpectedName { expected: None })
        );
    }
}
