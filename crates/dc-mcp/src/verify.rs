//! Checking a submitted program against its task (decision D11).
//!
//! A proposer writes a program in surface syntax (spec §10): `let`
//! definitions, one of them named [`ENTRY`]. The prelude is in scope, and
//! library definition `i` is in scope as `fn_i`. A [`Verifier`] checks a
//! program in three stages and stops at the first that fails:
//!
//! 1. **Compile.** The text parses and desugars, and defines [`ENTRY`].
//! 2. **Type.** The entry has the task's signature, with the prelude's and
//!    the library's type schemes.
//! 3. **Run.** The entry, with the prelude and the library expanded, gives
//!    the expected output on every example of the task, within the harness's
//!    example fuel ([`dc_tasks::harness::check_examples`]), and then on the
//!    task's gate, if it has one, within the gate's fuel
//!    ([`dc_tasks::harness::check_gate`], decision D25).
//!
//! A program that passes all three is kept as a core term with its prelude
//! and library references in place, so the compressor sees `map` as a token,
//! not its body (spec §10).

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use dc_core::types::{check, infer};
use dc_core::{sexpr, Arena, ExpandError, Expander, LibDef, LibId, Scheme, TermId};
use dc_surface::prelude::{Prelude, PreludeMode};
use dc_surface::TopLevel;
use dc_tasks::harness::{check_examples, check_gate, CheckFailure, HarnessConfig, RunFailure};
use dc_tasks::task::TaskFamily;

use crate::state::{Check, LibraryEntry, RunConfig};

/// The name of the definition a submitted program is checked by.
pub const ENTRY: &str = "solve";

/// What a library definition is called in programs: `fn_0`, `fn_1`, ...
pub fn library_name(index: usize) -> String {
    format!("fn_{index}")
}

/// What library entry `index` is called in payloads: the name a proposer
/// gave it (decision D23), or [`library_name`].
pub fn entry_name(index: usize, entry: &LibraryEntry) -> String {
    entry.name.clone().unwrap_or_else(|| library_name(index))
}

/// The longest error text kept for a program, in bytes. The rest is cut.
pub const MAX_ERROR_LEN: usize = 2000;

/// How many values of an output an error message shows.
const SHOWN_VALUES: u64 = 40;

/// The run's library could not be loaded. The run's state is wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BadLibrary {
    /// The entry.
    pub index: usize,
    /// Why.
    pub reason: String,
}

impl std::fmt::Display for BadLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "library entry {} is wrong: {}", self.index, self.reason)
    }
}

impl std::error::Error for BadLibrary {}

/// The fingerprint of the prelude this build compiles programs with
/// (`Prelude::fingerprint`), found once in each process. A new run stores it
/// ([`RunConfig::prelude`]).
///
/// # Panics
///
/// If the shipped prelude does not load, which the tests of dc-surface rule
/// out.
pub fn shipped_prelude() -> u64 {
    static SHIPPED: OnceLock<u64> = OnceLock::new();
    *SHIPPED.get_or_init(|| {
        Prelude::load()
            .expect("the shipped prelude loads")
            .fingerprint()
    })
}

/// The run's stored programs were compiled with another prelude than the
/// one in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreludeChanged {
    /// The fingerprint the run stores.
    pub run: u64,
    /// The fingerprint of the prelude in use.
    pub now: u64,
}

impl std::fmt::Display for PreludeChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the run's programs were compiled with another prelude (fingerprint {:016x}) than \
             this server's ({:016x}). They name prelude functions by position, so they cannot \
             be used. Serve the run with the dc-mcp build that made it",
            self.run, self.now
        )
    }
}

impl std::error::Error for PreludeChanged {}

/// Checks that a run with the settings `config` can be used under a prelude
/// with the fingerprint `now`: that it is the prelude the run's programs
/// were compiled with. A frontier member, a verified program or a library
/// entry names a prelude function as `lib_<PRELUDE_LIB_BASE + i>`, by its
/// position. Under a prelude with its definitions in another order the
/// reference can name another function of the same type, such as `min` for
/// `max`, and every type check still passes (decision D19, item 10).
///
/// Call it with [`shipped_prelude`] before a run's library, frontiers or
/// verified programs are parsed, and before a program is compiled for it.
pub fn check_prelude(config: &RunConfig, now: u64) -> Result<(), PreludeChanged> {
    if config.prelude == now {
        Ok(())
    } else {
        Err(PreludeChanged {
            run: config.prelude,
            now,
        })
    }
}

#[cfg(test)]
thread_local! {
    /// How many verifiers this thread has built, for tests.
    static BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many verifiers this thread has built. Tests use it to count builds.
#[cfg(test)]
pub(crate) fn builds() -> u64 {
    BUILDS.with(std::cell::Cell::get)
}

/// Checks programs against tasks under one library.
///
/// Building one loads the prelude and checks and expands the whole library,
/// so a server keeps the latest in a [`VerifierCache`]. A check adds to the
/// verifier's arena, so checks run on a clone, which is cheaper than a build.
#[derive(Clone, Debug)]
pub struct Verifier {
    arena: Arena,
    prelude: Prelude,
    top: TopLevel,
    expander: Expander,
    schemes: Vec<Scheme>,
    harness: HarnessConfig,
    /// The library, to build the verifier again after a panic.
    library: Vec<LibraryEntry>,
    /// A check panicked. The expander's tables may be half changed, so the
    /// next check builds the verifier again first.
    poisoned: bool,
}

impl Verifier {
    /// A verifier for programs written against `library`, with the prelude
    /// on.
    pub fn new(library: &[LibraryEntry]) -> Result<Verifier, BadLibrary> {
        #[cfg(test)]
        BUILDS.with(|b| b.set(b.get() + 1));
        let prelude = Prelude::load().map_err(|e| BadLibrary {
            index: 0,
            reason: format!("the prelude does not load: {e}"),
        })?;
        let mut arena = Arena::new();
        let mut top = prelude.top_level(PreludeMode::On);
        let mut defs = Vec::with_capacity(library.len());
        let mut schemes: Vec<Scheme> = Vec::with_capacity(library.len());
        for (index, entry) in library.iter().enumerate() {
            let bad = |reason: String| BadLibrary { index, reason };
            let term = sexpr::parse(&mut arena, &entry.term).map_err(|e| bad(e.to_string()))?;
            let scheme = infer(&arena, term, &prelude.with_libs(&schemes))
                .map_err(|e| bad(e.to_string()))?;
            // The learned expander must never see a prelude id.
            let term = prelude.expand(&mut arena, term);
            defs.push(LibDef {
                arity: entry.arity,
                term,
            });
            schemes.push(scheme);
            let id = u32::try_from(index).map_err(|_| bad("too many entries".to_owned()))?;
            let reference = arena.lib(LibId(id));
            top.define_inline(&arena, &library_name(index), reference)
                .map_err(|e| bad(e.to_string()))?;
            // A named entry is in scope by its name too (decision D23).
            if let Some(name) = &entry.name {
                top.define_inline(&arena, name, reference)
                    .map_err(|e| bad(e.to_string()))?;
            }
        }
        let expander = Expander::new(&mut arena, &defs).map_err(|e| {
            let index = match e {
                ExpandError::UnknownLib(LibId(l))
                | ExpandError::ForwardReference { lib: LibId(l), .. }
                | ExpandError::NotClosed { lib: LibId(l), .. }
                | ExpandError::TooFewLambdas { lib: LibId(l), .. } => l as usize,
            };
            BadLibrary {
                index,
                reason: e.to_string(),
            }
        })?;
        Ok(Verifier {
            arena,
            prelude,
            top,
            expander,
            schemes,
            harness: HarnessConfig::default(),
            library: library.to_vec(),
            poisoned: false,
        })
    }

    /// The library the verifier was built for.
    pub fn library(&self) -> &[LibraryEntry] {
        &self.library
    }

    /// The prelude the programs are compiled with.
    pub fn prelude(&self) -> &Prelude {
        &self.prelude
    }

    /// The library's type schemes: entry `i` is `fn_i`'s.
    pub fn schemes(&self) -> &[Scheme] {
        &self.schemes
    }

    /// Checks `text` against `family`. A panic in the checks is reported as a
    /// failure of the run stage, so the submission is still kept.
    pub fn check(&mut self, family: &dyn TaskFamily, text: &str) -> Check {
        if self.poisoned {
            match Verifier::new(&self.library) {
                Ok(fresh) => *self = fresh,
                Err(e) => {
                    return Check::Run {
                        error: format!("the server could not check the program: {e}"),
                    }
                }
            }
        }
        let checked = panic::catch_unwind(AssertUnwindSafe(|| self.check_stages(family, text)));
        let poisoned = checked.is_err();
        let check = checked.unwrap_or_else(|payload| {
            let why = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            Check::Run {
                error: format!("the server failed while checking the program: {why}"),
            }
        });
        self.poisoned = poisoned;
        cut(check)
    }

    fn check_stages(&mut self, family: &dyn TaskFamily, text: &str) -> Check {
        let program = match self.compile(text) {
            Ok(program) => program,
            Err(error) => return Check::Compile { error },
        };
        let libs = self.prelude.with_libs(&self.schemes);
        if let Err(e) = check(&self.arena, program, &libs, &family.signature()) {
            return Check::Type {
                error: e.to_string(),
            };
        }
        // The program compiled and typed, so a failure from here on is a
        // failure of the run stage.
        let stored = match sexpr::print(&self.arena, program) {
            Ok(stored) => stored,
            Err(e) => {
                return Check::Run {
                    error: format!("the program is too large to keep: {e}"),
                }
            }
        };
        let expanded = self.prelude.expand(&mut self.arena, program);
        let expanded = match self.expander.expand(&mut self.arena, expanded) {
            Ok(expanded) => expanded,
            Err(e) => {
                return Check::Run {
                    error: format!("the program does not expand: {e}"),
                }
            }
        };
        let checked = check_examples(family, &self.arena, expanded, &self.harness)
            .and_then(|passed| check_gate(family, &self.arena, &passed, &self.harness));
        match checked {
            Ok(_) => Check::Passed { program: stored },
            Err(CheckFailure::Type(e)) => Check::Type {
                error: e.to_string(),
            },
            Err(CheckFailure::Example { index, failure }) => Check::Run {
                error: format!("example {index}: {}", describe(&failure)),
            },
            Err(CheckFailure::Gate { index, failure }) => Check::Run {
                error: format!("large input {index}: {}", describe_hidden(&failure)),
            },
            Err(other) => Check::Run {
                error: format!("{other:?}"),
            },
        }
    }

    fn compile(&mut self, text: &str) -> Result<TermId, String> {
        let mut top = self.top.clone();
        let names = top.add_source("program", text).map_err(|e| e.to_string())?;
        if !names.iter().any(|n| n == ENTRY) {
            return Err(format!("the program has no definition named '{ENTRY}'"));
        }
        top.close_definition(&mut self.arena, ENTRY)
            .map_err(|e| e.to_string())
    }
}

/// The verifier of a run's latest library, kept between requests, so that a
/// server builds one verifier for each length of the library, not one for
/// each request.
///
/// A run's library only grows, so its length names the library. A cache
/// serves one run; it keeps one verifier, for the longest library it has
/// been given. A verifier is handed out only for a library equal to the one
/// it was built for, entry by entry, so a cache shared by mistake between
/// runs costs builds, not wrong checks.
#[derive(Debug, Default)]
pub struct VerifierCache {
    latest: Mutex<Option<Arc<Verifier>>>,
}

impl VerifierCache {
    /// The kept verifier, if it was built for exactly `library`.
    pub fn get(&self, library: &[LibraryEntry]) -> Option<Arc<Verifier>> {
        let latest = self.latest.lock().unwrap_or_else(PoisonError::into_inner);
        latest
            .as_ref()
            .filter(|v| v.library.as_slice() == library)
            .cloned()
    }

    /// A verifier for `library`: the kept one if it was built for exactly
    /// `library`, or a new one. A new one is kept unless the kept one is for
    /// a longer library.
    pub fn get_or_build(&self, library: &[LibraryEntry]) -> Result<Arc<Verifier>, BadLibrary> {
        if let Some(kept) = self.get(library) {
            return Ok(kept);
        }
        let built = Arc::new(Verifier::new(library)?);
        let mut latest = self.latest.lock().unwrap_or_else(PoisonError::into_inner);
        if latest
            .as_ref()
            .is_none_or(|v| v.library.len() <= library.len())
        {
            *latest = Some(Arc::clone(&built));
        }
        Ok(built)
    }
}

/// A short account of a failed run.
fn describe(failure: &RunFailure) -> String {
    match failure {
        RunFailure::WrongOutput { expected, actual } => format!(
            "expected {}, got {}",
            expected.display_within(SHOWN_VALUES),
            actual.display_within(SHOWN_VALUES)
        ),
        RunFailure::Error { kind, steps } => format!("runtime error {kind:?} after {steps} steps"),
        RunFailure::Timeout { fuel } => format!("no result within {fuel} steps"),
    }
}

/// A failure on a gate input, without the expected or the actual value or
/// the steps before an error. The gate is hidden from the prompt (decision
/// D25), and a program's output or step count could carry any feature of
/// the hidden input. Which input failed, and the kind of failure, still
/// tell a little about it.
fn describe_hidden(failure: &RunFailure) -> String {
    match failure {
        RunFailure::WrongOutput { .. } => "wrong output".to_owned(),
        RunFailure::Error { kind, .. } => format!("runtime error {kind:?}"),
        RunFailure::Timeout { .. } => describe(failure),
    }
}

/// `check` with its error text cut to [`MAX_ERROR_LEN`] bytes.
fn cut(check: Check) -> Check {
    let shorten = |mut s: String| {
        if s.len() > MAX_ERROR_LEN {
            let mut end = MAX_ERROR_LEN;
            while !s.is_char_boundary(end) {
                end -= 1;
            }
            s.truncate(end);
            s.push_str("...");
        }
        s
    };
    match check {
        Check::Compile { error } => Check::Compile {
            error: shorten(error),
        },
        Check::Type { error } => Check::Type {
            error: shorten(error),
        },
        Check::Run { error } => Check::Run {
            error: shorten(error),
        },
        other => other,
    }
}

/// The development task named `name`, or `None` (decision D14: the held-out
/// family is never among them).
pub fn task(name: &str) -> Option<Box<dyn TaskFamily>> {
    dc_tasks::families::development()
        .into_iter()
        .map(|v| v.family)
        .find(|f| f.name() == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dc_tasks::families::development;

    fn merge() -> (Box<dyn TaskFamily>, &'static str, &'static str) {
        let v = development()
            .into_iter()
            .find(|v| v.family.name() == "collapse/merge")
            .unwrap();
        let good = v.references.last().unwrap().entry;
        (v.family, v.source, good)
    }

    #[test]
    fn a_reference_solution_passes_and_keeps_its_prelude_references() {
        let (family, source, good) = merge();
        let mut v = Verifier::new(&[]).unwrap();
        let text = format!("{source}\nlet {ENTRY} = {good}\n");
        let Check::Passed { program } = v.check(&*family, &text) else {
            panic!("{:?}", v.check(&*family, &text));
        };
        // The stored program is closed core text; a program that uses a
        // prelude function refers to it, it does not contain it.
        let uses_prelude = format!("{source}\nlet {ENTRY} xs = reverse (reverse ({good} xs))\n");
        let Check::Passed { program: p2 } = v.check(&*family, &uses_prelude) else {
            panic!("reverse twice should pass");
        };
        assert!(!program.is_empty());
        let reverse = v.prelude().get("reverse").unwrap().lib();
        assert!(p2.contains(&format!("lib_{}", reverse.0)), "{p2}");
    }

    #[test]
    fn each_stage_reports_its_own_failure() {
        let (family, _, _) = merge();
        let mut v = Verifier::new(&[]).unwrap();
        type Expect = fn(&Check) -> bool;
        let cases: [(&str, Expect); 5] = [
            ("let solve xs = (", |c| matches!(c, Check::Compile { .. })),
            (
                "let other xs = xs",
                |c| matches!(c, Check::Compile { error } if error.contains("solve")),
            ),
            ("let solve xs = nosuchname xs", |c| {
                matches!(c, Check::Compile { .. })
            }),
            ("let solve xs = true", |c| matches!(c, Check::Type { .. })),
            (
                "let solve xs = xs",
                |c| matches!(c, Check::Run { error } if error.starts_with("example")),
            ),
        ];
        for (text, ok) in cases {
            let c = v.check(&*family, text);
            assert!(ok(&c), "{text}: {c:?}");
        }
    }

    #[test]
    fn a_program_that_runs_out_of_fuel_is_a_run_failure() {
        let (family, _, _) = merge();
        let mut v = Verifier::new(&[]).unwrap();
        let text = "let solve xs = loop 1000000000 xs (fun i acc -> acc)";
        let c = v.check(&*family, text);
        assert!(
            matches!(&c, Check::Run { error } if error.contains("no result within")),
            "{c:?}"
        );
    }

    #[test]
    fn library_definitions_are_in_scope_as_fn_i() {
        let (family, _, _) = merge();
        // fn_0 = λx. reverse (reverse x): the identity on lists, through the
        // prelude.
        let prelude = Prelude::load().unwrap();
        let reverse = prelude.get("reverse").unwrap().lib();
        let entry = LibraryEntry {
            name: None,
            doc: None,
            arity: 1,
            term: format!("(lam (lib_{r} (lib_{r} $0)))", r = reverse.0),
        };
        let mut v = Verifier::new(&[entry]).unwrap();
        assert_eq!(v.schemes().len(), 1);
        // fn_0 is the identity on lists, so this has the right type and the
        // wrong output; without fn_0 in scope it would not compile.
        let c = v.check(&*family, "let solve xs = fn_0 xs");
        assert!(matches!(&c, Check::Run { .. }), "{c:?}");
        let c = v.check(&*family, "let solve xs = fn_1 xs");
        assert!(matches!(&c, Check::Compile { .. }), "{c:?}");
    }

    #[test]
    fn a_library_entry_that_does_not_type_is_refused() {
        let entry = LibraryEntry {
            name: None,
            doc: None,
            arity: 1,
            term: "(lam (add $0 true))".into(),
        };
        assert_eq!(Verifier::new(&[entry]).unwrap_err().index, 0);
    }

    #[test]
    fn long_errors_are_cut() {
        let long = "x".repeat(3 * MAX_ERROR_LEN);
        let Check::Run { error } = cut(Check::Run { error: long }) else {
            unreachable!()
        };
        assert_eq!(error.len(), MAX_ERROR_LEN + 3);
    }

    #[test]
    fn only_development_tasks_are_found() {
        assert!(task("collapse/merge").is_some());
        assert!(task("no/such").is_none());
    }

    #[test]
    fn a_program_that_passes_the_examples_but_not_the_gate_does_not_pass() {
        // Decision D25: the naive dominance references give every example's
        // output, and run out of the gate's fuel.
        let v = development()
            .into_iter()
            .find(|v| v.family.name() == "dominance/weighted-inversions")
            .unwrap();
        let naive = v.references.first().unwrap();
        let good = v.references.last().unwrap();
        assert!(!naive.passes_gate && good.passes_gate);
        let mut verifier = Verifier::new(&[]).unwrap();
        let text = |entry: &str| format!("{}\nlet {ENTRY} = {entry}\n", v.source);
        match verifier.check(&*v.family, &text(naive.entry)) {
            Check::Run { error } => assert_eq!(
                error,
                format!(
                    "large input 0: no result within {} steps",
                    dc_tasks::families::dominance::GATE_FUEL
                )
            ),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            verifier.check(&*v.family, &text(good.entry)),
            Check::Passed { .. }
        ));
        // A wrong output on a gate input names neither the expected value
        // nor the program's: the program returns the input's length.
        let wrong = format!(
            "fun xs -> if length xs > 100 then length xs else {} xs",
            good.entry
        );
        match verifier.check(&*v.family, &text(&wrong)) {
            Check::Run { error } => assert_eq!(error, "large input 0: wrong output"),
            other => panic!("{other:?}"),
        }
        // Nor does a runtime error give the steps before it: here they
        // would depend on the first element.
        let error = format!(
            "fun xs -> if length xs > 100 then \
             1 / loop (match xs with | [] -> 0 | x :: _ -> x + 2000) 0 (fun i acc -> acc) \
             else {} xs",
            good.entry
        );
        match verifier.check(&*v.family, &text(&error)) {
            Check::Run { error } => {
                assert_eq!(error, "large input 0: runtime error DivisionByZero")
            }
            other => panic!("{other:?}"),
        }
    }
}
