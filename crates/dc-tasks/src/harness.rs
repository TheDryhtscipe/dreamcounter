//! The harness that checks candidate programs against a task family and runs
//! the growth sweep (spec §9).
//!
//! # Pipeline
//!
//! 1. [`check_examples`] type-checks a candidate against the family's
//!    signature and runs it on every example. Only a pass yields an
//!    [`ExamplesPassed`] token, and [`run_sweep`] takes that token rather than
//!    a program, so a sweep cannot run on a program that has not passed its
//!    examples (spec §9, point 2).
//! 2. [`FuelPlan::from_reference`] runs the family's reference solution over
//!    the sweep once and sets each size's fuel to
//!    [`HarnessConfig::fuel_multiplier`] times the reference's steps there,
//!    capped at [`HarnessConfig::max_fuel`] (spec §9, point 3). The
//!    reference's steps at a size are its *largest* over the size's seeds, so
//!    no seed gets less than ten times what the reference needed on it.
//! 3. [`run_sweep`] runs the candidate on [`HarnessConfig::seeds`] seeded
//!    inputs per size. Each output is checked against the oracle. A size
//!    whose runs all finish correctly records one exact step count per seed;
//!    the first failure at a size (timeout, runtime error, wrong output) is
//!    recorded for that size and the sweep goes on to the next size. Nothing
//!    is dropped: a timeout is a result, not a failure of the sweep.
//! 4. The resulting [`Sweep`] feeds a [`GrowthEstimator`]; [`measure`] does
//!    steps 1, 3 and 4 in one call. A wrong output or runtime error at any
//!    size, inside the fit window or below it, makes the candidate
//!    [`Measurement::Incorrect`] with no fit: only a timeout is not a failure.
//!
//! The [`ExamplesPassed`] token and the [`FuelPlan`] carry the family's name,
//! and the plan its number of seeds; [`run_sweep`] refuses them for another
//! family or seed count.
//!
//! [`check_generated`] checks a program against the oracle on generated
//! inputs without measuring it, to test families and reference solutions.
//! [`validate_family`] checks a family's own definition (sizes, examples
//! against the oracle, a repeatable generator) before anything runs, and
//! [`FuelPlan::from_reference`] refuses a family that fails it.
//! [`check_input_sizes`] checks that generated inputs have the size asked for.
//!
//! Every step count is a deterministic function of the program and the
//! input, and every input of `(n, seed)`, so a sweep repeats exactly.
//!
//! Candidates must be closed and fully expanded: the type check uses no
//! library abstractions, and the evaluator refuses unexpanded ones.

use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use dc_core::{types, Arena, Scheme, TermId, TypeError};
use dc_eval::{apply, ErrorKind, Outcome, Value};

use crate::growth::{
    GrowthError, GrowthEstimator, GrowthFit, PowerLogGrid, SizeObservation, Sweep,
};
use crate::task::{generate_input, TaskFamily};

/// The number of seeded inputs per size (spec §9).
pub const DEFAULT_SEEDS: u64 = 5;
/// Per-size fuel as a multiple of the reference solution's steps (spec §9).
pub const DEFAULT_FUEL_MULTIPLIER: u64 = 10;
/// The default global cap on fuel for one run.
///
/// The evaluator has no memory bound, so fuel is the only bound on memory: a
/// non-terminating `unfold` allocates about 16 bytes per step, so this cap
/// holds one runaway candidate to about 800 MB. A family whose reference
/// needs more than a tenth of this at its largest sizes gets less than ten
/// times the reference's steps there; such families declare smaller sweeps,
/// or a caller raises the cap on a machine that has the memory.
pub const DEFAULT_MAX_FUEL: u64 = 50_000_000;
/// The default fuel for one example run. Examples are small; this stops a
/// non-terminating candidate (`unfold` can loop) early.
pub const DEFAULT_EXAMPLE_FUEL: u64 = 10_000_000;

/// Harness settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HarnessConfig {
    /// Seeded inputs per size; seeds `0..seeds` are used.
    pub seeds: u64,
    /// Per-size fuel is this times the reference's steps at that size.
    pub fuel_multiplier: u64,
    /// The global cap on fuel for any one run.
    pub max_fuel: u64,
    /// Fuel for one example run, also capped by `max_fuel`.
    pub example_fuel: u64,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        HarnessConfig {
            seeds: DEFAULT_SEEDS,
            fuel_multiplier: DEFAULT_FUEL_MULTIPLIER,
            max_fuel: DEFAULT_MAX_FUEL,
            example_fuel: DEFAULT_EXAMPLE_FUEL,
        }
    }
}

/// Why one run of a program on one input did not give the expected output.
#[derive(Debug, Clone, PartialEq)]
pub enum RunFailure {
    /// The program finished with a different value.
    WrongOutput {
        /// What the oracle or example expects. Boxed, as are `actual`, to
        /// keep `Result<_, RunFailure>` small.
        expected: Box<Value>,
        /// What the program returned.
        actual: Box<Value>,
    },
    /// The program stopped with a runtime error.
    Error {
        /// The error.
        kind: ErrorKind,
        /// Steps taken before it.
        steps: u64,
    },
    /// The program ran out of fuel.
    Timeout {
        /// The fuel it had.
        fuel: u64,
    },
}

/// Why a program failed a check.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckFailure {
    /// The family itself is defective, so no program can be checked against
    /// it. Only [`FuelPlan::from_reference`] reports this.
    Family(FamilyDefect),
    /// The program does not have the family's signature.
    Type(TypeError),
    /// The program failed an example.
    Example {
        /// The example's position in [`TaskFamily::examples`].
        index: usize,
        /// What went wrong.
        failure: RunFailure,
    },
    /// The program failed on a generated input.
    Generated {
        /// The input size.
        n: u64,
        /// The input seed.
        seed: u64,
        /// What went wrong.
        failure: RunFailure,
    },
}

/// Runs `program` on `inputs` with `fuel` and compares the result with
/// `expected`. Returns the exact step count on success.
///
/// # Errors
///
/// The [`RunFailure`] if the result differs, or the run errs or times out.
pub fn run_on(
    arena: &Arena,
    program: TermId,
    inputs: &[Value],
    expected: &Value,
    fuel: u64,
) -> Result<u64, RunFailure> {
    match apply(arena, program, inputs, fuel) {
        Outcome::Value(actual, steps) => {
            if actual == *expected {
                Ok(steps)
            } else {
                Err(RunFailure::WrongOutput {
                    expected: Box::new(expected.clone()),
                    actual: Box::new(actual),
                })
            }
        }
        Outcome::Error(kind, steps) => Err(RunFailure::Error { kind, steps }),
        Outcome::Timeout(_) => Err(RunFailure::Timeout { fuel }),
    }
}

/// A defect in a task family's definition, found without running any program.
#[derive(Debug, Clone, PartialEq)]
pub enum FamilyDefect {
    /// [`TaskFamily::sizes`] is not strictly increasing and positive.
    BadSizes {
        /// The sizes the family declares.
        sizes: Vec<u64>,
    },
    /// The default fit window (the upper half of the sizes) has fewer sizes
    /// than estimator B needs, so the sweep is too short to fit growth.
    WindowTooSmall {
        /// The sizes in the window.
        points: usize,
        /// The sizes needed, [`PowerLogGrid::MIN_POINTS`].
        needed: usize,
    },
    /// The family has no examples, so the examples gate would pass anything
    /// of the right type.
    NoExamples,
    /// An example's stated output differs from the oracle's output on the
    /// example's inputs.
    ExampleDisagrees {
        /// The example's position in [`TaskFamily::examples`].
        index: usize,
        /// The output the example states.
        stated: Box<Value>,
        /// The output the oracle gives.
        oracle: Box<Value>,
    },
    /// The oracle panicked on an example's inputs.
    OraclePanicked {
        /// The example's position in [`TaskFamily::examples`].
        index: usize,
        /// The panic message, or a placeholder if it was not a string.
        message: String,
    },
    /// The generator gave the same input for two different seeds at every
    /// size, so the seeds of a sweep would all measure one input.
    IgnoresSeed {
        /// The two seeds compared.
        seeds: (u64, u64),
    },
    /// Two calls of the generator with the same `(n, seed)` gave different
    /// inputs.
    NotDeterministic {
        /// The size.
        n: u64,
        /// The seed.
        seed: u64,
    },
    /// A generated input does not have the size it was generated for.
    WrongInputSize {
        /// The size asked for.
        n: u64,
        /// The seed.
        seed: u64,
        /// The size of the input generated.
        actual: u64,
    },
}

/// Checks a family's definition before anything is run on it: its sizes are
/// strictly increasing and positive, the upper half of them holds at least
/// [`PowerLogGrid::MIN_POINTS`] sizes, it has examples and each agrees with
/// the oracle (an oracle that panics on an example is a defect, not a panic
/// here), its generator gives the same input twice for every size and seed
/// `0..config.seeds`, and seeds 0 and 1 give different inputs at some size.
/// [`FuelPlan::from_reference`] calls it.
///
/// # Errors
///
/// The first defect found, in the order above.
pub fn validate_family<F: TaskFamily + ?Sized>(
    family: &F,
    config: &HarnessConfig,
) -> Result<(), FamilyDefect> {
    let sizes = family.sizes();
    let increasing = sizes.windows(2).all(|w| w[0] < w[1]);
    if sizes.first() == Some(&0) || !increasing {
        return Err(FamilyDefect::BadSizes { sizes });
    }
    // The upper half of k sizes is the last ⌈k/2⌉ (growth::Window::UpperHalf).
    let points = sizes.len() - sizes.len() / 2;
    if points < PowerLogGrid::MIN_POINTS {
        return Err(FamilyDefect::WindowTooSmall {
            points,
            needed: PowerLogGrid::MIN_POINTS,
        });
    }
    let examples = family.examples();
    if examples.is_empty() {
        return Err(FamilyDefect::NoExamples);
    }
    for (index, e) in examples.iter().enumerate() {
        // `AssertUnwindSafe`: on a panic the family is reported as defective
        // and nothing it holds is used again here.
        let oracle = match panic::catch_unwind(AssertUnwindSafe(|| family.oracle(&e.inputs))) {
            Ok(v) => v,
            Err(payload) => {
                // `panic!` payloads are a `&str` or a `String`.
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "<non-string panic payload>".to_string());
                return Err(FamilyDefect::OraclePanicked { index, message });
            }
        };
        if oracle != e.output {
            return Err(FamilyDefect::ExampleDisagrees {
                index,
                stated: Box::new(e.output.clone()),
                oracle: Box::new(oracle),
            });
        }
    }
    for &n in &sizes {
        for seed in 0..config.seeds {
            if generate_input(family, n, seed) != generate_input(family, n, seed) {
                return Err(FamilyDefect::NotDeterministic { n, seed });
            }
        }
    }
    let seeds = (0, 1);
    if sizes
        .iter()
        .all(|&n| generate_input(family, n, seeds.0) == generate_input(family, n, seeds.1))
    {
        return Err(FamilyDefect::IgnoresSeed { seeds });
    }
    Ok(())
}

/// Checks that every generated input for the family's sizes and seeds
/// `0..seeds` has size `n`, as measured by `size_of`. The family format does
/// not say what an input's size is (a list's length, an integer's value, ...),
/// so the caller supplies the measure.
///
/// # Errors
///
/// The first input of the wrong size.
pub fn check_input_sizes<F: TaskFamily + ?Sized>(
    family: &F,
    seeds: u64,
    size_of: impl Fn(&[Value]) -> u64,
) -> Result<(), FamilyDefect> {
    for n in family.sizes() {
        for seed in 0..seeds {
            let actual = size_of(&generate_input(family, n, seed));
            if actual != n {
                return Err(FamilyDefect::WrongInputSize { n, seed, actual });
            }
        }
    }
    Ok(())
}

/// Proof that a program passed a family's type check and examples. Only
/// [`check_examples`] makes one, and it names the family, so
/// [`run_sweep`] refuses it for any other family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExamplesPassed {
    program: TermId,
    family: String,
}

impl ExamplesPassed {
    /// The program that passed.
    pub fn program(&self) -> TermId {
        self.program
    }

    /// The [`TaskFamily::name`] of the family whose examples it passed.
    pub fn family(&self) -> &str {
        &self.family
    }
}

/// Why [`run_sweep`] or [`measure`] could not run a sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SweepError {
    /// The [`ExamplesPassed`] token or the [`FuelPlan`] was made for another
    /// family.
    WrongFamily {
        /// The name of the family being swept.
        expected: String,
        /// The family the token or plan was made for.
        found: String,
    },
    /// The plan was made with a different number of seeds than the config
    /// the sweep runs with, so its fuel does not cover the same inputs.
    SeedsMismatch {
        /// The seeds the plan was made with.
        plan: u64,
        /// The seeds in the config.
        config: u64,
    },
    /// The plan's sizes do not form a valid [`Sweep`].
    Growth(GrowthError),
}

impl fmt::Display for SweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SweepError::WrongFamily { expected, found } => write!(
                f,
                "sweeping family {expected:?} with a token or plan made for {found:?}"
            ),
            SweepError::SeedsMismatch { plan, config } => write!(
                f,
                "the plan was made with {plan} seed(s) but the config has {config}"
            ),
            SweepError::Growth(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SweepError {}

impl From<GrowthError> for SweepError {
    fn from(e: GrowthError) -> Self {
        SweepError::Growth(e)
    }
}

/// Type-checks `program` against the family's signature, then runs it on
/// every example with [`HarnessConfig::example_fuel`] (capped by
/// [`HarnessConfig::max_fuel`]).
///
/// # Errors
///
/// The first failure: the type error, or the first example that fails.
pub fn check_examples<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    program: TermId,
    config: &HarnessConfig,
) -> Result<ExamplesPassed, CheckFailure> {
    let no_libs: &[Scheme] = &[];
    types::check(arena, program, no_libs, &family.signature()).map_err(CheckFailure::Type)?;
    let fuel = config.example_fuel.min(config.max_fuel);
    for (index, example) in family.examples().iter().enumerate() {
        run_on(arena, program, &example.inputs, &example.output, fuel)
            .map_err(|failure| CheckFailure::Example { index, failure })?;
    }
    Ok(ExamplesPassed {
        program,
        family: family.name().to_string(),
    })
}

/// Runs `program` on seeds `0..config.seeds` at each of `sizes` with
/// [`HarnessConfig::max_fuel`] and checks every output against the oracle.
///
/// # Errors
///
/// The first input on which the program fails.
pub fn check_generated<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    program: TermId,
    sizes: &[u64],
    config: &HarnessConfig,
) -> Result<(), CheckFailure> {
    for &n in sizes {
        for seed in 0..config.seeds {
            run_generated(family, arena, program, n, seed, config.max_fuel)
                .map_err(|failure| CheckFailure::Generated { n, seed, failure })?;
        }
    }
    Ok(())
}

/// Runs `program` on the input of `(n, seed)` and checks it against the oracle.
fn run_generated<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    program: TermId,
    n: u64,
    seed: u64,
    fuel: u64,
) -> Result<u64, RunFailure> {
    let inputs = generate_input(family, n, seed);
    let expected = family.oracle(&inputs);
    run_on(arena, program, &inputs, &expected, fuel)
}

/// The fuel for one size of a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeFuel {
    /// The input size.
    pub n: u64,
    /// The fuel for each run at this size.
    pub fuel: u64,
    /// The reference solution's largest step count over the size's seeds, or
    /// `None` if the reference did not finish within the global cap (the fuel
    /// is then the cap), or if the plan was not made from a reference.
    pub reference_steps: Option<u64>,
}

/// The per-size fuel of a sweep, for one family and one number of seeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuelPlan {
    family: String,
    seeds: u64,
    sizes: Vec<SizeFuel>,
}

impl FuelPlan {
    /// The plan from the family's reference solution over the family's sizes
    /// (spec §9): `min(fuel_multiplier × reference steps, max_fuel)` per size.
    ///
    /// The family is first checked with [`validate_family`]. The reference is
    /// then checked like a candidate (type and examples) and its
    /// every output is checked against the oracle: a reference that fails is
    /// a broken family. A reference that runs out of the global cap at a size
    /// is not a failure; that size gets the cap as its fuel.
    ///
    /// The family check assumes the default fit: [`Window::UpperHalf`](crate::growth::Window::UpperHalf) and
    /// estimator B's [`PowerLogGrid::MIN_POINTS`] (3). So a family with three
    /// or four sizes is refused here even if the caller will fit it with
    /// estimator A or with [`Window::All`](crate::growth::Window::All), where
    /// fewer points would do.
    ///
    /// # Errors
    ///
    /// [`CheckFailure::Family`] if the family is defective; otherwise the
    /// reference's first failure other than a timeout at the cap.
    pub fn from_reference<F: TaskFamily + ?Sized>(
        family: &F,
        arena: &Arena,
        reference: TermId,
        config: &HarnessConfig,
    ) -> Result<FuelPlan, CheckFailure> {
        validate_family(family, config).map_err(CheckFailure::Family)?;
        check_examples(family, arena, reference, config)?;
        let mut sizes = Vec::new();
        for n in family.sizes() {
            let mut most: Option<u64> = Some(0);
            for seed in 0..config.seeds {
                match run_generated(family, arena, reference, n, seed, config.max_fuel) {
                    Ok(steps) => most = most.map(|m| m.max(steps)),
                    Err(RunFailure::Timeout { .. }) => {
                        most = None;
                        break;
                    }
                    Err(failure) => return Err(CheckFailure::Generated { n, seed, failure }),
                }
            }
            let fuel = most.map_or(config.max_fuel, |m| {
                m.saturating_mul(config.fuel_multiplier)
                    .min(config.max_fuel)
            });
            sizes.push(SizeFuel {
                n,
                fuel,
                reference_steps: most,
            });
        }
        Ok(FuelPlan {
            family: family.name().to_string(),
            seeds: config.seeds,
            sizes,
        })
    }

    /// The same fuel, capped by [`HarnessConfig::max_fuel`], at every one of
    /// the family's sizes, with no reference.
    pub fn uniform<F: TaskFamily + ?Sized>(
        family: &F,
        fuel: u64,
        config: &HarnessConfig,
    ) -> FuelPlan {
        let fuel = fuel.min(config.max_fuel);
        FuelPlan {
            family: family.name().to_string(),
            seeds: config.seeds,
            sizes: family
                .sizes()
                .into_iter()
                .map(|n| SizeFuel {
                    n,
                    fuel,
                    reference_steps: None,
                })
                .collect(),
        }
    }

    /// The sizes and their fuel, in the family's order.
    pub fn sizes(&self) -> &[SizeFuel] {
        &self.sizes
    }

    /// The [`TaskFamily::name`] of the family the plan was made for.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// The number of seeds the plan was made with.
    pub fn seeds(&self) -> u64 {
        self.seeds
    }
}

/// What happened at one size of a sweep.
#[derive(Debug, Clone, PartialEq)]
pub enum SizeRun {
    /// Every seed finished with the right output; one step count per seed,
    /// in seed order.
    Completed {
        /// The step counts.
        steps: Vec<u64>,
    },
    /// The first seed that failed, and how. Later seeds at this size are not
    /// run.
    Failed {
        /// The seed.
        seed: u64,
        /// The failure.
        failure: RunFailure,
    },
}

/// One size of a sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct SizeRecord {
    /// The input size.
    pub n: u64,
    /// The fuel each run had.
    pub fuel: u64,
    /// What happened.
    pub run: SizeRun,
}

/// The full record of a sweep and the [`Sweep`] it gives the estimators.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepReport {
    /// One record per size of the plan, in the plan's order.
    pub records: Vec<SizeRecord>,
    /// The estimators' view: a timeout is [`SizeOutcome::TimedOut`](crate::growth::SizeOutcome::TimedOut);
    /// a runtime error or wrong output is
    /// [`SizeOutcome::Errored`](crate::growth::SizeOutcome::Errored).
    pub sweep: Sweep,
}

impl SweepReport {
    /// The smallest size at which a run failed for a reason other than a
    /// timeout (a wrong output or a runtime error), with the seed and the
    /// failure. `None` if every size completed correctly or timed out.
    pub fn first_incorrect(&self) -> Option<(u64, u64, &RunFailure)> {
        self.records.iter().find_map(|r| match &r.run {
            SizeRun::Failed {
                failure: RunFailure::Timeout { .. },
                ..
            }
            | SizeRun::Completed { .. } => None,
            SizeRun::Failed { seed, failure } => Some((r.n, *seed, failure)),
        })
    }
}

/// Runs the growth sweep for a program that has passed its examples.
///
/// # Errors
///
/// [`SweepError::WrongFamily`] if `passed` or `plan` was made for another
/// family, and [`SweepError::SeedsMismatch`] if `plan` was made with another
/// number of seeds; both are checked before anything runs.
/// [`SweepError::Growth`] if the plan's sizes do not form a valid [`Sweep`]
/// (zero or repeated sizes) or `config.seeds` is 0.
pub fn run_sweep<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    passed: &ExamplesPassed,
    plan: &FuelPlan,
    config: &HarnessConfig,
) -> Result<SweepReport, SweepError> {
    for found in [passed.family(), plan.family()] {
        if found != family.name() {
            return Err(SweepError::WrongFamily {
                expected: family.name().to_string(),
                found: found.to_string(),
            });
        }
    }
    if plan.seeds != config.seeds {
        return Err(SweepError::SeedsMismatch {
            plan: plan.seeds,
            config: config.seeds,
        });
    }
    let mut records = Vec::with_capacity(plan.sizes.len());
    let mut observations = Vec::with_capacity(plan.sizes.len());
    for &SizeFuel { n, fuel, .. } in &plan.sizes {
        let run = run_size(family, arena, passed.program, n, fuel, config.seeds);
        observations.push(match &run {
            SizeRun::Completed { steps } => SizeObservation::completed(n, steps.clone()),
            SizeRun::Failed {
                failure: RunFailure::Timeout { .. },
                ..
            } => SizeObservation::timed_out(n),
            SizeRun::Failed { .. } => SizeObservation::errored(n),
        });
        records.push(SizeRecord { n, fuel, run });
    }
    Ok(SweepReport {
        records,
        sweep: Sweep::new(observations)?,
    })
}

/// Runs seeds `0..seeds` at size `n`, stopping at the first failure.
fn run_size<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    program: TermId,
    n: u64,
    fuel: u64,
    seeds: u64,
) -> SizeRun {
    let mut steps = Vec::with_capacity(seeds as usize);
    for seed in 0..seeds {
        match run_generated(family, arena, program, n, seed, fuel) {
            Ok(s) => steps.push(s),
            Err(failure) => return SizeRun::Failed { seed, failure },
        }
    }
    SizeRun::Completed { steps }
}

/// The result of [`measure`].
#[derive(Debug, Clone, PartialEq)]
pub enum Measurement {
    /// The program failed its type check or examples; no sweep was run.
    Rejected(CheckFailure),
    /// The program passed its examples but gave a wrong output or a runtime
    /// error on a generated input during the sweep, at any size, inside the
    /// fit window or not. It is incorrect, so it gets no fit. Only a timeout
    /// is not a failure (spec §9). The full sweep is kept for the record.
    Incorrect {
        /// The sweep.
        report: SweepReport,
        /// The smallest size with a failure other than a timeout.
        n: u64,
        /// The seed that failed there.
        seed: u64,
        /// What went wrong.
        failure: RunFailure,
    },
    /// The sweep and the estimator's fit (which can refuse, for example when
    /// sizes in its window timed out). Every size completed with the right
    /// outputs or timed out.
    Measured {
        /// The sweep.
        report: SweepReport,
        /// The fit.
        fit: Result<GrowthFit, GrowthError>,
    },
}

/// Checks the examples, then sweeps and fits: the whole of spec §9 for one
/// candidate, given a plan made once per family.
///
/// # Errors
///
/// As for [`run_sweep`].
pub fn measure<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    program: TermId,
    plan: &FuelPlan,
    estimator: &dyn GrowthEstimator,
    config: &HarnessConfig,
) -> Result<Measurement, SweepError> {
    let passed = match check_examples(family, arena, program, config) {
        Ok(passed) => passed,
        Err(failure) => return Ok(Measurement::Rejected(failure)),
    };
    let report = run_sweep(family, arena, &passed, plan, config)?;
    if let Some((n, seed, failure)) = report.first_incorrect() {
        let failure = failure.clone();
        return Ok(Measurement::Incorrect {
            report,
            n,
            seed,
            failure,
        });
    }
    let fit = estimator.estimate(&report.sweep);
    Ok(Measurement::Measured { report, fit })
}
