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
//!    steps 1, 3 and 4 in one call.
//!
//! [`check_generated`] checks a program against the oracle on generated
//! inputs without measuring it, to test families and reference solutions.
//!
//! Every step count is a deterministic function of the program and the
//! input, and every input of `(n, seed)`, so a sweep repeats exactly.
//!
//! Candidates must be closed and fully expanded: the type check uses no
//! library abstractions, and the evaluator refuses unexpanded ones.

use dc_core::{types, Arena, Scheme, TermId, TypeError};
use dc_eval::{apply, ErrorKind, Outcome, Value};

use crate::growth::{GrowthError, GrowthEstimator, GrowthFit, SizeObservation, Sweep};
use crate::task::{generate_input, TaskFamily};

/// The number of seeded inputs per size (spec §9).
pub const DEFAULT_SEEDS: u64 = 5;
/// Per-size fuel as a multiple of the reference solution's steps (spec §9).
pub const DEFAULT_FUEL_MULTIPLIER: u64 = 10;
/// The default global cap on fuel for one run.
pub const DEFAULT_MAX_FUEL: u64 = 1_000_000_000;
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

/// Proof that a program passed its family's type check and examples. Only
/// [`check_examples`] makes one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExamplesPassed {
    program: TermId,
}

impl ExamplesPassed {
    /// The program that passed.
    pub fn program(&self) -> TermId {
        self.program
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
    Ok(ExamplesPassed { program })
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

/// The per-size fuel of a sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuelPlan {
    sizes: Vec<SizeFuel>,
}

impl FuelPlan {
    /// The plan from the family's reference solution over the family's sizes
    /// (spec §9): `min(fuel_multiplier × reference steps, max_fuel)` per size.
    ///
    /// The reference is checked like a candidate (type and examples) and its
    /// every output is checked against the oracle: a reference that fails is
    /// a broken family. A reference that runs out of the global cap at a size
    /// is not a failure; that size gets the cap as its fuel.
    ///
    /// # Errors
    ///
    /// The reference's first failure other than a timeout at the cap.
    pub fn from_reference<F: TaskFamily + ?Sized>(
        family: &F,
        arena: &Arena,
        reference: TermId,
        config: &HarnessConfig,
    ) -> Result<FuelPlan, CheckFailure> {
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
        Ok(FuelPlan { sizes })
    }

    /// The same fuel at every size, with no reference.
    pub fn uniform(sizes: &[u64], fuel: u64) -> FuelPlan {
        FuelPlan {
            sizes: sizes
                .iter()
                .map(|&n| SizeFuel {
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

/// Runs the growth sweep for a program that has passed its examples.
///
/// # Errors
///
/// [`GrowthError`] if the plan's sizes do not form a valid [`Sweep`] (zero
/// or repeated sizes) or `config.seeds` is 0.
pub fn run_sweep<F: TaskFamily + ?Sized>(
    family: &F,
    arena: &Arena,
    passed: &ExamplesPassed,
    plan: &FuelPlan,
    config: &HarnessConfig,
) -> Result<SweepReport, GrowthError> {
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
    /// The sweep and the estimator's fit (which can refuse, for example when
    /// sizes in its window timed out).
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
) -> Result<Measurement, GrowthError> {
    let passed = match check_examples(family, arena, program, config) {
        Ok(passed) => passed,
        Err(failure) => return Ok(Measurement::Rejected(failure)),
    };
    let report = run_sweep(family, arena, &passed, plan, config)?;
    let fit = estimator.estimate(&report.sweep);
    Ok(Measurement::Measured { report, fit })
}
