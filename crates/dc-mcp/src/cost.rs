//! What frontier programs cost on their task's sweep (bead
//! dreamcounter-08a350).
//!
//! When an iteration ends, [`measure_frontiers`] runs each frontier member
//! and each new program on its task's growth sweep with the `dc-tasks`
//! harness, before the step selects the new frontiers, and the costs of the
//! programs the step keeps are committed with the frontiers
//! ([`crate::iterate`]). For each program it records the step counts of each
//! seed at each size, their median, and estimator C's verdict
//! ([`ClassSelect::default`]).
//!
//! # The sweep
//!
//! Each task's fuel plan is made from its variant's first
//! reference solution that runs over the whole sweep
//! ([`dc_tasks::families::Reference::max_n`]): the naive reference for the
//! collapse and sorted-order families, the quadratic one for subsequence
//! count, whose naive reference is `2ⁿ`. Each size's fuel is ten times that
//! reference's steps there, within the harness's cap
//! ([`HarnessConfig::default`]). These are the plans the corpus tests
//! measure natural solutions with. A plan is made under the default
//! harness, memory limit included, whatever the run's memory limit
//! ([`RunConfig::max_memory`](crate::state::RunConfig::max_memory)), so every
//! run gives a task's programs the same fuel. The programs themselves run
//! under the run's memory limit. A process makes each plan once and keeps
//! it. A plan that cannot be made is tried once in each call of
//! [`measure_frontiers`], and the task's programs are refused.
//!
//! # What it changes
//!
//! Each program's cost gives its growth ([`growth`]) to the step, which
//! selects frontiers under `J(L)` at the run's λ (decisions D4 and D27). At
//! λ = 0 costs change nothing the step chooses.
//!
//! # Time
//!
//! Measuring runs without the run's lock, before the commit. It stops at a
//! deadline and when the call is cancelled, checked before each program's
//! examples and before each run of a program on the sweep. A program it did
//! not reach is [`ProgramCost::NotMeasured`]. The deadline is passed by
//! whatever had started when it came:
//!
//! * one run, bounded by its fuel, at most the harness's cap of 50 million
//!   steps;
//! * one program's examples, each run bounded by the harness's example fuel;
//! * making a task's fuel plan, when the process has none for the task: it
//!   runs the reference over the sweep, up to about ten seconds for a
//!   development task.
//!
//! A cancel during measuring stops it, and the call then ends as
//! interrupted with nothing written ([`crate::iterate`]).
//!
//! A cost is a function of the program's text, its task and the harness:
//! a library only grows, so a program's text names the same definitions in
//! every later iteration. A member whose text was in the task's frontier
//! before keeps the cost it had if that was [`ProgramCost::Measured`] or
//! [`ProgramCost::Incorrect`]. Otherwise it is measured again: a refusal can
//! come from a failure that does not recur. A run is served by one build (the prelude's
//! fingerprint must match), so the harness is the same; a run served by a
//! build with another harness or estimator keeps the earlier build's costs
//! for programs it does not measure again.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use dc_core::Arena;
use dc_driver::objective::{Growth, Refusal};
use dc_surface::prelude::{Prelude, PreludeMode};
use dc_tasks::families::development;
use dc_tasks::growth::{median_steps, ClassSelect, ClassVerdict, GrowthClass};
use dc_tasks::harness::{
    check_examples, run_sweep_until, FuelPlan, HarnessConfig, RunFailure, SizeRun,
};

use crate::state::{
    CostClass, CostVerdict, FailureKind, FrontierMember, LibraryEntry, ProgramCost, SizeCost,
    SizeFailure,
};
use crate::verify::{task, Verifier};

#[cfg(test)]
thread_local! {
    /// How many programs this thread has started to measure, for tests.
    static STARTED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// How many fuel plans this thread has tried to make, for tests.
    static PLANS_TRIED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many fuel plans this thread has tried to make.
#[cfg(test)]
pub(crate) fn plans_tried() -> u64 {
    PLANS_TRIED.with(std::cell::Cell::get)
}

/// How many programs this thread has started to measure. Tests use it to
/// see that a call measured, or did not.
#[cfg(test)]
pub(crate) fn started() -> u64 {
    STARTED.with(std::cell::Cell::get)
}

/// How long [`crate::iterate`] measures the frontiers of one iteration
/// before it records the rest as not measured.
pub const MEASURE_BUDGET: Duration = Duration::from_secs(600);

/// The reason recorded for a program the deadline came before.
pub const OUT_OF_TIME: &str = "the time for measuring this iteration's frontiers ran out";

/// The reason recorded for a program not measured because the call was
/// cancelled.
pub const CANCELLED: &str = "the call was cancelled while frontiers were measured";

/// The fuel plan of `name`'s sweep, made the first time it is asked for in
/// this process and kept. A plan that cannot be made is not kept, so a
/// later call makes it again. See the module docs.
///
/// # Errors
///
/// The task is not a development task, has no reference over the whole
/// sweep, or its reference fails; the text says which.
pub fn fuel_plan(name: &str) -> Result<Arc<FuelPlan>, String> {
    type Plans = Mutex<BTreeMap<String, Arc<FuelPlan>>>;
    static PLANS: OnceLock<Plans> = OnceLock::new();
    let plans = PLANS.get_or_init(Plans::default);
    // Held while a plan is made, so two threads do not both make it. A plan
    // that could not be made is not kept, so the next call tries again.
    let mut plans = plans.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(plan) = plans.get(name) {
        return Ok(Arc::clone(plan));
    }
    let plan = Arc::new(make_plan(name)?);
    plans.insert(name.to_owned(), Arc::clone(&plan));
    Ok(plan)
}

fn make_plan(name: &str) -> Result<FuelPlan, String> {
    #[cfg(test)]
    PLANS_TRIED.with(|p| p.set(p.get() + 1));
    let variant = development()
        .into_iter()
        .find(|v| v.family.name() == name)
        .ok_or_else(|| format!("there is no development task {name:?}"))?;
    let reference = variant
        .references
        .iter()
        .find(|r| r.max_n == u64::MAX)
        .ok_or_else(|| format!("the task {name:?} has no reference over its whole sweep"))?;
    let prelude = Prelude::load().map_err(|e| format!("the prelude does not load: {e}"))?;
    let mut top = prelude.top_level(PreludeMode::On);
    top.add_source("references", variant.source)
        .map_err(|e| format!("the references of {name:?} do not compile: {e}"))?;
    let mut arena = Arena::new();
    let term = top
        .close_definition(&mut arena, reference.entry)
        .map_err(|e| format!("the reference {:?} does not compile: {e}", reference.entry))?;
    let term = prelude.expand(&mut arena, term);
    FuelPlan::from_reference(&*variant.family, &arena, term, &HarnessConfig::default())
        .map_err(|e| format!("the reference {:?} fails: {e:?}", reference.entry))
}

/// Measures `program`, a stored program of `task_name` (core text with
/// prelude and library references) on the task's sweep. `verifier` is for
/// the library the program was written against; it is cloned, not changed.
/// `stop` is called before each run; once it returns `true` the result is
/// [`ProgramCost::NotMeasured`] with `stopped` as the reason.
pub fn measure_program(
    verifier: &Verifier,
    task_name: &str,
    program: &str,
    stop: &mut dyn FnMut() -> bool,
    stopped: &str,
) -> ProgramCost {
    let measured = panic::catch_unwind(AssertUnwindSafe(|| {
        measure_unguarded(verifier, task_name, program, stop, stopped)
    }));
    measured.unwrap_or_else(|payload| {
        let why = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        ProgramCost::Refused {
            reason: format!("measuring the program failed: {why}"),
        }
    })
}

fn measure_unguarded(
    verifier: &Verifier,
    task_name: &str,
    program: &str,
    stop: &mut dyn FnMut() -> bool,
    stopped: &str,
) -> ProgramCost {
    let refused = |reason: String| ProgramCost::Refused { reason };
    let Some(family) = task(task_name) else {
        return refused(format!("there is no development task {task_name:?}"));
    };
    let plan = match fuel_plan(task_name) {
        Ok(plan) => plan,
        Err(e) => return refused(e),
    };
    let mut v = verifier.clone();
    let term = match v.expand_stored(program) {
        Ok(term) => term,
        Err(e) => return refused(format!("the program does not expand: {e}")),
    };
    let config = *v.harness();
    if stop() {
        return ProgramCost::NotMeasured {
            reason: stopped.to_owned(),
        };
    }
    #[cfg(test)]
    STARTED.with(|s| s.set(s.get() + 1));
    let passed = match check_examples(&*family, v.arena(), term, &config) {
        Ok(passed) => passed,
        Err(e) => return refused(format!("the program fails its task's checks: {e:?}")),
    };
    let report = match run_sweep_until(&*family, v.arena(), &passed, &plan, &config, stop) {
        Ok(Some(report)) => report,
        Ok(None) => {
            return ProgramCost::NotMeasured {
                reason: stopped.to_owned(),
            }
        }
        Err(e) => return refused(format!("the sweep cannot run: {e}")),
    };
    let sizes: Vec<SizeCost> = report
        .records
        .iter()
        .map(|r| match &r.run {
            SizeRun::Completed { steps } => SizeCost {
                n: r.n,
                fuel: r.fuel,
                steps: steps.clone(),
                median: median_steps(steps),
                failed: None,
            },
            SizeRun::Failed { seed, failure } => SizeCost {
                n: r.n,
                fuel: r.fuel,
                steps: Vec::new(),
                median: None,
                failed: Some(SizeFailure {
                    seed: *seed,
                    kind: match failure {
                        RunFailure::Timeout { .. } => FailureKind::Timeout,
                        RunFailure::MemoryLimit { .. } => FailureKind::MemoryLimit,
                        RunFailure::Error { .. } => FailureKind::Error,
                        RunFailure::WrongOutput { .. } => FailureKind::WrongOutput,
                    },
                }),
            },
        })
        .collect();
    if let Some((n, seed, _)) = report.first_incorrect() {
        return ProgramCost::Incorrect { sizes, n, seed };
    }
    let verdict = match ClassSelect::default().fit(&report.sweep) {
        Err(e) => CostVerdict::NotFitted {
            reason: e.to_string(),
        },
        Ok(fit) => {
            let best = class(fit.best().class);
            match fit.verdict {
                ClassVerdict::Class(c) => CostVerdict::Class(class(c)),
                ClassVerdict::TooNoisy => CostVerdict::TooNoisy { best },
                ClassVerdict::NoiseUnknown => CostVerdict::NoiseUnknown { best },
                ClassVerdict::NoFit => CostVerdict::NoFit { best },
            }
        }
    };
    ProgramCost::Measured { sizes, verdict }
}

/// What `cost`, a frontier member's stored cost, says about its growth, for
/// the cost-aware objective (`dc_driver::objective`, decision D27). `None`
/// is a member of a run made before costs were measured.
///
/// * A measured member whose verdict names a class is fitted.
/// * A measured member whose verdict is `not_fitted` and that has a size
///   that timed out is timed out. Estimator C fits the whole sweep
///   ([`ClassSelect::default`]), so every timeout is inside its window, and
///   a timeout is why such a fit fails.
/// * Any other measured member was refused by estimator C.
/// * An incorrect member is incorrect.
/// * A refused or unmeasured member, or one with no cost, is not measured.
pub fn growth(cost: Option<&ProgramCost>) -> Growth {
    match cost {
        None | Some(ProgramCost::Refused { .. } | ProgramCost::NotMeasured { .. }) => {
            Growth::NotMeasured
        }
        Some(ProgramCost::Incorrect { .. }) => Growth::Incorrect,
        Some(ProgramCost::Measured { sizes, verdict }) => match verdict {
            CostVerdict::Class(c) => Growth::Fitted(growth_class(*c)),
            CostVerdict::TooNoisy { .. } => Growth::Refused(Refusal::TooNoisy),
            CostVerdict::NoFit { .. } => Growth::Refused(Refusal::NoFit),
            CostVerdict::NoiseUnknown { .. } => Growth::Refused(Refusal::NoiseUnknown),
            CostVerdict::NotFitted { .. } => {
                let timed_out = sizes.iter().any(|s| {
                    s.failed.as_ref().is_some_and(|f| {
                        matches!(f.kind, FailureKind::Timeout | FailureKind::MemoryLimit)
                    })
                });
                if timed_out {
                    Growth::TimedOut
                } else {
                    Growth::Refused(Refusal::NotFitted)
                }
            }
        },
    }
}

fn growth_class(c: CostClass) -> GrowthClass {
    match c {
        CostClass::Constant => GrowthClass::Constant,
        CostClass::Log => GrowthClass::Log,
        CostClass::Linear => GrowthClass::Linear,
        CostClass::NLogN => GrowthClass::NLogN,
        CostClass::Quadratic => GrowthClass::Quadratic,
        CostClass::Cubic => GrowthClass::Cubic,
        CostClass::Exponential => GrowthClass::Exponential,
    }
}

fn class(c: GrowthClass) -> CostClass {
    match c {
        GrowthClass::Constant => CostClass::Constant,
        GrowthClass::Log => CostClass::Log,
        GrowthClass::Linear => CostClass::Linear,
        GrowthClass::NLogN => CostClass::NLogN,
        GrowthClass::Quadratic => CostClass::Quadratic,
        GrowthClass::Cubic => CostClass::Cubic,
        GrowthClass::Exponential => CostClass::Exponential,
    }
}

/// When to stop measuring.
#[derive(Clone, Copy, Debug)]
pub struct Limits<'a> {
    /// No run starts after this.
    pub deadline: Instant,
    /// No run starts once this is set.
    pub cancel: Option<&'a AtomicBool>,
}

/// [`Limits`], and why measuring stopped, once it has. Once stopped it
/// stays stopped, with its first reason.
struct Stopper<'a> {
    limits: Limits<'a>,
    stopped: Option<&'static str>,
}

impl Stopper<'_> {
    fn stop(&mut self) -> bool {
        if self.stopped.is_none() {
            if self.limits.cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                self.stopped = Some(CANCELLED);
            } else if Instant::now() >= self.limits.deadline {
                self.stopped = Some(OUT_OF_TIME);
            }
        }
        self.stopped.is_some()
    }
}

/// The members of each task's new frontier, `kept` (program texts, in the
/// step's order), with their costs. `library` is the library the programs
/// were written against: the run's, with the step's new abstractions.
/// Each run of a program may hold `max_memory` bytes
/// ([`RunConfig::max_memory`](crate::state::RunConfig::max_memory)).
/// `previous` is each task's frontier before the step; a member whose text
/// was there keeps its cost if it was measured or incorrect. See the
/// module docs.
pub fn measure_frontiers(
    library: &[LibraryEntry],
    kept: &BTreeMap<String, Vec<String>>,
    previous: &BTreeMap<String, Vec<FrontierMember>>,
    max_memory: u64,
    limits: Limits<'_>,
) -> BTreeMap<String, Vec<FrontierMember>> {
    let verifier = Verifier::new(library, max_memory);
    let mut stopper = Stopper {
        limits,
        stopped: None,
    };
    let mut out = BTreeMap::new();
    for (name, programs) in kept {
        let before = previous.get(name).map(Vec::as_slice).unwrap_or_default();
        // Made at most once in this call, when a program of the task is
        // first measured.
        let mut plan: Option<Result<(), String>> = None;
        let mut members = Vec::with_capacity(programs.len());
        for program in programs {
            let known = before
                .iter()
                .find(|m| &m.program == program)
                .and_then(|m| m.cost.clone())
                .filter(|c| {
                    matches!(
                        c,
                        ProgramCost::Measured { .. } | ProgramCost::Incorrect { .. }
                    )
                });
            let cost = match (known, &verifier) {
                (Some(cost), _) => cost,
                (None, Err(e)) => ProgramCost::Refused {
                    reason: format!("the library does not load: {e}"),
                },
                (None, Ok(v)) => {
                    let cost = if stopper.stop() {
                        None
                    } else if let Err(e) = plan
                        .get_or_insert_with(|| fuel_plan(name).map(|_| ()))
                        .as_ref()
                    {
                        Some(ProgramCost::Refused { reason: e.clone() })
                    } else {
                        Some(measure_program(
                            v,
                            name,
                            program,
                            &mut || stopper.stop(),
                            "",
                        ))
                    };
                    match cost {
                        Some(ProgramCost::NotMeasured { .. }) | None => ProgramCost::NotMeasured {
                            reason: stopper.stopped.unwrap_or(OUT_OF_TIME).to_owned(),
                        },
                        Some(other) => other,
                    }
                }
            };
            members.push(FrontierMember {
                program: program.clone(),
                cost: Some(cost),
            });
        }
        out.insert(name.clone(), members);
    }
    out
}

#[cfg(test)]
#[path = "cost_tests.rs"]
mod tests;
