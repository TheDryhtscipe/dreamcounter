//! End-to-end tests of the task format and harness, with two small families
//! defined here: list sum and list reverse. Their reference programs are
//! written in core syntax without the prelude.

use dc_core::{sexpr, Arena, Scheme, TermId};
use dc_eval::{apply, Outcome, Value};
use dc_tasks::growth::{
    median_steps, GrowthError, GrowthFit, LogLogSlope, SizeOutcome, Sweep, DEFAULT_SIZES,
};
use dc_tasks::harness::{
    check_examples, check_generated, measure, run_sweep, CheckFailure, FuelPlan, HarnessConfig,
    Measurement, RunFailure, SizeRun, SweepError,
};
use dc_tasks::rng::TaskRng;
use dc_tasks::task::{generate_input, Example, Inputs, TaskFamily};

// ---------------------------------------------------------------------------
// Families

fn random_ints(n: u64, rng: &mut TaskRng) -> Value {
    let xs: Vec<i64> = (0..n).map(|_| rng.int_in(-1000, 1000)).collect();
    Value::int_list(&xs)
}

fn ints_of(v: &Value) -> Vec<i64> {
    match v {
        Value::List(l) => l
            .iter()
            .map(|x| x.as_int().expect("an Int element"))
            .collect(),
        other => panic!("expected a list, got {other}"),
    }
}

struct ListSum;

impl TaskFamily for ListSum {
    fn name(&self) -> &str {
        "list-sum"
    }
    fn signature(&self) -> Scheme {
        "List Int -> Int".parse().unwrap()
    }
    fn description(&self) -> &str {
        "Return the sum of a list of integers."
    }
    fn examples(&self) -> Vec<Example> {
        vec![
            Example::new(vec![Value::int_list(&[])], Value::Int(0)),
            Example::new(vec![Value::int_list(&[1, 2, 3])], Value::Int(6)),
            Example::new(vec![Value::int_list(&[-4, 10])], Value::Int(6)),
        ]
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        vec![random_ints(n, rng)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        Value::Int(ints_of(&inputs[0]).iter().sum())
    }
}

struct ListReverse;

impl TaskFamily for ListReverse {
    fn name(&self) -> &str {
        "list-reverse"
    }
    fn signature(&self) -> Scheme {
        "List a -> List a".parse().unwrap()
    }
    fn description(&self) -> &str {
        "Reverse a list."
    }
    fn examples(&self) -> Vec<Example> {
        vec![
            Example::new(vec![Value::int_list(&[])], Value::int_list(&[])),
            Example::new(vec![Value::int_list(&[5])], Value::int_list(&[5])),
            Example::new(
                vec![Value::int_list(&[1, 2, 3])],
                Value::int_list(&[3, 2, 1]),
            ),
        ]
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        vec![random_ints(n, rng)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        match &inputs[0] {
            Value::List(l) => {
                let mut xs: Vec<Value> = l.iter().cloned().collect();
                xs.reverse();
                Value::list(xs)
            }
            other => panic!("expected a list, got {other}"),
        }
    }
}

/// The sum of the negative elements. A reference that adds only negative
/// elements takes more steps on inputs with more of them, so its step counts
/// differ between seeds at the same size.
struct ListNegSum;

impl TaskFamily for ListNegSum {
    fn name(&self) -> &str {
        "list-neg-sum"
    }
    fn signature(&self) -> Scheme {
        "List Int -> Int".parse().unwrap()
    }
    fn description(&self) -> &str {
        "Return the sum of the negative integers in a list."
    }
    fn examples(&self) -> Vec<Example> {
        vec![
            Example::new(vec![Value::int_list(&[])], Value::Int(0)),
            Example::new(vec![Value::int_list(&[1, -2, 3, -4])], Value::Int(-6)),
        ]
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        vec![random_ints(n, rng)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        Value::Int(ints_of(&inputs[0]).iter().filter(|&&x| x < 0).sum())
    }
}

// ---------------------------------------------------------------------------
// Programs

/// `foldl (\acc x -> if x < 0 then acc + x else acc) 0`.
const NEG_SUM: &str = "(lam (foldl (lam (lam (if (lt $0 0) (add $1 $0) $1))) 0 $0))";

/// `foldl (\acc x -> acc + x) 0`.
const SUM: &str = "(lam (foldl (lam (lam (add $1 $0))) 0 $0))";
/// Counts instead of summing: right on `[]`, wrong on `[1, 2, 3]`.
const SUM_COUNTS: &str = "(lam (foldl (lam (lam (add $1 1))) 0 $0))";
/// Sums lists of at most 10 elements and returns 0 for longer ones, so it
/// passes every example and fails every generated input.
const SUM_SHORT_ONLY: &str =
    "(lam (if (lt 10 (length $0)) 0 (foldl (lam (lam (add $1 $0))) 0 $0)))";
/// `foldl (\acc x -> x :: acc) []`: linear.
const REV: &str = "(lam (foldl (lam (lam (cons $0 $1))) nil $0))";
/// `foldr (\x acc -> acc ++ [x]) []` with `++` as a `foldr`: quadratic.
const REV_QUADRATIC: &str = "(lam (foldr (lam (lam (foldr cons (cons $1 nil) $0))) nil $0))";
/// The identity: right on `[]` and `[5]`, wrong on `[1, 2, 3]`.
const REV_IDENTITY: &str = "(lam $0)";
/// Reverses, but only lists of `Int`: too specific for `List a -> List a`.
const REV_INT_ONLY: &str = "(lam (foldl (lam (lam (cons (add $0 0) $1))) nil $0))";

fn parse(arena: &mut Arena, src: &str) -> TermId {
    sexpr::parse(arena, src).unwrap()
}

/// A config whose example and global caps keep a runaway test small.
fn config() -> HarnessConfig {
    HarnessConfig {
        max_fuel: 50_000_000,
        ..HarnessConfig::default()
    }
}

// ---------------------------------------------------------------------------
// The format

#[test]
fn default_config_follows_the_spec() {
    let c = HarnessConfig::default();
    assert_eq!((c.seeds, c.fuel_multiplier), (5, 10));
    assert_eq!(ListSum.sizes(), DEFAULT_SIZES.to_vec());
}

#[test]
fn default_global_cap_keeps_a_runaway_run_under_a_gigabyte() {
    // A non-terminating `unfold` candidate allocates about 16 bytes per step
    // (measured in release: 159 MB peak RSS at 10M steps). Every run the
    // harness makes is capped by `max_fuel`, so the default cap bounds the
    // memory one looping candidate can take on a shared machine.
    const BYTES_PER_RUNAWAY_STEP: u64 = 16;
    let c = HarnessConfig::default();
    assert!(
        c.max_fuel * BYTES_PER_RUNAWAY_STEP <= 1 << 30,
        "max_fuel = {}",
        c.max_fuel
    );
    assert!(c.example_fuel <= c.max_fuel);
}

#[test]
fn generator_is_deterministic_in_size_and_seed() {
    for n in [16, 1024] {
        for seed in 0..5 {
            let a = generate_input(&ListSum, n, seed);
            let b = generate_input(&ListSum, n, seed);
            assert_eq!(a, b, "n = {n}, seed = {seed}");
            let xs = ints_of(&a[0]);
            assert_eq!(xs.len() as u64, n);
            assert!(xs.iter().all(|x| (-1000..=1000).contains(x)));
        }
        assert_ne!(
            generate_input(&ListSum, n, 0),
            generate_input(&ListSum, n, 1)
        );
    }
}

// ---------------------------------------------------------------------------
// Correctness checks

#[test]
fn oracle_agrees_with_the_reference_programs() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let rev = parse(&mut arena, REV);
    let rev2 = parse(&mut arena, REV_QUADRATIC);
    let c = config();
    check_generated(&ListSum, &arena, sum, &DEFAULT_SIZES, &c).unwrap();
    check_generated(&ListReverse, &arena, rev, &DEFAULT_SIZES, &c).unwrap();
    check_generated(&ListReverse, &arena, rev2, &[16, 64, 256], &c).unwrap();
}

#[test]
fn oracle_check_rejects_a_program_that_only_fits_the_examples() {
    let mut arena = Arena::new();
    let p = parse(&mut arena, SUM_SHORT_ONLY);
    let c = config();
    assert!(check_examples(&ListSum, &arena, p, &c).is_ok());
    match check_generated(&ListSum, &arena, p, &[16, 32], &c) {
        Err(CheckFailure::Generated {
            n: 16,
            seed: 0,
            failure: RunFailure::WrongOutput { actual, .. },
        }) => assert_eq!(*actual, Value::Int(0)),
        other => panic!("expected a wrong output at n = 16, seed 0; got {other:?}"),
    }
}

#[test]
fn examples_gate_rejects_wrong_and_ill_typed_programs() {
    let mut arena = Arena::new();
    let c = config();
    let counts = parse(&mut arena, SUM_COUNTS);
    match check_examples(&ListSum, &arena, counts, &c) {
        Err(CheckFailure::Example {
            index: 1,
            failure: RunFailure::WrongOutput { expected, actual },
        }) => assert_eq!((*expected, *actual), (Value::Int(6), Value::Int(3))),
        other => panic!("expected example 1 to fail; got {other:?}"),
    }
    let identity = parse(&mut arena, REV_IDENTITY);
    assert!(matches!(
        check_examples(&ListReverse, &arena, identity, &c),
        Err(CheckFailure::Example { index: 2, .. })
    ));
    // Right on every example, but its type is `List Int -> List Int`.
    let int_only = parse(&mut arena, REV_INT_ONLY);
    assert!(matches!(
        check_examples(&ListReverse, &arena, int_only, &c),
        Err(CheckFailure::Type(_))
    ));
    let sum = parse(&mut arena, SUM);
    assert!(matches!(
        check_examples(&ListReverse, &arena, sum, &c),
        Err(CheckFailure::Type(_))
    ));
    let rev = parse(&mut arena, REV);
    let passed = check_examples(&ListReverse, &arena, rev, &c).unwrap();
    assert_eq!(passed.program(), rev);
}

#[test]
fn example_fuel_bounds_a_runaway_candidate() {
    // `unfold` that never stops: `unfold (\s -> some (s, s)) 0`, ignoring the input.
    let mut arena = Arena::new();
    let p = parse(
        &mut arena,
        "(lam (foldl (lam (lam $1)) 0 (unfold (lam (some (pair $0 $0))) 0)))",
    );
    let c = HarnessConfig {
        example_fuel: 10_000,
        ..config()
    };
    match check_examples(&ListSum, &arena, p, &c) {
        Err(CheckFailure::Example {
            index: 0,
            failure: RunFailure::Timeout { fuel: 10_000 },
        }) => {}
        other => panic!("expected a timeout on example 0; got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Fuel plans and sweeps

fn alpha(fit: &GrowthFit) -> f64 {
    fit.alpha()
}

#[test]
fn reference_fuel_is_ten_times_the_reference_steps() {
    let mut arena = Arena::new();
    let rev = parse(&mut arena, REV);
    let plan = FuelPlan::from_reference(&ListReverse, &arena, rev, &config()).unwrap();
    let sizes: Vec<u64> = plan.sizes().iter().map(|s| s.n).collect();
    assert_eq!(sizes, DEFAULT_SIZES.to_vec());
    for s in plan.sizes() {
        let r = s.reference_steps.expect("the reference finishes");
        assert!(r > s.n, "reference steps {r} at n = {}", s.n);
        assert_eq!(s.fuel, 10 * r);
    }
}

#[test]
fn linear_program_has_slope_near_one() {
    let mut arena = Arena::new();
    let rev = parse(&mut arena, REV);
    let c = config();
    let plan = FuelPlan::from_reference(&ListReverse, &arena, rev, &c).unwrap();
    let passed = check_examples(&ListReverse, &arena, rev, &c).unwrap();
    let report = run_sweep(&ListReverse, &arena, &passed, &plan, &c).unwrap();
    assert_eq!(report.records.len(), DEFAULT_SIZES.len());
    for r in &report.records {
        match &r.run {
            SizeRun::Completed { steps } => assert_eq!(steps.len(), 5),
            other => panic!("n = {}: {other:?}", r.n),
        }
    }
    let fit = LogLogSlope::default().fit(&report.sweep).unwrap();
    assert!((fit.alpha - 1.0).abs() < 0.05, "alpha = {}", fit.alpha);
}

#[test]
fn quadratic_program_has_slope_near_two() {
    let mut arena = Arena::new();
    let rev2 = parse(&mut arena, REV_QUADRATIC);
    let c = config();
    let plan = FuelPlan::uniform(&ListReverse, c.max_fuel, &c);
    let passed = check_examples(&ListReverse, &arena, rev2, &c).unwrap();
    let report = run_sweep(&ListReverse, &arena, &passed, &plan, &c).unwrap();
    let fit = LogLogSlope::default().fit(&report.sweep).unwrap();
    assert!((fit.alpha - 2.0).abs() < 0.08, "alpha = {}", fit.alpha);
}

#[test]
fn quadratic_program_times_out_under_a_linear_reference_and_is_recorded() {
    let mut arena = Arena::new();
    let rev = parse(&mut arena, REV);
    let rev2 = parse(&mut arena, REV_QUADRATIC);
    let c = config();
    let plan = FuelPlan::from_reference(&ListReverse, &arena, rev, &c).unwrap();
    let passed = check_examples(&ListReverse, &arena, rev2, &c).unwrap();
    let report = run_sweep(&ListReverse, &arena, &passed, &plan, &c).unwrap();
    // Every size is recorded, small ones completed and the largest timed out.
    assert_eq!(report.records.len(), DEFAULT_SIZES.len());
    assert!(matches!(report.records[0].run, SizeRun::Completed { .. }));
    let last = report.records.last().unwrap();
    assert!(
        matches!(
            last.run,
            SizeRun::Failed {
                failure: RunFailure::Timeout { .. },
                ..
            }
        ),
        "{last:?}"
    );
    assert_eq!(
        report.sweep.observations().last().unwrap().outcome,
        SizeOutcome::TimedOut
    );
    assert!(matches!(
        LogLogSlope::default().fit(&report.sweep),
        Err(GrowthError::IncompleteWindow { .. })
    ));
}

#[test]
fn too_small_global_cap_records_timeouts_per_size() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let c = HarnessConfig {
        max_fuel: 1000,
        ..config()
    };
    let plan = FuelPlan::from_reference(&ListSum, &arena, sum, &c).unwrap();
    assert!(plan.sizes().iter().all(|s| s.fuel <= 1000));
    let passed = check_examples(&ListSum, &arena, sum, &c).unwrap();
    let report = run_sweep(&ListSum, &arena, &passed, &plan, &c).unwrap();
    let mut completed = 0;
    let mut timed_out = 0;
    for (r, s) in report.records.iter().zip(plan.sizes()) {
        assert_eq!(r.fuel, s.fuel);
        match &r.run {
            SizeRun::Completed { steps } => {
                completed += 1;
                assert!(steps.iter().all(|&x| x <= 1000));
            }
            SizeRun::Failed {
                failure: RunFailure::Timeout { fuel },
                ..
            } => {
                timed_out += 1;
                assert_eq!(*fuel, 1000);
                // The reference itself could not finish under the cap.
                assert_eq!(s.reference_steps, None);
            }
            other => panic!("n = {}: {other:?}", r.n),
        }
    }
    assert!(completed > 0 && timed_out > 0, "{completed} / {timed_out}");
}

#[test]
fn wrong_outputs_in_the_sweep_are_recorded_as_errors() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let short = parse(&mut arena, SUM_SHORT_ONLY);
    let c = config();
    let plan = FuelPlan::from_reference(&ListSum, &arena, sum, &c).unwrap();
    let passed = check_examples(&ListSum, &arena, short, &c).unwrap();
    let report = run_sweep(&ListSum, &arena, &passed, &plan, &c).unwrap();
    assert_eq!(report.records.len(), DEFAULT_SIZES.len());
    assert_eq!(report.sweep.observations().len(), DEFAULT_SIZES.len());
    for (r, o) in report.records.iter().zip(report.sweep.observations()) {
        assert!(
            matches!(
                r.run,
                SizeRun::Failed {
                    seed: 0,
                    failure: RunFailure::WrongOutput { .. }
                }
            ),
            "{r:?}"
        );
        assert_eq!(o.outcome, SizeOutcome::Errored);
    }
}

#[test]
fn sweep_is_deterministic() {
    let mut arena = Arena::new();
    let rev = parse(&mut arena, REV);
    let c = config();
    let plan = FuelPlan::from_reference(&ListReverse, &arena, rev, &c).unwrap();
    let passed = check_examples(&ListReverse, &arena, rev, &c).unwrap();
    let a = run_sweep(&ListReverse, &arena, &passed, &plan, &c).unwrap();
    let b = run_sweep(&ListReverse, &arena, &passed, &plan, &c).unwrap();
    assert_eq!(a.records.len(), DEFAULT_SIZES.len());
    assert_eq!(a, b);
    assert!(plan.sizes().iter().all(|s| s.reference_steps.is_some()));
    let plan_again = FuelPlan::from_reference(&ListReverse, &arena, rev, &c).unwrap();
    assert_eq!(plan, plan_again);
}

#[test]
fn a_broken_reference_is_rejected() {
    let mut arena = Arena::new();
    let short = parse(&mut arena, SUM_SHORT_ONLY);
    assert!(matches!(
        FuelPlan::from_reference(&ListSum, &arena, short, &config()),
        Err(CheckFailure::Generated { n: 16, .. })
    ));
}

#[test]
fn measure_runs_the_whole_pipeline_and_gates_on_examples() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let counts = parse(&mut arena, SUM_COUNTS);
    let c = config();
    let plan = FuelPlan::from_reference(&ListSum, &arena, sum, &c).unwrap();
    let est = LogLogSlope::default();
    match measure(&ListSum, &arena, sum, &plan, &est, &c).unwrap() {
        Measurement::Measured { report, fit } => {
            assert_eq!(report.records.len(), DEFAULT_SIZES.len());
            let a = alpha(&fit.unwrap());
            assert!((a - 1.0).abs() < 0.05, "alpha = {a}");
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        measure(&ListSum, &arena, counts, &plan, &est, &c).unwrap(),
        Measurement::Rejected(CheckFailure::Example { index: 1, .. })
    ));
}

/// Sums correctly, except that it returns 0 for lists of 4 to 99 elements:
/// right on every example, wrong at n = 16, 32 and 64, which lie below the
/// default fit window, and right at every size inside it.
const SUM_WRONG_BELOW_WINDOW: &str = "(lam (if (lt 3 (length $0)) \
     (if (lt (length $0) 100) 0 (foldl (lam (lam (add $1 $0))) 0 $0)) \
     (foldl (lam (lam (add $1 $0))) 0 $0)))";

#[test]
fn measure_rejects_wrong_outputs_even_below_the_fit_window() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let wrong = parse(&mut arena, SUM_WRONG_BELOW_WINDOW);
    let c = config();
    let plan = FuelPlan::from_reference(&ListSum, &arena, sum, &c).unwrap();
    let est = LogLogSlope::default();
    match measure(&ListSum, &arena, wrong, &plan, &est, &c).unwrap() {
        Measurement::Incorrect {
            report,
            n: 16,
            seed: 0,
            failure: RunFailure::WrongOutput { actual, .. },
        } => {
            assert_eq!(*actual, Value::Int(0));
            // The whole sweep is still recorded.
            assert_eq!(report.records.len(), DEFAULT_SIZES.len());
            let errored: Vec<u64> = report
                .sweep
                .observations()
                .iter()
                .filter(|o| o.outcome == SizeOutcome::Errored)
                .map(|o| o.n)
                .collect();
            assert_eq!(errored, vec![16, 32, 64]);
        }
        other => panic!("expected Incorrect at n = 16, seed 0; got {other:?}"),
    }
}

#[test]
fn measure_still_fits_when_only_timeouts_occur() {
    // A timeout is recorded, not a failure (spec §9): the quadratic reverse
    // under a linear reference times out at large sizes and is Measured, with
    // the estimator refusing over its incomplete window.
    let mut arena = Arena::new();
    let rev = parse(&mut arena, REV);
    let rev2 = parse(&mut arena, REV_QUADRATIC);
    let c = config();
    let plan = FuelPlan::from_reference(&ListReverse, &arena, rev, &c).unwrap();
    let est = LogLogSlope::default();
    match measure(&ListReverse, &arena, rev2, &plan, &est, &c).unwrap() {
        Measurement::Measured { fit, .. } => {
            assert!(matches!(fit, Err(GrowthError::IncompleteWindow { .. })))
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn sweep_refuses_a_token_or_plan_from_another_family_or_seed_count() {
    let mut arena = Arena::new();
    let sum = parse(&mut arena, SUM);
    let neg = parse(&mut arena, NEG_SUM);
    let c = config();
    let sum_plan = FuelPlan::from_reference(&ListSum, &arena, sum, &c).unwrap();
    let neg_plan = FuelPlan::from_reference(&ListNegSum, &arena, neg, &c).unwrap();
    let sum_passed = check_examples(&ListSum, &arena, sum, &c).unwrap();
    let neg_passed = check_examples(&ListNegSum, &arena, neg, &c).unwrap();
    let wrong_family = SweepError::WrongFamily {
        expected: "list-neg-sum".to_string(),
        found: "list-sum".to_string(),
    };
    // Examples passed for list-sum, swept as list-neg-sum.
    assert_eq!(
        run_sweep(&ListNegSum, &arena, &sum_passed, &neg_plan, &c),
        Err(wrong_family.clone())
    );
    // A plan made for list-sum, used for list-neg-sum.
    assert_eq!(
        run_sweep(&ListNegSum, &arena, &neg_passed, &sum_plan, &c),
        Err(wrong_family.clone())
    );
    let est = LogLogSlope::default();
    assert_eq!(
        measure(&ListNegSum, &arena, neg, &sum_plan, &est, &c),
        Err(wrong_family)
    );
    // A plan made with 5 seeds, swept with 3.
    let three = HarnessConfig { seeds: 3, ..c };
    assert_eq!(
        run_sweep(&ListNegSum, &arena, &neg_passed, &neg_plan, &three),
        Err(SweepError::SeedsMismatch { plan: 5, config: 3 })
    );
    assert!(run_sweep(&ListNegSum, &arena, &neg_passed, &neg_plan, &c).is_ok());
}

/// The exact steps of `program` on each seed's input at size `n`, run
/// directly on the evaluator rather than through the harness.
fn direct_steps<F: TaskFamily>(family: &F, arena: &Arena, program: TermId, n: u64) -> Vec<u64> {
    (0..5)
        .map(|seed| {
            let inputs = generate_input(family, n, seed);
            match apply(arena, program, &inputs, u64::MAX) {
                Outcome::Value(v, steps) => {
                    assert_eq!(v, family.oracle(&inputs));
                    steps
                }
                other => panic!("n = {n}, seed = {seed}: {other:?}"),
            }
        })
        .collect()
}

#[test]
fn sweep_runs_every_seed_and_fuel_covers_the_slowest_reference_run() {
    let mut arena = Arena::new();
    let neg = parse(&mut arena, NEG_SUM);
    let c = config();
    let plan = FuelPlan::from_reference(&ListNegSum, &arena, neg, &c).unwrap();
    let passed = check_examples(&ListNegSum, &arena, neg, &c).unwrap();
    let report = run_sweep(&ListNegSum, &arena, &passed, &plan, &c).unwrap();
    let mut medians = Vec::new();
    for ((r, o), s) in report
        .records
        .iter()
        .zip(report.sweep.observations())
        .zip(plan.sizes())
    {
        let expected = direct_steps(&ListNegSum, &arena, neg, r.n);
        let min = *expected.iter().min().unwrap();
        let max = *expected.iter().max().unwrap();
        // The family must make the seeds differ, or this test proves nothing.
        assert!(min < max, "n = {}: steps {expected:?} do not vary", r.n);
        assert_eq!(
            r.run,
            SizeRun::Completed {
                steps: expected.clone()
            }
        );
        assert_eq!(
            o.outcome,
            SizeOutcome::Completed {
                steps: expected.clone()
            }
        );
        assert_eq!(s.reference_steps, Some(max), "n = {}", r.n);
        assert_eq!(s.fuel, 10 * max, "n = {}", r.n);
        medians.push((r.n, median_steps(&expected).unwrap()));
    }
    assert_eq!(report.sweep.medians(), medians);
    let from_medians = Sweep::from_fn(&DEFAULT_SIZES, |n| {
        medians.iter().find(|&&(m, _)| m == n).unwrap().1
    })
    .unwrap();
    let est = LogLogSlope::default();
    assert_eq!(
        est.fit(&report.sweep).unwrap(),
        est.fit(&from_medians).unwrap()
    );
}
