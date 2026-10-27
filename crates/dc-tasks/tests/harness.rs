//! End-to-end tests of the task format and harness, with two small families
//! defined here: list sum and list reverse. Their reference programs are
//! written in core syntax without the prelude.

use dc_core::{sexpr, Arena, Scheme, TermId};
use dc_eval::Value;
use dc_tasks::growth::{GrowthError, GrowthFit, LogLogSlope, SizeOutcome, DEFAULT_SIZES};
use dc_tasks::harness::{
    check_examples, check_generated, measure, run_sweep, CheckFailure, FuelPlan, HarnessConfig,
    Measurement, RunFailure, SizeRun,
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

// ---------------------------------------------------------------------------
// Programs

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
    let plan = FuelPlan::uniform(&DEFAULT_SIZES, c.max_fuel);
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
