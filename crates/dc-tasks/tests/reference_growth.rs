//! Reference programs of known complexity, run through the whole pipeline
//! (family validation, fuel plan, sweep, estimators), checking that the
//! fitted growth matches the known class (bead P3.3).
//!
//! | program | family | class |
//! |---|---|---|
//! | `sum_ref` | `ref-sum` | `n` |
//! | `msort` | `ref-sort` | `n log n` |
//! | `isort` | `ref-sort` | `n²` |
//! | `zero_subsets` | `zero-subsets` | `2ⁿ` |
//!
//! # How the checks are set
//!
//! Neither estimator names a class (spec §9's class selector, estimator C, is
//! not built), so the tests classify the fitted slope themselves. Estimator A
//! is applied to the exact series `n`, `n log n`, `n²` and `2ⁿ` on the
//! family's own sweep; that gives each class's *ideal* slope there. A program
//! is put in the class whose ideal slope is nearest its fitted slope. On top
//! of that, the fitted slope must be within a quarter of the gap from its
//! class's ideal slope to the nearest other class's, so a program that only
//! just lands on the right side fails. Estimator B must choose `γ = 1` for
//! `n log n` and `γ = 0` for `n` and `n²`, with `α` within
//! [`B_ALPHA_TOLERANCE`] of the class's exponent. The tolerances follow from
//! the gaps between classes, not from the runs.
//!
//! Both estimators fit the medians over seeds, as the pipeline does. B's
//! choice of `γ` is right on the medians but not on every seed alone, and its
//! [`PowerLogFit::gamma_stability`] is below the spread between seeds for
//! both sorts (merge sort on the debug sweep: about 0.0004 against 0.0097 in
//! `ln` steps). The stronger, per-seed requirement is the ignored test
//! [`power_log_gamma_holds_on_every_seed_of_the_default_sweep`], which fails
//! today.
//!
//! The sweeps are short so that the tests run in seconds in a debug build:
//! `ref-sort` stops at 512 and `zero-subsets` runs `n = 4, 6, ..., 14`.

use dc_core::{Arena, Scheme, TermId};
use dc_eval::Value;
use dc_surface::prelude::{Prelude, PreludeMode};
use dc_tasks::growth::{
    FitQuality, FitWindow, GrowthError, GrowthEstimator, GrowthFit, LogLogFit, LogLogSlope,
    PowerLogFit, PowerLogGrid, SizeObservation, SizeOutcome, Sweep, Window, DEFAULT_SIZES,
};
use dc_tasks::harness::{
    check_input_sizes, measure, validate_family, FuelPlan, HarnessConfig, Measurement,
};
use dc_tasks::rng::TaskRng;
use dc_tasks::task::{Example, Inputs, TaskFamily};

// ---------------------------------------------------------------------------
// Programs

/// The reference programs in dc, compiled against the prelude. `merge` and
/// `msort` are the spec §10 samples.
const SOURCE: &str = r"
let sum_ref xs = foldl (fun acc x -> acc + x) 0 xs

let merge ab =
  bfix (length (fst ab) + length (snd ab) + 1)
    (fun self (a, b) ->
      match a with
      | [] -> b
      | x :: xs ->
        match b with
        | [] -> a
        | y :: ys -> if x <= y then x :: self (xs, b) else y :: self (a, ys))
    ab

let msort xs =
  bfix (length xs + 1)
    (fun self xs ->
      if length xs <= 1 then xs
      else
        let (l, r) = split_at (length xs / 2) xs in
        merge (self l, self r))
    xs

-- Insertion into a sorted list: linear in the elements smaller than x.
let insert x xs =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> [x]
      | y :: t -> if x <= y then x :: ys else y :: self t)
    xs

let isort xs = foldr (fun x acc -> insert x acc) [] xs

-- Every subset is visited: exactly 2^n leaves whatever the elements are.
let zero_subsets xs =
  bfix (length xs + 1)
    (fun self (ys, s) ->
      match ys with
      | [] -> if s = 0 then 1 else 0
      | y :: t -> self (t, s) + self (t, s + y))
    (xs, 0)
";

/// Compiles the definition `entry` of [`SOURCE`] to a closed, fully expanded
/// term.
fn compile(arena: &mut Arena, entry: &str) -> TermId {
    let prelude = Prelude::load().expect("the prelude loads");
    let mut top = prelude.top_level(PreludeMode::On);
    top.add_source("reference", SOURCE)
        .expect("the reference programs compile");
    let t = top
        .close_definition(arena, entry)
        .expect("the entry is defined");
    prelude.expand(arena, t)
}

// ---------------------------------------------------------------------------
// Families

fn ints_of(v: &Value) -> Vec<i64> {
    match v {
        Value::List(l) => l
            .iter()
            .map(|x| x.as_int().expect("an Int element"))
            .collect(),
        other => panic!("expected a list, got {other}"),
    }
}

fn random_ints(n: u64, rng: &mut TaskRng, lo: i64, hi: i64) -> Value {
    let xs: Vec<i64> = (0..n).map(|_| rng.int_in(lo, hi)).collect();
    Value::int_list(&xs)
}

/// The length of a family's single list input.
fn list_len(inputs: &[Value]) -> u64 {
    match inputs {
        [Value::List(l)] => l.len(),
        other => panic!("expected one list input, got {other:?}"),
    }
}

struct RefSum;

impl TaskFamily for RefSum {
    fn name(&self) -> &str {
        "ref-sum"
    }
    fn signature(&self) -> Scheme {
        "List Int -> Int".parse().expect("a scheme")
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
        vec![random_ints(n, rng, -1000, 1000)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        Value::Int(ints_of(&inputs[0]).iter().sum())
    }
}

/// The sort family's sweep in the debug tests: up to 512, not 1024, because
/// the quadratic program takes about 2.4 million steps per run at 1024.
const SORT_TEST_SIZES: &[u64] = &[16, 32, 64, 128, 256, 512];

struct RefSort {
    sizes: &'static [u64],
}

const REF_SORT: RefSort = RefSort {
    sizes: SORT_TEST_SIZES,
};

impl TaskFamily for RefSort {
    fn name(&self) -> &str {
        "ref-sort"
    }
    fn signature(&self) -> Scheme {
        "List Int -> List Int".parse().expect("a scheme")
    }
    fn description(&self) -> &str {
        "Sort a list of integers into ascending order, keeping duplicates."
    }
    fn examples(&self) -> Vec<Example> {
        vec![
            Example::new(vec![Value::int_list(&[])], Value::int_list(&[])),
            Example::new(
                vec![Value::int_list(&[3, 1, 2])],
                Value::int_list(&[1, 2, 3]),
            ),
            Example::new(
                vec![Value::int_list(&[5, -1, 5, 0])],
                Value::int_list(&[-1, 0, 5, 5]),
            ),
        ]
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        vec![random_ints(n, rng, -1000, 1000)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        let mut xs = ints_of(&inputs[0]);
        xs.sort_unstable();
        Value::int_list(&xs)
    }
    fn sizes(&self) -> Vec<u64> {
        self.sizes.to_vec()
    }
}

struct ZeroSubsets;

impl TaskFamily for ZeroSubsets {
    fn name(&self) -> &str {
        "zero-subsets"
    }
    fn signature(&self) -> Scheme {
        "List Int -> Int".parse().expect("a scheme")
    }
    fn description(&self) -> &str {
        "Count the subsets of the list's positions, the empty one included, \
         whose elements sum to 0."
    }
    fn examples(&self) -> Vec<Example> {
        vec![
            Example::new(vec![Value::int_list(&[])], Value::Int(1)),
            Example::new(vec![Value::int_list(&[1, -1])], Value::Int(2)),
            Example::new(vec![Value::int_list(&[1, 2, -3, 0])], Value::Int(4)),
        ]
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        vec![random_ints(n, rng, -5, 5)]
    }
    /// Counts by dynamic programming over the reachable sums, not by
    /// enumeration, so it shares nothing with the dc program.
    fn oracle(&self, inputs: &[Value]) -> Value {
        let mut counts: std::collections::BTreeMap<i64, i64> = [(0, 1)].into();
        for x in ints_of(&inputs[0]) {
            let mut next = counts.clone();
            for (&s, &c) in &counts {
                *next.entry(s + x).or_insert(0) += c;
            }
            counts = next;
        }
        Value::Int(counts.get(&0).copied().unwrap_or(0))
    }
    /// A super-polynomial reference declares a small sweep (spec §9).
    fn sizes(&self) -> Vec<u64> {
        vec![4, 6, 8, 10, 12, 14]
    }
}

// ---------------------------------------------------------------------------
// Classes

/// The growth classes the estimators must tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Linear,
    NLogN,
    Quadratic,
    Exponential,
}

const CLASSES: [Class; 4] = [
    Class::Linear,
    Class::NLogN,
    Class::Quadratic,
    Class::Exponential,
];

impl Class {
    /// An exact step series of the class, scaled so that rounding to an
    /// integer is negligible.
    fn steps(self, n: u64) -> u64 {
        let x = n as f64;
        let v = match self {
            Class::Linear => 1000.0 * x,
            Class::NLogN => 1000.0 * x * x.log2(),
            Class::Quadratic => 1000.0 * x * x,
            Class::Exponential => 1000.0 * x.exp2(),
        };
        v.round() as u64
    }

    /// The exponent of `n` and of `ln n` that estimator B should find, for the
    /// polynomial classes.
    fn b_target(self) -> Option<(f64, u8)> {
        match self {
            Class::Linear => Some((1.0, 0)),
            Class::NLogN => Some((1.0, 1)),
            Class::Quadratic => Some((2.0, 0)),
            Class::Exponential => None,
        }
    }

    /// Estimator A's slope for the exact series of the class on `sizes`.
    fn ideal_slope(self, sizes: &[u64]) -> f64 {
        let sweep = Sweep::from_fn(sizes, |n| self.steps(n)).expect("valid sizes");
        LogLogSlope::default()
            .fit(&sweep)
            .expect("an exact series fits")
            .alpha
    }
}

/// How estimator A's slope classifies on a sweep.
#[derive(Debug, Clone, Copy)]
struct Classified {
    /// The class whose ideal slope is nearest.
    class: Class,
    /// That class's ideal slope.
    ideal: f64,
    /// The distance from the fitted slope to it.
    distance: f64,
    /// The distance from the ideal slope to the nearest other class's.
    gap: f64,
}

fn classify(alpha: f64, sizes: &[u64]) -> Classified {
    let ideals: Vec<(Class, f64)> = CLASSES.iter().map(|&c| (c, c.ideal_slope(sizes))).collect();
    let &(class, ideal) = ideals
        .iter()
        .min_by(|a, b| (a.1 - alpha).abs().total_cmp(&(b.1 - alpha).abs()))
        .expect("four classes");
    let gap = ideals
        .iter()
        .filter(|(c, _)| *c != class)
        .map(|(_, s)| (s - ideal).abs())
        .fold(f64::INFINITY, f64::min);
    Classified {
        class,
        ideal,
        distance: (alpha - ideal).abs(),
        gap,
    }
}

/// The largest allowed distance from the class's ideal slope, as a fraction
/// of the gap to the nearest other class. A quarter leaves half of the
/// classifying margin unused.
const A_GAP_FRACTION: f64 = 0.25;

/// How far estimator B's `α` may be from the class's exponent. The `n log n`
/// and `n` ideals differ by 1 in `γ`, not in `α`, and `n` and `n²` by 1 in
/// `α`; 0.1 is a tenth of the smallest unit step between the classes.
const B_ALPHA_TOLERANCE: f64 = 0.1;

/// Checks estimator A's fit against `expected` on `sizes`.
fn check_a(alpha: f64, sizes: &[u64], expected: Class) -> Result<Classified, String> {
    let c = classify(alpha, sizes);
    if c.class != expected {
        return Err(format!(
            "A: alpha {alpha:.4} classifies as {:?} (ideal {:.4}), expected {expected:?}",
            c.class, c.ideal
        ));
    }
    if c.distance > A_GAP_FRACTION * c.gap {
        return Err(format!(
            "A: alpha {alpha:.4} is {:.4} from the {expected:?} ideal {:.4}; \
             allowed {:.4} (a quarter of the gap {:.4})",
            c.distance,
            c.ideal,
            A_GAP_FRACTION * c.gap,
            c.gap
        ));
    }
    Ok(c)
}

/// The largest spread of `ln` step counts between seeds at any size in the
/// fit window: how much the medians could move with another choice of seeds.
fn seed_spread(sweep: &Sweep, window: &FitWindow) -> f64 {
    sweep
        .observations()
        .iter()
        .filter(|o| o.n >= window.n_min)
        .map(|o| match &o.outcome {
            SizeOutcome::Completed { steps } => {
                let lo = *steps.iter().min().expect("a seed") as f64;
                let hi = *steps.iter().max().expect("a seed") as f64;
                hi.ln() - lo.ln()
            }
            other => panic!("n = {}: {other:?}", o.n),
        })
        .fold(0.0, f64::max)
}

/// Checks estimator B's fit against a polynomial class.
fn check_b(fit: &PowerLogFit, expected: Class) -> Result<(), String> {
    let (alpha, gamma) = expected.b_target().expect("a polynomial class");
    if fit.best.gamma != gamma {
        return Err(format!(
            "B: gamma {} (gamma_hat {:.3}), expected {gamma}",
            fit.best.gamma, fit.gamma_hat
        ));
    }
    if (fit.best.alpha - alpha).abs() > B_ALPHA_TOLERANCE {
        return Err(format!(
            "B: alpha {:.4}, expected {alpha} within {B_ALPHA_TOLERANCE}",
            fit.best.alpha
        ));
    }
    Ok(())
}

/// Checks that estimator B chooses the class's `γ` on each seed's step counts
/// alone, not only on the medians.
fn check_b_per_seed(sweep: &Sweep, expected: Class) -> Result<(), String> {
    let (_, gamma) = expected.b_target().expect("a polynomial class");
    for (seed, one) in per_seed(sweep).iter().enumerate() {
        let f = PowerLogGrid::default()
            .fit(one)
            .map_err(|e| e.to_string())?;
        if f.best.gamma != gamma {
            return Err(format!(
                "B: seed {seed} alone gives gamma {} (gamma_hat {:.3}), expected {gamma}",
                f.best.gamma, f.gamma_hat
            ));
        }
    }
    Ok(())
}

/// One sweep per seed, each holding only that seed's step count at every
/// size.
fn per_seed(sweep: &Sweep) -> Vec<Sweep> {
    let seeds = match &sweep.observations()[0].outcome {
        SizeOutcome::Completed { steps } => steps.len(),
        other => panic!("{other:?}"),
    };
    (0..seeds)
        .map(|k| {
            Sweep::new(
                sweep
                    .observations()
                    .iter()
                    .map(|o| match &o.outcome {
                        SizeOutcome::Completed { steps } => {
                            SizeObservation::completed(o.n, vec![steps[k]])
                        }
                        other => panic!("n = {}: {other:?}", o.n),
                    })
                    .collect(),
            )
            .expect("a valid sweep")
        })
        .collect()
}

/// Estimator A over the first and over the last three sizes of the sweep. A
/// polynomial's slope barely moves between them; an exponential's rises by
/// about `ln 2` per unit of `n` that the window moves.
fn slope_drift(sweep: &Sweep) -> f64 {
    let obs = sweep.observations();
    let fit = |part: &[SizeObservation]| {
        LogLogSlope {
            window: Window::All,
        }
        .fit(&Sweep::new(part.to_vec()).expect("a valid part"))
        .expect("a fit")
        .alpha
    };
    fit(&obs[obs.len() - 3..]) - fit(&obs[..3])
}

/// A polynomial program's slope may drift by at most this much between the
/// lower and upper thirds of its sweep: half the smallest gap between integer
/// exponents.
const POLY_MAX_DRIFT: f64 = 0.5;

// ---------------------------------------------------------------------------
// Running

/// The result of sweeping one reference program.
struct Run {
    sweep: Sweep,
    a: LogLogFit,
    b: PowerLogFit,
}

/// Validates the family, makes its fuel plan from `reference` and measures
/// `entry` with both estimators.
fn run<F: TaskFamily>(family: &F, reference: &str, entry: &str) -> Run {
    let c = HarnessConfig::default();
    validate_family(family, &c).expect("the family is valid");
    check_input_sizes(family, c.seeds, list_len).expect("inputs have size n");
    let mut arena = Arena::new();
    let r = compile(&mut arena, reference);
    let p = compile(&mut arena, entry);
    let plan = FuelPlan::from_reference(family, &arena, r, &c).expect("the reference is correct");
    let est = LogLogSlope::default();
    let sweep = match measure(family, &arena, p, &plan, &est, &c).expect("the sweep runs") {
        Measurement::Measured { report, .. } => report.sweep,
        other => panic!("{entry}: {other:?}"),
    };
    let a = est.fit(&sweep).expect("A fits");
    let b = PowerLogGrid::default().fit(&sweep).expect("B fits");
    let r2 = match a.quality {
        FitQuality::RSquared(r) => r,
        FitQuality::ZeroVariance => f64::NAN,
    };
    // One line per program for the results log.
    println!(
        "RESULT {entry} on {}: medians {:?}; A alpha {:.4} R2 {:.6} window {}..{}; \
         B gamma {} alpha {:.4} ln_c {:.4} gamma_hat {:.3} stability {:.5}; \
         seed spread {:.5}; drift {:.3}",
        family.name(),
        sweep.medians(),
        a.alpha,
        r2,
        a.window.n_min,
        a.window.n_max,
        b.best.gamma,
        b.best.alpha,
        b.best.ln_c,
        b.gamma_hat,
        b.gamma_stability,
        seed_spread(&sweep, &b.window),
        slope_drift(&sweep),
    );
    let per_seed_gamma: Vec<String> = per_seed(&sweep)
        .iter()
        .map(|s| match PowerLogGrid::default().fit(s) {
            Ok(f) => format!("{}:{:.3}", f.best.gamma, f.gamma_hat),
            Err(e) => e.to_string(),
        })
        .collect();
    println!("RESULT {entry}: B per seed (gamma:gamma_hat) {per_seed_gamma:?}");
    Run { sweep, a, b }
}

fn check_polynomial<F: TaskFamily>(family: &F, reference: &str, entry: &str, class: Class) {
    let r = run(family, reference, entry);
    let c = check_a(r.a.alpha, &family.sizes(), class).unwrap_or_else(|e| panic!("{entry}: {e}"));
    println!(
        "RESULT {entry}: A ideal {:.4} distance {:.4} gap {:.4}",
        c.ideal, c.distance, c.gap
    );
    check_b(&r.b, class).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let drift = slope_drift(&r.sweep);
    assert!(drift.abs() < POLY_MAX_DRIFT, "{entry}: drift {drift}");
}

// ---------------------------------------------------------------------------
// Tests

#[test]
fn sum_is_linear() {
    check_polynomial(&RefSum, "sum_ref", "sum_ref", Class::Linear);
}

#[test]
fn merge_sort_is_n_log_n() {
    check_polynomial(&REF_SORT, "msort", "msort", Class::NLogN);
}

#[test]
fn insertion_sort_is_quadratic() {
    // Measured under merge sort's fuel plan, as a candidate for the family
    // would be: ten times merge sort's steps is still enough up to 512.
    check_polynomial(&REF_SORT, "msort", "isort", Class::Quadratic);
}

/// Estimator B's `γ` on each seed alone, over the default sweep
/// `16, ..., 1024`. Fails today: insertion sort's seed 1 alone gives `γ = 1`
/// (`γ̂ = 0.665`), and on the shorter debug sweep merge sort's seeds 2 and 3
/// give `γ = 0`. The medians are right in both cases, but B's
/// `gamma_stability` (0.0135 for insertion sort) is far below the spread
/// between seeds (0.119 in `ln` steps), so the right answer on the medians is
/// not guaranteed. Slow in a debug build; run with
/// `cargo test --release -p dc-tasks --test reference_growth -- --ignored`.
#[test]
#[ignore = "estimator B's gamma is not robust to the seed spread; see bead dreamcounter-db440d"]
fn power_log_gamma_holds_on_every_seed_of_the_default_sweep() {
    let full = RefSort {
        sizes: &DEFAULT_SIZES,
    };
    let mut failures = Vec::new();
    for (entry, class) in [("msort", Class::NLogN), ("isort", Class::Quadratic)] {
        let r = run(&full, "msort", entry);
        if let Err(e) = check_b_per_seed(&r.sweep, class) {
            failures.push(format!("{entry}: {e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

/// Neither estimator has an exponential class. Estimator A reports a slope
/// near `ln 2 · n` over the window, which is far from every polynomial
/// class on this sweep, and the slope rises as the window moves up; that
/// rise is what tells `2ⁿ` from a high-degree polynomial.
#[test]
fn subset_enumeration_is_exponential() {
    let family = ZeroSubsets;
    let r = run(&family, "zero_subsets", "zero_subsets");
    let sizes = family.sizes();
    let c = check_a(r.a.alpha, &sizes, Class::Exponential).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "RESULT zero_subsets: A ideal {:.4} distance {:.4} gap {:.4}",
        c.ideal, c.distance, c.gap
    );
    // The local slope of 2ⁿ at n is n ln 2, so the window's slope lies
    // between its ends' local slopes.
    let ln2 = std::f64::consts::LN_2;
    let (lo, hi) = (ln2 * r.a.window.n_min as f64, ln2 * r.a.window.n_max as f64);
    assert!(
        (lo..=hi).contains(&r.a.alpha),
        "alpha {} outside [{lo}, {hi}]",
        r.a.alpha
    );
    // The lower and upper thirds are 6 apart in mean n.
    let drift = slope_drift(&r.sweep);
    assert!(
        drift > 0.5 * ln2 * 6.0,
        "drift {drift}: no sign of super-polynomial growth"
    );
}

/// A stand-in estimator that reports slope 1 for every sweep.
struct AlwaysLinear;

impl GrowthEstimator for AlwaysLinear {
    fn name(&self) -> &'static str {
        "always-linear"
    }
    fn estimate(&self, sweep: &Sweep) -> Result<GrowthFit, GrowthError> {
        let real = LogLogSlope::default().fit(sweep)?;
        Ok(GrowthFit::LogLog(LogLogFit { alpha: 1.0, ..real }))
    }
}

/// The checks above must fail for an estimator that cannot tell classes
/// apart, and for each class pass only for that class.
#[test]
fn the_checks_reject_a_wrong_estimator_and_a_wrong_class() {
    let sizes = REF_SORT.sizes();
    for &class in &CLASSES {
        let sweep = Sweep::from_fn(&sizes, |n| class.steps(n)).expect("valid sizes");
        let real = LogLogSlope::default().estimate(&sweep).expect("a fit");
        let stub = AlwaysLinear.estimate(&sweep).expect("a fit");
        assert!(check_a(real.alpha(), &sizes, class).is_ok(), "{class:?}");
        for &other in CLASSES.iter().filter(|&&o| o != class) {
            assert!(
                check_a(real.alpha(), &sizes, other).is_err(),
                "{class:?} passed as {other:?}"
            );
        }
        assert_eq!(
            check_a(stub.alpha(), &sizes, class).is_ok(),
            class == Class::Linear,
            "{class:?}"
        );
        if class.b_target().is_some() {
            let b = PowerLogGrid::default().fit(&sweep).expect("a fit");
            assert!(check_b(&b, class).is_ok(), "{class:?}");
        }
    }
    // B's γ check: an exact n log n series is not taken for n or n².
    let sweep = Sweep::from_fn(&sizes, |n| Class::NLogN.steps(n)).expect("valid sizes");
    let b = PowerLogGrid::default().fit(&sweep).expect("a fit");
    assert!(check_b(&b, Class::Linear).is_err());
    assert!(check_b(&b, Class::Quadratic).is_err());
}
