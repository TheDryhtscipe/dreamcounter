//! The sorted-order family (bead P3.5) through the whole pipeline: each
//! variant is a valid family, its fuel plan is made from the naive
//! reference, both references pass the examples and the oracle on every
//! generated input, and estimator C puts the naive reference in `n²` and the
//! good one in `n log n`, or refuses; it never names another class.
//!
//! C is checked on the medians over seeds and on each seed alone, as in
//! `reference_growth.rs`: a seed alone is fitted with the noise of one
//! seed's `ln` step count, measured on the full sweep.

use dc_core::{Arena, TermId};
use dc_eval::Value;
use dc_surface::prelude::{Prelude, PreludeMode};
use dc_tasks::families::sorted_order::{self, SortedOrder, Variant};
use dc_tasks::growth::{
    ClassFit, ClassSelect, ClassVerdict, FitWindow, GrowthClass, LogLogSlope, NoiseModel,
    SizeObservation, SizeOutcome, Sweep,
};
use dc_tasks::harness::{
    check_input_sizes, measure, validate_family, FuelPlan, HarnessConfig, Measurement,
};
use dc_tasks::task::TaskFamily;

/// Compiles the definition `entry` of [`sorted_order::SOURCE`] to a closed,
/// fully expanded term.
fn compile(arena: &mut Arena, entry: &str) -> TermId {
    let prelude = Prelude::load().expect("the prelude loads");
    let mut top = prelude.top_level(PreludeMode::On);
    top.add_source("sorted_order", sorted_order::SOURCE)
        .expect("the references compile");
    let t = top
        .close_definition(arena, entry)
        .expect("the entry is defined");
    prelude.expand(arena, t)
}

/// The length of the list every variant takes first.
fn list_len(inputs: &[Value]) -> u64 {
    match inputs.first() {
        Some(Value::List(l)) => l.len(),
        other => panic!("expected a list first, got {other:?}"),
    }
}

/// The noise of one seed's `ln` step count: the largest sample standard
/// deviation of the seeds' `ln` steps at a size in `window`.
fn single_seed_noise(sweep: &Sweep, window: &FitWindow) -> f64 {
    sweep
        .observations()
        .iter()
        .filter(|o| o.n >= window.n_min)
        .map(|o| match &o.outcome {
            SizeOutcome::Completed { steps } if steps.len() >= 2 => {
                let ys: Vec<f64> = steps.iter().map(|&s| (s as f64).ln()).collect();
                let k = ys.len() as f64;
                let mean = ys.iter().sum::<f64>() / k;
                let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / (k - 1.0);
                var.sqrt()
            }
            other => panic!("n = {}: {other:?}", o.n),
        })
        .fold(0.0, f64::max)
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
                        SizeOutcome::Completed { steps } => SizeObservation {
                            n: o.n,
                            outcome: SizeOutcome::Completed {
                                steps: vec![steps[k]],
                            },
                        },
                        other => panic!("{other:?}"),
                    })
                    .collect(),
            )
            .expect("a valid sweep")
        })
        .collect()
}

/// `Err` if `fit` names a class other than `expected`; whether it named one.
fn named(fit: &ClassFit, expected: GrowthClass, what: &str) -> Result<bool, String> {
    match fit.verdict {
        ClassVerdict::Class(k) if k == expected => Ok(true),
        ClassVerdict::Class(k) => Err(format!(
            "{what}: C names {k}, expected {expected} (misfits {:?})",
            fit.candidates
                .map(|c| (c.class.to_string(), (c.misfit * 1e4).round() / 1e4))
        )),
        ClassVerdict::TooNoisy | ClassVerdict::NoiseUnknown | ClassVerdict::NoFit => Ok(false),
    }
}

/// Measures `entry` under `plan` and checks C on the medians and on each
/// seed alone. Returns whether C named the class on the medians.
fn check_class(
    family: &SortedOrder,
    arena: &mut Arena,
    plan: &FuelPlan,
    entry: &str,
    expected: GrowthClass,
) -> bool {
    let c = HarnessConfig::default();
    let p = compile(arena, entry);
    let sweep = match measure(family, arena, p, plan, &LogLogSlope::default(), &c)
        .expect("the sweep runs")
    {
        Measurement::Measured { report, .. } => report.sweep,
        other => panic!("{entry}: {other:?}"),
    };
    let fit = ClassSelect::default().fit(&sweep).expect("C fits");
    println!(
        "RESULT {entry}: C {:?} tolerance {:.4} noise_margin {:.4} misfits {:?}",
        fit.verdict,
        fit.tolerance,
        fit.noise_margin,
        fit.candidates
            .map(|k| (k.class.to_string(), (k.misfit * 1e4).round() / 1e4))
    );
    let on_medians = named(&fit, expected, entry).unwrap_or_else(|e| panic!("{e}"));
    let noise = NoiseModel::Known(single_seed_noise(&sweep, &fit.window));
    let one = ClassSelect {
        noise_model: noise,
        ..ClassSelect::default()
    };
    for (seed, s) in per_seed(&sweep).iter().enumerate() {
        let f = one.fit(s).expect("C fits one seed");
        named(&f, expected, &format!("{entry} seed {seed}")).unwrap_or_else(|e| panic!("{e}"));
    }
    on_medians
}

fn check_variant(v: Variant) {
    let family = SortedOrder(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let mut arena = Arena::new();
    let naive = compile(&mut arena, v.naive());
    let plan = FuelPlan::from_reference(&family, &arena, naive, &c)
        .expect("the naive reference is correct");
    println!("PLAN {}: {:?}", v.naive(), plan.sizes());
    // The naive references' steps do not depend on the input (see SOURCE),
    // so C has no noise to refuse them for.
    let naive_named = check_class(
        &family,
        &mut arena,
        &plan,
        v.naive(),
        GrowthClass::Quadratic,
    );
    assert!(naive_named, "C does not name n^2 for {}", v.naive());
    // Refusing is allowed, but C names n log n for every good reference
    // today; a refusal would be a change worth seeing.
    let good_named = check_class(&family, &mut arena, &plan, v.good(), GrowthClass::NLogN);
    assert!(good_named, "C does not name n log n for {}", v.good());
}

#[test]
fn inversions() {
    check_variant(Variant::Inversions);
}

#[test]
fn pair_sums() {
    check_variant(Variant::PairSums);
}

#[test]
fn min_gap() {
    check_variant(Variant::MinGap);
}

#[test]
fn distinct() {
    check_variant(Variant::Distinct);
}

#[test]
fn lower_median() {
    check_variant(Variant::LowerMedian);
}

#[test]
fn variant_names_are_unique() {
    assert_eq!(sorted_order::all().len(), Variant::ALL.len());
    let names: std::collections::BTreeSet<String> = Variant::ALL
        .iter()
        .map(|&v| SortedOrder(v).name().to_owned())
        .collect();
    assert_eq!(names.len(), Variant::ALL.len(), "names are unique");
}
