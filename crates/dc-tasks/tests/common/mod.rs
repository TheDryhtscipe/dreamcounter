//! Helpers shared by the family tests: compiling references and checking
//! estimator C on a sweep, on the medians over seeds and on each seed alone.
//! A seed alone is fitted with the noise of one seed's `ln` step count,
//! measured on the full sweep.

use dc_core::{Arena, TermId};
use dc_eval::Value;
use dc_surface::prelude::{Prelude, PreludeMode};
use dc_tasks::growth::{
    ClassFit, ClassSelect, ClassVerdict, FitWindow, GrowthClass, LogLogSlope, NoiseModel,
    SizeObservation, SizeOutcome, Sweep,
};
use dc_tasks::harness::{measure, FuelPlan, HarnessConfig, Measurement};
use dc_tasks::task::TaskFamily;

/// Compiles the definition `entry` of `source` to a closed, fully expanded
/// term.
pub fn compile(arena: &mut Arena, source: &str, entry: &str) -> TermId {
    let prelude = Prelude::load().expect("the prelude loads");
    let mut top = prelude.top_level(PreludeMode::On);
    top.add_source("references", source)
        .expect("the references compile");
    let t = top
        .close_definition(arena, entry)
        .expect("the entry is defined");
    prelude.expand(arena, t)
}

/// The length of the list every variant takes first.
pub fn list_len(inputs: &[Value]) -> u64 {
    match inputs.first() {
        Some(Value::List(l)) => l.len(),
        other => panic!("expected a list first, got {other:?}"),
    }
}

/// The noise of one seed's `ln` step count: the largest sample standard
/// deviation of the seeds' `ln` steps at a size in `window`.
pub fn single_seed_noise(sweep: &Sweep, window: &FitWindow) -> f64 {
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
pub fn per_seed(sweep: &Sweep) -> Vec<Sweep> {
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
pub fn named(fit: &ClassFit, expected: GrowthClass, what: &str) -> Result<bool, String> {
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
pub fn check_class<F: TaskFamily>(
    family: &F,
    arena: &mut Arena,
    source: &str,
    plan: &FuelPlan,
    entry: &str,
    expected: GrowthClass,
) -> bool {
    let c = HarnessConfig::default();
    let p = compile(arena, source, entry);
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
