//! The subsequence-counting family (bead P3.6) through the whole pipeline:
//! each variant is a valid family; its fuel plan is made from the quadratic
//! reference; the quadratic and good references pass the examples and the
//! oracle on every generated input; estimator C names `n²` for the
//! quadratic reference and `n` for the good one, on the medians and on each
//! seed alone; and the naive (`2ⁿ`) reference times out on the sweep from
//! 32 on.
//!
//! The naive reference cannot be fitted on the family's sweep, so it is
//! checked on a short sweep of its own, where C must name `2ⁿ`.

mod common;

use common::{check_class, compile, list_len};
use dc_core::{Arena, Scheme};
use dc_eval::Value;
use dc_tasks::families::subsequence_count::{self, SubsequenceCount, Variant};
use dc_tasks::growth::{GrowthClass, LogLogSlope, SizeOutcome};
use dc_tasks::harness::{
    check_input_sizes, measure, validate_family, FuelPlan, HarnessConfig, Measurement,
};
use dc_tasks::rng::TaskRng;
use dc_tasks::task::{Example, Inputs, TaskFamily};

/// A variant with a sweep short enough for its naive reference to finish at
/// every size.
struct Short(SubsequenceCount);

impl TaskFamily for Short {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn signature(&self) -> Scheme {
        self.0.signature()
    }
    fn description(&self) -> &str {
        self.0.description()
    }
    fn examples(&self) -> Vec<Example> {
        self.0.examples()
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        self.0.generate(n, rng)
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        self.0.oracle(inputs)
    }
    fn sizes(&self) -> Vec<u64> {
        vec![6, 8, 10, 12, 14, 16]
    }
}

fn check_variant(v: Variant) {
    let family = SubsequenceCount(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let src = subsequence_count::SOURCE;
    let mut arena = Arena::new();
    let quad = compile(&mut arena, src, v.quadratic());
    let plan = FuelPlan::from_reference(&family, &arena, quad, &c)
        .expect("the quadratic reference is correct");
    println!("PLAN {}: {:?}", v.quadratic(), plan.sizes());
    assert!(
        plan.sizes().iter().all(|s| s.reference_steps.is_some()),
        "the quadratic reference finishes under the cap at every size"
    );
    let quadratic = check_class(
        &family,
        &mut arena,
        src,
        &plan,
        v.quadratic(),
        GrowthClass::Quadratic,
    );
    // C refuses the sign-changes one (no class fits within tolerance): its
    // n² term, about 54 n² steps, is only 4 to 16 times its linear term,
    // about 900 n, over the fit window 64..256, so the local slope is still
    // below 2 there. Refusing is allowed; naming another class is not.
    assert!(
        quadratic || v == Variant::SignChanges,
        "C does not name n^2 for {}",
        v.quadratic()
    );
    let good = check_class(
        &family,
        &mut arena,
        src,
        &plan,
        v.good(),
        GrowthClass::Linear,
    );
    assert!(good, "C does not name n for {}", v.good());

    // The naive reference passes the examples and gives no wrong answer, and
    // the plan cuts it off from 32 on at the latest (2^32 subsets).
    let naive = compile(&mut arena, src, v.naive());
    match measure(&family, &arena, naive, &plan, &LogLogSlope::default(), &c)
        .expect("the sweep runs")
    {
        Measurement::Measured { report, .. } => {
            for o in report.sweep.observations() {
                if o.n >= 32 {
                    assert!(
                        matches!(o.outcome, SizeOutcome::TimedOut),
                        "{} at {}: {:?}",
                        v.naive(),
                        o.n,
                        o.outcome
                    );
                }
            }
        }
        other => panic!("{}: {other:?}", v.naive()),
    }

    let short = Short(family);
    let plan = FuelPlan::from_reference(&short, &arena, naive, &c)
        .expect("the naive reference is correct on the short sweep");
    let naive_named = check_class(
        &short,
        &mut arena,
        src,
        &plan,
        v.naive(),
        GrowthClass::Exponential,
    );
    assert!(naive_named, "C does not name 2^n for {}", v.naive());
}

#[test]
fn sum_mod_7() {
    check_variant(Variant::SumMod7);
}

#[test]
fn close_steps() {
    check_variant(Variant::CloseSteps);
}

#[test]
fn no_three_in_a_row() {
    check_variant(Variant::NoThreeInARow);
}

#[test]
fn zigzag() {
    check_variant(Variant::Zigzag);
}

#[test]
fn sign_changes() {
    check_variant(Variant::SignChanges);
}

#[test]
fn variant_names_are_unique() {
    assert_eq!(subsequence_count::all().len(), Variant::ALL.len());
    let names: std::collections::BTreeSet<String> = Variant::ALL
        .iter()
        .map(|&v| SubsequenceCount(v).name().to_owned())
        .collect();
    assert_eq!(names.len(), Variant::ALL.len(), "names are unique");
}
