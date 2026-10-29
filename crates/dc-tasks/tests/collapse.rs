//! The collapse family (bead P3.7) through the whole pipeline: each variant
//! is a valid family; its fuel plan is made from the naive reference; the
//! naive and good references pass the examples and the oracle on every
//! generated input; and estimator C names `n²` for the naive reference and
//! `n` for the good one, on the medians and on each seed alone.

mod common;

use common::{check_class, compile, list_len};
use dc_core::Arena;
use dc_tasks::families::collapse::{self, Collapse, Variant};
use dc_tasks::growth::GrowthClass;
use dc_tasks::harness::{check_input_sizes, validate_family, FuelPlan, HarnessConfig};
use dc_tasks::task::TaskFamily;

fn check_variant(v: Variant) {
    let family = Collapse(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let src = collapse::SOURCE;
    let mut arena = Arena::new();
    let naive = compile(&mut arena, src, v.naive());
    let plan = FuelPlan::from_reference(&family, &arena, naive, &c)
        .expect("the naive reference is correct");
    println!("PLAN {}: {:?}", v.naive(), plan.sizes());
    assert!(
        plan.sizes().iter().all(|s| s.reference_steps.is_some()),
        "the naive reference finishes under the cap at every size"
    );
    let naive_named = check_class(
        &family,
        &mut arena,
        src,
        &plan,
        v.naive(),
        GrowthClass::Quadratic,
    );
    assert!(naive_named, "C does not name n^2 for {}", v.naive());
    let good_named = check_class(
        &family,
        &mut arena,
        src,
        &plan,
        v.good(),
        GrowthClass::Linear,
    );
    assert!(good_named, "C does not name n for {}", v.good());
}

#[test]
fn merge() {
    check_variant(Variant::Merge);
}

#[test]
fn cancel() {
    check_variant(Variant::Cancel);
}

#[test]
fn absorb() {
    check_variant(Variant::Absorb);
}

#[test]
fn absorb_count() {
    check_variant(Variant::AbsorbCount);
}

#[test]
fn variant_names_are_unique() {
    assert_eq!(collapse::all().len(), Variant::ALL.len());
    let names: std::collections::BTreeSet<String> = Variant::ALL
        .iter()
        .map(|&v| Collapse(v).name().to_owned())
        .collect();
    assert_eq!(names.len(), Variant::ALL.len(), "names are unique");
}
