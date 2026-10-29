//! The sorted-order family (bead P3.5) through the whole pipeline: each
//! variant is a valid family, its fuel plan is made from the naive
//! reference, both references pass the examples and the oracle on every
//! generated input, and estimator C puts the naive reference in `n²` and the
//! good one in `n log n`, or refuses; it never names another class.
//!
//! C is checked on the medians over seeds and on each seed alone, as in
//! `reference_growth.rs`: a seed alone is fitted with the noise of one
//! seed's `ln` step count, measured on the full sweep.

mod common;

use common::{check_class, compile, list_len};
use dc_core::Arena;
use dc_tasks::families::sorted_order::{self, SortedOrder, Variant};
use dc_tasks::growth::GrowthClass;
use dc_tasks::harness::{check_input_sizes, validate_family, FuelPlan, HarnessConfig};
use dc_tasks::task::TaskFamily;

fn check_variant(v: Variant) {
    let family = SortedOrder(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let mut arena = Arena::new();
    let naive = compile(&mut arena, sorted_order::SOURCE, v.naive());
    let plan = FuelPlan::from_reference(&family, &arena, naive, &c)
        .expect("the naive reference is correct");
    println!("PLAN {}: {:?}", v.naive(), plan.sizes());
    // The naive references' steps barely depend on the input (see SOURCE),
    // so C has little noise to refuse them for.
    let naive_named = check_class(
        &family,
        &mut arena,
        sorted_order::SOURCE,
        &plan,
        v.naive(),
        GrowthClass::Quadratic,
    );
    assert!(naive_named, "C does not name n^2 for {}", v.naive());
    // Refusing is allowed, but C names n log n for every good reference
    // today; a refusal would be a change worth seeing.
    let good_named = check_class(
        &family,
        &mut arena,
        sorted_order::SOURCE,
        &plan,
        v.good(),
        GrowthClass::NLogN,
    );
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
