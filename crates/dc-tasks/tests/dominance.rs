//! The dominance-counting family (decision D25) through the whole pipeline:
//! each variant is a valid family, its fuel plan is made from the naive
//! reference, and estimator C puts the naive reference in `n²` and the good
//! one in `n log n`. On each seed alone C names the class or refuses as too
//! noisy, on at most the seeds [`check_variant`] allows.
//!
//! The gate: the naive and good references pass every example. The good
//! reference passes the gate within a third of its fuel, and the naive one
//! runs out of the gate's fuel, so a program that looks at every pair of
//! positions does not solve the variant.

mod common;

use common::{check_class, compile, list_len, SeedsRefused};
use dc_core::Arena;
use dc_tasks::families::dominance::{self, Dominance, Variant, GATE_FUEL, GATE_N};
use dc_tasks::growth::GrowthClass;
use dc_tasks::harness::{
    check_examples, check_gate, check_input_sizes, run_on, validate_family, CheckFailure, FuelPlan,
    HarnessConfig, RunFailure, DEFAULT_EXAMPLE_FUEL,
};
use dc_tasks::task::TaskFamily;

fn check_variant(v: Variant) {
    let family = Dominance(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let mut arena = Arena::new();
    let naive = compile(&mut arena, dominance::SOURCE, v.naive());
    let plan = FuelPlan::from_reference(&family, &arena, naive, &c)
        .expect("the naive reference is correct");
    println!("PLAN {}: {:?}", v.naive(), plan.sizes());
    let naive_named = check_class(
        &family,
        &mut arena,
        dominance::SOURCE,
        &plan,
        v.naive(),
        GrowthClass::Quadratic,
        SeedsRefused::None,
    );
    assert!(naive_named, "C does not name n^2 for {}", v.naive());
    // C names n log n on the medians for every good reference. On a seed
    // alone it finds some good references too noisy (November 2026: seeds 1
    // and 2 of weighted inversions, seed 2 of pairs-2d and of increasing
    // triples, seed 4 of increasing subsequences, noise margins 0.052 to
    // 0.071) and names n log n on the others. It names no other class.
    let good_refused = match v {
        Variant::WeightedInversions => SeedsRefused::AtMost(2),
        _ => SeedsRefused::AtMost(1),
    };
    let good_named = check_class(
        &family,
        &mut arena,
        dominance::SOURCE,
        &plan,
        v.good(),
        GrowthClass::NLogN,
        good_refused,
    );
    assert!(good_named, "C does not name n log n for {}", v.good());
}

/// The gate, as the module comment says.
fn gate_turns_away_naive(v: Variant) {
    let family = Dominance(v);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let naive = compile(&mut arena, dominance::SOURCE, v.naive());
    let good = compile(&mut arena, dominance::SOURCE, v.good());
    for e in family.examples() {
        for p in [naive, good] {
            run_on(&arena, p, &e.inputs, &e.output, DEFAULT_EXAMPLE_FUEL)
                .unwrap_or_else(|f| panic!("{}: {f:?} on {:?}", family.name(), e.inputs));
        }
    }
    let passed = check_examples(&family, &arena, good, &c).expect("the good reference passes");
    let steps = check_gate(&family, &arena, &passed, &c)
        .unwrap_or_else(|f| panic!("{} on the gate: {f:?}", v.good()))
        .expect("a gate");
    println!("GATE {} at n = {GATE_N}: {steps} steps", v.good());
    assert!(
        steps < GATE_FUEL / 3,
        "{}: {steps} steps on the gate",
        v.good()
    );
    let passed = check_examples(&family, &arena, naive, &c).expect("the naive reference passes");
    assert_eq!(
        check_gate(&family, &arena, &passed, &c),
        Err(CheckFailure::Gate {
            failure: RunFailure::Timeout { fuel: GATE_FUEL }
        }),
        "{} on the gate",
        v.naive()
    );
}

#[test]
fn weighted_inversions() {
    check_variant(Variant::WeightedInversions);
}

#[test]
fn pairs_2d() {
    check_variant(Variant::Pairs2d);
}

#[test]
fn increasing_triples() {
    check_variant(Variant::IncreasingTriples);
}

#[test]
fn increasing_subsequences() {
    check_variant(Variant::IncreasingSubsequences);
}

#[test]
fn gate_weighted_inversions() {
    gate_turns_away_naive(Variant::WeightedInversions);
}

#[test]
fn gate_pairs_2d() {
    gate_turns_away_naive(Variant::Pairs2d);
}

#[test]
fn gate_increasing_triples() {
    gate_turns_away_naive(Variant::IncreasingTriples);
}

#[test]
fn gate_increasing_subsequences() {
    gate_turns_away_naive(Variant::IncreasingSubsequences);
}

#[test]
fn variant_names_are_unique() {
    assert_eq!(dominance::all().len(), Variant::ALL.len());
    let names: std::collections::BTreeSet<String> = Variant::ALL
        .iter()
        .map(|&v| Dominance(v).name().to_owned())
        .collect();
    assert_eq!(names.len(), Variant::ALL.len(), "names are unique");
}
