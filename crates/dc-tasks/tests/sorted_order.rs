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
use dc_tasks::harness::{
    check_input_sizes, run_on, validate_family, FuelPlan, HarnessConfig, RunFailure,
};
use dc_tasks::task::{generate_input, TaskFamily};

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

/// Two wrong pair-sums solutions, compiled after [`sorted_order::SOURCE`].
/// Each is right whenever at most one pair sums to exactly `t` and no two
/// equal elements do, so only inputs with several exact pairs tell them from
/// a right solution.
const PAIR_SUMS_MUTANTS: &str = r#"
-- The pairs that sum to less than t, plus 1 if any pair sums to exactly t.
let pair_sums_one_exact xs t =
  let below =
    bfix (length xs + 1)
      (fun self ys ->
        match ys with
        | [] -> 0
        | x :: rest -> count_where (fun y -> x + y < t) rest + self rest)
      xs in
  let exact =
    bfix (length xs + 1)
      (fun self ys ->
        match ys with
        | [] -> 0
        | x :: rest -> count_where (fun y -> x + y = t) rest + self rest)
      xs in
  below + (if exact > 0 then 1 else 0)

-- The pairs that sum to less than t, and those that sum to exactly t only
-- when their elements differ.
let pair_sums_exact_unequal xs t =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> 0
      | x :: rest ->
        count_where (fun y -> x + y < t || (x + y = t && (x < y || y < x))) rest
          + self rest)
    xs
"#;

/// The generated inputs alone reject both pair-sums mutants, each on several
/// of them; the examples are not needed. Each mutant is a near miss: it
/// passes every example but those with several exact pairs (3 and 4).
#[test]
fn pair_sums_sweep_rejects_wrong_exact_pair_counts() {
    let family = SortedOrder(Variant::PairSums);
    let c = HarnessConfig::default();
    let source = format!("{}\n{PAIR_SUMS_MUTANTS}", sorted_order::SOURCE);
    let mut arena = Arena::new();
    let mut too_few = Vec::new();
    for entry in ["pair_sums_one_exact", "pair_sums_exact_unequal"] {
        let p = compile(&mut arena, &source, entry);
        let failed_examples: Vec<usize> = family
            .examples()
            .iter()
            .enumerate()
            .filter(|(_, e)| run_on(&arena, p, &e.inputs, &e.output, c.example_fuel).is_err())
            .map(|(i, _)| i)
            .collect();
        assert!(
            !failed_examples.is_empty() && failed_examples.iter().all(|&i| i == 3 || i == 4),
            "{entry} fails examples {failed_examples:?}"
        );
        let mut wrong = Vec::new();
        for n in family.sizes() {
            for seed in 0..c.seeds {
                let inputs = generate_input(&family, n, seed);
                let expected = family.oracle(&inputs);
                match run_on(&arena, p, &inputs, &expected, c.max_fuel) {
                    Ok(_) => {}
                    Err(RunFailure::WrongOutput { expected, actual }) => {
                        wrong.push(format!("n = {n}, seed {seed}: {actual}, not {expected}"));
                    }
                    Err(other) => panic!("{entry} at n = {n}, seed {seed}: {other:?}"),
                }
            }
        }
        println!(
            "MUTANT {entry}: wrong on {} generated inputs: {wrong:?}",
            wrong.len()
        );
        if wrong.len() < 3 {
            too_few.push(format!(
                "{entry} is wrong on only {} generated inputs",
                wrong.len()
            ));
        }
    }
    assert!(too_few.is_empty(), "{too_few:?}");
}
