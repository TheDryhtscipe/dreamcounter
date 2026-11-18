//! The dominance-counting family (decision D25) through the whole pipeline:
//! each variant is a valid family, its fuel plan is made from the naive
//! reference, and estimator C puts the naive reference in `n²` and the good
//! one in `n log n`. On each seed alone C names the class or refuses as too
//! noisy, on at most the seeds [`check_variant`] allows.
//!
//! The gate: the naive and good references pass every example. The good
//! reference passes each gate input within a third of the gate's fuel, and
//! the naive one runs out of fuel on the first input. A program that is
//! `n²` only on sorted input (it inserts into an unbalanced search tree)
//! passes the examples and the random input, and runs out of fuel on the
//! sorted ones.

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
            index: 0,
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

/// Weighted inversions with an unbalanced search tree: `n log n` on random
/// input, `n²` on sorted input. Each node holds a value, its position, and
/// the count and position sum of its subtree.
const TREE_WEIGHTED_INVERSIONS: &str = r#"
let totals t =
  match t with
  | Leaf -> (0, 0)
  | Node (_, p, _) -> let (_, _, c, s) = p in (c, s)

let greater t x =
  bfix (size_hint + 1)
    (fun self t ->
      match t with
      | Leaf -> (0, 0)
      | Node (l, p, r) ->
        let (v, i, _, _) = p in
        if v > x then
          let (rc, rs) = totals r in
          let (gc, gs) = self l in
          (gc + rc + 1, gs + rs + i)
        else self r)
    t

let insert t x j =
  bfix (size_hint + 1)
    (fun self t ->
      match t with
      | Leaf -> Node (Leaf, (x, j, 1, j), Leaf)
      | Node (l, p, r) ->
        let (v, vi, c, s) = p in
        if x < v then Node (self l, (v, vi, c + 1, s + j), r)
        else Node (l, (v, vi, c + 1, s + j), self r))
    t

let tree_weighted_inversions xs =
  let (_, acc) =
    foldl
      (fun (t, acc) (x, j) ->
        let (gc, gs) = greater t x in
        (insert t x j, acc + gc * j - gs))
      (Leaf, 0) (zip xs (range 0 (length xs))) in
  acc
"#;

#[test]
fn the_gate_turns_away_a_program_that_is_quadratic_only_on_sorted_input() {
    let family = Dominance(Variant::WeightedInversions);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let p = compile(
        &mut arena,
        &TREE_WEIGHTED_INVERSIONS.replace("size_hint", "100000"),
        "tree_weighted_inversions",
    );
    let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
    let gate = family.gate().expect("a gate");
    let random = &gate.inputs[0];
    let steps = run_on(&arena, p, &random.inputs, &random.output, GATE_FUEL)
        .expect("it answers the random input");
    println!("TREE on the random gate input: {steps} steps");
    assert_eq!(
        check_gate(&family, &arena, &passed, &c),
        Err(CheckFailure::Gate {
            index: 1,
            failure: RunFailure::Timeout { fuel: GATE_FUEL }
        })
    );
}
