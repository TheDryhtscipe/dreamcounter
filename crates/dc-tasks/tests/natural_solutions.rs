//! Natural solutions (bead dreamcounter-0ea5b7): correct programs written
//! the ordinary way, with plain `if`s whose branches cost different amounts,
//! unlike the references, which are written so that their steps depend on
//! `n` alone. Estimator C must name their class on the medians, as for the
//! references, and on every seed alone, unless a test allows C to refuse
//! some seeds as too noisy. C never names another class (see
//! [`check_class`]).
//!
//! Two tests guard bead dreamcounter-0ea5b7, where C used to refuse these
//! solutions: `sorted_order_pair_sums` and
//! `subsequence_count_zigzag_quadratic`. The first fails without either half
//! of that change: generators that keep a seed's kind and threshold at every
//! size, and C taking out each seed's level. The second fails without the
//! generators. The other tests were named before that change too; they
//! check that natural solutions across the families still get their class.
//!
//! Insertion sort and the plain quadratic close-steps solution are the
//! exceptions on the medians, and these two, pair sums and the quadratic
//! zigzag solution on a seed alone (see their tests). Insertion sort is the
//! one the growth module documents: its cost varies with each input's order
//! by about `0.7 / √n`, so C may refuse it, but must not name a class other
//! than `n²`.

#[allow(dead_code)]
mod common;

use common::{check_class, compile, single_seed_noise, SeedsRefused};
use dc_core::Arena;
use dc_tasks::families::collapse::{self, Collapse};
use dc_tasks::families::sorted_order::{self, SortedOrder};
use dc_tasks::families::subsequence_count::{self, SubsequenceCount};
use dc_tasks::growth::{ClassSelect, GrowthClass, SizeObservation, Sweep, DEFAULT_SIZES};
use dc_tasks::harness::{FuelPlan, HarnessConfig};
use dc_tasks::task::TaskFamily;

/// Natural sorted-order solutions, compiled after [`sorted_order::SOURCE`].
const SORTED_ORDER: &str = r#"
let insert x ys =
  bfix (length ys + 1)
    (fun self ys ->
      match ys with
      | [] -> [x]
      | y :: rest -> if x <= y then x :: ys else y :: self rest)
    ys

let insertion_sort xs = foldl (fun acc x -> insert x acc) [] xs

let inversions_natural xs =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> 0
      | x :: rest -> length (filter (fun y -> x > y) rest) + self rest)
    xs

let pair_sums_natural xs t =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> 0
      | x :: rest -> length (filter (fun y -> x + y <= t) rest) + self rest)
    xs

let distinct_natural xs =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> 0
      | x :: rest ->
        match rest with
        | [] -> 1
        | y :: _ -> if x = y then self rest else 1 + self rest)
    (msort xs)

let min_gap_natural xs =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> max_int
      | x :: rest ->
        match rest with
        | [] -> max_int
        | y :: _ -> let m = self rest in if y - x < m then y - x else m)
    (msort xs)

let lower_median_insertion xs =
  if length xs = 0 then 0 else nth (insertion_sort xs) ((length xs - 1) / 2)
"#;

/// Natural subsequence-count solutions, compiled after
/// [`subsequence_count::SOURCE`].
const SUBSEQUENCE_COUNT: &str = r#"
let close_steps_quadratic_natural xs =
  let ends =
    foldl
      (fun ends x ->
        let f = foldl (fun acc (y, g) -> if absdiff x y <= 2 then addm acc g else acc) 1 ends in
        (x, f) :: ends)
      [] xs in
  foldl (fun acc (_, g) -> addm acc g) 1 ends

let zigzag_quadratic_natural xs =
  let ends =
    foldl
      (fun ends x ->
        let (u, d) =
          foldl
            (fun (u, d) (y, uy, dy) ->
              if y < x then (addm u (addm 1 dy), d)
              else if y > x then (u, addm d (addm 1 uy))
              else (u, d))
            (0, 0) ends in
        (x, u, d) :: ends)
      [] xs in
  foldl (fun acc (_, u, d) -> addm acc (addm 1 (addm u d))) 1 ends

let zigzag_natural xs =
  let zero10 = array_tabulate 10 (fun _ -> 0) in
  let (_, _, total) =
    foldl
      (fun (a, b, total) x ->
        let up = loop 10 0 (fun v acc -> if v < x then addm acc (array_get a v) else acc) in
        let down = loop 10 0 (fun v acc -> if v > x then addm acc (array_get b v) else acc) in
        ( array_set a x (addm (array_get a x) (addm 1 down)),
          array_set b x (addm (array_get b x) (addm 1 up)),
          addm total (addm 1 (addm up down)) ))
      (zero10, zero10, 1) xs in
  total
"#;

/// Natural collapse solutions, compiled after [`collapse::SOURCE`].
const COLLAPSE: &str = r#"
let merge_natural xs =
  reverse
    (foldl
      (fun st x ->
        bfix (length st + 2)
          (fun self (st, x) ->
            match st with
            | [] -> [x]
            | t :: rest -> if t = x then self (rest, x + 1) else x :: st)
          (st, x))
      [] xs)

let cancel_natural xs =
  reverse
    (foldl
      (fun st x ->
        match st with
        | [] -> [x]
        | t :: rest -> if t + x = 0 then rest else x :: st)
      [] xs)
"#;

/// Whether C names `expected` on the medians of `entry`, measured under
/// the plan made from `plan_entry`. Panics if C names another class on the
/// medians or on a seed alone, or refuses more seeds alone than `refused`
/// allows.
fn named<F: TaskFamily>(
    family: &F,
    source: &str,
    plan_entry: &str,
    entry: &str,
    expected: GrowthClass,
    refused: SeedsRefused,
) -> bool {
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let reference = compile(&mut arena, source, plan_entry);
    let plan =
        FuelPlan::from_reference(family, &arena, reference, &c).expect("the reference is correct");
    check_class(family, &mut arena, source, &plan, entry, expected, refused)
}

fn sorted_order_named(
    v: sorted_order::Variant,
    entry: &str,
    expected: GrowthClass,
    refused: SeedsRefused,
) -> bool {
    let source = format!("{}\n{SORTED_ORDER}", sorted_order::SOURCE);
    named(
        &SortedOrder(v),
        &source,
        v.naive(),
        entry,
        expected,
        refused,
    )
}

fn subsequence_count_named(
    v: subsequence_count::Variant,
    entry: &str,
    expected: GrowthClass,
    refused: SeedsRefused,
) -> bool {
    let source = format!("{}\n{SUBSEQUENCE_COUNT}", subsequence_count::SOURCE);
    named(
        &SubsequenceCount(v),
        &source,
        v.quadratic(),
        entry,
        expected,
        refused,
    )
}

fn collapse_named(
    v: collapse::Variant,
    entry: &str,
    expected: GrowthClass,
    refused: SeedsRefused,
) -> bool {
    let source = format!("{}\n{COLLAPSE}", collapse::SOURCE);
    named(&Collapse(v), &source, v.naive(), entry, expected, refused)
}

#[test]
fn sorted_order_inversions() {
    use sorted_order::Variant::Inversions;
    assert!(sorted_order_named(
        Inversions,
        "inversions_natural",
        GrowthClass::Quadratic,
        SeedsRefused::None,
    ));
}

/// On a seed alone, C finds two seeds too noisy (seeds 0 and 3 in November
/// 2026, noise margin 0.055) and names `n²` on the other three.
#[test]
fn sorted_order_pair_sums() {
    use sorted_order::Variant::PairSums;
    assert!(sorted_order_named(
        PairSums,
        "pair_sums_natural",
        GrowthClass::Quadratic,
        SeedsRefused::AtMost(2),
    ));
}

#[test]
fn sorted_order_distinct() {
    use sorted_order::Variant::Distinct;
    assert!(sorted_order_named(
        Distinct,
        "distinct_natural",
        GrowthClass::NLogN,
        SeedsRefused::None,
    ));
}

#[test]
fn sorted_order_min_gap() {
    use sorted_order::Variant::MinGap;
    assert!(sorted_order_named(
        MinGap,
        "min_gap_natural",
        GrowthClass::NLogN,
        SeedsRefused::None,
    ));
}

/// Refusing insertion sort is allowed, on the medians and on every seed
/// alone; naming another class is not. In November 2026 C refused every
/// seed alone as too noisy (noise margin 0.31).
#[test]
fn sorted_order_lower_median_by_insertion_sort() {
    use sorted_order::Variant::LowerMedian;
    let named = sorted_order_named(
        LowerMedian,
        "lower_median_insertion",
        GrowthClass::Quadratic,
        SeedsRefused::Any,
    );
    println!("lower_median_insertion named n^2: {named}");
}

/// Refusing is allowed; naming another class is not. Its cost depends on
/// how many earlier values are within 2 of each, which varies most at the
/// smallest size, 8: in November 2026 its `n²` misfit was 0.0112 with
/// seed-centred medians (0.045 before), just above the tolerance, and drawing
/// each seed's offset once rather than at each size left it at 0.0112. On a
/// seed alone, C then found three seeds too noisy (seeds 2, 3 and 4, noise
/// margin 0.059) and named `n²` on the other two.
#[test]
fn subsequence_count_close_steps_quadratic() {
    use subsequence_count::Variant::CloseSteps;
    let named = subsequence_count_named(
        CloseSteps,
        "close_steps_quadratic_natural",
        GrowthClass::Quadratic,
        SeedsRefused::AtMost(3),
    );
    println!("close_steps_quadratic_natural named n^2: {named}");
}

/// On a seed alone, C finds two seeds too noisy (seeds 1 and 2 in November
/// 2026, noise margin 0.20) and names `n²` on the other three.
#[test]
fn subsequence_count_zigzag_quadratic() {
    use subsequence_count::Variant::Zigzag;
    assert!(subsequence_count_named(
        Zigzag,
        "zigzag_quadratic_natural",
        GrowthClass::Quadratic,
        SeedsRefused::AtMost(2),
    ));
}

#[test]
fn subsequence_count_zigzag() {
    use subsequence_count::Variant::Zigzag;
    assert!(subsequence_count_named(
        Zigzag,
        "zigzag_natural",
        GrowthClass::Linear,
        SeedsRefused::None,
    ));
}

#[test]
fn collapse_merge() {
    use collapse::Variant::Merge;
    assert!(collapse_named(
        Merge,
        "merge_natural",
        GrowthClass::Linear,
        SeedsRefused::None,
    ));
}

#[test]
fn collapse_cancel() {
    use collapse::Variant::Cancel;
    assert!(collapse_named(
        Cancel,
        "cancel_natural",
        GrowthClass::Linear,
        SeedsRefused::None,
    ));
}

/// The noise of one seed that the check on each seed alone uses is C's own
/// noise on a sweep in seed order: seeds four levels apart, with a small
/// spread that changes from size to size on top. Without centring it would
/// be about the spread of the levels.
#[test]
fn single_seed_noise_is_cs_centred_noise() {
    let levels = [-0.3, -0.1, 0.1, 0.3];
    let k = levels.len();
    let obs: Vec<SizeObservation> = DEFAULT_SIZES
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let steps = (0..k)
                .map(|j| {
                    let wobble = 0.01 * (((3 * i + 5 * j) % 7) as f64 - 3.0);
                    (1e6 * n as f64 * (levels[j] + wobble).exp()).round() as u64
                })
                .collect();
            SizeObservation::completed(n, steps)
        })
        .collect();
    let in_order = Sweep::in_seed_order(obs.clone()).expect("a sweep in seed order");
    let fit = ClassSelect::default().fit(&in_order).expect("C fits");
    assert!(fit.seed_centred);
    let one = single_seed_noise(&in_order, &fit.window);
    let from_one = (std::f64::consts::FRAC_PI_2 * one * one / k as f64).sqrt();
    let c = fit.noise.expect("seeds");
    assert!((from_one - c).abs() < 1e-9, "{from_one} against C's {c}");
    let any = Sweep::new(obs).expect("a sweep");
    let raw = single_seed_noise(&any, &fit.window);
    assert!(raw > 0.2 && one < 0.05, "raw {raw}, centred {one}");
}
