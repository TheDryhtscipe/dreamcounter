//! The chain family (bead dreamcounter-4593ae) through the whole pipeline:
//! each variant is a valid family, its fuel plan is made from the naive
//! reference, and estimator C puts the naive reference in `n²`. The good
//! references are `n log² n`, a class C does not have; what C says of them
//! is printed and checked in [`check_variant`].
//!
//! The gate: the naive and good references pass every example. The good
//! reference passes each gate input within two thirds of the gate's fuel,
//! and the naive one runs out of fuel on the first input, as does a leaner
//! quadratic program. Two fast programs that a proposer model wrote in a
//! probe, without running them, pass the gate. The good reference with a
//! quicksort that takes its first element as pivot is fast on the random
//! input and runs out of fuel on the input whose `xs` are sorted. The good
//! reference with its merges replaced by merge sorts, still `n log² n`,
//! passes with a third of the fuel to spare. A program that searches only
//! the chain lengths whose least `x` and `y` are below a point passes the
//! other inputs and runs out of fuel on the antichain, with or without
//! skipping repeated points.

mod common;

use common::{check_class, compile, list_len, SeedsRefused};
use dc_core::Arena;
use dc_tasks::families::chains::{self, Chains, GateInput, Variant, GATE_FUEL, GATE_INPUTS};
use dc_tasks::growth::GrowthClass;
use dc_tasks::harness::{
    check_examples, check_gate, check_input_sizes, run_on, validate_family, CheckFailure, FuelPlan,
    HarnessConfig, RunFailure, DEFAULT_EXAMPLE_FUEL,
};
use dc_tasks::task::TaskFamily;

fn check_variant(v: Variant) {
    let family = Chains(v);
    let c = HarnessConfig::default();
    validate_family(&family, &c).expect("the family is valid");
    check_input_sizes(&family, c.seeds, list_len).expect("inputs have size n");
    let mut arena = Arena::new();
    let naive = compile(&mut arena, chains::SOURCE, v.naive());
    let plan = FuelPlan::from_reference(&family, &arena, naive, &c)
        .expect("the naive reference is correct");
    println!("PLAN {}: {:?}", v.naive(), plan.sizes());
    let naive_named = check_class(
        &family,
        &mut arena,
        chains::SOURCE,
        &plan,
        v.naive(),
        GrowthClass::Quadratic,
        // Measured: C names n^2 on every seed of every variant.
        SeedsRefused::AtMost(0),
    );
    assert!(naive_named, "C does not name n^2 for {}", v.naive());
    // C has no n log^2 n class. Between n log n and n^2 it names n log n,
    // or refuses; check_class fails if it names any other class. Measured:
    // it finds no fit on the medians, and each seed alone is too noisy.
    check_class(
        &family,
        &mut arena,
        chains::SOURCE,
        &plan,
        v.good(),
        GrowthClass::NLogN,
        SeedsRefused::Any,
    );
}

/// The examples and the gate, for the references, as the module comment
/// says.
fn gate_turns_away_naive(v: Variant) {
    let family = Chains(v);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let naive = compile(&mut arena, chains::SOURCE, v.naive());
    let good = compile(&mut arena, chains::SOURCE, v.good());
    for e in family.examples() {
        for p in [naive, good] {
            run_on(
                &arena,
                p,
                &e.inputs,
                &e.output,
                c.limits(DEFAULT_EXAMPLE_FUEL),
            )
            .unwrap_or_else(|f| panic!("{}: {f:?} on {:?}", family.name(), e.inputs));
        }
    }
    let passed = check_examples(&family, &arena, good, &c).expect("the good reference passes");
    let steps = check_gate(&family, &arena, &passed, &c)
        .unwrap_or_else(|f| panic!("{} on the gate: {f:?}", v.good()))
        .expect("a gate");
    println!("GATE {}: {steps} steps", v.good());
    assert!(
        steps < GATE_FUEL * 2 / 3,
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
fn count() {
    check_variant(Variant::Count);
}

#[test]
fn longest() {
    check_variant(Variant::Longest);
}

#[test]
fn heaviest() {
    check_variant(Variant::Heaviest);
}

#[test]
fn longest_count() {
    check_variant(Variant::LongestCount);
}

#[test]
fn gate_count() {
    gate_turns_away_naive(Variant::Count);
}

#[test]
fn gate_longest() {
    gate_turns_away_naive(Variant::Longest);
}

#[test]
fn gate_heaviest() {
    gate_turns_away_naive(Variant::Heaviest);
}

#[test]
fn gate_longest_count() {
    gate_turns_away_naive(Variant::LongestCount);
}

#[test]
fn variant_names_are_unique() {
    assert_eq!(chains::all().len(), Variant::ALL.len());
    let names: std::collections::BTreeSet<String> = Variant::ALL
        .iter()
        .map(|&v| Chains(v).name().to_owned())
        .collect();
    assert_eq!(names.len(), Variant::ALL.len(), "names are unique");
}

/// The longest chain by looking at every earlier position, with few steps
/// for each pair of positions: about 10.5. A review wrote one with about
/// 9.3; the gate allows fewer than 4.
const LEAN_QUADRATIC: &str = r#"
let lean xs ys =
  let (_, best) =
    foldl
      (fun (seen, best) (x, y) ->
        let f =
          1 + foldl
                (fun m (a, b, t) -> if a < x then (if b < y then (if m < t then t else m) else m) else m)
                0 seen in
        ((x, y, f) :: seen, if best < f then f else best))
      ([], 0) (zip xs ys) in
  best
"#;

#[test]
fn the_gate_turns_away_a_lean_quadratic_program() {
    let family = Chains(Variant::Longest);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let p = compile(&mut arena, LEAN_QUADRATIC, "lean");
    let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
    // Its steps at 1024 elements, for the record: about n^2 / 2 pairs.
    let input = dc_tasks::task::generate_input(&family, 1024, 0);
    let steps = run_on(
        &arena,
        p,
        &input,
        &family.oracle(&input),
        c.limits(GATE_FUEL),
    )
    .expect("it answers 1024 elements");
    println!(
        "LEAN at 1024: {steps} steps, {:.2} per pair",
        steps as f64 / (1024.0 * 1023.0 / 2.0)
    );
    assert_eq!(
        check_gate(&family, &arena, &passed, &c),
        Err(CheckFailure::Gate {
            index: 0,
            failure: RunFailure::Timeout { fuel: GATE_FUEL }
        })
    );
}

/// A longest-chain program a proposer model wrote in a probe (bead
/// dreamcounter-4593ae), without running it: divide and conquer over
/// positions, with a segment tree over `ys` in an array.
const PROPOSED_LONGEST: &str = r#"
let upd t y v =
  fst (loop 15 (t, y + 16384) (fun k acc ->
    let (a, p) = acc in
    (array_set a p (max (array_get a p) v), p / 2)))

let qry t y =
  snd (loop 14 (y + 16384, 0) (fun k acc ->
    let (r, res) = acc in
    if r mod 2 = 1 then (r / 2, max res (array_get t (r - 1)))
    else (r / 2, res)))

let bucket_sort key items =
  let bs = foldl (fun arr e ->
                    let k = key e in
                    array_set arr k (cons e (array_get arr k)))
                 (array_tabulate 16384 (fun k -> []))
                 (reverse items) in
  concat (array_to_list bs)

let contribute t0 mid s d =
  let r = foldl (fun st e ->
                   let (t, dd) = st in
                   let (x, y, i) = e in
                   if i < mid then (upd t y (array_get dd i), dd)
                   else
                     let q = qry t y in
                     if q + 1 > array_get dd i
                     then (t, array_set dd i (q + 1))
                     else (t, dd))
                (t0, d) s in
  snd r

let cdq t0 = bfix 20 (fun self a ->
  let (lo, hi, s, d) = a in
  if hi - lo <= 1 then d
  else
    let mid = (lo + hi) / 2 in
    let d1 = self (lo, mid, filter (fun e -> let (x, y, i) = e in i < mid) s, d) in
    let d2 = contribute t0 mid s d1 in
    self (mid, hi, filter (fun e -> let (x, y, i) = e in i >= mid) s, d2))

let solve xs ys =
  let n = length xs in
  let items = map (fun e -> let (i, p) = e in let (x, y) = p in (x, y, i))
                  (zip (range 0 n) (zip xs ys)) in
  let s1 = bucket_sort (fun e -> let (x, y, i) = e in 16383 - y) items in
  let s = bucket_sort (fun e -> let (x, y, i) = e in x) s1 in
  let d = cdq (array_tabulate 32768 (fun k -> 0))
              (0, n, s, array_tabulate n (fun k -> 1)) in
  foldl max 0 (array_to_list d)
"#;

/// A heaviest-chain program from the same probe: the same method, with
/// the structure cleared after each pass.
const PROPOSED_HEAVIEST: &str = r#"
let m = 16384

let seg_update t pos v =
  snd (loop 15 (m + pos, t) (fun _ st ->
    let (p, a) = st in
    if p < 1 then st
    else (p / 2, array_set a p (max (array_get a p) v))))

let seg_clear t pos =
  snd (loop 15 (m + pos, t) (fun _ st ->
    let (p, a) = st in
    if p < 1 then st
    else (p / 2, array_set a p 0)))

let seg_query t q =
  snd (loop 15 ((m, m + q), 0) (fun _ st ->
    let (lr, res) = st in
    let (l, r) = lr in
    if l < r then
      let a = if l mod 2 = 1 then max res (array_get t l) else res in
      let l2 = if l mod 2 = 1 then l + 1 else l in
      let b = if r mod 2 = 1 then max a (array_get t (r - 1)) else a in
      let r2 = if r mod 2 = 1 then r - 1 else r in
      ((l2 / 2, r2 / 2), b)
    else st))

let advance xa ya xr st =
  bfix 16400 (fun self s ->
    let (t, rest) = s in
    let (ls, rest2) = rest in
    let (ins, arr) = rest2 in
    match ls with
    [] -> s
    | h :: tl ->
      if array_get xa h < xr
      then self (seg_update t (array_get ya h) (array_get arr h), (tl, (h :: ins, arr)))
      else s) st

let cross_step xa ya st r =
  let st2 = advance xa ya (array_get xa r) st in
  let (t, rest) = st2 in
  let (ls, rest2) = rest in
  let (ins, arr) = rest2 in
  let q = seg_query t (array_get ya r) in
  (t, (ls, (ins, array_set arr r (max (array_get arr r) q))))

let cross xa ya t arr ls rs =
  let res = foldl (cross_step xa ya) (t, (ls, ([], arr))) rs in
  let (t1, rest) = res in
  let (ls1, rest2) = rest in
  let (ins, arr1) = rest2 in
  (foldl (fun tt i -> seg_clear tt (array_get ya i)) t1 ins, arr1)

let cdq xa ya wa =
  bfix 40 (fun self s ->
    let (lo, r1) = s in
    let (hi, r2) = r1 in
    let (ls, r3) = r2 in
    let (t, arr) = r3 in
    if hi - lo = 1 then
      (t, array_set arr lo (array_get arr lo + array_get wa lo))
    else
      let mid = (lo + hi) / 2 in
      let l = filter (fun i -> i < mid) ls in
      let r = filter (fun i -> not (i < mid)) ls in
      let s1 = self (lo, (mid, (l, (t, arr)))) in
      let (t1, arr1) = s1 in
      let (t2, arr2) = cross xa ya t1 arr1 l r in
      self (mid, (hi, (r, (t2, arr2)))))

let solve xs ys ws =
  let n = length xs in
  if n = 0 then 0
  else
    let xa = array_of_list xs in
    let ya = array_of_list ys in
    let wa = array_of_list ws in
    let buckets = foldl (fun b i ->
        let x = array_get xa i in
        array_set b x (i :: array_get b x))
      (array_tabulate m (fun _ -> []))
      (filter (fun i -> i < n) (range 0 n)) in
    let order = concat (array_to_list buckets) in
    let init = (0, (n, (order, (array_tabulate (2 * m) (fun _ -> 0), array_tabulate n (fun _ -> 0))))) in
    let res = cdq xa ya wa init in
    foldl max 0 (array_to_list (snd res))
"#;

/// A program a proposer wrote for `v` passes the examples and the gate.
/// One test per program, as each takes minutes of CPU.
fn proposed_program_passes_the_gate(v: Variant, source: &str) {
    let c = HarnessConfig::default();
    let family = Chains(v);
    let mut arena = Arena::new();
    let p = compile(&mut arena, source, "solve");
    let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
    let steps = check_gate(&family, &arena, &passed, &c)
        .unwrap_or_else(|f| panic!("{} on the gate: {f:?}", family.name()))
        .expect("a gate");
    println!("PROPOSED {}: {steps} steps", family.name());
}

#[test]
fn a_proposers_longest_program_passes_the_gate() {
    proposed_program_passes_the_gate(Variant::Longest, PROPOSED_LONGEST);
}

#[test]
fn a_proposers_heaviest_program_passes_the_gate() {
    proposed_program_passes_the_gate(Variant::Heaviest, PROPOSED_HEAVIEST);
}

/// The good reference with its two merges of sorted lists replaced by a
/// merge sort of the joined lists: still `n log² n`, with a larger constant,
/// as a proposer might well write it. It passes the gate on `v` with a third
/// of the fuel to spare. One test per variant, as each takes minutes of CPU.
fn the_gate_lets_through_a_less_careful_n_log2_n_program(v: Variant) {
    let joined_points = "merge_by by_x ls (self by_pos))";
    let tagged = "merge_by (fun (a, wa, _, _, _, _, _) (b, wb, _, _, _, _, _) -> a < b || (a = b && wa <= wb))\n      (tag 1 l) (tag 0 r)";
    assert!(chains::SOURCE.contains(joined_points) && chains::SOURCE.contains(tagged));
    let source = chains::SOURCE
        .replace(joined_points, "msort_by by_x (append ls (self by_pos)))")
        .replace(
            tagged,
            "msort_by (fun (a, wa, _, _, _, _, _) (b, wb, _, _, _, _, _) -> a < b || (a = b && wa <= wb))\n      (append (tag 1 l) (tag 0 r))",
        );
    let c = HarnessConfig::default();
    let family = Chains(v);
    let mut arena = Arena::new();
    let p = compile(&mut arena, &source, v.good());
    let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
    let steps = check_gate(&family, &arena, &passed, &c)
        .unwrap_or_else(|f| panic!("{} on the gate: {f:?}", family.name()))
        .expect("a gate");
    println!("CARELESS {}: {steps} steps", family.name());
    // Room to spare: the gate must allow about one and a half times this
    // program's steps, so that a program a little slower still passes.
    assert!(
        steps <= GATE_FUEL * 2 / 3,
        "{}: {steps} steps of {GATE_FUEL}",
        family.name()
    );
}

#[test]
fn the_gate_lets_through_a_less_careful_longest_program() {
    the_gate_lets_through_a_less_careful_n_log2_n_program(Variant::Longest);
}

#[test]
fn the_gate_lets_through_a_less_careful_longest_count_program() {
    the_gate_lets_through_a_less_careful_n_log2_n_program(Variant::LongestCount);
}

/// A sort that is `n²` on sorted input: a quicksort with the first element
/// as pivot.
const FIRST_PIVOT_QUICKSORT: &str = r#"
let msort_by le xs =
  bfix (length xs + 1)
    (fun self ys ->
      match ys with
      | [] -> []
      | p :: rest ->
        append (self (filter (fun y -> le y p) rest))
          (p :: self (filter (fun y -> not (le y p)) rest)))
    xs

"#;

#[test]
fn the_gate_turns_away_a_program_that_is_quadratic_only_on_sorted_xs() {
    // The good reference sorts each later half back into position order
    // from the order of xs. With xs sorted, that order is already position
    // order: the first-pivot quicksort's worst case.
    let anchor = "-- The step that passes values across the halves.";
    assert!(chains::SOURCE.contains(anchor));
    let source = chains::SOURCE
        .replace("let msort_by le xs =", "let msort_by_unused le xs =")
        .replace(anchor, &format!("{FIRST_PIVOT_QUICKSORT}{anchor}"));
    let family = Chains(Variant::Longest);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let p = compile(&mut arena, &source, Variant::Longest.good());
    let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
    let gate = family.gate().expect("a gate");
    let random = &gate.inputs[0];
    let steps = run_on(
        &arena,
        p,
        &random.inputs,
        &random.output,
        c.limits(GATE_FUEL),
    )
    .expect("it answers the random input");
    println!("QUICKSORT on the random gate input: {steps} steps");
    let sorted = GATE_INPUTS
        .iter()
        .position(|&g| g == GateInput::XsSorted)
        .expect("an input with xs sorted");
    // On sorted xs its recursion is n deep, and with the default memory
    // limit it fills that limit before it runs out of fuel. Either stops it
    // on this input.
    let result = check_gate(&family, &arena, &passed, &c);
    assert!(
        matches!(
            result,
            Err(CheckFailure::Gate {
                index,
                failure: RunFailure::Timeout { .. } | RunFailure::MemoryLimit { .. }
            }) if index == sorted
        ),
        "{result:?}"
    );
}

/// A longest-chain program a review wrote: it keeps the points of each
/// chain length in a list, with the least x and least y of each, and looks
/// for a point below only in a length whose least x and y are both below.
/// It is fast on random and sorted inputs, and `n²` when many points are
/// above the least x and y of a long list that has no point below them.
const LAYERED: &str = r#"
let layered xs ys =
  let n = length xs in
  let big = n + 1 in
  let st0 = (array_tabulate (n + 2) (fun _ -> []),
             (array_tabulate (n + 2) (fun _ -> big),
              (array_tabulate (n + 2) (fun _ -> big), 0))) in
  let hit x y bucket =
    bfix (length bucket + 1) (fun self l ->
      match l with
      | [] -> false
      | e :: rest -> let (a, b) = e in if a < x then (if b < y then true else self rest) else self rest) bucket in
  let find lv mx my top x y =
    bfix (top + 2) (fun self lev ->
      if lev < 1 then 0
      else if array_get mx lev < x then
        (if array_get my lev < y then (if hit x y (array_get lv lev) then lev else self (lev - 1)) else self (lev - 1))
      else self (lev - 1)) top in
  let final = foldl (fun st p ->
      let (x, y) = p in
      let (lv, r1) = st in
      let (mx, r2) = r1 in
      let (my, top) = r2 in
      let f = 1 + find lv mx my top x y in
      (array_set lv f ((x, y) :: array_get lv f),
       (array_set mx f (min x (array_get mx f)),
        (array_set my f (min y (array_get my f)), max top f)))) st0 (zip xs ys) in
  snd (snd (snd final))
"#;

#[test]
fn the_gate_turns_away_a_program_that_is_quadratic_on_an_antichain() {
    let family = Chains(Variant::Longest);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let antichain = GATE_INPUTS
        .iter()
        .position(|&g| g == GateInput::Antichain)
        .expect("an antichain input");
    // The same program, after it drops each point equal to the one before:
    // that changes no answer, and makes runs of equal points cheap.
    let skipping = format!("{LAYERED}{SKIP_REPEATS}");
    for (source, entry) in [(LAYERED, "layered"), (skipping.as_str(), "skipping")] {
        let p = compile(&mut arena, source, entry);
        let passed = check_examples(&family, &arena, p, &c).expect("it passes the examples");
        assert_eq!(
            check_gate(&family, &arena, &passed, &c),
            Err(CheckFailure::Gate {
                index: antichain,
                failure: RunFailure::Timeout { fuel: GATE_FUEL }
            }),
            "{entry}"
        );
    }
}

const SKIP_REPEATS: &str = r#"
let skipping xs ys =
  let kept =
    foldr
      (fun p acc ->
        match acc with
        | [] -> [p]
        | q :: _ ->
          let (a, b) = p in
          let (c, d) = q in
          if a = c && b = d then acc else p :: acc)
      [] (zip xs ys) in
  layered (map fst kept) (map snd kept)
"#;
