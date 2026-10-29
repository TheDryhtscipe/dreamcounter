//! The collapse family (bead P3.7) through the whole pipeline: each variant
//! is a valid family; its fuel plan is made from the naive reference; the
//! naive and good references pass the examples and the oracle on every
//! generated input; and estimator C names `n²` for the naive reference and
//! `n` for the good one, on the medians and on each seed alone. A good
//! reference that stops after a few rewrites per element fails the examples
//! and the generated inputs.

mod common;

use common::{check_class, compile, list_len};
use dc_core::{Arena, TermId};
use dc_eval::Value;
use dc_tasks::families::collapse::{self, Collapse, Variant};
use dc_tasks::growth::GrowthClass;
use dc_tasks::harness::{
    check_examples, check_generated, check_input_sizes, validate_family, CheckFailure, FuelPlan,
    HarnessConfig,
};
use dc_tasks::task::{generate_input, TaskFamily};

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

/// A wrong good reference: the stack fold, but each element pushed makes at
/// most `cap` rewrites, after which it is pushed as it is. It is right on
/// every input whose chains of rewrites from one element are no longer than
/// `cap`, and costs `n`, so only an input with a longer chain rejects it.
fn capped_source(cap: u64) -> String {
    format!(
        "{}
let collapse_push_capped redex replace st x =
  bfix (length st + 2)
    (fun self (st, x, k) ->
      match st with
      | [] -> ([x], k)
      | t :: rest ->
        if k < {cap} then
          if redex t x then
            match replace t x with
            | [] -> (rest, k + 1)
            | y :: _ -> self (rest, y, k + 1)
          else (x :: st, k)
        else (x :: st, k))
    (st, x, 0)

let collapse_capped_count redex replace xs =
  let (st, k) =
    foldl
      (fun (st, k) x ->
        let (st2, j) = collapse_push_capped redex replace st x in (st2, k + j))
      ([], 0) xs in
  (reverse st, k)

let merge_capped xs = fst (collapse_capped_count merge_redex merge_replace xs)

let absorb_capped xs = fst (collapse_capped_count absorb_redex absorb_replace xs)

let absorb_count_capped xs = snd (collapse_capped_count absorb_redex absorb_replace xs)
",
        collapse::SOURCE
    )
}

/// The capped stack fold of `v` with `cap`, compiled.
fn capped(arena: &mut Arena, v: Variant, cap: u64) -> TermId {
    let entry = match v {
        Variant::Merge => "merge_capped",
        Variant::Absorb => "absorb_capped",
        Variant::AbsorbCount => "absorb_count_capped",
        Variant::Cancel => panic!("cancel makes at most one rewrite per element"),
    };
    compile(arena, &capped_source(cap), entry)
}

/// The longest chain of rewrites from one element of `xs`, by the stack
/// fold.
fn longest_chain(v: Variant, xs: &[i64]) -> u64 {
    let mut stack: Vec<i64> = Vec::new();
    let mut longest = 0;
    for &x in xs {
        let (mut c, mut chain) = (x, 0);
        loop {
            let r = match (v, stack.last().copied()) {
                (Variant::Merge, Some(t)) if t == c => c + 1,
                (Variant::Absorb | Variant::AbsorbCount, Some(t)) if t >= c + 2 => (t + c) % 10,
                _ => break,
            };
            stack.pop();
            c = r;
            chain += 1;
        }
        stack.push(c);
        longest = longest.max(chain);
    }
    longest
}

/// The capped stack fold with a cap of 5, and with the largest cap the
/// examples reject, fails the examples alone; with a cap of 5, and with the
/// largest cap the sweep rejects, it fails the generated inputs alone. One
/// more than each largest cap passes, which is where a capped solution can
/// hide. At each size, the cap one below the longest chain in that size's
/// inputs fails there, and that chain is at least 6.
fn check_capped(v: Variant, examples_largest: u64, sweep_largest: u64) {
    let family = Collapse(v);
    let c = HarnessConfig::default();
    let mut arena = Arena::new();
    let caps = [
        5,
        examples_largest,
        examples_largest + 1,
        sweep_largest,
        sweep_largest + 1,
    ];
    for cap in caps {
        let p = capped(&mut arena, v, cap);
        let examples = check_examples(&family, &arena, p, &c);
        println!("CAPPED {v:?} cap {cap}: examples {examples:?}");
        let generated = check_generated(&family, &arena, p, &family.sizes(), &c);
        println!("CAPPED {v:?} cap {cap}: generated {generated:?}");
        if cap <= examples_largest {
            assert!(
                matches!(examples, Err(CheckFailure::Example { .. })),
                "{v:?}: the examples pass a cap of {cap}"
            );
        } else {
            assert!(
                examples.is_ok(),
                "{v:?}: the examples reject a cap of {cap}"
            );
        }
        if cap <= sweep_largest {
            assert!(
                matches!(generated, Err(CheckFailure::Generated { .. })),
                "{v:?}: the generated inputs pass a cap of {cap}"
            );
        } else {
            assert!(
                generated.is_ok(),
                "{v:?}: the generated inputs reject a cap of {cap}"
            );
        }
    }
    let mut chains = Vec::new();
    for n in family.sizes() {
        let chain = (0..c.seeds)
            .map(|seed| {
                let xs = generate_input(&family, n, seed);
                match &xs[0] {
                    Value::List(l) => longest_chain(
                        v,
                        &l.iter()
                            .map(|x| x.as_int().expect("an Int"))
                            .collect::<Vec<i64>>(),
                    ),
                    other => panic!("{other:?}"),
                }
            })
            .min()
            .expect("seeds");
        chains.push((n, chain));
        assert!(chain >= 6, "{v:?} at {n}: the longest chain is {chain}");
        let p = capped(&mut arena, v, chain - 1);
        assert!(
            matches!(
                check_generated(&family, &arena, p, &[n], &c),
                Err(CheckFailure::Generated { .. })
            ),
            "{v:?} at {n}: a cap of {} passes",
            chain - 1
        );
    }
    println!("CHAINS {v:?}: {chains:?}");
    let longest = chains.iter().map(|&(_, k)| k).max().expect("sizes");
    assert_eq!(longest - 1, sweep_largest, "{v:?}: {chains:?}");
}

#[test]
fn capped_merge_is_rejected() {
    check_capped(Variant::Merge, 8, 5);
}

#[test]
fn capped_absorb_is_rejected() {
    check_capped(Variant::Absorb, 7, 7);
}

#[test]
fn capped_absorb_count_is_rejected() {
    check_capped(Variant::AbsorbCount, 7, 7);
}
