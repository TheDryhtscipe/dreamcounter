//! The corpus-level run (bead P3.10): every oracle against every generator
//! across the sweep, for every variant of every development family in
//! [`families::development`].
//!
//! For each variant, at [`CORPUS_SEEDS`] seeds rather than the
//! [`DEFAULT_SEEDS`] of a sweep: the family is valid ([`validate_family`]),
//! its inputs have size `n`, and its inputs at the largest size differ for
//! every seed. Every reference has the family's signature, passes its
//! examples, and agrees with the oracle on every generated input it is run
//! on, under the default fuel cap. By default the good reference runs at
//! every size and the slower ones at the lower half of the sweep;
//! `every_reference_at_every_size`, ignored by default, runs all of them at
//! every size. An oracle that disagrees with the good reference on the last
//! seed only fails the run (see
//! [`a_wrong_answer_on_the_last_seed_fails_the_run`]).
//!
//! The family tests check the growth classes of the references; this run
//! checks correctness only, on more inputs. `corpus_cost_report`, ignored by
//! default, prints the references' median steps per size and C's verdict on
//! them, for the results log.

// This run uses only the compile helpers of the shared module.
#[allow(dead_code)]
mod common;

use std::collections::BTreeSet;
use std::thread;

use common::{compile, list_len};
use dc_core::{Arena, Scheme};
use dc_eval::Value;
use dc_tasks::families::{self, CorpusVariant, Reference};
use dc_tasks::growth::{ClassSelect, SizeObservation, Sweep};
use dc_tasks::harness::{
    check_examples, check_generated, check_input_sizes, run_on, validate_family, HarnessConfig,
    DEFAULT_SEEDS,
};
use dc_tasks::rng::TaskRng;
use dc_tasks::task::{generate_input, Example, Inputs, TaskFamily};

/// The seeds of the corpus run, `0..CORPUS_SEEDS`.
const CORPUS_SEEDS: u64 = 32;

/// The development families: collapse, sorted order, subsequence count.
const FAMILIES: usize = 3;

/// The configuration of the corpus run: the harness defaults at
/// [`CORPUS_SEEDS`] seeds.
fn corpus_config() -> HarnessConfig {
    HarnessConfig {
        seeds: CORPUS_SEEDS,
        ..HarnessConfig::default()
    }
}

/// Which sizes the slower references are run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    /// Every reference at every size of the sweep it can finish at.
    Whole,
    /// The good reference (the last) at every size; the others at the lower
    /// half of the sweep only, where they are cheap.
    LowerHalf,
}

/// The sizes of `family`'s sweep that `r` is run at: those up to
/// `r.max_n`, and under [`Coverage::LowerHalf`], unless `r` is the good
/// reference, only the lower half of the sweep, its first `⌊k/2⌋` of `k`
/// sizes.
fn sizes_for(family: &dyn TaskFamily, r: &Reference, good: bool, coverage: Coverage) -> Vec<u64> {
    let sizes = family.sizes();
    let keep = match coverage {
        Coverage::LowerHalf if !good => sizes.len() / 2,
        Coverage::LowerHalf | Coverage::Whole => sizes.len(),
    };
    sizes
        .into_iter()
        .take(keep)
        .filter(|&n| n <= r.max_n)
        .collect()
}

/// Checks one variant as the module comment says, at seeds `0..config.seeds`.
fn check_variant(
    v: &CorpusVariant,
    config: &HarnessConfig,
    coverage: Coverage,
) -> Result<(), String> {
    let family = v.family.as_ref();
    let name = family.name();
    validate_family(family, config).map_err(|d| format!("{name}: {d:?}"))?;
    check_input_sizes(family, config.seeds, list_len).map_err(|d| format!("{name}: {d:?}"))?;
    let largest = *family.sizes().last().expect("a sweep has sizes");
    let mut distinct: Vec<Inputs> = Vec::new();
    for seed in 0..config.seeds {
        let input = generate_input(family, largest, seed);
        if !distinct.contains(&input) {
            distinct.push(input);
        }
    }
    if distinct.len() as u64 != config.seeds {
        return Err(format!(
            "{name}: {} distinct inputs at n = {largest} over {} seeds",
            distinct.len(),
            config.seeds
        ));
    }
    let mut arena = Arena::new();
    let good = v.references.len() - 1;
    for (i, r) in v.references.iter().enumerate() {
        let p = compile(&mut arena, v.source, r.entry);
        check_examples(family, &arena, p, config).map_err(|f| format!("{}: {f:?}", r.entry))?;
        let sizes = sizes_for(family, r, i == good, coverage);
        if sizes.is_empty() {
            return Err(format!("{}: runs at no size of the sweep", r.entry));
        }
        check_generated(family, &arena, p, &sizes, config)
            .map_err(|f| format!("{}: {f:?}", r.entry))?;
    }
    Ok(())
}

/// Runs `check` on every variant of the corpus, one thread each (values are
/// `!Send`, so each thread builds its own corpus), and fails with every
/// variant's error.
fn for_every_variant(check: impl Fn(&CorpusVariant) -> Result<(), String> + Sync) {
    let count = families::development().len();
    let errors: Vec<String> = thread::scope(|s| {
        let handles: Vec<_> = (0..count)
            .map(|i| {
                let check = &check;
                s.spawn(move || check(&families::development()[i]))
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap_or_else(|_| Err("panicked".into())).err())
            .collect()
    });
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn the_corpus_holds_every_variant_once() {
    let corpus = families::development();
    let names: BTreeSet<&str> = corpus.iter().map(|v| v.family.name()).collect();
    assert_eq!(names.len(), corpus.len(), "variant names are unique");
    let expected = families::collapse::all().len()
        + families::sorted_order::all().len()
        + families::subsequence_count::all().len();
    assert_eq!(corpus.len(), expected);
    let families: BTreeSet<&str> = corpus.iter().map(|v| v.source).collect();
    assert_eq!(families.len(), FAMILIES);
    for v in &corpus {
        assert!(
            v.references.len() >= 2,
            "{v:?}: a naive and a good reference"
        );
        let entries: BTreeSet<&str> = v.references.iter().map(|r| r.entry).collect();
        assert_eq!(
            entries.len(),
            v.references.len(),
            "{v:?}: distinct references"
        );
    }
}

/// The good references at every size and the others at the lower half of
/// the sweep. The naive references at the upper half take most of the time
/// of the whole run (about 14 CPU-minutes in release in November 2026), too
/// long for every `cargo test`; [`every_reference_at_every_size`] runs them.
#[test]
fn every_reference_agrees_with_every_oracle_on_every_seed() {
    let config = corpus_config();
    for_every_variant(|v| check_variant(v, &config, Coverage::LowerHalf));
}

/// Every reference at every size it can finish at. Run with
/// `cargo test --release -p dc-tasks --test corpus -- --ignored`.
#[test]
#[ignore = "about 14 CPU-minutes in release; run before changing a family"]
fn every_reference_at_every_size() {
    let config = corpus_config();
    for_every_variant(|v| check_variant(v, &config, Coverage::Whole));
}

/// A variant whose oracle is wrong on one input only: the input of the
/// largest size and seed `seed`.
struct WrongOn {
    inner: Box<dyn TaskFamily>,
    seed: u64,
}

impl WrongOn {
    fn input(&self) -> Inputs {
        let largest = *self.inner.sizes().last().expect("a sweep has sizes");
        generate_input(self.inner.as_ref(), largest, self.seed)
    }
}

impl TaskFamily for WrongOn {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn signature(&self) -> Scheme {
        self.inner.signature()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn examples(&self) -> Vec<Example> {
        self.inner.examples()
    }
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        self.inner.generate(n, rng)
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        let right = self.inner.oracle(inputs);
        if inputs == self.input().as_slice() {
            match right {
                Value::Int(k) => Value::Int(k + 1),
                other => panic!("expected an Int answer, got {other}"),
            }
        } else {
            right
        }
    }
    fn sizes(&self) -> Vec<u64> {
        self.inner.sizes()
    }
}

/// The run is not vacuous: an oracle wrong only on the last corpus seed at
/// the largest size fails it, while a check at the sweep's own seeds passes.
#[test]
fn a_wrong_answer_on_the_last_seed_fails_the_run() {
    let seed = CORPUS_SEEDS - 1;
    let v = families::development()
        .into_iter()
        .find(|v| v.family.name().starts_with("sorted-order"))
        .expect("the corpus has a sorted-order variant");
    let largest = *v.family.sizes().last().expect("a sweep has sizes");
    let tampered = CorpusVariant {
        family: Box::new(WrongOn {
            inner: v.family,
            seed,
        }),
        source: v.source,
        references: v.references,
    };
    let at_sweep_seeds = HarnessConfig {
        seeds: DEFAULT_SEEDS,
        ..HarnessConfig::default()
    };
    assert_eq!(
        check_variant(&tampered, &at_sweep_seeds, Coverage::LowerHalf),
        Ok(())
    );
    let err = check_variant(&tampered, &corpus_config(), Coverage::LowerHalf)
        .expect_err("the wrong seed is found");
    let entry = tampered.references[tampered.references.len() - 1].entry;
    assert!(
        err.starts_with(entry)
            && err.contains(&format!("n: {largest}, seed: {seed}"))
            && err.contains("WrongOutput"),
        "{err}"
    );
}

/// Prints each reference's median steps at each size it is run at over the
/// sweep's [`DEFAULT_SEEDS`] seeds, and C's verdict on those medians when it
/// is run at the whole sweep. Run with
/// `cargo test --release -p dc-tasks --test corpus -- --ignored --nocapture`.
#[test]
#[ignore = "a report for the results log, not a check"]
fn corpus_cost_report() {
    let config = HarnessConfig::default();
    for v in families::development() {
        let family = v.family.as_ref();
        let mut arena = Arena::new();
        for r in &v.references {
            let p = compile(&mut arena, v.source, r.entry);
            let sizes = sizes_for(family, r, false, Coverage::Whole);
            let observations: Vec<SizeObservation> = sizes
                .iter()
                .map(|&n| {
                    let steps = (0..config.seeds)
                        .map(|seed| {
                            let inputs = generate_input(family, n, seed);
                            let expected = family.oracle(&inputs);
                            run_on(&arena, p, &inputs, &expected, config.max_fuel)
                                .unwrap_or_else(|f| panic!("{} n {n} seed {seed}: {f:?}", r.entry))
                        })
                        .collect();
                    SizeObservation::completed(n, steps)
                })
                .collect();
            let sweep = Sweep::in_seed_order(observations).expect("a valid sweep");
            let verdict = if sizes.len() == family.sizes().len() {
                let fit = ClassSelect::default().fit(&sweep).expect("C fits");
                format!("{:?}", fit.verdict)
            } else {
                "not fitted (short sweep)".to_owned()
            };
            println!(
                "COST {} {}: {:?} C {verdict}",
                family.name(),
                r.entry,
                sweep.medians()
            );
        }
    }
}
