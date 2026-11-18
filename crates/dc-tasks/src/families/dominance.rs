//! Dominance counting (bead dreamcounter-4b0daa, decision D25): count the
//! pairs or chains of positions that are in order both by position and by
//! value.
//!
//! | variant | answer |
//! |---|---|
//! | [`Variant::WeightedInversions`] | the sum of `j - i` over the pairs `i < j` with `xs[i] > xs[j]` |
//! | [`Variant::Pairs2d`] | the pairs `(i, j)` with `xs[i] < xs[j]` and `ys[i] < ys[j]` |
//! | [`Variant::IncreasingTriples`] | the triples `i < j < k` with `xs[i] < xs[j] < xs[k]` |
//! | [`Variant::IncreasingSubsequences`] | the non-empty strictly increasing subsequences, mod [`MODULUS`] |
//!
//! The family is the harder part of the development corpus. The pilot run
//! (results log, 2026-11-18) showed a strong proposer solving every other
//! development task with every program. Here each variant has a
//! [`Gate`]: seven inputs of [`GATE_N`] elements, each with [`GATE_FUEL`]
//! steps ([`GATE_INPUTS`]). A program passes only if it answers all seven
//! within that fuel. The gate checks only these inputs. They are chosen so
//! that the slow programs in the tests of `tests/dominance.rs` run out of
//! fuel on at least one: the naive references, which look at every pair of
//! positions; an unbalanced search tree (sorted input); a quicksort that
//! puts ties on one side (few values); a quicksort whose pivot is the
//! middle, first, or last element, or the median of those three
//! (organ-pipe order); and an array indexed by value, which costs
//! at least a step for each value in the range (a wide range). A program
//! that is slow only on some other input can pass. The good references
//! take at most 29 % of the fuel on any gate input (November 2026). The
//! description states the size, the value range, and the step limit, not
//! the inputs.
//!
//! Each variant has a naive dc reference that looks at every pair of
//! positions (`n²`), which times out on the gate, and a good one (`n log n`)
//! that passes it. They are in [`SOURCE`]. The good ones share a merge sort
//! that counts, for each element, the earlier elements of smaller value; for
//! increasing subsequences, a tree of sums over the ranks of the values.
//! dc v0 has no sort primitive, so a learned library can hold these. The
//! fuel plan is built from the naive reference.
//!
//! Inputs come in one kind per seed, the same at every size (decision D15):
//! seed `k` gets kind `k mod 3`. Kind 0 draws values from
//! `-WIDE..=WIDE`, so ties are rare; kind 1 from `0..=n / 8`, so each value
//! comes about 8 times and the strict comparisons matter; kind 2 is a walk
//! with steps in `-2..=2`, so long stretches go up or down and equal
//! neighbours are common. [`Variant::Pairs2d`] draws both lists in the
//! seed's kind.
//!
//! The oracles do not follow the references. They count with a Fenwick tree
//! over the sorted distinct values; this module's tests check them against
//! brute force.

use dc_core::Scheme;
use dc_eval::Value;

use crate::rng::TaskRng;
use crate::task::{generate_input, Example, Gate, Inputs, TaskFamily};

/// The reference solutions in dc surface syntax, compiled against the
/// prelude. [`Variant::naive`] and [`Variant::good`] name the definitions.
pub const SOURCE: &str = include_str!("dominance.dc");

/// The sweep, as in the sorted-order family: the naive references take a
/// few `n²` steps, so ten times their steps at 512 stays under the default
/// fuel cap.
pub const SIZES: &[u64] = &[16, 32, 64, 128, 256, 512];

/// The size of the gate.
pub const GATE_N: u64 = 8192;

/// The fuel of the gate: the harness's default global cap.
pub const GATE_FUEL: u64 = crate::harness::DEFAULT_MAX_FUEL;

/// Kind 0 draws values from `-WIDE..=WIDE`.
pub const WIDE: i64 = 1000;

/// [`Variant::IncreasingSubsequences`] counts mod this prime.
pub const MODULUS: i64 = 1_000_003;

/// A variant of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The sum of `j - i` over the pairs `i < j` with `xs[i] > xs[j]`.
    WeightedInversions,
    /// The pairs `(i, j)` with `xs[i] < xs[j]` and `ys[i] < ys[j]`.
    Pairs2d,
    /// The triples `i < j < k` with `xs[i] < xs[j] < xs[k]`.
    IncreasingTriples,
    /// The non-empty strictly increasing subsequences, mod [`MODULUS`].
    IncreasingSubsequences,
}

impl Variant {
    /// Every variant, in the order of the table above.
    pub const ALL: [Variant; 4] = [
        Variant::WeightedInversions,
        Variant::Pairs2d,
        Variant::IncreasingTriples,
        Variant::IncreasingSubsequences,
    ];

    /// The definition in [`SOURCE`] of the naive (`n²`) reference.
    pub fn naive(self) -> &'static str {
        match self {
            Variant::WeightedInversions => "weighted_inversions_naive",
            Variant::Pairs2d => "pairs_2d_naive",
            Variant::IncreasingTriples => "increasing_triples_naive",
            Variant::IncreasingSubsequences => "increasing_subsequences_naive",
        }
    }

    /// The definition in [`SOURCE`] of the good (`n log n`) reference.
    pub fn good(self) -> &'static str {
        match self {
            Variant::WeightedInversions => "weighted_inversions_good",
            Variant::Pairs2d => "pairs_2d_good",
            Variant::IncreasingTriples => "increasing_triples_good",
            Variant::IncreasingSubsequences => "increasing_subsequences_good",
        }
    }
}

/// One variant of the dominance-counting family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dominance(pub Variant);

/// Every variant as a family.
pub fn all() -> Vec<Dominance> {
    Variant::ALL.into_iter().map(Dominance).collect()
}

fn ints_of(v: &Value) -> Vec<i64> {
    match v {
        Value::List(l) => l
            .iter()
            .map(|x| x.as_int().expect("an Int element"))
            .collect(),
        other => panic!("expected a list of Int, got {other}"),
    }
}

fn list(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

/// A Fenwick tree of counts over ranks `0..n`.
struct Fenwick(Vec<i64>);

impl Fenwick {
    fn new(n: usize) -> Fenwick {
        Fenwick(vec![0; n + 1])
    }

    /// Adds `c` at rank `r`, mod `m`.
    fn add(&mut self, r: usize, c: i64, m: i64) {
        let mut i = r + 1;
        while i < self.0.len() {
            self.0[i] = (self.0[i] + c) % m;
            i += i & i.wrapping_neg();
        }
    }

    /// The sum over the ranks below `r`, mod `m`.
    fn below(&self, r: usize, m: i64) -> i64 {
        let mut i = r;
        let mut s = 0;
        while i > 0 {
            s = (s + self.0[i]) % m;
            i -= i & i.wrapping_neg();
        }
        s
    }
}

/// The rank of each element among the distinct values of `xs`.
fn ranks(xs: &[i64]) -> (Vec<usize>, usize) {
    let mut values = xs.to_vec();
    values.sort_unstable();
    values.dedup();
    let r = xs
        .iter()
        .map(|x| values.binary_search(x).expect("a value of xs"))
        .collect();
    (r, values.len())
}

/// No modulus: the counts of these variants fit in an `i64` at every size
/// the family uses.
const NO_MOD: i64 = i64::MAX;

/// For each position, the number of earlier elements of strictly smaller
/// value.
fn smaller_before(xs: &[i64]) -> Vec<i64> {
    let (r, m) = ranks(xs);
    let mut f = Fenwick::new(m);
    r.iter()
        .map(|&ri| {
            let c = f.below(ri, NO_MOD);
            f.add(ri, 1, NO_MOD);
            c
        })
        .collect()
}

fn weighted_inversions(xs: &[i64]) -> i64 {
    // An earlier larger element is an earlier smaller one of the negated
    // list. With the count c and the position sum s of those, position j
    // adds c j - s.
    let neg: Vec<i64> = xs.iter().map(|x| -x).collect();
    let (r, m) = ranks(&neg);
    let mut count = Fenwick::new(m);
    let mut sum = Fenwick::new(m);
    let mut total = 0;
    for (j, &rj) in r.iter().enumerate() {
        let c = count.below(rj, NO_MOD);
        let s = sum.below(rj, NO_MOD);
        total += c * j as i64 - s;
        count.add(rj, 1, NO_MOD);
        sum.add(rj, j as i64, NO_MOD);
    }
    total
}

fn pairs_2d(xs: &[i64], ys: &[i64]) -> i64 {
    // Sorted by x, and by y downwards among equal x, an earlier point of
    // smaller y always has smaller x.
    let mut ps: Vec<(i64, i64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    ps.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let ys_sorted: Vec<i64> = ps.iter().map(|p| p.1).collect();
    smaller_before(&ys_sorted).iter().sum()
}

fn increasing_triples(xs: &[i64]) -> i64 {
    let left = smaller_before(xs);
    let rev_neg: Vec<i64> = xs.iter().rev().map(|x| -x).collect();
    let mut right = smaller_before(&rev_neg);
    right.reverse();
    left.iter().zip(&right).map(|(l, r)| l * r).sum()
}

fn increasing_subsequences(xs: &[i64]) -> i64 {
    let (r, m) = ranks(xs);
    let mut f = Fenwick::new(m);
    let mut total = 0;
    for &ri in &r {
        let ending = (1 + f.below(ri, MODULUS)) % MODULUS;
        f.add(ri, ending, MODULUS);
        total = (total + ending) % MODULUS;
    }
    total
}

/// The kind of input seed `seed` gets: `seed mod 3`.
fn kind_of(seed: u64) -> u64 {
    seed % 3
}

/// `n` values of kind `kind`, as the module comment says.
fn values(kind: u64, n: u64, rng: &mut TaskRng) -> Vec<i64> {
    match kind {
        0 => (0..n).map(|_| rng.int_in(-WIDE, WIDE)).collect(),
        1 => {
            let top = (n / 8).max(1) as i64;
            (0..n).map(|_| rng.int_in(0, top)).collect()
        }
        _ => {
            let mut v = 0;
            (0..n)
                .map(|_| {
                    v += rng.int_in(-2, 2);
                    v
                })
                .collect()
        }
    }
}

/// One input of the gate. Each is there because some common way to count
/// is `n²` or worse on it and fast on the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateInput {
    /// The input of seed 0: values from `-WIDE..=WIDE`.
    Random,
    /// The input of seed 1 (many ties) sorted up: the worst case of an
    /// unbalanced search tree.
    SortedUp,
    /// The same, sorted down.
    SortedDown,
    /// The input of seed 2: a walk.
    Walk,
    /// Values from `0..=1`: the worst case of a quicksort that puts every
    /// tie on one side, and of a search tree that sends ties one way.
    FewValues,
    /// Values from `-WIDE_GATE..=WIDE_GATE`, nearly all distinct: the worst
    /// case of a program that keeps a slot or a step for each value in the
    /// range, such as an array indexed by value.
    WideRange,
    /// The values of [`GateInput::WideRange`] in organ-pipe order: up, then
    /// down, so the middle element is the largest. The worst case of a
    /// quicksort whose pivot is the middle, first, or last element, or the
    /// median of those three. A pivot at another fixed place, such as a
    /// third of the way, is not slow on it.
    OrganPipe,
}

/// The order of the gate's inputs.
pub const GATE_INPUTS: [GateInput; 7] = [
    GateInput::Random,
    GateInput::SortedUp,
    GateInput::SortedDown,
    GateInput::Walk,
    GateInput::FewValues,
    GateInput::WideRange,
    GateInput::OrganPipe,
];

/// The bound on values that the description states, and the value range of
/// [`GateInput::WideRange`]. A program that spends even a step on each value
/// in the range runs out of fuel: the range is 40 times [`GATE_FUEL`]. The
/// product of two values fits in an `Int`.
pub const WIDE_GATE: i64 = 1_000_000_000;

/// The seeds of the gate inputs that are not generated inputs. They are
/// passed to [`TaskRng::new`] with [`GATE_N`], apart from the generator's.
const FEW_VALUES_SEED: u64 = 1_000;
const WIDE_RANGE_SEED: u64 = 1_001;

/// A gate input of `family`. In [`Variant::Pairs2d`] each list is built
/// the same way.
fn gate_input(family: &Dominance, g: GateInput) -> Inputs {
    let lists = if family.0 == Variant::Pairs2d { 2 } else { 1 };
    let sorted = |down: bool| -> Inputs {
        generate_input(family, GATE_N, 1)
            .iter()
            .map(|l| {
                let mut xs = ints_of(l);
                xs.sort_unstable();
                if down {
                    xs.reverse();
                }
                list(&xs)
            })
            .collect()
    };
    let drawn = |seed: u64, lo: i64, hi: i64| -> Inputs {
        let mut rng = TaskRng::new(GATE_N, seed);
        (0..lists)
            .map(|_| {
                list(
                    &(0..GATE_N)
                        .map(|_| rng.int_in(lo, hi))
                        .collect::<Vec<i64>>(),
                )
            })
            .collect()
    };
    match g {
        GateInput::Random => generate_input(family, GATE_N, 0),
        GateInput::SortedUp => sorted(false),
        GateInput::SortedDown => sorted(true),
        GateInput::Walk => generate_input(family, GATE_N, 2),
        GateInput::FewValues => drawn(FEW_VALUES_SEED, 0, 1),
        GateInput::WideRange => drawn(WIDE_RANGE_SEED, -WIDE_GATE, WIDE_GATE),
        GateInput::OrganPipe => drawn(WIDE_RANGE_SEED, -WIDE_GATE, WIDE_GATE)
            .iter()
            .map(|l| {
                let mut xs = ints_of(l);
                xs.sort_unstable();
                let up = xs.iter().step_by(2);
                let down = xs.iter().skip(1).step_by(2).rev();
                list(&up.chain(down).copied().collect::<Vec<i64>>())
            })
            .collect(),
    }
}

/// The text of the size and step limits, the same for every variant.
macro_rules! limits {
    () => {
        " Lists have at most 8192 elements, each between -1000000000 and \
         1000000000. The program must finish within 50000000 evaluation steps \
         on a list of 8192 elements."
    };
}

impl TaskFamily for Dominance {
    fn name(&self) -> &str {
        match self.0 {
            Variant::WeightedInversions => "dominance/weighted-inversions",
            Variant::Pairs2d => "dominance/pairs-2d",
            Variant::IncreasingTriples => "dominance/increasing-triples",
            Variant::IncreasingSubsequences => "dominance/increasing-subsequences",
        }
    }

    fn signature(&self) -> Scheme {
        let s = match self.0 {
            Variant::Pairs2d => "List Int -> List Int -> Int",
            _ => "List Int -> Int",
        };
        s.parse().expect("a scheme")
    }

    fn description(&self) -> &str {
        match self.0 {
            Variant::WeightedInversions => concat!(
                "For every pair of positions i < j of the list (counting from 0) where \
                 the element at i is strictly greater than the element at j, take the \
                 distance j - i. Return the sum of these distances, or 0 if there is no \
                 such pair.",
                limits!()
            ),
            Variant::Pairs2d => concat!(
                "The two lists have the same length; position i stands for the point \
                 (xs[i], ys[i]). Count the ordered pairs of positions (i, j) where \
                 xs[i] < xs[j] and ys[i] < ys[j], both strictly. Points may repeat, \
                 and equal coordinates never count.",
                limits!()
            ),
            Variant::IncreasingTriples => concat!(
                "Count the triples of positions i < j < k of the list where the \
                 elements are strictly increasing: xs[i] < xs[j] < xs[k]. Equal \
                 elements never count.",
                limits!()
            ),
            Variant::IncreasingSubsequences => concat!(
                "Count the ways to choose a non-empty set of positions of the list \
                 such that, read from left to right, the elements at the chosen \
                 positions are strictly increasing. A single position always counts. \
                 Two choices are different if they choose different positions, even \
                 when the elements there are equal. Return the count mod 1000003.",
                limits!()
            ),
        }
    }

    fn examples(&self) -> Vec<Example> {
        let one = |xs: &[i64], out: i64| Example::new(vec![list(xs)], Value::Int(out));
        match self.0 {
            Variant::WeightedInversions => vec![
                one(&[], 0),
                one(&[1], 0),
                one(&[2, 1], 1),
                one(&[3, 1, 2], 3),
                one(&[2, 2, 1], 3),
                one(&[5, 0, 5, -1], 7),
                one(&[4, 1, 4, 2, 9, 1, 7], 18),
            ],
            Variant::Pairs2d => {
                let two = |xs: &[i64], ys: &[i64], out: i64| {
                    Example::new(vec![list(xs), list(ys)], Value::Int(out))
                };
                vec![
                    two(&[], &[], 0),
                    two(&[1, 2], &[1, 2], 1),
                    two(&[1, 2], &[2, 1], 0),
                    two(&[1, 1], &[1, 2], 0),
                    two(&[3, 1, 2], &[1, 2, 3], 1),
                    two(&[0, 0, 1, 1], &[0, 1, 0, 1], 1),
                    two(&[2, 5, 1, 4, 3], &[1, 4, 2, 5, 3], 8),
                ]
            }
            Variant::IncreasingTriples => vec![
                one(&[], 0),
                one(&[1, 2, 3], 1),
                one(&[3, 2, 1], 0),
                one(&[1, 1, 2], 0),
                one(&[1, 2, 2, 3], 2),
                one(&[2, 1, 3, 4], 2),
                one(&[4, 1, 4, 2, 9, 1, 7], 4),
            ],
            Variant::IncreasingSubsequences => vec![
                one(&[], 0),
                one(&[5], 1),
                one(&[1, 2], 3),
                one(&[2, 1], 2),
                one(&[1, 1], 2),
                one(&[1, 2, 3], 7),
                one(&[1, 3, 2], 5),
                one(&[2, 2, 3], 5),
                one(&[4, 1, 4, 2, 9, 1, 7], 22),
            ],
        }
    }

    /// The [`GateInput`]s in [`GATE_INPUTS`] order, each of [`GATE_N`]
    /// elements, with [`GATE_FUEL`] each.
    fn gate(&self) -> Option<Gate> {
        let inputs = GATE_INPUTS
            .iter()
            .map(|&g| {
                let i = gate_input(self, g);
                let out = self.oracle(&i);
                Example::new(i, out)
            })
            .collect();
        Some(Gate {
            inputs,
            fuel: GATE_FUEL,
        })
    }

    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let kind = kind_of(rng.seed());
        let xs = values(kind, n, rng);
        let mut inputs = vec![list(&xs)];
        if self.0 == Variant::Pairs2d {
            inputs.push(list(&values(kind, n, rng)));
        }
        inputs
    }

    fn oracle(&self, inputs: &[Value]) -> Value {
        let xs = ints_of(&inputs[0]);
        Value::Int(match self.0 {
            Variant::WeightedInversions => weighted_inversions(&xs),
            Variant::Pairs2d => pairs_2d(&xs, &ints_of(&inputs[1])),
            Variant::IncreasingTriples => increasing_triples(&xs),
            Variant::IncreasingSubsequences => increasing_subsequences(&xs),
        })
    }

    fn sizes(&self) -> Vec<u64> {
        SIZES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answers by brute force, independent of the oracles.
    fn brute(v: Variant, xs: &[i64], ys: &[i64]) -> i64 {
        let n = xs.len();
        match v {
            Variant::WeightedInversions => (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .filter(|&(i, j)| xs[i] > xs[j])
                .map(|(i, j)| (j - i) as i64)
                .sum(),
            Variant::Pairs2d => (0..n)
                .flat_map(|i| (0..n).map(move |j| (i, j)))
                .filter(|&(i, j)| xs[i] < xs[j] && ys[i] < ys[j])
                .count() as i64,
            Variant::IncreasingTriples => {
                let mut c = 0;
                for i in 0..n {
                    for j in i + 1..n {
                        for k in j + 1..n {
                            if xs[i] < xs[j] && xs[j] < xs[k] {
                                c += 1;
                            }
                        }
                    }
                }
                c
            }
            Variant::IncreasingSubsequences => {
                // Every non-empty subset of positions.
                (1_u32..1 << n)
                    .filter(|&mask| {
                        let chosen: Vec<i64> = (0..n)
                            .filter(|&i| mask >> i & 1 == 1)
                            .map(|i| xs[i])
                            .collect();
                        chosen.windows(2).all(|w| w[0] < w[1])
                    })
                    .count() as i64
                    % MODULUS
            }
        }
    }

    fn inputs(v: Variant, xs: &[i64], ys: &[i64]) -> Inputs {
        let mut i = vec![list(xs)];
        if v == Variant::Pairs2d {
            i.push(list(ys));
        }
        i
    }

    proptest::proptest! {
        #[test]
        fn oracles_agree_with_brute_force(
            pts in proptest::collection::vec((-6_i64..6, -6_i64..6), 0..13),
        ) {
            let xs: Vec<i64> = pts.iter().map(|p| p.0).collect();
            let ys: Vec<i64> = pts.iter().map(|p| p.1).collect();
            for v in Variant::ALL {
                proptest::prop_assert_eq!(
                    Dominance(v).oracle(&inputs(v, &xs, &ys)),
                    Value::Int(brute(v, &xs, &ys)),
                    "{:?} on {:?}, {:?}", v, xs, ys
                );
            }
        }
    }

    #[test]
    fn oracles_agree_with_brute_force_on_generated_inputs() {
        // Up to 64 elements, every kind; increasing subsequences by brute
        // force only at 16, as it enumerates every subset.
        for v in Variant::ALL {
            let f = Dominance(v);
            for n in [16, 32, 64] {
                if v == Variant::IncreasingSubsequences && n > 16 {
                    continue;
                }
                for seed in 0..6 {
                    let i = generate_input(&f, n, seed);
                    let ys = i.get(1).map(ints_of).unwrap_or_default();
                    assert_eq!(
                        f.oracle(&i),
                        Value::Int(brute(v, &ints_of(&i[0]), &ys)),
                        "{v:?}, n = {n}, seed {seed}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_gate_has_gate_n_elements_and_the_examples_are_small() {
        for f in all() {
            let gate = f.gate().expect("a gate");
            assert_eq!(gate.fuel, GATE_FUEL);
            assert_eq!(gate.inputs.len(), GATE_INPUTS.len());
            for e in &gate.inputs {
                for input in &e.inputs {
                    assert_eq!(ints_of(input).len() as u64, GATE_N, "{}", f.name());
                }
            }
            // The second and third are sorted up and down.
            for (e, down) in gate.inputs[1..].iter().zip([false, true]) {
                for input in &e.inputs {
                    let xs = ints_of(input);
                    assert!(
                        xs.windows(2)
                            .all(|w| if down { w[0] >= w[1] } else { w[0] <= w[1] }),
                        "{}",
                        f.name()
                    );
                }
            }
            // The input of few values has two, and the wide one has nearly
            // as many distinct values as elements, over a range far wider
            // than the list.
            let distinct = |e: &Example| -> Vec<(usize, i64)> {
                e.inputs
                    .iter()
                    .map(|l| {
                        let xs = ints_of(l);
                        let set: std::collections::BTreeSet<i64> = xs.iter().copied().collect();
                        let lo = *set.first().unwrap();
                        let hi = *set.last().unwrap();
                        (set.len(), hi - lo)
                    })
                    .collect()
            };
            let at = |g: GateInput| &gate.inputs[GATE_INPUTS.iter().position(|&x| x == g).unwrap()];
            for (d, _) in distinct(at(GateInput::FewValues)) {
                assert_eq!(d, 2, "{}", f.name());
            }
            for (d, range) in distinct(at(GateInput::WideRange)) {
                assert!(d as u64 > GATE_N * 99 / 100, "{}: {d}", f.name());
                assert!(range as u64 > 100 * GATE_N, "{}: {range}", f.name());
            }
            // The organ pipe goes up to its largest value, then down.
            for l in &at(GateInput::OrganPipe).inputs {
                let xs = ints_of(l);
                let top = xs.len() / 2;
                assert_eq!(xs[top], *xs.iter().max().unwrap(), "{}", f.name());
                assert!(xs[..=top].windows(2).all(|w| w[0] <= w[1]), "{}", f.name());
                assert!(xs[top..].windows(2).all(|w| w[0] >= w[1]), "{}", f.name());
            }
            for e in f.examples() {
                assert!(ints_of(&e.inputs[0]).len() <= 8, "{}", f.name());
            }
        }
    }

    #[test]
    fn the_description_states_the_size_and_step_limits() {
        // The stated limits must be the real ones.
        let text = limits!();
        assert!(
            text.contains(&format!("at most {GATE_N} elements")),
            "{text}"
        );
        assert!(text.contains(&format!("within {GATE_FUEL} ")), "{text}");
        assert!(
            text.contains(&format!("a list of {GATE_N} elements")),
            "{text}"
        );
        assert!(
            text.contains(&format!("between -{WIDE_GATE} and {WIDE_GATE}.")),
            "{text}"
        );
        for f in all() {
            assert!(f.description().ends_with(text), "{}", f.name());
        }
    }

    #[test]
    fn each_seed_keeps_its_kind_at_every_size() {
        for seed in 0..6 {
            for n in SIZES {
                let xs =
                    ints_of(&generate_input(&Dominance(Variant::IncreasingTriples), *n, seed)[0]);
                let in_tie_range = xs.iter().all(|&x| 0 <= x && x <= (n / 8) as i64);
                let walk = xs.windows(2).all(|w| (w[1] - w[0]).abs() <= 2);
                let observed = match (in_tie_range, walk) {
                    (true, _) => 1,
                    (false, true) => 2,
                    (false, false) => 0,
                };
                assert_eq!(observed, kind_of(seed), "seed {seed}, n = {n}: {xs:?}");
            }
        }
    }

    #[test]
    fn generated_inputs_have_ties_and_both_orders() {
        // Strict comparisons matter only when values repeat, and a count is
        // tested only if it is neither 0 nor everything.
        for v in Variant::ALL {
            let f = Dominance(v);
            for seed in 0..3 {
                let i = generate_input(&f, 128, seed);
                let xs = ints_of(&i[0]);
                let mut sorted = xs.clone();
                sorted.sort_unstable();
                sorted.dedup();
                if seed % 3 != 0 {
                    assert!(sorted.len() < xs.len(), "{v:?} seed {seed}: no ties");
                }
                let a = f.oracle(&i).as_int().expect("an Int");
                assert!(a > 0, "{v:?} seed {seed}: answer 0");
            }
        }
    }
}
