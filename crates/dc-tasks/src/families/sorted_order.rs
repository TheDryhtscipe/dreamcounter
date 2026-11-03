//! Sorted-order queries (bead P3.5): questions about a list of integers that
//! a scan answers once the list is sorted.
//!
//! The input is a list of `n` integers in `[-10⁶, 10⁶]`; [`Variant::PairSums`]
//! also takes a threshold `t`. The size is `n`. Ties and gaps matter
//! differently to each variant, so inputs come in three kinds. Seed `k` gets
//! kind `k mod 3` at every size, in the order below, so the default five
//! seeds have every kind. Each seed's step counts then follow one kind of
//! input across the sweep, and estimator C can see a program's growth
//! through them (bead dreamcounter-0ea5b7). The kinds:
//!
//! - tied: each element is drawn, with even odds, from the whole range or
//!   from a narrow one of about `n / 2` values (see [`narrow_bound`]), so the
//!   input has many equal values;
//! - spread: elements are drawn from about `2 n²` values (see
//!   [`spread_bound`]), so most inputs have none equal, but neighbours once
//!   sorted are close, and the least gap is small and not 0;
//! - wide: the range is cut into `n` equal slots and each element is drawn
//!   from the first half of its own slot, then the list is shuffled, so the
//!   least gap is at least `10⁶ / n` (about 2 000 at 512).
//!
//! The threshold of [`Variant::PairSums`] is chosen per seed in the same
//! way, so that a natural solution's cost, which depends on the answer, does
//! not jump between sizes. A seed's plan says whether the threshold is exact
//! or random, and where along its range it falls: at the bottom, at the top,
//! or at a quantile `q` in `[0, 1)` drawn from [`TaskRng::seed_stream`]. The
//! plan is the same at every size. Seed `k` of kind `k mod 3` takes the
//! plan `(k / 3) mod 6` in its kind's row of `PAIR_SUMS_PLANS`; each row
//! holds all six plans, so over seeds 0 to 17 every kind gets every plan.
//!
//! - An exact threshold is the sum of two elements, so some pair sums to
//!   exactly `t`. In a tied input it is taken from the sums of two narrow
//!   elements (`|x| <= narrow_bound(n)`) to which at least two pairs sum,
//!   preferring those to which two equal elements sum (see
//!   `tied_thresholds`). If there are none, and in the other kinds, it is
//!   taken from the sorted sums of all pairs `i < j`. Either way it is the
//!   first, the last, or the sum at the fraction `q` of the way along that
//!   list. When the list is all pair sums, its last is the largest, so every
//!   pair counts; the first and last of a tied input's list are not the
//!   least and largest pair sums.
//! - A random threshold is `-2 top`, `2 top`, or the value at the fraction
//!   `q` of the way along `-2 top..=2 top`, where `top` is the largest
//!   magnitude of an element. At `2 top` every pair counts. At `-2 top` in
//!   a wide input none does, as its elements are distinct.
//!
//! The default five seeds take these plans:
//!
//! | seed | kind | threshold |
//! |---|---|---|
//! | 0 | tied | exact, at `q`: several pairs sum to it wherever the input allows |
//! | 1 | spread | exact, at `q` |
//! | 2 | wide | random, `-2 top`: no pair counts |
//! | 3 | tied | random, at `q` |
//! | 4 | spread | random, `2 top`: every pair counts |
//!
//! So the default sweep has no wide input with an exact threshold or with
//! an answer strictly between 0 and every pair, and at seed 4, where every
//! pair counts, no pair sums to `t` exactly. Seeds 5 and up have these.
//!
//! With the default five seeds, the sweep has least gaps of 0, small ones
//! and large ones; thresholds that at least one pair sums to and that
//! several pairs sum to; and answers of 0 and of every pair. Unit tests
//! check this. So a solution that is wrong on
//! every input of one of those kinds fails the sweep, and so do two wrong
//! pair-sums solutions that are right whenever at most one pair sums to `t`
//! (`tests/sorted_order.rs`). The variants:
//!
//! | variant | answer |
//! |---|---|
//! | [`Variant::Inversions`] | pairs `i < j` with `xs[i] > xs[j]` |
//! | [`Variant::PairSums`] | pairs `i < j` with `xs[i] + xs[j] <= t` |
//! | [`Variant::MinGap`] | the least `|xs[i] - xs[j]|` over `i < j` |
//! | [`Variant::Distinct`] | the number of distinct values |
//! | [`Variant::LowerMedian`] | the element at index `(n - 1) / 2` once sorted |
//!
//! Each has a naive dc reference that looks at every pair of positions
//! (`n²`) and a good one that sorts by merge sort and then scans (`n log n`).
//! dc v0 has no sort primitive, so merge sort is the abstraction the
//! variants share. The references are in [`SOURCE`], to be compiled against
//! the prelude; [`Variant::naive`] and [`Variant::good`] name their
//! definitions. The fuel plan is built from the naive one.
//!
//! The oracles do not follow the references: inversions are counted with a
//! Fenwick tree, pair sums by binary search over the sorted list, and the
//! rest by Rust's sort and a scan.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use dc_core::Scheme;
use dc_eval::Value;

use crate::rng::TaskRng;
use crate::task::{Example, Inputs, TaskFamily};

/// The reference solutions in dc surface syntax, compiled against the
/// prelude. [`Variant::naive`] and [`Variant::good`] name the definitions.
pub const SOURCE: &str = include_str!("sorted_order.dc");

/// The sweep. It stops at 512 because the naive references take between
/// about `3.5 n²` and `14 n²` steps (October 2026): the most, the lower
/// median's, is 3.7 million at 512, so the plan's ten times that stays under
/// the default fuel cap of 50 million, and at 1024 it would not.
pub const SIZES: &[u64] = &[16, 32, 64, 128, 256, 512];

/// Elements, and the threshold of [`Variant::PairSums`], are drawn from
/// `-VALUE_BOUND..=VALUE_BOUND`.
pub const VALUE_BOUND: i64 = 1_000_000;

/// The narrow range elements of an input of size `n` are drawn from half the
/// time is `-narrow_bound(n)..=narrow_bound(n)`: about `n / 4` values each
/// side of 0, so equal values are common.
pub fn narrow_bound(n: u64) -> i64 {
    (n / 4).clamp(1, VALUE_BOUND as u64) as i64
}

/// Elements of a spread input of size `n` are drawn from
/// `-spread_bound(n)..=spread_bound(n)`: about `2 n²` values, so two equal
/// elements are unlikely but the least gap is small.
pub fn spread_bound(n: u64) -> i64 {
    n.saturating_mul(n).clamp(1, VALUE_BOUND as u64) as i64
}

/// A variant of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// Pairs `i < j` with `xs[i] > xs[j]`.
    Inversions,
    /// Pairs `i < j` with `xs[i] + xs[j] <= t`.
    PairSums,
    /// The least `|xs[i] - xs[j]|` over `i < j`, or `max_int` if `n < 2`.
    MinGap,
    /// The number of distinct values.
    Distinct,
    /// The element at index `(n - 1) / 2` in sorted order, or `0` if `n = 0`.
    LowerMedian,
}

impl Variant {
    /// Every variant, in the order of the table above.
    pub const ALL: [Variant; 5] = [
        Variant::Inversions,
        Variant::PairSums,
        Variant::MinGap,
        Variant::Distinct,
        Variant::LowerMedian,
    ];

    /// The definition in [`SOURCE`] of the naive (`n²`) reference.
    pub fn naive(self) -> &'static str {
        match self {
            Variant::Inversions => "inversions_naive",
            Variant::PairSums => "pair_sums_naive",
            Variant::MinGap => "min_gap_naive",
            Variant::Distinct => "distinct_naive",
            Variant::LowerMedian => "lower_median_naive",
        }
    }

    /// The definition in [`SOURCE`] of the good (`n log n`) reference.
    pub fn good(self) -> &'static str {
        match self {
            Variant::Inversions => "inversions_good",
            Variant::PairSums => "pair_sums_good",
            Variant::MinGap => "min_gap_good",
            Variant::Distinct => "distinct_good",
            Variant::LowerMedian => "lower_median_good",
        }
    }
}

/// One variant of the sorted-order family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SortedOrder(pub Variant);

/// Every variant as a family.
pub fn all() -> Vec<SortedOrder> {
    Variant::ALL.into_iter().map(SortedOrder).collect()
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

fn int_of(v: &Value) -> i64 {
    v.as_int().expect("an Int")
}

fn sorted(mut xs: Vec<i64>) -> Vec<i64> {
    xs.sort_unstable();
    xs
}

/// Inversions by a Fenwick tree over the ranks of the values: for each
/// element from the left, the elements before it that are greater.
fn inversions(xs: &[i64]) -> i64 {
    let mut ranks = xs.to_vec();
    ranks.sort_unstable();
    ranks.dedup();
    let mut tree = vec![0_i64; ranks.len() + 1];
    let mut count = 0;
    for (seen, x) in (0_i64..).zip(xs) {
        // 1-based rank of x; elements at most x so far are a prefix sum.
        let r = ranks.partition_point(|&v| v <= *x);
        let mut at_most = 0;
        let mut i = r;
        while i > 0 {
            at_most += tree[i];
            i &= i - 1;
        }
        count += seen - at_most;
        let mut i = r;
        while i < tree.len() {
            tree[i] += 1;
            i += i & i.wrapping_neg();
        }
    }
    count
}

/// Pairs with sum at most `t`, by binary search for each element's partners
/// among the elements after it in sorted order.
fn pair_sums(xs: &[i64], t: i64) -> i64 {
    let s = sorted(xs.to_vec());
    let mut count = 0;
    for (i, &x) in s.iter().enumerate() {
        let rest = &s[i + 1..];
        count += rest.partition_point(|&y| x + y <= t) as i64;
    }
    count
}

/// The pairs `i < j` with `xs[i] + xs[j] = t`.
#[cfg(test)]
fn exact_pairs(xs: &[i64], t: i64) -> i64 {
    pair_sums(xs, t) - pair_sums(xs, t - 1)
}

/// The thresholds a tied input's exact threshold is drawn from: the sums of
/// two narrow elements (`|x| <= narrow`) to which at least two pairs sum,
/// and of those, the ones to which a pair of equal elements sums, if any. A
/// pair with an element from the whole range is almost always the only pair
/// with its sum, so it cannot tell a right count of exact pairs from "1 if
/// there is one", nor one that skips pairs of equal elements.
fn tied_thresholds(xs: &[i64], narrow: i64) -> Vec<i64> {
    let near = sorted(xs.iter().copied().filter(|x| x.abs() <= narrow).collect());
    let mut sums: Vec<i64> = near
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| near[i + 1..].iter().map(move |&b| a + b))
        .collect();
    sums.sort_unstable();
    sums.dedup();
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for &x in xs {
        *counts.entry(x).or_insert(0) += 1;
    }
    // Pairs `i < j` with `xs[i] + xs[j] = t`, as `exact_pairs` counts them.
    let exact = |t: i64| -> i64 {
        counts
            .iter()
            .map(|(&v, &c)| match (t - v).cmp(&v) {
                Ordering::Greater => c * counts.get(&(t - v)).copied().unwrap_or(0),
                Ordering::Equal => c * (c - 1) / 2,
                Ordering::Less => 0,
            })
            .sum()
    };
    sums.retain(|&t| exact(t) >= 2);
    let twins = |t: i64| t % 2 == 0 && near.iter().filter(|&&x| 2 * x == t).count() >= 2;
    let with_twins: Vec<i64> = sums.iter().copied().filter(|&t| twins(t)).collect();
    if with_twins.is_empty() {
        sums
    } else {
        with_twins
    }
}

fn min_gap(xs: &[i64]) -> i64 {
    sorted(xs.to_vec())
        .windows(2)
        .map(|w| w[1] - w[0])
        .min()
        .unwrap_or(i64::MAX)
}

fn distinct(xs: &[i64]) -> i64 {
    let mut s = sorted(xs.to_vec());
    s.dedup();
    s.len() as i64
}

fn lower_median(xs: &[i64]) -> i64 {
    if xs.is_empty() {
        return 0;
    }
    sorted(xs.to_vec())[(xs.len() - 1) / 2]
}

fn list(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

/// The kind of input seed `seed` gets at every size: 0 tied, 1 spread, 2
/// wide.
fn kind_of(seed: u64) -> u64 {
    seed % 3
}

/// Where along its range a pair-sums threshold falls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// The first value of the list: the least pair sum, the least of a
    /// tied input's thresholds, or `-2 top`.
    Bottom,
    /// The last value of the list: the largest pair sum, the largest of a
    /// tied input's thresholds, or `2 top`.
    Top,
    /// The value at the seed's quantile, drawn from
    /// [`TaskRng::seed_stream`].
    Drawn,
}

/// A seed's pair-sums threshold: whether it is a pair sum ("exact") or
/// drawn from `-2 top..=2 top`, and where along that range it falls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PairSumsPlan {
    exact: bool,
    place: Place,
}

/// The pair-sums plans, a row for each kind (tied, spread, wide), each row
/// all six plans in some order. Seed `k` takes entry `(k / 3) mod 6` of row
/// `k mod 3`, so the place does not follow the kind. The default five seeds
/// take entries 0 and 1 of the tied and spread rows and entry 0 of the wide
/// row (see the module comment).
const PAIR_SUMS_PLANS: [[PairSumsPlan; 6]; 3] = {
    const fn p(exact: bool, place: Place) -> PairSumsPlan {
        PairSumsPlan { exact, place }
    }
    use Place::{Bottom, Drawn, Top};
    [
        [
            p(true, Drawn),
            p(false, Drawn),
            p(true, Top),
            p(false, Bottom),
            p(true, Bottom),
            p(false, Top),
        ],
        [
            p(true, Drawn),
            p(false, Top),
            p(false, Drawn),
            p(true, Bottom),
            p(true, Top),
            p(false, Bottom),
        ],
        [
            p(false, Bottom),
            p(true, Drawn),
            p(true, Top),
            p(false, Drawn),
            p(false, Top),
            p(true, Bottom),
        ],
    ]
};

fn pair_sums_plan(seed: u64) -> PairSumsPlan {
    PAIR_SUMS_PLANS[kind_of(seed) as usize][((seed / 3) % 6) as usize]
}

/// The index `place` picks in a list of `len > 0`: the first, the last, or
/// the fraction `q / 2⁶⁴` of the way along.
fn index_at(place: Place, q: u64, len: u64) -> u64 {
    match place {
        Place::Bottom => 0,
        Place::Top => len - 1,
        Place::Drawn => scale(q, len),
    }
}

/// `floor(q / 2⁶⁴ · len)`: the index the fraction `q / 2⁶⁴` of the way
/// along a list of `len`, below `len` whenever `len > 0`. In integers, so it
/// is the same on every platform.
fn scale(q: u64, len: u64) -> u64 {
    ((u128::from(q) * u128::from(len)) >> 64) as u64
}

impl TaskFamily for SortedOrder {
    fn name(&self) -> &str {
        match self.0 {
            Variant::Inversions => "sorted-order/inversions",
            Variant::PairSums => "sorted-order/pair-sums",
            Variant::MinGap => "sorted-order/min-gap",
            Variant::Distinct => "sorted-order/distinct",
            Variant::LowerMedian => "sorted-order/lower-median",
        }
    }

    fn signature(&self) -> Scheme {
        let s = match self.0 {
            Variant::PairSums => "List Int -> Int -> Int",
            _ => "List Int -> Int",
        };
        s.parse().expect("a scheme")
    }

    fn description(&self) -> &str {
        match self.0 {
            Variant::Inversions => {
                "Count the pairs of positions i < j in the list whose elements are out of \
                 order: the element at i is greater than the element at j."
            }
            Variant::PairSums => {
                "Given a list of integers and a threshold t, count the pairs of positions \
                 i < j whose elements sum to at most t."
            }
            Variant::MinGap => {
                "Return the smallest absolute difference between the elements at two \
                 different positions of the list. If the list has fewer than two \
                 elements, return max_int (9223372036854775807)."
            }
            Variant::Distinct => "Count the distinct values in the list.",
            Variant::LowerMedian => {
                "Return the lower median of the list: the element at index (n - 1) / 2, \
                 counting from 0 and rounding down, once the list of n elements is sorted \
                 into ascending order. For an empty list, return 0."
            }
        }
    }

    fn examples(&self) -> Vec<Example> {
        let one = |xs: &[i64], out: i64| Example::new(vec![list(xs)], Value::Int(out));
        match self.0 {
            Variant::Inversions => vec![
                one(&[], 0),
                one(&[3, 1, 2], 2),
                one(&[5, -1, 5, 0], 3),
                one(&[2, 2, 1, 1], 4),
                one(&[4, 1, 4, 2, 9, 1, 7], 8),
            ],
            Variant::PairSums => {
                let two = |xs: &[i64], t: i64, out: i64| {
                    Example::new(vec![list(xs), Value::Int(t)], Value::Int(out))
                };
                vec![
                    two(&[], 0, 0),
                    two(&[1, 2, 3], 4, 2),
                    two(&[-5, 10, 0, 2], 5, 4),
                    two(&[3, 3, 1, 5], 6, 4),
                    two(&[4, 1, 4, 2, 9, 1, 7], 5, 7),
                ]
            }
            Variant::MinGap => vec![
                one(&[], i64::MAX),
                one(&[7], i64::MAX),
                one(&[10, 3, 8], 2),
                one(&[4, -2, 4], 0),
                one(&[5, 1], 4),
                one(&[1, 2, 10], 1),
                one(&[0, 500, 1000], 500),
                one(&[-1_000_000, 1_000_000], 2_000_000),
                one(&[9, 2, 14, 5, 20, 11, 17], 2),
            ],
            Variant::Distinct => vec![
                one(&[], 0),
                one(&[3, 1, 3], 2),
                one(&[-1, 0, 1, 0, -1], 3),
                one(&[2, 2, 2, 2], 1),
                one(&[4, 1, 4, 2, 9, 1, 7], 5),
            ],
            Variant::LowerMedian => vec![
                one(&[], 0),
                one(&[5], 5),
                one(&[3, 1, 2], 2),
                one(&[4, -2, 9, 0], 0),
                one(&[2, 1, 1], 1),
                one(&[7, 7, 7, 1], 7),
                one(&[4, 1, 4, 2, 9, 1, 7], 4),
            ],
        }
    }

    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let kind = kind_of(rng.seed());
        let narrow = narrow_bound(n);
        let xs: Vec<i64> = match kind {
            0 => (0..n)
                .map(|_| {
                    let bound = if rng.bool() { narrow } else { VALUE_BOUND };
                    rng.int_in(-bound, bound)
                })
                .collect(),
            1 => {
                let spread = spread_bound(n);
                (0..n).map(|_| rng.int_in(-spread, spread)).collect()
            }
            _ => {
                let step = (2 * VALUE_BOUND / n.max(1) as i64).max(1);
                let mut xs: Vec<i64> = (0..n as i64)
                    .map(|k| -VALUE_BOUND + k * step + rng.int_in(0, step / 2))
                    .collect();
                for i in (1..xs.len()).rev() {
                    xs.swap(i, rng.below(i as u64 + 1) as usize);
                }
                xs
            }
        };
        let mut inputs = vec![list(&xs)];
        if self.0 == Variant::PairSums {
            let plan = pair_sums_plan(rng.seed());
            // The seed's quantile: a fraction `q / 2⁶⁴` of the way along.
            let q = rng.seed_stream().next_u64();
            let t = if xs.len() >= 2 && plan.exact {
                let tied = if kind == 0 {
                    tied_thresholds(&xs, narrow)
                } else {
                    Vec::new()
                };
                if tied.is_empty() {
                    let sums = sorted(
                        xs.iter()
                            .enumerate()
                            .flat_map(|(i, &a)| xs[i + 1..].iter().map(move |&b| a + b))
                            .collect(),
                    );
                    sums[index_at(plan.place, q, sums.len() as u64) as usize]
                } else {
                    tied[index_at(plan.place, q, tied.len() as u64) as usize]
                }
            } else {
                let top = xs.iter().map(|x| x.abs()).max().unwrap_or(0).max(1);
                -2 * top + index_at(plan.place, q, 4 * top as u64 + 1) as i64
            };
            inputs.push(Value::Int(t));
        }
        inputs
    }

    fn oracle(&self, inputs: &[Value]) -> Value {
        let xs = ints_of(&inputs[0]);
        Value::Int(match self.0 {
            Variant::Inversions => inversions(&xs),
            Variant::PairSums => pair_sums(&xs, int_of(&inputs[1])),
            Variant::MinGap => min_gap(&xs),
            Variant::Distinct => distinct(&xs),
            Variant::LowerMedian => lower_median(&xs),
        })
    }

    fn sizes(&self) -> Vec<u64> {
        SIZES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answers by brute force over all pairs, independent of the oracles.
    fn brute(v: Variant, xs: &[i64], t: i64) -> i64 {
        let pairs =
            || (0..xs.len()).flat_map(move |i| (i + 1..xs.len()).map(move |j| (xs[i], xs[j])));
        match v {
            Variant::Inversions => pairs().filter(|(a, b)| a > b).count() as i64,
            Variant::PairSums => pairs().filter(|(a, b)| a + b <= t).count() as i64,
            Variant::MinGap => pairs()
                .map(|(a, b)| (a - b).abs())
                .min()
                .unwrap_or(i64::MAX),
            Variant::Distinct => {
                (0..xs.len()).filter(|&i| !xs[..i].contains(&xs[i])).count() as i64
            }
            Variant::LowerMedian => {
                if xs.is_empty() {
                    0
                } else {
                    let k = (xs.len() as i64 - 1) / 2;
                    *xs.iter()
                        .find(|&&x| {
                            let less = xs.iter().filter(|&&y| y < x).count() as i64;
                            let le = xs.iter().filter(|&&y| y <= x).count() as i64;
                            less <= k && k < le
                        })
                        .expect("a median")
                }
            }
        }
    }

    /// The generated inputs of `v` over the sweep, five seeds per size.
    fn generated(v: Variant) -> Vec<Inputs> {
        let f = SortedOrder(v);
        SIZES
            .iter()
            .flat_map(|&n| (0..5).map(move |seed| (n, seed)))
            .map(|(n, seed)| f.generate(n, &mut TaskRng::new(n, seed)))
            .collect()
    }

    #[test]
    fn generated_inputs_test_both_ties_and_gaps() {
        // A sweep of only tied inputs answers 0 for the least gap almost
        // every time, so it cannot tell a solution that finds equal values
        // but gets gaps wrong from a right one; and the reverse for spread
        // inputs. Both kinds must be common.
        let gaps: Vec<i64> = generated(Variant::MinGap)
            .iter()
            .map(|i| min_gap(&ints_of(&i[0])))
            .collect();
        let zero = gaps.iter().filter(|&&g| g == 0).count();
        let small = gaps.iter().filter(|&&g| 0 < g && g <= 64).count();
        let large = gaps.iter().filter(|&&g| g > 1_000).count();
        assert!(zero >= 5 && small >= 5 && large >= 5, "least gaps {gaps:?}");
        // The threshold is often an exact pair sum, where <= and < differ;
        // some thresholds are the sum of several pairs, so a count of exact
        // pairs is tested and not only whether there is one; and in some of
        // those, two equal elements are an exact pair.
        let pair_sums_inputs: Vec<(Vec<i64>, i64)> = generated(Variant::PairSums)
            .iter()
            .map(|i| (ints_of(&i[0]), int_of(&i[1])))
            .collect();
        let exact: Vec<i64> = pair_sums_inputs
            .iter()
            .map(|(xs, t)| exact_pairs(xs, *t))
            .collect();
        let some = exact.iter().filter(|&&e| e >= 1).count();
        let several = exact.iter().filter(|&&e| e >= 2).count();
        let twins = pair_sums_inputs
            .iter()
            .zip(&exact)
            .filter(|((xs, t), &e)| e >= 2 && xs.iter().filter(|&&x| 2 * x == *t).count() >= 2)
            .count();
        assert!(
            some >= 6 && several >= 5 && twins >= 3,
            "exact pairs per threshold {exact:?}: {some} with one or more, {several} with \
             two or more, {twins} of those with two equal elements"
        );
    }

    /// The kind of input `xs` of size `n` is, worked out from the elements:
    /// 2 if each element lies in the first half of its own slot once sorted,
    /// 1 if every element is within the spread bound, and 0 otherwise.
    fn observed_kind(n: u64, xs: &[i64]) -> u64 {
        let step = (2 * VALUE_BOUND / n.max(1) as i64).max(1);
        let in_slots = sorted(xs.to_vec()).iter().zip(0_i64..).all(|(&x, k)| {
            let lo = -VALUE_BOUND + k * step;
            lo <= x && x <= lo + step / 2
        });
        if in_slots {
            2
        } else if xs.iter().all(|x| x.abs() <= spread_bound(n)) {
            1
        } else {
            0
        }
    }

    #[test]
    fn each_seed_gets_one_kind_and_threshold_mode_at_every_size() {
        // Estimator C sees a program's growth through each seed's step
        // counts, so a seed must not change kind, or threshold mode, from one
        // size to the next (bead dreamcounter-0ea5b7).
        let f = SortedOrder(Variant::PairSums);
        let mut kinds_seen = Vec::new();
        let mut modes_seen = Vec::new();
        for seed in 0..6 {
            let at = |n: u64| {
                let i = f.generate(n, &mut TaskRng::new(n, seed));
                (ints_of(&i[0]), int_of(&i[1]))
            };
            let kinds: Vec<u64> = f
                .sizes()
                .into_iter()
                .map(|n| observed_kind(n, &at(n).0))
                .collect();
            assert!(
                kinds.iter().all(|&k| k == seed % 3),
                "seed {seed}: kinds {kinds:?} over the sizes"
            );
            kinds_seen.push(kinds[0]);
            let plan = pair_sums_plan(seed);
            modes_seen.push((kinds[0], plan.exact));
            if plan.exact {
                // Exact: some pair sums to t at every size.
                for n in f.sizes() {
                    let (xs, t) = at(n);
                    assert!(exact_pairs(&xs, t) >= 1, "seed {seed}, n = {n}: t = {t}");
                }
            } else {
                // Random: t is the same fraction of the way along
                // `-2 top..=2 top` at every size.
                let fractions: Vec<f64> = f
                    .sizes()
                    .into_iter()
                    .map(|n| {
                        let (xs, t) = at(n);
                        let top = xs.iter().map(|x| x.abs()).max().unwrap_or(0).max(1);
                        (t + 2 * top) as f64 / (4 * top + 1) as f64
                    })
                    .collect();
                let (lo, hi) = fractions
                    .iter()
                    .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
                assert!(hi - lo < 1e-3, "seed {seed}: fractions {fractions:?}");
            }
        }
        assert_eq!(kinds_seen[..3], [0, 1, 2], "the first three seeds");
        // The mode does not follow the kind: each kind gets both.
        modes_seen.sort();
        assert_eq!(
            modes_seen,
            [
                (0, false),
                (0, true),
                (1, false),
                (1, true),
                (2, false),
                (2, true)
            ],
            "(kind, exact) over seeds 0 to 5"
        );
    }

    #[test]
    fn every_kind_gets_every_pair_sums_plan() {
        // Over seeds 0 to 17 each kind takes all six plans: exact or
        // random, at the bottom, the top or a drawn quantile.
        for kind in 0..3 {
            let mut plans: Vec<(bool, u8)> = (0..18)
                .filter(|k| kind_of(*k) == kind)
                .map(|k| {
                    let p = pair_sums_plan(k);
                    (p.exact, p.place as u8)
                })
                .collect();
            plans.sort();
            plans.dedup();
            assert_eq!(plans.len(), 6, "kind {kind}: {plans:?}");
        }
    }

    #[test]
    fn default_seeds_give_pair_sums_answers_of_none_and_all() {
        // A solution can be wrong only when no pair counts, or only when
        // every pair does. Seed 2 (wide, -2 top) has answer 0 and seed 4
        // (spread, 2 top) every pair, at every size; seeds 0, 1 and 3 fall
        // in between, and 0 and 1 are exact.
        let f = SortedOrder(Variant::PairSums);
        for n in f.sizes() {
            let at = |seed: u64| {
                let i = f.generate(n, &mut TaskRng::new(n, seed));
                (ints_of(&i[0]), int_of(&i[1]))
            };
            let answer = |seed: u64| {
                let (xs, t) = at(seed);
                pair_sums(&xs, t)
            };
            let all = (n * (n - 1) / 2) as i64;
            assert_eq!(answer(2), 0, "n = {n}");
            assert_eq!(answer(4), all, "n = {n}");
            for seed in [0, 1, 3] {
                let a = answer(seed);
                assert!(0 < a && a < all, "seed {seed}, n = {n}: {a} of {all}");
            }
            for seed in [0, 1] {
                let (xs, t) = at(seed);
                assert!(exact_pairs(&xs, t) >= 1, "seed {seed}, n = {n}: t = {t}");
            }
        }
    }

    proptest::proptest! {
        #[test]
        fn oracles_agree_with_brute_force(
            xs in proptest::collection::vec(-20_i64..20, 0..40),
            t in -40_i64..40,
        ) {
            for v in Variant::ALL {
                let f = SortedOrder(v);
                let mut inputs = vec![list(&xs)];
                if v == Variant::PairSums {
                    inputs.push(Value::Int(t));
                }
                proptest::prop_assert_eq!(
                    f.oracle(&inputs),
                    Value::Int(brute(v, &xs, t)),
                    "{:?} on {:?}, t = {}", v, xs, t
                );
            }
        }
    }
}
