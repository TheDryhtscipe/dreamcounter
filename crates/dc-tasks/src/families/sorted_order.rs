//! Sorted-order queries (bead P3.5): questions about a list of integers that
//! a scan answers once the list is sorted.
//!
//! The input is a list of `n` integers in `[-10⁶, 10⁶]`; [`Variant::PairSums`]
//! also takes a threshold `t`. The size is `n`. Ties matter differently to
//! each variant, so inputs come in two kinds, with even odds:
//!
//! - tied: each element is drawn, with even odds, from the whole range or
//!   from a narrow one of about `n / 2` values (see [`narrow_bound`]), so the
//!   input has many equal values;
//! - spread: elements are drawn from about `2 n²` values (see
//!   [`spread_bound`]), so most inputs have none equal, but neighbours once
//!   sorted are close, and the least gap is small and not 0.
//!
//! Half the time the threshold of [`Variant::PairSums`] is the sum of a
//! random pair of elements, so some pair sums to exactly `t`; otherwise it is
//! drawn from `[-10⁶, 10⁶]`. So a solution that mishandles ties, or gets
//! only ties right, fails the sweep. The variants:
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
                ]
            }
            Variant::MinGap => vec![
                one(&[], i64::MAX),
                one(&[7], i64::MAX),
                one(&[10, 3, 8], 2),
                one(&[4, -2, 4], 0),
                one(&[5, 1], 4),
                one(&[1, 2, 10], 1),
            ],
            Variant::Distinct => vec![
                one(&[], 0),
                one(&[3, 1, 3], 2),
                one(&[-1, 0, 1, 0, -1], 3),
                one(&[2, 2, 2, 2], 1),
            ],
            Variant::LowerMedian => vec![
                one(&[], 0),
                one(&[5], 5),
                one(&[3, 1, 2], 2),
                one(&[4, -2, 9, 0], 0),
                one(&[2, 1, 1], 1),
                one(&[7, 7, 7, 1], 7),
            ],
        }
    }

    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let xs: Vec<i64> = if rng.bool() {
            let narrow = narrow_bound(n);
            (0..n)
                .map(|_| {
                    let bound = if rng.bool() { narrow } else { VALUE_BOUND };
                    rng.int_in(-bound, bound)
                })
                .collect()
        } else {
            let spread = spread_bound(n);
            (0..n).map(|_| rng.int_in(-spread, spread)).collect()
        };
        let mut inputs = vec![list(&xs)];
        if self.0 == Variant::PairSums {
            let t = if xs.len() >= 2 && rng.bool() {
                let i = rng.below(xs.len() as u64) as usize;
                let j = (i + 1 + rng.below(xs.len() as u64 - 1) as usize) % xs.len();
                xs[i] + xs[j]
            } else {
                rng.int_in(-VALUE_BOUND, VALUE_BOUND)
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
        assert!(zero >= 6 && small >= 6, "least gaps {gaps:?}");
        // The threshold is often an exact pair sum, where <= and < differ.
        let exact = generated(Variant::PairSums)
            .iter()
            .filter(|i| {
                let (xs, t) = (ints_of(&i[0]), int_of(&i[1]));
                pair_sums(&xs, t) != pair_sums(&xs, t - 1)
            })
            .count();
        assert!(exact >= 6, "{exact} of 30 thresholds are a pair sum");
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
