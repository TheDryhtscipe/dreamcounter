//! Constrained subsequence counting (bead P3.6): count the subsets of a
//! list's positions whose chosen values, in order of position, satisfy a
//! local rule, mod [`MODULUS`].
//!
//! The input is a list of `n` integers; [`Variant::SignChanges`] also takes
//! a bound `m`. The size is `n`. A subset of positions is any set of indices
//! `0..n`, the empty one and the whole list included; two subsets that
//! choose different positions are different even if their values are equal.
//! The variants are rules invented for this corpus, so a solution cannot be
//! recalled from a textbook:
//!
//! | variant | values | the chosen values `c₁, …, c_k` must |
//! |---|---|---|
//! | [`Variant::SumMod7`] | `-3..=3` | sum to a multiple of 7 |
//! | [`Variant::CloseSteps`] | `0..=9` | have `|c_{i+1} - c_i| <= 2` |
//! | [`Variant::NoThreeInARow`] | `-3..=3` | not include three consecutive positions |
//! | [`Variant::Zigzag`] | `0..=9` | go strictly up and down in turn |
//! | [`Variant::SignChanges`] | `-3..=3` | change sign at most `m` times |
//!
//! Each variant has three dc references in [`SOURCE`]: a naive one that
//! enumerates all `2ⁿ` subsets, a quadratic one over the last chosen
//! position, and a good one that carries the counts of a small state in one
//! pass (`n`). The fuel plan is built from the quadratic one, the slowest
//! that finishes on the sweep: the naive one finishes only at its smallest
//! sizes, so a plan from it would give the fuel cap everywhere else, and
//! every exponential candidate would run to the cap at every such size
//! before timing out.
//!
//! The inputs come in kinds chosen per variant, one kind per seed at every
//! size (see [`TaskFamily::generate`] on [`SubsequenceCount`]), so that the
//! boundary of each rule is met often:
//! differences of exactly 2 and 3 for close steps, long runs of one value
//! for zigzag, zeros and runs of one sign for sign changes.
//!
//! The oracle is a Rust dynamic programme over (position, state), with the
//! counts of each state in a `BTreeMap` (for sum mod 7, an array over the
//! seven residues); a brute force over all subsets checks it in this
//! module's tests.

use std::collections::BTreeMap;

use dc_core::Scheme;
use dc_eval::Value;

use crate::rng::TaskRng;
use crate::task::{Example, Inputs, TaskFamily};

/// The reference solutions in dc surface syntax, compiled against the
/// prelude. [`Variant::naive`], [`Variant::quadratic`] and [`Variant::good`]
/// name the definitions; the fuel plan is built from the quadratic one.
pub const SOURCE: &str = include_str!("subsequence_count.dc");

/// Answers are counts mod this prime.
pub const MODULUS: i64 = 1_000_003;

/// The sweep. It stops at 256: that is enough to tell `n` from `n²`, and
/// the quadratic references, which set the fuel plan, take a quarter as
/// many steps there as at 512.
pub const SIZES: &[u64] = &[8, 16, 32, 64, 128, 256];

/// [`Variant::SignChanges`] draws its bound `m` from `0..=MAX_M`.
pub const MAX_M: i64 = 4;

/// A variant of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The chosen values sum to a multiple of 7. Values in `-3..=3`.
    SumMod7,
    /// Consecutive chosen values differ by at most 2. Values in `0..=9`.
    CloseSteps,
    /// No three consecutive positions are all chosen. Values in `-3..=3`.
    NoThreeInARow,
    /// The chosen values go strictly up and down in turn, starting either
    /// way. Values in `0..=9`.
    Zigzag,
    /// At most `m` sign changes between consecutive non-zero chosen values;
    /// 0 has no sign. Values in `-3..=3`.
    SignChanges,
}

impl Variant {
    /// Every variant, in the order of the table above.
    pub const ALL: [Variant; 5] = [
        Variant::SumMod7,
        Variant::CloseSteps,
        Variant::NoThreeInARow,
        Variant::Zigzag,
        Variant::SignChanges,
    ];

    /// The definition in [`SOURCE`] of the naive (`2ⁿ`) reference.
    pub fn naive(self) -> &'static str {
        match self {
            Variant::SumMod7 => "sum_mod_7_naive",
            Variant::CloseSteps => "close_steps_naive",
            Variant::NoThreeInARow => "no_three_in_a_row_naive",
            Variant::Zigzag => "zigzag_naive",
            Variant::SignChanges => "sign_changes_naive",
        }
    }

    /// The definition in [`SOURCE`] of the quadratic (`n²`) reference.
    pub fn quadratic(self) -> &'static str {
        match self {
            Variant::SumMod7 => "sum_mod_7_quadratic",
            Variant::CloseSteps => "close_steps_quadratic",
            Variant::NoThreeInARow => "no_three_in_a_row_quadratic",
            Variant::Zigzag => "zigzag_quadratic",
            Variant::SignChanges => "sign_changes_quadratic",
        }
    }

    /// The definition in [`SOURCE`] of the good (`n`) reference.
    pub fn good(self) -> &'static str {
        match self {
            Variant::SumMod7 => "sum_mod_7_good",
            Variant::CloseSteps => "close_steps_good",
            Variant::NoThreeInARow => "no_three_in_a_row_good",
            Variant::Zigzag => "zigzag_good",
            Variant::SignChanges => "sign_changes_good",
        }
    }

    /// The values elements are drawn from, `lo..=hi`.
    pub fn value_range(self) -> (i64, i64) {
        match self {
            Variant::CloseSteps | Variant::Zigzag => (0, 9),
            Variant::SumMod7 | Variant::NoThreeInARow | Variant::SignChanges => (-3, 3),
        }
    }
}

/// One variant of the subsequence-counting family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsequenceCount(pub Variant);

/// Every variant as a family.
pub fn all() -> Vec<SubsequenceCount> {
    Variant::ALL.into_iter().map(SubsequenceCount).collect()
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

fn list(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

/// The first `len` elements of `pattern` repeated.
fn repeat(pattern: &[i64], len: usize) -> Vec<i64> {
    pattern.iter().copied().cycle().take(len).collect()
}

/// The number of subsets of positions, mod [`MODULUS`], whose chosen values
/// keep the rule, where `step` gives the state after choosing a value, or
/// `None` if choosing it breaks the rule, and `skip` the state after
/// skipping a position.
fn count<S: Ord + Clone>(
    xs: &[i64],
    start: S,
    step: impl Fn(&S, i64) -> Option<S>,
    skip: impl Fn(&S) -> S,
) -> i64 {
    let mut counts: BTreeMap<S, i64> = BTreeMap::from([(start, 1)]);
    for &x in xs {
        let mut next: BTreeMap<S, i64> = BTreeMap::new();
        for (s, &c) in &counts {
            for t in [Some(skip(s)), step(s, x)].into_iter().flatten() {
                let e = next.entry(t).or_insert(0);
                *e = (*e + c) % MODULUS;
            }
        }
        counts = next;
    }
    counts.values().fold(0, |acc, &c| (acc + c) % MODULUS)
}

fn same<S: Clone>(s: &S) -> S {
    s.clone()
}

/// The state of the zigzag rule: nothing chosen, or the last chosen value
/// and the direction of the step that reached it.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Zig {
    Empty,
    One(i64),
    Rose(i64),
    Fell(i64),
}

fn oracle_count(v: Variant, xs: &[i64], m: i64) -> i64 {
    debug_assert!(m >= 0, "the bound m is at least 0");
    match v {
        Variant::SumMod7 => {
            // The counts of every residue, then the one of residue 0.
            let mut by_residue = [0_i64; 7];
            by_residue[0] = 1;
            for &x in xs {
                let mut next = by_residue;
                for (r, &c) in by_residue.iter().enumerate() {
                    let t = (r as i64 + x).rem_euclid(7) as usize;
                    next[t] = (next[t] + c) % MODULUS;
                }
                by_residue = next;
            }
            by_residue[0]
        }
        Variant::CloseSteps => count(
            xs,
            None,
            |last: &Option<i64>, x| match last {
                Some(l) if (x - l).abs() > 2 => None,
                _ => Some(Some(x)),
            },
            same,
        ),
        Variant::NoThreeInARow => count(xs, 0_u8, |&run, _| (run < 2).then_some(run + 1), |_| 0),
        Variant::Zigzag => count(
            xs,
            Zig::Empty,
            |z, x| match *z {
                Zig::Empty => Some(Zig::One(x)),
                Zig::One(l) if x > l => Some(Zig::Rose(x)),
                Zig::One(l) if x < l => Some(Zig::Fell(x)),
                Zig::Rose(l) if x < l => Some(Zig::Fell(x)),
                Zig::Fell(l) if x > l => Some(Zig::Rose(x)),
                _ => None,
            },
            same,
        ),
        Variant::SignChanges => count(
            xs,
            (0_i64, 0_i64),
            |&(changes, last_sign), x| {
                let sx = x.signum();
                if sx == 0 {
                    Some((changes, last_sign))
                } else if last_sign == 0 || last_sign == sx {
                    Some((changes, sx))
                } else {
                    (changes < m).then_some((changes + 1, sx))
                }
            },
            same,
        ),
    }
}

/// The bound `m` of [`Variant::SignChanges`] for seed `seed`, the same at
/// every size: `3 seed mod (MAX_M + 1)`. The kind is `seed mod 3`, so the
/// bound does not follow it: over seeds 0 to 14 every kind gets every
/// bound. The default five seeds give the uniform inputs bounds 0 and 4,
/// the half-zero ones 3 and 2, and the runs of signs 1.
fn bound_of(seed: u64) -> i64 {
    ((3 * seed) % (MAX_M as u64 + 1)) as i64
}

impl TaskFamily for SubsequenceCount {
    fn name(&self) -> &str {
        match self.0 {
            Variant::SumMod7 => "subsequence-count/sum-mod-7",
            Variant::CloseSteps => "subsequence-count/close-steps",
            Variant::NoThreeInARow => "subsequence-count/no-three-in-a-row",
            Variant::Zigzag => "subsequence-count/zigzag",
            Variant::SignChanges => "subsequence-count/sign-changes",
        }
    }

    fn signature(&self) -> Scheme {
        let s = match self.0 {
            Variant::SignChanges => "List Int -> Int -> Int",
            _ => "List Int -> Int",
        };
        s.parse().expect("a scheme")
    }

    fn description(&self) -> &str {
        match self.0 {
            Variant::SumMod7 => {
                "Count the ways to choose a set of positions of the list (any number of \
                 them, none or all included) such that the elements at the chosen \
                 positions sum to a multiple of 7 (0, 7, -7, 14, ...). Choosing no \
                 positions gives sum 0. Two choices are different if they choose \
                 different positions, even when the elements there are equal. Every \
                 element is between -3 and 3. Return the count mod 1000003."
            }
            Variant::CloseSteps => {
                "Count the ways to choose a set of positions of the list (any number of \
                 them, none or all included) such that, reading the elements at the \
                 chosen positions from left to right, each differs from the one before \
                 it by at most 2. Choosing no positions or one position always counts. \
                 Two choices are different if they choose different positions, even \
                 when the elements there are equal. Every element is between 0 and 9. \
                 Return the count mod 1000003."
            }
            Variant::NoThreeInARow => {
                "Count the ways to choose a set of positions of the list (any number of \
                 them, none or all included) such that no three consecutive positions \
                 i, i + 1, i + 2 are all chosen. Two choices are different if they \
                 choose different positions. Return the count mod 1000003."
            }
            Variant::Zigzag => {
                "Count the ways to choose a set of positions of the list (any number of \
                 them, none or all included) such that, reading the elements at the \
                 chosen positions from left to right, they go strictly up and strictly \
                 down in turn: each is greater than the one before it, then smaller, \
                 then greater, and so on, or smaller, then greater, and so on. No two \
                 consecutive chosen elements may be equal. Choosing no positions or one \
                 position always counts. Two choices are different if they choose \
                 different positions, even when the elements there are equal. Every \
                 element is between 0 and 9. Return the count mod 1000003."
            }
            Variant::SignChanges => {
                "Given a list of integers and a bound m >= 0, count the ways to choose a \
                 set of positions of the list (any number of them, none or all \
                 included) such that the elements at the chosen positions change sign \
                 at most m times. Read the chosen elements from left to right and \
                 ignore those equal to 0, which have no sign; a sign change is a \
                 positive element followed by a negative one, or a negative one \
                 followed by a positive one, in what remains. Two choices are \
                 different if they choose different positions, even when the elements \
                 there are equal. Every element is between -3 and 3. Return the count \
                 mod 1000003."
            }
        }
    }

    fn examples(&self) -> Vec<Example> {
        let one = |xs: &[i64], out: i64| Example::new(vec![list(xs)], Value::Int(out));
        let two = |xs: &[i64], m: i64, out: i64| {
            Example::new(vec![list(xs), Value::Int(m)], Value::Int(out))
        };
        // The longest examples have 12 elements, so that a correct solution
        // that enumerates every subset passes them within the example fuel;
        // so the counts stay under the modulus, and only the sweep checks it.
        let long = repeat(&[1, -2, 0, 3, -1, 2, -3], 12);
        match self.0 {
            Variant::SumMod7 => vec![
                one(&[], 1),
                one(&[0], 2),
                one(&[3, -3], 2),
                one(&[-3, 3, 0], 4),
                one(&[3, 3, 1, -3, -1], 7),
                one(&[-3, -3, -1, 2, 3, 3, -2], 19),
                one(&long, 588),
            ],
            Variant::CloseSteps => vec![
                one(&[], 1),
                one(&[5], 2),
                one(&[1, 3], 4),
                one(&[1, 4], 3),
                one(&[2, 2, 2], 8),
                one(&[0, 2, 5, 3, 9, 7], 13),
                one(&repeat(&[4, 6, 5, 3, 4], 12), 2_810),
            ],
            Variant::NoThreeInARow => vec![
                one(&[], 1),
                one(&[3], 2),
                one(&[1, 1], 4),
                one(&[1, 2, 3], 7),
                one(&[0, 0, 0, 0], 13),
                one(&[-3, 2, 0, 1, 1, -1, 3], 81),
                one(&long, 1_705),
            ],
            Variant::Zigzag => vec![
                one(&[], 1),
                one(&[4], 2),
                one(&[3, 3], 3),
                one(&[1, 2], 4),
                one(&[1, 2, 3], 7),
                one(&[1, 3, 2], 8),
                one(&[5, 5, 1, 9, 9, 0], 34),
                one(&repeat(&[4, 7, 4, 4, 1, 8], 12), 511),
            ],
            Variant::SignChanges => vec![
                two(&[], 0, 1),
                two(&[2, -1], 0, 3),
                two(&[2, -1], 1, 4),
                two(&[1, 0, -1], 0, 6),
                two(&[0, 0, 3], 0, 8),
                two(&[1, -1, 1, -1], 1, 13),
                two(&[1, -1, 1, -1], 2, 15),
                two(&[3, -2, 0, -1, 2, 0, -3, 1], 3, 244),
                two(&[1, -1, 1, -1, 1, -1], 10, 64),
                two(&[1, -1, 1, -1, 1, -1, 1, -1], 5, 253),
                two(&[1, -1, 1, -1, 1, -1, 1, -1], 6, 255),
                two(&long, 2, 1_852),
            ],
        }
    }

    /// An input of size `n`. Elements are drawn from
    /// [`Variant::value_range`] in one of the three kinds below. Seed `k` gets
    /// the `k mod 3`th kind, counting from 0, at every size, so the default
    /// five seeds have every kind. Each seed's step counts then follow one
    /// kind of input across the sweep, and estimator C can see a program's
    /// growth through them (bead dreamcounter-0ea5b7).
    ///
    /// - Sum mod 7 and no three in a row: uniform; or every element of one
    ///   sign (0 included), so sums run far from 0.
    /// - Close steps: uniform; values from a window of four; or values from
    ///   `{a, a + 2, a + 3}`, so differences of exactly 2 and 3 are common.
    /// - Zigzag: uniform; or two neighbouring values, so ties are common.
    /// - Sign changes: uniform; half the elements 0; or signs in runs of 1
    ///   to 8 elements. The bound `m` is `3 k mod (MAX_M + 1)` for seed
    ///   `k`, the same at every size, as a solution's cost can depend on it;
    ///   the default five seeds have every bound from 0 to `MAX_M`.
    ///
    /// The choices that set an input's character are drawn once per seed,
    /// from [`TaskRng::seed_stream`], so they are the same at every size:
    /// the sign of a one-sign input, the offset `a` of close steps and
    /// zigzag, and the first sign of the runs. The elements, which elements
    /// are 0, and the lengths of the runs are drawn afresh at each size.
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let (lo, hi) = self.0.value_range();
        let mut per_seed = rng.seed_stream();
        let xs: Vec<i64> = match (self.0, rng.seed() % 3) {
            (_, 0) => (0..n).map(|_| rng.int_in(lo, hi)).collect(),
            (Variant::SumMod7 | Variant::NoThreeInARow, _) => {
                let (a, b) = if per_seed.bool() { (0, 3) } else { (-3, 0) };
                (0..n).map(|_| rng.int_in(a, b)).collect()
            }
            (Variant::CloseSteps, 1) => {
                let a = per_seed.int_in(0, 6);
                (0..n).map(|_| rng.int_in(a, a + 3)).collect()
            }
            (Variant::CloseSteps, _) => {
                let a = per_seed.int_in(0, 6);
                (0..n)
                    .map(|_| a + [0, 2, 3][rng.below(3) as usize])
                    .collect()
            }
            (Variant::Zigzag, _) => {
                let a = per_seed.int_in(0, 8);
                (0..n).map(|_| a + rng.int_in(0, 1)).collect()
            }
            (Variant::SignChanges, 1) => (0..n)
                .map(|_| if rng.bool() { 0 } else { rng.int_in(lo, hi) })
                .collect(),
            (Variant::SignChanges, _) => {
                let mut xs = Vec::with_capacity(n as usize);
                let mut positive = per_seed.bool();
                while (xs.len() as u64) < n {
                    let run = rng.int_in(1, 8);
                    for _ in 0..run {
                        if (xs.len() as u64) < n {
                            let v = rng.int_in(0, 3);
                            xs.push(if positive { v } else { -v });
                        }
                    }
                    positive = !positive;
                }
                xs
            }
        };
        let mut inputs = vec![list(&xs)];
        if self.0 == Variant::SignChanges {
            inputs.push(Value::Int(bound_of(rng.seed())));
        }
        inputs
    }

    fn oracle(&self, inputs: &[Value]) -> Value {
        let xs = ints_of(&inputs[0]);
        let m = if self.0 == Variant::SignChanges {
            int_of(&inputs[1])
        } else {
            0
        };
        Value::Int(oracle_count(self.0, &xs, m))
    }

    fn sizes(&self) -> Vec<u64> {
        SIZES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the chosen values `c` satisfy the rule, checked on the
    /// sequence itself, independent of the oracle's states.
    fn allowed(v: Variant, c: &[i64], positions: &[usize], m: i64) -> bool {
        match v {
            Variant::SumMod7 => c.iter().sum::<i64>() % 7 == 0,
            Variant::CloseSteps => c.windows(2).all(|w| (w[1] - w[0]).abs() <= 2),
            Variant::NoThreeInARow => positions
                .windows(3)
                .all(|w| !(w[1] == w[0] + 1 && w[2] == w[1] + 1)),
            Variant::Zigzag => {
                let d: Vec<i64> = c.windows(2).map(|w| (w[1] - w[0]).signum()).collect();
                d.iter().all(|&s| s != 0) && d.windows(2).all(|w| w[0] != w[1])
            }
            Variant::SignChanges => {
                let s: Vec<i64> = c.iter().filter(|&&x| x != 0).map(|x| x.signum()).collect();
                s.windows(2).filter(|w| w[0] != w[1]).count() as i64 <= m
            }
        }
    }

    /// The count by enumerating every subset.
    fn brute(v: Variant, xs: &[i64], m: i64) -> i64 {
        let n = xs.len();
        (0_u32..1 << n)
            .filter(|mask| {
                let positions: Vec<usize> = (0..n).filter(|i| mask >> i & 1 == 1).collect();
                let c: Vec<i64> = positions.iter().map(|&i| xs[i]).collect();
                allowed(v, &c, &positions, m)
            })
            .count() as i64
            % MODULUS
    }

    #[test]
    fn examples_are_in_the_value_range() {
        for v in Variant::ALL {
            let (lo, hi) = v.value_range();
            for e in SubsequenceCount(v).examples() {
                let xs = ints_of(&e.inputs[0]);
                assert!(xs.iter().all(|&x| lo <= x && x <= hi), "{v:?}: {xs:?}");
            }
        }
    }

    #[test]
    fn short_examples_agree_with_brute_force() {
        for v in Variant::ALL {
            for e in SubsequenceCount(v).examples() {
                let xs = ints_of(&e.inputs[0]);
                if xs.len() <= 16 {
                    let m = e.inputs.get(1).map_or(0, int_of);
                    assert_eq!(e.output, Value::Int(brute(v, &xs, m)), "{v:?} on {xs:?}");
                }
            }
        }
    }

    /// The oracle on 100 elements, where every count exceeds both the
    /// modulus and `i64` unless it is reduced at every step,
    /// against counts worked out apart from it: every subset is allowed for
    /// sum mod 7 on zeros, close steps on equal values and sign changes on
    /// positive values; no three in a row allows the bit strings with no
    /// three ones in a row.
    #[test]
    fn counts_past_the_modulus() {
        // Written out, not `MODULUS`, so a wrong constant fails here.
        let p = 1_000_003_u128;
        let n = 100;
        let all = (1_u128 << n) % p;
        let mut no_three = [1_u128, 2, 4];
        for _ in 3..=n {
            no_three = [no_three[1], no_three[2], no_three.iter().sum()];
        }
        let no_three = no_three[2] % p;
        for (v, x, expected) in [
            (Variant::SumMod7, 0, all),
            (Variant::CloseSteps, 5, all),
            (Variant::SignChanges, 2, all),
            (Variant::NoThreeInARow, 1, no_three),
        ] {
            let xs = vec![x; n];
            assert_eq!(oracle_count(v, &xs, 0) as u128, expected, "{v:?}");
        }
    }

    /// Whether `xs` looks uniform for `v`, not one of its narrower kinds:
    /// both signs for sum mod 7 and no three in a row, more than two values
    /// for zigzag, and a span over 3 for close steps. Sign changes has no
    /// such test; its bound `m` is checked instead.
    fn looks_uniform(v: Variant, xs: &[i64]) -> Option<bool> {
        let span = xs.iter().max().unwrap_or(&0) - xs.iter().min().unwrap_or(&0);
        match v {
            Variant::SumMod7 | Variant::NoThreeInARow => {
                Some(xs.iter().any(|&x| x > 0) && xs.iter().any(|&x| x < 0))
            }
            Variant::Zigzag => Some(span > 1),
            Variant::CloseSteps => Some(span > 3),
            Variant::SignChanges => None,
        }
    }

    #[test]
    fn each_seed_gets_one_kind_at_every_size() {
        // Estimator C sees a program's growth through each seed's step
        // counts, so a seed must not change kind, or bound, from one size to
        // the next (bead dreamcounter-0ea5b7).
        for v in Variant::ALL {
            let f = SubsequenceCount(v);
            for seed in 0..6 {
                let inputs: Vec<Inputs> = f
                    .sizes()
                    .into_iter()
                    .map(|n| f.generate(n, &mut TaskRng::new(n, seed)))
                    .collect();
                // At 8 elements a uniform input can have one sign by chance
                // (seed 3 of sum mod 7 does), so the kind is read from 16 up.
                let uniform: Vec<Option<bool>> = inputs
                    .iter()
                    .map(|i| ints_of(&i[0]))
                    .filter(|xs| xs.len() >= 16)
                    .map(|xs| looks_uniform(v, &xs))
                    .collect();
                if v != Variant::SignChanges {
                    let expected = Some(seed % 3 == 0);
                    assert!(
                        uniform.iter().all(|&u| u == expected),
                        "{v:?}, seed {seed}: uniform {uniform:?} over the sizes"
                    );
                } else {
                    let bounds: Vec<i64> = inputs.iter().map(|i| int_of(&i[1])).collect();
                    assert!(
                        bounds.iter().all(|&m| m == bound_of(seed)),
                        "seed {seed}: bounds {bounds:?} over the sizes"
                    );
                }
            }
        }
    }

    /// What sets the character of a non-uniform input, read from it: the
    /// sign of a one-sign input (sum mod 7, no three in a row), the least
    /// value (close steps, zigzag: the offset `a` once `a` is drawn), or the
    /// sign of the first non-zero element (the runs of sign changes).
    fn character(v: Variant, xs: &[i64]) -> i64 {
        match v {
            Variant::SumMod7 | Variant::NoThreeInARow => xs.iter().sum::<i64>().signum(),
            Variant::CloseSteps | Variant::Zigzag => *xs.iter().min().unwrap_or(&0),
            Variant::SignChanges => xs.iter().find(|&&x| x != 0).map_or(0, |x| x.signum()),
        }
    }

    #[test]
    fn each_seed_keeps_its_sign_and_offset_at_every_size() {
        // A natural solution's cost can depend on the sign of a one-sign
        // input, the offset of close steps and zigzag, and where the runs of
        // signs start, so each is drawn once per seed (bead
        // dreamcounter-0ea5b7). Read from 32 up, where the least value is
        // the offset in every such input of these seeds.
        for v in Variant::ALL {
            let f = SubsequenceCount(v);
            let mut seen = Vec::new();
            for seed in (0..12).filter(|s| s % 3 != 0) {
                if v == Variant::SignChanges && seed % 3 != 2 {
                    continue;
                }
                let chars: Vec<i64> = f
                    .sizes()
                    .into_iter()
                    .filter(|&n| n >= 32)
                    .map(|n| character(v, &ints_of(&f.generate(n, &mut TaskRng::new(n, seed))[0])))
                    .collect();
                assert!(
                    chars.iter().all(|&c| c == chars[0]),
                    "{v:?}, seed {seed}: {chars:?} over the sizes"
                );
                seen.push(chars[0]);
            }
            // And the seeds do not all agree, so the test sees a real draw.
            assert!(
                seen.iter().any(|&c| c != seen[0]),
                "{v:?}: every seed gave {seen:?}"
            );
        }
    }

    #[test]
    fn sign_changes_bound_does_not_follow_the_kind() {
        // Over seeds 0 to 14 every kind (seed mod 3) gets every bound; the
        // default five have every bound, and the runs of seed 2 get 1.
        let mut pairs: Vec<(u64, i64)> = (0..15).map(|s| (s % 3, bound_of(s))).collect();
        pairs.sort();
        pairs.dedup();
        assert_eq!(pairs.len(), 15, "{pairs:?}");
        assert_eq!((0..5).map(bound_of).collect::<Vec<i64>>(), [0, 3, 1, 4, 2]);
    }

    proptest::proptest! {
        #[test]
        fn oracles_agree_with_brute_force(
            raw in proptest::collection::vec(0_i64..10, 0..13),
            m in 0_i64..6,
        ) {
            for v in Variant::ALL {
                let f = SubsequenceCount(v);
                let (lo, hi) = v.value_range();
                let xs: Vec<i64> = raw.iter().map(|&x| lo + x % (hi - lo + 1)).collect();
                let mut inputs = vec![list(&xs)];
                if v == Variant::SignChanges {
                    inputs.push(Value::Int(m));
                }
                proptest::prop_assert_eq!(
                    f.oracle(&inputs),
                    Value::Int(brute(v, &xs, m)),
                    "{:?} on {:?}, m = {}", v, xs, m
                );
            }
        }
    }
}
