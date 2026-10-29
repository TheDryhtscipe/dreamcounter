//! Collapse games (bead P3.7): a list of small integers is rewritten by a
//! local rule on adjacent pairs, always at the leftmost pair the rule
//! applies to, until no such pair is left.
//!
//! The input is a list of `n` integers; the size is `n`. A *redex* is a pair
//! of adjacent elements `x, y` (`x` just left of `y`) that the rule applies
//! to. One rewrite replaces the leftmost redex by what the rule makes of it,
//! in its place. The rules are invented for this corpus:
//!
//! | variant | values | the pair `x, y` is a redex when | and becomes | answer |
//! |---|---|---|---|---|
//! | [`Variant::Merge`] | `0..=3` | `x = y` | `x + 1` | the final list |
//! | [`Variant::Cancel`] | `-3..=3` | `x + y = 0` | nothing | the final list |
//! | [`Variant::Absorb`] | `0..=9` | `x - y >= 2` | `(x + y) mod 10` | the final list |
//! | [`Variant::AbsorbCount`] | `0..=9` | as absorb | as absorb | the number of rewrites |
//!
//! Each variant has two dc references in [`SOURCE`]: a naive one that finds
//! the leftmost redex from the front, rewrites it and starts again, and a
//! good one that folds the list once with a stack (`n`). The naive one costs
//! the sum, over the rewrites, of the redex's position, which is about `n²`
//! when the inputs make about `n` rewrites at a depth of about `n`; the
//! generator draws such inputs (see [`TaskFamily::generate`] on
//! [`Collapse`]). The fuel plan is built from the naive reference.
//!
//! Every seed's input of a size makes the same rewrites at the same depths,
//! so the references' step counts do not depend on the seed and estimator C
//! sees no seed noise. The price is that the sweep meets only the shapes of
//! that schedule: a redex alone, a rewrite whose result makes a new redex
//! with the element before it, two redexes sharing an element, and, except
//! for cancel, a cascade: a long chain of rewrites from one element. A
//! solution that stops after `k` rewrites per element pushed is right on
//! every input without a longer chain. The absorb cascade is a chain of 8,
//! the longest any absorb input has. The merge cascade is a chain of 6: a
//! chain of `k` merges needs at least `2^(k - 3) + 3` elements, as each
//! element is worth at most `2^3` and a merge keeps the sum of `2^x`, and
//! 11 is what a cycle has room for. So a merge solution capped at 6 or
//! more passes the sweep. The examples add merge chains of 7, 8 and 9 and
//! merge results above 4, so a cap of 9, 10 or 11 is the only one they and
//! the sweep miss that some input of at most 1024 elements catches.
//!
//! The oracle simulates the rewriting literally: find the leftmost redex,
//! splice in its replacement, repeat. This module's tests check it against a
//! stack fold written apart from it, which is the claim the good references
//! rest on: the stack never holds a redex, so the leftmost redex can only be
//! its top and the next element.

use std::collections::HashMap;

use dc_core::Scheme;
use dc_eval::Value;

use crate::growth::DEFAULT_SIZES;
use crate::rng::TaskRng;
use crate::task::{Example, Inputs, TaskFamily};

/// The reference solutions in dc surface syntax, compiled against the
/// prelude. [`Variant::naive`] and [`Variant::good`] name the definitions;
/// the fuel plan is built from the naive one.
pub const SOURCE: &str = include_str!("collapse.dc");

/// A variant of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// An equal pair `x, x` becomes `x + 1`. Values in `0..=3`.
    Merge,
    /// A pair summing to 0 disappears. Values in `-3..=3`.
    Cancel,
    /// `x` absorbs `y` when `x - y >= 2`, and the pair becomes
    /// `(x + y) mod 10`. Values in `0..=9`.
    Absorb,
    /// The number of rewrites under the absorb rule. Values in `0..=9`.
    AbsorbCount,
}

impl Variant {
    /// Every variant, in the order of the table above.
    pub const ALL: [Variant; 4] = [
        Variant::Merge,
        Variant::Cancel,
        Variant::Absorb,
        Variant::AbsorbCount,
    ];

    /// The definition in [`SOURCE`] of the naive (`n²`) reference.
    pub fn naive(self) -> &'static str {
        match self {
            Variant::Merge => "merge_naive",
            Variant::Cancel => "cancel_naive",
            Variant::Absorb => "absorb_naive",
            Variant::AbsorbCount => "absorb_count_naive",
        }
    }

    /// The definition in [`SOURCE`] of the good (`n`) reference.
    pub fn good(self) -> &'static str {
        match self {
            Variant::Merge => "merge_good",
            Variant::Cancel => "cancel_good",
            Variant::Absorb => "absorb_good",
            Variant::AbsorbCount => "absorb_count_good",
        }
    }

    /// The values input elements are drawn from, `lo..=hi`.
    pub fn value_range(self) -> (i64, i64) {
        match self {
            Variant::Merge => (0, 3),
            Variant::Cancel => (-3, 3),
            Variant::Absorb | Variant::AbsorbCount => (0, 9),
        }
    }

    /// The rule's rewrite of the pair `x, y`, or `None` if it is not a
    /// redex.
    fn rewrite(self, x: i64, y: i64) -> Option<Vec<i64>> {
        match self {
            Variant::Merge => (x == y).then(|| vec![x + 1]),
            Variant::Cancel => (x + y == 0).then(Vec::new),
            Variant::Absorb | Variant::AbsorbCount => {
                (x - y >= 2).then(|| vec![(x + y).rem_euclid(10)])
            }
        }
    }
}

/// The sweep of every variant but cancel. It starts at 32 because the
/// generator's cycle of units with a cascade is 16 elements: at 16 the
/// second half would hold only part of one, and the references' step
/// counts there would not follow the `n²` and `n` of the larger sizes.
pub const CASCADE_SIZES: &[u64] = &[32, 64, 128, 256, 512, 1024];

/// One variant of the collapse family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Collapse(pub Variant);

/// Every variant as a family.
pub fn all() -> Vec<Collapse> {
    Variant::ALL.into_iter().map(Collapse).collect()
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

/// The final list and the number of rewrites, by rewriting the leftmost
/// redex until there is none.
fn simulate(v: Variant, xs: &[i64]) -> (Vec<i64>, i64) {
    let mut xs = xs.to_vec();
    let mut rewrites = 0;
    loop {
        let redex = (0..xs.len().saturating_sub(1))
            .find_map(|i| v.rewrite(xs[i], xs[i + 1]).map(|r| (i, r)));
        match redex {
            Some((i, r)) => {
                xs.splice(i..i + 2, r);
                rewrites += 1;
            }
            None => return (xs, rewrites),
        }
    }
}

/// The stack (top last) after pushing each of `xs` onto `stack`, rewriting
/// while its top and the element pushed are a redex, and the position in
/// the stack of the left element of each rewrite, in order.
fn fold(v: Variant, stack: &[i64], xs: &[i64]) -> (Vec<i64>, Vec<usize>) {
    let mut stack = stack.to_vec();
    let mut at = Vec::new();
    for &x in xs {
        let mut pending = vec![x];
        while let Some(c) = pending.pop() {
            match stack.last().and_then(|&t| v.rewrite(t, c)) {
                Some(r) => {
                    stack.pop();
                    at.push(stack.len());
                    pending.extend(r);
                }
                None => stack.push(c),
            }
        }
    }
    (stack, at)
}

/// A unit of the generated inputs: a few elements, pushed on a stack whose
/// top is `t` at position 0, that make a fixed list of rewrites and leave
/// `t` in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    /// `x, y`: `x` is no redex with `t`, and `x, y` is one. One rewrite.
    Pair,
    /// `x, z, y` (for cancel, `x, z, y, w`): `z, y` is a redex, and what it
    /// leaves makes a redex with `x` (for cancel, `x, w` is one). Two
    /// rewrites, the second at `x` after the first at `z`.
    Nested,
    /// `x, y, z` where both `x, y` and `y, z` are redexes. Only the left one
    /// is rewritten, so a solution that rewrites another redex first gets
    /// another answer whenever the two disagree. One rewrite for merge; for
    /// absorb, two, the second where its result absorbs `z`, because no
    /// absorb unit with one rewrite leaves a top of 8 or 9 in place.
    Overlap,
    /// A long chain of rewrites from the last element. Merge: the 11
    /// elements `3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0`, which build `5, 4, 3, 2,
    /// 1, 0` and merge the last 0 six times into 6, ten rewrites in all.
    /// Absorb: `9, 9, 9, 9, 9, 9, 9, 7` on a top `t` of 2 or more: the 7 is
    /// absorbed down the nines into 6, 5, ..., 0, and `t` absorbs the 0
    /// into `t` again, a chain of eight rewrites, the longest any absorb
    /// input has. Not for cancel, whose rewrites leave nothing to chain.
    Cascade,
}

impl Unit {
    /// The number of elements.
    fn len(self, v: Variant) -> usize {
        match (self, v) {
            (Unit::Pair, _) => 2,
            (Unit::Nested, Variant::Cancel) => 4,
            (Unit::Nested | Unit::Overlap, _) => 3,
            (Unit::Cascade, Variant::Merge) => 11,
            (Unit::Cascade, _) => 8,
        }
    }

    /// The positions of the rewrites, as [`fold`] gives them on a stack
    /// holding only `t`.
    fn rewrites(self, v: Variant) -> &'static [usize] {
        match (self, v) {
            (Unit::Cascade, Variant::Merge) => &[1, 2, 1, 2, 6, 5, 4, 3, 2, 1],
            (Unit::Cascade, _) => &[7, 6, 5, 4, 3, 2, 1, 0],
            (Unit::Overlap, Variant::Absorb | Variant::AbsorbCount) => &[1, 1],
            (Unit::Pair | Unit::Overlap, _) => &[1],
            (Unit::Nested, _) => &[2, 1],
        }
    }

    /// Every unit of this kind on top `t`, with elements from `values`.
    fn options(self, v: Variant, t: i64, values: &[i64]) -> Vec<Vec<i64>> {
        // A cascade is a stack holding no redex and an element absorbed
        // down it: the merge stack is built of threes, and there are too
        // many absorb inputs of 8 elements to try each.
        let all: Vec<Vec<i64>> = match (self, v) {
            (Unit::Cascade, Variant::Merge) => vec![merge_cascade(6)],
            (Unit::Cascade, _) => chain_units(v, self.len(v) - 1, values),
            _ => (0..self.len(v)).fold(vec![Vec::new()], |all, _| {
                all.into_iter()
                    .flat_map(|u| {
                        values.iter().map(move |&x| {
                            let mut u = u.clone();
                            u.push(x);
                            u
                        })
                    })
                    .collect()
            }),
        };
        all.into_iter()
            .filter(|u| {
                let (stack, at) = fold(v, &[t], u);
                at == self.rewrites(v)
                    && stack.first() == Some(&t)
                    && (self != Unit::Overlap || v.rewrite(u[1], u[2]).is_some())
            })
            .collect()
    }
}

impl Variant {
    /// The units of the second half of a generated input, in turn. A cycle
    /// is 8 elements for cancel and 16 for the others, so it fits every
    /// size of [`Collapse::sizes`] exactly. Cancel has no overlap unit: its
    /// answer does not depend on the order of the rewrites, as the
    /// cancelled pairs nest like brackets. Merge has no nested unit: its
    /// cascade leaves room for only two units of 2 and 3 elements, and its
    /// chains already start from results that make a new redex with the
    /// element before them.
    fn cycle(self) -> &'static [Unit] {
        match self {
            Variant::Cancel => &[Unit::Pair, Unit::Nested, Unit::Pair],
            Variant::Merge => &[Unit::Pair, Unit::Overlap, Unit::Cascade],
            Variant::Absorb | Variant::AbsorbCount => {
                &[Unit::Cascade, Unit::Pair, Unit::Nested, Unit::Overlap]
            }
        }
    }
}

/// The units at step `i` of [`Variant::cycle`] on each top `t`, keyed by
/// `(i, t)`: those after which step `i + 1` (cyclically) has a unit on the
/// top they leave, so the cycle never runs out of units.
fn unit_table(v: Variant) -> HashMap<(usize, i64), Vec<Vec<i64>>> {
    let (lo, hi) = v.value_range();
    let values: Vec<i64> = (lo..=hi).collect();
    let cycle = v.cycle();
    let next_top = |t: i64, u: &[i64]| *fold(v, &[t], u).0.last().expect("t is kept");
    // Every top a unit can leave, starting from the input values.
    let mut table: HashMap<(usize, i64), Vec<Vec<i64>>> = HashMap::new();
    let mut tops: Vec<i64> = values.clone();
    while let Some(t) = tops.pop() {
        for (i, k) in cycle.iter().enumerate() {
            if let std::collections::hash_map::Entry::Vacant(e) = table.entry((i, t)) {
                let options = k.options(v, t, &values);
                tops.extend(options.iter().map(|u| next_top(t, u)));
                e.insert(options);
            }
        }
    }
    // Drop units after which the next step has none, until none is dropped.
    loop {
        let has_next =
            |table: &HashMap<(usize, i64), Vec<Vec<i64>>>, i: usize, t: i64, u: &[i64]| {
                table
                    .get(&((i + 1) % cycle.len(), next_top(t, u)))
                    .is_some_and(|o| !o.is_empty())
            };
        let keys: Vec<(usize, i64)> = table.keys().copied().collect();
        let mut dropped = false;
        for (i, t) in keys {
            let kept: Vec<Vec<i64>> = table[&(i, t)]
                .iter()
                .filter(|u| has_next(&table, i, t, u))
                .cloned()
                .collect();
            if kept.len() < table[&(i, t)].len() {
                dropped = true;
                table.insert((i, t), kept);
            }
        }
        if !dropped {
            return table;
        }
    }
}

/// The elements whose merges leave the single element `x` on a stack whose
/// top is no element of them: `x` itself up to 3, and twice the elements
/// for `x - 1` above.
fn merge_build(x: i64) -> Vec<i64> {
    if x <= 3 {
        vec![x]
    } else {
        let half = merge_build(x - 1);
        [half.clone(), half].concat()
    }
}

/// The elements that build the stack `k - 1, ..., 1, 0` (top last) and a 0
/// that merges `k` times into `k`: `2^(k - 3) + 3` elements from `k = 3`
/// on.
fn merge_cascade(k: i64) -> Vec<i64> {
    (0..k).rev().flat_map(merge_build).chain([0]).collect()
}

/// The inputs of `m + 1` elements whose last element makes a chain of `m`
/// rewrites down the `m` before it, which hold no redex. Each element is
/// from `values`.
fn chain_units(v: Variant, m: usize, values: &[i64]) -> Vec<Vec<i64>> {
    // The stack from the top down, the last result and the trigger.
    fn grow(
        v: Variant,
        m: usize,
        values: &[i64],
        down: &mut Vec<i64>,
        r: i64,
        c: i64,
        out: &mut Vec<Vec<i64>>,
    ) {
        if down.len() == m {
            out.push(down.iter().rev().copied().chain([c]).collect());
            return;
        }
        for &t in values {
            let below_last = down.last().is_none_or(|&p| v.rewrite(t, p).is_none());
            if let (Some(&[r2]), true) = (v.rewrite(t, r).as_deref(), below_last) {
                down.push(t);
                grow(v, m, values, down, r2, c, out);
                down.pop();
            }
        }
    }
    let mut out = Vec::new();
    for &c in values {
        grow(v, m, values, &mut Vec::new(), c, c, &mut out);
    }
    out
}

impl TaskFamily for Collapse {
    fn name(&self) -> &str {
        match self.0 {
            Variant::Merge => "collapse/merge",
            Variant::Cancel => "collapse/cancel",
            Variant::Absorb => "collapse/absorb",
            Variant::AbsorbCount => "collapse/absorb-count",
        }
    }

    fn signature(&self) -> Scheme {
        let s = match self.0 {
            Variant::AbsorbCount => "List Int -> Int",
            _ => "List Int -> List Int",
        };
        s.parse().expect("a scheme")
    }

    fn description(&self) -> &str {
        match self.0 {
            Variant::Merge => {
                "Rewrite the list until no two adjacent elements are equal. One rewrite \
                 takes the leftmost pair of adjacent equal elements x, x and replaces \
                 the two of them with the single element x + 1, in their place. Then \
                 look again from the start of the list. Return the list once no two \
                 adjacent elements are equal. Every element of the input is between 0 \
                 and 3."
            }
            Variant::Cancel => {
                "Rewrite the list until no two adjacent elements sum to 0. One rewrite \
                 takes the leftmost pair of adjacent elements x, y with x + y = 0 and \
                 removes both of them. Then look again from the start of the list. \
                 Return the list once no two adjacent elements sum to 0. Every element \
                 of the input is between -3 and 3."
            }
            Variant::Absorb => {
                "Rewrite the list until no element is at least 2 greater than the \
                 element just after it. One rewrite takes the leftmost pair of adjacent \
                 elements x, y (x just before y) with x - y >= 2 and replaces the two of \
                 them with the single element (x + y) mod 10, in their place. Then look \
                 again from the start of the list. Return the list once no such pair is \
                 left. Every element of the input is between 0 and 9."
            }
            Variant::AbsorbCount => {
                "Rewrite the list until no element is at least 2 greater than the \
                 element just after it. One rewrite takes the leftmost pair of adjacent \
                 elements x, y (x just before y) with x - y >= 2 and replaces the two of \
                 them with the single element (x + y) mod 10, in their place. Then look \
                 again from the start of the list. Return the number of rewrites made \
                 until no such pair is left. Every element of the input is between 0 \
                 and 9."
            }
        }
    }

    fn examples(&self) -> Vec<Example> {
        let to_list = |xs: &[i64], out: &[i64]| Example::new(vec![list(xs)], list(out));
        let to_int = |xs: &[i64], out: i64| Example::new(vec![list(xs)], Value::Int(out));
        // The absorb examples, with the final list and the number of
        // rewrites.
        let absorb: [(&[i64], &[i64], i64); 13] = [
            (&[], &[], 0),
            (&[7], &[7], 0),
            (&[5, 3], &[8], 1),
            (&[5, 4], &[5, 4], 0),
            (&[9, 3], &[2], 1),
            // Leftmost first: (6, 4) gives 0, and 0, 0 is not a redex. The
            // rightmost redex (4, 0) first would end in [0].
            (&[6, 4, 0], &[0, 0], 1),
            (&[1, 9, 8, 5, 2], &[1, 2, 2], 2),
            (&[8, 8, 7, 2, 5, 0], &[0, 0], 4),
            (&[0, 1, 2, 3, 2, 1, 0], &[0, 1, 2, 3, 2, 1, 0], 0),
            // A chain of four rewrites from one element: 9, 7 gives 6, which
            // 8 absorbs, then 6, then 2.
            (&[2, 6, 8, 9, 7], &[2], 4),
            (&[3, 4, 5, 6, 7, 8, 9, 2, 0, 9, 1, 4, 3], &[3, 4, 5, 9], 9),
            // Chains of seven and of eight, the longest there is: the 6 is
            // absorbed down the nines into 5, 4, 3, 2, 1, 0, and the 4
            // absorbs the 0; the 7 into 6, ..., 0, and the 3 absorbs the 0.
            (&[4, 9, 9, 9, 9, 9, 9, 6, 8], &[4, 8], 7),
            (&[1, 3, 9, 9, 9, 9, 9, 9, 9, 7, 5], &[1, 3, 5], 8),
        ];
        match self.0 {
            Variant::Merge => vec![
                to_list(&[], &[]),
                to_list(&[2], &[2]),
                to_list(&[1, 1], &[2]),
                // Leftmost first: the rightmost pair first would give [1, 2].
                to_list(&[1, 1, 1], &[2, 1]),
                to_list(&[2, 1, 1], &[3]),
                to_list(&[0, 0, 0, 0], &[2]),
                to_list(&[1, 0, 0, 2], &[3]),
                // A chain of four merges from the last element, and results
                // above 4.
                to_list(&[3, 2, 1, 0, 0], &[4]),
                to_list(&[3, 3, 3, 3], &[5]),
                to_list(&[3, 2, 2, 1, 1, 0, 0], &[4, 2, 1]),
                to_list(&[0, 1, 0, 1, 0, 1], &[0, 1, 0, 1, 0, 1]),
                to_list(&[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], &[6, 5]),
                // Chains of six and seven merges from the last element, onto
                // 5, 4, 3, 2, 1, 0 and 6, 5, 4, 3, 2, 1, 0 built of threes,
                // and of eight and nine, in 35 and 67 elements, longer than
                // the generated inputs make.
                to_list(&[1, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0], &[1, 6]),
                to_list(
                    &[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 0, 2],
                    &[7, 2],
                ),
                to_list(&merge_cascade(8), &[8]),
                to_list(&merge_cascade(9), &[9]),
            ],
            Variant::Cancel => vec![
                to_list(&[], &[]),
                to_list(&[0], &[0]),
                to_list(&[0, 0], &[]),
                to_list(&[2, 2], &[2, 2]),
                to_list(&[2, -2], &[]),
                to_list(&[3, -2], &[3, -2]),
                to_list(&[1, 2, -2, -1], &[]),
                to_list(&[2, -3, 3, 1, -1, -2, 1], &[1]),
                to_list(&[3, 1, -2, 2, -1, 0, 0, -3], &[]),
                to_list(&[1, 0, -1, 2, 0, 0, -2, 3], &[1, 0, -1, 3]),
            ],
            Variant::Absorb => absorb.iter().map(|(xs, out, _)| to_list(xs, out)).collect(),
            Variant::AbsorbCount => absorb.iter().map(|(xs, _, k)| to_int(xs, *k)).collect(),
        }
    }

    /// An input of size `n` whose rewrites follow the same schedule for
    /// every seed, so that the references' step counts do not depend on the
    /// seed; only the values do. The first `n / 2` elements have no redex,
    /// so the stack of the rewriting holds all of them. The rest are the
    /// units of a fixed cycle in turn (for absorb, cascade, pair, nested,
    /// overlap; for merge, pair, overlap, cascade; for cancel, pair,
    /// nested, pair), each rewritten on top of the stack without reaching
    /// below it (the absorb cascade rewrites the top itself, into its own
    /// value), so every rewrite is at a depth of at least `n / 2`. Elements
    /// too few for the next unit have no redex. Each element or unit is
    /// drawn uniformly from those in [`Variant::value_range`] that keep the
    /// schedule going.
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let v = self.0;
        let (lo, hi) = v.value_range();
        let table = unit_table(v);
        let cycle = v.cycle();
        let first = |t: i64| table.get(&(0, t)).is_some_and(|o| !o.is_empty());
        let mut xs: Vec<i64> = Vec::with_capacity(n as usize);
        // The stack of the rewriting so far, top last.
        let mut stack: Vec<i64> = Vec::with_capacity(n as usize);
        let mut step = 0;
        while (xs.len() as u64) < n {
            let at = xs.len() as u64;
            let top = stack.last().copied();
            match top {
                Some(t) if at >= n / 2 && cycle[step].len(v) as u64 <= n - at => {
                    let options = &table[&(step, t)];
                    let u = &options[rng.below(options.len() as u64) as usize];
                    xs.extend(u);
                    stack.pop();
                    stack.extend(fold(v, &[t], u).0);
                    step = (step + 1) % cycle.len();
                }
                _ => {
                    // The last element before the units is a top the first
                    // unit can follow.
                    let misses: Vec<i64> = (lo..=hi)
                        .filter(|&x| top.is_none_or(|t| v.rewrite(t, x).is_none()))
                        .filter(|&x| at + 1 != n / 2 || first(x))
                        .collect();
                    let x = misses[rng.below(misses.len() as u64) as usize];
                    xs.push(x);
                    stack.push(x);
                }
            }
        }
        vec![list(&xs)]
    }
    fn oracle(&self, inputs: &[Value]) -> Value {
        let (out, rewrites) = simulate(self.0, &ints_of(&inputs[0]));
        match self.0 {
            Variant::AbsorbCount => Value::Int(rewrites),
            _ => list(&out),
        }
    }

    /// [`CASCADE_SIZES`], and for cancel the default sweep.
    fn sizes(&self) -> Vec<u64> {
        match self.0 {
            Variant::Cancel => DEFAULT_SIZES.to_vec(),
            _ => CASCADE_SIZES.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The final list and the number of rewrites by one pass with a stack,
    /// written apart from [`simulate`] and [`Variant::rewrite`].
    fn stack_fold(v: Variant, xs: &[i64]) -> (Vec<i64>, i64) {
        let mut stack: Vec<i64> = Vec::new();
        let mut rewrites = 0;
        for &x in xs {
            let mut cur = Some(x);
            while let Some(c) = cur {
                match (v, stack.last().copied()) {
                    (Variant::Merge, Some(t)) if t == c => {
                        stack.pop();
                        cur = Some(c + 1);
                    }
                    (Variant::Cancel, Some(t)) if t == -c => {
                        stack.pop();
                        cur = None;
                    }
                    (Variant::Absorb | Variant::AbsorbCount, Some(t)) if t >= c + 2 => {
                        stack.pop();
                        cur = Some((t + c) % 10);
                    }
                    _ => {
                        stack.push(c);
                        cur = None;
                        continue;
                    }
                }
                rewrites += 1;
            }
        }
        (stack, rewrites)
    }

    #[test]
    fn examples_are_in_the_value_range() {
        for v in Variant::ALL {
            let (lo, hi) = v.value_range();
            for e in Collapse(v).examples() {
                let xs = ints_of(&e.inputs[0]);
                assert!(xs.iter().all(|&x| lo <= x && x <= hi), "{v:?}: {xs:?}");
            }
        }
    }

    #[test]
    fn examples_agree_with_the_stack_fold() {
        for v in Variant::ALL {
            for e in Collapse(v).examples() {
                let (out, k) = stack_fold(v, &ints_of(&e.inputs[0]));
                let expected = match v {
                    Variant::AbsorbCount => Value::Int(k),
                    _ => list(&out),
                };
                assert_eq!(e.output, expected, "{v:?} on {:?}", e.inputs[0]);
            }
        }
    }

    /// The leftmost redex matters: rewriting the rightmost one first gives
    /// another answer on these inputs, so an example catches a solution that
    /// rewrites in the wrong place.
    #[test]
    fn the_leftmost_redex_matters() {
        assert_eq!(simulate(Variant::Merge, &[1, 1, 1]).0, vec![2, 1]);
        assert_eq!(simulate(Variant::Absorb, &[6, 4, 0]).0, vec![0, 0]);
    }

    /// Every seed's input makes the same rewrites at the same depths, so
    /// the references' step counts do not depend on the seed: at every size
    /// of the variant's sweep, and at an odd size, each variant's inputs
    /// for seeds `0..5` make the same number of rewrites, at least `n / 5`,
    /// and the positions of the redexes the oracle's scan finds sum to the
    /// same total, at least `n² / 20` from 64 on. The inputs still differ.
    #[test]
    fn generated_inputs_keep_one_schedule() {
        for v in Variant::ALL {
            let f = Collapse(v);
            for n in f.sizes().into_iter().chain([101]) {
                let mut schedules = Vec::new();
                let mut inputs = Vec::new();
                for seed in 0..5 {
                    let xs = ints_of(&crate::task::generate_input(&f, n, seed)[0]);
                    let (lo, hi) = v.value_range();
                    assert!(xs.iter().all(|&x| lo <= x && x <= hi), "{v:?}: {xs:?}");
                    let (mut ys, mut k, mut depth) = (xs.clone(), 0_u64, 0_u64);
                    while let Some(i) = (0..ys.len().saturating_sub(1))
                        .find(|&i| v.rewrite(ys[i], ys[i + 1]).is_some())
                    {
                        let r = v.rewrite(ys[i], ys[i + 1]).expect("a redex");
                        ys.splice(i..i + 2, r);
                        k += 1;
                        depth += i as u64;
                    }
                    schedules.push((k, depth));
                    inputs.push(xs);
                }
                let (k, depth) = schedules[0];
                assert!(
                    schedules.iter().all(|&s| s == (k, depth)),
                    "{v:?} at {n}: {schedules:?}"
                );
                assert!(k >= n / 5, "{v:?} at {n}: {k} rewrites");
                assert!(n < 64 || depth >= n * n / 20, "{v:?} at {n}: depth {depth}");
                assert!(
                    inputs[1..].iter().any(|xs| *xs != inputs[0]),
                    "{v:?} at {n}"
                );
            }
        }
    }

    /// No absorb chain is longer than 8, and the cascade units end in a
    /// chain of 6 merges and of 8 absorbs, and are in the unit table.
    #[test]
    fn cascades_end_in_long_chains() {
        let values: Vec<i64> = (0..=9).collect();
        assert!(!chain_units(Variant::Absorb, 8, &values).is_empty());
        assert!(chain_units(Variant::Absorb, 9, &values).is_empty());
        for (v, chain) in [
            (Variant::Merge, 6),
            (Variant::Absorb, 8),
            (Variant::AbsorbCount, 8),
        ] {
            // The last rewrites, each one below the one before, are one
            // chain.
            let at = Unit::Cascade.rewrites(v);
            let last = at.windows(2).rev().take_while(|w| w[0] == w[1] + 1).count() + 1;
            assert_eq!(last, chain, "{v:?}: {at:?}");
            let i = v.cycle().iter().position(|&u| u == Unit::Cascade);
            let table = unit_table(v);
            let tops: Vec<i64> = table
                .iter()
                .filter(|(&(j, _), o)| Some(j) == i && !o.is_empty())
                .map(|(&(_, t), _)| t)
                .collect();
            assert!(!tops.is_empty(), "{v:?}: no cascade on any top");
        }
    }

    proptest::proptest! {
        /// The literal simulation and the stack fold agree: the rewriting at
        /// the leftmost redex is what one pass with a stack computes.
        #[test]
        fn simulation_agrees_with_the_stack_fold(
            raw in proptest::collection::vec(0_i64..10, 0..40),
        ) {
            for v in Variant::ALL {
                let (lo, hi) = v.value_range();
                let xs: Vec<i64> = raw.iter().map(|&x| lo + x % (hi - lo + 1)).collect();
                proptest::prop_assert_eq!(
                    simulate(v, &xs),
                    stack_fold(v, &xs),
                    "{:?} on {:?}", v, xs
                );
            }
        }
    }
}
