//! The test, the interval, the call, and the spread between runs of
//! decision D33.
//!
//! The unit of replication is a pair of runs, one in each arm. Each pair
//! gives one difference: the arm under test minus the control arm.
//!
//! * [`sign_flip_p`] is the exact paired sign-flip permutation test,
//!   two-sided: it enumerates all `2^S` sign patterns of the `S`
//!   differences.
//! * [`interval`] is the confidence interval for the mean difference by
//!   test inversion: every shift `δ` for which the test of the differences
//!   minus `δ` gives `p > α`.
//! * [`call`] is the call D33 fixes before any comparison runs.
//! * [`logit_spread`] estimates the spread between runs on the logit scale,
//!   for sizing the next comparison.

use serde::Serialize;

/// The most pairs the exact test takes. The interval checks about `2^S`
/// shifts against `2^S` patterns each, which takes seconds at 16.
pub const MAX_PAIRS: usize = 16;

/// Why a test cannot be run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestError {
    /// No differences were given.
    Empty,
    /// More than [`MAX_PAIRS`] differences were given.
    TooMany(usize),
    /// A difference is not a finite number.
    NotFinite,
}

impl std::fmt::Display for TestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestError::Empty => write!(f, "the test needs at least one pair"),
            TestError::TooMany(s) => {
                write!(f, "the exact test takes at most {MAX_PAIRS} pairs, not {s}")
            }
            TestError::NotFinite => write!(f, "a pair's difference is not a finite number"),
        }
    }
}

impl std::error::Error for TestError {}

/// The sums each sign pattern gives: for pattern `m`, `Σ s_i d_i` and
/// `Σ s_i`, where `s_i` is `-1` if bit `i` of `m` is set and `+1` if not.
/// Pattern 0 is the observed one.
struct Patterns {
    /// `Σ s_i d_i` for each pattern.
    sums: Vec<f64>,
    /// `Σ s_i` for each pattern.
    signs: Vec<f64>,
    /// `Σ |d_i|`, the scale for the tolerance on ties.
    scale: f64,
}

impl Patterns {
    fn of(d: &[f64]) -> Result<Patterns, TestError> {
        if d.is_empty() {
            return Err(TestError::Empty);
        }
        if d.len() > MAX_PAIRS {
            return Err(TestError::TooMany(d.len()));
        }
        if d.iter().any(|x| !x.is_finite()) {
            return Err(TestError::NotFinite);
        }
        let n = 1usize << d.len();
        let mut sums = Vec::with_capacity(n);
        let mut signs = Vec::with_capacity(n);
        for m in 0..n {
            let mut sum = 0.0;
            let mut sign = 0.0;
            for (i, x) in d.iter().enumerate() {
                if m >> i & 1 == 1 {
                    sum -= x;
                    sign -= 1.0;
                } else {
                    sum += x;
                    sign += 1.0;
                }
            }
            sums.push(sum);
            signs.push(sign);
        }
        let scale = d.iter().map(|x| x.abs()).sum();
        Ok(Patterns { sums, signs, scale })
    }

    /// The two-sided p of the differences minus `delta`: the share of sign
    /// patterns whose `|Σ s_i (d_i - δ)|` is at least the observed one.
    /// Ties count, within a tolerance for rounding, so `p` is never below
    /// its exact value.
    fn p(&self, delta: f64) -> f64 {
        let n = self.sums.len();
        let observed = (self.sums[0] - delta * self.signs[0]).abs();
        let tolerance = 1e-9 * (self.scale + self.signs[0] * delta.abs());
        let at_least = self
            .sums
            .iter()
            .zip(&self.signs)
            .filter(|(s, k)| (*s - delta * *k).abs() >= observed - tolerance)
            .count();
        // Both counts are at most 2^16, so they are exact in f64.
        at_least as f64 / n as f64
    }
}

/// The two-sided p of the exact paired sign-flip test of `d`: the share of
/// the `2^S` sign patterns whose `|Σ s_i d_i|` is at least `|Σ d_i|`.
///
/// # Errors
///
/// If `d` is empty, has more than [`MAX_PAIRS`] values, or has one that is
/// not finite.
pub fn sign_flip_p(d: &[f64]) -> Result<f64, TestError> {
    Ok(Patterns::of(d)?.p(0.0))
}

/// The confidence interval for the mean difference, by inverting the
/// sign-flip test.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Interval {
    /// The lower end, or `None` if the interval has none: with 5 pairs or
    /// fewer, no shift gives `p ≤ 0.05`.
    pub lower: Option<f64>,
    /// The upper end, or `None` if the interval has none.
    pub upper: Option<f64>,
    /// Whether some shift between the ends gives `p ≤ α`, so the set of
    /// shifts is not one interval and the ends are its hull.
    pub has_gaps: bool,
}

impl Interval {
    /// Whether the interval lies strictly between `-m` and `+m`.
    pub fn inside(&self, m: f64) -> bool {
        matches!((self.lower, self.upper), (Some(l), Some(u)) if l > -m && u < m)
    }
}

/// The interval for the mean of `d` at level `alpha`: every shift `δ` for
/// which the sign-flip test of `d_i - δ` gives `p > alpha`.
///
/// `p(δ)` can change only where `|Σ s_i (d_i - δ)| = |Σ (d_i - δ)|` for some
/// pattern `s`. That holds where `δ` is the mean of the `d_i` that `s` flips,
/// or of those it keeps. So the breaks are the means of the non-empty
/// subsets of `d`. Between two breaks `p` is constant, and at a break it is
/// at least its value on either side, so the ends are breaks. The function
/// checks `p` at every break, between each two, and beyond both ends.
///
/// # Errors
///
/// As for [`sign_flip_p`].
pub fn interval(d: &[f64], alpha: f64) -> Result<Interval, TestError> {
    let patterns = Patterns::of(d)?;
    let n = 1usize << d.len();
    let mut breaks: Vec<f64> = Vec::with_capacity(n - 1);
    for m in 1..n {
        let mut sum = 0.0;
        let mut count = 0.0;
        for (i, x) in d.iter().enumerate() {
            if m >> i & 1 == 1 {
                sum += x;
                count += 1.0;
            }
        }
        breaks.push(sum / count);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + b.abs()));
    // Each candidate shift: beyond the lowest break, then each break and
    // the midpoint after it, the last midpoint replaced by a point beyond
    // the highest break.
    let first = breaks[0];
    let last = breaks[breaks.len() - 1];
    let reach = 1.0 + (last - first).abs();
    let below = first - reach;
    let above = last + reach;
    let mut candidates: Vec<(f64, Option<f64>)> = Vec::with_capacity(2 * breaks.len() + 1);
    candidates.push((below, None));
    for (i, b) in breaks.iter().enumerate() {
        candidates.push((*b, Some(*b)));
        let next = breaks.get(i + 1).map_or(above, |n| (b + n) / 2.0);
        candidates.push((next, None));
    }
    let inside: Vec<bool> = candidates
        .iter()
        .map(|(delta, _)| patterns.p(*delta) > alpha)
        .collect();
    let Some(lo) = inside.iter().position(|x| *x) else {
        // No shift gives p > alpha. Not possible: p at the mean of all the
        // differences is 1. Kept as an empty interval for safety.
        return Ok(Interval {
            lower: Some(f64::NAN),
            upper: Some(f64::NAN),
            has_gaps: false,
        });
    };
    let hi = inside.iter().rposition(|x| *x).unwrap_or(lo);
    let has_gaps = inside[lo..=hi].iter().any(|x| !x);
    // An end that is a candidate beyond every break is open.
    let lower = candidates[lo].1;
    let upper = candidates[hi].1;
    Ok(Interval {
        lower: if lo == 0 { None } else { lower },
        upper: if hi == candidates.len() - 1 {
            None
        } else {
            upper
        },
        has_gaps,
    })
}

/// The call D33 fixes before any comparison runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Call {
    /// `p ≤ α`, and the mean difference favours the arm under test.
    Positive,
    /// `p ≤ α`, and the mean difference favours the control arm.
    Negative,
    /// `p > α`, and the whole interval lies strictly between `-m` and `+m`,
    /// the smallest effect that matters.
    Null,
    /// Any other result.
    Inconclusive,
}

/// The call for a mean difference `mean` with test result `p` and interval
/// `ci`, at level `alpha`, with `smallest` the smallest effect that matters.
pub fn call(mean: f64, p: f64, ci: &Interval, alpha: f64, smallest: f64) -> Call {
    if p <= alpha && mean > 0.0 {
        Call::Positive
    } else if p <= alpha && mean < 0.0 {
        Call::Negative
    } else if p > alpha && ci.inside(smallest) {
        Call::Null
    } else {
        Call::Inconclusive
    }
}

/// Programs passed and batches issued in one cell: one task of one run,
/// over the counted iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Programs that passed.
    pub passed: u64,
    /// Programs the batches issued asked for.
    pub issued: u64,
}

/// The spread between runs on the logit scale.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Spread {
    /// The variance between runs of a cell's logit, beyond what sampling
    /// gives, pooled over the groups; never below 0.
    pub between_run_variance: f64,
    /// The spread `σ` of D33's simulation model, `sqrt(variance / 2)`: that
    /// model gives each run and each task within a run a shift with spread
    /// `σ`, so a cell's logit varies between runs by `2σ²`.
    pub sigma: f64,
    /// The groups that went into the estimate.
    pub groups: usize,
    /// The cells that went into the estimate.
    pub cells: usize,
}

/// The method-of-moments estimate of the spread between runs. Each group is
/// one task in one arm, and holds that task's cell in each run of the arm.
/// For each group with two or more cells that issued programs, it takes the
/// variance over runs of the empirical logit `ln((x + ½) / (n - x + ½))`,
/// less the mean of its sampling variance `1/(x + ½) + 1/(n - x + ½)`. The
/// groups are pooled by their degrees of freedom. Returns `None` if no group
/// has two such cells.
pub fn logit_spread(groups: &[Vec<Cell>]) -> Option<Spread> {
    let mut weighted = 0.0;
    let mut freedom = 0.0;
    let mut used_groups = 0;
    let mut used_cells = 0;
    for group in groups {
        let cells: Vec<&Cell> = group.iter().filter(|c| c.issued > 0).collect();
        if cells.len() < 2 {
            continue;
        }
        let k = cells.len() as f64;
        let logits: Vec<f64> = cells
            .iter()
            .map(|c| {
                let x = c.passed as f64 + 0.5;
                let y = (c.issued - c.passed.min(c.issued)) as f64 + 0.5;
                (x / y).ln()
            })
            .collect();
        let sampling: f64 = cells
            .iter()
            .map(|c| {
                let x = c.passed as f64 + 0.5;
                let y = (c.issued - c.passed.min(c.issued)) as f64 + 0.5;
                1.0 / x + 1.0 / y
            })
            .sum::<f64>()
            / k;
        let mean = logits.iter().sum::<f64>() / k;
        let variance = logits.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / (k - 1.0);
        weighted += (variance - sampling) * (k - 1.0);
        freedom += k - 1.0;
        used_groups += 1;
        used_cells += cells.len();
    }
    if used_groups == 0 {
        return None;
    }
    let between = (weighted / freedom).max(0.0);
    Some(Spread {
        between_run_variance: between,
        sigma: (between / 2.0).sqrt(),
        groups: used_groups,
        cells: used_cells,
    })
}

/// The mean of `d`, or 0 if it is empty.
pub fn mean(d: &[f64]) -> f64 {
    if d.is_empty() {
        0.0
    } else {
        d.iter().sum::<f64>() / d.len() as f64
    }
}

/// The sample standard deviation of `d`, or `None` with fewer than two.
pub fn sd(d: &[f64]) -> Option<f64> {
    if d.len() < 2 {
        return None;
    }
    let m = mean(d);
    Some((d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64).sqrt())
}

#[cfg(test)]
#[path = "stats_tests.rs"]
mod tests;
