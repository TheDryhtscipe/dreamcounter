//! The cost-aware objective `J(L)` of decision D4, and the cost value of
//! each frontier member (decision D27, bead dreamcounter-183fbb).
//!
//! ```text
//! J(L) = DL(L) + Σ_t min_{p ∈ F_t} [DL(p | L) + λ · C(p)]
//! ```
//!
//! [`objective`] takes `DL(L)`, each task's members with their description
//! length `DL(p | L)` and their [`Growth`], and λ. Each task contributes its
//! [`Selection`]: the member with the smallest score, the earliest on a tie.
//!
//! # `C(p)`
//!
//! * A member whose class estimator C named gets [`class_cost`]: the class's
//!   position in [`GrowthClass::ALL`], from 0 for `1` to 6 for `2ⁿ`.
//! * A member that estimator C refused ([`Growth::Refused`]), or that was not
//!   measured ([`Growth::NotMeasured`]), gets the largest fitted `C` in its
//!   task plus 1, so it ranks after every fitted member of its task. If no
//!   member of the task is fitted, it gets [`NO_FITTED_REFUSAL_COST`], one
//!   past `2ⁿ`, so a task never looks cheaper because its cost could not be
//!   measured.
//! * A member with a size that timed out inside estimator C's fit window
//!   ([`Growth::TimedOut`]) gets [`TIMEOUT_COST`]. That is the smallest whole
//!   number above every fitted value (at most 6) and every refusal value (at
//!   most 7), so at equal description length a timed-out member loses to
//!   every other member at every λ > 0. A larger value would change only how
//!   a timeout trades against description length, which is a weighting
//!   choice that D4 leaves to P5's λ sweep.
//! * A member that gave a wrong output ([`Growth::Incorrect`]) is not a
//!   solution and is left out. A task whose members are all incorrect, or
//!   that has no members, contributes no term.
//!
//! Each [`MemberCost`] says why a member has no fitted class
//! ([`MemberCost::unfitted`]), so the cause can be logged with it.
//!
//! The scale is a placeholder, as spec §9's is: P5 compares values of λ on
//! the same corpus before any weighting is chosen.

use std::collections::BTreeMap;
use std::fmt;

use dc_tasks::growth::GrowthClass;
use serde::{Deserialize, Serialize};

/// `C(p)` of a member with a size that timed out inside the fit window. See
/// the module docs for why it is 8.
pub const TIMEOUT_COST: u32 = 8;

/// `C(p)` of a refused or unmeasured member of a task that has no fitted
/// member: one past `2ⁿ`.
pub const NO_FITTED_REFUSAL_COST: u32 = 7;

/// The largest λ accepted: `2⁶⁴`. Above it one step of `C` outweighs any
/// difference in description length, which is a `u64`, so a larger λ
/// selects the same members. The bound keeps every score and `J(L)` finite.
pub const MAX_LAMBDA: f64 = 18_446_744_073_709_551_616.0;

/// Why estimator C named no class for a sweep with no timeout in its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// The medians are too noisy to name a class.
    TooNoisy,
    /// No class fits.
    NoFit,
    /// Some size has one seed, so the noise is unknown.
    NoiseUnknown,
    /// Estimator C could not fit the sweep for another reason, such as a
    /// median of zero steps.
    NotFitted,
}

/// What is known about a frontier member's cost growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Growth {
    /// Estimator C named this class.
    Fitted(GrowthClass),
    /// A size inside estimator C's fit window timed out.
    TimedOut,
    /// Estimator C named no class.
    Refused(Refusal),
    /// The member was not measured: it could not be run on the sweep, or
    /// the time for measuring ran out, or its run predates measuring.
    NotMeasured,
    /// The member gave a wrong output or a runtime error on a generated
    /// input. It is not a solution.
    Incorrect,
}

/// Why a member that has a `C(p)` has no fitted class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unfitted {
    /// A size inside the fit window timed out.
    TimedOut,
    /// Estimator C named no class.
    Refused(Refusal),
    /// The member was not measured.
    NotMeasured,
}

impl fmt::Display for Unfitted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Unfitted::TimedOut => "timed out inside the fit window",
            Unfitted::Refused(Refusal::TooNoisy) => "refused by estimator C: too noisy",
            Unfitted::Refused(Refusal::NoFit) => "refused by estimator C: no fit",
            Unfitted::Refused(Refusal::NoiseUnknown) => "refused by estimator C: noise unknown",
            Unfitted::Refused(Refusal::NotFitted) => "refused by estimator C: not fitted",
            Unfitted::NotMeasured => "not measured",
        })
    }
}

/// One member of a task's frontier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    /// `DL(p | L)`, such as [`crate::stitch_cost`] under the library.
    pub dl: u64,
    /// What is known about its cost growth.
    pub growth: Growth,
}

/// A member's `C(p)`, and why it has no fitted class if it has none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemberCost {
    /// `C(p)`.
    pub c: u32,
    /// `None` for a fitted member.
    pub unfitted: Option<Unfitted>,
}

/// `C(p)` of a member whose class is `class`: its position in
/// [`GrowthClass::ALL`].
pub fn class_cost(class: GrowthClass) -> u32 {
    let position = GrowthClass::ALL
        .iter()
        .position(|&c| c == class)
        .unwrap_or(GrowthClass::ALL.len() - 1);
    // ALL has seven classes.
    u32::try_from(position).unwrap_or(u32::MAX)
}

/// Each member's `C(p)`, in order; `None` for an incorrect member. See the
/// module docs.
pub fn member_costs(members: &[Member]) -> Vec<Option<MemberCost>> {
    let refusal = members
        .iter()
        .filter_map(|m| match m.growth {
            Growth::Fitted(class) => Some(class_cost(class) + 1),
            _ => None,
        })
        .max()
        .unwrap_or(NO_FITTED_REFUSAL_COST);
    members
        .iter()
        .map(|m| {
            let (c, unfitted) = match m.growth {
                Growth::Incorrect => return None,
                Growth::Fitted(class) => (class_cost(class), None),
                Growth::TimedOut => (TIMEOUT_COST, Some(Unfitted::TimedOut)),
                Growth::Refused(r) => (refusal, Some(Unfitted::Refused(r))),
                Growth::NotMeasured => (refusal, Some(Unfitted::NotMeasured)),
            };
            Some(MemberCost { c, unfitted })
        })
        .collect()
}

/// The member a task contributes to `J(L)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Selection {
    /// Its position in the task's members.
    pub index: usize,
    /// Its `DL(p | L)`.
    pub dl: u64,
    /// Its `C(p)`, and the cause if it has no fitted class.
    pub cost: MemberCost,
    /// `DL(p | L) + λ · C(p)`, rounded to `f64`. The selection is made by
    /// comparing gaps ([`select`]), so in rare near-ties a member it beat can
    /// have a score one rounding smaller.
    pub score: f64,
}

/// Why `J(L)` cannot be computed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObjectiveError {
    /// λ must be finite, at least 0, and at most [`MAX_LAMBDA`].
    BadLambda(f64),
}

impl fmt::Display for ObjectiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjectiveError::BadLambda(l) => {
                write!(
                    f,
                    "lambda must be finite, at least 0 and at most {MAX_LAMBDA}, not {l}"
                )
            }
        }
    }
}

impl std::error::Error for ObjectiveError {}

fn check_lambda(lambda: f64) -> Result<(), ObjectiveError> {
    if lambda.is_finite() && (0.0..=MAX_LAMBDA).contains(&lambda) {
        Ok(())
    } else {
        Err(ObjectiveError::BadLambda(lambda))
    }
}

/// A valid λ: finite, at least 0, and at most [`MAX_LAMBDA`]. `-0` is kept
/// as `0`. Written as a number; a number that is not a valid λ is not read.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Lambda(f64);

// A Lambda is never NaN, so equality is an equivalence.
impl Eq for Lambda {}

impl Lambda {
    /// λ = 0: selection by description length alone.
    pub const ZERO: Lambda = Lambda(0.0);

    /// `value` as a λ.
    ///
    /// # Errors
    ///
    /// [`ObjectiveError::BadLambda`] if `value` is not finite, is negative,
    /// or is above [`MAX_LAMBDA`].
    pub fn new(value: f64) -> Result<Lambda, ObjectiveError> {
        check_lambda(value)?;
        // -0 + 0 is +0.
        Ok(Lambda(value + 0.0))
    }

    /// The value.
    pub fn get(self) -> f64 {
        self.0
    }

    /// Whether λ is 0.
    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
}

impl fmt::Display for Lambda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Serialize for Lambda {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Lambda {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Lambda, D::Error> {
        Lambda::new(f64::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

/// The member of `members` with the smallest `DL(p | L) + λ · C(p)`, the
/// earliest on a tie. `None` if every member is incorrect or there are none.
///
/// Two members are compared by `DL(a) − DL(b)` against `λ · (C(b) − C(a))`,
/// not by their rounded scores. So at equal description length the smaller
/// `C` wins at every λ > 0, however small: `λ · ΔC` is never 0 when λ > 0
/// and `ΔC ≥ 1`. Scores alone could round to the same value.
///
/// # Errors
///
/// [`ObjectiveError::BadLambda`] if λ is not finite, is negative, or is
/// above [`MAX_LAMBDA`].
pub fn select(members: &[Member], lambda: f64) -> Result<Option<Selection>, ObjectiveError> {
    check_lambda(lambda)?;
    let costs = member_costs(members);
    Ok(
        best(members, &costs, 0..members.len(), lambda).map(|index| {
            let member = members[index];
            let cost = costs[index].expect("best takes only correct members");
            Selection {
                index,
                dl: member.dl,
                cost,
                score: member.dl as f64 + lambda * f64::from(cost.c),
            }
        }),
    )
}

/// Of the members at `among`, in the order given, the correct one with the
/// smallest `DL(p | L) + λ · C(p)`, the earliest on a tie, compared as
/// [`select`] says. `costs` are [`member_costs`] of `members`.
fn best(
    members: &[Member],
    costs: &[Option<MemberCost>],
    among: impl IntoIterator<Item = usize>,
    lambda: f64,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for index in among {
        let Some(cost) = costs[index] else { continue };
        let better = |b: usize| {
            let b_cost = costs[b].expect("best holds only correct members");
            let dl_gap = (i128::from(members[index].dl) - i128::from(members[b].dl)) as f64;
            let cost_gap = lambda * (f64::from(b_cost.c) - f64::from(cost.c));
            dl_gap < cost_gap
        };
        if best.is_none_or(better) {
            best = Some(index);
        }
    }
    best
}

/// The positions of `members` in order of `DL(p | L) + λ · C(p)`: the
/// member [`select`] gives, then the one it gives from the rest, and so on.
/// Each member's `C(p)` is [`member_costs`] of all of `members`, so taking
/// members out does not change the others' values. Equal scores keep the
/// given order. Incorrect members come last, in the given order. At λ = 0
/// this is a stable sort of the correct members by description length.
///
/// # Errors
///
/// [`ObjectiveError::BadLambda`] if λ is not finite, is negative, or is
/// above [`MAX_LAMBDA`].
pub fn rank(members: &[Member], lambda: f64) -> Result<Vec<usize>, ObjectiveError> {
    check_lambda(lambda)?;
    let costs = member_costs(members);
    let mut left: Vec<usize> = (0..members.len()).collect();
    let mut order = Vec::with_capacity(members.len());
    while let Some(index) = best(members, &costs, left.iter().copied(), lambda) {
        order.push(index);
        left.retain(|&i| i != index);
    }
    // What is left is the incorrect members, in order.
    order.extend(left);
    Ok(order)
}

/// `J(L)` and the member each task contributes.
#[derive(Clone, Debug, PartialEq)]
pub struct Objective {
    /// `DL(L)`.
    pub library_dl: u64,
    /// λ.
    pub lambda: f64,
    /// Each task that contributes a term, with its selected member.
    pub tasks: BTreeMap<String, Selection>,
    /// `J(L)`.
    pub total: f64,
}

impl Objective {
    /// `Σ_t C(selected_t)`, the sum D4 logs as component (b).
    pub fn total_cost(&self) -> u64 {
        self.tasks.values().map(|s| u64::from(s.cost.c)).sum()
    }
}

/// `J(L)` for a library of description length `library_dl`, with each
/// task's frontier members in `frontiers`.
///
/// # Errors
///
/// [`ObjectiveError::BadLambda`] if λ is not finite, is negative, or is
/// above [`MAX_LAMBDA`].
pub fn objective(
    library_dl: u64,
    frontiers: &BTreeMap<String, Vec<Member>>,
    lambda: f64,
) -> Result<Objective, ObjectiveError> {
    check_lambda(lambda)?;
    let mut tasks = BTreeMap::new();
    let mut total = library_dl as f64;
    for (name, members) in frontiers {
        if let Some(selection) = select(members, lambda)? {
            total += selection.score;
            tasks.insert(name.clone(), selection);
        }
    }
    Ok(Objective {
        library_dl,
        lambda,
        tasks,
        total,
    })
}

#[cfg(test)]
#[path = "objective_tests.rs"]
mod tests;
