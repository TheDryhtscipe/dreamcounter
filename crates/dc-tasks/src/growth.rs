//! Cost growth estimation (spec §9, decision D5).
//!
//! A program is run at a sweep of input sizes `n` (by default
//! `16, 32, ..., 1024`), several seeded inputs per size. Step counts are exact
//! integers. This module turns such a sweep into an estimate of how cost grows
//! with `n`, so that programs can be compared on computational cost.
//!
//! Two estimators sit behind the [`GrowthEstimator`] trait:
//!
//! * [`LogLogSlope`] (estimator A, the default): ordinary least squares of
//!   `ln steps` on `ln n`, reporting the slope `α̂` and R².
//! * [`PowerLogGrid`] (estimator B): `steps ≈ c · n^α · (ln n)^γ` with
//!   `γ ∈ {0, 1, 2}` and `α` continuous, choosing the `γ` with the smallest
//!   residual sum of squares.
//!
//! Both fit over the same [`Window`] of the sweep, by default its upper half.
//!
//! # Conventions
//!
//! * **Median over seeds.** Each size's step count is the median of its seed
//!   counts. For an even number of seeds the *lower* median is used (the
//!   `(k/2)`-th smallest of `k` values, counting from 1), so the result is
//!   always an exact step count that was actually observed. See
//!   [`median_steps`].
//! * **Upper half.** For a sweep of `k` sizes sorted by `n`, the upper half is
//!   the last `⌈k/2⌉` sizes, so for odd `k` the middle size is included. The
//!   default 7-size sweep therefore fits over `n ∈ {128, 256, 512, 1024}`, the
//!   window the spec's `n log n` remark refers to. The window is chosen from
//!   all sizes in the sweep, completed or not, so two programs run on the same
//!   sweep are always compared over the same sizes.
//! * **Timeouts and errors.** A size that timed out or failed is recorded in
//!   the [`Sweep`] (spec §9: a timeout is recorded, not a failure). If such a
//!   size falls inside the fit window the estimators refuse with
//!   [`GrowthError::IncompleteWindow`] rather than fit over the remaining
//!   sizes, because dropping the largest, most expensive sizes would bias the
//!   slope downwards exactly for the programs that grow fastest. Incomplete
//!   sizes below the window do not affect the fit and are reported in
//!   [`FitWindow::incomplete_below`].
//! * **No NaN.** Every float in a returned fit is finite. An R² that is
//!   undefined because the step counts do not vary is reported as
//!   [`FitQuality::ZeroVariance`], not as NaN.

use std::fmt;

/// The default sweep of sizes, `16, 32, ..., 1024` (spec §9).
pub const DEFAULT_SIZES: [u64; 7] = [16, 32, 64, 128, 256, 512, 1024];

/// The outcome of running a program at one size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SizeOutcome {
    /// Every seeded input finished; one exact step count per seed.
    Completed {
        /// Step counts, one per seed, in any order.
        steps: Vec<u64>,
    },
    /// The run exhausted its fuel at this size.
    TimedOut,
    /// The run failed at this size for a reason other than fuel.
    Errored,
}

/// The observations for one size of a sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeObservation {
    /// The input size.
    pub n: u64,
    /// What happened at that size.
    pub outcome: SizeOutcome,
}

impl SizeObservation {
    /// A size at which every seed completed with the given step counts.
    pub fn completed(n: u64, steps: Vec<u64>) -> Self {
        Self {
            n,
            outcome: SizeOutcome::Completed { steps },
        }
    }

    /// A size at which the run timed out.
    pub fn timed_out(n: u64) -> Self {
        Self {
            n,
            outcome: SizeOutcome::TimedOut,
        }
    }

    /// A size at which the run failed.
    pub fn errored(n: u64) -> Self {
        Self {
            n,
            outcome: SizeOutcome::Errored,
        }
    }
}

/// A validated sweep: sizes strictly increasing, all positive, and every
/// completed size has at least one seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sweep {
    observations: Vec<SizeObservation>,
}

impl Sweep {
    /// Validates and sorts the observations by `n`.
    ///
    /// # Errors
    ///
    /// [`GrowthError::ZeroSize`] if some `n` is 0, [`GrowthError::DuplicateSize`]
    /// if two observations share an `n`, and [`GrowthError::NoSeeds`] if a
    /// completed size has an empty list of step counts.
    pub fn new(mut observations: Vec<SizeObservation>) -> Result<Self, GrowthError> {
        observations.sort_by_key(|o| o.n);
        for (i, o) in observations.iter().enumerate() {
            if o.n == 0 {
                return Err(GrowthError::ZeroSize);
            }
            if i > 0 && observations[i - 1].n == o.n {
                return Err(GrowthError::DuplicateSize { n: o.n });
            }
            if let SizeOutcome::Completed { steps } = &o.outcome {
                if steps.is_empty() {
                    return Err(GrowthError::NoSeeds { n: o.n });
                }
            }
        }
        Ok(Self { observations })
    }

    /// Builds a sweep with one seed per size from a step function, as used for
    /// synthetic series.
    ///
    /// # Errors
    ///
    /// As for [`Sweep::new`].
    pub fn from_fn(sizes: &[u64], steps: impl Fn(u64) -> u64) -> Result<Self, GrowthError> {
        Self::new(
            sizes
                .iter()
                .map(|&n| SizeObservation::completed(n, vec![steps(n)]))
                .collect(),
        )
    }

    /// The observations, sorted by strictly increasing `n`.
    pub fn observations(&self) -> &[SizeObservation] {
        &self.observations
    }

    /// The median step count at each completed size, as `(n, median)`, in
    /// increasing `n`. Timed-out and failed sizes are omitted.
    pub fn medians(&self) -> Vec<(u64, u64)> {
        self.observations
            .iter()
            .filter_map(|o| match &o.outcome {
                SizeOutcome::Completed { steps } => median_steps(steps).map(|m| (o.n, m)),
                SizeOutcome::TimedOut | SizeOutcome::Errored => None,
            })
            .collect()
    }
}

/// The median of the seed step counts: the lower median for an even count
/// (the `(k/2)`-th smallest of `k` values, counting from 1), so the result is
/// an observed value. `None` for an empty slice.
pub fn median_steps(steps: &[u64]) -> Option<u64> {
    if steps.is_empty() {
        return None;
    }
    let mut sorted = steps.to_vec();
    sorted.sort_unstable();
    Some(sorted[(sorted.len() - 1) / 2])
}

/// Which sizes of the sweep a fit uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Window {
    /// The last `⌈k/2⌉` of the `k` sizes (the default, spec §9).
    #[default]
    UpperHalf,
    /// Every size.
    All,
}

impl Window {
    /// The index of the first size in the window, for a sweep of `k` sizes.
    fn start(self, k: usize) -> usize {
        match self {
            // k - ceil(k/2) == floor(k/2)
            Window::UpperHalf => k / 2,
            Window::All => 0,
        }
    }
}

/// Why a growth fit could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrowthError {
    /// An observation had `n == 0`.
    ZeroSize,
    /// Two observations had the same `n`.
    DuplicateSize {
        /// The repeated size.
        n: u64,
    },
    /// A completed size had no seed step counts.
    NoSeeds {
        /// The size concerned.
        n: u64,
    },
    /// The fit window has fewer than two sizes.
    TooFewPoints {
        /// The number of sizes in the window.
        points: usize,
    },
    /// Sizes inside the fit window timed out or failed; the fit refuses rather
    /// than drop them.
    IncompleteWindow {
        /// Sizes in the window that timed out.
        timed_out: Vec<u64>,
        /// Sizes in the window that failed.
        errored: Vec<u64>,
    },
    /// A size in the window has a median step count of 0, whose logarithm is
    /// undefined.
    ZeroSteps {
        /// The size concerned.
        n: u64,
    },
    /// Estimator B needs `ln ln n`, which is undefined at `n = 1`.
    SizeTooSmall {
        /// The size concerned.
        n: u64,
        /// The smallest size the estimator accepts.
        min: u64,
    },
    /// The sizes in the window have no spread in `ln n`, so no slope exists.
    DegenerateSizes,
    /// A computed quantity was not finite.
    NonFinite,
}

impl fmt::Display for GrowthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrowthError::ZeroSize => write!(f, "sweep contains size n = 0"),
            GrowthError::DuplicateSize { n } => write!(f, "sweep contains size n = {n} twice"),
            GrowthError::NoSeeds { n } => write!(f, "size n = {n} has no seed step counts"),
            GrowthError::TooFewPoints { points } => {
                write!(f, "fit window has {points} size(s); at least 2 are needed")
            }
            GrowthError::IncompleteWindow { timed_out, errored } => write!(
                f,
                "fit window has incomplete sizes: timed out {timed_out:?}, errored {errored:?}"
            ),
            GrowthError::ZeroSteps { n } => write!(f, "median step count at n = {n} is 0"),
            GrowthError::SizeTooSmall { n, min } => {
                write!(
                    f,
                    "size n = {n} is below the minimum {min} for this estimator"
                )
            }
            GrowthError::DegenerateSizes => write!(f, "sizes in the fit window do not vary"),
            GrowthError::NonFinite => write!(f, "fit produced a non-finite value"),
        }
    }
}

impl std::error::Error for GrowthError {}

/// The sizes a fit used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitWindow {
    /// The smallest size in the window.
    pub n_min: u64,
    /// The largest size in the window.
    pub n_max: u64,
    /// The number of sizes fitted.
    pub points: usize,
    /// Sizes below the window that timed out or failed. They do not affect the
    /// fit; they are reported so that nothing is dropped silently.
    pub incomplete_below: Vec<u64>,
}

/// The quality of a least-squares fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FitQuality {
    /// The coefficient of determination, in `[0, 1]`. Rounding can push the
    /// computed value a hair outside that range; it is clamped.
    RSquared(f64),
    /// The fitted response has zero variance (every median step count in the
    /// window is equal), so R² is undefined. The fit is then exact.
    ZeroVariance,
}

/// The result of estimator A.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLogFit {
    /// The fitted slope `α̂` of `ln steps` on `ln n`.
    pub alpha: f64,
    /// The fitted intercept, `ln` of the constant factor.
    pub ln_c: f64,
    /// R², or a note that it is undefined.
    pub quality: FitQuality,
    /// The sizes fitted.
    pub window: FitWindow,
}

/// The fit of estimator B at one value of `γ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaFit {
    /// The exponent of `ln n`.
    pub gamma: u8,
    /// The fitted exponent of `n`.
    pub alpha: f64,
    /// The fitted `ln c`.
    pub ln_c: f64,
    /// The residual sum of squares, in `ln steps` units.
    pub rss: f64,
}

/// The result of estimator B: the chosen `γ` and every candidate's fit.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerLogFit {
    /// The fit with the smallest residual sum of squares. Ties go to the
    /// smaller `γ`.
    pub best: GammaFit,
    /// The fits for `γ = 0, 1, 2`, in that order.
    pub candidates: [GammaFit; 3],
    /// The sizes fitted.
    pub window: FitWindow,
}

impl PowerLogFit {
    /// The constant factor `c = exp(ln c)` of the best fit. It can overflow to
    /// infinity for extreme fits; [`GammaFit::ln_c`] is always finite.
    pub fn c(&self) -> f64 {
        self.best.ln_c.exp()
    }
}

/// The result of any growth estimator.
#[derive(Debug, Clone, PartialEq)]
pub enum GrowthFit {
    /// From estimator A.
    LogLog(LogLogFit),
    /// From estimator B.
    PowerLog(PowerLogFit),
}

impl GrowthFit {
    /// The fitted exponent of `n`: `α̂` for A, the best fit's `α` for B.
    pub fn alpha(&self) -> f64 {
        match self {
            GrowthFit::LogLog(f) => f.alpha,
            GrowthFit::PowerLog(f) => f.best.alpha,
        }
    }

    /// The sizes fitted.
    pub fn window(&self) -> &FitWindow {
        match self {
            GrowthFit::LogLog(f) => &f.window,
            GrowthFit::PowerLog(f) => &f.window,
        }
    }
}

/// A swappable way of estimating cost growth from a sweep (D5).
pub trait GrowthEstimator {
    /// A short name for logs and results tables.
    fn name(&self) -> &'static str;

    /// Fits the sweep.
    ///
    /// # Errors
    ///
    /// See [`GrowthError`]; an estimator never panics on bad input.
    fn estimate(&self, sweep: &Sweep) -> Result<GrowthFit, GrowthError>;
}

/// Estimator A: OLS slope of `ln steps` on `ln n` (the default, spec §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LogLogSlope {
    /// The sizes to fit over.
    pub window: Window,
}

impl LogLogSlope {
    /// Fits the sweep and returns the estimator's own result type.
    ///
    /// # Errors
    ///
    /// See [`GrowthError`].
    pub fn fit(&self, sweep: &Sweep) -> Result<LogLogFit, GrowthError> {
        let (points, window) = window_points(sweep, self.window, 1)?;
        let xs: Vec<f64> = points.iter().map(|&(n, _)| ln_u64(n)).collect();
        let ys: Vec<f64> = points.iter().map(|&(_, s)| ln_u64(s)).collect();
        let ols = ols(&xs, &ys)?;
        let quality = if ols.syy == 0.0 {
            FitQuality::ZeroVariance
        } else {
            FitQuality::RSquared((1.0 - ols.rss / ols.syy).clamp(0.0, 1.0))
        };
        Ok(LogLogFit {
            alpha: ols.slope,
            ln_c: ols.intercept,
            quality,
            window,
        })
    }
}

impl GrowthEstimator for LogLogSlope {
    fn name(&self) -> &'static str {
        "A:loglog-slope"
    }

    fn estimate(&self, sweep: &Sweep) -> Result<GrowthFit, GrowthError> {
        self.fit(sweep).map(GrowthFit::LogLog)
    }
}

/// Estimator B: `steps ≈ c · n^α · (ln n)^γ`, `γ ∈ {0, 1, 2}`, `α` continuous.
///
/// For each `γ` it fits `ln steps − γ · ln ln n` on `ln n` by OLS and keeps
/// the `γ` with the smallest residual sum of squares. It uses the same window
/// as estimator A by default, so small-`n` transients (such as the `+5` in
/// `3n + 5`) do not push it towards a spurious `γ`, and so the two estimators
/// see the same data. Every size in the window must be at least 2, since
/// `ln ln 1` is undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PowerLogGrid {
    /// The sizes to fit over.
    pub window: Window,
}

/// The grid of `γ` values estimator B searches.
pub const GAMMAS: [u8; 3] = [0, 1, 2];

impl PowerLogGrid {
    /// Fits the sweep and returns the estimator's own result type.
    ///
    /// # Errors
    ///
    /// See [`GrowthError`].
    pub fn fit(&self, sweep: &Sweep) -> Result<PowerLogFit, GrowthError> {
        let (points, window) = window_points(sweep, self.window, 2)?;
        let xs: Vec<f64> = points.iter().map(|&(n, _)| ln_u64(n)).collect();
        let lnln: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
        let ln_steps: Vec<f64> = points.iter().map(|&(_, s)| ln_u64(s)).collect();
        let mut candidates = [GammaFit {
            gamma: 0,
            alpha: 0.0,
            ln_c: 0.0,
            rss: 0.0,
        }; 3];
        for (slot, &gamma) in candidates.iter_mut().zip(GAMMAS.iter()) {
            let g = f64::from(gamma);
            let ys: Vec<f64> = ln_steps.iter().zip(&lnln).map(|(y, l)| y - g * l).collect();
            let fit = ols(&xs, &ys)?;
            *slot = GammaFit {
                gamma,
                alpha: fit.slope,
                ln_c: fit.intercept,
                rss: fit.rss,
            };
        }
        let mut best = candidates[0];
        for c in &candidates[1..] {
            if c.rss < best.rss {
                best = *c;
            }
        }
        Ok(PowerLogFit {
            best,
            candidates,
            window,
        })
    }
}

impl GrowthEstimator for PowerLogGrid {
    fn name(&self) -> &'static str {
        "B:power-log-grid"
    }

    fn estimate(&self, sweep: &Sweep) -> Result<GrowthFit, GrowthError> {
        self.fit(sweep).map(GrowthFit::PowerLog)
    }
}

/// `ln` of a positive integer.
fn ln_u64(v: u64) -> f64 {
    // u64 -> f64 is exact up to 2^53 and correctly rounded above; step counts
    // and sizes are far below the range where that matters for a logarithm.
    let f = v as f64;
    f.ln()
}

/// Selects the window and checks it: at least two sizes, none incomplete, no
/// zero medians, every size at least `min_n`.
fn window_points(
    sweep: &Sweep,
    window: Window,
    min_n: u64,
) -> Result<(Vec<(u64, u64)>, FitWindow), GrowthError> {
    let obs = sweep.observations();
    let start = window.start(obs.len());
    let in_window = &obs[start..];
    if in_window.len() < 2 {
        return Err(GrowthError::TooFewPoints {
            points: in_window.len(),
        });
    }
    let mut timed_out = Vec::new();
    let mut errored = Vec::new();
    let mut points = Vec::with_capacity(in_window.len());
    for o in in_window {
        match &o.outcome {
            SizeOutcome::Completed { steps } => {
                let m = median_steps(steps).ok_or(GrowthError::NoSeeds { n: o.n })?;
                points.push((o.n, m));
            }
            SizeOutcome::TimedOut => timed_out.push(o.n),
            SizeOutcome::Errored => errored.push(o.n),
        }
    }
    if !timed_out.is_empty() || !errored.is_empty() {
        return Err(GrowthError::IncompleteWindow { timed_out, errored });
    }
    for &(n, s) in &points {
        if n < min_n {
            return Err(GrowthError::SizeTooSmall { n, min: min_n });
        }
        if s == 0 {
            return Err(GrowthError::ZeroSteps { n });
        }
    }
    let incomplete_below = obs[..start]
        .iter()
        .filter(|o| !matches!(o.outcome, SizeOutcome::Completed { .. }))
        .map(|o| o.n)
        .collect();
    let fw = FitWindow {
        n_min: in_window[0].n,
        n_max: in_window[in_window.len() - 1].n,
        points: points.len(),
        incomplete_below,
    };
    Ok((points, fw))
}

/// An ordinary least-squares line `y = intercept + slope · x`.
#[derive(Debug, Clone, Copy)]
struct Ols {
    slope: f64,
    intercept: f64,
    rss: f64,
    syy: f64,
}

/// Fits `ys` on `xs` by OLS.
///
/// Both series are shifted by their first value before the means are taken.
/// That keeps the sums well conditioned, and it makes a constant series give
/// a slope, RSS and variance of exactly 0 rather than rounding noise, which is
/// what lets [`FitQuality::ZeroVariance`] be detected exactly.
fn ols(xs: &[f64], ys: &[f64]) -> Result<Ols, GrowthError> {
    debug_assert_eq!(xs.len(), ys.len());
    if xs.len() < 2 {
        return Err(GrowthError::TooFewPoints { points: xs.len() });
    }
    let (x0, y0) = (xs[0], ys[0]);
    let dx: Vec<f64> = xs.iter().map(|x| x - x0).collect();
    let dy: Vec<f64> = ys.iter().map(|y| y - y0).collect();
    let m = xs.len() as f64;
    let xbar = dx.iter().sum::<f64>() / m;
    let ybar = dy.iter().sum::<f64>() / m;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (x, y) in dx.iter().zip(&dy) {
        let (cx, cy) = (x - xbar, y - ybar);
        sxx += cx * cx;
        sxy += cx * cy;
        syy += cy * cy;
    }
    if sxx == 0.0 {
        return Err(GrowthError::DegenerateSizes);
    }
    let slope = sxy / sxx;
    let rss = dx
        .iter()
        .zip(&dy)
        .map(|(x, y)| {
            let r = (y - ybar) - slope * (x - xbar);
            r * r
        })
        .sum::<f64>();
    let intercept = (y0 + ybar) - slope * (x0 + xbar);
    let out = Ols {
        slope,
        intercept,
        rss,
        syy,
    };
    if [out.slope, out.intercept, out.rss, out.syy]
        .iter()
        .all(|v| v.is_finite())
    {
        Ok(out)
    } else {
        Err(GrowthError::NonFinite)
    }
}

#[cfg(test)]
mod tests;
