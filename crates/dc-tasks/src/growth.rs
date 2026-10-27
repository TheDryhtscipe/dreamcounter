//! Cost growth estimation (spec §9, decision D5).
//!
//! A program is run at a sweep of input sizes `n` (by default
//! `16, 32, ..., 1024`), several seeded inputs per size. Step counts are exact
//! integers. This module turns such a sweep into an estimate of how cost grows
//! with `n`, so that programs can be compared on computational cost.
//!
//! Three estimators sit behind the [`GrowthEstimator`] trait:
//!
//! * [`LogLogSlope`] (estimator A, the default): ordinary least squares of
//!   `ln steps` on `ln n`, reporting the slope `α̂` and R².
//! * [`PowerLogGrid`] (estimator B): `steps ≈ c · n^α · (ln n)^γ` with
//!   `γ ∈ {0, 1, 2}` and `α` continuous, choosing the `γ` with the smallest
//!   residual sum of squares.
//! * [`ClassSelect`] (estimator C): `steps ≈ c · f(n)` for `f` one of the
//!   [`GrowthClass`]es `1, log n, n, n log n, n², n³, 2ⁿ`, naming a class only
//!   when exactly one fits within a tolerance that allows for the spread
//!   between seeds. It refuses rather than guess.
//!
//! All three fit over the same [`Window`] of the sweep, by default its upper
//! half.
//!
//! # How decisive estimator B's `γ` is
//!
//! Neighbouring `γ` differ only by the `ln ln n` term, which is nearly linear
//! in `ln n` over a short sweep. Over the default window
//! `n ∈ {128, ..., 1024}`, moving `γ` one step from the truth raises B's RSS
//! by only about `2.0e-4`. So B's choice holds only for low noise in the
//! median step counts. [`PowerLogFit::gamma_stability`] reports the exact
//! margin: any change of at most that much to each `ln` median (a relative
//! change of about the same size to each step count) leaves `γ` unchanged,
//! and a worst-case change slightly larger flips it. For an exact `n log n`
//! series on the default sweep it is about `0.0035`, so B separates
//! `n log n` from `n^1.17` only when each median is accurate to about
//! 0.35 %. Deterministic step counts on fixed inputs meet that; medians that
//! vary by around 1 % with the seeded inputs may not. Check the reported
//! stability before relying on `γ`, or use estimator C, which fixes the
//! exponent and so separates the classes by much more (below).
//!
//! # How estimator C decides
//!
//! Each class has one free parameter, the constant `c`. C fits it in the
//! Chebyshev (minimax) sense: with `d_i = ln steps_i − ln f(n_i)` over the
//! window, `ln c` is the midpoint of the `d_i` and the class's *misfit* is
//! half their range, so every median step count is within a factor
//! `e^misfit` of `c · f(n)`. Half the range is a seminorm, so the triangle
//! inequality applies to it: a series within `τ` of two classes `f` and `g`
//! has `½ range(ln f − ln g) ≤ 2τ`. The *separation* of the window is the
//! smallest `½ range(ln f − ln g)` over all pairs of classes; over the
//! default window it is `½ ln(ln 1024 / ln 128) ≈ 0.178`, for `n` against
//! `n log n` (and `1` against `log n`).
//!
//! The *tolerance* `τ` is the larger of a floor, [`ClassSelect::misfit_fraction`]
//! (default ¼) times the separation, which absorbs small lower-order terms
//! such as the `+ 2n` in `n log₂ n + 2n` (but not `+ 4n`, a `NoFit`), and [`ClassSelect::noise_z`] (default 3)
//! times the seed noise: the largest standard error of a size's median `ln`
//! step count, `√(π/2) · s / √k` for `k` seeds with sample standard deviation
//! `s` of their `ln` steps (the large-sample standard error of a median). A
//! class is named only if it is the one class with misfit at most `τ`; if
//! none is, the verdict is [`ClassVerdict::NoFit`] (`n^1.5`, `n² log n`).
//! Half the separation bounds `τ`: below it at most one class can be within
//! `τ`, but at or above it a series between two classes can sit just inside
//! one and just outside the other, so C then refuses with
//! [`ClassVerdict::TooNoisy`] whatever the misfits. Noise can therefore only
//! make C refuse, never make it name a class.
//!
//! # What a verdict means
//!
//! A named class says that over the window every median step count is
//! within a factor `e^τ` of `c · f(n)`. It is a statement about the window,
//! not a proof of asymptotic growth. Growth close to a class is named as
//! that class: `n^1.2` as `n log n`, `n^1.04` as `n`, and `n log n + 30n`
//! (a large lower-order term) as `n` over `128..1024`, where `ln n` varies by
//! only a factor 1.43. Telling such cases apart needs larger sizes.
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
    /// The fit window has too few sizes for the estimator.
    TooFewPoints {
        /// The number of sizes in the window.
        points: usize,
        /// The number the estimator needs.
        needed: usize,
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
    /// Estimator B cannot tell `γ` values apart: over the window, `ln ln n`
    /// is (to rounding) a linear function of `ln n`, so every `γ` fits
    /// equally well.
    GammaUnidentifiable,
    /// An estimator parameter is out of its documented range.
    BadParameter {
        /// The parameter's field name.
        name: &'static str,
    },
}

impl fmt::Display for GrowthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrowthError::ZeroSize => write!(f, "sweep contains size n = 0"),
            GrowthError::DuplicateSize { n } => write!(f, "sweep contains size n = {n} twice"),
            GrowthError::NoSeeds { n } => write!(f, "size n = {n} has no seed step counts"),
            GrowthError::TooFewPoints { points, needed } => {
                write!(
                    f,
                    "fit window has {points} size(s); at least {needed} are needed"
                )
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
            GrowthError::BadParameter { name } => {
                write!(f, "estimator parameter {name} is out of range")
            }
            GrowthError::GammaUnidentifiable => {
                write!(
                    f,
                    "ln ln n is linear in ln n over the window; gamma is unidentifiable"
                )
            }
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
    /// The least-squares `γ` if `γ` were continuous. Each candidate's RSS is
    /// `RSS(γ̂) + r · (γ − γ̂)²`, where `r` is the RSS of `ln ln n` regressed
    /// on `ln n`, so the chosen `γ` is the grid point nearest `γ̂`.
    pub gamma_hat: f64,
    /// The largest `δ` such that changing each window size's `ln` median step
    /// count by less than `δ` cannot change the chosen `γ`. It is the
    /// distance from `γ̂` to the nearest decision boundary (0.5 or 1.5) times
    /// `r / Σ|p_i|`, where `p` are the residuals of `ln ln n` on `ln n`: a
    /// change `ε` moves `γ̂` by `Σ ε_i p_i / r`, at most `δ · Σ|p_i| / r`.
    /// The bound is tight: `ε = ±δ · sign(p)` reaches it. In ln units, so
    /// `0.0035` means about 0.35 % per step count. Always finite and `≥ 0`.
    pub gamma_stability: f64,
    /// The sizes fitted.
    pub window: FitWindow,
}

impl PowerLogFit {
    /// The constant factor `c = exp(ln c)` of the best fit. It can overflow to
    /// infinity for extreme fits; [`GammaFit::ln_c`] is always finite.
    pub fn c(&self) -> f64 {
        self.best.ln_c.exp()
    }

    /// The RSS of the runner-up `γ` minus the RSS of the best: how much worse
    /// the next-best candidate fits. Always `≥ 0`.
    pub fn rss_margin(&self) -> f64 {
        self.candidates
            .iter()
            .filter(|c| c.gamma != self.best.gamma)
            .map(|c| c.rss - self.best.rss)
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }
}

/// The result of any growth estimator.
#[derive(Debug, Clone, PartialEq)]
pub enum GrowthFit {
    /// From estimator A.
    LogLog(LogLogFit),
    /// From estimator B.
    PowerLog(PowerLogFit),
    /// From estimator C, boxed because it is several times the size of the
    /// others.
    Class(Box<ClassFit>),
}

impl GrowthFit {
    /// The fitted exponent of `n`: `α̂` for A, the best fit's `α` for B.
    /// `None` for C, which fits classes, not an exponent.
    pub fn alpha(&self) -> Option<f64> {
        match self {
            GrowthFit::LogLog(f) => Some(f.alpha),
            GrowthFit::PowerLog(f) => Some(f.best.alpha),
            GrowthFit::Class(_) => None,
        }
    }

    /// The sizes fitted.
    pub fn window(&self) -> &FitWindow {
        match self {
            GrowthFit::LogLog(f) => &f.window,
            GrowthFit::PowerLog(f) => &f.window,
            GrowthFit::Class(f) => &f.window,
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
        let (points, window) = window_points(sweep, self.window, 2, 1)?;
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
///
/// B needs at least [`PowerLogGrid::MIN_POINTS`] = 3 sizes in the window.
/// Each candidate has two free parameters, so on two points every `γ` fits
/// exactly and the choice of `γ` would be decided by rounding noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PowerLogGrid {
    /// The sizes to fit over.
    pub window: Window,
}

/// The grid of `γ` values estimator B searches.
pub const GAMMAS: [u8; 3] = [0, 1, 2];

impl PowerLogGrid {
    /// The fewest sizes in the window for which `γ` can be chosen.
    pub const MIN_POINTS: usize = 3;

    /// Fits the sweep and returns the estimator's own result type.
    ///
    /// # Errors
    ///
    /// See [`GrowthError`].
    pub fn fit(&self, sweep: &Sweep) -> Result<PowerLogFit, GrowthError> {
        let (points, window) = window_points(sweep, self.window, Self::MIN_POINTS, 2)?;
        let xs: Vec<f64> = points.iter().map(|&(n, _)| ln_u64(n)).collect();
        let lnln: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
        let ln_steps: Vec<f64> = points.iter().map(|&(_, s)| ln_u64(s)).collect();
        let lnln_fit = ols(&xs, &lnln)?;
        let mut gamma0_residuals = Vec::new();
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
            if gamma == 0 {
                gamma0_residuals = fit.residuals.clone();
            }
            *slot = GammaFit {
                gamma,
                alpha: fit.slope,
                ln_c: fit.intercept,
                rss: fit.rss,
            };
        }
        let best = select_best(&candidates);
        let (gamma_hat, gamma_stability) =
            gamma_decision(best.gamma, &gamma0_residuals, &lnln_fit.residuals)?;
        Ok(PowerLogFit {
            best,
            candidates,
            gamma_hat,
            gamma_stability,
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

/// A growth class of estimator C (spec §9): `steps ≈ c · f(n)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GrowthClass {
    /// `f(n) = 1`.
    Constant,
    /// `f(n) = ln n`.
    Log,
    /// `f(n) = n`.
    Linear,
    /// `f(n) = n ln n`.
    NLogN,
    /// `f(n) = n²`.
    Quadratic,
    /// `f(n) = n³`.
    Cubic,
    /// `f(n) = 2ⁿ`.
    Exponential,
}

impl GrowthClass {
    /// Every class, in order of growth. A class's position here is its index
    /// in [`ClassFit::candidates`].
    pub const ALL: [GrowthClass; 7] = [
        GrowthClass::Constant,
        GrowthClass::Log,
        GrowthClass::Linear,
        GrowthClass::NLogN,
        GrowthClass::Quadratic,
        GrowthClass::Cubic,
        GrowthClass::Exponential,
    ];

    /// `ln f(n)`, for `n ≥ 2` (`ln ln 1` is undefined).
    pub fn ln_f(self, n: u64) -> f64 {
        let ln_n = ln_u64(n);
        match self {
            GrowthClass::Constant => 0.0,
            GrowthClass::Log => ln_n.ln(),
            GrowthClass::Linear => ln_n,
            GrowthClass::NLogN => ln_n + ln_n.ln(),
            GrowthClass::Quadratic => 2.0 * ln_n,
            GrowthClass::Cubic => 3.0 * ln_n,
            // n ≤ 2^64, so n ln 2 is exact to a few ulps.
            GrowthClass::Exponential => n as f64 * std::f64::consts::LN_2,
        }
    }
}

impl fmt::Display for GrowthClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GrowthClass::Constant => "1",
            GrowthClass::Log => "log n",
            GrowthClass::Linear => "n",
            GrowthClass::NLogN => "n log n",
            GrowthClass::Quadratic => "n^2",
            GrowthClass::Cubic => "n^3",
            GrowthClass::Exponential => "2^n",
        })
    }
}

/// Estimator C's fit of one class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassCandidate {
    /// The class.
    pub class: GrowthClass,
    /// The minimax `ln c`: the midpoint of `ln steps − ln f(n)` over the
    /// window.
    pub ln_c: f64,
    /// Half the range of `ln steps − ln f(n)` over the window: every median
    /// step count is within a factor `e^misfit` of `c · f(n)`. In `ln` units,
    /// `≥ 0`.
    pub misfit: f64,
}

/// What estimator C concludes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassVerdict {
    /// Exactly one class is within tolerance.
    Class(GrowthClass),
    /// The seed noise puts the tolerance at or above half the separation, so
    /// several classes could fit and C does not choose. Raised before the
    /// misfits are looked at.
    TooNoisy,
    /// No class is within tolerance: the growth lies between classes (such as
    /// `n^1.5`) or outside them.
    NoFit,
}

/// The result of estimator C.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassFit {
    /// The conclusion.
    pub verdict: ClassVerdict,
    /// Every class's fit, in the order of [`GrowthClass::ALL`].
    pub candidates: [ClassCandidate; 7],
    /// The largest misfit at which a class counts as fitting, in `ln` units.
    pub tolerance: f64,
    /// The smallest `½ range(ln f − ln g)` over pairs of classes on the
    /// window: how far apart the closest classes are.
    pub separation: f64,
    /// The largest standard error of a median `ln` step count over the
    /// window, estimated from the sizes with at least two seeds. `None` if
    /// every size in the window has a single seed, so the noise is unknown
    /// and only the floor applies.
    pub noise: Option<f64>,
    /// The sizes fitted.
    pub window: FitWindow,
}

impl ClassFit {
    /// The class with the smallest misfit; on a tie, the slower-growing.
    pub fn best(&self) -> ClassCandidate {
        let mut best = self.candidates[0];
        for c in &self.candidates[1..] {
            if c.misfit < best.misfit {
                best = *c;
            }
        }
        best
    }

    /// The named class, if the verdict names one.
    pub fn class(&self) -> Option<GrowthClass> {
        match self.verdict {
            ClassVerdict::Class(c) => Some(c),
            ClassVerdict::TooNoisy | ClassVerdict::NoFit => None,
        }
    }
}

/// Estimator C: class selection among [`GrowthClass::ALL`] (spec §9). See the
/// module comment for how it decides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassSelect {
    /// The sizes to fit over.
    pub window: Window,
    /// The tolerance floor as a fraction of the separation, in `[0, ½)`. Below
    /// ½ the classes are exclusive for exact step counts; the default ¼ leaves
    /// half of that margin for lower-order terms.
    pub misfit_fraction: f64,
    /// How many standard errors of seed noise the tolerance allows; finite
    /// and `≥ 0`.
    pub noise_z: f64,
}

impl Default for ClassSelect {
    fn default() -> Self {
        Self {
            window: Window::UpperHalf,
            misfit_fraction: 0.25,
            noise_z: 3.0,
        }
    }
}

impl ClassSelect {
    /// The fewest sizes in the window. On two, every class's misfit is half
    /// a single difference, so the shape of growth is not tested.
    pub const MIN_POINTS: usize = 3;

    /// Fits the sweep and returns the estimator's own result type.
    ///
    /// # Errors
    ///
    /// See [`GrowthError`]. Besides the medians, every seed's step count in
    /// the window must be positive ([`GrowthError::ZeroSteps`]), since the
    /// noise is estimated from their logarithms. [`GrowthError::BadParameter`]
    /// if `misfit_fraction` or `noise_z` is out of range.
    pub fn fit(&self, sweep: &Sweep) -> Result<ClassFit, GrowthError> {
        if !(0.0..0.5).contains(&self.misfit_fraction) {
            return Err(GrowthError::BadParameter {
                name: "misfit_fraction",
            });
        }
        if !(self.noise_z.is_finite() && self.noise_z >= 0.0) {
            return Err(GrowthError::BadParameter { name: "noise_z" });
        }
        let (points, window) = window_points(sweep, self.window, Self::MIN_POINTS, 2)?;
        let ln_steps: Vec<f64> = points.iter().map(|&(_, s)| ln_u64(s)).collect();
        let ln_fs: Vec<Vec<f64>> = GrowthClass::ALL
            .iter()
            .map(|k| points.iter().map(|&(n, _)| k.ln_f(n)).collect())
            .collect();
        let candidates = std::array::from_fn(|i| {
            let d: Vec<f64> = ln_steps.iter().zip(&ln_fs[i]).map(|(y, f)| y - f).collect();
            let (lo, hi) = min_max(&d);
            ClassCandidate {
                class: GrowthClass::ALL[i],
                ln_c: lo + (hi - lo) / 2.0,
                misfit: (hi - lo) / 2.0,
            }
        });
        let mut separation = f64::INFINITY;
        for i in 0..ln_fs.len() {
            for j in i + 1..ln_fs.len() {
                let d: Vec<f64> = ln_fs[i].iter().zip(&ln_fs[j]).map(|(a, b)| a - b).collect();
                let (lo, hi) = min_max(&d);
                separation = separation.min((hi - lo) / 2.0);
            }
        }
        let noise = seed_noise(sweep, window.n_min)?;
        let tolerance =
            (self.misfit_fraction * separation).max(self.noise_z * noise.unwrap_or(0.0));
        // Exclusivity: with τ < separation / 2 at most one class is within τ.
        let verdict = if tolerance >= separation / 2.0 {
            ClassVerdict::TooNoisy
        } else {
            match candidates.iter().find(|c| c.misfit <= tolerance) {
                Some(c) => ClassVerdict::Class(c.class),
                None => ClassVerdict::NoFit,
            }
        };
        let finite = [tolerance, separation, noise.unwrap_or(0.0)]
            .iter()
            .chain(
                candidates
                    .iter()
                    .flat_map(|c: &ClassCandidate| [&c.ln_c, &c.misfit]),
            )
            .all(|v| v.is_finite());
        if !finite {
            return Err(GrowthError::NonFinite);
        }
        Ok(ClassFit {
            verdict,
            candidates,
            tolerance,
            separation,
            noise,
            window,
        })
    }
}

impl GrowthEstimator for ClassSelect {
    fn name(&self) -> &'static str {
        "C:class-select"
    }

    fn estimate(&self, sweep: &Sweep) -> Result<GrowthFit, GrowthError> {
        self.fit(sweep).map(|f| GrowthFit::Class(Box::new(f)))
    }
}

/// The smallest and largest of a non-empty slice.
fn min_max(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        })
}

/// The largest standard error of a median `ln` step count over the completed
/// sizes from `n_min` up: `√(π/2) · s / √k` for `k ≥ 2` seeds whose `ln`
/// steps have sample standard deviation `s`. Sizes with one seed say nothing
/// about the noise and are skipped; `None` if every size has one seed.
/// The caller has already checked that those sizes completed.
fn seed_noise(sweep: &Sweep, n_min: u64) -> Result<Option<f64>, GrowthError> {
    let mut noise: Option<f64> = None;
    for o in sweep.observations().iter().filter(|o| o.n >= n_min) {
        let SizeOutcome::Completed { steps } = &o.outcome else {
            continue;
        };
        if steps.contains(&0) {
            return Err(GrowthError::ZeroSteps { n: o.n });
        }
        if steps.len() < 2 {
            continue;
        }
        let ys: Vec<f64> = steps.iter().map(|&s| ln_u64(s)).collect();
        let k = ys.len() as f64;
        let mean = ys.iter().sum::<f64>() / k;
        let var = ys.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / (k - 1.0);
        let se = (std::f64::consts::FRAC_PI_2 * var / k).sqrt();
        noise = Some(noise.map_or(se, |m| m.max(se)));
    }
    Ok(noise)
}

/// The candidate with the smallest RSS; on a tie, the earliest (smallest `γ`,
/// the simpler model).
fn select_best(candidates: &[GammaFit; 3]) -> GammaFit {
    let mut best = candidates[0];
    for c in &candidates[1..] {
        if c.rss < best.rss {
            best = *c;
        }
    }
    best
}

/// The continuous least-squares `γ̂` and the stability radius of the chosen
/// `best` (see [`PowerLogFit::gamma_stability`]).
///
/// `py` are the residuals of `ln steps` on `ln n` (the `γ = 0` fit) and `pl`
/// the residuals of `ln ln n` on `ln n`. The `γ` fit's residuals are
/// `py − γ · pl`, so its RSS is a quadratic in `γ` minimised at
/// `γ̂ = ⟨py, pl⟩ / ‖pl‖²`.
fn gamma_decision(best: u8, py: &[f64], pl: &[f64]) -> Result<(f64, f64), GrowthError> {
    let r: f64 = pl.iter().map(|p| p * p).sum();
    let l1: f64 = pl.iter().map(|p| p.abs()).sum();
    if r <= 0.0 || l1 <= 0.0 {
        return Err(GrowthError::GammaUnidentifiable);
    }
    let dot: f64 = py.iter().zip(pl).map(|(y, p)| y * p).sum();
    let gamma_hat = dot / r;
    let g = f64::from(best);
    let lo = if best == GAMMAS[0] {
        f64::NEG_INFINITY
    } else {
        g - 0.5
    };
    let hi = if best == GAMMAS[GAMMAS.len() - 1] {
        f64::INFINITY
    } else {
        g + 0.5
    };
    // Rounding can put γ̂ a hair outside the chosen γ's band at a near-tie;
    // the choice is then as fragile as it gets, so the radius is 0.
    let distance = (gamma_hat - lo).min(hi - gamma_hat).max(0.0);
    let stability = distance * r / l1;
    if gamma_hat.is_finite() && stability.is_finite() {
        Ok((gamma_hat, stability))
    } else {
        Err(GrowthError::NonFinite)
    }
}

/// `ln` of a positive integer.
fn ln_u64(v: u64) -> f64 {
    // u64 -> f64 is exact up to 2^53 and correctly rounded above; step counts
    // and sizes are far below the range where that matters for a logarithm.
    let f = v as f64;
    f.ln()
}

/// Selects the window and checks it: at least `min_points` sizes, none
/// incomplete, no zero medians, every size at least `min_n`.
fn window_points(
    sweep: &Sweep,
    window: Window,
    min_points: usize,
    min_n: u64,
) -> Result<(Vec<(u64, u64)>, FitWindow), GrowthError> {
    let obs = sweep.observations();
    let start = window.start(obs.len());
    let in_window = &obs[start..];
    if in_window.len() < min_points {
        return Err(GrowthError::TooFewPoints {
            points: in_window.len(),
            needed: min_points,
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
#[derive(Debug, Clone)]
struct Ols {
    slope: f64,
    intercept: f64,
    rss: f64,
    syy: f64,
    /// `y_i − (intercept + slope · x_i)`, computed in centred form.
    residuals: Vec<f64>,
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
        return Err(GrowthError::TooFewPoints {
            points: xs.len(),
            needed: 2,
        });
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
    let residuals: Vec<f64> = dx
        .iter()
        .zip(&dy)
        .map(|(x, y)| (y - ybar) - slope * (x - xbar))
        .collect();
    let rss = residuals.iter().map(|r| r * r).sum::<f64>();
    let intercept = (y0 + ybar) - slope * (x0 + xbar);
    let out = Ols {
        slope,
        intercept,
        rss,
        syy,
        residuals,
    };
    if [out.slope, out.intercept, out.rss, out.syy]
        .iter()
        .chain(&out.residuals)
        .all(|v| v.is_finite())
    {
        Ok(out)
    } else {
        Err(GrowthError::NonFinite)
    }
}

#[cfg(test)]
mod tests;
