//! Tests for growth estimation on synthetic exact step series.
//!
//! Tolerances come from the maths, not from tuning:
//!
//! * **Exact power laws** (`c · n^k`, `c · n^k · (log2 n)^g`) are fitted with
//!   [`EXACT`] = 1e-9. Inputs are integers whose logarithms are computed to
//!   within a few ulps (about 1e-15 absolute for values below 45), and the
//!   OLS slope over the default window amplifies an error in `y` by at most
//!   `Σ|x − x̄| / Sxx ≈ 1.4`, so rounding error is around 1e-14. 1e-9 leaves
//!   five orders of margin and is still far below any difference of interest.
//! * **Series that are not exact power laws** (`3n + 5`, `n log n`) are
//!   checked against a band. The OLS slope is a weighted mean, with
//!   non-negative weights, of the secant slopes between pairs of points, and
//!   each secant slope is a mean of the local slope `d ln steps / d ln n` over
//!   its interval. So the OLS slope lies between the smallest and largest
//!   local slope over the window. For a monotone local slope those are its
//!   values at `n_min` and `n_max`.

use super::*;
use proptest::prelude::*;

/// Rounding tolerance for exact power-law series; see the module comment.
const EXACT: f64 = 1e-9;

/// `log2 n` for a power of two.
fn log2(n: u64) -> u64 {
    u64::from(n.trailing_zeros())
}

fn default_sweep(f: impl Fn(u64) -> u64) -> Sweep {
    Sweep::from_fn(&DEFAULT_SIZES, f).expect("valid sweep")
}

fn a(sweep: &Sweep) -> LogLogFit {
    LogLogSlope::default().fit(sweep).expect("A fits")
}

fn b(sweep: &Sweep) -> PowerLogFit {
    PowerLogGrid::default().fit(sweep).expect("B fits")
}

fn r_squared(q: FitQuality) -> f64 {
    match q {
        FitQuality::RSquared(r) => r,
        FitQuality::ZeroVariance => panic!("expected a defined R²"),
    }
}

/// Asserts `lo − EXACT ≤ v ≤ hi + EXACT`.
fn assert_in_band(v: f64, lo: f64, hi: f64) {
    assert!(
        v >= lo - EXACT && v <= hi + EXACT,
        "{v} not in [{lo}, {hi}]"
    );
}

/// The residual sum of squares of `ln ln n` regressed on `ln n` over the
/// default window. Moving `γ` one step away from the true value adds exactly
/// this to estimator B's RSS on an exact series, so it is the margin by which
/// B separates neighbouring `γ`.
fn lnln_residual() -> f64 {
    let xs: Vec<f64> = [128u64, 256, 512, 1024]
        .iter()
        .map(|&n| (n as f64).ln())
        .collect();
    let ys: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    ols(&xs, &ys).expect("fits").rss
}

// ---------------------------------------------------------------- median

#[test]
fn median_of_odd_count_is_middle_value() {
    assert_eq!(median_steps(&[50, 10, 30]), Some(30));
    assert_eq!(median_steps(&[9, 1, 7, 3, 5]), Some(5));
}

#[test]
fn median_of_even_count_is_lower_middle_value() {
    assert_eq!(median_steps(&[40, 10, 30, 20]), Some(20));
    assert_eq!(median_steps(&[2, 1]), Some(1));
}

#[test]
fn median_of_empty_is_none() {
    assert_eq!(median_steps(&[]), None);
}

#[test]
fn sweep_medians_skip_incomplete_sizes() {
    let sweep = Sweep::new(vec![
        SizeObservation::completed(32, vec![7, 3, 5]),
        SizeObservation::timed_out(64),
        SizeObservation::completed(16, vec![4, 2]),
        SizeObservation::errored(128),
    ])
    .expect("valid");
    assert_eq!(sweep.medians(), vec![(16, 2), (32, 5)]);
}

proptest! {
    #[test]
    fn median_is_an_element_with_lower_median_rank(
        mut v in prop::collection::vec(0u64..1000, 1..12)
    ) {
        let m = median_steps(&v).expect("non-empty");
        prop_assert!(v.contains(&m));
        let k = v.len();
        let le = v.iter().filter(|&&x| x <= m).count();
        let ge = v.iter().filter(|&&x| x >= m).count();
        // Lower median: at least ceil(k/2) values ≤ m, at least floor(k/2)+1 ≥ m.
        prop_assert!(le >= k.div_ceil(2));
        prop_assert!(ge > k / 2);
        v.reverse();
        prop_assert_eq!(median_steps(&v), Some(m));
    }
}

// ---------------------------------------------------------------- sweep validation

#[test]
fn sweep_sorts_by_size() {
    let sweep = Sweep::from_fn(&[64, 16, 32], |n| n).expect("valid");
    let ns: Vec<u64> = sweep.observations().iter().map(|o| o.n).collect();
    assert_eq!(ns, vec![16, 32, 64]);
}

#[test]
fn sweep_rejects_duplicate_size() {
    assert_eq!(
        Sweep::from_fn(&[16, 32, 16], |n| n),
        Err(GrowthError::DuplicateSize { n: 16 })
    );
}

#[test]
fn sweep_rejects_zero_size() {
    assert_eq!(
        Sweep::from_fn(&[0, 16], |n| n + 1),
        Err(GrowthError::ZeroSize)
    );
}

#[test]
fn sweep_rejects_completed_size_without_seeds() {
    assert_eq!(
        Sweep::new(vec![SizeObservation::completed(16, vec![])]),
        Err(GrowthError::NoSeeds { n: 16 })
    );
}

// ---------------------------------------------------------------- window

#[test]
fn upper_half_of_default_sweep_is_128_to_1024() {
    let fit = a(&default_sweep(|n| n));
    assert_eq!(
        fit.window,
        FitWindow {
            n_min: 128,
            n_max: 1024,
            points: 4,
            incomplete_below: vec![],
        }
    );
}

#[test]
fn upper_half_of_even_sweep_is_last_half() {
    let sweep = Sweep::from_fn(&[16, 32, 64, 128, 256, 512], |n| n).expect("valid");
    let w = a(&sweep).window;
    assert_eq!((w.n_min, w.n_max, w.points), (128, 512, 3));
}

#[test]
fn upper_half_uses_the_median_steps_of_each_size() {
    // Seeds disagree; medians are exactly n², so the slope is exactly 2. The
    // outlying seeds would pull the slope off 2 if they were averaged in.
    let obs = DEFAULT_SIZES
        .iter()
        .map(|&n| SizeObservation::completed(n, vec![n * n, 1, n * n * n, n * n, 7]))
        .collect();
    let fit = a(&Sweep::new(obs).expect("valid"));
    assert!((fit.alpha - 2.0).abs() < EXACT, "alpha {}", fit.alpha);
}

#[test]
fn window_all_uses_every_size() {
    // 3n + 5 has a larger transient at small n, so the full-sweep slope is
    // lower than the upper-half slope; check the window really changed.
    let sweep = default_sweep(|n| 3 * n + 5);
    let all = LogLogSlope {
        window: Window::All,
    }
    .fit(&sweep)
    .expect("fits");
    assert_eq!((all.window.n_min, all.window.points), (16, 7));
    assert!(all.alpha < a(&sweep).alpha);
}

// ---------------------------------------------------------------- estimator A

#[test]
fn a_on_linear_with_offset_lies_in_local_slope_band() {
    // Local slope of 3n + 5 is 3n / (3n + 5), increasing in n.
    let local = |n: f64| 3.0 * n / (3.0 * n + 5.0);
    let fit = a(&default_sweep(|n| 3 * n + 5));
    assert_in_band(fit.alpha, local(128.0), local(1024.0));
    assert!(r_squared(fit.quality) > 0.999_9);
}

#[test]
fn a_on_exact_powers_recovers_exponent() {
    for (k, f) in [
        (1.0, (|n| n) as fn(u64) -> u64),
        (2.0, |n| n * n),
        (3.0, |n| n * n * n),
    ] {
        let fit = a(&default_sweep(f));
        assert!((fit.alpha - k).abs() < EXACT, "k {k}: alpha {}", fit.alpha);
        assert!((r_squared(fit.quality) - 1.0).abs() < EXACT);
    }
}

#[test]
fn a_on_constant_has_zero_slope_and_undefined_r_squared() {
    let fit = a(&default_sweep(|_| 42));
    assert_eq!(fit.alpha, 0.0);
    assert_eq!(fit.quality, FitQuality::ZeroVariance);
    assert!((fit.ln_c - 42f64.ln()).abs() < EXACT);
}

/// The mean of repeated copies of a float is not always that float (for
/// example seven copies of `ln 5`), so a naive OLS reports rounding noise as
/// slope and variance. The fit must still give exactly 0 and
/// [`FitQuality::ZeroVariance`].
#[test]
fn a_on_constant_is_exact_where_naive_means_round() {
    let all = LogLogSlope {
        window: Window::All,
    };
    for (sizes, c) in [(&DEFAULT_SIZES[..], 5), (&DEFAULT_SIZES[..5], 6)] {
        let fit = all
            .fit(&Sweep::from_fn(sizes, |_| c).expect("valid"))
            .expect("fits");
        assert_eq!(fit.alpha, 0.0, "c {c}");
        assert_eq!(fit.quality, FitQuality::ZeroVariance, "c {c}");
    }
}

#[test]
fn a_on_n_log_n_reads_about_1_17() {
    // Local slope of n log n is 1 + 1/ln n, decreasing in n (spec §9).
    let local = |n: f64| 1.0 + 1.0 / n.ln();
    let fit = a(&default_sweep(|n| n * log2(n)));
    assert_in_band(fit.alpha, local(1024.0), local(128.0));
    assert!((fit.alpha - 1.17).abs() < 0.01, "alpha {}", fit.alpha);
}

#[test]
fn a_on_n_squared_log_n_lies_in_local_slope_band() {
    let local = |n: f64| 2.0 + 1.0 / n.ln();
    let fit = a(&default_sweep(|n| n * n * log2(n)));
    assert_in_band(fit.alpha, local(1024.0), local(128.0));
}

/// On `2^n` the local slope `d ln steps / d ln n` is `n ln 2`, so A's slope is
/// not a growth exponent at all: it is about `n ln 2` at the window, and it
/// rises without bound as the sweep moves to larger `n`. A large and
/// sweep-dependent `α̂` is the signal of super-polynomial growth.
#[test]
fn a_on_exponential_is_large_and_rises_with_the_sweep() {
    let pow2 = |n: u64| 1u64 << n;
    // Two-point windows: the slope is the secant (n2 − n1) ln 2 / ln 2 = n1.
    let s1 = a(&Sweep::from_fn(&[2, 4, 8, 16], pow2).expect("valid"));
    let s2 = a(&Sweep::from_fn(&[4, 8, 16, 32], pow2).expect("valid"));
    assert!((s1.alpha - 8.0).abs() < EXACT, "alpha {}", s1.alpha);
    assert!((s2.alpha - 16.0).abs() < EXACT, "alpha {}", s2.alpha);
    // Three-point window {8, 16, 32}: x = 3, 4, 5 and y = 8, 16, 32 in units
    // of ln 2, so the slope is (32 − 8) / 2 = 12, and the curvature shows as
    // R² < 1.
    let s3 = a(&Sweep::from_fn(&[2, 4, 8, 16, 32], pow2).expect("valid"));
    assert!((s3.alpha - 12.0).abs() < EXACT, "alpha {}", s3.alpha);
    assert!(r_squared(s3.quality) < 0.99);
    assert!(s1.alpha > 3.0 && s1.alpha < s3.alpha && s3.alpha < s2.alpha);
}

// ---------------------------------------------------------------- estimator B

#[test]
fn b_on_n_log_n_chooses_gamma_1_alpha_1() {
    let fit = b(&default_sweep(|n| n * log2(n)));
    assert_eq!(fit.best.gamma, 1);
    assert!(
        (fit.best.alpha - 1.0).abs() < EXACT,
        "alpha {}",
        fit.best.alpha
    );
    // steps = n · ln n / ln 2, so c = 1 / ln 2.
    assert!((fit.c() - 1.0 / 2f64.ln()).abs() < EXACT, "c {}", fit.c());
    assert!(fit.best.rss < EXACT);
}

#[test]
fn b_on_n_squared_log_n_chooses_gamma_1_alpha_2() {
    let fit = b(&default_sweep(|n| n * n * log2(n)));
    assert_eq!(fit.best.gamma, 1);
    assert!(
        (fit.best.alpha - 2.0).abs() < EXACT,
        "alpha {}",
        fit.best.alpha
    );
}

#[test]
fn b_on_exact_powers_chooses_gamma_0() {
    for (k, f) in [
        (0.0, (|_| 9) as fn(u64) -> u64),
        (1.0, |n| n),
        (2.0, |n| n * n),
        (3.0, |n| n * n * n),
    ] {
        let fit = b(&default_sweep(f));
        assert_eq!(fit.best.gamma, 0, "k {k}");
        assert!((fit.best.alpha - k).abs() < EXACT, "k {k}");
    }
}

#[test]
fn b_on_linear_with_offset_chooses_gamma_0_and_agrees_with_a() {
    // With γ = 0, B is the same regression as A over the same window.
    let sweep = default_sweep(|n| 3 * n + 5);
    let fit = b(&sweep);
    assert_eq!(fit.best.gamma, 0);
    assert!((fit.best.alpha - a(&sweep).alpha).abs() < EXACT);
}

#[test]
fn b_wrong_gamma_costs_exactly_the_lnln_residual() {
    // On an exact c · n · ln n series, γ = 0 and γ = 2 each leave ± ln ln n
    // unexplained, so their RSS is the ln ln n residual exactly.
    let r = lnln_residual();
    assert!(r > 1e-4, "margin {r}");
    let fit = b(&default_sweep(|n| n * log2(n)));
    assert!((fit.candidates[0].rss - r).abs() < EXACT);
    assert!((fit.candidates[2].rss - r).abs() < EXACT);
    assert_eq!(
        fit.candidates.map(|c| c.gamma),
        [0, 1, 2],
        "candidates in γ order"
    );
}

/// The residuals of `ln ln n` regressed on `ln n` over the default window,
/// computed independently of the module's OLS.
fn lnln_residuals() -> [f64; 4] {
    let xs = [128u64, 256, 512, 1024].map(|n| (n as f64).ln());
    let ls = xs.map(f64::ln);
    let xbar = xs.iter().sum::<f64>() / 4.0;
    let lbar = ls.iter().sum::<f64>() / 4.0;
    let sxx: f64 = xs.iter().map(|x| (x - xbar).powi(2)).sum();
    let sxl: f64 = xs
        .iter()
        .zip(&ls)
        .map(|(x, l)| (x - xbar) * (l - lbar))
        .sum();
    let slope = sxl / sxx;
    let mut out = [0.0; 4];
    for (o, (x, l)) in out.iter_mut().zip(xs.iter().zip(&ls)) {
        *o = (l - lbar) - slope * (x - xbar);
    }
    out
}

/// The stability radius of an exact series whose `γ̂` is an integer: `γ̂` is
/// 0.5 from the decision boundary, and a perturbation bounded by `δ` per
/// point moves `γ̂` by at most `δ · Σ|p_i| / Σ p_i²`, with `p` the `ln ln n`
/// residuals.
fn exact_stability() -> f64 {
    let p = lnln_residuals();
    let r: f64 = p.iter().map(|v| v * v).sum();
    let l1: f64 = p.iter().map(|v| v.abs()).sum();
    0.5 * r / l1
}

/// A sweep of `1e9 · n · ln n · e^(ε_i)` over the default sizes, with `ε_i`
/// applied to the `i`-th window size (`n = 128 · 2^i`). Rounding to integers
/// adds at most `0.5 / 1e11` to each `ln steps`, far below the margins tested.
fn perturbed_n_log_n(eps: [f64; 4]) -> Sweep {
    default_sweep(|n| {
        let e = if n >= 128 {
            eps[n.trailing_zeros() as usize - 7]
        } else {
            0.0
        };
        (1e9 * (n as f64) * (n as f64).ln() * e.exp()).round() as u64
    })
}

#[test]
fn b_reports_continuous_gamma_and_its_stability_on_n_log_n() {
    let fit = b(&default_sweep(|n| n * log2(n)));
    assert!((fit.gamma_hat - 1.0).abs() < 1e-6, "γ̂ {}", fit.gamma_hat);
    let s = exact_stability();
    assert!(
        (fit.gamma_stability - s).abs() < 1e-9,
        "{} vs {s}",
        fit.gamma_stability
    );
    // About 0.35 % per point in step-count terms, the figure the docs give.
    assert!(s > 0.0034 && s < 0.0036, "{s}");
    // The runner-up is one γ step away and pays the ln ln n residual.
    assert!((fit.rss_margin() - lnln_residual()).abs() < 1e-9);
}

/// The stability radius is tight: the worst-case perturbation `±δ · sign(p)`
/// keeps `γ` just below the radius and flips it just above, towards either
/// neighbour.
#[test]
fn b_gamma_flips_exactly_at_the_stability_radius() {
    let s = exact_stability();
    let sign = lnln_residuals().map(f64::signum);
    for (dir, flipped) in [(-1.0, 0u8), (1.0, 2u8)] {
        let below = b(&perturbed_n_log_n(sign.map(|v| dir * 0.98 * s * v)));
        assert_eq!(below.best.gamma, 1, "dir {dir}, 0.98 δ");
        assert!(
            below.gamma_stability > 0.0 && below.gamma_stability < 0.03 * s,
            "{}",
            below.gamma_stability
        );
        let above = b(&perturbed_n_log_n(sign.map(|v| dir * 1.02 * s * v)));
        assert_eq!(above.best.gamma, flipped, "dir {dir}, 1.02 δ");
    }
    // A ±1 % perturbation, as on noisy data, is well past the radius.
    let one_pct = b(&perturbed_n_log_n(sign.map(|v| -0.01 * v)));
    assert_eq!(one_pct.best.gamma, 0);
}

#[test]
fn b_gamma_decision_refuses_when_ln_ln_n_is_explained_by_ln_n() {
    assert_eq!(
        gamma_decision(0, &[0.1, -0.2, 0.1], &[0.0, 0.0, 0.0]),
        Err(GrowthError::GammaUnidentifiable)
    );
}

/// The spec's warning (§9): over `n ∈ [128, 1024]`, A gives `n log n` and a
/// pure power `n^β`, with `β` set to A's own reading of `n log n`, the same
/// slope. B separates them.
#[test]
fn a_cannot_separate_n_log_n_from_n_to_1_17_but_b_can() {
    let nlogn = default_sweep(|n| n * log2(n));
    let beta = a(&nlogn).alpha;
    let exact = |n: u64| 1000.0 * (n as f64).powf(beta);
    let power = default_sweep(|n| exact(n).round() as u64);

    // Rounding to integers perturbs ln steps by at most delta, which moves an
    // OLS slope by at most delta · Σ|x − x̄| / Sxx.
    let window = [128u64, 256, 512, 1024];
    let delta = window
        .iter()
        .map(|&n| ((exact(n).round()).ln() - exact(n).ln()).abs())
        .fold(0.0, f64::max);
    let xs: Vec<f64> = window.iter().map(|&n| (n as f64).ln()).collect();
    let xbar = xs.iter().sum::<f64>() / 4.0;
    let sxx: f64 = xs.iter().map(|x| (x - xbar).powi(2)).sum();
    let spread: f64 = xs.iter().map(|x| (x - xbar).abs()).sum();
    let slope_tol = delta * spread / sxx + EXACT;
    assert!(slope_tol < 1e-5, "rounding bound {slope_tol}");

    // A: indistinguishable.
    let a_power = a(&power).alpha;
    assert!((a_power - beta).abs() < slope_tol, "{a_power} vs {beta}");

    // B: γ = 1 for n log n, γ = 0 for the pure power.
    let b_nlogn = b(&nlogn);
    let b_power = b(&power);
    assert_eq!(b_nlogn.best.gamma, 1);
    assert_eq!(b_power.best.gamma, 0);
    assert!((b_power.best.alpha - beta).abs() < slope_tol);

    // The separation is large compared with rounding noise. The residual is
    // a projection, so on the pure power the γ = 0 RSS is at most m · δ², and
    // the γ = 1 RSS is at least (√r − √m · δ)² by the triangle inequality.
    let r = lnln_residual();
    let noise = 2.0 * delta; // √m · δ with m = 4
    assert!(b_power.candidates[0].rss <= noise * noise + EXACT);
    assert!(b_power.candidates[1].rss >= (r.sqrt() - noise).powi(2) - EXACT);
    assert!(noise * noise < r / 1e4, "noise {noise} vs margin {r}");
}

/// B is not meaningful on `2^n`: no candidate comes close to fitting. It
/// still returns finite numbers, with every RSS orders of magnitude above the
/// polynomial cases.
#[test]
fn b_on_exponential_fits_nothing_well() {
    let fit = b(&Sweep::from_fn(&[2, 4, 8, 16, 32], |n| 1u64 << n).expect("valid"));
    for c in fit.candidates {
        assert!(c.rss > 1.0, "γ {}: rss {}", c.gamma, c.rss);
        assert!(c.alpha > 3.0 && c.alpha.is_finite() && c.ln_c.is_finite());
    }
}

// ---------------------------------------------------------------- trait

#[test]
fn estimators_are_swappable_behind_the_trait() {
    let ests: Vec<Box<dyn GrowthEstimator>> = vec![
        Box::new(LogLogSlope::default()),
        Box::new(PowerLogGrid::default()),
    ];
    let sweep = default_sweep(|n| n * log2(n));
    let fits: Vec<GrowthFit> = ests
        .iter()
        .map(|e| e.estimate(&sweep).expect("fits"))
        .collect();
    assert!(matches!(fits[0], GrowthFit::LogLog(_)));
    assert!(matches!(fits[1], GrowthFit::PowerLog(_)));
    assert!((fits[0].alpha() - a(&sweep).alpha).abs() < EXACT);
    assert!((fits[1].alpha() - 1.0).abs() < EXACT);
    assert_eq!(fits[0].window(), fits[1].window());
    assert_ne!(ests[0].name(), ests[1].name());
}

// ---------------------------------------------------------------- degenerate input

#[test]
fn too_few_points_is_an_error() {
    let empty = Sweep::new(vec![]).expect("valid");
    assert_eq!(
        LogLogSlope::default().fit(&empty),
        Err(GrowthError::TooFewPoints {
            points: 0,
            needed: 2
        })
    );
    let two = Sweep::from_fn(&[16, 32], |n| n).expect("valid");
    assert_eq!(
        LogLogSlope::default().fit(&two),
        Err(GrowthError::TooFewPoints {
            points: 1,
            needed: 2
        })
    );
    assert_eq!(
        PowerLogGrid::default().fit(&two),
        Err(GrowthError::TooFewPoints {
            points: 1,
            needed: 3
        })
    );
    // The same two sizes are enough over the whole sweep.
    let all = LogLogSlope {
        window: Window::All,
    }
    .fit(&two)
    .expect("fits");
    assert!((all.alpha - 1.0).abs() < EXACT);
}

#[test]
fn b_needs_three_points_to_choose_gamma() {
    // Four sizes: the upper half has two, which A can fit but B cannot use,
    // because every γ fits two points exactly.
    let four = Sweep::from_fn(&[16, 32, 64, 128], |n| n * log2(n)).expect("valid");
    assert!(LogLogSlope::default().fit(&four).is_ok());
    assert_eq!(
        PowerLogGrid::default().fit(&four),
        Err(GrowthError::TooFewPoints {
            points: 2,
            needed: 3
        })
    );
    // Five sizes: three in the upper half, enough for B to find γ = 1.
    let five = Sweep::from_fn(&[16, 32, 64, 128, 256], |n| n * log2(n)).expect("valid");
    assert_eq!(b(&five).best.gamma, 1);
}

#[test]
fn b_ties_go_to_the_smaller_gamma() {
    let fit = |gamma, rss| GammaFit {
        gamma,
        alpha: 1.0,
        ln_c: 0.0,
        rss,
    };
    assert_eq!(
        select_best(&[fit(0, 0.5), fit(1, 0.5), fit(2, 0.5)]).gamma,
        0
    );
    assert_eq!(
        select_best(&[fit(0, 0.9), fit(1, 0.5), fit(2, 0.5)]).gamma,
        1
    );
    assert_eq!(
        select_best(&[fit(0, 0.9), fit(1, 0.7), fit(2, 0.5)]).gamma,
        2
    );
}

#[test]
fn timeout_in_window_is_refused_and_reported() {
    let mut obs: Vec<SizeObservation> = DEFAULT_SIZES
        .iter()
        .map(|&n| SizeObservation::completed(n, vec![n * n]))
        .collect();
    obs[5] = SizeObservation::timed_out(512);
    obs[6] = SizeObservation::errored(1024);
    let sweep = Sweep::new(obs).expect("valid");
    let expected = GrowthError::IncompleteWindow {
        timed_out: vec![512],
        errored: vec![1024],
    };
    assert_eq!(LogLogSlope::default().fit(&sweep), Err(expected.clone()));
    assert_eq!(PowerLogGrid::default().fit(&sweep), Err(expected));
}

#[test]
fn timeout_below_window_is_reported_not_fitted() {
    let mut obs: Vec<SizeObservation> = DEFAULT_SIZES
        .iter()
        .map(|&n| SizeObservation::completed(n, vec![n * n]))
        .collect();
    obs[1] = SizeObservation::timed_out(32);
    let fit = a(&Sweep::new(obs).expect("valid"));
    assert_eq!(fit.window.incomplete_below, vec![32]);
    assert_eq!(fit.window.points, 4);
    assert!((fit.alpha - 2.0).abs() < EXACT);
}

#[test]
fn zero_steps_in_window_is_an_error() {
    let sweep = default_sweep(|n| if n == 256 { 0 } else { n });
    assert_eq!(
        LogLogSlope::default().fit(&sweep),
        Err(GrowthError::ZeroSteps { n: 256 })
    );
    assert_eq!(
        PowerLogGrid::default().fit(&sweep),
        Err(GrowthError::ZeroSteps { n: 256 })
    );
}

#[test]
fn zero_steps_below_window_does_not_affect_the_fit() {
    let fit = a(&default_sweep(|n| if n == 16 { 0 } else { n }));
    assert!((fit.alpha - 1.0).abs() < EXACT);
}

#[test]
fn b_rejects_size_one_and_a_accepts_it() {
    let sweep = Sweep::from_fn(&[1, 2, 4], |n| 5 * n).expect("valid");
    let all_b = PowerLogGrid {
        window: Window::All,
    };
    assert_eq!(
        all_b.fit(&sweep),
        Err(GrowthError::SizeTooSmall { n: 1, min: 2 })
    );
    let all_a = LogLogSlope {
        window: Window::All,
    };
    assert!((all_a.fit(&sweep).expect("fits").alpha - 1.0).abs() < EXACT);
}

#[test]
fn ols_rejects_equal_x() {
    assert!(matches!(
        ols(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]),
        Err(GrowthError::DegenerateSizes)
    ));
}

#[test]
fn ols_rejects_non_finite_input() {
    assert!(matches!(
        ols(&[1.0, f64::INFINITY], &[1.0, 2.0]),
        Err(GrowthError::NonFinite)
    ));
    assert!(matches!(
        ols(&[1.0, 2.0], &[1.0, f64::NAN]),
        Err(GrowthError::NonFinite)
    ));
}

// ---------------------------------------------------------------- laws

proptest! {
    /// c · n^k · (log2 n)^g is recovered exactly by B, and A agrees with B
    /// when g = 0.
    #[test]
    fn exact_power_log_series_are_recovered(
        c in 1u64..1000, k in 0u32..=2, g in 0u32..=2
    ) {
        let sweep = default_sweep(|n| c * n.pow(k) * log2(n).pow(g));
        let fb = b(&sweep);
        prop_assert_eq!(u32::from(fb.best.gamma), g);
        prop_assert!((fb.best.alpha - f64::from(k)).abs() < EXACT);
        if g == 0 {
            prop_assert!((a(&sweep).alpha - f64::from(k)).abs() < EXACT);
        }
    }

    /// Scaling every step count by a constant moves only the intercept.
    #[test]
    fn slope_is_invariant_to_constant_factor(
        steps in prop::collection::vec(1u64..1_000_000, 7), c in 2u64..1000
    ) {
        let base = Sweep::from_fn(&DEFAULT_SIZES, |n| {
            steps[n.trailing_zeros() as usize - 4]
        }).expect("valid");
        let scaled = Sweep::from_fn(&DEFAULT_SIZES, |n| {
            c * steps[n.trailing_zeros() as usize - 4]
        }).expect("valid");
        let (fa, fs) = (a(&base), a(&scaled));
        prop_assert!((fa.alpha - fs.alpha).abs() < EXACT);
        prop_assert!((fs.ln_c - fa.ln_c - (c as f64).ln()).abs() < EXACT);
        let (ba, bs) = (b(&base), b(&scaled));
        prop_assert!((ba.best.alpha - bs.best.alpha).abs() < EXACT);
    }

    /// The slope lies between the smallest and largest secant slope of
    /// consecutive window points, and every returned float is finite.
    #[test]
    fn slope_is_bounded_by_consecutive_secants(
        steps in prop::collection::vec(1u64..1_000_000, 7)
    ) {
        let sweep = Sweep::from_fn(&DEFAULT_SIZES, |n| {
            steps[n.trailing_zeros() as usize - 4]
        }).expect("valid");
        let fit = a(&sweep);
        let w = &steps[3..];
        let secants: Vec<f64> = w
            .windows(2)
            .map(|p| ((p[1] as f64).ln() - (p[0] as f64).ln()) / 2f64.ln())
            .collect();
        let lo = secants.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = secants.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        prop_assert!(fit.alpha >= lo - EXACT && fit.alpha <= hi + EXACT);
        prop_assert!(fit.alpha.is_finite() && fit.ln_c.is_finite());
        if let FitQuality::RSquared(r) = fit.quality {
            prop_assert!((0.0..=1.0).contains(&r));
        }
        for c in b(&sweep).candidates {
            prop_assert!(c.alpha.is_finite() && c.ln_c.is_finite() && c.rss.is_finite());
        }
    }

    /// Any perturbation of the window's `ln steps` by less than the reported
    /// stability radius leaves B's choice of `γ` unchanged.
    #[test]
    fn gamma_is_unchanged_within_the_stability_radius(
        u in prop::array::uniform4(-0.99f64..0.99)
    ) {
        let clean = b(&perturbed_n_log_n([0.0; 4]));
        prop_assert_eq!(clean.best.gamma, 1);
        let s = clean.gamma_stability;
        prop_assert!(s > 0.0034);
        let noisy = b(&perturbed_n_log_n(u.map(|v| v * s)));
        prop_assert_eq!(noisy.best.gamma, 1);
    }

    /// B's RSS is a quadratic in `γ` with curvature `r` (the `ln ln n`
    /// residual) and minimum at `γ̂`; the chosen `γ` is the grid point
    /// nearest `γ̂`; the margin is the gap to the runner-up; the stability
    /// radius is never negative.
    #[test]
    fn gamma_geometry_laws(steps in prop::collection::vec(1u64..1_000_000, 7)) {
        let sweep = Sweep::from_fn(&DEFAULT_SIZES, |n| {
            steps[n.trailing_zeros() as usize - 4]
        }).expect("valid");
        let fit = b(&sweep);
        let r = lnln_residual();
        let g = fit.gamma_hat;
        prop_assert!(g.is_finite() && fit.gamma_stability.is_finite());
        prop_assert!(fit.gamma_stability >= 0.0);
        let base = fit.best.rss - r * (f64::from(fit.best.gamma) - g).powi(2);
        for c in fit.candidates {
            let predicted = base + r * (f64::from(c.gamma) - g).powi(2);
            prop_assert!((c.rss - predicted).abs() < 1e-9 * (1.0 + c.rss));
        }
        // Away from a boundary the chosen γ is the grid point nearest γ̂.
        if (g - g.floor() - 0.5).abs() > 1e-6 {
            prop_assert_eq!(f64::from(fit.best.gamma), g.round().clamp(0.0, 2.0));
        }
        let mut rss = fit.candidates.map(|c| c.rss);
        rss.sort_by(f64::total_cmp);
        prop_assert!((fit.rss_margin() - (rss[1] - rss[0])).abs() < 1e-12);
    }
}
