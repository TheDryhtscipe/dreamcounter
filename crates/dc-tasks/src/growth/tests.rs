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
    assert!((fits[0].alpha().expect("A has an exponent") - a(&sweep).alpha).abs() < EXACT);
    assert!((fits[1].alpha().expect("B has an exponent") - 1.0).abs() < EXACT);
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

// ------------------------------------------------------------- estimator C

/// Estimator C with the given noise model and the other defaults.
fn c_with(noise_model: NoiseModel) -> ClassSelect {
    ClassSelect {
        noise_model,
        ..ClassSelect::default()
    }
}

/// C on an exact series, such as one from [`Sweep::from_fn`] (one seed per
/// size): [`NoiseModel::Exact`].
fn c(sweep: &Sweep) -> ClassFit {
    c_with(NoiseModel::Exact).fit(sweep).expect("C fits")
}

/// C with the defaults, which estimate the noise from the seeds.
fn c_seeded(sweep: &Sweep) -> ClassFit {
    ClassSelect::default().fit(sweep).expect("C fits")
}

/// Half the range of `ln f₁ − ln f₂` over the default window for `f₁ = n log n`
/// and `f₂ = n`: `½ ln(ln 1024 / ln 128) = ½ ln(10/7)`. It is also the
/// separation of `log n` from `1`, and the smallest over all class pairs.
fn default_separation() -> f64 {
    0.5 * (10.0f64 / 7.0).ln()
}

/// The default tolerance on the default window: a quarter of the
/// separation, 0.0446, capped at 0.04.
fn default_tolerance() -> f64 {
    let d = ClassSelect::default();
    (d.misfit_fraction * default_separation()).min(d.max_tolerance)
}

/// A class and an exact step series of it.
type ClassCase = (GrowthClass, fn(u64) -> u64);

/// Sizes on which `2ⁿ` fits in a `u64`.
const EXP_SIZES: [u64; 6] = [10, 20, 30, 40, 50, 60];

#[test]
fn c_names_the_class_of_each_exact_series() {
    let cases: [ClassCase; 6] = [
        (GrowthClass::Constant, |_| 7),
        (GrowthClass::Log, |n| 1000 * log2(n)),
        (GrowthClass::Linear, |n| 3 * n),
        (GrowthClass::NLogN, |n| n * log2(n)),
        (GrowthClass::Quadratic, |n| n * n),
        (GrowthClass::Cubic, |n| n * n * n),
    ];
    for (class, f) in cases {
        let fit = c(&default_sweep(f));
        assert_eq!(fit.verdict, ClassVerdict::Class(class), "{class:?}");
        assert_eq!(fit.best().class, class);
        assert!(
            fit.best().misfit < EXACT,
            "{class:?}: {}",
            fit.best().misfit
        );
    }
    let exp = Sweep::from_fn(&EXP_SIZES, |n| 1 << n).expect("valid");
    assert_eq!(
        c(&exp).verdict,
        ClassVerdict::Class(GrowthClass::Exponential)
    );
}

#[test]
fn c_absorbs_lower_order_terms() {
    for (class, f) in [
        (GrowthClass::Linear, (|n| 3 * n + 5) as fn(u64) -> u64),
        // 1 + 2/log2 n runs from 1.29 to 1.2 over the window: misfit 0.035.
        (GrowthClass::NLogN, |n| n * log2(n) + 2 * n),
        (GrowthClass::Quadratic, |n| n * n / 4 + 3 * n),
    ] {
        let fit = c(&default_sweep(f));
        assert_eq!(fit.verdict, ClassVerdict::Class(class), "{fit:?}");
    }
}

#[test]
fn c_refuses_growth_outside_its_classes() {
    // n^1.5 and n² log n lie between classes; neither is within tolerance of
    // any class, so C says so rather than naming the nearest.
    for f in [
        (|n| ((n as f64).powf(1.5) * 1000.0) as u64) as fn(u64) -> u64,
        |n| n * n * log2(n),
    ] {
        let fit = c(&default_sweep(f));
        assert_eq!(fit.verdict, ClassVerdict::NoFit, "{fit:?}");
        assert!(fit.best().misfit > fit.tolerance);
    }
}

/// `2⁸, 2¹⁰, …, 2²⁰`.
const WIDE_SIZES: [u64; 7] = [1 << 8, 1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20];

/// Exact `100 · n^a` over `2⁸, 2¹⁰, …, 2²⁰`, all fitted: a window where
/// `ln n` varies by a factor 2.5 and the separation is 0.458.
fn wide_power_law(a: f64) -> (ClassSelect, Sweep) {
    let sweep = Sweep::from_fn(&WIDE_SIZES, |n| ((n as f64).powf(a) * 100.0).round() as u64)
        .expect("valid");
    let select = ClassSelect {
        window: Window::All,
        ..c_with(NoiseModel::Exact)
    };
    (select, sweep)
}

#[test]
fn c_tolerance_does_not_grow_with_a_wide_window() {
    // A quarter of the separation would be 0.115 here, within which exact
    // n^1.1 (misfit 0.072 from n log n) is named n log n. Capped, the
    // tolerance stays where it is on the default window, and the wider
    // window tells n^1.1 from n log n.
    let (select, sweep) = wide_power_law(1.1);
    let fit = select.fit(&sweep).expect("fits");
    assert!(fit.separation > 0.45, "{fit:?}");
    let nlogn = fit.candidates[GrowthClass::NLogN as usize].misfit;
    assert!(nlogn > 0.07 && nlogn < 0.075, "{fit:?}");
    assert_eq!(
        fit.tolerance,
        ClassSelect::default().max_tolerance,
        "{fit:?}"
    );
    assert_eq!(fit.verdict, ClassVerdict::NoFit, "{fit:?}");

    // Uncapped, it is named n log n: the cap is what refuses it.
    let uncapped = ClassSelect {
        max_tolerance: f64::INFINITY,
        ..select
    };
    let fit = uncapped.fit(&sweep).expect("fits");
    assert!((fit.tolerance - 0.25 * fit.separation).abs() < EXACT);
    assert_eq!(fit.verdict, ClassVerdict::Class(GrowthClass::NLogN));

    // A lower-order term that is large at the window's small end is refused:
    // ½ (ln(1 + 2/8) − ln(1 + 2/20)) = 0.064.
    let plus_2n = Sweep::from_fn(&WIDE_SIZES, |n| n * log2(n) + 2 * n).expect("valid");
    let fit = select.fit(&plus_2n).expect("fits");
    let nlogn = fit.candidates[GrowthClass::NLogN as usize].misfit;
    assert!((nlogn - 0.0639).abs() < 1e-3, "{fit:?}");
    assert_eq!(fit.verdict, ClassVerdict::NoFit, "{fit:?}");

    // The classes themselves are still named on the wide window.
    for (a, class) in [(1.0, GrowthClass::Linear), (2.0, GrowthClass::Quadratic)] {
        let (select, sweep) = wide_power_law(a);
        let fit = select.fit(&sweep).expect("fits");
        assert_eq!(fit.verdict, ClassVerdict::Class(class), "{fit:?}");
    }
}

#[test]
fn c_misfit_is_the_half_range_and_ln_c_the_midpoint() {
    // Against class n, ln(3n + 5) − ln n = ln(3 + 5/n), which falls with n.
    let fit = c(&default_sweep(|n| 3 * n + 5));
    let lin = fit.candidates[GrowthClass::Linear as usize];
    assert_eq!(lin.class, GrowthClass::Linear);
    let (hi, lo) = ((3.0 + 5.0 / 128.0f64).ln(), (3.0 + 5.0 / 1024.0f64).ln());
    assert!((lin.misfit - (hi - lo) / 2.0).abs() < EXACT, "{lin:?}");
    assert!((lin.ln_c - (hi + lo) / 2.0).abs() < EXACT, "{lin:?}");
}

#[test]
fn c_separation_tolerance_and_margin_on_an_exact_series() {
    let fit = c(&default_sweep(|n| 3 * n));
    assert!((fit.separation - default_separation()).abs() < EXACT);
    assert_eq!(fit.noise, Some(0.0), "exact: noise 0");
    assert_eq!(fit.noise_margin, 0.0);
    assert!((fit.tolerance - default_tolerance()).abs() < EXACT);
    // The default tolerance keeps classes exclusive: no series can lie
    // within tolerance of two classes.
    assert!(fit.tolerance < fit.separation / 2.0);
}

#[test]
fn c_refuses_one_seed_per_size_unless_told_the_noise() {
    let sweep = default_sweep(|n| n * n);
    let fit = c_seeded(&sweep);
    assert_eq!(fit.noise, None, "one seed per size: noise unknown");
    assert_eq!(fit.noise_margin, 0.0);
    assert_eq!(fit.verdict, ClassVerdict::NoiseUnknown);
    assert_eq!(fit.class(), None);

    // The same series, declared exact or with a known small noise, is named.
    let exact = c(&sweep);
    assert_eq!(exact.candidates, fit.candidates);
    assert_eq!(exact.verdict, ClassVerdict::Class(GrowthClass::Quadratic));
    let known = c_with(NoiseModel::Known(0.01)).fit(&sweep).expect("fits");
    assert_eq!(known.noise, Some(0.01));
    assert!((known.noise_margin - 0.03).abs() < EXACT, "{known:?}");
    assert_eq!(known.verdict, ClassVerdict::Class(GrowthClass::Quadratic));

    // A known noise large enough to bring n log n within reach of exact n
    // refuses.
    let lin = default_sweep(|n| 3 * n);
    let loud = c_with(NoiseModel::Known(0.1)).fit(&lin).expect("fits");
    assert!(
        loud.candidates[GrowthClass::NLogN as usize].misfit <= loud.tolerance + loud.noise_margin
    );
    assert_eq!(loud.verdict, ClassVerdict::TooNoisy, "{loud:?}");
}

/// A default sweep with five seeds per size: `base(n)` scaled by
/// `1 + δ · {−2, −1, 0, 1, 2}`.
fn seeded_sweep(base: impl Fn(u64) -> u64, delta: f64) -> Sweep {
    Sweep::new(
        DEFAULT_SIZES
            .iter()
            .map(|&n| {
                let b = base(n) as f64;
                let steps = [-2.0, -1.0, 0.0, 1.0, 2.0]
                    .iter()
                    .map(|k| (b * (1.0 + delta * k)).round() as u64)
                    .collect();
                SizeObservation::completed(n, steps)
            })
            .collect(),
    )
    .expect("valid")
}

/// The noise C should report for [`seeded_sweep`]: the largest, over the
/// window, of `√(π/2) · sd / √5`, the standard error of a median of five,
/// with `sd` the sample standard deviation of the seeds' `ln` steps.
fn seeded_noise(base: impl Fn(u64) -> u64, delta: f64) -> f64 {
    [128u64, 256, 512, 1024]
        .iter()
        .map(|&n| {
            let b = base(n) as f64;
            let ys: Vec<f64> = [-2.0, -1.0, 0.0, 1.0, 2.0]
                .iter()
                .map(|k| ((b * (1.0 + delta * k)).round()).ln())
                .collect();
            let m = ys.iter().sum::<f64>() / 5.0;
            let var = ys.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / 4.0;
            (std::f64::consts::FRAC_PI_2 * var / 5.0).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
fn c_seed_noise_sets_the_noise_margin_not_the_tolerance() {
    let delta = 0.03;
    let fit = c_seeded(&seeded_sweep(|n| 1000 * n, delta));
    let noise = seeded_noise(|n| 1000 * n, delta);
    assert!((fit.noise.expect("five seeds") - noise).abs() < EXACT);
    let z = ClassSelect::default().noise_z;
    assert!(
        z * noise > default_tolerance(),
        "the test needs a noise margin above the tolerance"
    );
    assert!(
        (fit.tolerance - default_tolerance()).abs() < EXACT,
        "{fit:?}"
    );
    assert!((fit.noise_margin - z * noise).abs() < EXACT, "{fit:?}");
    // n log n, a separation away from exact n, is beyond τ + m.
    assert!(fit.candidates[GrowthClass::NLogN as usize].misfit > fit.tolerance + fit.noise_margin);
    assert_eq!(fit.verdict, ClassVerdict::Class(GrowthClass::Linear));
}

#[test]
fn c_refuses_when_noise_brings_another_class_within_reach() {
    // The medians are exactly linear (the middle seed is the base), but
    // seeds this far apart put n log n within τ + m.
    let sweep = seeded_sweep(|n| 1000 * n, 0.15);
    let fit = c_seeded(&sweep);
    assert!(fit.candidates[GrowthClass::Linear as usize].misfit <= fit.tolerance);
    assert!(
        fit.candidates[GrowthClass::NLogN as usize].misfit <= fit.tolerance + fit.noise_margin,
        "{fit:?}"
    );
    assert_eq!(fit.verdict, ClassVerdict::TooNoisy);
    assert_eq!(
        c(&sweep).verdict,
        ClassVerdict::Class(GrowthClass::Linear),
        "the same medians, declared exact"
    );
}

#[test]
fn c_never_names_a_class_for_noisy_growth_between_classes() {
    // n^1.5 lies between n log n and n²: outside τ of both, but with seeds
    // 13 % apart n log n is within τ + m. C refuses.
    let f = |n: u64| ((n as f64).powf(1.5) * 1000.0) as u64;
    let sweep = seeded_sweep(f, 0.13);
    let fit = c_seeded(&sweep);
    let nlogn = fit.candidates[GrowthClass::NLogN as usize].misfit;
    assert!(nlogn > fit.tolerance, "{fit:?}");
    assert!(nlogn <= fit.tolerance + fit.noise_margin, "{fit:?}");
    assert_eq!(fit.verdict, ClassVerdict::TooNoisy, "{fit:?}");
    assert_eq!(c(&sweep).verdict, ClassVerdict::NoFit);
}

#[test]
fn c_seed_noise_does_not_name_growth_between_n_and_n_log_n() {
    // Review 2 of PR 19, HIGH 1: when seed noise widened the tolerance,
    // 1000 · n^1.1 and 1000 · n ln ln n with five seeds 3 % apart were named
    // n log n (misfits 0.0744 and 0.0765 against a widened tolerance of
    // 0.0799), though their exact medians are NoFit.
    let cases: [fn(u64) -> u64; 2] = [
        |n| (1000.0 * (n as f64).powf(1.1)) as u64,
        |n| (1000.0 * n as f64 * (n as f64).ln().ln()) as u64,
    ];
    for f in cases {
        let exact = c(&default_sweep(f));
        assert_eq!(exact.verdict, ClassVerdict::NoFit, "{exact:?}");
        for delta in [0.0, 0.02, 0.03, 0.04] {
            let fit = c_seeded(&seeded_sweep(f, delta));
            assert!(
                matches!(fit.verdict, ClassVerdict::TooNoisy | ClassVerdict::NoFit),
                "delta {delta}: {fit:?}"
            );
            assert!(fit.candidates[GrowthClass::NLogN as usize].misfit > fit.tolerance);
        }
    }
}

#[test]
fn c_noise_ignores_single_seed_sizes_but_not_the_others() {
    // As in the noisy linear sweep, but n = 1024 has one seed: the other
    // sizes still show the spread, so C still refuses.
    let mut obs = seeded_sweep(|n| 1000 * n, 0.15).observations().to_vec();
    obs[6] = SizeObservation::completed(1024, vec![1000 * 1024]);
    let fit = c_seeded(&Sweep::new(obs).expect("valid"));
    assert!(fit.noise.expect("other sizes have seeds") > 0.0);
    assert_eq!(fit.verdict, ClassVerdict::TooNoisy, "{fit:?}");
}

#[test]
fn c_rejects_parameters_out_of_range() {
    let sweep = default_sweep(|n| n);
    let mut cases = vec![
        (
            ClassSelect {
                max_tolerance: -0.01,
                ..ClassSelect::default()
            },
            "max_tolerance",
        ),
        (
            ClassSelect {
                max_tolerance: f64::NAN,
                ..ClassSelect::default()
            },
            "max_tolerance",
        ),
        (
            ClassSelect {
                misfit_fraction: 0.5,
                ..ClassSelect::default()
            },
            "misfit_fraction",
        ),
        (
            ClassSelect {
                misfit_fraction: -0.1,
                ..ClassSelect::default()
            },
            "misfit_fraction",
        ),
        (
            ClassSelect {
                misfit_fraction: f64::NAN,
                ..ClassSelect::default()
            },
            "misfit_fraction",
        ),
        (
            ClassSelect {
                noise_z: -1.0,
                ..ClassSelect::default()
            },
            "noise_z",
        ),
        (
            ClassSelect {
                noise_z: f64::INFINITY,
                ..ClassSelect::default()
            },
            "noise_z",
        ),
    ];
    for v in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.push((c_with(NoiseModel::Known(v)), "noise_model"));
    }
    for (est, name) in cases {
        assert_eq!(est.fit(&sweep), Err(GrowthError::BadParameter { name }));
    }
    assert!(c_with(NoiseModel::Known(0.0)).fit(&sweep).is_ok());
}

#[test]
fn c_refuses_incomplete_windows_and_short_ones() {
    let sweep = Sweep::new(
        DEFAULT_SIZES[..6]
            .iter()
            .map(|&n| SizeObservation::completed(n, vec![n]))
            .chain([SizeObservation::timed_out(1024)])
            .collect(),
    )
    .expect("valid");
    assert_eq!(
        ClassSelect::default().fit(&sweep),
        Err(GrowthError::IncompleteWindow {
            timed_out: vec![1024],
            errored: vec![]
        })
    );
    let short = Sweep::from_fn(&[16, 32, 64, 128], |n| n).expect("valid");
    assert_eq!(
        ClassSelect::default().fit(&short),
        Err(GrowthError::TooFewPoints {
            points: 2,
            needed: ClassSelect::MIN_POINTS
        })
    );
    let seed_zero = Sweep::new(
        DEFAULT_SIZES
            .iter()
            .map(|&n| SizeObservation::completed(n, vec![0, n, n]))
            .collect(),
    )
    .expect("valid");
    for model in [NoiseModel::FromSeeds, NoiseModel::Exact] {
        assert_eq!(
            c_with(model).fit(&seed_zero),
            Err(GrowthError::ZeroSteps { n: 128 })
        );
    }
}

#[test]
fn c_through_the_trait() {
    let est: &dyn GrowthEstimator = &c_with(NoiseModel::Exact);
    assert_eq!(est.name(), "C:class-select");
    let fit = est.estimate(&default_sweep(|n| n * n)).expect("fits");
    assert_eq!(fit.alpha(), None);
    match fit {
        GrowthFit::Class(f) => assert_eq!(f.class(), Some(GrowthClass::Quadratic)),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        LogLogSlope::default()
            .estimate(&default_sweep(|n| n))
            .expect("fits")
            .alpha()
            .map(f64::round),
        Some(1.0)
    );
}

proptest! {
    /// Whatever the step counts: every float is finite; the misfits, the
    /// separation and the noise match a recomputation from the seeds; the
    /// tolerance is the floor and below half the separation; the noise
    /// margin is `noise_z` times the noise; and the verdict follows the rule
    /// in the module comment, recomputed here from the misfits.
    #[test]
    fn c_verdict_laws(
        steps in prop::collection::vec(prop::collection::vec(1u64..1_000_000, 1..6), 7)
    ) {
        let seeds = steps[0].len();
        let per_size: Vec<Vec<u64>> = steps
            .iter()
            .map(|s| s.iter().copied().cycle().take(seeds).collect())
            .collect();
        let sweep = Sweep::new(
            DEFAULT_SIZES
                .iter()
                .zip(&per_size)
                .map(|(&n, s)| SizeObservation::completed(n, s.clone()))
                .collect(),
        ).expect("valid");
        let fit = c_seeded(&sweep);
        prop_assert!(fit.tolerance.is_finite() && fit.separation.is_finite());
        prop_assert!(fit.noise_margin.is_finite());
        prop_assert!(fit.candidates.iter().all(|k| k.misfit.is_finite() && k.ln_c.is_finite()));
        // Recompute the misfits and the separation from the medians here.
        let window = [128u64, 256, 512, 1024];
        let meds: Vec<f64> = sweep.medians()[3..].iter().map(|&(_, m)| (m as f64).ln()).collect();
        let half_range = |d: &[f64]| {
            let hi = d.iter().copied().fold(f64::MIN, f64::max);
            let lo = d.iter().copied().fold(f64::MAX, f64::min);
            (hi - lo) / 2.0
        };
        let lnf: Vec<Vec<f64>> = GrowthClass::ALL
            .iter()
            .map(|k| window.iter().map(|&n| k.ln_f(n)).collect())
            .collect();
        let mut sep = f64::INFINITY;
        for i in 0..7 {
            let d: Vec<f64> = meds.iter().zip(&lnf[i]).map(|(y, f)| y - f).collect();
            prop_assert!((fit.candidates[i].misfit - half_range(&d)).abs() < 1e-9);
            for j in i + 1..7 {
                let d: Vec<f64> = lnf[i].iter().zip(&lnf[j]).map(|(a, b)| a - b).collect();
                sep = sep.min(half_range(&d));
            }
        }
        prop_assert!((fit.separation - sep).abs() < 1e-9);
        // Recompute the noise: the largest standard error of a median over
        // the window, from the seeds' ln steps.
        let noise = (seeds >= 2).then(|| {
            per_size[3..]
                .iter()
                .map(|s| {
                    let ys: Vec<f64> = s.iter().map(|&v| (v as f64).ln()).collect();
                    let k = ys.len() as f64;
                    let m = ys.iter().sum::<f64>() / k;
                    let var = ys.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / (k - 1.0);
                    (std::f64::consts::FRAC_PI_2 * var / k).sqrt()
                })
                .fold(0.0, f64::max)
        });
        match (fit.noise, noise) {
            (None, None) => {}
            (Some(got), Some(want)) => prop_assert!((got - want).abs() < 1e-9),
            (got, want) => prop_assert!(false, "noise {got:?}, expected {want:?}"),
        }
        let defaults = ClassSelect::default();
        let floor = (defaults.misfit_fraction * sep).min(defaults.max_tolerance);
        prop_assert!((fit.tolerance - floor).abs() < 1e-9);
        prop_assert!(fit.tolerance < sep / 2.0);
        prop_assert!(
            (fit.noise_margin - defaults.noise_z * noise.unwrap_or(0.0)).abs() < 1e-9
        );
        let within = |limit: f64| -> Vec<GrowthClass> {
            fit.candidates.iter().filter(|k| k.misfit <= limit).map(|k| k.class).collect()
        };
        let named = within(fit.tolerance);
        let near = within(fit.tolerance + fit.noise_margin);
        prop_assert!(named.len() <= 1, "classes are exclusive within τ");
        let expected = match noise {
            None => ClassVerdict::NoiseUnknown,
            Some(_) if named.len() == 1 && near.len() == 1 => ClassVerdict::Class(named[0]),
            Some(_) if !near.is_empty() => ClassVerdict::TooNoisy,
            Some(_) => ClassVerdict::NoFit,
        };
        prop_assert_eq!(&fit.verdict, &expected);
    }

    /// With the medians fixed, noise only refuses: the verdict with any
    /// noise margin is the verdict for exact step counts or `TooNoisy`, a
    /// zero margin gives the exact verdict, and more noise refuses whenever
    /// less does. The series are near or between classes, `n^α (ln n)^γ`,
    /// with multiplicative seed jitter.
    #[test]
    fn c_noise_only_refuses(
        alpha in 0.0f64..3.2,
        gamma in -1.0f64..2.0,
        seeds in 1usize..6,
        jitter in prop::collection::vec(-1.0f64..1.0, 35),
        spread in 0.0f64..0.15,
        v in 0.0f64..0.06,
        z in 0.0f64..6.0,
    ) {
        let sweep = Sweep::new(
            DEFAULT_SIZES
                .iter()
                .enumerate()
                .map(|(i, &n)| {
                    let ln_n = (n as f64).ln();
                    let base = 1000.0 * (alpha * ln_n + gamma * ln_n.ln()).exp();
                    let steps = (0..seeds)
                        .map(|j| (base * (spread * jitter[i * 5 + j]).exp()).round().max(1.0) as u64)
                        .collect();
                    SizeObservation::completed(n, steps)
                })
                .collect(),
        ).expect("valid");
        let exact = c_with(NoiseModel::Exact).fit(&sweep).expect("fits");
        let refuses_or_agrees = |verdict: &ClassVerdict| {
            *verdict == exact.verdict || *verdict == ClassVerdict::TooNoisy
        };
        let with = |noise_z: f64, noise_model: NoiseModel| {
            ClassSelect { noise_z, noise_model, ..ClassSelect::default() }
                .fit(&sweep)
                .expect("fits")
        };

        let from_seeds = with(z, NoiseModel::FromSeeds);
        let silent = with(0.0, NoiseModel::FromSeeds);
        if from_seeds.noise.is_some() {
            prop_assert!(refuses_or_agrees(&from_seeds.verdict), "{from_seeds:?} vs {exact:?}");
            prop_assert_eq!(&silent.verdict, &exact.verdict);
        } else {
            prop_assert_eq!(&from_seeds.verdict, &ClassVerdict::NoiseUnknown);
            prop_assert_eq!(&silent.verdict, &ClassVerdict::NoiseUnknown);
        }

        let known = with(z, NoiseModel::Known(v));
        prop_assert!(refuses_or_agrees(&known.verdict), "{known:?} vs {exact:?}");
        prop_assert_eq!(&with(0.0, NoiseModel::Known(v)).verdict, &exact.verdict);
        let louder = with(z, NoiseModel::Known(2.0 * v));
        prop_assert!(refuses_or_agrees(&louder.verdict), "{louder:?} vs {exact:?}");
        if known.verdict == ClassVerdict::TooNoisy {
            prop_assert_eq!(&louder.verdict, &ClassVerdict::TooNoisy);
        }
    }
}
