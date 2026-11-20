use super::*;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The exact two-sided sign-flip p of whole-number differences, counted
/// with integer sums: an oracle that shares no code with the test.
fn exact_p(d: &[i64]) -> f64 {
    let observed: i64 = d.iter().sum::<i64>().abs();
    let n = 1u64 << d.len();
    let mut count = 0u64;
    for m in 0..n {
        let s: i64 = d
            .iter()
            .enumerate()
            .map(|(i, x)| if m >> i & 1 == 1 { -x } else { *x })
            .sum();
        if s.abs() >= observed {
            count += 1;
        }
    }
    count as f64 / n as f64
}

fn hundredths(d: &[i64]) -> Vec<f64> {
    d.iter().map(|x| *x as f64 / 100.0).collect()
}

#[test]
fn sign_flip_p_of_one_two_three_is_a_quarter() {
    // ±1 ±2 ±3 gives 6, 4, 2, 0, 0, -2, -4, -6; two of eight reach 6.
    assert_eq!(sign_flip_p(&[1.0, 2.0, 3.0]), Ok(0.25));
}

#[test]
fn six_differences_of_one_sign_give_the_smallest_p_d33_names() {
    let p = sign_flip_p(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    assert_eq!(p, 2.0 / 64.0);
    assert!(p <= 0.05);
    // With five pairs the smallest p is 0.0625, above 0.05.
    assert_eq!(sign_flip_p(&[0.1, 0.2, 0.3, 0.4, 0.5]), Ok(2.0 / 32.0));
}

#[test]
fn differences_that_cancel_give_p_one() {
    assert_eq!(sign_flip_p(&[1.0, -1.0]), Ok(1.0));
    assert_eq!(sign_flip_p(&[0.0, 0.0, 0.0]), Ok(1.0));
}

#[test]
fn sign_flip_p_matches_integer_counting_on_twelve_pairs() {
    let sets: [[i64; 12]; 3] = [
        [5, -2, 10, 7, 1, 12, 3, -4, 9, 6, 8, 2],
        [3, -3, 4, -4, 1, 0, 2, -2, 6, -1, 0, 5],
        [-10, -7, -12, -3, 2, -8, -1, -6, -9, 4, -5, -11],
    ];
    for d in sets {
        // Rates are fractions, so the differences are not whole numbers:
        // the tolerance on ties must still find the exact ties.
        let p = sign_flip_p(&hundredths(&d)).unwrap();
        assert_eq!(p, exact_p(&d), "{d:?}");
    }
}

#[test]
fn exact_ties_count_although_the_differences_round() {
    // 0.1 + 0.2 is not exactly 0.3 in floating point, but the pattern that
    // flips 0.3 alone ties the observed sum exactly in the reals:
    // |0.1 + 0.2 - 0.3| = 0, and the observed sum 0.1 + 0.2 + 0.3 - 0.6 = 0.
    let d = [0.1, 0.2, 0.3, -0.6];
    assert_eq!(sign_flip_p(&d), Ok(1.0));
    // Integer counting of the same differences in tenths.
    assert_eq!(exact_p(&[1, 2, 3, -6]), 1.0);
}

#[test]
fn the_test_refuses_what_it_cannot_enumerate() {
    assert_eq!(sign_flip_p(&[]), Err(TestError::Empty));
    assert_eq!(
        sign_flip_p(&[0.0; MAX_PAIRS + 1]),
        Err(TestError::TooMany(MAX_PAIRS + 1))
    );
    assert_eq!(sign_flip_p(&[0.1, f64::NAN]), Err(TestError::NotFinite));
}

fn shifted(d: &[f64], delta: f64) -> Vec<f64> {
    d.iter().map(|x| x - delta).collect()
}

#[test]
fn the_interval_ends_are_where_the_test_turns() {
    let sets: [[i64; 12]; 3] = [
        [5, -2, 10, 7, 1, 12, 3, -4, 9, 6, 8, 2],
        [3, -3, 4, -4, 1, 0, 2, -2, 6, -1, 0, 5],
        [-10, -7, -12, -3, 2, -8, -1, -6, -9, 4, -5, -11],
    ];
    for raw in sets {
        let d = hundredths(&raw);
        let ci = interval(&d, 0.05).unwrap();
        let (Some(lo), Some(hi)) = (ci.lower, ci.upper) else {
            panic!("12 pairs give a bounded interval: {ci:?}");
        };
        assert!(!ci.has_gaps, "{raw:?}: {ci:?}");
        let m = mean(&d);
        assert!(lo <= m && m <= hi, "{raw:?}: {lo} {m} {hi}");
        // Inside at the ends, outside just past them.
        assert!(sign_flip_p(&shifted(&d, lo)).unwrap() > 0.05, "{raw:?}");
        assert!(sign_flip_p(&shifted(&d, hi)).unwrap() > 0.05, "{raw:?}");
        assert!(
            sign_flip_p(&shifted(&d, lo - 1e-6)).unwrap() <= 0.05,
            "{raw:?}"
        );
        assert!(
            sign_flip_p(&shifted(&d, hi + 1e-6)).unwrap() <= 0.05,
            "{raw:?}"
        );
        // And every shift between them, on a fine grid, is inside.
        let steps = 400;
        for k in 0..=steps {
            let delta = lo + (hi - lo) * k as f64 / steps as f64;
            assert!(
                sign_flip_p(&shifted(&d, delta)).unwrap() > 0.05,
                "{raw:?} at {delta}"
            );
        }
    }
}

#[test]
fn the_interval_excludes_zero_exactly_when_the_test_rejects() {
    let sets: [[i64; 12]; 3] = [
        [5, -2, 10, 7, 1, 12, 3, -4, 9, 6, 8, 2],
        [3, -3, 4, -4, 1, 0, 2, -2, 6, -1, 0, 5],
        [-10, -7, -12, -3, 2, -8, -1, -6, -9, 4, -5, -11],
    ];
    let mut rejected = 0;
    for raw in sets {
        let d = hundredths(&raw);
        let p = sign_flip_p(&d).unwrap();
        let ci = interval(&d, 0.05).unwrap();
        let (Some(lo), Some(hi)) = (ci.lower, ci.upper) else {
            panic!("{ci:?}");
        };
        assert_eq!(
            p <= 0.05,
            !(lo <= 0.0 && 0.0 <= hi),
            "{raw:?}: p {p}, {ci:?}"
        );
        rejected += usize::from(p <= 0.05);
    }
    // The sets cover both outcomes, so the check is not vacuous.
    assert!((1..3).contains(&rejected), "{rejected}");
}

#[test]
fn five_pairs_give_an_interval_without_ends() {
    let ci = interval(&[0.1, 0.2, 0.3, 0.4, 0.5], 0.05).unwrap();
    assert_eq!((ci.lower, ci.upper), (None, None));
    assert!(!ci.inside(0.10));
}

#[test]
fn equal_differences_give_a_point_interval() {
    let ci = interval(&[0.1; 8], 0.05).unwrap();
    let (Some(lo), Some(hi)) = (ci.lower, ci.upper) else {
        panic!("{ci:?}");
    };
    assert!(
        (lo - 0.1).abs() < 1e-12 && (hi - 0.1).abs() < 1e-12,
        "{ci:?}"
    );
}

#[test]
fn calls_follow_d33() {
    let narrow = Interval {
        lower: Some(-0.04),
        upper: Some(0.06),
        has_gaps: false,
    };
    let wide = Interval {
        lower: Some(-0.04),
        upper: Some(0.16),
        has_gaps: false,
    };
    let open = Interval {
        lower: None,
        upper: None,
        has_gaps: false,
    };
    let high = Interval {
        lower: Some(0.02),
        upper: Some(0.2),
        has_gaps: false,
    };
    assert_eq!(call(0.11, 0.01, &high, 0.05, 0.10), Call::Positive);
    // p ≤ α, not p < α: 6 pairs of one sign give p = 0.03125, and the level
    // itself rejects too.
    assert_eq!(call(0.11, 0.05, &high, 0.05, 0.10), Call::Positive);
    assert_eq!(call(-0.11, 0.05, &high, 0.05, 0.10), Call::Negative);
    assert_eq!(call(0.01, 0.4, &narrow, 0.05, 0.10), Call::Null);
    assert_eq!(call(0.06, 0.2, &wide, 0.05, 0.10), Call::Inconclusive);
    assert_eq!(call(0.0, 1.0, &open, 0.05, 0.10), Call::Inconclusive);
    // The null band is each comparison's own: a86001 uses 0.05.
    assert_eq!(call(0.01, 0.4, &narrow, 0.05, 0.05), Call::Inconclusive);
    // An end on the band's edge is not inside it.
    let edge = Interval {
        lower: Some(-0.05),
        upper: Some(0.02),
        has_gaps: false,
    };
    assert_eq!(call(0.0, 0.9, &edge, 0.05, 0.05), Call::Inconclusive);
}

/// A standard normal draw (Box-Muller).
fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let u = |r: &mut ChaCha8Rng| ((r.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (a, b) = (u(rng), u(rng));
    (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
}

fn binomial(rng: &mut ChaCha8Rng, n: u64, p: f64) -> u64 {
    (0..n)
        .filter(|_| ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) < p)
        .count() as u64
}

/// Runs drawn from D33's simulation model: each run a shift, each task in a
/// run a further shift, both normal with spread `sigma`.
fn simulated(sigma: f64, runs: usize, issued: u64, seed: u64) -> Vec<Vec<Cell>> {
    let base = [1.1, 1.1, -1.1, 2.0];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut groups: Vec<Vec<Cell>> = vec![Vec::new(); base.len()];
    for _ in 0..runs {
        let shift = sigma * normal(&mut rng);
        for (t, b) in base.iter().enumerate() {
            let logit = b + shift + sigma * normal(&mut rng);
            let p = 1.0 / (1.0 + (-logit).exp());
            let passed = binomial(&mut rng, issued, p);
            groups[t].push(Cell { passed, issued });
        }
    }
    groups
}

#[test]
fn the_spread_estimate_recovers_a_known_sigma() {
    // 18 programs per task and run: 6 proposals in each of 3 counted
    // iterations, as in 70c43d.
    for (sigma, low, high) in [(0.3, 0.15, 0.45), (0.6, 0.45, 0.75)] {
        let s = logit_spread(&simulated(sigma, 400, 18, 7)).unwrap();
        assert!(
            low < s.sigma && s.sigma < high,
            "sigma {sigma}: estimated {s:?}"
        );
        assert_eq!((s.groups, s.cells), (4, 1600));
    }
    // With no spread the estimate is near 0, not the sampling noise.
    let s = logit_spread(&simulated(0.0, 400, 18, 8)).unwrap();
    assert!(s.sigma < 0.15, "{s:?}");
}

#[test]
fn the_spread_needs_two_runs_of_a_group() {
    let one = vec![vec![Cell {
        passed: 3,
        issued: 6,
    }]];
    assert_eq!(logit_spread(&one), None);
    let none_issued = vec![vec![
        Cell {
            passed: 0,
            issued: 0,
        },
        Cell {
            passed: 0,
            issued: 0,
        },
    ]];
    assert_eq!(logit_spread(&none_issued), None);
}

#[test]
fn mean_and_sd() {
    assert_eq!(mean(&[]), 0.0);
    assert_eq!(mean(&[1.0, 2.0, 3.0]), 2.0);
    assert_eq!(sd(&[1.0]), None);
    assert_eq!(sd(&[1.0, 2.0, 3.0]), Some(1.0));
}

#[test]
fn tiny_differences_are_not_all_ties() {
    // The tolerance on ties is relative: differences of 1e-12 are tested as
    // their size allows, not as all equal.
    let d: Vec<f64> = [1.0, 2.0, 3.0, 1.0, 2.0, 5.0]
        .iter()
        .map(|x| x * 1e-12)
        .collect();
    assert_eq!(sign_flip_p(&d), Ok(2.0 / 64.0));
}

#[test]
fn the_interval_scales_with_the_differences() {
    // The interval of differences scaled by k is the interval scaled by k,
    // at any scale: no tolerance in it is absolute.
    let raw = [5, -2, 10, 7, 1, 12, 3, -4, 9, 6, 8, 2];
    let base = interval(&hundredths(&raw), 0.05).unwrap();
    let (Some(lo), Some(hi)) = (base.lower, base.upper) else {
        panic!("{base:?}");
    };
    for k in [1e-14, 1e-9, 1e6, 1e12] {
        let d: Vec<f64> = hundredths(&raw).iter().map(|x| x * k).collect();
        let ci = interval(&d, 0.05).unwrap();
        let (Some(l), Some(h)) = (ci.lower, ci.upper) else {
            panic!("{k}: {ci:?}");
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(f64::MIN_POSITIVE);
        assert!(
            close(l, lo * k) && close(h, hi * k),
            "{k}: {ci:?} against {base:?}"
        );
    }
}

#[test]
fn sums_that_overflow_are_refused() {
    let d = [1e308; 6];
    assert_eq!(sign_flip_p(&d), Err(TestError::TooLarge));
    assert_eq!(interval(&d, 0.05), Err(TestError::TooLarge));
}

#[test]
fn differences_too_large_to_test_exactly_are_refused() {
    let d = [1.5e308, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(sign_flip_p(&d), Err(TestError::TooLarge));
    assert_eq!(interval(&d, 0.05), Err(TestError::TooLarge));
    assert_eq!(
        sign_flip_p(&[-8e307, 8e307, 1.0, 1.0, 1.0, 1.0]),
        Err(TestError::TooLarge)
    );
    // At the bound, the test runs.
    assert!(sign_flip_p(&[MAX_DIFFERENCE, 1.0]).is_ok());
}
