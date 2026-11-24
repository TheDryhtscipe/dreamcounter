use std::collections::BTreeMap;

use dc_tasks::growth::GrowthClass;

use super::*;

/// λ values above 0, from very small to very large.
const POSITIVE_LAMBDAS: [f64; 6] = [1e-9, 1e-3, 0.5, 1.0, 100.0, 1e9];

fn member(dl: u64, growth: Growth) -> Member {
    Member { dl, growth }
}

fn fitted(dl: u64, class: GrowthClass) -> Member {
    member(dl, Growth::Fitted(class))
}

fn too_noisy(dl: u64) -> Member {
    member(dl, Growth::Refused(Refusal::TooNoisy))
}

fn selected(members: &[Member], lambda: f64) -> Selection {
    select(members, lambda)
        .expect("lambda is valid")
        .expect("the task has a correct member")
}

#[test]
fn class_cost_is_the_position_in_order_of_growth() {
    let costs: Vec<u32> = GrowthClass::ALL.iter().map(|&c| class_cost(c)).collect();
    assert_eq!(costs, vec![0, 1, 2, 3, 4, 5, 6]);
}

#[test]
fn a_refused_member_gets_the_largest_fitted_cost_in_its_task_plus_one() {
    let members = [
        fitted(10, GrowthClass::Linear),
        too_noisy(10),
        fitted(10, GrowthClass::Quadratic),
        member(10, Growth::Refused(Refusal::NoFit)),
        member(10, Growth::NotMeasured),
    ];
    let costs = member_costs(&members);
    let c: Vec<u32> = costs.iter().map(|m| m.expect("correct").c).collect();
    assert_eq!(c, vec![2, 5, 4, 5, 5]);
    let causes: Vec<Option<Unfitted>> =
        costs.iter().map(|m| m.expect("correct").unfitted).collect();
    assert_eq!(
        causes,
        vec![
            None,
            Some(Unfitted::Refused(Refusal::TooNoisy)),
            None,
            Some(Unfitted::Refused(Refusal::NoFit)),
            Some(Unfitted::NotMeasured),
        ]
    );
}

#[test]
fn a_refused_member_of_a_task_with_no_fitted_member_gets_one_past_the_slowest_class() {
    let members = [
        too_noisy(10),
        member(10, Growth::NotMeasured),
        member(10, Growth::Refused(Refusal::NoiseUnknown)),
    ];
    for cost in member_costs(&members) {
        assert_eq!(cost.expect("correct").c, NO_FITTED_REFUSAL_COST);
    }
    assert_eq!(
        NO_FITTED_REFUSAL_COST,
        class_cost(GrowthClass::Exponential) + 1
    );
}

#[test]
fn an_incorrect_member_is_not_fitted_or_scored_and_a_timeout_gets_the_fixed_penalty() {
    let members = [
        member(10, Growth::Incorrect),
        member(10, Growth::TimedOut),
        fitted(10, GrowthClass::Exponential),
    ];
    let costs = member_costs(&members);
    assert_eq!(costs[0], None);
    assert_eq!(
        costs[1],
        Some(MemberCost {
            c: TIMEOUT_COST,
            unfitted: Some(Unfitted::TimedOut),
        })
    );
    // An incorrect member does not count as fitted for a refusal either.
    let costs = member_costs(&[member(10, Growth::Incorrect), too_noisy(10)]);
    assert_eq!(costs[1].expect("correct").c, NO_FITTED_REFUSAL_COST);
}

#[test]
fn the_timeout_penalty_exceeds_every_fitted_and_every_refusal_value() {
    let largest_fitted = GrowthClass::ALL
        .iter()
        .map(|&c| class_cost(c))
        .max()
        .expect("there are classes");
    assert_eq!(largest_fitted, 6);
    // A refusal is at most the largest fitted cost plus one, which is
    // NO_FITTED_REFUSAL_COST.
    assert_eq!(NO_FITTED_REFUSAL_COST, largest_fitted + 1);
    assert!(TIMEOUT_COST > largest_fitted + 1);
}

#[test]
fn a_task_with_one_fitted_and_one_refused_member_selects_the_fitted_one_at_every_positive_lambda() {
    for class in GrowthClass::ALL {
        // The refused member comes first, so a tie would select it.
        let members = [too_noisy(40), fitted(40, class)];
        for lambda in POSITIVE_LAMBDAS {
            let s = selected(&members, lambda);
            assert_eq!(s.index, 1, "class {class}, lambda {lambda}");
            assert_eq!(s.cost.unfitted, None);
            assert_eq!(s.cost.c, class_cost(class));
        }
    }
}

#[test]
fn a_timed_out_member_loses_to_a_refused_one_at_every_positive_lambda() {
    let tasks: [Vec<Member>; 3] = [
        vec![member(40, Growth::TimedOut), too_noisy(40)],
        vec![
            member(40, Growth::TimedOut),
            member(40, Growth::NotMeasured),
        ],
        // With a fitted member of the slowest class, the refusal is 7.
        vec![
            member(40, Growth::TimedOut),
            too_noisy(40),
            fitted(41, GrowthClass::Exponential),
        ],
    ];
    for members in &tasks {
        let costs = member_costs(members);
        assert!(costs[0].expect("correct").c > costs[1].expect("correct").c);
        for lambda in POSITIVE_LAMBDAS {
            let s = selected(members, lambda);
            // In the third task the fitted member wins at large λ; the
            // timed-out one never does.
            if members.len() == 2 || lambda < 1.0 {
                assert_eq!(s.index, 1, "{members:?} at lambda {lambda}");
            }
            assert_ne!(s.cost.unfitted, Some(Unfitted::TimedOut));
        }
    }
}

#[test]
fn a_timed_out_member_loses_to_every_fitted_member() {
    for class in GrowthClass::ALL {
        let members = [member(40, Growth::TimedOut), fitted(40, class)];
        for lambda in POSITIVE_LAMBDAS {
            assert_eq!(selected(&members, lambda).index, 1);
        }
    }
}

#[test]
fn at_lambda_zero_selection_is_by_description_length_with_the_earliest_on_a_tie() {
    let members = [
        fitted(50, GrowthClass::Linear),
        member(30, Growth::TimedOut),
        too_noisy(30),
    ];
    let s = selected(&members, 0.0);
    assert_eq!(s.index, 1);
    assert_eq!(s.score, 30.0);
    assert_eq!(s.cost.unfitted, Some(Unfitted::TimedOut));
}

#[test]
fn a_shorter_slower_member_wins_until_lambda_pays_for_the_difference() {
    // n² at 100, n at 110: the n program costs 2 less, so it wins once
    // 2λ > 10.
    let members = [
        fitted(100, GrowthClass::Quadratic),
        fitted(110, GrowthClass::Linear),
    ];
    assert_eq!(selected(&members, 4.9).index, 0);
    assert_eq!(selected(&members, 5.0).index, 0);
    assert_eq!(selected(&members, 5.1).index, 1);
}

#[test]
fn a_task_whose_members_are_all_unfitted_still_contributes_a_term() {
    let mut frontiers = BTreeMap::new();
    frontiers.insert(
        "all_unfitted".to_owned(),
        vec![member(70, Growth::TimedOut), too_noisy(80)],
    );
    frontiers.insert("fitted".to_owned(), vec![fitted(20, GrowthClass::Linear)]);
    let j = objective(1000, &frontiers, 2.0).expect("lambda is valid");
    let s = j.tasks.get("all_unfitted").expect("the task contributes");
    // 70 + 2·8 = 86 is less than 80 + 2·7 = 94: the timed-out member is
    // shorter by more than λ pays for.
    assert_eq!(s.index, 0);
    assert_eq!(s.score, 86.0);
    assert_eq!(s.cost.unfitted, Some(Unfitted::TimedOut));
    assert_eq!(j.total, 1000.0 + 86.0 + (20.0 + 2.0 * 2.0));
    assert_eq!(j.total_cost(), u64::from(TIMEOUT_COST) + 2);

    let mut refused_only = BTreeMap::new();
    refused_only.insert("t".to_owned(), vec![too_noisy(80)]);
    let j = objective(0, &refused_only, 1.0).expect("lambda is valid");
    let s = j.tasks.get("t").expect("the task contributes");
    assert_eq!(s.cost.c, NO_FITTED_REFUSAL_COST);
    assert_eq!(s.cost.unfitted, Some(Unfitted::Refused(Refusal::TooNoisy)));
    assert_eq!(j.total, 80.0 + 7.0);
}

#[test]
fn an_incorrect_member_is_never_selected_and_a_task_of_only_incorrect_members_has_no_term() {
    let members = [member(1, Growth::Incorrect), fitted(90, GrowthClass::Cubic)];
    assert_eq!(selected(&members, 0.0).index, 1);

    let mut frontiers = BTreeMap::new();
    frontiers.insert("wrong".to_owned(), vec![member(1, Growth::Incorrect)]);
    frontiers.insert("empty".to_owned(), Vec::new());
    let j = objective(5, &frontiers, 1.0).expect("lambda is valid");
    assert!(j.tasks.is_empty());
    assert_eq!(j.total, 5.0);
}

#[test]
fn lambda_must_be_finite_and_not_negative() {
    let frontiers = BTreeMap::from([("t".to_owned(), vec![too_noisy(1)])]);
    for lambda in [
        -1.0,
        -1e-12,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        MAX_LAMBDA * 2.0,
        f64::MAX,
    ] {
        assert!(matches!(
            objective(0, &frontiers, lambda),
            Err(ObjectiveError::BadLambda(_))
        ));
        assert!(matches!(
            select(&[too_noisy(1)], lambda),
            Err(ObjectiveError::BadLambda(_))
        ));
    }
}

#[test]
fn each_cause_has_a_log_line() {
    let lines: Vec<String> = [
        Unfitted::TimedOut,
        Unfitted::Refused(Refusal::TooNoisy),
        Unfitted::Refused(Refusal::NoFit),
        Unfitted::Refused(Refusal::NoiseUnknown),
        Unfitted::Refused(Refusal::NotFitted),
        Unfitted::NotMeasured,
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    let mut distinct = lines.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), lines.len());
}

#[test]
fn a_cost_difference_decides_a_tie_in_description_length_even_when_lambda_is_tiny() {
    // λ · ΔC is far below the spacing of f64 values near these lengths, so
    // the rounded scores are equal.
    for (dl, lambda) in [(40, 1e-15), (1_000_000, 1e-12), (u64::MAX / 2, 1e-3)] {
        let refused_first = [too_noisy(dl), fitted(dl, GrowthClass::Exponential)];
        assert_eq!(
            selected(&refused_first, lambda).index,
            1,
            "dl {dl}, lambda {lambda}"
        );
        let timeout_first = [member(dl, Growth::TimedOut), too_noisy(dl)];
        assert_eq!(
            selected(&timeout_first, lambda).index,
            1,
            "dl {dl}, lambda {lambda}"
        );
    }
}

#[test]
fn the_timeout_penalty_is_fixed_whatever_the_task_holds() {
    for others in [
        vec![],
        vec![fitted(10, GrowthClass::Constant)],
        vec![fitted(10, GrowthClass::Linear), too_noisy(10)],
        vec![member(10, Growth::Incorrect)],
    ] {
        let mut members = vec![member(10, Growth::TimedOut)];
        members.extend(others);
        let costs = member_costs(&members);
        assert_eq!(costs[0].expect("correct").c, TIMEOUT_COST, "{members:?}");
    }
}

#[test]
fn the_largest_lambda_keeps_j_finite() {
    let frontiers = BTreeMap::from([
        ("a".to_owned(), vec![member(u64::MAX, Growth::TimedOut)]),
        (
            "b".to_owned(),
            vec![fitted(u64::MAX, GrowthClass::Exponential)],
        ),
    ]);
    let j = objective(u64::MAX, &frontiers, MAX_LAMBDA).expect("lambda is valid");
    assert!(j.total.is_finite());
    // One step of C outweighs the largest gap in description length.
    let members = [
        fitted(u64::MAX, GrowthClass::Constant),
        fitted(0, GrowthClass::Log),
    ];
    assert_eq!(selected(&members, MAX_LAMBDA).index, 0);
}

#[test]
fn rank_at_lambda_zero_is_a_stable_sort_by_description_length_with_incorrect_members_last() {
    let members = [
        fitted(30, GrowthClass::Quadratic),
        member(10, Growth::Incorrect),
        fitted(20, GrowthClass::Linear),
        too_noisy(20),
        fitted(5, GrowthClass::Cubic),
    ];
    assert_eq!(rank(&members, 0.0).unwrap(), vec![4, 2, 3, 0, 1]);
}

#[test]
fn rank_puts_a_slower_member_after_a_longer_faster_one_once_lambda_pays_for_it() {
    let members = [
        fitted(10, GrowthClass::Quadratic),
        fitted(13, GrowthClass::Linear),
        fitted(11, GrowthClass::Cubic),
    ];
    // C is 4, 2 and 5. At λ = 1 the scores are 14, 15 and 16.
    assert_eq!(rank(&members, 1.0).unwrap(), vec![0, 1, 2]);
    // At λ = 2 they are 18, 17 and 21.
    assert_eq!(rank(&members, 2.0).unwrap(), vec![1, 0, 2]);
    // At a huge λ, C alone decides.
    assert_eq!(rank(&members, 1e9).unwrap(), vec![1, 0, 2]);
}

#[test]
fn the_first_member_rank_gives_is_the_one_select_gives() {
    let members = [
        fitted(40, GrowthClass::Quadratic),
        too_noisy(30),
        member(25, Growth::TimedOut),
        member(50, Growth::Incorrect),
        fitted(45, GrowthClass::Linear),
        member(30, Growth::NotMeasured),
    ];
    for lambda in std::iter::once(0.0).chain(POSITIVE_LAMBDAS) {
        let ranked = rank(&members, lambda).unwrap();
        assert_eq!(
            ranked[0],
            selected(&members, lambda).index,
            "lambda {lambda}"
        );
        let mut sorted = ranked.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..members.len()).collect::<Vec<_>>());
        assert_eq!(*ranked.last().unwrap(), 3, "the incorrect member is last");
    }
}

#[test]
fn rank_keeps_the_given_order_among_members_with_equal_scores() {
    let members = [
        fitted(12, GrowthClass::Linear),
        fitted(10, GrowthClass::Quadratic),
        fitted(12, GrowthClass::Linear),
        fitted(10, GrowthClass::Quadratic),
    ];
    // At λ = 1 every score is 14.
    assert_eq!(rank(&members, 1.0).unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn rank_refuses_a_bad_lambda() {
    assert_eq!(
        rank(&[fitted(1, GrowthClass::Linear)], -1.0),
        Err(ObjectiveError::BadLambda(-1.0))
    );
}

#[test]
fn a_lambda_is_finite_between_zero_and_the_largest_and_minus_zero_is_zero() {
    for bad in [f64::NAN, f64::INFINITY, -1.0, -1e-300, MAX_LAMBDA * 2.0] {
        assert!(Lambda::new(bad).is_err(), "{bad}");
    }
    let minus_zero = Lambda::new(-0.0).unwrap();
    assert!(minus_zero.is_zero());
    assert!(minus_zero.get().is_sign_positive());
    assert_eq!(minus_zero, Lambda::ZERO);
    assert_eq!(Lambda::default(), Lambda::ZERO);
    assert_eq!(Lambda::new(MAX_LAMBDA).unwrap().get(), MAX_LAMBDA);
    assert!(!Lambda::new(1e-300).unwrap().is_zero());
}

#[test]
fn a_lambda_is_written_as_a_number_and_a_bad_one_is_not_read() {
    let l = Lambda::new(0.25).unwrap();
    assert_eq!(serde_json::to_string(&l).unwrap(), "0.25");
    assert_eq!(serde_json::from_str::<Lambda>("0.25").unwrap(), l);
    assert_eq!(serde_json::from_str::<Lambda>("0").unwrap(), Lambda::ZERO);
    assert!(serde_json::from_str::<Lambda>("-1").is_err());
    assert!(serde_json::from_str::<Lambda>("1e30").is_err());
}

#[test]
fn an_unfitted_cause_is_written_in_snake_case() {
    let causes = [
        Unfitted::TimedOut,
        Unfitted::NotMeasured,
        Unfitted::Refused(Refusal::NoiseUnknown),
    ];
    assert_eq!(
        serde_json::to_string(&causes).unwrap(),
        r#"["timed_out","not_measured",{"refused":"noise_unknown"}]"#
    );
}
