use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::*;
use crate::state::{
    CostClass, CostVerdict, FailureKind, FrontierMember, LibraryEntry, ProgramCost, SizeCost,
    SizeFailure,
};
use crate::verify::{shipped_prelude, Verifier};

/// The fingerprint of the prelude the pilot run (bead dreamcounter-b1a2a8)
/// was made with. Its programs name prelude functions by position, so they
/// mean the same only under this prelude.
const PILOT_PRELUDE: u64 = 1_500_720_720_873_173_848;

fn entry(arity: u32, term: &str) -> LibraryEntry {
    LibraryEntry {
        arity,
        term: term.to_owned(),
        name: None,
        doc: None,
    }
}

/// The pilot run's library after iteration 1: entries 0 to 2 joined when
/// iteration 0 ended, 3 to 5 when iteration 1 did.
fn pilot_library() -> Vec<LibraryEntry> {
    vec![
        entry(2, "(lam (lam (mod (add $0 $1) 1000003)))"),
        entry(
            2,
            "(lam (lam (lam (let $0 (let (fst $0) (let (snd $1) (pair (add $1 (length \
             (lib_4294901765 $4 $0))) (cons $5 $0))))))))",
        ),
        entry(1, "(lam (add (length $0) 2))"),
        entry(
            1,
            "(lam (lam (lib_4294901767 (foldl (lam (lam (bfix (lib_2 $2) $3 (pair $1 $0)))) \
             nil $0))))",
        ),
        entry(1, "(lam (add $0 1))"),
        entry(2, "(lam (lam (lam (fst (foldr $1 (pair $2 nil) $0)))))"),
    ]
}

/// Frontier programs of the pilot run, as its journal holds them, with the
/// class estimator C should name.
const PILOT: [(&str, &str, CostClass); 6] = [
    // Iteration 0's frontiers.
    (
        "sorted-order/inversions",
        "(lam (fst (foldr (lam (lib_1 $0 (lam (lt $0 $1)))) (pair 0 nil) $0)))",
        CostClass::Quadratic,
    ),
    (
        "sorted-order/pair-sums",
        "(lam (lam (fst (foldr (lam (lib_1 $0 (lam (le (add $1 $0) $2)))) (pair 0 nil) $1))))",
        CostClass::Quadratic,
    ),
    (
        "collapse/cancel",
        "(lam (lib_4294901767 (foldl (lam (lam (list_case $1 (lam (cons $1 nil)) (lam (lam \
         (if (int_eq (add $1 $2) 0) $0 (cons $2 $3))))))) nil $0)))",
        CostClass::Linear,
    ),
    (
        "collapse/merge",
        "(let (lam (lam (bfix (lib_2 $1) (lam (lam (let $0 (let (fst $0) (let (snd $1) \
         (list_case $1 (lam (cons $1 nil)) (lam (lam (if (int_eq $1 $2) ($6 (pair $0 (add $2 \
         1))) (cons $2 $3)))))))))) (pair $1 $0)))) (lam (lib_4294901767 (foldl $1 nil $0))))",
        CostClass::Linear,
    ),
    // Iteration 1's, written with the abstractions Stitch found then.
    (
        "sorted-order/inversions",
        "(lib_5 0 (lam (lib_1 $0 (lam (lt $0 $1)))))",
        CostClass::Quadratic,
    ),
    (
        "collapse/absorb",
        "(lib_3 (lam (lam (let $0 (let (fst $0) (let (snd $1) (list_case $1 (lam (cons $1 \
         nil)) (lam (lam (if (le 2 (sub $1 $2)) ($6 (pair $0 (mod (add $1 $2) 10))) (cons $2 \
         $3)))))))))))",
        CostClass::Linear,
    ),
];

fn never() -> Limits<'static> {
    Limits {
        deadline: Instant::now() + Duration::from_secs(3600),
        cancel: None,
    }
}

fn verdict(cost: &ProgramCost) -> &CostVerdict {
    match cost {
        ProgramCost::Measured { verdict, .. } => verdict,
        other => panic!("not measured: {other:?}"),
    }
}

#[test]
fn the_pilot_runs_frontier_programs_get_the_expected_classes() {
    assert_eq!(
        shipped_prelude(),
        PILOT_PRELUDE,
        "the prelude changed, so the pilot's programs name other functions"
    );
    let v = Verifier::new(&pilot_library()).unwrap();
    for (task, program, expected) in PILOT {
        let cost = measure_program(&v, task, program, &mut || false, "");
        assert_eq!(
            verdict(&cost),
            &CostVerdict::Class(expected),
            "{task}: {program}\n{cost:?}"
        );
        let ProgramCost::Measured { sizes, .. } = &cost else {
            unreachable!()
        };
        let plan = fuel_plan(task).unwrap();
        assert_eq!(sizes.len(), plan.sizes().len());
        for (size, planned) in sizes.iter().zip(plan.sizes()) {
            assert_eq!((size.n, size.fuel), (planned.n, planned.fuel));
            assert_eq!(
                size.steps.len() as u64,
                plan.seeds(),
                "{task} at {}",
                size.n
            );
            assert!(size.failed.is_none());
            let mut sorted = size.steps.clone();
            sorted.sort_unstable();
            // The lower median, as dc_tasks::growth::median_steps takes it.
            assert_eq!(size.median, Some(sorted[(sorted.len() - 1) / 2]));
        }
    }
}

#[test]
fn a_program_that_is_wrong_on_the_sweep_is_incorrect_not_measured() {
    // A wrong program that fails its examples is refused, so this one is
    // right below 64 elements, where the examples are, and wrong from 64.
    let v = Verifier::new(&[]).unwrap();
    let good = PILOT[2].1;
    let wrong = format!("(lam (if (lt (length $0) 64) ({good} $0) nil))");
    let cost = measure_program(&v, "collapse/cancel", &wrong, &mut || false, "");
    let ProgramCost::Incorrect { sizes, n, seed } = &cost else {
        panic!("{cost:?}");
    };
    assert_eq!((*n, *seed), (64, 0), "{sizes:?}");
    assert!(sizes[0].median.is_some());
}

#[test]
fn a_program_that_fails_its_examples_or_does_not_expand_is_refused() {
    let v = Verifier::new(&[]).unwrap();
    for program in ["(lam $0)", "(lam (lib_7 $0))", "(lam"] {
        let cost = measure_program(&v, "collapse/cancel", program, &mut || false, "");
        assert!(
            matches!(cost, ProgramCost::Refused { .. }),
            "{program}: {cost:?}"
        );
    }
    let cost = measure_program(&v, "no/such", PILOT[2].1, &mut || false, "");
    assert!(matches!(cost, ProgramCost::Refused { .. }), "{cost:?}");
}

fn kept(task: &str, programs: &[&str]) -> BTreeMap<String, Vec<String>> {
    BTreeMap::from([(
        task.to_owned(),
        programs.iter().map(|p| (*p).to_owned()).collect(),
    )])
}

#[test]
fn measuring_stops_at_the_deadline_and_on_cancel_and_keeps_every_program() {
    let programs = [
        PILOT[2].1,
        "(lam (lib_4294901767 (lib_4294901767 (lib_4294901767 $0))))",
    ];
    let past = Limits {
        deadline: Instant::now(),
        cancel: None,
    };
    let out = measure_frontiers(
        &[],
        &kept("collapse/cancel", &programs),
        &BTreeMap::new(),
        past,
    );
    let members = &out["collapse/cancel"];
    assert_eq!(
        members
            .iter()
            .map(|m| m.program.as_str())
            .collect::<Vec<_>>(),
        programs
    );
    for m in members {
        assert_eq!(
            m.cost,
            Some(ProgramCost::NotMeasured {
                reason: OUT_OF_TIME.to_owned()
            })
        );
    }
    let cancel = AtomicBool::new(true);
    let cancelled = Limits {
        deadline: Instant::now() + Duration::from_secs(3600),
        cancel: Some(&cancel),
    };
    let out = measure_frontiers(
        &[],
        &kept("collapse/cancel", &programs),
        &BTreeMap::new(),
        cancelled,
    );
    assert!(out["collapse/cancel"].iter().all(|m| m.cost
        == Some(ProgramCost::NotMeasured {
            reason: CANCELLED.to_owned()
        })));
}

#[test]
fn a_deadline_inside_a_sweep_stops_it() {
    let v = Verifier::new(&[]).unwrap();
    let mut runs = 0;
    let cost = measure_program(
        &v,
        "collapse/cancel",
        PILOT[2].1,
        &mut || {
            runs += 1;
            runs > 7
        },
        "stopped",
    );
    assert_eq!(
        cost,
        ProgramCost::NotMeasured {
            reason: "stopped".to_owned()
        }
    );
    // Asked before the examples, then before each of the sweep's runs: the
    // examples and six runs, and none after it says stop.
    assert_eq!(runs, 8);

    // Asked before the examples first: stopped there, nothing runs. This
    // program fails its examples, so it would be refused had they run.
    let mut asked = 0;
    let cost = measure_program(
        &v,
        "collapse/cancel",
        "(lam $0)",
        &mut || {
            asked += 1;
            true
        },
        "stopped",
    );
    assert_eq!(
        cost,
        ProgramCost::NotMeasured {
            reason: "stopped".to_owned()
        }
    );
    assert_eq!(asked, 1);
}

#[test]
fn a_task_with_no_fuel_plan_refuses_its_programs_without_measuring_them() {
    let before = started();
    let tried = plans_tried();
    let out = measure_frontiers(
        &[],
        &kept("no/such", &[PILOT[2].1, PILOT[3].1]),
        &BTreeMap::new(),
        never(),
    );
    for m in &out["no/such"] {
        assert!(
            matches!(&m.cost, Some(ProgramCost::Refused { reason }) if reason.contains("no development task")),
            "{m:?}"
        );
    }
    assert_eq!(started(), before);
    assert_eq!(
        plans_tried(),
        tried + 1,
        "the plan was tried once in the call"
    );

    // After the deadline, no plan is tried: the programs are not measured.
    let tried = plans_tried();
    let past = Limits {
        deadline: Instant::now(),
        cancel: None,
    };
    let out = measure_frontiers(&[], &kept("no/such", &[PILOT[2].1]), &BTreeMap::new(), past);
    assert_eq!(
        out["no/such"][0].cost,
        Some(ProgramCost::NotMeasured {
            reason: OUT_OF_TIME.to_owned()
        })
    );
    assert_eq!(plans_tried(), tried);
}

#[test]
fn a_member_kept_from_the_previous_frontier_keeps_only_a_cost_that_measuring_gives_again() {
    let task = "collapse/cancel";
    // Costs no measuring gives, so that keeping them is seen: no sizes.
    let marked = ProgramCost::Incorrect {
        sizes: Vec::new(),
        n: 0,
        seed: 99,
    };
    let measured = ProgramCost::Measured {
        sizes: Vec::new(),
        verdict: CostVerdict::Class(CostClass::Cubic),
    };
    // Reversing twice: it fails cancel's examples, so only its kept cost
    // can make it measured.
    let twice = "(lam (lib_4294901767 (lib_4294901767 $0)))";
    let previous = BTreeMap::from([(
        task.to_owned(),
        vec![
            FrontierMember {
                program: PILOT[2].1.to_owned(),
                cost: Some(marked.clone()),
            },
            FrontierMember {
                program: "(lam (lib_4294901767 (lib_4294901767 (lib_4294901767 $0))))".to_owned(),
                cost: Some(ProgramCost::NotMeasured {
                    reason: OUT_OF_TIME.to_owned(),
                }),
            },
            FrontierMember {
                program: twice.to_owned(),
                cost: Some(measured.clone()),
            },
            FrontierMember {
                program: "(lam (lib_4294901767 $0))".to_owned(),
                cost: Some(ProgramCost::Refused {
                    reason: "a failure that need not recur".to_owned(),
                }),
            },
        ],
    )]);
    let programs = [
        PILOT[2].1,
        twice,
        "(lam (lib_4294901767 (lib_4294901767 (lib_4294901767 $0))))",
        "(lam (lib_4294901767 $0))",
    ];
    let out = measure_frontiers(&[], &kept(task, &programs), &previous, never());
    let members = &out[task];
    assert_eq!(members[0].cost, Some(marked));
    assert_eq!(members[1].cost, Some(measured));
    // Measured again: they were not measured, or were refused, before.
    // Reversing is not cancel's collapse, so they fail the examples.
    for m in &members[2..] {
        assert!(
            matches!(&m.cost, Some(ProgramCost::Refused { reason }) if reason.contains("checks")),
            "{m:?}"
        );
    }
}

fn measured(sizes: Vec<SizeCost>, verdict: CostVerdict) -> ProgramCost {
    ProgramCost::Measured { sizes, verdict }
}

fn size(n: u64, failed: Option<FailureKind>) -> SizeCost {
    SizeCost {
        n,
        fuel: 1000,
        steps: if failed.is_some() {
            Vec::new()
        } else {
            vec![10, 11, 12]
        },
        median: if failed.is_some() { None } else { Some(11) },
        failed: failed.map(|kind| SizeFailure { seed: 0, kind }),
    }
}

#[test]
fn a_stored_cost_gives_the_growth_the_objective_scores() {
    use dc_driver::objective::{Growth, Refusal};
    use dc_tasks::growth::GrowthClass;

    let done = || vec![size(16, None), size(32, None)];
    let best = CostClass::Linear;
    let cases = [
        (
            measured(done(), CostVerdict::Class(CostClass::Cubic)),
            Growth::Fitted(GrowthClass::Cubic),
        ),
        (
            measured(done(), CostVerdict::TooNoisy { best }),
            Growth::Refused(Refusal::TooNoisy),
        ),
        (
            measured(done(), CostVerdict::NoFit { best }),
            Growth::Refused(Refusal::NoFit),
        ),
        (
            measured(done(), CostVerdict::NoiseUnknown { best }),
            Growth::Refused(Refusal::NoiseUnknown),
        ),
        (
            measured(
                vec![size(16, None), size(32, Some(FailureKind::Timeout))],
                CostVerdict::NotFitted {
                    reason: "fit window has incomplete sizes".to_owned(),
                },
            ),
            Growth::TimedOut,
        ),
        (
            measured(
                done(),
                CostVerdict::NotFitted {
                    reason: "median step count at n = 16 is 0".to_owned(),
                },
            ),
            Growth::Refused(Refusal::NotFitted),
        ),
        // A timeout below the window does not stop C naming a class.
        (
            measured(
                vec![size(16, Some(FailureKind::Timeout)), size(32, None)],
                CostVerdict::Class(CostClass::Linear),
            ),
            Growth::Fitted(GrowthClass::Linear),
        ),
        (
            ProgramCost::Incorrect {
                sizes: done(),
                n: 16,
                seed: 0,
            },
            Growth::Incorrect,
        ),
        (
            ProgramCost::Refused {
                reason: "no plan".to_owned(),
            },
            Growth::NotMeasured,
        ),
        (
            ProgramCost::NotMeasured {
                reason: OUT_OF_TIME.to_owned(),
            },
            Growth::NotMeasured,
        ),
    ];
    for (cost, expected) in &cases {
        assert_eq!(growth(Some(cost)), *expected, "{cost:?}");
    }
    assert_eq!(growth(None), Growth::NotMeasured);

    // Every class maps to the class of the same name.
    let classes = [
        (CostClass::Constant, GrowthClass::Constant),
        (CostClass::Log, GrowthClass::Log),
        (CostClass::Linear, GrowthClass::Linear),
        (CostClass::NLogN, GrowthClass::NLogN),
        (CostClass::Quadratic, GrowthClass::Quadratic),
        (CostClass::Cubic, GrowthClass::Cubic),
        (CostClass::Exponential, GrowthClass::Exponential),
    ];
    for (stored, class) in classes {
        assert_eq!(
            growth(Some(&measured(done(), CostVerdict::Class(stored)))),
            Growth::Fitted(class)
        );
    }
}

#[test]
fn a_correct_program_slower_than_its_fuel_is_timed_out_for_the_objective() {
    use dc_driver::objective::{member_costs, Growth, Member, TIMEOUT_COST};
    use dc_tasks::growth::GrowthClass;

    let v = Verifier::new(&pilot_library()).unwrap();
    // The linear collapse/cancel program, after a cubic computation whose
    // result it drops: the quadratic inversions count of the list, once for
    // each element. The fuel is ten times the reference's steps, so the
    // large sizes run out of it.
    let good = PILOT[2].1;
    let inversions = "(fst (foldr (lam (lib_1 $0 (lam (lt $0 $1)))) (pair 0 nil) $2))";
    let slow = format!("(lam (let (foldl (lam (lam (add $1 {inversions}))) 0 $0) ({good} $1)))");
    let cost = measure_program(&v, "collapse/cancel", &slow, &mut || false, "");
    let ProgramCost::Measured { sizes, verdict } = &cost else {
        panic!("{cost:?}");
    };
    assert!(
        matches!(verdict, CostVerdict::NotFitted { .. }),
        "{verdict:?}"
    );
    assert!(
        sizes.iter().any(|s| s
            .failed
            .as_ref()
            .is_some_and(|f| f.kind == FailureKind::Timeout)),
        "{sizes:?}"
    );
    assert_eq!(growth(Some(&cost)), Growth::TimedOut);

    let fast = measure_program(&v, "collapse/cancel", good, &mut || false, "");
    assert_eq!(growth(Some(&fast)), Growth::Fitted(GrowthClass::Linear));
    let costs = member_costs(&[
        Member {
            dl: 1,
            growth: growth(Some(&cost)),
        },
        Member {
            dl: 1,
            growth: growth(Some(&fast)),
        },
    ]);
    assert_eq!(costs[0].map(|c| c.c), Some(TIMEOUT_COST));
    assert_eq!(costs[1].map(|c| c.c), Some(2));
}
