use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::*;
use crate::state::{CostClass, CostVerdict, FrontierMember, LibraryEntry, ProgramCost};
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
            assert_eq!(size.median, Some(sorted[sorted.len() / 2]));
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
    assert_eq!(
        runs, 8,
        "stop is asked before each run, and no run after it says stop"
    );
}

#[test]
fn a_member_kept_from_the_previous_frontier_keeps_its_cost_unless_not_measured() {
    let task = "collapse/cancel";
    let marked = ProgramCost::Refused {
        reason: "a cost from before".to_owned(),
    };
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
        ],
    )]);
    let programs = [
        PILOT[2].1,
        "(lam (lib_4294901767 (lib_4294901767 (lib_4294901767 $0))))",
    ];
    let out = measure_frontiers(&[], &kept(task, &programs), &previous, never());
    let members = &out[task];
    assert_eq!(members[0].cost, Some(marked));
    // Measured again: it was not measured before. Reversing three times
    // is not cancel's collapse, so it fails the examples.
    assert!(matches!(
        members[1].cost,
        Some(ProgramCost::Refused { ref reason }) if reason.contains("checks")
    ));
}
