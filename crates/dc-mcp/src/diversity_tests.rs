//! How varied passed programs are, on runs filled through the proposer's
//! calls.

use std::time::Duration;

use dc_driver::InProcess;
use tempfile::TempDir;

use super::*;
use crate::iterate::finish_iteration;
use crate::proposer::{check_pending, next_batch, submit};
use crate::state::{Arm, LibraryEntry, RunConfig, StitchSettings};
use crate::verify::{shipped_prelude, VerifierCache};

const MERGE: &str = "collapse/merge";
const CANCEL: &str = "collapse/cancel";

fn setup(quota: u32) -> (TempDir, Store, RunId) {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let config = RunConfig {
        max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
        autodoc: false,
        arm: Arm::Blind,
        library_learning: true,
        seed: 3,
        proposals_per_task: quota,
        programs_per_batch: 1,
        lease_ms: 1000,
        max_unchecked_per_batch: 3,
        frontier_size: 2,
        stitch: StitchSettings::default(),
        prelude: shipped_prelude(),
        compared_with: None,
    };
    store
        .create_run(&run, config, vec![MERGE.to_owned(), CANCEL.to_owned()])
        .unwrap();
    (dir, store, run)
}

/// The source of `task`'s reference number `which`, naive first.
fn reference(task: &str, which: usize) -> String {
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == task)
        .unwrap();
    let entry = v.references[which].entry;
    format!("{}\nlet solve = {entry}\n", v.source)
}

/// Hands out every open batch, each to a worker of its own, sends
/// `program(task, n)` for the `n`-th batch of `task`, and checks them.
fn fill(store: &Store, run: &RunId, now_ms: u64, program: impl Fn(&str, usize) -> String) {
    let verifiers = VerifierCache::default();
    let mut per_task: BTreeMap<String, usize> = BTreeMap::new();
    loop {
        let worker = format!("w{}-{now_ms}", per_task.values().sum::<usize>());
        let reply = next_batch(store, run, &worker, now_ms, &verifiers).unwrap();
        let Some(payload) = reply.batch else {
            break;
        };
        let task = store
            .read(run, |s| s.batch(payload.batch).unwrap().task.clone())
            .unwrap();
        let n = per_task.entry(task.clone()).or_insert(0);
        let text = program(&task, *n);
        *n += 1;
        submit(
            store,
            run,
            payload.batch,
            &worker,
            &payload.token,
            vec![text],
            None,
            now_ms,
        )
        .unwrap();
    }
    check_pending(store, run, &VerifierCache::default()).unwrap();
}

/// The task's line of an iteration's report.
fn line<'a>(report: &'a DiversityReport, iteration: u64, task: &str) -> &'a TaskDiversity {
    report
        .iterations
        .iter()
        .find(|i| i.iteration == iteration)
        .unwrap()
        .tasks
        .iter()
        .find(|t| t.task == task)
        .unwrap()
}

fn report_of(
    store: &Store,
    run: &RunId,
    iteration: Option<u64>,
    growth: Option<GrowthLimits<'_>>,
) -> DiversityReport {
    let passed = passed_programs(store, run).unwrap();
    store
        .read(run, |s| report(run, s, &passed, iteration, growth))
        .unwrap()
        .unwrap()
}

/// A merge program from the pilot run, with its variables named `xs`, `st`
/// and `x`, or `ys`, `acc` and `y`: the same core term, written twice.
fn pilot_merge(renamed: bool) -> String {
    let text = "let solve xs =\n  let n = length xs + 10 in\n  \
                let push = bfix n (fun self p -> let (st, x) = p in match st with \
                [] -> [x] | h :: t -> if h = x then self (t, x + 1) else x :: st) in\n  \
                reverse (foldl (fun st x -> push (st, x)) [] xs)";
    if renamed {
        text.replace("xs", "ys")
            .replace("st", "acc")
            .replace("(acc, x)", "(acc, y)")
            .replace("[x]", "[y]")
            .replace("h = x", "h = y")
            .replace("x + 1", "y + 1")
            .replace("x :: acc", "y :: acc")
            .replace("fun acc x ->", "fun acc y ->")
    } else {
        text.to_owned()
    }
}

#[test]
fn the_journal_gives_each_iterations_passed_programs_by_task() {
    let (_dir, store, run) = setup(3);
    // Merge gets the naive reference, the good one, then the naive one
    // again. Cancel gets one program that fails and two that do not compile.
    fill(&store, &run, 0, |task, n| match (task, n) {
        (MERGE, 0 | 2) => reference(MERGE, 0),
        (MERGE, _) => reference(MERGE, 1),
        (_, 0) => "let solve xs = xs".to_owned(),
        _ => "let solve = )".to_owned(),
    });
    let passed = passed_programs(&store, &run).unwrap();
    assert_eq!(passed.len(), 1, "{passed:?}");
    let by_task = &passed[&0];
    assert_eq!(by_task.keys().collect::<Vec<_>>(), [MERGE]);
    let merge = &by_task[MERGE];
    assert_eq!(merge.len(), 3);
    assert_eq!(merge[0], merge[2]);
    assert_ne!(merge[0], merge[1]);
}

#[test]
fn programs_that_differ_only_in_names_count_once_and_two_algorithms_twice() {
    let (_dir, store, run) = setup(3);
    fill(&store, &run, 0, |task, n| match (task, n) {
        (MERGE, 0) => pilot_merge(false),
        (MERGE, 1) => pilot_merge(true),
        (MERGE, _) => reference(MERGE, 1),
        _ => reference(CANCEL, 0),
    });
    let r = report_of(&store, &run, Some(0), None);
    let merge = line(&r, 0, MERGE);
    assert_eq!(merge.passed, 3);
    assert_eq!(merge.distinct_programs, 2);
    assert_eq!(merge.distinct_expanded, 2);
    // The most common group comes first.
    assert_eq!(merge.groups[0].count, 2);
    assert_eq!(merge.groups[1].count, 1);
    assert!(merge.groups.iter().all(|g| g.growth.is_none()));
    assert_eq!(merge.distinct_classes, None);
    assert_eq!(merge.groups_without_class, None);
    // Two of the three pairs are the same term, the third is not.
    let d = merge.mean_distance.unwrap();
    assert!(d > 0.0 && d < 2.0 / 3.0, "{d}");

    let cancel = line(&r, 0, CANCEL);
    assert_eq!(cancel.passed, 3);
    assert_eq!(cancel.distinct_programs, 1);
    assert_eq!(cancel.distinct_expanded, 1);
    assert_eq!(cancel.mean_distance, Some(0.0));

    let s = &r.iterations[0].summary;
    assert_eq!(s.tasks, 2);
    assert_eq!(s.passed, 6);
    assert_eq!(s.distinct_expanded, 3);
    assert_eq!(s.tasks_all_same, 1, "cancel's three programs are one");
    let mean = (d + 0.0) / 2.0;
    assert!((s.mean_distance.unwrap() - mean).abs() < 1e-12);
    assert_eq!(s.tasks_with_two_classes, None);
}

#[test]
fn a_call_of_a_library_function_and_the_function_written_out_expand_alike() {
    let library = [LibraryEntry {
        arity: 1,
        term: "(lam (add $0 1))".to_owned(),
        name: None,
        doc: None,
    }];
    let mut verifier = Verifier::new(&library, dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
    let programs = [
        "(lam (lib_0 $0))".to_owned(),
        "(lam (add $0 1))".to_owned(),
        "(lam (add $0 2))".to_owned(),
    ];
    let none: Option<fn(&str, Option<TermId>) -> ProgramCost> = None;
    let t = task_diversity("t", &programs, &mut verifier, none);
    assert_eq!(t.passed, 3);
    assert_eq!(t.distinct_programs, 3);
    assert_eq!(t.distinct_expanded, 2);
    assert_eq!(t.groups[0].program, programs[0]);
    assert_eq!(t.groups[0].count, 2);
    assert_eq!(t.groups[1].program, programs[2]);

    // Groups of the same count are in the order they first passed.
    let tied = [programs[2].clone(), programs[0].clone()];
    let t = task_diversity("t", &tied, &mut verifier, none);
    let order: Vec<&str> = t.groups.iter().map(|g| g.program.as_str()).collect();
    assert_eq!(order, [tied[0].as_str(), tied[1].as_str()]);
    let tied = [programs[0].clone(), programs[2].clone()];
    let t = task_diversity("t", &tied, &mut verifier, none);
    let order: Vec<&str> = t.groups.iter().map(|g| g.program.as_str()).collect();
    assert_eq!(order, [tied[0].as_str(), tied[1].as_str()]);
}

#[test]
fn the_distance_is_zero_for_one_term_one_for_nothing_shared_and_a_mean_over_pairs() {
    let a = "(lam (add $0 1))".to_owned();
    let b = "(lam (mul $0 2))".to_owned();
    let c = "(lam (add $0 2))".to_owned();
    assert_eq!(mean_distance(std::slice::from_ref(&a)), None);
    assert_eq!(mean_distance(&[a.clone(), a.clone()]), Some(0.0));
    // Two different atoms are not the same program.
    assert_eq!(
        mean_distance(&["lib_0".to_owned(), "lib_1".to_owned()]),
        Some(1.0)
    );
    assert_eq!(
        mean_distance(&["lib_0".to_owned(), "lib_0".to_owned()]),
        Some(0.0)
    );
    // a and b share no compound subterm.
    assert_eq!(mean_distance(&[a.clone(), b.clone()]), Some(1.0));
    // a and c share `(add $0)`, as applications are curried.
    let ac = mean_distance(&[a.clone(), c.clone()]).unwrap();
    assert!(ac > 0.0 && ac < 1.0, "{ac}");
    // Three programs: the mean of the three pairs.
    let three = mean_distance(&[a.clone(), a.clone(), c.clone()]).unwrap();
    assert!((three - 2.0 * ac / 3.0).abs() < 1e-12, "{three} {ac}");
}

#[test]
fn growth_names_the_classes_of_two_algorithms() {
    let (_dir, store, run) = setup(2);
    fill(&store, &run, 0, reference);
    let never = AtomicBool::new(false);
    let limits = GrowthLimits {
        deadline: Instant::now() + Duration::from_secs(600),
        cancel: &never,
    };
    let r = report_of(&store, &run, Some(0), Some(limits));
    let merge = line(&r, 0, MERGE);
    assert_eq!(merge.distinct_expanded, 2);
    let classes: Vec<Option<CostClass>> = merge
        .groups
        .iter()
        .map(|g| g.growth.as_ref().unwrap().class())
        .collect();
    assert!(classes.contains(&Some(CostClass::Quadratic)), "{merge:?}");
    assert!(classes.contains(&Some(CostClass::Linear)), "{merge:?}");
    assert_eq!(merge.distinct_classes, Some(2));
    assert_eq!(merge.groups_without_class, Some(0));
    assert_eq!(r.iterations[0].summary.tasks_with_two_classes, Some(2));
}

#[test]
fn past_the_deadline_a_program_is_not_measured_unless_the_frontier_measured_it() {
    let (_dir, store, run) = setup(2);
    fill(&store, &run, 0, reference);
    let never = AtomicBool::new(false);
    let past = GrowthLimits {
        deadline: Instant::now(),
        cancel: &never,
    };
    // Before the iteration ends, nothing has a cost.
    let before = report_of(&store, &run, Some(0), Some(past));
    for g in &line(&before, 0, MERGE).groups {
        assert_eq!(
            g.growth,
            Some(GrowthResult::NotMeasured {
                reason: OUT_OF_TIME.to_owned()
            })
        );
    }
    assert_eq!(line(&before, 0, MERGE).distinct_classes, Some(0));
    assert_eq!(line(&before, 0, MERGE).groups_without_class, Some(2));

    // Ending it keeps both programs of each task and measures them. Stitch
    // may rewrite them to call new entries; their costs still count.
    finish_iteration(&store, &run, &mut InProcess).unwrap();
    let after = report_of(&store, &run, Some(0), Some(past));
    let merge = line(&after, 0, MERGE);
    assert_eq!(merge.distinct_classes, Some(2), "{merge:?}");
}

#[test]
fn a_cancelled_call_records_why_it_stopped() {
    let (_dir, store, run) = setup(1);
    fill(&store, &run, 0, |task, _| reference(task, 0));
    let cancelled = AtomicBool::new(true);
    let stopped = GrowthLimits {
        deadline: Instant::now() + Duration::from_secs(600),
        cancel: &cancelled,
    };
    let r = report_of(&store, &run, Some(0), Some(stopped));
    assert_eq!(
        line(&r, 0, MERGE).groups[0].growth,
        Some(GrowthResult::NotMeasured {
            reason: CANCELLED.to_owned()
        })
    );
}

#[test]
fn each_program_counts_in_the_iteration_of_its_batch() {
    let (_dir, store, run) = setup(1);
    fill(&store, &run, 0, |task, _| reference(task, 0));
    finish_iteration(&store, &run, &mut InProcess).unwrap();
    // Iteration 0 gets the naive reference, iteration 1 the good one.
    fill(&store, &run, 1, |task, _| reference(task, 1));
    let r = report_of(&store, &run, None, None);
    assert_eq!(r.iterations.len(), 2);
    for (i, which) in [(0, 0), (1, 1)] {
        let merge = line(&r, i, MERGE);
        assert_eq!(merge.passed, 1, "iteration {i}");
        assert_eq!(merge.distinct_expanded, 1);
        let expected = store
            .read(&run, |s| {
                let mut v = Verifier::new(s.library(), s.config().max_memory).unwrap();
                let a = v.expand_stored(&merge.groups[0].program).unwrap();
                (a, v)
            })
            .unwrap();
        let (got, mut v) = expected;
        // The program is the reference given in that iteration, compiled.
        let mut check = v.clone();
        let family = crate::verify::task(MERGE).unwrap();
        let Check::Passed { program } = check.check(&*family, &reference(MERGE, which)) else {
            panic!("the reference passes");
        };
        assert_eq!(v.expand_stored(&program).unwrap(), got, "iteration {i}");
        // One program is not "all the same".
        assert_eq!(r.iterations[i as usize].summary.tasks_all_same, 0);
    }
}

#[test]
fn every_iteration_up_to_the_current_one_is_reported_and_a_later_one_refused() {
    let (_dir, store, run) = setup(1);
    fill(&store, &run, 0, |task, _| reference(task, 0));
    finish_iteration(&store, &run, &mut InProcess).unwrap();
    let all = report_of(&store, &run, None, None);
    let iterations: Vec<u64> = all.iterations.iter().map(|i| i.iteration).collect();
    assert_eq!(iterations, [0, 1]);
    assert_eq!(all.iterations[0].summary.passed, 2);
    assert_eq!(all.iterations[1].summary.passed, 0);
    assert_eq!(all.iterations[1].summary.tasks, 0);
    assert_eq!(all.iterations[1].summary.mean_distance, None);
    let passed = passed_programs(&store, &run).unwrap();
    let later = store
        .read(&run, |s| report(&run, s, &passed, Some(2), None))
        .unwrap()
        .unwrap_err();
    assert!(later.contains("has not started"), "{later}");
}
