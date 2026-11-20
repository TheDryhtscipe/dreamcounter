use super::*;

use std::fs;

use serde_json::{json, Value};

/// One task in one iteration: batches, programs, compiled, typed, passed.
type Row = (&'static str, u64, u64, u64, u64, u64);

/// By iteration and task, each group's count and growth as JSON.
type Growth = Vec<(u64, &'static str, Vec<(u64, Value)>)>;

struct RunData {
    run: String,
    seed: u64,
    library_learning: bool,
    programs_per_batch: u64,
    frontier_size: u64,
    /// By iteration, each task's row.
    iterations: Vec<(u64, Vec<Row>)>,
    /// By iteration and task, each group's count and growth as JSON.
    growth: Option<Growth>,
    foreign: u64,
    /// Submissions waiting to be checked, in every iteration.
    pending_checks: u64,
    /// Whether the last iteration has not finished.
    unfinished: bool,
}

const TASKS: [&str; 2] = ["chains/count", "chains/heaviest"];

fn run(run: &str, seed: u64, library: bool, iterations: Vec<(u64, Vec<Row>)>) -> RunData {
    RunData {
        run: run.to_owned(),
        seed,
        library_learning: library,
        programs_per_batch: 1,
        frontier_size: 6,
        iterations,
        growth: None,
        foreign: 0,
        pending_checks: 0,
        unfinished: false,
    }
}

fn write(dir: &Path, r: &RunData) {
    fs::create_dir_all(dir).unwrap();
    let last = r.iterations.iter().map(|(i, _)| *i).max().unwrap_or(0);
    let status = json!({
        "run": r.run, "arm": "blind", "seed": r.seed, "proposals_per_task": 6,
        "programs_per_batch": r.programs_per_batch, "lease_seconds": 1800,
        "max_unchecked_per_batch": 3, "frontier_size": r.frontier_size,
        "stitch": if r.library_learning { json!({"iterations": 3}) } else { Value::Null },
        "iteration": if r.unfinished { last } else { last + 1 },
        "tasks": TASKS.iter().map(|t| json!({"name": t, "frontier_size": 1})).collect::<Vec<_>>(),
        "library_size": 0, "autodoc": true, "library_learning": r.library_learning,
        "max_memory_mib": 512, "compared_with": null, "approach_hints": false,
        "show_received": true, "unnamed": 0,
        "log_counts": {"compression": 0, "note": 0, "recovery": 0},
        "batches_issued": 0, "current": {}
    });
    fs::write(dir.join("status.json"), status.to_string()).unwrap();
    let iterations: Vec<Value> = r
        .iterations
        .iter()
        .map(|(i, rows)| {
            let passed: u64 = rows.iter().map(|r| r.5).sum();
            json!({
                "iteration": i, "quota_in": true, "pending_checks": r.pending_checks, "tasks_solved": 0,
                "submissions": 0, "late": 1, "repeats": 0, "foreign": r.foreign,
                "refused_repeats": 0, "refused_foreign": 0, "payload_bytes": 0,
                "submission_bytes": 0, "models": [],
                "library_uptake": {"passed": passed, "using_library": passed / 2, "entries": []},
                "tasks": rows.iter().map(|(n, b, p, c, t, s)| json!({
                    "name": n, "batches": b, "received": p, "programs": p, "compiled": c,
                    "typed": t, "passed": s, "solved": *s > 0
                })).collect::<Vec<_>>()
            })
        })
        .collect();
    fs::write(
        dir.join("proposal_stats.json"),
        json!({"run": r.run, "iterations": iterations}).to_string(),
    )
    .unwrap();
    if let Some(growth) = &r.growth {
        let mut by_iter: BTreeMap<u64, Vec<Value>> = BTreeMap::new();
        for (i, task, groups) in growth {
            let passed: u64 = groups.iter().map(|g| g.0).sum();
            by_iter.entry(*i).or_default().push(json!({
                "task": task, "passed": passed,
                "groups": groups.iter().map(|(n, g)| json!({
                    "program": "(lam $0)", "count": n, "growth": g
                })).collect::<Vec<_>>()
            }));
        }
        let its: Vec<Value> = by_iter
            .into_iter()
            .map(|(i, tasks)| json!({"iteration": i, "tasks": tasks}))
            .collect();
        fs::write(
            dir.join("diversity.json"),
            json!({"run": r.run, "iterations": its}).to_string(),
        )
        .unwrap();
    }
}

fn spec(metric: Metric, pairs: &[(u64, &str, &str)]) -> Spec {
    Spec {
        name: "test".to_owned(),
        metric,
        tasks: TASKS.iter().map(|t| (*t).to_owned()).collect(),
        counted_iterations: vec![1, 2],
        check_iteration: Some(0),
        smallest_effect: 0.10,
        alpha: 0.05,
        arms: Arms {
            test: "library".to_owned(),
            control: "baseline".to_owned(),
        },
        arm_settings: vec!["library_learning".to_owned(), "stitch".to_owned()],
        arm_values: ArmValues::default(),
        planned_pairs: None,
        allow_unfinished: false,
        worse_than_n_tasks: Vec::new(),
        pairs: pairs
            .iter()
            .map(|(seed, t, c)| PairSpec {
                seed: *seed,
                test: RunSpec {
                    records: PathBuf::from(t),
                    logs: Vec::new(),
                },
                control: RunSpec {
                    records: PathBuf::from(c),
                    logs: Vec::new(),
                },
            })
            .collect(),
    }
}

/// Six pairs. In iterations 1 and 2 the library arm passes `4 + k` of 12
/// programs over both tasks in pair `k`, the baseline 4: differences
/// k / 12 for k = 1..6. In iteration 0 both pass 6 of 12.
fn six_pairs(dir: &Path) -> Spec {
    let mut pairs = Vec::new();
    for k in 1..=6u64 {
        let seed = 7000 + k;
        let it0: Vec<Row> = vec![
            ("chains/count", 6, 6, 6, 6, 3),
            ("chains/heaviest", 6, 6, 6, 6, 3),
        ];
        // Library arm: over iterations 1 and 2, 24 issued, 8 + 2k passed.
        let lib = |extra: u64| -> Vec<Row> {
            vec![
                ("chains/count", 6, 6, 6, 6, 3 + extra),
                ("chains/heaviest", 6, 5, 4, 4, 1),
            ]
        };
        // Baseline: over iterations 1 and 2, 24 issued, 8 passed. One batch
        // a task got no submission, one program did not compile.
        let base: Vec<Row> = vec![
            ("chains/count", 6, 6, 6, 6, 3),
            ("chains/heaviest", 6, 5, 4, 4, 1),
        ];
        let t = format!("lib-{seed}");
        let c = format!("base-{seed}");
        write(
            &dir.join(&t),
            &run(
                &t,
                seed,
                true,
                vec![(0, it0.clone()), (1, lib(k)), (2, lib(k))],
            ),
        );
        let mut b = run(
            &c,
            seed,
            false,
            vec![(0, it0), (1, base.clone()), (2, base)],
        );
        b.foreign = 2;
        write(&dir.join(&c), &b);
        pairs.push((seed, t, c));
    }
    let refs: Vec<(u64, &str, &str)> = pairs
        .iter()
        .map(|(s, t, c)| (*s, t.as_str(), c.as_str()))
        .collect();
    spec(Metric::PassRate, &refs)
}

#[test]
fn the_pass_rate_is_over_batches_issued_in_the_counted_iterations() {
    let dir = tempfile::tempdir().unwrap();
    let spec = six_pairs(dir.path());
    let report = analyse(&spec, dir.path()).unwrap();
    let m = &report.primary.metric;
    for (k, p) in (1..=6u64).zip(&m.pairs) {
        // 24 issued over two tasks and two iterations; iteration 0 left out.
        assert_eq!(p.test, Some((8 + 2 * k) as f64 / 24.0), "pair {k}");
        // The baseline's no-submission batches count as fails: 8 of 24, not
        // 8 of the 22 programs that came in.
        assert_eq!(p.control, Some(8.0 / 24.0), "pair {k}");
        let d = p.difference.unwrap();
        assert!((d - k as f64 / 12.0).abs() < 1e-12, "pair {k}: {d}");
    }
    let t = m.test.as_ref().unwrap();
    assert!((t.mean - 21.0 / 72.0).abs() < 1e-12, "{}", t.mean);
    // Six differences of one sign: p = 2/64.
    assert_eq!(t.p, 2.0 / 64.0);
    assert_eq!(report.primary.call, Call::Positive);
    // The baseline's failures, by stage, over both tasks and counted
    // iterations of six runs. Each iteration of a run: 1 not submitted and
    // 1 not compiled (heaviest), and 3 + 3 failed examples, of 12 issued.
    let f = report.failures[1];
    assert_eq!(
        (
            f.issued,
            f.not_submitted,
            f.not_compiled,
            f.not_typed,
            f.failed_examples,
            f.passed
        ),
        (144, 12, 12, 0, 72, 48)
    );
}

#[test]
fn iteration_zero_is_a_check_not_data() {
    let dir = tempfile::tempdir().unwrap();
    let spec = six_pairs(dir.path());
    let report = analyse(&spec, dir.path()).unwrap();
    let check = report.check.as_ref().unwrap();
    assert_eq!(check.iteration, 0);
    for p in &check.metric.pairs {
        assert_eq!((p.test, p.control), (Some(0.5), Some(0.5)));
    }
    assert!(!check.differs);
    assert_eq!(check.metric.test.as_ref().unwrap().p, 1.0);
    assert!(report
        .warnings
        .iter()
        .all(|w| !w.contains("iteration 0 differs")));
}

#[test]
fn a_check_iteration_that_differs_is_flagged() {
    let dir = tempfile::tempdir().unwrap();
    let mut pairs = Vec::new();
    for k in 1..=6u64 {
        let seed = 100 + k;
        let rows = |passed: u64| -> Vec<Row> {
            vec![
                ("chains/count", 6, 6, 6, 6, passed),
                ("chains/heaviest", 6, 6, 6, 6, passed),
            ]
        };
        let t = format!("t{k}");
        let c = format!("c{k}");
        write(
            &dir.path().join(&t),
            &run(
                &t,
                seed,
                true,
                vec![(0, rows(5)), (1, rows(3)), (2, rows(3))],
            ),
        );
        write(
            &dir.path().join(&c),
            &run(
                &c,
                seed,
                false,
                vec![(0, rows(1)), (1, rows(3)), (2, rows(3))],
            ),
        );
        pairs.push((seed, t, c));
    }
    let refs: Vec<(u64, &str, &str)> = pairs
        .iter()
        .map(|(s, t, c)| (*s, t.as_str(), c.as_str()))
        .collect();
    let report = analyse(&spec(Metric::PassRate, &refs), dir.path()).unwrap();
    let check = report.check.as_ref().unwrap();
    assert!(check.differs);
    assert!(report
        .warnings
        .iter()
        .any(|w| w.contains("iteration 0 differs")));
    // The counted iterations are equal: a null result, as the interval is
    // the point 0.
    assert_eq!(report.primary.metric.test.as_ref().unwrap().p, 1.0);
    assert_eq!(report.primary.call, Call::Null);
}

#[test]
fn foreign_submissions_are_reported_not_counted() {
    let dir = tempfile::tempdir().unwrap();
    let spec = six_pairs(dir.path());
    let report = analyse(&spec, dir.path()).unwrap();
    let base = report.runs.iter().find(|r| r.run == "base-7001").unwrap();
    assert!(base.iterations.iter().all(|i| i.foreign == 2));
    assert_eq!(base.iterations.len(), 3);
    assert!(!base.iterations[0].counted && base.iterations[1].counted);
}

#[test]
fn class_ranks_follow_d27() {
    assert_eq!(class_rank(CostClass::Constant), 0);
    assert_eq!(class_rank(CostClass::Linear), 2);
    assert_eq!(class_rank(CostClass::NLogN), 3);
    assert_eq!(class_rank(CostClass::Quadratic), 4);
    assert_eq!(class_rank(CostClass::Exponential), 6);
}

#[test]
fn the_cheap_rate_counts_only_fitted_classes_up_to_n_log_n() {
    let dir = tempfile::tempdir().unwrap();
    let class = |c: &str| json!({"verdict": {"class": c}});
    let mut pairs = Vec::new();
    for k in 1..=6u64 {
        let seed = 5000 + k;
        let rows: Vec<Row> = vec![
            ("chains/count", 6, 6, 6, 6, 6),
            ("chains/heaviest", 6, 6, 6, 6, 0),
        ];
        let its = vec![(0, rows.clone()), (1, rows.clone()), (2, rows)];
        // Iterations 1 and 2: the test arm has k cheap programs in each
        // (classes n log n and n), the control 1. The rest are n², too
        // noisy, not fitted (a timeout), refused by the sweep, or
        // incorrect: none is cheap.
        let groups = |cheap: u64| -> Vec<(u64, Value)> {
            let mut g = vec![(1, class("n log n"))];
            if cheap > 1 {
                g.push((cheap - 1, class("n")));
            }
            let rest = 6 - cheap;
            let fill = [
                class("n^2"),
                json!({"verdict": {"too_noisy": {"best": "n"}}}),
                json!({"verdict": {"not_fitted": {"reason": "fit window has incomplete sizes: timed out [16, 32]"}}}),
                json!({"refused": {"reason": "over the memory limit"}}),
                json!("incorrect"),
            ];
            for (j, f) in fill.iter().enumerate() {
                let n = rest / 5 + u64::from((j as u64) < rest % 5);
                if n > 0 {
                    g.push((n, f.clone()));
                }
            }
            g
        };
        let growth = |cheap: u64| {
            Some(
                [0u64, 1, 2]
                    .iter()
                    .flat_map(|i| {
                        [
                            (*i, "chains/count", groups(if *i == 0 { 1 } else { cheap })),
                            (*i, "chains/heaviest", Vec::new()),
                        ]
                    })
                    .collect(),
            )
        };
        let t = format!("lam-{seed}");
        let c = format!("zero-{seed}");
        let mut rt = run(&t, seed, true, its.clone());
        rt.growth = growth(k);
        let mut rc = run(&c, seed, true, its);
        rc.growth = growth(1);
        write(&dir.path().join(&t), &rt);
        write(&dir.path().join(&c), &rc);
        pairs.push((seed, t, c));
    }
    let refs: Vec<(u64, &str, &str)> = pairs
        .iter()
        .map(|(s, t, c)| (*s, t.as_str(), c.as_str()))
        .collect();
    let mut s = spec(Metric::CheapRate, &refs);
    s.arm_settings = Vec::new();
    s.smallest_effect = 0.05;
    s.worse_than_n_tasks = vec!["chains/count".to_owned()];
    let report = analyse(&s, dir.path()).unwrap();
    for (k, p) in (1..=6u64).zip(&report.primary.metric.pairs) {
        // 2k cheap of 24 issued over both tasks and both counted iterations.
        assert_eq!(p.test, Some((2 * k) as f64 / 24.0), "pair {k}");
        assert_eq!(p.control, Some(2.0 / 24.0), "pair {k}");
    }
    // Pair 1 has difference 0, so not every difference has one sign.
    let t = report.primary.metric.test.as_ref().unwrap();
    assert!(t.p > 2.0 / 64.0, "{}", t.p);
    let per_passed = report
        .exploratory
        .iter()
        .find(|c| c.name == "cheap programs per passed program")
        .unwrap();
    // 12 passed over the counted iterations (heaviest passes none).
    assert_eq!(per_passed.pairs[2].test, Some(6.0 / 12.0));
    let worse = report
        .exploratory
        .iter()
        .find(|c| {
            c.name
                .starts_with("share of passed programs with a class worse than n")
        })
        .unwrap();
    // Pair 1's control, in each counted iteration: the n log n program and
    // the one n² program are worse than n; the 4 without a fitted class do
    // not count either way.
    assert_eq!(worse.pairs[0].control, Some(4.0 / 12.0));
    assert!(report
        .not_computed
        .iter()
        .any(|n| n.contains("dreamcounter-211560")));
}

#[test]
fn the_cheap_rate_needs_growth() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = vec![
        ("chains/count", 6, 6, 6, 6, 6),
        ("chains/heaviest", 6, 6, 6, 6, 0),
    ];
    let its = vec![(0, rows.clone()), (1, rows.clone()), (2, rows)];
    write(&dir.path().join("a"), &run("a", 1, true, its.clone()));
    write(&dir.path().join("b"), &run("b", 1, true, its));
    let err = analyse(&spec(Metric::CheapRate, &[(1, "a", "b")]), dir.path()).unwrap_err();
    assert!(err.0.contains("no diversity.json"), "{err}");
}

#[test]
fn a_run_short_of_a_counted_iteration_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = vec![
        ("chains/count", 6, 6, 6, 6, 6),
        ("chains/heaviest", 6, 6, 6, 6, 0),
    ];
    write(
        &dir.path().join("a"),
        &run("a", 1, true, vec![(0, rows.clone()), (1, rows.clone())]),
    );
    write(
        &dir.path().join("b"),
        &run(
            "b",
            1,
            false,
            vec![(0, rows.clone()), (1, rows.clone()), (2, rows)],
        ),
    );
    let err = analyse(&spec(Metric::PassRate, &[(1, "a", "b")]), dir.path()).unwrap_err();
    assert!(err.0.contains("has not reached iteration 2"), "{err}");
}

#[test]
fn settings_seeds_and_arm_settings_are_checked_within_each_pair() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = vec![
        ("chains/count", 6, 6, 6, 6, 3),
        ("chains/heaviest", 6, 6, 6, 6, 3),
    ];
    let its = vec![(0, rows.clone()), (1, rows.clone()), (2, rows)];
    let mut a = run("a", 1, true, its.clone());
    a.frontier_size = 4;
    // Both runs learn a library, though the arms should differ in it; and
    // the control has another seed than its pair.
    let b = run("b", 2, true, its);
    write(&dir.path().join("a"), &a);
    write(&dir.path().join("b"), &b);
    let report = analyse(&spec(Metric::PassRate, &[(1, "a", "b")]), dir.path()).unwrap();
    let d = report.deviations.join("\n");
    assert!(
        d.contains("setting frontier_size differs: 4 in a, 6 in b"),
        "{d}"
    );
    assert!(
        d.contains("setting library_learning is the same in both runs"),
        "{d}"
    );
    assert!(d.contains("the baseline run b has seed 2"), "{d}");
    assert!(d.contains("setting stitch is the same in both runs"), "{d}");
    assert!(!d.contains("setting show_received"), "{d}");
    // One pair: the test cannot reject, and the interval has no ends.
    assert_eq!(report.primary.call, Call::Inconclusive);
}

#[test]
fn several_programs_per_batch_scale_the_issued_count_and_warn() {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Row> = vec![
        ("chains/count", 2, 6, 6, 6, 3),
        ("chains/heaviest", 2, 6, 6, 6, 3),
    ];
    let its = vec![(0, rows.clone()), (1, rows.clone()), (2, rows)];
    let mut a = run("a", 1, true, its.clone());
    a.programs_per_batch = 3;
    let mut b = run("b", 1, false, its);
    b.programs_per_batch = 3;
    write(&dir.path().join("a"), &a);
    write(&dir.path().join("b"), &b);
    let report = analyse(&spec(Metric::PassRate, &[(1, "a", "b")]), dir.path()).unwrap();
    assert_eq!(report.primary.metric.pairs[0].test, Some(12.0 / 24.0));
    assert!(report
        .warnings
        .iter()
        .any(|w| w.contains("3 programs per batch")));
}

#[test]
fn a_spec_reads_from_json_and_refuses_unknown_fields() {
    let text = r#"{
        "name": "library versus baseline, chain family",
        "metric": "pass_rate",
        "tasks": ["chains/count"],
        "counted_iterations": [1, 2, 3],
        "check_iteration": 0,
        "smallest_effect": 0.1,
        "arms": {"test": "library", "control": "baseline"},
        "arm_settings": ["library_learning"],
        "pairs": [{"seed": 7001, "test": {"records": "r/a", "logs": ["l/a"]},
                   "control": {"records": "r/b"}}]
    }"#;
    let s: Spec = serde_json::from_str(text).unwrap();
    assert_eq!(
        (s.metric, s.alpha, s.check_iteration),
        (Metric::PassRate, 0.05, Some(0))
    );
    assert_eq!(s.pairs[0].test.logs, [PathBuf::from("l/a")]);
    let bad = text.replace("\"smallest_effect\"", "\"smallest\"");
    assert!(serde_json::from_str::<Spec>(&bad).is_err());
}

/// Rows where each task passes `passed` of 6 in each of iterations 0 to 2.
fn plain(passed: u64) -> Vec<(u64, Vec<Row>)> {
    let rows: Vec<Row> = vec![
        ("chains/count", 6, 6, 6, 6, passed),
        ("chains/heaviest", 6, 6, 6, 6, passed),
    ];
    vec![(0, rows.clone()), (1, rows.clone()), (2, rows)]
}

/// Writes runs `t<k>` (library) and `c<k>` (baseline) for seeds
/// `first + k`, k from 0, and returns their pairs.
fn plain_pairs(dir: &Path, first: u64, n: u64) -> Vec<(u64, String, String)> {
    (0..n)
        .map(|k| {
            let seed = first + k;
            let (t, c) = (format!("t{seed}"), format!("c{seed}"));
            write(&dir.join(&t), &run(&t, seed, true, plain(4)));
            write(&dir.join(&c), &run(&c, seed, false, plain(3)));
            (seed, t, c)
        })
        .collect()
}

fn refs(pairs: &[(u64, String, String)]) -> Vec<(u64, &str, &str)> {
    pairs
        .iter()
        .map(|(s, t, c)| (*s, t.as_str(), c.as_str()))
        .collect()
}

#[test]
fn a_pair_with_its_arms_swapped_is_a_deviation() {
    let dir = tempfile::tempdir().unwrap();
    let mut pairs = plain_pairs(dir.path(), 7001, 2);
    // The second pair lists its baseline run as the arm under test.
    let (s, t, c) = pairs.pop().unwrap();
    pairs.push((s, c, t));
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.arm_values
        .test
        .insert("library_learning".to_owned(), json!(true));
    sp.arm_values
        .control
        .insert("library_learning".to_owned(), json!(false));
    let report = analyse(&sp, dir.path()).unwrap();
    let d = report.deviations.join("\n");
    assert!(
        d.contains(
            "pair 7002: the library run c7002 has library_learning false, but its arm has true"
        ),
        "{d}"
    );
    assert!(!d.contains("pair 7001"), "{d}");
}

#[test]
fn seeds_out_of_order_or_too_few_pairs_are_deviations() {
    let dir = tempfile::tempdir().unwrap();
    let mut pairs = plain_pairs(dir.path(), 7001, 3);
    // Pair 7002 dropped: 7001, 7003.
    pairs.remove(1);
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.planned_pairs = Some(12);
    let report = analyse(&sp, dir.path()).unwrap();
    let d = report.deviations.join("\n");
    assert!(d.contains("not consecutive from 7001"), "{d}");
    assert!(
        d.contains("the spec has 2 pairs, but 12 were planned"),
        "{d}"
    );
    // In order and as planned: neither.
    let pairs = plain_pairs(dir.path(), 8001, 3);
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.planned_pairs = Some(3);
    let report = analyse(&sp, dir.path()).unwrap();
    assert!(report.deviations.is_empty(), "{:?}", report.deviations);
}

#[test]
fn a_listed_twice_task_iteration_or_run_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let pairs = plain_pairs(dir.path(), 7001, 2);
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.tasks.push("chains/count".to_owned());
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(err.0.contains("task chains/count is listed twice"), "{err}");
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.counted_iterations.push(1);
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(err.0.contains("iteration 1 is counted twice"), "{err}");
    let mut sp = spec(Metric::PassRate, &refs(&pairs));
    sp.pairs[1].control.records = PathBuf::from("c7001");
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(
        err.0.contains("run c7001 is in more than one place"),
        "{err}"
    );
}

#[test]
fn an_unfinished_iteration_or_pending_checks_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = run("a", 1, true, plain(3));
    a.unfinished = true;
    write(&dir.path().join("a"), &a);
    write(&dir.path().join("b"), &run("b", 1, false, plain(3)));
    let mut sp = spec(Metric::PassRate, &[(1, "a", "b")]);
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(
        err.0.contains("run a iteration 2 has not finished"),
        "{err}"
    );
    // Allowed for checking the script, with a warning.
    sp.allow_unfinished = true;
    let report = analyse(&sp, dir.path()).unwrap();
    assert!(report
        .warnings
        .iter()
        .any(|w| w.contains("run a iteration 2 has not finished")));
    let mut b = run("b", 1, false, plain(3));
    b.pending_checks = 1;
    write(&dir.path().join("b"), &b);
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(err.0.contains("waiting to be checked"), "{err}");
}

/// A cheap-rate pair whose test run's count task has `groups` in every
/// iteration (6 passed), and whose control has 6 programs of class n².
fn cheap_pair(dir: &Path, groups: Vec<(u64, Value)>, passed: u64) -> Spec {
    let rows: Vec<Row> = vec![
        ("chains/count", 6, 6, 6, 6, passed),
        ("chains/heaviest", 6, 6, 6, 6, 0),
    ];
    let its = vec![(0, rows.clone()), (1, rows.clone()), (2, rows)];
    let growth = |g: &Vec<(u64, Value)>| -> Option<Growth> {
        Some(
            [0u64, 1, 2]
                .iter()
                .flat_map(|i| {
                    [
                        (*i, "chains/count", g.clone()),
                        (*i, "chains/heaviest", Vec::new()),
                    ]
                })
                .collect(),
        )
    };
    let mut a = run("a", 1, true, its.clone());
    a.growth = growth(&groups);
    let mut b = run("b", 1, true, its);
    b.growth = growth(&vec![(passed, json!({"verdict": {"class": "n^2"}}))]);
    write(&dir.join("a"), &a);
    write(&dir.join("b"), &b);
    let mut sp = spec(Metric::CheapRate, &[(1, "a", "b")]);
    sp.arm_settings = Vec::new();
    sp
}

#[test]
fn the_cheap_rate_refuses_programs_that_were_not_measured() {
    let dir = tempfile::tempdir().unwrap();
    let sp = cheap_pair(
        dir.path(),
        vec![
            (3, json!({"verdict": {"class": "n log n"}})),
            (
                3,
                json!({"not_measured": {"reason": "the time for measuring ran out"}}),
            ),
        ],
        6,
    );
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(err.0.contains("was not measured"), "{err}");
    // The pass rate does not need growth, so it still runs.
    let mut sp = sp;
    sp.metric = Metric::PassRate;
    assert!(analyse(&sp, dir.path()).is_ok());
}

#[test]
fn the_cheap_rate_refuses_diversity_that_disagrees_with_the_stats() {
    let dir = tempfile::tempdir().unwrap();
    // 4 programs in the groups, 6 passed in proposal_stats.
    let sp = cheap_pair(
        dir.path(),
        vec![(4, json!({"verdict": {"class": "n log n"}}))],
        6,
    );
    let err = analyse(&sp, dir.path()).unwrap_err();
    assert!(
        err.0
            .contains("diversity.json has 4 passed programs, proposal_stats.json 6"),
        "{err}"
    );
}

#[test]
fn programs_the_sweep_refused_are_not_cheap_and_warned_about() {
    let dir = tempfile::tempdir().unwrap();
    let sp = cheap_pair(
        dir.path(),
        vec![
            (4, json!({"verdict": {"class": "n"}})),
            (2, json!({"refused": {"reason": "over the memory limit"}})),
        ],
        6,
    );
    let report = analyse(&sp, dir.path()).unwrap();
    // 4 cheap of 12 issued in each counted iteration.
    assert_eq!(report.primary.metric.pairs[0].test, Some(8.0 / 24.0));
    assert!(report
        .warnings
        .iter()
        .any(|w| w.contains("could not be run on the growth sweep")));
}
