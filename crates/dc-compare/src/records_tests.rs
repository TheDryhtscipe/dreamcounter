//! The mirrored types read what dc-mcp writes. Each test builds dc-mcp's own
//! report value, serializes it as the server does, and reads it back.

use super::*;

use dc_mcp::diversity as dv;
use dc_mcp::server as sv;

fn task_stats(name: &str, batches: u64, passed: u64) -> sv::TaskStats {
    sv::TaskStats {
        name: name.to_owned(),
        batches,
        received: batches,
        programs: batches,
        compiled: batches,
        typed: batches,
        passed,
        known_mistake: Some(2),
        solved: passed > 0,
    }
}

fn iteration_stats(iteration: u64) -> sv::IterationStats {
    sv::IterationStats {
        iteration,
        quota_in: true,
        pending_checks: 2,
        tasks_solved: 1,
        submissions: 7,
        late: 3,
        repeats: 4,
        foreign: 5,
        refused_repeats: 0,
        refused_foreign: 6,
        payload_bytes: 100,
        submission_bytes: 50,
        models: vec![sv::ModelStats {
            model: Some("claude-sonnet-5-5".to_owned()),
            submissions: 7,
            programs: 7,
        }],
        library_uptake: sv::UptakeStats {
            passed: 3,
            using_library: 2,
            entries: vec![sv::EntryUse {
                entry: 1,
                name: Some("pairs".to_owned()),
                programs: 2,
                tasks: vec!["chains/count".to_owned()],
            }],
        },
        tasks: vec![task_stats("chains/count", 6, 3)],
    }
}

#[test]
fn proposal_stats_read_back() {
    let report = sv::ProposalStats {
        run: "r".to_owned(),
        iterations: vec![iteration_stats(0), iteration_stats(1)],
    };
    let text = serde_json::to_string(&report).unwrap();
    let read: ProposalStats = serde_json::from_str(&text).unwrap();
    assert_eq!(read.run, "r");
    let it = &read.iterations[1];
    assert_eq!((it.iteration, it.quota_in, it.pending_checks), (1, true, 2));
    assert_eq!(
        (it.late, it.repeats, it.foreign, it.refused_foreign),
        (3, 4, 5, 6)
    );
    assert_eq!(
        (it.library_uptake.passed, it.library_uptake.using_library),
        (3, 2)
    );
    let e = &it.library_uptake.entries[0];
    assert_eq!(
        (e.entry, e.name.as_deref(), e.programs),
        (1, Some("pairs"), 2)
    );
    assert_eq!(e.tasks, ["chains/count"]);
    let t = &it.tasks[0];
    assert_eq!(
        (
            t.name.as_str(),
            t.batches,
            t.programs,
            t.compiled,
            t.typed,
            t.passed,
            t.solved
        ),
        ("chains/count", 6, 6, 6, 6, 3, true)
    );
    assert_eq!(t.known_mistake, Some(2));
}

#[test]
fn run_status_reads_back_and_keeps_its_settings() {
    let status = sv::RunStatus {
        run: "r".to_owned(),
        arm: "blind".to_owned(),
        seed: 7001,
        proposals_per_task: 6,
        programs_per_batch: 1,
        lease_seconds: 1800,
        max_unchecked_per_batch: 3,
        frontier_size: 6,
        stitch: Some(sv::StitchStatus {
            iterations: 3,
            max_arity: 2,
            wall_seconds: 600,
            memory_mib: 4096,
        }),
        iteration: 2,
        tasks: vec![sv::TaskStatus {
            name: "chains/count".to_owned(),
            frontier_size: 3,
        }],
        library_size: 4,
        autodoc: true,
        library_learning: true,
        max_memory_mib: 512,
        compared_with: None,
        approach_hints: false,
        show_received: true,
        refuse_bucketing: true,
        seed_entries: vec!["bucket_push_index".to_owned()],
        unnamed: 0,
        log_counts: sv::LogCounts {
            compression: 1,
            note: 0,
            recovery: 0,
        },
        batches_issued: 12,
        current: iteration_stats(2),
    };
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("status.json"),
        serde_json::to_string(&status).unwrap(),
    )
    .unwrap();
    let stats = sv::ProposalStats {
        run: "r".to_owned(),
        iterations: vec![iteration_stats(0)],
    };
    fs::write(
        dir.path().join("proposal_stats.json"),
        serde_json::to_string(&stats).unwrap(),
    )
    .unwrap();
    let records = RunRecords::read(dir.path(), &[]).unwrap();
    let s = &records.status;
    assert_eq!(
        (
            s.run.as_str(),
            s.seed,
            s.programs_per_batch,
            s.iteration,
            s.library_learning
        ),
        ("r", 7001, 1, 2, true)
    );
    assert_eq!(s.tasks[0].name, "chains/count");
    let settings = records.settings();
    // How the run was made is kept; where it is is not.
    for kept in [
        "arm",
        "proposals_per_task",
        "programs_per_batch",
        "lease_seconds",
        "frontier_size",
        "stitch",
        "autodoc",
        "library_learning",
        "max_memory_mib",
        "approach_hints",
        "show_received",
        "refuse_bucketing",
        "seed_entries",
    ] {
        assert!(settings.contains_key(kept), "{kept}");
    }
    for dropped in [
        "run",
        "seed",
        "iteration",
        "library_size",
        "current",
        "batches_issued",
    ] {
        assert!(!settings.contains_key(dropped), "{dropped}");
    }
}

#[test]
fn diversity_reads_back_with_each_kind_of_growth() {
    use dc_mcp::state::CostClass as C;
    let group = |count, growth| dv::ProgramGroup {
        program: "(lam $0)".to_owned(),
        count,
        growth: Some(growth),
    };
    let report = dv::DiversityReport {
        run: "r".to_owned(),
        iterations: vec![dv::IterationDiversity {
            iteration: 1,
            tasks: vec![dv::TaskDiversity {
                task: "sorted-order/inversions".to_owned(),
                passed: 9,
                distinct_programs: 5,
                distinct_expanded: 5,
                mean_distance: Some(0.5),
                distinct_classes: Some(2),
                groups_without_class: Some(3),
                groups: vec![
                    group(3, dv::GrowthResult::Verdict(CostVerdict::Class(C::NLogN))),
                    group(
                        2,
                        dv::GrowthResult::Verdict(CostVerdict::TooNoisy { best: C::Linear }),
                    ),
                    group(1, dv::GrowthResult::Incorrect),
                    group(
                        1,
                        dv::GrowthResult::NotMeasured {
                            reason: "out of time".to_owned(),
                        },
                    ),
                    group(
                        2,
                        dv::GrowthResult::Verdict(CostVerdict::NotFitted {
                            reason: "timed out [16, 32]".to_owned(),
                        }),
                    ),
                ],
            }],
            summary: dv::DiversitySummary {
                tasks: 1,
                passed: 9,
                distinct_expanded: 5,
                tasks_all_same: 0,
                mean_distance: Some(0.5),
                tasks_with_two_classes: Some(0),
            },
        }],
    };
    let text = serde_json::to_string(&report).unwrap();
    let read: DiversityReport = serde_json::from_str(&text).unwrap();
    let t = &read.iterations[0].tasks[0];
    assert_eq!((read.iterations[0].iteration, t.passed), (1, 9));
    let classes: Vec<Option<CostClass>> = t
        .groups
        .iter()
        .map(|g| g.growth.as_ref().and_then(GrowthResult::class))
        .collect();
    assert_eq!(classes, [Some(C::NLogN), None, None, None, None]);
    let counts: Vec<u64> = t.groups.iter().map(|g| g.count).collect();
    assert_eq!(counts, [3, 2, 1, 1, 2]);
}

#[test]
fn library_reads_back() {
    let report = sv::LibraryReport {
        run: "r".to_owned(),
        iteration: 3,
        entries: vec![sv::LibraryItem {
            index: 0,
            arity: 2,
            term: "(lam (lam $1))".to_owned(),
            name: Some("first".to_owned()),
            doc: Some("The first of two.".to_owned()),
            iteration_added: 1,
        }],
    };
    let read: LibraryReport =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    let e = &read.entries[0];
    assert_eq!(read.iteration, 3);
    assert_eq!(
        (e.index, e.arity, e.name.as_deref(), e.iteration_added),
        (0, 2, Some("first"), 1)
    );
    assert_eq!(e.doc.as_deref(), Some("The first of two."));
}

/// The shape of a dc-propose summary, from a real launch (probe2,
/// 2026-11-19), cut to two sessions.
const SUMMARY: &str = r#"{
  "launch": "p3f12ebfc",
  "run": "probe2",
  "model": "claude-sonnet-5-5",
  "state_dir": "/runs/state",
  "interrupted": false,
  "names_pending_at_start": 0,
  "sessions": [
    {"worker": "p3f12ebfc-0", "kind": "batch", "outcome": "submitted", "timed_out": false,
     "model": "claude-sonnet-5-5", "next_batch_calls": 1, "submit_calls": 1,
     "leased_batches": [2], "elapsed_ms": 19036, "batch": 2, "received": 1,
     "usage": {"input_tokens": 6, "cache_creation_input_tokens": 6448,
               "cache_read_input_tokens": 6339, "output_tokens": 1983,
               "output_tokens_details": {"thinking_tokens": 1110}, "service_tier": "standard"},
     "total_cost_usd": 0.0462679, "num_turns": 3, "duration_ms": 17502},
    {"worker": "p3f12ebfc-1", "kind": "batch", "outcome": "failed", "timed_out": false,
     "model": null, "next_batch_calls": 0, "submit_calls": 0, "leased_batches": [],
     "elapsed_ms": 900, "detail": "exited"}
  ]
}"#;

#[test]
fn launch_summaries_are_found_under_the_log_directories_and_kept_by_run() {
    let dir = tempfile::tempdir().unwrap();
    let records = dir.path().join("rec");
    fs::create_dir(&records).unwrap();
    fs::write(
        records.join("status.json"),
        r#"{"run":"probe2","seed":1,"programs_per_batch":1,"iteration":0,
            "tasks":[],"library_learning":false}"#,
    )
    .unwrap();
    fs::write(
        records.join("proposal_stats.json"),
        r#"{"run":"probe2","iterations":[]}"#,
    )
    .unwrap();
    let logs = dir.path().join("logs");
    fs::create_dir_all(logs.join("iter0")).unwrap();
    fs::create_dir_all(logs.join("other")).unwrap();
    fs::write(logs.join("iter0/summary.json"), SUMMARY).unwrap();
    fs::write(
        logs.join("other/summary.json"),
        SUMMARY.replace("\"probe2\"", "\"probe3\""),
    )
    .unwrap();
    // The directory listed twice, and a subdirectory of it, count once.
    let read =
        RunRecords::read(&records, &[logs.clone(), logs.join("iter0"), logs.clone()]).unwrap();
    assert_eq!(read.launches.len(), 1);
    // A copy of the same launch's summary elsewhere counts once too.
    let copy = dir.path().join("copy");
    fs::create_dir(&copy).unwrap();
    fs::copy(logs.join("iter0/summary.json"), copy.join("summary.json")).unwrap();
    let both = RunRecords::read(&records, &[logs.clone(), copy]).unwrap();
    assert_eq!(both.launches.len(), 1);
    let l = &read.launches[0];
    assert_eq!(
        (l.launch.as_str(), l.run.as_str(), l.interrupted),
        ("p3f12ebfc", "probe2", false)
    );
    let u = l.sessions[0].usage.unwrap();
    assert_eq!(
        (
            u.input_tokens,
            u.cache_creation_input_tokens,
            u.cache_read_input_tokens,
            u.output_tokens
        ),
        (6, 6448, 6339, 1983)
    );
    assert_eq!(l.sessions[0].total_cost_usd, Some(0.0462679));
    assert!(l.sessions[1].usage.is_none() && l.sessions[1].total_cost_usd.is_none());
    assert_eq!(l.sessions[1].outcome, "failed");
}

#[test]
fn records_of_different_runs_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(
        dir.path().join("status.json"),
        r#"{"run":"a","seed":1,"programs_per_batch":1,"iteration":0,"tasks":[],"library_learning":false}"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("proposal_stats.json"),
        r#"{"run":"b","iterations":[]}"#,
    )
    .unwrap();
    let err = RunRecords::read(dir.path(), &[]).unwrap_err();
    assert!(err.0.contains("of run b"), "{err}");
}
