//! Ending iterations of runs filled through the proposer's calls, with real
//! Stitch runs in this process and with runners that act in the middle of
//! the step.

use std::ops::ControlFlow;

use dc_compress::worker::WorkerError;
use dc_compress::{CompressError, Compression, FrontierProgram};
use dc_core::{Arena, LibDef};
use dc_driver::{CompressionReport, InProcess};
use tempfile::TempDir;

use super::*;
use crate::proposer::{next_batch, submit, Refused};
use crate::state::{Arm, Check, CostVerdict, ProgramCost, RunConfig, Standing};

const MERGE: &str = "collapse/merge";
const CANCEL: &str = "collapse/cancel";
const ABSORB: &str = "collapse/absorb";
const TASKS: [&str; 3] = [MERGE, CANCEL, ABSORB];

/// The frontier size of the tests' runs.
const K: u32 = 2;

fn config(quota: u32, prelude: u64) -> RunConfig {
    RunConfig {
        max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
        autodoc: false,
        arm: Arm::Blind,
        library_learning: true,
        seed: 3,
        proposals_per_task: quota,
        programs_per_batch: 1,
        lease_ms: 1000,
        max_unchecked_per_batch: 3,
        frontier_size: K,
        stitch: StitchSettings::default(),
        prelude,
        compared_with: None,
        approach_hints: false,
        show_received: false,
    }
}

fn setup(quota: u32) -> (TempDir, Store, RunId) {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
    store
        .create_run(&run, config(quota, shipped_prelude()), tasks)
        .unwrap();
    (dir, store, run)
}

/// A program that solves `task`: its reference number `which`, counted
/// round its references, naive first.
fn solution(task: &str, which: usize) -> String {
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == task)
        .unwrap();
    let entry = v.references[which % v.references.len()].entry;
    format!("{}\nlet solve = {entry}\n", v.source)
}

fn check(store: &Store, run: &RunId) -> usize {
    check_pending(store, run, &VerifierCache::default())
        .unwrap()
        .checked
}

/// Hands out every batch the run has open at `now_ms`, each to a worker of
/// its own, submits a solution for each, and checks them. The `n`-th batch
/// of a task gets the task's reference `n`.
fn fill(store: &Store, run: &RunId, now_ms: u64) {
    fill_with(store, run, now_ms, solution);
}

/// As `fill`, with `program(task, n)` sent for the `n`-th batch of `task`.
fn fill_with(store: &Store, run: &RunId, now_ms: u64, program: impl Fn(&str, usize) -> String) {
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
        let program = program(&task, *n);
        *n += 1;
        submit(
            store,
            run,
            payload.batch,
            &worker,
            &payload.token,
            vec![program],
            None,
            now_ms,
        )
        .unwrap();
    }
    check(store, run);
    assert!(store.read(run, RunState::quota_in).unwrap());
}

fn events(store: &Store, run: &RunId) -> Vec<Event> {
    let mut all = Vec::new();
    store
        .for_each_event(run, |e| {
            all.push(e);
            ControlFlow::Continue(())
        })
        .unwrap();
    all
}

/// The run's journal, as it is on disk.
fn journal(dir: &TempDir) -> Vec<u8> {
    std::fs::read(dir.path().join("r").join("journal")).unwrap()
}

fn lines(journal: &[u8]) -> usize {
    journal.iter().filter(|b| **b == b'\n').count()
}

/// Runs Stitch in this process, after it has run `before` once.
struct Hooked<'a> {
    before: Option<Box<dyn FnOnce() + 'a>>,
}

impl<'a> Hooked<'a> {
    fn new(before: impl FnOnce() + 'a) -> Hooked<'a> {
        Hooked {
            before: Some(Box::new(before)),
        }
    }
}

impl Stitch for Hooked<'_> {
    fn run(
        &mut self,
        arena: &mut Arena,
        prelude: Option<&Prelude>,
        libs: &[LibDef],
        programs: &[FrontierProgram],
        cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        if let Some(before) = self.before.take() {
            before();
        }
        InProcess.run(arena, prelude, libs, programs, cfg)
    }
}

/// A Stitch run that is cancelled.
struct Cancelled;

impl Stitch for Cancelled {
    fn run(
        &mut self,
        _arena: &mut Arena,
        _prelude: Option<&Prelude>,
        _libs: &[LibDef],
        _programs: &[FrontierProgram],
        _cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        Err(CompressError::Worker(WorkerError::Cancelled))
    }
}

#[test]
fn ending_an_iteration_commits_the_whole_step_as_one_record() {
    let (dir, store, run) = setup(2);
    fill(&store, &run, 0);
    let before = events(&store, &run).len();
    let records = lines(&journal(&dir));
    let verified = store.read(&run, |s| s.verified().len()).unwrap();
    assert_eq!(verified, 6, "two solutions for each of three tasks");

    let done = finish_iteration(&store, &run, &mut InProcess).unwrap();

    assert_eq!(done.iteration, 0);
    assert_eq!(done.used, 6);
    let CompressionReport::Ran { kept, .. } = &done.report.compression else {
        panic!("{:?}", done.report.compression);
    };
    assert_eq!(*kept, done.new_abstractions);
    // The three tasks' solutions share code, so Stitch finds something.
    assert!(done.new_abstractions > 0, "{:?}", done.report.compression);

    // One record holds the step.
    assert_eq!(lines(&journal(&dir)), records + 1);
    let added = &events(&store, &run)[before..];
    let mut kinds = Vec::new();
    for event in added {
        kinds.push(match event {
            Event::FrontierSet { task, members } => {
                assert!(!members.is_empty() && members.len() <= K as usize);
                task.as_str()
            }
            Event::LibraryExtended { entries } => {
                assert_eq!(entries.len(), done.new_abstractions);
                "library"
            }
            Event::Logged(record) => {
                assert_eq!(record.kind, LogKind::Compression);
                assert_eq!(record.iteration, 0);
                assert_eq!(record.task, None);
                // The log holds the report the caller got.
                let logged: IterationReport = serde_json::from_str(&record.text).unwrap();
                assert_eq!(logged, done.report);
                "log"
            }
            Event::IterationFinished { iteration, used } => {
                assert_eq!((*iteration, *used), (0, 6));
                "finished"
            }
            other => panic!("{other:?}"),
        });
    }
    assert_eq!(kinds, [MERGE, CANCEL, ABSORB, "library", "log", "finished"]);

    // The run is in the next iteration, with the step's result.
    let state = store.state(&run).unwrap();
    assert_eq!(state.iteration(), 1);
    assert_eq!(state.library().len(), done.new_abstractions);
    assert_eq!(state.log_count(LogKind::Compression), 1);
    assert!(state.verified().is_empty());
    assert!(!state.quota_in());
    // The new library and frontiers load: the next iteration's programs are
    // checked against them, and a second step takes them as its input.
    fill(&store, &run, 10_000);
    let again = finish_iteration(&store, &run, &mut InProcess).unwrap();
    assert_eq!(again.iteration, 1);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 2);
}

/// Each task's kept programs and their costs, as the run's state holds them
/// after `iteration` ended.
fn kept(store: &Store, run: &RunId, iteration: u64) -> BTreeMap<String, Vec<FrontierMember>> {
    store
        .read(run, |s| s.kept_frontiers(iteration).cloned().unwrap())
        .unwrap()
}

fn programs(kept: &BTreeMap<String, Vec<FrontierMember>>) -> BTreeMap<String, Vec<String>> {
    kept.iter()
        .map(|(t, ms)| (t.clone(), ms.iter().map(|m| m.program.clone()).collect()))
        .collect()
}

#[test]
fn the_kept_programs_are_measured_and_measuring_changes_nothing_else() {
    // Two runs filled alike; one has no time to measure.
    let (_d1, measured, run) = setup(2);
    let (_d2, unmeasured, _) = setup(2);
    fill(&measured, &run, 0);
    fill(&unmeasured, &run, 0);
    let a = finish_iteration(&measured, &run, &mut InProcess).unwrap();
    let b =
        finish_iteration_with_budget(&unmeasured, &run, &mut InProcess, Duration::ZERO).unwrap();
    assert_eq!(a, b, "the step's outcome does not depend on measuring");
    let with = kept(&measured, &run, 0);
    let without = kept(&unmeasured, &run, 0);
    assert_eq!(programs(&with), programs(&without));
    assert_eq!(
        measured.read(&run, |s| s.library().to_vec()).unwrap(),
        unmeasured.read(&run, |s| s.library().to_vec()).unwrap()
    );
    assert!(a.new_abstractions > 0);

    // Every member was measured, under the library with the step's new
    // abstractions: some kept programs use them.
    let mut uses_new = 0;
    for (task, members) in &with {
        assert_eq!(members.len(), K as usize, "{task}");
        for m in members {
            let Some(ProgramCost::Measured { verdict, sizes }) = &m.cost else {
                panic!("{task}: {:?}", m.cost);
            };
            assert!(
                matches!(verdict, CostVerdict::Class(_)),
                "{task}: {verdict:?}"
            );
            assert!(
                sizes.iter().all(|s| s.median.is_some()),
                "{task}: {sizes:?}"
            );
            uses_new += usize::from((0..a.new_abstractions).any(|i| {
                m.program.contains(&format!("lib_{i} ")) || m.program.contains(&format!("lib_{i})"))
            }));
        }
    }
    assert!(uses_new > 0, "{with:?}");
    for members in without.values() {
        for m in members {
            assert_eq!(
                m.cost,
                Some(ProgramCost::NotMeasured {
                    reason: crate::cost::OUT_OF_TIME.to_owned()
                })
            );
        }
    }

    // The next iteration measures what was not measured, and keeps the
    // costs of programs it measured before.
    fill(&measured, &run, 10_000);
    fill(&unmeasured, &run, 10_000);
    finish_iteration(&measured, &run, &mut InProcess).unwrap();
    finish_iteration(&unmeasured, &run, &mut InProcess).unwrap();
    let with = kept(&measured, &run, 1);
    let caught_up = kept(&unmeasured, &run, 1);
    assert_eq!(with, caught_up);
    assert!(with
        .values()
        .flatten()
        .all(|m| matches!(m.cost, Some(ProgramCost::Measured { .. }))));
    // Iteration 0's frontiers are still there to read.
    assert_eq!(programs(&kept(&measured, &run, 0)), programs(&without));
}

#[test]
fn an_iteration_that_lacks_programs_or_checks_does_not_end() {
    let (dir, store, run) = setup(1);
    let before = journal(&dir);
    let short = |e: FinishError| match e {
        FinishError::QuotaNotIn {
            iteration: 0,
            short,
            pending_checks,
        } => (short, pending_checks),
        other => panic!("{other:?}"),
    };
    let refused = finish_iteration(&store, &run, &mut InProcess).unwrap_err();
    assert_eq!(short(refused), (TASKS.map(str::to_owned).to_vec(), 0));
    assert_eq!(journal(&dir), before, "a refused step wrote to the run");

    // Every program is in, but one is not checked.
    let verifiers = VerifierCache::default();
    for (i, _) in TASKS.iter().enumerate() {
        let worker = format!("w{i}");
        let payload = next_batch(&store, &run, &worker, 0, &verifiers)
            .unwrap()
            .batch
            .unwrap();
        let task = store
            .read(&run, |s| s.batch(payload.batch).unwrap().task.clone())
            .unwrap();
        submit(
            &store,
            &run,
            payload.batch,
            &worker,
            &payload.token,
            vec![solution(&task, 0)],
            None,
            0,
        )
        .unwrap();
        if i == 0 {
            assert_eq!(check(&store, &run), 1);
        }
    }
    let before = journal(&dir);
    let refused = finish_iteration(&store, &run, &mut InProcess).unwrap_err();
    assert_eq!(short(refused), (Vec::new(), 2));
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

#[test]
fn a_step_that_another_server_beat_writes_nothing() {
    let (dir, store, run) = setup(1);
    fill(&store, &run, 0);
    let other = Store::open(dir.path()).unwrap();
    let mut after_other = Vec::new();
    // Another server ends the iteration while this one runs Stitch.
    let mut stitch = Hooked::new(|| {
        finish_iteration(&other, &run, &mut InProcess).unwrap();
        after_other = journal(&dir);
    });
    let lost = finish_iteration(&store, &run, &mut stitch).unwrap_err();
    drop(stitch);
    assert!(
        matches!(lost, FinishError::AlreadyFinished { iteration: 0 }),
        "{lost:?}"
    );
    assert!(!after_other.is_empty());
    assert_eq!(journal(&dir), after_other, "the beaten step wrote");
    let state = store.state(&run).unwrap();
    assert_eq!(state.iteration(), 1);
    assert_eq!(state.log_count(LogKind::Compression), 1);
}

#[test]
fn a_program_that_comes_in_during_the_step_is_checked_and_kept_for_the_next() {
    let (_dir, store, run) = setup(1);
    let verifiers = VerifierCache::default();
    // A proposer takes a batch and lets its lease end.
    let slow = next_batch(&store, &run, "slow", 0, &verifiers)
        .unwrap()
        .batch
        .unwrap();
    let task = store
        .read(&run, |s| s.batch(slow.batch).unwrap().task.clone())
        .unwrap();
    // Others fill the iteration, its task too.
    fill(&store, &run, 5000);
    assert_eq!(store.read(&run, |s| s.verified().len()).unwrap(), 3);

    let other = Store::open(_dir.path()).unwrap();
    let mut stitch = Hooked::new(|| {
        // The slow proposer submits while Stitch runs.
        submit(
            &other,
            &run,
            slow.batch,
            "slow",
            &slow.token,
            vec![solution(&task, 1)],
            None,
            6000,
        )
        .unwrap();
        assert_eq!(other.read(&run, |s| s.pending_checks().len()).unwrap(), 1);
    });
    let done = finish_iteration(&store, &run, &mut stitch).unwrap();
    drop(stitch);

    // It was not one of the step's programs.
    assert_eq!(done.used, 3);
    let state = store.state(&run).unwrap();
    assert_eq!(state.iteration(), 1);
    assert!(state.pending_checks().is_empty());
    // It was checked, and waits for the next step.
    assert_eq!(state.verified().len(), 1);
    assert_eq!(state.verified()[0].task, task);
    let late: Vec<Standing> = events(&store, &run)
        .into_iter()
        .filter_map(|e| match e {
            Event::Submitted(s) if s.batch == slow.batch => Some(s.standing),
            _ => None,
        })
        .collect();
    assert_eq!(late, [Standing::Expired]);
}

#[test]
fn an_interrupted_step_writes_nothing_and_can_be_run_again() {
    let (dir, store, run) = setup(1);
    fill(&store, &run, 0);
    let before = journal(&dir);
    let stopped = finish_iteration(&store, &run, &mut Cancelled).unwrap_err();
    assert!(
        matches!(stopped, FinishError::Interrupted(_)),
        "{stopped:?}"
    );
    assert!(stopped.to_string().contains("Call again"), "{stopped}");
    assert_eq!(journal(&dir), before, "an interrupted step wrote");
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);

    finish_iteration(&store, &run, &mut InProcess).unwrap();
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 1);
}

#[test]
fn a_prelude_in_another_order_has_another_fingerprint_and_is_refused() {
    // Two functions of one type: under the other order a stored reference
    // to the first would name the second, and still type.
    let one = Prelude::from_source("p", "let first x y = x + y\nlet second x y = x - y\n").unwrap();
    let other =
        Prelude::from_source("p", "let second x y = x - y\nlet first x y = x + y\n").unwrap();
    assert_eq!(one.entries()[0].scheme(), other.entries()[0].scheme());
    let made_under_one = config(1, one.fingerprint());
    assert_eq!(check_prelude(&made_under_one, one.fingerprint()), Ok(()));
    assert_eq!(
        check_prelude(&made_under_one, other.fingerprint()),
        Err(PreludeChanged {
            run: one.fingerprint(),
            now: other.fingerprint(),
        })
    );
}

#[test]
fn a_run_made_under_another_prelude_is_not_served() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let other = Prelude::from_source("p", "let second x y = x - y\n").unwrap();
    assert_ne!(other.fingerprint(), shipped_prelude());
    store
        .create_run(&run, config(1, other.fingerprint()), vec![MERGE.to_owned()])
        .unwrap();
    let before = journal(&dir);
    let verifiers = VerifierCache::default();

    let step = finish_iteration(&store, &run, &mut InProcess).unwrap_err();
    assert!(matches!(step, FinishError::Prelude(_)), "{step:?}");
    let batch = next_batch(&store, &run, "w", 0, &verifiers).unwrap_err();
    assert!(
        matches!(&batch, Refused::State(why) if why.contains("another prelude")),
        "{batch:?}"
    );
    let checks = check_pending(&store, &run, &verifiers).unwrap_err();
    assert!(
        matches!(&checks, Refused::State(why) if why.contains("another prelude")),
        "{checks:?}"
    );
    assert_eq!(journal(&dir), before, "a refused call wrote to the run");
}

/// A run with library learning off (the LLM-only baseline, D24) ends its
/// iterations with frontiers and no Stitch run. Its programs share code, as
/// in [`ending_an_iteration_commits_the_whole_step_as_one_record`], so a
/// Stitch run would find abstractions; and a step that tried to start its
/// worker would fail, since there is none.
#[test]
fn a_run_without_library_learning_ends_its_iterations_without_stitch() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let config = RunConfig {
        library_learning: false,
        ..config(2, shipped_prelude())
    };
    let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
    store.create_run(&run, config, tasks).unwrap();
    let missing = WorkerStart {
        program: dir.path().join("no-such-worker"),
        args: Vec::new(),
    };
    let cancel = AtomicBool::new(false);
    for iteration in 0..2 {
        fill(&store, &run, iteration * 10_000);
        let before = events(&store, &run).len();
        let done = finish_iteration_in_worker(&store, &run, &missing, &cancel).unwrap();
        assert_eq!(done.iteration, iteration);
        assert_eq!(done.used, 6);
        assert_eq!(done.new_abstractions, 0);
        assert_eq!(done.report.compression, CompressionReport::Off);
        let added = &events(&store, &run)[before..];
        assert!(
            !added
                .iter()
                .any(|e| matches!(e, Event::LibraryExtended { .. })),
            "{added:?}"
        );
        let state = store.state(&run).unwrap();
        assert!(state.library().is_empty());
        for task in TASKS {
            let members = state.frontier(task).unwrap();
            assert!(!members.is_empty() && members.len() <= K as usize, "{task}");
        }
    }
}

/// A run's memory limit (`RunConfig::max_memory`) reaches the checks of
/// its submissions and the measuring of its frontiers. The run has a limit
/// of 1 MiB. A program that makes an array of a billion elements on every
/// input stops at that limit on its first example; under the default limit
/// it would run out of fuel first. The kept programs make an array of
/// 50,000 elements, about 1.6 MB, from 512 elements: they pass their
/// examples, and their sweeps stop at the limit from 512.
#[test]
fn a_runs_memory_limit_reaches_its_checks_and_its_costs() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let config = RunConfig {
        library_learning: false,
        max_memory: 1 << 20,
        ..config(2, shipped_prelude())
    };
    store
        .create_run(&run, config, vec![CANCEL.to_owned()])
        .unwrap();
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == CANCEL)
        .unwrap();
    let with_array = |which: usize, from: u64, length: u64| {
        let e = v.references[which % v.references.len()].entry;
        format!(
            "{}\nlet solve = fun xs -> if length xs < {from} then {e} xs \
             else (if 0 < array_size (array_tabulate {length} (fun i -> i)) then {e} xs \
             else {e} xs)\n",
            v.source
        )
    };

    let verifiers = VerifierCache::default();
    let payload = next_batch(&store, &run, "w", 0, &verifiers)
        .unwrap()
        .batch
        .unwrap();
    let runaway = with_array(0, 0, 1_000_000_000);
    submit(
        &store,
        &run,
        payload.batch,
        "w",
        &payload.token,
        vec![runaway],
        None,
        0,
    )
    .unwrap();
    check(&store, &run);
    let checks: Vec<Check> = events(&store, &run)
        .into_iter()
        .filter_map(|e| match e {
            Event::Checked(c) if c.batch == payload.batch => Some(c.checks),
            _ => None,
        })
        .flatten()
        .collect();
    match checks.as_slice() {
        [Check::Run { error }] => assert!(
            error.ends_with("no result within the memory limit of 1048576 bytes"),
            "{error}"
        ),
        other => panic!("{other:?}"),
    }

    fill_with(&store, &run, 10_000, |_, n| with_array(n, 512, 50_000));
    let missing = WorkerStart {
        program: dir.path().join("no-such-worker"),
        args: Vec::new(),
    };
    finish_iteration_in_worker(&store, &run, &missing, &AtomicBool::new(false)).unwrap();
    let members = kept(&store, &run, 0).remove(CANCEL).unwrap();
    assert!(!members.is_empty());
    for m in &members {
        let Some(ProgramCost::Measured { sizes, .. }) = &m.cost else {
            panic!("{:?}", m.cost);
        };
        for size in sizes {
            if size.n < 512 {
                assert!(size.median.is_some(), "{size:?}");
            } else {
                assert_eq!(
                    size.failed.as_ref().map(|f| f.kind),
                    Some(crate::state::FailureKind::MemoryLimit),
                    "{size:?}"
                );
            }
        }
    }
}

/// A worker that never answers: `sleep`, which does not read its request.
fn sleeper() -> WorkerStart {
    WorkerStart {
        program: "sleep".into(),
        args: vec!["600".into()],
    }
}

#[test]
fn setting_the_cancel_flag_stops_a_step_in_a_worker_and_nothing_is_written() {
    let (dir, store, run) = setup(1);
    fill(&store, &run, 0);
    let before = journal(&dir);
    let cancel = AtomicBool::new(false);
    let started = std::time::Instant::now();
    let stopped = std::thread::scope(|scope| {
        let step = scope.spawn(|| finish_iteration_in_worker(&store, &run, &sleeper(), &cancel));
        std::thread::sleep(Duration::from_millis(300));
        assert!(!step.is_finished(), "the step did not wait for its worker");
        cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        step.join().unwrap()
    });
    let stopped = stopped.unwrap_err();
    assert!(
        matches!(stopped, FinishError::Interrupted(_)),
        "{stopped:?}"
    );
    assert!(started.elapsed() < STUCK_WALL);
    assert_eq!(journal(&dir), before);
}

#[test]
fn a_worker_that_cannot_be_started_is_an_interruption_not_a_finished_iteration() {
    let (dir, store, run) = setup(1);
    fill(&store, &run, 0);
    let before = journal(&dir);
    let missing = WorkerStart {
        program: dir.path().join("no-such-worker"),
        args: Vec::new(),
    };
    let cancel = AtomicBool::new(false);
    let failed = finish_iteration_in_worker(&store, &run, &missing, &cancel).unwrap_err();
    assert!(matches!(failed, FinishError::Interrupted(_)), "{failed:?}");
    assert_eq!(journal(&dir), before);
}

#[test]
fn a_worker_stopped_at_the_runs_time_limit_ends_the_iteration_without_abstractions() {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    let settings = RunConfig {
        autodoc: false,
        stitch: StitchSettings {
            wall_ms: 200,
            ..StitchSettings::default()
        },
        ..config(1, shipped_prelude())
    };
    let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
    store.create_run(&run, settings, tasks).unwrap();
    // Reference 1 of each collapse task, the good one: cheap to measure, so
    // that measuring the kept programs adds little to the time below.
    fill_with(&store, &run, 0, |task, _| solution(task, 1));
    let cancel = AtomicBool::new(false);
    // The fuel plans that measuring the frontiers needs are made once per
    // process. Made here, they are not part of the time below.
    for task in TASKS {
        crate::cost::fuel_plan(task).unwrap();
    }
    // The run's limit, not the worker's default of ten minutes, stops it.
    let started = std::time::Instant::now();
    let done = finish_iteration_in_worker(&store, &run, &sleeper(), &cancel).unwrap();
    assert!(started.elapsed() < STUCK_WALL);
    let CompressionReport::Failed { error, .. } = &done.report.compression else {
        panic!("{:?}", done.report.compression);
    };
    assert!(error.contains("200"), "{error}");
    assert_eq!(done.new_abstractions, 0);
    let state = store.state(&run).unwrap();
    assert_eq!(state.iteration(), 1);
    assert!(state.library().is_empty());
    assert_eq!(state.frontier(MERGE).map(<[_]>::len), Some(1));
}

#[test]
fn a_server_that_shuts_down_stops_its_step() {
    use crate::server::RunArg;

    let (dir, store, run, server, runtime) = stuck_server();
    let before = journal(&dir);
    let stopper = server.clone();
    let stop = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        stopper.shut_down();
    });
    let started = std::time::Instant::now();
    let reply = runtime.block_on(server.finish_iteration_for_test(
        RunArg {
            run: "r".to_owned(),
        },
        std::future::pending(),
    ));
    stop.join().unwrap();
    let why = reply.unwrap_err();
    assert!(why.contains("interrupted"), "{why}");
    assert!(started.elapsed() >= Duration::from_millis(300));
    assert!(started.elapsed() < STUCK_WALL);
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

/// The time limit of the Stitch runs of `stuck_server`'s run.
const STUCK_WALL: Duration = Duration::from_secs(20);

/// A server whose worker does not answer within the run's limit of
/// `STUCK_WALL`, on a run that is ready to end, and a runtime with a clock.
fn stuck_server() -> (
    TempDir,
    Store,
    RunId,
    crate::server::DcServer,
    tokio::runtime::Runtime,
) {
    let dir = TempDir::new().unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r").unwrap();
    // A step that is not stopped ends at this limit, without the error the
    // tests look for, so a test fails then and does not wait long.
    let mut config = config(1, shipped_prelude());
    config.stitch.wall_ms = STUCK_WALL.as_millis() as u64;
    let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
    store.create_run(&run, config, tasks).unwrap();
    fill(&store, &run, 0);
    let server = crate::server::DcServer::new(Store::open(dir.path()).unwrap(), sleeper());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    (dir, store, run, server, runtime)
}

#[test]
fn a_call_the_client_cancels_stops_its_step_and_writes_nothing() {
    use crate::server::RunArg;

    let (dir, store, run, server, runtime) = stuck_server();
    let before = journal(&dir);
    let started = std::time::Instant::now();
    let reply = runtime.block_on(server.finish_iteration_for_test(
        RunArg {
            run: "r".to_owned(),
        },
        async { tokio::time::sleep(Duration::from_millis(300)).await },
    ));
    let why = reply.unwrap_err();
    assert!(why.contains("interrupted"), "{why}");
    assert!(started.elapsed() >= Duration::from_millis(300));
    assert!(started.elapsed() < STUCK_WALL);
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

#[test]
fn a_call_whose_future_is_dropped_stops_its_step_and_writes_nothing() {
    use crate::server::RunArg;

    let (dir, store, run, server, runtime) = stuck_server();
    let before = journal(&dir);
    let started = std::time::Instant::now();
    let dropped = runtime.block_on(async {
        tokio::time::timeout(
            Duration::from_millis(300),
            server.finish_iteration_for_test(
                RunArg {
                    run: "r".to_owned(),
                },
                std::future::pending(),
            ),
        )
        .await
    });
    assert!(dropped.is_err(), "the call ended before it was dropped");
    // Dropping the runtime waits for the step's thread, which ends once its
    // worker is stopped.
    drop(runtime);
    assert!(started.elapsed() < STUCK_WALL);
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

#[test]
fn a_cancel_that_comes_when_no_worker_runs_still_stops_the_commit() {
    let (dir, store, run) = setup(1);
    // No program solves its task, so the step has nothing for Stitch and
    // starts no worker that the flag could stop.
    fill_with(&store, &run, 0, |_, _| "let solve = 1\n".to_owned());
    assert!(store.read(&run, |s| s.verified().is_empty()).unwrap());
    let before = journal(&dir);
    let cancel = AtomicBool::new(true);
    let stopped = finish_iteration_in_worker(&store, &run, &sleeper(), &cancel).unwrap_err();
    assert!(
        matches!(stopped, FinishError::Interrupted(_)),
        "{stopped:?}"
    );
    assert_eq!(journal(&dir), before);
}

#[test]
fn a_cancel_stops_measuring_and_an_uncancelled_call_measures() {
    // No library learning, so no Stitch worker starts and the flag is
    // first read by measuring.
    let setup_run = || {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let run = RunId::new("r").unwrap();
        let settings = RunConfig {
            library_learning: false,
            ..config(1, shipped_prelude())
        };
        let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
        store.create_run(&run, settings, tasks).unwrap();
        fill_with(&store, &run, 0, |task, _| solution(task, 1));
        (dir, store, run)
    };
    let (dir, store, run) = setup_run();
    let before = journal(&dir);
    let cancel = AtomicBool::new(true);
    let started = crate::cost::started();
    let stopped = finish_iteration_in_worker(&store, &run, &sleeper(), &cancel).unwrap_err();
    assert!(
        matches!(stopped, FinishError::Interrupted(_)),
        "{stopped:?}"
    );
    assert_eq!(crate::cost::started(), started, "no program was measured");
    assert_eq!(journal(&dir), before);

    // The same call, not cancelled, measures every kept program within the
    // server's budget.
    let (_dir, store, run) = setup_run();
    let cancel = AtomicBool::new(false);
    finish_iteration_in_worker(&store, &run, &sleeper(), &cancel).unwrap();
    assert_eq!(crate::cost::started(), started + TASKS.len() as u64);
    for members in kept(&store, &run, 0).values() {
        assert!(
            members
                .iter()
                .all(|m| matches!(m.cost, Some(ProgramCost::Measured { .. }))),
            "{members:?}"
        );
    }
}

#[test]
fn a_step_on_a_run_that_was_removed_and_made_again_writes_nothing() {
    let (dir, store, run) = setup(1);
    fill(&store, &run, 0);
    let other = Store::open(dir.path()).unwrap();
    let mut remade = Vec::new();
    // While this server runs Stitch, the run is removed and a run of the
    // same id is made and filled: iteration 0 again, with an empty library.
    let mut stitch = Hooked::new(|| {
        std::fs::remove_dir_all(dir.path().join("r")).unwrap();
        let tasks = TASKS.iter().map(|t| (*t).to_owned()).collect();
        other
            .create_run(&run, config(1, shipped_prelude()), tasks)
            .unwrap();
        fill(&other, &run, 0);
        remade = journal(&dir);
    });
    let lost = finish_iteration(&store, &run, &mut stitch).unwrap_err();
    drop(stitch);
    assert!(
        matches!(lost, FinishError::AlreadyFinished { iteration: 0 }),
        "{lost:?}"
    );
    assert!(!remade.is_empty());
    assert_eq!(journal(&dir), remade, "the step wrote to the new run");
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

#[test]
fn a_cancelled_notification_stops_the_step_of_its_call() {
    use rmcp::ServiceExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let (dir, store, run, server, runtime) = stuck_server();
    let before = journal(&dir);
    runtime.block_on(async {
        let (ours, theirs) = tokio::io::duplex(1 << 16);
        let (read, mut write) = tokio::io::split(ours);
        let mut lines = BufReader::new(read).lines();
        let serving = tokio::spawn({
            let server = server.clone();
            async move {
                let service = server.serve(tokio::io::split(theirs)).await.unwrap();
                let _ = service.waiting().await;
            }
        });
        let mut send = async |message: serde_json::Value| {
            write
                .write_all(format!("{message}\n").as_bytes())
                .await
                .unwrap();
        };
        send(serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": { "name": "test", "version": "0" }
            }
        }))
        .await;
        let hello = lines.next_line().await.unwrap().unwrap();
        assert!(hello.contains("\"id\":1"), "{hello}");
        send(serde_json::json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).await;
        send(serde_json::json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": { "name": "dc_finish_iteration", "arguments": { "run": "r" } }
        }))
        .await;
        // The call runs, and would go on for the run's time limit.
        let deadline = tokio::time::Instant::now() + STUCK_WALL / 2;
        while server.finish_calls_running() != 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the call did not start"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(server.finish_calls_running(), 1);
        send(serde_json::json!({
            "jsonrpc": "2.0", "method": "notifications/cancelled",
            "params": { "requestId": 2, "reason": "test" }
        }))
        .await;
        while server.finish_calls_running() != 0 {
            assert!(tokio::time::Instant::now() < deadline, "the call went on");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        serving.abort();
    });
    drop(runtime);
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}

#[test]
fn a_call_that_starts_after_the_server_shut_down_is_stopped() {
    use crate::server::RunArg;

    let (dir, store, run, server, runtime) = stuck_server();
    let before = journal(&dir);
    server.shut_down();
    let started = std::time::Instant::now();
    let reply = runtime.block_on(server.finish_iteration_for_test(
        RunArg {
            run: "r".to_owned(),
        },
        std::future::pending(),
    ));
    let why = reply.unwrap_err();
    assert!(why.contains("interrupted"), "{why}");
    assert!(started.elapsed() < STUCK_WALL);
    assert_eq!(journal(&dir), before);
    assert_eq!(store.read(&run, RunState::iteration).unwrap(), 0);
}
