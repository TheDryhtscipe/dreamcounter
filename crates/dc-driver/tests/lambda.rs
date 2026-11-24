//! The step's selection under `J(L)` at λ > 0 (decision D4, bead
//! dreamcounter-211560): ranking and trimming the pool, the members given to
//! Stitch, ordering again after compression, and the selection report.

use std::collections::BTreeMap;

use dc_compress::{CompressConfig, CompressError, Compression, FrontierProgram};
use dc_core::{sexpr, Arena, LibDef, Scheme};
use dc_driver::objective::{Refusal, Unfitted};
use dc_driver::{
    run_iteration, Candidate, DriverConfig, Fate, Growth, Growths, IterationInput,
    IterationOutcome, Lambda, MemberScore, Source, Stitch, TaskSpec,
};
use dc_stitch::LibNames;
use dc_surface::prelude::Prelude;
use dc_tasks::growth::GrowthClass;

fn task(name: &str, signature: &str) -> TaskSpec {
    TaskSpec {
        name: name.to_owned(),
        signature: signature.parse::<Scheme>().unwrap(),
    }
}

fn candidate(task: &str, program: &str) -> Candidate {
    Candidate {
        task: task.to_owned(),
        program: program.to_owned(),
    }
}

fn config(k: usize, lambda: f64) -> DriverConfig {
    DriverConfig {
        frontier_size: k,
        compress: Some(CompressConfig::default()),
        refuse_bucketing: false,
        lambda: Lambda::new(lambda).unwrap(),
    }
}

fn fitted(class: GrowthClass) -> Growth {
    Growth::Fitted(class)
}

/// A Stitch runner that finds nothing, and records the programs of each
/// call as text.
#[derive(Default)]
struct Nothing {
    calls: Vec<Vec<(String, String)>>,
}

impl Stitch for Nothing {
    fn run(
        &mut self,
        arena: &mut Arena,
        _prelude: Option<&Prelude>,
        _libs: &[LibDef],
        programs: &[FrontierProgram],
        _cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        self.calls.push(
            programs
                .iter()
                .map(|p| (p.task.clone(), sexpr::print(arena, p.program).unwrap()))
                .collect(),
        );
        Ok(Compression {
            new_libs: Vec::new(),
            rewritten: programs.iter().map(|p| p.program).collect(),
            names: LibNames::new(),
            stitch_steps: 0,
            rejected: None,
        })
    }
}

/// A Stitch runner that checks it is given `given`, as text, and returns
/// `libs` and `rewritten`.
struct Fixed {
    given: Vec<&'static str>,
    libs: Vec<(u32, &'static str)>,
    rewritten: Vec<&'static str>,
}

impl Stitch for Fixed {
    fn run(
        &mut self,
        arena: &mut Arena,
        _prelude: Option<&Prelude>,
        _libs: &[LibDef],
        programs: &[FrontierProgram],
        _cfg: &CompressConfig,
    ) -> Result<Compression, CompressError> {
        let given: Vec<String> = programs
            .iter()
            .map(|p| sexpr::print(arena, p.program).unwrap())
            .collect();
        assert_eq!(given, self.given, "the programs given to Stitch");
        Ok(Compression {
            new_libs: self
                .libs
                .iter()
                .map(|&(arity, body)| LibDef {
                    arity,
                    term: sexpr::parse(arena, body).unwrap(),
                })
                .collect(),
            rewritten: self
                .rewritten
                .iter()
                .map(|p| sexpr::parse(arena, p).unwrap())
                .collect(),
            names: LibNames::new(),
            stitch_steps: self.libs.len(),
            rejected: None,
        })
    }
}

struct Case<'a> {
    tasks: &'a [TaskSpec],
    frontiers: BTreeMap<String, Vec<String>>,
    candidates: &'a [Candidate],
    growths: Option<&'a Growths>,
}

impl Case<'_> {
    fn run(&self, cfg: &DriverConfig, stitch: &mut dyn Stitch) -> IterationOutcome {
        run_iteration(
            IterationInput {
                tasks: self.tasks,
                library: &[],
                frontiers: &self.frontiers,
                candidates: self.candidates,
                prelude: None,
                growth: self.growths,
            },
            cfg,
            stitch,
        )
        .unwrap()
    }
}

fn candidates_growth(growth: &[Growth]) -> Growths {
    Growths {
        members: BTreeMap::new(),
        candidates: growth.to_vec(),
    }
}

/// "5" is the shortest program and the slowest; "(add 2 3)" is longer and
/// faster.
const SHORT_SLOW: &str = "5";
const LONG_FAST: &str = "(add 2 3)";

fn short_slow_and_long_fast() -> (Vec<TaskSpec>, Vec<Candidate>, Growths) {
    (
        vec![task("t", "Int")],
        vec![candidate("t", SHORT_SLOW), candidate("t", LONG_FAST)],
        candidates_growth(&[
            fitted(GrowthClass::Quadratic),
            fitted(GrowthClass::Linear),
        ]),
    )
}

#[test]
fn at_lambda_zero_what_is_known_of_growth_changes_nothing_the_step_chooses() {
    let tasks = [task("t", "Int"), task("u", "Int")];
    let candidates = [
        candidate("t", "(add 1 (add 2 2))"),
        candidate("t", SHORT_SLOW),
        candidate("u", LONG_FAST),
        candidate("t", "(add 4 1)"),
        candidate("u", "7"),
    ];
    let growths = candidates_growth(&[
        fitted(GrowthClass::Constant),
        Growth::TimedOut,
        Growth::Incorrect,
        fitted(GrowthClass::Linear),
        Growth::Refused(Refusal::NoFit),
    ]);
    let blind = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: None,
    };
    let seeing = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    for k in [1, 2, 5] {
        let (mut s1, mut s2) = (Nothing::default(), Nothing::default());
        let a = blind.run(&config(k, 0.0), &mut s1);
        let b = seeing.run(&config(k, 0.0), &mut s2);
        assert_eq!(a.frontiers, b.frontiers, "k {k}");
        assert_eq!(a.sources, b.sources);
        assert_eq!(a.new_library, b.new_library);
        assert_eq!(a.report.programs, b.report.programs);
        assert_eq!(a.report.compression, b.report.compression);
        assert_eq!(s1.calls, s2.calls);
    }
}

#[test]
fn a_large_lambda_puts_a_faster_longer_program_first() {
    let (tasks, candidates, growths) = short_slow_and_long_fast();
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    let zero = case.run(&config(5, 0.0), &mut Nothing::default());
    assert_eq!(zero.frontiers["t"], vec![SHORT_SLOW, LONG_FAST]);
    let large = case.run(&config(5, 1e6), &mut Nothing::default());
    assert_eq!(large.frontiers["t"], vec![LONG_FAST, SHORT_SLOW]);
    assert_eq!(
        large.sources["t"],
        vec![
            Source::Candidate { index: 1 },
            Source::Candidate { index: 0 }
        ]
    );
}

#[test]
fn a_small_lambda_does_not_pay_for_the_longer_program() {
    let (tasks, candidates, growths) = short_slow_and_long_fast();
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    // C differs by 2, and the description lengths by far more than 0.002.
    let out = case.run(&config(5, 1e-3), &mut Nothing::default());
    assert_eq!(out.frontiers["t"], vec![SHORT_SLOW, LONG_FAST]);
}

#[test]
fn with_room_for_one_a_large_lambda_trims_the_slower_shorter_program() {
    let (tasks, candidates, growths) = short_slow_and_long_fast();
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    let out = case.run(&config(1, 1e6), &mut Nothing::default());
    assert_eq!(out.frontiers["t"], vec![LONG_FAST]);
    let fate = |index| {
        out.report
            .programs
            .iter()
            .find(|p| p.source == Source::Candidate { index })
            .unwrap()
            .fate
            .clone()
    };
    assert_eq!(fate(0), Fate::Trimmed);
    assert_eq!(fate(1), Fate::Kept);
}

#[test]
fn an_incorrect_program_ranks_last_at_a_positive_lambda() {
    let tasks = [task("t", "Int")];
    let candidates = [candidate("t", SHORT_SLOW), candidate("t", LONG_FAST)];
    let growths = candidates_growth(&[Growth::Incorrect, Growth::TimedOut]);
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    let out = case.run(&config(5, 1e-9), &mut Nothing::default());
    assert_eq!(out.frontiers["t"], vec![LONG_FAST, SHORT_SLOW]);
}

#[test]
fn stitch_is_given_only_correct_members_no_costlier_than_the_selected_one() {
    let tasks = [task("t", "Int"), task("u", "Int")];
    let candidates = [
        candidate("t", SHORT_SLOW),
        candidate("t", LONG_FAST),
        candidate("t", "(add 1 (add 1 3))"),
        candidate("t", "(add 4 1)"),
        candidate("u", "(mul 2 2)"),
        candidate("u", "(add 2 2)"),
    ];
    // In t, at λ = 1 the shortest program, linear, is selected: the
    // constant one is longer, the quadratic one longer still, and the last
    // is incorrect. In u every program is incorrect.
    let growths = candidates_growth(&[
        fitted(GrowthClass::Linear),
        fitted(GrowthClass::Quadratic),
        fitted(GrowthClass::Constant),
        Growth::Incorrect,
        Growth::Incorrect,
        Growth::Incorrect,
    ]);
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    let mut stitch = Nothing::default();
    let out = case.run(&config(5, 1.0), &mut stitch);
    assert_eq!(out.frontiers["t"][0], SHORT_SLOW);
    assert_eq!(stitch.calls.len(), 1);
    let mut given: Vec<String> = stitch.calls[0].iter().map(|(_, p)| p.clone()).collect();
    given.sort();
    assert_eq!(given, vec!["(add 1 (add 1 3))", SHORT_SLOW]);

    // At λ = 0 Stitch is given every member, as before λ was a setting.
    let mut stitch = Nothing::default();
    case.run(&config(5, 0.0), &mut stitch);
    assert_eq!(stitch.calls[0].len(), 6);
}

#[test]
fn after_compression_each_frontier_is_ordered_again_under_j() {
    let tasks = [task("t", "Int")];
    let candidates = [
        candidate("t", SHORT_SLOW),
        candidate("t", "(add 1 2)"),
        candidate("t", "(add 3 4)"),
    ];
    let growths = candidates_growth(&[
        fitted(GrowthClass::Quadratic),
        fitted(GrowthClass::Linear),
        fitted(GrowthClass::Linear),
    ]);
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    // Before compression the two linear programs come first, and only they
    // go to Stitch. The rewriting makes the second of them the shortest
    // linear program, so it is selected after compression; the quadratic
    // program, shortest of all, stays last.
    let mut stitch = Fixed {
        given: vec!["(add 1 2)", "(add 3 4)"],
        libs: vec![(1, "(lam (add $0 4))")],
        rewritten: vec!["(add 1 2)", "(lib_0 3)"],
    };
    let out = case.run(&config(5, 1e6), &mut stitch);
    assert_eq!(out.new_library.len(), 1);
    assert_eq!(out.frontiers["t"], vec!["(lib_0 3)", "(add 1 2)", SHORT_SLOW]);
    assert_eq!(
        out.sources["t"],
        vec![
            Source::Candidate { index: 2 },
            Source::Candidate { index: 1 },
            Source::Candidate { index: 0 },
        ]
    );
}

#[test]
fn the_report_gives_each_members_cost_and_cause_and_the_lambda_zero_selection() {
    let tasks = [task("t", "Int"), task("u", "Int"), task("v", "Int")];
    let candidates = [
        candidate("t", SHORT_SLOW),
        candidate("t", LONG_FAST),
        candidate("u", "7"),
        candidate("u", "(add 3 3)"),
        candidate("u", "(add 1 (add 3 3))"),
        candidate("u", "(add 2 (add 3 3))"),
        candidate("v", "8"),
    ];
    let growths = candidates_growth(&[
        fitted(GrowthClass::Quadratic),
        fitted(GrowthClass::Linear),
        Growth::TimedOut,
        Growth::Refused(Refusal::TooNoisy),
        fitted(GrowthClass::Log),
        Growth::NotMeasured,
        Growth::Incorrect,
    ]);
    let case = Case {
        tasks: &tasks,
        frontiers: BTreeMap::new(),
        candidates: &candidates,
        growths: Some(&growths),
    };
    let out = case.run(&config(5, 1e6), &mut Nothing::default());
    let report = out.report.selection.expect("the step reports its selection");
    assert_eq!(report.lambda, Lambda::new(1e6).unwrap());

    let t = &report.tasks["t"];
    assert_eq!(out.frontiers["t"], vec![LONG_FAST, SHORT_SLOW]);
    assert_eq!((t.selected, t.baseline), (Some(0), Some(1)));
    assert_eq!(
        t.members.iter().map(|m| m.c).collect::<Vec<_>>(),
        vec![Some(2), Some(4)]
    );

    // In u, log n is fitted (C 1); the refused and unmeasured members get 2;
    // the timed-out member gets 8. The fitted one is selected; λ = 0 takes
    // the shortest, the timed-out one.
    let u = &report.tasks["u"];
    assert_eq!(
        out.frontiers["u"],
        vec!["(add 1 (add 3 3))", "(add 3 3)", "(add 2 (add 3 3))", "7"]
    );
    let scores: Vec<(Option<u32>, Option<Unfitted>)> =
        u.members.iter().map(|m| (m.c, m.unfitted)).collect();
    assert_eq!(
        scores,
        vec![
            (Some(1), None),
            (Some(2), Some(Unfitted::Refused(Refusal::TooNoisy))),
            (Some(2), Some(Unfitted::NotMeasured)),
            (Some(8), Some(Unfitted::TimedOut)),
        ]
    );
    assert_eq!((u.selected, u.baseline), (Some(0), Some(3)));
    assert!(u.members.iter().all(|m| m.dl.is_some()));

    // v has only an incorrect member: no selection, no cost.
    let v = &report.tasks["v"];
    assert_eq!((v.selected, v.baseline), (None, None));
    assert_eq!(
        v.members,
        vec![MemberScore {
            dl: v.members[0].dl,
            c: None,
            unfitted: None
        }]
    );

    assert_eq!(report.total_cost, 2 + 1);
    assert_eq!(report.baseline_total_cost, 4 + 8);
    assert_eq!(report.changed, 2);
}

#[test]
fn members_carry_their_growth_by_position_in_the_frontier() {
    let tasks = [task("t", "Int")];
    let frontiers: BTreeMap<String, Vec<String>> =
        [("t".to_owned(), vec![SHORT_SLOW.to_owned()])].into();
    let candidates = [candidate("t", LONG_FAST)];
    let growths = Growths {
        members: [("t".to_owned(), vec![fitted(GrowthClass::Cubic)])].into(),
        candidates: vec![fitted(GrowthClass::Constant)],
    };
    let case = Case {
        tasks: &tasks,
        frontiers,
        candidates: &candidates,
        growths: Some(&growths),
    };
    let out = case.run(&config(5, 1e6), &mut Nothing::default());
    assert_eq!(out.frontiers["t"], vec![LONG_FAST, SHORT_SLOW]);
    assert_eq!(
        out.sources["t"],
        vec![
            Source::Candidate { index: 0 },
            Source::Member {
                task: "t".to_owned(),
                index: 0
            }
        ]
    );
    let t = &out.report.selection.unwrap().tasks["t"];
    assert_eq!(
        t.members.iter().map(|m| m.c).collect::<Vec<_>>(),
        vec![Some(0), Some(5)]
    );
}

#[test]
fn a_report_from_before_selection_was_logged_still_reads() {
    let json = r#"{"programs":[],"compression":"off"}"#;
    let report: dc_driver::IterationReport = serde_json::from_str(json).unwrap();
    assert_eq!(report.selection, None);
}
