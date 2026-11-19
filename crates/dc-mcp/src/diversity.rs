//! How varied the programs that passed each task are (bead
//! dreamcounter-07f465, epic dreamcounter-664459).
//!
//! The library-learning loop can only keep a cheaper program if one is
//! proposed. The pilot run found that two proposer sessions often wrote the
//! same program for a task. [`report`] measures that for each iteration and
//! task, over the programs that passed:
//!
//! * **distinct programs:** distinct core terms as stored. Core terms use de
//!   Bruijn indices, so programs that differ only in the names of their
//!   variables, or in surface sugar, count once.
//! * **distinct expanded programs:** distinct terms once prelude and learned
//!   library references are written out, each call reduced with its
//!   arguments put in. A program that calls a library function counts as
//!   the same as one that writes out what the call reduces to. A
//!   proposer's own redex, such as a lambda applied where it is written,
//!   is not reduced.
//! * **mean distance:** the mean over pairs of passed programs of the
//!   Jaccard distance between the sets of their compound subterms (every
//!   subterm that has a subterm of its own, and the whole program however
//!   small), as stored. Terms are
//!   hash-consed, so two programs share a subterm exactly when they contain
//!   the same piece of code. 0 means the two are the same term; 1 means they
//!   share no compound piece. Library and prelude references are atoms
//!   here, so a call of a library function and the function written out
//!   count as different. Pairs of identical programs count, with distance
//!   0.
//! * **growth class**, if asked for: each distinct expanded program is run
//!   on its task's growth sweep ([`crate::cost::measure_program`]), and the
//!   report says how many distinct classes estimator C names. A program
//!   whose expansion is that of a member of the iteration's kept frontier
//!   with a measured or incorrect cost keeps that cost, since the expansion
//!   is what runs. A program is measured at most once in a call, however
//!   many iterations it passed in.
//!
//! Programs that did not pass are left out. Programs come from the run's
//! journal: the checks of the batches issued in each iteration
//! ([`passed_programs`]). Only one submission of each batch is checked, so
//! each program counted is one a batch asked for.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use dc_core::term::children;
use dc_core::{sexpr, Arena, TermId};
use rmcp::schemars;
use serde::Serialize;

use crate::cost::measure_program;
use crate::state::{Check, CostClass, CostVerdict, Event, ProgramCost, RunId, RunState};
use crate::store::{Store, StoreError};
use crate::verify::Verifier;

/// The programs that passed, by iteration and then task, each task's in the
/// order their checks were written.
pub type PassedPrograms = BTreeMap<u64, BTreeMap<String, Vec<String>>>;

/// Reads the programs that passed in `run` from its journal.
///
/// # Errors
///
/// The store's error if the journal cannot be read.
pub fn passed_programs(store: &Store, run: &RunId) -> Result<PassedPrograms, StoreError> {
    let mut batches: HashMap<u64, (u64, String)> = HashMap::new();
    let mut passed = PassedPrograms::new();
    store.for_each_event(run, |event| {
        match event {
            Event::BatchIssued(b) => {
                batches.insert(b.batch, (b.iteration, b.task));
            }
            Event::Checked(c) => {
                // A run's state refuses a check of a batch it has not
                // issued, so every check has its batch.
                if let Some((iteration, task)) = batches.get(&c.batch) {
                    for check in c.checks {
                        if let Check::Passed { program } = check {
                            passed
                                .entry(*iteration)
                                .or_default()
                                .entry(task.clone())
                                .or_default()
                                .push(program);
                        }
                    }
                }
            }
            _ => {}
        }
        ControlFlow::Continue(())
    })?;
    Ok(passed)
}

/// What a distinct expanded program costs, in short.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GrowthResult {
    /// It ran over the whole sweep; estimator C's verdict.
    Verdict(CostVerdict),
    /// It gave a wrong output or an error on a generated input.
    Incorrect,
    /// It could not be run on the sweep.
    Refused {
        /// Why.
        reason: String,
    },
    /// The time for measuring ran out, or the call was cancelled.
    NotMeasured {
        /// Why.
        reason: String,
    },
}

impl GrowthResult {
    fn of(cost: &ProgramCost) -> GrowthResult {
        match cost {
            ProgramCost::Measured { verdict, .. } => GrowthResult::Verdict(verdict.clone()),
            ProgramCost::Incorrect { .. } => GrowthResult::Incorrect,
            ProgramCost::Refused { reason } => GrowthResult::Refused {
                reason: reason.clone(),
            },
            ProgramCost::NotMeasured { reason } => GrowthResult::NotMeasured {
                reason: reason.clone(),
            },
        }
    }

    /// The class C names, if it names one.
    fn class(&self) -> Option<CostClass> {
        match self {
            GrowthResult::Verdict(CostVerdict::Class(c)) => Some(*c),
            _ => None,
        }
    }
}

/// One distinct expanded program of a task.
#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct ProgramGroup {
    /// The first of the programs that expand to it, as stored: a core term
    /// as an s-expression.
    pub program: String,
    /// How many passed programs expand to it.
    pub count: u64,
    /// What it costs on the task's growth sweep, if growth was asked for.
    pub growth: Option<GrowthResult>,
}

/// How varied one task's passed programs are in one iteration.
#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct TaskDiversity {
    /// The task.
    pub task: String,
    /// Programs that passed.
    pub passed: u64,
    /// Distinct core terms among them, as stored.
    pub distinct_programs: u64,
    /// Distinct terms once prelude and library references are expanded.
    pub distinct_expanded: u64,
    /// The mean Jaccard distance between the compound subterms of each pair
    /// of passed programs. None with fewer than two.
    pub mean_distance: Option<f64>,
    /// How many distinct growth classes C names among the groups. None if
    /// growth was not asked for.
    pub distinct_classes: Option<u64>,
    /// How many groups have no class: C named none (too noisy, no fit, not
    /// fitted), or the program was incorrect, refused or not measured. None
    /// if growth was not asked for.
    pub groups_without_class: Option<u64>,
    /// The distinct expanded programs, most common first, then in the order
    /// they first passed.
    pub groups: Vec<ProgramGroup>,
}

/// The tasks of one iteration, summed up.
#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct DiversitySummary {
    /// Tasks with at least one passed program.
    pub tasks: u64,
    /// Programs that passed, over all tasks.
    pub passed: u64,
    /// Distinct expanded programs, summed over tasks.
    pub distinct_expanded: u64,
    /// Tasks with two or more passed programs that all expand to the same
    /// term.
    pub tasks_all_same: u64,
    /// The mean of the tasks' mean distances, over tasks that have one.
    pub mean_distance: Option<f64>,
    /// Tasks whose groups C puts in two or more classes. None if growth was
    /// not asked for.
    pub tasks_with_two_classes: Option<u64>,
}

/// How varied one iteration's passed programs are.
#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct IterationDiversity {
    /// The iteration.
    pub iteration: u64,
    /// Each task of the run, in the run's order.
    pub tasks: Vec<TaskDiversity>,
    /// All tasks together.
    pub summary: DiversitySummary,
}

/// The result of `dc_diversity`.
#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct DiversityReport {
    /// The run's id.
    pub run: String,
    /// Each iteration asked for, in order.
    pub iterations: Vec<IterationDiversity>,
}

/// When to stop measuring growth.
#[derive(Clone, Copy, Debug)]
pub struct GrowthLimits<'a> {
    /// No run of a program starts after this.
    pub deadline: Instant,
    /// No run of a program starts once this is set.
    pub cancel: &'a AtomicBool,
}

/// Why measuring stopped, as a group's growth says.
const OUT_OF_TIME: &str = "the time for measuring ran out";
const CANCELLED: &str = "the call was cancelled";

/// The diversity of `iteration` of the run, or of every iteration up to the
/// current one. `passed` is [`passed_programs`] of the run whose state is
/// `state`. With `growth`, each distinct expanded program is measured on
/// its task's sweep, within the limits.
///
/// # Errors
///
/// If `iteration` has not started, or the run's library does not load.
pub fn report(
    run: &RunId,
    state: &RunState,
    passed: &PassedPrograms,
    iteration: Option<u64>,
    growth: Option<GrowthLimits<'_>>,
) -> Result<DiversityReport, String> {
    let current = state.iteration();
    let wanted: Vec<u64> = match iteration {
        Some(i) if i > current => {
            return Err(format!(
                "iteration {i} has not started; the run is in iteration {current}"
            ))
        }
        Some(i) => vec![i],
        None => (0..=current).collect(),
    };
    // The library only grows, and its entries keep their index, so the
    // whole library expands the programs of every iteration.
    let base = Verifier::new(state.library(), state.config().max_memory)
        .map_err(|e| format!("the run's library does not load: {e}"))?;
    let mut expanding = base.clone();
    // Costs measured in this call, by task and expanded term, so a program
    // that passed in several iterations is measured once.
    let mut measured: HashMap<(String, TermId), ProgramCost> = HashMap::new();
    let empty = BTreeMap::new();
    let mut iterations = Vec::with_capacity(wanted.len());
    for i in wanted {
        let by_task = passed.get(&i).unwrap_or(&empty);
        let kept = state.kept_frontiers(i);
        let mut tasks: Vec<TaskDiversity> = Vec::with_capacity(state.tasks().len());
        for task in state.tasks() {
            let programs = by_task.get(task).map_or(&[][..], Vec::as_slice);
            // The kept frontier's measured costs, by expanded term. Stitch
            // may have rewritten a kept program to call a new entry, so its
            // text can differ from the program that passed; its expansion,
            // which is what runs, does not.
            let known: Vec<(TermId, ProgramCost)> = kept
                .and_then(|k| k.get(task))
                .map_or(&[][..], Vec::as_slice)
                .iter()
                .filter_map(|m| {
                    let cost = m.cost.as_ref().filter(|c| {
                        matches!(
                            c,
                            ProgramCost::Measured { .. } | ProgramCost::Incorrect { .. }
                        )
                    })?;
                    let term = expanding.expand_stored(&m.program).ok()?;
                    Some((term, cost.clone()))
                })
                .collect();
            let measure = growth.map(|limits| {
                let base = &base;
                let known = &known;
                let measured = &mut measured;
                move |program: &str, term: Option<TermId>| -> ProgramCost {
                    if let Some((_, cost)) = known.iter().find(|(t, _)| Some(*t) == term) {
                        return cost.clone();
                    }
                    let key = term.map(|t| (task.clone(), t));
                    if let Some(cost) = key.as_ref().and_then(|k| measured.get(k)) {
                        return cost.clone();
                    }
                    let mut stop = || {
                        limits.cancel.load(Ordering::SeqCst) || Instant::now() >= limits.deadline
                    };
                    let cost = match measure_program(base, task, program, &mut stop, OUT_OF_TIME) {
                        // The flag is only ever set, so if it is set now it
                        // may be why measuring stopped.
                        ProgramCost::NotMeasured { .. } if limits.cancel.load(Ordering::SeqCst) => {
                            ProgramCost::NotMeasured {
                                reason: CANCELLED.to_owned(),
                            }
                        }
                        cost => cost,
                    };
                    if let (Some(k), false) = (key, matches!(cost, ProgramCost::NotMeasured { .. }))
                    {
                        measured.insert(k, cost.clone());
                    }
                    cost
                }
            });
            tasks.push(task_diversity(task, programs, &mut expanding, measure));
        }
        let summary = summarise(&tasks);
        iterations.push(IterationDiversity {
            iteration: i,
            tasks,
            summary,
        });
    }
    Ok(DiversityReport {
        run: run.as_str().to_owned(),
        iterations,
    })
}

/// The diversity of `programs`, the passed programs of `task`. `expanding`
/// expands them; its arena grows. `measure` gives a program's cost.
fn task_diversity(
    task: &str,
    programs: &[String],
    expanding: &mut Verifier,
    mut measure: Option<impl FnMut(&str, Option<TermId>) -> ProgramCost>,
) -> TaskDiversity {
    // Distinct stored terms, and distinct expanded ones. A program that does
    // not expand (a passed program always does) is its own group.
    let distinct_programs = programs.iter().collect::<HashSet<_>>().len() as u64;
    let mut index: HashMap<Result<TermId, &str>, usize> = HashMap::new();
    let mut groups: Vec<(ProgramGroup, usize, Option<TermId>)> = Vec::new();
    for (order, program) in programs.iter().enumerate() {
        let key = expanding
            .expand_stored(program)
            .map_err(|_| program.as_str());
        match index.get(&key) {
            Some(&g) => groups[g].0.count += 1,
            None => {
                index.insert(key, groups.len());
                groups.push((
                    ProgramGroup {
                        program: program.clone(),
                        count: 1,
                        growth: None,
                    },
                    order,
                    key.ok(),
                ));
            }
        }
    }
    groups.sort_by(|(a, ai, _), (b, bi, _)| b.count.cmp(&a.count).then(ai.cmp(bi)));
    let classes = measure.as_mut().map(|measure| {
        let mut classes: Vec<CostClass> = Vec::new();
        let mut without = 0u64;
        for (g, _, term) in &mut groups {
            let result = GrowthResult::of(&measure(&g.program, *term));
            match result.class() {
                Some(c) if !classes.contains(&c) => classes.push(c),
                Some(_) => {}
                None => without += 1,
            }
            g.growth = Some(result);
        }
        (classes.len() as u64, without)
    });
    let groups: Vec<ProgramGroup> = groups.into_iter().map(|(g, _, _)| g).collect();
    TaskDiversity {
        task: task.to_owned(),
        passed: programs.len() as u64,
        distinct_programs,
        distinct_expanded: groups.len() as u64,
        mean_distance: mean_distance(programs),
        distinct_classes: classes.map(|(c, _)| c),
        groups_without_class: classes.map(|(_, w)| w),
        groups,
    }
}

/// The compound subterms of `term`: every subterm that has a subterm, and
/// `term` itself, so that two different atoms are not the same.
fn compound_subterms(arena: &Arena, term: TermId) -> HashSet<TermId> {
    let mut seen = HashSet::from([term]);
    let mut stack = children(&arena.get(term));
    while let Some(t) = stack.pop() {
        let kids = children(&arena.get(t));
        if kids.is_empty() || !seen.insert(t) {
            continue;
        }
        stack.extend(kids);
    }
    seen
}

/// The Jaccard distance between two sets: 0 if both are empty.
fn jaccard_distance(a: &HashSet<TermId>, b: &HashSet<TermId>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    let shared = a.intersection(b).count();
    1.0 - shared as f64 / union as f64
}

/// The mean Jaccard distance between the compound subterms of each pair of
/// `programs`, parsed as stored into one arena. None with fewer than two
/// programs that parse.
fn mean_distance(programs: &[String]) -> Option<f64> {
    let mut arena = Arena::new();
    let terms: Vec<TermId> = programs
        .iter()
        .filter_map(|p| sexpr::parse(&mut arena, p).ok())
        .collect();
    let sets: Vec<HashSet<TermId>> = terms
        .into_iter()
        .map(|t| compound_subterms(&arena, t))
        .collect();
    if sets.len() < 2 {
        return None;
    }
    let mut total = 0.0;
    let mut pairs = 0u64;
    for (i, a) in sets.iter().enumerate() {
        for b in &sets[i + 1..] {
            total += jaccard_distance(a, b);
            pairs += 1;
        }
    }
    Some(total / pairs as f64)
}

/// The mean of some values, or None if there are none.
fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0u64), |(s, n), v| (s + v, n + 1));
    (n > 0).then(|| sum / n as f64)
}

fn summarise(tasks: &[TaskDiversity]) -> DiversitySummary {
    DiversitySummary {
        tasks: tasks.iter().filter(|t| t.passed > 0).count() as u64,
        passed: tasks.iter().map(|t| t.passed).sum(),
        distinct_expanded: tasks.iter().map(|t| t.distinct_expanded).sum(),
        tasks_all_same: tasks
            .iter()
            .filter(|t| t.passed >= 2 && t.distinct_expanded == 1)
            .count() as u64,
        mean_distance: mean(tasks.iter().filter_map(|t| t.mean_distance)),
        tasks_with_two_classes: tasks
            .iter()
            .map(|t| t.distinct_classes.map(|c| u64::from(c >= 2)))
            .sum(),
    }
}

#[cfg(test)]
#[path = "diversity_tests.rs"]
mod tests;
