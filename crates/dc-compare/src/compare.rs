//! One arm comparison of decision D33, from its spec to its report.
//!
//! A [`Spec`] names the pairs of runs, the primary metric, the tasks and the
//! iterations it counts, and the smallest effect that matters. [`analyse`]
//! reads every run's records and gives a [`Report`]: each pair's
//! difference, the sign-flip test, its interval and the call; the
//! iteration-0 check; the spread between runs; the secondary metrics,
//! labelled exploratory; each run's details; and every deviation found.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use dc_driver::objective::class_cost;
use dc_mcp::state::CostClass;
use dc_tasks::growth::GrowthClass;
use serde::{Deserialize, Serialize};

use crate::records::{GrowthResult, RecordError, RunRecords};
use crate::stats::{self, Call, Cell, Interval, Spread, TestError};

/// The primary metric of a comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Programs passed over programs the batches issued asked for
    /// (dreamcounter-70c43d).
    PassRate,
    /// Programs passed whose growth class is `n log n` or better (rank 3 or
    /// less in D27), over programs the batches issued asked for
    /// (dreamcounter-a86001).
    CheapRate,
}

/// The two arms' labels.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arms {
    /// The arm under test.
    pub test: String,
    /// The control arm.
    pub control: String,
}

/// Where one run's records are.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSpec {
    /// The run's records directory.
    pub records: PathBuf,
    /// The dc-propose log directories to search for its launch summaries.
    #[serde(default)]
    pub logs: Vec<PathBuf>,
}

/// One pair: a run in each arm with the same seed.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairSpec {
    /// The pair's seed.
    pub seed: u64,
    /// The run in the arm under test.
    pub test: RunSpec,
    /// The run in the control arm.
    pub control: RunSpec,
}

/// Settings of `dc_run_status` with the value each arm's runs must have.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmValues {
    /// The arm under test's values, by setting.
    #[serde(default)]
    pub test: BTreeMap<String, serde_json::Value>,
    /// The control arm's values, by setting.
    #[serde(default)]
    pub control: BTreeMap<String, serde_json::Value>,
}

fn default_alpha() -> f64 {
    0.05
}

/// A comparison, as fixed before its runs (D33). Relative paths are taken
/// from the spec file's directory.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// The comparison's name.
    pub name: String,
    /// The primary metric.
    pub metric: Metric,
    /// The tasks the primary metric counts, in the order to report them.
    pub tasks: Vec<String>,
    /// The iterations the primary metric counts.
    pub counted_iterations: Vec<u64>,
    /// The iteration that is a check, not data (D33: iteration 0).
    #[serde(default)]
    pub check_iteration: Option<u64>,
    /// The smallest effect that matters: the null call needs the interval
    /// strictly inside `(-m, +m)`.
    pub smallest_effect: f64,
    /// The test's level.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// The arms' labels.
    pub arms: Arms,
    /// The settings of `dc_run_status` the arms differ in. Any other
    /// setting that differs within a pair is a deviation.
    #[serde(default)]
    pub arm_settings: Vec<String>,
    /// The value each arm's runs must have for some settings of
    /// `dc_run_status`, so a pair with its arms swapped is a deviation.
    #[serde(default)]
    pub arm_values: ArmValues,
    /// The number of pairs fixed before the runs (D33: 12). A spec with
    /// another number is a deviation.
    #[serde(default)]
    pub planned_pairs: Option<usize>,
    /// Accept counted or check iterations that have not finished. Only for
    /// checking the script on runs that stopped early; it is warned about.
    #[serde(default)]
    pub allow_unfinished: bool,
    /// Tasks for the secondary share of passed programs worse than `n`.
    #[serde(default)]
    pub worse_than_n_tasks: Vec<String>,
    /// The pairs, in seed order.
    pub pairs: Vec<PairSpec>,
}

/// Why a comparison cannot be analysed.
#[derive(Debug)]
pub struct CompareError(pub String);

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompareError {}

impl From<RecordError> for CompareError {
    fn from(e: RecordError) -> Self {
        CompareError(e.0)
    }
}

impl From<TestError> for CompareError {
    fn from(e: TestError) -> Self {
        CompareError(e.to_string())
    }
}

/// The rank of a class in D27: its position from `1` (0) to `2ⁿ` (6).
pub fn class_rank(c: CostClass) -> u32 {
    class_cost(match c {
        CostClass::Constant => GrowthClass::Constant,
        CostClass::Log => GrowthClass::Log,
        CostClass::Linear => GrowthClass::Linear,
        CostClass::NLogN => GrowthClass::NLogN,
        CostClass::Quadratic => GrowthClass::Quadratic,
        CostClass::Cubic => GrowthClass::Cubic,
        CostClass::Exponential => GrowthClass::Exponential,
    })
}

/// The largest rank a cheap program's class has: `n log n`.
pub const CHEAP_RANK: u32 = 3;

/// What came in for one task in one iteration of one run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Counts {
    /// Programs the batches issued asked for: batches times
    /// `programs_per_batch`. A batch with no submission counts as programs
    /// that did not pass.
    pub issued: u64,
    /// Programs checked.
    pub programs: u64,
    /// Programs that compiled.
    pub compiled: u64,
    /// Programs that compiled and typed.
    pub typed: u64,
    /// Programs that passed.
    pub passed: u64,
    /// Passed programs whose class has rank [`CHEAP_RANK`] or less. `None`
    /// without growth in the records.
    pub cheap: Option<u64>,
    /// Passed programs with a fitted class.
    pub classed: Option<u64>,
    /// Passed programs with a fitted class worse than `n`.
    pub worse_than_n: Option<u64>,
}

impl Counts {
    fn add(&mut self, o: &Counts) {
        self.issued += o.issued;
        self.programs += o.programs;
        self.compiled += o.compiled;
        self.typed += o.typed;
        self.passed += o.passed;
        let sum = |a: Option<u64>, b: Option<u64>| a.zip(b).map(|(a, b)| a + b);
        self.cheap = sum(self.cheap, o.cheap);
        self.classed = sum(self.classed, o.classed);
        self.worse_than_n = sum(self.worse_than_n, o.worse_than_n);
    }

    fn zero(growth: bool) -> Counts {
        let g = growth.then_some(0);
        Counts {
            cheap: g,
            classed: g,
            worse_than_n: g,
            ..Counts::default()
        }
    }

    /// The metric's numerator.
    fn hits(&self, metric: Metric) -> Option<u64> {
        match metric {
            Metric::PassRate => Some(self.passed),
            Metric::CheapRate => self.cheap,
        }
    }
}

fn ratio(a: u64, b: u64) -> Option<f64> {
    // Counts of programs are far below 2^53, so they are exact in f64.
    (b > 0).then(|| a as f64 / b as f64)
}

/// One run, read and checked.
struct Run {
    records: RunRecords,
    /// Counts by iteration and task.
    counts: BTreeMap<u64, BTreeMap<String, Counts>>,
}

impl Run {
    fn counts(&self, iterations: &[u64], tasks: &[String]) -> Counts {
        let growth = self.records.diversity.is_some();
        let mut total = Counts::zero(growth);
        for i in iterations {
            for t in tasks {
                if let Some(c) = self.counts.get(i).and_then(|m| m.get(t)) {
                    total.add(c);
                }
            }
        }
        total
    }

    fn rate(&self, metric: Metric, iterations: &[u64], tasks: &[String]) -> Option<f64> {
        let c = self.counts(iterations, tasks);
        ratio(c.hits(metric)?, c.issued)
    }
}

/// Reads one run and counts what came in, noting what does not fit the
/// spec.
fn load(
    spec: &Spec,
    base: &Path,
    run: &RunSpec,
    warnings: &mut Vec<String>,
) -> Result<Run, CompareError> {
    let dir = base.join(&run.records);
    let logs: Vec<PathBuf> = run.logs.iter().map(|l| base.join(l)).collect();
    let records = RunRecords::read(&dir, &logs)?;
    let id = records.status.run.clone();
    let names: BTreeSet<&str> = records
        .status
        .tasks
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    for t in spec.tasks.iter().chain(&spec.worse_than_n_tasks) {
        if !names.contains(t.as_str()) {
            return Err(CompareError(format!("run {id} has no task {t}")));
        }
    }
    let per_batch = u64::from(records.status.programs_per_batch);
    if per_batch != 1 {
        warnings.push(format!(
            "run {id} asks for {per_batch} programs per batch, not 1 (D33); \
             its rates are over batches issued times {per_batch}"
        ));
    }
    let mut wanted: Vec<u64> = spec.counted_iterations.clone();
    wanted.extend(spec.check_iteration);
    let need_growth = spec.metric == Metric::CheapRate;
    if need_growth && records.diversity.is_none() {
        return Err(CompareError(format!(
            "run {id} has no diversity.json, and the cheap rate needs dc_diversity with growth"
        )));
    }
    let mut counts = BTreeMap::new();
    for i in &wanted {
        let Some(stats) = records.iteration(*i) else {
            return Err(CompareError(format!(
                "run {id} has not reached iteration {i}; D33 runs the pair again with the \
                 next unused seed"
            )));
        };
        if stats.pending_checks > 0 {
            return Err(CompareError(format!(
                "run {id} iteration {i} has {} submissions waiting to be checked; run \
                 dc_check_pending, then export the records again",
                stats.pending_checks
            )));
        }
        let finished = *i < records.status.iteration;
        if !finished || !stats.quota_in {
            let msg = format!(
                "run {id} iteration {i} {}: batches whose lease has not ended count as fails",
                if finished {
                    "finished without its quota"
                } else {
                    "has not finished"
                }
            );
            if !finished && !spec.allow_unfinished {
                return Err(CompareError(format!(
                    "{msg}. Finish the iteration, or set allow_unfinished to check the script"
                )));
            }
            warnings.push(msg);
        }
        let diversity = records.diversity_of(*i);
        if records.diversity.is_some() && diversity.is_none() {
            let msg = format!("run {id}: diversity.json has no iteration {i}");
            if need_growth {
                return Err(CompareError(msg));
            }
            warnings.push(msg);
        }
        let mut by_task = BTreeMap::new();
        for t in &stats.tasks {
            let mut c = Counts {
                issued: t.batches * per_batch,
                programs: t.programs,
                compiled: t.compiled,
                typed: t.typed,
                passed: t.passed,
                ..Counts::default()
            };
            if let Some(d) = diversity {
                let groups = d.tasks.iter().find(|g| g.task == t.name);
                let groups = groups.map_or(&[][..], |g| g.groups.as_slice());
                let in_groups: u64 = groups.iter().map(|g| g.count).sum();
                if in_groups != t.passed {
                    let msg = format!(
                        "run {id} iteration {i} {}: diversity.json has {in_groups} passed \
                         programs, proposal_stats.json {}; export both after the run's checks",
                        t.name, t.passed
                    );
                    if need_growth {
                        return Err(CompareError(msg));
                    }
                    warnings.push(msg);
                }
                let mut cheap = 0;
                let mut classed = 0;
                let mut worse = 0;
                for g in groups {
                    let Some(growth) = &g.growth else {
                        return Err(CompareError(format!(
                            "run {id} iteration {i} {}: diversity.json has no growth; \
                             run dc_diversity with growth true",
                            t.name
                        )));
                    };
                    match growth {
                        GrowthResult::NotMeasured { reason } if need_growth => {
                            return Err(CompareError(format!(
                                "run {id} iteration {i} {}: a program was not measured \
                                 ({reason}); run dc_diversity with a longer deadline_seconds",
                                t.name
                            )));
                        }
                        GrowthResult::Refused { reason } => warnings.push(format!(
                            "run {id} iteration {i} {}: {} passed programs could not be run \
                             on the growth sweep ({reason}); they count as not cheap",
                            t.name, g.count
                        )),
                        _ => {}
                    }
                    if let Some(class) = growth.class() {
                        let rank = class_rank(class);
                        classed += g.count;
                        if rank <= CHEAP_RANK {
                            cheap += g.count;
                        }
                        if rank > class_rank(CostClass::Linear) {
                            worse += g.count;
                        }
                    }
                }
                c.cheap = Some(cheap);
                c.classed = Some(classed);
                c.worse_than_n = Some(worse);
            }
            by_task.insert(t.name.clone(), c);
        }
        counts.insert(*i, by_task);
    }
    Ok(Run { records, counts })
}

/// A test of the pairs' differences of one metric.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Test {
    /// Pairs in the test.
    pub pairs: usize,
    /// The mean difference: the arm under test minus the control.
    pub mean: f64,
    /// The standard deviation of the differences.
    pub sd: Option<f64>,
    /// The two-sided p of the exact sign-flip test.
    pub p: f64,
    /// The interval for the mean difference at the comparison's level, by
    /// test inversion.
    pub interval: Interval,
}

fn test(d: &[f64], alpha: f64) -> Result<Test, CompareError> {
    Ok(Test {
        pairs: d.len(),
        mean: stats::mean(d),
        sd: stats::sd(d),
        p: stats::sign_flip_p(d)?,
        interval: stats::interval(d, alpha)?,
    })
}

/// One pair's values of a metric.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairValue {
    /// The pair's seed.
    pub seed: u64,
    /// The run in the arm under test.
    pub test_run: String,
    /// The run in the control arm.
    pub control_run: String,
    /// The metric in the arm under test.
    pub test: Option<f64>,
    /// The metric in the control arm.
    pub control: Option<f64>,
    /// The difference, if both are defined.
    pub difference: Option<f64>,
}

/// A metric over the pairs, with its test if every pair has a difference.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Compared {
    /// The metric's name.
    pub name: String,
    /// Each pair's values.
    pub pairs: Vec<PairValue>,
    /// The mean in the arm under test, over pairs where it is defined.
    pub test_mean: Option<f64>,
    /// The mean in the control arm, over pairs where it is defined.
    pub control_mean: Option<f64>,
    /// The test, if every pair has a difference.
    pub test: Option<Test>,
    /// Why there is no test, if there is none.
    pub untested: Option<String>,
}

fn compared(
    name: &str,
    pairs: &[(u64, &Run, &Run)],
    alpha: f64,
    value: &dyn Fn(&Run) -> Option<f64>,
) -> Result<Compared, CompareError> {
    let rows: Vec<PairValue> = pairs
        .iter()
        .map(|(seed, t, c)| {
            let (vt, vc) = (value(t), value(c));
            PairValue {
                seed: *seed,
                test_run: t.records.status.run.clone(),
                control_run: c.records.status.run.clone(),
                test: vt,
                control: vc,
                difference: vt.zip(vc).map(|(a, b)| a - b),
            }
        })
        .collect();
    let defined = |f: &dyn Fn(&PairValue) -> Option<f64>| {
        let v: Vec<f64> = rows.iter().filter_map(f).collect();
        (!v.is_empty()).then(|| stats::mean(&v))
    };
    let test_mean = defined(&|r| r.test);
    let control_mean = defined(&|r| r.control);
    let d: Option<Vec<f64>> = rows.iter().map(|r| r.difference).collect();
    let (tested, untested) = match d {
        Some(d) if !d.is_empty() => (Some(test(&d, alpha)?), None),
        Some(_) => (None, Some("no pairs".to_owned())),
        None => (
            None,
            Some("some pair has no value in an arm (nothing issued, or nothing passed)".to_owned()),
        ),
    };
    Ok(Compared {
        name: name.to_owned(),
        pairs: rows,
        test_mean,
        control_mean,
        test: tested,
        untested,
    })
}

/// The primary result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Primary {
    /// The metric over the pairs, with its test.
    pub metric: Compared,
    /// The call.
    pub call: Call,
}

/// The iteration-0 check.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    /// The iteration checked.
    pub iteration: u64,
    /// The primary metric on that iteration alone, with its test.
    pub metric: Compared,
    /// Whether the test rejects at the comparison's level: then the setup
    /// differs between arms in some other way, and D33 asks for the cause
    /// before the primary result is read.
    pub differs: bool,
}

/// What failed, summed over an arm's runs, the spec's tasks and the counted
/// iterations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Failures {
    /// Programs the batches issued asked for.
    pub issued: u64,
    /// Asked for but never checked: no submission, or fewer programs.
    pub not_submitted: u64,
    /// Checked but did not compile.
    pub not_compiled: u64,
    /// Compiled but did not have the task's type.
    pub not_typed: u64,
    /// Typed but failed an example.
    pub failed_examples: u64,
    /// Passed.
    pub passed: u64,
}

impl Failures {
    fn of(c: &Counts) -> Failures {
        Failures {
            issued: c.issued,
            not_submitted: c.issued.saturating_sub(c.programs),
            not_compiled: c.programs.saturating_sub(c.compiled),
            not_typed: c.compiled.saturating_sub(c.typed),
            failed_examples: c.typed.saturating_sub(c.passed),
            passed: c.passed,
        }
    }

    fn add(&mut self, o: &Failures) {
        self.issued += o.issued;
        self.not_submitted += o.not_submitted;
        self.not_compiled += o.not_compiled;
        self.not_typed += o.not_typed;
        self.failed_examples += o.failed_examples;
        self.passed += o.passed;
    }
}

/// Tokens and cost of a run's sessions, from dc-propose's summaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Tokens {
    /// Sessions of every kind.
    pub sessions: u64,
    /// Naming sessions.
    pub naming_sessions: u64,
    /// Sessions with a reported usage.
    pub with_usage: u64,
    /// Input tokens, cache reads and writes included.
    pub input_tokens: u64,
    /// Output tokens.
    pub output_tokens: u64,
    /// The cost Claude Code reports at API prices, in USD.
    pub cost_usd: f64,
    /// Launches that were interrupted.
    pub interrupted_launches: u64,
}

fn tokens(r: &RunRecords) -> Tokens {
    let mut t = Tokens::default();
    for l in &r.launches {
        t.interrupted_launches += u64::from(l.interrupted);
        for s in &l.sessions {
            t.sessions += 1;
            t.naming_sessions += u64::from(s.kind == "naming");
            if let Some(u) = s.usage {
                t.with_usage += 1;
                t.input_tokens +=
                    u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens;
                t.output_tokens += u.output_tokens;
            }
            t.cost_usd += s.total_cost_usd.unwrap_or(0.0);
        }
    }
    t
}

/// One iteration of one run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IterationRow {
    /// The iteration.
    pub iteration: u64,
    /// Whether it is counted, or the check.
    pub counted: bool,
    /// The spec's tasks solved.
    pub tasks_solved: u64,
    /// Each spec task's counts.
    pub tasks: BTreeMap<String, Counts>,
    /// Programs that passed, over all the run's tasks.
    pub passed_all_tasks: u64,
    /// Of those, the programs that refer to a learned entry.
    pub using_library: u64,
    /// Each entry referred to, with how many passed programs refer to it.
    pub entries_used: Vec<(u64, Option<String>, u64)>,
    /// Foreign submissions kept, and refused: left out of every rate.
    pub foreign: u64,
    /// Late and repeat submissions: kept, not checked.
    pub late_or_repeat: u64,
}

/// One library entry of a run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Entry {
    /// Its index.
    pub index: u64,
    /// Its arity.
    pub arity: u32,
    /// Its name, if any.
    pub name: Option<String>,
    /// Its doc, if any.
    pub doc: Option<String>,
    /// The iteration whose end added it.
    pub iteration_added: u64,
}

/// One run's details.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunRow {
    /// The run's id.
    pub run: String,
    /// Its arm's label.
    pub arm: String,
    /// The pair's seed.
    pub seed: u64,
    /// Where its records are.
    pub records: String,
    /// Each counted or checked iteration.
    pub iterations: Vec<IterationRow>,
    /// Its library at its end, if recorded.
    pub library: Option<Vec<Entry>>,
    /// What failed, over the spec's tasks and the counted iterations.
    pub failures: Failures,
    /// Its sessions' tokens and cost.
    pub tokens: Tokens,
}

fn run_row(spec: &Spec, run: &Run, arm: &str, seed: u64) -> RunRow {
    let r = &run.records;
    let mut wanted: Vec<u64> = spec.counted_iterations.clone();
    wanted.extend(spec.check_iteration);
    wanted.sort_unstable();
    wanted.dedup();
    let iterations = wanted
        .iter()
        .filter_map(|i| {
            let stats = r.iteration(*i)?;
            let tasks: BTreeMap<String, Counts> = spec
                .tasks
                .iter()
                .filter_map(|t| Some((t.clone(), *run.counts.get(i)?.get(t)?)))
                .collect();
            Some(IterationRow {
                iteration: *i,
                counted: spec.counted_iterations.contains(i),
                tasks_solved: stats
                    .tasks
                    .iter()
                    .filter(|t| t.solved && spec.tasks.contains(&t.name))
                    .count() as u64,
                tasks,
                passed_all_tasks: stats.library_uptake.passed,
                using_library: stats.library_uptake.using_library,
                entries_used: stats
                    .library_uptake
                    .entries
                    .iter()
                    .map(|e| (e.entry, e.name.clone(), e.programs))
                    .collect(),
                foreign: stats.foreign + stats.refused_foreign,
                late_or_repeat: stats.late + stats.repeats,
            })
        })
        .collect();
    let mut failures = Failures::default();
    for i in &spec.counted_iterations {
        for t in &spec.tasks {
            if let Some(c) = run.counts.get(i).and_then(|m| m.get(t)) {
                failures.add(&Failures::of(c));
            }
        }
    }
    RunRow {
        run: r.status.run.clone(),
        arm: arm.to_owned(),
        seed,
        records: r.dir.display().to_string(),
        iterations,
        library: r.library.as_ref().map(|l| {
            l.entries
                .iter()
                .map(|e| Entry {
                    index: e.index,
                    arity: e.arity,
                    name: e.name.clone(),
                    doc: e.doc.clone(),
                    iteration_added: e.iteration_added,
                })
                .collect()
        }),
        failures,
        tokens: tokens(r),
    }
}

/// The spread between runs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpreadReport {
    /// The standard deviation of the pairs' primary differences.
    pub difference_sd: Option<f64>,
    /// The spread on the logit scale, from each spec task's primary rate in
    /// each run, over the counted iterations, with arms as separate groups.
    pub logit: Option<Spread>,
}

/// The report of a comparison.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    /// The comparison's name.
    pub name: String,
    /// The primary metric.
    pub metric: Metric,
    /// The arms.
    pub arms: Arms,
    /// The test's level.
    pub alpha: f64,
    /// The smallest effect that matters.
    pub smallest_effect: f64,
    /// The tasks the primary metric counts.
    pub tasks: Vec<String>,
    /// The iterations it counts.
    pub counted_iterations: Vec<u64>,
    /// The primary result.
    pub primary: Primary,
    /// The iteration-0 check, if the spec names one.
    pub check: Option<Check>,
    /// The spread between runs.
    pub spread: SpreadReport,
    /// Secondary metrics. Exploratory: they get no call.
    pub exploratory: Vec<Compared>,
    /// What failed, summed over each arm's runs: the arm under test, then
    /// the control.
    pub failures: [Failures; 2],
    /// Each run.
    pub runs: Vec<RunRow>,
    /// Differences between the runs of a pair, or from the spec, that D33
    /// records as deviations.
    pub deviations: Vec<String>,
    /// Anything else that may make a figure less than it seems.
    pub warnings: Vec<String>,
    /// What D33 asks for that these records cannot give, and why.
    pub not_computed: Vec<String>,
}

/// Compares two runs of a pair: the seed, the tasks, and the settings.
fn pair_deviations(spec: &Spec, seed: u64, t: &Run, c: &Run, out: &mut Vec<String>) {
    for (arm, run) in [(&spec.arms.test, t), (&spec.arms.control, c)] {
        if run.records.status.seed != seed {
            out.push(format!(
                "pair {seed}: the {arm} run {} has seed {}",
                run.records.status.run, run.records.status.seed
            ));
        }
    }
    let names = |r: &Run| -> Vec<String> {
        r.records
            .status
            .tasks
            .iter()
            .map(|t| t.name.clone())
            .collect()
    };
    if names(t) != names(c) {
        out.push(format!(
            "pair {seed}: the runs have different tasks: {:?} and {:?}",
            names(t),
            names(c)
        ));
    }
    for (arm, run, values) in [
        (&spec.arms.test, t, &spec.arm_values.test),
        (&spec.arms.control, c, &spec.arm_values.control),
    ] {
        let settings = run.records.settings();
        for (k, want) in values {
            let got = settings.get(k);
            if got != Some(want) {
                out.push(format!(
                    "pair {seed}: the {arm} run {} has {k} {}, but its arm has {want}",
                    run.records.status.run,
                    got.map_or("absent".to_owned(), ToString::to_string),
                ));
            }
        }
    }
    let (st, sc) = (t.records.settings(), c.records.settings());
    let keys: BTreeSet<&String> = st.keys().chain(sc.keys()).collect();
    for k in keys {
        let (a, b) = (st.get(k), sc.get(k));
        let differs = a != b;
        let expected = spec.arm_settings.contains(k);
        if differs && !expected {
            out.push(format!(
                "pair {seed}: setting {k} differs: {} in {}, {} in {}",
                a.map_or("absent".to_owned(), ToString::to_string),
                t.records.status.run,
                b.map_or("absent".to_owned(), ToString::to_string),
                c.records.status.run
            ));
        } else if !differs && expected {
            out.push(format!(
                "pair {seed}: setting {k} is the same in both runs, but the arms should differ in it"
            ));
        }
    }
}

/// Analyses the comparison `spec`, whose relative paths start at `base`.
///
/// # Errors
///
/// If a run's records cannot be read, a run lacks a task or iteration the
/// spec counts, the cheap rate lacks growth, the spec is malformed, or the
/// primary metric is undefined for some run.
pub fn analyse(spec: &Spec, base: &Path) -> Result<Report, CompareError> {
    if spec.pairs.is_empty() {
        return Err(CompareError("the spec has no pairs".to_owned()));
    }
    if spec.tasks.is_empty() || spec.counted_iterations.is_empty() {
        return Err(CompareError(
            "the spec needs at least one task and one counted iteration".to_owned(),
        ));
    }
    // Written so that NaN fails too.
    let valid = spec.alpha > 0.0 && spec.alpha < 1.0 && spec.smallest_effect > 0.0;
    if !valid {
        return Err(CompareError(
            "alpha must be in (0, 1) and smallest_effect above 0".to_owned(),
        ));
    }
    if let Some(i) = spec.check_iteration {
        if spec.counted_iterations.contains(&i) {
            return Err(CompareError(format!(
                "iteration {i} is both counted and the check"
            )));
        }
    }
    let mut seeds = BTreeSet::new();
    for p in &spec.pairs {
        if !seeds.insert(p.seed) {
            return Err(CompareError(format!("seed {} is in two pairs", p.seed)));
        }
    }
    for (what, list) in [
        ("task", &spec.tasks),
        ("worse-than-n task", &spec.worse_than_n_tasks),
    ] {
        let mut seen = BTreeSet::new();
        for t in list {
            if !seen.insert(t) {
                return Err(CompareError(format!("{what} {t} is listed twice")));
            }
        }
    }
    let mut seen = BTreeSet::new();
    for i in &spec.counted_iterations {
        if !seen.insert(i) {
            return Err(CompareError(format!("iteration {i} is counted twice")));
        }
    }
    let mut warnings = Vec::new();
    let mut deviations = Vec::new();
    // D33: each comparison uses its seeds in order and skips none, and runs
    // a fixed number of pairs.
    let first = spec.pairs[0].seed;
    let in_order = spec
        .pairs
        .iter()
        .zip(0u64..)
        .all(|(p, k)| first.checked_add(k) == Some(p.seed));
    if !in_order {
        let listed: Vec<u64> = spec.pairs.iter().map(|p| p.seed).collect();
        deviations.push(format!(
            "the pairs' seeds {listed:?} are not consecutive from {first} in order (D33)"
        ));
    }
    match spec.planned_pairs {
        Some(n) if n != spec.pairs.len() => deviations.push(format!(
            "the spec has {} pairs, but {n} were planned",
            spec.pairs.len()
        )),
        Some(_) => {}
        None => {
            warnings.push("the spec names no planned_pairs, so the count is not checked".to_owned())
        }
    }
    if spec.check_iteration.is_none() {
        warnings.push("the spec names no check_iteration; D33 checks iteration 0".to_owned());
    }
    if spec.allow_unfinished {
        warnings.push(
            "allow_unfinished is set: iterations that have not finished are counted".to_owned(),
        );
    }
    if !spec.arm_settings.is_empty()
        && spec.arm_values.test.is_empty()
        && spec.arm_values.control.is_empty()
    {
        warnings.push(
            "the spec names no arm_values, so a pair with its arms swapped is not found".to_owned(),
        );
    }
    let mut loaded: Vec<(u64, Run, Run)> = Vec::with_capacity(spec.pairs.len());
    for p in &spec.pairs {
        let t = load(spec, base, &p.test, &mut warnings)?;
        let c = load(spec, base, &p.control, &mut warnings)?;
        pair_deviations(spec, p.seed, &t, &c, &mut deviations);
        loaded.push((p.seed, t, c));
    }
    let mut ids = BTreeSet::new();
    for (_, t, c) in &loaded {
        for r in [t, c] {
            if !ids.insert(r.records.status.run.clone()) {
                return Err(CompareError(format!(
                    "run {} is in more than one place in the spec",
                    r.records.status.run
                )));
            }
        }
    }
    let pairs: Vec<(u64, &Run, &Run)> = loaded.iter().map(|(s, t, c)| (*s, t, c)).collect();
    let metric = spec.metric;
    let counted = &spec.counted_iterations;
    let tasks = &spec.tasks;
    let primary_metric = compared(
        match metric {
            Metric::PassRate => "pass rate",
            Metric::CheapRate => "cheap rate",
        },
        &pairs,
        spec.alpha,
        &|r| r.rate(metric, counted, tasks),
    )?;
    let Some(primary_test) = &primary_metric.test else {
        return Err(CompareError(format!(
            "the primary metric is undefined for some run: {}",
            primary_metric.untested.clone().unwrap_or_default()
        )));
    };
    let call = stats::call(
        primary_test.mean,
        primary_test.p,
        &primary_test.interval,
        spec.alpha,
        spec.smallest_effect,
    );
    if primary_test.interval.has_gaps {
        warnings.push(
            "the primary interval's set of shifts has gaps; the ends given are its hull".to_owned(),
        );
    }
    let check = match spec.check_iteration {
        None => None,
        Some(i) => {
            let m = compared(
                &format!("{} in iteration {i}", primary_metric.name),
                &pairs,
                spec.alpha,
                &|r| r.rate(metric, &[i], tasks),
            )?;
            let differs = m.test.as_ref().is_some_and(|t| t.p <= spec.alpha);
            if differs {
                warnings.push(format!(
                    "iteration {i} differs between arms at p <= {}: D33 asks for the cause \
                     before the primary result is read",
                    spec.alpha
                ));
            }
            Some(Check {
                iteration: i,
                metric: m,
                differs,
            })
        }
    };
    let mut groups: Vec<Vec<Cell>> = Vec::new();
    for arm in 0..2 {
        for t in tasks {
            let cells = pairs
                .iter()
                .filter_map(|(_, a, b)| {
                    let run = if arm == 0 { a } else { b };
                    let c = run.counts(counted, std::slice::from_ref(t));
                    Some(Cell {
                        passed: c.hits(metric)?,
                        issued: c.issued,
                    })
                })
                .collect();
            groups.push(cells);
        }
    }
    let differences: Vec<f64> = primary_metric
        .pairs
        .iter()
        .filter_map(|p| p.difference)
        .collect();
    let spread = SpreadReport {
        difference_sd: stats::sd(&differences),
        logit: stats::logit_spread(&groups),
    };
    let mut exploratory = Vec::new();
    for t in tasks {
        let one = std::slice::from_ref(t);
        exploratory.push(compared(
            &format!("pass rate of {t}"),
            &pairs,
            spec.alpha,
            &|r| r.rate(Metric::PassRate, counted, one),
        )?);
    }
    exploratory.push(compared(
        "tasks solved per counted iteration",
        &pairs,
        spec.alpha,
        &|r| {
            let solved: u64 = counted
                .iter()
                .filter_map(|i| r.counts.get(i))
                .map(|m| {
                    tasks
                        .iter()
                        .filter(|t| m.get(*t).is_some_and(|c| c.passed > 0))
                        .count() as u64
                })
                .sum();
            Some(solved as f64 / counted.len() as f64)
        },
    )?);
    if metric == Metric::CheapRate {
        for t in tasks {
            let one = std::slice::from_ref(t);
            exploratory.push(compared(
                &format!("cheap rate of {t}"),
                &pairs,
                spec.alpha,
                &|r| r.rate(Metric::CheapRate, counted, one),
            )?);
        }
        exploratory.push(compared(
            "cheap programs per passed program",
            &pairs,
            spec.alpha,
            &|r| {
                let c = r.counts(counted, tasks);
                ratio(c.cheap?, c.passed)
            },
        )?);
    }
    if !spec.worse_than_n_tasks.is_empty() {
        let w = &spec.worse_than_n_tasks;
        exploratory.push(compared(
            &format!(
                "share of passed programs with a class worse than n, on {}",
                w.join(", ")
            ),
            &pairs,
            spec.alpha,
            &|r| {
                let c = r.counts(counted, w);
                ratio(c.worse_than_n?, c.passed)
            },
        )?);
    }
    exploratory.push(compared(
        "share of passed programs that refer to a learned entry, all tasks",
        &pairs,
        spec.alpha,
        &|r| {
            let (mut used, mut passed) = (0, 0);
            for i in counted {
                if let Some(s) = r.records.iteration(*i) {
                    used += s.library_uptake.using_library;
                    passed += s.library_uptake.passed;
                }
            }
            ratio(used, passed)
        },
    )?);
    let with_logs = pairs
        .iter()
        .all(|(_, a, b)| !a.records.launches.is_empty() && !b.records.launches.is_empty());
    if with_logs {
        exploratory.push(compared(
            "reported cost per run, USD (whole run, every iteration)",
            &pairs,
            spec.alpha,
            &|r| Some(tokens(&r.records).cost_usd),
        )?);
        exploratory.push(compared(
            "output tokens per run (whole run, every iteration)",
            &pairs,
            spec.alpha,
            &|r| Some(tokens(&r.records).output_tokens as f64),
        )?);
    } else {
        warnings.push(
            "some run has no launch summaries, so tokens and cost are not compared".to_owned(),
        );
    }
    let mut failures = [Failures::default(), Failures::default()];
    let mut runs = Vec::new();
    for (seed, t, c) in &pairs {
        let rt = run_row(spec, t, &spec.arms.test, *seed);
        let rc = run_row(spec, c, &spec.arms.control, *seed);
        failures[0].add(&rt.failures);
        failures[1].add(&rc.failures);
        runs.push(rt);
        runs.push(rc);
    }
    let mut not_computed = vec![
        "causes of failure finer than compile, type and examples (such as positions out of \
         order): the records count only those stages; the checks' messages are in the journal"
            .to_owned(),
        "tokens and cost for each iteration: launch summaries give them for each run only \
         (dreamcounter-f3bcfa)"
            .to_owned(),
    ];
    if metric == Metric::CheapRate {
        not_computed.push(
            "manipulation check, C(selected), DL(L) and sum DL(selected): the records hold no \
             frontier selections yet (dreamcounter-211560)"
                .to_owned(),
        );
    }
    Ok(Report {
        name: spec.name.clone(),
        metric,
        arms: spec.arms.clone(),
        alpha: spec.alpha,
        smallest_effect: spec.smallest_effect,
        tasks: tasks.clone(),
        counted_iterations: counted.clone(),
        primary: Primary {
            metric: primary_metric,
            call,
        },
        check,
        spread,
        exploratory,
        failures,
        runs,
        deviations,
        warnings,
        not_computed,
    })
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
