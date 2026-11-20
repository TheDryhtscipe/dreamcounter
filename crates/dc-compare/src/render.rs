//! The report as text, for reading and for the results log.

use std::fmt::Write as _;

use crate::compare::{Compared, Report};
use crate::stats::{Call, Interval};

fn num(x: Option<f64>) -> String {
    x.map_or("-".to_owned(), |v| format!("{v:.4}"))
}

fn interval(i: &Interval) -> String {
    let end = |e: Option<f64>, open: &str| e.map_or(open.to_owned(), |v| format!("{v:+.4}"));
    format!(
        "[{}, {}]{}",
        end(i.lower, "-inf"),
        end(i.upper, "+inf"),
        if i.has_gaps {
            " (hull; the set has gaps)"
        } else {
            ""
        }
    )
}

fn call(c: Call) -> &'static str {
    match c {
        Call::Positive => "POSITIVE",
        Call::Negative => "NEGATIVE",
        Call::Null => "NULL",
        Call::Inconclusive => "INCONCLUSIVE",
    }
}

fn compared(out: &mut String, c: &Compared, with_pairs: bool, alpha: f64) {
    let _ = writeln!(
        out,
        "  {}: test arm mean {}, control mean {}",
        c.name,
        num(c.test_mean),
        num(c.control_mean)
    );
    match (&c.test, &c.untested) {
        (Some(t), _) => {
            let _ = writeln!(
                out,
                "    mean difference {:+.4}, sd {}, p {:.4}, {}% interval {}, {} pairs",
                t.mean,
                num(t.sd),
                t.p,
                (1.0 - alpha) * 100.0,
                interval(&t.interval),
                t.pairs
            );
        }
        (None, Some(why)) => {
            let _ = writeln!(out, "    no test: {why}");
        }
        (None, None) => {}
    }
    if with_pairs {
        for p in &c.pairs {
            let _ = writeln!(
                out,
                "    seed {}: {} {} - {} {} = {}",
                p.seed,
                p.test_run,
                num(p.test),
                p.control_run,
                num(p.control),
                p.difference.map_or("-".to_owned(), |d| format!("{d:+.4}"))
            );
        }
    }
}

/// The report as text.
pub fn text(r: &Report, commit: Option<&str>) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "Comparison: {}", r.name);
    if let Some(c) = commit {
        let _ = writeln!(out, "dc-compare commit: {c}");
    }
    let _ = writeln!(
        out,
        "Arms: {} (test) against {} (control). Tasks: {}. Counted iterations: {:?}.",
        r.arms.test,
        r.arms.control,
        r.tasks.join(", "),
        r.counted_iterations
    );
    let _ = writeln!(
        out,
        "Level {}, smallest effect that matters {}.",
        r.alpha, r.smallest_effect
    );
    let _ = writeln!(out, "\nPRIMARY ({:?})", r.metric);
    compared(&mut out, &r.primary.metric, true, r.alpha);
    let _ = writeln!(out, "  CALL: {}", call(r.primary.call));
    if let Some(c) = &r.check {
        let _ = writeln!(out, "\nCHECK (iteration {}, not data)", c.iteration);
        compared(&mut out, &c.metric, true, r.alpha);
        let _ = writeln!(
            out,
            "  {}",
            if c.differs {
                "DIFFERS: find the cause before reading the primary result (D33)"
            } else {
                "no difference at the level"
            }
        );
    }
    let _ = writeln!(out, "\nSPREAD BETWEEN RUNS");
    let _ = writeln!(
        out,
        "  sd of the pair differences: {}",
        num(r.spread.difference_sd)
    );
    match &r.spread.logit {
        Some(s) => {
            let _ = writeln!(
                out,
                "  logit scale: sigma {:.3} (variance between runs {:.3}, {} groups, {} cells)",
                s.sigma, s.between_run_variance, s.groups, s.cells
            );
        }
        None => {
            let _ = writeln!(out, "  logit scale: needs two runs in an arm");
        }
    }
    let _ = writeln!(out, "\nEXPLORATORY (no call)");
    for c in &r.exploratory {
        compared(&mut out, c, false, r.alpha);
    }
    let _ = writeln!(out, "\nFAILURES (spec tasks, counted iterations)");
    for (arm, f) in [&r.arms.test, &r.arms.control].iter().zip(&r.failures) {
        let _ = writeln!(
            out,
            "  {arm}: issued {}, not submitted {}, not compiled {}, not typed {}, \
             failed examples {}, passed {}",
            f.issued, f.not_submitted, f.not_compiled, f.not_typed, f.failed_examples, f.passed
        );
    }
    let _ = writeln!(out, "\nRUNS");
    for run in &r.runs {
        let t = &run.tokens;
        let _ = writeln!(
            out,
            "  {} ({}, pair seed {}): {} sessions ({} naming, {} with usage), {} input and {} \
             output tokens, {:.2} USD reported",
            run.run,
            run.arm,
            run.seed,
            t.sessions,
            t.naming_sessions,
            t.with_usage,
            t.input_tokens,
            t.output_tokens,
            t.cost_usd
        );
        for i in &run.iterations {
            let tasks: Vec<String> = i
                .tasks
                .iter()
                .map(|(n, c)| match c.cheap {
                    Some(ch) => format!("{n} {}/{} ({ch} cheap)", c.passed, c.issued),
                    None => format!("{n} {}/{}", c.passed, c.issued),
                })
                .collect();
            let _ = writeln!(
                out,
                "    iteration {}{}: solved {}; {}; library used by {}/{} passed; \
                 foreign {}, late or repeat {}",
                i.iteration,
                if i.counted { "" } else { " (check)" },
                i.tasks_solved,
                tasks.join(", "),
                i.using_library,
                i.passed_all_tasks,
                i.foreign,
                i.late_or_repeat
            );
            for (entry, name, programs) in &i.entries_used {
                let _ = writeln!(
                    out,
                    "      entry {entry} ({}) used by {programs}",
                    name.as_deref().unwrap_or("no name")
                );
            }
        }
        if let Some(lib) = &run.library {
            for e in lib {
                let _ = writeln!(
                    out,
                    "    library {}: {} /{}, added at the end of iteration {}: {}",
                    e.index,
                    e.name.as_deref().unwrap_or("no name"),
                    e.arity,
                    e.iteration_added,
                    e.doc.as_deref().unwrap_or("")
                );
            }
        }
    }
    for (title, lines) in [
        ("DEVIATIONS", &r.deviations),
        ("WARNINGS", &r.warnings),
        ("NOT COMPUTED", &r.not_computed),
    ] {
        let _ = writeln!(out, "\n{title}");
        if lines.is_empty() {
            let _ = writeln!(out, "  none");
        }
        for l in lines {
            let _ = writeln!(out, "  {l}");
        }
    }
    out
}
