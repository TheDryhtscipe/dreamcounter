//! Throughput benchmark for the dc evaluator (spec §11).
//!
//! For each program in [`dc_bench::programs`] at its full input size, the
//! benchmark reports steps/second and evaluations/second, on one thread and on
//! several threads at once (one evaluator per thread, each with its own inputs).
//!
//! Method: a warm-up of about `--warmup` seconds sets the batch size, so that one
//! batch takes about `--batch` seconds on one thread. Then `--reps` timed batches
//! run, and the median rate is reported. In the multi-threaded run every thread
//! runs the same batch size, the threads start each repetition together at a
//! barrier, and the rate is all threads' evaluations over the wall time from the
//! first start to the last finish. Every evaluation's result is checked against
//! the Rust reference, and its step count against the first run's, inside the
//! timed loop; a failed check stops the benchmark.
//!
//! Usage: `dc-bench [--threads N] [--reps R] [--batch SECONDS] [--warmup SECONDS]`

use std::process::ExitCode;
use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use dc_bench::{programs, BenchError, Compiled};

#[derive(Debug, Clone, Copy)]
struct Options {
    threads: usize,
    reps: usize,
    batch: Duration,
    warmup: Duration,
}

fn parse_options() -> Result<Options, String> {
    let mut o = Options {
        threads: 16,
        reps: 5,
        batch: Duration::from_millis(500),
        warmup: Duration::from_millis(300),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let bad = || format!("bad value for {flag}: {value}");
        match flag.as_str() {
            "--threads" => o.threads = value.parse().map_err(|_| bad())?,
            "--reps" => o.reps = value.parse().map_err(|_| bad())?,
            "--batch" => o.batch = Duration::from_secs_f64(value.parse().map_err(|_| bad())?),
            "--warmup" => o.warmup = Duration::from_secs_f64(value.parse().map_err(|_| bad())?),
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if o.threads == 0 || o.reps == 0 {
        return Err("--threads and --reps must be at least 1".to_owned());
    }
    Ok(o)
}

#[derive(Debug)]
enum Failure {
    Bench(BenchError),
    StepsChanged {
        program: &'static str,
        first: u64,
        now: u64,
    },
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Bench(e) => write!(f, "{e}"),
            Failure::StepsChanged {
                program,
                first,
                now,
            } => write!(f, "{program}: step count changed from {first} to {now}"),
        }
    }
}

impl From<BenchError> for Failure {
    fn from(e: BenchError) -> Failure {
        Failure::Bench(e)
    }
}

/// Runs `count` checked evaluations; each must take `steps` steps.
fn batch(c: &Compiled, inst: &dc_bench::Instance, steps: u64, count: u64) -> Result<(), Failure> {
    for _ in 0..count {
        let now = c.run(inst)?;
        if now != steps {
            return Err(Failure::StepsChanged {
                program: c.program().name,
                first: steps,
                now,
            });
        }
    }
    Ok(())
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    let m = xs.len() / 2;
    if xs.len() % 2 == 1 {
        xs[m]
    } else {
        (xs[m - 1] + xs[m]) / 2.0
    }
}

#[derive(Debug)]
struct Row {
    name: &'static str,
    n: i64,
    steps: u64,
    batch: u64,
    single: f64,
    multi: f64,
}

/// One-thread run: returns (steps per evaluation, batch size, median evals/s).
fn single(c: &Compiled, o: &Options) -> Result<(u64, u64, f64), Failure> {
    let p = c.program();
    let inst = p.instance(p.full_n);
    let steps = c.run(&inst)?;
    let start = Instant::now();
    let mut done = 0u64;
    while start.elapsed() < o.warmup {
        batch(c, &inst, steps, 1)?;
        done += 1;
    }
    let per_eval = start.elapsed().as_secs_f64() / done as f64;
    let count = ((o.batch.as_secs_f64() / per_eval).round() as u64).max(1);
    let mut rates = vec![];
    for _ in 0..o.reps {
        let t = Instant::now();
        batch(c, &inst, steps, count)?;
        rates.push(count as f64 / t.elapsed().as_secs_f64());
    }
    Ok((steps, count, median(rates)))
}

/// Multi-thread run: median aggregate evals/s over the repetitions.
fn multi(c: &Compiled, o: &Options, steps: u64, count: u64) -> Result<f64, Failure> {
    let barrier = Barrier::new(o.threads);
    let spans = thread::scope(|s| {
        let handles: Vec<_> = (0..o.threads)
            .map(|_| {
                s.spawn(|| -> Result<Vec<(Instant, Instant)>, Failure> {
                    let p = c.program();
                    // dc values are not Send: each thread builds its own inputs.
                    let inst = p.instance(p.full_n);
                    // Repetition 0 is an unreported warm-up. After a failed
                    // check the thread stops evaluating but still meets every
                    // barrier, so the other threads do not wait forever.
                    let mut failure = None;
                    let mut spans = vec![];
                    for r in 0..=o.reps {
                        barrier.wait();
                        let t = Instant::now();
                        if failure.is_none() {
                            failure = batch(c, &inst, steps, count).err();
                        }
                        if r > 0 {
                            spans.push((t, Instant::now()));
                        }
                    }
                    match failure {
                        Some(e) => Err(e),
                        None => Ok(spans),
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("benchmark thread panicked"))
            .collect::<Result<Vec<_>, Failure>>()
    })?;
    let rates = (0..o.reps)
        .map(|r| {
            let first = spans.iter().map(|s| s[r].0).min().expect("threads");
            let last = spans.iter().map(|s| s[r].1).max().expect("threads");
            (o.threads as u64 * count) as f64 / (last - first).as_secs_f64()
        })
        .collect();
    Ok(median(rates))
}

fn run(o: &Options) -> Result<Vec<Row>, Failure> {
    let mut rows = vec![];
    for p in programs() {
        let c = p.compile()?;
        let (steps, count, single) = single(&c, o)?;
        let multi = multi(&c, o, steps, count)?;
        eprintln!("{}: done", p.name);
        rows.push(Row {
            name: p.name,
            n: p.full_n,
            steps,
            batch: count,
            single,
            multi,
        });
    }
    Ok(rows)
}

fn main() -> ExitCode {
    let o = match parse_options() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("dc-bench: {e}");
            return ExitCode::from(2);
        }
    };
    let rows = match run(&o) {
        Ok(rows) => rows,
        Err(e) => {
            eprintln!("dc-bench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let t = o.threads;
    println!(
        "threads = {t}, reps = {}, batch = {:.2} s, warm-up = {:.2} s; rates are medians",
        o.reps,
        o.batch.as_secs_f64(),
        o.warmup.as_secs_f64()
    );
    println!(
        "| program | n | steps/eval | batch | 1T evals/s | 1T Msteps/s | {t}T evals/s | {t}T Msteps/s | {t}T/1T |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for r in rows {
        let ms = |evals: f64| evals * r.steps as f64 / 1e6;
        println!(
            "| {} | {} | {} | {} | {:.1} | {:.2} | {:.1} | {:.2} | {:.2} |",
            r.name,
            r.n,
            r.steps,
            r.batch,
            r.single,
            ms(r.single),
            r.multi,
            ms(r.multi),
            r.multi / r.single
        );
    }
    ExitCode::SUCCESS
}
