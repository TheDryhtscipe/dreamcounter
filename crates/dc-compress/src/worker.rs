//! Runs Stitch in a child process, with a time and a memory limit.
//!
//! Stitch's time and memory grow quickly with the size of its input, it panics
//! on input it cannot handle, and its allocation failures and stack overflows
//! abort the process. [`Worker::run`] runs it in a separate process, so none of
//! this reaches the caller:
//!
//! - The caller waits at most [`Limits::wall`]. A worker still running then is
//!   killed, and the result is [`WorkerError::TimedOut`].
//! - The worker limits its own address space to [`Limits::memory_bytes`]
//!   (`RLIMIT_AS`). An allocation over the limit fails, and the worker aborts.
//!   The result is [`WorkerError::OutOfMemory`].
//! - A panic, an abort or any other exit that is not a success gives
//!   [`WorkerError::Failed`], with the start of the worker's standard error.
//!
//! The worker is a program that calls [`serve`]: the `dc-stitch-worker` binary
//! of this crate ([`WORKER_BIN`]), or another binary that calls [`serve`] when
//! it is started as a worker. [`Worker::program`] and [`Worker::args`] say how
//! to start it.
//!
//! # Protocol
//!
//! The caller starts the worker with three environment variables:
//! [`MEMORY_ENV`] (the memory limit in bytes), [`CPU_ENV`] (a limit on CPU
//! seconds) and [`PARENT_ENV`] (the caller's process id). It writes one
//! [`StitchRequest`] as JSON on the worker's standard input and closes it. The
//! worker writes Stitch's steps as a JSON array of [`StitchStep`]s on its
//! standard output, and exits with status 0.
//!
//! The output is not trusted: [`crate::compress_isolated`] checks the steps with
//! [`crate::accept_steps`], as it checks the steps of an in-process run.
//!
//! # When the caller dies
//!
//! On Linux the worker asks to be killed when its parent dies
//! (`PR_SET_PDEATHSIG`), and exits at once if its parent is already not the
//! process named in [`PARENT_ENV`]. The signal is sent when the parent's thread
//! that started the worker exits, so start a worker from a thread that waits
//! for it, as [`Worker::run`] does. On every platform the CPU limit stops a
//! worker whose caller is gone: it is [`Limits::wall`] plus [`CPU_SLACK`],
//! rounded up to whole seconds.

use std::ffi::OsString;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitCode, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::{run_stitch, CompressConfig, StitchStep};

/// The name of this crate's worker binary.
pub const WORKER_BIN: &str = "dc-stitch-worker";

/// The environment variable that holds the worker's memory limit, in bytes.
pub const MEMORY_ENV: &str = "DC_STITCH_WORKER_MEMORY_BYTES";

/// The environment variable that holds the worker's limit on CPU time, in
/// seconds.
pub const CPU_ENV: &str = "DC_STITCH_WORKER_CPU_SECONDS";

/// The environment variable that holds the process id of the worker's caller.
pub const PARENT_ENV: &str = "DC_STITCH_WORKER_PARENT";

/// How much longer than [`Limits::wall`] the worker's CPU limit is. The CPU
/// limit only stops a worker whose caller is gone, so it must not fire first.
pub const CPU_SLACK: Duration = Duration::from_secs(5);

/// How much of the worker's standard error is kept, in bytes.
const STDERR_KEPT: usize = 64 << 10;

/// The longest wait between two checks on a running worker.
const POLL_MAX: Duration = Duration::from_millis(50);

/// The exit status of a worker whose request or limits were wrong.
const BAD_REQUEST: u8 = 2;

/// What a worker is asked to run: Stitch's input, as text.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StitchRequest {
    /// The programs, encoded for Stitch.
    pub programs: Vec<String>,
    /// The task label of each program.
    pub tasks: Vec<String>,
    /// [`CompressConfig::iterations`].
    pub iterations: usize,
    /// [`CompressConfig::max_arity`].
    pub max_arity: u32,
    /// The number of abstractions the library already has.
    pub library_len: usize,
}

/// The limits on one worker run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// The longest the caller waits for the worker.
    pub wall: Duration,
    /// The worker's address space, in bytes. This counts memory the worker has
    /// reserved, not only what it uses: Stitch's thread reserves a stack of
    /// 256 MiB. Set it well above that. In a release build, two lists of 100
    /// elements took 0.02 seconds under a limit of 400 MiB and 0.8 seconds
    /// under 280 MiB.
    ///
    /// It also bounds what the caller reads from the worker's standard output.
    pub memory_bytes: u64,
}

impl Default for Limits {
    /// Ten minutes and 4 GiB.
    fn default() -> Limits {
        Limits {
            wall: Duration::from_secs(600),
            memory_bytes: 4 << 30,
        }
    }
}

impl Limits {
    /// The worker's limit on CPU time: [`Limits::wall`] plus [`CPU_SLACK`],
    /// rounded up to whole seconds.
    pub fn cpu_seconds(&self) -> u64 {
        let total = self.wall.saturating_add(CPU_SLACK);
        let whole = total.as_secs();
        if total.subsec_nanos() > 0 {
            whole.saturating_add(1)
        } else {
            whole
        }
    }
}

/// How to start a worker, and its limits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Worker {
    /// The program to start.
    pub program: PathBuf,
    /// The arguments to give it.
    pub args: Vec<OsString>,
    /// The limits on each run.
    pub limits: Limits,
}

/// Why a worker run gave no steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerError {
    /// The worker could not be started.
    Spawn(io::ErrorKind),
    /// The worker's output could not be read.
    Read(io::ErrorKind),
    /// The worker did not finish within [`Limits::wall`], and was killed.
    TimedOut {
        /// The limit.
        wall: Duration,
    },
    /// An allocation in the worker went over [`Limits::memory_bytes`].
    OutOfMemory {
        /// The limit.
        memory_bytes: u64,
        /// The start of the worker's standard error.
        stderr: String,
    },
    /// The worker exited with a status that is not a success, or was killed by
    /// a signal, for another reason: a panic in Stitch, for example.
    Failed {
        /// How the worker exited.
        status: ExitStatus,
        /// The start of the worker's standard error.
        stderr: String,
    },
    /// The worker wrote more than [`Limits::memory_bytes`] on its standard
    /// output.
    OutputTooLarge {
        /// The limit.
        bytes: u64,
    },
    /// The worker's output is not a JSON array of steps.
    BadOutput {
        /// The parse error.
        error: String,
    },
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerError::Spawn(kind) => write!(f, "the Stitch worker could not start: {kind}"),
            WorkerError::Read(kind) => {
                write!(f, "the Stitch worker's output could not be read: {kind}")
            }
            WorkerError::TimedOut { wall } => {
                write!(
                    f,
                    "the Stitch worker took more than {wall:?} and was stopped"
                )
            }
            WorkerError::OutOfMemory {
                memory_bytes,
                stderr,
            } => write!(
                f,
                "the Stitch worker went over its memory limit of {memory_bytes} bytes: {}",
                stderr.trim_end()
            ),
            WorkerError::Failed { status, stderr } => {
                write!(
                    f,
                    "the Stitch worker failed ({status}): {}",
                    stderr.trim_end()
                )
            }
            WorkerError::OutputTooLarge { bytes } => {
                write!(f, "the Stitch worker wrote more than {bytes} bytes")
            }
            WorkerError::BadOutput { error } => {
                write!(f, "the Stitch worker's output does not parse: {error}")
            }
        }
    }
}

impl std::error::Error for WorkerError {}

impl Worker {
    /// A worker started as `program` with no arguments, with the default
    /// limits.
    pub fn new(program: impl Into<PathBuf>) -> Worker {
        Worker {
            program: program.into(),
            args: Vec::new(),
            limits: Limits::default(),
        }
    }

    /// Runs `request` in a new worker process, and returns Stitch's steps.
    ///
    /// The calling thread waits for the worker. A worker still running after
    /// [`Limits::wall`] is killed.
    ///
    /// The caller reads the worker's output until its pipes close. A worker
    /// must not start processes that keep them open after it exits: the call
    /// would wait for those too.
    pub fn run(&self, request: &StitchRequest) -> Result<Vec<StitchStep>, WorkerError> {
        let input =
            serde_json::to_vec(request).expect("a request of strings and numbers serialises");
        let child = Command::new(&self.program)
            .args(&self.args)
            .env(MEMORY_ENV, self.limits.memory_bytes.to_string())
            .env(CPU_ENV, self.limits.cpu_seconds().to_string())
            .env(PARENT_ENV, std::process::id().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| WorkerError::Spawn(e.kind()))?;
        let mut child = Reaped(child);
        let stdin = child.0.stdin.take().expect("stdin is piped");
        let stdout = child.0.stdout.take().expect("stdout is piped");
        let stderr = child.0.stderr.take().expect("stderr is piped");
        let output_cap = usize::try_from(self.limits.memory_bytes).unwrap_or(usize::MAX);
        let deadline = Instant::now().checked_add(self.limits.wall);
        let (status, out, err) = std::thread::scope(|scope| {
            scope.spawn(move || {
                let mut stdin = stdin;
                // A worker that exits without reading its input makes this
                // fail. Its exit status says why.
                let _ = stdin.write_all(&input);
            });
            let out = scope.spawn(move || read_capped(stdout, output_cap));
            let err = scope.spawn(move || read_capped(stderr, STDERR_KEPT));
            let status = wait_until(&mut child.0, deadline);
            if status.is_err() {
                // The readers below wait for the worker's pipes to close.
                let _ = child.0.kill();
                let _ = child.0.wait();
            }
            // Each reader ends when the worker has exited or been killed.
            let join = |h: std::thread::ScopedJoinHandle<'_, io::Result<Capped>>| {
                h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))
            };
            (status, join(out), join(err))
        });
        let status = status.map_err(|e| WorkerError::Read(e.kind()))?;
        let Some(status) = status else {
            return Err(WorkerError::TimedOut {
                wall: self.limits.wall,
            });
        };
        let stderr = err
            .map(|c| String::from_utf8_lossy(&c.bytes).into_owned())
            .unwrap_or_default();
        if !status.success() {
            // Rust's allocation error handler writes this line, then aborts.
            return Err(if stderr.contains("memory allocation of ") {
                WorkerError::OutOfMemory {
                    memory_bytes: self.limits.memory_bytes,
                    stderr,
                }
            } else {
                WorkerError::Failed { status, stderr }
            });
        }
        let out = out.map_err(|e| WorkerError::Read(e.kind()))?;
        if out.overflowed {
            return Err(WorkerError::OutputTooLarge {
                bytes: self.limits.memory_bytes,
            });
        }
        serde_json::from_slice(&out.bytes).map_err(|e| WorkerError::BadOutput {
            error: e.to_string(),
        })
    }
}

/// A child process that is killed and reaped when this is dropped, unless it
/// has already exited.
struct Reaped(Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        if !matches!(self.0.try_wait(), Ok(Some(_))) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

/// Waits for `child` until `deadline`. Returns `None` if it was still running
/// then: it has been killed and reaped. With no deadline, waits for as long as
/// it runs.
fn wait_until(child: &mut Child, deadline: Option<Instant>) -> io::Result<Option<ExitStatus>> {
    let Some(deadline) = deadline else {
        return child.wait().map(Some);
    };
    let mut poll = Duration::from_millis(1);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Some(status));
        }
        let now = Instant::now();
        if now >= deadline {
            child.kill()?;
            child.wait()?;
            return Ok(None);
        }
        std::thread::sleep(poll.min(deadline - now));
        poll = (poll * 2).min(POLL_MAX);
    }
}

/// What [`read_capped`] read.
struct Capped {
    /// The first bytes read, at most the cap.
    bytes: Vec<u8>,
    /// Whether there were more.
    overflowed: bool,
}

/// Reads `r` to its end. Keeps at most `cap` bytes, and reads and drops the
/// rest, so that the writer does not block.
fn read_capped(mut r: impl Read, cap: usize) -> io::Result<Capped> {
    let mut bytes = Vec::new();
    (&mut r)
        .take(u64::try_from(cap).unwrap_or(u64::MAX))
        .read_to_end(&mut bytes)?;
    let rest = io::copy(&mut r, &mut io::sink())?;
    Ok(Capped {
        bytes,
        overflowed: rest > 0,
    })
}

/// The worker's side: reads a [`StitchRequest`] on standard input, runs Stitch,
/// and writes its steps on standard output. Call this from the `main` of a
/// program that is started as a worker, and return its exit code.
///
/// It first sets the limits named in [`MEMORY_ENV`] and [`CPU_ENV`], and on
/// Linux asks to be killed when its parent dies. A missing variable sets no
/// limit. A variable that does not parse, or a limit that cannot be set, exits
/// with status 2, as does a request that does not parse. A panic in Stitch
/// exits with Rust's panic status, 101.
pub fn serve() -> ExitCode {
    match serve_inner() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            let _ = writeln!(io::stderr(), "{}: {message}", WORKER_BIN);
            ExitCode::from(BAD_REQUEST)
        }
    }
}

fn serve_inner() -> Result<(), String> {
    set_limits()?;
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| format!("reading the request: {e}"))?;
    let request: StitchRequest =
        serde_json::from_slice(&input).map_err(|e| format!("the request does not parse: {e}"))?;
    drop(input);
    if request.programs.len() != request.tasks.len() {
        return Err(format!(
            "{} programs and {} task labels",
            request.programs.len(),
            request.tasks.len()
        ));
    }
    let steps = if request.programs.is_empty() {
        Vec::new()
    } else {
        let cfg = CompressConfig {
            iterations: request.iterations,
            max_arity: request.max_arity,
        };
        run_stitch(&request.programs, request.tasks, &cfg, request.library_len)
            .map_err(|e| e.to_string())?
    };
    let output = serde_json::to_vec(&steps).expect("steps of strings and numbers serialise");
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&output)
        .and_then(|()| stdout.flush())
        .map_err(|e| format!("writing the steps: {e}"))
}

/// Reads the limit in the environment variable `name`, if it is set.
fn env_limit(name: &str) -> Result<Option<u64>, String> {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .map(Some)
            .map_err(|e| format!("{name}={v:?} does not parse: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(e) => Err(format!("{name}: {e}")),
    }
}

/// Sets the limits named in the environment, and the parent-death signal.
#[cfg(unix)]
fn set_limits() -> Result<(), String> {
    use rustix::process::{setrlimit, Resource, Rlimit};
    if let Some(bytes) = env_limit(MEMORY_ENV)? {
        setrlimit(
            Resource::As,
            Rlimit {
                current: Some(bytes),
                maximum: Some(bytes),
            },
        )
        .map_err(|e| format!("setting the memory limit: {e}"))?;
    }
    if let Some(seconds) = env_limit(CPU_ENV)? {
        // SIGXCPU at the soft limit; SIGKILL at the hard limit if SIGXCPU is
        // caught or ignored.
        setrlimit(
            Resource::Cpu,
            Rlimit {
                current: Some(seconds),
                maximum: Some(seconds.saturating_add(1)),
            },
        )
        .map_err(|e| format!("setting the CPU limit: {e}"))?;
    }
    if let Some(parent) = env_limit(PARENT_ENV)? {
        die_with_parent(parent)?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_limits() -> Result<(), String> {
    for name in [MEMORY_ENV, CPU_ENV] {
        if env_limit(name)?.is_some() {
            return Err(format!("{name}: limits are not supported on this platform"));
        }
    }
    Ok(())
}

/// Asks to be killed when the parent dies, then checks that the parent is
/// still `parent`: if it died first, the request came too late.
#[cfg(target_os = "linux")]
fn die_with_parent(parent: u64) -> Result<(), String> {
    use rustix::process::{getppid, set_parent_process_death_signal, Pid, Signal};
    set_parent_process_death_signal(Some(Signal::KILL))
        .map_err(|e| format!("setting the parent-death signal: {e}"))?;
    let now = Pid::as_raw(getppid());
    if u64::try_from(now).ok() != Some(parent) {
        return Err(format!("the parent {parent} has exited"));
    }
    Ok(())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn die_with_parent(_parent: u64) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_limit_is_the_wall_limit_plus_the_slack_rounded_up() {
        let limits = |wall| Limits {
            wall,
            memory_bytes: 1,
        };
        assert_eq!(limits(Duration::from_secs(10)).cpu_seconds(), 15);
        assert_eq!(limits(Duration::from_millis(10_001)).cpu_seconds(), 16);
        assert_eq!(limits(Duration::ZERO).cpu_seconds(), 5);
        assert_eq!(limits(Duration::MAX).cpu_seconds(), u64::MAX);
    }
}
