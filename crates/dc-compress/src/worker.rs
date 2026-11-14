//! Runs Stitch in a child process, with a time and a memory limit.
//!
//! Stitch's time and memory grow quickly with the size of its input, it panics
//! on input it cannot handle, and its allocation failures and stack overflows
//! abort the process. [`Worker::run`] runs it in a separate process, so that
//! these stay in that process:
//!
//! - The caller waits at most [`Limits::wall`]. A worker still running then is
//!   killed, and the result is [`WorkerError::TimedOut`]. The caller can also
//!   stop a run early with its cancel flag ([`WorkerError::Cancelled`]).
//! - The worker limits its own address space to [`Limits::memory_bytes`]
//!   (`RLIMIT_AS`). An allocation over the limit fails, and the worker aborts.
//!   The result is [`WorkerError::OutOfMemory`]. A limit too small for the
//!   stack of Stitch's thread gives [`WorkerError::NoThread`].
//! - The worker limits its own CPU time, as a backstop (see below). A worker
//!   stopped by a CPU limit gives [`WorkerError::CpuLimit`].
//! - A panic, an abort or any other exit that is not a success gives
//!   [`WorkerError::Failed`], with the start and the end of the worker's
//!   standard error.
//! - The caller keeps at most [`Limits::output_bytes`] of the worker's output.
//!
//! The limits need Unix. Elsewhere the worker refuses to run with them, so
//! every run fails. dc runs on Linux.
//!
//! The worker is a program that calls [`serve`]: the `dc-stitch-worker` binary
//! of this crate ([`WORKER_BIN`]), or another binary that calls [`serve`] when
//! it is started as a worker. [`Worker::program`] and [`Worker::args`] say how
//! to start it. The program must be the worker itself, or a wrapper that
//! replaces itself with the worker (`exec`). A wrapper that starts the worker
//! as its own child makes every run fail, because the worker's parent is then
//! not the caller (see below).
//!
//! # Protocol
//!
//! The caller starts the worker with three environment variables:
//! [`MEMORY_ENV`] (the memory limit in bytes), [`CPU_ENV`] (a limit on CPU
//! seconds) and [`PARENT_ENV`] (the caller's process id). It writes one
//! [`StitchRequest`] as JSON on the worker's standard input and closes it. The
//! worker writes Stitch's steps as a JSON array of [`StitchStep`]s on its
//! standard output, and exits with status 0. It exits with status 2 when the
//! request or a limit is wrong, with 3 when the thread that runs Stitch cannot
//! start, and with 101 when Stitch panics. It sets its core file size to 0
//! and, on Linux, marks itself as not dumpable, so that a failed run does not
//! dump core.
//!
//! The output is not trusted: [`crate::compress_isolated`] checks the steps with
//! [`crate::accept_steps`], as it checks the steps of an in-process run. Both
//! sides refuse fields they do not know, so a worker built from other source
//! fails if the format changed. A worker built from other source with the same
//! format is not detected.
//!
//! # When the caller dies
//!
//! On Linux the worker asks to be killed when its parent dies
//! (`PR_SET_PDEATHSIG`), and exits at once if its parent is already not the
//! process named in [`PARENT_ENV`]. The signal is sent when the parent's thread
//! that started the worker exits, so start a worker from a thread that waits
//! for it, as [`Worker::run`] does. On Unix the CPU limit also stops a worker
//! whose caller is gone: it is [`Limits::wall`] plus [`CPU_SLACK`], rounded up
//! to whole seconds.
//!
//! A hard limit the worker inherits, which it cannot raise, lowers the limit it
//! sets. The worker then says so on its standard error. An inherited hard CPU
//! limit can stop a run before [`Limits::wall`]. The worker replaces the soft
//! limits it inherits with its own.
//!
//! The caller writes the request on a pipe. If the worker exits before it has
//! read it all, the write fails with `SIGPIPE` unless the caller ignores that
//! signal. Rust programs ignore it.

use std::ffi::OsString;
use std::fmt;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, ExitCode, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// How much of the start and of the end of the worker's standard error is
/// kept, in bytes each. A panic message comes first, and an allocation failure
/// last.
const STDERR_KEPT: usize = 32 << 10;

/// The longest wait between two checks on a running worker.
const POLL_MAX: Duration = Duration::from_millis(50);

/// The exit status of a worker whose request or limits were wrong.
const BAD_REQUEST: u8 = 2;

/// The exit status of a worker whose Stitch thread could not start. Under a
/// memory limit, that is because the limit is too small for its stack.
const NO_THREAD: u8 = 3;

/// The marker [`read_head_tail`] puts where it left bytes out.
const LEFT_OUT: &str = "\n[...]\n";

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
    /// The longest the caller waits for the worker. The CPU limit is at most
    /// about 584 years, the largest the kernel keeps. A wall limit too large to
    /// add to the current time, such as [`Duration::MAX`], sets none.
    pub wall: Duration,
    /// The worker's address space, in bytes. This counts memory the worker has
    /// reserved, not only what it uses: Stitch's thread reserves a stack of
    /// 256 MiB, and below about 260 MiB the thread cannot start. Set it well
    /// above that. In a release build, two lists of 100 elements took 0.02
    /// seconds under a limit of 400 MiB and 0.8 seconds under 280 MiB.
    ///
    /// A lower hard limit that the worker inherits applies in its place.
    pub memory_bytes: u64,
    /// The most bytes the caller keeps of the worker's standard output. The
    /// caller holds the output and the steps parsed from it at the same time.
    pub output_bytes: u64,
}

impl Default for Limits {
    /// Ten minutes, 4 GiB of address space, and 256 MiB of output.
    fn default() -> Limits {
        Limits {
            wall: Duration::from_secs(600),
            memory_bytes: 4 << 30,
            output_bytes: 256 << 20,
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
    /// Waiting for the worker, or reading its output, failed.
    Io(io::ErrorKind),
    /// The worker did not finish within [`Limits::wall`], and was killed.
    TimedOut {
        /// The limit.
        wall: Duration,
    },
    /// The cancel flag was set, and the worker was killed.
    Cancelled,
    /// An allocation in the worker failed, and the worker aborted. Under the
    /// memory limit, that means the run went over it.
    OutOfMemory {
        /// The limit asked for. The worker's standard error says if a lower
        /// inherited limit applied.
        memory_bytes: u64,
        /// The start and the end of the worker's standard error.
        stderr: String,
    },
    /// The thread that runs Stitch could not start. Under the memory limit,
    /// that is usually because the limit is too small for its stack.
    NoThread {
        /// The memory limit asked for.
        memory_bytes: u64,
        /// The start and the end of the worker's standard error.
        stderr: String,
    },
    /// The worker was stopped by its CPU limit (`SIGXCPU`). The worker's own
    /// limit comes after [`Limits::wall`], so this means a lower hard limit it
    /// inherited applied. A worker that ignores `SIGXCPU`, or whose inherited
    /// hard limit is 0, is killed at the hard limit instead, and gives
    /// [`WorkerError::Failed`].
    CpuLimit {
        /// The start and the end of the worker's standard error.
        stderr: String,
    },
    /// The worker exited with a status that is not a success, or was killed by
    /// a signal, for another reason: a panic in Stitch, for example.
    Failed {
        /// How the worker exited.
        status: ExitStatus,
        /// The start and the end of the worker's standard error.
        stderr: String,
    },
    /// The worker wrote more than [`Limits::output_bytes`] on its standard
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
            WorkerError::Io(kind) => write!(
                f,
                "waiting for the Stitch worker or reading its output failed: {kind}"
            ),
            WorkerError::Cancelled => f.write_str("the Stitch worker was cancelled"),
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
            WorkerError::NoThread {
                memory_bytes,
                stderr,
            } => write!(
                f,
                "the Stitch worker could not start Stitch's thread under a memory limit of \
                 {memory_bytes} bytes: {}",
                stderr.trim_end()
            ),
            WorkerError::CpuLimit { stderr } => write!(
                f,
                "the Stitch worker was stopped by its CPU limit: {}",
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
    /// [`Limits::wall`] is killed. Setting `cancel` from another thread kills
    /// the worker too, within about 50 milliseconds.
    ///
    /// The caller reads the worker's output until its pipes close. A worker
    /// must not start processes that keep them open after it exits: the call
    /// would wait for those too.
    pub fn run(
        &self,
        request: &StitchRequest,
        cancel: &AtomicBool,
    ) -> Result<Vec<StitchStep>, WorkerError> {
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
        let deadline = Instant::now().checked_add(self.limits.wall);
        let (status, out, err) = std::thread::scope(|scope| {
            scope.spawn(move || {
                let mut stdin = stdin;
                // A worker that exits without reading its input makes this
                // fail. Its exit status says why.
                let _ = stdin.write_all(&input);
            });
            let output_bytes = self.limits.output_bytes;
            let out = scope.spawn(move || read_capped(stdout, output_bytes));
            let err = scope.spawn(move || read_head_tail(stderr, STDERR_KEPT));
            let ended = wait_until(&mut child.0, deadline, cancel);
            if ended.is_err() {
                // The readers below wait for the worker's pipes to close.
                let _ = child.0.kill();
                let _ = child.0.wait();
            }
            // Each reader ends when the worker has exited or been killed.
            let out = out.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
            let err = err.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
            (ended, out, err)
        });
        let status = match status.map_err(|e| WorkerError::Io(e.kind()))? {
            Ended::Exited(status) => status,
            Ended::TimedOut => {
                return Err(WorkerError::TimedOut {
                    wall: self.limits.wall,
                })
            }
            Ended::Cancelled => return Err(WorkerError::Cancelled),
        };
        let stderr = err
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default();
        if !status.success() {
            let memory_bytes = self.limits.memory_bytes;
            return Err(if allocation_failed(&status, &stderr) {
                WorkerError::OutOfMemory {
                    memory_bytes,
                    stderr,
                }
            } else if status.code() == Some(i32::from(NO_THREAD)) {
                WorkerError::NoThread {
                    memory_bytes,
                    stderr,
                }
            } else if signal_of(&status) == Some(SIGXCPU) {
                WorkerError::CpuLimit { stderr }
            } else {
                WorkerError::Failed { status, stderr }
            });
        }
        let out = out.map_err(|e| WorkerError::Io(e.kind()))?;
        if out.overflowed {
            return Err(WorkerError::OutputTooLarge {
                bytes: self.limits.output_bytes,
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

/// How [`wait_until`] ended.
enum Ended {
    /// The worker exited on its own.
    Exited(ExitStatus),
    /// The deadline passed, and the worker was killed.
    TimedOut,
    /// The cancel flag was set, and the worker was killed.
    Cancelled,
}

/// Waits for `child` until `deadline` (with none, for as long as it runs), or
/// until `cancel` is set. A child still running then is killed and reaped.
fn wait_until(
    child: &mut Child,
    deadline: Option<Instant>,
    cancel: &AtomicBool,
) -> io::Result<Ended> {
    let mut poll = Duration::from_millis(1);
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Ended::Exited(status));
        }
        let now = Instant::now();
        let stop = if cancel.load(Ordering::Relaxed) {
            Some(Ended::Cancelled)
        } else if deadline.is_some_and(|d| now >= d) {
            Some(Ended::TimedOut)
        } else {
            None
        };
        if let Some(stop) = stop {
            child.kill()?;
            let status = child.wait()?;
            // The child can exit between `try_wait` and `kill`: then it ended
            // on its own.
            return Ok(if killed(&status) {
                stop
            } else {
                Ended::Exited(status)
            });
        }
        let sleep = match deadline {
            Some(d) => poll.min(d - now),
            None => poll,
        };
        std::thread::sleep(sleep);
        poll = (poll * 2).min(POLL_MAX);
    }
}

const SIGABRT: i32 = 6;
const SIGKILL: i32 = 9;
const SIGXCPU: i32 = 24;

/// The signal that ended a process with `status`, if one did.
#[cfg(unix)]
fn signal_of(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of(_status: &ExitStatus) -> Option<i32> {
    None
}

/// Whether `status` is that of a process killed by `SIGKILL`. Off Unix, any
/// status after a kill counts.
fn killed(status: &ExitStatus) -> bool {
    cfg!(not(unix)) || signal_of(status) == Some(SIGKILL)
}

/// Whether the worker aborted on a failed allocation. Rust's allocation error
/// handler writes a line `memory allocation of N bytes failed`, then aborts.
/// Text from Stitch, such as a panic message that quotes a program, is not
/// followed by an abort.
fn allocation_failed(status: &ExitStatus, stderr: &str) -> bool {
    signal_of(status) == Some(SIGABRT)
        && stderr
            .lines()
            .any(|l| l.starts_with("memory allocation of ") && l.ends_with(" bytes failed"))
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
fn read_capped(mut r: impl Read, cap: u64) -> io::Result<Capped> {
    let mut bytes = Vec::new();
    (&mut r).take(cap).read_to_end(&mut bytes)?;
    let rest = io::copy(&mut r, &mut io::sink())?;
    Ok(Capped {
        bytes,
        overflowed: rest > 0,
    })
}

/// Reads `r` to its end. Keeps its first `keep` bytes and its last `keep`
/// bytes, with [`LEFT_OUT`] between them if bytes were left out.
fn read_head_tail(mut r: impl Read, keep: usize) -> io::Result<Vec<u8>> {
    let mut head = Vec::new();
    (&mut r)
        .take(u64::try_from(keep).unwrap_or(u64::MAX))
        .read_to_end(&mut head)?;
    let mut tail: Vec<u8> = Vec::new();
    let mut left_out = false;
    let mut buf = vec![0u8; 8 << 10];
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > 2 * keep {
            tail.drain(..tail.len() - keep);
            left_out = true;
        }
    }
    if tail.len() > keep {
        tail.drain(..tail.len() - keep);
        left_out = true;
    }
    if left_out {
        head.extend_from_slice(LEFT_OUT.as_bytes());
    }
    head.extend_from_slice(&tail);
    Ok(head)
}

/// The worker's side: reads a [`StitchRequest`] on standard input, runs Stitch,
/// and writes its steps on standard output. Call this from the `main` of a
/// program that is started as a worker, and return its exit code.
///
/// It first sets the limits named in [`MEMORY_ENV`] and [`CPU_ENV`], and on
/// Linux asks to be killed when its parent dies. A missing variable sets no
/// limit, and a hard limit already in place lowers the limit set. A variable
/// that does not parse, or a limit that cannot be set, exits with status 2, as
/// does a request that does not parse. If the thread that runs Stitch cannot
/// start, it exits with status 3. A panic in Stitch exits with Rust's panic
/// status, 101.
pub fn serve() -> ExitCode {
    let (message, code) = match serve_inner() {
        Ok(()) => return ExitCode::SUCCESS,
        Err(Refused::BadRequest(m)) => (m, BAD_REQUEST),
        Err(Refused::NoThread(m)) => (m, NO_THREAD),
    };
    let _ = writeln!(io::stderr(), "{WORKER_BIN}: {message}");
    ExitCode::from(code)
}

/// Why [`serve`] gave no steps.
enum Refused {
    /// The request or a limit is wrong.
    BadRequest(String),
    /// The thread that runs Stitch could not start.
    NoThread(String),
}

impl From<String> for Refused {
    fn from(message: String) -> Refused {
        Refused::BadRequest(message)
    }
}

fn serve_inner() -> Result<(), Refused> {
    set_limits()?;
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| format!("reading the request: {e}"))?;
    let request: StitchRequest =
        serde_json::from_slice(&input).map_err(|e| format!("the request does not parse: {e}"))?;
    drop(input);
    if request.programs.len() != request.tasks.len() {
        return Err(Refused::BadRequest(format!(
            "{} programs and {} task labels",
            request.programs.len(),
            request.tasks.len()
        )));
    }
    let steps = if request.programs.is_empty() {
        Vec::new()
    } else {
        let cfg = CompressConfig {
            iterations: request.iterations,
            max_arity: request.max_arity,
        };
        run_stitch(&request.programs, request.tasks, &cfg, request.library_len)
            .map_err(|e| Refused::NoThread(e.to_string()))?
    };
    let output = serde_json::to_vec(&steps).expect("steps of strings and numbers serialise");
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(&output)
        .and_then(|()| stdout.flush())
        .map_err(|e| Refused::BadRequest(format!("writing the steps: {e}")))
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
    use rustix::process::Resource;
    // Lowering a limit to 0 is always allowed. A core limit of 0 stops a core
    // file, but not a core piped to a handler such as apport.
    set_limit(Resource::Core, 0, 0).map_err(|e| format!("setting the core file size: {e}"))?;
    not_dumpable()?;
    if let Some(bytes) = env_limit(MEMORY_ENV)? {
        let hard = inherited_hard(Resource::As).map_or(bytes, |m| bytes.min(m));
        set_limit(Resource::As, hard, hard)
            .map_err(|e| format!("setting the memory limit: {e}"))?;
        if hard < bytes {
            note(&format!(
                "the memory limit is {hard} bytes, the inherited hard limit"
            ));
        }
    }
    if let Some(asked) = env_limit(CPU_ENV)? {
        // A limit too large for the kernel's count of nanoseconds wraps to a
        // small one, so the hard limit is at most MAX_CPU_SECONDS.
        let seconds = asked.min(MAX_CPU_SECONDS - 1);
        // SIGXCPU at the soft limit; SIGKILL at the hard limit if SIGXCPU is
        // caught or ignored. The soft limit stays below the hard limit, so that
        // the caller sees SIGXCPU.
        let wanted = seconds + 1;
        let inherited = inherited_hard(Resource::Cpu);
        let hard = inherited.map_or(wanted, |m| wanted.min(m));
        let soft = seconds.min(hard.saturating_sub(1));
        set_limit(Resource::Cpu, soft, hard).map_err(|e| format!("setting the CPU limit: {e}"))?;
        if inherited.is_some_and(|m| m < wanted) {
            note(&format!(
                "the CPU limit is {soft} s, below the inherited hard limit of {hard} s"
            ));
        }
    }
    if let Some(parent) = env_limit(PARENT_ENV)? {
        die_with_parent(parent)?;
    }
    Ok(())
}

/// The largest CPU limit, in seconds, that the kernel keeps: it counts CPU time
/// in nanoseconds, in 64 bits. The worker sets no larger hard limit.
#[cfg(unix)]
const MAX_CPU_SECONDS: u64 = u64::MAX / 1_000_000_000;

/// Asks the kernel not to dump core for this process, by any means.
#[cfg(target_os = "linux")]
fn not_dumpable() -> Result<(), String> {
    use rustix::process::{set_dumpable_behavior, DumpableBehavior};
    set_dumpable_behavior(DumpableBehavior::NotDumpable)
        .map_err(|e| format!("turning off core dumps: {e}"))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn not_dumpable() -> Result<(), String> {
    Ok(())
}

/// The hard limit on `resource` the worker inherited, or `None` if there is
/// none. A process cannot raise its hard limit.
#[cfg(unix)]
fn inherited_hard(resource: rustix::process::Resource) -> Option<u64> {
    rustix::process::getrlimit(resource).maximum
}

/// Sets the soft and hard limits on `resource`.
#[cfg(unix)]
fn set_limit(resource: rustix::process::Resource, soft: u64, hard: u64) -> io::Result<()> {
    rustix::process::setrlimit(
        resource,
        rustix::process::Rlimit {
            current: Some(soft),
            maximum: Some(hard),
        },
    )
    .map_err(io::Error::from)
}

/// Writes a line on standard error. A failure to write it is ignored.
#[cfg(unix)]
fn note(message: &str) {
    let _ = writeln!(io::stderr(), "{WORKER_BIN}: {message}");
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
            ..Limits::default()
        };
        assert_eq!(limits(Duration::from_secs(10)).cpu_seconds(), 15);
        assert_eq!(limits(Duration::from_millis(10_001)).cpu_seconds(), 16);
        assert_eq!(limits(Duration::ZERO).cpu_seconds(), 5);
        assert_eq!(limits(Duration::MAX).cpu_seconds(), u64::MAX);
    }

    #[test]
    fn read_head_tail_keeps_both_ends() {
        let read = |len: usize, keep: usize| {
            let data: Vec<u8> = (0..len).map(|i| b'a' + (i % 26) as u8).collect();
            let kept = read_head_tail(io::Cursor::new(&data), keep).expect("read");
            (data, kept)
        };
        // Short enough: all of it, with no marker.
        let (data, kept) = read(100, 64);
        assert_eq!(kept, data);
        let (data, kept) = read(128, 64);
        assert_eq!(kept, data);
        // Longer: the first and the last 64 bytes, with the marker between.
        for len in [129, 1_000, 100_000] {
            let (data, kept) = read(len, 64);
            let mut want = data[..64].to_vec();
            want.extend_from_slice(LEFT_OUT.as_bytes());
            want.extend_from_slice(&data[len - 64..]);
            assert_eq!(kept, want, "len = {len}");
        }
    }
}
