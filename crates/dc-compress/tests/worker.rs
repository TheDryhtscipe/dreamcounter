//! Stitch in a worker process: the same result as in the caller's process, and
//! a run over its time or memory limit, a cancelled run, a panic in Stitch, or a
//! caller that dies does not reach the caller.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use dc_compress::worker::{
    Limits, StitchRequest, Worker, WorkerError, CPU_ENV, MEMORY_ENV, PARENT_ENV,
};
use dc_compress::{compress, compress_isolated, CompressConfig, CompressError, FrontierProgram};
use dc_core::{sexpr, Arena, LibDef};

const WORKER: &str = env!("CARGO_BIN_EXE_dc-stitch-worker");

/// A cancel flag that is never set.
static NEVER: AtomicBool = AtomicBool::new(false);

fn worker(wall: Duration, memory_bytes: u64) -> Worker {
    Worker {
        limits: Limits {
            wall,
            memory_bytes,
            ..Limits::default()
        },
        ..Worker::new(WORKER)
    }
}

/// A worker whose program is `program` with `args`, not a Stitch worker.
fn stand_in(program: &str, args: &[&str], limits: Limits) -> Worker {
    Worker {
        program: program.into(),
        args: args.iter().map(Into::into).collect(),
        limits,
    }
}

fn frontier(a: &mut Arena, srcs: &[&str]) -> Vec<FrontierProgram> {
    srcs.iter()
        .enumerate()
        .map(|(i, s)| FrontierProgram {
            task: format!("task_{i}"),
            program: sexpr::parse(a, s).unwrap_or_else(|e| panic!("{s}: {e}")),
        })
        .collect()
}

/// The Stitch text of the list `[0, 2, 4, ..., 2(n - 1)]`, as `cons` cells.
fn list_text(n: usize) -> String {
    let mut s = "nil".to_string();
    for i in (0..n).rev() {
        s = format!("(cons int_{} {s})", 2 * i);
    }
    s
}

/// A request for two copies of [`list_text`]`(n)`. Stitch's time and memory
/// grow quickly with `n`. In a release build, `n` = 1,000 took 3 seconds and
/// 217 MB, and `n` = 1,500 took 7.8 seconds. Test builds are slower.
fn two_lists(n: usize) -> StitchRequest {
    StitchRequest {
        programs: vec![list_text(n); 2],
        tasks: vec!["a".to_string(), "b".to_string()],
        iterations: 3,
        max_arity: 2,
        library_len: 0,
    }
}

fn exit_code(status: &ExitStatus) -> Option<i32> {
    status.code()
}

#[cfg(unix)]
fn signal(status: &ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[test]
fn a_worker_gives_what_stitch_gives_in_process() {
    // Two rounds, the second with the first round's abstractions as the
    // library, over corpora from the bridge tests.
    let corpus = [
        "(mul 3 (add 1 (let 5 (mul $0 $0))))",
        "(mul 3 (add 1 (let 6 (sub $0 2))))",
        "(mul 3 (add 1 (let 7 (add $0 $0))))",
        "(pair (if (lt 1 2) 3 4) (if (lt 1 2) 5 6))",
        "(pair (if (lt 1 2) 7 8) (if (lt 1 2) 9 10))",
        "(pair (if (lt 1 2) 11 12) (if (lt 1 2) 13 14))",
    ];
    let mut a = Arena::new();
    let programs = frontier(&mut a, &corpus);
    let cfg = CompressConfig::default();
    let w = Worker::new(WORKER);
    let first = compress(&mut a, &[], &programs, &cfg).expect("in process");
    let isolated = compress_isolated(&mut a, &[], &programs, &cfg, &w, &NEVER).expect("worker");
    assert_eq!(isolated, first);
    assert!(!first.new_libs.is_empty(), "Stitch found no abstraction");

    let libs: Vec<LibDef> = first.new_libs.clone();
    let again: Vec<FrontierProgram> = programs
        .iter()
        .zip(&first.rewritten)
        .map(|(p, r)| FrontierProgram {
            task: p.task.clone(),
            program: *r,
        })
        .collect();
    let second = compress(&mut a, &libs, &again, &cfg).expect("in process");
    let isolated = compress_isolated(&mut a, &libs, &again, &cfg, &w, &NEVER).expect("worker");
    assert_eq!(isolated, second);
}

#[test]
fn a_run_over_the_time_limit_is_stopped() {
    let wall = Duration::from_secs(1);
    let w = worker(wall, Limits::default().memory_bytes);
    let start = Instant::now();
    let result = w.run(&two_lists(1500), &NEVER);
    let took = start.elapsed();
    assert_eq!(result, Err(WorkerError::TimedOut { wall }));
    assert!(took < Duration::from_secs(10), "took {took:?}");
}

#[test]
fn a_cancelled_run_is_stopped() {
    let cancel = AtomicBool::new(false);
    let w = worker(Duration::from_secs(300), Limits::default().memory_bytes);
    let start = Instant::now();
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(Duration::from_millis(300));
            cancel.store(true, Ordering::Relaxed);
        });
        w.run(&two_lists(1500), &cancel)
    });
    let took = start.elapsed();
    assert_eq!(result, Err(WorkerError::Cancelled));
    assert!(took < Duration::from_secs(10), "took {took:?}");
}

#[test]
fn a_run_over_the_memory_limit_is_stopped() {
    // Stitch's thread reserves a stack of 256 MiB, which counts towards the
    // limit. Two lists of 1,000 elements use about 217 MB more.
    let memory_bytes = 384 << 20;
    let w = worker(Duration::from_secs(300), memory_bytes);
    match w.run(&two_lists(1000), &NEVER) {
        Err(WorkerError::OutOfMemory {
            memory_bytes: m,
            stderr,
        }) => {
            assert_eq!(m, memory_bytes);
            assert!(stderr.contains("memory allocation of"), "{stderr}");
        }
        other => panic!("expected OutOfMemory, got {other:?}"),
    }
}

#[test]
fn the_same_run_under_a_larger_memory_limit_finishes() {
    // The control for the test above: the limit stops that run, not the
    // workload.
    let w = worker(Duration::from_secs(300), 1 << 30);
    let steps = w.run(&two_lists(1000), &NEVER).expect("the run fits");
    assert!(!steps.is_empty());
}

#[test]
fn a_limit_too_small_for_stitch_s_stack_is_reported() {
    for memory_bytes in [0, 200 << 20] {
        match worker(Duration::from_secs(60), memory_bytes).run(&two_lists(10), &NEVER) {
            Err(WorkerError::NoThread {
                memory_bytes: m,
                stderr,
            }) => {
                assert_eq!(m, memory_bytes);
                assert!(stderr.contains("could not be started"), "{stderr}");
            }
            other => panic!("{memory_bytes}: expected NoThread, got {other:?}"),
        }
    }
}

/// Runs a stand-in worker that is `sh -c script`.
#[cfg(unix)]
fn run_shell(script: &str) -> Result<Vec<dc_compress::StitchStep>, WorkerError> {
    // A shell that aborts is not the worker, so it would leave a core file or
    // a crash report. A core limit of 0 stops both.
    let script = format!("ulimit -c 0; {script}");
    stand_in("sh", &["-c", &script], Limits::default()).run(&two_lists(1), &NEVER)
}

#[cfg(unix)]
#[test]
fn out_of_memory_needs_an_abort_and_the_allocation_line() {
    // Rust's allocation error handler writes the line and then aborts.
    let line = "echo 'memory allocation of 8 bytes failed' >&2";
    match run_shell(&format!("{line}; kill -ABRT $$")) {
        Err(WorkerError::OutOfMemory { .. }) => {}
        other => panic!("abort with the line: expected OutOfMemory, got {other:?}"),
    }
    // The line without an abort, as when a panic message quotes it.
    match run_shell(&format!("{line}; exit 1")) {
        Err(WorkerError::Failed { status, .. }) => assert_eq!(exit_code(&status), Some(1)),
        other => panic!("line without abort: expected Failed, got {other:?}"),
    }
    // An abort with the words inside a longer line.
    let quoted = "echo 'parse error in: (memory allocation of 8 bytes failed' >&2";
    match run_shell(&format!("{quoted}; kill -ABRT $$")) {
        Err(WorkerError::Failed { status, .. }) => assert_eq!(signal(&status), Some(6)),
        other => panic!("abort with a quoted line: expected Failed, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn an_inherited_hard_cpu_limit_is_reported_as_the_cpu_limit() {
    // The wrapper lowers the hard CPU limit to 2 seconds and replaces itself
    // with the worker, so the worker's parent is still the caller.
    let w = Worker {
        program: "sh".into(),
        args: vec![
            "-c".into(),
            "ulimit -t 2 && exec \"$0\"".into(),
            WORKER.into(),
        ],
        limits: Limits {
            wall: Duration::from_secs(300),
            ..Limits::default()
        },
    };
    match w.run(&two_lists(1500), &NEVER) {
        Err(WorkerError::CpuLimit { stderr }) => {
            assert!(stderr.contains("the CPU limit is 1 seconds"), "{stderr}");
        }
        other => panic!("expected CpuLimit, got {other:?}"),
    }
}

#[test]
fn a_panic_in_stitch_is_reported_not_raised() {
    // Stitch unwraps its parse of each program.
    let request = StitchRequest {
        programs: vec!["(((".to_string(), "(((".to_string()],
        tasks: vec!["a".to_string(), "b".to_string()],
        iterations: 1,
        max_arity: 2,
        library_len: 0,
    };
    match Worker::new(WORKER).run(&request, &NEVER) {
        Err(WorkerError::Failed { status, stderr }) => {
            assert_eq!(exit_code(&status), Some(101), "{stderr}");
            assert!(stderr.contains("mismatched parens"), "{stderr}");
        }
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn compress_isolated_reports_a_worker_error() {
    let mut a = Arena::new();
    let programs = frontier(&mut a, &["(pair 1 2)", "(pair 1 3)"]);
    let missing = Worker::new("/nonexistent/dc-stitch-worker");
    assert_eq!(
        compress_isolated(
            &mut a,
            &[],
            &programs,
            &CompressConfig::default(),
            &missing,
            &NEVER
        ),
        Err(CompressError::Worker(WorkerError::Spawn(
            std::io::ErrorKind::NotFound
        )))
    );
}

#[test]
fn a_worker_that_does_not_exit_is_killed_at_the_deadline() {
    let wall = Duration::from_millis(200);
    let limits = Limits {
        wall,
        ..Limits::default()
    };
    let start = Instant::now();
    assert_eq!(
        stand_in("sleep", &["30"], limits).run(&two_lists(1), &NEVER),
        Err(WorkerError::TimedOut { wall })
    );
    let took = start.elapsed();
    assert!(took < Duration::from_secs(5), "took {took:?}");
}

#[test]
fn a_worker_that_exits_with_a_failure_is_reported() {
    match stand_in("false", &[], Limits::default()).run(&two_lists(1), &NEVER) {
        Err(WorkerError::Failed { status, .. }) => assert_eq!(exit_code(&status), Some(1)),
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn output_that_is_not_steps_is_refused() {
    match stand_in("echo", &["[1, 2]"], Limits::default()).run(&two_lists(1), &NEVER) {
        Err(WorkerError::BadOutput { .. }) => {}
        other => panic!("expected BadOutput, got {other:?}"),
    }
}

#[test]
fn output_over_the_output_limit_is_refused() {
    let limits = Limits {
        output_bytes: 1000,
        ..Limits::default()
    };
    assert_eq!(
        stand_in("head", &["-c", "2000", "/dev/zero"], limits).run(&two_lists(1), &NEVER),
        Err(WorkerError::OutputTooLarge { bytes: 1000 })
    );
}

/// Starts the worker binary through `sh -c script`, with `envs`, on the request
/// `input`, and waits for it with no limit of the caller's own. The script gets
/// the worker's path as `$1`.
fn run_worker_via(
    script: &str,
    envs: &[(&str, String)],
    input: &StitchRequest,
) -> (ExitStatus, String) {
    let mut child = Command::new("sh")
        .args(["-c", script, "sh", WORKER])
        .envs(envs.iter().map(|(k, v)| (*k, v.as_str())))
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start the worker");
    let body = serde_json::to_vec(input).expect("serialise");
    child
        .stdin
        .take()
        .expect("piped")
        .write_all(&body)
        .expect("write the request");
    let out = child.wait_with_output().expect("wait");
    (
        out.status,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// [`run_worker_via`] with no wrapper.
fn run_worker_directly(envs: &[(&str, String)], input: &StitchRequest) -> (ExitStatus, String) {
    run_worker_via("exec \"$1\"", envs, input)
}

#[cfg(unix)]
#[test]
fn the_cpu_limit_stops_a_worker_on_its_own() {
    // The backstop for a worker whose caller no longer waits for it.
    let start = Instant::now();
    let (status, stderr) = run_worker_directly(&[(CPU_ENV, "1".to_string())], &two_lists(1500));
    let took = start.elapsed();
    assert_eq!(signal(&status), Some(24), "SIGXCPU: {status} {stderr}");
    assert!(took < Duration::from_secs(30), "took {took:?}");
}

/// The soft and hard values of the limit `name` of process `pid`, as
/// `/proc/<pid>/limits` shows them.
#[cfg(target_os = "linux")]
fn proc_limit(pid: u32, name: &str) -> Option<(String, String)> {
    let limits = std::fs::read_to_string(format!("/proc/{pid}/limits")).ok()?;
    let line = limits.lines().find(|l| l.starts_with(name))?;
    let mut values = line[name.len()..].split_whitespace();
    Some((values.next()?.to_string(), values.next()?.to_string()))
}

/// Soft and hard values, as `/proc/<pid>/limits` shows them.
#[cfg(target_os = "linux")]
type Shown = Vec<Option<(String, String)>>;

/// Starts the worker through `sh -c script` with `cpu` and `memory` in its
/// environment. The worker blocks reading a request this test never writes, so
/// its limits can be read. Returns the limits in `want` as the worker shows
/// them once they match, or after 5 seconds, and whether
/// `/proc/<pid>/status` was owned by root then: the kernel does that for a process that is not
/// dumpable.
#[cfg(target_os = "linux")]
fn worker_limits(
    script: &str,
    cpu: &str,
    memory: &str,
    want: &[(&str, (&str, &str))],
) -> (Shown, Shown, bool) {
    use std::os::unix::fs::MetadataExt;
    let mut child = Command::new("sh")
        .args(["-c", script, "sh", WORKER])
        .env(CPU_ENV, cpu)
        .env(MEMORY_ENV, memory)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start the worker");
    let pid = child.id();
    let read = || -> Shown { want.iter().map(|(name, _)| proc_limit(pid, name)).collect() };
    let wanted: Shown = want
        .iter()
        .map(|(_, (s, h))| Some((s.to_string(), h.to_string())))
        .collect();
    let root_owned =
        || std::fs::metadata(format!("/proc/{pid}/status")).is_ok_and(|m| m.uid() == 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut got = read();
    while (got != wanted || !root_owned()) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        got = read();
    }
    let not_dumpable = root_owned();
    let _ = child.kill();
    let _ = child.wait();
    (got, wanted, not_dumpable)
}

#[cfg(target_os = "linux")]
#[test]
fn the_worker_sets_its_limits_below_inherited_hard_limits() {
    // The wrapper allows core files and lowers the hard CPU and address-space
    // limits, then replaces itself with the worker.
    let (got, wanted, not_dumpable) = worker_limits(
        "ulimit -S -c 1024 && ulimit -t 100 && ulimit -v 2097152 && exec \"$1\"",
        "1000",
        &(4u64 << 30).to_string(),
        &[
            ("Max core file size", ("0", "0")),
            ("Max cpu time", ("99", "100")),
            ("Max address space", ("2147483648", "2147483648")),
        ],
    );
    assert_eq!(got, wanted);
    assert!(not_dumpable, "the worker is still dumpable");
}

/// The largest CPU limit in seconds that the kernel keeps: it counts CPU time
/// in nanoseconds, in 64 bits.
#[cfg(target_os = "linux")]
const MAX_CPU_SECONDS: u64 = u64::MAX / 1_000_000_000;

#[cfg(target_os = "linux")]
#[test]
fn a_cpu_limit_too_large_for_the_kernel_is_lowered_to_the_largest_it_keeps() {
    // A hard limit of MAX_CPU_SECONDS + 1 wraps to under a second in the
    // kernel, and 2^63 + 4 seconds wraps to 4 seconds.
    let soft = (MAX_CPU_SECONDS - 1).to_string();
    let hard = MAX_CPU_SECONDS.to_string();
    for cpu in [MAX_CPU_SECONDS - 1, MAX_CPU_SECONDS, (1u64 << 63) + 4, u64::MAX] {
        let (got, wanted, _) = worker_limits(
            "exec \"$1\"",
            &cpu.to_string(),
            &(1u64 << 30).to_string(),
            &[("Max cpu time", (&soft, &hard))],
        );
        assert_eq!(got, wanted, "CPU limit {cpu}");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn a_cpu_limit_too_large_for_the_kernel_stays_below_an_inherited_hard_limit() {
    // As Duration::MAX gives. The soft limit stays below the inherited hard
    // limit, so the run ends with SIGXCPU and not SIGKILL.
    let (got, wanted, _) = worker_limits(
        "ulimit -t 2 && exec \"$1\"",
        &u64::MAX.to_string(),
        &(1u64 << 30).to_string(),
        &[("Max cpu time", ("1", "2"))],
    );
    assert_eq!(got, wanted);
}

#[cfg(unix)]
#[test]
fn a_bad_limit_is_refused() {
    let (status, stderr) = run_worker_directly(&[(MEMORY_ENV, "lots".to_string())], &two_lists(1));
    assert_eq!(exit_code(&status), Some(2), "{stderr}");
    assert!(stderr.contains(MEMORY_ENV), "{stderr}");
}

#[cfg(target_os = "linux")]
#[test]
fn a_worker_whose_parent_is_already_gone_exits() {
    let (status, stderr) =
        run_worker_directly(&[(PARENT_ENV, u64::MAX.to_string())], &two_lists(1));
    assert_eq!(exit_code(&status), Some(2), "{stderr}");
    assert!(stderr.contains("has exited"), "{stderr}");
}

/// Whether process `pid` is running: it exists and is not a zombie.
#[cfg(target_os = "linux")]
fn running(pid: u32) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        // The state follows the command name in parentheses.
        Ok(stat) => stat
            .rsplit_once(") ")
            .is_some_and(|(_, rest)| !rest.starts_with('Z') && !rest.starts_with('X')),
        Err(_) => false,
    }
}

#[cfg(target_os = "linux")]
#[test]
fn a_worker_dies_with_its_parent() {
    // A shell starts the worker and exits. The worker reads its request from a
    // pipe this test holds open and never writes to, so it cannot finish on its
    // own. Only the parent-death signal stops it.
    let script =
        format!("exec 3<&0; {PARENT_ENV}=$$ \"$1\" <&3 >/dev/null 2>&1 & echo $!; sleep 0.5");
    let mut shell = Command::new("sh")
        .args(["-c", &script, "sh", WORKER])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("run the shell");
    let held = shell.stdin.take().expect("piped");
    let mut line = String::new();
    BufReader::new(shell.stdout.take().expect("piped"))
        .read_line(&mut line)
        .expect("the worker's pid");
    let pid: u32 = line.trim().parse().expect("a pid");
    assert!(shell.wait().expect("the shell exits").success());
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut alive = running(pid);
    while alive && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        alive = running(pid);
    }
    if alive {
        // Do not leave it running.
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
    }
    drop(held);
    assert!(!alive, "the worker {pid} outlived its parent");
}
