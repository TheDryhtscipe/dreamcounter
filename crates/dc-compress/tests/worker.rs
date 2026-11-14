//! Stitch in a worker process: the same result as in the caller's process, and
//! a run over its time or memory limit, a panic in Stitch, or a caller that dies
//! does not reach the caller.

use std::io::Write;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use dc_compress::worker::{
    Limits, StitchRequest, Worker, WorkerError, CPU_ENV, MEMORY_ENV, PARENT_ENV,
};
use dc_compress::{compress, compress_isolated, CompressConfig, CompressError, FrontierProgram};
use dc_core::{sexpr, Arena, LibDef};

const WORKER: &str = env!("CARGO_BIN_EXE_dc-stitch-worker");

fn worker(wall: Duration, memory_bytes: u64) -> Worker {
    Worker {
        limits: Limits { wall, memory_bytes },
        ..Worker::new(WORKER)
    }
}

/// A worker whose program is `program` with `args`, not a Stitch worker.
fn stand_in(program: &str, args: &[&str], wall: Duration, memory_bytes: u64) -> Worker {
    Worker {
        program: program.into(),
        args: args.iter().map(Into::into).collect(),
        limits: Limits { wall, memory_bytes },
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
/// grow quickly with `n`: in a release build, `n` = 1,000 took 3 seconds and
/// 217 MB, and `n` = 1,500 more than 2 seconds of CPU time.
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
    let isolated = compress_isolated(&mut a, &[], &programs, &cfg, &w).expect("worker");
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
    let isolated = compress_isolated(&mut a, &libs, &again, &cfg, &w).expect("worker");
    assert_eq!(isolated, second);
}

#[test]
fn a_run_over_the_time_limit_is_stopped() {
    let wall = Duration::from_secs(1);
    let w = worker(wall, Limits::default().memory_bytes);
    let start = Instant::now();
    let result = w.run(&two_lists(1500));
    let took = start.elapsed();
    assert_eq!(result, Err(WorkerError::TimedOut { wall }));
    assert!(took < Duration::from_secs(10), "took {took:?}");
}

#[test]
fn a_run_over_the_memory_limit_is_stopped() {
    // Stitch's thread reserves a stack of 256 MiB, which counts towards the
    // limit. Two lists of 1,000 elements use about 217 MB more.
    let memory_bytes = 384 << 20;
    let w = worker(Duration::from_secs(300), memory_bytes);
    match w.run(&two_lists(1000)) {
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
    // The control for the test above: 384 MiB is what stops it, not the
    // workload. A small list needs little more than the stack.
    let w = worker(Duration::from_secs(300), 384 << 20);
    let steps = w.run(&two_lists(100)).expect("a small run fits");
    assert!(!steps.is_empty());
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
    match Worker::new(WORKER).run(&request) {
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
        compress_isolated(&mut a, &[], &programs, &CompressConfig::default(), &missing),
        Err(CompressError::Worker(WorkerError::Spawn(
            std::io::ErrorKind::NotFound
        )))
    );
}

#[test]
fn a_worker_that_does_not_exit_is_killed_at_the_deadline() {
    let wall = Duration::from_millis(200);
    let w = stand_in("sleep", &["30"], wall, 1 << 30);
    let start = Instant::now();
    assert_eq!(w.run(&two_lists(1)), Err(WorkerError::TimedOut { wall }));
    let took = start.elapsed();
    assert!(took < Duration::from_secs(5), "took {took:?}");
}

#[test]
fn a_worker_that_exits_with_a_failure_is_reported() {
    match stand_in("false", &[], Duration::from_secs(30), 1 << 30).run(&two_lists(1)) {
        Err(WorkerError::Failed { status, .. }) => assert_eq!(exit_code(&status), Some(1)),
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn output_that_is_not_steps_is_refused() {
    let w = stand_in("echo", &["[1, 2]"], Duration::from_secs(30), 1 << 30);
    match w.run(&two_lists(1)) {
        Err(WorkerError::BadOutput { .. }) => {}
        other => panic!("expected BadOutput, got {other:?}"),
    }
}

#[test]
fn output_over_the_memory_limit_is_refused() {
    let w = stand_in(
        "head",
        &["-c", "2000", "/dev/zero"],
        Duration::from_secs(30),
        1000,
    );
    assert_eq!(
        w.run(&two_lists(1)),
        Err(WorkerError::OutputTooLarge { bytes: 1000 })
    );
}

/// Starts the worker binary with `envs` on the request `input`, and waits for
/// it with no limit of the caller's own.
fn run_worker_directly(envs: &[(&str, String)], input: &StitchRequest) -> (ExitStatus, String) {
    let mut child = Command::new(WORKER)
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
    // A shell starts the worker on a long run and exits at once. The worker
    // would run for seconds; the parent-death signal kills it.
    let dir = std::env::temp_dir().join(format!("dc-worker-parent-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let request = dir.join("request.json");
    std::fs::write(
        &request,
        serde_json::to_vec(&two_lists(1500)).expect("serialise"),
    )
    .expect("write the request");
    let script = format!("{PARENT_ENV}=$$ \"$1\" < \"$2\" > /dev/null 2>&1 & echo $!; sleep 0.5");
    let out = Command::new("sh")
        .args(["-c", &script, "sh"])
        .arg(WORKER)
        .arg(&request)
        .output()
        .expect("run the shell");
    let pid: u32 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("the worker's pid");
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
    let _ = std::fs::remove_dir_all(&dir);
    assert!(!alive, "the worker {pid} outlived its parent");
}
