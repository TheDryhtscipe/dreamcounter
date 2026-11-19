//! The threads that run checks and tool calls (bead dreamcounter-c8e05a).
//!
//! Every job that may block, such as checking submissions, measuring costs
//! or reading a journal, runs on Tokio's blocking pool through [`spawn`].
//! The pool has up to 512 threads, and glibc keeps the memory each thread
//! frees in that thread's arena. A check whose program runs to the memory
//! limit leaves its thread holding about the limit, resident, after the
//! program's values are freed, so k such threads held about k times the
//! limit. [`spawn`] gives that memory back when the job ends, and
//! [`releasing`] when one part of a long job ends.
//!
//! Whether to give back is decided from the count `dc_alloc` keeps for the
//! job's thread. Memory a job allocates and another thread frees after the
//! job ended is not seen: it goes back to the job thread's arena, and stays
//! there until a later job gives back, since giving back covers every arena.

use tokio::task::JoinHandle;

/// A job whose thread's heap rose more than this above where it started
/// gives back the memory the system allocator then holds free. It keeps
/// calls that read a little, such as polling tools, from walking every
/// arena; above it, giving back costs far less than the work that grew the
/// heap.
pub(crate) const RELEASE_ABOVE: i64 = 16 << 20;

/// Runs `work` on Tokio's blocking pool. When it ends, or unwinds, after
/// its thread's heap rose more than [`RELEASE_ABOVE`] bytes, the free
/// memory of every arena goes back to the system. Must be called within a
/// Tokio runtime.
pub(crate) fn spawn<T, F>(work: F) -> JoinHandle<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let _release = Release::from_now();
        work()
    })
}

/// Runs `work` on this thread. When it ends, or unwinds, after the thread's
/// heap rose more than [`RELEASE_ABOVE`] bytes, the free memory of every
/// arena goes back to the system. A job that runs many checks calls it for
/// each, so that it does not keep one check's memory while it waits.
pub(crate) fn releasing<T>(work: impl FnOnce() -> T) -> T {
    let _release = Release::from_now();
    work()
}

/// Gives back free memory when dropped, if the thread's heap rose more than
/// [`RELEASE_ABOVE`] since it was made.
#[derive(Debug)]
struct Release {
    base: i64,
}

impl Release {
    fn from_now() -> Release {
        dc_alloc::reset_peak();
        Release {
            base: dc_alloc::live_bytes(),
        }
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        if dc_alloc::peak_bytes().saturating_sub(self.base) > RELEASE_ABOVE {
            dc_alloc::release_free_memory();
        }
    }
}

#[cfg(all(test, target_os = "linux", target_env = "gnu"))]
mod tests {
    use std::sync::{Arc, Barrier};

    use dc_core::{sexpr, Arena, Scheme};
    use dc_eval::Value;
    use dc_tasks::harness::{check_examples, CheckFailure, HarnessConfig, RunFailure};
    use dc_tasks::rng::TaskRng;
    use dc_tasks::task::{Example, Inputs, TaskFamily};

    use super::spawn;

    /// The memory limit of each runaway run.
    const LIMIT: u64 = 64 << 20;
    /// How many runaway jobs run at once, each on a thread of its own.
    const JOBS: usize = 4;
    /// How far the resident memory may stay above where it was before the
    /// jobs, once they are done.
    const BOUND: u64 = LIMIT / 2;

    const CHILD: &str = "DC_MCP_BLOCKING_TEST_CHILD";

    /// A family of type `Int -> Int` with one example, for the runaway
    /// program to fail.
    struct Runaway;

    impl TaskFamily for Runaway {
        fn name(&self) -> &str {
            "runaway"
        }

        fn signature(&self) -> Scheme {
            "Int -> Int".parse().unwrap()
        }

        fn description(&self) -> &str {
            "Return 0."
        }

        fn examples(&self) -> Vec<Example> {
            vec![Example {
                inputs: vec![Value::Int(0)],
                output: Value::Int(0),
            }]
        }

        fn generate(&self, _n: u64, _rng: &mut TaskRng) -> Inputs {
            vec![Value::Int(0)]
        }

        fn oracle(&self, _inputs: &[Value]) -> Value {
            Value::Int(0)
        }
    }

    /// Checks a program that builds a cons list until it reaches the memory
    /// limit, and says whether it did.
    fn runaway_check() -> bool {
        let mut arena = Arena::new();
        let p = sexpr::parse(
            &mut arena,
            "(lam (length (loop 1000000000 nil (lam (lam (cons $1 $0))))))",
        )
        .unwrap();
        let config = HarnessConfig {
            max_memory: LIMIT,
            // Enough fuel to hold four times the limit, at 16 bytes a step.
            example_fuel: 4 * LIMIT / 16,
            ..HarnessConfig::default()
        };
        matches!(
            check_examples(&Runaway, &arena, p, &config),
            Err(CheckFailure::Example {
                failure: RunFailure::MemoryLimit { .. },
                ..
            })
        )
    }

    /// A field of `/proc/self/status`, in bytes.
    fn status_bytes(field: &str) -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let line = status
            .lines()
            .find_map(|line| line.strip_prefix(field))
            .unwrap_or_else(|| panic!("no {field} in /proc/self/status"));
        line.trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .unwrap()
            << 10
    }

    /// Runs [`JOBS`] runaway checks at once through [`spawn`], and prints
    /// the resident memory before, at the peak, and after. Run only as the
    /// child of the test below, so that nothing else runs in its process.
    fn child() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        std::fs::write("/proc/self/clear_refs", "5").unwrap();
        let before = status_bytes("VmRSS:");
        let reached = runtime.block_on(async {
            // Each job waits for the others before it ends, so the jobs are
            // on different threads and every thread is alive at the end, as
            // the threads of a busy server are.
            let barrier = Arc::new(Barrier::new(JOBS));
            let jobs: Vec<_> = (0..JOBS)
                .map(|_| {
                    let barrier = Arc::clone(&barrier);
                    spawn(move || {
                        let reached = runaway_check();
                        barrier.wait();
                        reached
                    })
                })
                .collect();
            let mut reached = 0;
            for job in jobs {
                reached += usize::from(job.await.unwrap());
            }
            reached
        });
        let peak = status_bytes("VmHWM:");
        let after = status_bytes("VmRSS:");
        println!("\nresult {reached} {before} {peak} {after}");
    }

    #[test]
    fn runaway_jobs_on_many_threads_leave_little_resident_memory() {
        if std::env::var_os(CHILD).is_some() {
            child();
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "blocking::tests::runaway_jobs_on_many_threads_leave_little_resident_memory",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed: {stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let line = stdout
            .lines()
            .find_map(|line| line.strip_prefix("result "))
            .unwrap_or_else(|| panic!("no result from the child: {stdout}"));
        let f: Vec<u64> = line.split(' ').map(|n| n.parse().unwrap()).collect();
        let (reached, before, peak, after) = (f[0], f[1], f[2], f[3]);
        let mib = |b: u64| b >> 20;
        println!(
            "{reached} of {JOBS} reached the limit; resident {} MiB before, peak {} MiB, {} MiB \
             after",
            mib(before),
            mib(peak),
            mib(after)
        );
        assert_eq!(reached, JOBS as u64, "every job must reach the limit");
        // The jobs held their memory at once: without that, the test could
        // pass with nothing to give back.
        assert!(
            peak.saturating_sub(before) >= JOBS as u64 * LIMIT * 9 / 10,
            "the peak was only {} MiB above the start",
            mib(peak.saturating_sub(before))
        );
        assert!(
            after.saturating_sub(before) <= BOUND,
            "{JOBS} runaway jobs left {} MiB resident; the bound is {} MiB",
            mib(after.saturating_sub(before)),
            mib(BOUND)
        );
    }
}
