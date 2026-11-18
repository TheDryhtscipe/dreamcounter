//! The memory limit bounds the resident memory of the process, not only the
//! allocator's count (bead dreamcounter-a3db93).
//!
//! This file holds one test, so the process runs nothing else while it
//! measures its own peak resident set (`VmHWM`). Linux only.
//!
//! Each run is measured from the resident memory before it. Memory the
//! allocator keeps after earlier runs is resident too, and is not part of
//! this bound: see `dc_tasks::harness::DEFAULT_MAX_MEMORY`.

#![cfg(target_os = "linux")]

use dc_core::{sexpr, Arena};
use dc_tasks::harness::{check_examples, CheckFailure, HarnessConfig, RunFailure};
use dc_tasks::task::{Example, Inputs, TaskFamily};

use dc_core::Scheme;
use dc_eval::Value;

/// A family of type `Int -> Int` with one example, for the runaway
/// candidates to fail.
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

    fn generate(&self, _n: u64, _rng: &mut dc_tasks::rng::TaskRng) -> Inputs {
        vec![Value::Int(0)]
    }

    fn oracle(&self, _inputs: &[Value]) -> Value {
        Value::Int(0)
    }
}

/// A field of `/proc/self/status`, in kB.
fn status_kb(field: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status
        .lines()
        .find(|l| l.starts_with(field))
        .unwrap_or_else(|| panic!("no {field} in /proc/self/status"));
    line[field.len()..]
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn one_runaway_run_keeps_the_peak_resident_memory_near_the_limit() {
    const LIMIT: u64 = 64 << 20;
    // Each runaway candidate gets fuel to hold four times the limit, so with
    // no memory limit the peak would be at least 256 MB above the start.
    let runaways: [(&str, u64); 5] = [
        // A list of ints, and a list of pairs: dropping them when the run
        // stops frees them one cell at a time.
        (
            "(lam (length (loop 1000000000 nil (lam (lam (cons $1 $0))))))",
            16,
        ),
        (
            "(lam (length (loop 1000000000 nil (lam (lam (cons (pair $1 $1) $0))))))",
            32,
        ),
        (
            "(lam (array_size (array_tabulate 1000000000000 (lam $0))))",
            32,
        ),
        (
            "(lam (length (unfold (lam (some (pair $0 (add $0 1)))) 0)))",
            16,
        ),
        (
            "(lam (array_size (fst (loop 1000000000 (pair (array_tabulate 8192 (lam 0)) nil) \
             (lam (lam (pair (array_set (fst $0) (mod $1 8192) $1) (cons (fst $0) (snd $0)))))))))",
            300,
        ),
    ];
    for (src, per_step) in runaways {
        // Freed memory stays resident for this thread, so each run is
        // measured from its own start: writing 5 to `clear_refs` resets the
        // peak to the resident memory now.
        std::fs::write("/proc/self/clear_refs", "5").unwrap();
        let start = status_kb("VmRSS:") << 10;
        let mut arena = Arena::new();
        let p = sexpr::parse(&mut arena, src).unwrap();
        let config = HarnessConfig {
            max_memory: LIMIT,
            example_fuel: 4 * LIMIT / per_step,
            ..HarnessConfig::default()
        };
        let got = check_examples(&Runaway, &arena, p, &config);
        let peak = status_kb("VmHWM:") << 10;
        // The allocator's own overhead (headers, rounding of small blocks)
        // puts the resident memory a little above the count it keeps:
        // measured on Linux with glibc, 64 to 76 MB for a 64 MiB limit.
        assert!(
            peak.saturating_sub(start) <= LIMIT + LIMIT / 4,
            "{src}: peak resident memory {} MB above the start, limit {} MB",
            peak.saturating_sub(start) >> 20,
            LIMIT >> 20
        );
        assert!(
            matches!(
                got,
                Err(CheckFailure::Example {
                    failure: RunFailure::MemoryLimit { .. },
                    ..
                })
            ),
            "{src}: {got:?}"
        );
    }
}
