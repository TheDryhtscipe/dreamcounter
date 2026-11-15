//! The checker handles the deepest programs a proposer can submit on a small
//! stack, so no stage recurses once per level of a program.
//!
//! A stack overflow aborts the process, so the checks run in a child: this
//! test binary run again, which checks the programs on a thread with a
//! [`SMALL_STACK`] stack. The parent asserts that the child finished.

mod common;

use std::process::Command;

use common::{assert_checks, deepest_programs};
use dc_mcp::verify::{task, Verifier};

/// The child's stack, in bytes, sized for the test profile (opt-level 1, as
/// CI runs it). There the checks need between 64 and 72 KiB, and a plain
/// recursive walk over the compiled term of the deepest program, more than
/// 5000 levels, does not fit. A larger stack would let such a walk through.
/// An opt-level-0 build needs between 224 and 256 KiB, so there the test is
/// at its limit.
const SMALL_STACK: usize = 256 * 1024;

/// Set in the child to the size of its stack, in bytes.
const STACK_ENV: &str = "DC_MCP_DEEP_TEST_STACK";

/// What the child prints, with the number of programs, once their checks are
/// as expected.
const CHECKED: &str = "dc-mcp deep test: checked";

/// The child: checks the programs on a thread with the stack `STACK_ENV`
/// says. It is not a test of its own, so it does nothing unless the parent
/// set the variable.
#[test]
fn checker_child() {
    let Some(stack) = std::env::var_os(STACK_ENV) else {
        return;
    };
    let stack: usize = stack.to_str().unwrap().parse().unwrap();
    let programs = deepest_programs();
    let texts: Vec<String> = programs.iter().map(|p| p.text.clone()).collect();
    let checks = std::thread::Builder::new()
        .stack_size(stack)
        .spawn(move || {
            let family = task("collapse/merge").unwrap();
            let mut verifier = Verifier::new(&[]).unwrap();
            texts
                .iter()
                .map(|t| verifier.check(&*family, t))
                .collect::<Vec<_>>()
        })
        .unwrap()
        .join()
        .unwrap();
    assert_checks(&programs, &checks);
    println!("{CHECKED} {}", programs.len());
}

#[test]
fn the_deepest_programs_are_checked_on_a_small_stack() {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "checker_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(STACK_ENV, SMALL_STACK.to_string())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "the checks did not finish on a {SMALL_STACK}-byte stack: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    // The child ran every check: a child that did nothing passes too.
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("{CHECKED} {}", deepest_programs().len());
    assert!(stdout.contains(&want), "{stdout}");
}
