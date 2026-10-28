//! Running the deep part of a recursion on a thread of its own.
//!
//! The parser and the desugarer recurse once per level of nesting, and in a debug
//! build a level costs up to about 11 KiB of stack in the parser and 8 KiB in the
//! desugarer. At [`crate::MAX_DEPTH`] that is more than a 1 MiB stack holds. So
//! each goes on, from level [`HANDOFF_DEPTH`] down, on a new thread with a
//! [`DEEP_STACK`] stack, as the pretty-printer does for a deep term. Hand-written
//! programs seldom nest that deeply, so they compile without starting a thread.

/// The level at which the parser or the desugarer moves to a thread of its own.
/// Recursing this deep takes at most about 300 KiB of the caller's stack in a
/// debug build.
pub(crate) const HANDOFF_DEPTH: usize = 24;

/// The stack size of that thread: about eight times what the levels from
/// [`HANDOFF_DEPTH`] to [`crate::MAX_DEPTH`] need in a debug build.
pub(crate) const DEEP_STACK: usize = 8 << 20;

/// Runs `f` on a new thread with a [`DEEP_STACK`] stack and returns its result.
/// A panic in `f` is passed on. Panics if the system cannot start the thread.
pub(crate) fn on_deep_stack<T: Send>(name: &str, f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name(name.into())
            .stack_size(DEEP_STACK)
            .spawn_scoped(scope, f)
            .expect("the system starts a thread to compile deeply nested input on")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}
