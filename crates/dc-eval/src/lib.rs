//! CEK evaluator for dc with exact step counting and fuel (spec §4 to §7).
//!
//! [`eval`] evaluates a closed core term; [`apply`] evaluates a closed program and
//! applies it to input values. Both return an [`Outcome`]: a value, a runtime error
//! or a timeout, each with the exact number of steps taken. The step count is a
//! deterministic function of the term and the inputs; [`cost`] holds the whole cost
//! model in one table.
//!
//! The machine keeps its continuation on an explicit stack, so deep recursion in dc
//! does not grow the Rust stack. Evaluation, dropping, structural equality and
//! formatting of values use explicit work lists instead of host recursion over
//! the value structure (see [`value`]); the one bounded exception is `imbl`'s own
//! tree inside a single array. Values can share parts, so print or log them with
//! [`Value::display_within`] or [`Outcome::display_within`], whose output is
//! bounded; `Display` writes the whole value. The `dc_core::sexpr` parser and printer used to
//! build test terms do recurse over the term.
//!
//! ```
//! use dc_core::{sexpr, Arena};
//! use dc_eval::{eval, Outcome, Value, UNLIMITED};
//!
//! let mut arena = Arena::new();
//! let t = sexpr::parse(&mut arena, "((lam (add $0 1)) 41)").unwrap();
//! // One β step and one `add`.
//! match eval(&arena, t, UNLIMITED) {
//!     Outcome::Value(v, steps) => assert_eq!((v, steps), (Value::Int(42), 2)),
//!     other => panic!("{other}"),
//! }
//! ```

pub mod cost;
mod machine;
pub mod value;

pub use machine::{
    apply, apply_within, eval, eval_within, ErrorKind, Limits, Outcome, OutcomeWithin, Stuck,
    UNLIMITED,
};

/// The heap count behind [`Limits::memory`] (bead dreamcounter-a3db93).
///
/// Every binary that links `dc-eval` allocates through `dc_alloc`'s counting
/// allocator, which keeps a count of each thread's live heap bytes. An
/// evaluation runs on one thread, so the bytes it holds are its thread's live
/// bytes less those when it started. That covers everything the machine
/// allocates: environments, closures, continuation frames, memo tables,
/// lists, trees, arrays (`imbl`'s chunks included) and work lists. The inputs
/// of [`apply_within`], built before the evaluation starts, are not counted.
///
/// # How far above the limit an evaluation can get
///
/// The machine checks the limit before every step, so after each step the
/// evaluation holds at most the limit plus what that one step allocated.
/// The steps that could allocate a lot are checked inside:
///
/// - the continuation stack and an `unfold`'s output grow only after a check
///   that the larger buffer fits, so they never take an evaluation over;
/// - `array_of_list`, `array_to_list`, `foldr` (which copies its list) and the
///   end of `unfold` (which builds its list) check every 1024 elements, so
///   they overshoot by at most 1024 elements.
///
/// What a step can still allocate unchecked is a constant number of values,
/// environments and frames for each term node it passes through; an
/// `array_set` path copy (a few `imbl` chunks); and the scratch space of an
/// `eq`, a `memorec` key, or a reachability walk over memo tables, which is at
/// most proportional to values the evaluation already holds. The tests in
/// `tests/memory.rs` measure the overshoot for each kind of runaway program.
///
/// The count is of bytes asked of the allocator. The resident memory of the
/// process is higher by the allocator's own overhead and by freed memory it
/// keeps; `dc-tasks`'s `tests/peak_rss.rs` measures that.
pub mod memory {
    pub use dc_alloc::{live_bytes, peak_bytes, reset_peak};
}
pub use value::{DisplayWithin, Env, Fun, List, MemoKey, Tree, Value, DEBUG_NODES};
