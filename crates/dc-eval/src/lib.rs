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

mod budget;
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
/// allocates: environments, closures, continuation frames, memo tables and
/// keys, lists, trees, arrays (`imbl`'s chunks included) and scratch space.
/// The inputs of [`apply_within`], built before the evaluation starts, are
/// not counted; scratch space over them is.
///
/// # The bound
///
/// An evaluation's count stays within its limit plus a few kilobytes. The
/// machine checks the limit before every step. Within a step, every buffer
/// that can grow in proportion to the data the evaluation holds grows only
/// after a check that the larger buffer fits: the continuation stack, an
/// `unfold`'s output, `tree_fold`'s work lists, memo tables, the list of
/// memo tables, and the scratch space of `eq`, of building a memo key and of
/// a reachability walk. `array_of_list`, `array_to_list`, `foldr`'s copy of
/// its list and the end of `unfold` check every 1024 elements. Dropping a
/// list or tree frees it one cell or node at a time, so it needs no memory in
/// proportion to its length.
///
/// What a step can still allocate unchecked is bounded by the program text
/// and the kind of step, not by the data: the values, environments and
/// frames of the term nodes it passes through, an `array_set`'s path copy (a
/// few `imbl` chunks), and up to 1024 elements of a collection step. The
/// tests in `tests/memory.rs` hold the measured peak of each kind of runaway
/// program within the limit plus 64 KiB.
///
/// The count is of bytes asked of the allocator. The resident memory of the
/// process is higher by the allocator's overhead, and by freed memory it
/// keeps for the thread that freed it; `dc-tasks`'s `tests/peak_rss.rs`
/// measures that for one run.
pub mod memory {
    pub use dc_alloc::{live_bytes, peak_bytes, reset_peak};
}
pub use value::{DisplayWithin, Env, Fun, List, MemoKey, Tree, Value, DEBUG_NODES};
