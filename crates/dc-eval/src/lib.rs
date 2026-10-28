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

pub use machine::{apply, eval, ErrorKind, Outcome, OutcomeWithin, Stuck, UNLIMITED};
pub use value::{DisplayWithin, Env, Fun, List, MemoKey, Tree, Value, DEBUG_NODES};
