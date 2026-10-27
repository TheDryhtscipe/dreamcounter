//! CEK evaluator for dc with exact step counting and fuel (spec §4 to §7).
//!
//! [`eval`] evaluates a closed core term; [`apply`] evaluates a closed program and
//! applies it to input values. Both return an [`Outcome`]: a value, a runtime error
//! or a timeout, each with the exact number of steps taken. The step count is a
//! deterministic function of the term and the inputs; [`cost`] holds the whole cost
//! model in one table.
//!
//! The machine keeps its continuation on an explicit stack and never recurses on the
//! host stack over terms or values, so deep recursion in dc cannot overflow the Rust
//! stack.
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

pub use machine::{apply, eval, ErrorKind, Outcome, Stuck, UNLIMITED};
pub use value::{Env, Fun, List, MemoKey, Tree, Value};
