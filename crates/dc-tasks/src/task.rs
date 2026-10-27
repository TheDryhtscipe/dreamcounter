//! The task format (spec §9).
//!
//! A task family is a problem stated for every input size `n`: "sum a list of
//! `n` integers", say. It gives
//!
//! * a dc type signature that every solution must have,
//! * a natural-language description, which is what the LLM sees,
//! * example inputs and outputs for the prompt,
//! * a generator `generate(n, seed)` of inputs of size `n`,
//! * an oracle in Rust, independent of dc, that gives the expected output for
//!   any input, and
//! * the sweep of sizes over which cost growth is measured.
//!
//! The generator receives a [`TaskRng`] seeded from `(n, seed)` rather than
//! the seed itself, so a family cannot reach for another source of randomness
//! by accident: the same `(n, seed)` always gives the same input. Use
//! [`generate_input`] to call it.
//!
//! Values are `!Send`, so a family builds them on the thread that uses them;
//! the harness is single-threaded.

use dc_core::Scheme;
use dc_eval::Value;

use crate::growth::DEFAULT_SIZES;
use crate::rng::TaskRng;

/// One input to a task: the arguments a solution is applied to, in order.
pub type Inputs = Vec<Value>;

/// An input and its expected output, shown in the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// The arguments, in order.
    pub inputs: Inputs,
    /// The expected result.
    pub output: Value,
}

impl Example {
    /// An example with the given arguments and expected result.
    pub fn new(inputs: Inputs, output: Value) -> Example {
        Example { inputs, output }
    }
}

/// A task family (spec §9).
pub trait TaskFamily {
    /// A short, unique identifier, for logs and results tables.
    fn name(&self) -> &str;

    /// The type every solution must have. Its type variables are rigid: a
    /// solution must work for every choice of them.
    fn signature(&self) -> Scheme;

    /// The task in natural language, as shown to the LLM.
    fn description(&self) -> &str;

    /// Example inputs and outputs for the prompt. A candidate must reproduce
    /// all of them before its cost is measured.
    fn examples(&self) -> Vec<Example>;

    /// An input of size `n`, drawn from `rng` and nothing else.
    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs;

    /// The expected output for `inputs`, computed in Rust without dc.
    fn oracle(&self, inputs: &[Value]) -> Value;

    /// The sizes of the growth sweep, strictly increasing and positive. The
    /// default is `16, 32, ..., 1024`; families whose reference solutions are
    /// super-polynomial declare smaller sweeps.
    fn sizes(&self) -> Vec<u64> {
        DEFAULT_SIZES.to_vec()
    }
}

/// The input of size `n` for seed `seed`: `family.generate(n, TaskRng::new(n, seed))`.
pub fn generate_input<F: TaskFamily + ?Sized>(family: &F, n: u64, seed: u64) -> Inputs {
    family.generate(n, &mut TaskRng::new(n, seed))
}
