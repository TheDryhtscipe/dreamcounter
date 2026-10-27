//! Task families, input generators, oracles and growth estimation (spec §9).
//!
//! * [`task`]: the task format, [`task::TaskFamily`].
//! * [`rng`]: the seeded random source generators draw from.
//! * [`harness`]: checking candidates and running the growth sweep.
//! * [`growth`]: growth estimators over a sweep.

pub mod growth;
pub mod harness;
pub mod rng;
pub mod task;
