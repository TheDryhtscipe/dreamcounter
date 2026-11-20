//! Analysis of a DreamCounter arm comparison (decision D33): each run's
//! metrics from its saved records, the exact paired sign-flip test over
//! pairs of runs, its interval by test inversion, the call, the iteration-0
//! check, the spread between runs, and exploratory secondary metrics.
//!
//! The `dc-compare` binary reads a comparison spec and prints the report.

pub mod compare;
pub mod records;
pub mod render;
pub mod stats;
