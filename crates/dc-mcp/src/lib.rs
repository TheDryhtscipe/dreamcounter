//! The MCP server that owns the state of a DreamCounter run (decision D11).
//!
//! A run is one execution of the library-learning loop: a task set, a library,
//! a frontier of programs for each task, and an iteration number, with a log
//! of everything handed out and received. The server keeps each run on disk
//! and writes every change before it acknowledges it. So a run survives a
//! restart and can be spread over many sessions.
//!
//! * [`state`]: a run's state and the events that change it.
//! * [`journal`]: the file format that holds a run's events.
//! * [`store`]: runs on disk, with locking between servers. Its module docs
//!   say what a commit guarantees and what happens to a torn tail.
//! * [`server`]: the MCP tools.
//!
//! # Tools
//!
//! The server has the operator's tools only: `dc_create_run`, `dc_run_status`,
//! `dc_add_note`, and `dc_list_runs`. They show a run's library size and frontier sizes, which
//! a proposer in the blind arm must not see. Do not connect a proposer session
//! to this server.

pub mod journal;
pub mod server;
pub mod state;
pub mod store;
