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
//! * [`verify`]: checking a submitted program against its task.
//! * [`prompt`]: the payload handed to a proposer.
//! * [`proposer`]: handing out batches and taking in submissions.
//! * [`iterate`]: ending an iteration with the step of `dc_driver`.
//! * [`cost`]: measuring what frontier programs cost on their task's sweep.
//! * `blocking`: the threads that run checks and tool calls, which give
//!   back the memory a large evaluation freed.
//!
//! # Tools
//!
//! The binary serves one of two tool sets. The operator's server
//! (`dc-mcp --state-dir <dir>`) has `dc_create_run`, `dc_run_status`,
//! `dc_proposal_stats`, `dc_frontiers`, `dc_check_pending`,
//! `dc_finish_iteration`, `dc_add_note`, and `dc_list_runs`. They show a run's
//! library, frontier sizes and results, which a proposer in the blind arm must
//! not see. A proposer's server serves one run in one mode and has only that
//! mode's two tools: `dc-mcp --state-dir <dir> --proposer <run>` has
//! `dc_next_batch` and `dc_submit`, and `dc-mcp --state-dir <dir> --namer
//! <run>` has `dc_next_naming_job` and `dc_submit_naming` (decision D23).
//! Connect a proposer session only to a proposer's server.
//!
//! Started as `dc-mcp --stitch-worker`, the binary is the Stitch worker
//! process of `dc_finish_iteration` and serves nothing (see [`iterate`]).

mod blocking;
pub mod cost;
pub mod iterate;
pub mod journal;
pub mod naming;
pub mod prompt;
pub mod proposer;
pub mod server;
pub mod state;
pub mod store;
pub mod verify;
