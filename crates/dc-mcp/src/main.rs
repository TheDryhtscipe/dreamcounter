//! `dc-mcp --state-dir <dir>`: serves the operator's tools for the runs in
//! `<dir>` over stdio.
//!
//! `dc-mcp --state-dir <dir> --proposer <run>`: serves a proposer's tools for
//! the run `<run>` in `<dir>` over stdio. The run must exist.
//!
//! `dc-mcp --stitch-worker`: runs Stitch once, as the worker process of the
//! operator's `dc_finish_iteration` (`dc_compress::worker`). The server
//! starts it; it reads a request on standard input.

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use dc_mcp::iterate::{WorkerStart, WORKER_ARG};
use dc_mcp::proposer::SystemClock;
use dc_mcp::server::{DcServer, ProposerServer};
use dc_mcp::state::RunId;
use dc_mcp::store::Store;
use rmcp::ServiceExt;

const USAGE: &str = "usage: dc-mcp --state-dir <dir> [--proposer <run>]\n       \
                     dc-mcp --stitch-worker";

/// The stack of the runtime's threads, where tool calls and the checks of
/// submitted programs run. A stack overflow aborts the server, and a proposer's
/// server checks what is left unchecked again when it starts, so a program
/// that overflowed the checker would stop the server each time. So the stack
/// is set here, not left to Tokio's default. Checking the deepest program a
/// proposer can submit needs less than 128 KiB in the tests' build
/// (`tests/deep.rs` checks them on 256 KiB); 8 MiB, what dc-surface gives
/// deep parsing, leaves a wide margin for builds with larger frames.
const THREAD_STACK_BYTES: usize = 8 << 20;

/// The state directory, and the run if the server is a proposer's.
fn arguments() -> Option<(PathBuf, Option<String>)> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [flag, dir] if flag == "--state-dir" => Some((PathBuf::from(dir), None)),
        [flag, dir, role, run] if flag == "--state-dir" && role == "--proposer" => {
            Some((PathBuf::from(dir), Some(run.to_str()?.to_owned())))
        }
        _ => None,
    }
}

fn main() -> ExitCode {
    // Before anything else: a worker sets its own limits and reads only
    // its request.
    let mut args = std::env::args_os().skip(1);
    if args.next().is_some_and(|a| a == WORKER_ARG) && args.next().is_none() {
        return dc_compress::worker::serve();
    }
    let Some((dir, proposer)) = arguments() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match serve(&dir, proposer.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dc-mcp: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Serves until the client closes the connection.
fn serve(dir: &std::path::Path, proposer: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let store = Store::open(dir)?;
    // A proposer's run must exist: its server cannot make one.
    let proposer = match proposer {
        None => None,
        Some(run) => {
            let run = RunId::new(run)?;
            store.state(&run)?;
            Some(run)
        }
    };
    // A mistyped directory would otherwise look like a store with no runs.
    let shown = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_owned());
    // Standard error may be closed.
    let _ = writeln!(
        std::io::stderr(),
        "dc-mcp: serving {} in {}",
        proposer.as_ref().map_or_else(
            || "the runs".to_owned(),
            |r| format!("proposers of run {r}")
        ),
        shown.display()
    );
    // One thread for the protocol. Tool calls run on the blocking pool.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .thread_stack_size(THREAD_STACK_BYTES)
        .enable_all()
        .build()?;
    let checks_in_background = proposer.is_some();
    let served = runtime.block_on(async {
        match proposer {
            None => {
                let server = DcServer::new(store, WorkerStart::this_binary()?);
                let service = server.clone().serve(rmcp::transport::stdio()).await;
                let served = match service {
                    Ok(service) => service.waiting().await.map(|_| ()).map_err(Into::into),
                    Err(e) => Err(e.into()),
                };
                // The client is gone. A Stitch run would go on for as long
                // as its limit allows; stop it, and its call writes nothing.
                server.shut_down();
                return served;
            }
            Some(run) => {
                let server = ProposerServer::new(store, run, Arc::new(SystemClock));
                // Check what a server that stopped left unchecked.
                server.check_in_background();
                let service = server.serve(rmcp::transport::stdio()).await?;
                service.waiting().await?;
            }
        }
        Ok(())
    });
    // A background check may be running or waiting to try again. Its work is
    // committed whole or not at all, so it can be left: the next proposer
    // server takes it up.
    // The operator's server waits for its tool calls to finish. A call that
    // was running Stitch finishes at once: `shut_down` stopped its worker.
    if checks_in_background {
        runtime.shutdown_timeout(std::time::Duration::from_secs(1));
    }
    served
}
