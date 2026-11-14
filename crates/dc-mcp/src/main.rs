//! `dc-mcp --state-dir <dir>`: serves the operator's tools for the runs in
//! `<dir>` over stdio.
//!
//! `dc-mcp --state-dir <dir> --proposer <run>`: serves a proposer's tools for
//! the run `<run>` in `<dir>` over stdio. The run must exist.

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use dc_mcp::proposer::SystemClock;
use dc_mcp::server::{DcServer, ProposerServer};
use dc_mcp::state::RunId;
use dc_mcp::store::Store;
use rmcp::ServiceExt;

const USAGE: &str = "usage: dc-mcp --state-dir <dir> [--proposer <run>]";

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
        .enable_all()
        .build()?;
    let served = runtime.block_on(async {
        match proposer {
            None => {
                let service = DcServer::new(store).serve(rmcp::transport::stdio()).await?;
                service.waiting().await?;
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
    runtime.shutdown_timeout(std::time::Duration::from_secs(1));
    served
}
