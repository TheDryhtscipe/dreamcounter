//! `dc-mcp --state-dir <dir>`: serves the runs in `<dir>` over stdio.

use std::path::PathBuf;
use std::process::ExitCode;

use dc_mcp::server::DcServer;
use dc_mcp::store::Store;
use rmcp::ServiceExt;

const USAGE: &str = "usage: dc-mcp --state-dir <dir>";

fn state_dir() -> Option<PathBuf> {
    let mut args = std::env::args_os().skip(1);
    let (flag, dir, rest) = (args.next()?, args.next()?, args.next());
    (flag == "--state-dir" && rest.is_none()).then(|| PathBuf::from(dir))
}

fn main() -> ExitCode {
    let Some(dir) = state_dir() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match serve(&dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dc-mcp: {error}");
            ExitCode::FAILURE
        }
    }
}

/// Serves until the client closes the connection.
fn serve(dir: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    let store = Store::open(dir)?;
    // One thread: every tool call is a short file operation, and the store
    // makes callers take turns in any case.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let service = DcServer::new(store).serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}
