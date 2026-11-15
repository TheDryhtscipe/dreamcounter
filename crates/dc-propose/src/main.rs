//! `dc-propose`: starts locked-down proposer sessions for a dc-mcp run.
//!
//! ```text
//! dc-propose --state-dir <dir> --run <run> --model <model id> --log-dir <dir>
//!            --sessions <n> [--parallel <n>] [--session-timeout <seconds>]
//!            [--dc-mcp <path>] [--claude <path>]
//! ```
//!
//! Writes `summary.json` to the log directory and prints it. Exits 0 when
//! every session submitted or found no batch, 1 when one failed, and 2 on a
//! usage error. The project lead starts every run (D11 requirement 5).

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use dc_propose::{Config, Launcher};

const USAGE: &str = "usage: dc-propose --state-dir <dir> --run <run> --model <model id> \
    --log-dir <dir> --sessions <n> [--parallel <n>] [--session-timeout <seconds>] \
    [--dc-mcp <path>] [--claude <path>]";

/// The `dc-mcp` beside this executable, as cargo builds them.
fn sibling_dc_mcp() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("dc-mcp")))
        .unwrap_or_else(|| PathBuf::from("dc-mcp"))
}

fn arguments() -> Result<Config, String> {
    let mut args = std::env::args_os().skip(1);
    let mut state_dir = None;
    let mut run = None;
    let mut model = None;
    let mut log_dir = None;
    let mut sessions = None;
    let mut parallel = 4;
    let mut timeout = 1200;
    let mut dc_mcp = None;
    let mut claude = PathBuf::from("claude");
    while let Some(flag) = args.next() {
        let value: OsString = args.next().ok_or(format!("{flag:?} needs a value"))?;
        let text = || {
            value
                .to_str()
                .map(str::to_owned)
                .ok_or("not UTF-8".to_owned())
        };
        let number = || text()?.parse::<u64>().map_err(|e| format!("{flag:?}: {e}"));
        match flag.to_str() {
            Some("--state-dir") => state_dir = Some(PathBuf::from(&value)),
            Some("--run") => run = Some(text()?),
            Some("--model") => model = Some(text()?),
            Some("--log-dir") => log_dir = Some(PathBuf::from(&value)),
            Some("--sessions") => sessions = Some(number()?),
            Some("--parallel") => parallel = number()?,
            Some("--session-timeout") => timeout = number()?,
            Some("--dc-mcp") => dc_mcp = Some(PathBuf::from(&value)),
            Some("--claude") => claude = PathBuf::from(&value),
            _ => return Err(format!("unknown flag {flag:?}")),
        }
    }
    let size = |n: u64| usize::try_from(n).map_err(|e| e.to_string());
    Ok(Config {
        state_dir: state_dir.ok_or("--state-dir is needed")?,
        run: run.ok_or("--run is needed")?,
        model: model.ok_or("--model is needed")?,
        log_dir: log_dir.ok_or("--log-dir is needed")?,
        sessions: size(sessions.ok_or("--sessions is needed")?)?,
        parallel: size(parallel)?,
        session_timeout: Duration::from_secs(timeout),
        dc_mcp: dc_mcp.unwrap_or_else(sibling_dc_mcp),
        claude,
    })
}

fn main() -> ExitCode {
    let config = match arguments() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("dc-propose: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let summary_path = config.log_dir.join("summary.json");
    let launcher = match Launcher::new(config) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("dc-propose: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("dc-propose: launch {}", launcher.id());
    let sessions = launcher.launch();
    let summary = launcher.summary(&sessions);
    let text = serde_json::to_string_pretty(&summary).unwrap_or_default();
    if let Err(e) = std::fs::write(&summary_path, &text) {
        eprintln!("dc-propose: cannot write {}: {e}", summary_path.display());
    }
    println!("{text}");
    if sessions.iter().any(|s| s.outcome.is_failure()) {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
