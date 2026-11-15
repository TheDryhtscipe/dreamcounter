//! Starts proposer sessions for a dc-mcp run (D11, D21).
//!
//! Each session is a headless Claude Code process (`claude -p`) that runs in
//! a new, empty directory of its own. It has no built-in tools, loads no user,
//! project or local settings (so no rules, hooks or plugins), and has one MCP
//! server: the run's proposer server, `dc-mcp --proposer`, whose two tools are
//! the only ones it may call. Its brief tells it to take one batch, write the
//! programs the batch's prompt asks for, and submit them, so each batch is
//! written in a context of its own (D11 requirement 3).
//!
//! The launcher reads each session's stream-json transcript as it comes. The
//! session's `init` event must come before anything it does, and it must show
//! the lockdown: exactly the two tools, exactly the one server, connected, the
//! session's own directory, and the model asked for. A session that shows
//! anything else is killed, and no further session starts. A session that
//! leaves anything in its directory fails the same way. Every transcript,
//! command line and MCP configuration is kept in the log directory, with a
//! summary of every session.
//!
//! The launcher does not create runs or finish iterations: those are the
//! operator's tools. It starts sessions until it has started the number
//! asked for, or until a session finds no batch to take.

use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rustix::process::{kill_process_group, Pid, Signal};
use serde_json::{json, Map, Value};

/// The name of the run's server in each session's MCP configuration.
pub const SERVER: &str = "dc";
/// The proposer's tool that hands out a batch, as the session names it.
pub const NEXT_BATCH: &str = "mcp__dc__dc_next_batch";
/// The proposer's tool that takes a batch's programs, as the session names it.
pub const SUBMIT: &str = "mcp__dc__dc_submit";
/// The prefix of each session's directory name.
pub const DIR_PREFIX: &str = "dc-proposer-";

/// What the launcher is asked to do.
#[derive(Clone, Debug)]
pub struct Config {
    /// dc-mcp's state directory.
    pub state_dir: PathBuf,
    /// The run the sessions propose for.
    pub run: String,
    /// The model each session runs on: the full id the session reports.
    pub model: String,
    /// The `dc-mcp` executable.
    pub dc_mcp: PathBuf,
    /// The `claude` executable.
    pub claude: PathBuf,
    /// Where transcripts, command lines and the summary go.
    pub log_dir: PathBuf,
    /// How many sessions to start at most.
    pub sessions: usize,
    /// How many sessions run at once at most.
    pub parallel: usize,
    /// How long a session may run before it is killed.
    pub session_timeout: Duration,
}

/// How a session ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// It took a batch and the server received its programs.
    Submitted {
        /// The batch.
        batch: u64,
        /// How many programs the server received.
        received: u64,
    },
    /// The server had no batch for it.
    NoBatch {
        /// What the server said.
        detail: String,
    },
    /// It showed more than the lockdown allows, or could not show that it
    /// was locked down. It was stopped, and no further session starts.
    Lockdown {
        /// What it showed.
        detail: String,
    },
    /// It ran past its time and was killed.
    Timeout,
    /// It ended without submitting, or could not be started.
    Failed {
        /// Why.
        detail: String,
    },
}

impl Outcome {
    /// The outcome's name in the summary.
    pub fn name(&self) -> &'static str {
        match self {
            Outcome::Submitted { .. } => "submitted",
            Outcome::NoBatch { .. } => "no_batch",
            Outcome::Lockdown { .. } => "lockdown",
            Outcome::Timeout => "timeout",
            Outcome::Failed { .. } => "failed",
        }
    }

    /// Whether the launch counts this session as a failure.
    pub fn is_failure(&self) -> bool {
        !matches!(self, Outcome::Submitted { .. } | Outcome::NoBatch { .. })
    }
}

/// One session's record in the summary.
#[derive(Clone, Debug)]
pub struct Session {
    /// The worker id the session was given.
    pub worker: String,
    /// How it ended.
    pub outcome: Outcome,
    /// The model it reported, if it reported one.
    pub model: Option<String>,
    /// The `usage`, `total_cost_usd`, `num_turns` and `duration_ms` of its
    /// final `result` event, as it gave them.
    pub result: Map<String, Value>,
    /// How many times it called `dc_next_batch` and `dc_submit`.
    pub calls: (u64, u64),
    /// How long it ran, in milliseconds, as the launcher saw it.
    pub elapsed_ms: u64,
}

impl Session {
    fn failed(worker: &str, outcome: Outcome) -> Self {
        Session {
            worker: worker.to_owned(),
            outcome,
            model: None,
            result: Map::new(),
            calls: (0, 0),
            elapsed_ms: 0,
        }
    }

    fn to_json(&self) -> Value {
        let mut s = Map::new();
        s.insert("worker".into(), json!(self.worker));
        s.insert("outcome".into(), json!(self.outcome.name()));
        s.insert("model".into(), json!(self.model));
        s.insert("next_batch_calls".into(), json!(self.calls.0));
        s.insert("submit_calls".into(), json!(self.calls.1));
        s.insert("elapsed_ms".into(), json!(self.elapsed_ms));
        match &self.outcome {
            Outcome::Submitted { batch, received } => {
                s.insert("batch".into(), json!(batch));
                s.insert("received".into(), json!(received));
            }
            Outcome::NoBatch { detail }
            | Outcome::Lockdown { detail }
            | Outcome::Failed { detail } => {
                s.insert("detail".into(), json!(detail));
            }
            Outcome::Timeout => {}
        }
        for (k, v) in &self.result {
            s.insert(k.clone(), v.clone());
        }
        Value::Object(s)
    }
}

/// An error that stops the launch before it starts a session.
#[derive(Debug)]
pub struct LaunchError(pub String);

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LaunchError {}

fn io_error(what: &str, path: &Path, e: &std::io::Error) -> LaunchError {
    LaunchError(format!("{what} {}: {e}", path.display()))
}

/// The brief a session gets as its whole system prompt.
pub fn brief(worker: &str, model: &str) -> String {
    format!(
        "You write programs for a research server. You have two tools, \
         dc_next_batch and dc_submit, and nothing else. No one will answer \
         questions, so do not ask any.\n\
         \n\
         1. Call dc_next_batch with worker \"{worker}\". Call it once.\n\
         2. If its reply has no batch, stop and reply NO_BATCH.\n\
         3. Otherwise, read the batch's prompt and write the programs it asks \
         for, as it says. You cannot run them; work them out yourself, and \
         check them before you submit.\n\
         4. Call dc_submit with worker \"{worker}\", the batch's number and \
         token, your programs, and model \"{model}\". Only your first \
         submission that the server takes counts, so submit once. If it \
         replies with an error, do what the error says.\n\
         5. Reply with one line: SUBMITTED and the batch's number.\n"
    )
}

/// The MCP configuration of a session: the run's proposer server only.
pub fn mcp_config(dc_mcp: &Path, state_dir: &Path, run: &str) -> Value {
    json!({"mcpServers": {SERVER: {
        "type": "stdio",
        "command": dc_mcp,
        "args": ["--state-dir", state_dir, "--proposer", run],
    }}})
}

/// The arguments `claude` is started with.
pub fn claude_args(model: &str, mcp_config: &Path, brief: &str) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "-p",
        "Begin.",
        "--model",
        model,
        // No user, project or local settings: no rules, hooks or plugins.
        "--restricted",
        // No built-in tools.
        "--tools",
        "",
        "--strict-mcp-config",
        "--mcp-config",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    args.push(mcp_config.into());
    args.extend(
        [
            "--allowedTools",
            NEXT_BATCH,
            SUBMIT,
            "--system-prompt",
            brief,
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        .iter()
        .map(OsString::from),
    );
    args
}

/// Checks a session's `init` event against the lockdown, for a session that
/// runs in `cwd` (canonical) on `model`. `Err` says what it showed instead.
pub fn check_init(init: &Value, cwd: &Path, model: &str) -> Result<(), String> {
    let mut tools: Vec<&str> = init["tools"]
        .as_array()
        .ok_or("the init event lists no tools")?
        .iter()
        .map(|t| t.as_str().unwrap_or("?"))
        .collect();
    tools.sort_unstable();
    let mut want = [NEXT_BATCH, SUBMIT];
    want.sort_unstable();
    if tools != want {
        return Err(format!("the session has the tools {tools:?}"));
    }
    let servers: Vec<(&str, &str)> = init["mcp_servers"]
        .as_array()
        .ok_or("the init event lists no MCP servers")?
        .iter()
        .map(|s| {
            (
                s["name"].as_str().unwrap_or("?"),
                s["status"].as_str().unwrap_or("?"),
            )
        })
        .collect();
    if servers != [(SERVER, "connected")] {
        return Err(format!(
            "the session has the MCP servers {servers:?} (name, status)"
        ));
    }
    let reported = init["cwd"].as_str().unwrap_or("?");
    if !Path::new(reported).canonicalize().is_ok_and(|p| p == cwd) {
        return Err(format!("the session runs in {reported}"));
    }
    let reported = init["model"].as_str().unwrap_or("?");
    if reported != model {
        return Err(format!("the session runs on the model {reported}"));
    }
    Ok(())
}

/// The text of a `tool_result` block's content.
fn result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

/// How a session ended, from its events, and how many times it called
/// `dc_next_batch` and `dc_submit`. A received submission wins over
/// anything else; then a reply with no batch; otherwise the session failed,
/// and the last tool error or its last words say how.
pub fn classify(events: &[Value]) -> (Outcome, (u64, u64)) {
    let mut names: Vec<(String, String)> = Vec::new();
    let mut calls = (0, 0);
    let mut submitted = None;
    let mut no_batch = None;
    let mut error = None;
    let mut last_words = None;
    for e in events {
        for b in e["message"]["content"].as_array().into_iter().flatten() {
            match b["type"].as_str() {
                Some("tool_use") => {
                    let name = b["name"].as_str().unwrap_or("?").to_owned();
                    if name == NEXT_BATCH {
                        calls.0 += 1;
                    } else if name == SUBMIT {
                        calls.1 += 1;
                    }
                    names.push((b["id"].as_str().unwrap_or("").to_owned(), name));
                }
                Some("tool_result") => {
                    let id = b["tool_use_id"].as_str().unwrap_or("");
                    let name = names
                        .iter()
                        .find(|(i, _)| i == id)
                        .map_or("?", |(_, n)| n.as_str());
                    let text = result_text(&b["content"]);
                    if b["is_error"].as_bool() == Some(true) {
                        error = Some(format!("{name}: {text}"));
                        continue;
                    }
                    let reply: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
                    if name == SUBMIT {
                        if let (Some(batch), Some(received)) =
                            (reply["batch"].as_u64(), reply["received"].as_u64())
                        {
                            submitted = Some(Outcome::Submitted { batch, received });
                        }
                    } else if name == NEXT_BATCH && reply.get("batch").is_some_and(Value::is_null) {
                        let wait = &reply["wait"];
                        let reason = wait["reason"].as_str().unwrap_or("no reason given");
                        let detail = match wait["retry_after_seconds"].as_u64() {
                            Some(s) => format!("{reason}; ask again in {s} s"),
                            None => reason.to_owned(),
                        };
                        no_batch = Some(Outcome::NoBatch { detail });
                    }
                }
                _ => {}
            }
        }
        if e["type"] == "result" {
            last_words = e["result"].as_str().map(str::to_owned);
        }
    }
    let outcome = submitted.or(no_batch).unwrap_or_else(|| Outcome::Failed {
        detail: error
            .or(last_words.map(|t| format!("the session ended with: {t}")))
            .unwrap_or_else(|| "the session ended without submitting".to_owned()),
    });
    (outcome, calls)
}

/// The fields of the last `result` event that the summary keeps.
fn result_fields(events: &[Value]) -> Map<String, Value> {
    let mut kept = Map::new();
    if let Some(r) = events.iter().rev().find(|e| e["type"] == "result") {
        for k in ["usage", "total_cost_usd", "num_turns", "duration_ms"] {
            if let Some(v) = r.get(k) {
                kept.insert(k.to_owned(), v.clone());
            }
        }
    }
    kept
}

/// What a directory holds, by name.
fn entries(dir: &Path) -> Result<Vec<String>, std::io::Error> {
    let mut names = Vec::new();
    for e in fs::read_dir(dir)? {
        names.push(e?.file_name().to_string_lossy().into_owned());
    }
    names.sort();
    Ok(names)
}

/// Kills `child`'s process group, which holds the session and what it
/// started (its MCP server), and reaps `child`.
fn kill_group(child: &mut Child) {
    let _ = kill_process_group(Pid::from_child(child), Signal::KILL);
    let _ = child.kill();
    let _ = child.wait();
}

/// Waits for `child` to exit until `deadline`, then kills its group.
/// Returns whether it exited by itself.
fn reap(child: &mut Child, deadline: Instant) -> bool {
    loop {
        if let Ok(Some(_)) = child.try_wait() {
            return true;
        }
        if Instant::now() >= deadline {
            kill_group(child);
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The launcher, set up for one launch.
#[derive(Debug)]
pub struct Launcher {
    config: Config,
    /// The state directory, canonical.
    state_dir: PathBuf,
    /// Directories a session's directory must not be inside.
    forbidden: Vec<PathBuf>,
    /// The prefix of this launch's worker ids.
    launch: String,
}

impl Launcher {
    /// Checks `config`, makes the log directory, and gives the launch a
    /// random id.
    pub fn new(config: Config) -> Result<Self, LaunchError> {
        if config.sessions == 0 || config.parallel == 0 {
            return Err(LaunchError(
                "--sessions and --parallel must be at least 1".into(),
            ));
        }
        let state_dir = config
            .state_dir
            .canonicalize()
            .map_err(|e| io_error("cannot open the state directory", &config.state_dir, &e))?;
        fs::create_dir_all(&config.log_dir)
            .map_err(|e| io_error("cannot make the log directory", &config.log_dir, &e))?;
        let log_dir = config
            .log_dir
            .canonicalize()
            .map_err(|e| io_error("cannot open the log directory", &config.log_dir, &e))?;
        let mut forbidden = vec![state_dir.clone(), log_dir];
        // The launcher most likely runs in the repository. The root would
        // forbid everything, so it is left out.
        if let Ok(here) = std::env::current_dir().and_then(|d| d.canonicalize()) {
            if here.parent().is_some() {
                forbidden.push(here);
            }
        }
        let mut bytes = [0u8; 4];
        getrandom::fill(&mut bytes)
            .map_err(|e| LaunchError(format!("cannot make a launch id: {e}")))?;
        let launch = format!(
            "p{}",
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
        );
        Ok(Launcher {
            config,
            state_dir,
            forbidden,
            launch,
        })
    }

    /// This launch's id, the prefix of its worker ids.
    pub fn id(&self) -> &str {
        &self.launch
    }

    /// Starts the sessions, at most `parallel` at once, and returns their
    /// records in the order they started. No session starts after one finds
    /// no batch or fails the lockdown.
    pub fn launch(&self) -> Vec<Session> {
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let done: Mutex<Vec<(usize, Session)>> = Mutex::new(Vec::new());
        let workers = self.config.parallel.min(self.config.sessions);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    let n = next.fetch_add(1, Ordering::SeqCst);
                    if n >= self.config.sessions {
                        return;
                    }
                    let session = self.session(n);
                    if matches!(
                        session.outcome,
                        Outcome::NoBatch { .. } | Outcome::Lockdown { .. }
                    ) {
                        stop.store(true, Ordering::SeqCst);
                    }
                    let _ = writeln!(
                        std::io::stderr(),
                        "dc-propose: {} {}",
                        session.worker,
                        session.outcome.name()
                    );
                    done.lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push((n, session));
                });
            }
        });
        let mut done = done
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        done.sort_by_key(|(n, _)| *n);
        done.into_iter().map(|(_, s)| s).collect()
    }

    /// The summary of a launch, as written to `summary.json`.
    pub fn summary(&self, sessions: &[Session]) -> Value {
        json!({
            "launch": self.launch,
            "run": self.config.run,
            "model": self.config.model,
            "state_dir": self.state_dir,
            "sessions": sessions.iter().map(Session::to_json).collect::<Vec<_>>(),
        })
    }

    /// Runs session `n` to its end.
    fn session(&self, n: usize) -> Session {
        let worker = format!("{}-{n}", self.launch);
        let started = Instant::now();
        let mut session = self
            .run_session(&worker, started)
            .unwrap_or_else(|e| Session::failed(&worker, Outcome::Failed { detail: e.0 }));
        session.elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        session
    }

    fn run_session(&self, worker: &str, started: Instant) -> Result<Session, LaunchError> {
        let c = &self.config;
        let dir = tempfile::Builder::new()
            .prefix(DIR_PREFIX)
            .tempdir()
            .map_err(|e| LaunchError(format!("cannot make a session directory: {e}")))?;
        let cwd = dir
            .path()
            .canonicalize()
            .map_err(|e| io_error("cannot open", dir.path(), &e))?;
        if let Some(f) = self.forbidden.iter().find(|f| cwd.starts_with(f)) {
            return Ok(Session::failed(
                worker,
                Outcome::Lockdown {
                    detail: format!(
                        "the session's directory {} is inside {}; set TMPDIR elsewhere",
                        cwd.display(),
                        f.display()
                    ),
                },
            ));
        }
        let log = |suffix: &str| c.log_dir.join(format!("{worker}{suffix}"));
        let config_path = log(".mcp.json");
        fs::write(
            &config_path,
            mcp_config(&c.dc_mcp, &self.state_dir, &c.run).to_string(),
        )
        .map_err(|e| io_error("cannot write", &config_path, &e))?;
        let config_path = config_path
            .canonicalize()
            .map_err(|e| io_error("cannot open", &config_path, &e))?;
        let args = claude_args(&c.model, &config_path, &brief(worker, &c.model));
        let command = json!({
            "program": c.claude,
            "args": args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>(),
            "cwd": cwd,
        });
        fs::write(log(".cmd.json"), command.to_string())
            .map_err(|e| io_error("cannot write", &log(".cmd.json"), &e))?;
        let stderr =
            File::create(log(".err")).map_err(|e| io_error("cannot write", &log(".err"), &e))?;
        let mut transcript = File::create(log(".jsonl"))
            .map_err(|e| io_error("cannot write", &log(".jsonl"), &e))?;
        let mut child = Command::new(&c.claude)
            .args(&args)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(stderr)
            // A group of its own, so a kill takes what it started too.
            .process_group(0)
            .spawn()
            .map_err(|e| io_error("cannot start", &c.claude, &e))?;
        let stdout = child.stdout.take().expect("stdout is piped");
        let (send, lines) = mpsc::channel::<Result<String, String>>();
        // Not joined: after a kill, a process that left the group could
        // still hold the pipe open.
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let failed = line.is_err();
                if send.send(line.map_err(|e| e.to_string())).is_err() || failed {
                    return;
                }
            }
        });

        let deadline = started
            .checked_add(c.session_timeout)
            .unwrap_or_else(|| started + Duration::from_secs(u64::from(u32::MAX)));
        let mut events: Vec<Value> = Vec::new();
        let mut model = None;
        let mut init_seen = false;
        let mut stopped: Option<Outcome> = None;
        loop {
            match lines.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(Ok(line)) => {
                    let _ = writeln!(transcript, "{line}");
                    let Ok(event) = serde_json::from_str::<Value>(&line) else {
                        continue;
                    };
                    let is_init = event["type"] == "system" && event["subtype"] == "init";
                    if is_init && !init_seen {
                        init_seen = true;
                        model = event["model"].as_str().map(str::to_owned);
                        if let Err(detail) = check_init(&event, &cwd, &c.model) {
                            stopped = Some(Outcome::Lockdown { detail });
                            break;
                        }
                    } else if !init_seen && event["type"] != "system" {
                        stopped = Some(Outcome::Lockdown {
                            detail: format!(
                                "the session sent a {} event before its init event",
                                event["type"]
                            ),
                        });
                        break;
                    }
                    events.push(event);
                }
                Ok(Err(e)) => {
                    stopped = Some(Outcome::Failed {
                        detail: format!("cannot read the session's output: {e}"),
                    });
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    stopped = Some(Outcome::Timeout);
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        if stopped.is_some() {
            kill_group(&mut child);
        } else if !reap(&mut child, deadline) {
            stopped = Some(Outcome::Timeout);
        }

        let left = entries(&cwd).map_err(|e| io_error("cannot list", &cwd, &e))?;
        let (classified, calls) = classify(&events);
        let outcome = match stopped {
            Some(o @ Outcome::Lockdown { .. }) => o,
            _ if !left.is_empty() => Outcome::Lockdown {
                detail: format!("the session left {left:?} in its directory"),
            },
            Some(o) => o,
            None if !init_seen => Outcome::Lockdown {
                detail: "the session never sent its init event".to_owned(),
            },
            None => classified,
        };
        // The directory is removed here.
        drop(dir);
        Ok(Session {
            worker: worker.to_owned(),
            outcome,
            model,
            result: result_fields(&events),
            calls,
            elapsed_ms: 0,
        })
    }
}
