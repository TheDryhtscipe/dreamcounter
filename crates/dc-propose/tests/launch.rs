//! Tests of the `dc-propose` binary against a fake `claude` that records how
//! it was started and prints a canned stream-json transcript.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const MODEL: &str = "claude-test-1";
const NEXT: &str = "mcp__dc__dc_next_batch";
const SUBMIT: &str = "mcp__dc__dc_submit";

/// The fake `claude`. Each call records its arguments (NUL-separated), its
/// working directory and what that directory holds, then plays the first
/// file left in `queue/` (in name order), or `default` when the queue is
/// empty. In a played file, `@CWD@` becomes the working directory and
/// `@SCRATCH@` a scratch directory that it makes as Claude Code does (under
/// `fake/cc/`, named after the working directory); a line `@SLEEP n` sleeps,
/// a line `@TOUCH name` makes a file, and a line `@BG` starts a background
/// `sleep 300` that holds no pipe of the call and records its pid in `bg.<pid>`.
const FAKE: &str = r#"#!/usr/bin/env bash
set -u
d="$FAKE_DIR"
id="$$"
cwd="$(pwd -P)"
printf '%s\0' "$@" > "$d/argv.$id"
printf '%s' "$cwd" > "$d/cwd.$id"
ls -A > "$d/ls.$id"
env > "$d/env.$id"
scratch="$d/cc/${cwd//\//-}/u1/scratchpad"
mkdir -p "$scratch"
mkdir "$d/running.$id"
ls -d "$d"/running.* | wc -l > "$d/conc.$id"
out="$d/default"
for f in $(ls "$d/queue" 2>/dev/null | sort); do
  if mv "$d/queue/$f" "$d/taken.$id" 2>/dev/null; then out="$d/taken.$id"; break; fi
done
while IFS= read -r line; do
  case "$line" in
    @SLEEP*) sleep "${line#@SLEEP }" ;;
    @TOUCH*) touch "${line#@TOUCH }" ;;
    @BG) sleep 300 </dev/null >/dev/null 2>&1 & echo $! > "$d/bg.$id" ;;
    *) line="${line//@CWD@/$cwd}"; printf '%s\n' "${line//@SCRATCH@/$scratch}" ;;
  esac
done < "$out"
rmdir "$d/running.$id"
"#;

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        for d in ["fake", "fake/queue", "state", "logs"] {
            fs::create_dir(root.path().join(d)).unwrap();
        }
        let claude = root.path().join("claude");
        fs::write(&claude, FAKE).unwrap();
        fs::set_permissions(&claude, fs::Permissions::from_mode(0o755)).unwrap();
        let fixture = Fixture { root };
        fixture.set_default(&submitted(0));
        fixture
    }

    fn path(&self, p: &str) -> PathBuf {
        self.root.path().join(p)
    }

    fn set_default(&self, lines: &[String]) {
        fs::write(self.path("fake/default"), lines.join("\n") + "\n").unwrap();
    }

    fn queue(&self, name: &str, lines: &[String]) {
        fs::write(self.path("fake/queue").join(name), lines.join("\n") + "\n").unwrap();
    }

    /// Runs `dc-propose` with `extra` after the usual arguments.
    fn run(&self, extra: &[&str]) -> Output {
        self.command(extra).output().unwrap()
    }

    /// The `dc-propose` command with `extra` after the usual arguments.
    fn command(&self, extra: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_dc-propose"));
        command
            .args(["--state-dir"])
            .arg(self.path("state"))
            .args([
                "--run",
                "r1",
                "--model",
                MODEL,
                "--dc-mcp",
                "/opt/dc/dc-mcp",
            ])
            .arg("--claude")
            .arg(self.path("claude"))
            .arg("--log-dir")
            .arg(self.path("logs"))
            .args(extra)
            .args(["--pass-env", "FAKE_DIR"])
            .env("FAKE_DIR", self.path("fake"))
            .env("DC_PROPOSE_TEST_SECRET", "s3cret");
        command
    }

    /// The pids of the background processes the fake's calls started.
    fn background(&self) -> Vec<u32> {
        fs::read_dir(self.path("fake"))
            .unwrap()
            .filter_map(|e| {
                let name = e.unwrap().file_name().into_string().unwrap();
                name.strip_prefix("bg.")?;
                Some(self.read(&format!("fake/{name}")).trim().parse().unwrap())
            })
            .collect()
    }

    /// Panics if a background process of the fake is still alive, after
    /// killing it.
    fn assert_no_background_left(&self) {
        let pids = self.background();
        assert!(!pids.is_empty(), "the fake started no background process");
        // A killed process's parent may take a moment to reap it.
        std::thread::sleep(Duration::from_millis(300));
        let alive: Vec<u32> = pids.into_iter().filter(|p| alive(*p)).collect();
        for p in &alive {
            let _ = Command::new("kill").args(["-9", &p.to_string()]).status();
        }
        assert!(alive.is_empty(), "left running: {alive:?}");
    }

    /// The pids of the fake's calls, in the order they started.
    fn calls(&self) -> Vec<String> {
        let mut calls: Vec<(std::time::SystemTime, String)> = fs::read_dir(self.path("fake"))
            .unwrap()
            .filter_map(|e| {
                let e = e.unwrap();
                let name = e.file_name().into_string().unwrap();
                let pid = name.strip_prefix("argv.")?.to_owned();
                Some((e.metadata().unwrap().modified().unwrap(), pid))
            })
            .collect();
        calls.sort();
        calls.into_iter().map(|(_, pid)| pid).collect()
    }

    fn argv(&self, pid: &str) -> Vec<String> {
        let raw = fs::read(self.path(&format!("fake/argv.{pid}"))).unwrap();
        let text = String::from_utf8(raw).unwrap();
        let mut args: Vec<String> = text.split('\0').map(str::to_owned).collect();
        assert_eq!(args.pop().as_deref(), Some(""));
        args
    }

    fn read(&self, p: &str) -> String {
        fs::read_to_string(self.path(p)).unwrap()
    }

    fn summary(&self) -> Value {
        serde_json::from_str(&self.read("logs/summary.json")).unwrap()
    }
}

/// Whether process `pid` is alive and not a zombie.
fn alive(pid: u32) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|s| {
        s.rsplit(')')
            .next()
            .is_some_and(|r| !r.trim_start().starts_with('Z'))
    })
}

fn init(tools: &[&str], servers: Value, model: &str, cwd: &str) -> String {
    json!({
        "type": "system", "subtype": "init", "cwd": cwd, "tools": tools,
        "mcp_servers": servers, "model": model,
        "plugins": [{"name": "cc-plugin-privacy", "path": "builtin",
                     "source": "cc-plugin-privacy@builtin"}],
        "scratchpad_path": "@SCRATCH@",
    })
    .to_string()
}

fn good_init() -> String {
    init(
        &[NEXT, SUBMIT],
        json!([{"name": "dc", "status": "connected"}]),
        MODEL,
        "@CWD@",
    )
}

fn tool_use(id: &str, name: &str, input: Value) -> String {
    asking(id, name, input, 2000)
}

/// A tool call from a request of `tokens` input tokens.
fn asking(id: &str, name: &str, input: Value, tokens: u64) -> String {
    json!({"type": "assistant", "message": {
        "content": [{"type": "tool_use", "id": id, "name": name, "input": input}],
        "usage": {"input_tokens": 2, "cache_creation_input_tokens": tokens - 1000,
                  "cache_read_input_tokens": 998, "output_tokens": 30},
    }})
    .to_string()
}

fn tool_result(id: &str, content: &str, is_error: bool) -> String {
    json!({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": id, "content": content, "is_error": is_error}
    ]}})
    .to_string()
}

fn result(text: &str) -> String {
    json!({
        "type": "result", "subtype": "success", "is_error": false, "result": text,
        "duration_ms": 1200, "num_turns": 3, "total_cost_usd": 0.01,
        "usage": {"input_tokens": 10, "output_tokens": 20,
                  "cache_read_input_tokens": 30, "cache_creation_input_tokens": 40},
    })
    .to_string()
}

fn submitted(batch: u64) -> Vec<String> {
    vec![
        good_init(),
        tool_use("t1", NEXT, json!({"worker": "w"})),
        tool_result(
            "t1",
            &json!({"batch": {"batch": batch, "token": "ab", "programs_wanted": 1,
                              "lease_seconds": 1800, "prompt": "write solve"}, "wait": null})
            .to_string(),
            false,
        ),
        tool_use(
            "t2",
            SUBMIT,
            json!({"batch": batch, "programs": ["let solve x = x"]}),
        ),
        tool_result(
            "t2",
            &json!({"batch": batch, "received": 1}).to_string(),
            false,
        ),
        result(&format!("SUBMITTED {batch}")),
    ]
}

fn no_batch() -> Vec<String> {
    vec![
        good_init(),
        tool_use("t1", NEXT, json!({"worker": "w"})),
        tool_result(
            "t1",
            &json!({"batch": null, "wait": {"reason": "every slot is taken",
                                              "retry_after_seconds": 60}})
            .to_string(),
            false,
        ),
        result("NO_BATCH"),
    ]
}

fn value_after<'a>(args: &'a [String], flag: &str) -> &'a str {
    let i = args
        .iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("no {flag} in {args:?}"));
    &args[i + 1]
}

fn values_after<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let i = args
        .iter()
        .position(|a| a == flag)
        .unwrap_or_else(|| panic!("no {flag} in {args:?}"));
    args[i + 1..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .map(String::as_str)
        .collect()
}

fn outcomes(summary: &Value) -> Vec<String> {
    summary["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["outcome"].as_str().unwrap().to_owned())
        .collect()
}

fn under(path: &Path, dir: &Path) -> bool {
    path.starts_with(dir.canonicalize().unwrap())
}

#[test]
fn a_session_runs_locked_down_in_an_empty_directory() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "1"]);
    assert!(out.status.success(), "{out:?}");
    let calls = f.calls();
    assert_eq!(calls.len(), 1);
    let args = f.argv(&calls[0]);

    // Print mode, a prompt, and a transcript the launcher can read.
    assert_eq!(args[0], "-p");
    assert_eq!(value_after(&args, "--output-format"), "stream-json");
    assert!(args.contains(&"--verbose".to_owned()));
    // No user, project or local settings (rules, hooks, plugins), and no
    // built-in tools at all.
    assert!(args.contains(&"--restricted".to_owned()));
    assert_eq!(value_after(&args, "--tools"), "");
    // Only the run's proposer server, and only its two tools allowed.
    assert!(args.contains(&"--strict-mcp-config".to_owned()));
    let mut allowed = values_after(&args, "--allowedTools");
    allowed.sort_unstable();
    assert_eq!(allowed, [NEXT, SUBMIT]);
    let config: Value =
        serde_json::from_str(&fs::read_to_string(value_after(&args, "--mcp-config")).unwrap())
            .unwrap();
    let servers = config["mcpServers"].as_object().unwrap();
    assert_eq!(servers.len(), 1);
    assert_eq!(servers["dc"]["command"], "/opt/dc/dc-mcp");
    let state = f.path("state").canonicalize().unwrap();
    assert_eq!(
        servers["dc"]["args"],
        json!(["--state-dir", state.to_str().unwrap(), "--proposer", "r1"])
    );
    assert_eq!(value_after(&args, "--model"), MODEL);
    assert!(args.contains(&"--no-session-persistence".to_owned()));
    for banned in ["--add-dir", "--dangerously-skip-permissions", "--settings"] {
        assert!(!args.contains(&banned.to_owned()), "{banned} in {args:?}");
    }

    // The brief names the worker id and the model to report.
    let summary = f.summary();
    let worker = summary["sessions"][0]["worker"]
        .as_str()
        .unwrap()
        .to_owned();
    let brief = value_after(&args, "--system-prompt");
    assert!(brief.contains(&format!("\"{worker}\"")), "{brief}");
    assert!(brief.contains(MODEL), "{brief}");

    // The session started in an empty directory of its own, outside the
    // state, the logs and the launcher's directory, and it is gone now.
    assert_eq!(f.read(&format!("fake/ls.{}", calls[0])), "");
    let cwd = PathBuf::from(f.read(&format!("fake/cwd.{}", calls[0])));
    assert!(cwd
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .starts_with("dc-proposer-"));
    assert!(!under(&cwd, &f.path("state")));
    assert!(!under(&cwd, &f.path("logs")));
    assert!(!under(&cwd, &std::env::current_dir().unwrap()));
    assert!(!cwd.exists());

    let session = &summary["sessions"][0];
    assert_eq!(session["outcome"], "submitted");
    assert_eq!(session["batch"], 0);
    assert_eq!(session["received"], 1);
    assert_eq!(session["model"], MODEL);
    assert_eq!(session["usage"]["output_tokens"], 20);
    assert_eq!(summary["run"], "r1");
    // The transcript is kept as the session wrote it.
    let log = f.read(&format!("logs/{worker}.jsonl"));
    assert!(log.contains("SUBMITTED 0"), "{log}");
    assert!(log.contains(cwd.to_str().unwrap()));
}

#[test]
fn every_session_gets_its_own_worker_id_and_directory() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "3", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    let calls = f.calls();
    assert_eq!(calls.len(), 3);
    let mut briefs: Vec<String> = calls
        .iter()
        .map(|c| value_after(&f.argv(c), "--system-prompt").to_owned())
        .collect();
    let mut cwds: Vec<String> = calls
        .iter()
        .map(|c| f.read(&format!("fake/cwd.{c}")))
        .collect();
    briefs.dedup();
    cwds.sort();
    cwds.dedup();
    assert_eq!(briefs.len(), 3);
    assert_eq!(cwds.len(), 3);
    let summary = f.summary();
    let mut workers: Vec<&str> = summary["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["worker"].as_str().unwrap())
        .collect();
    workers.sort_unstable();
    workers.dedup();
    assert_eq!(workers.len(), 3);
}

#[test]
fn sessions_run_in_parallel_up_to_the_limit() {
    let f = Fixture::new();
    let mut slow = vec!["@SLEEP 1".to_owned()];
    slow.extend(submitted(0));
    f.set_default(&slow);
    let out = f.run(&["--sessions", "4", "--parallel", "2"]);
    assert!(out.status.success(), "{out:?}");
    let calls = f.calls();
    assert_eq!(calls.len(), 4);
    let most = calls
        .iter()
        .map(|c| {
            f.read(&format!("fake/conc.{c}"))
                .trim()
                .parse::<u32>()
                .unwrap()
        })
        .max()
        .unwrap();
    assert_eq!(most, 2);
}

#[test]
fn no_session_starts_after_one_finds_no_batch() {
    let f = Fixture::new();
    f.queue("1", &submitted(0));
    f.queue("2", &no_batch());
    let out = f.run(&["--sessions", "5", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(f.calls().len(), 2);
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["submitted", "no_batch"]);
    assert_eq!(
        summary["sessions"][1]["detail"],
        "every slot is taken; ask again in 60 s"
    );
}

/// Runs one session whose init event is `init`, then sleeps; checks that the
/// launcher stops it at once, records a lockdown failure that mentions
/// `why`, starts no other session, and fails.
fn refuses(init: String, why: &str) {
    let f = Fixture::new();
    let mut lines = vec![init, "@SLEEP 30".to_owned()];
    lines.extend(submitted(0).into_iter().skip(1));
    f.set_default(&lines);
    let started = Instant::now();
    let out = f.run(&["--sessions", "3", "--parallel", "1"]);
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(f.calls().len(), 1);
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains(why), "{detail}");
}

#[test]
fn a_session_with_another_tool_is_stopped() {
    refuses(
        init(
            &[NEXT, SUBMIT, "Read"],
            json!([{"name": "dc", "status": "connected"}]),
            MODEL,
            "@CWD@",
        ),
        "Read",
    );
}

#[test]
fn a_session_without_the_server_is_stopped() {
    refuses(
        init(
            &[NEXT, SUBMIT],
            json!([{"name": "dc", "status": "failed"}]),
            MODEL,
            "@CWD@",
        ),
        "failed",
    );
}

#[test]
fn a_session_with_another_server_is_stopped() {
    refuses(
        init(
            &[NEXT, SUBMIT],
            json!([{"name": "dc", "status": "connected"},
                   {"name": "files", "status": "connected"}]),
            MODEL,
            "@CWD@",
        ),
        "files",
    );
}

#[test]
fn a_session_on_another_model_is_stopped() {
    refuses(
        init(
            &[NEXT, SUBMIT],
            json!([{"name": "dc", "status": "connected"}]),
            "claude-other-2",
            "@CWD@",
        ),
        "claude-other-2",
    );
}

#[test]
fn a_session_in_another_directory_is_stopped() {
    refuses(
        init(
            &[NEXT, SUBMIT],
            json!([{"name": "dc", "status": "connected"}]),
            MODEL,
            "/home",
        ),
        "/home",
    );
}

#[test]
fn a_session_that_acts_before_its_init_event_is_stopped() {
    refuses(tool_use("t0", NEXT, json!({"worker": "w"})), "init");
}

#[test]
fn a_session_that_leaves_files_fails_the_lockdown() {
    let f = Fixture::new();
    let mut lines = submitted(0);
    lines.insert(1, "@TOUCH notes.txt".to_owned());
    f.set_default(&lines);
    let out = f.run(&["--sessions", "2", "--parallel", "1"]);
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(f.calls().len(), 1);
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("notes.txt"), "{detail}");
}

#[test]
fn a_session_past_its_time_is_killed() {
    let f = Fixture::new();
    let mut lines = vec![good_init(), "@BG".to_owned(), "@SLEEP 30".to_owned()];
    lines.extend(submitted(0).into_iter().skip(1));
    f.set_default(&lines);
    let started = Instant::now();
    let out = f.run(&["--sessions", "1", "--session-timeout", "1"]);
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(outcomes(&f.summary()), ["timeout"]);
    // What the session started went with it.
    f.assert_no_background_left();
}

#[test]
fn a_refused_submission_is_a_failure() {
    let f = Fixture::new();
    let mut lines = submitted(0);
    lines[4] = tool_result("t2", "the token does not match the batch", true);
    lines[5] = result("FAILED");
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1"]);
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["failed"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("the token does not match"), "{detail}");
}

#[test]
fn bad_arguments_are_a_usage_error() {
    let out = Command::new(env!("CARGO_BIN_EXE_dc-propose"))
        .args(["--run", "r1"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage"));
}

#[test]
fn a_system_event_before_the_init_event_is_stopped() {
    // Hook output comes as system events; anything before init is refused.
    refuses(
        json!({"type": "system", "subtype": "hook_response", "output": "context"}).to_string(),
        "before its init event",
    );
}

#[test]
fn a_session_with_a_user_plugin_is_stopped() {
    let mut event: Value = serde_json::from_str(&good_init()).unwrap();
    event["plugins"].as_array_mut().unwrap().push(
        json!({"name": "notes", "path": "/home/u/.claude/plugins/notes",
                     "source": "notes@market"}),
    );
    refuses(event.to_string(), "notes@market");
}

#[test]
fn a_session_directory_inside_the_logs_is_refused() {
    let f = Fixture::new();
    fs::create_dir(f.path("logs/tmp")).unwrap();
    let out = f
        .command(&["--sessions", "2", "--parallel", "1"])
        .env("TMPDIR", f.path("logs/tmp"))
        .output()
        .unwrap();
    assert!(!out.status.success(), "{out:?}");
    assert!(f.calls().is_empty());
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("TMPDIR"), "{detail}");
}

#[test]
fn what_a_session_started_ends_when_it_ends() {
    let f = Fixture::new();
    let mut lines = submitted(0);
    lines.insert(1, "@BG".to_owned());
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(outcomes(&f.summary()), ["submitted"]);
    f.assert_no_background_left();
}

#[test]
fn a_session_that_hangs_after_its_result_is_ended() {
    let f = Fixture::new();
    let mut lines = submitted(3);
    lines.push("@SLEEP 30".to_owned());
    f.set_default(&lines);
    let started = Instant::now();
    let out = f.run(&["--sessions", "1", "--session-timeout", "60"]);
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["submitted"]);
    assert_eq!(summary["sessions"][0]["batch"], 3);
}

#[test]
fn a_session_killed_after_it_submitted_still_submitted() {
    let f = Fixture::new();
    let mut lines = submitted(2);
    // No result event: the session hangs after the server took its programs.
    lines[5] = "@SLEEP 30".to_owned();
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1", "--session-timeout", "2"]);
    assert!(out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["submitted"]);
    assert_eq!(summary["sessions"][0]["batch"], 2);
    assert_eq!(summary["sessions"][0]["timed_out"], true);
}

#[test]
fn an_interrupt_ends_every_session_and_writes_the_summary() {
    let f = Fixture::new();
    f.set_default(&[good_init(), "@BG".to_owned(), "@SLEEP 60".to_owned()]);
    let mut child = f
        .command(&["--sessions", "4", "--parallel", "2"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    // Wait until both sessions run.
    let started = Instant::now();
    while f.background().len() < 2 {
        assert!(started.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_millis(50));
    }
    let interrupted = Instant::now();
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    let status = child.wait().unwrap();
    assert!(interrupted.elapsed() < Duration::from_secs(10));
    assert!(!status.success());
    assert_eq!(f.calls().len(), 2);
    assert_eq!(outcomes(&f.summary()), ["interrupted", "interrupted"]);
    f.assert_no_background_left();
    for c in f.calls() {
        assert!(!Path::new(&f.read(&format!("fake/cwd.{c}"))).exists());
    }
}

#[test]
fn the_scratch_directory_claude_makes_for_a_session_is_removed() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "1"]);
    assert!(out.status.success(), "{out:?}");
    let call = &f.calls()[0];
    let cwd = f.read(&format!("fake/cwd.{call}"));
    let made = f.path("fake/cc").join(cwd.replace('/', "-"));
    assert!(f.path("fake/cc").is_dir());
    assert!(!made.exists(), "{made:?} is left");
}

#[test]
fn a_session_gets_only_the_allowed_environment() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "1"]);
    assert!(out.status.success(), "{out:?}");
    let env = f.read(&format!("fake/env.{}", f.calls()[0]));
    assert!(env.lines().any(|l| l.starts_with("FAKE_DIR=")), "{env}");
    assert!(env.lines().any(|l| l.starts_with("HOME=")), "{env}");
    assert!(!env.contains("DC_PROPOSE_TEST_SECRET"), "{env}");
}

#[test]
fn a_session_with_a_large_first_request_is_stopped() {
    let f = Fixture::new();
    let mut lines = submitted(0);
    lines[1] = asking("t1", NEXT, json!({"worker": "w"}), 9000);
    lines.insert(2, "@SLEEP 30".to_owned());
    f.set_default(&lines);
    let started = Instant::now();
    let out = f.run(&["--sessions", "2", "--parallel", "1"]);
    assert!(started.elapsed() < Duration::from_secs(20));
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("9000 input tokens"), "{detail}");
    // The cap is the launcher's flag.
    let f = Fixture::new();
    lines.retain(|l| !l.starts_with("@SLEEP"));
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1", "--max-context-tokens", "10000"]);
    assert!(out.status.success(), "{out:?}");
}

#[test]
fn a_session_directory_inside_a_git_repository_is_refused() {
    let f = Fixture::new();
    fs::create_dir_all(f.path("repo/.git")).unwrap();
    fs::create_dir(f.path("repo/tmp")).unwrap();
    let out = f
        .command(&["--sessions", "1"])
        .env("TMPDIR", f.path("repo/tmp"))
        .output()
        .unwrap();
    assert!(!out.status.success(), "{out:?}");
    assert!(f.calls().is_empty());
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("repo"), "{detail}");
}
