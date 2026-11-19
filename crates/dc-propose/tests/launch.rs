//! Tests of the `dc-propose` binary against a fake `claude` that records how
//! it was started and prints a canned stream-json transcript.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Duration, Instant};

use dc_mcp::state::{Arm, Event, LibraryEntry, RunConfig, RunId, StitchSettings};
use dc_mcp::store::Store;
use serde_json::{json, Value};

const MODEL: &str = "claude-test-1";
const NEXT: &str = "mcp__dc__dc_next_batch";
const SUBMIT: &str = "mcp__dc__dc_submit";
const NEXT_NAMING: &str = "mcp__dc__dc_next_naming_job";
const SUBMIT_NAMING: &str = "mcp__dc__dc_submit_naming";

/// The fake `claude`. Each call records its arguments (NUL-separated), its
/// working directory and what that directory holds, then plays the first
/// file left in `queue/` (in name order), or `default` when the queue is
/// empty. In a played file, `@CWD@` becomes the working directory; a line
/// `@SLEEP n` sleeps, a line `@TOUCH name` makes a file, a line `@CLOSE`
/// closes standard output, a line `@BG` starts a background `sleep 300`
/// that holds no pipe of the call and records its pid in `bg.<pid>`, and a
/// line `@BGOUT` does the same with a `sleep 300` that keeps standard output.
const FAKE: &str = r#"#!/usr/bin/env bash
set -u
d="$FAKE_DIR"
id="$$"
cwd="$(pwd -P)"
printf '%s\0' "$@" > "$d/argv.$id"
printf '%s' "$cwd" > "$d/cwd.$id"
ls -A > "$d/ls.$id"
env > "$d/env.$id"
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
    @CLOSE) exec 1>&- ;;
    @BGOUT) sleep 300 </dev/null 2>/dev/null & echo $! > "$d/bg.$id" ;;
    *) printf '%s\n' "${line//@CWD@/$cwd}" ;;
  esac
done < "$out"
rmdir "$d/running.$id"
"#;

struct Fixture {
    root: tempfile::TempDir,
}

impl Fixture {
    /// A fixture whose run has no function waiting to be named.
    fn new() -> Self {
        Fixture::with_names_pending(0)
    }

    /// A fixture whose run has `pending` library functions waiting to be
    /// named (autodoc on if there are any).
    fn with_names_pending(pending: usize) -> Self {
        let root = tempfile::tempdir().unwrap();
        for d in ["fake", "fake/queue", "state", "logs"] {
            fs::create_dir(root.path().join(d)).unwrap();
        }
        let claude = root.path().join("claude");
        fs::write(&claude, FAKE).unwrap();
        fs::set_permissions(&claude, fs::Permissions::from_mode(0o755)).unwrap();
        let fixture = Fixture { root };
        fixture.set_default(&submitted(0));
        let store = Store::open(&fixture.path("state")).unwrap();
        let run = RunId::new("r1").unwrap();
        let config = RunConfig {
            autodoc: pending > 0,
            arm: Arm::Blind,
            library_learning: true,
            max_memory: 512 << 20,
            seed: 3,
            proposals_per_task: 1,
            programs_per_batch: 1,
            lease_ms: 60_000,
            max_unchecked_per_batch: 3,
            frontier_size: 5,
            stitch: StitchSettings::default(),
            prelude: dc_mcp::verify::shipped_prelude(),
            compared_with: None,
            approach_hints: false,
            show_received: false,
        };
        store
            .create_run(&run, config, vec!["collapse/merge".into()])
            .unwrap();
        if pending > 0 {
            let entries = (0..pending)
                .map(|_| LibraryEntry {
                    arity: 1,
                    term: "(lam $0)".into(),
                    ..LibraryEntry::default()
                })
                .collect();
            store
                .append(&run, vec![Event::LibraryExtended { entries }])
                .unwrap();
        }
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

/// A batch session told that `count` new functions wait to be named.
fn names_pending(count: u64) -> Vec<String> {
    vec![
        good_init(),
        tool_use("t1", NEXT, json!({"worker": "w"})),
        tool_result(
            "t1",
            &json!({"batch": null, "wait": {"reason": "new library functions are being named",
                                              "retry_after_seconds": 5,
                                              "names_pending": count}})
            .to_string(),
            false,
        ),
        result("NO_BATCH"),
    ]
}

fn naming_init() -> String {
    init(
        &[NEXT_NAMING, SUBMIT_NAMING],
        json!([{"name": "dc", "status": "connected"}]),
        MODEL,
        "@CWD@",
    )
}

/// A naming session that names job `job`'s function `ident`.
fn named(job: u64) -> Vec<String> {
    vec![
        naming_init(),
        tool_use("t1", NEXT_NAMING, json!({"worker": "w"})),
        tool_result(
            "t1",
            &json!({"job": {"job": job, "token": "ab", "lease_seconds": 1800,
                            "prompt": "name it"}, "wait": null})
            .to_string(),
            false,
        ),
        tool_use(
            "t2",
            SUBMIT_NAMING,
            json!({"job": job, "token": "ab", "name": "ident", "doc": "Returns x."}),
        ),
        tool_result(
            "t2",
            &json!({"job": job, "name": "ident", "doc": "Returns x.", "refused": []}).to_string(),
            false,
        ),
        result(&format!("NAMED {job}")),
    ]
}

/// A naming session that finds no job, with `names_pending` as the wait says.
fn no_job(reason: &str, names_pending: u64) -> Vec<String> {
    vec![
        naming_init(),
        tool_use("t1", NEXT_NAMING, json!({"worker": "w"})),
        tool_result(
            "t1",
            &json!({"job": null, "wait": {"reason": reason, "retry_after_seconds": 60,
                                            "names_pending": names_pending}})
            .to_string(),
            false,
        ),
        result("NO_JOB"),
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
    briefs.sort();
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

#[test]
fn an_interrupt_ends_a_session_that_closed_its_output() {
    let f = Fixture::new();
    f.set_default(&[
        good_init(),
        "@BG".to_owned(),
        "@CLOSE".to_owned(),
        "@SLEEP 60".to_owned(),
    ]);
    let mut child = f
        .command(&["--sessions", "1", "--session-timeout", "120"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while f.background().is_empty() {
        assert!(started.elapsed() < Duration::from_secs(20));
        std::thread::sleep(Duration::from_millis(50));
    }
    // Let the launcher see the output close.
    std::thread::sleep(Duration::from_millis(500));
    let interrupted = Instant::now();
    Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    let status = child.wait().unwrap();
    assert!(interrupted.elapsed() < Duration::from_secs(10));
    assert!(!status.success());
    assert_eq!(outcomes(&f.summary()), ["interrupted"]);
    f.assert_no_background_left();
}

#[test]
fn the_batch_a_session_leased_is_recorded_whatever_its_outcome() {
    let f = Fixture::new();
    let mut lines = submitted(4);
    lines[4] = tool_result("t2", "the token does not match the batch", true);
    lines[5] = result("FAILED");
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1"]);
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["failed"]);
    assert_eq!(summary["sessions"][0]["leased_batches"], json!([4]));
}

#[test]
fn a_session_directory_inside_the_state_directory_is_refused() {
    let f = Fixture::new();
    fs::create_dir(f.path("state/tmp")).unwrap();
    let out = f
        .command(&["--sessions", "1"])
        .env("TMPDIR", f.path("state/tmp"))
        .output()
        .unwrap();
    assert!(!out.status.success(), "{out:?}");
    assert!(f.calls().is_empty());
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("state"), "{detail}");
}

#[test]
fn the_launcher_may_run_in_the_directory_that_holds_the_sessions() {
    let f = Fixture::new();
    fs::create_dir_all(f.path("work/tmp")).unwrap();
    let out = f
        .command(&["--sessions", "1"])
        .current_dir(f.path("work"))
        .env("TMPDIR", f.path("work/tmp"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    assert_eq!(outcomes(&f.summary()), ["submitted"]);
}

#[test]
fn a_first_request_without_usage_is_stopped() {
    let mut lines = submitted(0);
    lines[1] = json!({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": NEXT, "input": {"worker": "w"}}
    ]}})
    .to_string();
    let f = Fixture::new();
    lines.insert(2, "@SLEEP 30".to_owned());
    f.set_default(&lines);
    let out = f.run(&["--sessions", "2", "--parallel", "1"]);
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["lockdown"]);
    let detail = summary["sessions"][0]["detail"].as_str().unwrap();
    assert!(detail.contains("no usage"), "{detail}");
}

#[test]
fn zero_parallel_sessions_is_a_usage_error() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "1", "--parallel", "0"]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
}

#[test]
fn every_batch_a_session_leased_is_recorded() {
    let f = Fixture::new();
    let mut lines = submitted(3);
    let again = submitted(6);
    // A second dc_next_batch call and reply, then no submission.
    lines.truncate(3);
    lines.push(again[1].clone());
    lines.push(again[2].clone());
    lines.push(result("gave up"));
    f.set_default(&lines);
    let out = f.run(&["--sessions", "1"]);
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["failed"]);
    assert_eq!(summary["sessions"][0]["leased_batches"], json!([3, 6]));
    assert_eq!(summary["sessions"][0]["next_batch_calls"], 2);
}

#[test]
fn a_session_that_exits_while_its_output_is_held_open_ends_at_once() {
    let f = Fixture::new();
    // The session exits without a result; a process it started keeps the
    // pipe open.
    f.set_default(&[good_init(), "@BGOUT".to_owned()]);
    let started = Instant::now();
    let out = f.run(&["--sessions", "1", "--session-timeout", "60"]);
    assert!(started.elapsed() < Duration::from_secs(15));
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["failed"]);
    assert_eq!(summary["sessions"][0]["timed_out"], false);
    f.assert_no_background_left();
}

#[test]
fn zero_sessions_is_a_usage_error() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "0"]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
}

const HELD: &str = "every function still to name is held by another job";
const NONE: &str = "no function waits to be named";

/// The kinds and outcomes of a launch's sessions, as `kind:outcome`.
fn kinds(summary: &Value) -> Vec<String> {
    summary["sessions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            format!(
                "{}:{}",
                s["kind"].as_str().unwrap(),
                s["outcome"].as_str().unwrap()
            )
        })
        .collect()
}

/// `transcript` with a pause of `seconds` after its init event.
fn slow(mut transcript: Vec<String>, seconds: u64) -> Vec<String> {
    transcript.insert(1, format!("@SLEEP {seconds}"));
    transcript
}

#[test]
fn the_launch_names_new_functions_before_it_asks_for_batches() {
    let f = Fixture::with_names_pending(1);
    f.queue("1", &named(0));
    let out = f.run(&["--sessions", "3", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(summary["names_pending_at_start"], 1);
    // Once the one pending name is given, batches start.
    assert_eq!(
        kinds(&summary),
        ["naming:named", "batch:submitted", "batch:submitted"]
    );
    let session = &summary["sessions"][0];
    assert_eq!(session["job"], 0);
    assert_eq!(session["name"], "ident");
    assert_eq!(session["refused"], json!([]));
    assert_eq!(session["next_naming_job_calls"], 1);
    assert_eq!(session["submit_naming_calls"], 1);
    assert_eq!(session["leased_jobs"], json!([0]));

    let calls = f.calls();
    assert_eq!(calls.len(), 3);
    for (i, call) in calls.iter().enumerate() {
        let args = f.argv(call);
        let mut allowed = values_after(&args, "--allowedTools");
        allowed.sort_unstable();
        // The server lists only the kind's tools, so none is denied.
        assert!(!args.contains(&"--disallowedTools".to_owned()), "{args:?}");
        let brief = value_after(&args, "--system-prompt");
        let worker = summary["sessions"][i]["worker"].as_str().unwrap();
        assert!(brief.contains(worker), "{brief}");
        // Each kind's MCP configuration, kept in the log directory, starts
        // the server in that kind's mode, so the server lists only the
        // kind's two tools.
        let config_path = value_after(&args, "--mcp-config");
        assert!(
            config_path.ends_with(&format!("{worker}.mcp.json")),
            "{config_path}"
        );
        let config: Value =
            serde_json::from_str(&fs::read_to_string(config_path).unwrap()).unwrap();
        let state = f.path("state").canonicalize().unwrap();
        let mode = if i == 0 { "--namer" } else { "--proposer" };
        assert_eq!(
            config["mcpServers"]["dc"]["args"],
            json!(["--state-dir", state.to_str().unwrap(), mode, "r1"]),
            "session {i}"
        );
        if i == 0 {
            assert_eq!(allowed, [NEXT_NAMING, SUBMIT_NAMING]);
            assert!(brief.contains("dc_next_naming_job"), "{brief}");
            assert!(!brief.contains("dc_next_batch"), "{brief}");
        } else {
            assert_eq!(allowed, [NEXT, SUBMIT]);
            assert!(!brief.contains("naming"), "{brief}");
        }
    }
}

#[test]
fn a_launch_whose_run_has_no_names_pending_starts_with_batches() {
    let f = Fixture::new();
    let out = f.run(&["--sessions", "2", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(summary["names_pending_at_start"], 0);
    assert_eq!(kinds(&summary), ["batch:submitted", "batch:submitted"]);
}

#[test]
fn a_launch_for_a_run_that_does_not_exist_starts_no_session() {
    let f = Fixture::new();
    let out = f
        .command(&["--sessions", "2"])
        .args(["--run", "r2"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    assert!(f.calls().is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cannot read run r2"), "{stderr}");
}

#[test]
fn names_that_become_pending_during_a_launch_are_named_before_more_batches() {
    let f = Fixture::new();
    // The operator ends an iteration that adds a function during the launch.
    f.queue("1", &submitted(0));
    f.queue("2", &names_pending(1));
    f.queue("3", &named(0));
    let out = f.run(&["--sessions", "4", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        kinds(&f.summary()),
        [
            "batch:submitted",
            "batch:names_pending",
            "naming:named",
            "batch:submitted"
        ]
    );
}

#[test]
fn a_naming_session_that_finds_nothing_to_name_starts_batches_again() {
    let f = Fixture::new();
    // Two names pending, but another proposer gives them.
    f.queue("1", &names_pending(2));
    f.queue("2", &no_job(NONE, 0));
    let out = f.run(&["--sessions", "3", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        kinds(&f.summary()),
        [
            "batch:names_pending",
            "naming:nothing_to_name",
            "batch:submitted"
        ]
    );
}

#[test]
fn a_naming_session_with_the_batch_tools_is_stopped() {
    let f = Fixture::new();
    f.queue("1", &names_pending(1));
    // The default transcript's init shows the batch tools.
    let out = f.run(&["--sessions", "3", "--parallel", "1"]);
    assert!(!out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["names_pending", "lockdown"]);
    let detail = summary["sessions"][1]["detail"].as_str().unwrap();
    assert!(detail.contains(NEXT), "{detail}");
}

#[test]
fn the_launch_stops_when_every_function_to_name_is_held_elsewhere() {
    let f = Fixture::new();
    f.queue("1", &names_pending(1));
    f.queue("2", &no_job(HELD, 1));
    let out = f.run(&["--sessions", "5", "--parallel", "1"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(f.calls().len(), 2);
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["names_pending", "naming_held"]);
    assert_eq!(
        summary["sessions"][1]["detail"],
        format!("{HELD}; ask again in 60 s")
    );
}

#[test]
fn no_more_naming_sessions_run_at_once_than_names_are_pending() {
    let f = Fixture::with_names_pending(1);
    f.queue("1", &slow(named(0), 1));
    // Any further naming session would play the default, a batch session's
    // transcript, and fail the lockdown. No batch session starts while the
    // name is being given.
    let out = f.run(&["--sessions", "5", "--parallel", "4"]);
    assert!(out.status.success(), "{out:?}");
    let mut got = kinds(&f.summary());
    got.sort();
    let mut want = vec!["naming:named"];
    want.extend(["batch:submitted"; 4]);
    want.sort_unstable();
    assert_eq!(got, want);
    let calls = f.calls();
    assert_eq!(f.read(&format!("fake/conc.{}", calls[0])).trim(), "1");
    assert!(f.argv(&calls[1]).iter().any(|a| a == NEXT));
}

#[test]
fn a_session_that_finds_the_names_held_by_this_launch_waits_for_them() {
    let f = Fixture::new();
    f.queue("1", &names_pending(2));
    f.queue("2", &names_pending(2));
    f.queue("3", &slow(named(0), 2));
    f.queue("4", &no_job(HELD, 2));
    f.queue("5", &no_job(HELD, 1));
    let out = f.run(&["--sessions", "6", "--parallel", "2"]);
    assert!(out.status.success(), "{out:?}");
    // The first session that found a name held did not end the launch: no
    // naming session started until this launch's other one had ended. The
    // next found the other name still held, with none running, and ended it.
    let calls = f.calls();
    assert_eq!(calls.len(), 5);
    assert_eq!(f.read(&format!("fake/conc.{}", calls[4])).trim(), "1");
    let mut got = kinds(&f.summary());
    got.sort();
    assert_eq!(
        got,
        [
            "batch:names_pending",
            "batch:names_pending",
            "naming:named",
            "naming:naming_held",
            "naming:naming_held"
        ]
    );
}

#[test]
fn a_naming_held_from_an_earlier_phase_does_not_stop_the_launch() {
    let f = Fixture::new();
    f.queue("1", &names_pending(2));
    f.queue("2", &names_pending(2));
    f.queue("3", &slow(no_job(HELD, 2), 2));
    // Another proposer named both meanwhile.
    f.queue("4", &no_job(NONE, 0));
    let out = f.run(&["--sessions", "6", "--parallel", "2"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(f.calls().len(), 6);
    let mut got = kinds(&f.summary());
    got.sort();
    assert_eq!(
        got,
        [
            "batch:names_pending",
            "batch:names_pending",
            "batch:submitted",
            "batch:submitted",
            "naming:naming_held",
            "naming:nothing_to_name"
        ]
    );
}

#[test]
fn names_pending_from_an_earlier_phase_starts_no_naming_session() {
    let f = Fixture::new();
    f.queue("1", &names_pending(1));
    f.queue("2", &slow(names_pending(1), 2));
    f.queue("3", &named(0));
    // Any naming session after the name is given would play the default, a
    // batch session's transcript, and fail the lockdown. The default is slow,
    // so that sessions are still to start when session 2 ends.
    f.set_default(&slow(submitted(0), 1));
    let out = f.run(&["--sessions", "6", "--parallel", "2"]);
    assert!(out.status.success(), "{out:?}");
    let mut got = kinds(&f.summary());
    got.sort();
    assert_eq!(
        got,
        [
            "batch:names_pending",
            "batch:names_pending",
            "batch:submitted",
            "batch:submitted",
            "batch:submitted",
            "naming:named"
        ]
    );
}

#[test]
fn a_naming_session_killed_after_its_answer_was_taken_still_named() {
    let f = Fixture::new();
    f.queue("1", &names_pending(1));
    let mut lines = named(3);
    // No result event: the session hangs after the server took its answer.
    lines[5] = "@SLEEP 30".to_owned();
    f.queue("2", &lines);
    let out = f.run(&[
        "--sessions",
        "2",
        "--session-timeout",
        "2",
        "--parallel",
        "1",
    ]);
    assert!(out.status.success(), "{out:?}");
    let summary = f.summary();
    assert_eq!(outcomes(&summary), ["names_pending", "named"]);
    assert_eq!(summary["sessions"][1]["job"], 3);
    assert_eq!(summary["sessions"][1]["timed_out"], true);
}

/// A session that takes work, asks again while it holds it, and stops.
fn asks_again_while_holding(
    kind_next: &str,
    work: Value,
    wait: Value,
    init: String,
) -> Vec<String> {
    vec![
        init,
        tool_use("t1", kind_next, json!({"worker": "w"})),
        tool_result("t1", &work.to_string(), false),
        tool_use("t2", kind_next, json!({"worker": "w"})),
        tool_result("t2", &wait.to_string(), false),
        result("NO_BATCH"),
    ]
}

#[test]
fn a_batch_session_that_asks_again_while_it_holds_a_batch_failed_and_the_launch_goes_on() {
    let f = Fixture::new();
    f.queue(
        "1",
        &asks_again_while_holding(
            NEXT,
            json!({"batch": {"batch": 0, "token": "ab", "programs_wanted": 1,
                             "lease_seconds": 1800, "prompt": "write solve"}, "wait": null}),
            json!({"batch": null, "wait": {"reason": "you hold batch 0", "retry_after_seconds": 9,
                                            "names_pending": 0}}),
            good_init(),
        ),
    );
    let out = f.run(&["--sessions", "2", "--parallel", "1"]);
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(kinds(&f.summary()), ["batch:failed", "batch:submitted"]);
}

#[test]
fn a_naming_session_that_asks_again_while_it_holds_a_job_failed_and_the_launch_goes_on() {
    let f = Fixture::with_names_pending(2);
    f.queue(
        "1",
        &asks_again_while_holding(
            NEXT_NAMING,
            json!({"job": {"job": 0, "token": "ab", "lease_seconds": 1800, "prompt": "name it"},
                   "wait": null}),
            json!({"job": null, "wait": {"reason": "you hold naming job 0",
                                          "retry_after_seconds": 9, "names_pending": 2}}),
            naming_init(),
        ),
    );
    f.queue("2", &named(1));
    f.queue("3", &named(2));
    let out = f.run(&["--sessions", "4", "--parallel", "1"]);
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(
        kinds(&f.summary()),
        [
            "naming:failed",
            "naming:named",
            "naming:named",
            "batch:submitted"
        ]
    );
}
