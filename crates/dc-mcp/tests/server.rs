//! The server binary, driven over stdio as an MCP client drives it.
//!
//! The client here is written against the MCP wire format directly, with no
//! code shared with the server, so it checks what a real client would see.

use std::io::{BufRead, BufReader, Lines, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

mod common;

use common::{assert_checks, deepest_programs};
use dc_mcp::prompt::PromptTable;
use dc_mcp::state::{Check, Event, FrontierMember, LibraryEntry, LogKind, LogRecord, RunId};
use dc_mcp::store::Store;
use serde_json::{json, Value};

struct Client {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    next_id: u64,
}

impl Client {
    /// Starts the server on `dir` and does the MCP handshake.
    fn start(dir: &Path) -> Client {
        Client::start_with_stderr(dir, Stdio::null())
    }

    /// As `start`, with the server's standard error connected to `stderr`.
    fn start_with_stderr(dir: &Path, stderr: Stdio) -> Client {
        Client::start_with(dir, &[], stderr)
    }

    /// Starts a proposer's server for `run` in `dir`.
    fn start_proposer(dir: &Path, run: &str) -> Client {
        Client::start_with(dir, &["--proposer", run], Stdio::null())
    }

    /// Starts the server on `dir` with `args` after the state directory.
    fn start_with(dir: &Path, args: &[&str], stderr: Stdio) -> Client {
        Client::start_binary(Path::new(env!("CARGO_BIN_EXE_dc-mcp")), dir, args, stderr)
    }

    /// As `start_with`, with the server started from the file `binary`.
    fn start_binary(binary: &Path, dir: &Path, args: &[&str], stderr: Stdio) -> Client {
        let mut child = Command::new(binary)
            .arg("--state-dir")
            .arg(dir)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let mut client = Client {
            child,
            stdin,
            lines,
            next_id: 1,
        };
        let hello = client.request(
            "initialize",
            json!({
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": { "name": "dc-mcp-test", "version": "0" }
            }),
        );
        assert!(hello["capabilities"]["tools"].is_object(), "{hello}");
        client.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }));
        client
    }

    fn send(&mut self, message: &Value) {
        writeln!(self.stdin, "{message}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Sends a request and returns its id, without waiting for the response.
    fn start_request(&mut self, method: &str, params: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }));
        id
    }

    /// The next response from the server: its id and its `result`.
    fn next_response(&mut self) -> (u64, Value) {
        loop {
            let line = self
                .lines
                .next()
                .expect("the server closed stdout")
                .unwrap();
            let message: Value = serde_json::from_str(&line).unwrap();
            if let Some(id) = message["id"].as_u64() {
                assert!(message["error"].is_null(), "{message}");
                return (id, message["result"].clone());
            }
        }
    }

    /// Sends a request and returns its whole response, error or not. No
    /// other request may be waiting for a response.
    fn raw_request(&mut self, method: &str, params: Value) -> Value {
        let id = self.start_request(method, params);
        loop {
            let line = self
                .lines
                .next()
                .expect("the server closed stdout")
                .unwrap();
            let message: Value = serde_json::from_str(&line).unwrap();
            if message["id"].as_u64() == Some(id) {
                return message;
            }
        }
    }

    /// Sends a request and returns the `result` of its response. No other
    /// request may be waiting for a response.
    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.start_request(method, params);
        let (answered, result) = self.next_response();
        assert_eq!(answered, id);
        result
    }

    /// Calls a tool. Returns its structured content, or its error text.
    fn call(&mut self, tool: &str, arguments: Value) -> Result<Value, String> {
        let result = self.request(
            "tools/call",
            json!({ "name": tool, "arguments": arguments }),
        );
        if result["isError"] == json!(true) {
            Err(result["content"][0]["text"].as_str().unwrap().to_owned())
        } else {
            Ok(result["structuredContent"].clone())
        }
    }

    /// Kills the server with SIGKILL, as a crash would.
    fn kill(mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }
}

#[test]
fn the_server_lists_its_tools() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut client = Client::start(dir.path());
    let tools = client.request("tools/list", json!({}));
    let mut names: Vec<&str> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "dc_add_note",
            "dc_check_pending",
            "dc_create_run",
            "dc_finish_iteration",
            "dc_list_runs",
            "dc_proposal_stats",
            "dc_run_status"
        ]
    );
    for tool in tools["tools"].as_array().unwrap() {
        assert!(
            tool["description"].as_str().is_some_and(|d| d.len() > 20),
            "{tool}"
        );
        assert_eq!(tool["inputSchema"]["type"], "object", "{tool}");
    }
    client.kill();
}

#[test]
fn a_run_survives_a_kill_in_the_middle_of_an_iteration() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut client = Client::start(dir.path());
    assert_eq!(
        client.call("dc_list_runs", json!({})).unwrap(),
        json!({ "runs": [] })
    );

    let created = client
        .call(
            "dc_create_run",
            json!({ "run": "r1", "seed": 5, "proposals_per_task": 2, "frontier_size": 5 }),
        )
        .unwrap();
    assert_eq!(created["run"], "r1");
    assert_eq!(created["arm"], "blind");
    assert_eq!(created["seed"], 5);
    assert_eq!(created["proposals_per_task"], 2);
    assert_eq!(created["iteration"], 0);
    assert_eq!(created["library_size"], 0);
    // With no task list, the run gets the whole development corpus.
    let corpus: Vec<String> = dc_tasks::families::development()
        .iter()
        .map(|v| v.family.name().to_owned())
        .collect();
    assert!(corpus.len() > 3);
    let tasks = created["tasks"].as_array().unwrap();
    let names: Vec<&str> = tasks.iter().map(|t| t["name"].as_str().unwrap()).collect();
    assert_eq!(names, corpus);
    assert!(tasks.iter().all(|t| t["frontier_size"] == 0));

    // A second server process on the same directory takes the run through an
    // iteration and into the next one. Here it is a store in this process.
    let run = RunId::new("r1").unwrap();
    let other = Store::open(dir.path()).unwrap();
    let log = |kind, iteration, task: &str| {
        Event::Logged(LogRecord {
            kind,
            iteration,
            task: Some(task.to_owned()),
            text: "text".into(),
        })
    };
    let member = |program: &str| FrontierMember {
        program: program.into(),
    };
    other
        .append(&run, {
            vec![
                log(LogKind::Compression, 0, &corpus[0]),
                log(LogKind::Note, 0, &corpus[0]),
                Event::FrontierSet {
                    task: corpus[0].clone(),
                    members: vec![member("(lam 0)"), member("(lam (lam 1))")],
                },
                Event::LibraryExtended {
                    entries: vec![LibraryEntry {
                        name: None,
                        doc: None,
                        arity: 1,
                        term: "(lam 0)".into(),
                    }],
                },
                Event::IterationFinished {
                    iteration: 0,
                    used: 0,
                },
            ]
        })
        .unwrap();
    other
        .append(&run, vec![log(LogKind::Compression, 1, &corpus[1])])
        .unwrap();

    // The first server sees those changes.
    let before = client
        .call("dc_run_status", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(before["iteration"], 1);
    assert_eq!(before["library_size"], 1);
    assert_eq!(before["tasks"][0]["frontier_size"], 2);
    assert_eq!(before["tasks"][1]["frontier_size"], 0);
    assert_eq!(
        before["log_counts"],
        json!({ "compression": 2, "note": 1, "recovery": 0 })
    );

    client.kill();
    let mut client = Client::start(dir.path());
    assert_eq!(
        client
            .call("dc_run_status", json!({ "run": "r1" }))
            .unwrap(),
        before
    );
    assert_eq!(
        client.call("dc_list_runs", json!({})).unwrap(),
        json!({ "runs": ["r1"] })
    );
    client.kill();
}

#[test]
fn a_server_killed_while_it_writes_keeps_every_note_it_acknowledged() {
    let dir = tempfile::TempDir::new().unwrap();
    let run = RunId::new("r1").unwrap();
    let mut client = Client::start(dir.path());
    let task = dc_tasks::families::development()[0]
        .family
        .name()
        .to_owned();
    client
        .call(
            "dc_create_run",
            json!({ "run": "r1", "seed": 5, "proposals_per_task": 2, "frontier_size": 5, "tasks": [task] }),
        )
        .unwrap();
    // Take the run into its second iteration, so the kill is inside one.
    Store::open(dir.path())
        .unwrap()
        .append(
            &run,
            vec![Event::IterationFinished {
                iteration: 0,
                used: 0,
            }],
        )
        .unwrap();

    // Ask for many notes at once, and kill the server when some are
    // acknowledged and the rest are still being written.
    const ASKED: usize = 300;
    const WAITED_FOR: usize = 40;
    let text = |i: usize| format!("note {i} {}", "x".repeat(2000));
    let mut ids = std::collections::HashMap::new();
    for i in 0..ASKED {
        let id = client.start_request(
            "tools/call",
            json!({ "name": "dc_add_note", "arguments": { "run": "r1", "text": text(i) } }),
        );
        ids.insert(id, i);
    }
    let mut acknowledged = Vec::new();
    while acknowledged.len() < WAITED_FOR {
        let (id, result) = client.next_response();
        assert_eq!(result["isError"], json!(false), "{result}");
        assert_eq!(result["structuredContent"]["iteration"], 1);
        acknowledged.push(ids[&id]);
    }
    client.kill();

    let logged: Vec<String> = Store::open(dir.path())
        .unwrap()
        .log(&run)
        .unwrap()
        .into_iter()
        .map(|record| {
            assert_eq!(record.iteration, 1);
            record.text
        })
        .collect();
    for i in &acknowledged {
        assert!(
            logged.contains(&text(*i)),
            "note {i} was acknowledged and is not in the run"
        );
    }
    let mut distinct = logged.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), logged.len(), "a note is in the run twice");
    assert!(
        logged.len() < ASKED,
        "the server finished before it was killed"
    );

    // A new server has the same run and goes on with it.
    let mut client = Client::start(dir.path());
    let status = client
        .call("dc_run_status", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(status["iteration"], 1);
    assert_eq!(status["log_counts"]["note"], logged.len());
    let status = client
        .call(
            "dc_add_note",
            json!({ "run": "r1", "text": "after the restart" }),
        )
        .unwrap();
    assert_eq!(status["log_counts"]["note"], logged.len() + 1);
    client.kill();
}

#[test]
fn a_locked_run_does_not_hold_up_the_rest_of_the_server() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut client = Client::start(dir.path());
    let task = dc_tasks::families::development()[0]
        .family
        .name()
        .to_owned();
    for run in ["held", "free"] {
        client
            .call(
                "dc_create_run",
                json!({ "run": run, "seed": 1, "proposals_per_task": 1, "frontier_size": 5, "tasks": [task] }),
            )
            .unwrap();
    }
    let lock = std::fs::File::open(dir.path().join("held")).unwrap();
    lock.lock().unwrap();

    let waiting = client.start_request(
        "tools/call",
        json!({ "name": "dc_add_note", "arguments": { "run": "held", "text": "waits" } }),
    );
    // While that call waits for the lock, the server answers these.
    let ping = client.start_request("ping", json!({}));
    assert_eq!(client.next_response().0, ping);
    let list = client.start_request(
        "tools/call",
        json!({ "name": "dc_list_runs", "arguments": {} }),
    );
    let (id, result) = client.next_response();
    assert_eq!(id, list);
    assert_eq!(
        result["structuredContent"],
        json!({ "runs": ["free", "held"] })
    );
    let other = client.start_request(
        "tools/call",
        json!({ "name": "dc_add_note", "arguments": { "run": "free", "text": "goes" } }),
    );
    let (id, result) = client.next_response();
    assert_eq!(id, other);
    assert_eq!(result["structuredContent"]["log_counts"]["note"], 1);

    drop(lock);
    let (id, result) = client.next_response();
    assert_eq!(id, waiting);
    assert_eq!(result["structuredContent"]["log_counts"]["note"], 1);
    client.kill();
}

#[test]
fn a_torn_tail_is_reported_in_the_run_even_with_standard_error_closed() {
    let dir = tempfile::TempDir::new().unwrap();
    let task = dc_tasks::families::development()[0]
        .family
        .name()
        .to_owned();
    let mut client = Client::start(dir.path());
    client
        .call(
            "dc_create_run",
            json!({ "run": "r1", "seed": 1, "proposals_per_task": 1, "frontier_size": 5, "tasks": [task] }),
        )
        .unwrap();
    client
        .call("dc_add_note", json!({ "run": "r1", "text": "one" }))
        .unwrap();
    client.kill();
    let journal = dir.path().join("r1").join("journal");
    let mut bytes = std::fs::read(&journal).unwrap();
    bytes.extend_from_slice(b"2 0000 half a record");
    std::fs::write(&journal, bytes).unwrap();

    // A server whose standard error nobody reads: every write to it fails.
    let mut client = Client::start_with_stderr(dir.path(), Stdio::piped());
    drop(client.child.stderr.take());
    let status = client
        .call("dc_run_status", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(status["log_counts"]["recovery"], 0);
    for (i, text) in ["two", "three"].into_iter().enumerate() {
        let status = client
            .call("dc_add_note", json!({ "run": "r1", "text": text }))
            .unwrap();
        assert_eq!(status["log_counts"]["note"], 2 + i);
        assert_eq!(status["log_counts"]["recovery"], 1);
    }
    let mut files: Vec<String> = std::fs::read_dir(dir.path().join("r1"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(files.len(), 2, "{files:?}");
    assert_eq!(files[0], "journal");
    assert_eq!(
        std::fs::read(dir.path().join("r1").join(&files[1])).unwrap(),
        b"2 0000 half a record"
    );
    client.kill();
}

#[test]
fn bad_requests_are_tool_errors_and_change_nothing() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut client = Client::start(dir.path());
    let first = dc_tasks::families::development()[0]
        .family
        .name()
        .to_owned();
    let create = |run: &str, tasks: Value| json!({ "run": run, "seed": 1, "proposals_per_task": 1, "frontier_size": 5, "tasks": tasks });

    let error = client
        .call("dc_run_status", json!({ "run": "none" }))
        .unwrap_err();
    assert!(error.contains("no run"), "{error}");
    // A Stitch setting that is refused is named as the tool names it, in
    // the tool's unit.
    for (setting, value, said) in [
        (
            "stitch_iterations",
            0,
            "stitch_iterations must be at least 1",
        ),
        (
            "stitch_wall_seconds",
            0,
            "stitch_wall_seconds must be at least 1",
        ),
        (
            "stitch_memory_mib",
            511,
            "stitch_memory_mib must be at least 512",
        ),
    ] {
        let mut arguments = create("s1", json!([first]));
        arguments[setting] = json!(value);
        let error = client.call("dc_create_run", arguments).unwrap_err();
        assert_eq!(error, said);
    }
    let error = client
        .call("dc_run_status", json!({ "run": "s1" }))
        .unwrap_err();
    assert!(error.contains("no run"), "{error}");
    let error = client
        .call("dc_create_run", create("../up", json!([first])))
        .unwrap_err();
    assert!(error.contains("run id"), "{error}");
    let error = client
        .call("dc_run_status", json!({ "run": "a/b" }))
        .unwrap_err();
    assert!(error.contains("run id"), "{error}");
    let error = client
        .call("dc_add_note", json!({ "run": "none", "text": "x" }))
        .unwrap_err();
    assert!(error.contains("no run"), "{error}");
    // A long bad argument is not repeated in full.
    let error = client
        .call("dc_run_status", json!({ "run": "A".repeat(100_000) }))
        .unwrap_err();
    assert!(
        error.contains("run id") && error.len() < 400,
        "{}",
        error.len()
    );
    let long_task = "t".repeat(100_000);
    let error = client
        .call("dc_create_run", create("r1", json!([long_task])))
        .unwrap_err();
    assert!(
        error.contains("no task") && error.len() < 2000,
        "{}",
        error.len()
    );
    // Arguments a tool does not have are refused, not ignored.
    let id = client.start_request(
        "tools/call",
        json!({ "name": "dc_list_runs", "arguments": { "extra": 1 } }),
    );
    let refused = loop {
        let line = client.lines.next().unwrap().unwrap();
        let message: Value = serde_json::from_str(&line).unwrap();
        if message["id"] == json!(id) {
            break message;
        }
    };
    assert!(
        !refused["error"].is_null() || refused["result"]["isError"] == json!(true),
        "{refused}"
    );
    let error = client
        .call("dc_create_run", create("r1", json!(["no-such-task"])))
        .unwrap_err();
    assert!(error.contains("no-such-task"), "{error}");
    let error = client
        .call("dc_create_run", create("r1", json!([])))
        .unwrap_err();
    assert!(error.contains("at least one task"), "{error}");
    let error = client
        .call("dc_create_run", create("r1", json!([first, first])))
        .unwrap_err();
    assert!(error.contains("twice"), "{error}");
    let error = client
        .call(
            "dc_create_run",
            json!({ "run": "r1", "seed": 1, "proposals_per_task": 0, "frontier_size": 5 }),
        )
        .unwrap_err();
    assert!(error.contains("proposals_per_task"), "{error}");
    assert_eq!(
        client.call("dc_list_runs", json!({})).unwrap(),
        json!({ "runs": [] })
    );
    // Only the store's own directory was touched.
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);

    let made = client
        .call("dc_create_run", create("r1", json!([first])))
        .unwrap();
    assert_eq!(
        made["tasks"],
        json!([{ "name": first, "frontier_size": 0 }])
    );
    let error = client
        .call("dc_create_run", create("r1", json!([first])))
        .unwrap_err();
    assert!(error.contains("already"), "{error}");
    assert_eq!(
        client
            .call("dc_run_status", json!({ "run": "r1" }))
            .unwrap(),
        made
    );
    client.kill();
}

#[test]
fn the_binary_refuses_to_start_without_a_state_directory() {
    let output = Command::new(env!("CARGO_BIN_EXE_dc-mcp")).output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("--state-dir"));
}

/// The names of the tools a server lists, sorted.
fn tool_names(client: &mut Client) -> Vec<String> {
    let tools = client.request("tools/list", json!({}));
    let mut names: Vec<String> = tools["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    names.sort_unstable();
    names
}

/// A program that solves collapse/merge: its good reference.
fn solves_merge() -> String {
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == "collapse/merge")
        .unwrap();
    format!(
        "{}\nlet solve = {}\n",
        v.source,
        v.references.last().unwrap().entry
    )
}

#[test]
fn a_proposer_sees_only_its_tools_and_gets_no_results() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start(dir.path());
    let created = operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 1, "frontier_size": 5,
                "tasks": ["collapse/merge", "collapse/cancel"]
            }),
        )
        .unwrap();
    // Naming is off unless asked for (D23).
    assert_eq!(created["autodoc"], false);
    assert_eq!(created["unnamed"], 0);

    let mut proposer = Client::start_proposer(dir.path(), "r1");
    assert_eq!(
        tool_names(&mut proposer),
        [
            "dc_next_batch",
            "dc_next_naming_job",
            "dc_submit",
            "dc_submit_naming"
        ]
    );
    // An operator's tool is not there for a proposer.
    for (tool, arguments) in [
        ("dc_run_status", json!({ "run": "r1" })),
        ("dc_list_runs", json!({})),
        ("dc_proposal_stats", json!({ "run": "r1" })),
    ] {
        let refused = proposer.raw_request(
            "tools/call",
            json!({ "name": tool, "arguments": arguments }),
        );
        assert_eq!(refused["error"]["message"], "tool not found", "{refused}");
    }

    let first = proposer
        .call("dc_next_batch", json!({ "worker": "agent-1" }))
        .unwrap();
    assert_eq!(first["batch"]["batch"], 0);
    assert_eq!(first["batch"]["programs_wanted"], 1);
    let prompt = first["batch"]["prompt"].as_str().unwrap();
    assert!(prompt.contains("solve : List Int -> List Int"), "{prompt}");
    assert!(!prompt.contains("collapse/"));
    let second = proposer
        .call("dc_next_batch", json!({ "worker": "agent-2" }))
        .unwrap();
    assert_eq!(second["batch"]["batch"], 1);
    let none = proposer
        .call("dc_next_batch", json!({ "worker": "agent-3" }))
        .unwrap();
    assert!(none["batch"].is_null(), "{none}");
    assert!(none["wait"]["retry_after_seconds"].as_u64().unwrap() >= 1);

    // A right and a wrong program get the same kind of reply.
    let good = proposer
        .call(
            "dc_submit",
            json!({
                "worker": "agent-1", "batch": 0, "token": first["batch"]["token"],
                "programs": [solves_merge()], "model": "test-model"
            }),
        )
        .unwrap();
    let bad = proposer
        .call(
            "dc_submit",
            json!({
                "worker": "agent-2", "batch": 1, "token": second["batch"]["token"],
                "programs": ["let solve = ("]
            }),
        )
        .unwrap();
    assert_eq!(good, json!({ "batch": 0, "received": 1 }));
    assert_eq!(bad, json!({ "batch": 1, "received": 1 }));
    let unknown = proposer.call(
        "dc_submit",
        json!({ "worker": "agent-1", "batch": 9, "token": "t", "programs": [] }),
    );
    assert!(unknown.unwrap_err().contains("batch 9 was not issued"));
    // Batch 1's token does not open batch 0, even with batch 0's worker id.
    let wrong = proposer.call(
        "dc_submit",
        json!({
            "worker": "agent-1", "batch": 0, "token": second["batch"]["token"],
            "programs": []
        }),
    );
    assert!(wrong.unwrap_err().contains("token"));
    // Batch 0 keeps three foreign submissions; the next is counted, not kept.
    for kept in [true, true, false] {
        let again = proposer.call(
            "dc_submit",
            json!({
                "worker": "agent-1", "batch": 0, "token": "not-the-token",
                "programs": ["let solve xs = xs"]
            }),
        );
        let why = again.unwrap_err();
        assert_eq!(why.contains("counted but not kept"), !kept, "{why}");
    }

    // The operator sees what came in, once the server has checked it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let stats = loop {
        let stats = operator
            .call("dc_proposal_stats", json!({ "run": "r1" }))
            .unwrap();
        if stats["iterations"][0]["pending_checks"] == 0 {
            break stats;
        }
        assert!(std::time::Instant::now() < deadline, "{stats}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let it = &stats["iterations"][0];
    assert_eq!(it["tasks"][0]["received"], 1);
    assert_eq!(it["quota_in"], true);
    assert_eq!(it["tasks_solved"], 1);
    assert_eq!(it["submissions"], 5);
    assert_eq!(it["foreign"], 3);
    assert_eq!(it["refused_foreign"], 1);
    assert_eq!(it["refused_repeats"], 0);
    assert_eq!(it["tasks"][0]["passed"], 1);
    assert_eq!(it["tasks"][1]["compiled"], 0);
    let status = operator
        .call("dc_run_status", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(status["batches_issued"], 2);
    assert_eq!(status["current"]["quota_in"], true);
    assert_eq!(status["current"]["refused_foreign"], 1);

    // Payloads and submissions are in the journal as they were sent.
    proposer.kill();
    operator.kill();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r1").unwrap();
    let mut payloads = Vec::new();
    let mut programs = Vec::new();
    let mut prompts = PromptTable::default();
    store
        .for_each_event(&run, |e| {
            prompts.see(&e);
            match e {
                Event::BatchIssued(b) => payloads.push(prompts.payload_text(&b).unwrap()),
                Event::Submitted(s) => programs.push((s.programs, s.model)),
                _ => {}
            }
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let sent: Value = serde_json::from_str(&payloads[0]).unwrap();
    assert_eq!(sent, first["batch"]);
    assert_eq!(
        programs,
        [
            (vec![solves_merge()], Some("test-model".to_owned())),
            (vec!["let solve = (".to_owned()], None),
            (vec![], None),
            // The refused fourth foreign submission is not here.
            (vec!["let solve xs = xs".to_owned()], None),
            (vec!["let solve xs = xs".to_owned()], None),
        ]
    );
    assert_eq!(store.refusals(&run).unwrap().iteration(0).foreign, 1);
}

/// The checks of the run's checked submissions, in the order they were
/// checked.
fn checks(dir: &Path, run: &str) -> Vec<Vec<Check>> {
    let mut checks = Vec::new();
    Store::open(dir)
        .unwrap()
        .for_each_event(&RunId::new(run).unwrap(), |e| {
            if let Event::Checked(c) = e {
                checks.push(c.checks);
            }
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    checks
}

/// Waits until the run has no submission waiting to be checked.
fn wait_for_checks(operator: &mut Client, proposer: &mut Client, run: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        // A stack overflow in a check aborts the proposer's server.
        assert!(
            proposer.child.try_wait().unwrap().is_none(),
            "the proposer's server stopped"
        );
        let stats = operator
            .call("dc_proposal_stats", json!({ "run": run }))
            .unwrap();
        if stats["iterations"][0]["pending_checks"] == 0 {
            return;
        }
        assert!(std::time::Instant::now() < deadline, "{stats}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

#[test]
fn the_deepest_programs_a_proposer_can_send_do_not_crash_the_server() {
    let programs = deepest_programs();
    let texts: Vec<String> = programs.iter().map(|p| p.text.clone()).collect();
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start(dir.path());
    // One task, so every batch is its.
    operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 64, "frontier_size": 5,
                "programs_per_batch": 32, "tasks": ["collapse/merge"]
            }),
        )
        .unwrap();

    // Sent to a live server, which checks them in the background.
    let mut proposer = Client::start_proposer(dir.path(), "r1");
    let first = proposer
        .call("dc_next_batch", json!({ "worker": "agent-1" }))
        .unwrap();
    proposer
        .call(
            "dc_submit",
            json!({
                "worker": "agent-1", "batch": first["batch"]["batch"],
                "token": first["batch"]["token"], "programs": texts
            }),
        )
        .unwrap();
    wait_for_checks(&mut operator, &mut proposer, "r1");
    let done = checks(dir.path(), "r1");
    assert_eq!(done.len(), 1);
    assert_checks(&programs, &done[0]);
    proposer.kill();

    // Left unchecked by a server that stopped, and checked when the next one
    // starts.
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r1").unwrap();
    let held = dc_mcp::proposer::next_batch(&store, &run, "agent-2", 0, &Default::default())
        .unwrap()
        .batch
        .unwrap();
    dc_mcp::proposer::submit(
        &store,
        &run,
        held.batch,
        "agent-2",
        &held.token,
        texts,
        None,
        1,
    )
    .unwrap();
    let mut proposer = Client::start_proposer(dir.path(), "r1");
    wait_for_checks(&mut operator, &mut proposer, "r1");
    let done = checks(dir.path(), "r1");
    assert_eq!(done.len(), 2);
    assert_checks(&programs, &done[1]);
    proposer.kill();

    // The program that passed, in the frontier of a task, is too deep to show
    // in another task's prompt, and is left out.
    let deep = done[0]
        .iter()
        .find_map(|c| match c {
            Check::Passed { program } => Some(program.clone()),
            _ => None,
        })
        .unwrap();
    operator
        .call(
            "dc_create_run",
            json!({
                "run": "r2", "seed": 5, "proposals_per_task": 1, "frontier_size": 5,
                "tasks": ["collapse/merge", "collapse/cancel"]
            }),
        )
        .unwrap();
    let r2 = RunId::new("r2").unwrap();
    let frontiers = |program: &str| -> Vec<Event> {
        ["collapse/merge", "collapse/cancel"]
            .map(|task| Event::FrontierSet {
                task: task.to_owned(),
                members: vec![FrontierMember {
                    program: program.to_owned(),
                }],
            })
            .to_vec()
    };
    const SOLVED: &str = "Programs that solve other tasks";
    store.append(&r2, frontiers(&deep)).unwrap();
    let mut proposer = Client::start_proposer(dir.path(), "r2");
    let batch = proposer
        .call("dc_next_batch", json!({ "worker": "agent-1" }))
        .unwrap();
    let prompt = batch["batch"]["prompt"].as_str().unwrap();
    assert!(!prompt.contains(SOLVED), "{prompt}");
    // A program that can be shown is shown.
    store.append(&r2, frontiers("(lam $0)")).unwrap();
    let batch = proposer
        .call("dc_next_batch", json!({ "worker": "agent-2" }))
        .unwrap();
    let prompt = batch["batch"]["prompt"].as_str().unwrap();
    assert!(prompt.contains(SOLVED), "{prompt}");
    proposer.kill();
    operator.kill();
}

#[test]
fn the_operator_can_check_what_a_stopped_proposer_left() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start(dir.path());
    operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 1, "frontier_size": 5,
                "tasks": ["collapse/merge"]
            }),
        )
        .unwrap();
    // A proposer server took a submission and stopped before checking it.
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r1").unwrap();
    let sent = dc_mcp::proposer::next_batch(&store, &run, "a", 0, &Default::default())
        .unwrap()
        .batch
        .unwrap();
    dc_mcp::proposer::submit(
        &store,
        &run,
        0,
        "a",
        &sent.token,
        vec![solves_merge()],
        None,
        1,
    )
    .unwrap();
    let stats = operator
        .call("dc_proposal_stats", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(stats["iterations"][0]["pending_checks"], 1);
    assert_eq!(stats["iterations"][0]["quota_in"], false);

    let report = operator
        .call("dc_check_pending", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(
        report,
        json!({ "run": "r1", "checked": 1, "pending_checks": 0, "claimed_elsewhere": [] })
    );
    let stats = operator
        .call("dc_proposal_stats", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(stats["iterations"][0]["quota_in"], true);
    assert_eq!(stats["iterations"][0]["tasks"][0]["passed"], 1);
}

/// A check whose claimant stopped is taken up when a proposer waits for it,
/// though no submission comes in and no server starts.
#[test]
fn a_proposer_that_waits_for_an_abandoned_check_gets_it_done() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start(dir.path());
    operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 1, "frontier_size": 5,
                "tasks": ["collapse/merge"]
            }),
        )
        .unwrap();
    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r1").unwrap();
    let sent = dc_mcp::proposer::next_batch(&store, &run, "a", 0, &Default::default())
        .unwrap()
        .batch
        .unwrap();
    dc_mcp::proposer::submit(
        &store,
        &run,
        0,
        "a",
        &sent.token,
        vec![solves_merge()],
        None,
        1,
    )
    .unwrap();
    // Another checker claims the check, so the new server's first checks
    // leave it alone.
    let claim = store.try_claim_check(&run, 0).unwrap().unwrap();
    let mut proposer = Client::start_proposer(dir.path(), "r1");
    let wait = proposer
        .call("dc_next_batch", json!({ "worker": "b" }))
        .unwrap();
    assert!(
        wait["wait"]["reason"]
            .as_str()
            .unwrap()
            .contains("being checked"),
        "{wait}"
    );
    // Time for the server's checks so far to finish.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let pending = |operator: &mut Client| {
        operator
            .call("dc_proposal_stats", json!({ "run": "r1" }))
            .unwrap()["iterations"][0]["pending_checks"]
            .clone()
    };
    assert_eq!(pending(&mut operator), 1);
    // The operator sees which batch another checker holds.
    let report = operator
        .call("dc_check_pending", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(
        report,
        json!({ "run": "r1", "checked": 0, "pending_checks": 1, "claimed_elsewhere": [0] })
    );

    // The claimant stops without checking.
    drop(claim);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        proposer
            .call("dc_next_batch", json!({ "worker": "b" }))
            .unwrap();
        if pending(&mut operator) == 0 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the abandoned check was not taken up"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(checks(dir.path(), "r1").len(), 1);
    proposer.kill();
    operator.kill();
}

#[test]
fn a_proposer_server_needs_a_run_that_exists() {
    let dir = tempfile::TempDir::new().unwrap();
    for run in ["nosuch", "Not A Run"] {
        let out = Command::new(env!("CARGO_BIN_EXE_dc-mcp"))
            .arg("--state-dir")
            .arg(dir.path())
            .args(["--proposer", run])
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(!out.status.success(), "{run}");
        assert!(!String::from_utf8_lossy(&out.stderr).is_empty());
    }
    let out = Command::new(env!("CARGO_BIN_EXE_dc-mcp"))
        .arg("--state-dir")
        .arg(dir.path())
        .arg("--proposer")
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

/// A program that solves `task`: its good reference.
fn solves(task: &str) -> String {
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == task)
        .unwrap();
    format!(
        "{}\nlet solve = {}\n",
        v.source,
        v.references.last().unwrap().entry
    )
}

/// The whole loop through the binaries: proposers fill an iteration, the
/// operator ends it, Stitch runs in a worker process that is this binary
/// started again, and proposers then work against the new library.
#[test]
fn the_operator_ends_an_iteration_and_proposers_get_the_new_library() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start(dir.path());
    let tasks = ["collapse/merge", "collapse/cancel", "collapse/absorb"];
    let created = operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 1, "frontier_size": 2,
                "stitch_wall_seconds": 120, "stitch_memory_mib": 2048,
                "autodoc": true, "tasks": tasks
            }),
        )
        .unwrap();
    assert_eq!(created["frontier_size"], 2);
    assert_eq!(
        created["stitch"],
        json!({ "iterations": 3, "max_arity": 2, "wall_seconds": 120, "memory_mib": 2048 })
    );

    // Not before the iteration's programs are in.
    let early = operator
        .call("dc_finish_iteration", json!({ "run": "r1" }))
        .unwrap_err();
    assert!(early.contains("cannot end yet"), "{early}");

    let store = Store::open(dir.path()).unwrap();
    let run = RunId::new("r1").unwrap();
    let mut proposer = Client::start_proposer(dir.path(), "r1");
    for i in 0..tasks.len() {
        let worker = format!("agent-{i}");
        let got = proposer
            .call("dc_next_batch", json!({ "worker": worker }))
            .unwrap();
        let batch = got["batch"]["batch"].as_u64().unwrap();
        assert!(!got["batch"]["prompt"].as_str().unwrap().contains("fn_0"));
        let task = store
            .read(&run, |s| s.batch(batch).unwrap().task.clone())
            .unwrap();
        proposer
            .call(
                "dc_submit",
                json!({
                    "worker": worker, "batch": batch, "token": got["batch"]["token"],
                    "programs": [solves(&task)]
                }),
            )
            .unwrap();
    }
    wait_for_checks(&mut operator, &mut proposer, "r1");

    let done = operator
        .call("dc_finish_iteration", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(done["finished_iteration"], 0);
    assert_eq!(done["programs_used"], 3);
    // The worker process ran Stitch to the end. A worker that could not be
    // started is an error, and one that failed says "failed".
    assert_eq!(done["compression"], "ran", "{done}");
    assert_eq!(done["compression_error"], Value::Null);
    let found = done["new_abstractions"].as_u64().unwrap();
    assert!(found > 0, "{done}");
    let status = &done["status"];
    assert_eq!(status["iteration"], 1);
    assert_eq!(status["library_size"].as_u64().unwrap(), found);
    assert_eq!(status["log_counts"]["compression"], 1);
    for task in status["tasks"].as_array().unwrap() {
        assert_eq!(task["frontier_size"], 1, "{task}");
    }
    assert_eq!(status["current"]["quota_in"], false);
    assert_eq!(status["unnamed"].as_u64().unwrap(), found);

    // The same again is refused: iteration 1 has no programs yet.
    let again = operator
        .call("dc_finish_iteration", json!({ "run": "r1" }))
        .unwrap_err();
    assert!(again.contains("iteration 1 cannot end yet"), "{again}");
    assert_eq!(created["autodoc"], true);

    // No batch until each new function is named (decision D23).
    let waiting = proposer
        .call("dc_next_batch", json!({ "worker": "agent-9" }))
        .unwrap();
    assert_eq!(waiting["batch"], Value::Null);
    let reason = waiting["wait"]["reason"].as_str().unwrap();
    assert!(reason.contains("dc_next_naming_job"), "{reason}");
    for k in 0..found {
        let worker = format!("namer-{k}");
        let job = proposer
            .call("dc_next_naming_job", json!({ "worker": worker }))
            .unwrap();
        let prompt = job["job"]["prompt"].as_str().unwrap();
        assert!(prompt.contains(&format!("let fn_{k} = ")), "{prompt}");
        let named = proposer
            .call(
                "dc_submit_naming",
                json!({
                    "worker": worker, "job": job["job"]["job"], "token": job["job"]["token"],
                    "name": format!("abs_{k}"), "doc": format!("Abstraction {k}.")
                }),
            )
            .unwrap();
        assert_eq!(named["name"], format!("abs_{k}"), "{named}");
    }
    let none = proposer
        .call("dc_next_naming_job", json!({ "worker": "namer-x" }))
        .unwrap();
    assert_eq!(none["job"], Value::Null, "{none}");
    let status = operator
        .call("dc_run_status", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(status["unnamed"], 0, "{status}");

    // The proposer's server, which was running all along, hands out work
    // for iteration 1 with the new library, and checks a program that uses
    // it.
    let got = proposer
        .call("dc_next_batch", json!({ "worker": "agent-9" }))
        .unwrap();
    let prompt = got["batch"]["prompt"].as_str().unwrap();
    assert!(prompt.contains("-- Abstraction 0.\nabs_0 : "), "{prompt}");
    assert!(!prompt.contains("fn_0"), "{prompt}");
    let batch = got["batch"]["batch"].as_u64().unwrap();
    let task = store
        .read(&run, |s| s.batch(batch).unwrap().task.clone())
        .unwrap();
    proposer
        .call(
            "dc_submit",
            json!({
                "worker": "agent-9", "batch": batch, "token": got["batch"]["token"],
                "programs": [format!("{}\nlet unused = abs_0\n", solves(&task))]
            }),
        )
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let passed = loop {
        let stats = operator
            .call("dc_proposal_stats", json!({ "run": "r1" }))
            .unwrap();
        let now = &stats["iterations"][1];
        if now["pending_checks"] == 0 && now["submissions"] == 1 {
            break now["tasks_solved"].clone();
        }
        assert!(std::time::Instant::now() < deadline, "{stats}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert_eq!(passed, 1);
}

/// Makes the run `r1` of three tasks and gives each task one program that
/// solves it, through a proposer's server.
fn a_run_ready_to_end(dir: &Path, operator: &mut Client) {
    let tasks = ["collapse/merge", "collapse/cancel", "collapse/absorb"];
    operator
        .call(
            "dc_create_run",
            json!({
                "run": "r1", "seed": 5, "proposals_per_task": 1, "frontier_size": 2,
                "stitch_wall_seconds": 120, "stitch_memory_mib": 2048,
                "tasks": tasks
            }),
        )
        .unwrap();
    let store = Store::open(dir).unwrap();
    let run = RunId::new("r1").unwrap();
    let mut proposer = Client::start_proposer(dir, "r1");
    for i in 0..tasks.len() {
        let worker = format!("agent-{i}");
        let got = proposer
            .call("dc_next_batch", json!({ "worker": worker }))
            .unwrap();
        let batch = got["batch"]["batch"].as_u64().unwrap();
        let task = store
            .read(&run, |s| s.batch(batch).unwrap().task.clone())
            .unwrap();
        proposer
            .call(
                "dc_submit",
                json!({
                    "worker": worker, "batch": batch, "token": got["batch"]["token"],
                    "programs": [solves(&task)]
                }),
            )
            .unwrap();
    }
    wait_for_checks(operator, &mut proposer, "r1");
}

/// A server whose file was replaced after it started, as an upgrade does,
/// still starts its own build as the worker.
#[cfg(target_os = "linux")]
#[test]
fn a_server_whose_file_was_replaced_still_runs_its_own_worker() {
    use std::os::unix::fs::PermissionsExt;

    let bin = tempfile::TempDir::new().unwrap();
    let binary = bin.path().join("dc-mcp");
    std::fs::copy(env!("CARGO_BIN_EXE_dc-mcp"), &binary).unwrap();
    let dir = tempfile::TempDir::new().unwrap();
    let mut operator = Client::start_binary(&binary, dir.path(), &[], Stdio::null());
    a_run_ready_to_end(dir.path(), &mut operator);

    // Another program takes the path. Started as the worker, it would fail.
    let other = bin.path().join("other");
    std::fs::write(&other, "#!/bin/sh\nexit 2\n").unwrap();
    std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::rename(&other, &binary).unwrap();
    assert_eq!(Command::new(&binary).status().unwrap().code(), Some(2));

    let done = operator
        .call("dc_finish_iteration", json!({ "run": "r1" }))
        .unwrap();
    assert_eq!(done["compression"], "ran", "{done}");
    assert!(done["new_abstractions"].as_u64().unwrap() > 0, "{done}");
}

#[test]
fn the_binary_is_its_own_stitch_worker() {
    // A worker reads a request on standard input and refuses one that does
    // not parse, as `dc-stitch-worker` does.
    let mut child = Command::new(env!("CARGO_BIN_EXE_dc-mcp"))
        .arg("--stitch-worker")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"not a request")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("dc-stitch-worker"), "{stderr}");
    assert!(!stderr.contains("usage"), "{stderr}");
    assert!(output.stdout.is_empty());

    // With anything after it, the argument is not the worker's.
    let output = Command::new(env!("CARGO_BIN_EXE_dc-mcp"))
        .args(["--stitch-worker", "--state-dir"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("usage"));
}
