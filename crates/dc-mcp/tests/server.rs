//! The server binary, driven over stdio as an MCP client drives it.
//!
//! The client here is written against the MCP wire format directly, with no
//! code shared with the server, so it checks what a real client would see.

use std::io::{BufRead, BufReader, Lines, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use dc_mcp::state::{Event, FrontierMember, LibraryEntry, LogKind, LogRecord, RunId};
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
        let mut child = Command::new(env!("CARGO_BIN_EXE_dc-mcp"))
            .arg("--state-dir")
            .arg(dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
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
            "dc_create_run",
            "dc_list_runs",
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
            json!({ "run": "r1", "seed": 5, "proposals_per_task": 2 }),
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
                log(LogKind::Payload, 0, &corpus[0]),
                log(LogKind::Submission, 0, &corpus[0]),
                Event::FrontierSet {
                    task: corpus[0].clone(),
                    members: vec![member("(lam 0)"), member("(lam (lam 1))")],
                },
                Event::LibraryExtended {
                    entries: vec![LibraryEntry {
                        body: "(lam 0)".into(),
                    }],
                },
                Event::IterationFinished { iteration: 0 },
            ]
        })
        .unwrap();
    other
        .append(&run, vec![log(LogKind::Payload, 1, &corpus[1])])
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
        json!({ "payload": 2, "submission": 1, "evaluation": 0, "compression": 0, "note": 0 })
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
            json!({ "run": "r1", "seed": 5, "proposals_per_task": 2, "tasks": [task] }),
        )
        .unwrap();
    // Take the run into its second iteration, so the kill is inside one.
    Store::open(dir.path())
        .unwrap()
        .append(&run, vec![Event::IterationFinished { iteration: 0 }])
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
                json!({ "run": run, "seed": 1, "proposals_per_task": 1, "tasks": [task] }),
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
fn bad_requests_are_tool_errors_and_change_nothing() {
    let dir = tempfile::TempDir::new().unwrap();
    let mut client = Client::start(dir.path());
    let first = dc_tasks::families::development()[0]
        .family
        .name()
        .to_owned();
    let create = |run: &str, tasks: Value| json!({ "run": run, "seed": 1, "proposals_per_task": 1, "tasks": tasks });

    let error = client
        .call("dc_run_status", json!({ "run": "none" }))
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
            json!({ "run": "r1", "seed": 1, "proposals_per_task": 0 }),
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
