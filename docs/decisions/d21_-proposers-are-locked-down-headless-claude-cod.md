---
id: doc-amd
title: 'D21: Proposers are locked-down headless Claude Code sessions that dc-propose
  starts, one batch each'
type: decision
scope: internal
created_at: '2026-11-15T19:37:53.144373Z'
updated_at: '2026-11-15T20:05:10.574330Z'
---

- **Date:** 2026-11-15. **Status:** accepted by Claude as technical design (autonomous lane). Open to change by the project lead. Bead dreamcounter-50d667 (P4.3). Follows D11 (doc-d9s) and D20 (doc-hw6). The brief's wording is a research choice the project lead may want to change.
- **Decision:**
  1. **A launcher, not a Workflow script.** `dc-propose` (crate `crates/dc-propose`) starts each proposer as a headless Claude Code process (`claude -p`). D11 asked for "a workflow in a session". A Claude Code Workflow agent runs in the calling session's directory, with its tools, rules and context, so it cannot meet D11 requirement 1.
  2. **One batch for each session.** The brief tells a session to call `dc_next_batch` once, write the programs, and call `dc_submit` once. With `programs_per_batch` at 1, each program comes from its own context (D11 requirement 3). Each session gets a worker id of its own: the launch id and its number.
  3. **The lockdown.** Each session runs with these flags:
     - `--restricted`: no user, project or local settings, so no rules, hooks or installed plugins;
     - `--tools ""`: no built-in tools;
     - `--strict-mcp-config` with one server, `dc-mcp --state-dir <dir> --proposer <run>`;
     - `--allowedTools` for its two tools only;
     - `--system-prompt <brief>`, which replaces Claude Code's default system prompt;
     - `--no-session-persistence`;
     - `--model <id>`.
  4. **Where a session runs.** It runs in a new, empty directory made by `mktemp` (`dc-proposer-*` in `TMPDIR`). The launcher refuses a directory inside the state directory, the log directory, or any git repository. A session gets only these environment variables: `HOME`, `PATH`, `USER`, `LOGNAME`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TERM`, `TMPDIR`, and any that `--pass-env` names.
  5. **Checks on each session.** The launcher reads the session's stream-json transcript as it comes. The first event must be `init`. It must list exactly the two tools, exactly the one server (connected), only built-in plugins, the session's own directory, and the model asked for. The first request must have no more input tokens than `--max-context-tokens` (4,000 by default; a locked-down session's first request has about 1,800 to 2,300). If a check fails, or the session leaves anything in its directory, the launcher kills the session's process group and starts no more sessions.
  6. **Stopping.** The launcher starts sessions until it has started `--sessions` of them, or until one finds no batch. Worker ids are unique, so "no batch" means every open slot is leased or waits for its check. The operator starts the next launch. A session is killed after `--session-timeout` (1200 s by default). Programs the server took still count. SIGINT, SIGTERM or SIGHUP kills every running session and its processes, and the launcher still writes the summary.
  7. **Records.** The log directory holds each session's command line and environment names (`<worker>.cmd.json`), MCP configuration, stream-json transcript and standard error, and `summary.json`. For each session, the summary gives the outcome, the batch it submitted, every batch it leased (`leased_batches`, whatever the outcome, since their slots stay leased), the model the session reported, its tool calls, and the usage, cost and turns from its `result` event (D11 requirement 4).
  8. **The project lead starts every run** (D11 requirement 5). The launcher neither makes runs nor finishes iterations.
- **Alternatives:**
  - (a) Claude Code Workflow agents: they inherit the caller's directory, tools and context (see item 1).
  - (b) `--bare`: it also skips CLAUDE.md files and hooks, but it signs in only with an API key, which costs money (Q5).
  - (c) A brief that asks the session not to read files, with the default tools: it rests on the model obeying.
  - (d) A long-lived proposer that takes many batches: its samples would share a context (D11 requirement 3).
- **Contamination check (2026-11-15, Claude Code 2.1.318, `claude-haiku-5-5`).** An independent reviewer ran 6 real sessions: 3 through `dc-propose` and 3 probes with the recorded flags and hostile prompts. One probe ran in a planted git repository with CLAUDE.md, AGENTS.md, rules, skills, agents, a hook and `.mcp.json`, each holding a marker string.
  - **Reach.** No session reached the repository, the state directory, the docs (the hypothesis, D14), the task family code, the held-out family, the home directory or the network. Every session had exactly the two tools. No marker string reached a session, and the hook did not run.
  - **Context that loads:**
    - a. Claude Code's one-line prefix ("You are a Claude agent, built on Anthropic's Claude Agent SDK."), then the brief;
    - b. the two tool schemas;
    - c. a `# userEmail` block with the account's email address, from the sign-in;
    - d. an environment block: the session's directory, whether it is a git repository, platform, shell, OS version, model name and id, knowledge cutoff, token budget, and the date;
    - e. the prompt "Begin.", then the batch.
  - **Context that does not load:** user rules, CLAUDE.md or AGENTS.md files, the skills listing, agents, hooks, memory, git status.
  - **Evidence.** The tool list in `init` is the strong evidence. The sessions' own recitations and the token counts support the context list. The check did not capture the raw request.
  - **Changes after the check.** The environment allowlist (item 4), the token cap (item 5), and the git-repository rule (item 4) were added. Before the allowlist, a launch from inside a Claude Code session passed that session's variables on, and the environment block changed with them.
- **Limits:**
  - **The account email loads into every session.** No flag removes it short of `--bare`. It is not research contamination, but transcripts that hold it should not be shared.
  - **The lockdown depends on Claude Code's flags.** A later version could load context the checks do not see. The token cap leaves about 2,000 tokens of room, so it catches large additions only. Record the Claude Code version with each run.
  - **SIGKILL to the launcher leaves its sessions running.** They keep their directories, and no summary is written. Setting a parent-death signal on each session would need `pre_exec`, which is `unsafe`, and the workspace forbids `unsafe`. Stop the launcher with SIGINT or SIGTERM.
  - **A process that leaves its session's process group** (`setsid`) outlives the kill. Claude Code 2.1.318 keeps `dc-mcp` in the group.
  - **The payload shows how many batches came before.** Batch numbers are run-wide, and `dc_submit` names a batch that was never issued. Bead dreamcounter-cea994.
  - **Sampling settings are Claude Code's.** Temperature cannot be set (D11).
  - **A lease held by a session that failed** keeps its slots for `lease_seconds` (D20), and the next launch finds no batch until then.
- **Use** (run locally, from the repository; `--model` must be the full id the session reports):
  - `cargo build --release -p dc-propose -p dc-mcp`
  - `target/release/dc-propose --state-dir <state> --run <run> --model claude-opus-5-5 --log-dir <logs> --sessions 20 --parallel 4`
