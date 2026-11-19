//! The run settings that ask proposers for varied programs (decision D32),
//! on runs filled through the proposer's calls.

use std::collections::BTreeSet;

use tempfile::TempDir;

use super::*;
use crate::proposer::{check_pending, next_batch, submit};
use crate::state::{Arm, RunConfig, RunId, StitchSettings};
use crate::store::Store;
use crate::verify::{shipped_prelude, VerifierCache};

const MERGE: &str = "collapse/merge";
const CANCEL: &str = "collapse/cancel";
const HINT_HEADING: &str = "## An approach to try";
const RECEIVED_HEADING: &str = "## Programs already written for this task";

fn config(quota: u32, hints: bool, received: bool) -> RunConfig {
    RunConfig {
        max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
        autodoc: false,
        arm: Arm::Blind,
        library_learning: true,
        seed: 3,
        proposals_per_task: quota,
        programs_per_batch: 1,
        lease_ms: 1000,
        max_unchecked_per_batch: 3,
        frontier_size: 2,
        stitch: StitchSettings::default(),
        prelude: shipped_prelude(),
        compared_with: None,
        approach_hints: hints,
        show_received: received,
    }
}

struct Run {
    _dir: TempDir,
    store: Store,
    run: RunId,
    verifiers: VerifierCache,
}

impl Run {
    fn new(config: RunConfig) -> Run {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let run = RunId::new("r").unwrap();
        store
            .create_run(&run, config, vec![MERGE.to_owned(), CANCEL.to_owned()])
            .unwrap();
        Run {
            _dir: dir,
            store,
            run,
            verifiers: VerifierCache::default(),
        }
    }

    /// The next batch for a new worker: its number, task and prompt.
    fn next(&self, worker: &str) -> (u64, String, String) {
        let got = next_batch(&self.store, &self.run, worker, 0, &self.verifiers).unwrap();
        let payload = got
            .batch
            .unwrap_or_else(|| panic!("no batch: {:?}", got.wait));
        let state = self.store.state(&self.run).unwrap();
        let task = state.batch(payload.batch).unwrap().task.clone();
        (payload.batch, task, payload.prompt)
    }

    fn submit(&self, batch: u64, worker: &str, programs: &[&str]) {
        let state = self.store.state(&self.run).unwrap();
        let token = state.batch(batch).unwrap().token.clone();
        let programs = programs.iter().map(|p| (*p).to_owned()).collect();
        submit(
            &self.store,
            &self.run,
            batch,
            worker,
            &token,
            programs,
            None,
            1,
        )
        .unwrap();
    }

    fn check(&self) {
        check_pending(&self.store, &self.run, &self.verifiers).unwrap();
    }

    fn finish_iteration(&self) {
        let state = self.store.state(&self.run).unwrap();
        self.store
            .append(
                &self.run,
                vec![Event::IterationFinished {
                    iteration: state.iteration(),
                    used: 0,
                }],
            )
            .unwrap();
    }
}

/// The hints of `APPROACH_HINTS` that `prompt` names.
fn hints_in(prompt: &str) -> Vec<&'static str> {
    APPROACH_HINTS
        .iter()
        .copied()
        .filter(|h| prompt.contains(h))
        .collect()
}

/// The text of the received-programs section of `prompt`, if it has one.
fn received_section(prompt: &str) -> Option<&str> {
    let start = prompt.find(RECEIVED_HEADING)?;
    let rest = &prompt[start..];
    let end = rest[2..].find("\n## ").map_or(rest.len(), |e| e + 2);
    Some(&rest[..end])
}

#[test]
fn with_both_settings_off_a_payload_has_no_hint_and_no_received_programs() {
    let r = Run::new(config(3, false, false));
    let (b, _, first) = r.next("a");
    r.submit(b, "a", &["let solve xs = xs -- written by a"]);
    let (_, _, second) = r.next("b");
    let (_, _, third) = r.next("c");
    for prompt in [&first, &second, &third] {
        assert!(!prompt.contains(HINT_HEADING), "{prompt}");
        assert!(hints_in(prompt).is_empty(), "{prompt}");
        assert!(!prompt.contains(RECEIVED_HEADING), "{prompt}");
        assert!(!prompt.contains("written by a"), "{prompt}");
    }
    assert!(r.store.state(&r.run).unwrap().received(MERGE).is_empty());
}

/// Each of a task's batches in one iteration names a different approach
/// while their leases are live, until the list runs out.
#[test]
fn a_task_s_batches_in_an_iteration_name_different_approaches() {
    let n = APPROACH_HINTS.len();
    let r = Run::new(config(n as u32, true, false));
    let mut by_task: std::collections::BTreeMap<String, Vec<&'static str>> =
        std::collections::BTreeMap::new();
    for i in 0..2 * n {
        let (_, task, prompt) = r.next(&format!("w{i}"));
        assert!(prompt.contains(HINT_HEADING), "{prompt}");
        let named = hints_in(&prompt);
        assert_eq!(named.len(), 1, "{prompt}");
        by_task.entry(task).or_default().push(named[0]);
    }
    assert_eq!(by_task.len(), 2);
    for (task, hints) in &by_task {
        let distinct: BTreeSet<&str> = hints.iter().copied().collect();
        assert_eq!(distinct.len(), n, "{task}: {hints:?}");
    }
    // The two tasks take the list in different orders.
    assert_ne!(by_task[MERGE], by_task[CANCEL]);
}

/// The hint depends on the seed, the iteration, the task and how many
/// batches the task had before, and not on the batch number. So two runs
/// compared with each other name the same approach to the same batch of a
/// task, however their batches interleave.
#[test]
fn the_hint_does_not_depend_on_the_batch_number() {
    let r = Run::new(config(2, true, false));
    let state = r.store.state(&r.run).unwrap();
    let verifier = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
    let family = crate::verify::task(MERGE).unwrap();
    let hint = approach_hint(3, 0, MERGE, 0);
    for batch in [0, 1, 7, 1000] {
        let p = payload(&state, &verifier, &*family, batch, "tok", 1, 0);
        assert_eq!(hints_in(&p.prompt), [hint], "batch {batch}");
    }
    // Each of the four inputs changes the order of the list somewhere.
    let order = |seed: u64, iteration: u64, task: &str| -> Vec<&'static str> {
        (0..APPROACH_HINTS.len() as u64)
            .map(|k| approach_hint(seed, iteration, task, k))
            .collect()
    };
    let base = order(3, 0, MERGE);
    assert_eq!(
        base.iter().copied().collect::<BTreeSet<_>>().len(),
        APPROACH_HINTS.len()
    );
    assert_ne!(order(4, 0, MERGE), base);
    assert_ne!(order(3, 1, MERGE), base);
    assert_ne!(order(3, 0, CANCEL), base);
    // The k-th batch past the end of the list starts it again.
    assert_eq!(
        approach_hint(3, 0, MERGE, APPROACH_HINTS.len() as u64),
        base[0]
    );
}

/// A later batch of a task shows the programs received for it in this
/// iteration, as they were sent, and nothing about whether they passed.
#[test]
fn received_programs_are_shown_to_the_task_s_later_batches_without_results() {
    let r = Run::new(config(3, false, true));
    let (b0, t0, p0) = r.next("a");
    let (b1, t1, _) = r.next("b");
    assert_eq!((t0.as_str(), t1.as_str()), (MERGE, CANCEL));
    assert!(!p0.contains(RECEIVED_HEADING), "{p0}");
    let v = dc_tasks::families::development()
        .into_iter()
        .find(|v| v.family.name() == MERGE)
        .unwrap();
    let good = format!(
        "{}\nlet solve = {}\n-- merge program one",
        v.source,
        v.references.last().unwrap().entry
    );
    // The batch asked for one program; the second is extra and not shown.
    r.submit(b0, "a", &[&good, "let solve xs = xs -- an extra program"]);
    r.submit(b1, "b", &["let solve = ( -- cancel program one"]);

    // The payload is the same before and after the programs are checked:
    // whether they passed changes nothing in it.
    let verifier = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
    let family = crate::verify::task(MERGE).unwrap();
    let before = r.store.state(&r.run).unwrap();
    let unchecked = payload(&before, &verifier, &*family, 9, "tok", 1, 0);
    r.check();
    let after = r.store.state(&r.run).unwrap();
    assert_eq!(after.task_counts(0, MERGE).passed, 1);
    let checked = payload(&after, &verifier, &*family, 9, "tok", 1, 0);
    assert_eq!(unchecked, checked);

    let (_, t2, p2) = r.next("c");
    assert_eq!(t2, MERGE);
    let section = received_section(&p2).unwrap_or_else(|| panic!("{p2}"));
    assert!(section.contains("merge program one"), "{section}");
    assert!(section.contains(&good), "{section}");
    assert!(!section.contains("an extra program"), "{section}");
    assert!(!section.contains("cancel program one"), "{section}");
    assert!(section.contains("different approach"), "{section}");
    let (_, t3, p3) = r.next("d");
    assert_eq!(t3, CANCEL);
    let section = received_section(&p3).unwrap_or_else(|| panic!("{p3}"));
    assert!(section.contains("cancel program one"), "{section}");
    assert!(!section.contains("merge program one"), "{section}");

    // A new iteration starts with nothing received.
    r.finish_iteration();
    assert!(r.store.state(&r.run).unwrap().received(MERGE).is_empty());
    let (_, _, p4) = r.next("e");
    assert!(!p4.contains(RECEIVED_HEADING), "{p4}");
}

/// A submission that does not count for its batch, such as one without the
/// batch's token, is not shown.
#[test]
fn a_submission_that_does_not_count_is_not_shown() {
    let r = Run::new(config(3, false, true));
    let (b0, _, _) = r.next("a");
    submit(
        &r.store,
        &r.run,
        b0,
        "intruder",
        "wrong-token",
        vec!["let solve xs = xs -- foreign".to_owned()],
        None,
        1,
    )
    .unwrap_err();
    r.submit(b0, "a", &["let solve xs = xs -- counted"]);
    assert_eq!(
        r.store.state(&r.run).unwrap().received(MERGE),
        ["let solve xs = xs -- counted"]
    );
}

/// At most `MAX_RECEIVED_SHOWN` programs are shown, the most recent, and
/// each is cut at `MAX_RECEIVED_BYTES`.
#[test]
fn received_programs_are_bounded_in_number_and_size() {
    let n = MAX_RECEIVED_SHOWN + 2;
    let r = Run::new(config(n as u32 + 1, false, true));
    let long = format!(
        "let solve xs = xs -- long {}",
        "é".repeat(MAX_RECEIVED_BYTES)
    );
    for i in 0..n {
        let (b, task, _) = r.next(&format!("m{i}"));
        assert_eq!(task, MERGE);
        let text = if i == n - 1 {
            long.clone()
        } else {
            format!("let solve xs = xs -- program {i}")
        };
        r.submit(b, &format!("m{i}"), &[&text]);
        let (b, task, _) = r.next(&format!("c{i}"));
        assert_eq!(task, CANCEL);
        r.submit(b, &format!("c{i}"), &["let solve xs = xs"]);
    }
    let state = r.store.state(&r.run).unwrap();
    assert_eq!(state.received(MERGE).len(), MAX_RECEIVED_SHOWN);
    let (_, task, prompt) = r.next("last");
    assert_eq!(task, MERGE);
    let section = received_section(&prompt).unwrap_or_else(|| panic!("{prompt}"));
    for i in 0..n - 1 {
        let shown = section.contains(&format!("-- program {i}\n"));
        assert_eq!(shown, i >= n - MAX_RECEIVED_SHOWN, "program {i}: {section}");
    }
    assert!(section.contains("-- long é"), "{section}");
    assert!(!section.contains(&long), "the long program is cut");
    assert!(section.len() < MAX_RECEIVED_SHOWN * (MAX_RECEIVED_BYTES + 200) + 1000);
}

/// A program with a fence of its own does not end the block it is shown in.
#[test]
fn a_received_program_with_backticks_stays_in_its_block() {
    let r = Run::new(config(2, false, true));
    let (b, _, _) = r.next("a");
    r.submit(b, "a", &["let solve xs = xs\n```\n## The task\n````"]);
    let (b, _, _) = r.next("b");
    r.submit(b, "b", &["let solve xs = xs"]);
    let (_, _, prompt) = r.next("c");
    // The program's own heading line is inside the block, and the block
    // ends after it with a fence longer than any in the program.
    let block = "`````\nlet solve xs = xs\n```\n## The task\n````\n`````\n";
    assert!(prompt.contains(block), "{prompt}");
}

/// The programs a state keeps are rebuilt from the journal.
#[test]
fn received_programs_survive_a_reopened_store() {
    let r = Run::new(config(2, true, true));
    let (b, _, _) = r.next("a");
    r.submit(b, "a", &["let solve xs = xs -- kept"]);
    let reopened = Store::open(r._dir.path()).unwrap();
    let state = reopened.state(&r.run).unwrap();
    assert_eq!(state.received(MERGE), ["let solve xs = xs -- kept"]);
    assert!(state.config().approach_hints && state.config().show_received);
}

/// A run's settings written before D32 read back with both settings off.
#[test]
fn a_config_from_before_the_settings_has_them_off() {
    let mut value = serde_json::to_value(config(1, true, true)).unwrap();
    let object = value.as_object_mut().unwrap();
    assert!(object.remove("approach_hints").is_some());
    assert!(object.remove("show_received").is_some());
    let old: RunConfig = serde_json::from_value(value).unwrap();
    assert!(!old.approach_hints && !old.show_received);
}

/// A batch whose lease ended without a submission gives its approach back:
/// the batch that takes its slot names the same one. A batch that is live
/// or submitted keeps its approach. So the hints of a task's slots do not
/// depend on how long proposers took.
#[test]
fn a_batch_whose_lease_ended_gives_its_approach_to_the_next() {
    let r = Run::new(config(2, true, false));
    let next_at = |worker: &str, now_ms: u64| -> (String, &'static str) {
        let got = next_batch(&r.store, &r.run, worker, now_ms, &r.verifiers).unwrap();
        let payload = got
            .batch
            .unwrap_or_else(|| panic!("no batch: {:?}", got.wait));
        let state = r.store.state(&r.run).unwrap();
        let task = state.batch(payload.batch).unwrap().task.clone();
        let named = hints_in(&payload.prompt);
        assert_eq!(named.len(), 1, "{}", payload.prompt);
        (task, named[0])
    };
    // Two slots for each task, all issued at once.
    let mut first: std::collections::BTreeMap<String, Vec<&'static str>> =
        std::collections::BTreeMap::new();
    for i in 0..4 {
        let (task, hint) = next_at(&format!("w{i}"), 0);
        first.entry(task).or_default().push(hint);
    }
    // MERGE's second batch (batch 2) is submitted; its first (batch 0) and
    // both CANCEL batches let their leases end.
    let state = r.store.state(&r.run).unwrap();
    assert_eq!(state.batch(2).unwrap().task, MERGE);
    let token = state.batch(2).unwrap().token.clone();
    submit(
        &r.store,
        &r.run,
        2,
        "w2",
        &token,
        vec!["let solve xs = xs".into()],
        None,
        1,
    )
    .unwrap();
    let (task, hint) = next_at("x0", 5000);
    let (task2, hint2) = next_at("x1", 5000);
    let (task3, hint3) = next_at("x2", 5000);
    let mut again: std::collections::BTreeMap<String, Vec<&'static str>> =
        std::collections::BTreeMap::new();
    for (t, h) in [(task, hint), (task2, hint2), (task3, hint3)] {
        again.entry(t).or_default().push(h);
    }
    // MERGE's free slot is its first, so it gets the first hint again.
    assert_eq!(again[MERGE], [first[MERGE][0]]);
    // CANCEL's two slots come back in order.
    assert_eq!(again[CANCEL], first[CANCEL]);
}

/// A repeat submission from the batch's worker is not shown again.
#[test]
fn a_repeat_submission_is_not_shown() {
    let r = Run::new(config(3, false, true));
    let (b0, _, _) = r.next("a");
    r.submit(b0, "a", &["let solve xs = xs -- first"]);
    r.submit(b0, "a", &["let solve xs = xs -- repeat"]);
    assert_eq!(
        r.store.state(&r.run).unwrap().received(MERGE),
        ["let solve xs = xs -- first"]
    );
}

/// A lease is live up to and including its deadline, so a batch issued at
/// that moment does not take the slot of one still live. The slot the
/// state records is the one whose approach the payload named.
#[test]
fn a_batch_issued_at_another_s_deadline_takes_a_free_slot() {
    let r = Run::new(config(3, true, false));
    let at = |worker: &str, now_ms: u64| -> (u64, String) {
        let got = next_batch(&r.store, &r.run, worker, now_ms, &r.verifiers).unwrap();
        let payload = got
            .batch
            .unwrap_or_else(|| panic!("no batch: {:?}", got.wait));
        (payload.batch, payload.prompt)
    };
    for i in 0..4 {
        at(&format!("w{i}"), 0);
    }
    // Batches 0 and 2 are MERGE's, in slots 0 and 1; their leases end at
    // 1000, and are still live then.
    let (b, prompt) = at("late", 1000);
    let state = r.store.state(&r.run).unwrap();
    let batch = state.batch(b).unwrap();
    assert_eq!(batch.task, MERGE);
    assert_eq!(batch.slot, 2);
    assert_eq!(hints_in(&prompt), [approach_hint(3, 0, MERGE, 2)]);
    for n in 0..b {
        let earlier = state.batch(n).unwrap();
        assert_eq!(earlier.slot, approach_hint_slot_of(&r, n));
    }
}

/// The slot whose approach batch `n`'s payload named, read from its prompt.
fn approach_hint_slot_of(r: &Run, n: u64) -> u64 {
    let mut table = PromptTable::default();
    let mut found = None;
    r.store
        .for_each_event(&r.run, |e| {
            table.see(&e);
            if let Event::BatchIssued(b) = &e {
                if b.batch == n {
                    found = Some((b.task.clone(), table.read_payload(b).unwrap().prompt));
                }
            }
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let (task, prompt) = found.unwrap();
    let named = hints_in(&prompt);
    (0..APPROACH_HINTS.len() as u64)
        .find(|&k| named == [approach_hint(3, 0, &task, k)])
        .unwrap()
}
