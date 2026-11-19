//! The payload handed to a proposer with each batch (decision D11).
//!
//! The payload is all a proposer sees (D11 requirement 1), so it carries
//! everything needed to write a program: how to answer, a short guide to the
//! dc language, the type of every primitive and prelude function, the
//! library, the task, its examples, and a few programs that solve other tasks
//! of the run. The types are taken from the code, not written by hand, so
//! they cannot drift from what the type checker accepts.
//!
//! The payload does not show the task's name, any cost, or any frontier
//! size, and it gives no result of any submission. The programs that solve
//! other tasks are frontier members, so they come from earlier submissions
//! that passed: that is how LILO shows solved tasks.
//!
//! Two run settings add to the payload (decision D32). With
//! `approach_hints` on, it names an approach to try, drawn by the seed from
//! [`APPROACH_HINTS`]. With `show_received` on, it shows the programs
//! already received for its task in this iteration, passed or not, which
//! can include the proposer's own.

use dc_core::{prim::Prim, sexpr, types::prim_scheme, Arena, LibId};
use dc_surface::pretty;
use dc_tasks::task::TaskFamily;
use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::state::{BatchIssued, Event, IssuedPayload, PromptPart, RunState};
use crate::verify::{entry_name, library_name, Verifier, ENTRY};

/// The most programs of other tasks a payload shows.
pub const MAX_SOLVED_EXAMPLES: usize = 3;

/// How many values of one example input or output the payload shows.
const SHOWN_VALUES: u64 = 2000;

/// What `dc_next_batch` returns when it hands out a batch.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    /// The batch's number. Pass it to `dc_submit`.
    pub batch: u64,
    /// The batch's token. Pass it to `dc_submit`.
    pub token: String,
    /// How many programs to write.
    pub programs_wanted: u32,
    /// How long the batch is held for you, in seconds. After that it may be
    /// given to someone else, but a submission is still accepted.
    pub lease_seconds: u64,
    /// The task, the language, and how to answer.
    pub prompt: String,
}

/// The language guide. The types of primitives and prelude functions follow
/// it in the prompt. The tests in `prompt_guide_tests.rs` check its claims
/// against the parser and type checker by reading them out of this text.
pub(crate) const GUIDE: &str = "\
## The dc language

dc is a small, strict, purely functional language with OCaml-like syntax.
Comments run from `--` to the end of the line.

A program is a sequence of top-level definitions `let name params = expr`.
A definition can use the definitions before it. There is no recursive
`let`: recursion goes only through the combinators `foldl`, `foldr`,
`loop`, `unfold`, `bfix` and `memorec` described below.

Expressions:
  let x = e1 in e2            let f a b = e1 in e2      let (a, b) = e1 in e2
  fun x y -> e                fun (a, b) -> e           if c then e1 else e2
  match xs with [] -> e1 | h :: t -> e2
  match o with None -> e1 | Some x -> e2
  match t with Leaf -> e1 | Node (l, x, r) -> e2
  f x y                       (a, b, c)                 [1; 2; 3]
  x :: xs   + - * / mod   = (on Int)  < <= > >=   == (structural)   && || not
Patterns are one level deep: write a nested match for a nested pattern.
`( + )` and the other operators can be passed as functions, except
`( > )`, `( >= )`, `( && )` and `( || )`.

Types: Int (64-bit; overflow is an error), Bool, Unit, pairs (a, b), List a,
Option a, Array a (persistent, indexed from 0), Tree a, and functions. Types
are inferred. A `Key` type is Int, Bool, Unit, or a tuple of those.

Recursion:
  foldl f z xs, foldr f z xs   fold over a list.
  loop n z f                   f (n-1) (... (f 0 z)); z when n <= 0.
  unfold f s                   the list of values f produces until it
                               returns None.
  bfix n f x                   f self x, where self is bfix (n-1) f. The
                               depth of recursion is at most n; deeper is an
                               error. Use for example `bfix (length xs + 1)`.
  memorec n f x                as bfix, with results cached by argument (a
                               Key type). n bounds the number of distinct
                               arguments computed.
";

/// The instructions at the top of every prompt.
fn instructions(wanted: u32, batch: u64, token: &str) -> String {
    let programs = if wanted == 1 {
        "one program in the dc language that solves".to_owned()
    } else {
        format!("{wanted} different programs in the dc language that solve")
    };
    format!(
        "Write {programs} the task below.\n\
         Each program is the full text of a dc program: one or more `let` definitions, \
         one of which is `{ENTRY}`. `{ENTRY}` takes the task's inputs as arguments, in \
         order, and returns the output.\n\
         Submit with the dc_submit tool: batch {batch}, token {token}, and the programs as \
         a list of strings, one string for each program. You will get an acknowledgement, not \
         results: no program is run for you.\n"
    )
}

/// One line for each primitive: `name : type`.
pub(crate) fn primitives() -> String {
    let mut out = String::from(
        "## Primitives\n\nEach can be used by its name. The operators are add (+), sub (-), \
         mul (*), div (/), mod, int_eq (=), lt (<), le (<=), eq (==) and cons (::).\n\n",
    );
    for &p in Prim::ALL {
        out.push_str(&format!("{} : {}\n", p.name(), prim_scheme(p)));
    }
    out
}

/// One line for each prelude function: `name : type`.
pub(crate) fn prelude(verifier: &Verifier) -> String {
    let mut out = String::from("## Prelude functions\n\n");
    for e in verifier.prelude().entries() {
        out.push_str(&format!("{} : {}\n", e.name(), e.scheme()));
    }
    out
}

/// The names programs use for library references: prelude names and `fn_i`.
pub(crate) fn lib_name(verifier: &Verifier) -> impl Fn(LibId) -> String + Sync + '_ {
    move |id| match verifier.prelude().name(id) {
        Some(name) => name.to_owned(),
        None => {
            let i = id.0 as usize;
            verifier
                .library()
                .get(i)
                .map_or_else(|| library_name(i), |e| entry_name(i, e))
        }
    }
}

/// `text`, a closed core term, as surface syntax, or `None` if it cannot be
/// printed.
pub(crate) fn surface(verifier: &Verifier, text: &str) -> Option<String> {
    let mut arena = Arena::new();
    let term = sexpr::parse(&mut arena, text).ok()?;
    pretty(&arena, term, lib_name(verifier)).ok()
}

/// The library section: each definition's name, type and body.
fn library(state: &RunState, verifier: &Verifier) -> String {
    if state.library().is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Library\n\nThese functions are in scope. A use of one means exactly its \
         definition written out.\n\n",
    );
    for (i, (entry, scheme)) in state.library().iter().zip(verifier.schemes()).enumerate() {
        let name = entry_name(i, entry);
        if let Some(doc) = &entry.doc {
            out.push_str(&format!("-- {doc}\n"));
        }
        out.push_str(&format!("{name} : {scheme}\n"));
        match surface(verifier, &entry.term) {
            Some(body) => out.push_str(&format!("let {name} = {body}\n\n")),
            None => out.push_str("(its definition is too large to show)\n\n"),
        }
    }
    out
}

/// A value as the payload shows it as an argument: in brackets if it has a
/// space or starts with a minus sign, so it reads as one argument.
fn shown(v: &dc_eval::Value) -> String {
    let s = v.display_within(SHOWN_VALUES).to_string();
    let bare = !(s.starts_with('(') || s.starts_with('['));
    if bare && (s.contains(' ') || s.starts_with('-')) {
        format!("({s})")
    } else {
        s
    }
}

/// The task section: its description, type and examples. A task's gate
/// (decision D25) is not shown: its answers would let a program pass it
/// without computing them. The description states its size and fuel.
fn task(family: &dyn TaskFamily) -> String {
    let mut out = format!(
        "## The task\n\n{}\n\n{ENTRY} : {}\n\nExamples:\n",
        family.description(),
        family.signature()
    );
    for example in family.examples() {
        let args: Vec<String> = example.inputs.iter().map(shown).collect();
        out.push_str(&format!(
            "  {ENTRY} {} = {}\n",
            args.join(" "),
            example.output.display_within(SHOWN_VALUES)
        ));
    }
    out
}

/// A step of the SplitMix64 generator: a well-mixed 64-bit value from `x`.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Programs that solve other tasks of the run: the first member of each
/// other task's frontier, up to [`MAX_SOLVED_EXAMPLES`] of them, chosen by
/// the run's seed and the batch number.
fn solved(state: &RunState, verifier: &Verifier, this: &str, batch: u64) -> String {
    let mut pool: Vec<(&str, &str)> = state
        .tasks()
        .iter()
        .filter(|t| t.as_str() != this)
        .filter_map(|t| {
            let first = state.frontier(t)?.first()?;
            Some((t.as_str(), first.program.as_str()))
        })
        .collect();
    let mut r = mix(state.config().seed ^ mix(batch));
    // A partial Fisher-Yates shuffle picks the first few.
    let take = pool.len().min(MAX_SOLVED_EXAMPLES);
    for i in 0..take {
        r = mix(r);
        let j = i + (r % (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let mut out = String::new();
    for (name, program) in pool.into_iter().take(take) {
        let (Some(family), Some(text)) = (crate::verify::task(name), surface(verifier, program))
        else {
            continue;
        };
        if out.is_empty() {
            out.push_str("## Programs that solve other tasks\n\n");
        }
        out.push_str(&format!(
            "Task: {}\n{ENTRY} : {}\nlet {ENTRY} = {text}\n\n",
            family.description(),
            family.signature()
        ));
    }
    out
}

/// The approaches a payload can name when its run has
/// [`RunConfig::approach_hints`](crate::state::RunConfig::approach_hints) on
/// (decision D32). Each is a technique the dc language supports, and none
/// depends on the task.
pub const APPROACH_HINTS: [&str; 9] = [
    "one pass with foldl that carries what it needs in an accumulator (a tuple if it \
     needs several values)",
    "foldr, building the answer from the end of the list back to the start",
    "for each element, a second fold over the list (a nested loop)",
    "put the elements in order first (write the sort yourself, for example a merge sort \
     with bfix), then work on the ordered list",
    "an Array: build one with array_of_list or array_tabulate, then read it by index \
     inside loop",
    "a Tree: insert the elements into a binary search tree, then walk it with tree_fold \
     or tree_case",
    "memorec: a recursive function of a key whose results are cached",
    "divide and conquer with bfix: split the input into two halves, solve each half, \
     and combine the results",
    "loop or unfold over an index or a counter instead of over the list itself",
];

/// The most received programs a payload shows (decision D32).
pub const MAX_RECEIVED_SHOWN: usize = 4;

/// The most bytes of one received program a payload shows.
pub const MAX_RECEIVED_BYTES: usize = 4000;

/// The approach named to the `k`-th batch (from 0) that `task` is given in
/// `iteration` of a run with `seed`.
///
/// Each task and iteration has its own order of [`APPROACH_HINTS`], a
/// shuffle drawn from the seed, the iteration and the task's name, and the
/// `k`-th batch takes item `k` of it, starting again past the end. The batch
/// number plays no part, so two runs compared with each other name the same
/// approach to a task's `k`-th batch however their batches interleave.
pub fn approach_hint(seed: u64, iteration: u64, task: &str, k: u64) -> &'static str {
    // FNV-1a: a hash of the name that is the same on every platform.
    let name = task.bytes().fold(0xCBF2_9CE4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    });
    let mut r = mix(seed ^ mix(iteration ^ mix(name)));
    let mut order: Vec<usize> = (0..APPROACH_HINTS.len()).collect();
    // A Fisher-Yates shuffle.
    for i in (1..order.len()).rev() {
        r = mix(r);
        let j = (r % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let at = (k % APPROACH_HINTS.len() as u64) as usize;
    APPROACH_HINTS[order[at]]
}

/// The section that names an approach to try, for a batch of `task` in a
/// run with [`RunConfig::approach_hints`](crate::state::RunConfig::approach_hints)
/// on; empty with it off. The batch is the task's first in the iteration
/// that has not been issued yet.
fn hint(state: &RunState, task: &str) -> String {
    if !state.config().approach_hints {
        return String::new();
    }
    let iteration = state.iteration();
    let k = state.task_counts(iteration, task).batches;
    format!(
        "## An approach to try\n\nTry to solve the task with this approach: {}. If it does \
         not fit the task, use another approach that does.\n",
        approach_hint(state.config().seed, iteration, task, k)
    )
}

/// `text` cut to at most [`MAX_RECEIVED_BYTES`] bytes, at a character
/// boundary, with a note if it was cut.
fn cut(text: &str) -> String {
    if text.len() <= MAX_RECEIVED_BYTES {
        return text.to_owned();
    }
    let mut end = MAX_RECEIVED_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n-- (the rest of this program is not shown)",
        &text[..end]
    )
}

/// The section that shows the programs received for `task` in this
/// iteration, without results, for a run with
/// [`RunConfig::show_received`](crate::state::RunConfig::show_received) on;
/// empty with it off or with none received. Each program is in a fenced
/// block whose fence is longer than any run of backticks in it.
fn received(state: &RunState, task: &str) -> String {
    let programs = state.received(task);
    if !state.config().show_received || programs.is_empty() {
        return String::new();
    }
    let mut out = String::from(
        "## Programs already written for this task\n\nThese programs were sent for this \
         task in this round, before your batch. They are shown as they were sent, with no \
         results: any of them may be wrong. Write programs that take a different approach \
         from each of them.\n\n",
    );
    for program in programs {
        let text = cut(program);
        let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
        let fence = "`".repeat(longest.max(2) + 1);
        out.push_str(&format!("{fence}\n{text}\n{fence}\n\n"));
    }
    out
}

/// A piece of a prompt as it is built. A shared piece is the same for many
/// batches (the guide, the types, the library, a task), so the run stores it
/// once and payloads name it; the others are written out in each batch's
/// event (dreamcounter-20c6e5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    /// The text.
    pub text: String,
    /// Whether the piece is stored once in the run.
    pub shared: bool,
}

/// The payload of `batch`, a batch for `family` that asks for `wanted`
/// programs, under the state's library.
pub fn payload(
    state: &RunState,
    verifier: &Verifier,
    family: &dyn TaskFamily,
    batch: u64,
    token: &str,
    wanted: u32,
) -> Payload {
    payload_in_pieces(state, verifier, family, batch, token, wanted).0
}

/// [`payload`], with the pieces its prompt is made of, in order: the prompt
/// is their texts joined with nothing between them.
pub fn payload_in_pieces(
    state: &RunState,
    verifier: &Verifier,
    family: &dyn TaskFamily,
    batch: u64,
    token: &str,
    wanted: u32,
) -> (Payload, Vec<Piece>) {
    let sections = [
        (instructions(wanted, batch, token), false),
        (GUIDE.to_owned(), true),
        (primitives(), true),
        (prelude(verifier), true),
        (library(state, verifier), true),
        (solved(state, verifier, family.name(), batch), false),
        (task(family), true),
        (hint(state, family.name()), true),
        (received(state, family.name()), false),
    ];
    let mut pieces = Vec::new();
    for (i, (text, shared)) in sections.into_iter().enumerate() {
        if text.is_empty() {
            continue;
        }
        if !pieces.is_empty() {
            pieces.push(Piece {
                text: "\n".to_owned(),
                shared: false,
            });
        }
        if i == 0 {
            pieces.extend(split_instructions(text, batch, token));
        } else {
            pieces.push(Piece { text, shared });
        }
    }
    let payload = Payload {
        batch,
        token: token.to_owned(),
        programs_wanted: wanted,
        lease_seconds: state.config().lease_ms / 1000,
        prompt: pieces.iter().map(|p| p.text.as_str()).collect(),
    };
    (payload, pieces)
}

/// The instructions in pieces: the batch number and token, which change
/// with each batch, and the text around them, which is shared.
fn split_instructions(text: String, batch: u64, token: &str) -> Vec<Piece> {
    let own = format!("batch {batch}, token {token},");
    let Some(at) = text.find(&own) else {
        return vec![Piece {
            text,
            shared: false,
        }];
    };
    let (before, rest) = text.split_at(at);
    let after = &rest[own.len()..];
    [(before, true), (own.as_str(), false), (after, true)]
        .into_iter()
        .filter(|(t, _)| !t.is_empty())
        .map(|(t, shared)| Piece {
            text: t.to_owned(),
            shared,
        })
        .collect()
}

/// Why a payload does not read back from its event.
#[derive(Debug)]
pub enum PayloadError {
    /// The event names a prompt that the events seen so far did not store.
    UnknownPrompt(u64),
    /// The payload's JSON text does not read or write.
    Json(serde_json::Error),
}

impl std::fmt::Display for PayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PayloadError::UnknownPrompt(id) => write!(f, "no prompt {id} is stored"),
            PayloadError::Json(e) => write!(f, "the payload's JSON: {e}"),
        }
    }
}

impl std::error::Error for PayloadError {}

/// The prompts a run has stored, read from its events, so that the payload
/// of each [`BatchIssued`] event reads back as it was sent.
///
/// Give it the run's events in order with [`PromptTable::see`]; a payload
/// reads back once the events before it have been seen. It keeps one copy
/// of each stored prompt's text, however many payloads name it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromptTable {
    texts: Vec<String>,
}

impl PromptTable {
    /// Takes note of `event` if it stores a prompt. A prompt whose id is not
    /// the next one is left out, so payloads that name it do not read back.
    /// The run's state refuses such an event, so a run the store reads has
    /// none.
    pub fn see(&mut self, event: &Event) {
        if let Event::PromptStored(stored) = event {
            if stored.id == self.texts.len() as u64 {
                self.texts.push(stored.text.clone());
            }
        }
    }

    /// The payload of `issued`, read back.
    pub fn read_payload(&self, issued: &BatchIssued) -> Result<Payload, PayloadError> {
        let parts = match &issued.payload {
            IssuedPayload::Whole(text) => {
                return serde_json::from_str(text).map_err(PayloadError::Json)
            }
            IssuedPayload::Parts(parts) => parts,
        };
        let mut prompt = String::new();
        for part in &parts.prompt {
            match part {
                PromptPart::Text(text) => prompt.push_str(text),
                PromptPart::Stored(id) => prompt.push_str(
                    usize::try_from(*id)
                        .ok()
                        .and_then(|i| self.texts.get(i))
                        .ok_or(PayloadError::UnknownPrompt(*id))?,
                ),
            }
        }
        Ok(Payload {
            batch: issued.batch,
            token: issued.token.clone(),
            programs_wanted: issued.wanted,
            lease_seconds: parts.lease_seconds,
            prompt,
        })
    }

    /// The JSON text of the payload of `issued`, exactly as it was sent.
    pub fn payload_text(&self, issued: &BatchIssued) -> Result<String, PayloadError> {
        match &issued.payload {
            IssuedPayload::Whole(text) => Ok(text.clone()),
            IssuedPayload::Parts(_) => {
                serde_json::to_string(&self.read_payload(issued)?).map_err(PayloadError::Json)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Arm, Event, FrontierMember, LibraryEntry, RunConfig};

    #[test]
    fn an_argument_is_bracketed_when_it_would_not_read_as_one() {
        use dc_eval::Value;
        assert_eq!(shown(&Value::Int(-3)), "(-3)");
        assert_eq!(shown(&Value::Int(3)), "3");
        assert_eq!(shown(&Value::Bool(true)), "true");
    }

    fn state(tasks: &[&str]) -> RunState {
        RunState::create(
            RunConfig {
                max_memory: dc_tasks::harness::DEFAULT_MAX_MEMORY,
                autodoc: false,
                arm: Arm::Blind,
                library_learning: true,
                seed: 11,
                proposals_per_task: 2,
                programs_per_batch: 1,
                lease_ms: 90_000,
                max_unchecked_per_batch: 3,
                frontier_size: 5,
                stitch: crate::state::StitchSettings::default(),
                prelude: crate::verify::shipped_prelude(),
                compared_with: None,
                approach_hints: false,
                show_received: false,
            },
            tasks.iter().map(|t| (*t).to_owned()).collect(),
        )
        .unwrap()
    }

    #[test]
    fn a_payload_has_the_task_its_examples_and_the_language() {
        let s = state(&["collapse/merge", "collapse/cancel"]);
        let v = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 4, "tok", 1);
        assert_eq!((p.batch, p.programs_wanted, p.lease_seconds), (4, 1, 90));
        assert!(p.prompt.contains(family.description()));
        assert!(
            p.prompt.contains("solve : List Int -> List Int"),
            "{}",
            p.prompt
        );
        assert!(p.prompt.contains("batch 4"));
        assert!(p.prompt.contains("bfix : "));
        assert!(p.prompt.contains("map : "));
        let first = &family.examples()[0];
        assert!(p.prompt.contains(&format!("= {}", first.output)));
        // The task's name and the other tasks' names are not shown.
        assert!(!p.prompt.contains("collapse/"), "{}", p.prompt);
        // No library and no solved tasks yet.
        assert!(!p.prompt.contains("## Library"));
        assert!(!p.prompt.contains("## Programs that solve"));
    }

    #[test]
    fn a_gate_is_not_shown() {
        let s = state(&["dominance/increasing-triples"]);
        let v = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task("dominance/increasing-triples").unwrap();
        let p = payload(&s, &v, &*family, 0, "tok", 1);
        // Every small example is shown, and nothing else in its form.
        let examples = family.examples();
        for e in &examples {
            let line = format!(
                "  {ENTRY} {} = {}\n",
                shown(&e.inputs[0]),
                e.output.display_within(SHOWN_VALUES)
            );
            assert!(p.prompt.contains(&line), "{}", p.prompt);
        }
        assert_eq!(
            p.prompt.matches(&format!("  {ENTRY} ")).count(),
            examples.len(),
            "{}",
            p.prompt
        );
        // No gate input is shown, nor an answer no small example has (the
        // input sorted down has no increasing triple, like a small example).
        for e in family.gate().expect("a gate").inputs {
            assert!(!p.prompt.contains(&shown(&e.inputs[0])));
            if examples.iter().all(|x| x.output != e.output) {
                assert!(!p.prompt.contains(&format!("= {}\n", e.output)));
            }
        }
        // A shown gate input would add at least SHOWN_VALUES values.
        assert!(
            p.prompt.len() < 3 * SHOWN_VALUES as usize,
            "{}",
            p.prompt.len()
        );
    }

    #[test]
    fn the_library_and_solved_tasks_are_shown_in_surface_syntax() {
        let mut s = state(&["collapse/merge", "collapse/cancel", "collapse/absorb"]);
        let v0 = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let reverse = v0.prelude().get("reverse").unwrap().lib().0;
        let entry = LibraryEntry {
            name: None,
            doc: None,
            arity: 1,
            term: format!("(lam (lib_{reverse} (lib_{reverse} $0)))"),
        };
        s.apply(&Event::LibraryExtended {
            entries: vec![entry.clone()],
        })
        .unwrap();
        s.apply(&Event::FrontierSet {
            task: "collapse/cancel".into(),
            members: vec![FrontierMember {
                program: "(lam (lib_0 $0))".into(),
                cost: None,
            }],
        })
        .unwrap();
        let v = Verifier::new(&[entry], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 0, "tok", 2);
        assert!(p
            .prompt
            .contains("Write 2 different programs in the dc language that solve the"));
        assert!(p.prompt.contains("fn_0 : List "), "{}", p.prompt);
        assert!(p.prompt.contains("let fn_0 = fun"), "{}", p.prompt);
        assert!(p.prompt.contains("reverse (reverse"), "{}", p.prompt);
        let cancel = crate::verify::task("collapse/cancel").unwrap();
        assert!(p.prompt.contains(cancel.description()));
        assert!(p.prompt.contains("let solve = fun"), "{}", p.prompt);
        // A task with an empty frontier gives no example program.
        let absorb = crate::verify::task("collapse/absorb").unwrap();
        assert!(!p.prompt.contains(absorb.description()));
    }

    #[test]
    fn the_solved_examples_depend_on_the_seed_and_batch_only() {
        let names = [
            "collapse/merge",
            "collapse/cancel",
            "collapse/absorb",
            "collapse/absorb-count",
        ];
        let mut s = state(&names);
        for t in &names[1..] {
            s.apply(&Event::FrontierSet {
                task: (*t).into(),
                members: vec![FrontierMember {
                    program: "(lam $0)".into(),
                    cost: None,
                }],
            })
            .unwrap();
        }
        let v = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task(names[0]).unwrap();
        let a = payload(&s, &v, &*family, 3, "tok", 1);
        assert_eq!(a, payload(&s, &v, &*family, 3, "tok", 1));
        // Three other tasks have a program, and at most three are shown.
        let shown = a.prompt.matches("\nTask: ").count();
        assert_eq!(shown, MAX_SOLVED_EXAMPLES);
        // Over many batches, the order of the shown tasks changes.
        let orders: std::collections::BTreeSet<String> = (0..20)
            .map(|b| {
                let p = payload(&s, &v, &*family, b, "tok", 1).prompt;
                let start = p.find("## Programs that solve").unwrap();
                let end = p.find("## The task").unwrap();
                p[start..end].to_owned()
            })
            .collect();
        assert!(orders.len() > 1);
    }

    #[test]
    fn a_whole_payload_reads_back_from_its_event() {
        let s = state(&["collapse/merge"]);
        let v = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 0, "tok", 1);
        assert!(p.prompt.contains("batch 0, token tok,"), "{}", p.prompt);
        let text = serde_json::to_string(&p).unwrap();
        let issued = BatchIssued {
            batch: 0,
            iteration: 0,
            task: "collapse/merge".into(),
            wanted: 1,
            worker: "w".into(),
            token: "tok".into(),
            issued_ms: 0,
            deadline_ms: 1,
            payload: IssuedPayload::Whole(text.clone()),
        };
        let table = PromptTable::default();
        assert_eq!(table.read_payload(&issued).unwrap(), p);
        assert_eq!(table.payload_text(&issued).unwrap(), text);
    }

    /// The prompt as `payload` built it before it was made of pieces: the
    /// sections that are not empty, joined by line breaks.
    fn prompt_as_before(
        state: &RunState,
        verifier: &Verifier,
        family: &dyn TaskFamily,
        batch: u64,
        token: &str,
        wanted: u32,
    ) -> String {
        [
            instructions(wanted, batch, token),
            GUIDE.to_owned(),
            primitives(),
            prelude(verifier),
            library(state, verifier),
            solved(state, verifier, family.name(), batch),
            task(family),
        ]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
    }

    /// Proposers see no change: the pieces make the prompt that was built
    /// before, and only the batch's own text and the solved programs, which
    /// change with the batch, are not shared.
    #[test]
    fn the_pieces_make_the_prompt_as_it_was_built_before() {
        let names = ["collapse/merge", "collapse/cancel"];
        let mut s = state(&names);
        s.apply(&Event::FrontierSet {
            task: names[1].into(),
            members: vec![FrontierMember {
                program: "(lam $0)".into(),
                cost: None,
            }],
        })
        .unwrap();
        let v = Verifier::new(&[], dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap();
        let family = crate::verify::task(names[0]).unwrap();
        for (batch, wanted) in [(0, 1), (7, 3)] {
            let (p, pieces) = payload_in_pieces(&s, &v, &*family, batch, "tok", wanted);
            assert_eq!(
                p.prompt,
                prompt_as_before(&s, &v, &*family, batch, "tok", wanted)
            );
            let own: String = pieces
                .iter()
                .filter(|p| !p.shared)
                .map(|p| p.text.as_str())
                .collect();
            let solved = solved(&s, &v, family.name(), batch);
            assert!(!solved.is_empty());
            assert!(own.contains(&format!("batch {batch}, token tok,")));
            assert!(own.contains(&solved));
            assert!(own.len() < solved.len() + 64, "{own}");
        }
    }
}

#[cfg(test)]
#[path = "variety_tests.rs"]
mod variety_tests;

#[cfg(test)]
#[path = "prompt_guide_tests.rs"]
mod guide_tests;
