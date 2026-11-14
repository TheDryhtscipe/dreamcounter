//! The payload handed to a proposer with each batch (decision D11).
//!
//! The payload is all a proposer sees (D11 requirement 1), so it carries
//! everything needed to write a program: how to answer, a short guide to the
//! dc language, the type of every primitive and prelude function, the
//! library, the task, its examples, and a few programs that solve other tasks
//! of the run. The types are taken from the code, not written by hand, so
//! they cannot drift from what the type checker accepts.
//!
//! The payload does not show the task's name, any cost, any frontier size, or
//! anything about earlier submissions.

use dc_core::{prim::Prim, sexpr, types::prim_scheme, Arena, LibId};
use dc_surface::pretty;
use dc_tasks::task::TaskFamily;
use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::state::{BatchIssued, RunState};
use crate::verify::{library_name, Verifier, ENTRY};

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
    /// How many programs to write.
    pub programs_wanted: u32,
    /// How long the batch is held for you, in seconds. After that it may be
    /// given to someone else, but a submission is still accepted.
    pub lease_seconds: u64,
    /// The task, the language, and how to answer.
    pub prompt: String,
}

/// The language guide. The types of primitives and prelude functions follow
/// it in the prompt.
const GUIDE: &str = "\
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
fn instructions(wanted: u32, batch: u64) -> String {
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
         Submit with the dc_submit tool: batch {batch}, and the programs as a list of \
         strings, one string for each program. You will get an acknowledgement, not \
         results: no program is run for you.\n"
    )
}

/// One line for each primitive: `name : type`.
fn primitives() -> String {
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
fn prelude(verifier: &Verifier) -> String {
    let mut out = String::from("## Prelude functions\n\n");
    for e in verifier.prelude().entries() {
        out.push_str(&format!("{} : {}\n", e.name(), e.scheme()));
    }
    out
}

/// The names programs use for library references: prelude names and `fn_i`.
fn lib_name(verifier: &Verifier) -> impl Fn(LibId) -> String + Sync + '_ {
    move |id| match verifier.prelude().name(id) {
        Some(name) => name.to_owned(),
        None => library_name(id.0 as usize),
    }
}

/// `text`, a closed core term, as surface syntax, or `None` if it cannot be
/// printed.
fn surface(verifier: &Verifier, text: &str) -> Option<String> {
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
        let name = library_name(i);
        out.push_str(&format!("{name} : {scheme}\n"));
        match surface(verifier, &entry.term) {
            Some(body) => out.push_str(&format!("let {name} = {body}\n\n")),
            None => out.push('\n'),
        }
    }
    out
}

/// A value as the payload shows it, in brackets if it has a space.
fn shown(v: &dc_eval::Value) -> String {
    let s = v.display_within(SHOWN_VALUES).to_string();
    if s.contains(' ') && !(s.starts_with('(') || s.starts_with('[')) {
        format!("({s})")
    } else {
        s
    }
}

/// The task section: its description, type and examples.
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

/// The payload of `batch`, a batch for `family` that asks for `wanted`
/// programs, under the state's library.
pub fn payload(
    state: &RunState,
    verifier: &Verifier,
    family: &dyn TaskFamily,
    batch: u64,
    wanted: u32,
) -> Payload {
    let sections = [
        instructions(wanted, batch),
        GUIDE.to_owned(),
        primitives(),
        prelude(verifier),
        library(state, verifier),
        solved(state, verifier, family.name(), batch),
        task(family),
    ];
    let prompt = sections
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");
    Payload {
        batch,
        programs_wanted: wanted,
        lease_seconds: state.config().lease_ms / 1000,
        prompt,
    }
}

/// The payload a [`BatchIssued`] event holds, read back.
pub fn read_payload(issued: &BatchIssued) -> Result<Payload, serde_json::Error> {
    serde_json::from_str(&issued.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Arm, Event, FrontierMember, LibraryEntry, RunConfig};

    fn state(tasks: &[&str]) -> RunState {
        RunState::create(
            RunConfig {
                arm: Arm::Blind,
                seed: 11,
                proposals_per_task: 2,
                programs_per_batch: 1,
                lease_ms: 90_000,
            },
            tasks.iter().map(|t| (*t).to_owned()).collect(),
        )
        .unwrap()
    }

    #[test]
    fn a_payload_has_the_task_its_examples_and_the_language() {
        let s = state(&["collapse/merge", "collapse/cancel"]);
        let v = Verifier::new(&[]).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 4, 1);
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
    fn the_library_and_solved_tasks_are_shown_in_surface_syntax() {
        let mut s = state(&["collapse/merge", "collapse/cancel", "collapse/absorb"]);
        let v0 = Verifier::new(&[]).unwrap();
        let reverse = v0.prelude().get("reverse").unwrap().lib().0;
        let entry = LibraryEntry {
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
            }],
        })
        .unwrap();
        let v = Verifier::new(&[entry]).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 0, 2);
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
                }],
            })
            .unwrap();
        }
        let v = Verifier::new(&[]).unwrap();
        let family = crate::verify::task(names[0]).unwrap();
        let a = payload(&s, &v, &*family, 3, 1);
        assert_eq!(a, payload(&s, &v, &*family, 3, 1));
        // Three other tasks have a program, and at most three are shown.
        let shown = a.prompt.matches("\nTask: ").count();
        assert_eq!(shown, MAX_SOLVED_EXAMPLES);
        // Over many batches, the order of the shown tasks changes.
        let orders: std::collections::BTreeSet<String> = (0..20)
            .map(|b| {
                let p = payload(&s, &v, &*family, b, 1).prompt;
                let start = p.find("## Programs that solve").unwrap();
                let end = p.find("## The task").unwrap();
                p[start..end].to_owned()
            })
            .collect();
        assert!(orders.len() > 1);
    }

    #[test]
    fn a_payload_reads_back_from_its_event() {
        let s = state(&["collapse/merge"]);
        let v = Verifier::new(&[]).unwrap();
        let family = crate::verify::task("collapse/merge").unwrap();
        let p = payload(&s, &v, &*family, 0, 1);
        let issued = BatchIssued {
            batch: 0,
            iteration: 0,
            task: "collapse/merge".into(),
            wanted: 1,
            worker: "w".into(),
            issued_ms: 0,
            deadline_ms: 1,
            payload: serde_json::to_string(&p).unwrap(),
        };
        assert_eq!(read_payload(&issued).unwrap(), p);
    }
}
