//! Naming new abstractions: LILO's AutoDoc as a pulled job (decision D23).
//!
//! When an iteration adds entries to the library, each gets a naming job. A
//! proposer pulls one with `dc_next_naming_job`, reads the abstraction, and
//! answers with `dc_submit_naming`: a name and a one-line doc. The server
//! checks both. A name that fails the check ends the job all the same, and
//! the entry keeps its generated name `fn_i`; a doc that fails is dropped.
//! While the run's `autodoc` setting is on, no batch is issued until every
//! entry has a naming result, so each batch of an iteration sees the same
//! library text.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use dc_core::Prim;
use dc_surface::prelude::Prelude;
use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::prompt::{prelude, primitives, surface, GUIDE};
use crate::proposer::{new_token, Refused, Wait, MAX_NAME_LEN};
use crate::state::{LibraryEntry, NamingDone, NamingIssued, RunId, RunState};
use crate::store::Store;
use crate::verify::{
    check_prelude, entry_name, library_name, shipped_prelude, Verifier, VerifierCache, ENTRY,
};

/// The longest name, in bytes.
pub const MAX_ENTRY_NAME_LEN: usize = 40;

/// The shortest name, in bytes: names such as `x` or `f` read as local
/// variables, which programs often bind and so shadow.
pub const MIN_ENTRY_NAME_LEN: usize = 3;

/// The longest doc, in bytes.
pub const MAX_DOC_LEN: usize = 400;

/// How many frontier programs that use the abstraction a job shows.
const USES_SHOWN: usize = 3;

/// The names a program has in scope before the library: the primitives'
/// and the shipped prelude's functions.
fn builtin_names() -> &'static BTreeSet<String> {
    static NAMES: OnceLock<BTreeSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let prelude = Prelude::load().expect("the shipped prelude loads");
        let prelude = prelude.entries().iter().map(|e| e.name().to_owned());
        let prims = Prim::ALL.iter().map(|p| p.name().to_owned());
        prelude.chain(prims).collect()
    })
}

/// Whether `name` is `fn_` followed by digits only, the form of a generated
/// name.
fn is_generated(name: &str) -> bool {
    name.strip_prefix("fn_")
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// Checks `name` as the name of library entry `index` of `library`: the
/// checks of [`check_name_form`], and it is not a primitive or prelude name.
///
/// The server runs this check. Replaying a journal runs only
/// [`check_name_form`], so that a later build whose primitives or prelude
/// gain a name a run already used still reads the run (its prelude
/// fingerprint check then refuses work on it, if the prelude changed).
pub fn check_name(name: &str, index: usize, library: &[LibraryEntry]) -> Result<(), String> {
    check_name_form(name, index, library)?;
    if builtin_names().contains(name) {
        return Err(format!("{name:?} is a primitive or prelude function"));
    }
    Ok(())
}

/// Checks `name` as the name of library entry `index` of `library`: it is a
/// variable name of [`MIN_ENTRY_NAME_LEN`] to [`MAX_ENTRY_NAME_LEN`] bytes,
/// and it is not the entry point, a generated name, or another entry's name.
pub fn check_name_form(name: &str, index: usize, library: &[LibraryEntry]) -> Result<(), String> {
    if name.len() < MIN_ENTRY_NAME_LEN || name.len() > MAX_ENTRY_NAME_LEN {
        return Err(format!(
            "the name must have {MIN_ENTRY_NAME_LEN} to {MAX_ENTRY_NAME_LEN} bytes"
        ));
    }
    if !dc_surface::is_variable_name(name) {
        return Err(format!(
            "{name:?} is not a variable name: it must start with a lower-case letter or '_', \
             have only letters, digits, '_' and '\\'', and not be a keyword"
        ));
    }
    if name == ENTRY {
        return Err(format!("{name:?} is the name of the entry point"));
    }
    if is_generated(name) {
        return Err(format!(
            "{name:?} has the form of a generated name (fn_ and digits)"
        ));
    }
    let taken = library
        .iter()
        .enumerate()
        .any(|(j, e)| j != index && e.name.as_deref() == Some(name));
    if taken {
        return Err(format!("{name:?} names another library function"));
    }
    Ok(())
}

/// Checks a doc: at most [`MAX_DOC_LEN`] bytes, not blank, and only
/// printable ASCII, so it is one line with no control, format or separator
/// characters (such as U+2028 or a bidi override) that could change how
/// other proposers' prompts read.
pub fn check_doc(doc: &str) -> Result<(), String> {
    if doc.trim().is_empty() || doc.len() > MAX_DOC_LEN {
        return Err(format!(
            "the doc must have 1 to {MAX_DOC_LEN} bytes and not be blank"
        ));
    }
    if !doc.bytes().all(|b| (0x20..0x7f).contains(&b)) {
        return Err("the doc must be one line of printable ASCII".to_owned());
    }
    Ok(())
}

/// A naming job as the proposer gets it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NamingPayload {
    /// The job's number. Pass it to `dc_submit_naming`.
    pub job: u64,
    /// The job's token. Pass it to `dc_submit_naming`.
    pub token: String,
    /// How long the job is held for you, in seconds.
    pub lease_seconds: u64,
    /// The abstraction, how it is used, and how to answer.
    pub prompt: String,
}

/// What `dc_next_naming_job` returns: a job, or why there is none now.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NextNamingJob {
    /// The job, if there is one.
    pub job: Option<NamingPayload>,
    /// Why there is no job, if there is none.
    pub wait: Option<Wait>,
}

/// What `dc_submit_naming` returns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct NamingReceipt {
    /// The job.
    pub job: u64,
    /// The name the function now has: yours, or its generated name if yours
    /// was refused.
    pub name: String,
    /// The doc it now has, if any.
    pub doc: Option<String>,
    /// What was refused, and why.
    pub refused: Vec<String>,
}

/// Whether the s-expression `program` refers to library entry `entry`
/// (`lib_<entry>` not followed by another digit).
fn uses(program: &str, entry: u64) -> bool {
    let needle = format!("lib_{entry}");
    program.match_indices(&needle).any(|(at, _)| {
        let before = program[..at].chars().next_back();
        let after = program[at + needle.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
            && !after.is_some_and(|c| c.is_ascii_digit())
    })
}

/// The prompt of a naming job for `entry`.
pub fn naming_prompt(
    state: &RunState,
    verifier: &Verifier,
    entry: u64,
    job: u64,
    token: &str,
) -> String {
    let index = entry as usize;
    let lib = state.library();
    let this = &lib[index];
    let current = entry_name(index, this);
    let scheme = verifier
        .schemes()
        .get(index)
        .map_or_else(|| "?".to_owned(), ToString::to_string);
    let body = surface(verifier, &this.term)
        .unwrap_or_else(|| "(its definition is too large to show)".to_owned());
    let mut out = format!(
        "You are naming one function in a library of dc programs. The library was \
         learned by compressing programs that solve tasks; this function is a pattern \
         those programs share. Give it a short, descriptive name and a one-line doc \
         that says what it does, so that people writing new programs can use it.\n\n\
         Call dc_submit_naming with job {job}, token \"{token}\", your name, and your \
         doc. Call it once.\n\n\
         Rules for the name: a lower-case letter or '_' first, then letters, digits, \
         '_' or '\\''; at most {MAX_ENTRY_NAME_LEN} bytes; not a keyword, not \
         `{ENTRY}`, not fn_ followed by digits, and not a name listed below. Rules \
         for the doc: one line, at most {MAX_DOC_LEN} bytes. If the name breaks a \
         rule the function keeps the name {current}; if the doc does, it has no \
         doc.\n\n\
         ## The function\n\n{current} : {scheme}\nlet {current} = {body}\n\n",
    );
    let mut shown = 0;
    for task in state.tasks() {
        if shown == USES_SHOWN {
            break;
        }
        let Some(frontier) = state.frontier(task) else {
            continue;
        };
        if let Some(m) = frontier.iter().find(|m| uses(&m.program, entry)) {
            if shown == 0 {
                out.push_str("## Programs that use it\n\n");
            }
            if let Some(text) = surface(verifier, &m.program) {
                // No task name: payloads never show one (see `prompt`).
                out.push_str(&format!("{text}\n\n"));
                shown += 1;
            }
        }
    }
    let others: Vec<String> = lib
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != index)
        .map(|(j, e)| entry_name(j, e))
        .collect();
    if !others.is_empty() {
        out.push_str(&format!(
            "## Other library functions (names taken)\n\n{}\n\n",
            others.join(", ")
        ));
    }
    out.push_str(GUIDE);
    out.push('\n');
    out.push_str(&primitives());
    out.push('\n');
    out.push_str(&prelude(verifier));
    out
}

/// Hands `worker` a naming job of `run` at `now_ms`, or says why there is
/// none.
pub fn next_naming_job(
    store: &Store,
    run: &RunId,
    worker: &str,
    now_ms: u64,
    verifiers: &VerifierCache,
) -> Result<NextNamingJob, Refused> {
    check_worker(worker)?;
    store.commit(run, |state| {
        check_prelude(state.config(), shipped_prelude())
            .map_err(|e| Refused::State(e.to_string()))?;
        if !state.config().autodoc {
            return Err(Refused::Request(
                "this run does not name abstractions (autodoc is off)".to_owned(),
            ));
        }
        if let Some((job, j)) = state.live_naming(now_ms).find(|(_, j)| j.worker == worker) {
            let left = (j.deadline_ms - now_ms).div_ceil(1000).max(1);
            let reply = NextNamingJob {
                job: None,
                wait: Some(Wait {
                    reason: format!(
                        "you hold naming job {job}; answer it with dc_submit_naming before \
                         you ask for another. If you no longer have it, it is released \
                         when its lease ends, in {left} seconds"
                    ),
                    retry_after_seconds: left,
                }),
            };
            return Ok((reply, Vec::new()));
        }
        let Some(entry) = state.next_unnamed(now_ms) else {
            let wait = match state.live_naming(now_ms).map(|(_, j)| j.deadline_ms).min() {
                Some(first) => Wait {
                    reason: "every function still to name is held by another job".to_owned(),
                    retry_after_seconds: (first - now_ms).div_ceil(1000).max(1),
                },
                None => Wait {
                    reason: "no function waits to be named".to_owned(),
                    retry_after_seconds: 60,
                },
            };
            return Ok((
                NextNamingJob {
                    job: None,
                    wait: Some(wait),
                },
                Vec::new(),
            ));
        };
        let verifier = verifiers
            .get_or_build(state.library())
            .map_err(|e| Refused::State(e.to_string()))?;
        let job = state.naming_job_count();
        let token = new_token()?;
        let prompt = naming_prompt(state, &verifier, entry, job, &token);
        let lease_ms = state.config().lease_ms;
        let event = crate::state::Event::NamingIssued(NamingIssued {
            job,
            entry,
            worker: worker.to_owned(),
            token: token.clone(),
            issued_ms: now_ms,
            deadline_ms: now_ms.saturating_add(lease_ms),
            prompt: prompt.clone(),
        });
        let reply = NextNamingJob {
            job: Some(NamingPayload {
                job,
                token,
                lease_seconds: lease_ms / 1000,
                prompt,
            }),
            wait: None,
        };
        Ok((reply, vec![event]))
    })
}

fn check_worker(worker: &str) -> Result<(), Refused> {
    if worker.is_empty() || worker.len() > MAX_NAME_LEN {
        return Err(Refused::Request(format!(
            "the worker id must have 1 to {MAX_NAME_LEN} bytes"
        )));
    }
    Ok(())
}

/// Takes `worker`'s answer to naming job `job` of `run`. The name and the
/// doc are checked; what fails is refused and the rest kept (decision D23,
/// item 5). The answer ends the job either way. An answer with the wrong
/// worker or token, or for an entry already named, is refused and writes
/// nothing.
pub fn submit_naming(
    store: &Store,
    run: &RunId,
    worker: &str,
    job: u64,
    token: &str,
    name: &str,
    doc: &str,
) -> Result<NamingReceipt, Refused> {
    check_worker(worker)?;
    store.commit(run, |state| {
        let Some(issued) = state.naming_job(job) else {
            return Err(Refused::Request(format!("naming job {job} was not issued")));
        };
        if issued.worker != worker || !crate::state::same_secret(&issued.token, token) {
            return Err(Refused::Request(format!(
                "the worker or token does not match naming job {job}"
            )));
        }
        let entry = issued.entry;
        if state.is_named(entry) {
            return Err(Refused::Request(format!(
                "the function of naming job {job} is already named"
            )));
        }
        let index = entry as usize;
        let mut refused = Vec::new();
        let name = match check_name(name, index, state.library()) {
            Ok(()) => Some(name.to_owned()),
            Err(why) => {
                refused.push(format!("name: {why}"));
                None
            }
        };
        let doc = match check_doc(doc) {
            Ok(()) => Some(doc.to_owned()),
            Err(why) => {
                refused.push(format!("doc: {why}"));
                None
            }
        };
        let receipt = NamingReceipt {
            job,
            name: name.clone().unwrap_or_else(|| library_name(index)),
            doc: doc.clone(),
            refused: refused.clone(),
        };
        let event = crate::state::Event::NamingDone(NamingDone {
            job,
            entry,
            name,
            doc,
            refused,
        });
        Ok((receipt, vec![event]))
    })
}

#[cfg(test)]
#[path = "naming_tests.rs"]
mod tests;
