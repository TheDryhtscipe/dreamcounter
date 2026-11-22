//! Seed entries: library entries a run is given when it is made
//! ([`RunConfig::seed_entries`](crate::state::RunConfig::seed_entries),
//! decision D36).
//!
//! They join the library when iteration 0 ends, after the entries Stitch
//! finds then, each with its name and doc, so no naming job is issued for
//! them. In the settings, `lib_k` in a seed entry's term is seed entry `k`.
//! [`placed`] numbers those references as the library does.

use std::collections::HashMap;

use dc_core::{sexpr, Arena, LibId, Term};

use crate::naming::{check_doc, check_name};
use crate::state::LibraryEntry;
use crate::verify::Verifier;

/// The most seed entries a run may have.
pub const MAX_SEED_ENTRIES: usize = 16;
/// The most bytes a seed entry's term may have.
pub const MAX_SEED_TERM_LEN: usize = 4096;

/// Checks the seed entries of a new run, as a library of their own: there
/// are at most [`MAX_SEED_ENTRIES`]; each has a term of at most
/// [`MAX_SEED_TERM_LEN`] bytes, a name [`check_name`] accepts and a doc
/// [`check_doc`] accepts; and each term parses, types and expands with the
/// prelude and the entries before it, with at least as many leading lambdas
/// as its arity. An entry with no parameters must be a lambda, a literal, a
/// primitive or a library reference: any other is computed again at each use
/// (decision D34). `max_memory` is the run's limit for an evaluation; no
/// program runs here.
pub fn check(entries: &[LibraryEntry], max_memory: u64) -> Result<(), String> {
    if entries.len() > MAX_SEED_ENTRIES {
        return Err(format!(
            "seed_entries has {} entries; the most is {MAX_SEED_ENTRIES}",
            entries.len()
        ));
    }
    for (i, entry) in entries.iter().enumerate() {
        let bad = |why: String| format!("seed entry {i}: {why}");
        if entry.term.len() > MAX_SEED_TERM_LEN {
            return Err(bad(format!(
                "its term has more than {MAX_SEED_TERM_LEN} bytes"
            )));
        }
        let (Some(name), Some(doc)) = (&entry.name, &entry.doc) else {
            return Err(bad("it needs a name and a doc".to_owned()));
        };
        check_name(name, i, entries).map_err(bad)?;
        check_doc(doc).map_err(bad)?;
    }
    Verifier::new(entries, max_memory)
        .map_err(|e| format!("seed entry {}: {}", e.index, e.reason))?;
    let mut arena = Arena::new();
    for (i, entry) in entries.iter().enumerate() {
        let bad = |why: String| format!("seed entry {i}: {why}");
        let term = sexpr::parse(&mut arena, &entry.term).map_err(|e| bad(e.to_string()))?;
        let value = matches!(
            arena.get(term),
            Term::Lam(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) | Term::Lib(_)
        );
        if entry.arity == 0 && !value {
            return Err(bad(
                "it has no parameters and its term does work, which each use would do again"
                    .to_owned(),
            ));
        }
    }
    Ok(())
}

/// The seed entries as they join a library that holds `base` entries: each
/// reference to seed entry `k` becomes a reference to library entry
/// `base + k`. Other references, which are to prelude functions, stay.
pub fn placed(entries: &[LibraryEntry], base: usize) -> Result<Vec<LibraryEntry>, String> {
    let mut arena = Arena::new();
    let mut map = HashMap::new();
    for k in 0..entries.len() {
        let id = |n: usize| {
            u32::try_from(n)
                .map(LibId)
                .map_err(|_| "the library has too many entries".to_owned())
        };
        let (from, to) = (arena.lib(id(k)?), arena.lib(id(base + k)?));
        map.insert(from, to);
    }
    let mut memo = HashMap::new();
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let bad = |why: String| format!("seed entry {i}: {why}");
            let term = sexpr::parse(&mut arena, &entry.term).map_err(|e| bad(e.to_string()))?;
            let term = dc_driver::replace_nodes(&mut arena, term, &map, &mut memo);
            Ok(LibraryEntry {
                term: sexpr::print(&arena, term).map_err(|e| bad(e.to_string()))?,
                ..entry.clone()
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "seed_tests.rs"]
pub(crate) mod tests;
