//! Seed entries: the checks at run creation, and their numbering in the
//! library.

use dc_surface::prelude::Prelude;

use super::*;

const MEMORY: u64 = dc_tasks::harness::DEFAULT_MAX_MEMORY;

fn entry(arity: u32, term: &str, name: &str) -> LibraryEntry {
    LibraryEntry {
        arity,
        term: term.to_owned(),
        name: Some(name.to_owned()),
        doc: Some(format!("What {name} does.")),
    }
}

/// The two entries of run lib-7109 of comparison 2fa7a8 that decision D36
/// seeds, with the second one's reference numbered within the list.
pub(crate) fn bucketing_entries() -> Vec<LibraryEntry> {
    vec![
        entry(
            1,
            "(lam (lam (lam (let (array_get $2 $0) \
             (array_set $2 $0 (cons $1 (array_get $2 $0)))))))",
            "bucket_push_index",
        ),
        entry(
            2,
            "(lam (lam (foldl (lib_0 $0) (array_tabulate 16384 (lam nil)) $1)))",
            "group_indices_by_key",
        ),
    ]
}

fn refused(entries: &[LibraryEntry], part: &str) {
    let e = check(entries, MEMORY).unwrap_err();
    assert!(e.contains(part), "{e}");
}

#[test]
fn the_entries_of_decision_d36_are_accepted() {
    check(&bucketing_entries(), MEMORY).unwrap();
    check(&[], MEMORY).unwrap();
}

#[test]
fn a_seed_entry_that_does_not_fit_is_refused_with_its_index() {
    let good = bucketing_entries();
    // The second entry first: it uses an entry that is not before it.
    refused(&[good[1].clone(), good[0].clone()], "seed entry 0");
    // A term that does not parse, one that does not type, and one with
    // fewer leading lambdas than its arity.
    refused(&[entry(1, "(lam $0", "broken")], "seed entry 0");
    refused(
        &[good[0].clone(), entry(1, "(lam (add $0 true))", "wrong")],
        "seed entry 1",
    );
    refused(&[entry(2, "(lam (add $0 1))", "short")], "seed entry 0");
    // No name, no doc, a name another entry has, and a prelude name.
    let unnamed = LibraryEntry {
        name: None,
        ..good[0].clone()
    };
    refused(&[unnamed], "needs a name and a doc");
    let undocumented = LibraryEntry {
        doc: None,
        ..good[0].clone()
    };
    refused(&[undocumented], "needs a name and a doc");
    let twice = LibraryEntry {
        name: good[0].name.clone(),
        ..good[1].clone()
    };
    refused(&[good[0].clone(), twice], "names another library function");
    refused(&[entry(1, "(lam $0)", "map")], "primitive or prelude");
    // Too long, and too many.
    let long = format!("(lam (add $0 {}))", "1".repeat(MAX_SEED_TERM_LEN));
    refused(&[entry(1, &long, "long")], "bytes");
    let many: Vec<LibraryEntry> = (0..=MAX_SEED_ENTRIES)
        .map(|i| entry(1, "(lam $0)", &format!("same_{i}")))
        .collect();
    refused(&many, "the most is 16");
    check(&many[..MAX_SEED_ENTRIES], MEMORY).unwrap();
}

#[test]
fn placing_numbers_each_reference_to_a_seed_entry_as_the_library_does() {
    let prelude = Prelude::load().unwrap();
    let map = format!("lib_{}", prelude.get("map").unwrap().lib().0);
    let seeds = vec![
        entry(1, "(lam (add $0 1))", "inc"),
        entry(1, "(lam (lib_0 (lib_0 $0)))", "inc_twice"),
        entry(
            1,
            &format!("(lam ({map} lib_1 (cons (lib_0 0) $0)))"),
            "bump",
        ),
    ];
    check(&seeds, MEMORY).unwrap();
    // With nothing before them they are as given.
    assert_eq!(placed(&seeds, 0).unwrap(), seeds);
    // After one entry, seed 0 is lib_1 and seed 1 is lib_2: a reference
    // moved to lib_1 is not moved again. The prelude function stays.
    let after = placed(&seeds, 1).unwrap();
    let terms: Vec<&str> = after.iter().map(|e| e.term.as_str()).collect();
    assert_eq!(
        terms,
        [
            "(lam (add $0 1))",
            "(lam (lib_1 (lib_1 $0)))",
            &format!("(lam ({map} lib_2 (cons (lib_1 0) $0)))"),
        ]
    );
    // Names, docs and arities are kept.
    for (a, s) in after.iter().zip(&seeds) {
        assert_eq!((a.arity, &a.name, &a.doc), (s.arity, &s.name, &s.doc));
    }
    // The placed entries are a library after any entry of one parameter.
    let mut library = vec![entry(1, "(lam (mul $0 2))", "double")];
    library.extend(after);
    Verifier::new(&library, MEMORY).unwrap();
}
