//! The rule for a bucketing abstraction (decision D36).
//!
//! A term is bucketing if it holds a term that puts a value onto the list
//! held in an array slot. That is one of two forms:
//!
//! - `array_set A K (cons V (array_get A K))`, where the two `A` terms are
//!   the same term and the two `K` terms are the same term;
//! - `let (array_get A K) B`, where `B` holds `array_set A' K' (cons V x)`,
//!   `x` is the variable the `let` binds, and `A'` and `K'` are `A` and `K`
//!   as they read at that place.
//!
//! The rule is about the text of the term. It does not follow a library
//! reference: [`is_bucketing`] is told which entries are bucketing. It does
//! not see a term that takes one of these forms only when an argument is put
//! in for a parameter.

use std::collections::HashSet;

use dc_core::term::children;
use dc_core::{Arena, LibId, Prim, Term, TermId};

/// Whether `root` is bucketing: it holds one of the forms in the module
/// docs, or a reference to a library entry `l` for which `entry(l)` is true.
pub(crate) fn is_bucketing(arena: &Arena, root: TermId, entry: impl Fn(u32) -> bool) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let term = arena.get(id);
        match term {
            Term::Lib(LibId(l)) if entry(l) => return true,
            Term::Let(bound, body) if puts_onto_named_list(arena, bound, body) => return true,
            _ => {}
        }
        if puts_onto_slot(arena, id) {
            return true;
        }
        stack.extend(children(&term));
    }
    false
}

/// `id` as a primitive applied to exactly `N` arguments, if it is one.
fn applied<const N: usize>(arena: &Arena, id: TermId, prim: Prim) -> Option<[TermId; N]> {
    let mut args = [id; N];
    let mut head = id;
    for slot in args.iter_mut().rev() {
        let Term::App(f, a) = arena.get(head) else {
            return None;
        };
        *slot = a;
        head = f;
    }
    (arena.get(head) == Term::Prim(prim)).then_some(args)
}

/// Whether `id` is `array_set A K (cons V (array_get A K))`.
fn puts_onto_slot(arena: &Arena, id: TermId) -> bool {
    let Some([a, k, list]) = applied::<3>(arena, id, Prim::ArraySet) else {
        return false;
    };
    let Some([_, tail]) = applied::<2>(arena, list, Prim::Cons) else {
        return false;
    };
    applied::<2>(arena, tail, Prim::ArrayGet) == Some([a, k])
}

/// Whether `bound` is `array_get A K` and `body`, the body of the `let` that
/// binds it, holds `array_set A' K' (cons V x)` with `x` the bound variable
/// and `A'`, `K'` the terms `A`, `K` as they read under the binders between.
fn puts_onto_named_list(arena: &Arena, bound: TermId, body: TermId) -> bool {
    let Some([a, k]) = applied::<2>(arena, bound, Prim::ArrayGet) else {
        return false;
    };
    // Each node with the number of binders between it and the `let`'s body.
    let mut seen = HashSet::new();
    let mut stack = vec![(body, 0_u32)];
    while let Some((id, depth)) = stack.pop() {
        if !seen.insert((id, depth)) {
            continue;
        }
        if let Some([a2, k2, list]) = applied::<3>(arena, id, Prim::ArraySet) {
            if let Some([_, tail]) = applied::<2>(arena, list, Prim::Cons) {
                // The `let` itself is one binder more.
                let by = depth + 1;
                if arena.get(tail) == Term::Var(depth)
                    && same_shifted(arena, a, a2, by)
                    && same_shifted(arena, k, k2, by)
                {
                    return true;
                }
            }
        }
        match arena.get(id) {
            Term::Lam(b) => stack.push((b, depth + 1)),
            Term::Let(v, b) => {
                stack.push((v, depth));
                stack.push((b, depth + 1));
            }
            other => stack.extend(children(&other).into_iter().map(|c| (c, depth))),
        }
    }
    false
}

/// Whether `inner` is `outer` with each free variable raised by `by`: the
/// same term, read under `by` more binders.
fn same_shifted(arena: &Arena, outer: TermId, inner: TermId, by: u32) -> bool {
    // Each pair with the number of binders inside the two terms above it.
    let mut stack = vec![(outer, inner, 0_u32)];
    let mut seen = HashSet::new();
    while let Some((x, y, cutoff)) = stack.pop() {
        if !seen.insert((x, y, cutoff)) {
            continue;
        }
        match (arena.get(x), arena.get(y)) {
            (Term::Var(i), Term::Var(j)) => {
                let want = if i < cutoff { Some(i) } else { i.checked_add(by) };
                if want != Some(j) {
                    return false;
                }
            }
            (Term::Lam(b), Term::Lam(c)) => stack.push((b, c, cutoff + 1)),
            (Term::Let(v, b), Term::Let(w, c)) => {
                stack.push((v, w, cutoff));
                stack.push((b, c, cutoff + 1));
            }
            (s, t) => {
                let (cs, ct) = (children(&s), children(&t));
                // Leaves, and nodes of one kind with the same children up to
                // the shift. Two nodes of different kinds differ.
                if std::mem::discriminant(&s) != std::mem::discriminant(&t) {
                    return false;
                }
                if cs.is_empty() {
                    if s != t {
                        return false;
                    }
                } else {
                    stack.extend(cs.into_iter().zip(ct).map(|(p, q)| (p, q, cutoff)));
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use dc_core::sexpr;

    fn bucketing(src: &str) -> bool {
        let mut arena = Arena::new();
        let term = sexpr::parse(&mut arena, src).unwrap();
        is_bucketing(&arena, term, |_| false)
    }

    #[test]
    fn the_entries_of_comparison_2fa7a8_are_bucketing() {
        // lib-7101, lib-7104 and lib-7109: push an index onto its key's list.
        assert!(bucketing(
            "(lam (lam (lam (let (array_get $2 $0) \
             (array_set $2 $0 (cons $1 (array_get $2 $0)))))))"
        ));
        // lib-7108: the same step inside a fold.
        assert!(bucketing(
            "(lam (lam (foldl (lam (lam (let (array_get $2 $0) \
             (array_set $2 $0 (cons $1 (array_get $2 $0)))))) \
             (array_tabulate 16384 (lam nil)) $1)))"
        ));
    }

    #[test]
    fn a_list_named_first_counts() {
        // buckets, key and value are $2, $1 and $0; the let names the old list.
        assert!(bucketing(
            "(lam (lam (lam (let (array_get $2 $1) (array_set $3 $2 (cons $1 $0))))))"
        ));
        // The same under one more binder inside the let.
        assert!(bucketing(
            "(lam (lam (lam (let (array_get $2 $1) \
             (let 7 (array_set $4 $3 (cons $2 $1)))))))"
        ));
    }

    #[test]
    fn another_slot_or_another_array_does_not_count() {
        // The list read is of another slot.
        assert!(!bucketing(
            "(lam (lam (lam (array_set $2 $1 (cons $0 (array_get $2 $0))))))"
        ));
        // The list read is of another array.
        assert!(!bucketing(
            "(lam (lam (lam (lam (array_set $3 $1 (cons $0 (array_get $2 $1)))))))"
        ));
        // A new list in the slot, and a value that is not a list.
        assert!(!bucketing("(lam (lam (lam (array_set $2 $1 (cons $0 nil)))))"));
        assert!(!bucketing(
            "(lam (lam (lam (array_set $2 $1 (add $0 (array_get $2 $1))))))"
        ));
        // With the list named first: another array, another slot, and a
        // variable that is not the one the let binds.
        assert!(!bucketing(
            "(lam (lam (lam (let (array_get $2 $1) (array_set $2 $2 (cons $1 $0))))))"
        ));
        assert!(!bucketing(
            "(lam (lam (lam (let (array_get $2 $1) (array_set $3 $1 (cons $1 $0))))))"
        ));
        assert!(!bucketing(
            "(lam (lam (lam (let (array_get $2 $1) (array_set $3 $2 (cons $1 $1))))))"
        ));
        // lib-7108's flatten_buckets reads the lists and puts nothing on one.
        assert!(!bucketing(
            "(lam (foldl (lam (lam (cons (array_get $2 $0) $1))) nil $0))"
        ));
    }

    #[test]
    fn a_reference_to_a_bucketing_entry_counts() {
        let mut arena = Arena::new();
        // lib-7109's group_indices_by_key.
        let term = sexpr::parse(
            &mut arena,
            "(lam (lam (foldl (lib_1 $0) (array_tabulate 16384 (lam nil)) $1)))",
        )
        .unwrap();
        assert!(is_bucketing(&arena, term, |l| l == 1));
        assert!(!is_bucketing(&arena, term, |l| l == 0));
    }
}
