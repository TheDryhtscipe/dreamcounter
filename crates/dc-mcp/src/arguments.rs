//! Binding the arguments of library uses once (decision D35).
//!
//! A use of a library entry means its definition written out, with each
//! argument in place of its parameter (D1, `dc_core::expand`). So an
//! argument is evaluated wherever the entry's body evaluates the parameter:
//! not at all, once, or many times if the body uses the parameter inside a
//! function or in several places. A proposer reads `fn_3 (build n xs)` as a
//! call in a strict language, where `build n xs` is evaluated once. In run
//! lib-7108 an entry used its parameter inside the function it mapped over
//! 16384 indices, and the programs that passed it a call rebuilt that call's
//! result 16384 times (bead dreamcounter-a2602b).
//!
//! [`bind_arguments`] changes a program a proposer wrote, before it is typed
//! and stored: where an argument of a library use is not free to evaluate
//! and the entry can evaluate that parameter more than once in one use
//! ([`repeated_parameters`]), the argument is bound with a `let` around the
//! use, and the use takes the bound variable. `let` and a variable cost no
//! steps, so the program costs what the proposer's reading says.
//!
//! An argument for a parameter that the entry evaluates at most once stays
//! where it is. Such an entry can be a guard, `fn_2 x n = if n = 0 then 0
//! else x`, and the programs it was learned from rely on `x` not being
//! evaluated when `n` is 0. The rule counts occurrences, so it protects a
//! guard only if the guarded parameter occurs once on each path, and not
//! inside a function: see [`repeated_parameters`].
//!
//! A use with fewer arguments than the entry has parameters is a function,
//! and each call of it would evaluate the arguments it has again. So every
//! argument of such a use that is not free to evaluate is bound.
//!
//! Programs that the compressor rewrites are not changed: their uses come
//! from D1's rewriting, which keeps each program's steps as they were.

use std::collections::HashMap;

use dc_core::term::children;
use dc_core::{Arena, LibId, Prim, Term, TermId};

/// A count of evaluations: 0, 1, or this for more than one.
const MANY: u8 = 2;

/// For each parameter of a library entry, in argument order, whether one
/// use of the entry can evaluate it more than once.
///
/// `entry` is the entry's closed term `λ^arity. body` with the library
/// references in it expanded. A reference left in it counts as using
/// nothing. A parameter counts as evaluated more than once if it occurs
/// inside a function in the body, since the function can be called any
/// number of times, or if it occurs more than once on one path through the
/// body: the two branches of an `if` are different paths; both sides of
/// `and` and of `or` are on one. The two branches of `option_case`,
/// `list_case` and `tree_case` (a `match`) are different paths too, and not
/// functions, where each is written with the lambdas the case calls it
/// with. The count is of occurrences, so it can say "more than once" of a
/// parameter that every run evaluates once: one used twice in a branch that
/// a guard skips, or inside a function that is called once. If
/// `entry` has fewer than `arity` leading lambdas, every parameter counts as
/// evaluated more than once.
pub(crate) fn repeated_parameters(arena: &Arena, entry: TermId, arity: u32) -> Vec<bool> {
    let k = arity as usize;
    let mut body = entry;
    for _ in 0..k {
        match arena.get(body) {
            Term::Lam(b) => body = b,
            _ => return vec![true; k],
        }
    }
    // The counts of each parameter in a node, by the node and the number of
    // binders of the body it is under.
    let mut memo: HashMap<(TermId, u32), Vec<u8>> = HashMap::new();
    let mut stack = vec![(body, 0u32, false)];
    while let Some((id, depth, ready)) = stack.pop() {
        if memo.contains_key(&(id, depth)) {
            continue;
        }
        let node = arena.get(id);
        let under = depth.saturating_add(1);
        let case = case_parts(arena, id, depth);
        let parts: Vec<(TermId, u32)> = match (&case, node) {
            (Some(parts), _) => parts.to_vec(),
            (None, Term::Lam(b)) => vec![(b, under)],
            (None, Term::Let(x, b)) => vec![(x, depth), (b, under)],
            (None, other) => children(&other).into_iter().map(|c| (c, depth)).collect(),
        };
        if !ready {
            stack.push((id, depth, true));
            for &(c, d) in &parts {
                if !memo.contains_key(&(c, d)) {
                    stack.push((c, d, false));
                }
            }
            continue;
        }
        let count = |i: usize, j: usize| memo[&parts[i]][j];
        // What is looked at, then one of two branches: an `if` or a case.
        let paths = case.is_some() || matches!(node, Term::If(..));
        let counts: Vec<u8> = (0..k)
            .map(|j| match node {
                // Parameter `j` under `depth` binders is `$(depth + k - 1 - j)`.
                Term::Var(v) => {
                    let parameter = (k - 1 - j) as u64 + u64::from(depth);
                    u8::from(u64::from(v) == parameter)
                }
                Term::Lam(_) => {
                    if count(0, j) == 0 {
                        0
                    } else {
                        MANY
                    }
                }
                _ if paths => count(0, j) + count(1, j).max(count(2, j)),
                _ => (0..parts.len()).map(|i| count(i, j)).sum(),
            })
            .map(|c| c.min(MANY))
            .collect();
        memo.insert((id, depth), counts);
    }
    memo[&(body, 0)].iter().map(|&c| c >= MANY).collect()
}

/// If `id`, under `depth` binders, is `option_case`, `list_case` or
/// `tree_case` applied to all three arguments, and each branch is written
/// with at least the lambdas the case calls it with: what the case looks
/// at, and the body of each branch under those lambdas, each with the
/// number of binders it is under. The case calls one branch, once, so the
/// bodies are two paths. A branch written any other way, a variable say,
/// makes this `None`, and the case then counts as any application.
fn case_parts(arena: &Arena, id: TermId, depth: u32) -> Option<[(TermId, u32); 3]> {
    let Term::App(f, second) = arena.get(id) else {
        return None;
    };
    let Term::App(f, first) = arena.get(f) else {
        return None;
    };
    let Term::App(f, looked_at) = arena.get(f) else {
        return None;
    };
    // The empty branch is called with unit; the other with the parts.
    let called_with = match arena.get(f) {
        Term::Prim(Prim::OptionCase) => [1, 1],
        Term::Prim(Prim::ListCase) => [1, 2],
        Term::Prim(Prim::TreeCase) => [1, 3],
        _ => return None,
    };
    let body = |mut branch: TermId, lambdas: u32| {
        for _ in 0..lambdas {
            match arena.get(branch) {
                Term::Lam(b) => branch = b,
                _ => return None,
            }
        }
        Some((branch, depth.saturating_add(lambdas)))
    };
    Some([
        (looked_at, depth),
        body(first, called_with[0])?,
        body(second, called_with[1])?,
    ])
}

/// `root` with every free variable moved up by `by`: the term to put under
/// `by` new binders. Walks with a work list, each node once for each number
/// of binders it is under.
fn shift(arena: &mut Arena, root: TermId, by: u32) -> TermId {
    if by == 0 {
        return root;
    }
    let mut memo: HashMap<(TermId, u32), TermId> = HashMap::new();
    let mut stack = vec![(root, 0u32, false)];
    while let Some((id, depth, ready)) = stack.pop() {
        if memo.contains_key(&(id, depth)) {
            continue;
        }
        let node = arena.get(id);
        let under = depth.saturating_add(1);
        let parts: Vec<(TermId, u32)> = match node {
            Term::Lam(b) => vec![(b, under)],
            Term::Let(x, b) => vec![(x, depth), (b, under)],
            other => children(&other).into_iter().map(|c| (c, depth)).collect(),
        };
        if !ready {
            stack.push((id, depth, true));
            for &(c, d) in &parts {
                if !memo.contains_key(&(c, d)) {
                    stack.push((c, d, false));
                }
            }
            continue;
        }
        let m = |i: usize| memo[&parts[i]];
        let out = match node {
            Term::Var(v) if v >= depth => arena.var(v.saturating_add(by)),
            Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit => id,
            Term::Prim(_) | Term::Lib(_) => id,
            Term::Lam(_) => arena.lam(m(0)),
            Term::App(..) => arena.app(m(0), m(1)),
            Term::Let(..) => arena.let_(m(0), m(1)),
            Term::If(..) => arena.if_(m(0), m(1), m(2)),
            Term::And(..) => arena.and(m(0), m(1)),
            Term::Or(..) => arena.or(m(0), m(1)),
        };
        memo.insert((id, depth), out);
    }
    memo[&(root, 0)]
}

/// The head of the application `id` and the number of arguments it is
/// applied to: `(f, 2)` for `f a b`, and `(id, 0)` for a term that is not
/// an application. `memo` makes the walks down a long application linear
/// in its length in total.
fn head(arena: &Arena, id: TermId, memo: &mut HashMap<TermId, (TermId, u32)>) -> (TermId, u32) {
    let mut path = Vec::new();
    let mut at = id;
    let (h, mut n) = loop {
        if let Some(&known) = memo.get(&at) {
            break known;
        }
        match arena.get(at) {
            Term::App(f, _) => {
                path.push(at);
                at = f;
            }
            _ => break (at, 0),
        }
    };
    for p in path.into_iter().rev() {
        n = n.saturating_add(1);
        memo.insert(p, (h, n));
    }
    (h, n)
}

/// Whether evaluating `term` costs no steps and does no work, so that
/// writing it out several times is harmless: a variable, a literal, a
/// function, a primitive, or a reference to a library entry with
/// parameters or to a prelude function. A primitive with no arguments
/// (`nil`, `none`, `leaf`) costs one step, and counts as free here.
fn free_to_evaluate(arena: &Arena, term: TermId, repeated: &[Vec<bool>]) -> bool {
    match arena.get(term) {
        Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) | Term::Lam(_) => {
            true
        }
        // An entry with no parameters is its body, which can do work.
        Term::Lib(LibId(l)) => repeated.get(l as usize).is_none_or(|r| !r.is_empty()),
        Term::App(..) | Term::Let(..) | Term::If(..) | Term::And(..) | Term::Or(..) => false,
    }
}

/// `root`, a program compiled from a proposer's text, with the arguments of
/// its library uses bound once where the module docs say.
///
/// `repeated[i]` is [`repeated_parameters`] of library entry `i`; a
/// reference to any other library (a prelude function) is left alone, since
/// it is a function already. In a use `fn_i a0 ... a(n-1)`, each argument
/// `aj` that is one of the entry's (`j` less than its arity), is not free to
/// evaluate and has `repeated[i][j]` is bound, in argument order:
/// `let x = aj in fn_i ... x ...`. So the bound arguments are evaluated
/// first, in order, and each once, even if the entry would not have
/// evaluated one. In a use with fewer arguments than the entry has
/// parameters, every argument that is not free to evaluate is bound,
/// whatever `repeated[i]` says: the use is a function, and each call of it
/// would evaluate them again. The result is the same term whatever order the
/// program's parts are visited in, and binding a program again changes
/// nothing. Walks with a work list.
pub(crate) fn bind_arguments(arena: &mut Arena, root: TermId, repeated: &[Vec<bool>]) -> TermId {
    let mut heads: HashMap<TermId, (TermId, u32)> = HashMap::new();
    let mut done: HashMap<TermId, TermId> = HashMap::new();
    let mut stack = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if done.contains_key(&id) {
            continue;
        }
        let node = arena.get(id);
        // A use of a learned entry: its head, its entry and its arguments.
        // The applications inside it, `fn_i a0` in `fn_i a0 a1`, are not
        // visited as terms of their own.
        let (h, n) = head(arena, id, &mut heads);
        let entry = match arena.get(h) {
            Term::Lib(LibId(l)) if n > 0 => repeated.get(l as usize),
            _ => None,
        };
        let parts: Vec<TermId> = match entry {
            Some(_) => {
                let mut args = Vec::with_capacity(n as usize);
                let mut at = id;
                while let Term::App(f, a) = arena.get(at) {
                    args.push(a);
                    at = f;
                }
                args.reverse();
                args
            }
            None => children(&node),
        };
        if !ready {
            stack.push((id, true));
            stack.extend(
                parts
                    .iter()
                    .filter(|c| !done.contains_key(c))
                    .map(|&c| (c, false)),
            );
            continue;
        }
        let m = |i: usize| done[&parts[i]];
        let out = match (entry, node) {
            (Some(entry), _) => {
                let args: Vec<TermId> = (0..parts.len()).map(m).collect();
                bind_use(arena, h, entry, &args, repeated)
            }
            (None, Term::Lam(_)) => arena.lam(m(0)),
            (None, Term::App(..)) => arena.app(m(0), m(1)),
            (None, Term::Let(..)) => arena.let_(m(0), m(1)),
            (None, Term::If(..)) => arena.if_(m(0), m(1), m(2)),
            (None, Term::And(..)) => arena.and(m(0), m(1)),
            (None, Term::Or(..)) => arena.or(m(0), m(1)),
            (
                None,
                Term::Var(_)
                | Term::Int(_)
                | Term::Bool(_)
                | Term::Unit
                | Term::Prim(_)
                | Term::Lib(_),
            ) => id,
        };
        done.insert(id, out);
    }
    done[&root]
}

/// The use `head args...` of an entry whose [`repeated_parameters`] are
/// `entry`, with the arguments that need it bound around it.
fn bind_use(
    arena: &mut Arena,
    head: TermId,
    entry: &[bool],
    args: &[TermId],
    repeated: &[Vec<bool>],
) -> TermId {
    // A use with too few arguments is a function: each call of it would
    // evaluate the arguments again.
    let function = args.len() < entry.len();
    let bound: Vec<usize> = (0..args.len().min(entry.len()))
        .filter(|&j| (function || entry[j]) && !free_to_evaluate(arena, args[j], repeated))
        .collect();
    // An entry has few parameters; a count that does not fit is not bound.
    let Ok(lets) = u32::try_from(bound.len()) else {
        return arena.apps(head, args);
    };
    // Inside the `let`s, the bound argument `p` is `$(lets - 1 - p)`, and
    // every other argument is under all of them.
    let mut next = 0u32;
    let mut inner = Vec::with_capacity(args.len());
    for (j, &a) in args.iter().enumerate() {
        if bound.get(next as usize) == Some(&j) {
            inner.push(arena.var(lets - 1 - next));
            next += 1;
        } else {
            inner.push(shift(arena, a, lets));
        }
    }
    let mut out = arena.apps(head, &inner);
    // The bound argument `p` is under the `let`s of those before it.
    for (p, &j) in bound.iter().enumerate().rev() {
        let value = shift(arena, args[j], p as u32);
        out = arena.let_(value, out);
    }
    out
}

#[cfg(test)]
#[path = "arguments_tests.rs"]
mod tests;
