//! Core terms in a hash-consed arena (spec §3).
//!
//! Terms use de Bruijn indices: `Var(0)` is the nearest enclosing binder. `Lam` and
//! `Let` bind one variable each (`Let` binds index 0 in its body). Each distinct
//! node is stored once, so two structurally equal terms built in the same arena get
//! the same [`TermId`], and comparing terms is comparing ids.

use rustc_hash::FxHashMap;

use crate::prim::Prim;

#[cfg(test)]
thread_local! {
    /// Calls of [`Arena::get`] on this thread: the tests compare it with the
    /// work a pass over terms is charged for.
    pub(crate) static GETS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Index of a term in an [`Arena`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TermId(u32);

impl TermId {
    /// The raw index, for use as a table key.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Index of a library abstraction (spec §8).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LibId(pub u32);

/// One core term node. Children are [`TermId`]s in the same arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Term {
    /// De Bruijn variable.
    Var(u32),
    /// Lambda abstraction binding index 0 in its body.
    Lam(TermId),
    /// Application of a function to one argument.
    App(TermId, TermId),
    /// `Let(bound, body)`: evaluates `bound`, then `body` with it at index 0.
    Let(TermId, TermId),
    /// Integer literal.
    Int(i64),
    /// Boolean literal.
    Bool(bool),
    /// The unit value.
    Unit,
    /// A primitive or combinator.
    Prim(Prim),
    /// A reference to a library abstraction.
    Lib(LibId),
    /// `If(cond, then, else)`: lazy in both branches.
    If(TermId, TermId, TermId),
    /// Short-circuit conjunction.
    And(TermId, TermId),
    /// Short-circuit disjunction.
    Or(TermId, TermId),
}

/// A hash-consing store of terms.
#[derive(Debug, Default, Clone)]
pub struct Arena {
    nodes: Vec<Term>,
    index: FxHashMap<Term, TermId>,
}

impl Arena {
    /// An empty arena.
    pub fn new() -> Arena {
        Arena::default()
    }

    /// Returns the id of `term`, adding it if it is new.
    ///
    /// # Panics
    ///
    /// Panics if a child id is not in this arena, or if the arena holds 2³² terms.
    pub fn intern(&mut self, term: Term) -> TermId {
        if let Some(&id) = self.index.get(&term) {
            return id;
        }
        for child in children(&term) {
            assert!(
                child.index() < self.nodes.len(),
                "child {child:?} is not in this arena"
            );
        }
        let id = TermId(u32::try_from(self.nodes.len()).expect("arena is full"));
        self.nodes.push(term);
        self.index.insert(term, id);
        id
    }

    /// The node stored at `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not in this arena.
    pub fn get(&self, id: TermId) -> Term {
        #[cfg(test)]
        GETS.with(|g| g.set(g.get() + 1));
        self.nodes[id.index()]
    }

    /// The number of distinct nodes stored.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the arena holds no terms.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// `Var(i)`.
    pub fn var(&mut self, i: u32) -> TermId {
        self.intern(Term::Var(i))
    }

    /// `Lam(body)`.
    pub fn lam(&mut self, body: TermId) -> TermId {
        self.intern(Term::Lam(body))
    }

    /// `App(f, a)`.
    pub fn app(&mut self, f: TermId, a: TermId) -> TermId {
        self.intern(Term::App(f, a))
    }

    /// `f a0 a1 ...`, applied left to right.
    pub fn apps(&mut self, f: TermId, args: &[TermId]) -> TermId {
        args.iter().fold(f, |acc, &a| self.app(acc, a))
    }

    /// `Let(bound, body)`.
    pub fn let_(&mut self, bound: TermId, body: TermId) -> TermId {
        self.intern(Term::Let(bound, body))
    }

    /// Integer literal.
    pub fn int(&mut self, n: i64) -> TermId {
        self.intern(Term::Int(n))
    }

    /// Boolean literal.
    pub fn bool(&mut self, b: bool) -> TermId {
        self.intern(Term::Bool(b))
    }

    /// The unit literal.
    pub fn unit(&mut self) -> TermId {
        self.intern(Term::Unit)
    }

    /// A primitive.
    pub fn prim(&mut self, p: Prim) -> TermId {
        self.intern(Term::Prim(p))
    }

    /// A library reference.
    pub fn lib(&mut self, l: LibId) -> TermId {
        self.intern(Term::Lib(l))
    }

    /// `If(c, t, e)`.
    pub fn if_(&mut self, c: TermId, t: TermId, e: TermId) -> TermId {
        self.intern(Term::If(c, t, e))
    }

    /// `And(a, b)`.
    pub fn and(&mut self, a: TermId, b: TermId) -> TermId {
        self.intern(Term::And(a, b))
    }

    /// `Or(a, b)`.
    pub fn or(&mut self, a: TermId, b: TermId) -> TermId {
        self.intern(Term::Or(a, b))
    }

    /// The number of binders a context must supply for `id` to be closed: one more
    /// than the largest free de Bruijn index, or 0 for a closed term.
    ///
    /// It walks with an explicit stack and visits each distinct node once, so its
    /// cost is linear in the number of distinct nodes under `id`, however deep or
    /// shared they are.
    pub fn free_depth(&self, id: TermId) -> u32 {
        let mut memo = FxHashMap::default();
        self.free_depths(id, &mut memo);
        memo[&id]
    }

    /// Records in `memo` the [`free_depth`](Arena::free_depth) of `id` and of every
    /// node under it that `memo` does not already hold.
    pub(crate) fn free_depths(&self, id: TermId, memo: &mut FxHashMap<TermId, u32>) {
        // A node is pushed with `false` to be expanded, and again with `true` to be
        // computed once its children are done.
        let mut stack = vec![(id, false)];
        while let Some((t, children_done)) = stack.pop() {
            if memo.contains_key(&t) {
                continue;
            }
            let node = self.get(t);
            if !children_done {
                stack.push((t, true));
                stack.extend(
                    children(&node)
                        .into_iter()
                        .filter(|c| !memo.contains_key(c))
                        .map(|c| (c, false)),
                );
                continue;
            }
            let d = |c: TermId| memo[&c];
            let depth = match node {
                Term::Var(i) => i.saturating_add(1),
                Term::Lam(b) => d(b).saturating_sub(1),
                Term::Let(x, b) => d(x).max(d(b).saturating_sub(1)),
                t => children(&t).into_iter().map(d).max().unwrap_or(0),
            };
            memo.insert(t, depth);
        }
    }

    /// The number of nodes in `id` written out as a tree, counting a shared
    /// subterm at every use, if that is at most `limit`; `None` if it is more.
    ///
    /// Terms are DAGs, so this can be exponential in the number of distinct
    /// nodes: `(if c t t)` nested `n` deep has `3 * 2^n - 2`. It walks with an
    /// explicit stack and visits each distinct node once, and every distinct node
    /// counts at least one, so it stops after at most `limit + 1` of them. Its
    /// cost is linear in the smaller of `limit` and the number of distinct nodes
    /// under `id`.
    pub fn tree_size(&self, id: TermId, limit: usize) -> Option<usize> {
        let mut memo: FxHashMap<TermId, usize> = FxHashMap::default();
        let mut expanded = 0usize;
        // A node is pushed with `false` to be expanded, and again with `true` to be
        // summed once its children are done.
        let mut stack = vec![(id, false)];
        while let Some((t, children_done)) = stack.pop() {
            if memo.contains_key(&t) {
                continue;
            }
            let node = self.get(t);
            if !children_done {
                expanded += 1;
                if expanded > limit {
                    return None;
                }
                stack.push((t, true));
                stack.extend(
                    children(&node)
                        .into_iter()
                        .filter(|c| !memo.contains_key(c))
                        .map(|c| (c, false)),
                );
                continue;
            }
            let size = children(&node)
                .into_iter()
                .try_fold(1usize, |n, c| n.checked_add(memo[&c]))
                .filter(|&n| n <= limit)?;
            memo.insert(t, size);
        }
        Some(memo[&id])
    }

    /// Whether `id` has no free variables.
    pub fn is_closed(&self, id: TermId) -> bool {
        self.free_depth(id) == 0
    }
}

/// The direct children of a node, in left-to-right order.
pub fn children(term: &Term) -> Vec<TermId> {
    match *term {
        Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) | Term::Lib(_) => {
            vec![]
        }
        Term::Lam(b) => vec![b],
        Term::App(a, b) | Term::Let(a, b) | Term::And(a, b) | Term::Or(a, b) => vec![a, b],
        Term::If(a, b, c) => vec![a, b, c],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_structures_share_an_id() {
        let mut a = Arena::new();
        let x = a.var(0);
        let one = a.int(1);
        let add = a.prim(Prim::Add);
        let body1 = a.apps(add, &[x, one]);
        let f1 = a.lam(body1);
        let before = a.len();
        let x2 = a.var(0);
        let one2 = a.int(1);
        let add2 = a.prim(Prim::Add);
        let body2 = a.apps(add2, &[x2, one2]);
        let f2 = a.lam(body2);
        assert_eq!(f1, f2);
        assert_eq!(a.len(), before, "rebuilding an existing term added nodes");
    }

    #[test]
    fn different_structures_get_different_ids() {
        let mut a = Arena::new();
        let v0 = a.var(0);
        let v1 = a.var(1);
        assert_ne!(a.lam(v0), a.lam(v1));
        let t = a.bool(true);
        let one = a.int(1);
        assert_ne!(t, one);
        let x = a.int(0);
        assert_ne!(a.app(x, one), a.app(one, x));
    }

    #[test]
    fn free_depth_counts_binders() {
        let mut a = Arena::new();
        let v0 = a.var(0);
        let v2 = a.var(2);
        let id = a.lam(v0);
        assert!(a.is_closed(id));
        assert_eq!(a.free_depth(v2), 3);
        let l = a.lam(v2);
        assert_eq!(a.free_depth(l), 2);
        // let binds index 0 in the body only, not in the bound expression.
        let one = a.int(1);
        let let_ok = a.let_(one, v0);
        assert!(a.is_closed(let_ok));
        let let_free = a.let_(v0, v0);
        assert_eq!(a.free_depth(let_free), 1);
    }

    #[test]
    fn free_depth_of_deep_terms_needs_no_host_recursion() {
        const DEEP: u32 = 100_000;
        let mut a = Arena::new();
        // $DEEP under DEEP lambdas: free, with depth 1 left over.
        let mut t = a.var(DEEP);
        for _ in 0..DEEP {
            t = a.lam(t);
        }
        assert_eq!(a.free_depth(t), 1);
        assert!(!a.is_closed(t));
        let closed = a.lam(t);
        assert!(a.is_closed(closed));
        // Deep through every kind of node, closed by lets.
        let mut u = a.var(0);
        let one = a.int(1);
        for i in 0..DEEP {
            u = match i % 4 {
                0 => a.let_(one, u),
                1 => a.app(u, one),
                2 => a.if_(one, one, u),
                _ => a.and(u, one),
            };
        }
        assert_eq!(a.free_depth(u), 0);
    }

    #[test]
    fn free_depth_visits_shared_subterms_once() {
        // A DAG of about 100 nodes that is a tree of 2^100 nodes.
        let mut a = Arena::new();
        let c = a.var(3);
        let mut t = a.var(0);
        for _ in 0..100 {
            t = a.if_(c, t, t);
        }
        let start = std::time::Instant::now();
        assert_eq!(a.free_depth(t), 4);
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
    }

    /// `(if $1 t t)` nested `n` deep over `$0`: `n + 2` distinct nodes, and
    /// `3 * 2^n - 2` nodes written out as a tree.
    fn shared_ifs(a: &mut Arena, n: usize) -> TermId {
        let c = a.var(1);
        let mut t = a.var(0);
        for _ in 0..n {
            t = a.if_(c, t, t);
        }
        t
    }

    fn gets() -> u64 {
        GETS.with(std::cell::Cell::get)
    }

    #[test]
    fn tree_size_counts_shared_subterms_at_every_use() {
        let mut a = Arena::new();
        for n in 0..20 {
            let t = shared_ifs(&mut a, n);
            let size = 3 * (1usize << n) - 2;
            assert_eq!(a.tree_size(t, usize::MAX), Some(size), "n = {n}");
            assert_eq!(a.tree_size(t, size), Some(size), "n = {n}");
            assert_eq!(a.tree_size(t, size - 1), None, "n = {n}");
        }
        // Every kind of node counts one: 16 here.
        let t = crate::sexpr::parse(
            &mut a,
            "(let 5 (if (lt $0 3) true (and false (or true (lib_2 unit)))))",
        )
        .unwrap();
        assert_eq!(a.tree_size(t, 100), Some(16));
        assert_eq!(a.tree_size(t, 0), None);
    }

    #[test]
    fn tree_size_of_a_huge_dag_is_linear_in_its_distinct_nodes() {
        // 102 distinct nodes, 3 * 2^100 - 2 written out: more than a usize holds.
        let mut a = Arena::new();
        let t = shared_ifs(&mut a, 100);
        for limit in [1 << 20, usize::MAX] {
            let before = gets();
            assert_eq!(a.tree_size(t, limit), None);
            let reads = gets() - before;
            assert!(reads <= 2 * 102, "{reads} reads for limit {limit}");
        }
        // With a limit below the number of distinct nodes, the walk stops early.
        let before = gets();
        assert_eq!(a.tree_size(t, 10), None);
        let reads = gets() - before;
        assert!(reads <= 2 * 11, "{reads} reads for limit 10");
    }

    #[test]
    #[should_panic(expected = "is not in this arena")]
    fn foreign_child_ids_are_rejected() {
        let mut big = Arena::new();
        for i in 0..10 {
            big.int(i);
        }
        let foreign = big.int(9);
        let mut small = Arena::new();
        small.lam(foreign);
    }
}
