//! Hindley-Milner type inference with let-polymorphism (spec §2).
//!
//! The algorithm is Algorithm J over a union-find store of type nodes, with
//! level-based generalisation (Rémy): every unbound variable records the `let` depth
//! at which it was created, and a `let` generalises exactly the variables of its
//! bound expression whose level is deeper than the `let` itself. So generalisation
//! never scans the environment.
//!
//! **No host recursion.** Terms and types are walked with explicit work stacks, so
//! the depth of a term is limited by memory, not by the thread's stack (spec §4 keeps
//! the evaluator off host recursion for the same reason). Types written out as
//! [`Type`] trees are limited to [`MAX_TYPE_NODES`] constructors and
//! [`MAX_TYPE_DEPTH`] levels; a larger one is a [`TypeErrorKind::TypeTooLarge`]
//! error.
//!
//! **Shared structure.** Types in the store are DAGs: `(lam (pair $0 $0))` applied
//! `n` times has a type of `n` nodes that is `2^n` nodes written out. Every walk over
//! the store (occurs check, generalisation, instantiation, `Key` checks and
//! unification) visits each node at most once per walk, so its cost is linear in the
//! size of the DAG, not of the tree. A single unification can still cost time linear
//! in the size of the types involved, so inference is not linear in the worst case,
//! but it is polynomial.
//!
//! **Shared subterms.** Terms are DAGs too: `(if c t t)` nested `n` deep has `n`
//! distinct nodes but `2^n` as a tree. A non-leaf subterm reached by more than one
//! edge of the DAG is inferred once per scope and reused (see
//! [`TypeChecker::walk`] for why that is sound), so the walk is linear in the
//! number of distinct nodes, not in the size of the tree.
//!
//! **Checking.** [`TypeChecker::check`] pushes the expected type into the term:
//! through `lam` (parameter and body), the body of `let`, both branches of `if`, and
//! the result of an application before its arguments. So a clash with a task
//! signature is reported at the subterm where it happens, not at the whole term.
//!
//! **`Key` constraints.** `memorec` keys must be `Int`, `Bool`, `Unit` or tuples of
//! those (spec §6). A variable may carry a `Key` flag. Binding a flagged variable to
//! `Int`, `Bool` or `Unit` discharges it; binding it to a pair flags both
//! components; binding it to any other type is a [`TypeErrorKind::BadMemoKey`] error.
//! The flag is kept when a `let` generalises the variable and copied when the scheme
//! is instantiated, so the restriction holds through let-bound helpers and library
//! abstractions as well. A closed term whose key type is still a variable at the end
//! gets a scheme with a `Key` constraint on it, e.g.
//! `Key a => ((a -> b) -> a -> b) -> a -> b`.

use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;

use rustc_hash::{FxHashMap, FxHashSet};

use super::{Scheme, TyVar, Type, TypeError, TypeErrorKind};
use crate::prim::Prim;
use crate::term::{children, Arena, LibId, Term, TermId};

/// The types of library abstractions (spec §8), looked up by [`LibId`].
///
/// Implemented for slices and vectors (indexed by `LibId.0`) and for hash and
/// B-tree maps keyed by `LibId`.
pub trait LibEnv {
    /// The scheme of library abstraction `id`, or `None` if it is not defined.
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme>;
}

impl LibEnv for [Scheme] {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        self.get(id.0 as usize)
    }
}

impl LibEnv for Vec<Scheme> {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        self.as_slice().lib_scheme(id)
    }
}

impl<S: BuildHasher> LibEnv for HashMap<LibId, Scheme, S> {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        self.get(&id)
    }
}

impl LibEnv for BTreeMap<LibId, Scheme> {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        self.get(&id)
    }
}

/// The most type constructors and variables a [`Type`] produced by the checker may
/// have, written out as a tree.
pub const MAX_TYPE_NODES: usize = 10_000;

/// The deepest nesting a [`Type`] produced by the checker may have.
pub const MAX_TYPE_DEPTH: usize = 500;

/// Index of a node in the type store.
type Ty = u32;

/// The level of a generalised (quantified) variable.
const GENERIC: u32 = u32::MAX;

/// Pre-allocated nodes for the base types; base types never change, so sharing one
/// node each is safe.
const INT: Ty = 0;
const BOOL: Ty = 1;
const UNIT: Ty = 2;

/// Longest subterm text put into an error message.
const MAX_TERM_TEXT: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Head {
    Int,
    Bool,
    Unit,
    Pair,
    List,
    Option,
    Array,
    Tree,
    Arrow,
}

impl Head {
    fn arity(self) -> usize {
        match self {
            Head::Int | Head::Bool | Head::Unit => 0,
            Head::List | Head::Option | Head::Array | Head::Tree => 1,
            Head::Pair | Head::Arrow => 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Node {
    /// Unified with another node.
    Link(Ty),
    /// An unbound variable at a `let` depth, or [`GENERIC`] when quantified.
    Var { level: u32, key: bool },
    /// A rigid variable from an expected type in [`TypeChecker::check`]. It equals
    /// only itself.
    Rigid { key: bool },
    /// A type constructor applied to `arity` children; unused children are ignored.
    Con(Head, Ty, Ty),
}

/// Why two types could not be unified. The types are store nodes.
#[derive(Clone, Copy, Debug)]
enum UnifyError {
    /// Different constructors (left from the expected side, right from the found).
    Clash(Ty, Ty),
    /// A variable would contain itself.
    Occurs(Ty, Ty),
    /// A non-key type where a `Key` type is required.
    Key(Ty),
}

/// A type written out as a [`Type`] would exceed [`MAX_TYPE_NODES`] or
/// [`MAX_TYPE_DEPTH`].
#[derive(Clone, Copy, Debug)]
struct TooLarge;

#[derive(Clone, Copy, Debug)]
struct Binding {
    ty: Ty,
    /// Whether the type may contain generalised variables (a `let` binding).
    poly: bool,
    /// Unique per push in one walk, so a scope can be recognised (see
    /// [`MemoKey`]).
    serial: u32,
}

/// An expected type pushed into a subterm by [`TypeChecker::check`], with the
/// description of that subterm's position used if it does not fit.
#[derive(Clone, Copy, Debug)]
struct Want {
    ty: Ty,
    context: &'static str,
}

impl Want {
    fn at(ty: Ty, context: &'static str) -> Want {
        Want { ty, context }
    }
}

/// What to do with the type of a subterm once it is known: the continuation of the
/// term walk, kept on an explicit stack instead of the host stack.
#[derive(Clone, Copy, Debug)]
enum Frame {
    /// After a lambda body: drop the binder; the lambda has type `param -> body`.
    Lam { param: Ty },
    /// After a lambda body checked against the result of `arrow`: drop the binder;
    /// the lambda has type `arrow`.
    LamChecked { arrow: Ty },
    /// After the bound expression of a `let`: generalise it and go on to `body`.
    LetBound { body: TermId, want: Option<Want> },
    /// After the body of a `let`: drop the binder.
    LetBody,
    /// After the condition of an `if`.
    IfCond {
        cond: TermId,
        then: TermId,
        els: TermId,
        want: Option<Want>,
    },
    /// After the then branch of an `if` checked against an expected type.
    IfThenChecked { els: TermId, want: Want },
    /// After the then branch of an `if` without an expected type.
    IfThen { els: TermId },
    /// After the else branch of an `if` without an expected type.
    IfElse { then_ty: Ty, els: TermId },
    /// After the first operand of `and`/`or`.
    BoolFirst {
        first: TermId,
        second: TermId,
        op: &'static str,
    },
    /// After the second operand of `and`/`or`.
    BoolSecond { second: TermId, op: &'static str },
    /// After the head of an application spine.
    AppHead { sp: Spine, want: Option<Want> },
    /// After argument `i` of an application spine.
    AppArg {
        sp: Spine,
        i: usize,
        param: Ty,
        ret: Ty,
        want: Option<Want>,
    },
    /// Unify the subterm's type with `want`.
    Check { want: Want, term: TermId },
    /// After the first inference of a shared subterm in a scope: generalise its
    /// type, keep it under `key`, and return an instance.
    Memo { key: MemoKey },
}

/// A shared subterm and the scope it is inferred in: `None` for a closed subterm,
/// whose type does not depend on the scope, or else the serial of the innermost
/// binding in scope. Bindings form a stack, so the same innermost binding means
/// the very same bindings all the way out.
type MemoKey = (TermId, Option<u32>);

/// An application `app` = `head a1 ... an`, with the arguments at
/// `TypeChecker::spine[start..]`.
#[derive(Clone, Copy, Debug)]
struct Spine {
    app: TermId,
    head: TermId,
    start: usize,
}

/// One step of the term walk.
enum Step {
    /// Find the type of a subterm, optionally against an expected type.
    Eval(TermId, Option<Want>),
    /// Infer a shared subterm afresh (not from the memo), with no expected type.
    EvalShared(TermId),
    /// A subterm has this type; hand it to the top frame.
    Return(Ty),
}

/// A reusable type checker. Reusing one across many terms keeps its allocations.
#[derive(Debug, Default)]
pub struct TypeChecker {
    nodes: Vec<Node>,
    level: u32,
    env: Vec<Binding>,
    /// Instantiation map: generalised variables and constructor nodes to copies.
    inst: FxHashMap<Ty, Ty>,
    /// Visit marks for walks over the store; a node is visited in the current walk
    /// when its mark equals `epoch`.
    marks: Vec<u32>,
    epoch: u32,
    /// Scratch stacks, kept to reuse their allocations.
    work: Vec<Ty>,
    pairs: Vec<(Ty, Ty)>,
    seen_pairs: FxHashSet<(Ty, Ty)>,
    frames: Vec<Frame>,
    spine: Vec<TermId>,
    /// Non-leaf subterms of the current root reached by more than one edge.
    shared: FxHashSet<TermId>,
    /// Free depths of the shared subterms and the nodes under them.
    free_depths: FxHashMap<TermId, u32>,
    /// Generalised types of shared subterms already inferred, by scope.
    memo: FxHashMap<MemoKey, Ty>,
    next_serial: u32,
    /// Walks every term as a tree, without reusing shared subterms: the reference
    /// the tests compare the memoised walk with.
    #[cfg(test)]
    pub(super) tree_walk: bool,
}

/// Where a unification happens, for the error message.
struct Site {
    term: TermId,
    context: String,
}

impl Site {
    fn new(term: TermId, context: impl Into<String>) -> Site {
        Site {
            term,
            context: context.into(),
        }
    }
}

impl TypeChecker {
    /// A new checker.
    pub fn new() -> TypeChecker {
        TypeChecker::default()
    }

    /// Infers the principal type scheme of `term`, closed under `libs`.
    ///
    /// Free de Bruijn variables are errors. The result is normalised (see
    /// [`Scheme`]).
    pub fn infer<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        term: TermId,
        libs: &L,
    ) -> Result<Scheme, TypeError> {
        self.reset();
        let t = self.walk(arena, libs, term, None)?;
        self.export_scheme(t).map_err(|TooLarge| {
            let mut e = plain_error(arena, term, too_large());
            e.context = "the type of the whole term".into();
            e
        })
    }

    /// Checks that `term` has type `expected`.
    ///
    /// The variables of `expected` are rigid: the term must work for every choice of
    /// them, so `(lam (add $0 1))` does not check against `a -> a`. A variable may be
    /// used as a `memorec` key only if `expected` gives it a `Key` constraint.
    ///
    /// The expected type is pushed into the term (see the module documentation), so
    /// an error names the subterm where the clash happens.
    pub fn check<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        term: TermId,
        libs: &L,
        expected: &Scheme,
    ) -> Result<(), TypeError> {
        self.reset();
        let mut rigid: FxHashMap<TyVar, Ty> = FxHashMap::default();
        for v in expected.vars() {
            let key = expected.keys().contains(&v);
            rigid.insert(v, self.push(Node::Rigid { key }));
        }
        let want = self.import(expected.ty(), &mut |_, v| rigid[&v]);
        let want = Want::at(want, "the whole term, checked against its expected type");
        self.walk(arena, libs, term, Some(want)).map(|_| ())
    }

    fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::Con(Head::Int, 0, 0));
        self.nodes.push(Node::Con(Head::Bool, 0, 0));
        self.nodes.push(Node::Con(Head::Unit, 0, 0));
        self.level = 1;
        self.env.clear();
        self.frames.clear();
        self.spine.clear();
        self.shared.clear();
        self.free_depths.clear();
        self.memo.clear();
        self.next_serial = 0;
    }

    /// Pushes a binding for a new innermost variable.
    fn bind_var(&mut self, ty: Ty, poly: bool) {
        let serial = self.next_serial;
        self.next_serial = serial.checked_add(1).expect("fewer than 2^32 binders");
        self.env.push(Binding { ty, poly, serial });
    }

    /// Finds the non-leaf subterms of `root` that more than one edge of the DAG
    /// reaches.
    fn find_shared(&mut self, arena: &Arena, root: TermId) {
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut edges: FxHashSet<TermId> = FxHashSet::default();
        let mut stack = vec![root];
        seen.insert(root);
        while let Some(id) = stack.pop() {
            for c in children(&arena.get(id)) {
                if !edges.insert(c) && !children(&arena.get(c)).is_empty() {
                    self.shared.insert(c);
                }
                if seen.insert(c) {
                    stack.push(c);
                }
            }
        }
    }

    // ------------------------------------------------------------ store

    fn push(&mut self, n: Node) -> Ty {
        let id = Ty::try_from(self.nodes.len()).expect("fewer than 2^32 type nodes");
        self.nodes.push(n);
        id
    }

    fn node(&self, t: Ty) -> Node {
        self.nodes[t as usize]
    }

    fn set(&mut self, t: Ty, n: Node) {
        self.nodes[t as usize] = n;
    }

    fn fresh(&mut self) -> Ty {
        self.push(Node::Var {
            level: self.level,
            key: false,
        })
    }

    fn con1(&mut self, h: Head, a: Ty) -> Ty {
        self.push(Node::Con(h, a, 0))
    }

    fn con2(&mut self, h: Head, a: Ty, b: Ty) -> Ty {
        self.push(Node::Con(h, a, b))
    }

    fn arrow(&mut self, a: Ty, b: Ty) -> Ty {
        self.con2(Head::Arrow, a, b)
    }

    fn fun(&mut self, args: &[Ty], ret: Ty) -> Ty {
        args.iter().rev().fold(ret, |acc, &a| self.arrow(a, acc))
    }

    fn list(&mut self, a: Ty) -> Ty {
        self.con1(Head::List, a)
    }

    fn option(&mut self, a: Ty) -> Ty {
        self.con1(Head::Option, a)
    }

    fn array(&mut self, a: Ty) -> Ty {
        self.con1(Head::Array, a)
    }

    fn tree(&mut self, a: Ty) -> Ty {
        self.con1(Head::Tree, a)
    }

    fn pair(&mut self, a: Ty, b: Ty) -> Ty {
        self.con2(Head::Pair, a, b)
    }

    /// Follows links to the representative node, compressing the path.
    fn resolve(&mut self, t: Ty) -> Ty {
        let mut root = t;
        while let Node::Link(next) = self.node(root) {
            root = next;
        }
        let mut cur = t;
        while let Node::Link(next) = self.node(cur) {
            self.set(cur, Node::Link(root));
            cur = next;
        }
        root
    }

    /// Starts a new walk over the store: no node is marked visited.
    fn new_walk(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.epoch = 1;
        }
    }

    /// Marks `t` visited in the current walk; false if it already was.
    fn first_visit(&mut self, t: Ty) -> bool {
        let i = t as usize;
        if i >= self.marks.len() {
            self.marks.resize(self.nodes.len().max(i + 1), 0);
        }
        if self.marks[i] == self.epoch {
            false
        } else {
            self.marks[i] = self.epoch;
            true
        }
    }

    /// Pushes the children of constructor node `n`, rightmost first, so they are
    /// popped left to right.
    fn push_children(work: &mut Vec<Ty>, n: Node) {
        if let Node::Con(h, a, b) = n {
            if h.arity() >= 2 {
                work.push(b);
            }
            if h.arity() >= 1 {
                work.push(a);
            }
        }
    }

    // ------------------------------------------------------------ unification

    /// Unifies `expected` with `found`. On failure, the store keeps the partial
    /// unification; inference stops at the first error, so that is harmless.
    fn unify(&mut self, expected: Ty, found: Ty) -> Result<(), UnifyError> {
        let mut pairs = std::mem::take(&mut self.pairs);
        let mut seen = std::mem::take(&mut self.seen_pairs);
        pairs.clear();
        seen.clear();
        pairs.push((expected, found));
        let mut result = Ok(());
        while let Some((x, y)) = pairs.pop() {
            let a = self.resolve(x);
            let b = self.resolve(y);
            if a == b {
                continue;
            }
            let step = match (self.node(a), self.node(b)) {
                (Node::Var { level: la, key: ka }, Node::Var { level: lb, key: kb }) => {
                    self.set(a, Node::Link(b));
                    self.set(
                        b,
                        Node::Var {
                            level: la.min(lb),
                            key: ka || kb,
                        },
                    );
                    Ok(())
                }
                (Node::Var { level, key }, _) => self.bind(a, level, key, b),
                (_, Node::Var { level, key }) => self.bind(b, level, key, a),
                (Node::Con(ha, a1, a2), Node::Con(hb, b1, b2)) if ha == hb => {
                    // Constructor nodes are not linked to each other (that would
                    // change how the expected side prints in an error), so a pair
                    // met again through shared structure is skipped here instead.
                    if ha.arity() > 0 && seen.insert((a, b)) {
                        if ha.arity() >= 2 {
                            pairs.push((a2, b2));
                        }
                        pairs.push((a1, b1));
                    }
                    Ok(())
                }
                _ => Err(UnifyError::Clash(a, b)),
            };
            if step.is_err() {
                result = step;
                break;
            }
        }
        self.pairs = pairs;
        self.seen_pairs = seen;
        result
    }

    /// Binds unbound variable `v` (at `level`, with `key`) to non-variable `t`.
    fn bind(&mut self, v: Ty, level: u32, key: bool, t: Ty) -> Result<(), UnifyError> {
        self.occurs_adjust(v, level, t)?;
        if key {
            self.require_key(t)?;
        }
        self.set(v, Node::Link(t));
        Ok(())
    }

    /// Fails if `v` occurs in `t`; otherwise lowers the level of every variable in
    /// `t` to at most `level`, so `t` is not generalised earlier than `v` would be.
    fn occurs_adjust(&mut self, v: Ty, level: u32, t: Ty) -> Result<(), UnifyError> {
        self.new_walk();
        let mut work = std::mem::take(&mut self.work);
        work.clear();
        work.push(t);
        let mut result = Ok(());
        while let Some(x) = work.pop() {
            let r = self.resolve(x);
            if !self.first_visit(r) {
                continue;
            }
            let n = self.node(r);
            match n {
                Node::Var { level: l, key } => {
                    if r == v {
                        result = Err(UnifyError::Occurs(v, t));
                        break;
                    }
                    if l > level {
                        self.set(r, Node::Var { level, key });
                    }
                }
                Node::Con(..) => Self::push_children(&mut work, n),
                Node::Rigid { .. } | Node::Link(_) => {}
            }
        }
        self.work = work;
        result
    }

    /// Requires `t` to be a key type, flagging variables inside it.
    fn require_key(&mut self, t: Ty) -> Result<(), UnifyError> {
        self.new_walk();
        let mut work = std::mem::take(&mut self.work);
        work.clear();
        work.push(t);
        let mut result = Ok(());
        while let Some(x) = work.pop() {
            let r = self.resolve(x);
            if !self.first_visit(r) {
                continue;
            }
            let n = self.node(r);
            match n {
                Node::Var { level, .. } => self.set(r, Node::Var { level, key: true }),
                Node::Rigid { key: true } | Node::Con(Head::Int | Head::Bool | Head::Unit, ..) => {}
                Node::Con(Head::Pair, ..) => Self::push_children(&mut work, n),
                _ => {
                    result = Err(UnifyError::Key(r));
                    break;
                }
            }
        }
        self.work = work;
        result
    }

    // ------------------------------------------------------------ polymorphism

    /// Quantifies the variables of `t` created deeper than the current level.
    fn generalize(&mut self, t: Ty) {
        self.new_walk();
        let mut work = std::mem::take(&mut self.work);
        work.clear();
        work.push(t);
        while let Some(x) = work.pop() {
            let r = self.resolve(x);
            if !self.first_visit(r) {
                continue;
            }
            let n = self.node(r);
            match n {
                Node::Var { level, key } if level > self.level && level != GENERIC => {
                    self.set(
                        r,
                        Node::Var {
                            level: GENERIC,
                            key,
                        },
                    );
                }
                Node::Con(..) => Self::push_children(&mut work, n),
                _ => {}
            }
        }
        self.work = work;
    }

    /// A copy of `t` with fresh variables for its quantified ones. Subtrees without
    /// quantified variables are shared, not copied, and each node is copied once, so
    /// shared structure stays shared.
    fn instantiate(&mut self, t: Ty) -> Ty {
        let mut map = std::mem::take(&mut self.inst);
        map.clear();
        let root = self.resolve(t);
        // A node is pushed with `true` to be expanded; a constructor is pushed again
        // with `false` to be rebuilt once its children are done. Entries are
        // resolved roots.
        let mut expand: Vec<(Ty, bool)> = vec![(root, true)];
        while let Some((r, first)) = expand.pop() {
            if first {
                if map.contains_key(&r) {
                    continue;
                }
                match self.node(r) {
                    Node::Var {
                        level: GENERIC,
                        key,
                    } => {
                        let n = self.push(Node::Var {
                            level: self.level,
                            key,
                        });
                        map.insert(r, n);
                    }
                    Node::Con(h, a, b) if h.arity() > 0 => {
                        expand.push((r, false));
                        if h.arity() >= 2 {
                            let rb = self.resolve(b);
                            expand.push((rb, true));
                        }
                        let ra = self.resolve(a);
                        expand.push((ra, true));
                    }
                    _ => {
                        map.insert(r, r);
                    }
                }
            } else if let Node::Con(h, a, b) = self.node(r) {
                let ra = self.resolve(a);
                let na = map.get(&ra).copied().unwrap_or(ra);
                let (rb, nb) = if h.arity() >= 2 {
                    let rb = self.resolve(b);
                    (rb, map.get(&rb).copied().unwrap_or(rb))
                } else {
                    (b, b)
                };
                let out = if na == ra && nb == rb {
                    r
                } else {
                    self.con2(h, na, nb)
                };
                map.insert(r, out);
            }
        }
        let out = map.get(&root).copied().unwrap_or(root);
        self.inst = map;
        out
    }

    /// Builds a public type into the store; `var` supplies the node for a variable.
    fn import(&mut self, t: &Type, var: &mut impl FnMut(&mut Self, TyVar) -> Ty) -> Ty {
        match t {
            Type::Int => INT,
            Type::Bool => BOOL,
            Type::Unit => UNIT,
            Type::Var(v) => var(self, *v),
            Type::List(a) => {
                let a = self.import(a, var);
                self.list(a)
            }
            Type::Option(a) => {
                let a = self.import(a, var);
                self.option(a)
            }
            Type::Array(a) => {
                let a = self.import(a, var);
                self.array(a)
            }
            Type::Tree(a) => {
                let a = self.import(a, var);
                self.tree(a)
            }
            Type::Pair(a, b) => {
                let a = self.import(a, var);
                let b = self.import(b, var);
                self.pair(a, b)
            }
            Type::Arrow(a, b) => {
                let a = self.import(a, var);
                let b = self.import(b, var);
                self.arrow(a, b)
            }
        }
    }

    /// A fresh instance of `s` at the current level.
    fn instantiate_scheme(&mut self, s: &Scheme) -> Ty {
        let mut map: FxHashMap<TyVar, Ty> = FxHashMap::default();
        self.import(s.ty(), &mut |this, v| {
            *map.entry(v).or_insert_with(|| {
                this.push(Node::Var {
                    level: this.level,
                    key: s.keys().contains(&v),
                })
            })
        })
    }

    /// The public form of `t`; `names` maps variable nodes to [`TyVar`]s and grows as
    /// new variables are met. Fails if the result would exceed [`MAX_TYPE_NODES`] or
    /// [`MAX_TYPE_DEPTH`].
    fn export(&mut self, t: Ty, names: &mut FxHashMap<Ty, TyVar>) -> Result<Type, TooLarge> {
        enum Item {
            Visit(Ty, usize),
            Build(Head),
        }
        let mut todo = vec![Item::Visit(t, 1)];
        let mut out: Vec<Type> = vec![];
        let mut count = 0usize;
        while let Some(item) = todo.pop() {
            match item {
                Item::Visit(x, depth) => {
                    count += 1;
                    if count > MAX_TYPE_NODES || depth > MAX_TYPE_DEPTH {
                        return Err(TooLarge);
                    }
                    let r = self.resolve(x);
                    match self.node(r) {
                        Node::Var { .. } | Node::Rigid { .. } | Node::Link(_) => {
                            let next = TyVar(
                                u32::try_from(names.len()).expect("fewer than 2^32 variables"),
                            );
                            out.push(Type::Var(*names.entry(r).or_insert(next)));
                        }
                        Node::Con(h, a, b) => match h {
                            Head::Int => out.push(Type::Int),
                            Head::Bool => out.push(Type::Bool),
                            Head::Unit => out.push(Type::Unit),
                            _ => {
                                todo.push(Item::Build(h));
                                if h.arity() >= 2 {
                                    todo.push(Item::Visit(b, depth + 1));
                                }
                                todo.push(Item::Visit(a, depth + 1));
                            }
                        },
                    }
                }
                Item::Build(h) => {
                    let mut pop = || out.pop().expect("children are built before parents");
                    let built = match h {
                        Head::List => Type::list(pop()),
                        Head::Option => Type::option(pop()),
                        Head::Array => Type::array(pop()),
                        Head::Tree => Type::tree(pop()),
                        Head::Pair | Head::Arrow => {
                            let b = pop();
                            let a = pop();
                            if h == Head::Pair {
                                Type::pair(a, b)
                            } else {
                                Type::arrow(a, b)
                            }
                        }
                        Head::Int | Head::Bool | Head::Unit => {
                            unreachable!("base types are built when visited")
                        }
                    };
                    out.push(built);
                }
            }
        }
        Ok(out.pop().expect("the root is built last"))
    }

    fn export_scheme(&mut self, t: Ty) -> Result<Scheme, TooLarge> {
        let mut names = FxHashMap::default();
        let ty = self.export(t, &mut names)?;
        let keys: Vec<TyVar> = names
            .iter()
            .filter(|(&node, _)| {
                matches!(
                    self.node(node),
                    Node::Var { key: true, .. } | Node::Rigid { key: true }
                )
            })
            .map(|(_, &v)| v)
            .collect();
        Ok(Scheme::with_keys(ty, &keys))
    }

    // ------------------------------------------------------------ primitives

    /// A fresh instance of the type of `p` (spec §5, §6).
    fn prim_type(&mut self, p: Prim) -> Ty {
        match p {
            Prim::Add | Prim::Sub | Prim::Mul | Prim::Div | Prim::Mod => self.fun(&[INT, INT], INT),
            Prim::IntEq | Prim::Lt | Prim::Le => self.fun(&[INT, INT], BOOL),
            Prim::Not => self.fun(&[BOOL], BOOL),
            Prim::Eq => {
                let a = self.fresh();
                self.fun(&[a, a], BOOL)
            }
            Prim::Pair => {
                let (a, b) = (self.fresh(), self.fresh());
                let ab = self.pair(a, b);
                self.fun(&[a, b], ab)
            }
            Prim::Fst | Prim::Snd => {
                let (a, b) = (self.fresh(), self.fresh());
                let ab = self.pair(a, b);
                let out = if p == Prim::Fst { a } else { b };
                self.arrow(ab, out)
            }
            Prim::None => {
                let a = self.fresh();
                self.option(a)
            }
            Prim::Some => {
                let a = self.fresh();
                let oa = self.option(a);
                self.arrow(a, oa)
            }
            Prim::OptionCase => {
                let (a, b) = (self.fresh(), self.fresh());
                let oa = self.option(a);
                let on_none = self.arrow(UNIT, b);
                let on_some = self.arrow(a, b);
                self.fun(&[oa, on_none, on_some], b)
            }
            Prim::Nil => {
                let a = self.fresh();
                self.list(a)
            }
            Prim::Cons => {
                let a = self.fresh();
                let la = self.list(a);
                self.fun(&[a, la], la)
            }
            Prim::ListCase => {
                let (a, b) = (self.fresh(), self.fresh());
                let la = self.list(a);
                let on_nil = self.arrow(UNIT, b);
                let on_cons = self.fun(&[a, la], b);
                self.fun(&[la, on_nil, on_cons], b)
            }
            Prim::Length => {
                let a = self.fresh();
                let la = self.list(a);
                self.arrow(la, INT)
            }
            Prim::ArrayTabulate => {
                let a = self.fresh();
                let f = self.arrow(INT, a);
                let arr = self.array(a);
                self.fun(&[INT, f], arr)
            }
            Prim::ArrayGet => {
                let a = self.fresh();
                let arr = self.array(a);
                self.fun(&[arr, INT], a)
            }
            Prim::ArraySet => {
                let a = self.fresh();
                let arr = self.array(a);
                self.fun(&[arr, INT, a], arr)
            }
            Prim::ArraySize => {
                let a = self.fresh();
                let arr = self.array(a);
                self.arrow(arr, INT)
            }
            Prim::ArrayOfList => {
                let a = self.fresh();
                let (la, arr) = (self.list(a), self.array(a));
                self.arrow(la, arr)
            }
            Prim::ArrayToList => {
                let a = self.fresh();
                let (la, arr) = (self.list(a), self.array(a));
                self.arrow(arr, la)
            }
            Prim::Leaf => {
                let a = self.fresh();
                self.tree(a)
            }
            Prim::Node => {
                let a = self.fresh();
                let ta = self.tree(a);
                self.fun(&[ta, a, ta], ta)
            }
            Prim::TreeCase => {
                let (a, b) = (self.fresh(), self.fresh());
                let ta = self.tree(a);
                let on_leaf = self.arrow(UNIT, b);
                let on_node = self.fun(&[ta, a, ta], b);
                self.fun(&[ta, on_leaf, on_node], b)
            }
            Prim::TreeFold => {
                let (a, b) = (self.fresh(), self.fresh());
                let ta = self.tree(a);
                let step = self.fun(&[b, a, b], b);
                self.fun(&[b, step, ta], b)
            }
            Prim::Foldl => {
                let (a, b) = (self.fresh(), self.fresh());
                let la = self.list(a);
                let step = self.fun(&[b, a], b);
                self.fun(&[step, b, la], b)
            }
            Prim::Foldr => {
                let (a, b) = (self.fresh(), self.fresh());
                let la = self.list(a);
                let step = self.fun(&[a, b], b);
                self.fun(&[step, b, la], b)
            }
            Prim::Loop => {
                let b = self.fresh();
                let step = self.fun(&[INT, b], b);
                self.fun(&[INT, b, step], b)
            }
            Prim::Unfold => {
                let (s, a) = (self.fresh(), self.fresh());
                let as_ = self.pair(a, s);
                let next = self.option(as_);
                let step = self.arrow(s, next);
                let la = self.list(a);
                self.fun(&[step, s], la)
            }
            Prim::Bfix => {
                let (a, b) = (self.fresh(), self.fresh());
                let ab = self.arrow(a, b);
                let open = self.fun(&[ab, a], b);
                self.fun(&[INT, open, a], b)
            }
            Prim::Memorec => {
                let k = self.push(Node::Var {
                    level: self.level,
                    key: true,
                });
                let v = self.fresh();
                let kv = self.arrow(k, v);
                let open = self.fun(&[kv, k], v);
                self.fun(&[INT, open, k], v)
            }
        }
    }

    // ------------------------------------------------------------ inference

    /// Finds the type of `root`, against `want` if given, walking the term with an
    /// explicit stack of [`Frame`]s.
    ///
    /// A shared subterm (one the DAG reaches by more than one edge) is inferred
    /// once per scope, as if it were `let`-bound at that point: the first time, at
    /// one level deeper, after which its type is generalised and kept; every use,
    /// the first included, gets a fresh instance. This is sound and complete
    /// because let-expansion preserves typing in Hindley-Milner: `let x = e in
    /// C[x, x]` is typable with type τ exactly when `C[e, e]` is, provided `e` is
    /// typable (Damas and Milner 1982; here there are no side effects, so no value
    /// restriction). The memo key ([`MemoKey`]) makes sure both uses see the same
    /// bindings for the free variables of `e`, so the let could be placed at that
    /// point: closed subterms need no bindings, open ones are reused only under the
    /// same innermost binding, which means the same scope. The types of those
    /// bindings may be refined between the uses, but the kept type stays linked to
    /// them, so it is refined along with them. `Key` flags are kept by
    /// generalisation and copied by instantiation, as for `let`.
    ///
    /// A shared subterm is inferred without the expected type of its position;
    /// the type is checked against it afterwards. So in [`TypeChecker::check`] a
    /// clash inside a shared subterm is reported at that subterm, not deeper.
    fn walk<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        root: TermId,
        want: Option<Want>,
    ) -> Result<Ty, TypeError> {
        #[cfg(test)]
        let skip = self.tree_walk;
        #[cfg(not(test))]
        let skip = false;
        if !skip {
            self.find_shared(arena, root);
        }
        let mut step = Step::Eval(root, want);
        loop {
            step = match step {
                Step::Eval(id, want) => {
                    if self.shared.contains(&id) {
                        self.shared_subterm(arena, id, want)?
                    } else {
                        self.descend(arena, libs, id, want)?
                    }
                }
                Step::EvalShared(id) => self.descend(arena, libs, id, None)?,
                Step::Return(t) => match self.frames.pop() {
                    None => return Ok(t),
                    Some(frame) => self.resume(arena, frame, t)?,
                },
            };
        }
    }

    /// Starts on shared subterm `id`: reuses its kept type in this scope, or infers
    /// it afresh (see [`TypeChecker::walk`]).
    fn shared_subterm(
        &mut self,
        arena: &Arena,
        id: TermId,
        want: Option<Want>,
    ) -> Result<Step, TypeError> {
        if !self.free_depths.contains_key(&id) {
            arena.free_depths(id, &mut self.free_depths);
        }
        let scope = if self.free_depths[&id] == 0 {
            None
        } else {
            self.env.last().map(|b| b.serial)
        };
        let key = (id, scope);
        if let Some(&kept) = self.memo.get(&key) {
            let t = self.instantiate(kept);
            if let Some(w) = want {
                self.expect(arena, w.ty, t, Site::new(id, w.context))?;
            }
            return Ok(Step::Return(t));
        }
        if let Some(w) = want {
            self.frames.push(Frame::Check { want: w, term: id });
        }
        self.level += 1;
        self.frames.push(Frame::Memo { key });
        Ok(Step::EvalShared(id))
    }

    /// Starts on subterm `id`: either its type is known at once, or frames are
    /// pushed and a child is evaluated first.
    fn descend<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        id: TermId,
        want: Option<Want>,
    ) -> Result<Step, TypeError> {
        let t = match arena.get(id) {
            Term::Var(i) => {
                let depth = self.env.len();
                if i as usize >= depth {
                    return Err(plain_error(
                        arena,
                        id,
                        TypeErrorKind::UnboundVariable {
                            index: i,
                            in_scope: u32::try_from(depth).unwrap_or(u32::MAX),
                        },
                    ));
                }
                let b = self.env[depth - 1 - i as usize];
                if b.poly {
                    self.instantiate(b.ty)
                } else {
                    b.ty
                }
            }
            Term::Int(_) => INT,
            Term::Bool(_) => BOOL,
            Term::Unit => UNIT,
            Term::Prim(p) => self.prim_type(p),
            Term::Lib(l) => match libs.lib_scheme(l) {
                Some(s) => self.instantiate_scheme(s),
                None => return Err(plain_error(arena, id, TypeErrorKind::UnknownLib(l))),
            },
            Term::Lam(body) => {
                if let Some(w) = want {
                    let rw = self.resolve(w.ty);
                    if let Node::Con(Head::Arrow, p, r) = self.node(rw) {
                        self.bind_var(p, false);
                        self.frames.push(Frame::LamChecked { arrow: rw });
                        let ctx = "the body of `lam` (its type is fixed by the expected type)";
                        return Ok(Step::Eval(body, Some(Want::at(r, ctx))));
                    }
                    self.frames.push(Frame::Check { want: w, term: id });
                }
                let a = self.fresh();
                self.bind_var(a, false);
                self.frames.push(Frame::Lam { param: a });
                return Ok(Step::Eval(body, None));
            }
            Term::Let(bound, body) => {
                self.level += 1;
                self.frames.push(Frame::LetBound { body, want });
                return Ok(Step::Eval(bound, None));
            }
            Term::If(cond, then, els) => {
                self.frames.push(Frame::IfCond {
                    cond,
                    then,
                    els,
                    want,
                });
                return Ok(Step::Eval(cond, None));
            }
            Term::And(first, second) | Term::Or(first, second) => {
                let op = if matches!(arena.get(id), Term::And(..)) {
                    "and"
                } else {
                    "or"
                };
                if let Some(w) = want {
                    self.frames.push(Frame::Check { want: w, term: id });
                }
                self.frames.push(Frame::BoolFirst { first, second, op });
                return Ok(Step::Eval(first, None));
            }
            Term::App(..) => {
                let start = self.spine.len();
                let mut head = id;
                while let Term::App(f, a) = arena.get(head) {
                    self.spine.push(a);
                    head = f;
                }
                self.spine[start..].reverse();
                let sp = Spine {
                    app: id,
                    head,
                    start,
                };
                self.frames.push(Frame::AppHead { sp, want });
                return Ok(Step::Eval(head, None));
            }
        };
        if let Some(w) = want {
            self.expect(arena, w.ty, t, Site::new(id, w.context))?;
        }
        Ok(Step::Return(t))
    }

    /// Continues `frame` now that its pending subterm has type `t`.
    fn resume(&mut self, arena: &Arena, frame: Frame, t: Ty) -> Result<Step, TypeError> {
        Ok(match frame {
            Frame::Lam { param } => {
                self.env.pop();
                Step::Return(self.arrow(param, t))
            }
            Frame::LamChecked { arrow } => {
                self.env.pop();
                Step::Return(arrow)
            }
            Frame::LetBound { body, want } => {
                self.level -= 1;
                self.generalize(t);
                self.bind_var(t, true);
                self.frames.push(Frame::LetBody);
                let ctx = "the body of `let` (its type is fixed by the expected type)";
                Step::Eval(body, want.map(|w| Want::at(w.ty, ctx)))
            }
            Frame::LetBody => {
                self.env.pop();
                Step::Return(t)
            }
            Frame::IfCond {
                cond,
                then,
                els,
                want,
            } => {
                self.expect(arena, BOOL, t, Site::new(cond, "the condition of `if`"))?;
                match want {
                    Some(w) => {
                        self.frames.push(Frame::IfThenChecked { els, want: w });
                        let ctx =
                            "the then branch of `if` (its type is fixed by the expected type)";
                        Step::Eval(then, Some(Want::at(w.ty, ctx)))
                    }
                    None => {
                        self.frames.push(Frame::IfThen { els });
                        Step::Eval(then, None)
                    }
                }
            }
            Frame::IfThenChecked { els, want } => {
                let ctx = "the else branch of `if` (its type is fixed by the expected type)";
                Step::Eval(els, Some(Want::at(want.ty, ctx)))
            }
            Frame::IfThen { els } => {
                self.frames.push(Frame::IfElse { then_ty: t, els });
                Step::Eval(els, None)
            }
            Frame::IfElse { then_ty, els } => {
                let ctx = "the else branch of `if` (it must have the same type as the then \
                           branch)";
                self.expect(arena, then_ty, t, Site::new(els, ctx))?;
                Step::Return(then_ty)
            }
            Frame::BoolFirst { first, second, op } => {
                let ctx = format!("the first operand of `{op}`");
                self.expect(arena, BOOL, t, Site::new(first, ctx))?;
                self.frames.push(Frame::BoolSecond { second, op });
                Step::Eval(second, None)
            }
            Frame::BoolSecond { second, op } => {
                let ctx = format!("the second operand of `{op}`");
                self.expect(arena, BOOL, t, Site::new(second, ctx))?;
                Step::Return(BOOL)
            }
            Frame::Check { want, term } => {
                self.expect(arena, want.ty, t, Site::new(term, want.context))?;
                Step::Return(t)
            }
            Frame::Memo { key } => {
                self.level -= 1;
                self.generalize(t);
                self.memo.insert(key, t);
                Step::Return(self.instantiate(t))
            }
            Frame::AppHead { sp, want } => {
                let pending = match want {
                    Some(w) => self.expect_result(arena, sp, t, w)?,
                    None => None,
                };
                self.next_arg(arena, sp, 0, t, pending)?
            }
            Frame::AppArg {
                sp,
                i,
                param,
                ret,
                want,
            } => {
                let arg = self.spine[sp.start + i];
                let ctx = format!("argument {} of `{}`", i + 1, short(arena, sp.head));
                self.expect(arena, param, t, Site::new(arg, ctx))?;
                self.next_arg(arena, sp, i + 1, ret, want)?
            }
        })
    }

    /// In checking mode, unifies the result type of application `app` with `want`
    /// before its arguments are looked at, when the head's type shows how many
    /// arguments it takes. Returns `want` back if that has to wait until the
    /// arguments are done (the head is applied to too many arguments, which the
    /// argument walk reports).
    fn expect_result(
        &mut self,
        arena: &Arena,
        sp: Spine,
        tf: Ty,
        want: Want,
    ) -> Result<Option<Want>, TypeError> {
        let n = self.spine.len() - sp.start;
        let mut cur = tf;
        for _ in 0..n {
            match self.split_arrow(arena, sp, cur)? {
                Some((_, ret)) => cur = ret,
                None => return Ok(Some(want)),
            }
        }
        self.expect(arena, want.ty, cur, Site::new(sp.app, want.context))?;
        Ok(None)
    }

    /// Goes on to argument `i` of `sp`, with `tf` the type of its head applied to the
    /// first `i` arguments.
    fn next_arg(
        &mut self,
        arena: &Arena,
        sp: Spine,
        i: usize,
        tf: Ty,
        want: Option<Want>,
    ) -> Result<Step, TypeError> {
        let n = self.spine.len() - sp.start;
        if i == n {
            self.spine.truncate(sp.start);
            if let Some(w) = want {
                self.expect(arena, w.ty, tf, Site::new(sp.app, w.context))?;
            }
            return Ok(Step::Return(tf));
        }
        let Some((param, ret)) = self.split_arrow(arena, sp, tf)? else {
            let rf = self.resolve(tf);
            let mut names = FxHashMap::default();
            let kind = match self.export(rf, &mut names) {
                Ok(found) => TypeErrorKind::NotAFunction {
                    found,
                    applied_to: n,
                    accepted: i,
                },
                Err(TooLarge) => too_large(),
            };
            return Err(plain_error(arena, sp.app, kind));
        };
        self.frames.push(Frame::AppArg {
            sp,
            i,
            param,
            ret,
            want,
        });
        Ok(Step::Eval(self.spine[sp.start + i], None))
    }

    /// The parameter and result types of `tf`, the type of the head of `sp` applied
    /// to some of its arguments. A variable is unified with a fresh arrow; `None` if `tf` is
    /// some other type, so not a function.
    fn split_arrow(
        &mut self,
        arena: &Arena,
        sp: Spine,
        tf: Ty,
    ) -> Result<Option<(Ty, Ty)>, TypeError> {
        let rf = self.resolve(tf);
        match self.node(rf) {
            Node::Con(Head::Arrow, p, r) => Ok(Some((p, r))),
            Node::Var { .. } => {
                let (p, r) = (self.fresh(), self.fresh());
                let want = self.arrow(p, r);
                let ctx = format!("the function position of `{}`", short(arena, sp.app));
                self.expect(arena, want, rf, Site::new(sp.head, ctx))?;
                Ok(Some((p, r)))
            }
            _ => Ok(None),
        }
    }

    /// Unifies `expected` with `found`, turning a failure into an error at `site`.
    fn expect(
        &mut self,
        arena: &Arena,
        expected: Ty,
        found: Ty,
        site: Site,
    ) -> Result<(), TypeError> {
        self.unify(expected, found)
            .map_err(|e| self.error(arena, e, expected, found, site))
    }

    // ------------------------------------------------------------ errors

    /// Turns a unification failure between `expected` and `found` at `site` into a
    /// [`TypeError`]. All types in it share one variable renaming.
    fn error(
        &mut self,
        arena: &Arena,
        err: UnifyError,
        expected: Ty,
        found: Ty,
        site: Site,
    ) -> TypeError {
        let (kind, note) = self
            .error_kind(err, expected, found)
            .unwrap_or_else(|TooLarge| (too_large(), None));
        TypeError {
            kind: Box::new(kind),
            term: site.term,
            term_text: short(arena, site.term),
            context: site.context,
            note,
        }
    }

    fn error_kind(
        &mut self,
        err: UnifyError,
        expected: Ty,
        found: Ty,
    ) -> Result<(TypeErrorKind, Option<String>), TooLarge> {
        let mut names = FxHashMap::default();
        let mut note = None;
        let kind = match err {
            UnifyError::Clash(x, y) => {
                let exp = self.export(expected, &mut names)?;
                let fnd = self.export(found, &mut names)?;
                let cx = self.export(x, &mut names)?;
                let cy = self.export(y, &mut names)?;
                if self.is_rigid(x) || self.is_rigid(y) {
                    note = Some(
                        "type variables in the expected type are rigid: the term must work \
                         for every choice of them, so it may not fix them to a particular \
                         type or to each other"
                            .into(),
                    );
                }
                let conflict = if cx == exp && cy == fnd {
                    None
                } else {
                    Some((cx, cy))
                };
                TypeErrorKind::Mismatch {
                    expected: exp,
                    found: fnd,
                    conflict,
                }
            }
            UnifyError::Occurs(v, t) => {
                let var = self.export(v, &mut names)?;
                let ty = self.export(t, &mut names)?;
                TypeErrorKind::InfiniteType { var, ty }
            }
            UnifyError::Key(t) => {
                let found = self.export(t, &mut names)?;
                if self.is_rigid(t) {
                    note = Some(format!(
                        "the expected type must declare `Key {found} =>` to use `{found}` \
                         as a memorec key"
                    ));
                }
                TypeErrorKind::BadMemoKey { found }
            }
        };
        Ok((kind, note))
    }

    fn is_rigid(&mut self, t: Ty) -> bool {
        let r = self.resolve(t);
        matches!(self.node(r), Node::Rigid { .. })
    }
}

fn too_large() -> TypeErrorKind {
    TypeErrorKind::TypeTooLarge {
        max_nodes: MAX_TYPE_NODES,
        max_depth: MAX_TYPE_DEPTH,
    }
}

fn plain_error(arena: &Arena, term: TermId, kind: TypeErrorKind) -> TypeError {
    TypeError {
        kind: Box::new(kind),
        term,
        term_text: short(arena, term),
        context: String::new(),
        note: None,
    }
}

/// `id` in core syntax (as [`crate::sexpr::print`] writes it), cut to
/// [`MAX_TERM_TEXT`] bytes followed by ` ...`. It stops writing once past the limit,
/// so its cost does not depend on the size or depth of the term.
pub(super) fn short(arena: &Arena, id: TermId) -> String {
    let mut out = String::new();
    crate::sexpr::print_bounded(arena, id, MAX_TERM_TEXT, &mut out);
    if out.len() > MAX_TERM_TEXT {
        let mut cut = MAX_TERM_TEXT;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str(" ...");
    }
    out
}

/// Infers the principal type scheme of `term`. See [`TypeChecker::infer`].
pub fn infer<L: LibEnv + ?Sized>(
    arena: &Arena,
    term: TermId,
    libs: &L,
) -> Result<Scheme, TypeError> {
    TypeChecker::new().infer(arena, term, libs)
}

/// Checks `term` against `expected`. See [`TypeChecker::check`].
pub fn check<L: LibEnv + ?Sized>(
    arena: &Arena,
    term: TermId,
    libs: &L,
    expected: &Scheme,
) -> Result<(), TypeError> {
    TypeChecker::new().check(arena, term, libs, expected)
}

/// The type scheme of primitive `p` (spec §5, §6).
pub fn prim_scheme(p: Prim) -> Scheme {
    let mut tc = TypeChecker::new();
    tc.reset();
    let t = tc.prim_type(p);
    tc.export_scheme(t)
        .expect("primitive types are far below the size limits")
}
