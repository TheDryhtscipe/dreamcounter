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
//! edge of the DAG is inferred once for each different set of bindings its free
//! variables refer to, and the result is reused (see [`TypeChecker::walk`] for why
//! that is sound). The DAG here takes an application spine `(f a1 ... an)` as one
//! node, so `t` is shared in `(add t 1)` and `(add t 2)`, which share `(add t)`.
//! Bindings count as the same when both are monomorphic (the subterm's type is
//! kept as a function of theirs), or when both are polymorphic with the same type
//! up to renaming of their quantified variables, free variables included. So
//! `(if c t t)`, `(let t t)` and `((lam t) t)`, nested `n` deep, take time linear
//! in `n`, however far out the binders of `t`'s variables are, and so does a
//! helper bound again at every level whose type mentions a lambda's variable. A
//! use of a shared subterm costs time linear in the number of its distinct free
//! variables, or, past 64 of them, in the number of binders out to the outermost
//! one it uses.
//!
//! An ill-typed term is walked a second time, so that its error is the one
//! tree-walking HM reports, which names the subterm where the clash happens
//! (see [`TypeChecker::run`]). That walk reuses shared subterms too, and goes
//! into one only where its use fails, so its cost is polynomial, not
//! exponential, in the size of the DAG.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// `None` for a monomorphic binding. For a polymorphic one (a `let` binding
    /// with something to quantify), the [`Canon`] id of its type: the same for
    /// every binding whose type has the same shape, the same free variables and
    /// quantified variables that differ only in name.
    class: Option<u32>,
}

/// A node of a type, with its quantified variables numbered in order of first
/// occurrence, its other variables given by their store node, and its children
/// given as [`Canon`] ids. So two types get the same id when they are equal up to
/// renaming of their quantified variables, and they stay equal: anything that
/// later binds one of their free variables binds it in both.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Canon {
    Var {
        index: u32,
        key: bool,
    },
    /// An unbound variable that is not quantified, or a rigid one.
    Free(Ty),
    Con(Head, u32, u32),
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
    /// After the first inference of a shared subterm for its [`Scope`]: generalise
    /// its type, keep it, and return an instance. The details are on top of
    /// `TypeChecker::pending`.
    Memo,
    /// After a use of a shared subterm tried from the innermost [`Checkpoint`]:
    /// the use worked, so drop the checkpoint.
    Tried,
}

/// The most distinct free variables recorded for a subterm (see [`FreeVars`]).
const MAX_FREE_VARS: usize = 64;

/// The free variables of a subterm, for keying the memo of shared subterms.
#[derive(Clone, Debug)]
struct FreeVars {
    /// One more than the largest free de Bruijn index; 0 for a closed subterm.
    depth: u32,
    /// The free de Bruijn indices, ascending, or `None` if there are more than
    /// [`MAX_FREE_VARS`]. Then the subterm is taken to use all `depth` innermost
    /// bindings.
    vars: Option<Box<[u32]>>,
}

impl FreeVars {
    /// The indices a [`Scope`] has a [`Slot`] for, in its order.
    fn indices(&self) -> Vec<u32> {
        match &self.vars {
            Some(v) => v.to_vec(),
            None => (0..self.depth).collect(),
        }
    }
}

/// What the type of a shared subterm depends on: one [`Slot`] for each binding its
/// free variables refer to, in the order of [`FreeVars::indices`]. Bindings it
/// does not use do not matter, however far out its variables reach.
type Scope = Vec<Slot>;

/// A binding as far as the type of a shared subterm that can see it depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Slot {
    /// A monomorphic binding. The subterm is inferred with a fresh variable in
    /// its place, and the kept type takes that variable as a parameter, so the
    /// kept type does not depend on the binding.
    Mono,
    /// A polymorphic binding, by its [`Canon`] id: every binding of that id
    /// gives the subterm the same type. The kept type may share the free
    /// variables of the binding's type, as the type of a `let` body would.
    Poly(u32),
}

/// A shared subterm and what its type depends on.
type MemoKey = (TermId, Scope);

/// A shared subterm being inferred for the memo.
#[derive(Debug)]
struct Pending {
    key: MemoKey,
    /// The fresh variables standing for its monomorphic bindings, in slot order.
    params: Vec<Ty>,
    /// The bindings replaced by those variables, by position in `env`, to put
    /// back afterwards.
    saved: Vec<(usize, Binding)>,
}

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
    /// Walk a subterm in place, as part of the term around it, without the memo
    /// for it, against an expected type if given.
    InPlace(TermId, Option<Want>),
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
    /// Free variables of the shared subterms and the nodes under them.
    free_vars: FxHashMap<TermId, FreeVars>,
    /// Generalised types of shared subterms already inferred, by scope.
    memo: FxHashMap<MemoKey, Ty>,
    /// Shared subterms being inferred, innermost last; one per [`Frame::Memo`].
    pending: Vec<Pending>,
    /// Ids of the [`Canon`] nodes met in this walk.
    canon: FxHashMap<Canon, u32>,
    /// For each [`Canon`] id, whether that type has no free variables.
    canon_closed: Vec<bool>,
    /// Walks every term as a tree, without reusing shared subterms: the
    /// reference the tests compare the memoised walk with.
    pub(super) tree_walk: bool,
    /// Tries each use of a shared subterm from a [`Checkpoint`], and walks it
    /// again in place if that fails (see [`TypeChecker::run`]).
    locate: bool,
    /// The uses of shared subterms being tried, innermost last.
    checkpoints: Vec<Checkpoint>,
    /// The old contents of the store nodes changed since the oldest checkpoint,
    /// in the order changed. Nodes made after the innermost checkpoint are left
    /// out: going back to any checkpoint leaves nothing that refers to them,
    /// except kept types copied by [`TypeChecker::freeze`], which never change.
    trail: Vec<(Ty, Node)>,
    /// The memo keys added since the oldest checkpoint, in the order added.
    memo_log: Vec<MemoKey>,
    /// The keys with a closed scope whose fresh inference failed while uses were
    /// being tried. It fails whatever the state of the store, so their uses are
    /// walked in place at once (see [`TypeChecker::shared_subterm`]).
    failed: FxHashSet<MemoKey>,
    /// Steps taken by the term walk of the current call.
    steps: u64,
    /// The most steps of the term walk one call may take; more is a panic. Lets
    /// the tests bound the work without a timer.
    #[cfg(test)]
    pub(super) step_limit: Option<u64>,
}

/// The state of the checker before a use of a shared subterm is tried, to go
/// back to if the use fails (see [`TypeChecker::run`]).
#[derive(Clone, Debug)]
struct Checkpoint {
    /// The shared subterm and the expected type of its position.
    id: TermId,
    want: Option<Want>,
    /// The use's memo key, if its scope is closed.
    closed_key: Option<MemoKey>,
    nodes: usize,
    trail: usize,
    frames: usize,
    spine: usize,
    env: usize,
    pending: usize,
    memo_log: usize,
    level: u32,
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
        let t = self.run(arena, libs, term, None)?;
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
        self.run(arena, libs, term, Some(expected)).map(|_| ())
    }

    /// Finds the type of `term`, checked against `expected` if given.
    ///
    /// If that fails, and the walk reused shared subterms, the term is walked
    /// again to find the error tree-walking HM would report, which names the
    /// subterm where the clash happens. The first walk checks a reused subterm
    /// as a whole (see [`TypeChecker::walk`]), so its error can name a whole
    /// shared subterm, and could depend on whether the same text appears
    /// elsewhere in the term.
    ///
    /// The second walk tries each use of a shared subterm as the first does,
    /// from a [`Checkpoint`]. If the use fails, the checker goes back to the
    /// checkpoint and walks that subterm in place, as the tree walk would,
    /// trying the uses of shared subterms inside it in the same way. A use that
    /// works makes the same unifications, up to renaming of fresh variables, as
    /// the tree walk of that subterm does, in a different order; unification
    /// succeeds or fails on the whole set, whatever the order. So the walk takes
    /// the same unifications as the tree walk, up to the one that fails, and
    /// reports the same error. An error unwinds to the innermost use being
    /// tried, so every use is walked in place at most once for each use being
    /// tried around it. A use whose subterm, inferred afresh for a closed scope,
    /// fails is walked in place at once from then on, since it fails in every
    /// state of the store; without that, a chain of shared subterms over a leaf
    /// that is ill typed on its own takes time exponential in its length.
    fn run<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        term: TermId,
        expected: Option<&Scheme>,
    ) -> Result<Ty, TypeError> {
        let first = self.walk_from_scratch(arena, libs, term, expected);
        let Err(e) = first else {
            return first;
        };
        if self.tree_walk || self.shared.is_empty() {
            return Err(e);
        }
        self.locate = true;
        let again = self.walk_from_scratch(arena, libs, term, expected);
        self.locate = false;
        debug_assert!(again.is_err(), "both walks accept the same terms");
        match again {
            Err(located) => Err(located),
            Ok(_) => Err(e),
        }
    }

    /// Resets the checker and walks `term`, against `expected` if given.
    fn walk_from_scratch<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        term: TermId,
        expected: Option<&Scheme>,
    ) -> Result<Ty, TypeError> {
        self.reset();
        let want = expected.map(|expected| {
            let mut rigid: FxHashMap<TyVar, Ty> = FxHashMap::default();
            for v in expected.vars() {
                // Scheme keys are sorted.
                let key = expected.keys().binary_search(&v).is_ok();
                rigid.insert(v, self.push(Node::Rigid { key }));
            }
            let want = self.import(expected.ty(), &mut |_, v| rigid[&v]);
            Want::at(want, "the whole term, checked against its expected type")
        });
        self.walk(arena, libs, term, want)
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
        self.free_vars.clear();
        self.memo.clear();
        self.pending.clear();
        self.canon.clear();
        self.canon_closed.clear();
        self.steps = 0;
        self.checkpoints.clear();
        self.trail.clear();
        self.memo_log.clear();
        self.failed.clear();
    }

    /// Pushes a binding for a new innermost variable.
    fn bind_var(&mut self, ty: Ty, poly: bool) {
        let class = poly.then(|| self.canonical(ty));
        self.env.push(Binding { ty, class });
    }

    /// The [`Canon`] id of `t`.
    fn canonical(&mut self, t: Ty) -> u32 {
        let mut ids: FxHashMap<Ty, u32> = FxHashMap::default();
        let mut vars = 0u32;
        // A constructor is pushed with `false` to be expanded, and again with `true`
        // once its children are done. Children are popped left to right, so the
        // variables are numbered in order of first occurrence.
        let root = self.resolve(t);
        let mut stack = vec![(root, false)];
        while let Some((r, children_done)) = stack.pop() {
            if ids.contains_key(&r) {
                continue;
            }
            let canon = match self.node(r) {
                Node::Var {
                    level: GENERIC,
                    key,
                } => {
                    vars += 1;
                    Canon::Var {
                        index: vars - 1,
                        key,
                    }
                }
                Node::Var { .. } | Node::Rigid { .. } => Canon::Free(r),
                Node::Link(_) => unreachable!("the stack holds resolved nodes"),
                Node::Con(h, a, b) => {
                    let (a, b) = (self.resolve(a), self.resolve(b));
                    if !children_done {
                        stack.push((r, true));
                        if h.arity() >= 2 {
                            stack.push((b, false));
                        }
                        if h.arity() >= 1 {
                            stack.push((a, false));
                        }
                        continue;
                    }
                    let id = |c: Ty, used: bool| if used { ids[&c] } else { 0 };
                    Canon::Con(h, id(a, h.arity() >= 1), id(b, h.arity() >= 2))
                }
            };
            let next = u32::try_from(self.canon.len()).expect("fewer than 2^32 types");
            let id = *self.canon.entry(canon).or_insert(next);
            if id == next {
                let closed = match canon {
                    Canon::Var { .. } => true,
                    Canon::Free(_) => false,
                    Canon::Con(h, a, b) => {
                        let closed = |c: u32| self.canon_closed[c as usize];
                        (h.arity() < 1 || closed(a)) && (h.arity() < 2 || closed(b))
                    }
                };
                self.canon_closed.push(closed);
            }
            ids.insert(r, id);
        }
        ids[&root]
    }

    /// Finds the non-leaf subterms of `root` that more than one edge of the DAG
    /// reaches, where the DAG is the one the term walk sees: an application spine
    /// `(f a1 ... an)` is one node with children `f, a1, ..., an`, since the walk
    /// never infers a prefix `(f a1 ... ai)` on its own. So `t` in `(add t 1)` and
    /// `(add t 2)` counts as reached twice, although both reach it through the one
    /// node `(add t)`.
    fn find_shared(&mut self, arena: &Arena, root: TermId) {
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut edges: FxHashSet<TermId> = FxHashSet::default();
        let mut stack = vec![root];
        let mut kids = Vec::new();
        seen.insert(root);
        while let Some(id) = stack.pop() {
            kids.clear();
            let node = arena.get(id);
            if let Term::App(..) = node {
                let mut head = id;
                while let Term::App(f, a) = arena.get(head) {
                    kids.push(a);
                    head = f;
                }
                kids.push(head);
            } else {
                kids.extend(children(&node));
            }
            for &c in &kids {
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
        let i = t as usize;
        if self.checkpoints.last().is_some_and(|cp| i < cp.nodes) {
            self.trail.push((t, self.nodes[i]));
        }
        self.nodes[i] = n;
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

    /// Quantifies the variables of `t` created deeper than the current level, and
    /// says whether `t` then has any quantified variable.
    fn generalize(&mut self, t: Ty) -> bool {
        let mut poly = false;
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
                Node::Var { level, key } if level > self.level => {
                    poly = true;
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
        poly
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
                    // Scheme keys are sorted.
                    key: s.keys().binary_search(&v).is_ok(),
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
    /// once per [`Scope`], as if it were `let`-bound at that point: the first
    /// time, at one level deeper, after which its type is generalised and kept;
    /// every use, the first included, gets a fresh instance. This is sound and
    /// complete because let-expansion preserves typing in Hindley-Milner: `let x =
    /// e in C[x, x]` is typable with type τ exactly when `C[e, e]` is, provided `e`
    /// is typable (Damas and Milner 1982; here there are no side effects, so no
    /// value restriction).
    ///
    /// The free variables of `e` are what the memo key ([`MemoKey`]) is about.
    /// A monomorphic one, `y`, is abstracted: `e` is inferred with a fresh
    /// variable for `y`, and what is kept is the type of `(lam y. e)`, whose
    /// parameter is unified with the type of `y` at every use. That is the
    /// expansion `let f = (lam y. e) in C[f y, f y]`, which types the same as
    /// `C[e, e]` because every occurrence of a monomorphic `y` has the one type.
    /// A polymorphic one cannot be abstracted (HM has no polymorphic parameters),
    /// so the key holds it, by its [`Canon`] id: two bindings whose types are the
    /// same up to renaming of their quantified variables, with the same free
    /// variables, give `e` the same type, whose free variables the kept type then
    /// shares, as a `let` would. Bindings refined between uses refine the kept
    /// type with them, and refine both such bindings alike.
    /// `Key` flags are kept by generalisation and copied by instantiation, as for
    /// `let`.
    ///
    /// A shared subterm is inferred without the expected type of its position;
    /// the type is checked against it afterwards, and the types of the
    /// monomorphic bindings it uses are checked against its kept type at each use.
    /// So a clash found that way is reported at the whole shared subterm. When
    /// that happens, [`TypeChecker::run`] walks the term again to find where the
    /// clash is.
    fn walk<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        root: TermId,
        want: Option<Want>,
    ) -> Result<Ty, TypeError> {
        if !self.tree_walk {
            self.find_shared(arena, root);
        }
        let mut step = Step::Eval(root, want);
        loop {
            self.steps += 1;
            #[cfg(test)]
            {
                if let Some(limit) = self.step_limit {
                    assert!(
                        self.steps <= limit,
                        "the term walk took more than {limit} steps"
                    );
                }
            }
            let next = match step {
                Step::Eval(id, want) => {
                    if self.shared.contains(&id) {
                        self.shared_subterm(arena, id, want)
                    } else {
                        self.descend(arena, libs, id, want)
                    }
                }
                Step::EvalShared(id) => self.descend(arena, libs, id, None),
                Step::InPlace(id, want) => self.descend(arena, libs, id, want),
                Step::Return(t) => match self.frames.pop() {
                    None => return Ok(t),
                    Some(frame) => self.resume(arena, frame, t),
                },
            };
            step = match next {
                Ok(next) => next,
                // Only the second walk of [`TypeChecker::run`] has checkpoints.
                Err(e) => match self.checkpoints.pop() {
                    None => return Err(e),
                    Some(cp) => {
                        // A use whose key is not in the memo failed while its
                        // subterm was being inferred afresh.
                        if let Some(key) = &cp.closed_key {
                            if !self.memo.contains_key(key) {
                                self.failed.insert(key.clone());
                            }
                        }
                        self.back_to(cp)
                    }
                },
            };
        }
    }

    /// Starts on shared subterm `id`: reuses its kept type for this [`Scope`], or
    /// infers it afresh (see [`TypeChecker::walk`]).
    fn shared_subterm(
        &mut self,
        arena: &Arena,
        id: TermId,
        want: Option<Want>,
    ) -> Result<Step, TypeError> {
        if !self.free_vars.contains_key(&id) {
            free_vars(arena, id, &mut self.free_vars);
        }
        let fv = &self.free_vars[&id];
        let len = self.env.len();
        if fv.depth as usize > len {
            // A free variable is unbound: walk it as a tree to report it.
            if self.locate {
                return Ok(Step::InPlace(id, want));
            }
            return self.descend_unshared(id, want);
        }
        let indices = fv.indices();
        let mut scope = Vec::with_capacity(indices.len());
        for &i in &indices {
            let b = self.env[len - 1 - i as usize];
            scope.push(match b.class {
                None => Slot::Mono,
                Some(c) => Slot::Poly(c),
            });
        }
        let key = (id, scope);
        if self.locate {
            if self.failed.contains(&key) {
                // Fresh inference fails in every context, so this use fails too:
                // trying it again would redo that work at every level of sharing.
                return Ok(Step::InPlace(id, want));
            }
            self.checkpoints.push(Checkpoint {
                id,
                want,
                closed_key: self.scope_closed(&key.1).then(|| key.clone()),
                nodes: self.nodes.len(),
                trail: self.trail.len(),
                frames: self.frames.len(),
                spine: self.spine.len(),
                env: self.env.len(),
                pending: self.pending.len(),
                memo_log: self.memo_log.len(),
                level: self.level,
            });
        }
        if let Some(&kept) = self.memo.get(&key) {
            let t = self.instantiate(kept);
            let t = self.apply_scope(arena, id, &key.1, &indices, t)?;
            if let Some(w) = want {
                self.expect(arena, w.ty, t, Site::new(id, w.context))?;
            }
            if self.locate {
                self.drop_checkpoint();
            }
            return Ok(Step::Return(t));
        }
        if self.locate {
            self.frames.push(Frame::Tried);
        }
        if let Some(w) = want {
            self.frames.push(Frame::Check { want: w, term: id });
        }
        // The monomorphic bindings it uses are replaced, in place, by fresh
        // variables one level deeper, which the kept type takes as parameters.
        // The other bindings stay as they are.
        self.level += 1;
        let mut params = Vec::new();
        let mut saved = Vec::new();
        for (slot, &i) in key.1.iter().zip(&indices) {
            if *slot == Slot::Mono {
                let pos = len - 1 - i as usize;
                let p = self.fresh();
                params.push(p);
                saved.push((pos, self.env[pos]));
                self.env[pos] = Binding { ty: p, class: None };
            }
        }
        self.pending.push(Pending { key, params, saved });
        self.frames.push(Frame::Memo);
        Ok(Step::EvalShared(id))
    }

    /// How many keys the last call marked as failed.
    #[cfg(test)]
    pub(super) fn failed_keys(&self) -> usize {
        self.failed.len()
    }

    /// Whether no binding in `scope` has a type with free variables: then
    /// inferring a subterm for it uses only nodes it makes itself.
    fn scope_closed(&self, scope: &Scope) -> bool {
        scope.iter().all(|slot| match *slot {
            Slot::Mono => true,
            Slot::Poly(c) => self.canon_closed[c as usize],
        })
    }

    /// Drops the innermost checkpoint, once the use it was for has worked.
    fn drop_checkpoint(&mut self) {
        self.checkpoints.pop();
        if self.checkpoints.is_empty() {
            self.trail.clear();
            self.memo_log.clear();
        }
    }

    /// The kept type to put in the memo for `key` while uses are being tried.
    ///
    /// If the scope has no binding whose type has free variables, inferring the
    /// subterm used only nodes it made itself, so its kept type holds for that
    /// scope in any state of the store. Then it is copied out of reach of going
    /// back to a checkpoint, and stays in the memo. Otherwise going back to a
    /// checkpoint before it removes it.
    fn keep_past_checkpoints(&mut self, key: &MemoKey, kept: Ty) -> Ty {
        let closed = self.scope_closed(&key.1);
        if let Some(frozen) = closed.then(|| self.freeze(kept)).flatten() {
            return frozen;
        }
        self.memo_log.push(key.clone());
        kept
    }

    /// A copy of `t` in new nodes, with no links, or `None` if `t` has a
    /// variable that is not quantified. No unification changes the copy: it has
    /// only quantified variables, which instantiation replaces, and constructor
    /// nodes, which unification never changes.
    fn freeze(&mut self, t: Ty) -> Option<Ty> {
        let mut map: FxHashMap<Ty, Ty> = FxHashMap::default();
        let root = self.resolve(t);
        // As in `instantiate`: `true` to expand, `false` to rebuild.
        let mut stack: Vec<(Ty, bool)> = vec![(root, true)];
        while let Some((r, first)) = stack.pop() {
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
                            level: GENERIC,
                            key,
                        });
                        map.insert(r, n);
                    }
                    Node::Var { .. } | Node::Rigid { .. } | Node::Link(_) => return None,
                    Node::Con(h, a, b) => {
                        if r <= UNIT {
                            map.insert(r, r);
                            continue;
                        }
                        stack.push((r, false));
                        if h.arity() >= 2 {
                            let rb = self.resolve(b);
                            stack.push((rb, true));
                        }
                        if h.arity() >= 1 {
                            let ra = self.resolve(a);
                            stack.push((ra, true));
                        }
                    }
                }
            } else if let Node::Con(h, a, b) = self.node(r) {
                let copy = |this: &mut Self, c: Ty, used: bool| {
                    if used {
                        let rc = this.resolve(c);
                        map[&rc]
                    } else {
                        0
                    }
                };
                let na = copy(self, a, h.arity() >= 1);
                let nb = copy(self, b, h.arity() >= 2);
                let out = self.con2(h, na, nb);
                map.insert(r, out);
            }
        }
        map.get(&root).copied()
    }

    /// Puts the checker back as it was at `cp`, which has been popped, and walks
    /// its shared subterm in place.
    fn back_to(&mut self, cp: Checkpoint) -> Step {
        for (t, n) in self.trail.drain(cp.trail..).rev() {
            self.nodes[t as usize] = n;
        }
        self.frames.truncate(cp.frames);
        self.spine.truncate(cp.spine);
        for p in self.pending.drain(cp.pending..).rev() {
            for (pos, b) in p.saved.into_iter().rev() {
                if let Some(slot) = self.env.get_mut(pos) {
                    *slot = b;
                }
            }
        }
        self.env.truncate(cp.env);
        for key in self.memo_log.drain(cp.memo_log..) {
            self.memo.remove(&key);
        }
        self.level = cp.level;
        if self.checkpoints.is_empty() {
            self.trail.clear();
            self.memo_log.clear();
        }
        Step::InPlace(cp.id, cp.want)
    }

    /// Walks shared subterm `id` without the memo, checking it against `want`.
    fn descend_unshared(&mut self, id: TermId, want: Option<Want>) -> Result<Step, TypeError> {
        if let Some(w) = want {
            self.frames.push(Frame::Check { want: w, term: id });
        }
        Ok(Step::EvalShared(id))
    }

    /// Takes instance `t` of the kept type of shared subterm `id` and, for each
    /// monomorphic binding in `scope` (whose slots are for the de Bruijn
    /// `indices`), unifies the binding's type with the parameter that stands for
    /// it; returns the type of the subterm.
    fn apply_scope(
        &mut self,
        arena: &Arena,
        id: TermId,
        scope: &Scope,
        indices: &[u32],
        mut t: Ty,
    ) -> Result<Ty, TypeError> {
        let len = self.env.len();
        for (slot, &i) in scope.iter().zip(indices) {
            if *slot != Slot::Mono {
                continue;
            }
            let r = self.resolve(t);
            let Node::Con(Head::Arrow, param, rest) = self.node(r) else {
                unreachable!("a kept type has one parameter per monomorphic binding");
            };
            let bound = self.env[len - 1 - i as usize].ty;
            let ctx = "this shared subterm, which uses a variable at a type that does not \
                       fit its binder";
            self.expect(arena, bound, param, Site::new(id, ctx))?;
            t = rest;
        }
        Ok(t)
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
                if b.class.is_some() {
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
                // A let whose type has nothing to quantify binds a monomorphic
                // variable, so shared subterms that use it can be reused (see
                // [`Scope`]).
                let poly = self.generalize(t);
                self.bind_var(t, poly);
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
            Frame::Memo => {
                let Pending { key, params, saved } = self
                    .pending
                    .pop()
                    .expect("a pending shared subterm per memo frame");
                for (pos, b) in saved {
                    self.env[pos] = b;
                }
                self.level -= 1;
                let kept = self.fun(&params, t);
                self.generalize(kept);
                let t = self.instantiate(kept);
                let (id, scope) = key.clone();
                let kept = if self.checkpoints.is_empty() {
                    kept
                } else {
                    self.keep_past_checkpoints(&key, kept)
                };
                // Kept before the scope is applied: a key in the memo is one
                // whose fresh inference worked, which is how a failed use tells
                // a failed fresh inference from a binder that does not fit.
                self.memo.insert(key, kept);
                let indices = self.free_vars[&id].indices();
                let t = self.apply_scope(arena, id, &scope, &indices, t)?;
                Step::Return(t)
            }
            Frame::Tried => {
                self.drop_checkpoint();
                Step::Return(t)
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

/// Records in `memo` the [`FreeVars`] of `id` and of every node under it that
/// `memo` does not already hold. It walks with an explicit stack and visits each
/// distinct node once; a node's list is at most [`MAX_FREE_VARS`] long, so the
/// cost is linear in the number of distinct nodes.
fn free_vars(arena: &Arena, id: TermId, memo: &mut FxHashMap<TermId, FreeVars>) {
    // A node is pushed with `false` to be expanded, and again with `true` to be
    // computed once its children are done.
    let mut stack = vec![(id, false)];
    while let Some((t, children_done)) = stack.pop() {
        if memo.contains_key(&t) {
            continue;
        }
        let node = arena.get(t);
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
        // The free variables of each child, shifted out of the binders between
        // `t` and the child.
        let parts: Vec<(TermId, u32)> = match node {
            Term::Lam(b) => vec![(b, 1)],
            Term::Let(x, b) => vec![(x, 0), (b, 1)],
            ref other => children(other).into_iter().map(|c| (c, 0)).collect(),
        };
        let fv = match node {
            Term::Var(i) => FreeVars {
                depth: i.saturating_add(1),
                vars: Some(Box::new([i])),
            },
            _ => {
                let mut depth = 0u32;
                let mut vars: Option<Vec<u32>> = Some(Vec::new());
                for (c, shift) in parts {
                    let child = &memo[&c];
                    depth = depth.max(child.depth.saturating_sub(shift));
                    match (&mut vars, &child.vars) {
                        (Some(acc), Some(cv)) => {
                            acc.extend(cv.iter().filter(|&&i| i >= shift).map(|&i| i - shift));
                        }
                        _ => vars = None,
                    }
                }
                let vars = vars.and_then(|mut v| {
                    v.sort_unstable();
                    v.dedup();
                    (v.len() <= MAX_FREE_VARS).then(|| v.into_boxed_slice())
                });
                FreeVars { depth, vars }
            }
        };
        memo.insert(t, fv);
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
