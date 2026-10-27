//! Hindley-Milner type inference with let-polymorphism (spec §2).
//!
//! The algorithm is Algorithm J over a union-find store of type nodes, with
//! level-based generalisation (Rémy): every unbound variable records the `let` depth
//! at which it was created, and a `let` generalises exactly the variables of its
//! bound expression whose level is deeper than the `let` itself. So generalisation
//! never scans the environment, and unification is near-linear.
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

use rustc_hash::FxHashMap;

use super::{Scheme, TyVar, Type, TypeError, TypeErrorKind};
use crate::prim::Prim;
use crate::sexpr;
use crate::term::{Arena, LibId, Term, TermId};

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

#[derive(Clone, Copy, Debug)]
struct Binding {
    ty: Ty,
    /// Whether the type may contain generalised variables (a `let` binding).
    poly: bool,
}

/// A reusable type checker. Reusing one across many terms keeps its allocations.
#[derive(Debug, Default)]
pub struct TypeChecker {
    nodes: Vec<Node>,
    level: u32,
    env: Vec<Binding>,
    inst: FxHashMap<Ty, Ty>,
}

/// Where a unification happens, for the error message.
struct Site {
    term: TermId,
    context: String,
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
        let t = self.infer_term(arena, libs, term)?;
        Ok(self.export_scheme(t))
    }

    /// Checks that `term` has type `expected`.
    ///
    /// The variables of `expected` are rigid: the term must work for every choice of
    /// them, so `(lam (add $0 1))` does not check against `a -> a`. A variable may be
    /// used as a `memorec` key only if `expected` gives it a `Key` constraint.
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
        let found = self.infer_term(arena, libs, term)?;
        self.unify(want, found).map_err(|e| {
            let site = Site {
                term,
                context: "the whole term, checked against its expected type".into(),
            };
            self.error(arena, e, want, found, site)
        })
    }

    fn reset(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::Con(Head::Int, 0, 0));
        self.nodes.push(Node::Con(Head::Bool, 0, 0));
        self.nodes.push(Node::Con(Head::Unit, 0, 0));
        self.level = 1;
        self.env.clear();
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

    // ------------------------------------------------------------ unification

    /// Unifies `expected` with `found`. On failure, the store keeps the partial
    /// unification; inference stops at the first error, so that is harmless.
    fn unify(&mut self, expected: Ty, found: Ty) -> Result<(), UnifyError> {
        let a = self.resolve(expected);
        let b = self.resolve(found);
        if a == b {
            return Ok(());
        }
        match (self.node(a), self.node(b)) {
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
            (Node::Con(ha, a1, a2), Node::Con(hb, b1, b2)) if ha == hb => match ha.arity() {
                0 => Ok(()),
                1 => self.unify(a1, b1),
                _ => {
                    self.unify(a1, b1)?;
                    self.unify(a2, b2)
                }
            },
            _ => Err(UnifyError::Clash(a, b)),
        }
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
        let r = self.resolve(t);
        match self.node(r) {
            Node::Var { level: l, key } => {
                if r == v {
                    return Err(UnifyError::Occurs(v, t));
                }
                if l > level {
                    self.set(r, Node::Var { level, key });
                }
                Ok(())
            }
            Node::Rigid { .. } | Node::Link(_) => Ok(()),
            Node::Con(h, a, b) => {
                let n = h.arity();
                if n >= 1 {
                    self.occurs_adjust(v, level, a)
                        .map_err(|_| UnifyError::Occurs(v, t))?;
                }
                if n >= 2 {
                    self.occurs_adjust(v, level, b)
                        .map_err(|_| UnifyError::Occurs(v, t))?;
                }
                Ok(())
            }
        }
    }

    /// Requires `t` to be a key type, flagging variables inside it.
    fn require_key(&mut self, t: Ty) -> Result<(), UnifyError> {
        let r = self.resolve(t);
        match self.node(r) {
            Node::Var { level, .. } => {
                self.set(r, Node::Var { level, key: true });
                Ok(())
            }
            Node::Rigid { key: true } => Ok(()),
            Node::Con(Head::Int | Head::Bool | Head::Unit, _, _) => Ok(()),
            Node::Con(Head::Pair, a, b) => {
                self.require_key(a)?;
                self.require_key(b)
            }
            _ => Err(UnifyError::Key(r)),
        }
    }

    // ------------------------------------------------------------ polymorphism

    /// Quantifies the variables of `t` created deeper than the current level.
    fn generalize(&mut self, t: Ty) {
        let r = self.resolve(t);
        match self.node(r) {
            Node::Var { level, key } if level > self.level && level != GENERIC => {
                self.set(
                    r,
                    Node::Var {
                        level: GENERIC,
                        key,
                    },
                );
            }
            Node::Con(h, a, b) => {
                let n = h.arity();
                if n >= 1 {
                    self.generalize(a);
                }
                if n >= 2 {
                    self.generalize(b);
                }
            }
            _ => {}
        }
    }

    /// A copy of `t` with fresh variables for its quantified ones. Subtrees without
    /// quantified variables are shared, not copied.
    fn instantiate(&mut self, t: Ty) -> Ty {
        let mut map = std::mem::take(&mut self.inst);
        map.clear();
        let out = self.inst_rec(t, &mut map);
        self.inst = map;
        out
    }

    fn inst_rec(&mut self, t: Ty, map: &mut FxHashMap<Ty, Ty>) -> Ty {
        let r = self.resolve(t);
        match self.node(r) {
            Node::Var {
                level: GENERIC,
                key,
            } => {
                if let Some(&n) = map.get(&r) {
                    return n;
                }
                let n = self.push(Node::Var {
                    level: self.level,
                    key,
                });
                map.insert(r, n);
                n
            }
            Node::Con(h, a, b) => {
                let n = h.arity();
                let na = if n >= 1 { self.inst_rec(a, map) } else { a };
                let nb = if n >= 2 { self.inst_rec(b, map) } else { b };
                if na == a && nb == b {
                    r
                } else {
                    self.con2(h, na, nb)
                }
            }
            _ => r,
        }
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
    /// new variables are met.
    fn export(&mut self, t: Ty, names: &mut FxHashMap<Ty, TyVar>) -> Type {
        let r = self.resolve(t);
        match self.node(r) {
            Node::Var { .. } | Node::Rigid { .. } | Node::Link(_) => {
                let next = TyVar(u32::try_from(names.len()).expect("fewer than 2^32 variables"));
                Type::Var(*names.entry(r).or_insert(next))
            }
            Node::Con(h, a, b) => match h {
                Head::Int => Type::Int,
                Head::Bool => Type::Bool,
                Head::Unit => Type::Unit,
                Head::List => Type::list(self.export(a, names)),
                Head::Option => Type::option(self.export(a, names)),
                Head::Array => Type::array(self.export(a, names)),
                Head::Tree => Type::tree(self.export(a, names)),
                Head::Pair => {
                    let a = self.export(a, names);
                    Type::pair(a, self.export(b, names))
                }
                Head::Arrow => {
                    let a = self.export(a, names);
                    Type::arrow(a, self.export(b, names))
                }
            },
        }
    }

    fn export_scheme(&mut self, t: Ty) -> Scheme {
        let mut names = FxHashMap::default();
        let ty = self.export(t, &mut names);
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
        Scheme::with_keys(ty, &keys)
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

    fn infer_term<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        id: TermId,
    ) -> Result<Ty, TypeError> {
        match arena.get(id) {
            Term::Var(i) => {
                let depth = self.env.len();
                if i as usize >= depth {
                    return Err(self.plain_error(
                        arena,
                        id,
                        TypeErrorKind::UnboundVariable {
                            index: i,
                            in_scope: u32::try_from(depth).unwrap_or(u32::MAX),
                        },
                    ));
                }
                let b = self.env[depth - 1 - i as usize];
                Ok(if b.poly { self.instantiate(b.ty) } else { b.ty })
            }
            Term::Lam(body) => {
                let a = self.fresh();
                self.env.push(Binding { ty: a, poly: false });
                let r = self.infer_term(arena, libs, body)?;
                self.env.pop();
                Ok(self.arrow(a, r))
            }
            Term::Let(bound, body) => {
                self.level += 1;
                let t = self.infer_term(arena, libs, bound)?;
                self.level -= 1;
                self.generalize(t);
                self.env.push(Binding { ty: t, poly: true });
                let r = self.infer_term(arena, libs, body)?;
                self.env.pop();
                Ok(r)
            }
            Term::Int(_) => Ok(INT),
            Term::Bool(_) => Ok(BOOL),
            Term::Unit => Ok(UNIT),
            Term::Prim(p) => Ok(self.prim_type(p)),
            Term::Lib(l) => match libs.lib_scheme(l) {
                Some(s) => Ok(self.instantiate_scheme(s)),
                None => Err(self.plain_error(arena, id, TypeErrorKind::UnknownLib(l))),
            },
            Term::If(c, t, e) => {
                self.expect_bool(arena, libs, c, "the condition of `if`")?;
                let tt = self.infer_term(arena, libs, t)?;
                let te = self.infer_term(arena, libs, e)?;
                self.unify(tt, te).map_err(|err| {
                    let site = Site {
                        term: e,
                        context: "the else branch of `if` (it must have the same type as \
                                  the then branch)"
                            .into(),
                    };
                    self.error(arena, err, tt, te, site)
                })?;
                Ok(tt)
            }
            Term::And(x, y) | Term::Or(x, y) => {
                let op = if matches!(arena.get(id), Term::And(..)) {
                    "and"
                } else {
                    "or"
                };
                self.expect_bool(arena, libs, x, &format!("the first operand of `{op}`"))?;
                self.expect_bool(arena, libs, y, &format!("the second operand of `{op}`"))?;
                Ok(BOOL)
            }
            Term::App(..) => self.infer_app(arena, libs, id),
        }
    }

    fn expect_bool<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        id: TermId,
        context: &str,
    ) -> Result<(), TypeError> {
        let t = self.infer_term(arena, libs, id)?;
        self.unify(BOOL, t).map_err(|e| {
            let site = Site {
                term: id,
                context: context.into(),
            };
            self.error(arena, e, BOOL, t, site)
        })
    }

    /// Infers an application spine `head a1 ... an` argument by argument, so errors
    /// can say "argument i of `head`".
    fn infer_app<L: LibEnv + ?Sized>(
        &mut self,
        arena: &Arena,
        libs: &L,
        id: TermId,
    ) -> Result<Ty, TypeError> {
        let mut args = vec![];
        let mut head = id;
        while let Term::App(f, a) = arena.get(head) {
            args.push(a);
            head = f;
        }
        args.reverse();
        let mut tf = self.infer_term(arena, libs, head)?;
        for (i, &arg) in args.iter().enumerate() {
            let rf = self.resolve(tf);
            let (param, ret) = match self.node(rf) {
                Node::Con(Head::Arrow, p, r) => (p, r),
                Node::Var { .. } => {
                    let (p, r) = (self.fresh(), self.fresh());
                    let want = self.arrow(p, r);
                    self.unify(rf, want).map_err(|e| {
                        let site = Site {
                            term: head,
                            context: format!("the function position of `{}`", short(arena, id)),
                        };
                        self.error(arena, e, want, rf, site)
                    })?;
                    (p, r)
                }
                _ => {
                    let mut names = FxHashMap::default();
                    let found = self.export(rf, &mut names);
                    return Err(self.plain_error(
                        arena,
                        id,
                        TypeErrorKind::NotAFunction {
                            found,
                            applied_to: args.len(),
                            accepted: i,
                        },
                    ));
                }
            };
            let ta = self.infer_term(arena, libs, arg)?;
            self.unify(param, ta).map_err(|e| {
                let site = Site {
                    term: arg,
                    context: format!("argument {} of `{}`", i + 1, short(arena, head)),
                };
                self.error(arena, e, param, ta, site)
            })?;
            tf = ret;
        }
        Ok(tf)
    }

    // ------------------------------------------------------------ errors

    fn plain_error(&self, arena: &Arena, term: TermId, kind: TypeErrorKind) -> TypeError {
        TypeError {
            kind: Box::new(kind),
            term,
            term_text: short(arena, term),
            context: String::new(),
            note: None,
        }
    }

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
        let mut names = FxHashMap::default();
        let mut note = None;
        let kind = match err {
            UnifyError::Clash(x, y) => {
                let exp = self.export(expected, &mut names);
                let fnd = self.export(found, &mut names);
                let cx = self.export(x, &mut names);
                let cy = self.export(y, &mut names);
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
                let var = self.export(v, &mut names);
                let ty = self.export(t, &mut names);
                TypeErrorKind::InfiniteType { var, ty }
            }
            UnifyError::Key(t) => {
                let found = self.export(t, &mut names);
                if self.is_rigid(t) {
                    note = Some(format!(
                        "the expected type must declare `Key {found} =>` to use `{found}` \
                         as a memorec key"
                    ));
                }
                TypeErrorKind::BadMemoKey { found }
            }
        };
        TypeError {
            kind: Box::new(kind),
            term: site.term,
            term_text: short(arena, site.term),
            context: site.context,
            note,
        }
    }

    fn is_rigid(&mut self, t: Ty) -> bool {
        let r = self.resolve(t);
        matches!(self.node(r), Node::Rigid { .. })
    }
}

/// `id` in core syntax, cut to [`MAX_TERM_TEXT`] bytes.
fn short(arena: &Arena, id: TermId) -> String {
    let mut s = sexpr::print(arena, id);
    if s.len() > MAX_TERM_TEXT {
        let mut cut = MAX_TERM_TEXT;
        while !s.is_char_boundary(cut) {
            cut -= 1;
        }
        s.truncate(cut);
        s.push_str(" ...");
    }
    s
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
}
