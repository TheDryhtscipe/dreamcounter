//! Expansion of library abstractions (spec §8, decision D1).
//!
//! A library abstraction is stored in the closed form `λ^k. body` that
//! `dc_stitch::decode_abstraction` produces: the parameter `#i`, found under `d`
//! binders of `body`, is the de Bruijn variable `$(d + k - 1 - i)`. Abstractions have
//! macro semantics: a use `L a0 ... a(k-1)` means exactly `body[#i := ai]`, each
//! argument shifted by the number of body binders it is placed under. Expansion
//! replaces every library reference with that term before evaluation, so a
//! refactored program evaluates to the same value in the same number of steps as the
//! program it was refactored from (spec §7, invariant 2).
//!
//! - **Unsaturated uses**, with `n < k` arguments (bare `L` included), are
//!   η-expanded to `λy(n) ... λy(k-1). L a0 ... a(n-1) y(n) ... y(k-1)` and then
//!   expanded. Applying the result costs one β step per η binder.
//! - **Over-saturated uses** `L a0 ... a(k-1) b ...` expand to the saturated
//!   expansion applied to the expanded `b ...`.
//! - A library body may use libraries with smaller ids, saturated or not. A use of
//!   itself or of a later library is an error, which rules out cycles.
//!
//! # Sharing and cost
//!
//! The arena is hash-consed, so the expanded term is a DAG: a parameter used twice
//! refers to one node. Expansion works top down. Each task is "expand node `t` with
//! its free variables renamed by `ρ`", where `ρ` is an order-preserving renaming
//! built from the binders the node has been moved under, or "instantiate body node
//! `b` at body depth `d` for one use site". Tasks are memoised, and a renaming is
//! cut down to the variables the node actually has free, so a closed subterm is
//! expanded once wherever it occurs. An argument substituted under binders is
//! expanded directly at its final position instead of being expanded and then
//! shifted.
//!
//! A renaming is a hash-consed list of the indices where its shift grows, each
//! stored relative to the one before. Moving it under a binder or under a use site
//! builds one list cell, so the renamings for a chain of nested uses with open
//! arguments, binders in the arguments included, take memory linear in its length.
//! Renaming a variable `$j` walks the breakpoints at or below `j`, of which there are
//! at most `j + 1`. Cutting a renaming down costs one step per breakpoint kept and is
//! memoised for each renaming and depth. Time is therefore linear in the input times
//! the number of breakpoints a variable or a cut walks: a variable deep inside many
//! nested open uses that refers to all their binders makes it quadratic.
//!
//! An application is expanded one argument at a time, except that a library use
//! takes its arguments up to the library's arity together, so prefixes of a long
//! application spine are shared. Every traversal uses an explicit stack, so the
//! depth of a term is limited only by memory.

use std::fmt;

use rustc_hash::FxHashMap;

use crate::term::{children, Arena, LibId, Term, TermId};

/// A library abstraction: `term` is the closed term `λ^arity. body`, in which the
/// parameter `#i` under `d` binders of `body` is `$(d + arity - 1 - i)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LibDef {
    /// The number of parameters.
    pub arity: u32,
    /// The closed term `λ^arity. body`.
    pub term: TermId,
}

/// Why a library or a program could not be expanded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpandError {
    /// A reference to a library that is not defined.
    UnknownLib(LibId),
    /// A library body refers to itself or to a later library.
    ForwardReference {
        /// The library whose body holds the reference.
        lib: LibId,
        /// The library it refers to.
        refers_to: LibId,
    },
    /// A library term has free variables.
    NotClosed {
        /// The library.
        lib: LibId,
        /// One more than its largest free de Bruijn index.
        free_depth: u32,
    },
    /// A library term has fewer leading lambdas than its arity.
    TooFewLambdas {
        /// The library.
        lib: LibId,
        /// Its declared arity.
        arity: u32,
        /// The number of leading lambdas its term has.
        lambdas: u32,
    },
}

impl fmt::Display for ExpandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ExpandError::UnknownLib(LibId(l)) => write!(f, "unknown library lib_{l}"),
            ExpandError::ForwardReference {
                lib: LibId(l),
                refers_to: LibId(r),
            } if l == r => write!(f, "library lib_{l} refers to itself"),
            ExpandError::ForwardReference {
                lib: LibId(l),
                refers_to: LibId(r),
            } => write!(f, "library lib_{l} refers to the later library lib_{r}"),
            ExpandError::NotClosed {
                lib: LibId(l),
                free_depth,
            } => write!(
                f,
                "library lib_{l} is not closed: it needs {free_depth} enclosing binders"
            ),
            ExpandError::TooFewLambdas {
                lib: LibId(l),
                arity,
                lambdas,
            } => write!(
                f,
                "library lib_{l} has arity {arity} but only {lambdas} leading lambdas"
            ),
        }
    }
}

impl std::error::Error for ExpandError {}

/// Expands every library reference in `root`, using `libs` indexed by [`LibId`].
///
/// To expand many programs with the same libraries, build one [`Expander`] and
/// reuse it: it checks and expands the library bodies once.
pub fn expand(arena: &mut Arena, libs: &[LibDef], root: TermId) -> Result<TermId, ExpandError> {
    Expander::new(arena, libs)?.expand(arena, root)
}

/// Checked libraries, ready to expand programs.
///
/// An expander belongs to the arena it was built with: its library bodies and memo
/// tables hold ids in that arena. Use it only with that arena.
///
/// Between programs it keeps only the expanded library bodies and two tables with
/// one entry per arena node, free depths and application heads, which grow as the
/// arena does. The tables
/// for one program are dropped when its expansion ends, so they are not shared
/// between programs.
#[derive(Debug)]
pub struct Expander {
    /// For each library: its arity and its body with every library use expanded.
    /// The parameters are free in the body: `#i` at depth 0 is `$(arity - 1 - i)`.
    bodies: Vec<(u32, TermId)>,
    /// The number of libraries, including those still being prepared.
    total: usize,
    memo: FxHashMap<Task, TermId>,
    sites: Vec<Site>,
    renamings: Renamings,
    free: FreeDepth,
    heads: SpineHeads,
    /// The number of subtasks planned, a measure of the work done.
    #[cfg(test)]
    planned: usize,
}

impl Expander {
    /// Checks `libs` and expands each library body. Library `i` may use libraries
    /// `0 .. i` only.
    pub fn new(arena: &mut Arena, libs: &[LibDef]) -> Result<Expander, ExpandError> {
        let mut ex = Expander {
            bodies: Vec::with_capacity(libs.len()),
            total: libs.len(),
            memo: FxHashMap::default(),
            sites: Vec::new(),
            renamings: Renamings::new(),
            free: FreeDepth::default(),
            heads: SpineHeads::default(),
            #[cfg(test)]
            planned: 0,
        };
        for (i, def) in libs.iter().enumerate() {
            let lib = LibId(u32::try_from(i).expect("more than 2^32 libraries"));
            let free_depth = ex.free.depth(arena, def.term);
            if free_depth != 0 {
                return Err(ExpandError::NotClosed { lib, free_depth });
            }
            let lambdas = leading_lambdas(arena, def.term, def.arity);
            if lambdas < def.arity {
                return Err(ExpandError::TooFewLambdas {
                    lib,
                    arity: def.arity,
                    lambdas,
                });
            }
            // Library uses inside the body are checked against `bodies.len() == i`.
            let expanded = ex.run_once(arena, Task::Expand(def.term, IDENTITY), Some(lib))?;
            let mut body = expanded;
            for _ in 0..def.arity {
                match arena.get(body) {
                    Term::Lam(b) => body = b,
                    other => unreachable!("expansion removed a leading lambda: {other:?}"),
                }
            }
            ex.bodies.push((def.arity, body));
        }
        Ok(ex)
    }

    /// Expands every library reference in `root`. The result contains no
    /// [`Term::Lib`] node.
    pub fn expand(&mut self, arena: &mut Arena, root: TermId) -> Result<TermId, ExpandError> {
        self.run_once(arena, Task::Expand(root, IDENTITY), None)
    }

    /// [`Expander::run`], then drops the tables built for it, whether or not it
    /// succeeded.
    fn run_once(
        &mut self,
        arena: &mut Arena,
        root: Task,
        preparing: Option<LibId>,
    ) -> Result<TermId, ExpandError> {
        let out = self.run(arena, root, preparing);
        self.memo.clear();
        self.sites.clear();
        self.renamings.clear();
        out
    }

    /// Runs `root` and everything it depends on with an explicit stack. `preparing`
    /// is the library whose body is being expanded, if any; only libraries before
    /// it may be used.
    fn run(
        &mut self,
        arena: &mut Arena,
        root: Task,
        preparing: Option<LibId>,
    ) -> Result<TermId, ExpandError> {
        let mut stack = vec![Frame::Visit(root)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Visit(task) => {
                    if self.memo.contains_key(&task) {
                        continue;
                    }
                    match self.plan(arena, task, preparing)? {
                        Plan::Done(t) => {
                            self.memo.insert(task, t);
                        }
                        Plan::Build(shape, kids) => {
                            #[cfg(test)]
                            {
                                self.planned += kids.len();
                            }
                            let pending: Vec<Task> = kids
                                .iter()
                                .copied()
                                .filter(|k| !self.memo.contains_key(k))
                                .collect();
                            stack.push(Frame::Build(task, shape, kids));
                            stack.extend(pending.into_iter().rev().map(Frame::Visit));
                        }
                    }
                }
                Frame::Build(task, shape, kids) => {
                    if self.memo.contains_key(&task) {
                        continue;
                    }
                    let results: Vec<TermId> = kids.iter().map(|k| self.memo[k]).collect();
                    let t = build(arena, shape, &results);
                    self.memo.insert(task, t);
                }
            }
        }
        Ok(self.memo[&root])
    }

    /// What `task` needs: either its result now, or a list of subtasks and how to
    /// combine their results.
    fn plan(
        &mut self,
        arena: &mut Arena,
        task: Task,
        preparing: Option<LibId>,
    ) -> Result<Plan, ExpandError> {
        match task {
            Task::Expand(t, r) => self.plan_expand(arena, t, r, preparing),
            Task::Inst(b, d, site) => Ok(self.plan_inst(arena, b, d, site)),
        }
    }

    /// The task "expand `t`, renaming its free variables by `r`", with `r` cut down
    /// to the variables `t` has free so that equal tasks share one memo entry.
    fn expand_task(&mut self, arena: &Arena, t: TermId, r: RenId) -> Task {
        let fd = self.free.depth(arena, t);
        Task::Expand(t, self.renamings.restrict(r, fd))
    }

    fn plan_expand(
        &mut self,
        arena: &mut Arena,
        t: TermId,
        r: RenId,
        preparing: Option<LibId>,
    ) -> Result<Plan, ExpandError> {
        let node = arena.get(t);
        Ok(match node {
            Term::Var(j) => Plan::Done(arena.var(self.renamings.apply(r, j))),
            Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) => Plan::Done(t),
            Term::Lam(b) => {
                let lifted = self.renamings.lift(r);
                Plan::Build(
                    Shape::Rebuild(node),
                    vec![self.expand_task(arena, b, lifted)],
                )
            }
            Term::Let(x, b) => {
                let lifted = self.renamings.lift(r);
                let kids = vec![
                    self.expand_task(arena, x, r),
                    self.expand_task(arena, b, lifted),
                ];
                Plan::Build(Shape::Rebuild(node), kids)
            }
            Term::If(..) | Term::And(..) | Term::Or(..) => {
                let kids = children(&node)
                    .into_iter()
                    .map(|c| self.expand_task(arena, c, r))
                    .collect();
                Plan::Build(Shape::Rebuild(node), kids)
            }
            Term::App(..) | Term::Lib(_) => {
                // A library use takes the spine only up to the library's arity: an
                // over-saturated use is the saturated one applied to the rest, one
                // argument at a time. So no spine walked is longer than an arity, and
                // each prefix of a long spine is its own memoised task.
                let (head, n) = self.heads.get(arena, t);
                let lib = match arena.get(head) {
                    Term::Lib(l) => self
                        .bodies
                        .get(l.0 as usize)
                        .is_none_or(|&(arity, _)| n <= arity)
                        .then_some(l),
                    _ => None,
                };
                match (lib, node) {
                    (Some(l), _) => {
                        let (_, args) = spine(arena, t);
                        self.plan_use(arena, l, &args, r, preparing)?
                    }
                    (None, Term::App(f, x)) => Plan::Build(
                        Shape::Rebuild(node),
                        vec![self.expand_task(arena, f, r), self.expand_task(arena, x, r)],
                    ),
                    (None, other) => unreachable!("a library is its own head: {other:?}"),
                }
            }
        })
    }

    /// A use `l args` whose context renames free variables by `r`.
    fn plan_use(
        &mut self,
        arena: &Arena,
        l: LibId,
        args: &[TermId],
        r: RenId,
        preparing: Option<LibId>,
    ) -> Result<Plan, ExpandError> {
        let idx = l.0 as usize;
        if idx >= self.bodies.len() {
            return Err(match preparing {
                Some(lib) if idx < self.total => {
                    ExpandError::ForwardReference { lib, refers_to: l }
                }
                _ => ExpandError::UnknownLib(l),
            });
        }
        let (arity, body) = self.bodies[idx];
        let k = arity as usize;
        let supplied = args.len().min(k);
        let eta = arity - u32::try_from(supplied).expect("arity fits u32");
        let site = SiteId(u32::try_from(self.sites.len()).expect("more than 2^32 use sites"));
        self.sites.push(Site {
            arity,
            args: args[..supplied].to_vec(),
            renaming: r,
            eta,
        });
        let mut kids = Vec::with_capacity(1 + args.len() - supplied);
        kids.push(Task::Inst(body, 0, site));
        for &a in &args[supplied..] {
            kids.push(self.expand_task(arena, a, r));
        }
        Ok(Plan::Build(Shape::Use { eta }, kids))
    }

    /// Instantiate body node `b`, at depth `d` inside the body, for use site `site`.
    fn plan_inst(&mut self, arena: &mut Arena, b: TermId, d: u32, site: SiteId) -> Plan {
        if self.free.depth(arena, b) <= d {
            // No parameter occurs in `b`, and the body is already expanded.
            return Plan::Done(b);
        }
        let node = arena.get(b);
        match node {
            Term::Var(j) => {
                // `j >= d`, so this is parameter #i.
                let s = &self.sites[site.0 as usize];
                let i = s.arity - 1 - (j - d);
                if let Some(&a) = s.args.get(i as usize) {
                    // The argument moves under the η binders and `d` body binders.
                    let (renaming, by) = (s.renaming, d + s.eta);
                    let r = self.renamings.after_shift(renaming, by);
                    Plan::Build(Shape::Same, vec![self.expand_task(arena, a, r)])
                } else {
                    // An η binder: y(i) is `$(arity - 1 - i)` at depth 0 of the body.
                    Plan::Done(arena.var(j))
                }
            }
            Term::Lam(c) => Plan::Build(Shape::Rebuild(node), vec![Task::Inst(c, d + 1, site)]),
            Term::Let(x, c) => Plan::Build(
                Shape::Rebuild(node),
                vec![Task::Inst(x, d, site), Task::Inst(c, d + 1, site)],
            ),
            _ => Plan::Build(
                Shape::Rebuild(node),
                children(&node)
                    .into_iter()
                    .map(|c| Task::Inst(c, d, site))
                    .collect(),
            ),
        }
    }
}

/// The number of leading lambdas of `t`, counting at most `max`.
fn leading_lambdas(arena: &Arena, mut t: TermId, max: u32) -> u32 {
    let mut n = 0;
    while n < max {
        match arena.get(t) {
            Term::Lam(b) => {
                n += 1;
                t = b;
            }
            _ => break,
        }
    }
    n
}

/// The head of an application spine and its arguments, left to right.
fn spine(arena: &Arena, t: TermId) -> (TermId, Vec<TermId>) {
    let mut args = vec![];
    let mut head = t;
    while let Term::App(f, a) = arena.get(head) {
        args.push(a);
        head = f;
    }
    args.reverse();
    (head, args)
}

/// Combines subtask results.
fn build(arena: &mut Arena, shape: Shape, kids: &[TermId]) -> TermId {
    match shape {
        Shape::Same => kids[0],
        Shape::Use { eta } => {
            let mut t = arena.apps(kids[0], &kids[1..]);
            for _ in 0..eta {
                t = arena.lam(t);
            }
            t
        }
        Shape::Rebuild(node) => {
            let new = match node {
                Term::Lam(_) => Term::Lam(kids[0]),
                Term::Let(..) => Term::Let(kids[0], kids[1]),
                Term::App(..) => Term::App(kids[0], kids[1]),
                Term::And(..) => Term::And(kids[0], kids[1]),
                Term::Or(..) => Term::Or(kids[0], kids[1]),
                Term::If(..) => Term::If(kids[0], kids[1], kids[2]),
                leaf => leaf,
            };
            arena.intern(new)
        }
    }
}

/// A unit of expansion work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Task {
    /// Expand a program (or library) node, renaming its free variables.
    Expand(TermId, RenId),
    /// Instantiate an expanded body node, at this depth inside the body, for a use
    /// site.
    Inst(TermId, u32, SiteId),
}

/// How to combine the results of a task's subtasks.
#[derive(Clone, Copy, Debug)]
enum Shape {
    /// The node with its children replaced by the results, in order.
    Rebuild(Term),
    /// A library use: the instantiated body applied to the extra arguments, under
    /// `eta` lambdas.
    Use { eta: u32 },
    /// The single result.
    Same,
}

#[derive(Debug)]
enum Plan {
    Done(TermId),
    Build(Shape, Vec<Task>),
}

#[derive(Debug)]
enum Frame {
    Visit(Task),
    Build(Task, Shape, Vec<Task>),
}

/// Index of a [`Site`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SiteId(u32);

/// One library use being instantiated.
#[derive(Debug)]
struct Site {
    arity: u32,
    /// The supplied arguments, at most `arity`, as unexpanded terms of the use site.
    args: Vec<TermId>,
    /// The renaming of the use site's free variables.
    renaming: RenId,
    /// The number of η binders around the instantiated body.
    eta: u32,
}

/// Index of an interned renaming. 0 is the identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RenId(u32);

const IDENTITY: RenId = RenId(0);

/// One breakpoint of a renaming and the renaming's remaining breakpoints.
///
/// The breakpoint sits `gap` indices above the previous one (above 0 for the first)
/// and raises the shift by `delta`.
#[derive(Clone, Copy, Debug)]
struct Point {
    gap: u32,
    delta: u32,
    rest: RenId,
    /// The position of the last breakpoint, relative to the previous breakpoint's.
    last: u32,
}

/// Order-preserving renamings of de Bruijn indices, `j ↦ j + shift(j)`.
///
/// `shift` is 0 below the first breakpoint and grows by a positive amount at each
/// breakpoint. A renaming is a hash-consed list of breakpoints from the lowest
/// index up, each stored relative to the one before it. Going under a binder
/// changes only the first breakpoint's position, and shifting the result changes
/// only the first breakpoint's increase, so both build one list cell and share the
/// rest. Each distinct renaming is interned once.
#[derive(Debug)]
struct Renamings {
    /// The cells; entry 0 stands for the identity and is never read.
    points: Vec<Point>,
    index: FxHashMap<(u32, u32, RenId), RenId>,
    /// Memo of [`Renamings::restrict`] for renamings it had to cut.
    restricted: FxHashMap<(RenId, u32), RenId>,
}

impl Renamings {
    fn new() -> Renamings {
        let unused = Point {
            gap: 0,
            delta: 0,
            rest: IDENTITY,
            last: 0,
        };
        Renamings {
            points: vec![unused],
            index: FxHashMap::default(),
            restricted: FxHashMap::default(),
        }
    }

    /// Drops every renaming but the identity.
    fn clear(&mut self) {
        self.points.truncate(1);
        self.index.clear();
        self.restricted.clear();
    }

    fn get(&self, r: RenId) -> Option<Point> {
        (r != IDENTITY).then(|| self.points[r.0 as usize])
    }

    /// The renaming with a first breakpoint `gap` above 0, raising the shift by
    /// `delta`, followed by the breakpoints of `rest`.
    fn cons(&mut self, gap: u32, delta: u32, rest: RenId) -> RenId {
        debug_assert!(delta > 0, "a breakpoint must raise the shift");
        if let Some(&id) = self.index.get(&(gap, delta, rest)) {
            return id;
        }
        let last = match self.get(rest) {
            None => gap,
            Some(p) => {
                debug_assert!(p.gap > 0, "breakpoints must be distinct");
                gap.checked_add(p.last).expect("de Bruijn index overflow")
            }
        };
        let id = RenId(u32::try_from(self.points.len()).expect("more than 2^32 renamings"));
        self.points.push(Point {
            gap,
            delta,
            rest,
            last,
        });
        self.index.insert((gap, delta, rest), id);
        id
    }

    /// `r(j)`. This walks the breakpoints at or below `j`.
    fn apply(&self, r: RenId, j: u32) -> u32 {
        let (mut at, mut shift, mut r) = (0u32, 0u32, r);
        while let Some(p) = self.get(r) {
            at = at.saturating_add(p.gap);
            if at > j {
                break;
            }
            shift += p.delta;
            r = p.rest;
        }
        j.checked_add(shift).expect("de Bruijn index overflow")
    }

    /// The renaming under one more binder: `0 ↦ 0`, `j + 1 ↦ r(j) + 1`.
    fn lift(&mut self, r: RenId) -> RenId {
        match self.get(r) {
            None => r,
            Some(p) => {
                let gap = p.gap.checked_add(1).expect("de Bruijn index overflow");
                self.cons(gap, p.delta, p.rest)
            }
        }
    }

    /// `r` followed by shifting every index up by `by`.
    fn after_shift(&mut self, r: RenId, by: u32) -> RenId {
        if by == 0 {
            return r;
        }
        match self.get(r) {
            Some(p) if p.gap == 0 => {
                let delta = p.delta.checked_add(by).expect("de Bruijn index overflow");
                self.cons(0, delta, p.rest)
            }
            _ => self.cons(0, by, r),
        }
    }

    /// The number of breakpoints stored for all interned renamings.
    #[cfg(test)]
    fn stored_points(&self) -> usize {
        self.points.len() - 1
    }

    /// `r` without its breakpoints at `depth` or above. The indices from `depth` up
    /// are not free in the term it will rename, so how they map does not matter, and
    /// equal tasks get equal renamings.
    ///
    /// Cutting costs one step per breakpoint kept, and is memoised.
    fn restrict(&mut self, r: RenId, depth: u32) -> RenId {
        // The kept cells whose rest must be cut, with the depth left at each.
        let mut path: Vec<(RenId, u32, Point)> = vec![];
        let (mut cur, mut left) = (r, depth);
        let tail = loop {
            let Some(p) = self.get(cur) else {
                break cur;
            };
            if p.last < left {
                break cur;
            }
            if p.gap >= left {
                break IDENTITY;
            }
            if let Some(&done) = self.restricted.get(&(cur, left)) {
                break done;
            }
            path.push((cur, left, p));
            left -= p.gap;
            cur = p.rest;
        };
        let mut acc = tail;
        for (id, left, p) in path.into_iter().rev() {
            acc = self.cons(p.gap, p.delta, acc);
            self.restricted.insert((id, left), acc);
        }
        acc
    }
}

/// A memo of the head of each application spine and its number of arguments.
#[derive(Debug, Default)]
struct SpineHeads {
    memo: Vec<Option<(TermId, u32)>>,
}

impl SpineHeads {
    /// The head of `t`'s spine and how many arguments it is applied to. Each node's
    /// answer is computed once, so this costs amortised constant time.
    fn get(&mut self, arena: &Arena, t: TermId) -> (TermId, u32) {
        if self.memo.len() < arena.len() {
            self.memo.resize(arena.len(), None);
        }
        let mut path = vec![];
        let mut cur = t;
        let (head, mut n) = loop {
            if let Some(known) = self.memo[cur.index()] {
                break known;
            }
            match arena.get(cur) {
                Term::App(f, _) => {
                    path.push(cur);
                    cur = f;
                }
                _ => break (cur, 0),
            }
        };
        for &p in path.iter().rev() {
            n = n.checked_add(1).expect("more than 2^32 arguments");
            self.memo[p.index()] = Some((head, n));
        }
        (head, n)
    }
}

/// A memo of [`Arena::free_depth`], computed with an explicit stack.
#[derive(Debug, Default)]
struct FreeDepth {
    memo: Vec<Option<u32>>,
}

impl FreeDepth {
    fn depth(&mut self, arena: &Arena, root: TermId) -> u32 {
        if self.memo.len() < arena.len() {
            self.memo.resize(arena.len(), None);
        }
        if let Some(d) = self.memo[root.index()] {
            return d;
        }
        let mut stack = vec![(root, false)];
        while let Some((t, ready)) = stack.pop() {
            if self.memo[t.index()].is_some() {
                continue;
            }
            let node = arena.get(t);
            if !ready {
                stack.push((t, true));
                for c in children(&node) {
                    if self.memo[c.index()].is_none() {
                        stack.push((c, false));
                    }
                }
                continue;
            }
            let get = |c: TermId| self.memo[c.index()].expect("child computed first");
            let d = match node {
                Term::Var(i) => i + 1,
                Term::Lam(b) => get(b).saturating_sub(1),
                Term::Let(x, b) => get(x).max(get(b).saturating_sub(1)),
                other => children(&other).into_iter().map(get).max().unwrap_or(0),
            };
            self.memo[t.index()] = Some(d);
        }
        self.memo[root.index()].expect("root computed")
    }
}

#[cfg(test)]
mod tests;
