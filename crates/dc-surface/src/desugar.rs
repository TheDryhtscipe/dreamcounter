//! Desugaring surface syntax to core terms (spec §10), and the top-level
//! environment that turns a program into one closed core term.
//!
//! # Rules and their cost
//!
//! Evaluation is strict and left to right; every β step and primitive application
//! costs a step, while `let`, variable lookup, literals and closure creation are
//! free (spec §7). Each rule below keeps the evaluation order of the source and
//! adds no step that the source construct does not itself denote.
//!
//! | Surface | Core | Steps added by the sugar |
//! |---|---|---|
//! | `f a b` | `(f a b)` | none |
//! | `fun p q -> e` | `(lam (lam e))`, i.e. `fun p -> fun q -> e` | none |
//! | `fun (a, b) -> e` | `(lam (let (fst $0) (let (snd $1) e)))` | 2 per call (`fst`, `snd`) |
//! | `fun (a, b, c) -> e` | `(lam (let (fst $0) (let (snd $1) (let (fst $0) (let (snd $1) e)))))` | 4 per call |
//! | `_` inside a tuple binder | no projection for that component | none |
//! | `fun _ -> e`, `fun () -> e` | `(lam e)` (the argument is not checked to be unit) | none |
//! | `let x = e1 in e2` | `(let e1 e2)` | none |
//! | `let f a b = e1 in e2` | `(let (lam (lam e1)) e2)` | none |
//! | `let (a, b) = e1 in e2` | `(let e1 (let (fst $0) (let (snd $1) e2)))` | 2 |
//! | `let _ = e1 in e2` | `(let e1 e2)` (`e1` is still evaluated) | none |
//! | `match xs with [] -> e1 \| h :: t -> e2` | `(list_case xs (lam e1) (lam (lam e2)))` | none beyond `list_case` and its thunk β (spec §5) |
//! | `match o with None -> e1 \| Some x -> e2` | `(option_case o (lam e1) (lam e2))` | as above |
//! | `match t with Leaf -> e1 \| Node (l, x, r) -> e2` | `(tree_case t (lam e1) (lam (lam (lam e2))))` | as above |
//! | `_ -> e` as the second arm | lambdas for the missing case, binding nothing | as above |
//! | `match e with (a, b) -> body` (one arm) | `let (a, b) = e in body` | 2 |
//! | `a + b`, `-`, `*`, `/`, `mod` | `(add a b)`, `sub`, `mul`, `div`, `mod` | none |
//! | `a = b`, `a == b`, `a < b`, `a <= b` | `(int_eq a b)`, `eq`, `lt`, `le` | none |
//! | `a > b` | `(let a (lt b' $0))`, `b'` shifted under the let | none (the `let` keeps `a` first) |
//! | `a > b`, when `a` or `b` is a variable or literal | `(lt b a)` | none (order of a free, infallible operand is unobservable) |
//! | `a >= b` | as `>`, with `le` | none |
//! | `a && b`, `a \|\| b` | `(and a b)`, `(or a b)` | none |
//! | `if c then t else e` | `(if c t e)` | none |
//! | `-5` | the literal `-5` | none |
//! | `-e` (not a literal) | `(sub 0 e)` | 1 |
//! | `h :: t` | `(cons h t)` | none |
//! | `[]`, `[a; b]` | `nil`, `(cons a (cons b nil))` | none |
//! | `()`, `(a, b)`, `(a, b, c)` | `unit`, `(pair a b)`, `(pair a (pair b c))` | none |
//! | `None`, `Some e`, `Leaf` | `none`, `(some e)`, `leaf` | none |
//! | `Node (l, x, r)` | `(node l x r)`: the tuple is not built | none |
//! | `Node`, `Some` alone | the curried primitive (`Node l x r` also works) | none |
//! | `( + )`, `( mod )`, `( :: )`, ... | the primitive (`add`, `mod`, `cons`, ...) | none |
//!
//! `( > )`, `( >= )`, `( && )` and `( || )` are rejected rather than wrapped in a
//! lambda, which would add two hidden β steps.
//!
//! # Names
//!
//! A name resolves to, in order: the nearest local binder; the latest top-level
//! definition or inline name before the current definition; a primitive by its
//! canonical name ([`Prim::from_name`]). `not`, `fst`, `foldl`, `bfix` and the rest
//! are primitives this way.
//!
//! # Recursion
//!
//! There is no recursion by name (spec §6). A top-level definition, or a local
//! `let` whose right side is a function (`let f x = ...` or `let f = fun ...`),
//! may not mention its own name, even if an earlier definition has that name; the
//! error tells the writer to use `bfix` or `memorec`. A local value `let x = x + 1`
//! refers to the outer `x`, as in OCaml.
//!
//! # Patterns
//!
//! Patterns are one level deep. A binder (a `fun` parameter, a `let` pattern, or a
//! constructor field) is a name, `_`, `()`, or a flat tuple of names and `_`. A match
//! arm is one constructor of a list, option or tree with binders as fields, or `_`.
//! Anything deeper is an [`ErrorKind::NestedPattern`] error suggesting a second
//! match or let.

use std::collections::BTreeSet;
use std::rc::Rc;

use dc_core::{Arena, Prim, TermId};

use crate::ast::{Arm, BinOp, Ctor, Def, Expr, ExprKind, LetLhs, Pat, PatKind};
use crate::error::{ErrorKind, Source, Span, SurfaceError};
use crate::parser;

type DResult<T> = Result<T, SurfaceError>;

/// A top-level name.
#[derive(Clone, Debug)]
pub(crate) struct Global {
    pub name: String,
    pub kind: GlobalKind,
}

#[derive(Clone, Debug)]
pub(crate) enum GlobalKind {
    /// A definition from source, compiled on demand.
    Def {
        def: Rc<Def>,
        source: Rc<Source>,
        /// Indices of the globals its body refers to.
        deps: Vec<usize>,
    },
    /// A closed core term substituted at each use.
    Inline(TermId),
}

/// An entry in the desugarer's scope.
#[derive(Clone, Debug)]
enum Binder {
    /// A named local binder.
    Named(String),
    /// A binder the source cannot name (a tuple binder's argument, `_`, a thunk's
    /// unit argument).
    Hidden(u32),
    /// A top-level definition, bound by a `let` around the entry.
    Global(usize),
    /// An inline top-level name: occupies no de Bruijn slot.
    Inline(usize),
}

impl Binder {
    fn occupies_slot(&self) -> bool {
        !matches!(self, Binder::Inline(_))
    }
}

/// One component of a tuple binder.
#[derive(Clone, Debug)]
enum Comp {
    Named(String),
    Wild,
}

/// The shape of a binder after validation.
#[derive(Clone, Debug)]
enum Shape {
    Named(String),
    Wild,
    Tuple(Vec<Comp>),
}

/// Where a binder appears, for error messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinderSite {
    Param,
    Let,
    Field,
}

/// The families of a match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    List,
    Option,
    Tree,
}

impl Family {
    fn describe(self) -> &'static str {
        match self {
            Family::List => "a list",
            Family::Option => "an option",
            Family::Tree => "a tree",
        }
    }
    fn empty_pat(self) -> &'static str {
        match self {
            Family::List => "[]",
            Family::Option => "None",
            Family::Tree => "Leaf",
        }
    }
    fn full_pat(self) -> &'static str {
        match self {
            Family::List => "h :: t",
            Family::Option => "Some x",
            Family::Tree => "Node (l, x, r)",
        }
    }
    fn arity(self) -> usize {
        match self {
            Family::List => 2,
            Family::Option => 1,
            Family::Tree => 3,
        }
    }
    fn prim(self) -> Prim {
        match self {
            Family::List => Prim::ListCase,
            Family::Option => Prim::OptionCase,
            Family::Tree => Prim::TreeCase,
        }
    }
}

/// How one arm's pattern was classified.
enum ArmCase<'p> {
    Empty(Family),
    Full(Family, Vec<&'p Pat>),
    CatchAll,
}

pub(crate) struct Ctx<'a> {
    arena: &'a mut Arena,
    globals: &'a [Global],
    scope: Vec<Binder>,
    /// Function definitions being compiled: (name, scope length at their start).
    defining: Vec<(String, usize)>,
    /// Globals referred to.
    pub deps: BTreeSet<usize>,
    src: Rc<Source>,
    next_hidden: u32,
    /// In a validation pass, inline names compile to a placeholder: the arena is a
    /// scratch arena that does not hold their terms.
    validating: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(
        arena: &'a mut Arena,
        globals: &'a [Global],
        src: Rc<Source>,
        validating: bool,
    ) -> Ctx<'a> {
        Ctx {
            arena,
            globals,
            scope: vec![],
            defining: vec![],
            deps: BTreeSet::new(),
            src,
            next_hidden: 0,
            validating,
        }
    }

    /// Brings globals into scope, in order.
    pub fn push_global(&mut self, g: usize) {
        let b = match self.globals[g].kind {
            GlobalKind::Def { .. } => Binder::Global(g),
            GlobalKind::Inline(_) => Binder::Inline(g),
        };
        self.scope.push(b);
    }

    pub fn set_source(&mut self, src: Rc<Source>) {
        self.src = src;
    }

    fn err(&self, kind: ErrorKind, span: Span, msg: impl Into<String>) -> SurfaceError {
        self.src.error(kind, span, msg)
    }

    fn push_hidden(&mut self) -> u32 {
        let h = self.next_hidden;
        self.next_hidden += 1;
        self.scope.push(Binder::Hidden(h));
        h
    }

    fn hidden_var(&mut self, h: u32) -> TermId {
        let mut slot = 0u32;
        for b in self.scope.iter().rev() {
            if let Binder::Hidden(x) = b {
                if *x == h {
                    return self.arena.var(slot);
                }
            }
            if b.occupies_slot() {
                slot += 1;
            }
        }
        unreachable!("hidden binder {h} is not in scope")
    }

    /// Compiles a top-level definition's right-hand side.
    pub fn definition(&mut self, def: &Def) -> DResult<TermId> {
        self.defining.push((def.name.clone(), self.scope.len()));
        let t = self.lambda(&def.params, &def.body, BinderSite::Param);
        self.defining.pop();
        t
    }

    // ---- names ----

    fn self_reference(&self, name: &str, span: Span) -> SurfaceError {
        self.err(
            ErrorKind::SelfReference,
            span,
            format!(
                "'{name}' refers to itself, but dc has no recursion by name (no 'let rec'). \
                 Use foldl, foldr or loop for structural recursion, or write \
                 'bfix FUEL (fun self x -> ... self y ...) x' (memorec for memoised \
                 recursion), where FUEL bounds the recursion depth, for example \
                 'length xs + 1'. If you meant an earlier definition also called \
                 '{name}', rename this one."
            ),
        )
    }

    /// Whether `name` is being defined by a function definition that encloses
    /// scope position `pos` from outside (so a use resolving there is recursion).
    fn is_self(&self, name: &str, pos: Option<usize>) -> bool {
        self.defining
            .iter()
            .any(|(n, start)| n == name && pos.is_none_or(|p| p < *start))
    }

    fn lookup(&mut self, name: &str, span: Span) -> DResult<TermId> {
        let mut slot = 0u32;
        for pos in (0..self.scope.len()).rev() {
            let b = &self.scope[pos];
            let hit = match b {
                Binder::Named(n) => n == name,
                Binder::Global(g) | Binder::Inline(g) => self.globals[*g].name == name,
                Binder::Hidden(_) => false,
            };
            if hit {
                if self.is_self(name, Some(pos)) {
                    return Err(self.self_reference(name, span));
                }
                return Ok(match *b {
                    Binder::Global(g) => {
                        self.deps.insert(g);
                        self.arena.var(slot)
                    }
                    Binder::Inline(g) => {
                        self.deps.insert(g);
                        match self.globals[g].kind {
                            GlobalKind::Inline(t) if !self.validating => t,
                            _ => self.arena.unit(),
                        }
                    }
                    _ => self.arena.var(slot),
                });
            }
            if b.occupies_slot() {
                slot += 1;
            }
        }
        if self.is_self(name, None) {
            return Err(self.self_reference(name, span));
        }
        if let Some(p) = Prim::from_name(name) {
            return Ok(self.arena.prim(p));
        }
        Err(self.unknown_name(name, span))
    }

    fn unknown_name(&self, name: &str, span: Span) -> SurfaceError {
        let hint = match name {
            "fold_left" | "fold" | "foldLeft" | "reduce" => Some("foldl"),
            "fold_right" | "foldRight" => Some("foldr"),
            "fix" | "rec" | "recurse" => Some("bfix or memorec"),
            "hd" | "head" | "tl" | "tail" => Some("match xs with [] -> ... | h :: t -> ..."),
            "len" | "size" | "List" => Some("length"),
            "abs" => Some("if x < 0 then 0 - x else x"),
            _ => None,
        };
        let mut msg = format!("unknown name '{name}'");
        if let Some(h) = hint {
            msg.push_str(&format!("; in dc use {h}"));
            return self.err(ErrorKind::UnknownName, span, msg);
        }
        let mut candidates: Vec<(usize, String)> = vec![];
        let mut consider = |cand: &str| {
            let d = edit_distance(name, cand);
            let limit = if name.chars().count() <= 3 { 1 } else { 2 };
            if d <= limit && !candidates.iter().any(|(_, c)| c == cand) {
                candidates.push((d, cand.to_owned()));
            }
        };
        for b in &self.scope {
            match b {
                Binder::Named(n) => consider(n),
                Binder::Global(g) | Binder::Inline(g) => consider(&self.globals[*g].name),
                Binder::Hidden(_) => {}
            }
        }
        for p in Prim::ALL {
            consider(p.name());
        }
        candidates.sort();
        if !candidates.is_empty() {
            let list: Vec<String> = candidates
                .iter()
                .take(3)
                .map(|(_, c)| format!("'{c}'"))
                .collect();
            msg.push_str(&format!("; did you mean {}?", list.join(" or ")));
        }
        self.err(ErrorKind::UnknownName, span, msg)
    }

    // ---- binders ----

    /// A tuple binder with a non-name component.
    fn nested_binder_error(&self, inner: &Pat) -> SurfaceError {
        let suggestion = "Bind the inner part to a name and take it apart with a second \
                          let: for example 'fun (ab, c) -> let (a, b) = ab in ...' instead \
                          of 'fun ((a, b), c) -> ...'";
        self.err(
            ErrorKind::NestedPattern,
            inner.span,
            format!(
                "patterns are one level deep in dc, and this tuple pattern contains \
                 another pattern. {suggestion}"
            ),
        )
    }

    /// Validates a binder pattern.
    fn shape(&self, p: &Pat, site: BinderSite) -> DResult<Shape> {
        match &p.kind {
            PatKind::Var(n) => Ok(Shape::Named(n.clone())),
            PatKind::Wild | PatKind::Unit => Ok(Shape::Wild),
            PatKind::Tuple(elems) => {
                let mut comps = vec![];
                for e in elems {
                    match &e.kind {
                        PatKind::Var(n) => {
                            if comps.iter().any(|c| matches!(c, Comp::Named(m) if m == n)) {
                                return Err(self.err(
                                    ErrorKind::Pattern,
                                    e.span,
                                    format!("the name '{n}' is bound twice in this pattern"),
                                ));
                            }
                            comps.push(Comp::Named(n.clone()));
                        }
                        PatKind::Wild => comps.push(Comp::Wild),
                        _ => return Err(self.nested_binder_error(e)),
                    }
                }
                Ok(Shape::Tuple(comps))
            }
            _ => Err(match site {
                BinderSite::Field => self.err(
                    ErrorKind::NestedPattern,
                    p.span,
                    "patterns are one level deep in dc: a constructor's fields can only \
                     be names, '_' or a flat tuple of names. Bind the field to a name and \
                     take it apart in a second match, for example \
                     '| x :: rest -> (match rest with | [] -> ... | y :: t -> ...)' \
                     instead of '| x :: y :: t -> ...'",
                ),
                BinderSite::Param | BinderSite::Let => self.err(
                    ErrorKind::Pattern,
                    p.span,
                    "a parameter or 'let' can only bind a name, '_', or a flat tuple of \
                     names like '(a, b)'. To take apart a list, option or tree, bind a \
                     name and use 'match name with ...'",
                ),
            }),
        }
    }

    /// `fun p1 ... pn -> body` (or just `body` when there are no parameters). Each
    /// parameter's tuple projections come directly under its own lambda.
    fn lambda(&mut self, params: &[Pat], body: &Expr, site: BinderSite) -> DResult<TermId> {
        let Some((p, rest)) = params.split_first() else {
            return self.expr(body);
        };
        let shape = self.shape(p, site)?;
        let mark = self.scope.len();
        let inner = match shape {
            Shape::Named(n) => {
                self.scope.push(Binder::Named(n));
                self.lambda(rest, body, site)
            }
            Shape::Wild => {
                self.push_hidden();
                self.lambda(rest, body, site)
            }
            Shape::Tuple(comps) => {
                let h = self.push_hidden();
                self.destructure(h, &comps, |s| s.lambda(rest, body, site))
            }
        };
        self.scope.truncate(mark);
        Ok(self.arena.lam(inner?))
    }

    /// Binds tuple components from the hidden variable `src`, then runs `k`.
    /// `(c1, c2, ..., cn)` is `(c1, (c2, ..., cn))`: `c1 = fst src`, and the rest
    /// come from `snd src`. Wildcards are not projected, and the `snd` is skipped
    /// when everything after it is a wildcard. Binders stay pushed; the caller
    /// truncates the scope.
    fn destructure<F>(&mut self, src: u32, comps: &[Comp], k: F) -> DResult<TermId>
    where
        F: FnOnce(&mut Self) -> DResult<TermId>,
    {
        let (first, rest) = comps.split_first().expect("tuples have two or more parts");
        match first {
            Comp::Named(n) => {
                let fst = self.arena.prim(Prim::Fst);
                let v = self.hidden_var(src);
                let proj = self.arena.app(fst, v);
                self.scope.push(Binder::Named(n.clone()));
                let body = self.destructure_rest(src, rest, k)?;
                Ok(self.arena.let_(proj, body))
            }
            Comp::Wild => self.destructure_rest(src, rest, k),
        }
    }

    fn destructure_rest<F>(&mut self, src: u32, rest: &[Comp], k: F) -> DResult<TermId>
    where
        F: FnOnce(&mut Self) -> DResult<TermId>,
    {
        let needed = rest.iter().any(|c| matches!(c, Comp::Named(_)));
        if !needed {
            return k(self);
        }
        let snd = self.arena.prim(Prim::Snd);
        let v = self.hidden_var(src);
        let proj = self.arena.app(snd, v);
        let body = if let [Comp::Named(n)] = rest {
            self.scope.push(Binder::Named(n.clone()));
            k(self)?
        } else {
            let h = self.push_hidden();
            self.destructure(h, rest, k)?
        };
        Ok(self.arena.let_(proj, body))
    }

    // ---- expressions ----

    fn prim(&mut self, p: Prim) -> TermId {
        self.arena.prim(p)
    }

    fn expr(&mut self, e: &Expr) -> DResult<TermId> {
        match &e.kind {
            ExprKind::Int(n) => Ok(self.arena.int(*n)),
            ExprKind::Bool(b) => Ok(self.arena.bool(*b)),
            ExprKind::Unit => Ok(self.arena.unit()),
            ExprKind::Var(name) => self.lookup(name, e.span),
            ExprKind::Ctor(c) => Ok(self.prim(ctor_prim(*c))),
            ExprKind::OpSection(op) => match op_prim(*op) {
                Some(p) => Ok(self.prim(p)),
                None => Err(self.err(
                    ErrorKind::Syntax,
                    e.span,
                    format!("'( {} )' is not available as a function", op.text()),
                )),
            },
            ExprKind::App(head, args) => self.app(head, args),
            ExprKind::Bin(op, a, b) => self.binop(*op, a, b),
            ExprKind::Neg(a) => {
                let sub = self.prim(Prim::Sub);
                let zero = self.arena.int(0);
                let ta = self.expr(a)?;
                Ok(self.arena.apps(sub, &[zero, ta]))
            }
            ExprKind::Tuple(elems) => self.tuple(elems),
            ExprKind::List(elems) => {
                let ts = elems
                    .iter()
                    .map(|x| self.expr(x))
                    .collect::<DResult<Vec<_>>>()?;
                let mut acc = self.prim(Prim::Nil);
                let cons = self.prim(Prim::Cons);
                for &t in ts.iter().rev() {
                    acc = self.arena.apps(cons, &[t, acc]);
                }
                Ok(acc)
            }
            ExprKind::Fun(params, body) => self.lambda(params, body, BinderSite::Param),
            ExprKind::Let(lhs, rhs, body) => self.let_(lhs, rhs, body),
            ExprKind::If(c, t, f) => {
                let c = self.expr(c)?;
                let t = self.expr(t)?;
                let f = self.expr(f)?;
                Ok(self.arena.if_(c, t, f))
            }
            ExprKind::Match(s, arms) => self.match_(e.span, s, arms),
        }
    }

    fn tuple(&mut self, elems: &[Expr]) -> DResult<TermId> {
        let ts = elems
            .iter()
            .map(|x| self.expr(x))
            .collect::<DResult<Vec<_>>>()?;
        let pair = self.prim(Prim::Pair);
        let mut acc = ts[ts.len() - 1];
        for &t in ts[..ts.len() - 1].iter().rev() {
            acc = self.arena.apps(pair, &[t, acc]);
        }
        Ok(acc)
    }

    fn app(&mut self, head: &Expr, args: &[Expr]) -> DResult<TermId> {
        if head.kind == ExprKind::Ctor(Ctor::Node) {
            if let ExprKind::Tuple(fields) = &args[0].kind {
                if fields.len() != 3 {
                    return Err(self.err(
                        ErrorKind::Syntax,
                        args[0].span,
                        "Node takes three fields: Node (left, value, right)",
                    ));
                }
                let mut t = self.prim(Prim::Node);
                for f in fields.iter().chain(&args[1..]) {
                    let ta = self.expr(f)?;
                    t = self.arena.app(t, ta);
                }
                return Ok(t);
            }
        }
        let mut t = self.expr(head)?;
        for a in args {
            let ta = self.expr(a)?;
            t = self.arena.app(t, ta);
        }
        Ok(t)
    }

    fn binop(&mut self, op: BinOp, a: &Expr, b: &Expr) -> DResult<TermId> {
        match op {
            BinOp::And | BinOp::Or => {
                let ta = self.expr(a)?;
                let tb = self.expr(b)?;
                Ok(if op == BinOp::And {
                    self.arena.and(ta, tb)
                } else {
                    self.arena.or(ta, tb)
                })
            }
            BinOp::Gt | BinOp::Ge => {
                let p = self.prim(if op == BinOp::Gt { Prim::Lt } else { Prim::Le });
                let ta = self.expr(a)?;
                let tb = self.expr(b)?;
                if self.is_trivial(ta) || self.is_trivial(tb) {
                    return Ok(self.arena.apps(p, &[tb, ta]));
                }
                // Evaluate a first, then b: let-bind a and compile b under it.
                let mark = self.scope.len();
                let h = self.push_hidden();
                let tb = self.expr(b);
                let va = self.hidden_var(h);
                self.scope.truncate(mark);
                let body = self.arena.apps(p, &[tb?, va]);
                Ok(self.arena.let_(ta, body))
            }
            _ => {
                let p = self.prim(op_prim(op).expect("arithmetic and comparison ops are prims"));
                let ta = self.expr(a)?;
                let tb = self.expr(b)?;
                Ok(self.arena.apps(p, &[ta, tb]))
            }
        }
    }

    /// A term whose evaluation is free and cannot fail: moving it past another
    /// operand changes neither result nor cost.
    fn is_trivial(&self, t: TermId) -> bool {
        use dc_core::Term;
        matches!(
            self.arena.get(t),
            Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit
        )
    }

    fn function_like(lhs: &LetLhs, rhs: &Expr) -> bool {
        match lhs {
            LetLhs::Name { params, .. } => {
                !params.is_empty() || matches!(rhs.kind, ExprKind::Fun(..))
            }
            LetLhs::Pattern(_) => false,
        }
    }

    fn let_(&mut self, lhs: &LetLhs, rhs: &Expr, body: &Expr) -> DResult<TermId> {
        let mark = self.scope.len();
        match lhs {
            LetLhs::Name { name, params, .. } => {
                let t = if Self::function_like(lhs, rhs) {
                    self.defining.push((name.clone(), self.scope.len()));
                    let t = self.lambda(params, rhs, BinderSite::Param);
                    self.defining.pop();
                    t?
                } else {
                    self.expr(rhs)?
                };
                self.scope.push(Binder::Named(name.clone()));
                let b = self.expr(body);
                self.scope.truncate(mark);
                Ok(self.arena.let_(t, b?))
            }
            LetLhs::Pattern(p) => {
                let shape = self.shape(p, BinderSite::Let)?;
                let t = self.expr(rhs)?;
                self.bind_shape(t, shape, body)
            }
        }
    }

    /// `let SHAPE = t in body`, with `t` already compiled.
    fn bind_shape(&mut self, t: TermId, shape: Shape, body: &Expr) -> DResult<TermId> {
        let mark = self.scope.len();
        let inner = match shape {
            Shape::Named(n) => {
                self.scope.push(Binder::Named(n));
                self.expr(body)
            }
            Shape::Wild => {
                self.push_hidden();
                self.expr(body)
            }
            Shape::Tuple(comps) => {
                let h = self.push_hidden();
                self.destructure(h, &comps, |s| s.expr(body))
            }
        };
        self.scope.truncate(mark);
        Ok(self.arena.let_(t, inner?))
    }

    fn not_a_constructor_pattern(&self, p: &Pat) -> SurfaceError {
        self.err(
            ErrorKind::Pattern,
            p.span,
            "match in dc only takes apart lists ('[]' / 'h :: t'), options ('None' / \
             'Some x') and trees ('Leaf' / 'Node (l, x, r)'), one constructor per arm. \
             To compare with a number or boolean use 'if ... then ... else'",
        )
    }

    fn classify<'p>(&self, p: &'p Pat) -> DResult<ArmCase<'p>> {
        Ok(match &p.kind {
            PatKind::Nil => ArmCase::Empty(Family::List),
            PatKind::Cons(h, t) => ArmCase::Full(Family::List, vec![h, t]),
            PatKind::Ctor(Ctor::None, None) => ArmCase::Empty(Family::Option),
            PatKind::Ctor(Ctor::Leaf, None) => ArmCase::Empty(Family::Tree),
            PatKind::Ctor(Ctor::Some, Some(x)) => ArmCase::Full(Family::Option, vec![x]),
            PatKind::Ctor(Ctor::Node, Some(x)) => match &x.kind {
                PatKind::Tuple(fields) if fields.len() == 3 => {
                    ArmCase::Full(Family::Tree, fields.iter().collect())
                }
                _ => {
                    return Err(self.err(
                        ErrorKind::Pattern,
                        x.span,
                        "a Node pattern has three fields: 'Node (l, x, r)'",
                    ))
                }
            },
            PatKind::Ctor(c @ (Ctor::None | Ctor::Leaf), Some(_)) => {
                return Err(self.err(
                    ErrorKind::Pattern,
                    p.span,
                    format!("'{}' has no fields", c.name()),
                ))
            }
            PatKind::Ctor(c, None) => {
                return Err(self.err(
                    ErrorKind::Pattern,
                    p.span,
                    format!(
                        "'{}' needs a pattern for its fields: {}",
                        c.name(),
                        if *c == Ctor::Some {
                            "'Some x'"
                        } else {
                            "'Node (l, x, r)'"
                        }
                    ),
                ))
            }
            PatKind::Wild => ArmCase::CatchAll,
            PatKind::List(_) => {
                return Err(self.err(
                    ErrorKind::NestedPattern,
                    p.span,
                    "a list pattern like '[x]' or '[x; y]' is a nested pattern ('x :: []'), \
                     and patterns are one level deep in dc. Match 'x :: rest', then match \
                     'rest' in a second match",
                ))
            }
            PatKind::Var(n) => {
                return Err(self.err(
                    ErrorKind::Pattern,
                    p.span,
                    format!(
                        "a catch-all arm in dc is written '_ -> ...'; to name the whole \
                         value, bind it first: 'let {n} = ... in match {n} with ...'"
                    ),
                ))
            }
            PatKind::Int(_) | PatKind::Bool(_) | PatKind::Unit | PatKind::Tuple(_) => {
                return Err(self.not_a_constructor_pattern(p))
            }
        })
    }

    fn match_(&mut self, span: Span, s: &Expr, arms: &[Arm]) -> DResult<TermId> {
        // One arm binding the whole value: a let.
        if let [arm] = arms {
            if matches!(
                arm.pat.kind,
                PatKind::Var(_) | PatKind::Wild | PatKind::Tuple(_)
            ) {
                let shape = self.shape(&arm.pat, BinderSite::Let)?;
                let t = self.expr(s)?;
                return self.bind_shape(t, shape, &arm.body);
            }
        }
        let mut family: Option<(Family, Span)> = None;
        let mut empty: Option<&Arm> = None;
        let mut full: Option<(&Arm, Vec<&Pat>)> = None;
        let mut catch_all: Option<&Arm> = None;
        for arm in arms {
            if let Some(c) = catch_all {
                return Err(self.err(
                    ErrorKind::Pattern,
                    arm.pat.span,
                    format!(
                        "this arm can never be reached: the '_' arm at {} already \
                         covers it (put '_' last)",
                        self.src.describe_position(c.pat.span.start)
                    ),
                ));
            }
            let case = self.classify(&arm.pat)?;
            let fam = match &case {
                ArmCase::Empty(f) | ArmCase::Full(f, _) => Some(*f),
                ArmCase::CatchAll => None,
            };
            if let Some(f) = fam {
                match family {
                    Some((g, gspan)) if g != f => {
                        return Err(self.err(
                            ErrorKind::Pattern,
                            arm.pat.span,
                            format!(
                                "this arm matches {}, but the arm at {} matches {}; \
                                 all arms of one match must take apart the same type",
                                f.describe(),
                                self.src.describe_position(gspan.start),
                                g.describe()
                            ),
                        ))
                    }
                    _ => family = Some((f, arm.pat.span)),
                }
            }
            let dup = |s: &Self| {
                s.err(
                    ErrorKind::Pattern,
                    arm.pat.span,
                    "this case is already matched by an earlier arm",
                )
            };
            match case {
                ArmCase::Empty(_) if empty.is_some() => return Err(dup(self)),
                ArmCase::Full(..) if full.is_some() => return Err(dup(self)),
                ArmCase::Empty(_) => empty = Some(arm),
                ArmCase::Full(_, fields) => full = Some((arm, fields)),
                ArmCase::CatchAll => {
                    if empty.is_some() && full.is_some() {
                        return Err(self.err(
                            ErrorKind::Pattern,
                            arm.pat.span,
                            "this '_' arm can never be reached: both cases are already matched",
                        ));
                    }
                    catch_all = Some(arm);
                }
            }
        }
        let Some((fam, _)) = family else {
            return Err(self.err(
                ErrorKind::Pattern,
                span,
                "this match has no constructor arm; to bind a value use 'let'",
            ));
        };
        // A nested match in an earlier arm takes all the following arms: say so.
        let nested_hint = if arms
            .last()
            .is_some_and(|a| matches!(a.body.kind, ExprKind::Match(..)))
        {
            ". If an arm of this match is itself a match, the inner match takes \
             every arm after it: put the inner match in parentheses"
        } else {
            ""
        };
        if catch_all.is_none() {
            if empty.is_none() {
                return Err(self.err(
                    ErrorKind::Pattern,
                    span,
                    format!(
                        "this match on {} has no arm for '{}'; add '| {} -> ...' or \
                         '| _ -> ...'{nested_hint}",
                        fam.describe(),
                        fam.empty_pat(),
                        fam.empty_pat()
                    ),
                ));
            }
            if full.is_none() {
                return Err(self.err(
                    ErrorKind::Pattern,
                    span,
                    format!(
                        "this match on {} has no arm for '{}'; add '| {} -> ...' or \
                         '| _ -> ...'{nested_hint}",
                        fam.describe(),
                        fam.full_pat(),
                        fam.full_pat()
                    ),
                ));
            }
        }
        // Validate field binders before compiling anything, so the nested-pattern
        // error is reported even if a body has another error.
        if let Some((_, fields)) = &full {
            for f in fields {
                self.shape(f, BinderSite::Field)?;
            }
        }
        let ts = self.expr(s)?;
        let empty_fn = match empty {
            Some(arm) => self.wild_lambdas(1, &arm.body)?,
            None => self.wild_lambdas(1, &catch_all.expect("checked").body)?,
        };
        let full_fn = match full {
            Some((arm, fields)) => {
                let params: Vec<Pat> = fields.into_iter().cloned().collect();
                self.lambda(&params, &arm.body, BinderSite::Field)?
            }
            None => self.wild_lambdas(fam.arity(), &catch_all.expect("checked").body)?,
        };
        let case = self.prim(fam.prim());
        Ok(self.arena.apps(case, &[ts, empty_fn, full_fn]))
    }

    /// `fun _ ... _ -> body` with `n` hidden binders.
    fn wild_lambdas(&mut self, n: usize, body: &Expr) -> DResult<TermId> {
        let mark = self.scope.len();
        for _ in 0..n {
            self.push_hidden();
        }
        let b = self.expr(body);
        self.scope.truncate(mark);
        let mut t = b?;
        for _ in 0..n {
            t = self.arena.lam(t);
        }
        Ok(t)
    }

    /// Compiles an entry expression in the current scope.
    pub fn entry(&mut self, e: &Expr) -> DResult<TermId> {
        self.expr(e)
    }
}

fn ctor_prim(c: Ctor) -> Prim {
    match c {
        Ctor::None => Prim::None,
        Ctor::Some => Prim::Some,
        Ctor::Leaf => Prim::Leaf,
        Ctor::Node => Prim::Node,
    }
}

/// The primitive an operator maps to, if it is a plain primitive application.
pub(crate) fn op_prim(op: BinOp) -> Option<Prim> {
    Some(match op {
        BinOp::Add => Prim::Add,
        BinOp::Sub => Prim::Sub,
        BinOp::Mul => Prim::Mul,
        BinOp::Div => Prim::Div,
        BinOp::Mod => Prim::Mod,
        BinOp::IntEq => Prim::IntEq,
        BinOp::Eq => Prim::Eq,
        BinOp::Lt => Prim::Lt,
        BinOp::Le => Prim::Le,
        BinOp::Cons => Prim::Cons,
        BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or => return None,
    })
}

/// Levenshtein distance over characters.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur.push(sub.min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

// ---- the top level ----

/// An ordered set of top-level names that programs and expressions are compiled
/// against: definitions from source (the prelude, then the program) and inline
/// names for closed core terms (for example library abstractions).
///
/// Later names shadow earlier ones. [`TopLevel::close_definition`] and
/// [`TopLevel::close_expression`] produce one closed core term: the definitions
/// the entry needs (transitively), in source order, as nested `let`s around the
/// entry. Definitions the entry does not use are left out, so the cost of a
/// program does not depend on what else the prelude defines.
#[derive(Clone, Debug, Default)]
pub struct TopLevel {
    globals: Vec<Global>,
}

impl TopLevel {
    /// An empty top level: only primitives are in scope.
    pub fn new() -> TopLevel {
        TopLevel::default()
    }

    /// The top-level names, oldest first (shadowed names included).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.globals.iter().map(|g| g.name.as_str())
    }

    /// Parses and checks a program (a sequence of `let` definitions) and adds its
    /// definitions. `source_name` appears in error messages. On error nothing is
    /// added. Returns the names defined, in order.
    pub fn add_source(&mut self, source_name: &str, text: &str) -> DResult<Vec<String>> {
        let src = Rc::new(Source::new(source_name, text));
        let defs = parser::parse_program(&src)?;
        let mut globals = self.globals.clone();
        let mut names = vec![];
        for def in defs {
            let deps = {
                let mut scratch = Arena::new();
                let mut ctx = Ctx::new(&mut scratch, &globals, src.clone(), true);
                for g in 0..globals.len() {
                    ctx.push_global(g);
                }
                ctx.definition(&def)?;
                ctx.deps.into_iter().collect()
            };
            names.push(def.name.clone());
            globals.push(Global {
                name: def.name.clone(),
                kind: GlobalKind::Def {
                    def: Rc::new(def),
                    source: src.clone(),
                    deps,
                },
            });
        }
        self.globals = globals;
        Ok(names)
    }

    /// Adds `name` for the closed term `term` (in `arena`), substituted directly at
    /// each use. Meant for terms whose evaluation is free, such as library
    /// references ([`dc_core::Term::Lib`]) or lambdas; a term that costs steps to
    /// evaluate would be charged at every use.
    pub fn define_inline(&mut self, arena: &Arena, name: &str, term: TermId) -> DResult<()> {
        if !arena.is_closed(term) {
            return Err(SurfaceError {
                kind: ErrorKind::UnknownName,
                source_name: "<inline>".to_owned(),
                line: 0,
                column: 0,
                span: Span::default(),
                message: format!("the term given for '{name}' has free variables"),
            });
        }
        self.globals.push(Global {
            name: name.to_owned(),
            kind: GlobalKind::Inline(term),
        });
        Ok(())
    }

    fn reachable(&self, roots: impl IntoIterator<Item = usize>) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        let mut stack: Vec<usize> = roots.into_iter().collect();
        while let Some(g) = stack.pop() {
            if seen.insert(g) {
                if let GlobalKind::Def { deps, .. } = &self.globals[g].kind {
                    stack.extend(deps.iter().copied());
                }
            }
        }
        seen
    }

    /// Compiles the reachable definitions into `ctx`'s scope; returns their terms.
    fn bind_reachable(
        ctx: &mut Ctx<'_>,
        globals: &[Global],
        keep: &BTreeSet<usize>,
    ) -> DResult<Vec<TermId>> {
        let mut terms = vec![];
        for &g in keep {
            if let GlobalKind::Def { def, source, .. } = &globals[g].kind {
                ctx.set_source(source.clone());
                terms.push(ctx.definition(def)?);
            }
            ctx.push_global(g);
        }
        Ok(terms)
    }

    fn wrap(arena: &mut Arena, defs: &[TermId], mut body: TermId) -> TermId {
        for &d in defs.iter().rev() {
            body = arena.let_(d, body);
        }
        body
    }

    /// The closed term for the latest top-level name `name`: its definition, inside
    /// `let`s for the definitions it needs.
    pub fn close_definition(&self, arena: &mut Arena, name: &str) -> DResult<TermId> {
        let Some(g) = self.globals.iter().rposition(|x| x.name == name) else {
            return Err(SurfaceError {
                kind: ErrorKind::UnknownName,
                source_name: "<entry>".to_owned(),
                line: 0,
                column: 0,
                span: Span::default(),
                message: format!("there is no top-level definition named '{name}'"),
            });
        };
        if let GlobalKind::Inline(t) = self.globals[g].kind {
            return Ok(t);
        }
        let keep = self.reachable([g]);
        let src = Rc::new(Source::new("<entry>", ""));
        let mut ctx = Ctx::new(arena, &self.globals, src, false);
        let mut defs = Self::bind_reachable(&mut ctx, &self.globals, &keep)?;
        // The entry is the last definition kept: `let d in $0` is just `d`.
        let last = defs.pop().expect("the entry is a definition");
        Ok(Self::wrap(arena, &defs, last))
    }

    /// Compiles the expression `text` against the top level into a closed term.
    pub fn close_expression(
        &self,
        arena: &mut Arena,
        source_name: &str,
        text: &str,
    ) -> DResult<TermId> {
        let src = Rc::new(Source::new(source_name, text));
        let e = parser::parse_expression(&src)?;
        let deps = {
            let mut scratch = Arena::new();
            let mut ctx = Ctx::new(&mut scratch, &self.globals, src.clone(), true);
            for g in 0..self.globals.len() {
                ctx.push_global(g);
            }
            ctx.entry(&e)?;
            ctx.deps
        };
        let keep = self.reachable(deps);
        let mut ctx = Ctx::new(arena, &self.globals, src.clone(), false);
        let defs = Self::bind_reachable(&mut ctx, &self.globals, &keep)?;
        ctx.set_source(src);
        let body = ctx.entry(&e)?;
        Ok(Self::wrap(arena, &defs, body))
    }
}

/// Compiles a closed expression with only the primitives in scope.
pub fn compile_expression(arena: &mut Arena, text: &str) -> DResult<TermId> {
    TopLevel::new().close_expression(arena, "expression", text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_examples() {
        assert_eq!(edit_distance("lenght", "length"), 2);
        assert_eq!(edit_distance("foldl", "foldr"), 1);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("same", "same"), 0);
    }
}
