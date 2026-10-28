//! Printing core terms back as surface syntax, with generated names.
//!
//! The output is meant to be read by the LLM (for example to show it library
//! abstractions) and compiles back to the same core term. The sugar the desugarer
//! produces is recognised and printed back: `match` for `list_case`,
//! `option_case` and `tree_case` with lambda branches, tuple binders for
//! `fst`/`snd` lets on an otherwise unused variable, infix operators, `>` for the
//! let-bound comparison, tuples, list literals and constructors.
//!
//! [`pretty`] checks its own output: it compiles the printed text and compares the
//! result with the input term. If the sugared text does not compile back to the
//! same term, it prints again without sugar (only `fun`, `let`, application,
//! `if`, `&&`, `||` and infix operators), which always round-trips.
//!
//! # Limits
//!
//! The printer recurses on the host stack, once or a few times per level of the
//! core term, and a stack overflow aborts the process. So it first measures, with
//! an explicit stack, how deep the term nests, and refuses a term deeper than
//! [`MAX_PRETTY_DEPTH`] with a [`PrettyError::TooDeep`] error. Every node counts
//! one level, including each application in a spine `f a b c`. The limit is five
//! times what the front end accepts ([`crate::MAX_DEPTH`]), so it also covers a
//! closed program wrapped in one `let` per definition it uses, for up to about 400
//! definitions. The printer stays recursive because its sugar recognition and
//! name scoping read much more plainly that way, and a deeper term could not be
//! read back by the front end anyway.
//!
//! A term deeper than [`INLINE_DEPTH`] is printed on a thread of its own with a
//! [`PRINT_STACK`] stack, so how much stack the caller has left does not matter.
//! A term within the depth limit needs at most about 2 MiB of stack in a debug
//! build.
//!
//! Terms are DAGs, and the printer writes a shared subterm out at every use, so
//! `(if c t t)` nested `n` deep prints `2^n` subterms. So the printer also refuses
//! a term with more than [`MAX_PRETTY_NODES`] nodes written out as a tree, with a
//! [`PrettyError::TooLarge`] error.

use std::collections::{HashMap, HashSet};
use std::fmt;

use dc_core::term::children;
use dc_core::{sexpr, Arena, LibId, Prim, Term, TermId};

use crate::desugar::TopLevel;
use crate::lexer::is_keyword;

const P_OPEN: u8 = 0;
const P_OR: u8 = 1;
const P_AND: u8 = 2;
const P_CMP: u8 = 3;
const P_CONS: u8 = 4;
const P_ADD: u8 = 5;
const P_MUL: u8 = 6;
const P_UNARY: u8 = 7;
const P_APP: u8 = 8;
const P_ATOM: u8 = 9;

/// A name pushed for a binder the printed text never mentions.
const HIDDEN: &str = "%hidden";

/// The deepest core term [`pretty`] prints; see the module documentation.
pub const MAX_PRETTY_DEPTH: usize = 500;

/// The most nodes a core term [`pretty`] prints may have, written out as a tree.
pub const MAX_PRETTY_NODES: usize = 100_000;

// `pretty` checks its output with `sexpr::print`, which must take any term it prints.
const _: () = assert!(MAX_PRETTY_NODES <= sexpr::MAX_PRINT_NODES);

/// The deepest term printed on the caller's thread. In a debug build, 256 KiB of
/// stack left is enough to print any term this deep, and to start the printing
/// thread for a deeper one.
pub const INLINE_DEPTH: usize = 16;

/// The stack size of the thread a deeper term is printed on: eight times what
/// the deepest term needs in a debug build.
pub const PRINT_STACK: usize = 16 << 20;

/// Why [`pretty`] refused a term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrettyError {
    /// The term nests more deeply than [`MAX_PRETTY_DEPTH`].
    TooDeep,
    /// The term has more than [`MAX_PRETTY_NODES`] nodes written out as a tree.
    TooLarge,
}

impl fmt::Display for PrettyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrettyError::TooDeep => write!(
                f,
                "the term nests more than {MAX_PRETTY_DEPTH} levels deep, too deep to \
                 print as surface syntax"
            ),
            PrettyError::TooLarge => write!(
                f,
                "the term has more than {MAX_PRETTY_NODES} nodes written out, too many \
                 to print as surface syntax"
            ),
        }
    }
}

impl std::error::Error for PrettyError {}

/// Prints `id` as surface syntax. Library references print as `lib_name(l)`.
/// Free variables (when `id` is open) print as `free0`, `free1`, ... with `free0`
/// the innermost.
///
/// Fails if `id` nests more than [`MAX_PRETTY_DEPTH`] levels deep or has more
/// than [`MAX_PRETTY_NODES`] nodes written out. Panics if a term deeper than
/// [`INLINE_DEPTH`] comes and the system cannot start a thread to print it on.
pub fn pretty(
    arena: &Arena,
    id: TermId,
    lib_name: impl Fn(LibId) -> String + Sync,
) -> Result<String, PrettyError> {
    let height = measure(arena, id)?;
    Ok(on_enough_stack(height, || {
        let sugared = Printer::new(arena, &lib_name, true).top(id);
        if round_trips(arena, id, &sugared, &lib_name) {
            return sugared;
        }
        Printer::new(arena, &lib_name, false).top(id)
    }))
}

/// Prints without checking the result. With `sugar` false only `fun`, `let`,
/// application, `if`, `&&`, `||`, infix operators and tuples are used; that form
/// always compiles back to the same term, and [`pretty`] falls back to it.
/// Exposed for tests.
#[doc(hidden)]
pub fn pretty_unchecked(
    arena: &Arena,
    id: TermId,
    lib_name: impl Fn(LibId) -> String + Sync,
    sugar: bool,
) -> Result<String, PrettyError> {
    let height = measure(arena, id)?;
    Ok(on_enough_stack(height, || {
        Printer::new(arena, &lib_name, sugar).top(id)
    }))
}

/// Runs `print` for a term `height` levels deep: on this thread if the term is
/// at most [`INLINE_DEPTH`] deep, else on a new thread with a [`PRINT_STACK`]
/// stack.
fn on_enough_stack(height: usize, print: impl FnOnce() -> String + Send) -> String {
    if height <= INLINE_DEPTH {
        return print();
    }
    std::thread::scope(|scope| {
        let printer = std::thread::Builder::new()
            .name("dc-pretty".into())
            .stack_size(PRINT_STACK)
            .spawn_scoped(scope, print)
            .expect("the system starts a thread to print a deep term on");
        printer
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

/// The height of `id`: the most levels from it down to a leaf. Fails if that is
/// more than [`MAX_PRETTY_DEPTH`], or if `id` has more than [`MAX_PRETTY_NODES`]
/// nodes written out as a tree. It walks with an explicit stack and visits each
/// distinct node once.
fn measure(arena: &Arena, id: TermId) -> Result<usize, PrettyError> {
    // The height and tree size of each finished node.
    let mut done: HashMap<TermId, (usize, usize)> = HashMap::new();
    let mut stack = vec![(id, 1usize, false)];
    while let Some((t, depth, children_done)) = stack.pop() {
        if depth > MAX_PRETTY_DEPTH {
            return Err(PrettyError::TooDeep);
        }
        if let Some(&(h, _)) = done.get(&t) {
            // Done already, reached again by another path.
            if depth + h - 1 > MAX_PRETTY_DEPTH {
                return Err(PrettyError::TooDeep);
            }
            continue;
        }
        let cs = children(&arena.get(t));
        if children_done {
            let h = 1 + cs.iter().map(|c| done[c].0).max().unwrap_or(0);
            let size = cs.iter().fold(1usize, |n, c| n.saturating_add(done[c].1));
            if size > MAX_PRETTY_NODES {
                return Err(PrettyError::TooLarge);
            }
            done.insert(t, (h, size));
            continue;
        }
        stack.push((t, depth, true));
        for c in cs {
            match done.get(&c) {
                Some(&(h, _)) if depth + h > MAX_PRETTY_DEPTH => return Err(PrettyError::TooDeep),
                Some(_) => {}
                None => stack.push((c, depth + 1, false)),
            }
        }
    }
    Ok(done[&id].0)
}

/// The library abstractions `id` refers to.
fn libs_in(arena: &Arena, id: TermId) -> HashSet<LibId> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack = vec![id];
    while let Some(t) = stack.pop() {
        if !seen.insert(t) {
            continue;
        }
        let node = arena.get(t);
        if let Term::Lib(l) = node {
            out.insert(l);
        }
        stack.extend(children(&node));
    }
    out
}

fn round_trips(arena: &Arena, id: TermId, text: &str, lib_name: &impl Fn(LibId) -> String) -> bool {
    let mut scratch = Arena::new();
    let mut env = TopLevel::new();
    for l in libs_in(arena, id) {
        let t = scratch.lib(l);
        if env.define_inline(&scratch, &lib_name(l), t).is_err() {
            return false;
        }
    }
    let depth = arena.free_depth(id);
    let mut wrapped = String::new();
    if depth > 0 {
        wrapped.push_str("fun");
        for j in (0..depth).rev() {
            wrapped.push_str(&format!(" free{j}"));
        }
        wrapped.push_str(" -> ");
    }
    wrapped.push_str(text);
    let Ok(back) = env.close_expression(&mut scratch, "pretty", &wrapped) else {
        return false;
    };
    let mut expected = String::new();
    for _ in 0..depth {
        expected.push_str("(lam ");
    }
    // `pretty` measured `id` first, so it is within MAX_PRETTY_NODES, which is
    // within what `sexpr::print` takes.
    expected.push_str(&sexpr::print(arena, id).expect("a measured term is within the print limit"));
    for _ in 0..depth {
        expected.push(')');
    }
    // A term too large to print is larger than `id`, so it is not the same term.
    sexpr::print(&scratch, back).is_ok_and(|text| text == expected)
}

fn infix(p: Prim) -> Option<(&'static str, u8, u8, u8)> {
    // (text, precedence, left operand needs, right operand needs)
    Some(match p {
        Prim::Add => ("+", P_ADD, P_ADD, P_ADD + 1),
        Prim::Sub => ("-", P_ADD, P_ADD, P_ADD + 1),
        Prim::Mul => ("*", P_MUL, P_MUL, P_MUL + 1),
        Prim::Div => ("/", P_MUL, P_MUL, P_MUL + 1),
        Prim::Mod => ("mod", P_MUL, P_MUL, P_MUL + 1),
        Prim::IntEq => ("=", P_CMP, P_CMP + 1, P_CMP + 1),
        Prim::Eq => ("==", P_CMP, P_CMP + 1, P_CMP + 1),
        Prim::Lt => ("<", P_CMP, P_CMP + 1, P_CMP + 1),
        Prim::Le => ("<=", P_CMP, P_CMP + 1, P_CMP + 1),
        Prim::Cons => ("::", P_CONS, P_CONS + 1, P_CONS),
        _ => return None,
    })
}

/// One binder in a printed parameter or let pattern.
enum Binding {
    /// A name (or `_` when unused).
    Name(String),
    /// A tuple pattern, with its components (`_` for wildcards).
    Tuple(Vec<String>),
}

impl Binding {
    fn text(&self) -> String {
        match self {
            Binding::Name(n) => n.clone(),
            Binding::Tuple(cs) => format!("({})", cs.join(", ")),
        }
    }
}

/// A recognised tuple destructuring: component kinds and the binders it pushes
/// after the destructured variable, in order (`true` = named component, `false` =
/// hidden tail).
struct TupleShape {
    named: Vec<bool>,
    pushes: Vec<bool>,
    rest: TermId,
}

struct Printer<'a, F: Fn(LibId) -> String> {
    arena: &'a Arena,
    lib_name: &'a F,
    sugar: bool,
    names: Vec<String>,
    used: HashSet<String>,
    reserved: HashSet<String>,
    uses_memo: HashMap<(TermId, u32), bool>,
}

impl<'a, F: Fn(LibId) -> String> Printer<'a, F> {
    fn new(arena: &'a Arena, lib_name: &'a F, sugar: bool) -> Self {
        Printer {
            arena,
            lib_name,
            sugar,
            names: vec![],
            used: HashSet::new(),
            reserved: HashSet::new(),
            uses_memo: HashMap::new(),
        }
    }

    fn top(&mut self, id: TermId) -> String {
        self.reserved = libs_in(self.arena, id)
            .into_iter()
            .map(|l| (self.lib_name)(l))
            .collect();
        self.go(id, P_OPEN)
    }

    fn fresh(&mut self, hint: &str) -> String {
        let mut n = 0usize;
        loop {
            let cand = if n == 0 {
                hint.to_owned()
            } else {
                format!("{hint}{n}")
            };
            if !self.used.contains(&cand)
                && !self.reserved.contains(&cand)
                && !is_keyword(&cand)
                && Prim::from_name(&cand).is_none()
            {
                self.used.insert(cand.clone());
                return cand;
            }
            n += 1;
        }
    }

    /// Whether de Bruijn index `k` occurs free in `id`.
    fn uses(&mut self, id: TermId, k: u32) -> bool {
        if let Some(&b) = self.uses_memo.get(&(id, k)) {
            return b;
        }
        let r = match self.arena.get(id) {
            Term::Var(i) => i == k,
            Term::Lam(b) => self.uses(b, k + 1),
            Term::Let(x, b) => self.uses(x, k) || self.uses(b, k + 1),
            t => children(&t).into_iter().any(|c| self.uses(c, k)),
        };
        self.uses_memo.insert((id, k), r);
        r
    }

    fn is_proj(&self, id: TermId, p: Prim, var: u32) -> bool {
        match self.arena.get(id) {
            Term::App(f, a) => {
                self.arena.get(f) == Term::Prim(p) && self.arena.get(a) == Term::Var(var)
            }
            _ => false,
        }
    }

    /// Recognises the desugaring of a tuple binder on the variable at index `h` in
    /// the scope of `t` (see `desugar::Ctx::destructure`).
    fn tuple_shape(&mut self, t: TermId, h: u32) -> Option<TupleShape> {
        let (first, t1, h1, mut pushes) = match self.arena.get(t) {
            Term::Let(x, b) if self.is_proj(x, Prim::Fst, h) => (true, b, h + 1, vec![true]),
            _ => (false, t, h, vec![]),
        };
        match self.arena.get(t1) {
            Term::Let(x, b2) if self.is_proj(x, Prim::Snd, h1) => {
                if let Some(inner) = self.tuple_shape(b2, 0) {
                    let tail_index = u32::try_from(inner.pushes.len()).ok()?;
                    if !self.uses(inner.rest, tail_index) {
                        let mut named = vec![first];
                        named.extend(inner.named);
                        pushes.push(false);
                        pushes.extend(inner.pushes);
                        return Some(TupleShape {
                            named,
                            pushes,
                            rest: inner.rest,
                        });
                    }
                }
                pushes.push(true);
                Some(TupleShape {
                    named: vec![first, true],
                    pushes,
                    rest: b2,
                })
            }
            _ if first => Some(TupleShape {
                named: vec![true, false],
                pushes,
                rest: t1,
            }),
            _ => None,
        }
    }

    /// Tries to read `body` (just inside a binder for the variable at index 0) as a
    /// tuple destructuring of that variable that leaves it otherwise unused.
    fn tuple_binder(&mut self, body: TermId) -> Option<TupleShape> {
        if !self.sugar {
            return None;
        }
        let shape = self.tuple_shape(body, 0)?;
        // Every hidden binder (the variable itself and any tails) must be unused.
        let total = u32::try_from(shape.pushes.len()).ok()?;
        if self.uses(shape.rest, total) {
            return None;
        }
        for (i, named) in shape.pushes.iter().enumerate() {
            let index = total - 1 - u32::try_from(i).ok()?;
            if !named && self.uses(shape.rest, index) {
                return None;
            }
        }
        Some(shape)
    }

    /// Pushes the binders of a recognised tuple shape and returns the pattern.
    fn push_tuple(&mut self, shape: &TupleShape, hint: &str) -> Binding {
        self.names.push(HIDDEN.to_owned());
        let mut fresh_names = vec![];
        for &named in &shape.pushes {
            if named {
                let n = self.fresh(hint);
                self.names.push(n.clone());
                fresh_names.push(n);
            } else {
                self.names.push(HIDDEN.to_owned());
            }
        }
        let mut fresh_names = fresh_names.into_iter();
        let comps = shape
            .named
            .iter()
            .map(|&named| {
                if named {
                    fresh_names.next().unwrap_or_else(|| "_".to_owned())
                } else {
                    "_".to_owned()
                }
            })
            .collect();
        Binding::Tuple(comps)
    }

    /// Collects up to `max` lambdas starting at `id`, with tuple binders. Pushes
    /// their names; returns the bindings, the body and how many names were pushed.
    fn params(&mut self, id: TermId, max: usize, hint: &str) -> (Vec<Binding>, TermId, usize) {
        let mark = self.names.len();
        let mut out = vec![];
        let mut cur = id;
        while out.len() < max {
            let Term::Lam(b) = self.arena.get(cur) else {
                break;
            };
            if let Some(shape) = self.tuple_binder(b) {
                out.push(self.push_tuple(&shape, hint));
                cur = shape.rest;
            } else if self.uses(b, 0) {
                let n = self.fresh(hint);
                self.names.push(n.clone());
                out.push(Binding::Name(n));
                cur = b;
            } else {
                self.names.push(HIDDEN.to_owned());
                out.push(Binding::Name("_".to_owned()));
                cur = b;
            }
        }
        (out, cur, self.names.len() - mark)
    }

    fn pop(&mut self, n: usize) {
        let len = self.names.len();
        self.names.truncate(len - n);
    }

    fn wrap(s: String, prec: u8, need: u8) -> String {
        if prec < need {
            format!("({s})")
        } else {
            s
        }
    }

    fn spine(&self, id: TermId) -> (TermId, Vec<TermId>) {
        let mut args = vec![];
        let mut head = id;
        while let Term::App(f, a) = self.arena.get(head) {
            args.push(a);
            head = f;
        }
        args.reverse();
        (head, args)
    }

    fn is_trivial(&self, id: TermId) -> bool {
        matches!(
            self.arena.get(id),
            Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit
        )
    }

    fn go(&mut self, id: TermId, need: u8) -> String {
        match self.arena.get(id) {
            Term::Var(i) => {
                let i = i as usize;
                if i < self.names.len() {
                    self.names[self.names.len() - 1 - i].clone()
                } else {
                    format!("free{}", i - self.names.len())
                }
            }
            Term::Int(n) if n < 0 => Self::wrap(n.to_string(), P_UNARY, need),
            Term::Int(n) => n.to_string(),
            Term::Bool(b) => b.to_string(),
            Term::Unit => "()".to_owned(),
            Term::Prim(p) => match p {
                Prim::Nil => "[]".to_owned(),
                Prim::None => "None".to_owned(),
                Prim::Leaf => "Leaf".to_owned(),
                Prim::Mod => "( mod )".to_owned(),
                p => p.name().to_owned(),
            },
            Term::Lib(l) => (self.lib_name)(l),
            Term::Lam(_) => {
                let (ps, body, pushed) = self.params(id, usize::MAX, "x");
                let b = self.go(body, P_OPEN);
                self.pop(pushed);
                let ps: Vec<String> = ps.iter().map(Binding::text).collect();
                Self::wrap(format!("fun {} -> {b}", ps.join(" ")), P_OPEN, need)
            }
            Term::Let(x, b) => self.let_(x, b, need),
            Term::If(c, t, e) => {
                let c = self.go(c, P_OPEN);
                let t = self.go(t, P_OPEN);
                let e = self.go(e, P_OPEN);
                Self::wrap(format!("if {c} then {t} else {e}"), P_OPEN, need)
            }
            Term::And(a, b) => {
                let a = self.go(a, P_AND + 1);
                let b = self.go(b, P_AND);
                Self::wrap(format!("{a} && {b}"), P_AND, need)
            }
            Term::Or(a, b) => {
                let a = self.go(a, P_OR + 1);
                let b = self.go(b, P_OR);
                Self::wrap(format!("{a} || {b}"), P_OR, need)
            }
            Term::App(..) => self.app(id, need),
        }
    }

    fn let_(&mut self, x: TermId, b: TermId, need: u8) -> String {
        if self.sugar {
            // a > b: (let a (lt b' $0)) with b' not using $0.
            let (head, args) = self.spine(b);
            if let (Term::Prim(p @ (Prim::Lt | Prim::Le)), [b1, v]) =
                (self.arena.get(head), &args[..])
            {
                if self.arena.get(*v) == Term::Var(0)
                    && !self.uses(*b1, 0)
                    && !self.is_trivial(x)
                    && !self.is_trivial(*b1)
                {
                    let lhs = self.go(x, P_CMP + 1);
                    self.names.push(HIDDEN.to_owned());
                    let rhs = self.go(*b1, P_CMP + 1);
                    self.pop(1);
                    let op = if p == Prim::Lt { ">" } else { ">=" };
                    return Self::wrap(format!("{lhs} {op} {rhs}"), P_CMP, need);
                }
            }
            if let Some(shape) = self.tuple_binder(b) {
                let rhs = self.go(x, P_OPEN);
                let before = self.names.len();
                let pat = self.push_tuple(&shape, "v");
                let body = self.go(shape.rest, P_OPEN);
                let pushed = self.names.len() - before;
                self.pop(pushed);
                return Self::wrap(
                    format!("let {} = {rhs} in {body}", pat.text()),
                    P_OPEN,
                    need,
                );
            }
        }
        let text = if let Term::Lam(_) = self.arena.get(x) {
            let (ps, fbody, pushed) = self.params(x, usize::MAX, "x");
            let fb = self.go(fbody, P_OPEN);
            self.pop(pushed);
            let ps: Vec<String> = ps.iter().map(Binding::text).collect();
            let name = if self.uses(b, 0) {
                self.fresh("f")
            } else {
                "_".to_owned()
            };
            self.names.push(if name == "_" {
                HIDDEN.to_owned()
            } else {
                name.clone()
            });
            let body = self.go(b, P_OPEN);
            self.pop(1);
            if name == "_" {
                format!("let _ = fun {} -> {fb} in {body}", ps.join(" "))
            } else {
                format!("let {name} {} = {fb} in {body}", ps.join(" "))
            }
        } else {
            let rhs = self.go(x, P_OPEN);
            let name = if self.uses(b, 0) {
                self.fresh("v")
            } else {
                "_".to_owned()
            };
            self.names.push(if name == "_" {
                HIDDEN.to_owned()
            } else {
                name.clone()
            });
            let body = self.go(b, P_OPEN);
            self.pop(1);
            format!("let {name} = {rhs} in {body}")
        };
        Self::wrap(text, P_OPEN, need)
    }

    fn generic_app(&mut self, head: TermId, args: &[TermId], need: u8) -> String {
        let mut s = self.go(head, P_ATOM);
        for &a in args {
            s.push(' ');
            s.push_str(&self.go(a, P_ATOM));
        }
        Self::wrap(s, P_APP, need)
    }

    fn app(&mut self, id: TermId, need: u8) -> String {
        let (head, args) = self.spine(id);
        let Term::Prim(p) = self.arena.get(head) else {
            return self.generic_app(head, &args, need);
        };
        if let Some((op, prec, ln, rn)) = infix(p) {
            if args.len() >= 2 {
                if self.sugar && p == Prim::Cons && args.len() == 2 {
                    if let Some(items) = self.list_items(id) {
                        let parts: Vec<String> =
                            items.iter().map(|&t| self.go(t, P_OPEN)).collect();
                        return format!("[{}]", parts.join("; "));
                    }
                }
                let a = self.go(args[0], ln);
                let b = self.go(args[1], rn);
                let s = format!("{a} {op} {b}");
                if args.len() == 2 {
                    return Self::wrap(s, prec, need);
                }
                let mut s = format!("({s})");
                for &x in &args[2..] {
                    s.push(' ');
                    s.push_str(&self.go(x, P_ATOM));
                }
                return Self::wrap(s, P_APP, need);
            }
        }
        if p == Prim::Pair && args.len() == 2 {
            let mut items = vec![args[0]];
            let mut rest = args[1];
            if self.sugar {
                loop {
                    let (h2, a2) = self.spine(rest);
                    if self.arena.get(h2) == Term::Prim(Prim::Pair) && a2.len() == 2 {
                        items.push(a2[0]);
                        rest = a2[1];
                    } else {
                        break;
                    }
                }
            }
            items.push(rest);
            let parts: Vec<String> = items.iter().map(|&t| self.go(t, P_OPEN)).collect();
            return format!("({})", parts.join(", "));
        }
        if self.sugar {
            match (p, args.len()) {
                (Prim::Node, 3) => {
                    let parts: Vec<String> = args.iter().map(|&t| self.go(t, P_OPEN)).collect();
                    return Self::wrap(format!("Node ({})", parts.join(", ")), P_APP, need);
                }
                (Prim::Some, 1) => {
                    let a = self.go(args[0], P_ATOM);
                    return Self::wrap(format!("Some {a}"), P_APP, need);
                }
                (Prim::ListCase | Prim::OptionCase | Prim::TreeCase, 3) => {
                    if let Some(s) = self.match_(p, &args) {
                        return Self::wrap(s, P_OPEN, need);
                    }
                }
                _ => {}
            }
        }
        self.generic_app(head, &args, need)
    }

    fn list_items(&self, id: TermId) -> Option<Vec<TermId>> {
        let mut items = vec![];
        let mut cur = id;
        loop {
            match self.arena.get(cur) {
                Term::Prim(Prim::Nil) => return Some(items),
                Term::App(..) => {
                    let (h, a) = self.spine(cur);
                    if self.arena.get(h) != Term::Prim(Prim::Cons) || a.len() != 2 {
                        return None;
                    }
                    items.push(a[0]);
                    cur = a[1];
                }
                _ => return None,
            }
        }
    }

    fn match_(&mut self, p: Prim, args: &[TermId]) -> Option<String> {
        let (empty_pat, arity) = match p {
            Prim::ListCase => ("[]", 2),
            Prim::OptionCase => ("None", 1),
            _ => ("Leaf", 3),
        };
        let Term::Lam(e1) = self.arena.get(args[1]) else {
            return None;
        };
        if self.uses(e1, 0) {
            return None;
        }
        let scrutinee = self.go(args[0], P_OPEN);
        self.names.push(HIDDEN.to_owned());
        // Not the last arm: an open form here would swallow the next arm.
        let empty_body = self.go(e1, P_OR);
        self.pop(1);
        let (fields, body, pushed) = self.params(args[2], arity, "x");
        if fields.len() != arity {
            self.pop(pushed);
            return None;
        }
        let full_body = self.go(body, P_OPEN);
        self.pop(pushed);
        let f: Vec<String> = fields.iter().map(Binding::text).collect();
        let full_pat = match p {
            Prim::ListCase => format!("{} :: {}", f[0], f[1]),
            Prim::OptionCase => format!("Some {}", f[0]),
            _ => format!("Node ({}, {}, {})", f[0], f[1], f[2]),
        };
        Some(format!(
            "match {scrutinee} with | {empty_pat} -> {empty_body} | {full_pat} -> {full_body}"
        ))
    }
}
