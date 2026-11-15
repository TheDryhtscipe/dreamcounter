//! The prelude: `map`, `filter`, `append` and the other list functions of spec §10,
//! written in dc (decision D7).
//!
//! The source is [`PRELUDE_SOURCE`]. [`Prelude::load`] compiles each definition to
//! a closed core term and infers its type scheme; a definition that does not
//! type-check is an error, so a loaded prelude is always well typed.
//!
//! # Representation
//!
//! A program compiled against the prelude ([`Prelude::top_level`] with
//! [`PreludeMode::On`]) refers to a prelude function `f` as the library reference
//! `Term::Lib(id)`, where `id` is `f`'s id in the reserved block that starts at
//! [`PRELUDE_LIB_BASE`]. So prelude names stay identifiable in core terms:
//!
//! - The type checker gives `id` the scheme of `f`: [`Prelude`] is a
//!   [`LibEnv`], and [`Prelude::with_libs`] layers it over learned abstractions.
//! - The Stitch encoding prints `id` as a token named `f` (give `dc_stitch` a
//!   `LibNames` table built from [`Prelude::entries`]), so the compressor treats
//!   `map` as an opaque primitive and never sees its body.
//! - Before evaluation, [`Prelude::expand`] replaces each prelude reference by
//!   `f`'s closed definition.
//!
//! **A prelude reference is a function, not a macro.** Learned abstractions
//! (spec §8) have macro semantics: an application is replaced by the substituted
//! body. A prelude reference is replaced by the closed term of its definition
//! (a `lam`, or a literal), and applying it is ordinary application, charged one
//! β step per argument. Evaluating a `lam` or a literal is free, so this costs
//! exactly what the same definition costs as a top-level `let` in the program:
//! the step count of a prelude function is the honest cost of its definition.
//! Learned abstractions keep the ids below the block (`LibId(0)`, `LibId(1)`,
//! ..., which matches Stitch's `fn_0`, `fn_1`, ...), and [`Prelude::expand`]
//! leaves every id outside the block alone.
//!
//! # Order of expansion
//!
//! Expand prelude references **first**: run [`Prelude::expand`] on the program
//! and on each learned abstraction's closed term `λ^k. body`, then give the
//! results to the learned-abstraction expander. That expander then never sees a
//! prelude id, so it can reject every id it does not define. The order is sound
//! because every prelude definition is closed and contains no library
//! references: substituting arguments into a learned body never reaches inside
//! one, so expanding prelude references before or after the learned
//! abstractions gives the same term. [`Prelude::expand`] keeps a learned term's
//! leading lambdas, so its arity is unchanged.
//!
//! # Ablation
//!
//! [`PreludeMode::Off`] compiles programs with only the primitives in scope, so a
//! use of `map` is an unknown-name error. [`PreludeMode::On`] is the default.
//!
//! ```
//! use dc_core::Arena;
//! use dc_eval::{eval, Outcome, UNLIMITED};
//! use dc_surface::prelude::{Prelude, PreludeMode};
//!
//! let prelude = Prelude::load().unwrap();
//! let env = prelude.top_level(PreludeMode::On);
//! let mut arena = Arena::new();
//! let t = env
//!     .close_expression(&mut arena, "example", "sum (map (fun x -> x * x) [1; 2; 3])")
//!     .unwrap();
//! let t = prelude.expand(&mut arena, t);
//! assert!(matches!(eval(&arena, t, UNLIMITED), Outcome::Value(..)));
//!
//! let off = prelude.top_level(PreludeMode::Off);
//! assert!(off.close_expression(&mut arena, "example", "map").is_err());
//! ```

use std::collections::HashMap;
use std::fmt;

use dc_core::types::{infer, LibEnv};
use dc_core::{term, Arena, LibId, Scheme, Term, TermId, TypeError};

use crate::desugar::copy_term;
use crate::{SurfaceError, TopLevel};

/// The prelude's dc source.
pub const PRELUDE_SOURCE: &str = include_str!("prelude.dc");

/// The first [`LibId`] of the block reserved for prelude functions. The prelude's
/// `i`-th definition is `LibId(PRELUDE_LIB_BASE + i)`.
pub const PRELUDE_LIB_BASE: u32 = 0xFFFF_0000;

/// Whether programs are compiled with the prelude in scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PreludeMode {
    /// The prelude's names are in scope (the default).
    #[default]
    On,
    /// Only the primitives are in scope: the ablation of spec §10.
    Off,
}

/// One prelude definition.
#[derive(Clone, Debug)]
pub struct PreludeEntry {
    name: String,
    lib: LibId,
    term: TermId,
    scheme: Scheme,
}

impl PreludeEntry {
    /// The name the definition is written under, for example `map`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The library id that programs use to refer to it.
    pub fn lib(&self) -> LibId {
        self.lib
    }

    /// Its principal type scheme.
    pub fn scheme(&self) -> &Scheme {
        &self.scheme
    }
}

/// Why the prelude could not be loaded.
#[derive(Clone, Debug)]
pub enum PreludeError {
    /// The source did not parse or desugar.
    Surface(SurfaceError),
    /// A definition did not type-check.
    Type {
        /// The definition.
        name: String,
        /// The type error.
        error: TypeError,
    },
    /// A name is defined twice.
    Duplicate(String),
}

impl fmt::Display for PreludeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PreludeError::Surface(e) => write!(f, "prelude: {e}"),
            PreludeError::Type { name, error } => {
                write!(
                    f,
                    "prelude definition '{name}' does not type-check: {error}"
                )
            }
            PreludeError::Duplicate(name) => write!(f, "prelude defines '{name}' twice"),
        }
    }
}

impl std::error::Error for PreludeError {}

impl From<SurfaceError> for PreludeError {
    fn from(e: SurfaceError) -> PreludeError {
        PreludeError::Surface(e)
    }
}

/// The compiled, type-checked prelude.
#[derive(Clone, Debug)]
pub struct Prelude {
    /// Holds the closed term of every definition.
    arena: Arena,
    entries: Vec<PreludeEntry>,
}

impl Prelude {
    /// Compiles and type-checks [`PRELUDE_SOURCE`].
    pub fn load() -> Result<Prelude, PreludeError> {
        Prelude::from_source("prelude", PRELUDE_SOURCE)
    }

    /// Compiles and type-checks `text` as a prelude. Each definition becomes the
    /// closed term [`TopLevel::close_definition`] gives it, so a definition that
    /// uses an earlier one carries that one's definition in a `let`.
    pub fn from_source(source_name: &str, text: &str) -> Result<Prelude, PreludeError> {
        let mut top = TopLevel::new();
        let names = top.add_source(source_name, text)?;
        let mut arena = Arena::new();
        let mut entries: Vec<PreludeEntry> = Vec::with_capacity(names.len());
        for (i, name) in names.into_iter().enumerate() {
            if entries.iter().any(|e| e.name == name) {
                return Err(PreludeError::Duplicate(name));
            }
            let index = u32::try_from(i).expect("fewer than 2^32 prelude definitions");
            let term = top.close_definition(&mut arena, &name)?;
            let scheme = infer(&arena, term, &NO_LIBS[..]).map_err(|error| PreludeError::Type {
                name: name.clone(),
                error,
            })?;
            entries.push(PreludeEntry {
                name,
                lib: LibId(PRELUDE_LIB_BASE + index),
                term,
                scheme,
            });
        }
        Ok(Prelude { arena, entries })
    }

    /// The definitions, in source order.
    pub fn entries(&self) -> &[PreludeEntry] {
        &self.entries
    }

    fn entry(&self, id: LibId) -> Option<&PreludeEntry> {
        let i = id.0.checked_sub(PRELUDE_LIB_BASE)?;
        self.entries.get(usize::try_from(i).ok()?)
    }

    /// The entry named `name`.
    pub fn get(&self, name: &str) -> Option<&PreludeEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// The name of prelude reference `id`, or `None` if `id` is not one.
    pub fn name(&self, id: LibId) -> Option<&str> {
        self.entry(id).map(|e| e.name.as_str())
    }

    /// The closed definition of the entry named `name`, copied into `arena`.
    pub fn definition(&self, arena: &mut Arena, name: &str) -> Option<TermId> {
        let e = self.get(name)?;
        Some(copy_term(&self.arena, e.term, arena, &mut HashMap::new()))
    }

    /// The names programs are compiled against: the prelude's names when `mode`
    /// is [`PreludeMode::On`], none otherwise. Add the program with
    /// [`TopLevel::add_source`]; its own definitions shadow prelude names.
    pub fn top_level(&self, mode: PreludeMode) -> TopLevel {
        let mut top = TopLevel::new();
        if mode == PreludeMode::On {
            let mut scratch = Arena::new();
            for e in &self.entries {
                let r = scratch.lib(e.lib);
                top.define_inline(&scratch, &e.name, r)
                    .expect("a library reference is closed");
            }
        }
        top
    }

    /// `term` with every prelude reference replaced by its closed definition, ready
    /// for evaluation. Other library references are left as they are.
    pub fn expand(&self, arena: &mut Arena, term: TermId) -> TermId {
        // Definitions copied into `arena`, and copies of `arena`'s own nodes.
        let mut from_prelude: HashMap<TermId, TermId> = HashMap::new();
        let mut done: HashMap<TermId, TermId> = HashMap::new();
        let mut stack = vec![(term, false)];
        while let Some((id, children_done)) = stack.pop() {
            if done.contains_key(&id) {
                continue;
            }
            let node = arena.get(id);
            if !children_done {
                stack.push((id, true));
                for c in term::children(&node) {
                    if !done.contains_key(&c) {
                        stack.push((c, false));
                    }
                }
                continue;
            }
            let m = |c: TermId| done[&c];
            let new = match node {
                Term::Lib(l) => match self.entry(l) {
                    Some(e) => copy_term(&self.arena, e.term, arena, &mut from_prelude),
                    None => id,
                },
                Term::Var(_) | Term::Int(_) | Term::Bool(_) | Term::Unit | Term::Prim(_) => id,
                Term::Lam(b) => arena.lam(m(b)),
                Term::App(a, b) => {
                    let (a, b) = (m(a), m(b));
                    arena.app(a, b)
                }
                Term::Let(a, b) => {
                    let (a, b) = (m(a), m(b));
                    arena.let_(a, b)
                }
                Term::And(a, b) => {
                    let (a, b) = (m(a), m(b));
                    arena.and(a, b)
                }
                Term::Or(a, b) => {
                    let (a, b) = (m(a), m(b));
                    arena.or(a, b)
                }
                Term::If(a, b, c) => {
                    let (a, b, c) = (m(a), m(b), m(c));
                    arena.if_(a, b, c)
                }
            };
            done.insert(id, new);
        }
        done[&term]
    }

    /// A [`LibEnv`] that gives prelude references their schemes and looks every
    /// other id up in `libs` (for example learned abstractions).
    pub fn with_libs<'a, L: LibEnv + ?Sized>(&'a self, libs: &'a L) -> WithPrelude<'a, L> {
        WithPrelude {
            prelude: self,
            libs,
        }
    }

    /// A fingerprint of the prelude's definitions: every definition's position,
    /// name and closed term, in source order. Comments and layout do not change
    /// it; reordering, renaming, adding or changing a definition does. It does
    /// not cover what the primitives do (a primitive is hashed by its name) or
    /// [`PRELUDE_LIB_BASE`].
    ///
    /// A prelude reference is positional (`LibId(PRELUDE_LIB_BASE + i)`), so a
    /// stored program that uses one is only meaningful against the prelude it
    /// was compiled with. Store the fingerprint with such programs and compare
    /// it before using them: two definitions of the same type, such as `max`
    /// and `min`, can swap ids without a type error.
    ///
    /// The hash is 64-bit FNV-1a over the term structure, so it is the same in
    /// every process and on every platform. It guards against mistakes, not
    /// against a crafted prelude.
    pub fn fingerprint(&self) -> u64 {
        let mut h = Fnv::new();
        h.u64(self.entries.len() as u64);
        let mut memo = HashMap::new();
        for e in &self.entries {
            h.bytes(e.name.as_bytes());
            h.u64(term_hash(&self.arena, e.term, &mut memo));
        }
        h.finish()
    }
}

/// 64-bit FNV-1a.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn u64(&mut self, x: u64) {
        x.to_le_bytes().into_iter().for_each(|b| self.byte(b));
    }

    /// `b`, prefixed by its length so that two strings cannot run together.
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        b.iter().for_each(|&b| self.byte(b));
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// The structural hash of `term`: a node's tag and contents, and its children's
/// hashes. `memo` holds the hash of each node done, so a shared subterm is hashed
/// once. Iterative, so a deep term cannot overflow the stack.
fn term_hash(arena: &Arena, term: TermId, memo: &mut HashMap<TermId, u64>) -> u64 {
    let mut stack = vec![(term, false)];
    while let Some((id, children_done)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        let node = arena.get(id);
        if !children_done {
            stack.push((id, true));
            stack.extend(term::children(&node).into_iter().map(|c| (c, false)));
            continue;
        }
        let mut h = Fnv::new();
        let (tag, payload): (u8, u64) = match node {
            Term::Var(i) => (0, u64::from(i)),
            Term::Lam(_) => (1, 0),
            Term::App(..) => (2, 0),
            Term::Let(..) => (3, 0),
            Term::Int(i) => (4, u64::from_le_bytes(i.to_le_bytes())),
            Term::Bool(b) => (5, u64::from(b)),
            Term::Unit => (6, 0),
            Term::Prim(p) => {
                h.bytes(p.name().as_bytes());
                (7, 0)
            }
            Term::Lib(LibId(l)) => (8, u64::from(l)),
            Term::If(..) => (9, 0),
            Term::And(..) => (10, 0),
            Term::Or(..) => (11, 0),
        };
        h.byte(tag);
        h.u64(payload);
        for c in term::children(&node) {
            h.u64(memo[&c]);
        }
        memo.insert(id, h.finish());
    }
    memo[&term]
}

const NO_LIBS: [Scheme; 0] = [];

impl LibEnv for Prelude {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        self.entry(id).map(|e| &e.scheme)
    }
}

/// The prelude layered over other library schemes; see [`Prelude::with_libs`].
#[derive(Debug)]
pub struct WithPrelude<'a, L: ?Sized> {
    prelude: &'a Prelude,
    libs: &'a L,
}

impl<L: LibEnv + ?Sized> LibEnv for WithPrelude<'_, L> {
    fn lib_scheme(&self, id: LibId) -> Option<&Scheme> {
        match self.prelude.entry(id) {
            Some(e) => Some(&e.scheme),
            None => self.libs.lib_scheme(id),
        }
    }
}
