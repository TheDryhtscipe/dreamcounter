//! Types, type schemes and Hindley-Milner type inference (spec §2, §5, §6).
//!
//! ```text
//! τ ::= Int | Bool | Unit | (τ, τ) | List τ | Option τ | Array τ | Tree τ | τ -> τ | α
//! ```
//!
//! A [`Scheme`] is a type whose variables are all universally quantified, plus a set of
//! **`Key` constraints**. `Key a` says that `a` may only be instantiated to a valid
//! `memorec` key type: `Int`, `Bool`, `Unit`, or a tuple of key types (spec §6). The
//! constraint is how the key restriction survives let-polymorphism and library
//! abstractions: `memorec` itself has the scheme
//! `Key a => Int -> ((a -> b) -> a -> b) -> a -> b`.
//!
//! Text form, used by [`Display`](fmt::Display) and by [`str::parse`]:
//!
//! ```text
//! Key a, Key b => Int -> List a -> (a, b) -> Option (a -> b)
//! ```
//!
//! Arrows associate to the right. A type constructor applies to one atom, so
//! `List (List a)` needs its brackets. Tuples nest to the right: `(a, b, c)` is
//! `(a, (b, c))`.
//!
//! Inference lives in [`infer`], [`check`] and [`TypeChecker`].

use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::{Hash, Hasher};

use rustc_hash::FxHashMap;

mod error;
mod infer;
mod parse;
#[cfg(test)]
mod tests;

pub use error::{TypeError, TypeErrorKind};
pub use infer::{
    check, infer, prim_scheme, LibEnv, TypeChecker, DEFAULT_WORK_BUDGET, MAX_TYPE_DEPTH,
    MAX_TYPE_NODES,
};
pub use parse::TypeParseError;

/// A type variable. It prints as `a`, `b`, ..., `z`, `a1`, `b1`, ...
#[derive(Clone, Copy, Debug, Eq, PartialOrd, Ord)]
pub struct TyVar(pub u32);

/// Equality on the number. In test builds each comparison is counted in `probes`,
/// so the tests can measure the work of a lookup whether it is a hash-map probe or a
/// scan of a list.
impl PartialEq for TyVar {
    fn eq(&self, other: &TyVar) -> bool {
        #[cfg(test)]
        probes::add(1);
        self.0 == other.0
    }
}

/// Hashes the number, as a derived `Hash` would; it agrees with [`PartialEq`].
impl Hash for TyVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Display for TyVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let letter = char::from(b'a' + (self.0 % 26) as u8);
        let round = self.0 / 26;
        if round == 0 {
            write!(f, "{letter}")
        } else {
            write!(f, "{letter}{round}")
        }
    }
}

/// A dc type (spec §2).
///
/// **Depth.** Dropping, cloning, comparing, hashing, printing and type-checking
/// against a `Type` recurse on the host stack, one level per level of nesting. Types
/// parsed from text and types the checker produces are at most [`MAX_TYPE_DEPTH`]
/// deep, which these walks handle on an ordinary thread stack. A type built directly
/// from the variants or constructors here has no such bound, so code building one
/// from outside input must keep it within [`MAX_TYPE_DEPTH`] itself, or go through
/// the text form.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Type {
    /// 64-bit signed integers.
    Int,
    /// Booleans.
    Bool,
    /// The unit type.
    Unit,
    /// A pair. Longer tuples nest to the right.
    Pair(Box<Type>, Box<Type>),
    /// A list.
    List(Box<Type>),
    /// An option.
    Option(Box<Type>),
    /// A persistent array.
    Array(Box<Type>),
    /// A built-in binary tree.
    Tree(Box<Type>),
    /// A function.
    Arrow(Box<Type>, Box<Type>),
    /// A type variable.
    Var(TyVar),
}

impl Type {
    /// `(a, b)`.
    pub fn pair(a: Type, b: Type) -> Type {
        Type::Pair(Box::new(a), Box::new(b))
    }

    /// `List a`.
    pub fn list(a: Type) -> Type {
        Type::List(Box::new(a))
    }

    /// `Option a`.
    pub fn option(a: Type) -> Type {
        Type::Option(Box::new(a))
    }

    /// `Array a`.
    pub fn array(a: Type) -> Type {
        Type::Array(Box::new(a))
    }

    /// `Tree a`.
    pub fn tree(a: Type) -> Type {
        Type::Tree(Box::new(a))
    }

    /// `a -> b`.
    pub fn arrow(a: Type, b: Type) -> Type {
        Type::Arrow(Box::new(a), Box::new(b))
    }

    /// `a0 -> a1 -> ... -> ret`.
    pub fn fun(args: impl IntoIterator<Item = Type>, ret: Type) -> Type {
        let args: Vec<Type> = args.into_iter().collect();
        args.into_iter()
            .rev()
            .fold(ret, |acc, a| Type::arrow(a, acc))
    }

    /// The variable `TyVar(i)`.
    pub fn var(i: u32) -> Type {
        Type::Var(TyVar(i))
    }

    /// The type variables in order of first occurrence, left to right, without
    /// repeats. It takes one hash-map lookup per variable occurrence, so time linear
    /// in the size of the type for a well-spread hash.
    pub fn vars(&self) -> Vec<TyVar> {
        self.numbered_vars().0
    }

    /// The type variables in order of first occurrence, and each variable's position
    /// in that order.
    fn numbered_vars(&self) -> (Vec<TyVar>, FxHashMap<TyVar, u32>) {
        let mut order = vec![];
        let mut index = FxHashMap::default();
        self.collect_vars(&mut order, &mut index);
        (order, index)
    }

    fn collect_vars(&self, order: &mut Vec<TyVar>, index: &mut FxHashMap<TyVar, u32>) {
        match self {
            Type::Int | Type::Bool | Type::Unit => {}
            Type::Var(v) => {
                if let Entry::Vacant(e) = index.entry(*v) {
                    e.insert(u32::try_from(order.len()).expect("fewer than 2^32 type variables"));
                    order.push(*v);
                }
            }
            Type::List(a) | Type::Option(a) | Type::Array(a) | Type::Tree(a) => {
                a.collect_vars(order, index)
            }
            Type::Pair(a, b) | Type::Arrow(a, b) => {
                a.collect_vars(order, index);
                b.collect_vars(order, index);
            }
        }
    }

    /// Applies `f` to every variable.
    pub fn map_vars(&self, f: &mut impl FnMut(TyVar) -> TyVar) -> Type {
        match self {
            Type::Int => Type::Int,
            Type::Bool => Type::Bool,
            Type::Unit => Type::Unit,
            Type::Var(v) => Type::Var(f(*v)),
            Type::List(a) => Type::list(a.map_vars(f)),
            Type::Option(a) => Type::option(a.map_vars(f)),
            Type::Array(a) => Type::array(a.map_vars(f)),
            Type::Tree(a) => Type::tree(a.map_vars(f)),
            Type::Pair(a, b) => Type::pair(a.map_vars(f), b.map_vars(f)),
            Type::Arrow(a, b) => Type::arrow(a.map_vars(f), b.map_vars(f)),
        }
    }

    /// The number of arrows along the result spine: `Int -> (a -> b) -> c` has 2.
    pub fn arrow_count(&self) -> usize {
        match self {
            Type::Arrow(_, r) => 1 + r.arrow_count(),
            _ => 0,
        }
    }

    fn fmt_arrow(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Arrow(a, b) => {
                if matches!(**a, Type::Arrow(..)) {
                    f.write_str("(")?;
                    a.fmt_arrow(f)?;
                    f.write_str(")")?;
                } else {
                    a.fmt_app(f)?;
                }
                f.write_str(" -> ")?;
                b.fmt_arrow(f)
            }
            _ => self.fmt_app(f),
        }
    }

    fn fmt_app(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (name, arg) = match self {
            Type::List(a) => ("List", a),
            Type::Option(a) => ("Option", a),
            Type::Array(a) => ("Array", a),
            Type::Tree(a) => ("Tree", a),
            _ => return self.fmt_atom(f),
        };
        write!(f, "{name} ")?;
        arg.fmt_atom(f)
    }

    fn fmt_atom(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Int => f.write_str("Int"),
            Type::Bool => f.write_str("Bool"),
            Type::Unit => f.write_str("Unit"),
            Type::Var(v) => write!(f, "{v}"),
            Type::Pair(a, b) => {
                f.write_str("(")?;
                a.fmt_arrow(f)?;
                let mut rest = b;
                while let Type::Pair(x, y) = &**rest {
                    f.write_str(", ")?;
                    x.fmt_arrow(f)?;
                    rest = y;
                }
                f.write_str(", ")?;
                rest.fmt_arrow(f)?;
                f.write_str(")")
            }
            _ => {
                f.write_str("(")?;
                self.fmt_arrow(f)?;
                f.write_str(")")
            }
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_arrow(f)
    }
}

/// A type scheme: a type with every variable universally quantified, and `Key`
/// constraints on some of them.
///
/// Schemes are kept in a normal form: variables are numbered `0, 1, 2, ...` in order
/// of first occurrence and the key list is sorted. So `==` on schemes is equality up
/// to renaming of bound variables.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scheme {
    ty: Type,
    keys: Vec<TyVar>,
}

impl Scheme {
    /// Quantifies every variable of `ty`, with no `Key` constraints.
    pub fn new(ty: Type) -> Scheme {
        Scheme::with_keys(ty, &[])
    }

    /// Quantifies every variable of `ty` and puts a `Key` constraint on each variable
    /// in `keys`. Entries of `keys` that do not occur in `ty` are dropped. It takes one
    /// hash-map lookup per variable occurrence and per key, plus a sort of the keys
    /// that occur.
    pub fn with_keys(ty: Type, keys: &[TyVar]) -> Scheme {
        let (_, index) = ty.numbered_vars();
        let rename = |v: TyVar| index.get(&v).map(|&i| TyVar(i));
        let ty = ty.map_vars(&mut |v| {
            rename(v).expect("every variable of the type is in its own variable index")
        });
        let mut keys: Vec<TyVar> = keys.iter().filter_map(|&v| rename(v)).collect();
        keys.sort();
        keys.dedup();
        Scheme { ty, keys }
    }

    /// The body of the scheme. Its variables are the quantified ones.
    pub fn ty(&self) -> &Type {
        &self.ty
    }

    /// The variables carrying a `Key` constraint.
    pub fn keys(&self) -> &[TyVar] {
        &self.keys
    }

    /// The quantified variables, in order of first occurrence.
    pub fn vars(&self) -> Vec<TyVar> {
        self.ty.vars()
    }
}

impl From<Type> for Scheme {
    fn from(ty: Type) -> Scheme {
        Scheme::new(ty)
    }
}

impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, k) in self.keys.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "Key {k}")?;
        }
        if !self.keys.is_empty() {
            f.write_str(" => ")?;
        }
        write!(f, "{}", self.ty)
    }
}

/// A count of the equality comparisons made on type variables ([`TyVar`]) and on
/// variable names in the type parser, a measure of lookup work that the tests check
/// without a timer.
///
/// A hash-map lookup that finds its key makes about one comparison and a scan of a
/// list makes one per element passed, so lookup work that grows faster than linear
/// shows up in the count. Ordered comparisons (sorting, binary search) are not
/// counted, and neither is work that compares the bare numbers or strings.
#[cfg(test)]
pub(crate) mod probes {
    use std::cell::Cell;

    thread_local! {
        static PROBES: Cell<u64> = const { Cell::new(0) };
    }

    /// Records `n` comparisons.
    pub(crate) fn add(n: usize) {
        PROBES.with(|p| p.set(p.get() + n as u64));
    }

    /// The comparisons made on this thread since the last call.
    pub(crate) fn take() -> u64 {
        PROBES.with(|p| p.replace(0))
    }
}

#[cfg(test)]
mod var_tests;
