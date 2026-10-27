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

use std::fmt;

mod error;
mod infer;
mod parse;
#[cfg(test)]
mod tests;

pub use error::{TypeError, TypeErrorKind};
pub use infer::{check, infer, prim_scheme, LibEnv, TypeChecker};
pub use parse::TypeParseError;

/// A type variable. It prints as `a`, `b`, ..., `z`, `a1`, `b1`, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TyVar(pub u32);

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
    /// repeats.
    pub fn vars(&self) -> Vec<TyVar> {
        let mut out = vec![];
        self.collect_vars(&mut out);
        out
    }

    fn collect_vars(&self, out: &mut Vec<TyVar>) {
        match self {
            Type::Int | Type::Bool | Type::Unit => {}
            Type::Var(v) => {
                if !out.contains(v) {
                    out.push(*v);
                }
            }
            Type::List(a) | Type::Option(a) | Type::Array(a) | Type::Tree(a) => a.collect_vars(out),
            Type::Pair(a, b) | Type::Arrow(a, b) => {
                a.collect_vars(out);
                b.collect_vars(out);
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
    /// in `keys`. Entries of `keys` that do not occur in `ty` are dropped.
    pub fn with_keys(ty: Type, keys: &[TyVar]) -> Scheme {
        let order = ty.vars();
        let rename = |v: TyVar| {
            let i = order
                .iter()
                .position(|&w| w == v)
                .expect("every variable of the type is in its own variable list");
            TyVar(u32::try_from(i).expect("fewer than 2^32 type variables"))
        };
        let ty = ty.map_vars(&mut |v| rename(v));
        let mut keys: Vec<TyVar> = keys
            .iter()
            .filter(|v| order.contains(v))
            .map(|&v| rename(v))
            .collect();
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
