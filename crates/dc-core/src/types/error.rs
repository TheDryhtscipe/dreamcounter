//! Type errors, written to be read and acted on by a language model.
//!
//! Every error names the offending subterm in core syntax and says where it sits
//! ("argument 2 of `foldl`"). Types inside one error share one renaming of type
//! variables to `a`, `b`, `c`, ..., so the same letter means the same variable in
//! the expected and found types.

use std::fmt;

use super::Type;
use crate::term::{LibId, TermId};

/// What went wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TypeErrorKind {
    /// The subterm has type `found` where `expected` was required. `conflict` is the
    /// innermost pair of types that could not be made equal, when it differs from
    /// the whole types. Types are shown as they stand when unification failed, so
    /// parts unified before the clash already show their unified form.
    Mismatch {
        /// The type the context requires.
        expected: Type,
        /// The type the subterm has.
        found: Type,
        /// The innermost clash, e.g. `Int` against `Bool` inside two list types.
        conflict: Option<(Type, Type)>,
    },
    /// Unification would need `var = ty` where `ty` contains `var`, an infinite type.
    /// A typical cause is applying a variable to itself.
    InfiniteType {
        /// The variable.
        var: Type,
        /// The type that contains it.
        ty: Type,
    },
    /// A term is applied to more arguments than its type accepts.
    NotAFunction {
        /// The type left after `accepted` arguments, which is not a function.
        found: Type,
        /// How many arguments the term is applied to.
        applied_to: usize,
        /// How many arguments it accepts.
        accepted: usize,
    },
    /// A de Bruijn variable points past the binders in scope.
    UnboundVariable {
        /// The de Bruijn index.
        index: u32,
        /// How many binders are in scope at that point.
        in_scope: u32,
    },
    /// A library reference that the library environment does not define.
    UnknownLib(LibId),
    /// A type that is not a valid `memorec` key (`Int`, `Bool`, `Unit` or a tuple of
    /// those) was used where a key is required.
    BadMemoKey {
        /// The offending type, or the part of it that is not a key type.
        found: Type,
    },
    /// A type involved is too large to build or print: more than `max_nodes`
    /// constructors in total, or nested more than `max_depth` levels deep. Types that
    /// share structure can be exponentially larger written out than in the checker.
    TypeTooLarge {
        /// The limit on the number of type constructors.
        max_nodes: usize,
        /// The limit on nesting depth.
        max_depth: usize,
    },
    /// Checking the term took more than `limit` units of work (see
    /// [`TypeChecker::set_work_budget`](super::TypeChecker::set_work_budget)), so it
    /// was stopped. The term may be well typed; it is refused because terms whose
    /// shared subterms see many differently typed polymorphic bindings can take
    /// time exponential in their size.
    TooMuchWork {
        /// The work budget that was exceeded.
        limit: u64,
    },
}

/// A type error at one subterm.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    /// What went wrong. Boxed to keep `Result<_, TypeError>` small on the
    /// success path.
    pub kind: Box<TypeErrorKind>,
    /// The offending subterm.
    pub term: TermId,
    /// The offending subterm in core syntax, shortened if very long.
    pub term_text: String,
    /// Where the subterm sits, e.g. "argument 2 of `foldl`". Empty when the error is
    /// about the subterm itself.
    pub context: String,
    /// Extra advice, when there is some.
    pub note: Option<String>,
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("type error")?;
        if !self.context.is_empty() {
            write!(f, " in {}", self.context)?;
        }
        f.write_str(": ")?;
        match &*self.kind {
            TypeErrorKind::Mismatch {
                expected,
                found,
                conflict,
            } => {
                write!(f, "expected `{expected}`, found `{found}`")?;
                if let Some((e, g)) = conflict {
                    write!(f, "\n  conflict: `{e}` does not match `{g}`")?;
                }
            }
            TypeErrorKind::InfiniteType { var, ty } => write!(
                f,
                "infinite type: `{var}` would have to equal `{ty}`, which contains it \
                 (is something applied to itself?)"
            )?,
            TypeErrorKind::NotAFunction {
                found,
                applied_to,
                accepted,
            } => {
                if *accepted == 0 {
                    write!(
                        f,
                        "this has type `{found}`, which is not a function, \
                         but it is applied to {applied_to} argument(s)"
                    )?;
                } else {
                    write!(
                        f,
                        "this is applied to {applied_to} argument(s), but it takes only \
                         {accepted}: after {accepted} its type is `{found}`, which is not \
                         a function"
                    )?;
                }
            }
            TypeErrorKind::UnboundVariable { index, in_scope } => write!(
                f,
                "unbound variable `${index}`: only {in_scope} variable(s) are in scope \
                 here (`$0` to `${}`)",
                in_scope.saturating_sub(1)
            )?,
            TypeErrorKind::UnknownLib(LibId(l)) => {
                write!(f, "unknown library abstraction `lib_{l}`")?
            }
            TypeErrorKind::BadMemoKey { found } => write!(
                f,
                "`{found}` cannot be a memorec key; keys must be Int, Bool, Unit or \
                 tuples of those"
            )?,
            TypeErrorKind::TypeTooLarge {
                max_nodes,
                max_depth,
            } => write!(
                f,
                "the type here is too large (more than {max_nodes} type constructors \
                 written out, or nested more than {max_depth} deep); a common cause \
                 is a function that duplicates its argument's type, such as \
                 `(lam (pair $0 $0))`, applied many times"
            )?,
            TypeErrorKind::TooMuchWork { limit } => write!(
                f,
                "type checking this term took more than {limit} units of work, so it \
                 was stopped; a common cause is a shared subterm used under many \
                 different choices of polymorphic `let`-bound helpers"
            )?,
        }
        write!(f, "\n  term: {}", self.term_text)?;
        if let Some(note) = &self.note {
            write!(f, "\n  note: {note}")?;
        }
        Ok(())
    }
}

impl std::error::Error for TypeError {}
