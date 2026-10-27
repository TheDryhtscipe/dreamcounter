//! Term arena with hash-consing, types, and type inference for the dc language (spec §2, §3).

pub mod expand;
pub mod prim;
pub mod sexpr;
pub mod term;
pub mod types;

pub use expand::{expand, ExpandError, Expander, LibDef};
pub use prim::Prim;
pub use term::{Arena, LibId, Term, TermId};
pub use types::{Scheme, TyVar, Type, TypeChecker, TypeError, TypeErrorKind};
