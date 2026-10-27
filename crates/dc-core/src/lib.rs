//! Term arena with hash-consing, types, and type inference for the dc language (spec §2, §3).

pub mod prim;
pub mod sexpr;
pub mod term;

pub use prim::Prim;
pub use term::{Arena, LibId, Term, TermId};
