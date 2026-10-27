//! Lexer, parser, desugarer and pretty-printer for dc surface syntax (spec §10).
//!
//! The surface language is a small OCaml-like subset, written mostly by an LLM.
//! A program is a sequence of top-level definitions `let name args = expr`;
//! comments run from `--` to the end of the line. See the grammar
//! below and the desugaring table in [`desugar`].
//!
//! # Use
//!
//! ```
//! use dc_core::{sexpr, Arena};
//! use dc_surface::TopLevel;
//!
//! let mut env = TopLevel::new();
//! // Definitions added first are in scope for later ones. The standard prelude
//! // is added as library references instead: see [`prelude`].
//! env.add_source("helpers", "let twice f x = f (f x)").unwrap();
//! env.add_source("program", "let main = twice (fun x -> x * 2) 5").unwrap();
//! let mut arena = Arena::new();
//! let main = env.close_definition(&mut arena, "main").unwrap();
//! assert_eq!(
//!     sexpr::print(&arena, main),
//!     "(let (lam (lam ($1 ($1 $0)))) ($0 (lam (mul $0 2)) 5))"
//! );
//! ```
//!
//! A closed program is the definitions its entry needs, as nested `let`s around
//! the entry, in source order; definitions it does not use are dropped. Errors
//! ([`SurfaceError`]) give the source name, line and column and say what was
//! expected, in words meant for the writer to act on.
//!
//! # Grammar
//!
//! ```text
//! program ::= ('let' name param* '=' expr)*
//! expr    ::= 'let' name param* '=' expr 'in' expr
//!           | 'let' ('(' binder, ... ')' | '_') '=' expr 'in' expr
//!           | 'fun' param+ '->' expr
//!           | 'if' expr 'then' expr 'else' expr
//!           | 'match' expr 'with' '|'? pat '->' expr ('|' pat '->' expr)*
//!           | expr op expr | '-' expr | expr expr+ | atom
//! op      ::= || && (right) | = == < <= > >= (non-chaining) | :: (right)
//!           | + - (left) | * / mod (left)          -- loosest to tightest
//! atom    ::= int | name | true | false | () | (expr) | (expr, expr, ...)
//!           | [] | [expr; expr; ...] | None | Some | Leaf | Node | ( op )
//! param   ::= name | _ | () | (name-or-_, name-or-_, ...)
//! pat     ::= [] | p :: p | None | Some p | Leaf | Node (p, p, p) | _
//!             -- p is a param: patterns are one level deep
//! ```
//!
//! `Node (l, x, r)` as an expression is `node l x r` (the fields are not built into
//! a tuple); `Node l x r` means the same. `not`, `fst`, `foldl`, `bfix` and every
//! other primitive are ordinary names ([`dc_core::Prim::from_name`]).

pub mod ast;
pub mod desugar;
mod error;
mod lexer;
mod parser;
pub mod prelude;
pub mod pretty;

use dc_core::Arena;

pub use desugar::{compile_expression, TopLevel};
pub use error::{ErrorKind, Span, SurfaceError};
pub use pretty::pretty;

/// The deepest nesting the front end accepts; deeper input is an
/// [`ErrorKind::TooDeep`] error instead of a stack overflow, which would abort the
/// process.
///
/// Brackets, operators, `let`/`fun`/`if`/`match`, and each element of a list,
/// tuple, argument list or parameter list count towards it (see the parser's
/// module docs for the exact measure). Anything within the limit parses,
/// desugars and pretty-prints on a 1 MiB stack in a debug build. Hand-written
/// programs nest far less deeply than this; a long list literal is the case most
/// likely to reach it.
pub const MAX_DEPTH: usize = 100;

/// Parses a program into its surface definitions without desugaring them.
pub fn parse_program(source_name: &str, text: &str) -> Result<Vec<ast::Def>, SurfaceError> {
    parser::parse_program(&error::Source::new(source_name, text))
}

/// Parses one expression into surface syntax without desugaring it.
pub fn parse_expression(source_name: &str, text: &str) -> Result<ast::Expr, SurfaceError> {
    parser::parse_expression(&error::Source::new(source_name, text))
}

/// Compiles `text`, a whole program, and closes it over the definition `entry`.
/// Only primitives are in scope besides the program's own definitions; use
/// [`TopLevel`] to compile against a prelude.
pub fn compile_program(
    arena: &mut Arena,
    source_name: &str,
    text: &str,
    entry: &str,
) -> Result<dc_core::TermId, SurfaceError> {
    let mut env = TopLevel::new();
    env.add_source(source_name, text)?;
    env.close_definition(arena, entry)
}
