//! Serialisation of dc core terms to and from the Stitch format (spec §10).
//!
//! [Stitch](https://github.com/mlb2251/stitch) reads and writes programs as
//! s-expressions in the syntax of its `lambdas` crate:
//!
//! ```text
//! e ::= $i          de Bruijn variable
//!     | #i          abstraction parameter (only inside abstraction bodies)
//!     | (lam e)     lambda
//!     | (e e e ...) application, left-associated
//!     | token       any other run of characters without whitespace or parentheses
//! ```
//!
//! Stitch treats every token as an opaque primitive, compares tokens by spelling,
//! and prints left-nested applications flattened, so `((f a) b)` reads back as
//! `(f a b)`.
//!
//! # Encoding of core terms
//!
//! | Core term | Stitch form |
//! |---|---|
//! | `Var(i)` | `$i` |
//! | `Lam(b)` | `(lam b)` |
//! | `App(f, a)` | `(f a)`, spines flattened: `(f a b)` |
//! | `Let(x, b)` | `(let_ x (lam b))` |
//! | `If(c, t, e)` | `(if_ c t e)` |
//! | `And(a, b)` | `(and_ a b)` |
//! | `Or(a, b)` | `(or_ a b)` |
//! | `Int(n)` | `int_n`, for example `int_3`, `int_-3` |
//! | `Bool(b)` | `true`, `false` |
//! | `Unit` | `unit` |
//! | `Prim(p)` | the canonical name, for example `add`, `foldl` |
//! | `Lib(l)` | the name a [`LibNames`] table gives `l`, for example `fn_0` |
//!
//! `Let` binds a variable, so its body goes under a real `lam`. Stitch then sees
//! the binder and treats the body's `$0` as bound, as it must when it decides which
//! subterms are closed enough to abstract. `If`, `And` and `Or` are lazy special
//! forms, not functions; they print as reserved tokens applied to their operands,
//! which is enough for compression, and decoding restores the special form
//! exactly. The reserved tokens [`LET`], [`IF`], [`AND`], [`OR`] end in `_`, which
//! no primitive name does, and [`LibNames`] refuses them as abstraction names.
//!
//! Each integer literal is its own token, so Stitch treats `int_3` and `int_4` as
//! unrelated primitives. Stitch's tokeniser accepts `-` inside a token, so negative
//! literals are written with a plain minus sign (`int_-9223372036854775808` for
//! `i64::MIN`). Decoding accepts only the canonical spelling `int_{n}` that Rust's
//! `i64` `Display` gives, so each literal has exactly one token.
//!
//! [`decode`] inverts [`encode`]: for every term `t` (closed, or decoded with
//! [`decode_open`] at its free depth), `decode(encode(t)) == t`.
//!
//! # Stitch input that has no core equivalent
//!
//! Stitch may cut an abstraction or a metavariable through a special form, for
//! example the abstraction body `(if_ #0 #1)`, the argument `(let_ x)`, or
//! `(let_ #0 #1)` whose binder `lam` has become a parameter. These have no core
//! term, so decoding reports [`DecodeError::MalformedSpecialForm`]. Stitch's
//! `--no-curried-bodies` and `--no-curried-metavars` flags rule out the first two
//! shapes. The third has no Stitch flag, so the bridge must drop abstractions
//! that fail to decode, or Stitch must be changed to treat the `lam` under
//! `let_` as part of the pattern.
//!
//! # Library abstractions
//!
//! Stitch reports an abstraction as a name (for example `fn_0`), an arity `k` and
//! a body in which `#0 ... #(k-1)` are the parameters. A rewritten program applies
//! the name to its arguments in parameter order: `(fn_0 a0 a1)` stands for the body
//! with `#0 := a0` and `#1 := a1`. Each argument is a term of the use site's scope,
//! so a parameter that occurs under `d` binders inside the body stands for its
//! argument shifted up by `d`.
//!
//! [`decode_abstraction`] turns such a body into the closed core term
//! `λ^k. body'`, in which the parameter `#i`, found under `d` binders of the
//! body, becomes the de Bruijn variable `$(d + k - 1 - i)`. The first parameter is
//! bound by the outermost lambda. So `(lam (add #0 $0))` with arity 1 decodes to
//! `(lam (lam (add $1 $0)))`, and β-reducing `λ^k. body'` applied to `a0 ... a(k-1)`
//! gives exactly the substitution above (spec §8 macro semantics). Expanding
//! abstractions is the evaluator's job (bead P2.10), not this crate's.

pub mod codec;
pub mod expr;
pub mod names;

pub use codec::{
    decode, decode_abstraction, decode_open, encode, from_expr, to_expr, DecodeError, EncodeError,
    AND, IF, INT_PREFIX, LET, OR,
};
pub use expr::{Expr, ParseError};
pub use names::{LibNames, NameError};
