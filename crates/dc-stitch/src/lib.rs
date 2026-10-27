//! Serialisation of dc core terms to and from the Stitch format (spec §10).
//!
//! [Stitch](https://github.com/mlb2251/stitch) reads and writes programs as
//! s-expressions in the syntax of its `lambdas` crate:
//!
//! ```text
//! e ::= $i          de Bruijn variable
//!     | #i          abstraction parameter (only inside abstraction bodies)
//!     | (lam e)     lambda
//!     | (lam_N e)   lambda tagged N, a non-negative integer
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
//! | `Let(x, b)` | `(let_ x (lam_0 b))` |
//! | `If(c, t, e)` | `(if_ c t e)` |
//! | `And(a, b)` | `(and_ a b)` |
//! | `Or(a, b)` | `(or_ a b)` |
//! | `Int(n)` | `int_n`, for example `int_3`, `int_-3` |
//! | `Bool(b)` | `true`, `false` |
//! | `Unit` | `unit` |
//! | `Prim(p)` | the canonical name, for example `add`, `foldl` |
//! | `Lib(l)` | the name a [`LibNames`] table gives `l`, for example `fn_0` |
//!
//! `Let` binds a variable, so its body goes under a real lambda. Stitch then sees
//! the binder and treats the body's `$0` as bound, as it must when it decides which
//! subterms are closed enough to abstract. The binder is the lambda tagged
//! [`LET_LAMBDA_TAG`] (`lam_0`), which appears nowhere else in the encoding, so a
//! let's binder never unifies with an ordinary `lam` and decoding can tell the two
//! apart. `If`, `And` and `Or` are lazy special
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
//! Encoding, decoding, parsing and printing use explicit work lists, not host
//! recursion, so a deep term (for example a list literal of many thousands of
//! elements, which nests one `cons` per element) is limited only by memory.
//!
//! # Stitch flags the encoding needs
//!
//! Stitch may cut an abstraction or a metavariable through a special form. Such
//! shapes have no core term, so decoding reports
//! [`DecodeError::MalformedSpecialForm`]. Run Stitch with these flags to rule
//! them out:
//!
//! - `--fused-lambda-tags 0`: Stitch then never roots an abstraction at a `lam_0`
//!   node and never puts a parameter there (Stitch's `fused_lambda_location`, used
//!   for both match roots and metavariable locations). Without it Stitch could
//!   learn `fn_0 = (lam_0 (add $0 int_1))` and rewrite a program to
//!   `(let_ int_5 fn_0)`, or learn the body `(let_ #0 #1)`. Abstractions can
//!   still match inside a let's body, or contain a whole let.
//! - `--no-curried-bodies` and `--no-curried-metavars`: these rule out an
//!   abstraction body such as `(if_ #0 #1)` and an argument such as `(let_ x)`.
//!
//! Stitch's eta-long mode does not keep lambda tags (`rewriting.rs`), so do not
//! use `--eta-long` with this encoding.
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
    AND, IF, INT_PREFIX, LET, LET_LAMBDA_TAG, OR,
};
pub use expr::{Expr, ParseError};
pub use names::{LibNames, NameError};
