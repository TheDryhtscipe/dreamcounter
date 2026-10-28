//! Parsing types and schemes from their text form.
//!
//! ```text
//! scheme ::= [ "Key" var { "," "Key" var } "=>" ] type
//! type   ::= app [ "->" type ]
//! app    ::= ("List" | "Option" | "Array" | "Tree") atom | atom
//! atom   ::= "Int" | "Bool" | "Unit" | var | "(" type ")" | "(" type "," type { "," type } ")"
//! var    ::= a lower-case identifier
//! ```
//!
//! Variables get [`TyVar`] numbers in order of first appearance. Schemes are
//! normalised, so any variable names may be used.
//!
//! **Bounded depth.** Type text may come from outside (task signatures, library
//! schemes read from files), and most walks over [`Type`] (drop, print, compare,
//! import into the checker) recurse on the host stack. So text is an error, not a
//! stack overflow, when the parsed type would be deeper than [`MAX_TYPE_DEPTH`] (the
//! bound on types the checker builds, counted the same way: a leaf is one level) or
//! when its brackets plus the root would nest deeper than that. The bounds differ
//! because brackets with no constructor, as in `((Int))`, deepen the text but not the
//! type, while a flat tuple `(a, b, c, ...)` deepens the type (pairs nest to the
//! right) but not the text. Text printed from a type within the bound parses back.
//! The parser itself keeps an explicit stack and does not recurse: at the bound, a
//! recursive descent needs megabytes of stack in a debug build.

use std::fmt;
use std::str::FromStr;

use super::{Scheme, TyVar, Type, MAX_TYPE_DEPTH};

/// A type or scheme parse error, with the byte offset where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeParseError {
    /// Byte offset into the input.
    pub offset: usize,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for TypeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for TypeParseError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok<'a> {
    Open,
    Close,
    Comma,
    Arrow,
    Implies,
    Ident(&'a str),
}

fn tokenize(src: &str) -> Result<Vec<(Tok<'_>, usize)>, TypeParseError> {
    let bytes = src.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        let start = i;
        let tok = match c {
            b'(' => Tok::Open,
            b')' => Tok::Close,
            b',' => Tok::Comma,
            b'-' if bytes.get(i + 1) == Some(&b'>') => {
                i += 1;
                Tok::Arrow
            }
            b'=' if bytes.get(i + 1) == Some(&b'>') => {
                i += 1;
                Tok::Implies
            }
            c if c.is_ascii_whitespace() => {
                i += 1;
                continue;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                out.push((Tok::Ident(&src[start..i]), start));
                continue;
            }
            _ => {
                return Err(TypeParseError {
                    offset: i,
                    message: format!("unexpected character '{}'", char::from(c)),
                })
            }
        };
        i += 1;
        out.push((tok, start));
    }
    Ok(out)
}

struct Parser<'a> {
    toks: Vec<(Tok<'a>, usize)>,
    pos: usize,
    end: usize,
    names: Vec<&'a str>,
}

/// A part of a type that [`Parser::ty`] has begun and that waits for more input.
/// Each `start` is the byte offset where the part's text begins.
enum Frame {
    /// A constructor, waiting for its argument `atom`.
    Ctor {
        ctor: fn(Type) -> Type,
        start: usize,
    },
    /// `lhs ->`, waiting for the result `type`; `depth` is the depth of `lhs`.
    Arrow {
        lhs: Type,
        depth: usize,
        start: usize,
    },
    /// `(` and the items so far with their depths, waiting for the next item.
    Tuple {
        items: Vec<(Type, usize)>,
        start: usize,
    },
}

/// The "too deep" error at `offset`.
fn too_deep<T>(offset: usize) -> Result<T, TypeParseError> {
    Err(TypeParseError {
        offset,
        message: format!("type nested deeper than {MAX_TYPE_DEPTH} levels"),
    })
}

/// The depth of a constructor over children at most `depth` deep, if that is within
/// [`MAX_TYPE_DEPTH`]; `offset` is where the constructor's text starts.
fn above(depth: usize, offset: usize) -> Result<usize, TypeParseError> {
    if depth >= MAX_TYPE_DEPTH {
        too_deep(offset)
    } else {
        Ok(depth + 1)
    }
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<Tok<'a>> {
        self.toks.get(self.pos).map(|t| t.0)
    }

    fn offset(&self) -> usize {
        self.toks.get(self.pos).map_or(self.end, |t| t.1)
    }

    fn err<T>(&self, message: impl Into<String>) -> Result<T, TypeParseError> {
        Err(TypeParseError {
            offset: self.offset(),
            message: message.into(),
        })
    }

    fn expect(&mut self, tok: Tok<'_>, what: &str) -> Result<(), TypeParseError> {
        if self.peek() == Some(tok) {
            self.pos += 1;
            Ok(())
        } else {
            self.err(format!("expected {what}"))
        }
    }

    fn var(&mut self, name: &'a str) -> TyVar {
        let i = match self.names.iter().position(|&n| n == name) {
            Some(i) => i,
            None => {
                self.names.push(name);
                self.names.len() - 1
            }
        };
        TyVar(u32::try_from(i).expect("fewer than 2^32 type variables"))
    }

    /// Parses a `type` with an explicit stack of [`Frame`]s, so the host stack does
    /// not grow with the nesting of the text. Every type is built with its depth
    /// known, so none deeper than [`MAX_TYPE_DEPTH`] is ever built (or dropped).
    fn ty(&mut self) -> Result<Type, TypeParseError> {
        let mut frames: Vec<Frame> = vec![];
        // Brackets open, plus one for the root: the nesting of the grammar.
        let mut nesting = 1;
        'ty: loop {
            // A `type` starts here; first its `app`, then its `atom`.
            let start = self.offset();
            let ctor: Option<fn(Type) -> Type> = match self.peek() {
                Some(Tok::Ident("List")) => Some(Type::list),
                Some(Tok::Ident("Option")) => Some(Type::option),
                Some(Tok::Ident("Array")) => Some(Type::array),
                Some(Tok::Ident("Tree")) => Some(Type::tree),
                _ => None,
            };
            if let Some(ctor) = ctor {
                self.pos += 1;
                frames.push(Frame::Ctor { ctor, start });
            }
            let leaf = match self.peek() {
                Some(Tok::Ident("Int")) => Type::Int,
                Some(Tok::Ident("Bool")) => Type::Bool,
                Some(Tok::Ident("Unit")) => Type::Unit,
                Some(Tok::Ident(name)) if name.starts_with(|c: char| c.is_ascii_lowercase()) => {
                    Type::Var(self.var(name))
                }
                Some(Tok::Ident(name)) => {
                    return self.err(format!(
                        "unknown type '{name}' (types are Int, Bool, Unit, List, Option, \
                         Array, Tree; variables are lower case)"
                    ))
                }
                Some(Tok::Open) => {
                    if nesting >= MAX_TYPE_DEPTH {
                        return too_deep(self.offset());
                    }
                    nesting += 1;
                    frames.push(Frame::Tuple {
                        items: vec![],
                        start: self.offset(),
                    });
                    self.pos += 1;
                    continue 'ty;
                }
                Some(_) => return self.err("expected a type"),
                None => return self.err("unexpected end of input"),
            };
            let leaf_start = self.offset();
            self.pos += 1;
            // `t` is a complete `atom` of depth `d` whose text starts at `s`. Close
            // the frames it completes, until one needs more input.
            let (mut t, mut d, mut s) = (leaf, 1, leaf_start);
            loop {
                if let Some(&Frame::Ctor { ctor, start }) = frames.last() {
                    frames.pop();
                    d = above(d, start)?;
                    t = ctor(t);
                    s = start;
                }
                // `t` is a complete `app`.
                if self.peek() == Some(Tok::Arrow) {
                    self.pos += 1;
                    frames.push(Frame::Arrow {
                        lhs: t,
                        depth: d,
                        start: s,
                    });
                    continue 'ty;
                }
                // `t` is a complete `type`: it is the result of any arrows waiting.
                while let Some(Frame::Arrow { .. }) = frames.last() {
                    let Some(Frame::Arrow { lhs, depth, start }) = frames.pop() else {
                        unreachable!("the last frame is an arrow");
                    };
                    d = above(depth.max(d), start)?;
                    t = Type::arrow(lhs, t);
                }
                let (mut items, start) = match frames.pop() {
                    None => return Ok(t),
                    Some(Frame::Tuple { items, start }) => (items, start),
                    Some(Frame::Ctor { .. } | Frame::Arrow { .. }) => {
                        unreachable!("a constructor waits for an atom, and arrows are closed")
                    }
                };
                items.push((t, d));
                if self.peek() == Some(Tok::Comma) {
                    self.pos += 1;
                    frames.push(Frame::Tuple { items, start });
                    continue 'ty;
                }
                self.expect(Tok::Close, "')'")?;
                nesting -= 1;
                // Pairs nest to the right. Check the depth before building any of
                // them, so a long tuple never becomes a pair chain too deep to drop.
                let (mut acc, mut depth) = items.pop().expect("at least one item");
                for &(_, item) in items.iter().rev() {
                    depth = above(item.max(depth), start)?;
                }
                for (item, _) in items.into_iter().rev() {
                    acc = Type::pair(item, acc);
                }
                (t, d, s) = (acc, depth, start);
            }
        }
    }

    fn keys(&mut self) -> Result<Vec<TyVar>, TypeParseError> {
        let has_context = self.toks.iter().any(|t| t.0 == Tok::Implies);
        if !has_context {
            return Ok(vec![]);
        }
        let mut keys = vec![];
        loop {
            self.expect(Tok::Ident("Key"), "'Key' constraint")?;
            match self.peek() {
                Some(Tok::Ident(n)) if n.starts_with(|c: char| c.is_ascii_lowercase()) => {
                    self.pos += 1;
                    keys.push(self.var(n));
                }
                _ => return self.err("expected a type variable after 'Key'"),
            }
            match self.peek() {
                Some(Tok::Comma) => self.pos += 1,
                Some(Tok::Implies) => {
                    self.pos += 1;
                    return Ok(keys);
                }
                _ => return self.err("expected ',' or '=>'"),
            }
        }
    }

    fn finish(&self) -> Result<(), TypeParseError> {
        if self.pos < self.toks.len() {
            self.err("unexpected input after type")
        } else {
            Ok(())
        }
    }
}

fn parser(src: &str) -> Result<Parser<'_>, TypeParseError> {
    Ok(Parser {
        toks: tokenize(src)?,
        pos: 0,
        end: src.len(),
        names: vec![],
    })
}

impl FromStr for Type {
    type Err = TypeParseError;

    /// Parses a type. Variables are numbered in order of first appearance. Text
    /// nested deeper than [`MAX_TYPE_DEPTH`] is an error.
    fn from_str(src: &str) -> Result<Type, TypeParseError> {
        let mut p = parser(src)?;
        let t = p.ty()?;
        p.finish()?;
        Ok(t)
    }
}

impl FromStr for Scheme {
    type Err = TypeParseError;

    /// Parses a scheme such as `Key a => a -> List a`. Text nested deeper than
    /// [`MAX_TYPE_DEPTH`] is an error.
    fn from_str(src: &str) -> Result<Scheme, TypeParseError> {
        let mut p = parser(src)?;
        let keys = p.keys()?;
        let t = p.ty()?;
        p.finish()?;
        Ok(Scheme::with_keys(t, &keys))
    }
}
