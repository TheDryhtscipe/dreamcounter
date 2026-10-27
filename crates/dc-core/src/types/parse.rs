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

use std::fmt;
use std::str::FromStr;

use super::{Scheme, TyVar, Type};

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

    fn ty(&mut self) -> Result<Type, TypeParseError> {
        let a = self.app()?;
        if self.peek() == Some(Tok::Arrow) {
            self.pos += 1;
            let b = self.ty()?;
            Ok(Type::arrow(a, b))
        } else {
            Ok(a)
        }
    }

    fn app(&mut self) -> Result<Type, TypeParseError> {
        let ctor: Option<fn(Type) -> Type> = match self.peek() {
            Some(Tok::Ident("List")) => Some(Type::list),
            Some(Tok::Ident("Option")) => Some(Type::option),
            Some(Tok::Ident("Array")) => Some(Type::array),
            Some(Tok::Ident("Tree")) => Some(Type::tree),
            _ => None,
        };
        match ctor {
            Some(c) => {
                self.pos += 1;
                Ok(c(self.atom()?))
            }
            None => self.atom(),
        }
    }

    fn atom(&mut self) -> Result<Type, TypeParseError> {
        match self.peek() {
            Some(Tok::Ident("Int")) => {
                self.pos += 1;
                Ok(Type::Int)
            }
            Some(Tok::Ident("Bool")) => {
                self.pos += 1;
                Ok(Type::Bool)
            }
            Some(Tok::Ident("Unit")) => {
                self.pos += 1;
                Ok(Type::Unit)
            }
            Some(Tok::Ident(name)) if name.starts_with(|c: char| c.is_ascii_lowercase()) => {
                self.pos += 1;
                Ok(Type::Var(self.var(name)))
            }
            Some(Tok::Ident(name)) => self.err(format!(
                "unknown type '{name}' (types are Int, Bool, Unit, List, Option, Array, Tree; \
                 variables are lower case)"
            )),
            Some(Tok::Open) => {
                self.pos += 1;
                let first = self.ty()?;
                let mut items = vec![first];
                while self.peek() == Some(Tok::Comma) {
                    self.pos += 1;
                    items.push(self.ty()?);
                }
                self.expect(Tok::Close, "')'")?;
                let last = items.pop().expect("at least one item");
                Ok(items
                    .into_iter()
                    .rev()
                    .fold(last, |acc, t| Type::pair(t, acc)))
            }
            Some(_) => self.err("expected a type"),
            None => self.err("unexpected end of input"),
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

    /// Parses a type. Variables are numbered in order of first appearance.
    fn from_str(src: &str) -> Result<Type, TypeParseError> {
        let mut p = parser(src)?;
        let t = p.ty()?;
        p.finish()?;
        Ok(t)
    }
}

impl FromStr for Scheme {
    type Err = TypeParseError;

    /// Parses a scheme such as `Key a => a -> List a`.
    fn from_str(src: &str) -> Result<Scheme, TypeParseError> {
        let mut p = parser(src)?;
        let keys = p.keys()?;
        let t = p.ty()?;
        p.finish()?;
        Ok(Scheme::with_keys(t, &keys))
    }
}
