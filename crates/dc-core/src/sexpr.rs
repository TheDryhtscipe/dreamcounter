//! Core syntax: an s-expression form of core terms, for printing, tests and tools.
//!
//! ```text
//! e ::= $i                      de Bruijn variable
//!     | (lam e)                 lambda
//!     | (e e e ...)             application, left-associated
//!     | (let e e)               let: the body sees the bound value at $0
//!     | (if e e e) | (and e e) | (or e e)
//!     | 42 | -7 | true | false | unit
//!     | add | foldl | ...       primitive, by canonical name
//!     | lib_3                   library abstraction 3
//! ```
//!
//! [`print`] always flattens application spines, so `parse(print(t)) == t` for every
//! term `t`.

use std::fmt::{self, Write};

use crate::prim::Prim;
use crate::term::{Arena, LibId, Term, TermId};

/// Prints `id` in core syntax.
pub fn print(arena: &Arena, id: TermId) -> String {
    let mut out = String::new();
    write_term(arena, id, &mut out).expect("writing to a String cannot fail");
    out
}

fn write_term(arena: &Arena, id: TermId, out: &mut String) -> fmt::Result {
    match arena.get(id) {
        Term::Var(i) => write!(out, "${i}"),
        Term::Int(n) => write!(out, "{n}"),
        Term::Bool(b) => write!(out, "{b}"),
        Term::Unit => out.write_str("unit"),
        Term::Prim(p) => out.write_str(p.name()),
        Term::Lib(LibId(l)) => write!(out, "lib_{l}"),
        Term::Lam(b) => write_form(arena, "lam", &[b], out),
        Term::Let(x, b) => write_form(arena, "let", &[x, b], out),
        Term::If(c, t, e) => write_form(arena, "if", &[c, t, e], out),
        Term::And(x, y) => write_form(arena, "and", &[x, y], out),
        Term::Or(x, y) => write_form(arena, "or", &[x, y], out),
        Term::App(..) => {
            let mut spine = vec![];
            let mut head = id;
            while let Term::App(f, a) = arena.get(head) {
                spine.push(a);
                head = f;
            }
            out.write_char('(')?;
            write_term(arena, head, out)?;
            for &a in spine.iter().rev() {
                out.write_char(' ')?;
                write_term(arena, a, out)?;
            }
            out.write_char(')')
        }
    }
}

fn write_form(arena: &Arena, keyword: &str, args: &[TermId], out: &mut String) -> fmt::Result {
    out.write_char('(')?;
    out.write_str(keyword)?;
    for &a in args {
        out.write_char(' ')?;
        write_term(arena, a, out)?;
    }
    out.write_char(')')
}

/// A parse error, with the byte offset where it was found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    /// Byte offset into the input.
    pub offset: usize,
    /// What went wrong.
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parses one term in core syntax into `arena`.
pub fn parse(arena: &mut Arena, src: &str) -> Result<TermId, ParseError> {
    let tokens = tokenize(src)?;
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
        end: src.len(),
    };
    let id = parser.term(arena)?;
    if let Some(tok) = parser.tokens.get(parser.pos) {
        return Err(ParseError {
            offset: tok.offset,
            message: "unexpected input after term".into(),
        });
    }
    Ok(id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tok<'a> {
    Open,
    Close,
    Atom(&'a str),
}

#[derive(Debug, Clone, Copy)]
struct Token<'a> {
    tok: Tok<'a>,
    offset: usize,
}

fn tokenize(src: &str) -> Result<Vec<Token<'_>>, ParseError> {
    let mut tokens = vec![];
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => {
                tokens.push(Token {
                    tok: Tok::Open,
                    offset: i,
                });
                i += 1;
            }
            b')' => {
                tokens.push(Token {
                    tok: Tok::Close,
                    offset: i,
                });
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let start = i;
                while i < bytes.len()
                    && !matches!(bytes[i], b'(' | b')')
                    && !bytes[i].is_ascii_whitespace()
                {
                    i += 1;
                }
                tokens.push(Token {
                    tok: Tok::Atom(&src[start..i]),
                    offset: start,
                });
            }
        }
    }
    Ok(tokens)
}

struct Parser<'a, 't> {
    tokens: &'t [Token<'a>],
    pos: usize,
    end: usize,
}

impl Parser<'_, '_> {
    fn next(&mut self) -> Result<Token<'_>, ParseError> {
        let tok = self.tokens.get(self.pos).copied().ok_or(ParseError {
            offset: self.end,
            message: "unexpected end of input".into(),
        })?;
        self.pos += 1;
        Ok(tok)
    }

    fn peek_close(&self) -> bool {
        matches!(
            self.tokens.get(self.pos),
            Some(Token {
                tok: Tok::Close,
                ..
            })
        )
    }

    fn term(&mut self, arena: &mut Arena) -> Result<TermId, ParseError> {
        let tok = self.next()?;
        match tok.tok {
            Tok::Close => Err(ParseError {
                offset: tok.offset,
                message: "unexpected ')'".into(),
            }),
            Tok::Atom(a) => atom(arena, a, tok.offset),
            Tok::Open => {
                let head_offset = self.tokens.get(self.pos).map_or(self.end, |t| t.offset);
                let keyword = match self.tokens.get(self.pos) {
                    Some(Token {
                        tok: Tok::Atom(a), ..
                    }) => keyword_arity(a).map(|n| (*a, n)),
                    _ => None,
                };
                let id = if let Some((kw, n)) = keyword {
                    self.pos += 1;
                    let mut args = Vec::with_capacity(n);
                    for _ in 0..n {
                        if self.peek_close() {
                            return Err(ParseError {
                                offset: head_offset,
                                message: format!("'{kw}' takes {n} argument(s)"),
                            });
                        }
                        args.push(self.term(arena)?);
                    }
                    match kw {
                        "lam" => arena.lam(args[0]),
                        "let" => arena.let_(args[0], args[1]),
                        "if" => arena.if_(args[0], args[1], args[2]),
                        "and" => arena.and(args[0], args[1]),
                        _ => arena.or(args[0], args[1]),
                    }
                } else {
                    if self.peek_close() {
                        return Err(ParseError {
                            offset: head_offset,
                            message: "empty application '()'".into(),
                        });
                    }
                    let head = self.term(arena)?;
                    let mut args = vec![];
                    while !self.peek_close() {
                        if self.pos >= self.tokens.len() {
                            break;
                        }
                        args.push(self.term(arena)?);
                    }
                    if args.is_empty() {
                        return Err(ParseError {
                            offset: head_offset,
                            message: "application needs at least one argument".into(),
                        });
                    }
                    arena.apps(head, &args)
                };
                match self.next()?.tok {
                    Tok::Close => Ok(id),
                    _ => Err(ParseError {
                        offset: self.tokens[self.pos - 1].offset,
                        message: "expected ')'".into(),
                    }),
                }
            }
        }
    }
}

fn keyword_arity(atom: &str) -> Option<usize> {
    match atom {
        "lam" => Some(1),
        "let" | "and" | "or" => Some(2),
        "if" => Some(3),
        _ => None,
    }
}

fn atom(arena: &mut Arena, a: &str, offset: usize) -> Result<TermId, ParseError> {
    let err = |message: String| ParseError { offset, message };
    if let Some(rest) = a.strip_prefix('$') {
        let i = rest
            .parse::<u32>()
            .map_err(|_| err(format!("bad variable '{a}'")))?;
        return Ok(arena.var(i));
    }
    if let Some(rest) = a.strip_prefix("lib_") {
        let l = rest
            .parse::<u32>()
            .map_err(|_| err(format!("bad library reference '{a}'")))?;
        return Ok(arena.lib(LibId(l)));
    }
    match a {
        "true" => return Ok(arena.bool(true)),
        "false" => return Ok(arena.bool(false)),
        "unit" => return Ok(arena.unit()),
        _ => {}
    }
    if keyword_arity(a).is_some() {
        return Err(err(format!(
            "'{a}' is a keyword and must open a form: ({a} ...)"
        )));
    }
    if let Some(p) = Prim::from_name(a) {
        return Ok(arena.prim(p));
    }
    if a.starts_with(|c: char| c == '-' || c.is_ascii_digit()) {
        return a
            .parse::<i64>()
            .map(|n| arena.int(n))
            .map_err(|_| err(format!("bad integer '{a}'")));
    }
    Err(err(format!("unknown name '{a}'")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn round_trip(src: &str) {
        let mut a = Arena::new();
        let id = parse(&mut a, src).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert_eq!(print(&a, id), src);
    }

    #[test]
    fn prints_and_parses_examples() {
        round_trip("(lam (add $0 1))");
        round_trip("(foldl (lam (lam (add $1 $0))) 0 $0)");
        round_trip("(let 5 (if (lt $0 3) true (and false (or true false))))");
        round_trip("((lam $0) -9223372036854775808)");
        round_trip("(lib_7 nil unit none leaf)");
    }

    #[test]
    fn application_is_left_associated() {
        let mut a = Arena::new();
        let parsed = parse(&mut a, "(add 1 2)").unwrap();
        let add = a.prim(Prim::Add);
        let one = a.int(1);
        let two = a.int(2);
        let inner = a.app(add, one);
        assert_eq!(parsed, a.app(inner, two));
    }

    #[test]
    fn rejects_malformed_input() {
        for (src, expect) in [
            ("", "unexpected end of input"),
            ("()", "empty application"),
            ("(add)", "at least one argument"),
            ("(lam)", "'lam' takes 1"),
            ("(if true 1)", "'if' takes 3"),
            ("(lam $0 $1)", "expected ')'"),
            ("lam", "is a keyword"),
            ("fix", "unknown name 'fix'"),
            ("$x", "bad variable"),
            ("99999999999999999999", "bad integer"),
            ("1 2", "unexpected input after term"),
            (")", "unexpected ')'"),
            ("(add 1 2", "unexpected end of input"),
        ] {
            let mut a = Arena::new();
            let e = parse(&mut a, src).expect_err(src);
            assert!(e.message.contains(expect), "{src:?}: got {e}");
        }
    }

    fn arb_term(arena: std::rc::Rc<std::cell::RefCell<Arena>>) -> impl Strategy<Value = TermId> {
        let leaf = prop_oneof![
            (0u32..4).prop_map(Term::Var),
            any::<i64>().prop_map(Term::Int),
            any::<bool>().prop_map(Term::Bool),
            Just(Term::Unit),
            proptest::sample::select(Prim::ALL).prop_map(Term::Prim),
            (0u32..5).prop_map(|l| Term::Lib(LibId(l))),
        ];
        let a0 = arena.clone();
        let leaf = leaf.prop_map(move |t| a0.borrow_mut().intern(t));
        leaf.prop_recursive(5, 48, 3, move |inner| {
            let a = arena.clone();
            prop_oneof![
                inner.clone().prop_map(Term::Lam),
                (inner.clone(), inner.clone()).prop_map(|(x, y)| Term::App(x, y)),
                (inner.clone(), inner.clone()).prop_map(|(x, y)| Term::Let(x, y)),
                (inner.clone(), inner.clone(), inner.clone())
                    .prop_map(|(x, y, z)| Term::If(x, y, z)),
                (inner.clone(), inner.clone()).prop_map(|(x, y)| Term::And(x, y)),
                (inner.clone(), inner).prop_map(|(x, y)| Term::Or(x, y)),
            ]
            .prop_map(move |t| a.borrow_mut().intern(t))
        })
    }

    #[test]
    fn parse_inverts_print() {
        let arena = std::rc::Rc::new(std::cell::RefCell::new(Arena::new()));
        let mut runner = proptest::test_runner::TestRunner::default();
        runner
            .run(&arb_term(arena.clone()), |id| {
                let text = print(&arena.borrow(), id);
                let mut fresh = Arena::new();
                let back =
                    parse(&mut fresh, &text).map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(print(&fresh, back), text.clone());
                // Parsing into the original arena must find the very same node.
                let again = parse(&mut arena.borrow_mut(), &text)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(again, id);
                Ok(())
            })
            .unwrap();
    }
}
