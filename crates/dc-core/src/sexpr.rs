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
//! [`print`](fn@print) always flattens application spines, so `parse(print(t)) == t` for every
//! term `t` it prints. It refuses a term with more than [`MAX_PRINT_NODES`]
//! nodes written out as a tree; [`print_bounded`] writes a prefix of any term.

use std::fmt::{self, Write as _};

use crate::prim::Prim;
use crate::term::{Arena, LibId, Term, TermId};

/// The most nodes a term [`print`](fn@print) prints may have, written out as a
/// tree (see [`Arena::tree_size`]). Each node adds at most 21 bytes of text (the
/// most is a 20-digit integer literal and the space before it), so the text of a
/// term within the limit is at most 21 MiB.
pub const MAX_PRINT_NODES: usize = 1 << 20;

/// Why [`print`](fn@print) refused a term: it has more than [`MAX_PRINT_NODES`]
/// nodes written out as a tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TooLarge;

impl fmt::Display for TooLarge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the term has more than {MAX_PRINT_NODES} nodes written out, too many to print"
        )
    }
}

impl std::error::Error for TooLarge {}

/// Prints `id` in core syntax: the whole term, so that [`parse`] reads it back.
///
/// Shared subterms are printed at each use, so the text of a DAG built in an
/// arena can be exponentially longer than its number of distinct nodes. So it
/// first measures the term written out as a tree, at a cost linear in the
/// number of distinct nodes, and refuses one with more than
/// [`MAX_PRINT_NODES`] nodes. For text that only needs to show a term, such as
/// an error message, use [`print_bounded`], which writes a prefix of any term.
///
/// It walks with an explicit stack, so any depth of term prints without risk of
/// a stack overflow.
pub fn print(arena: &Arena, id: TermId) -> Result<String, TooLarge> {
    arena.tree_size(id, MAX_PRINT_NODES).ok_or(TooLarge)?;
    let mut out = String::new();
    let whole = print_bounded(arena, id, usize::MAX, &mut out);
    debug_assert!(whole, "print_bounded stops only past its limit");
    Ok(out)
}

/// Appends `id` in core syntax to `out`, stopping once the text is known to be
/// longer than `limit` bytes. Returns whether the whole term was written; if
/// not, what was written is a prefix of it.
///
/// Its cost is linear in `limit`, whatever the size of the term. Each item
/// visited writes at least one byte, so it visits at most `limit + 1` of them.
/// The head of an application spine is at its far end, so the spine is walked
/// before anything of it is written; every argument met on the way writes at
/// least two bytes (a space and the argument) once it is printed. So once more
/// than `limit / 2` arguments have been met in all, the text is longer than
/// `limit`, and the walk stops there, after writing the opening `(`.
pub fn print_bounded(arena: &Arena, id: TermId, limit: usize, out: &mut String) -> bool {
    enum Item {
        Term(TermId),
        Text(&'static str),
    }
    let mut args_left = (limit / 2).saturating_add(1);
    let mut todo = vec![Item::Term(id)];
    while let Some(item) = todo.pop() {
        if out.len() > limit {
            return false;
        }
        let t = match item {
            Item::Text(s) => {
                out.push_str(s);
                continue;
            }
            Item::Term(t) => t,
        };
        let (keyword, children): (&str, &[TermId]) = match arena.get(t) {
            Term::Var(i) => {
                let _ = write!(out, "${i}");
                continue;
            }
            Term::Int(n) => {
                let _ = write!(out, "{n}");
                continue;
            }
            Term::Bool(b) => {
                let _ = write!(out, "{b}");
                continue;
            }
            Term::Unit => {
                out.push_str("unit");
                continue;
            }
            Term::Prim(p) => {
                out.push_str(p.name());
                continue;
            }
            Term::Lib(LibId(l)) => {
                let _ = write!(out, "lib_{l}");
                continue;
            }
            Term::Lam(b) => ("lam", &[b]),
            Term::Let(x, b) => ("let", &[x, b]),
            Term::If(c, th, e) => ("if", &[c, th, e]),
            Term::And(x, y) => ("and", &[x, y]),
            Term::Or(x, y) => ("or", &[x, y]),
            Term::App(..) => {
                // The spine is flattened: `(f a b)`, not `((f a) b)`. Arguments are
                // met last first, which is the order to push them.
                out.push('(');
                todo.push(Item::Text(")"));
                let mut head = t;
                while let Term::App(f, a) = arena.get(head) {
                    if args_left == 0 {
                        return false;
                    }
                    args_left -= 1;
                    todo.push(Item::Term(a));
                    todo.push(Item::Text(" "));
                    head = f;
                }
                todo.push(Item::Term(head));
                continue;
            }
        };
        out.push('(');
        out.push_str(keyword);
        todo.push(Item::Text(")"));
        for &c in children.iter().rev() {
            todo.push(Item::Term(c));
            todo.push(Item::Text(" "));
        }
    }
    true
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

/// An open form waiting for more subterms: the parser's explicit stack, in place of
/// recursion on the host stack.
enum Open<'a> {
    /// `(kw ...` with the arguments read so far; `kw` takes `arity` of them.
    Form {
        keyword: &'a str,
        arity: usize,
        args: Vec<TermId>,
        head_offset: usize,
    },
    /// `(head args ...`; `head` is `None` until it has been read.
    App {
        head: Option<TermId>,
        args: Vec<TermId>,
        head_offset: usize,
    },
}

impl<'a> Parser<'a, '_> {
    fn next(&mut self) -> Result<Token<'a>, ParseError> {
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

    fn arity_error(keyword: &str, arity: usize, head_offset: usize) -> ParseError {
        ParseError {
            offset: head_offset,
            message: format!("'{keyword}' takes {arity} argument(s)"),
        }
    }

    /// Consumes the `)` that closes a form.
    fn close(&mut self) -> Result<(), ParseError> {
        match self.next()?.tok {
            Tok::Close => Ok(()),
            _ => Err(ParseError {
                offset: self.tokens[self.pos - 1].offset,
                message: "expected ')'".into(),
            }),
        }
    }

    /// Reads one term. Open forms are kept on an explicit stack, so the nesting
    /// depth is limited by memory, not by the host stack.
    fn term(&mut self, arena: &mut Arena) -> Result<TermId, ParseError> {
        let mut stack: Vec<Open<'a>> = vec![];
        loop {
            // Read the start of a subterm: an atom is done at once, `(` opens a form.
            let tok = self.next()?;
            let mut done = match tok.tok {
                Tok::Close => {
                    return Err(ParseError {
                        offset: tok.offset,
                        message: "unexpected ')'".into(),
                    })
                }
                Tok::Atom(a) => atom(arena, a, tok.offset)?,
                Tok::Open => {
                    let head_offset = self.tokens.get(self.pos).map_or(self.end, |t| t.offset);
                    let keyword = match self.tokens.get(self.pos) {
                        Some(Token {
                            tok: Tok::Atom(a), ..
                        }) => keyword_arity(a).map(|n| (*a, n)),
                        _ => None,
                    };
                    if let Some((keyword, arity)) = keyword {
                        self.pos += 1;
                        if self.peek_close() {
                            return Err(Self::arity_error(keyword, arity, head_offset));
                        }
                        stack.push(Open::Form {
                            keyword,
                            arity,
                            args: Vec::with_capacity(arity),
                            head_offset,
                        });
                    } else {
                        if self.peek_close() {
                            return Err(ParseError {
                                offset: head_offset,
                                message: "empty application '()'".into(),
                            });
                        }
                        stack.push(Open::App {
                            head: None,
                            args: vec![],
                            head_offset,
                        });
                    }
                    continue;
                }
            };
            // Hand the finished subterm to the innermost open form; close every form
            // that it completes.
            loop {
                let Some(open) = stack.last_mut() else {
                    return Ok(done);
                };
                match open {
                    Open::Form {
                        keyword,
                        arity,
                        args,
                        head_offset,
                    } => {
                        args.push(done);
                        if args.len() < *arity {
                            if self.peek_close() {
                                return Err(Self::arity_error(keyword, *arity, *head_offset));
                            }
                            break;
                        }
                        done = match *keyword {
                            "lam" => arena.lam(args[0]),
                            "let" => arena.let_(args[0], args[1]),
                            "if" => arena.if_(args[0], args[1], args[2]),
                            "and" => arena.and(args[0], args[1]),
                            _ => arena.or(args[0], args[1]),
                        };
                    }
                    Open::App {
                        head,
                        args,
                        head_offset,
                    } => {
                        if head.is_none() {
                            *head = Some(done);
                        } else {
                            args.push(done);
                        }
                        if !self.peek_close() && self.pos < self.tokens.len() {
                            break;
                        }
                        if args.is_empty() {
                            return Err(ParseError {
                                offset: *head_offset,
                                message: "application needs at least one argument".into(),
                            });
                        }
                        let f = head.expect("the head is read before any argument");
                        done = arena.apps(f, args);
                    }
                }
                self.close()?;
                stack.pop();
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
        assert_eq!(print(&a, id).unwrap(), src);
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

    /// Far deeper than a recursive walk survives on a test thread's stack.
    const DEEP: usize = 100_000;

    #[test]
    fn deep_lambdas_print_and_parse_without_host_recursion() {
        let mut a = Arena::new();
        let mut t = a.var(0);
        for _ in 0..DEEP {
            t = a.lam(t);
        }
        let text = print(&a, t).unwrap();
        assert_eq!(text.len(), DEEP * "(lam )".len() + "$0".len());
        assert!(text.starts_with("(lam (lam "));
        assert_eq!(parse(&mut a, &text), Ok(t));
        let mut fresh = Arena::new();
        let back = parse(&mut fresh, &text).unwrap();
        assert_eq!(fresh.len(), a.len());
        assert_eq!(print(&fresh, back).unwrap(), text);
    }

    #[test]
    fn deep_arguments_heads_and_forms_print_and_parse() {
        let mut a = Arena::new();
        let f = a.prim(Prim::Not);
        let one = a.int(1);
        let t0 = a.bool(true);
        // Nested in argument position, in head position and in each form slot.
        let mut arg = t0;
        let mut head = f;
        let mut forms = one;
        for i in 0..DEEP {
            arg = a.app(f, arg);
            head = if i % 2 == 0 {
                a.app(head, one)
            } else {
                a.lam(head)
            };
            let c = a.bool(i % 3 == 0);
            forms = match i % 5 {
                0 => a.if_(c, forms, one),
                1 => a.let_(one, forms),
                2 => a.and(c, forms),
                3 => a.or(forms, c),
                _ => a.app(forms, one),
            };
        }
        for t in [arg, head, forms] {
            let text = print(&a, t).unwrap();
            assert_eq!(parse(&mut a, &text), Ok(t));
        }
    }

    #[test]
    fn deep_malformed_input_is_an_error() {
        let open = "(not ".repeat(DEEP);
        let mut a = Arena::new();
        let e = parse(&mut a, &open).unwrap_err();
        assert!(e.message.contains("at least one argument"), "{e}");
        let e = parse(&mut a, &format!("{open}(lam")).unwrap_err();
        assert!(e.message.contains("unexpected end of input"), "{e}");
        let unbalanced = format!("{open}true{}", ")".repeat(DEEP + 1));
        let e = parse(&mut a, &unbalanced).unwrap_err();
        assert!(e.message.contains("unexpected input after term"), "{e}");
        let short = format!("{open}true{}", ")".repeat(DEEP - 1));
        let e = parse(&mut a, &short).unwrap_err();
        assert!(e.message.contains("unexpected end of input"), "{e}");
    }

    /// `(if true t t)` nested `n` deep over `$0`: `n + 3` distinct nodes, and
    /// `3 * 2^n - 2` nodes written out as a tree.
    fn shared_ifs(a: &mut Arena, n: usize) -> TermId {
        let c = a.bool(true);
        let mut t = a.var(0);
        for _ in 0..n {
            t = a.if_(c, t, t);
        }
        t
    }

    #[test]
    fn print_refuses_terms_over_the_node_limit() {
        let mut a = Arena::new();
        // 3 * 2^19 - 2 = 1_572_862 nodes written out, from 21 distinct ones.
        let t = shared_ifs(&mut a, 19);
        match print(&a, t) {
            Err(e) => assert_eq!(e, TooLarge),
            Ok(text) => panic!("printed {} bytes over the node limit", text.len()),
        }
        // `(and t t)` nested 19 deep is 2^20 - 1 nodes written out; one `lam`
        // around it reaches the limit exactly, and a second goes over.
        let mut t = a.bool(false);
        for _ in 0..19 {
            t = a.and(t, t);
        }
        let at_limit = a.lam(t);
        assert_eq!(a.tree_size(at_limit, usize::MAX), Some(MAX_PRINT_NODES));
        let text = print(&a, at_limit).expect("a term at the limit prints");
        assert_eq!(parse(&mut a, &text), Ok(at_limit));
        let over = a.lam(at_limit);
        assert_eq!(print(&a, over), Err(TooLarge));
    }

    #[test]
    fn print_refuses_a_huge_dag_in_linear_time() {
        // 103 distinct nodes, 3 * 2^100 - 2 written out.
        let mut a = Arena::new();
        let t = shared_ifs(&mut a, 100);
        let before = crate::term::GETS.with(std::cell::Cell::get);
        assert_eq!(print(&a, t), Err(TooLarge));
        let reads = crate::term::GETS.with(std::cell::Cell::get) - before;
        assert!(reads <= 2 * 103, "{reads} reads");
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
                let text = print(&arena.borrow(), id).unwrap();
                let mut fresh = Arena::new();
                let back =
                    parse(&mut fresh, &text).map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(print(&fresh, back).unwrap(), text.clone());
                // Parsing into the original arena must find the very same node.
                let again = parse(&mut arena.borrow_mut(), &text)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(again, id);
                Ok(())
            })
            .unwrap();
    }
}
