//! Stitch's expression syntax, as an owned tree with a parser and printer.
//!
//! The grammar and printing follow Stitch's `lambdas` crate (`parse_expr.rs`,
//! revision `2c9bfd0`, the one Stitch pins): tokens are split on whitespace and
//! parentheses, `$i` is a variable, `#i` a parameter, `(lam e)` a lambda, and
//! every other token a primitive. Application spines print flattened.
//!
//! This parser is stricter than Stitch's in ways Stitch's own output never
//! exercises, so that malformed input is an error rather than a guess:
//!
//! - lambda and variable tags (`lam_3`, `$0_3`) and holes (`??`) are rejected;
//! - `(e)` with a single element is rejected (Stitch reads it as `e`, but never
//!   prints it);
//! - `(lambda e)` is accepted as a synonym for `(lam e)`, as in Stitch.

use std::fmt::{self, Write as _};

/// A Stitch expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// De Bruijn variable `$i`.
    Var(u32),
    /// Abstraction parameter `#i`.
    IVar(u32),
    /// `(lam body)`.
    Lam(Box<Expr>),
    /// Application of a function to one argument.
    App(Box<Expr>, Box<Expr>),
    /// Any other token.
    Sym(String),
}

impl Expr {
    /// `(lam body)`.
    pub fn lam(body: Expr) -> Expr {
        Expr::Lam(Box::new(body))
    }

    /// `(f a)`.
    pub fn app(f: Expr, a: Expr) -> Expr {
        Expr::App(Box::new(f), Box::new(a))
    }

    /// A token.
    pub fn sym(s: impl Into<String>) -> Expr {
        Expr::Sym(s.into())
    }

    /// Splits an application spine into its head and its arguments, left to right.
    /// A non-application is its own head with no arguments.
    pub fn spine(&self) -> (&Expr, Vec<&Expr>) {
        let mut args = vec![];
        let mut head = self;
        while let Expr::App(f, a) = head {
            args.push(&**a);
            head = f;
        }
        args.reverse();
        (head, args)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(i) => write!(f, "${i}"),
            Expr::IVar(i) => write!(f, "#{i}"),
            Expr::Sym(s) => f.write_str(s),
            Expr::Lam(b) => write!(f, "(lam {b})"),
            Expr::App(..) => {
                let (head, args) = self.spine();
                write!(f, "({head}")?;
                for a in args {
                    write!(f, " {a}")?;
                }
                f.write_char(')')
            }
        }
    }
}

/// Prints `e` the way Stitch prints it.
pub fn print(e: &Expr) -> String {
    e.to_string()
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

/// Parses one Stitch expression.
pub fn parse(src: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(src);
    let mut parser = Parser {
        tokens: &tokens,
        pos: 0,
        end: src.len(),
    };
    let e = parser.expr()?;
    if let Some(tok) = tokens.get(parser.pos) {
        return Err(ParseError {
            offset: tok.offset,
            message: "unexpected input after expression".into(),
        });
    }
    Ok(e)
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

/// Splits on whitespace and parentheses, as Stitch does.
fn tokenize(src: &str) -> Vec<Token<'_>> {
    let mut tokens = vec![];
    let mut chars = src.char_indices().peekable();
    while let Some(&(i, c)) = chars.peek() {
        if c == '(' || c == ')' {
            let tok = if c == '(' { Tok::Open } else { Tok::Close };
            tokens.push(Token { tok, offset: i });
            chars.next();
        } else if c.is_whitespace() {
            chars.next();
        } else {
            let mut end = src.len();
            while let Some(&(j, c)) = chars.peek() {
                if c.is_whitespace() || c == '(' || c == ')' {
                    end = j;
                    break;
                }
                chars.next();
            }
            tokens.push(Token {
                tok: Tok::Atom(&src[i..end]),
                offset: i,
            });
        }
    }
    tokens
}

struct Parser<'a, 't> {
    tokens: &'t [Token<'a>],
    pos: usize,
    end: usize,
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

    fn peek(&self) -> Option<Tok<'a>> {
        self.tokens.get(self.pos).map(|t| t.tok)
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        let tok = self.next()?;
        let open = match tok.tok {
            Tok::Close => {
                return Err(ParseError {
                    offset: tok.offset,
                    message: "unexpected ')'".into(),
                })
            }
            Tok::Atom(a) => return atom(a, tok.offset),
            Tok::Open => tok.offset,
        };
        let err = |message: &str| ParseError {
            offset: open,
            message: message.into(),
        };
        let is_lam = matches!(self.peek(), Some(Tok::Atom("lam" | "lambda")));
        if is_lam {
            self.pos += 1;
        }
        let mut items = vec![];
        loop {
            match self.peek() {
                Some(Tok::Close) => {
                    self.pos += 1;
                    break;
                }
                Some(_) => items.push(self.expr()?),
                None => {
                    return Err(ParseError {
                        offset: self.end,
                        message: "unexpected end of input".into(),
                    })
                }
            }
        }
        if is_lam {
            return match <[Expr; 1]>::try_from(items) {
                Ok([body]) => Ok(Expr::lam(body)),
                Err(_) => Err(err("'lam' takes exactly one argument")),
            };
        }
        let mut items = items.into_iter();
        let head = items.next().ok_or_else(|| err("empty application '()'"))?;
        let mut e = head;
        let mut applied = false;
        for a in items {
            e = Expr::app(e, a);
            applied = true;
        }
        if !applied {
            return Err(err("application needs at least one argument"));
        }
        Ok(e)
    }
}

/// Parses a decimal index made only of ASCII digits.
fn index(digits: &str) -> Option<u32> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn atom(a: &str, offset: usize) -> Result<Expr, ParseError> {
    let err = |message: String| ParseError { offset, message };
    if let Some(rest) = a.strip_prefix('$') {
        return index(rest)
            .map(Expr::Var)
            .ok_or_else(|| err(format!("bad variable '{a}' (tags are not supported)")));
    }
    if let Some(rest) = a.strip_prefix('#') {
        return index(rest)
            .map(Expr::IVar)
            .ok_or_else(|| err(format!("bad parameter '{a}'")));
    }
    if a == "lam" || a == "lambda" {
        return Err(err(format!("'{a}' must open a form: ({a} ...)")));
    }
    if a.starts_with("lam_") || a.starts_with("lambda_") {
        return Err(err(format!("tagged lambda '{a}' is not supported")));
    }
    if a == "??" {
        return Err(err("hole '??' is not an expression".into()));
    }
    Ok(Expr::sym(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(i: u32) -> Expr {
        Expr::Var(i)
    }

    #[test]
    fn prints_like_stitch() {
        let add = Expr::sym("add");
        let e = Expr::lam(Expr::app(Expr::app(add, v(0)), Expr::IVar(1)));
        assert_eq!(print(&e), "(lam (add $0 #1))");
        // A right-nested application keeps its parentheses.
        let f = Expr::sym("f");
        let g = Expr::sym("g");
        let e = Expr::app(f.clone(), Expr::app(g, v(2)));
        assert_eq!(print(&e), "(f (g $2))");
        // A lambda in head position is parenthesised and the spine flattened.
        let e = Expr::app(Expr::app(Expr::lam(v(0)), f), Expr::sym("x"));
        assert_eq!(print(&e), "((lam $0) f x)");
        assert_eq!(print(&Expr::sym("int_-3")), "int_-3");
    }

    #[test]
    fn parses_left_associated_applications() {
        let parsed = parse("(f a b)").unwrap();
        let expected = Expr::app(Expr::app(Expr::sym("f"), Expr::sym("a")), Expr::sym("b"));
        assert_eq!(parsed, expected);
        // Stitch's printer flattens, so the nested spelling reads the same.
        assert_eq!(parse("((f a) b)").unwrap(), expected);
    }

    #[test]
    fn parses_variables_parameters_lambdas_and_tokens() {
        let parsed = parse("  (lambda (lam ($0 #12 int_-9223372036854775808)))\n").unwrap();
        let expected = Expr::lam(Expr::lam(Expr::app(
            Expr::app(v(0), Expr::IVar(12)),
            Expr::sym("int_-9223372036854775808"),
        )));
        assert_eq!(parsed, expected);
    }

    #[test]
    fn print_then_parse_is_identity_on_examples() {
        for src in [
            "$0",
            "#3",
            "fn_0",
            "(lam $0)",
            "(lam (lam ($1 (#0 $0) #1)))",
            "(let_ int_5 (lam (if_ (lt $0 int_3) true (and_ false (or_ true false)))))",
            "((lam $0) (f a) (lam b))",
        ] {
            let e = parse(src).unwrap_or_else(|err| panic!("{src}: {err}"));
            assert_eq!(print(&e), src);
        }
    }

    #[test]
    fn rejects_malformed_input() {
        for (src, expect) in [
            ("", "unexpected end of input"),
            ("   ", "unexpected end of input"),
            ("()", "empty application"),
            ("(f)", "at least one argument"),
            ("(lam)", "'lam' takes exactly one argument"),
            ("(lam a b)", "'lam' takes exactly one argument"),
            ("lam", "'lam' must open a form"),
            ("(f lam)", "'lam' must open a form"),
            ("(lam_3 a)", "tagged lambda"),
            ("$0_3", "bad variable"),
            ("$x", "bad variable"),
            ("$", "bad variable"),
            ("$+1", "bad variable"),
            ("$99999999999", "bad variable"),
            ("#x", "bad parameter"),
            ("#-1", "bad parameter"),
            ("??", "hole"),
            ("a b", "unexpected input after expression"),
            (")", "unexpected ')'"),
            ("(f a", "unexpected end of input"),
        ] {
            let e = parse(src).expect_err(src);
            assert!(e.message.contains(expect), "{src:?}: got {e}");
        }
    }
}
