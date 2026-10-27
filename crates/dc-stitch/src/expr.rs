//! Stitch's expression syntax, as an owned tree with a parser and printer.
//!
//! The grammar and printing follow Stitch's `lambdas` crate (`parse_expr.rs`,
//! revision `2c9bfd0`, the one Stitch pins): tokens are split on whitespace and
//! parentheses, `$i` is a variable, `#i` a parameter, `(lam e)` a lambda,
//! `(lam_N e)` a lambda tagged `N`, and every other token a primitive.
//! Application spines print flattened.
//!
//! This parser is stricter than Stitch's in ways Stitch's own output never
//! exercises, so that malformed input is an error rather than a guess:
//!
//! - variable tags (`$0_3`) and holes (`??`) are rejected;
//! - `(e)` with a single element is rejected (Stitch reads it as `e`, but never
//!   prints it);
//! - `(lambda e)` and `(lambda_N e)` are accepted as synonyms for `(lam e)` and
//!   `(lam_N e)`, as in Stitch.
//!
//! Nothing here recurses on the host stack: parsing, printing, cloning, comparing
//! and dropping all use explicit work lists, so an expression as deep as memory
//! allows (for example a long list literal, which nests one `cons` per element)
//! is handled without a stack overflow.

use std::fmt::{self, Write as _};
use std::mem;

/// A Stitch expression.
///
/// `Clone`, `PartialEq`, `Debug`, `Display` and `Drop` are written by hand so that
/// none of them recurses on the host stack.
pub enum Expr {
    /// De Bruijn variable `$i`.
    Var(u32),
    /// Abstraction parameter `#i`.
    IVar(u32),
    /// `(lam body)`.
    Lam(Box<Expr>),
    /// `(lam_tag body)`: a lambda carrying a Stitch tag. Stitch's
    /// `--fused-lambda-tags` flag forbids abstractions rooted at, and parameters
    /// placed on, lambdas with the listed tags.
    TaggedLam(u32, Box<Expr>),
    /// Application of a function to one argument.
    App(Box<Expr>, Box<Expr>),
    /// Any other token.
    Sym(String),
}

/// The largest lambda tag Stitch accepts: its tags are non-negative `i32`s.
const MAX_TAG: u32 = i32::MAX as u32;

impl Expr {
    /// `(lam body)`.
    pub fn lam(body: Expr) -> Expr {
        Expr::Lam(Box::new(body))
    }

    /// `(lam_tag body)`.
    pub fn tagged_lam(tag: u32, body: Expr) -> Expr {
        Expr::TaggedLam(tag, Box::new(body))
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

    /// A copy of this node with no children, for use while the children are
    /// rebuilt separately.
    fn shallow(&self) -> Shallow {
        match self {
            Expr::Var(i) => Shallow::Leaf(Expr::Var(*i)),
            Expr::IVar(i) => Shallow::Leaf(Expr::IVar(*i)),
            Expr::Sym(s) => Shallow::Leaf(Expr::Sym(s.clone())),
            Expr::Lam(_) => Shallow::Lam(None),
            Expr::TaggedLam(t, _) => Shallow::Lam(Some(*t)),
            Expr::App(..) => Shallow::App,
        }
    }

    /// Moves this node's children into `out`, leaving cheap leaves in their place.
    fn take_children(&mut self, out: &mut Vec<Expr>) {
        match self {
            Expr::Lam(b) | Expr::TaggedLam(_, b) => out.push(mem::replace(&mut **b, Expr::Var(0))),
            Expr::App(f, a) => {
                out.push(mem::replace(&mut **f, Expr::Var(0)));
                out.push(mem::replace(&mut **a, Expr::Var(0)));
            }
            Expr::Var(_) | Expr::IVar(_) | Expr::Sym(_) => {}
        }
    }
}

/// A node without its children: see [`Expr::shallow`].
enum Shallow {
    Leaf(Expr),
    Lam(Option<u32>),
    App,
}

impl Drop for Expr {
    fn drop(&mut self) {
        let mut pending = vec![];
        self.take_children(&mut pending);
        while let Some(mut e) = pending.pop() {
            e.take_children(&mut pending);
            // `e` now has no deep children, so dropping it here is shallow.
        }
    }
}

impl Clone for Expr {
    fn clone(&self) -> Expr {
        enum Job<'a> {
            Visit(&'a Expr),
            Build(Shallow),
        }
        let mut jobs = vec![Job::Visit(self)];
        let mut built: Vec<Expr> = vec![];
        while let Some(job) = jobs.pop() {
            match job {
                Job::Visit(e) => match e.shallow() {
                    Shallow::Leaf(leaf) => built.push(leaf),
                    node => {
                        jobs.push(Job::Build(node));
                        match e {
                            Expr::Lam(b) | Expr::TaggedLam(_, b) => jobs.push(Job::Visit(b)),
                            Expr::App(f, a) => {
                                jobs.push(Job::Visit(a));
                                jobs.push(Job::Visit(f));
                            }
                            Expr::Var(_) | Expr::IVar(_) | Expr::Sym(_) => {}
                        }
                    }
                },
                Job::Build(Shallow::Leaf(leaf)) => built.push(leaf),
                Job::Build(Shallow::Lam(tag)) => {
                    let body = pop(&mut built);
                    built.push(match tag {
                        None => Expr::lam(body),
                        Some(t) => Expr::tagged_lam(t, body),
                    });
                }
                Job::Build(Shallow::App) => {
                    let a = pop(&mut built);
                    let f = pop(&mut built);
                    built.push(Expr::app(f, a));
                }
            }
        }
        pop(&mut built)
    }
}

/// Pops a value a work list has just pushed.
fn pop(built: &mut Vec<Expr>) -> Expr {
    built
        .pop()
        .expect("every build job follows the jobs that push its operands")
}

impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        let mut pairs = vec![(self, other)];
        while let Some(pair) = pairs.pop() {
            match pair {
                (Expr::Var(x), Expr::Var(y)) | (Expr::IVar(x), Expr::IVar(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Expr::Sym(x), Expr::Sym(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Expr::Lam(x), Expr::Lam(y)) => pairs.push((x, y)),
                (Expr::TaggedLam(s, x), Expr::TaggedLam(t, y)) => {
                    if s != t {
                        return false;
                    }
                    pairs.push((x, y));
                }
                (Expr::App(f, x), Expr::App(g, y)) => {
                    pairs.push((x, y));
                    pairs.push((f, g));
                }
                _ => return false,
            }
        }
        true
    }
}

impl Eq for Expr {}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Expr({self})")
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        enum Job<'a> {
            Visit(&'a Expr),
            Text(&'static str),
        }
        let mut jobs = vec![Job::Visit(self)];
        while let Some(job) = jobs.pop() {
            let e = match job {
                Job::Text(s) => {
                    f.write_str(s)?;
                    continue;
                }
                Job::Visit(e) => e,
            };
            match e {
                Expr::Var(i) => write!(f, "${i}")?,
                Expr::IVar(i) => write!(f, "#{i}")?,
                Expr::Sym(s) => f.write_str(s)?,
                Expr::Lam(b) => {
                    f.write_str("(lam ")?;
                    jobs.push(Job::Text(")"));
                    jobs.push(Job::Visit(b));
                }
                Expr::TaggedLam(t, b) => {
                    write!(f, "(lam_{t} ")?;
                    jobs.push(Job::Text(")"));
                    jobs.push(Job::Visit(b));
                }
                Expr::App(..) => {
                    let (head, args) = e.spine();
                    f.write_char('(')?;
                    jobs.push(Job::Text(")"));
                    for a in args.into_iter().rev() {
                        jobs.push(Job::Visit(a));
                        jobs.push(Job::Text(" "));
                    }
                    jobs.push(Job::Visit(head));
                }
            }
        }
        Ok(())
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

/// An open parenthesis whose closing one has not been read yet.
struct Frame {
    /// Byte offset of the `(`.
    open: usize,
    /// `Some(tag)` for a lambda form (`tag` is `None` for plain `lam`), `None` for
    /// an application.
    lam: Option<Option<u32>>,
    /// The expressions read so far inside the parentheses.
    items: Vec<Expr>,
}

impl Frame {
    /// Builds the expression for this form once its `)` has been read.
    fn finish(self) -> Result<Expr, ParseError> {
        let err = |message: &str| ParseError {
            offset: self.open,
            message: message.into(),
        };
        if let Some(tag) = self.lam {
            return match <[Expr; 1]>::try_from(self.items) {
                Ok([body]) => Ok(match tag {
                    None => Expr::lam(body),
                    Some(t) => Expr::tagged_lam(t, body),
                }),
                Err(_) => Err(err("'lam' takes exactly one argument")),
            };
        }
        let mut items = self.items.into_iter();
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

/// Parses one Stitch expression.
pub fn parse(src: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(src);
    let eof = || ParseError {
        offset: src.len(),
        message: "unexpected end of input".into(),
    };
    let mut frames: Vec<Frame> = vec![];
    let mut pos = 0;
    let result = loop {
        let tok = tokens.get(pos).copied().ok_or_else(eof)?;
        pos += 1;
        let done = match tok.tok {
            Tok::Close => match frames.pop() {
                Some(frame) => frame.finish()?,
                None => {
                    return Err(ParseError {
                        offset: tok.offset,
                        message: "unexpected ')'".into(),
                    })
                }
            },
            Tok::Atom(a) => atom(a, tok.offset)?,
            Tok::Open => {
                let lam = match tokens.get(pos) {
                    Some(&Token {
                        tok: Tok::Atom(a),
                        offset,
                    }) => lambda_head(a, offset)?,
                    _ => None,
                };
                if lam.is_some() {
                    pos += 1;
                }
                frames.push(Frame {
                    open: tok.offset,
                    lam,
                    items: vec![],
                });
                continue;
            }
        };
        match frames.last_mut() {
            Some(frame) => frame.items.push(done),
            None => break done,
        }
    };
    if let Some(tok) = tokens.get(pos) {
        return Err(ParseError {
            offset: tok.offset,
            message: "unexpected input after expression".into(),
        });
    }
    Ok(result)
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

/// Parses a decimal index made only of ASCII digits.
fn index(digits: &str) -> Option<u32> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Whether the token after a `(` opens a lambda form: `Some(None)` for `lam` or
/// `lambda`, `Some(Some(tag))` for `lam_tag` or `lambda_tag`, `None` otherwise.
fn lambda_head(a: &str, offset: usize) -> Result<Option<Option<u32>>, ParseError> {
    if a == "lam" || a == "lambda" {
        return Ok(Some(None));
    }
    let Some(tag) = a.strip_prefix("lam_").or_else(|| a.strip_prefix("lambda_")) else {
        return Ok(None);
    };
    match index(tag) {
        Some(t) if t <= MAX_TAG => Ok(Some(Some(t))),
        _ => Err(ParseError {
            offset,
            message: format!("bad lambda tag in '{a}'"),
        }),
    }
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
    if a == "lam" || a == "lambda" || a.starts_with("lam_") || a.starts_with("lambda_") {
        return Err(err(format!("'{a}' must open a form: ({a} ...)")));
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
    fn parses_tagged_lambdas_as_stitch_does() {
        // Stitch's lambdas crate reads `lam_N` and `lambda_N` as a lambda tagged N
        // (a non-negative i32) and prints it back as `lam_N`.
        let parsed = parse("(lam_0 (lambda_7 (lam $0)))").unwrap();
        let expected = Expr::TaggedLam(0, Box::new(Expr::TaggedLam(7, Box::new(Expr::lam(v(0))))));
        assert_eq!(parsed, expected);
        assert_eq!(print(&parsed), "(lam_0 (lam_7 (lam $0)))");
        assert_ne!(parsed, parse("(lam_0 (lam_8 (lam $0)))").unwrap());
        assert_ne!(parsed, parse("(lam (lam_7 (lam $0)))").unwrap());
        assert_eq!(
            parse("(lam_2147483647 x)").unwrap(),
            Expr::TaggedLam(2147483647, Box::new(Expr::sym("x")))
        );
    }

    /// Nesting this deep overflows a test thread's stack if any of parsing,
    /// printing, cloning, comparing or dropping recurses on the host stack.
    const DEEP: usize = 200_000;

    #[test]
    fn deep_lambda_nesting_does_not_use_the_host_stack() {
        let src = format!("{}x{}", "(lam ".repeat(DEEP), ")".repeat(DEEP));
        let e = parse(&src).unwrap();
        assert_eq!(print(&e), src);
        let copy = e.clone();
        assert_eq!(copy, e);
        let other = parse(&format!("{}y{}", "(lam ".repeat(DEEP), ")".repeat(DEEP))).unwrap();
        assert_ne!(other, e);
        let mut depth = 0;
        let mut cur = &e;
        while let Expr::Lam(b) = cur {
            depth += 1;
            cur = b;
        }
        assert_eq!((depth, cur), (DEEP, &Expr::sym("x")));
        drop((e, copy, other));
    }

    #[test]
    fn deep_application_nesting_does_not_use_the_host_stack() {
        // Right-nested arguments, as in a long cons list.
        let src = format!("{}nil{}", "(cons a ".repeat(DEEP), ")".repeat(DEEP));
        let e = parse(&src).unwrap();
        assert_eq!(print(&e), src);
        assert_eq!(e.clone(), e);
        // Left-nested heads read as one flat spine.
        let nested = format!("{}f{}", "(".repeat(DEEP), " a)".repeat(DEEP));
        let e = parse(&nested).unwrap();
        let (head, args) = e.spine();
        assert_eq!((head, args.len()), (&Expr::sym("f"), DEEP));
        let flat = format!("(f{})", " a".repeat(DEEP));
        assert_eq!(print(&e), flat);
        assert_eq!(parse(&flat).unwrap(), e);
    }

    #[test]
    fn deep_unclosed_input_is_an_error() {
        let src = "(lam ".repeat(DEEP);
        let err = parse(&src).unwrap_err();
        assert_eq!(err.message, "unexpected end of input");
        assert_eq!(err.offset, src.len());
    }

    #[test]
    fn print_then_parse_is_identity_on_examples() {
        for src in [
            "$0",
            "#3",
            "fn_0",
            "(lam $0)",
            "(lam (lam ($1 (#0 $0) #1)))",
            "(let_ int_5 (lam_0 (if_ (lt $0 int_3) true (and_ false (or_ true false)))))",
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
            ("(lam_x a)", "bad lambda tag"),
            ("(lam_-1 a)", "bad lambda tag"),
            ("(lam_2147483648 a)", "bad lambda tag"),
            ("(lam_3)", "'lam' takes exactly one argument"),
            ("lam_3", "must open a form"),
            ("(f lambda_0)", "must open a form"),
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
