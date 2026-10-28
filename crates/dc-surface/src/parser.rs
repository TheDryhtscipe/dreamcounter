//! Recursive-descent parser for dc surface syntax.
//!
//! Grammar, lowest precedence first (all binary operators as in OCaml):
//!
//! ```text
//! program ::= def*
//! def     ::= 'let' name param* '=' expr
//! expr    ::= 'let' name param* '=' expr 'in' expr
//!           | 'let' ( '(' pat, ... ')' | '_' ) '=' expr 'in' expr
//!           | 'fun' param+ '->' expr
//!           | 'if' expr 'then' expr 'else' expr
//!           | 'match' expr 'with' '|'? pat '->' expr ('|' pat '->' expr)*
//!           | binary
//! binary  ::= operators, loosest first:
//!             ||  (right)   &&  (right)
//!             = == < <= > >=  (non-associative: 'a < b < c' is an error)
//!             ::  (right)   + -  (left)   * / mod  (left)
//! unary   ::= '-' unary | app      ('-' before an integer literal makes a
//!                                   negative literal; 'let', 'fun', 'if' and
//!                                   'match' may also appear here, extending as
//!                                   far right as possible)
//! app     ::= atom atom*
//! atom    ::= int | name | Ctor | true | false | '()' | '(' op ')'
//!           | '(' expr ')' | '(' expr ',' expr (',' expr)* ')'
//!           | '[' ']' | '[' expr (';' expr)* ';'? ']'
//! ```
//!
//! As in OCaml, `f -1` is a subtraction; write `f (-1)`. A tuple needs parentheses.
//!
//! # Nesting limit
//!
//! The parser, the desugarer and the pretty-printer recurse on the host stack, and
//! a stack overflow aborts the process. So the parser rejects input nested more
//! than [`MAX_DEPTH`] deep with an [`ErrorKind::TooDeep`] error, in two checks:
//!
//! 1. While parsing, each nested expression, operand of a right-associative or
//!    unary operator, and pattern counts one level, as does each operator of a
//!    chain like `a + b + c` (which is built in a loop, and is deep all the same).
//! 2. After parsing, [`check_depth`] measures, without recursion, how deeply the
//!    core term will nest: each node is one level, and a list, tuple, argument
//!    list or parameter list adds a level per element, since each becomes one
//!    more nested application, lambda or `let` in core.
//!
//! From level [`HANDOFF_DEPTH`] of the first measure down, parsing goes on on a
//! thread of its own (see [`crate::stack`]).

use crate::ast::{Arm, BinOp, Ctor, Def, Expr, ExprKind, LetLhs, Pat, PatKind};
use crate::error::{ErrorKind, Source, Span, SurfaceError};
use crate::lexer::{lex, Tok, Token};
use crate::stack::{on_deep_stack, HANDOFF_DEPTH};
use crate::MAX_DEPTH;

type PResult<T> = Result<T, SurfaceError>;

/// Parses a whole program: a sequence of top-level definitions.
pub(crate) fn parse_program(src: &Source) -> PResult<Vec<Def>> {
    let mut p = Parser::new(src)?;
    let mut defs = vec![];
    loop {
        match p.peek() {
            Tok::Eof => break,
            Tok::Let => {
                let def = p.top_def()?;
                check_depth(src, &def.body, 1 + pats_weight(&def.params))?;
                defs.push(def);
            }
            _ => {
                return Err(
                    p.expected("a top-level definition 'let name args = expr' or end of input")
                )
            }
        }
    }
    Ok(defs)
}

/// Parses one expression that must use the whole input.
pub(crate) fn parse_expression(src: &Source) -> PResult<Expr> {
    let mut p = Parser::new(src)?;
    let e = p.expr()?;
    if *p.peek() != Tok::Eof {
        return Err(p.expected("end of input after the expression"));
    }
    check_depth(src, &e, 0)?;
    Ok(e)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Assoc {
    Left,
    Right,
    None,
}

fn binop_info(t: &Tok) -> Option<(BinOp, u8, Assoc)> {
    Some(match t {
        Tok::OrOr => (BinOp::Or, 1, Assoc::Right),
        Tok::AndAnd => (BinOp::And, 2, Assoc::Right),
        Tok::Eq => (BinOp::IntEq, 3, Assoc::None),
        Tok::EqEq => (BinOp::Eq, 3, Assoc::None),
        Tok::Lt => (BinOp::Lt, 3, Assoc::None),
        Tok::Le => (BinOp::Le, 3, Assoc::None),
        Tok::Gt => (BinOp::Gt, 3, Assoc::None),
        Tok::Ge => (BinOp::Ge, 3, Assoc::None),
        Tok::ColonColon => (BinOp::Cons, 4, Assoc::Right),
        Tok::Plus => (BinOp::Add, 5, Assoc::Left),
        Tok::Minus => (BinOp::Sub, 5, Assoc::Left),
        Tok::Star => (BinOp::Mul, 6, Assoc::Left),
        Tok::Slash => (BinOp::Div, 6, Assoc::Left),
        Tok::Mod => (BinOp::Mod, 6, Assoc::Left),
        _ => return None,
    })
}

/// The error for input nested deeper than [`MAX_DEPTH`].
fn too_deep(src: &Source, span: Span) -> SurfaceError {
    src.error(
        ErrorKind::TooDeep,
        span,
        format!(
            "this expression is nested more than {MAX_DEPTH} levels deep (brackets, \
             operators, let/fun/if/match, and the elements of a list, tuple or \
             argument list all count). Split it into separate top-level definitions"
        ),
    )
}

/// How many levels a binder pattern adds in core: one per node.
fn pat_weight(p: &Pat) -> usize {
    1 + match &p.kind {
        PatKind::Tuple(ps) | PatKind::List(ps) => pats_weight(ps),
        PatKind::Cons(a, b) => pat_weight(a) + pat_weight(b),
        PatKind::Ctor(_, Some(a)) => pat_weight(a),
        _ => 0,
    }
}

fn pats_weight(ps: &[Pat]) -> usize {
    ps.iter().map(pat_weight).sum()
}

/// Checks, without recursion, that `e` (starting at depth `base`) and the core
/// term it becomes nest at most [`MAX_DEPTH`] deep.
fn check_depth(src: &Source, e: &Expr, base: usize) -> PResult<()> {
    let mut stack: Vec<(&Expr, usize)> = vec![(e, base + 1)];
    while let Some((e, d)) = stack.pop() {
        if d > MAX_DEPTH {
            return Err(too_deep(src, e.span));
        }
        match &e.kind {
            ExprKind::Int(_)
            | ExprKind::Bool(_)
            | ExprKind::Unit
            | ExprKind::Var(_)
            | ExprKind::Ctor(_)
            | ExprKind::OpSection(_) => {}
            ExprKind::App(f, args) => {
                let inner = d + args.len();
                stack.push((f, inner));
                stack.extend(args.iter().map(|a| (a, inner)));
            }
            ExprKind::Bin(op, a, b) => {
                // '>' and '>=' may add a let.
                let extra = usize::from(matches!(op, BinOp::Gt | BinOp::Ge));
                stack.push((a, d + 1));
                stack.push((b, d + 1 + extra));
            }
            ExprKind::Neg(a) => stack.push((a, d + 1)),
            ExprKind::Tuple(es) | ExprKind::List(es) => {
                stack.extend(es.iter().enumerate().map(|(i, x)| (x, d + 1 + i)));
            }
            ExprKind::Fun(ps, body) => stack.push((body, d + pats_weight(ps))),
            ExprKind::Let(lhs, rhs, body) => {
                let w = match lhs {
                    LetLhs::Name { params, .. } => 1 + pats_weight(params),
                    LetLhs::Pattern(p) => pat_weight(p),
                };
                stack.push((rhs, d + w));
                stack.push((body, d + w));
            }
            ExprKind::If(c, t, f) => {
                stack.extend([(&**c, d + 1), (&**t, d + 1), (&**f, d + 1)]);
            }
            ExprKind::Match(scrutinee, arms) => {
                stack.push((scrutinee, d + 1));
                for arm in arms {
                    stack.push((&arm.body, d + 1 + pat_weight(&arm.pat)));
                }
            }
        }
    }
    Ok(())
}

fn starts_atom(t: &Tok) -> bool {
    matches!(
        t,
        Tok::Int(_)
            | Tok::Ident(_)
            | Tok::Ctor(_)
            | Tok::True
            | Tok::False
            | Tok::LParen
            | Tok::LBracket
            | Tok::Underscore
    )
}

fn starts_pat_atom(t: &Tok) -> bool {
    matches!(
        t,
        Tok::Int(_)
            | Tok::Ident(_)
            | Tok::Ctor(_)
            | Tok::True
            | Tok::False
            | Tok::LParen
            | Tok::LBracket
            | Tok::Underscore
            | Tok::Minus
    )
}

struct Parser<'s> {
    toks: Vec<Token>,
    pos: usize,
    src: &'s Source,
    /// Current nesting of recursive calls; see the module docs.
    depth: usize,
}

impl<'s> Parser<'s> {
    fn new(src: &'s Source) -> PResult<Parser<'s>> {
        Ok(Parser {
            toks: lex(src)?,
            pos: 0,
            src,
            depth: 0,
        })
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos + 1 < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn err(&self, kind: ErrorKind, span: Span, msg: impl Into<String>) -> SurfaceError {
        self.src.error(kind, span, msg)
    }

    /// "expected WHAT, found TOKEN" at the current token.
    fn expected(&self, what: &str) -> SurfaceError {
        self.err(
            ErrorKind::Syntax,
            self.span(),
            format!("expected {what}, found {}", self.peek().describe()),
        )
    }

    fn expect(&mut self, t: &Tok, what: &str) -> PResult<Span> {
        if self.peek() == t {
            Ok(self.bump().span)
        } else {
            Err(self.expected(what))
        }
    }

    /// Rejects `let rec` with the dc alternative.
    fn reject_rec(&self) -> PResult<()> {
        if *self.peek() == Tok::Rec {
            return Err(self.err(
                ErrorKind::SelfReference,
                self.span(),
                "dc has no 'let rec': recursion goes through the combinators. \
                 Use foldl/foldr/loop for structural recursion, or write \
                 'let f x = bfix FUEL (fun self x -> ... self y ...) x' (or memorec \
                 for memoised recursion), where FUEL bounds the recursion depth, \
                 for example 'length xs + 1'",
            ));
        }
        Ok(())
    }

    fn top_def(&mut self) -> PResult<Def> {
        self.bump(); // let
        self.reject_rec()?;
        let (name, name_span) = match self.peek().clone() {
            Tok::Ident(n) => (n, self.bump().span),
            _ => {
                return Err(self.expected(
                    "the name being defined after 'let' (a top-level definition binds \
                     one name; take tuples apart inside its body)",
                ))
            }
        };
        let params = self.params()?;
        self.expect(&Tok::Eq, &format!("'=' in the definition of '{name}'"))?;
        let body = self.expr()?;
        match self.peek() {
            Tok::Eof | Tok::Let => {}
            Tok::In => {
                return Err(self.err(
                    ErrorKind::Syntax,
                    self.span(),
                    format!(
                        "unexpected 'in' after the top-level definition of '{name}': \
                         top-level definitions have no 'in' (only local 'let ... in' \
                         inside an expression does)"
                    ),
                ))
            }
            _ => {
                return Err(self.expected(&format!(
                    "a new top-level 'let' or end of input after the definition of '{name}'"
                )))
            }
        }
        Ok(Def {
            name,
            name_span,
            params,
            body,
        })
    }

    /// Parameters: pattern atoms up to the terminator token.
    fn params(&mut self) -> PResult<Vec<Pat>> {
        let mut params = vec![];
        while starts_pat_atom(self.peek()) {
            params.push(self.pat_atom()?);
        }
        Ok(params)
    }

    /// Runs `f` one nesting level deeper, or fails past [`MAX_DEPTH`]. The level
    /// [`HANDOFF_DEPTH`] runs on a thread of its own.
    fn nested<T: Send>(&mut self, f: impl FnOnce(&mut Self) -> PResult<T> + Send) -> PResult<T> {
        if self.depth >= MAX_DEPTH {
            return Err(too_deep(self.src, self.span()));
        }
        self.depth += 1;
        let r = if self.depth == HANDOFF_DEPTH {
            on_deep_stack("dc-parse", || f(self))
        } else {
            f(self)
        };
        self.depth -= 1;
        r
    }

    fn expr(&mut self) -> PResult<Expr> {
        self.nested(Self::expr_inner)
    }

    fn expr_inner(&mut self) -> PResult<Expr> {
        match self.peek() {
            Tok::Let => self.let_expr(),
            Tok::Fun => self.fun_expr(),
            Tok::If => self.if_expr(),
            Tok::Match => self.match_expr(),
            _ => self.binary(0),
        }
    }

    fn let_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span; // let
        self.reject_rec()?;
        let lhs = match self.peek().clone() {
            Tok::Ident(name) => {
                let span = self.bump().span;
                let params = self.params()?;
                LetLhs::Name { name, span, params }
            }
            Tok::LParen | Tok::Underscore => LetLhs::Pattern(self.pat_atom()?),
            _ => {
                return Err(
                    self.expected("a name, '_' or a tuple pattern like '(a, b)' after 'let'")
                )
            }
        };
        self.expect(&Tok::Eq, "'=' in 'let'")?;
        let rhs = self.expr()?;
        let what = match &lhs {
            LetLhs::Name { name, .. } => format!(
                "'in' after the value of 'let {name} = ...' (a local definition is \
                 'let x = e in body')"
            ),
            LetLhs::Pattern(_) => "'in' after the value of 'let'".to_owned(),
        };
        self.expect(&Tok::In, &what)?;
        let body = self.expr()?;
        let span = start.to(body.span);
        Ok(Expr {
            kind: ExprKind::Let(lhs, Box::new(rhs), Box::new(body)),
            span,
        })
    }

    fn fun_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span; // fun
        let params = self.params()?;
        if params.is_empty() {
            return Err(self.expected(
                "at least one parameter after 'fun' (a name, '_', or a tuple like '(a, b)')",
            ));
        }
        self.expect(&Tok::Arrow, "'->' after the parameters of 'fun'")?;
        let body = self.expr()?;
        let span = start.to(body.span);
        Ok(Expr {
            kind: ExprKind::Fun(params, Box::new(body)),
            span,
        })
    }

    fn if_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span; // if
        let c = self.expr()?;
        self.expect(&Tok::Then, "'then' after the condition of 'if'")?;
        let t = self.expr()?;
        self.expect(
            &Tok::Else,
            "'else': every 'if' in dc needs an 'else' branch",
        )?;
        let e = self.expr()?;
        let span = start.to(e.span);
        Ok(Expr {
            kind: ExprKind::If(Box::new(c), Box::new(t), Box::new(e)),
            span,
        })
    }

    fn match_expr(&mut self) -> PResult<Expr> {
        let start = self.bump().span; // match
        let scrutinee = self.expr()?;
        self.expect(&Tok::With, "'with' after the value being matched")?;
        self.eat(&Tok::Bar);
        let mut arms = vec![];
        loop {
            let pat = self.pattern()?;
            self.expect(&Tok::Arrow, "'->' after the pattern of a match arm")?;
            let body = self.expr()?;
            arms.push(Arm { pat, body });
            if !self.eat(&Tok::Bar) {
                break;
            }
        }
        let span = start.to(arms.last().map_or(start, |a| a.body.span));
        Ok(Expr {
            kind: ExprKind::Match(Box::new(scrutinee), arms),
            span,
        })
    }

    fn binary(&mut self, min_prec: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        // Each operator of a left-associative chain nests the tree one level
        // deeper without a recursive call, so it counts here.
        let mut chain = 0;
        while let Some((op, prec, assoc)) = binop_info(self.peek()) {
            if prec < min_prec {
                break;
            }
            chain += 1;
            if self.depth + chain > MAX_DEPTH {
                return Err(too_deep(self.src, self.span()));
            }
            self.bump();
            let next_min = if assoc == Assoc::Right {
                prec
            } else {
                prec + 1
            };
            let rhs = self.nested(|p| p.binary(next_min))?;
            if assoc == Assoc::None {
                if let Some((op2, prec2, _)) = binop_info(self.peek()) {
                    if prec2 == prec {
                        return Err(self.err(
                            ErrorKind::Syntax,
                            self.span(),
                            format!(
                                "comparisons do not chain: 'a {} b {} c' is not allowed; \
                                 write 'a {} b && b {} c' or add parentheses",
                                op.text(),
                                op2.text(),
                                op.text(),
                                op2.text()
                            ),
                        ));
                    }
                }
            }
            let span = lhs.span.to(rhs.span);
            lhs = Expr {
                kind: ExprKind::Bin(op, Box::new(lhs), Box::new(rhs)),
                span,
            };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        match self.peek() {
            Tok::Minus => {
                let start = self.bump().span;
                if let Tok::Int(n) = *self.peek() {
                    let lit = self.bump().span;
                    let value = negate(n).ok_or_else(|| {
                        self.err(
                            ErrorKind::Lex,
                            start.to(lit),
                            format!("integer literal -{n} is too large for a 64-bit Int"),
                        )
                    })?;
                    let head = Expr {
                        kind: ExprKind::Int(value),
                        span: start.to(lit),
                    };
                    return self.app_args(head);
                }
                let e = self.nested(Self::unary)?;
                let span = start.to(e.span);
                Ok(Expr {
                    kind: ExprKind::Neg(Box::new(e)),
                    span,
                })
            }
            Tok::Let | Tok::Fun | Tok::If | Tok::Match => self.expr(),
            _ => {
                let head = self.atom()?;
                self.app_args(head)
            }
        }
    }

    fn app_args(&mut self, head: Expr) -> PResult<Expr> {
        let mut args = vec![];
        while starts_atom(self.peek()) {
            args.push(self.atom()?);
        }
        if args.is_empty() {
            return Ok(head);
        }
        let span = head.span.to(args[args.len() - 1].span);
        Ok(Expr {
            kind: ExprKind::App(Box::new(head), args),
            span,
        })
    }

    fn ctor(&self, name: &str, span: Span) -> PResult<Ctor> {
        Ok(match name {
            "None" => Ctor::None,
            "Some" => Ctor::Some,
            "Leaf" => Ctor::Leaf,
            "Node" => Ctor::Node,
            _ => {
                let hint = match name {
                    "True" | "False" => format!(" (booleans are '{}')", name.to_lowercase()),
                    "Nil" => " (the empty list is '[]')".to_owned(),
                    "Cons" => " (list cons is 'x :: xs')".to_owned(),
                    "Just" => " (use 'Some')".to_owned(),
                    "Nothing" => " (use 'None')".to_owned(),
                    _ => String::new(),
                };
                return Err(self.err(
                    ErrorKind::UnknownName,
                    span,
                    format!(
                        "unknown constructor '{name}'{hint}: dc has no user-defined types; \
                         its constructors are None, Some, Leaf and Node"
                    ),
                ));
            }
        })
    }

    fn atom(&mut self) -> PResult<Expr> {
        let tok = self.bump();
        let span = tok.span;
        let kind = match tok.tok {
            Tok::Int(n) => match i64::try_from(n) {
                Ok(v) => ExprKind::Int(v),
                Err(_) => {
                    return Err(self.err(
                        ErrorKind::Lex,
                        span,
                        format!(
                            "integer literal {n} is too large for a 64-bit Int \
                             (the largest is {})",
                            i64::MAX
                        ),
                    ))
                }
            },
            Tok::Ident(name) => ExprKind::Var(name),
            Tok::Ctor(name) => ExprKind::Ctor(self.ctor(&name, span)?),
            Tok::True => ExprKind::Bool(true),
            Tok::False => ExprKind::Bool(false),
            Tok::Underscore => {
                return Err(self.err(
                    ErrorKind::Syntax,
                    span,
                    "'_' can only be used as a pattern, not as a value",
                ))
            }
            Tok::LParen => return self.paren(span),
            Tok::LBracket => return self.list(span),
            other => {
                return Err(self.err(
                    ErrorKind::Syntax,
                    span,
                    format!("expected an expression, found {}", other.describe()),
                ))
            }
        };
        Ok(Expr { kind, span })
    }

    fn close(&mut self, closer: &Tok, opener: &str, open: Span) -> PResult<Span> {
        let what = format!(
            "'{}' to close the '{opener}' at {}",
            match closer {
                Tok::RParen => ")",
                _ => "]",
            },
            self.src.describe_position(open.start)
        );
        self.expect(closer, &what)
    }

    fn paren(&mut self, open: Span) -> PResult<Expr> {
        if *self.peek() == Tok::RParen {
            let end = self.bump().span;
            return Ok(Expr {
                kind: ExprKind::Unit,
                span: open.to(end),
            });
        }
        if *self.peek_at(1) == Tok::RParen {
            if let Some((op, _, _)) = binop_info(self.peek()) {
                let op_span = self.bump().span;
                let end = self.bump().span;
                return match op {
                    BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or => {
                        let hint = match op {
                            BinOp::Gt => "use '( < )' with the arguments swapped",
                            BinOp::Ge => "use '( <= )' with the arguments swapped",
                            _ => "write 'fun a b -> a OP b' (it is a short-circuit form, not a function)",
                        };
                        Err(self.err(
                            ErrorKind::Syntax,
                            op_span,
                            format!("'( {} )' is not available as a function: {hint}", op.text()),
                        ))
                    }
                    _ => Ok(Expr {
                        kind: ExprKind::OpSection(op),
                        span: open.to(end),
                    }),
                };
            }
        }
        let first = self.expr()?;
        if *self.peek() == Tok::Comma {
            let mut elems = vec![first];
            while self.eat(&Tok::Comma) {
                elems.push(self.expr()?);
            }
            let end = self.close(&Tok::RParen, "(", open)?;
            return Ok(Expr {
                kind: ExprKind::Tuple(elems),
                span: open.to(end),
            });
        }
        let end = self.close(&Tok::RParen, "(", open)?;
        Ok(Expr {
            kind: first.kind,
            span: open.to(end),
        })
    }

    fn list(&mut self, open: Span) -> PResult<Expr> {
        let mut elems = vec![];
        while *self.peek() != Tok::RBracket {
            elems.push(self.expr()?);
            if *self.peek() == Tok::Comma {
                return Err(self.err(
                    ErrorKind::Syntax,
                    self.span(),
                    "list elements are separated by ';' in dc, as in OCaml: [1; 2; 3] \
                     (write [(a, b)] for a list holding one tuple)",
                ));
            }
            if !self.eat(&Tok::Semi) {
                break;
            }
        }
        let end = self.close(&Tok::RBracket, "[", open)?;
        Ok(Expr {
            kind: ExprKind::List(elems),
            span: open.to(end),
        })
    }

    // ---- patterns ----

    fn pattern(&mut self) -> PResult<Pat> {
        self.nested(Self::pattern_inner)
    }

    fn pattern_inner(&mut self) -> PResult<Pat> {
        let head = self.pat_app()?;
        if *self.peek() == Tok::ColonColon {
            self.bump();
            let tail = self.pattern()?;
            let span = head.span.to(tail.span);
            return Ok(Pat {
                kind: PatKind::Cons(Box::new(head), Box::new(tail)),
                span,
            });
        }
        Ok(head)
    }

    fn pat_app(&mut self) -> PResult<Pat> {
        if let Tok::Ctor(name) = self.peek().clone() {
            let span = self.bump().span;
            let c = self.ctor(&name, span)?;
            if starts_pat_atom(self.peek()) && *self.peek() != Tok::Minus {
                let arg = self.pat_atom()?;
                let span = span.to(arg.span);
                return Ok(Pat {
                    kind: PatKind::Ctor(c, Some(Box::new(arg))),
                    span,
                });
            }
            return Ok(Pat {
                kind: PatKind::Ctor(c, None),
                span,
            });
        }
        self.pat_atom()
    }

    fn pat_atom(&mut self) -> PResult<Pat> {
        let tok = self.bump();
        let span = tok.span;
        let kind = match tok.tok {
            Tok::Ident(n) => PatKind::Var(n),
            Tok::Underscore => PatKind::Wild,
            Tok::True => PatKind::Bool(true),
            Tok::False => PatKind::Bool(false),
            Tok::Int(n) => PatKind::Int(i64::try_from(n).unwrap_or(i64::MAX)),
            Tok::Minus => match *self.peek() {
                Tok::Int(n) => {
                    self.bump();
                    PatKind::Int(negate(n).unwrap_or(i64::MIN))
                }
                _ => return Err(self.expected("an integer after '-' in a pattern")),
            },
            Tok::Ctor(name) => PatKind::Ctor(self.ctor(&name, span)?, None),
            Tok::LBracket => {
                let mut elems = vec![];
                while *self.peek() != Tok::RBracket {
                    elems.push(self.pattern()?);
                    if !self.eat(&Tok::Semi) {
                        break;
                    }
                }
                let end = self.close(&Tok::RBracket, "[", span)?;
                let kind = if elems.is_empty() {
                    PatKind::Nil
                } else {
                    PatKind::List(elems)
                };
                return Ok(Pat {
                    kind,
                    span: span.to(end),
                });
            }
            Tok::LParen => {
                if *self.peek() == Tok::RParen {
                    let end = self.bump().span;
                    return Ok(Pat {
                        kind: PatKind::Unit,
                        span: span.to(end),
                    });
                }
                let first = self.pattern()?;
                let mut elems = vec![first];
                while self.eat(&Tok::Comma) {
                    elems.push(self.pattern()?);
                }
                let end = self.close(&Tok::RParen, "(", span)?;
                if elems.len() == 1 {
                    let p = elems.remove(0);
                    return Ok(Pat {
                        kind: p.kind,
                        span: span.to(end),
                    });
                }
                return Ok(Pat {
                    kind: PatKind::Tuple(elems),
                    span: span.to(end),
                });
            }
            other => {
                return Err(self.err(
                    ErrorKind::Syntax,
                    span,
                    format!("expected a pattern, found {}", other.describe()),
                ))
            }
        };
        Ok(Pat { kind, span })
    }
}

/// `-n` as an i64, if it fits (2^63 is allowed and gives `i64::MIN`).
fn negate(n: u64) -> Option<i64> {
    if n == 1u64 << 63 {
        Some(i64::MIN)
    } else {
        i64::try_from(n).ok().map(|v| -v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Expr {
        parse_expression(&Source::new("t", s)).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn nested_match_in_last_arm_takes_the_remaining_arms() {
        let e = parse("match a with [] -> 0 | x :: t -> match t with [] -> 1 | _ -> 2");
        let ExprKind::Match(_, arms) = e.kind else {
            panic!("not a match")
        };
        assert_eq!(arms.len(), 2);
        assert!(matches!(arms[1].body.kind, ExprKind::Match(_, ref inner) if inner.len() == 2));
    }

    #[test]
    fn spans_cover_whole_expressions() {
        let e = parse("  f x (y + 1)");
        assert_eq!(e.span, Span::new(2, 13));
    }

    #[test]
    fn program_is_a_sequence_of_definitions() {
        let defs = parse_program(&Source::new("t", "let a = 1\nlet f x y = x\n")).unwrap();
        let names: Vec<_> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["a", "f"]);
        assert_eq!(defs[1].params.len(), 2);
    }
}
