//! Surface abstract syntax with source spans.
//!
//! The parser accepts patterns of any depth here, so that the desugarer can reject
//! nested ones with a message that names the exact pattern and suggests the
//! one-level rewrite.

use crate::error::Span;

/// A built-in constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ctor {
    /// `None`, the empty option.
    None,
    /// `Some x`.
    Some,
    /// `Leaf`, the empty tree.
    Leaf,
    /// `Node (l, x, r)`.
    Node,
}

impl Ctor {
    /// The constructor's surface name.
    pub fn name(self) -> &'static str {
        match self {
            Ctor::None => "None",
            Ctor::Some => "Some",
            Ctor::Leaf => "Leaf",
            Ctor::Node => "Node",
        }
    }
}

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    /// `||`
    Or,
    /// `&&`
    And,
    /// `=` (integer equality)
    IntEq,
    /// `==` (structural equality)
    Eq,
    /// `<`
    Lt,
    /// `<=`
    Le,
    /// `>`
    Gt,
    /// `>=`
    Ge,
    /// `::`
    Cons,
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `mod`
    Mod,
}

impl BinOp {
    /// The operator as written.
    pub fn text(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::IntEq => "=",
            BinOp::Eq => "==",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Cons => "::",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "mod",
        }
    }
}

/// An expression and its span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    /// What the expression is.
    pub kind: ExprKind,
    /// Where it is.
    pub span: Span,
}

/// Expression forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprKind {
    /// Integer literal (a leading `-` is already folded in).
    Int(i64),
    /// `true` / `false`.
    Bool(bool),
    /// `()`.
    Unit,
    /// A name.
    Var(String),
    /// A constructor used as a value or function.
    Ctor(Ctor),
    /// An operator in parentheses, used as a function: `( + )`.
    OpSection(BinOp),
    /// `f a b ...` (at least one argument).
    App(Box<Expr>, Vec<Expr>),
    /// `a op b`.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// `-e` where `e` is not an integer literal.
    Neg(Box<Expr>),
    /// `(a, b, ...)` with at least two elements.
    Tuple(Vec<Expr>),
    /// `[a; b; ...]`, possibly empty.
    List(Vec<Expr>),
    /// `fun p1 p2 ... -> body`.
    Fun(Vec<Pat>, Box<Expr>),
    /// `let ... = rhs in body`.
    Let(LetLhs, Box<Expr>, Box<Expr>),
    /// `if c then t else e`.
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    /// `match e with arms`.
    Match(Box<Expr>, Vec<Arm>),
}

/// The left side of a `let`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LetLhs {
    /// `let name p1 p2 ... = ...` (no parameters for a plain value).
    Name {
        /// The bound name.
        name: String,
        /// Where the name is.
        span: Span,
        /// Parameters, if this defines a function.
        params: Vec<Pat>,
    },
    /// `let (a, b) = ...` or `let _ = ...`.
    Pattern(Pat),
}

/// One `pattern -> body` arm of a match.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arm {
    /// The pattern.
    pub pat: Pat,
    /// The body.
    pub body: Expr,
}

/// A pattern and its span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pat {
    /// What the pattern is.
    pub kind: PatKind,
    /// Where it is.
    pub span: Span,
}

/// Pattern forms, of any depth (the desugarer limits the depth).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatKind {
    /// A name.
    Var(String),
    /// `_`.
    Wild,
    /// `()`.
    Unit,
    /// An integer literal.
    Int(i64),
    /// `true` / `false`.
    Bool(bool),
    /// `(p, q, ...)`.
    Tuple(Vec<Pat>),
    /// `[]`.
    Nil,
    /// `[p; q; ...]` with at least one element.
    List(Vec<Pat>),
    /// `p :: q`.
    Cons(Box<Pat>, Box<Pat>),
    /// A constructor with an optional argument pattern.
    Ctor(Ctor, Option<Box<Pat>>),
}

/// A top-level definition `let name p1 ... = body`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Def {
    /// The defined name.
    pub name: String,
    /// Where the name is.
    pub name_span: Span,
    /// Parameters.
    pub params: Vec<Pat>,
    /// The body.
    pub body: Expr,
}
