//! Source text, spans and errors.
//!
//! Every error carries a 1-based line and column, so that it can be shown to the
//! writer (usually an LLM) as `source:line:column: message`.

use std::fmt;

/// A byte range in a source text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Span {
    /// First byte.
    pub start: usize,
    /// One past the last byte.
    pub end: usize,
}

impl Span {
    /// The span from `start` to `end`.
    pub fn new(start: usize, end: usize) -> Span {
        Span { start, end }
    }

    /// The smallest span covering both.
    pub fn to(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

/// A named source text: a program, the prelude, or an expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Source {
    pub name: String,
    pub text: String,
}

impl Source {
    pub fn new(name: &str, text: &str) -> Source {
        Source {
            name: name.to_owned(),
            text: text.to_owned(),
        }
    }

    /// The 1-based line and column (in characters) of a byte offset.
    pub fn position(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.text.len());
        let before = &self.text[..offset];
        let line = before.matches('\n').count() + 1;
        let line_start = before.rfind('\n').map_or(0, |i| i + 1);
        let column = before[line_start..].chars().count() + 1;
        (line, column)
    }

    /// "line L, column C" for use inside messages.
    pub fn describe_position(&self, offset: usize) -> String {
        let (l, c) = self.position(offset);
        format!("line {l}, column {c}")
    }

    pub fn error(&self, kind: ErrorKind, span: Span, message: impl Into<String>) -> SurfaceError {
        let (line, column) = self.position(span.start);
        SurfaceError {
            kind,
            source_name: self.name.clone(),
            line,
            column,
            span,
            message: message.into(),
        }
    }
}

/// What sort of mistake an error reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// A character or literal the lexer cannot read.
    Lex,
    /// Tokens in an order the grammar does not allow.
    Syntax,
    /// A pattern that is not allowed where it appears, or a match that does not
    /// cover its cases.
    Pattern,
    /// A pattern nested inside another pattern (spec §10: patterns are one level
    /// deep).
    NestedPattern,
    /// A definition that refers to itself, or `let rec` (spec §6: there is no
    /// recursion by name).
    SelfReference,
    /// A name that is not bound locally, defined at top level, or a primitive.
    UnknownName,
    /// An expression nested more deeply than [`crate::MAX_DEPTH`] allows.
    TooDeep,
}

/// An error in surface source, with its position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceError {
    /// The sort of error.
    pub kind: ErrorKind,
    /// The name the source was given (for example `program` or `prelude`).
    pub source_name: String,
    /// 1-based line, or 0 when the error has no position (a missing entry point).
    pub line: usize,
    /// 1-based column in characters, or 0 when the error has no position.
    pub column: usize,
    /// The byte range the error points at.
    pub span: Span,
    /// What went wrong and, where possible, how to fix it.
    pub message: String,
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line == 0 {
            write!(f, "{}: {}", self.source_name, self.message)
        } else {
            write!(
                f,
                "{}:{}:{}: {}",
                self.source_name, self.line, self.column, self.message
            )
        }
    }
}

impl std::error::Error for SurfaceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_one_based_and_count_characters() {
        let s = Source::new("t", "ab\nçd\n");
        assert_eq!(s.position(0), (1, 1));
        assert_eq!(s.position(1), (1, 2));
        assert_eq!(s.position(3), (2, 1));
        // 'ç' is two bytes; 'd' is the second character on line 2.
        assert_eq!(s.position(5), (2, 2));
        assert_eq!(s.position(7), (3, 1));
        assert_eq!(s.position(100), (3, 1));
    }
}
