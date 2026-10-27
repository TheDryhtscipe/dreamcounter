//! Lexer for dc surface syntax.
//!
//! Comments run from `--` to the end of the line. Names are `[a-z_][A-Za-z0-9_']*`;
//! capitalised names are constructors. Integer literals are unsigned here; the
//! parser folds a preceding `-` into the literal.
//!
//! Characters that belong to other languages' syntax (`\`, `.`, `!=`, `<>`, `=>`,
//! `(*`, `:`, `@`, strings) are rejected with a hint at the dc way of writing it.

use crate::error::{ErrorKind, Source, Span, SurfaceError};

/// A token kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Tok {
    Int(u64),
    Ident(String),
    Ctor(String),
    Let,
    Rec,
    In,
    Fun,
    If,
    Then,
    Else,
    Match,
    With,
    True,
    False,
    Mod,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Arrow,
    Bar,
    Underscore,
    Eq,
    EqEq,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    ColonColon,
    AndAnd,
    OrOr,
    Eof,
}

impl Tok {
    /// How the token is named in error messages.
    pub fn describe(&self) -> String {
        match self {
            Tok::Int(n) => format!("integer {n}"),
            Tok::Ident(s) => format!("name '{s}'"),
            Tok::Ctor(s) => format!("constructor '{s}'"),
            Tok::Eof => "end of input".to_owned(),
            other => format!("'{}'", other.text()),
        }
    }

    fn text(&self) -> &'static str {
        match self {
            Tok::Let => "let",
            Tok::Rec => "rec",
            Tok::In => "in",
            Tok::Fun => "fun",
            Tok::If => "if",
            Tok::Then => "then",
            Tok::Else => "else",
            Tok::Match => "match",
            Tok::With => "with",
            Tok::True => "true",
            Tok::False => "false",
            Tok::Mod => "mod",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Comma => ",",
            Tok::Semi => ";",
            Tok::Arrow => "->",
            Tok::Bar => "|",
            Tok::Underscore => "_",
            Tok::Eq => "=",
            Tok::EqEq => "==",
            Tok::Lt => "<",
            Tok::Le => "<=",
            Tok::Gt => ">",
            Tok::Ge => ">=",
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::ColonColon => "::",
            Tok::AndAnd => "&&",
            Tok::OrOr => "||",
            Tok::Int(_) | Tok::Ident(_) | Tok::Ctor(_) | Tok::Eof => "",
        }
    }
}

/// A token and where it is.
#[derive(Clone, Debug)]
pub(crate) struct Token {
    pub tok: Tok,
    pub span: Span,
}

fn keyword(word: &str) -> Option<Tok> {
    Some(match word {
        "let" => Tok::Let,
        "rec" => Tok::Rec,
        "in" => Tok::In,
        "fun" => Tok::Fun,
        "if" => Tok::If,
        "then" => Tok::Then,
        "else" => Tok::Else,
        "match" => Tok::Match,
        "with" => Tok::With,
        "true" => Tok::True,
        "false" => Tok::False,
        "mod" => Tok::Mod,
        "_" => Tok::Underscore,
        _ => return None,
    })
}

/// Whether `word` is reserved and cannot be used as a name.
pub(crate) fn is_keyword(word: &str) -> bool {
    keyword(word).is_some()
}

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'\''
}

/// Splits `src` into tokens, ending with [`Tok::Eof`].
pub(crate) fn lex(src: &Source) -> Result<Vec<Token>, SurfaceError> {
    let text = src.text.as_str();
    let b = text.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    let err = |start: usize, len: usize, msg: String| {
        src.error(ErrorKind::Lex, Span::new(start, start + len), msg)
    };
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if c == b'-' && b.get(i + 1) == Some(&b'-') {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let start = i;
        let next = b.get(i + 1).copied();
        let tok = if c.is_ascii_digit() {
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            if i < b.len() && is_name_byte(b[i]) {
                return Err(err(
                    start,
                    i + 1 - start,
                    "a number must not run into letters (names cannot start with a digit)"
                        .to_owned(),
                ));
            }
            let digits = &text[start..i];
            match digits.parse::<u64>() {
                Ok(n) => Tok::Int(n),
                Err(_) => {
                    return Err(err(
                        start,
                        i - start,
                        format!("integer literal {digits} is too large for a 64-bit Int"),
                    ))
                }
            }
        } else if c.is_ascii_alphabetic() || c == b'_' {
            while i < b.len() && is_name_byte(b[i]) {
                i += 1;
            }
            let word = &text[start..i];
            if b.get(i) == Some(&b'.') && c.is_ascii_uppercase() {
                return Err(err(
                    start,
                    i + 1 - start,
                    format!(
                        "dc has no modules: write the function name alone \
                         (for example 'length', not '{word}.length')"
                    ),
                ));
            }
            match keyword(word) {
                Some(t) => t,
                None if c.is_ascii_uppercase() => Tok::Ctor(word.to_owned()),
                None => Tok::Ident(word.to_owned()),
            }
        } else {
            let (tok, len) =
                match (c, next) {
                    (b'(', Some(b'*')) => {
                        return Err(err(
                            start,
                            2,
                            "dc comments start with '--' and run to the end of the line; \
                         '(*' is not a comment (for the multiplication function write '( * )')"
                                .to_owned(),
                        ))
                    }
                    (b'(', _) => (Tok::LParen, 1),
                    (b')', _) => (Tok::RParen, 1),
                    (b'[', _) => (Tok::LBracket, 1),
                    (b']', _) => (Tok::RBracket, 1),
                    (b',', _) => (Tok::Comma, 1),
                    (b';', Some(b';')) => {
                        return Err(err(
                            start,
                            2,
                            "dc does not use ';;': start each top-level definition with 'let'"
                                .to_owned(),
                        ))
                    }
                    (b';', _) => (Tok::Semi, 1),
                    (b'-', Some(b'>')) => (Tok::Arrow, 2),
                    (b'-', _) => (Tok::Minus, 1),
                    (b'|', Some(b'|')) => (Tok::OrOr, 2),
                    (b'|', _) => (Tok::Bar, 1),
                    (b'&', Some(b'&')) => (Tok::AndAnd, 2),
                    (b'=', Some(b'=')) => (Tok::EqEq, 2),
                    (b'=', Some(b'>')) => return Err(err(
                        start,
                        2,
                        "use '->' after the parameters of 'fun' and after match patterns, not '=>'"
                            .to_owned(),
                    )),
                    (b'=', _) => (Tok::Eq, 1),
                    (b'<', Some(b'=')) => (Tok::Le, 2),
                    (b'<', Some(b'>')) | (b'!', Some(b'=')) => {
                        return Err(err(
                            start,
                            2,
                            "dc has no not-equal operator: write 'not (a = b)' for integers \
                         or 'not (a == b)' for other values"
                                .to_owned(),
                        ))
                    }
                    (b'<', _) => (Tok::Lt, 1),
                    (b'>', Some(b'=')) => (Tok::Ge, 2),
                    (b'>', _) => (Tok::Gt, 1),
                    (b'+', _) => (Tok::Plus, 1),
                    (b'*', _) => (Tok::Star, 1),
                    (b'/', _) => (Tok::Slash, 1),
                    (b':', Some(b':')) => (Tok::ColonColon, 2),
                    _ => {
                        let ch = text[start..].chars().next().unwrap_or('?');
                        let hint = match ch {
                            '\\' => " (dc writes functions as 'fun x -> e')",
                            '.' => " (dc has no modules or records)",
                            ':' => " (dc has no type annotations; '::' is list cons)",
                            '&' => " (conjunction is '&&')",
                            '!' => " (use 'not' for negation)",
                            '@' => " (append lists with the prelude function 'append xs ys')",
                            '"' | '\'' | '^' => " (dc has no strings or characters)",
                            '{' | '}' => " (dc has no records; use tuples)",
                            _ => "",
                        };
                        return Err(err(
                            start,
                            ch.len_utf8(),
                            format!("unexpected character '{ch}'{hint}"),
                        ));
                    }
                };
            i += len;
            tok
        };
        out.push(Token {
            tok,
            span: Span::new(start, i),
        });
    }
    out.push(Token {
        tok: Tok::Eof,
        span: Span::new(b.len(), b.len()),
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<Tok> {
        lex(&Source::new("t", s))
            .unwrap()
            .into_iter()
            .map(|t| t.tok)
            .collect()
    }

    #[test]
    fn lexes_operators_longest_first() {
        assert_eq!(
            toks("a<=b::c->d||e&&f==g>=h"),
            vec![
                Tok::Ident("a".into()),
                Tok::Le,
                Tok::Ident("b".into()),
                Tok::ColonColon,
                Tok::Ident("c".into()),
                Tok::Arrow,
                Tok::Ident("d".into()),
                Tok::OrOr,
                Tok::Ident("e".into()),
                Tok::AndAnd,
                Tok::Ident("f".into()),
                Tok::EqEq,
                Tok::Ident("g".into()),
                Tok::Ge,
                Tok::Ident("h".into()),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn comments_and_keywords() {
        assert_eq!(
            toks("let x' = Some _y -- comment -> ignored\nin mod"),
            vec![
                Tok::Let,
                Tok::Ident("x'".into()),
                Tok::Eq,
                Tok::Ctor("Some".into()),
                Tok::Ident("_y".into()),
                Tok::In,
                Tok::Mod,
                Tok::Eof
            ]
        );
    }

    #[test]
    fn spans_cover_the_token() {
        let t = lex(&Source::new("t", "  abc <=")).unwrap();
        assert_eq!(t[0].span, Span::new(2, 5));
        assert_eq!(t[1].span, Span::new(6, 8));
        assert_eq!(t[2].span, Span::new(8, 8));
    }

    #[test]
    fn rejects_digits_running_into_names() {
        assert!(lex(&Source::new("t", "3x")).is_err());
    }
}
