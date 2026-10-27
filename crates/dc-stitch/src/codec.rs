//! Encoding core terms as Stitch expressions and decoding them back.
//!
//! See the crate docs for the encoding table and the abstraction convention.

use std::fmt;

use dc_core::{Arena, LibId, Prim, Term, TermId};

use crate::expr::{self, Expr, ParseError};
use crate::names::LibNames;

/// Reserved token for `Let`: `(let_ bound (lam body))`.
pub const LET: &str = "let_";
/// Reserved token for `If`: `(if_ cond then else)`.
pub const IF: &str = "if_";
/// Reserved token for `And`: `(and_ a b)`.
pub const AND: &str = "and_";
/// Reserved token for `Or`: `(or_ a b)`.
pub const OR: &str = "or_";
/// Prefix of integer literal tokens: `int_3`, `int_-3`.
pub const INT_PREFIX: &str = "int_";

/// Why a core term could not be encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// The term refers to a library abstraction the name table does not name.
    UnnamedLib(LibId),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::UnnamedLib(l) => write!(f, "library {} has no Stitch name", l.0),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Why a Stitch expression could not be decoded into a core term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// The text is not a Stitch expression.
    Parse(ParseError),
    /// A token that is not a literal, primitive, reserved token or named abstraction.
    UnknownToken(String),
    /// An `int_` token whose suffix is not the canonical spelling of an `i64`.
    BadInteger(String),
    /// A variable `$index` with no binder for it.
    FreeVariable {
        /// The variable's index.
        index: u32,
        /// The number of binders in scope where it occurs, including the context.
        binders: u32,
    },
    /// A parameter `#i` outside an abstraction body.
    ParameterInProgram(u32),
    /// A parameter `#index` with `index >= arity`.
    ParameterOutOfRange {
        /// The parameter's index.
        index: u32,
        /// The abstraction's arity.
        arity: u32,
    },
    /// A reserved token not applied to the operands its special form needs.
    MalformedSpecialForm {
        /// The reserved token.
        token: &'static str,
        /// What is wrong.
        message: String,
    },
    /// A de Bruijn index would not fit in a `u32`.
    TooDeep,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Parse(e) => write!(f, "parse error {e}"),
            DecodeError::UnknownToken(s) => write!(f, "unknown token '{s}'"),
            DecodeError::BadInteger(s) => write!(f, "bad integer literal '{s}'"),
            DecodeError::FreeVariable { index, binders } => {
                write!(
                    f,
                    "free variable ${index} with {binders} binder(s) in scope"
                )
            }
            DecodeError::ParameterInProgram(i) => {
                write!(f, "parameter #{i} outside an abstraction body")
            }
            DecodeError::ParameterOutOfRange { index, arity } => {
                write!(f, "parameter #{index} in an abstraction of arity {arity}")
            }
            DecodeError::MalformedSpecialForm { token, message } => {
                write!(f, "malformed '{token}': {message}")
            }
            DecodeError::TooDeep => f.write_str("de Bruijn index does not fit in a u32"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<ParseError> for DecodeError {
    fn from(e: ParseError) -> DecodeError {
        DecodeError::Parse(e)
    }
}

/// Converts the core term `id` to a Stitch expression.
pub fn to_expr(arena: &Arena, id: TermId, names: &LibNames) -> Result<Expr, EncodeError> {
    let enc = |t: TermId| to_expr(arena, t, names);
    let form = |token: &str, args: Vec<Expr>| args.into_iter().fold(Expr::sym(token), Expr::app);
    Ok(match arena.get(id) {
        Term::Var(i) => Expr::Var(i),
        Term::Lam(b) => Expr::lam(enc(b)?),
        Term::App(f, a) => Expr::app(enc(f)?, enc(a)?),
        Term::Let(x, b) => form(LET, vec![enc(x)?, Expr::lam(enc(b)?)]),
        Term::If(c, t, e) => form(IF, vec![enc(c)?, enc(t)?, enc(e)?]),
        Term::And(x, y) => form(AND, vec![enc(x)?, enc(y)?]),
        Term::Or(x, y) => form(OR, vec![enc(x)?, enc(y)?]),
        Term::Int(n) => Expr::sym(format!("{INT_PREFIX}{n}")),
        Term::Bool(b) => Expr::sym(if b { "true" } else { "false" }),
        Term::Unit => Expr::sym("unit"),
        Term::Prim(p) => Expr::sym(p.name()),
        Term::Lib(l) => Expr::sym(names.name(l).ok_or(EncodeError::UnnamedLib(l))?),
    })
}

/// Prints the core term `id` in Stitch's input format.
pub fn encode(arena: &Arena, id: TermId, names: &LibNames) -> Result<String, EncodeError> {
    to_expr(arena, id, names).map(|e| expr::print(&e))
}

/// Converts a Stitch expression to a core term with `context` binders in scope:
/// variables `$i` with `i` at least the binders inside `e` plus `context` are an
/// error. Parameters `#i` are an error.
pub fn from_expr(
    arena: &mut Arena,
    e: &Expr,
    names: &LibNames,
    context: u32,
) -> Result<TermId, DecodeError> {
    Decoder {
        names,
        context,
        arity: None,
    }
    .term(arena, e, 0)
}

/// Decoding state that does not change while descending.
struct Decoder<'n> {
    names: &'n LibNames,
    /// Binders enclosing the whole expression.
    context: u32,
    /// `Some(k)` when decoding the body of an abstraction of `k` parameters.
    arity: Option<u32>,
}

/// The reserved token `s`, as a `'static` str, with its number of operands.
fn special_form(s: &str) -> Option<(&'static str, usize)> {
    match s {
        LET => Some((LET, 2)),
        IF => Some((IF, 3)),
        AND => Some((AND, 2)),
        OR => Some((OR, 2)),
        _ => None,
    }
}

impl Decoder<'_> {
    /// Decodes `e`, found under `depth` binders of the expression being decoded.
    fn term(&self, arena: &mut Arena, e: &Expr, depth: u32) -> Result<TermId, DecodeError> {
        match e {
            Expr::Var(i) => {
                let binders = depth
                    .checked_add(self.context)
                    .ok_or(DecodeError::TooDeep)?;
                if *i < binders {
                    Ok(arena.var(*i))
                } else {
                    Err(DecodeError::FreeVariable { index: *i, binders })
                }
            }
            Expr::IVar(i) => match self.arity {
                None => Err(DecodeError::ParameterInProgram(*i)),
                Some(k) if *i >= k => Err(DecodeError::ParameterOutOfRange {
                    index: *i,
                    arity: k,
                }),
                Some(k) => {
                    let index = depth.checked_add(k - 1 - i).ok_or(DecodeError::TooDeep)?;
                    Ok(arena.var(index))
                }
            },
            Expr::Lam(b) => {
                let inner = depth.checked_add(1).ok_or(DecodeError::TooDeep)?;
                let body = self.term(arena, b, inner)?;
                Ok(arena.lam(body))
            }
            Expr::Sym(s) => self.atom(arena, s),
            Expr::App(..) => {
                let (head, args) = e.spine();
                let (applied, rest) = match head {
                    Expr::Sym(s) => match special_form(s) {
                        Some((token, n)) => {
                            if args.len() < n {
                                return Err(DecodeError::MalformedSpecialForm {
                                    token,
                                    message: format!("needs {n} operands, got {}", args.len()),
                                });
                            }
                            let form = self.special(arena, token, &args[..n], depth)?;
                            (form, &args[n..])
                        }
                        None => (self.atom(arena, s)?, &args[..]),
                    },
                    _ => (self.term(arena, head, depth)?, &args[..]),
                };
                let mut out = applied;
                for a in rest {
                    let a = self.term(arena, a, depth)?;
                    out = arena.app(out, a);
                }
                Ok(out)
            }
        }
    }

    /// Builds the special form `token` from exactly its operands.
    fn special(
        &self,
        arena: &mut Arena,
        token: &'static str,
        ops: &[&Expr],
        depth: u32,
    ) -> Result<TermId, DecodeError> {
        if token == LET {
            let Expr::Lam(body) = ops[1] else {
                return Err(DecodeError::MalformedSpecialForm {
                    token,
                    message: format!("the body must be a (lam ...), got {}", ops[1]),
                });
            };
            let bound = self.term(arena, ops[0], depth)?;
            let inner = depth.checked_add(1).ok_or(DecodeError::TooDeep)?;
            let body = self.term(arena, body, inner)?;
            return Ok(arena.let_(bound, body));
        }
        // `ops` has exactly the operand count `special_form` gives `token`.
        let x = self.term(arena, ops[0], depth)?;
        let y = self.term(arena, ops[1], depth)?;
        Ok(match token {
            IF => {
                let z = self.term(arena, ops[2], depth)?;
                arena.if_(x, y, z)
            }
            AND => arena.and(x, y),
            _ => arena.or(x, y),
        })
    }

    fn atom(&self, arena: &mut Arena, s: &str) -> Result<TermId, DecodeError> {
        if let Some((token, n)) = special_form(s) {
            return Err(DecodeError::MalformedSpecialForm {
                token,
                message: format!("must head an application to {n} operands"),
            });
        }
        match s {
            "true" => return Ok(arena.bool(true)),
            "false" => return Ok(arena.bool(false)),
            "unit" => return Ok(arena.unit()),
            _ => {}
        }
        if let Some(p) = Prim::from_name(s) {
            return Ok(arena.prim(p));
        }
        if let Some(digits) = s.strip_prefix(INT_PREFIX) {
            return match digits.parse::<i64>() {
                Ok(n) if n.to_string() == digits => Ok(arena.int(n)),
                _ => Err(DecodeError::BadInteger(s.to_string())),
            };
        }
        if let Some(l) = self.names.id(s) {
            return Ok(arena.lib(l));
        }
        Err(DecodeError::UnknownToken(s.to_string()))
    }
}

/// Decodes a closed Stitch program, such as one of Stitch's rewritten programs.
pub fn decode(arena: &mut Arena, src: &str, names: &LibNames) -> Result<TermId, DecodeError> {
    decode_open(arena, src, names, 0)
}

/// Decodes a Stitch program that may refer to `context` enclosing binders.
pub fn decode_open(
    arena: &mut Arena,
    src: &str,
    names: &LibNames,
    context: u32,
) -> Result<TermId, DecodeError> {
    let e = expr::parse(src)?;
    from_expr(arena, &e, names, context)
}

/// Decodes the body of a Stitch abstraction of `arity` parameters into the closed
/// core term `λ^arity. body'`, where `#i` under `d` binders of the body becomes
/// `$(d + arity - 1 - i)` (see the crate docs). Free `$i` in the body are an error.
pub fn decode_abstraction(
    arena: &mut Arena,
    body: &str,
    arity: u32,
    names: &LibNames,
) -> Result<TermId, DecodeError> {
    let e = expr::parse(body)?;
    let decoder = Decoder {
        names,
        context: 0,
        arity: Some(arity),
    };
    let mut out = decoder.term(arena, &e, 0)?;
    for _ in 0..arity {
        out = arena.lam(out);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dc_core::sexpr;

    use proptest::prelude::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn names() -> LibNames {
        LibNames::sequential("fn_", 5).unwrap()
    }

    fn core(arena: &mut Arena, src: &str) -> TermId {
        sexpr::parse(arena, src).unwrap_or_else(|e| panic!("{src}: {e}"))
    }

    /// Encodes the core-syntax term `src` and checks the Stitch text, then decodes it
    /// back and checks it is the same node.
    fn check_encoding(src: &str, stitch: &str) {
        let mut a = Arena::new();
        let id = core(&mut a, src);
        let n = names();
        assert_eq!(encode(&a, id, &n).unwrap(), stitch, "encoding {src}");
        let ctx = a.free_depth(id);
        let back = decode_open(&mut a, stitch, &n, ctx).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sexpr::print(&a, back), src, "decoding {stitch}");
        assert_eq!(back, id);
    }

    #[test]
    fn hand_checked_encodings() {
        check_encoding("(lam (add $0 1))", "(lam (add $0 int_1))");
        check_encoding(
            "(foldl (lam (lam (add $1 $0))) 0 $0)",
            "(foldl (lam (lam (add $1 $0))) int_0 $0)",
        );
        check_encoding("(let 5 (mul $0 $0))", "(let_ int_5 (lam (mul $0 $0)))");
        check_encoding(
            "(if (and true false) (or false true) unit)",
            "(if_ (and_ true false) (or_ false true) unit)",
        );
        check_encoding("-3", "int_-3");
        check_encoding("-9223372036854775808", "int_-9223372036854775808");
        check_encoding("9223372036854775807", "int_9223372036854775807");
        check_encoding("(lib_2 nil none leaf)", "(fn_2 nil none leaf)");
        check_encoding("((lam $0) (lam $0))", "((lam $0) (lam $0))");
        check_encoding("(cons 1 (cons 2 nil))", "(cons int_1 (cons int_2 nil))");
    }

    #[test]
    fn let_binder_is_a_real_lambda_to_stitch() {
        // The body of a let sees the bound value at $0, so Stitch must see a binder:
        // the closed term `let 1 in $0` must not print with a free `$0` at top level.
        let mut a = Arena::new();
        let id = core(&mut a, "(let 1 $0)");
        let e = to_expr(&a, id, &names()).unwrap();
        let (head, args) = e.spine();
        assert_eq!(head, &Expr::sym(LET));
        assert_eq!(args, vec![&Expr::sym("int_1"), &Expr::lam(Expr::Var(0))]);
    }

    #[test]
    fn special_forms_applied_to_more_arguments_round_trip() {
        // Stitch prints ((if_ c t e) x) as (if_ c t e x); decoding must split it back.
        check_encoding("((if true $0 $1) 7)", "(if_ true $0 $1 int_7)");
        check_encoding("((let 1 $0) 2 3)", "(let_ int_1 (lam $0) int_2 int_3)");
        check_encoding("((and true false) unit)", "(and_ true false unit)");
        check_encoding("((or true false) unit)", "(or_ true false unit)");
        let mut a = Arena::new();
        let n = names();
        let nested = decode(&mut a, "((if_ true int_1 int_2) int_3)", &n).unwrap();
        let flat = decode(&mut a, "(if_ true int_1 int_2 int_3)", &n).unwrap();
        assert_eq!(nested, flat);
        assert_eq!(sexpr::print(&a, flat), "((if true 1 2) 3)");
    }

    #[test]
    fn unnamed_libraries_cannot_be_encoded() {
        let mut a = Arena::new();
        let id = core(&mut a, "(lib_7 1)");
        assert_eq!(
            encode(&a, id, &names()),
            Err(EncodeError::UnnamedLib(LibId(7)))
        );
    }

    #[test]
    fn library_names_come_from_the_table() {
        let mut t = LibNames::new();
        t.insert(LibId(0), "fn_4").unwrap();
        let mut a = Arena::new();
        let id = core(&mut a, "(lib_0 1)");
        assert_eq!(encode(&a, id, &t).unwrap(), "(fn_4 int_1)");
        assert_eq!(decode(&mut a, "(fn_4 int_1)", &t).unwrap(), id);
        assert_eq!(
            decode(&mut a, "(fn_0 int_1)", &t),
            Err(DecodeError::UnknownToken("fn_0".into()))
        );
    }

    #[test]
    fn decode_errors() {
        let n = names();
        let cases: Vec<(&str, DecodeError)> = vec![
            ("foo", DecodeError::UnknownToken("foo".into())),
            ("(add int_1 bar)", DecodeError::UnknownToken("bar".into())),
            ("fn_5", DecodeError::UnknownToken("fn_5".into())),
            ("int_x", DecodeError::BadInteger("int_x".into())),
            ("int_", DecodeError::BadInteger("int_".into())),
            ("int_03", DecodeError::BadInteger("int_03".into())),
            ("int_+3", DecodeError::BadInteger("int_+3".into())),
            ("int_-0", DecodeError::BadInteger("int_-0".into())),
            (
                "int_9223372036854775808",
                DecodeError::BadInteger("int_9223372036854775808".into()),
            ),
            (
                "$0",
                DecodeError::FreeVariable {
                    index: 0,
                    binders: 0,
                },
            ),
            (
                "(lam (lam $2))",
                DecodeError::FreeVariable {
                    index: 2,
                    binders: 2,
                },
            ),
            (
                "(let_ $0 (lam $0))",
                DecodeError::FreeVariable {
                    index: 0,
                    binders: 0,
                },
            ),
            ("(lam #0)", DecodeError::ParameterInProgram(0)),
        ];
        for (src, expected) in cases {
            let mut a = Arena::new();
            assert_eq!(decode(&mut a, src, &n), Err(expected), "{src}");
        }
        let mut a = Arena::new();
        assert!(matches!(
            decode(&mut a, "(f", &n),
            Err(DecodeError::Parse(_))
        ));
    }

    #[test]
    fn open_decoding_respects_the_context() {
        let n = names();
        let mut a = Arena::new();
        let v1 = a.var(1);
        assert_eq!(decode_open(&mut a, "$1", &n, 2), Ok(v1));
        assert_eq!(
            decode_open(&mut a, "(lam $3)", &n, 2),
            Err(DecodeError::FreeVariable {
                index: 3,
                binders: 3
            })
        );
    }

    #[test]
    fn malformed_special_forms_are_errors() {
        let n = names();
        for (src, token) in [
            ("if_", IF),
            ("(if_ true int_1)", IF),
            ("(add if_)", IF),
            ("(and_ true)", AND),
            ("or_", OR),
            ("(let_ int_1)", LET),
            // The let body must be a syntactic lambda: Stitch cutting a parameter
            // through the binder yields this shape.
            ("(let_ int_1 int_2)", LET),
            ("(let_ int_1 fn_0)", LET),
        ] {
            let mut a = Arena::new();
            match decode(&mut a, src, &n) {
                Err(DecodeError::MalformedSpecialForm { token: t, .. }) => {
                    assert_eq!(t, token, "{src}")
                }
                other => panic!("{src}: {other:?}"),
            }
        }
    }

    #[test]
    fn decodes_a_one_parameter_abstraction() {
        let n = names();
        let mut a = Arena::new();
        let got = decode_abstraction(&mut a, "(lam (add #0 $0))", 1, &n).unwrap();
        let expected = core(&mut a, "(lam (lam (add $1 $0)))");
        assert_eq!(got, expected);
        assert!(a.is_closed(got));
    }

    #[test]
    fn abstraction_parameters_bind_outermost_first() {
        let n = names();
        let mut a = Arena::new();
        // #0 is bound by the outer lambda, #1 by the inner one.
        let got = decode_abstraction(&mut a, "(sub #0 #1)", 2, &n).unwrap();
        assert_eq!(sexpr::print(&a, got), "(lam (lam (sub $1 $0)))");
        // Under a let binder the indices shift by one.
        let got = decode_abstraction(&mut a, "(let_ #1 (lam (add $0 #0)))", 2, &n).unwrap();
        assert_eq!(sexpr::print(&a, got), "(lam (lam (let $0 (add $0 $2))))");
        // A parameter in function position, and a reference to an earlier abstraction.
        let got = decode_abstraction(&mut a, "(fn_1 (lam (#0 $0)))", 1, &n).unwrap();
        assert_eq!(sexpr::print(&a, got), "(lam (lib_1 (lam ($1 $0))))");
        // Arity 0: the body itself.
        let got = decode_abstraction(&mut a, "(mul int_2 int_3)", 0, &n).unwrap();
        assert_eq!(sexpr::print(&a, got), "(mul 2 3)");
    }

    /// Applying the decoded abstraction to arguments and β-reducing gives the body
    /// with each parameter replaced by its argument, which is Stitch's reading of
    /// `(fn_k a0 a1 ...)`.
    #[test]
    fn decoded_abstraction_beta_reduces_to_the_substitution() {
        let n = names();
        let mut a = Arena::new();
        let abs = decode_abstraction(&mut a, "(lam (cons #1 (cons #0 $0)))", 2, &n).unwrap();
        let x = a.int(10);
        let y = a.int(20);
        let reduced = beta_apply(&mut a, abs, &[x, y]);
        assert_eq!(sexpr::print(&a, reduced), "(lam (cons 20 (cons 10 $0)))");
    }

    /// Substitutes closed `args` for the outer lambdas of `f`, outermost first.
    fn beta_apply(a: &mut Arena, f: TermId, args: &[TermId]) -> TermId {
        let mut f = f;
        for &arg in args {
            assert!(a.is_closed(arg));
            let Term::Lam(body) = a.get(f) else {
                panic!("not a lambda")
            };
            f = subst(a, body, 0, arg);
        }
        f
    }

    /// `t[$depth := arg]` for closed `arg`, lowering the variables above `depth`.
    fn subst(a: &mut Arena, t: TermId, depth: u32, arg: TermId) -> TermId {
        let node = match a.get(t) {
            Term::Var(i) if i == depth => return arg,
            Term::Var(i) if i > depth => Term::Var(i - 1),
            Term::Lam(b) => Term::Lam(subst(a, b, depth + 1, arg)),
            Term::Let(x, b) => {
                let x = subst(a, x, depth, arg);
                Term::Let(x, subst(a, b, depth + 1, arg))
            }
            Term::App(f, x) => {
                let f = subst(a, f, depth, arg);
                Term::App(f, subst(a, x, depth, arg))
            }
            Term::If(c, x, y) => {
                let c = subst(a, c, depth, arg);
                let x = subst(a, x, depth, arg);
                Term::If(c, x, subst(a, y, depth, arg))
            }
            Term::And(x, y) => {
                let x = subst(a, x, depth, arg);
                Term::And(x, subst(a, y, depth, arg))
            }
            Term::Or(x, y) => {
                let x = subst(a, x, depth, arg);
                Term::Or(x, subst(a, y, depth, arg))
            }
            other => other,
        };
        a.intern(node)
    }

    #[test]
    fn abstraction_errors() {
        let n = names();
        let mut a = Arena::new();
        assert_eq!(
            decode_abstraction(&mut a, "(add #0 #1)", 1, &n),
            Err(DecodeError::ParameterOutOfRange { index: 1, arity: 1 })
        );
        assert_eq!(
            decode_abstraction(&mut a, "#0", 0, &n),
            Err(DecodeError::ParameterOutOfRange { index: 0, arity: 0 })
        );
        assert_eq!(
            decode_abstraction(&mut a, "(lam (add #0 $1))", 1, &n),
            Err(DecodeError::FreeVariable {
                index: 1,
                binders: 1
            })
        );
        assert_eq!(
            decode_abstraction(&mut a, "(foo #0)", 1, &n),
            Err(DecodeError::UnknownToken("foo".into()))
        );
        assert!(matches!(
            decode_abstraction(&mut a, "(if_ #0 #1)", 2, &n),
            Err(DecodeError::MalformedSpecialForm { token: IF, .. })
        ));
        assert!(matches!(
            decode_abstraction(&mut a, "(let_ #0 #1)", 2, &n),
            Err(DecodeError::MalformedSpecialForm { token: LET, .. })
        ));
        assert_eq!(
            decode_abstraction(&mut a, "(lam (lam #0))", u32::MAX, &n),
            Err(DecodeError::TooDeep)
        );
    }

    fn arb_term(arena: Rc<RefCell<Arena>>) -> impl Strategy<Value = TermId> {
        let leaf = prop_oneof![
            (0u32..4).prop_map(Term::Var),
            any::<i64>().prop_map(Term::Int),
            prop_oneof![Just(i64::MIN), Just(i64::MAX), Just(-1i64), Just(0i64)]
                .prop_map(Term::Int),
            any::<bool>().prop_map(Term::Bool),
            Just(Term::Unit),
            proptest::sample::select(Prim::ALL).prop_map(Term::Prim),
            (0u32..5).prop_map(|l| Term::Lib(LibId(l))),
        ];
        let a0 = arena.clone();
        let leaf = leaf.prop_map(move |t| a0.borrow_mut().intern(t));
        leaf.prop_recursive(6, 64, 3, move |inner| {
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
    fn decode_inverts_encode() {
        let n = names();
        let arena = Rc::new(RefCell::new(Arena::new()));
        let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig {
            cases: 2048,
            ..ProptestConfig::default()
        });
        runner
            .run(&arb_term(arena.clone()), |id| {
                let text = encode(&arena.borrow(), id, &n)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let ctx = arena.borrow().free_depth(id);
                // Decoding into the same arena must find the very same node.
                let back = decode_open(&mut arena.borrow_mut(), &text, &n, ctx)
                    .map_err(|e| TestCaseError::fail(format!("{text}: {e}")))?;
                prop_assert_eq!(back, id, "{}", text);
                // And into a fresh arena, the same term.
                let mut fresh = Arena::new();
                let back = decode_open(&mut fresh, &text, &n, ctx)
                    .map_err(|e| TestCaseError::fail(format!("{text}: {e}")))?;
                prop_assert_eq!(
                    sexpr::print(&fresh, back),
                    sexpr::print(&arena.borrow(), id)
                );
                // The text is a fixed point of Stitch's parse-then-print.
                let e = expr::parse(&text).map_err(|e| TestCaseError::fail(e.to_string()))?;
                prop_assert_eq!(expr::print(&e), text.clone());
                // One fewer context binder than needed is a free-variable error.
                if ctx > 0 {
                    let short = decode_open(&mut fresh, &text, &n, ctx - 1);
                    prop_assert!(
                        matches!(short, Err(DecodeError::FreeVariable { .. })),
                        "{}: {:?}",
                        text,
                        short
                    );
                }
                Ok(())
            })
            .unwrap();
    }

    /// Replaces each free variable of `e` (index `binders + j`, `j < arity`) with
    /// the parameter `#(arity - 1 - j)`: the inverse of the abstraction convention.
    fn abstract_free_vars(e: &Expr, binders: u32, arity: u32) -> Expr {
        match e {
            Expr::Var(i) if *i >= binders => Expr::IVar(arity - 1 - (i - binders)),
            Expr::Lam(b) => Expr::lam(abstract_free_vars(b, binders + 1, arity)),
            Expr::App(f, x) => Expr::app(
                abstract_free_vars(f, binders, arity),
                abstract_free_vars(x, binders, arity),
            ),
            other => other.clone(),
        }
    }

    #[test]
    fn decode_abstraction_closes_over_parameters() {
        let n = names();
        let arena = Rc::new(RefCell::new(Arena::new()));
        let mut runner = proptest::test_runner::TestRunner::default();
        runner
            .run(&arb_term(arena.clone()), |id| {
                let arity = arena.borrow().free_depth(id);
                let e = to_expr(&arena.borrow(), id, &n)
                    .map_err(|e| TestCaseError::fail(e.to_string()))?;
                let body = expr::print(&abstract_free_vars(&e, 0, arity));
                let mut a = arena.borrow_mut();
                let got = decode_abstraction(&mut a, &body, arity, &n)
                    .map_err(|e| TestCaseError::fail(format!("{body}: {e}")))?;
                let mut expected = id;
                for _ in 0..arity {
                    expected = a.lam(expected);
                }
                prop_assert_eq!(got, expected, "{}", body);
                prop_assert!(a.is_closed(got));
                Ok(())
            })
            .unwrap();
    }
}
