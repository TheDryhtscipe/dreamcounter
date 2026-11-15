//! Tests that what the language guide ([`GUIDE`]) and the primitives section
//! ([`primitives`]) say about dc holds for the parser and type checker the
//! server uses.
//!
//! The claims are read out of the guide's text, not copied into the tests, so
//! the guide can be rewritten without touching this file as long as it keeps
//! the same shape: an indented `Expressions:` block of code forms in columns
//! with one row of operators, an indented `Recursion:` block, code in
//! backticks, a sentence that lists the operators that cannot be passed as
//! functions after the word "except", and so on. A rewrite that changes the
//! shape makes an extractor below fail with a message that names it; then
//! change the extractor, not the claim. The few code spans in backticks that
//! are not expressions are in [`OTHER_SPANS`], each with its own check. The
//! types the placeholder names stand for (`n` an Int, `f` a function) are in
//! [`PLACEHOLDERS`], and the number of expression forms is
//! [`EXPRESSION_FORMS`].
//!
//! Programs are compiled as the verifier compiles them: with the prelude on,
//! through [`TopLevel::add_source`] and [`TopLevel::close_definition`], and
//! typed with [`infer`] against the prelude.

use super::*;
use dc_core::types::infer;
use dc_core::{Scheme, TermId, Type};
use dc_surface::prelude::{Prelude, PreludeMode};

/// The name of the definition each snippet is compiled as.
const PROBE: &str = "guide_probe";

/// The words the lexer reserves: they are never holes in a snippet.
const KEYWORDS: &[&str] = &[
    "let", "rec", "in", "fun", "if", "then", "else", "match", "with", "true", "false", "mod",
];

/// Whether a type is of the kind a placeholder stands for.
type IsType = fn(&Type) -> bool;

/// What the guide's placeholder names stand for, where the name says so:
/// `n` is a count, `f` a function, `xs` a list, and so on. A placeholder that
/// is free in a snippet (not bound by it) must have this type there, so a
/// form written with its arguments in the wrong order, such as `loop n f z`,
/// fails. Other placeholders, such as `z` or `e1`, may have any type.
const PLACEHOLDERS: &[(&str, &str, IsType)] = &[
    ("n", "Int", |t| *t == Type::Int),
    ("f", "a function", |t| matches!(t, Type::Arrow(..))),
    ("xs", "a list", |t| matches!(t, Type::List(_))),
    ("o", "an option", |t| matches!(t, Type::Option(_))),
    ("t", "a tree", |t| matches!(t, Type::Tree(_))),
];

/// How many code forms the `Expressions:` block shows. If the guide changes,
/// change this to match; it is here so that two forms run together into one
/// column are noticed.
const EXPRESSION_FORMS: usize = 13;

/// How many operators the `Expressions:` block lists. If the guide changes,
/// change this to match; it is here so that a dropped operator is noticed.
const GUIDE_OPERATORS: usize = 14;

/// A check of one code span that is not an expression.
type SpanCheck = fn(&Probe);

/// The code spans in the guide that are not expressions, and how each is
/// checked. Every other span in backticks must compile and type as an
/// expression, or be an operator in brackets (`( + )`), which
/// [`the_operators_the_guide_passes_as_functions_can_be_and_the_excluded_cannot`]
/// checks.
const OTHER_SPANS: &[(&str, SpanCheck)] = &[
    // "Comments run from `--` to the end of the line."
    ("--", |p| {
        p.infer_closed("1 -- not code: ( let")
            .unwrap_or_else(|e| panic!("a comment does not end at the line's end: {e}"));
    }),
    // "A program is a sequence of top-level definitions `let name params = expr`."
    ("let name params = expr", |_| {
        let defs = dc_surface::parse_program("guide", "let name params = expr")
            .unwrap_or_else(|e| panic!("the top-level form does not parse: {e}"));
        assert_eq!(defs.len(), 1);
    }),
    // "A definition can use the definitions before it. There is no recursive
    // `let`".
    ("let", |p| {
        p.compile_program("let g x = x\nlet h y = g y", "h")
            .unwrap_or_else(|e| panic!("a definition cannot use an earlier one: {e}"));
        let recursive = p.compile_program("let f x = f x", "f");
        assert!(recursive.is_err(), "a recursive let compiles");
    }),
    // "A `Key` type is Int, Bool, Unit, or a tuple of those."
    ("Key", check_key_types),
];

/// Compiles and types snippets of the guide.
struct Probe {
    prelude: Prelude,
}

impl Probe {
    fn new() -> Probe {
        Probe {
            prelude: Prelude::load().unwrap(),
        }
    }

    /// Whether `name` is a primitive or a prelude function.
    fn in_scope(&self, name: &str) -> bool {
        Prim::from_name(name).is_some() || self.prelude.get(name).is_some()
    }

    /// The holes of `snippet`, in order: the lower-case names in it that are
    /// not keywords and not in scope, such as `e1` in `let x = e1 in e2`.
    fn holes(&self, snippet: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut word = String::new();
        for c in snippet.chars().chain(std::iter::once(' ')) {
            if c.is_ascii_alphanumeric() || c == '_' || c == '\'' {
                word.push(c);
                continue;
            }
            let starts_lower = word.starts_with(|c: char| c.is_ascii_lowercase());
            if starts_lower
                && !KEYWORDS.contains(&word.as_str())
                && !self.in_scope(&word)
                && !out.contains(&word)
            {
                out.push(word.clone());
            }
            word.clear();
        }
        out
    }

    /// Compiles `text` as the verifier does and closes it over `entry`.
    fn compile_program(&self, text: &str, entry: &str) -> Result<(Arena, TermId), String> {
        let mut top = self.prelude.top_level(PreludeMode::On);
        top.add_source("guide", text).map_err(|e| e.to_string())?;
        let mut arena = Arena::new();
        let term = top
            .close_definition(&mut arena, entry)
            .map_err(|e| e.to_string())?;
        Ok((arena, term))
    }

    fn infer_program(&self, text: &str) -> Result<Scheme, String> {
        let (arena, term) = self.compile_program(text, PROBE)?;
        infer(&arena, term, &self.prelude).map_err(|e| e.to_string())
    }

    /// The type of the expression `text`, which has no holes.
    fn infer_closed(&self, text: &str) -> Result<Scheme, String> {
        self.infer_program(&format!("let {PROBE} = {text}\n"))
    }

    /// The type of `snippet` with each of its free holes a parameter. A free
    /// hole whose name is in [`PLACEHOLDERS`] must have the type given there.
    fn infer_open(&self, snippet: &str) -> Result<Scheme, String> {
        let holes = self.holes(snippet);
        let with = |params: &[&String]| {
            let params: Vec<&str> = params.iter().map(|s| s.as_str()).collect();
            format!("let {PROBE} {} = {snippet}\n", params.join(" "))
        };
        // A hole is free if the snippet does not compile without it as a
        // parameter; otherwise the snippet binds it, as `x` in `fun x -> e`.
        let free: Vec<&String> = holes
            .iter()
            .filter(|h| {
                let others: Vec<&String> = holes.iter().filter(|o| o != h).collect();
                self.compile_program(&with(&others), PROBE).is_err()
            })
            .collect();
        let scheme = self.infer_program(&with(&free))?;
        let mut ty = scheme.ty();
        for hole in free {
            let Type::Arrow(param, rest) = ty else {
                return Err(format!("`{snippet}` has fewer parameters than holes"));
            };
            if let Some((_, what, ok)) = PLACEHOLDERS.iter().find(|(n, _, _)| n == hole) {
                if !ok(param) {
                    return Err(format!(
                        "in `{snippet}`, `{hole}` should be {what} but has type {param}"
                    ));
                }
            }
            ty = rest;
        }
        Ok(scheme)
    }
}

/// The code spans of `text`, the parts in backticks.
fn spans(text: &str) -> Vec<&str> {
    let parts: Vec<&str> = text.split('`').collect();
    assert!(parts.len() % 2 == 1, "a backtick is not closed: {text}");
    parts.into_iter().skip(1).step_by(2).collect()
}

/// The sentence of the guide that starts with the first `start` outside a
/// code span, up to its full stop.
fn sentence_from(start: &str) -> &'static str {
    let from = GUIDE
        .find(start)
        .unwrap_or_else(|| panic!("the guide has no {start:?}: change the extractor"));
    let rest = &GUIDE[from..];
    let mut in_code = false;
    for (i, c) in rest.char_indices() {
        match c {
            '`' => in_code = !in_code,
            '.' if !in_code => return &rest[..i],
            _ => {}
        }
    }
    panic!("the sentence from {start:?} has no full stop")
}

/// The lines of the indented block under the line `heading`.
fn block(heading: &str) -> Vec<&'static str> {
    let mut lines = GUIDE.lines().skip_while(|l| l.trim() != heading);
    assert!(
        lines.next().is_some(),
        "the guide has no {heading:?} block: change the extractor"
    );
    let block: Vec<&str> = lines.take_while(|l| l.starts_with("  ")).collect();
    assert!(!block.is_empty(), "the {heading:?} block is empty");
    block
}

/// The columns of a line of a block: the text between runs of two or more
/// spaces.
fn columns(line: &str) -> Vec<&str> {
    line.split("  ")
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect()
}

/// Whether a column of the `Expressions:` block lists operators rather than
/// showing a code form: it starts with an operator symbol.
fn is_operator_column(column: &str) -> bool {
    !column.starts_with(|c: char| c.is_ascii_alphanumeric() || c == '(' || c == '[')
}

/// The code forms the `Expressions:` block shows.
fn expression_forms() -> Vec<&'static str> {
    let forms: Vec<&str> = block("Expressions:")
        .into_iter()
        .flat_map(columns)
        .filter(|c| !is_operator_column(c))
        .collect();
    assert_eq!(
        forms.len(),
        EXPRESSION_FORMS,
        "the Expressions block shows {} forms, not {EXPRESSION_FORMS}: if the guide \
         changed, change EXPRESSION_FORMS; else two columns ran together: {forms:?}",
        forms.len()
    );
    forms
}

/// The operators the `Expressions:` block lists, each with the remark in
/// brackets after its group, if any: `=` has `on Int`.
fn guide_operators() -> Vec<(&'static str, Option<&'static str>)> {
    let mut out = Vec::new();
    for column in block("Expressions:")
        .into_iter()
        .flat_map(columns)
        .filter(|c| is_operator_column(c))
    {
        let (ops, remark) = match column.split_once('(') {
            Some((ops, remark)) => {
                let remark = remark
                    .strip_suffix(')')
                    .unwrap_or_else(|| panic!("an unclosed remark in {column:?}"));
                (ops, Some(remark))
            }
            None => (column, None),
        };
        for op in ops.split_whitespace() {
            out.push((op, remark));
        }
    }
    assert_eq!(
        out.len(),
        GUIDE_OPERATORS,
        "the Expressions block lists {} operators, not {GUIDE_OPERATORS}: if the guide \
         changed, change GUIDE_OPERATORS: {out:?}",
        out.len()
    );
    out
}

/// The operator inside a span such as `( + )`, if the span is one.
fn section(span: &str) -> Option<&str> {
    span.strip_prefix("( ")?.strip_suffix(" )")
}

/// The operators the guide says cannot be passed as functions: the
/// bracketed operators after "except".
fn excluded_operators() -> Vec<&'static str> {
    let excluded: Vec<&str> = spans(sentence_from("except"))
        .into_iter()
        .filter_map(section)
        .collect();
    assert!(!excluded.is_empty(), "no operator follows \"except\"");
    excluded
}

/// The entries of the `Recursion:` block: each combinator applied to its
/// arguments as the guide writes it, such as `loop n z f`.
fn recursion_forms() -> Vec<&'static str> {
    let forms: Vec<&str> = block("Recursion:")
        .into_iter()
        // An entry starts after two spaces; the lines that carry on a
        // description start further in.
        .filter(|l| !l.starts_with("   "))
        .filter_map(|l| columns(l).into_iter().next())
        .flat_map(|c| c.split(", "))
        .collect();
    assert!(forms.len() >= 5, "too few recursion forms: {forms:?}");
    forms
}

#[test]
fn every_expression_form_in_the_guide_compiles_and_types() {
    let p = Probe::new();
    for form in expression_forms() {
        p.infer_open(form)
            .unwrap_or_else(|e| panic!("the guide's form `{form}` fails: {e}"));
    }
}

/// Whether `op` is written between its arguments: a symbol, or a reserved
/// word such as `mod`. A name such as `not` is an ordinary function.
fn is_infix(op: &str) -> bool {
    !op.starts_with(|c: char| c.is_ascii_alphabetic()) || KEYWORDS.contains(&op)
}

/// `op` used as the guide says: between two arguments, or before one if it
/// is a function name such as `not`.
fn applied(p: &Probe, op: &str) -> (String, Scheme) {
    let text = if is_infix(op) {
        format!("fun a b -> a {op} b")
    } else {
        format!("fun a -> {op} a")
    };
    match p.infer_closed(&text) {
        Ok(scheme) => (text, scheme),
        Err(e) => panic!("the guide's operator `{op}` fails in `{text}`: {e}"),
    }
}

#[test]
fn every_operator_in_the_guide_types_as_stated() {
    let p = Probe::new();
    for (op, remark) in guide_operators() {
        let (text, scheme) = applied(&p, op);
        match remark {
            None => {}
            Some("on Int") => assert_eq!(
                scheme.to_string(),
                "Int -> Int -> Bool",
                "the guide says `{op}` is on Int"
            ),
            // Structural: it compares values of any type, lists included.
            Some("structural") => {
                assert!(!scheme.vars().is_empty(), "`{text}` : {scheme}");
                p.infer_closed(&format!("[1] {op} [2]"))
                    .unwrap_or_else(|e| panic!("`{op}` does not compare lists: {e}"));
            }
            Some(other) => panic!("the guide's remark ({other}) on `{op}` has no check"),
        }
    }
}

#[test]
fn the_operators_the_guide_passes_as_functions_can_be_and_the_excluded_cannot() {
    let p = Probe::new();
    let excluded = excluded_operators();
    // Every excluded operator is one the guide lists.
    let ops: Vec<&str> = guide_operators().into_iter().map(|(op, _)| op).collect();
    for op in &excluded {
        assert!(ops.contains(op), "`( {op} )` is not a listed operator");
    }
    // The bracketed operators in the guide's text, such as `( + )`, and every
    // listed operator.
    let bracketed: Vec<&str> = spans(GUIDE).into_iter().filter_map(section).collect();
    for op in bracketed.into_iter().chain(ops) {
        let as_function = format!("( {op} )");
        let result = p.infer_closed(&as_function);
        if excluded.contains(&op) {
            assert!(
                matches!(&result, Err(e) if e.contains("is not available as a function")),
                "the guide says `{as_function}` cannot be passed as a function, \
                 but it gives: {result:?}"
            );
            continue;
        }
        let scheme = result.unwrap_or_else(|e| panic!("`{as_function}` is not a function: {e}"));
        // As a function it is the operator itself.
        let (text, applied) = applied(&p, op);
        assert_eq!(
            scheme.to_string(),
            applied.to_string(),
            "`{as_function}` and `{text}` differ"
        );
    }
}

#[test]
fn the_recursion_combinators_are_primitives_with_the_arguments_shown() {
    let p = Probe::new();
    let forms = recursion_forms();
    let mut names: Vec<&str> = Vec::new();
    for form in &forms {
        let mut words = form.split_whitespace();
        let name = words.next().unwrap();
        let prim = Prim::from_name(name).unwrap_or_else(|| panic!("`{name}` is not a primitive"));
        assert_eq!(words.count(), prim.arity(), "`{form}` has the wrong arity");
        p.infer_open(form)
            .unwrap_or_else(|e| panic!("the guide's form `{form}` fails: {e}"));
        names.push(name);
    }
    // The combinators the guide says recursion goes through are the ones the
    // block describes.
    let mut said: Vec<&str> = spans(sentence_from("recursion goes only through"));
    said.sort_unstable();
    names.sort_unstable();
    assert_eq!(
        said, names,
        "the combinators the guide names differ from the Recursion block's"
    );
}

/// "A `Key` type is Int, Bool, Unit, or a tuple of those."
fn check_key_types(p: &Probe) {
    let sentence = sentence_from("A `Key` type is ");
    let listed = sentence
        .strip_prefix("A `Key` type is ")
        .and_then(|s| s.strip_suffix(", or a tuple of those"))
        .unwrap_or_else(|| panic!("the Key sentence changed shape: change the extractor"));
    let mut values: Vec<&str> = Vec::new();
    for ty in listed.split(", ") {
        let value = match ty {
            "Int" => "1",
            "Bool" => "true",
            "Unit" => "()",
            other => panic!("the guide's key type {other} has no check"),
        };
        values.push(value);
    }
    let key = |value: &str| p.infer_open(&format!("memorec n f {value}"));
    for value in &values {
        key(value).unwrap_or_else(|e| panic!("{value} is not a key: {e}"));
    }
    let tuple = format!("({}, ({}))", values[0], values.join(", "));
    key(&tuple).unwrap_or_else(|e| panic!("{tuple} is not a key: {e}"));
    // Something else is not a key.
    assert!(key("[1]").is_err(), "a list is a key");
}

#[test]
fn the_types_the_guide_names_are_the_type_checkers() {
    let types = sentence_from("Types: ");
    let all: String = Prim::ALL
        .iter()
        .map(|&p| prim_scheme(p).to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let shown: Vec<&str> = all.split(|c: char| !c.is_ascii_alphanumeric()).collect();
    let named: Vec<&str> = types
        .split(|c: char| !c.is_ascii_alphanumeric())
        .skip(1) // "Types"
        .filter(|w| w.starts_with(|c: char| c.is_ascii_uppercase()))
        .collect();
    assert!(
        named.len() >= 7,
        "too few type names in the guide: {named:?}"
    );
    for name in named {
        assert!(shown.contains(&name), "no primitive's type has {name}");
    }
}

#[test]
fn every_code_span_in_the_guide_is_checked() {
    let p = Probe::new();
    let guide_spans = spans(GUIDE);
    for (span, check) in OTHER_SPANS {
        assert!(
            guide_spans.contains(span),
            "`{span}` is no longer in the guide: remove its check"
        );
        check(&p);
    }
    for span in guide_spans {
        if section(span).is_some() || OTHER_SPANS.iter().any(|(s, _)| *s == span) {
            continue;
        }
        p.infer_open(span)
            .unwrap_or_else(|e| panic!("the guide's code `{span}` fails: {e}"));
    }
}

#[test]
fn the_operator_names_in_primitives_are_primitives() {
    let text = primitives();
    const START: &str = "The operators are ";
    let from = text.find(START).expect("the operator sentence") + START.len();
    let list = &text[from..from + text[from..].find(".\n").expect("its full stop")];
    let p = Probe::new();
    let mut named: Vec<&str> = Vec::new();
    for item in list.split(", ").flat_map(|i| i.split(" and ")) {
        let (name, op) = match item.split_once(" (") {
            Some((name, op)) => (name, op.strip_suffix(')').expect("a closing bracket")),
            None => (item, item),
        };
        assert!(Prim::from_name(name).is_some(), "{name} is not a primitive");
        // The operator is that primitive.
        let (arena, term) = p
            .compile_program(&format!("let {PROBE} = ( {op} )"), PROBE)
            .unwrap_or_else(|e| panic!("`( {op} )` fails: {e}"));
        assert_eq!(sexpr::print(&arena, term).unwrap(), name, "( {op} )");
        named.push(op);
    }
    // They are the guide's infix operators that can be passed as functions,
    // and the ones its code forms use, such as `::` in `x :: xs`.
    let excluded = excluded_operators();
    let passable: Vec<&str> = guide_operators()
        .into_iter()
        .map(|(op, _)| op)
        .filter(|op| is_infix(op) && !excluded.contains(op))
        .collect();
    for op in &passable {
        assert!(named.contains(op), "primitives() does not name `{op}`");
    }
    let used: Vec<&str> = expression_forms()
        .into_iter()
        .flat_map(str::split_whitespace)
        .collect();
    for op in &named {
        assert!(
            passable.contains(op) || used.contains(op),
            "primitives() names `{op}`, which the guide does not show"
        );
    }
}

#[test]
fn every_primitive_can_be_used_by_its_name() {
    // primitives(): "Each can be used by its name."
    assert!(primitives().contains("Each can be used by its name."));
    let p = Probe::new();
    let failing: Vec<String> = Prim::ALL
        .iter()
        // Left out: `mod` is a reserved word, so the primitive `mod` cannot
        // be used by its name, only as `( mod )`. The claim is false for it
        // (dreamcounter-68e7e8); the assert below pins the exception.
        .filter(|prim| prim.name() != "mod")
        .filter_map(|prim| {
            let e = p.infer_closed(prim.name()).err()?;
            Some(format!("{}: {e}", prim.name()))
        })
        .collect();
    assert!(failing.is_empty(), "cannot be used by name: {failing:?}");
    assert!(
        p.infer_closed("mod").is_err(),
        "bare `mod` now compiles: dreamcounter-68e7e8 is fixed, so drop the \
         `mod` exception in this test"
    );
}

#[test]
fn no_placeholder_name_is_in_scope() {
    // A placeholder that names a primitive or prelude function is not a hole,
    // so its type would go unchecked.
    let p = Probe::new();
    for (name, _, _) in PLACEHOLDERS {
        assert!(
            !p.in_scope(name),
            "the placeholder `{name}` is a primitive or prelude name, so it is \
             never a hole: rename it in the guide and PLACEHOLDERS"
        );
    }
}
