//! Error tests: each names the kind, the position, and words the writer can act on.

use dc_core::Arena;
use dc_surface::{compile_expression, ErrorKind, SurfaceError, TopLevel};

fn program_error(src: &str) -> SurfaceError {
    let mut env = TopLevel::new();
    match env.add_source("prog", src) {
        Ok(names) => panic!("expected an error, but defined {names:?}"),
        Err(e) => e,
    }
}

fn expr_error(src: &str) -> SurfaceError {
    let mut arena = Arena::new();
    match compile_expression(&mut arena, src) {
        Ok(_) => panic!("expected an error for {src:?}"),
        Err(e) => e,
    }
}

fn assert_mentions(e: &SurfaceError, words: &[&str]) {
    for w in words {
        assert!(e.message.contains(w), "missing {w:?} in: {e}");
    }
}

#[test]
fn nested_list_pattern_is_rejected_with_a_second_match_suggestion() {
    let e = program_error("let f xs =\n  match xs with\n  | [] -> 0\n  | x :: y :: t -> x\n");
    assert_eq!(e.kind, ErrorKind::NestedPattern);
    assert_eq!((e.line, e.column), (4, 10), "{e}");
    assert_mentions(&e, &["one level deep", "second match"]);
}

#[test]
fn nested_constructor_and_tuple_patterns_are_rejected() {
    let e = program_error("let f o = match o with None -> 0 | Some (Some x) -> x");
    assert_eq!(e.kind, ErrorKind::NestedPattern);
    assert_mentions(&e, &["one level deep"]);

    let e = program_error("let f ((a, b), c) = a");
    assert_eq!(e.kind, ErrorKind::NestedPattern);
    assert_eq!((e.line, e.column), (1, 8), "{e}");
    assert_mentions(&e, &["let (a, b) ="]);

    let e = program_error("let f xs = match xs with [] -> 0 | [x] -> x | _ -> 1");
    assert_eq!(e.kind, ErrorKind::NestedPattern);
}

#[test]
fn self_reference_points_at_bfix_and_memorec() {
    let e = program_error("let len xs =\n  match xs with\n  | [] -> 0\n  | _ :: t -> 1 + len t\n");
    assert_eq!(e.kind, ErrorKind::SelfReference);
    assert_eq!((e.line, e.column), (4, 19), "{e}");
    assert_mentions(&e, &["'len'", "bfix", "memorec"]);
}

#[test]
fn self_reference_is_an_error_even_when_an_earlier_definition_has_the_name() {
    let e = program_error("let f x = x\nlet f x = f x");
    assert_eq!(e.kind, ErrorKind::SelfReference);
    assert_eq!((e.line, e.column), (2, 11), "{e}");
}

#[test]
fn local_function_self_reference_is_an_error() {
    let e = expr_error("let go n = if n = 0 then 0 else go (n - 1) in go 3");
    assert_eq!(e.kind, ErrorKind::SelfReference);
    let e = expr_error("let go = fun n -> go n in go 3");
    assert_eq!(e.kind, ErrorKind::SelfReference);
}

#[test]
fn let_rec_is_rejected_with_the_alternative() {
    let e = program_error("let rec f x = f x");
    assert_eq!(e.kind, ErrorKind::SelfReference);
    assert_eq!((e.line, e.column), (1, 5));
    assert_mentions(&e, &["let rec", "bfix", "memorec"]);
}

#[test]
fn unknown_name_suggests_near_candidates() {
    let e = program_error("let f xs = lenght xs");
    assert_eq!(e.kind, ErrorKind::UnknownName);
    assert_eq!((e.line, e.column), (1, 12));
    assert_mentions(&e, &["unknown name 'lenght'", "did you mean", "'length'"]);

    let e = program_error("let total xs = 0\nlet g = totl");
    assert_mentions(&e, &["'total'"]);

    let e = program_error("let g = zzzzzz");
    assert_eq!(e.kind, ErrorKind::UnknownName);
    assert!(!e.message.contains("did you mean"), "{e}");
}

#[test]
fn unknown_names_from_other_languages_get_a_hint() {
    let e = program_error("let f xs = fold_left (fun a x -> a + x) 0 xs");
    assert_mentions(&e, &["foldl"]);
}

#[test]
fn unclosed_paren_reports_where_it_was_opened() {
    let e = program_error("let f x =\n  (x + 1\n");
    assert_eq!(e.kind, ErrorKind::Syntax);
    assert_eq!((e.line, e.column), (3, 1), "{e}");
    assert_mentions(&e, &["expected ')'", "line 2, column 3", "end of input"]);
    let shown = e.to_string();
    assert!(shown.starts_with("prog:3:1: "), "{shown}");
}

#[test]
fn unclosed_bracket_reports_where_it_was_opened() {
    let e = expr_error("[1; 2");
    assert_mentions(&e, &["expected ']'", "line 1, column 1"]);
}

#[test]
fn missing_else_is_explained() {
    let e = expr_error("if true then 1");
    assert_eq!(e.kind, ErrorKind::Syntax);
    assert_mentions(&e, &["'else'"]);
}

#[test]
fn missing_in_is_explained() {
    let e = program_error("let f x =\n  let y = x + 1\nlet g = 2\n");
    assert_eq!(e.kind, ErrorKind::Syntax);
    assert_eq!((e.line, e.column), (3, 1), "{e}");
    assert_mentions(&e, &["expected 'in'", "found 'let'"]);
}

#[test]
fn chained_comparisons_are_rejected() {
    let e = expr_error("1 < 2 < 3");
    assert_mentions(&e, &["&&"]);
}

#[test]
fn list_with_commas_is_rejected() {
    let e = expr_error("[1, 2, 3]");
    assert_eq!((e.line, e.column), (1, 3));
    assert_mentions(&e, &["';'"]);
}

#[test]
fn other_language_syntax_gets_a_hint() {
    assert_mentions(&expr_error("1 != 2"), &["not (a = b)"]);
    assert_mentions(&expr_error("1 <> 2"), &["not (a = b)"]);
    assert_mentions(&expr_error("\\x -> x"), &["fun x ->"]);
    assert_mentions(&expr_error("List.length []"), &["length"]);
    assert_mentions(&expr_error("(* c *) 1"), &["--"]);
    assert_mentions(&expr_error("fun x => x"), &["->"]);
}

#[test]
fn match_must_cover_both_cases() {
    let e = expr_error("match [] with [] -> 0");
    assert_eq!(e.kind, ErrorKind::Pattern);
    assert_mentions(&e, &["h :: t"]);
    let e = expr_error("match None with Some x -> x");
    assert_mentions(&e, &["None"]);
}

#[test]
fn tree_patterns_are_checked() {
    // Both tree cases must be covered.
    let e = expr_error("match Leaf with Leaf -> 0");
    assert_eq!(e.kind, ErrorKind::Pattern);
    assert_mentions(&e, &["Node (l, x, r)"]);
    let e = expr_error("match Leaf with Node (l, x, r) -> x");
    assert_eq!(e.kind, ErrorKind::Pattern);
    assert_mentions(&e, &["Leaf"]);
    // A Node pattern has exactly three fields.
    let e = expr_error("match Leaf with Leaf -> 0 | Node (l, r) -> 1");
    assert_mentions(&e, &["three fields"]);
    // Leaf has no fields.
    let e = expr_error("match Leaf with Leaf x -> 0 | Node (l, x, r) -> 1");
    assert_eq!(e.kind, ErrorKind::Pattern);
    // Patterns are one level deep, inside Node too.
    let e = expr_error("match Leaf with Leaf -> 0 | Node (Leaf, x, r) -> x");
    assert_eq!(e.kind, ErrorKind::NestedPattern);
    assert_mentions(&e, &["one level deep"]);
    // Tree and list constructors do not mix in one match.
    let e = expr_error("match Leaf with Leaf -> 0 | h :: t -> 1");
    assert_eq!(e.kind, ErrorKind::Pattern);
}

#[test]
fn node_expression_takes_three_fields() {
    let e = expr_error("Node (Leaf, 1)");
    assert_mentions(&e, &["three fields"]);
}

#[test]
fn unparenthesised_inner_match_is_explained() {
    // The inner match takes the '| [] -> 0' arm, so the outer one lacks '[]'.
    let e = expr_error("match [] with x :: t -> match t with [] -> 1 | _ -> 2 | [] -> 0");
    assert_eq!(e.kind, ErrorKind::Pattern);
    assert_mentions(&e, &["parentheses"]);
    let mut arena = Arena::new();
    compile_expression(
        &mut arena,
        "match [] with x :: t -> (match t with [] -> 1 | _ -> 2) | [] -> 0",
    )
    .unwrap_or_else(|e| panic!("parenthesised version should compile: {e}"));
}

#[test]
fn match_cannot_mix_types_or_match_numbers() {
    let e = expr_error("match [] with [] -> 0 | Some x -> x");
    assert_eq!(e.kind, ErrorKind::Pattern);
    let e = expr_error("match 3 with 0 -> 1 | _ -> 2");
    assert_eq!(e.kind, ErrorKind::Pattern);
    assert_mentions(&e, &["if"]);
}

#[test]
fn greater_than_has_no_section() {
    let e = expr_error("( > )");
    assert_mentions(&e, &["( < )"]);
}

#[test]
fn integer_too_large() {
    let e = expr_error("9223372036854775808");
    assert_mentions(&e, &["too large"]);
}

#[test]
fn top_level_garbage_after_definition() {
    let e = program_error("let f x = x )");
    assert_eq!((e.line, e.column), (1, 13));
    assert_mentions(&e, &["')'"]);
}

#[test]
fn entry_must_exist() {
    let env = TopLevel::new();
    let mut arena = Arena::new();
    let e = env.close_definition(&mut arena, "main").unwrap_err();
    assert_eq!(e.kind, ErrorKind::UnknownName);
    assert_mentions(&e, &["'main'"]);
}
