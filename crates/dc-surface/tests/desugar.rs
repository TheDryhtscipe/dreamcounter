//! Desugaring tests: surface programs against hand-written core s-expressions.

use dc_core::{sexpr, Arena, Term, TermId};
use dc_surface::{compile_expression, TopLevel};

/// A stand-in for the prelude (bead P2.11), enough for the spec §10 examples to
/// resolve `max`, `min_int` and `split_at`.
const TEST_PRELUDE: &str = r#"
-- test prelude
let max a b = if a < b then b else a
let min_int = -9223372036854775808
let reverse xs = foldl (fun acc x -> x :: acc) [] xs
let split_at n xs =
  let r = foldl (fun (i, front, back) x ->
                   if i < n then (i + 1, x :: front, back)
                   else (i + 1, front, x :: back))
                (0, [], []) xs in
  let (_, front, back) = r in
  (reverse front, reverse back)
"#;

/// The examples of spec §10, verbatim.
const SPEC_EXAMPLES: &str = r#"
let sum xs = foldl (fun acc x -> acc + x) 0 xs

let max_subarray xs =
  fst (foldl (fun (best, cur) x ->
         let cur2 = max x (cur + x) in (max best cur2, cur2))
       (min_int, 0) xs)

let merge ab =
  bfix (length (fst ab) + length (snd ab) + 1)
    (fun self (a, b) ->
      match a with
      | [] -> b
      | x :: xs ->
        match b with
        | [] -> a
        | y :: ys -> if x <= y then x :: self (xs, b) else y :: self (a, ys))
    ab

let msort xs =
  bfix (length xs + 1)
    (fun self xs ->
      if length xs <= 1 then xs
      else
        let (l, r) = split_at (length xs / 2) xs in
        merge (self l, self r))
    xs

let fib n =
  memorec (n + 1)
    (fun self k -> if k < 2 then k else self (k - 1) + self (k - 2))
    n
"#;

fn spec_env() -> TopLevel {
    let mut env = TopLevel::new();
    env.add_source("prelude", TEST_PRELUDE)
        .unwrap_or_else(|e| panic!("{e}"));
    env.add_source("spec", SPEC_EXAMPLES)
        .unwrap_or_else(|e| panic!("{e}"));
    env
}

fn close(env: &TopLevel, name: &str) -> (Arena, TermId) {
    let mut arena = Arena::new();
    let id = env
        .close_definition(&mut arena, name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    (arena, id)
}

fn program(src: &str, entry: &str) -> String {
    let mut env = TopLevel::new();
    env.add_source("test", src)
        .unwrap_or_else(|e| panic!("{e}"));
    let (arena, id) = close(&env, entry);
    assert!(arena.is_closed(id), "{entry} is not closed");
    sexpr::print(&arena, id)
}

fn expr(src: &str) -> String {
    let mut arena = Arena::new();
    let id = compile_expression(&mut arena, src).unwrap_or_else(|e| panic!("{src}: {e}"));
    sexpr::print(&arena, id)
}

/// Counts the `let`s wrapped around the entry: one per top-level definition kept.
fn outer_lets(arena: &Arena, mut id: TermId) -> usize {
    let mut n = 0;
    while let Term::Let(_, body) = arena.get(id) {
        n += 1;
        id = body;
    }
    n
}

#[test]
fn spec_sum_desugars_to_expected_core() {
    let env = spec_env();
    let (arena, id) = close(&env, "sum");
    assert_eq!(
        sexpr::print(&arena, id),
        "(lam (foldl (lam (lam (add $1 $0))) 0 $0))"
    );
}

#[test]
fn spec_merge_desugars_to_expected_core() {
    let env = spec_env();
    let (arena, id) = close(&env, "merge");
    assert_eq!(
        sexpr::print(&arena, id),
        "(lam (bfix (add (add (length (fst $0)) (length (snd $0))) 1) \
         (lam (lam (let (fst $0) (let (snd $1) \
         (list_case $1 (lam $1) (lam (lam \
         (list_case $2 (lam $4) (lam (lam \
         (if (le $3 $1) (cons $3 ($7 (pair $2 $4))) (cons $1 ($7 (pair $5 $0)))))))))))))) \
         $0))"
    );
}

#[test]
fn spec_fib_desugars_to_expected_core() {
    let env = spec_env();
    let (arena, id) = close(&env, "fib");
    assert_eq!(
        sexpr::print(&arena, id),
        "(lam (memorec (add $0 1) \
         (lam (lam (if (lt $0 2) $0 (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) $0))"
    );
}

#[test]
fn spec_max_subarray_keeps_only_its_dependencies() {
    let env = spec_env();
    let (arena, id) = close(&env, "max_subarray");
    assert!(arena.is_closed(id));
    // max and min_int, in that order, wrapped around the definition itself.
    assert_eq!(outer_lets(&arena, id), 2);
    let text = sexpr::print(&arena, id);
    assert!(
        text.starts_with("(let (lam (lam (if (lt $1 $0) $0 $1))) (let -9223372036854775808 (lam "),
        "{text}"
    );
}

#[test]
fn spec_msort_pulls_in_merge_and_split_at_transitively() {
    let env = spec_env();
    let (arena, id) = close(&env, "msort");
    assert!(arena.is_closed(id));
    // reverse (needed by split_at), split_at, merge; not max, min_int, sum or fib.
    assert_eq!(outer_lets(&arena, id), 3);
}

#[test]
fn match_on_list_becomes_list_case_with_a_thunk() {
    assert_eq!(
        program(
            "let head_or d xs = match xs with [] -> d | h :: t -> h",
            "head_or"
        ),
        "(lam (lam (list_case $0 (lam $2) (lam (lam $1)))))"
    );
}

#[test]
fn match_arms_may_come_in_either_order_and_use_a_catch_all() {
    assert_eq!(
        program("let f xs = match xs with | h :: _ -> h | [] -> 0", "f"),
        "(lam (list_case $0 (lam 0) (lam (lam $1))))"
    );
    assert_eq!(
        program("let f xs = match xs with [] -> 0 | _ -> 1", "f"),
        "(lam (list_case $0 (lam 0) (lam (lam 1))))"
    );
}

#[test]
fn match_on_option_and_tree() {
    assert_eq!(
        program("let get d o = match o with None -> d | Some x -> x", "get"),
        "(lam (lam (option_case $0 (lam $2) (lam $0))))"
    );
    assert_eq!(
        program(
            "let root d t = match t with Leaf -> d | Node (l, x, r) -> x",
            "root"
        ),
        "(lam (lam (tree_case $0 (lam $2) (lam (lam (lam $1))))))"
    );
}

#[test]
fn constructor_field_may_be_a_flat_tuple_binder() {
    // unfold-style: Some (x, s). The pair is destructured inside the lambda.
    assert_eq!(
        program(
            "let f o = match o with None -> 0 | Some (x, s) -> x + s",
            "f"
        ),
        "(lam (option_case $0 (lam 0) (lam (let (fst $0) (let (snd $1) (add $1 $0))))))"
    );
}

#[test]
fn tuple_binder_uses_fst_and_snd_on_a_fresh_variable() {
    assert_eq!(
        program("let swap = fun (a, b) -> (b, a)", "swap"),
        "(lam (let (fst $0) (let (snd $1) (pair $0 $1))))"
    );
}

#[test]
fn triple_binder_destructures_the_right_nested_pair() {
    assert_eq!(
        program("let f (a, b, c) = (a, b, c)", "f"),
        "(lam (let (fst $0) (let (snd $1) (let (fst $0) (let (snd $1) (pair $3 (pair $1 $0)))))))"
    );
}

#[test]
fn wildcard_components_are_not_projected() {
    assert_eq!(
        program("let f (_, b, _) = b", "f"),
        "(lam (let (snd $0) (let (fst $0) $0)))"
    );
    assert_eq!(program("let f (a, _) = a", "f"), "(lam (let (fst $0) $0))");
}

#[test]
fn multi_argument_fun_is_nested_lambdas_with_destructuring_after_each() {
    assert_eq!(
        program("let f = fun (a, b) c -> a + c", "f"),
        "(lam (let (fst $0) (let (snd $1) (lam (add $2 $0)))))"
    );
}

#[test]
fn let_forms() {
    assert_eq!(expr("let x = 1 in x + x"), "(let 1 (add $0 $0))");
    assert_eq!(
        expr("let f a b = a - b in f 3 4"),
        "(let (lam (lam (sub $1 $0))) ($0 3 4))"
    );
    assert_eq!(
        expr("let (a, b) = (1, 2) in a * b"),
        "(let (pair 1 2) (let (fst $0) (let (snd $1) (mul $1 $0))))"
    );
    assert_eq!(expr("let _ = 1 in 2"), "(let 1 2)");
    // A value let may refer to an outer binding of the same name (shadowing).
    assert_eq!(
        expr("let x = 1 in let x = x + 1 in x"),
        "(let 1 (let (add $0 1) $0))"
    );
}

#[test]
fn greater_than_keeps_left_to_right_order() {
    // Both operands need evaluating: the left one is let-bound first.
    assert_eq!(
        program("let g f x = f x > x + 1", "g"),
        "(lam (lam (let ($1 $0) (lt (add $1 1) $0))))"
    );
    assert_eq!(
        program("let g f x = f x >= x + 1", "g"),
        "(lam (lam (let ($1 $0) (le (add $1 1) $0))))"
    );
    // If either side is a variable or literal, swapping cannot change order or cost.
    assert_eq!(
        program("let gt a b = a > b", "gt"),
        "(lam (lam (lt $0 $1)))"
    );
    assert_eq!(
        program("let g f x = f x > 0", "g"),
        "(lam (lam (lt 0 ($1 $0))))"
    );
}

#[test]
fn operator_precedence_and_associativity() {
    assert_eq!(expr("1 + 2 * 3 - 4"), "(sub (add 1 (mul 2 3)) 4)");
    assert_eq!(expr("10 / 2 / 5"), "(div (div 10 2) 5)");
    assert_eq!(expr("7 mod 3 * 2"), "(mul (mod 7 3) 2)");
    assert_eq!(expr("1 :: 2 :: []"), "(cons 1 (cons 2 nil))");
    assert_eq!(expr("1 + 1 :: []"), "(cons (add 1 1) nil)");
    assert_eq!(
        expr("1 = 2 || 1 < 2 && true"),
        "(or (int_eq 1 2) (and (lt 1 2) true))"
    );
    assert_eq!(expr("true && false && true"), "(and true (and false true))");
    assert_eq!(expr("[1] == [1]"), "(eq (cons 1 nil) (cons 1 nil))");
    assert_eq!(expr("1 <= 2"), "(le 1 2)");
}

#[test]
fn unary_minus() {
    assert_eq!(expr("-5"), "-5");
    assert_eq!(expr("- 5"), "-5");
    assert_eq!(expr("-9223372036854775808"), "-9223372036854775808");
    assert_eq!(expr("let x = 3 in -x"), "(let 3 (sub 0 $0))");
    assert_eq!(expr("let x = 3 in -x * 2"), "(let 3 (mul (sub 0 $0) 2))");
    assert_eq!(
        expr("let f = fun x -> x in -f 2"),
        "(let (lam $0) (sub 0 ($0 2)))"
    );
    // As in OCaml, `f -1` is a subtraction; write `f (-1)` to pass -1.
    assert_eq!(expr("let f = 3 in f -1"), "(let 3 (sub $0 1))");
    assert_eq!(
        expr("let f = fun x -> x in f (-1)"),
        "(let (lam $0) ($0 -1))"
    );
}

#[test]
fn literals_tuples_lists_and_constructors() {
    assert_eq!(expr("()"), "unit");
    assert_eq!(expr("(1, true)"), "(pair 1 true)");
    assert_eq!(expr("(1, 2, 3)"), "(pair 1 (pair 2 3))");
    assert_eq!(expr("(1, (2, 3))"), "(pair 1 (pair 2 3))");
    assert_eq!(expr("[]"), "nil");
    assert_eq!(expr("[1; 2; 3]"), "(cons 1 (cons 2 (cons 3 nil)))");
    assert_eq!(expr("[1; 2;]"), "(cons 1 (cons 2 nil))");
    assert_eq!(expr("None"), "none");
    assert_eq!(expr("Some 3"), "(some 3)");
    assert_eq!(expr("Leaf"), "leaf");
    assert_eq!(expr("Node (Leaf, 1, Leaf)"), "(node leaf 1 leaf)");
    assert_eq!(expr("Node Leaf 1 Leaf"), "(node leaf 1 leaf)");
    assert_eq!(expr("foldl ( + ) 0 [1]"), "(foldl add 0 (cons 1 nil))");
    assert_eq!(expr("( mod )"), "mod");
    assert_eq!(expr("( :: )"), "cons");
    assert_eq!(expr("not (1 = 2)"), "(not (int_eq 1 2))");
    assert_eq!(expr("if true then 1 else 2"), "(if true 1 2)");
}

#[test]
fn application_takes_arguments_by_juxtaposition() {
    assert_eq!(
        expr("array_get (array_tabulate 3 (fun i -> i * i)) 2"),
        "(array_get (array_tabulate 3 (lam (mul $0 $0))) 2)"
    );
}

#[test]
fn open_forms_extend_to_the_right_as_operands() {
    assert_eq!(
        expr("1 + if true then 2 else 3 * 4"),
        "(add 1 (if true 2 (mul 3 4)))"
    );
    assert_eq!(
        expr("1 :: match [] with [] -> [] | _ -> []"),
        "(cons 1 (list_case nil (lam nil) (lam (lam nil))))"
    );
}

#[test]
fn single_arm_match_on_a_tuple_is_a_let() {
    assert_eq!(
        expr("match (1, 2) with (a, b) -> b"),
        "(let (pair 1 2) (let (fst $0) (let (snd $1) $0)))"
    );
}

#[test]
fn comments_run_to_end_of_line() {
    assert_eq!(expr("1 -- the answer\n + 2 -- more"), "(add 1 2)");
}

#[test]
fn local_binders_shadow_primitives_and_definitions() {
    assert_eq!(expr("fun length -> length"), "(lam $0)");
    assert_eq!(program("let f x = x\nlet g f = f 1", "g"), "(lam ($0 1))");
}

#[test]
fn later_definitions_shadow_earlier_ones() {
    let src = "let k = 1\nlet a = k\nlet k = 2\nlet b = (a, k)";
    assert_eq!(program(src, "b"), "(let 1 (let $0 (let 2 (pair $1 $0))))");
}

#[test]
fn close_expression_sees_definitions_and_prelude() {
    let env = spec_env();
    let mut arena = Arena::new();
    let id = env
        .close_expression(&mut arena, "query", "sum [1; 2; 3]")
        .unwrap();
    assert_eq!(
        sexpr::print(&arena, id),
        "(let (lam (foldl (lam (lam (add $1 $0))) 0 $0)) ($0 (cons 1 (cons 2 (cons 3 nil)))))"
    );
}

#[test]
fn inline_names_are_substituted_without_a_binder() {
    let mut arena = Arena::new();
    let lib = arena.lib(dc_core::LibId(4));
    let mut env = TopLevel::new();
    env.define_inline(&arena, "twice", lib).unwrap();
    env.add_source("p", "let f x = twice x x").unwrap();
    let id = env.close_definition(&mut arena, "f").unwrap();
    assert_eq!(sexpr::print(&arena, id), "(lam (lib_4 $0 $0))");
}

#[test]
fn inline_terms_must_be_closed() {
    let mut arena = Arena::new();
    let open = arena.var(0);
    let closed = arena.lam(open);
    let mut env = TopLevel::new();
    env.define_inline(&arena, "good", closed).unwrap();
    assert!(env.define_inline(&arena, "bad", open).is_err());
    assert_eq!(env.names().collect::<Vec<_>>(), vec!["good"]);
}

#[test]
fn a_failed_source_adds_no_definitions() {
    let mut env = TopLevel::new();
    assert_eq!(env.add_source("p", "let z = 0").unwrap(), vec!["z"]);
    assert!(env.add_source("p", "let a = 1\nlet b = nope").is_err());
    assert_eq!(env.names().collect::<Vec<_>>(), vec!["z"]);
}
