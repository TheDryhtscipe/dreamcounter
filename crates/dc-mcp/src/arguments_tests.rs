use dc_core::{sexpr, Arena, TermId};
use dc_eval::{apply_within, Limits, Outcome, Value, UNLIMITED};
use dc_surface::prelude::Prelude;

use super::repeated_parameters;
use crate::state::LibraryEntry;
use crate::verify::Verifier;

fn entry(arity: u32, term: &str) -> LibraryEntry {
    LibraryEntry {
        name: None,
        doc: None,
        arity,
        term: term.to_owned(),
    }
}

fn prelude_id(name: &str) -> u32 {
    Prelude::load().unwrap().get(name).unwrap().lib().0
}

/// `fn_0 n xs`: the sum over `i` in `0..n` of `length xs`. It uses `xs`
/// inside the function it folds with, as lib-7108's `flatten_buckets` used
/// its array.
fn length_times() -> LibraryEntry {
    entry(
        2,
        &format!(
            "(lam (lam (foldl (lam (lam (add $1 (length $2)))) 0 (lib_{range} 0 $1))))",
            range = prelude_id("range"),
        ),
    )
}

/// `fn_k x n`: 0 if `n` is 0, and `x` otherwise (lib-7108's
/// `zero_if_second_zero`).
fn zero_if() -> LibraryEntry {
    entry(2, "(lam (lam (if (int_eq $0 0) 0 $1)))")
}

fn verifier(library: &[LibraryEntry]) -> Verifier {
    Verifier::new(library, dc_tasks::harness::DEFAULT_MAX_MEMORY).unwrap()
}

fn compiled(v: &mut Verifier, text: &str) -> TermId {
    v.compile(text).unwrap_or_else(|e| panic!("{text}: {e}"))
}

fn run(v: &mut Verifier, text: &str, inputs: &[Value]) -> Outcome {
    let program = compiled(v, text);
    let stored = sexpr::print(v.arena(), program).unwrap();
    let expanded = v.expand_stored(&stored).unwrap();
    let limits = Limits {
        fuel: 10_000_000,
        memory: UNLIMITED,
    };
    apply_within(v.arena(), expanded, inputs, limits)
}

fn ints(n: i64) -> Value {
    Value::list((0..n).map(Value::Int))
}

fn repeated(arity: u32, term: &str) -> Vec<bool> {
    let mut arena = Arena::new();
    let t = sexpr::parse(&mut arena, term).unwrap();
    repeated_parameters(&arena, t, arity)
}

#[test]
fn an_argument_used_inside_a_function_is_computed_once() {
    let mut v = verifier(&[length_times()]);
    let inline = "let solve xs = fn_0 100 (map (fun x -> x + 1) xs)";
    let bound = "let solve xs = let m = map (fun x -> x + 1) xs in fn_0 100 m";
    // The same term, so the same steps: the arena shares equal terms.
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
    let Outcome::Value(value, steps) = run(&mut v, inline, &[ints(50)]) else {
        panic!("{}", run(&mut v, inline, &[ints(50)]));
    };
    assert_eq!(value, Value::Int(5000));
    // One map over 50 elements, at 3 steps an element, and 100 rounds of the
    // fold. Written out at each use, the map alone would run 100 times: over
    // 15,000 steps.
    assert!(steps < 2000, "{steps} steps");
}

#[test]
fn an_argument_used_at_most_once_is_left_where_it_is() {
    let mut v = verifier(&[zero_if()]);
    let text = "let solve n = fn_0 (10 / n) n";
    let program = compiled(&mut v, text);
    assert_eq!(
        sexpr::print(v.arena(), program).unwrap(),
        "(lam (lib_0 (div 10 $0) $0))"
    );
    // The entry never uses its first parameter when n is 0, so the division
    // by zero does not happen.
    assert!(matches!(
        run(&mut v, text, &[Value::Int(0)]),
        Outcome::Value(Value::Int(0), _)
    ));
}

#[test]
fn free_variables_of_the_arguments_keep_their_meaning() {
    // fn_0 a b = a + a + b + b: both parameters are used twice.
    let twice = entry(2, "(lam (lam (add (add $1 $1) (add $0 $0))))");
    let mut v = verifier(&[twice, zero_if(), length_times()]);
    let inline = "let solve a b = fn_0 (a + b) (a * b)";
    let bound = "let solve a b = let x = a + b in let y = a * b in fn_0 x y";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
    assert!(matches!(
        run(&mut v, inline, &[Value::Int(3), Value::Int(4)]),
        Outcome::Value(Value::Int(38), _)
    ));
    // Only the second parameter of fn_2 is used more than once: the first
    // argument stays in place, under the new binding.
    let inline = "let solve a b = fn_2 (a + b) (range a b)";
    let bound = "let solve a b = let r = range a b in fn_2 (a + b) r";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
    // A use inside an argument is bound inside it.
    let inline = "let solve a b = fn_0 (fn_0 (a + b) 1) b";
    let bound = "let solve a b = let y = (let x = a + b in fn_0 x 1) in fn_0 y b";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
}

#[test]
fn arguments_that_cost_nothing_are_left_where_they_are() {
    let apply_twice = entry(2, "(lam (lam ($1 ($1 $0))))");
    let mut v = verifier(&[apply_twice]);
    let text = "let solve a = fn_0 (fun x -> x + a) a";
    let program = compiled(&mut v, text);
    assert_eq!(
        sexpr::print(v.arena(), program).unwrap(),
        "(lam (lib_0 (lam (add $0 $1)) $0))"
    );
}

#[test]
fn a_use_with_fewer_or_more_arguments_binds_those_it_has() {
    let twice = entry(2, "(lam (lam (add (add $1 $1) (add $0 $0))))");
    let compose = entry(2, "(lam (lam ($1 ($1 $0))))");
    let mut v = verifier(&[twice, compose]);
    let inline = "let solve a = map (fn_0 (a * a)) [1; 2]";
    let bound = "let solve a = map (let x = a * a in fn_0 x) [1; 2]";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
    // fn_1 f x y: the third argument is not the entry's, and stays.
    let inline = "let solve a = fn_1 (fn_0 (a * a)) (fun y -> y + 1) (a - 1)";
    let bound =
        "let solve a = let f = (let x = a * a in fn_0 x) in fn_1 f (fun y -> y + 1) (a - 1)";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
}

#[test]
fn a_parameter_is_repeated_if_one_use_can_evaluate_it_twice() {
    // Inside a function.
    assert_eq!(repeated(1, "(lam (lam (add $0 $1)))"), [true]);
    // Twice in a row.
    assert_eq!(repeated(1, "(lam (add $0 $0))"), [true]);
    // Once, and once on each side of an `if`.
    assert_eq!(
        repeated(2, "(lam (lam (if (lt $0 0) $1 (add $1 1))))"),
        [false, false]
    );
    // In the condition and in one branch.
    assert_eq!(
        repeated(2, "(lam (lam (if (lt $0 0) $1 $0)))"),
        [false, true]
    );
    // Not used.
    assert_eq!(repeated(2, "(lam (lam $0))"), [false, false]);
    // Under a `let`, which is not a function.
    assert_eq!(repeated(1, "(lam (let 1 (add $0 $1)))"), [false]);
    // In what a `let` binds and in its body.
    assert_eq!(repeated(1, "(lam (let $0 (add $0 $1)))"), [true]);
    assert_eq!(
        repeated(2, "(lam (lam (let $1 (add $0 $2))))"),
        [true, false]
    );
    // Both sides of `and` and `or` can be evaluated.
    assert_eq!(repeated(1, "(lam (and $0 $0))"), [true]);
    assert_eq!(repeated(0, "(add 1 2)"), [] as [bool; 0]);
}

#[test]
fn a_parameter_passed_on_to_an_entry_that_repeats_it_is_repeated() {
    // fn_1 x = fn_0 3 x, and fn_0 uses its second parameter inside a function.
    let through = entry(1, "(lam (lib_0 3 $0))");
    let mut v = verifier(&[length_times(), through]);
    let inline = "let solve xs = fn_1 (map (fun x -> x + 1) xs)";
    let bound = "let solve xs = let m = map (fun x -> x + 1) xs in fn_1 m";
    assert_eq!(compiled(&mut v, inline), compiled(&mut v, bound));
}

/// The two entries of run lib-7108 whose use together ran out of steps
/// (bead dreamcounter-a2602b), as its journal has them.
#[test]
fn the_entries_of_run_lib_7108_cost_one_pass_over_the_buckets() {
    let (concat, map, range) = (prelude_id("concat"), prelude_id("map"), prelude_id("range"));
    let flatten_buckets = entry(
        1,
        &format!("(lam (lib_{concat} (lib_{map} (lam (array_get $1 $0)) (lib_{range} 0 16384))))"),
    );
    let bucket_indices_by_value = entry(
        2,
        &format!(
            "(lam (lam (foldl (lam (lam (let (array_get $2 $0) (array_set $2 $0 (cons $1 \
             (array_get $2 $0)))))) (array_tabulate 16384 (lam nil)) (lib_{range} 0 $1))))"
        ),
    );
    let mut v = verifier(&[flatten_buckets, bucket_indices_by_value]);
    let text = "let solve xs = fn_0 (fn_1 (length xs) (array_of_list xs))";
    let Outcome::Value(value, steps) = run(&mut v, text, &[ints(2)]) else {
        panic!("{}", run(&mut v, text, &[ints(2)]));
    };
    assert_eq!(value, Value::list([Value::Int(0), Value::Int(1)]));
    // Building the buckets takes about 49,000 steps and flattening them
    // about 230,000. Written out at each use, the buckets were built 16384
    // times: about 800,000,000 steps, against a limit of 10,000,000.
    assert!(steps < 400_000, "{steps} steps");
}
