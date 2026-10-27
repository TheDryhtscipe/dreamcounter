//! Exact step counts for core forms and constant-cost primitives (P2.6).
//! Each test states its arithmetic; the cost table is `dc_eval::cost`.

mod common;

use common::*;
use dc_eval::{Outcome, Value};

#[test]
fn beta_plus_add() {
    // β 1 + add 1 = 2.
    assert_value("((lam (add $0 1)) 41)", int(42), 2);
}

#[test]
fn literals_lambdas_and_variables_are_free() {
    assert_value("42", int(42), 0);
    assert_value("-7", int(-7), 0);
    assert_value("true", Value::Bool(true), 0);
    assert_value("unit", Value::Unit, 0);
    match run("(lam (div $0 0))") {
        Outcome::Value(v, 0) => assert!(v.is_fun()),
        other => panic!("{other}"),
    }
}

#[test]
fn let_binding_is_free() {
    // add 1; the let and the two lookups are free.
    assert_value("(let 5 (add $0 $0))", int(10), 1);
    // mul 1.
    assert_value("(let 2 (let 3 (mul $1 $0)))", int(6), 1);
    // The bound expression is charged as usual: add 1 + mul 1.
    assert_value("(let (add 1 2) (mul $0 $0))", int(9), 2);
}

#[test]
fn curried_application_costs_one_beta_per_argument() {
    // β 2 + sub 1 = 3.
    assert_value("((lam (lam (sub $1 $0))) 10 3)", int(7), 3);
    // β 3 + add 2 = 5.
    assert_value("((lam (lam (lam (add $2 (add $1 $0))))) 1 2 3)", int(6), 5);
}

#[test]
fn partial_application_of_a_primitive_is_free() {
    match run("(add 1)") {
        Outcome::Value(v, 0) => assert!(v.is_fun()),
        other => panic!("{other}"),
    }
    // The saturating application is the only charge: add 1.
    assert_value("(let (add 1) ($0 2))", int(3), 1);
    // A partial application used twice: add 1 + add 1 + outer add 1 = 3.
    assert_value("(let (add 10) (add ($0 1) ($0 2)))", int(23), 3);
}

#[test]
fn nullary_primitives_cost_one() {
    assert_value("nil", ints(&[]), 1);
    assert_value("none", Value::none(), 1);
    assert_value("leaf", Value::Tree(dc_eval::Tree::leaf()), 1);
}

#[test]
fn constant_cost_primitives() {
    // nil 1 + cons 1 + cons 1 + length 1 = 4.
    assert_value("(length (cons 1 (cons 2 nil)))", int(2), 4);
    // pair 1 + fst 1.
    assert_value("(fst (pair 1 2))", int(1), 2);
    assert_value("(snd (pair 1 2))", int(2), 2);
    assert_value("(some 3)", Value::some(int(3)), 1);
    assert_value("(not true)", Value::Bool(false), 1);
    assert_value("(int_eq 3 3)", Value::Bool(true), 1);
    assert_value("(lt 3 2)", Value::Bool(false), 1);
    assert_value("(le 2 2)", Value::Bool(true), 1);
    assert_value("(mul 6 7)", int(42), 1);
    assert_value("(sub 6 7)", int(-1), 1);
}

#[test]
fn div_and_mod_truncate_toward_zero() {
    assert_value("(div -7 2)", int(-3), 1);
    assert_value("(mod -7 2)", int(-1), 1);
    assert_value("(mod 7 -2)", int(1), 1);
}

#[test]
fn if_charges_one_decision_and_skips_the_other_branch() {
    // lt 1 + if 1; the else branch would divide by zero.
    assert_value("(if (lt 1 2) 10 (div 1 0))", int(10), 2);
    // if 1.
    assert_value("(if false (div 1 0) 20)", int(20), 1);
}

#[test]
fn and_or_decide_once_and_short_circuit() {
    // and 1; the right operand is not evaluated.
    assert_value("(and false (int_eq (div 1 0) 0))", Value::Bool(false), 1);
    assert_value("(or true (int_eq (div 1 0) 0))", Value::Bool(true), 1);
    // and 1 + lt 1 = 2.
    assert_value("(and true (lt 1 2))", Value::Bool(true), 2);
    assert_value("(or false (lt 2 1))", Value::Bool(false), 2);
}

#[test]
fn option_case_charges_one_plus_the_branch_call() {
    // some 1 + option_case 1 + β 1 + add 1 = 4.
    assert_value("(option_case (some 3) (lam 0) (lam (add $0 1)))", int(4), 4);
    // none 1 + option_case 1 + β 1 (the thunk applied to unit) = 3.
    assert_value("(option_case none (lam 0) (lam $0))", int(0), 3);
}

#[test]
fn list_case_charges_one_plus_the_branch_call() {
    // nil 1 + cons 1 + list_case 1 + β 2 = 5.
    assert_value("(list_case (cons 1 nil) (lam 0) (lam (lam $1)))", int(1), 5);
    // The tail is passed second: β 2 + list_case 1 + length 1 = 4 on an input.
    assert_value_on(
        "(lam (list_case $0 (lam 0) (lam (lam (length $0)))))",
        &[ints(&[7, 8, 9])],
        int(2),
        // β 1 (the program) + list_case 1 + β 2 + length 1 = 5.
        5,
    );
    // nil 1 + list_case 1 + β 1 = 3.
    assert_value("(list_case nil (lam 0) (lam (lam $1)))", int(0), 3);
}

#[test]
fn eq_charges_value_nodes_compared() {
    // One node pair each.
    assert_value("(eq 3 3)", Value::Bool(true), 1);
    assert_value("(eq 3 4)", Value::Bool(false), 1);
    // pair 1 + pair 1 + nodes (pair, 1, 2) 3 = 5.
    assert_value("(eq (pair 1 2) (pair 1 2))", Value::Bool(true), 5);
    // pair 2 + nodes (pair, 1 vs 0) 2: stops at the first difference.
    assert_value("(eq (pair 1 2) (pair 0 2))", Value::Bool(false), 4);
    // pair 2 + nodes (pair, 1, 2 vs 3) 3.
    assert_value("(eq (pair 1 2) (pair 1 3))", Value::Bool(false), 5);
    // some 2 + nodes (option, 1) 2.
    assert_value("(eq (some 1) (some 1))", Value::Bool(true), 4);
    // none 1 + some 1 + nodes (option) 1.
    assert_value("(eq none (some 1))", Value::Bool(false), 3);
}

#[test]
fn eq_on_lists_counts_cells_heads_and_nil() {
    let p = "(lam (lam (eq $1 $0)))";
    // β 2 + nodes (cell, 1, cell, 2, nil) 5 = 7.
    assert_value_on(p, &[ints(&[1, 2]), ints(&[1, 2])], Value::Bool(true), 7);
    // β 2 + nodes (cell, 1, cell, 2 vs 3) 4 = 6.
    assert_value_on(p, &[ints(&[1, 2]), ints(&[1, 3])], Value::Bool(false), 6);
    // β 2 + nodes (cell, 1, nil vs cell) 3 = 5.
    assert_value_on(p, &[ints(&[1]), ints(&[1, 2])], Value::Bool(false), 5);
    // β 2 + nodes (nil) 1 = 3.
    assert_value_on(p, &[ints(&[]), ints(&[])], Value::Bool(true), 3);
}

#[test]
fn applying_a_program_to_inputs_is_one_beta_per_input() {
    // β 2 + add 1 = 3.
    assert_value_on("(lam (lam (add $1 $0)))", &[int(2), int(3)], int(5), 3);
    // A primitive program: partial application free, add 1.
    assert_value_on("add", &[int(2), int(3)], int(5), 1);
    // Inputs are values: building them is not charged.
    assert_value_on("(lam (length $0))", &[range1(1000)], int(1000), 2);
}
