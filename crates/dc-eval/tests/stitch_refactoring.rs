//! Spec §7 invariant 2 (refactoring invariance) end to end, for hand-written
//! abstractions in Stitch's format: the abstraction goes through
//! `dc_stitch::decode_abstraction` straight into a `LibDef`, the rewritten program
//! through `dc_stitch::decode`, and the expansion must be the original program, run
//! to the same outcome in the same number of steps on random inputs, with the same
//! type.
//!
//! The abstractions are the shapes that matter for cost: a duplicated parameter
//! (evaluated at each occurrence), parameters in lazy positions (`if`, `and`, `or`,
//! under a lambda), parameters under binders of the body (shifted), arguments that
//! bind variables, a let in the body, recursion combinators, and an abstraction
//! built on an earlier one. Several of them raise runtime errors on some inputs,
//! and the errors must match too.

use dc_core::types::infer;
use dc_core::{expand, sexpr, Arena, LibDef, Scheme, TermId};
use dc_eval::{apply, Value, UNLIMITED};
use dc_stitch::{decode, decode_abstraction, LibNames};
use proptest::prelude::*;

/// Abstractions in library order (`fn_0`, `fn_1`, ...) as `(body, arity)`, a
/// program rewritten with them, and the original program. Every program is
/// `λxs. λn. ...` over a list and an Int.
struct Case {
    abstractions: &'static [(&'static str, u32)],
    rewritten: &'static str,
    original: &'static str,
}

const CASES: &[Case] = &[
    // A fold, used twice, once on an argument that is a computation.
    Case {
        abstractions: &[("(foldl (lam (lam (add $1 $0))) int_0 #0)", 1)],
        rewritten: "(lam (lam (add (fn_0 $1) (fn_0 (cons $0 $1)))))",
        original: "(lam (lam (add (foldl (lam (lam (add $1 $0))) 0 $1) \
                   (foldl (lam (lam (add $1 $0))) 0 (cons $0 $1)))))",
    },
    // A guarded division: #0 occurs twice and #1 sits in a lazy branch.
    Case {
        abstractions: &[("(if_ (int_eq #0 int_0) int_0 (div #1 #0))", 2)],
        rewritten: "(lam (lam (fn_0 (sub $0 int_3) (length $1))))",
        original: "(lam (lam (if (int_eq (sub $0 3) 0) 0 (div (length $1) (sub $0 3)))))",
    },
    // map: the function parameter sits under the fold's two binders, and the
    // argument is a lambda that refers to the program's input.
    Case {
        abstractions: &[("(foldr (lam (lam (cons (#0 $1) $0))) nil #1)", 2)],
        rewritten: "(lam (lam (fn_0 (lam (mul $0 $1)) $1)))",
        original: "(lam (lam (foldr (lam (lam (cons ((lam (mul $0 $3)) $1) $0))) nil $1)))",
    },
    // Memoised fib, used twice; the parameter is both the fuel and the argument.
    Case {
        abstractions: &[(
            "(memorec (add #0 int_1) (lam (lam (if_ (lt $0 int_2) $0 \
             (add ($1 (sub $0 int_1)) ($1 (sub $0 int_2)))))) #0)",
            1,
        )],
        rewritten: "(lam (lam (pair (fn_0 (mod $0 int_30)) (fn_0 (length $1)))))",
        original: "(lam (lam (pair \
                   (memorec (add (mod $0 30) 1) (lam (lam (if (lt $0 2) $0 \
                     (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) (mod $0 30)) \
                   (memorec (add (length $1) 1) (lam (lam (if (lt $0 2) $0 \
                     (add ($1 (sub $0 1)) ($1 (sub $0 2)))))) (length $1)))))",
    },
    // Factorial with a fuel parameter: FuelExhausted when n is at least the fuel,
    // Overflow from 21!.
    Case {
        abstractions: &[(
            "(bfix #0 (lam (lam (if_ (le $0 int_0) int_1 (mul $0 ($1 (sub $0 int_1)))))) #1)",
            2,
        )],
        rewritten: "(lam (lam (fn_0 (length $1) $0)))",
        original: "(lam (lam (bfix (length $1) \
                   (lam (lam (if (le $0 0) 1 (mul $0 ($1 (sub $0 1)))))) $0)))",
    },
    // A let in the body: #1 is under the let's binder. Out-of-range indices fail.
    Case {
        abstractions: &[(
            "(let_ (length #0) (lam_0 (array_get (array_tabulate $0 (lam (mul $0 $0))) #1)))",
            2,
        )],
        rewritten: "(lam (lam (fn_0 $1 $0)))",
        original: "(lam (lam (let (length $1) \
                   (array_get (array_tabulate $0 (lam (mul $0 $0))) $1))))",
    },
    // Short-circuit forms: the division by #0 runs only when 0 < #0.
    Case {
        abstractions: &[(
            "(and_ (lt int_0 #0) (or_ (int_eq (div int_100 #0) #1) (lt #1 #0)))",
            2,
        )],
        rewritten: "(lam (lam (fn_0 $0 (length $1))))",
        original: "(lam (lam (and (lt 0 $0) \
                   (or (int_eq (div 100 $0) (length $1)) (lt (length $1) $0)))))",
    },
    // A later abstraction built on an earlier one, used under a binder of the
    // program. Long lists of large elements overflow.
    Case {
        abstractions: &[("(add #0 int_1)", 1), ("(lam (fn_0 (mul #0 $0)))", 1)],
        rewritten: "(lam (lam (foldl (lam (fn_1 $0)) $0 $1)))",
        original: "(lam (lam (foldl (lam (lam (add (mul $1 $0) 1))) $0 $1)))",
    },
];

fn names() -> LibNames {
    LibNames::sequential("fn_", 4).expect("names")
}

/// The schemes of `libs`, each inferred from its definition with the earlier ones.
fn lib_schemes(a: &Arena, libs: &[LibDef]) -> Vec<Scheme> {
    let mut out: Vec<Scheme> = Vec::with_capacity(libs.len());
    for def in libs {
        let s = infer(a, def.term, &out)
            .unwrap_or_else(|e| panic!("{}: {e}", sexpr::print(a, def.term)));
        out.push(s);
    }
    out
}

/// A case decoded and expanded: the arena, the original and the expansion.
struct Decoded {
    arena: Arena,
    original: TermId,
    expanded: TermId,
}

/// Decodes and expands `case`, checking that the expansion is the original term
/// and that the rewritten program has the expansion's type.
fn decode_case(case: &Case) -> Decoded {
    let n = names();
    let mut a = Arena::new();
    let libs: Vec<LibDef> = case
        .abstractions
        .iter()
        .map(|&(body, arity)| LibDef {
            arity,
            term: decode_abstraction(&mut a, body, arity, &n)
                .unwrap_or_else(|e| panic!("{body}: {e}")),
        })
        .collect();
    let p =
        decode(&mut a, case.rewritten, &n).unwrap_or_else(|e| panic!("{}: {e}", case.rewritten));
    let original =
        sexpr::parse(&mut a, case.original).unwrap_or_else(|e| panic!("{}: {e}", case.original));
    let expanded = expand(&mut a, &libs, p).unwrap_or_else(|e| panic!("{}: {e}", case.rewritten));
    assert_eq!(
        sexpr::print(&a, expanded),
        sexpr::print(&a, original),
        "{}",
        case.rewritten
    );
    assert_eq!(expanded, original);

    let schemes = lib_schemes(&a, &libs);
    let refactored_type =
        infer(&a, p, &schemes).unwrap_or_else(|e| panic!("{}: {e}", case.rewritten));
    let no_libs: &[Scheme] = &[];
    let expanded_type = infer(&a, expanded, no_libs).expect("expansion types");
    assert_eq!(refactored_type, expanded_type, "{}", case.rewritten);
    Decoded {
        arena: a,
        original,
        expanded,
    }
}

#[test]
fn hand_written_abstractions_decode_and_expand_to_the_original() {
    for case in CASES {
        decode_case(case);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn hand_written_abstractions_preserve_outcome_and_steps(
        which in 0..CASES.len(),
        xs in prop::collection::vec(-3i64..10, 0..30),
        n in -3i64..26,
    ) {
        let d = decode_case(&CASES[which]);
        let inputs = [Value::int_list(&xs), Value::Int(n)];
        let want = apply(&d.arena, d.original, &inputs, UNLIMITED);
        let got = apply(&d.arena, d.expanded, &inputs, UNLIMITED);
        prop_assert_eq!(got.to_string(), want.to_string(), "{}", CASES[which].rewritten);
    }
}

#[test]
fn fixed_inputs_reach_errors_and_guards() {
    // Fixed inputs that drive each error case to its error, and each guard to the
    // branch that skips a failing division, so the property above compares errors
    // and lazy positions as well as values.
    let expected: &[(usize, &[i64], i64, &str)] = &[
        // The guard keeps the division by zero from running.
        (1, &[1, 2], 3, "Value(0, "),
        (6, &[1, 2], 0, "Value(false, "),
        (3, &[], -3, "Error(FuelExhausted"),
        (4, &[1, 2, 3], 3, "Error(FuelExhausted"),
        (4, &[0; 25], 22, "Error(Overflow"),
        (5, &[1, 2], 2, "Error(IndexOutOfBounds"),
        (7, &[9; 25], 9, "Error(Overflow"),
    ];
    for &(i, xs, n, prefix) in expected {
        let d = decode_case(&CASES[i]);
        let inputs = [Value::int_list(xs), Value::Int(n)];
        let o = apply(&d.arena, d.expanded, &inputs, UNLIMITED);
        assert!(o.to_string().starts_with(prefix), "case {i}: {o}");
    }
}
