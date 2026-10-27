//! Spec §7 invariants 1 (determinism) and 3 (monotone fuel) on random well-typed
//! programs, and type soundness: a well-typed program never gets stuck.
//!
//! `invariants.rs` checks the same invariants on hand-picked programs with
//! hand-derived step counts. Here a typed generator builds programs from every
//! recursion combinator, lazy special forms, lets, arrays and runtime errors, so
//! the invariants are checked on shapes nobody picked.

use std::cell::Cell;

use dc_core::types::infer;
use dc_core::{sexpr, Arena, Scheme, TermId};
use dc_eval::{apply, ErrorKind, Outcome, Value};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

/// The fuel that stands in for "unlimited": generated programs that need more are
/// treated as non-terminating.
const CAP: u64 = 20_000;

/// SplitMix64: a small deterministic generator, so a seed rebuilds the same program
/// on any thread.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        let width = u64::try_from(hi - lo + 1).expect("lo <= hi");
        lo + i64::try_from(self.below(width)).expect("small range")
    }
}

/// The types the generator builds terms of. `Unit` and `Fun` (`Int -> Int`) occur
/// as binder types only, or as a function position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    Int,
    Bool,
    List,
    Fun,
    Unit,
}

/// Generates well-typed core terms as `sexpr` source. `env` lists the binder types,
/// innermost last.
struct Gen<'r> {
    rng: &'r mut Rng,
}

impl Gen<'_> {
    /// A variable of type `ty` from `env`, if there is one.
    fn var(&mut self, ty: Ty, env: &[Ty]) -> Option<String> {
        let found: Vec<usize> = env
            .iter()
            .rev()
            .enumerate()
            .filter(|&(_, &t)| t == ty)
            .map(|(i, _)| i)
            .collect();
        if found.is_empty() {
            None
        } else {
            let i = found[self.rng.below(found.len() as u64) as usize];
            Some(format!("${i}"))
        }
    }

    fn with(env: &[Ty], extra: &[Ty]) -> Vec<Ty> {
        let mut e = env.to_vec();
        e.extend_from_slice(extra);
        e
    }

    fn term(&mut self, ty: Ty, env: &[Ty], depth: u32) -> String {
        let leaf = depth == 0 || self.rng.below(6) == 0;
        match ty {
            Ty::Int => self.int(env, depth, leaf),
            Ty::Bool => self.boolean(env, depth, leaf),
            Ty::List => self.list(env, depth, leaf),
            Ty::Fun => match self.var(Ty::Fun, env) {
                Some(v) if leaf || self.rng.below(2) == 0 => v,
                _ => format!(
                    "(lam {})",
                    self.term(Ty::Int, &Self::with(env, &[Ty::Int]), depth)
                ),
            },
            Ty::Unit => "unit".to_string(),
        }
    }

    fn int(&mut self, env: &[Ty], depth: u32, leaf: bool) -> String {
        if leaf {
            return match self.var(Ty::Int, env) {
                Some(v) if self.rng.below(3) != 0 => v,
                _ => self.rng.range(-3, 10).to_string(),
            };
        }
        let d = depth - 1;
        match self.rng.below(15) {
            0..=2 => {
                let op = ["add", "sub", "mul", "div", "mod"][self.rng.below(5) as usize];
                format!(
                    "({op} {} {})",
                    self.term(Ty::Int, env, d),
                    self.term(Ty::Int, env, d)
                )
            }
            3 => format!(
                "(if {} {} {})",
                self.term(Ty::Bool, env, d),
                self.term(Ty::Int, env, d),
                self.term(Ty::Int, env, d)
            ),
            4 => {
                let bound = [Ty::Int, Ty::List, Ty::Fun][self.rng.below(3) as usize];
                format!(
                    "(let {} {})",
                    self.term(bound, env, d),
                    self.term(Ty::Int, &Self::with(env, &[bound]), d)
                )
            }
            5 => format!("(length {})", self.term(Ty::List, env, d)),
            // foldl and foldr with a two-argument Int function: acc and x are both Int.
            6 | 7 => {
                let fold = if self.rng.below(2) == 0 {
                    "foldl"
                } else {
                    "foldr"
                };
                format!(
                    "({fold} (lam (lam {})) {} {})",
                    self.term(Ty::Int, &Self::with(env, &[Ty::Int, Ty::Int]), d),
                    self.term(Ty::Int, env, d),
                    self.term(Ty::List, env, d)
                )
            }
            8 => format!(
                "(loop {} {} (lam (lam {})))",
                self.small_count(env, d),
                self.term(Ty::Int, env, d),
                self.term(Ty::Int, &Self::with(env, &[Ty::Int, Ty::Int]), d)
            ),
            // bfix and memorec: the body sees self (Int -> Int) and the argument.
            9 | 10 => {
                let comb = if self.rng.below(2) == 0 {
                    "bfix"
                } else {
                    "memorec"
                };
                // Usually enough fuel for the argument; sometimes not.
                let fuel = if self.rng.below(4) == 0 {
                    self.small_count(env, d)
                } else {
                    "16".to_string()
                };
                format!(
                    "({comb} {fuel} (lam (lam {})) {})",
                    self.recursive_body(env, d),
                    self.term(Ty::Int, env, d)
                )
            }
            11 => {
                let f = self.term(Ty::Fun, env, d);
                format!("({f} {})", self.term(Ty::Int, env, d))
            }
            12 => format!(
                "(array_get (array_tabulate {} (lam {})) {})",
                self.small_count(env, d),
                self.term(Ty::Int, &Self::with(env, &[Ty::Int]), d),
                self.index(env, d)
            ),
            13 => format!(
                "(list_case {} (lam {}) (lam (lam {})))",
                self.term(Ty::List, env, d),
                self.term(Ty::Int, &Self::with(env, &[Ty::Unit]), d),
                self.term(Ty::Int, &Self::with(env, &[Ty::Int, Ty::List]), d)
            ),
            _ => format!(
                "(tree_fold 0 (lam (lam (lam {}))) (node (node leaf {} leaf) {} leaf))",
                self.term(Ty::Int, &Self::with(env, &[Ty::Int, Ty::Int, Ty::Int]), d),
                self.term(Ty::Int, env, d),
                self.term(Ty::Int, env, d)
            ),
        }
    }

    /// A recursive body `if x <= 0 then base else step`, where `step` calls self on
    /// a smaller argument, so most recursions finish.
    fn recursive_body(&mut self, env: &[Ty], d: u32) -> String {
        let inner = Self::with(env, &[Ty::Fun, Ty::Int]);
        let base = self.term(Ty::Int, &inner, d);
        let step = self.term(Ty::Int, &inner, d);
        let k = self.rng.range(1, 2);
        format!("(if (le $0 0) {base} (add {step} ($1 (sub $0 {k}))))")
    }

    /// An array index: usually a small literal, sometimes any Int.
    fn index(&mut self, env: &[Ty], d: u32) -> String {
        if self.rng.below(4) == 0 {
            self.term(Ty::Int, env, d)
        } else {
            self.rng.range(0, 2).to_string()
        }
    }

    /// An Int that is usually a small count, so loops and fuels stay small.
    fn small_count(&mut self, env: &[Ty], d: u32) -> String {
        if self.rng.below(3) == 0 {
            format!("(mod {} 12)", self.term(Ty::Int, env, d))
        } else {
            self.rng.range(1, 12).to_string()
        }
    }

    fn boolean(&mut self, env: &[Ty], depth: u32, leaf: bool) -> String {
        if leaf {
            return match self.var(Ty::Bool, env) {
                Some(v) if self.rng.below(2) == 0 => v,
                _ => ["true", "false"][self.rng.below(2) as usize].to_string(),
            };
        }
        let d = depth - 1;
        match self.rng.below(7) {
            0 | 1 => {
                let op = ["lt", "le", "int_eq"][self.rng.below(3) as usize];
                format!(
                    "({op} {} {})",
                    self.term(Ty::Int, env, d),
                    self.term(Ty::Int, env, d)
                )
            }
            2 => format!(
                "(eq {} {})",
                self.term(Ty::List, env, d),
                self.term(Ty::List, env, d)
            ),
            3 => format!("(not {})", self.term(Ty::Bool, env, d)),
            4 => format!(
                "(and {} {})",
                self.term(Ty::Bool, env, d),
                self.term(Ty::Bool, env, d)
            ),
            5 => format!(
                "(or {} {})",
                self.term(Ty::Bool, env, d),
                self.term(Ty::Bool, env, d)
            ),
            _ => format!(
                "(if {} {} {})",
                self.term(Ty::Bool, env, d),
                self.term(Ty::Bool, env, d),
                self.term(Ty::Bool, env, d)
            ),
        }
    }

    fn list(&mut self, env: &[Ty], depth: u32, leaf: bool) -> String {
        if leaf {
            return match self.var(Ty::List, env) {
                Some(v) if self.rng.below(4) != 0 => v,
                _ => "nil".to_string(),
            };
        }
        let d = depth - 1;
        match self.rng.below(7) {
            0 | 1 => format!(
                "(cons {} {})",
                self.term(Ty::Int, env, d),
                self.term(Ty::List, env, d)
            ),
            2 => format!(
                "(foldr cons {} {})",
                self.term(Ty::List, env, d),
                self.term(Ty::List, env, d)
            ),
            3 => format!(
                "(foldl (lam (lam (cons $0 $1))) nil {})",
                self.term(Ty::List, env, d)
            ),
            4 => format!(
                "(array_to_list (array_tabulate {} (lam {})))",
                self.small_count(env, d),
                self.term(Ty::Int, &Self::with(env, &[Ty::Int]), d)
            ),
            5 => format!(
                "(unfold (lam (if (lt 0 $0) (some (pair {} (sub $0 1))) none)) {})",
                self.term(Ty::Int, &Self::with(env, &[Ty::Int]), d),
                self.small_count(env, d)
            ),
            _ => format!(
                "(array_to_list (array_set (array_of_list {}) {} {}))",
                self.term(Ty::List, env, d),
                self.index(env, d),
                self.term(Ty::Int, env, d)
            ),
        }
    }
}

/// A program `λxs. λn. body` and its inputs, both determined by `seed`.
fn program(seed: u64) -> (String, Vec<i64>, i64) {
    let mut rng = Rng(seed);
    let ty = [Ty::Int, Ty::Int, Ty::Bool, Ty::List][rng.below(4) as usize];
    let depth = 3 + rng.below(4) as u32;
    let body = Gen { rng: &mut rng }.term(ty, &[Ty::List, Ty::Int], depth);
    let len = rng.below(10);
    let xs = (0..len).map(|_| rng.range(-5, 12)).collect();
    let n = rng.range(-3, 14);
    (format!("(lam (lam {body}))"), xs, n)
}

fn inputs(xs: &[i64], n: i64) -> Vec<Value> {
    vec![Value::int_list(xs), Value::Int(n)]
}

/// Parses `src` into `arena` and checks that it types with no libraries.
fn parse_typed(arena: &mut Arena, src: &str) -> TermId {
    let t = sexpr::parse(arena, src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let no_libs: &[Scheme] = &[];
    if let Err(e) = infer(arena, t, no_libs) {
        panic!("generated program does not type: {src}: {e}");
    }
    t
}

fn run(src: &str, xs: &[i64], n: i64, fuel: u64) -> Outcome {
    let mut arena = Arena::new();
    let t = parse_typed(&mut arena, src);
    apply(&arena, t, &inputs(xs, n), fuel)
}

fn runner(cases: u32) -> TestRunner {
    TestRunner::new(Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    })
}

/// Counts of outcome kinds, so a test can check that it is not vacuous.
#[derive(Default)]
struct Kinds {
    values: Cell<usize>,
    errors: Cell<usize>,
    long: Cell<usize>,
}

impl Kinds {
    fn record(&self, o: &Outcome) {
        match o {
            Outcome::Value(..) => self.values.set(self.values.get() + 1),
            Outcome::Error(..) => self.errors.set(self.errors.get() + 1),
            Outcome::Timeout(_) => {}
        }
        if o.steps() >= 50 {
            self.long.set(self.long.get() + 1);
        }
    }

    #[track_caller]
    fn check(&self, cases: usize) {
        let (v, e, l) = (self.values.get(), self.errors.get(), self.long.get());
        assert!(v * 2 >= cases, "only {v} of {cases} programs gave a value");
        assert!(
            e * 10 >= cases,
            "only {e} of {cases} programs gave an error"
        );
        assert!(
            l * 3 >= cases,
            "only {l} of {cases} programs took 50 steps or more"
        );
    }
}

const CASES: u32 = 384;

/// The same program and inputs give the same outcome, steps included: run twice,
/// in an arena where the program's term ids differ, and on other threads.
#[test]
fn random_programs_are_deterministic() {
    let kinds = Kinds::default();
    let result = runner(CASES).run(&any::<u64>(), |seed| {
        let (src, xs, n) = program(seed);
        let first = run(&src, &xs, n, CAP);
        kinds.record(&first);
        let want = first.to_string();
        prop_assert_eq!(
            run(&src, &xs, n, CAP).to_string(),
            want.clone(),
            "rerun of {}",
            src
        );

        // Other programs interned first move this one's ids.
        let mut arena = Arena::new();
        for k in 1..4 {
            parse_typed(&mut arena, &program(seed ^ k).0);
        }
        let t = parse_typed(&mut arena, &src);
        let shifted = apply(&arena, t, &inputs(&xs, n), CAP);
        prop_assert_eq!(shifted.to_string(), want.clone(), "shifted ids for {}", src);

        let others: Vec<String> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..2)
                .map(|_| s.spawn(|| run(&src, &xs, n, CAP).to_string()))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread"))
                .collect()
        });
        for o in others {
            prop_assert_eq!(o, want.clone(), "other thread for {}", src);
        }
        Ok(())
    });
    if let Err(e) = result {
        panic!("{e}");
    }
    kinds.check(CASES as usize);
}

/// Monotone fuel: an outcome reached within fuel `f` is the outcome at every fuel
/// above `f`; below the full cost, the run times out, at a count above the fuel
/// and no later than the full run.
#[test]
fn random_programs_have_monotone_fuel() {
    let kinds = Kinds::default();
    let result = runner(CASES).run(
        &(any::<u64>(), any::<u64>(), 0u64..2000),
        |(seed, f, extra)| {
            let (src, xs, n) = program(seed);
            let full = run(&src, &xs, n, CAP);
            kinds.record(&full);
            let total = full.steps();
            let fuel = f % (total + 3);
            let o = run(&src, &xs, n, fuel);
            match (&full, &o) {
                (Outcome::Timeout(_), Outcome::Timeout(s)) => prop_assert!(*s > fuel),
                (_, _) if fuel >= total => {
                    prop_assert_eq!(o.to_string(), full.to_string(), "fuel {} for {}", fuel, src);
                }
                (_, Outcome::Timeout(s)) => {
                    prop_assert!(
                        *s > fuel && *s <= total,
                        "timeout at {} with fuel {}: {}",
                        s,
                        fuel,
                        src
                    );
                }
                _ => prop_assert!(false, "fuel {} < {} gave {} for {}", fuel, total, o, src),
            }
            if !matches!(o, Outcome::Timeout(_)) {
                let more = run(&src, &xs, n, fuel + extra);
                prop_assert_eq!(
                    more.to_string(),
                    o.to_string(),
                    "fuel {} + {} for {}",
                    fuel,
                    extra,
                    src
                );
            }
            Ok(())
        },
    );
    if let Err(e) = result {
        panic!("{e}");
    }
    kinds.check(CASES as usize);
}

/// A program that types never reaches a stuck state (spec §4: every evaluation
/// ends in a value, a runtime error or a timeout).
#[test]
fn random_well_typed_programs_never_get_stuck() {
    let result = runner(1024).run(&any::<u64>(), |seed| {
        let (src, xs, n) = program(seed);
        let o = run(&src, &xs, n, CAP);
        prop_assert!(
            !matches!(o, Outcome::Error(ErrorKind::Stuck(_), _)),
            "{} on {:?}, {}: {}",
            src,
            xs,
            n,
            o
        );
        Ok(())
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
