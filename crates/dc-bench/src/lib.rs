//! The throughput benchmark programs (spec §11).
//!
//! The set is fixed: `sum` and `max_subarray` on n = 10⁴, `msort` on n = 10³,
//! naive `fib 20`, memo `fib 60`, and a binary search over a 10⁵ array. The
//! programs are written in dc ([`SOURCE`]) and compiled against the prelude.
//! Every run is checked against a Rust reference result ([`Compiled::run`]), so a
//! wrong evaluator cannot post a number.
//!
//! [`Program`] values hold no dc values, so they can be shared between threads;
//! the inputs ([`Instance`]) are built on the thread that evaluates them, because
//! dc values are not `Send`. A [`Compiled`] program is a read-only term arena and
//! can be shared by reference.

use std::fmt;

use dc_core::types::infer;
use dc_core::{Arena, TermId, TypeError};
use dc_eval::{apply, Outcome, Value};
use dc_surface::prelude::{Prelude, PreludeError, PreludeMode};
use dc_surface::SurfaceError;

/// The benchmark programs in dc. `sum`, `max_subarray`, `merge` and `msort` are the
/// spec §10 samples verbatim (`sum` shadows the prelude's own).
pub const SOURCE: &str = r"
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

let fib_naive n =
  bfix (n + 1)
    (fun self k -> if k < 2 then k else self (k - 1) + self (k - 2))
    n

let fib_memo n =
  memorec (n + 1)
    (fun self k -> if k < 2 then k else self (k - 1) + self (k - 2))
    n

-- The index of target in the sorted array arr, or -1. The search halves the
-- range [lo, hi) each call, so a depth fuel of 64 always suffices.
let bsearch arr target =
  bfix 64
    (fun self (lo, hi) ->
      if hi <= lo then -1
      else
        let mid = lo + (hi - lo) / 2 in
        let x = array_get arr mid in
        if x = target then mid
        else if x < target then self (mid + 1, hi)
        else self (lo, mid))
    (0, array_size arr)
";

/// The global step budget of one run: far above what any program here takes.
pub const FUEL: u64 = 10_000_000_000;

/// What a program computes, which fixes its inputs and Rust reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Sum of n integers in [-1000, 1000].
    Sum,
    /// Maximum subarray sum of n integers in [-1000, 1000].
    MaxSubarray,
    /// Merge sort of n integers in [0, 10⁶).
    Msort,
    /// The n-th Fibonacci number by naive tree recursion.
    FibNaive,
    /// The n-th Fibonacci number by memoised recursion.
    FibMemo,
    /// Binary search for one present element of the array `[0; 3; 6; ...]` of
    /// length n.
    Bsearch,
}

/// One benchmark program.
#[derive(Clone, Copy, Debug)]
pub struct Program {
    /// Its name in reports.
    pub name: &'static str,
    /// The top-level definition in [`SOURCE`] that is the program.
    pub entry: &'static str,
    /// The input size it is benchmarked at.
    pub full_n: i64,
    /// A small input size, for tests.
    pub small_n: i64,
    /// What it computes.
    pub kind: Kind,
}

/// The benchmark programs, in report order.
pub fn programs() -> Vec<Program> {
    let p = |name, full_n, small_n, kind| Program {
        name,
        entry: name,
        full_n,
        small_n,
        kind,
    };
    vec![
        p("sum", 10_000, 50, Kind::Sum),
        p("max_subarray", 10_000, 50, Kind::MaxSubarray),
        p("msort", 1_000, 20, Kind::Msort),
        p("fib_naive", 20, 8, Kind::FibNaive),
        p("fib_memo", 60, 20, Kind::FibMemo),
        p("bsearch", 100_000, 100, Kind::Bsearch),
    ]
}

/// A deterministic integer sequence (SplitMix64), so inputs are fixed across runs.
fn ints(seed: u64, n: i64, lo: i64, hi: i64) -> Vec<i64> {
    let mut s = seed;
    let span = (hi - lo) as u64;
    (0..n.max(0))
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            lo + (z % span) as i64
        })
        .collect()
}

fn signed_list(n: i64) -> Vec<i64> {
    ints(1, n, -1000, 1001)
}

fn msort_list(n: i64) -> Vec<i64> {
    ints(2, n, 0, 1_000_000)
}

fn bsearch_array(n: i64) -> Vec<i64> {
    (0..n).map(|i| 3 * i).collect()
}

fn bsearch_target(n: i64) -> i64 {
    3 * (n * 7 / 10)
}

fn fib(n: i64) -> i64 {
    let (mut a, mut b) = (0i64, 1i64);
    for _ in 0..n {
        (a, b) = (b, a + b);
    }
    a
}

impl Program {
    /// The inputs at size `n`.
    pub fn inputs(&self, n: i64) -> Vec<Value> {
        match self.kind {
            Kind::Sum | Kind::MaxSubarray => vec![Value::int_list(&signed_list(n))],
            Kind::Msort => vec![Value::int_list(&msort_list(n))],
            Kind::FibNaive | Kind::FibMemo => vec![Value::Int(n)],
            Kind::Bsearch => vec![
                Value::array(bsearch_array(n).into_iter().map(Value::Int)),
                Value::Int(bsearch_target(n)),
            ],
        }
    }

    /// The Rust reference result at size `n`.
    pub fn expected(&self, n: i64) -> Value {
        match self.kind {
            Kind::Sum => Value::Int(signed_list(n).iter().sum()),
            Kind::MaxSubarray => {
                // Kadane's algorithm, from the same start as the dc program.
                let (mut best, mut cur) = (i64::MIN, 0i64);
                for x in signed_list(n) {
                    cur = x.max(cur + x);
                    best = best.max(cur);
                }
                Value::Int(best)
            }
            Kind::Msort => {
                let mut xs = msort_list(n);
                xs.sort_unstable();
                Value::int_list(&xs)
            }
            Kind::FibNaive | Kind::FibMemo => Value::Int(fib(n)),
            Kind::Bsearch => {
                let i = bsearch_array(n)
                    .binary_search(&bsearch_target(n))
                    .map_or(-1, |i| i as i64);
                Value::Int(i)
            }
        }
    }

    /// The inputs and expected result at size `n`.
    pub fn instance(&self, n: i64) -> Instance {
        Instance::new(self.inputs(n), self.expected(n))
    }

    /// Compiles the program against the prelude, type-checks it and expands the
    /// prelude references.
    pub fn compile(&self) -> Result<Compiled, BenchError> {
        let prelude = Prelude::load().map_err(BenchError::Prelude)?;
        let mut top = prelude.top_level(PreludeMode::On);
        top.add_source("bench", SOURCE)
            .map_err(BenchError::Compile)?;
        let mut arena = Arena::new();
        let t = top
            .close_definition(&mut arena, self.entry)
            .map_err(BenchError::Compile)?;
        infer(&arena, t, &prelude).map_err(BenchError::Type)?;
        let term = prelude.expand(&mut arena, t);
        Ok(Compiled {
            program: *self,
            arena,
            term,
        })
    }
}

/// The inputs of one run and the result it must give.
#[derive(Debug)]
pub struct Instance {
    inputs: Vec<Value>,
    expected: Value,
}

impl Instance {
    /// An instance from inputs and the result they must give.
    pub fn new(inputs: Vec<Value>, expected: Value) -> Instance {
        Instance { inputs, expected }
    }
}

/// A program compiled to a closed core term, ready to run.
#[derive(Debug)]
pub struct Compiled {
    program: Program,
    arena: Arena,
    term: TermId,
}

/// At most this many characters of a value appear in an error.
const SHOWN: usize = 200;

fn shown(v: &impl fmt::Display) -> String {
    let s = v.to_string();
    match s.char_indices().nth(SHOWN) {
        Some((i, _)) => format!("{}...", &s[..i]),
        None => s,
    }
}

impl Compiled {
    /// The program this was compiled from.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Runs the program once on `inst` and checks the result against the
    /// expected one. Returns the number of steps taken.
    pub fn run(&self, inst: &Instance) -> Result<u64, BenchError> {
        match apply(&self.arena, self.term, &inst.inputs, FUEL) {
            Outcome::Value(v, steps) if v == inst.expected => Ok(steps),
            Outcome::Value(v, _) => Err(BenchError::WrongResult {
                program: self.program.name,
                got: shown(&v),
                expected: shown(&inst.expected),
            }),
            other => Err(BenchError::NotAValue {
                program: self.program.name,
                outcome: shown(&other),
            }),
        }
    }
}

/// Why a benchmark program could not be compiled or gave a wrong result.
#[derive(Debug)]
pub enum BenchError {
    /// The prelude failed to load.
    Prelude(PreludeError),
    /// [`SOURCE`] failed to compile.
    Compile(SurfaceError),
    /// The program failed to type-check.
    Type(TypeError),
    /// Evaluation ended in an error or a timeout.
    NotAValue {
        /// The program's name.
        program: &'static str,
        /// The outcome, shortened.
        outcome: String,
    },
    /// Evaluation gave a value other than the Rust reference.
    WrongResult {
        /// The program's name.
        program: &'static str,
        /// The value it gave, shortened.
        got: String,
        /// The reference value, shortened.
        expected: String,
    },
}

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BenchError::Prelude(e) => write!(f, "prelude: {e}"),
            BenchError::Compile(e) => write!(f, "compile: {e}"),
            BenchError::Type(e) => write!(f, "type: {e}"),
            BenchError::NotAValue { program, outcome } => {
                write!(f, "{program}: evaluation gave {outcome}")
            }
            BenchError::WrongResult {
                program,
                got,
                expected,
            } => write!(f, "{program}: got {got}, expected {expected}"),
        }
    }
}

impl std::error::Error for BenchError {}
