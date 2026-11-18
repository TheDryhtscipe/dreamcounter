//! Chains (bead dreamcounter-4593ae): dynamic programmes over the chains of
//! two lists. A chain is a set of positions `i1 < i2 < ... < ik` along which
//! both lists strictly increase: `xs[i1] < xs[i2] < ...` and
//! `ys[i1] < ys[i2] < ...`. A single position is a chain.
//!
//! | variant | answer |
//! |---|---|
//! | [`Variant::Count`] | the number of chains, mod [`MODULUS`] |
//! | [`Variant::Longest`] | the most positions in a chain |
//! | [`Variant::Heaviest`] | the largest sum of a third list `ws` over a chain |
//! | [`Variant::LongestCount`] | the number of chains of the most positions, mod [`MODULUS`] |
//!
//! Each answer comes from the value of the chains that end at each
//! position, which is built from the values at the earlier positions below
//! it in both lists. The variants differ only in how those values combine:
//! a sum, a maximum, a maximum with weights, or a maximum with a count of
//! ties.
//!
//! The family is the harder part of the development corpus, after the
//! [`dominance`](super::dominance) family proved easy (results log,
//! 2026-11-18). A fast program needs the values at earlier positions to be
//! final before it uses them, so the method is divide and conquer over
//! positions taken in order: solve the earlier half, pass its values to the
//! later half by a sweep in the order of `xs` with a structure over `ys`,
//! then solve the later half. That is `n log² n`. Counting the pairs of
//! positions in order in both lists, which needs no such order, is easier:
//! in a probe before this family was built, programs written without being
//! run solved such counts every time, and solved these variants half the
//! time (bead dreamcounter-4593ae notes).
//!
//! Each variant has a [`Gate`]: [`GATE_INPUTS`], each of [`GATE_N`]
//! elements, each with [`GATE_FUEL`] steps. A program passes only if it
//! answers each within that fuel. The gate checks only these inputs. They
//! are chosen so that the slow programs in the tests of `tests/chains.rs`
//! run out of fuel on at least one. The naive references, which look at
//! every earlier position, need about `8 n²/2` steps at the least, over
//! 250,000,000. The good references take about 80,000,000, and about 88,000,000 for
//! longest-count. The
//! gate's fuel is above the harness's global cap
//! ([`DEFAULT_MAX_FUEL`](crate::harness::DEFAULT_MAX_FUEL)): at a size where
//! `n²` and `n log² n` programs are far enough apart, the good ones need
//! more than that cap. The description states the size, the value range
//! and the step limit, not the inputs.
//!
//! Values are between 0 and `n - 1` for inputs of `n` elements, and between
//! 0 and [`GATE_N`]` - 1` on the gate, so that a program can index an array
//! by value without one larger than the list (bead dreamcounter-a3db93).
//!
//! The references are in [`SOURCE`]. The good ones share one divide and
//! conquer, `chain_dp`, which takes the way to combine values as an
//! argument, so a learned library can hold it. The fuel plan is built from
//! the naive reference.
//!
//! Inputs come in one kind per seed, the same at every size (decision D15):
//! seed `k` gets kind `k mod 3`. Kind 0 draws both lists from `0..n`;
//! kind 1 from `0..=n / 8`, so each value comes about 8 times and the strict
//! comparisons matter; kind 2 draws `xs` from `0..n` and puts each `ys`
//! near its `xs`, so chains are long. [`Variant::Heaviest`] draws `ws` from
//! `0..n` in every kind.
//!
//! The oracles do not follow the references: they fill in the value at each
//! position from every earlier position, in Rust. This module's tests check
//! them against a count over every set of positions.

use dc_core::Scheme;
use dc_eval::Value;

use crate::rng::TaskRng;
use crate::task::{generate_input, Example, Gate, Inputs, TaskFamily};

/// The reference solutions in dc surface syntax, compiled against the
/// prelude. [`Variant::naive`] and [`Variant::good`] name the definitions.
pub const SOURCE: &str = include_str!("chains.dc");

/// The sweep, as in the dominance family: the naive references take a few
/// `n²` steps, so ten times their steps at 512 stays under the default
/// fuel cap.
pub const SIZES: &[u64] = &[16, 32, 64, 128, 256, 512];

/// The size of the gate.
pub const GATE_N: u64 = 8192;

/// The fuel of the gate: the harness's default gate cap.
pub const GATE_FUEL: u64 = crate::harness::DEFAULT_MAX_GATE_FUEL;

/// [`Variant::Count`] and [`Variant::LongestCount`] count mod this prime.
pub const MODULUS: i64 = 1_000_003;

/// A variant of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// The number of chains, mod [`MODULUS`].
    Count,
    /// The most positions in a chain.
    Longest,
    /// The largest sum of `ws` over the positions of a chain.
    Heaviest,
    /// The number of chains of the most positions, mod [`MODULUS`].
    LongestCount,
}

impl Variant {
    /// Every variant, in the order of the table above.
    pub const ALL: [Variant; 4] = [
        Variant::Count,
        Variant::Longest,
        Variant::Heaviest,
        Variant::LongestCount,
    ];

    /// The definition in [`SOURCE`] of the naive (`n²`) reference.
    pub fn naive(self) -> &'static str {
        match self {
            Variant::Count => "chains_naive",
            Variant::Longest => "longest_chain_naive",
            Variant::Heaviest => "heaviest_chain_naive",
            Variant::LongestCount => "longest_chains_naive",
        }
    }

    /// The definition in [`SOURCE`] of the good (`n log² n`) reference.
    pub fn good(self) -> &'static str {
        match self {
            Variant::Count => "chains_good",
            Variant::Longest => "longest_chain_good",
            Variant::Heaviest => "heaviest_chain_good",
            Variant::LongestCount => "longest_chains_good",
        }
    }

    /// The number of lists a solution takes: three for
    /// [`Variant::Heaviest`], two for the others.
    pub fn lists(self) -> usize {
        if self == Variant::Heaviest {
            3
        } else {
            2
        }
    }
}

/// One variant of the chain family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chains(pub Variant);

/// Every variant as a family.
pub fn all() -> Vec<Chains> {
    Variant::ALL.into_iter().map(Chains).collect()
}

fn ints_of(v: &Value) -> Vec<i64> {
    match v {
        Value::List(l) => l
            .iter()
            .map(|x| x.as_int().expect("an Int element"))
            .collect(),
        other => panic!("expected a list of Int, got {other}"),
    }
}

fn list(xs: &[i64]) -> Value {
    Value::int_list(xs)
}

/// The answer of variant `v`, from the value at each position. `ws` is
/// used only by [`Variant::Heaviest`].
fn answer(v: Variant, xs: &[i64], ys: &[i64], ws: &[i64]) -> i64 {
    let n = xs.len();
    let below = |i: usize, j: usize| xs[i] < xs[j] && ys[i] < ys[j];
    match v {
        Variant::Count => {
            // The chains that end at j: j alone, and those that end below
            // it, extended by j.
            let mut f = vec![0_i64; n];
            for j in 0..n {
                let mut s = 1;
                for (i, &t) in f[..j].iter().enumerate() {
                    if below(i, j) {
                        s = (s + t) % MODULUS;
                    }
                }
                f[j] = s;
            }
            f.iter().fold(0, |a, &b| (a + b) % MODULUS)
        }
        Variant::Longest | Variant::Heaviest => {
            let mut f = vec![0_i64; n];
            for j in 0..n {
                let w = if v == Variant::Heaviest { ws[j] } else { 1 };
                let best = (0..j).filter(|&i| below(i, j)).map(|i| f[i]).max();
                f[j] = w + best.unwrap_or(0);
            }
            f.iter().copied().max().unwrap_or(0)
        }
        Variant::LongestCount => {
            // (length, count) of the longest chains that end at j.
            let mut f = vec![(0_i64, 0_i64); n];
            for j in 0..n {
                let mut best = (0, 1);
                for (i, &(l, c)) in f[..j].iter().enumerate() {
                    if below(i, j) {
                        if l > best.0 {
                            best = (l, c);
                        } else if l == best.0 {
                            best.1 = (best.1 + c) % MODULUS;
                        }
                    }
                }
                f[j] = (best.0 + 1, best.1);
            }
            let top = f.iter().map(|p| p.0).max().unwrap_or(0);
            f.iter()
                .filter(|p| p.0 == top)
                .fold(0, |a, p| (a + p.1) % MODULUS)
        }
    }
}

/// The kind of input seed `seed` gets: `seed mod 3`.
fn kind_of(seed: u64) -> u64 {
    seed % 3
}

/// One input of the gate. Each is there to turn away a slow way to compute
/// the answer that may be fast on the others. A test shows this only for
/// [`GateInput::XsSorted`], with a quicksort that picks the first element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateInput {
    /// The input of seed 0: both lists from `0..GATE_N`.
    Random,
    /// That input with `xs` sorted up: every earlier position is below in
    /// `xs`, so a program that looks at each of those positions is `n²`.
    /// The worst case of an unbalanced search tree keyed by `xs`.
    XsSorted,
    /// `xs` sorted up and `ys` sorted down: no chain has two positions, yet
    /// every earlier position is below in `xs`.
    Crossing,
    /// Both lists sorted up: chains of nearly every position. The worst
    /// case of a program that works chain length by chain length.
    BothSorted,
    /// Values from `0..=1`: the worst case of a quicksort that puts every
    /// tie on one side, and of a search tree that sends ties one way.
    FewValues,
    /// Both lists of [`GateInput::Random`] in organ-pipe order: up, then
    /// down, so the middle element is the largest. The worst case of a
    /// quicksort whose pivot is the middle, first, or last element.
    OrganPipe,
}

/// The order of the gate's inputs.
pub const GATE_INPUTS: [GateInput; 6] = [
    GateInput::Random,
    GateInput::XsSorted,
    GateInput::Crossing,
    GateInput::BothSorted,
    GateInput::FewValues,
    GateInput::OrganPipe,
];

/// The seed of the gate's input of few values, and of its weights, apart
/// from the generator's. Passed to [`TaskRng::new`] with [`GATE_N`].
const FEW_VALUES_SEED: u64 = 1_000;
const WEIGHTS_SEED: u64 = 1_001;

fn sorted(xs: &[i64], down: bool) -> Vec<i64> {
    let mut s = xs.to_vec();
    s.sort_unstable();
    if down {
        s.reverse();
    }
    s
}

fn organ_pipe(xs: &[i64]) -> Vec<i64> {
    let s = sorted(xs, false);
    let up = s.iter().step_by(2);
    let down = s.iter().skip(1).step_by(2).rev();
    up.chain(down).copied().collect()
}

/// A gate input of `family`. For [`Variant::Heaviest`] the weights are the
/// same on every input.
fn gate_input(family: &Chains, g: GateInput) -> Inputs {
    let random = generate_input(family, GATE_N, 0);
    let xs = ints_of(&random[0]);
    let ys = ints_of(&random[1]);
    let (xs, ys) = match g {
        GateInput::Random => (xs, ys),
        GateInput::XsSorted => (sorted(&xs, false), ys),
        GateInput::Crossing => (sorted(&xs, false), sorted(&ys, true)),
        GateInput::BothSorted => (sorted(&xs, false), sorted(&ys, false)),
        GateInput::FewValues => {
            let mut rng = TaskRng::new(GATE_N, FEW_VALUES_SEED);
            let mut draw = || (0..GATE_N).map(|_| rng.int_in(0, 1)).collect::<Vec<i64>>();
            (draw(), draw())
        }
        GateInput::OrganPipe => (organ_pipe(&xs), organ_pipe(&ys)),
    };
    let mut inputs = vec![list(&xs), list(&ys)];
    if family.0 == Variant::Heaviest {
        let mut rng = TaskRng::new(GATE_N, WEIGHTS_SEED);
        let top = GATE_N as i64 - 1;
        let ws: Vec<i64> = (0..GATE_N).map(|_| rng.int_in(0, top)).collect();
        inputs.push(list(&ws));
    }
    inputs
}

/// The text of the size and step limits for two lists.
macro_rules! limits {
    () => {
        " Both lists have the same length, at most 8192 elements, each between \
         0 and 8191. The program must finish within 150000000 evaluation \
         steps on lists of 8192 elements."
    };
}

/// The same, for three lists.
macro_rules! limits_3 {
    () => {
        " The three lists have the same length, at most 8192 elements, each \
         between 0 and 8191. The program must finish within 150000000 \
         evaluation steps on lists of 8192 elements."
    };
}

/// What a chain is, the same for every variant.
macro_rules! chain {
    () => {
        "A chain is a set of positions i1 < i2 < ... < ik (counting from 0) \
         along which both lists strictly increase: xs[i1] < xs[i2] < ... < \
         xs[ik] and ys[i1] < ys[i2] < ... < ys[ik]. A single position is a \
         chain; the empty set is not. "
    };
}

impl TaskFamily for Chains {
    fn name(&self) -> &str {
        match self.0 {
            Variant::Count => "chains/count",
            Variant::Longest => "chains/longest",
            Variant::Heaviest => "chains/heaviest",
            Variant::LongestCount => "chains/longest-count",
        }
    }

    fn signature(&self) -> Scheme {
        let s = match self.0 {
            Variant::Heaviest => "List Int -> List Int -> List Int -> Int",
            _ => "List Int -> List Int -> Int",
        };
        s.parse().expect("a scheme")
    }

    fn description(&self) -> &str {
        match self.0 {
            Variant::Count => concat!(
                chain!(),
                "Count the chains, and return the count mod 1000003.",
                limits!()
            ),
            Variant::Longest => concat!(
                chain!(),
                "Return the number of positions in the longest chain, or 0 if \
                 the lists are empty.",
                limits!()
            ),
            Variant::Heaviest => concat!(
                chain!(),
                "The weight of a chain is the sum of ws at its positions. \
                 Return the largest weight of a chain, or 0 if the lists are \
                 empty.",
                limits_3!()
            ),
            Variant::LongestCount => concat!(
                chain!(),
                "Count the chains that have the most positions, and return the \
                 count mod 1000003, or 0 if the lists are empty.",
                limits!()
            ),
        }
    }

    fn examples(&self) -> Vec<Example> {
        let pairs: [(&[i64], &[i64], &[i64]); 9] = [
            (&[], &[], &[]),
            (&[1, 2], &[1, 2], &[3, 4]),
            (&[1, 2], &[2, 1], &[3, 4]),
            (&[1, 1], &[1, 2], &[5, 1]),
            (&[0, 1, 2], &[2, 0, 1], &[1, 2, 3]),
            (&[1, 2, 3], &[1, 2, 3], &[1, 1, 1]),
            (&[0, 0, 1, 1], &[0, 1, 0, 1], &[1, 2, 3, 4]),
            (
                &[2, 5, 1, 4, 3, 6],
                &[1, 4, 2, 5, 3, 6],
                &[2, 7, 1, 8, 2, 8],
            ),
            (&[3, 0, 1, 2, 4], &[0, 1, 2, 3, 4], &[9, 1, 1, 1, 0]),
        ];
        pairs
            .iter()
            .map(|&(xs, ys, ws)| {
                let mut inputs = vec![list(xs), list(ys)];
                if self.0 == Variant::Heaviest {
                    inputs.push(list(ws));
                }
                Example::new(inputs, Value::Int(answer(self.0, xs, ys, ws)))
            })
            .collect()
    }

    /// The [`GateInput`]s in [`GATE_INPUTS`] order, each of [`GATE_N`]
    /// elements, with [`GATE_FUEL`] each.
    fn gate(&self) -> Option<Gate> {
        let inputs = GATE_INPUTS
            .iter()
            .map(|&g| {
                let i = gate_input(self, g);
                let out = self.oracle(&i);
                Example::new(i, out)
            })
            .collect();
        Some(Gate {
            inputs,
            fuel: GATE_FUEL,
        })
    }

    fn generate(&self, n: u64, rng: &mut TaskRng) -> Inputs {
        let kind = kind_of(rng.seed());
        let top = n as i64 - 1;
        let draw = |rng: &mut TaskRng, hi: i64| -> Vec<i64> {
            (0..n).map(|_| rng.int_in(0, hi)).collect()
        };
        let (xs, ys) = match kind {
            0 => (draw(rng, top), draw(rng, top)),
            1 => {
                let hi = (n / 8).max(1).min(n.saturating_sub(1)) as i64;
                (draw(rng, hi), draw(rng, hi))
            }
            _ => {
                let xs = draw(rng, top);
                let near = (n / 16).max(1) as i64;
                let ys = xs
                    .iter()
                    .map(|&x| (x + rng.int_in(-near, near)).clamp(0, top))
                    .collect();
                (xs, ys)
            }
        };
        let mut inputs = vec![list(&xs), list(&ys)];
        if self.0 == Variant::Heaviest {
            inputs.push(list(&draw(rng, top)));
        }
        inputs
    }

    fn oracle(&self, inputs: &[Value]) -> Value {
        let xs = ints_of(&inputs[0]);
        let ys = ints_of(&inputs[1]);
        let ws = inputs.get(2).map(ints_of).unwrap_or_default();
        Value::Int(answer(self.0, &xs, &ys, &ws))
    }

    fn sizes(&self) -> Vec<u64> {
        SIZES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answers from every set of positions, independent of the
    /// oracles.
    fn brute(v: Variant, xs: &[i64], ys: &[i64], ws: &[i64]) -> i64 {
        let n = xs.len();
        let chains: Vec<Vec<usize>> = (1_u32..1 << n)
            .map(|mask| {
                (0..n)
                    .filter(|&i| mask >> i & 1 == 1)
                    .collect::<Vec<usize>>()
            })
            .filter(|c| {
                c.windows(2)
                    .all(|w| xs[w[0]] < xs[w[1]] && ys[w[0]] < ys[w[1]])
            })
            .collect();
        match v {
            Variant::Count => chains.len() as i64 % MODULUS,
            Variant::Longest => chains.iter().map(|c| c.len() as i64).max().unwrap_or(0),
            Variant::Heaviest => chains
                .iter()
                .map(|c| c.iter().map(|&i| ws[i]).sum::<i64>())
                .max()
                .unwrap_or(0),
            Variant::LongestCount => {
                let top = chains.iter().map(Vec::len).max().unwrap_or(0);
                if top == 0 {
                    0
                } else {
                    chains.iter().filter(|c| c.len() == top).count() as i64 % MODULUS
                }
            }
        }
    }

    fn inputs(v: Variant, xs: &[i64], ys: &[i64], ws: &[i64]) -> Inputs {
        let mut i = vec![list(xs), list(ys)];
        if v == Variant::Heaviest {
            i.push(list(ws));
        }
        i
    }

    proptest::proptest! {
        #[test]
        fn oracles_agree_with_brute_force(
            pts in proptest::collection::vec((0_i64..5, 0_i64..5, 0_i64..20), 0..13),
        ) {
            let xs: Vec<i64> = pts.iter().map(|p| p.0).collect();
            let ys: Vec<i64> = pts.iter().map(|p| p.1).collect();
            let ws: Vec<i64> = pts.iter().map(|p| p.2).collect();
            for v in Variant::ALL {
                proptest::prop_assert_eq!(
                    Chains(v).oracle(&inputs(v, &xs, &ys, &ws)),
                    Value::Int(brute(v, &xs, &ys, &ws)),
                    "{:?} on {:?}, {:?}, {:?}", v, xs, ys, ws
                );
            }
        }
    }

    #[test]
    fn oracles_agree_with_brute_force_on_generated_inputs() {
        // Brute force enumerates every set of positions, so only at 16.
        for v in Variant::ALL {
            let f = Chains(v);
            for seed in 0..6 {
                let i = generate_input(&f, 16, seed);
                let ws = i.get(2).map(ints_of).unwrap_or_default();
                assert_eq!(
                    f.oracle(&i),
                    Value::Int(brute(v, &ints_of(&i[0]), &ints_of(&i[1]), &ws)),
                    "{v:?}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn the_examples_are_the_brute_force_answers() {
        for v in Variant::ALL {
            for e in Chains(v).examples() {
                let ws = e.inputs.get(2).map(ints_of).unwrap_or_default();
                let want = brute(v, &ints_of(&e.inputs[0]), &ints_of(&e.inputs[1]), &ws);
                assert_eq!(e.output, Value::Int(want), "{v:?} on {:?}", e.inputs);
            }
        }
    }

    #[test]
    fn the_gate_inputs_are_what_they_say() {
        for f in all() {
            let gate = f.gate().expect("a gate");
            assert_eq!(gate.fuel, GATE_FUEL);
            assert_eq!(gate.inputs.len(), GATE_INPUTS.len());
            let at = |g: GateInput| {
                let e = &gate.inputs[GATE_INPUTS.iter().position(|&x| x == g).unwrap()];
                (ints_of(&e.inputs[0]), ints_of(&e.inputs[1]), e.inputs.len())
            };
            for g in GATE_INPUTS {
                let (xs, ys, lists) = at(g);
                assert_eq!(lists, f.0.lists(), "{}", f.name());
                for l in [&xs, &ys] {
                    assert_eq!(l.len() as u64, GATE_N, "{} {g:?}", f.name());
                    assert!(
                        l.iter().all(|&x| (0..GATE_N as i64).contains(&x)),
                        "{} {g:?}",
                        f.name()
                    );
                }
            }
            let up = |l: &[i64]| l.windows(2).all(|w| w[0] <= w[1]);
            let down = |l: &[i64]| l.windows(2).all(|w| w[0] >= w[1]);
            let (xs, ys, _) = at(GateInput::XsSorted);
            assert!(up(&xs) && !up(&ys) && !down(&ys), "{}", f.name());
            let (xs, ys, _) = at(GateInput::Crossing);
            assert!(up(&xs) && down(&ys), "{}", f.name());
            let (xs, ys, _) = at(GateInput::BothSorted);
            assert!(up(&xs) && up(&ys), "{}", f.name());
            let (xs, ys, _) = at(GateInput::FewValues);
            for l in [&xs, &ys] {
                let set: std::collections::BTreeSet<i64> = l.iter().copied().collect();
                assert_eq!(set.len(), 2, "{}", f.name());
            }
            let (xs, ys, _) = at(GateInput::OrganPipe);
            for l in [&xs, &ys] {
                let top = l.len() / 2;
                assert_eq!(l[top], *l.iter().max().unwrap(), "{}", f.name());
                assert!(up(&l[..=top]) && down(&l[top..]), "{}", f.name());
            }
            for e in f.examples() {
                assert!(ints_of(&e.inputs[0]).len() <= 8, "{}", f.name());
            }
        }
    }

    #[test]
    fn the_gate_answers_are_not_trivial() {
        // The crossing input has no chain of two positions; the input with
        // both lists sorted has very long ones.
        let longest = Chains(Variant::Longest).gate().expect("a gate");
        let ans = |i: usize| longest.inputs[i].output.as_int().expect("an Int");
        let crossing = GATE_INPUTS
            .iter()
            .position(|&g| g == GateInput::Crossing)
            .unwrap();
        let both = GATE_INPUTS
            .iter()
            .position(|&g| g == GateInput::BothSorted)
            .unwrap();
        assert_eq!(ans(crossing), 1);
        assert!(ans(both) > 1000, "{}", ans(both));
        assert!(ans(0) > 10, "{}", ans(0));
    }

    #[test]
    fn the_description_states_the_size_and_step_limits() {
        for text in [limits!(), limits_3!()] {
            assert!(
                text.contains(&format!("at most {GATE_N} elements")),
                "{text}"
            );
            assert!(text.contains(&format!("within {GATE_FUEL} ")), "{text}");
            assert!(
                text.contains(&format!("lists of {GATE_N} elements")),
                "{text}"
            );
            assert!(
                text.contains(&format!("between 0 and {}.", GATE_N - 1)),
                "{text}"
            );
        }
        for f in all() {
            let limits = if f.0 == Variant::Heaviest {
                limits_3!()
            } else {
                limits!()
            };
            assert!(f.description().ends_with(limits), "{}", f.name());
            assert!(f.description().starts_with(chain!()), "{}", f.name());
        }
    }

    #[test]
    fn each_seed_keeps_its_kind_at_every_size() {
        for seed in 0..6 {
            for &n in SIZES {
                let i = generate_input(&Chains(Variant::Count), n, seed);
                let xs = ints_of(&i[0]);
                let ys = ints_of(&i[1]);
                let tie_range = xs.iter().chain(&ys).all(|&x| x <= (n / 8) as i64);
                let near = xs
                    .iter()
                    .zip(&ys)
                    .all(|(x, y)| (x - y).abs() <= (n / 16) as i64);
                let observed = match (tie_range, near) {
                    (true, _) => 1,
                    (false, true) => 2,
                    (false, false) => 0,
                };
                assert_eq!(observed, kind_of(seed), "seed {seed}, n = {n}");
                assert!(
                    xs.iter().chain(&ys).all(|&x| (0..n as i64).contains(&x)),
                    "seed {seed}, n = {n}"
                );
            }
        }
    }

    #[test]
    fn generated_inputs_have_ties_and_chains() {
        for v in Variant::ALL {
            let f = Chains(v);
            for seed in 0..3 {
                let i = generate_input(&f, 128, seed);
                let xs = ints_of(&i[0]);
                let mut distinct = xs.clone();
                distinct.sort_unstable();
                distinct.dedup();
                assert!(distinct.len() < xs.len(), "{v:?} seed {seed}: no ties");
                let longest = answer(Variant::Longest, &xs, &ints_of(&i[1]), &[]);
                assert!(longest > 2, "{v:?} seed {seed}: longest chain {longest}");
            }
        }
    }
}
