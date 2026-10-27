//! The cost model: every step the machine charges, in one table (spec §5 to §7).
//!
//! A **step** is the unit of cost. The machine charges only through the constants
//! and functions in this module, so this table is the whole cost model.
//!
//! | Event | Steps | Notes |
//! |---|---|---|
//! | variable lookup, literal, `lam` (closure creation), `let` binding | 0 | spec §7 |
//! | β step: applying a closure to one argument | 1 | [`BETA`]; `f a b` with a two-lambda `f` is 2 |
//! | evaluating a primitive of arity ≥ 1 to a function value | 0 | |
//! | applying a primitive to fewer arguments than its arity | 0 | [`PARTIAL`] |
//! | evaluating a nullary primitive: `nil`, `none`, `leaf` | 1 | [`PRIM`] |
//! | `add sub mul div mod int_eq lt le not pair fst snd some cons length node` | 1 | [`PRIM`], charged when saturated |
//! | `eq a b` | value-node pairs compared | [`eq`]; see below |
//! | `option_case`, `list_case`, `tree_case` | 1 + calls | the branch call is β steps; the empty branch is called with `unit` |
//! | `if`, `and`, `or` | 1 per decision | [`DECISION`]; `and`/`or` decide once, on their left operand |
//! | `foldl f z xs`, `foldr f z xs` | 1 per element + calls | [`ITERATION`]; empty list: 0 |
//! | `loop n z f` | 1 per iteration + calls | `n ≤ 0`: 0 |
//! | `unfold g s` | 1 per produced element + calls | the final call that returns `none` costs only its own steps |
//! | `array_tabulate n f` | max(n, 0) + calls | 1 per element, charged before each call of `f` |
//! | `array_get`, `array_size` | 1 | |
//! | `array_set a i v` | 1 + ⌈log₂(n+1)⌉, n = size of `a` | [`array_set`] |
//! | `array_of_list`, `array_to_list` | n + 1 | [`array_convert`] |
//! | `tree_fold z f t` | 1 per internal node + calls | `leaf`s cost 0 |
//! | `bfix n F x`, and each call of its `self` | 1 per call + calls inside `F` | [`ITERATION`]; the fuel check is free |
//! | `memorec n F x`, and each call of its `self` | hit: 1 + key size; miss: 1 + key size + calls inside `F` | [`memo_call`] |
//!
//! Readings of the spec's cost table that the spec leaves open. Each is the reading
//! that matches what the machine does:
//!
//! - **Charge timing.** A primitive's cost is charged when it is applied to its last
//!   argument, before it computes its result. A per-element or per-call cost of a
//!   combinator is charged just before the call of the function argument for that
//!   element. `tree_fold` charges a node when it combines the node's two subtree
//!   results, so in post-order.
//! - **Failing primitives are charged.** A primitive that raises an error (`div` or
//!   `mod` by zero, overflow, an array index out of bounds) is charged its full cost
//!   first: the step count of the error includes the failing application. `eq` that
//!   reaches a function is charged the node pairs compared up to and including the
//!   pair with the function.
//! - **Failing recursive calls are not charged.** A `bfix` call with fuel ≤ 0 and a
//!   `memorec` miss with no misses left raise `FuelExhausted` without charging, since
//!   the call does not happen and the fuel check is free (spec §6). An invalid
//!   `memorec` key raises `InvalidMemoKey` without charging.
//! - **The first `bfix`/`memorec` call is a call.** `bfix n F x` itself is charged 1
//!   like every recursive call, and needs `n ≥ 1`. So `bfix (length xs + 1)` allows
//!   exactly the `length xs + 1` nested calls of a list recursion. `memorec n F x`
//!   is a memo call on `x`, and with an empty table it is a miss.
//! - **`eq` node count.** Scalars, pairs, options, arrays and each list cell, `nil`,
//!   tree node and `leaf` count one node each. The comparison is pre-order and left
//!   to right, and stops at the first difference. Arrays of different sizes differ at
//!   their own node.
//! - **Timeout steps.** The machine stops with `Timeout(s)` at the first charge that
//!   takes the count `s` above the fuel. So `s > fuel`, and `s = fuel + 1` when that
//!   charge is a single step. A program that finishes in `s` steps finishes at every
//!   fuel `≥ s` with the same outcome (spec §7, invariant 3).
//! - **Applying a program to inputs.** [`crate::apply`] evaluates the program term
//!   (free for a `lam`) and applies it to each input in turn. Each application of a
//!   closure is one β step; a primitive program is charged as a primitive.

/// One β-reduction: applying a closure to one argument.
pub const BETA: u64 = 1;

/// A primitive applied to fewer arguments than its arity.
pub const PARTIAL: u64 = 0;

/// A saturated primitive with constant cost, and a nullary primitive.
pub const PRIM: u64 = 1;

/// One `if`, `and` or `or` decision.
pub const DECISION: u64 = 1;

/// One combinator iteration: a fold or loop element, an `unfold` output, a
/// `tabulate` element, a `tree_fold` node, or a `bfix` call.
pub const ITERATION: u64 = 1;

/// `eq` that compared `nodes` value-node pairs.
pub fn eq(nodes: u64) -> u64 {
    nodes
}

/// `array_set` on an array of size `n`: 1 + ⌈log₂(n+1)⌉.
pub fn array_set(n: u64) -> u64 {
    // ⌈log₂(n+1)⌉ is the number of bits needed to write n.
    1 + u64::from(u64::BITS - n.leading_zeros())
}

/// `array_of_list` or `array_to_list` on `n` elements.
pub fn array_convert(n: u64) -> u64 {
    n.saturating_add(1)
}

/// A `memorec` call with a key of `key_size` value nodes, hit or miss.
pub fn memo_call(key_size: u64) -> u64 {
    1 + key_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_set_is_one_plus_ceil_log2_of_size_plus_one() {
        // n: 0 1 2 3 4 7 8; ceil(log2(n+1)): 0 1 2 2 3 3 4.
        let got: Vec<u64> = [0, 1, 2, 3, 4, 7, 8].map(array_set).to_vec();
        assert_eq!(got, vec![1, 2, 3, 3, 4, 4, 5]);
        assert_eq!(array_set(u64::MAX), 65);
    }
}
