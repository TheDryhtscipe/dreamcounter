//! The task families of the corpus (spec §9, plan items P3.5 to P3.7).
//!
//! A family is a few variant [`TaskFamily`](crate::task::TaskFamily)
//! instances that share one idea, so that an abstraction found for one
//! variant pays off on the others. Each variant comes with two reference
//! solutions in dc: a naive one, which sets the fuel plan, and a good one of
//! a lower growth class.
//!
//! * [`collapse`]: rewriting a list at the leftmost redex until none is
//!   left.
//! * [`sorted_order`]: queries answered by sorting first.
//! * [`subsequence_count`]: counting the subsets of positions that keep a
//!   local rule.

pub mod collapse;
pub mod sorted_order;
pub mod subsequence_count;
