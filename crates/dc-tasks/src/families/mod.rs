//! The task families of the corpus (spec §9, plan items P3.5 to P3.7).
//!
//! A family is a few variant [`TaskFamily`]
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
//!
//! [`development`] lists every variant of these families with its reference
//! solutions in one place, for checks that run over the whole corpus. A new
//! family or reference must be added to it by hand.

pub mod collapse;
pub mod sorted_order;
pub mod subsequence_count;

use crate::task::TaskFamily;

/// One variant of the development corpus and its reference solutions.
pub struct CorpusVariant {
    /// The variant as a family.
    pub family: Box<dyn TaskFamily>,
    /// The dc source that defines the references.
    pub source: &'static str,
    /// The references, naive first and good (lowest growth) last. Every one
    /// is a correct solution of `family`.
    pub references: Vec<Reference>,
}

impl std::fmt::Debug for CorpusVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CorpusVariant")
            .field("family", &self.family.name())
            .field("references", &self.references)
            .finish_non_exhaustive()
    }
}

/// A reference solution and the sizes it is run at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reference {
    /// Its definition in the variant's source.
    pub entry: &'static str,
    /// The largest sweep size it is run at. It is [`u64::MAX`], the whole
    /// sweep, except for a reference too slow to finish at the larger
    /// sizes under the harness's fuel cap, such as a `2ⁿ` one.
    pub max_n: u64,
}

impl Reference {
    /// A reference run over the whole sweep.
    pub const fn whole(entry: &'static str) -> Reference {
        Reference {
            entry,
            max_n: u64::MAX,
        }
    }
}

/// The largest size the `2ⁿ` subsequence-count references are run at: they
/// take more than `2ⁿ` steps, so at 32 they need more than 4·10⁹, far past
/// the default fuel cap.
pub const EXPONENTIAL_MAX_N: u64 = 16;

/// Every variant of every development family, with its references, in
/// family order: collapse, sorted order, subsequence count. The held-out
/// family is not here (decision D14).
pub fn development() -> Vec<CorpusVariant> {
    let mut corpus = Vec::new();
    for v in collapse::Variant::ALL {
        corpus.push(CorpusVariant {
            family: Box::new(collapse::Collapse(v)),
            source: collapse::SOURCE,
            references: vec![Reference::whole(v.naive()), Reference::whole(v.good())],
        });
    }
    for v in sorted_order::Variant::ALL {
        corpus.push(CorpusVariant {
            family: Box::new(sorted_order::SortedOrder(v)),
            source: sorted_order::SOURCE,
            references: vec![Reference::whole(v.naive()), Reference::whole(v.good())],
        });
    }
    for v in subsequence_count::Variant::ALL {
        corpus.push(CorpusVariant {
            family: Box::new(subsequence_count::SubsequenceCount(v)),
            source: subsequence_count::SOURCE,
            references: vec![
                Reference {
                    entry: v.naive(),
                    max_n: EXPONENTIAL_MAX_N,
                },
                Reference::whole(v.quadratic()),
                Reference::whole(v.good()),
            ],
        });
    }
    corpus
}
