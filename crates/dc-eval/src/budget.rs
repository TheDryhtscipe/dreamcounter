//! The memory an evaluation may still allocate ([`crate::Limits::memory`]).
//!
//! Every buffer that one step of the machine can grow in proportion to the
//! data it holds (the continuation stack, an `unfold`'s output, `tree_fold`'s
//! work lists, a memo table, the scratch space of `eq`, of a memo key and of a
//! reachability walk) grows through [`Budget::ensure_vec`] or
//! [`Budget::ensure_map`]. They check that the larger buffer fits in the limit
//! before they allocate it, so a step cannot take an evaluation far above its
//! limit by growing one of them.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hash};
use std::mem::size_of;

/// An evaluation's memory limit, and its thread's live heap bytes when it
/// started.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Budget {
    limit: u64,
    base: i64,
}

/// The evaluation would go over its memory limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OverMemory;

/// The bytes of `n` values of type `T`.
pub(crate) fn bytes_of<T>(n: usize) -> u64 {
    u64::try_from(n.saturating_mul(size_of::<T>())).unwrap_or(u64::MAX)
}

impl Budget {
    /// No limit: every check passes and no buffer grows early.
    pub(crate) const UNLIMITED: Budget = Budget {
        limit: u64::MAX,
        base: 0,
    };

    /// A limit of `limit` bytes beyond what this thread holds now.
    pub(crate) fn new(limit: u64) -> Budget {
        if limit == u64::MAX {
            return Budget::UNLIMITED;
        }
        Budget {
            limit,
            base: crate::memory::live_bytes(),
        }
    }

    fn unlimited(&self) -> bool {
        self.limit == u64::MAX
    }

    /// The bytes held beyond the base.
    fn used(&self) -> u64 {
        u64::try_from(crate::memory::live_bytes().saturating_sub(self.base)).unwrap_or(0)
    }

    /// The bytes that can still be allocated.
    fn room(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }

    /// Whether the evaluation holds more than half its limit.
    pub(crate) fn past_half(&self) -> bool {
        !self.unlimited() && self.used() > self.limit / 2
    }

    /// Fails if the evaluation holds more than its limit, or would after
    /// allocating `extra` more bytes.
    pub(crate) fn check(&self, extra: u64) -> Result<(), OverMemory> {
        if !self.unlimited() && self.used().saturating_add(extra) > self.limit {
            return Err(OverMemory);
        }
        Ok(())
    }

    /// Makes room in `v` for `extra` more elements without a reallocation that
    /// the limit has not allowed. A full buffer doubles if the limit allows
    /// it, and otherwise grows only as far as the limit allows; if that is
    /// fewer than `extra` elements, this fails.
    pub(crate) fn ensure_vec<T>(&self, v: &mut Vec<T>, extra: usize) -> Result<(), OverMemory> {
        if self.unlimited() || v.capacity() - v.len() >= extra || size_of::<T>() == 0 {
            return Ok(());
        }
        let need = extra - (v.capacity() - v.len());
        // The count sees a reallocation as its change in size. The system
        // allocator may copy the buffer, holding the old one too for a
        // moment, or (with glibc, above its mmap threshold) remap its pages;
        // either way only resident memory sees it, not the count.
        let room = usize::try_from(self.room() / bytes_of::<T>(1)).unwrap_or(usize::MAX);
        let more = v.capacity().max(4).max(need).min(room);
        if more < need {
            return Err(OverMemory);
        }
        // `reserve_exact` counts from the length, not the capacity.
        v.reserve_exact(v.capacity() - v.len() + more);
        debug_assert!(v.capacity() - v.len() >= extra);
        Ok(())
    }

    /// Makes room in `m` for one more entry, checking first that the larger
    /// table it may move to fits in the limit.
    pub(crate) fn ensure_map<K: Eq + Hash, V, S: BuildHasher>(
        &self,
        m: &mut HashMap<K, V, S>,
    ) -> Result<(), OverMemory> {
        if self.unlimited() || m.len() < m.capacity() {
            return Ok(());
        }
        let more = m.capacity().max(4);
        self.check(table_bytes::<(K, V)>(m.len() + more))?;
        m.reserve(more);
        Ok(())
    }

    /// As [`Budget::ensure_map`], for a set.
    pub(crate) fn ensure_set<T: Eq + Hash, S: BuildHasher>(
        &self,
        s: &mut HashSet<T, S>,
    ) -> Result<(), OverMemory> {
        if self.unlimited() || s.len() < s.capacity() {
            return Ok(());
        }
        let more = s.capacity().max(4);
        self.check(table_bytes::<T>(s.len() + more))?;
        s.reserve(more);
        Ok(())
    }
}

/// An upper bound on the bytes of a hash table that holds `items` entries of
/// type `T`: the standard library's table has a power of two of buckets, at
/// least 8/7 of its items, with one control byte each, plus a group of
/// control bytes.
pub(crate) fn table_bytes<T>(items: usize) -> u64 {
    let buckets = items
        .saturating_mul(8)
        .div_ceil(7)
        .max(4)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    bytes_of::<T>(buckets)
        .saturating_add(u64::try_from(buckets).unwrap_or(u64::MAX))
        .saturating_add(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_bytes_bounds_what_a_growing_map_allocates() {
        let mut m: HashMap<u64, u64> = HashMap::new();
        for i in 0..100_000u64 {
            if m.len() == m.capacity() {
                let before = crate::memory::live_bytes();
                let more = m.capacity().max(4);
                m.reserve(more);
                let after = crate::memory::live_bytes();
                let grown = u64::try_from(after - before).unwrap_or(0);
                assert!(
                    grown <= table_bytes::<(u64, u64)>(m.len() + more),
                    "len {} grew by {grown}",
                    m.len()
                );
            }
            m.insert(i, i);
        }
    }

    #[test]
    fn ensure_vec_grows_only_within_the_limit() {
        let b = Budget::new(1 << 20);
        let mut v: Vec<u64> = Vec::new();
        let mut stopped = false;
        for i in 0..1_000_000u64 {
            if b.ensure_vec(&mut v, 1).is_err() {
                stopped = true;
                break;
            }
            v.push(i);
        }
        assert!(stopped);
        // The last growth filled what the limit left, not more.
        // The buffer grew to fill the limit, not only to half of it.
        assert!(v.capacity() * 8 > 3 << 18, "capacity {}", v.capacity());
        assert!(v.capacity() * 8 <= 1 << 20, "capacity {}", v.capacity());
    }
}
