//! The seeded random source for task input generators (spec §9).
//!
//! Every generated input is a pure function of the family, the size `n` and
//! the seed. [`TaskRng::new`] derives the generator's stream from `(n, seed)`
//! alone, so the same pair gives the same input on every run, platform and
//! thread.
//!
//! # Choices that hold for a seed across the sweep
//!
//! A generator that draws a kind of input (many ties, say, or none) from the
//! `(n, seed)` stream gives each seed a different kind at each size. A
//! program whose cost depends on the kind then has step counts that jump
//! between sizes, and estimator C cannot see its growth through them
//! (bead dreamcounter-0ea5b7). Draw such choices from the seed instead:
//! [`TaskRng::seed`] for a choice spread evenly over the seeds (the kind
//! `seed % 3`, say), and [`TaskRng::seed_stream`] for a random one. Both are
//! the same at every size, so each seed's step counts follow one kind of
//! input across the sweep.
//!
//! # Why the stream is stable across versions
//!
//! The stream is ChaCha8 (`rand_chacha`, pinned to an exact version in the
//! workspace manifest). ChaCha's keystream is fixed by its specification, and
//! the seed is built here byte by byte, not through a seed-expansion helper.
//! The sampling methods ([`TaskRng::below`], [`TaskRng::int_in`]) are
//! implemented in this module rather than taken from `rand`'s distributions,
//! whose algorithms have changed between `rand` releases. The golden-value
//! test in this module fails if any of that changes.

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Fills the last 16 bytes of the ChaCha key. Changing it changes every
/// generated input, so it carries a version.
const DOMAIN: &[u8; 16] = b"dc-tasks/input/1";

/// As [`DOMAIN`], for [`TaskRng::seed_stream`].
const SEED_DOMAIN: &[u8; 16] = b"dc-tasks/seeds/1";

/// A deterministic random source for one generated input.
#[derive(Debug, Clone)]
pub struct TaskRng {
    inner: ChaCha8Rng,
    seed: u64,
}

impl TaskRng {
    /// The stream for size `n` and seed `seed`.
    pub fn new(n: u64, seed: u64) -> TaskRng {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&n.to_le_bytes());
        key[8..16].copy_from_slice(&seed.to_le_bytes());
        key[16..].copy_from_slice(DOMAIN);
        TaskRng {
            inner: ChaCha8Rng::from_seed(key),
            seed,
        }
    }

    /// The seed this stream was made for. The same at every size, so a
    /// choice made from it holds for the seed across the sweep.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// A fresh stream derived from the seed alone: the same at every size,
    /// and apart from every `(n, seed)` stream. For random choices that
    /// should hold for the seed across the sweep. Each call starts the
    /// stream again from its beginning.
    pub fn seed_stream(&self) -> TaskRng {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&self.seed.to_le_bytes());
        key[16..].copy_from_slice(SEED_DOMAIN);
        TaskRng {
            inner: ChaCha8Rng::from_seed(key),
            seed: self.seed,
        }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// A uniform integer in `0..bound`, without modulo bias.
    ///
    /// # Panics
    ///
    /// If `bound` is 0.
    pub fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "TaskRng::below: bound must be positive");
        // Values below `threshold` would make `x % bound` favour small results;
        // reject them. `threshold = 2^64 mod bound`.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next_u64();
            if x >= threshold {
                return x % bound;
            }
        }
    }

    /// A uniform integer in `lo..=hi`.
    ///
    /// # Panics
    ///
    /// If `lo > hi`.
    pub fn int_in(&mut self, lo: i64, hi: i64) -> i64 {
        assert!(lo <= hi, "TaskRng::int_in: empty range {lo}..={hi}");
        let span = (i128::from(hi) - i128::from(lo)) as u128 + 1;
        let offset = match u64::try_from(span) {
            Ok(span) => self.below(span),
            // The whole i64 range: every 64-bit pattern is a value.
            Err(_) => self.next_u64(),
        };
        (i128::from(lo) + i128::from(offset)) as i64
    }

    /// A fair coin.
    pub fn bool(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_size_and_seed_give_the_same_stream() {
        let mut a = TaskRng::new(64, 3);
        let mut b = TaskRng::new(64, 3);
        let xs: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_eq!(xs, ys);
    }

    #[test]
    fn different_sizes_or_seeds_give_different_streams() {
        let first = |n, seed| TaskRng::new(n, seed).next_u64();
        assert_ne!(first(64, 3), first(64, 4));
        assert_ne!(first(64, 3), first(128, 3));
        // n and seed are not interchangeable.
        assert_ne!(first(3, 64), first(64, 3));
    }

    /// Pins the stream. If this fails, every generated task input has changed:
    /// a `rand_chacha` upgrade or a change to the seeding or sampling here.
    /// Recorded results then no longer reproduce. The values were checked
    /// against an independent ChaCha8 block function written from RFC 8439
    /// (8 rounds, 64-bit block counter, zero nonce, words taken low word first).
    #[test]
    fn golden_values() {
        let mut r = TaskRng::new(16, 0);
        let raw: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
        assert_eq!(raw, GOLDEN_RAW, "raw ChaCha8 stream for (n=16, seed=0)");
        let mut r = TaskRng::new(1024, 4);
        let ints: Vec<i64> = (0..6).map(|_| r.int_in(-1000, 1000)).collect();
        assert_eq!(
            ints, GOLDEN_INTS,
            "int_in(-1000, 1000) for (n=1024, seed=4)"
        );
    }

    #[test]
    fn seed_stream_depends_on_the_seed_alone() {
        let first = |n, seed| TaskRng::new(n, seed).seed_stream().next_u64();
        assert_eq!(first(16, 3), first(1024, 3));
        assert_ne!(first(16, 3), first(16, 4));
        assert_eq!(TaskRng::new(16, 3).seed(), 3);
        assert_eq!(TaskRng::new(16, 3).seed_stream().seed(), 3);
        // Apart from the input streams, including (n = 0, seed).
        for n in [0, 16, 1024] {
            assert_ne!(first(n, 3), TaskRng::new(n, 3).next_u64());
        }
        // Drawing from the input stream does not move the seed stream.
        let mut r = TaskRng::new(16, 3);
        r.next_u64();
        assert_eq!(r.seed_stream().next_u64(), first(16, 3));
    }

    /// Pins the seed stream, as [`golden_values`] pins the input streams.
    /// Recorded from this implementation; the key layout is the input
    /// stream's with the seed in the first eight bytes, zeros in the next
    /// eight and [`SEED_DOMAIN`] after.
    #[test]
    fn seed_stream_golden_values() {
        let raw: Vec<u64> = {
            let mut r = TaskRng::new(512, 2).seed_stream();
            (0..3).map(|_| r.next_u64()).collect()
        };
        assert_eq!(raw, GOLDEN_SEED_RAW, "seed stream for seed 2");
    }

    const GOLDEN_SEED_RAW: [u64; 3] = [
        6030082216158595254,
        2466040114500341188,
        10803117707397758781,
    ];

    const GOLDEN_RAW: [u64; 3] = [
        16912961933745333717,
        7987587475100509616,
        9604533920838116691,
    ];
    const GOLDEN_INTS: [i64; 6] = [-981, -497, 799, 921, 83, -657];

    #[test]
    fn below_stays_in_range_and_hits_every_value() {
        let mut r = TaskRng::new(7, 7);
        let mut seen = [0u32; 7];
        for _ in 0..7000 {
            let x = r.below(7);
            assert!(x < 7);
            seen[x as usize] += 1;
        }
        // Each bucket expects 1000; a fair source lands well inside 800..1200.
        assert!(seen.iter().all(|&c| (800..1200).contains(&c)), "{seen:?}");
    }

    #[test]
    fn below_one_is_always_zero() {
        let mut r = TaskRng::new(1, 1);
        assert!((0..100).all(|_| r.below(1) == 0));
    }

    #[test]
    fn int_in_respects_bounds_including_the_full_range() {
        let mut r = TaskRng::new(9, 9);
        for _ in 0..1000 {
            let x = r.int_in(-3, 3);
            assert!((-3..=3).contains(&x));
        }
        assert!((0..100).all(|_| r.int_in(5, 5) == 5));
        // The full range must not panic or loop; both signs should appear.
        let full: Vec<i64> = (0..64).map(|_| r.int_in(i64::MIN, i64::MAX)).collect();
        assert!(full.iter().any(|&x| x < 0) && full.iter().any(|&x| x > 0));
        let top: Vec<i64> = (0..100).map(|_| r.int_in(i64::MAX - 1, i64::MAX)).collect();
        assert!(top.iter().all(|&x| x >= i64::MAX - 1));
        assert!(top.contains(&i64::MAX) && top.contains(&(i64::MAX - 1)));
    }

    #[test]
    fn bool_gives_both_values() {
        let mut r = TaskRng::new(2, 2);
        let trues = (0..1000).filter(|_| r.bool()).count();
        assert!((400..600).contains(&trues), "{trues}");
    }
}
