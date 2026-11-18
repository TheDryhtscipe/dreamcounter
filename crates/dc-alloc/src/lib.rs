//! A global allocator that counts each thread's live heap bytes (bead
//! dreamcounter-a3db93).
//!
//! The evaluator (`dc-eval`) bounds the memory of one run with this count: a
//! run is on one thread, so the thread's live bytes minus its live bytes when
//! the run started are what the run holds. [`live_bytes`] reads the count and
//! [`peak_bytes`] the highest it has been since [`reset_peak`].
//!
//! Every binary that links this crate uses [`CountingAlloc`] as its global
//! allocator: the crate declares it with `#[global_allocator]`. A second
//! global allocator anywhere in the same binary is a build error, so the count
//! cannot silently stop working.
//!
//! The count is of the bytes asked for, as `Layout::size`, not of what the
//! system allocator rounds them up to or keeps after a free, so the resident
//! memory of the process can be somewhat above it. Memory that one thread
//! allocates and another frees moves the count of both: the count of a thread
//! that frees more than it allocated goes below zero.
//!
//! ```
//! let before = dc_alloc::live_bytes();
//! let v: Vec<u8> = Vec::with_capacity(1 << 20);
//! assert!(dc_alloc::live_bytes() - before >= 1 << 20);
//! drop(v);
//! assert_eq!(dc_alloc::live_bytes(), before);
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // `const` initialisers with no destructor: reading or writing these
    // never allocates, so the allocator can use them, even while the thread
    // is being torn down.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

/// The global allocator: the system allocator, plus a count of each thread's
/// live bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct CountingAlloc;

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Adds `n` bytes (negative for a free) to this thread's count.
fn add(n: i64) {
    // `try_with` fails only after the thread's locals are destroyed; the
    // count of that thread no longer matters then.
    let _ = LIVE.try_with(|live| {
        let now = live.get().wrapping_add(n);
        live.set(now);
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

/// A `Layout` size as a signed count. Sizes are at most `isize::MAX`, so the
/// conversion is exact.
fn bytes(size: usize) -> i64 {
    size as i64
}

// SAFETY: every method forwards to `System` with the arguments it was given
// and returns what `System` returned, so `System`'s guarantees carry over
// unchanged. The only other work is `add`, which updates two thread-local
// `Cell`s: it does not allocate, does not unwind, and does not touch the
// memory being allocated or freed.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's guarantees for `layout` are `System`'s.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(bytes(layout.size()));
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as for `alloc`.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            add(bytes(layout.size()));
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by this allocator, so by `System`,
        // with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        add(-bytes(layout.size()));
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as for `dealloc`, and the caller's guarantees for
        // `new_size` are `System`'s.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            add(bytes(new_size) - bytes(layout.size()));
        }
        p
    }
}

/// The bytes this thread has allocated and not freed, less any it freed that
/// other threads allocated.
pub fn live_bytes() -> i64 {
    LIVE.with(Cell::get)
}

/// The highest [`live_bytes`] has been on this thread since the last
/// [`reset_peak`], or since the thread started.
pub fn peak_bytes() -> i64 {
    PEAK.with(Cell::get)
}

/// Sets this thread's peak to its live bytes now.
pub fn reset_peak() {
    let now = live_bytes();
    PEAK.with(|p| p.set(now));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vec_moves_the_count_by_its_capacity_and_back() {
        let before = live_bytes();
        let mut v: Vec<u64> = Vec::with_capacity(1000);
        assert_eq!(live_bytes() - before, 8000);
        v.reserve_exact(2000);
        assert_eq!(live_bytes() - before, 8 * v.capacity() as i64);
        drop(v);
        assert_eq!(live_bytes(), before);
    }

    #[test]
    fn the_peak_keeps_the_highest_count_until_reset() {
        reset_peak();
        let before = live_bytes();
        drop(vec![0u8; 1 << 20]);
        assert!(peak_bytes() - before >= 1 << 20);
        assert_eq!(live_bytes(), before);
        reset_peak();
        assert_eq!(peak_bytes(), live_bytes());
    }

    #[test]
    fn another_threads_allocations_do_not_count_here() {
        let before = live_bytes();
        let held = std::thread::spawn(|| {
            let base = live_bytes();
            let v = vec![0u8; 1 << 20];
            assert_eq!(live_bytes() - base, 1 << 20);
            std::hint::black_box(v.len())
        })
        .join()
        .unwrap();
        assert_eq!(held, 1 << 20);
        // Spawning and joining allocate a little here; the 1 MiB is not.
        assert!((live_bytes() - before).abs() < 1 << 16);
    }
}
