//! Time per native stage, summed over threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
pub enum Stage {
    Read,
    Decode,
    Convert,
    Resize,
}

static STAGE_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
pub static EMITTED: AtomicU64 = AtomicU64::new(0);
pub static DECODED: AtomicU64 = AtomicU64::new(0);

pub fn timed<T>(stage: Stage, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = f();
    STAGE_NS[stage as usize].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

/// Seconds per stage (read, decode, convert, resize; summed over threads) and frames emitted /
/// decoded since the last reset.
pub fn profile(reset: bool) -> ([f64; 4], u64, u64) {
    let take = |a: &AtomicU64| {
        if reset {
            a.swap(0, Ordering::Relaxed)
        } else {
            a.load(Ordering::Relaxed)
        }
    };
    let seconds = STAGE_NS.each_ref().map(|a| take(a) as f64 / 1e9);
    (seconds, take(&EMITTED), take(&DECODED))
}
