//! A seeded permutation of [0, n), evaluated one position at a time: an epoch order over tens
//! of millions of images without an O(n) table per epoch (or per batch).
//!
//! A balanced Feistel network on the smallest even number of bits covering n, with a keyed
//! SplitMix64 round function, is a bijection of [0, 4^b); positions it maps outside [0, n) are
//! mapped again until they land inside (cycle walking, under 4 passes on average).

const ROUNDS: u64 = 6;

/// SplitMix64's finalizer: a fast, well-mixing 64-bit hash.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The image at `position` of the permutation of [0, n) keyed by `seed`.
pub fn permute(n: u64, seed: u64, position: u64) -> u64 {
    assert!(position < n, "position {position} of {n}");
    // half the bits of the domain: 4^half >= n
    let mut half = 1;
    while half < 32 && (1u64 << (2 * half)) < n {
        half += 1;
    }
    let mask = (1u64 << half) - 1;
    let key = mix(seed);

    let mut x = position;
    loop {
        let mut left = x >> half;
        let mut right = x & mask;
        for round in 0..ROUNDS {
            let f = mix(key ^ mix(right ^ (round << 56))) & mask;
            (left, right) = (right, left ^ f);
        }
        x = (left << half) | right;
        if x < n {
            return x;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_a_permutation() {
        for n in [1u64, 2, 3, 17, 1000, 4097] {
            for seed in 0..3 {
                let mut seen: Vec<u64> = (0..n).map(|p| permute(n, seed, p)).collect();
                seen.sort_unstable();
                assert!(seen.iter().copied().eq(0..n), "n {n} seed {seed}");
            }
        }
        assert_ne!(
            (0..100).map(|p| permute(100, 1, p)).collect::<Vec<_>>(),
            (0..100).map(|p| permute(100, 2, p)).collect::<Vec<_>>()
        );
    }
}
