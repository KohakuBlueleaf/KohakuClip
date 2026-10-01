//! Explicit x86-64 SIMD kernels for the resize, picked at runtime, each bit-exact with the plain
//! Rust version in `resize.rs` (checked by the tests below):
//!
//! - `transpose`: 16 x 16 byte blocks with SSE2 unpacks (baseline x86-64).
//! - `weighted_rows`: the Q14 weighted sum of 2-6 rows. Pairs of rows are interleaved as 16-bit
//!   values and multiplied with `madd` by a (w0, w1) weight pair, 16 pixels per step (AVX2) or
//!   32 with the multiply-add fused (AVX-512 VNNI, `vpdpwssd`).
//!
//! Other targets use the plain versions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Which `weighted_rows` kernel this CPU runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Plain,
    Avx2,
    Avx512,
}

pub fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        // KOHAKUCLIP_SIMD=plain|avx2 caps the level (for measurements)
        let cap = std::env::var("KOHAKUCLIP_SIMD").unwrap_or_default();
        if cap == "plain" {
            return Level::Plain;
        }
        #[cfg(target_arch = "x86_64")]
        {
            let avx512 =
                is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni");
            if avx512 && cap != "avx2" {
                return Level::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Level::Avx2;
            }
        }
        Level::Plain
    })
}

// ------------------------------------------------------------------ transpose
/// dst[x][y] = src[y][x] for an (h, w) matrix: 16 x 16 SIMD blocks, plain edges.
pub fn transpose(src: &[u8], h: usize, w: usize, dst: &mut [u8]) {
    assert!(src.len() >= h * w && dst.len() >= h * w);
    #[cfg(target_arch = "x86_64")]
    if level() != Level::Plain {
        let (hb, wb) = (h / 16 * 16, w / 16 * 16);
        for y in (0..hb).step_by(16) {
            for x in (0..wb).step_by(16) {
                // SAFETY: the block [y, y + 16) x [x, x + 16) lies inside both matrices
                unsafe { transpose16(src, w, y, x, dst, h) };
            }
        }
        transpose_plain(src, h, w, dst, 0..h, wb..w);
        transpose_plain(src, h, w, dst, hb..h, 0..wb);
        return;
    }
    transpose_blocked(src, h, w, dst);
}

/// The plain transpose in 32 x 32 blocks.
fn transpose_blocked(src: &[u8], h: usize, w: usize, dst: &mut [u8]) {
    for yb in (0..h).step_by(32) {
        for xb in (0..w).step_by(32) {
            transpose_plain(src, h, w, dst, yb..(yb + 32).min(h), xb..(xb + 32).min(w));
        }
    }
}

fn transpose_plain(
    src: &[u8],
    h: usize,
    w: usize,
    dst: &mut [u8],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) {
    assert!(src.len() >= h * w && dst.len() >= h * w);
    assert!(rows.end <= h && cols.end <= w);
    for x in cols {
        for y in rows.clone() {
            // SAFETY: x < w and y < h (asserted above); unchecked indexing lets it vectorize
            unsafe { *dst.get_unchecked_mut(x * h + y) = *src.get_unchecked(y * w + x) };
        }
    }
}

/// Transpose the 16 x 16 block at (y, x) of `src` (row length `w`) into `dst` (row length `h`).
///
/// # Safety
/// The block must lie inside both matrices.
#[cfg(target_arch = "x86_64")]
unsafe fn transpose16(src: &[u8], w: usize, y: usize, x: usize, dst: &mut [u8], h: usize) {
    // SAFETY: SSE2 is baseline on x86-64; the caller keeps every load and store in bounds
    unsafe {
        let mut r = [_mm_setzero_si128(); 16];
        for (i, row) in r.iter_mut().enumerate() {
            *row = _mm_loadu_si128(src.as_ptr().add((y + i) * w + x) as *const __m128i);
        }
        // four rounds of unpacks: 8-, 16-, 32-, 64-bit interleaves
        let mut a = [_mm_setzero_si128(); 16];
        for i in 0..8 {
            a[2 * i] = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
            a[2 * i + 1] = _mm_unpackhi_epi8(r[2 * i], r[2 * i + 1]);
        }
        let mut b = [_mm_setzero_si128(); 16];
        for i in 0..4 {
            for k in 0..2 {
                let (p, q) = (a[4 * i + k], a[4 * i + 2 + k]);
                b[4 * i + 2 * k] = _mm_unpacklo_epi16(p, q);
                b[4 * i + 2 * k + 1] = _mm_unpackhi_epi16(p, q);
            }
        }
        let mut c = [_mm_setzero_si128(); 16];
        for i in 0..2 {
            for k in 0..4 {
                let (p, q) = (b[8 * i + k], b[8 * i + 4 + k]);
                c[8 * i + 2 * k] = _mm_unpacklo_epi32(p, q);
                c[8 * i + 2 * k + 1] = _mm_unpackhi_epi32(p, q);
            }
        }
        for k in 0..8 {
            let (p, q) = (c[k], c[8 + k]);
            let lo = _mm_unpacklo_epi64(p, q);
            let hi = _mm_unpackhi_epi64(p, q);
            _mm_storeu_si128(
                dst.as_mut_ptr().add((x + 2 * k) * h + y) as *mut __m128i,
                lo,
            );
            _mm_storeu_si128(
                dst.as_mut_ptr().add((x + 2 * k + 1) * h + y) as *mut __m128i,
                hi,
            );
        }
    }
}

// ------------------------------------------------------------------ weighted rows
/// dst[x] = clamp((sum_j weights[j] * rows[j][x] + 2^13) >> 14, 0, 255); returns how many
/// leading pixels were done (the caller finishes the rest with the plain version).
pub fn weighted_rows(rows: &[&[u8]], weights: &[i16], dst: &mut [u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    match level() {
        // SAFETY: the CPU supports the instructions (runtime check in `level`)
        Level::Avx512 => return unsafe { weighted_rows_avx512(rows, weights, dst) },
        // SAFETY: as above
        Level::Avx2 => return unsafe { weighted_rows_avx2(rows, weights, dst) },
        Level::Plain => {}
    }
    0
}

/// (w0, w1) as one 32-bit lane of two 16-bit weights.
fn weight_pair(weights: &[i16], j: usize) -> i32 {
    let w0 = weights[j] as u16 as i32;
    let w1 = weights.get(j + 1).copied().unwrap_or(0) as u16 as i32;
    w0 | (w1 << 16)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn weighted_rows_avx2(rows: &[&[u8]], weights: &[i16], dst: &mut [u8]) -> usize {
    let len = dst.len();
    let done = len / 16 * 16;
    let zero = [0u8; 16];
    for x in (0..done).step_by(16) {
        let mut lo = _mm256_set1_epi32(1 << 13);
        let mut hi = lo;
        for j in (0..rows.len()).step_by(2) {
            let a = &rows[j][x..x + 16];
            let b = rows.get(j + 1).map_or(&zero[..], |r| &r[x..x + 16]);
            // SAFETY: both slices hold 16 bytes
            let (a, b) = unsafe {
                (
                    _mm_loadu_si128(a.as_ptr() as *const __m128i),
                    _mm_loadu_si128(b.as_ptr() as *const __m128i),
                )
            };
            let a = _mm256_cvtepu8_epi16(a);
            let b = _mm256_cvtepu8_epi16(b);
            let w = _mm256_set1_epi32(weight_pair(weights, j));
            lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(a, b), w));
            hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(a, b), w));
        }
        // per 128-bit lane: pixels (0-3, 4-7) and (8-11, 12-15) -> 16-bit -> 8-bit, then the
        // two lanes' low halves next to each other
        let packed = _mm256_packs_epi32(_mm256_srai_epi32(lo, 14), _mm256_srai_epi32(hi, 14));
        let bytes = _mm256_packus_epi16(packed, packed);
        let ordered = _mm256_permute4x64_epi64(bytes, 0b1000);
        let out = &mut dst[x..x + 16];
        // SAFETY: `out` holds 16 bytes
        unsafe {
            _mm_storeu_si128(
                out.as_mut_ptr() as *mut __m128i,
                _mm256_castsi256_si128(ordered),
            )
        };
    }
    done
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
fn weighted_rows_avx512(rows: &[&[u8]], weights: &[i16], dst: &mut [u8]) -> usize {
    let len = dst.len();
    let done = len / 32 * 32;
    let zero = [0u8; 32];
    for x in (0..done).step_by(32) {
        let mut lo = _mm512_set1_epi32(1 << 13);
        let mut hi = lo;
        for j in (0..rows.len()).step_by(2) {
            let a = &rows[j][x..x + 32];
            let b = rows.get(j + 1).map_or(&zero[..], |r| &r[x..x + 32]);
            // SAFETY: both slices hold 32 bytes
            let (a, b) = unsafe {
                (
                    _mm256_loadu_si256(a.as_ptr() as *const __m256i),
                    _mm256_loadu_si256(b.as_ptr() as *const __m256i),
                )
            };
            let a = _mm512_cvtepu8_epi16(a);
            let b = _mm512_cvtepu8_epi16(b);
            let w = _mm512_set1_epi32(weight_pair(weights, j));
            lo = _mm512_dpwssd_epi32(lo, _mm512_unpacklo_epi16(a, b), w);
            hi = _mm512_dpwssd_epi32(hi, _mm512_unpackhi_epi16(a, b), w);
        }
        // per 128-bit lane as in AVX2; packus leaves lane k's 8 pixels in its low 8 bytes
        let packed = _mm512_packs_epi32(_mm512_srai_epi32(lo, 14), _mm512_srai_epi32(hi, 14));
        let bytes = _mm512_packus_epi16(packed, packed);
        let order = _mm512_set_epi64(0, 0, 0, 0, 6, 4, 2, 0);
        let ordered = _mm512_permutexvar_epi64(order, bytes);
        let out = &mut dst[x..x + 32];
        // SAFETY: `out` holds 32 bytes
        unsafe {
            _mm256_storeu_si256(
                out.as_mut_ptr() as *mut __m256i,
                _mm512_castsi512_si256(ordered),
            )
        };
    }
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_matches_plain() {
        let (h, w) = (37, 70);
        let src: Vec<u8> = (0..h * w).map(|i| (i * 7 % 256) as u8).collect();
        let mut simd = vec![0u8; h * w];
        let mut plain = vec![0u8; h * w];
        transpose(&src, h, w, &mut simd);
        transpose_plain(&src, h, w, &mut plain, 0..h, 0..w);
        assert_eq!(simd, plain);
    }

    #[test]
    fn weighted_rows_match_plain() {
        let len = 100;
        let data: Vec<Vec<u8>> = (0..5)
            .map(|j| (0..len).map(|i| ((i * 13 + j * 101) % 256) as u8).collect())
            .collect();
        for taps in 1..=5 {
            let rows: Vec<&[u8]> = data[..taps].iter().map(|r| &r[..]).collect();
            let weights: Vec<i16> = (0..taps).map(|j| 16384 / taps as i16 + j as i16).collect();
            let mut simd = vec![0u8; len];
            let done = weighted_rows(&rows, &weights, &mut simd);
            for x in 0..done {
                let sum: i32 = (0..taps)
                    .map(|j| weights[j] as i32 * rows[j][x] as i32)
                    .sum();
                let plain = ((sum + (1 << 13)) >> 14).clamp(0, 255) as u8;
                assert_eq!(simd[x], plain, "taps {taps} x {x}");
            }
        }
    }
}
