//! YUV -> RGB conversion of output pixels (BT.601 / BT.709, limited or full range).

/// YUV -> RGB coefficients (Q14), per colorspace and range.
pub struct Matrix {
    luma: i32,
    luma_offset: i32,
    r_v: i32,
    g_u: i32,
    g_v: i32,
    b_u: i32,
}

impl Matrix {
    pub fn new(bt709: bool, full_range: bool) -> Self {
        // (Kr-derived, ...) coefficients: R = Y + r_v V, G = Y + g_u U + g_v V, B = Y + b_u U
        let (r_v, g_u, g_v, b_u) = if bt709 {
            (1.5748, -0.187324, -0.468124, 1.8556)
        } else {
            (1.402, -0.344136, -0.714136, 1.772)
        };
        let chroma_scale = if full_range { 1.0 } else { 255.0 / 224.0 };
        let q14 = |c: f64| (c * chroma_scale * 16384.0).round() as i32;
        Self {
            luma: if full_range { 16384 } else { 19077 },
            luma_offset: if full_range { 0 } else { 16 },
            r_v: q14(r_v),
            g_u: q14(g_u),
            g_v: q14(g_v),
            b_u: q14(b_u),
        }
    }
}

/// Y, U, V at output resolution -> RGB planes (CHW) with flips.
#[allow(clippy::too_many_arguments)]
pub fn to_rgb(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    m: &Matrix,
    oh: usize,
    ow: usize,
    hflip: bool,
    vflip: bool,
    out: &mut [u8],
) {
    let (r_plane, rest) = out.split_at_mut(oh * ow);
    let (g_plane, b_plane) = rest.split_at_mut(oh * ow);
    for row in 0..oh {
        let src = row * ow..(row + 1) * ow;
        let dst_row = if vflip { oh - 1 - row } else { row };
        let dst = dst_row * ow..(dst_row + 1) * ow;
        let r = &mut r_plane[dst.clone()];
        let g = &mut g_plane[dst.clone()];
        let b = &mut b_plane[dst];
        convert_row(&y[src.clone()], &u[src.clone()], &v[src], m, r, g, b);
        if hflip {
            r.reverse();
            g.reverse();
            b.reverse();
        }
    }
}

/// One row: straight-line integer math over zipped slices (no indexing, no branches), so the
/// compiler vectorizes it.
fn convert_row(y: &[u8], u: &[u8], v: &[u8], m: &Matrix, r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
    let pixel = |c: i32| ((c + (1 << 17)) >> 18).clamp(0, 255) as u8;
    let pixels = y.iter().zip(u).zip(v);
    let outputs = r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut());
    for (((&y, &u), &v), ((r, g), b)) in pixels.zip(outputs) {
        let luma = (y as i32 - m.luma_offset) * m.luma * 16;
        let cb = (u as i32 - 128) * 16;
        let cr = (v as i32 - 128) * 16;
        *r = pixel(luma + m.r_v * cr);
        *g = pixel(luma + m.g_u * cb + m.g_v * cr);
        *b = pixel(luma + m.b_u * cb);
    }
}

/// R, G, B planes -> full-range BT.601 Y, Cb, Cr planes of the same size (the JPEG convention;
/// Q14 fixed point, rounded).
pub fn rgb_to_yuv(r: &[u8], g: &[u8], b: &[u8], y: &mut [u8], u: &mut [u8], v: &mut [u8]) {
    let q = |c: f64| (c * 16384.0).round() as i32;
    let (yr, yg, yb) = (q(0.299), q(0.587), q(0.114));
    let (ur, ug, ub) = (q(-0.168_736), q(-0.331_264), q(0.5));
    let (vr, vg, vb) = (q(0.5), q(-0.418_688), q(-0.081_312));
    let half = 1 << 13;
    let center = 128 << 14;
    let pixel = |c: i32| (c >> 14).clamp(0, 255) as u8;

    let inputs = r.iter().zip(g).zip(b);
    let outputs = y.iter_mut().zip(u.iter_mut()).zip(v.iter_mut());
    for (((&r, &g), &b), ((y, u), v)) in inputs.zip(outputs) {
        let (r, g, b) = (r as i32, g as i32, b as i32);
        *y = pixel(yr * r + yg * g + yb * b + half);
        *u = pixel(ur * r + ug * g + ub * b + center + half);
        *v = pixel(vr * r + vg * g + vb * b + center + half);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RGB -> YCbCr -> RGB (full range BT.601) returns within rounding.
    #[test]
    fn rgb_yuv_round_trip() {
        let n = 4096;
        let r: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();
        let g: Vec<u8> = (0..n).map(|i| (i * 91 % 256) as u8).collect();
        let b: Vec<u8> = (0..n).map(|i| (i * 53 % 256) as u8).collect();
        let (mut y, mut u, mut v) = (vec![0; n], vec![0; n], vec![0; n]);
        rgb_to_yuv(&r, &g, &b, &mut y, &mut u, &mut v);
        let mut back = vec![0u8; 3 * n];
        to_rgb(
            &y,
            &u,
            &v,
            &Matrix::new(false, true),
            1,
            n,
            false,
            false,
            &mut back,
        );
        for (channel, original) in back.chunks_exact(n).zip([&r, &g, &b]) {
            for (&got, &want) in channel.iter().zip(original.iter()) {
                assert!((got as i32 - want as i32).abs() <= 2, "{got} vs {want}");
            }
        }
    }
}
