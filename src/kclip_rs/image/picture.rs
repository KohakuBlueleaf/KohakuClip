//! A decoded picture as planes, whatever produced it: a video frame, a JPEG, a still image.

use super::Plane;

/// How a picture's planes encode color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    /// Y, Cb, Cr. Any chroma subsampling: the chroma planes' own sizes say which.
    Yuv { bt709: bool, full_range: bool },
    /// R, G, B planes of the same size.
    Rgb,
    /// Luma only (full range); the other two planes are unused.
    Gray,
}

impl Color {
    /// The colorspace name the yuv modes report: "bt709" / "bt601", "-full" for full range.
    pub fn name(self) -> &'static str {
        match self {
            Color::Yuv {
                bt709: true,
                full_range: false,
            } => "bt709",
            Color::Yuv {
                bt709: true,
                full_range: true,
            } => "bt709-full",
            Color::Yuv {
                bt709: false,
                full_range: false,
            } => "bt601",
            Color::Yuv {
                bt709: false,
                full_range: true,
            }
            | Color::Gray => "bt601-full",
            Color::Rgb => "rgb",
        }
    }
}

/// Three planes (Y / Cb / Cr, R / G / B, or luma alone) and how to read them.
#[derive(Clone, Copy)]
pub struct Picture<'a> {
    pub planes: [Plane<'a>; 3],
    pub color: Color,
}

impl Picture<'_> {
    /// Size of the full-resolution plane (luma, or any RGB plane).
    pub fn size(&self) -> (usize, usize) {
        (self.planes[0].h, self.planes[0].w)
    }
}

/// Packed RGB / RGBA rows (`h` rows of `stride` bytes, `channels` bytes per pixel) split into R,
/// G, B planes of `buffer` (alpha dropped).
pub fn split_packed<'a>(
    src: &[u8],
    stride: usize,
    (h, w): (usize, usize),
    channels: usize,
    buffer: &'a mut Vec<u8>,
) -> Picture<'a> {
    buffer.resize(3 * h * w, 0);
    let (r, rest) = buffer.split_at_mut(h * w);
    let (g, b) = rest.split_at_mut(h * w);
    for y in 0..h {
        let row = &src[y * stride..y * stride + w * channels];
        let out = y * w..(y + 1) * w;
        let pixels = row.chunks_exact(channels);
        let outputs = r[out.clone()]
            .iter_mut()
            .zip(g[out.clone()].iter_mut())
            .zip(b[out].iter_mut());
        for (pixel, ((r, g), b)) in pixels.zip(outputs) {
            *r = pixel[0];
            *g = pixel[1];
            *b = pixel[2];
        }
    }

    let plane = |data: &'a [u8]| Plane {
        data,
        stride: w,
        h,
        w,
    };
    let (r, rest) = buffer.split_at(h * w);
    let (g, b) = rest.split_at(h * w);
    Picture {
        planes: [plane(r), plane(g), plane(b)],
        color: Color::Rgb,
    }
}
