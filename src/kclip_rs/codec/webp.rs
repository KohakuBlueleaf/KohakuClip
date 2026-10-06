//! WebP images through libwebp directly: decoded straight to its Y / Cb / Cr 4:2:0 planes
//! (no chroma upsampling, no color conversion; limited-range BT.601), into per-thread buffers.

use std::cell::RefCell;
use std::ffi::c_int;

use crate::image::{Color, Picture, Plane};

#[link(name = "webp")]
unsafe extern "C" {
    fn WebPGetInfo(data: *const u8, size: usize, width: *mut c_int, height: *mut c_int) -> c_int;
    #[allow(clippy::too_many_arguments)]
    fn WebPDecodeYUVInto(
        data: *const u8,
        size: usize,
        luma: *mut u8,
        luma_size: usize,
        luma_stride: c_int,
        u: *mut u8,
        u_size: usize,
        u_stride: c_int,
        v: *mut u8,
        v_size: usize,
        v_stride: c_int,
    ) -> *mut u8;
}

thread_local! {
    static PLANES: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Decode one WebP image and run `f` on its planes.
pub fn decode<T>(data: &[u8], f: impl FnOnce(&Picture) -> Result<T, String>) -> Result<T, String> {
    let (mut w, mut h) = (0, 0);
    // SAFETY: `data` is readable for its length; w and h are written on success
    if unsafe { WebPGetInfo(data.as_ptr(), data.len(), &mut w, &mut h) } == 0 {
        return Err("webp: not a WebP image".into());
    }
    let (h, w) = (h as usize, w as usize);
    let (ch, cw) = (h.div_ceil(2), w.div_ceil(2));

    PLANES.with_borrow_mut(|buffer| {
        buffer.resize(h * w + 2 * ch * cw, 0);
        let (luma, chroma) = buffer.split_at_mut(h * w);
        let (u, v) = chroma.split_at_mut(ch * cw);
        // SAFETY: each plane buffer holds its plane at the given stride
        let got = unsafe {
            WebPDecodeYUVInto(
                data.as_ptr(),
                data.len(),
                luma.as_mut_ptr(),
                luma.len(),
                w as c_int,
                u.as_mut_ptr(),
                u.len(),
                cw as c_int,
                v.as_mut_ptr(),
                v.len(),
                cw as c_int,
            )
        };
        if got.is_null() {
            return Err("webp: decoding failed".into());
        }

        let (luma, chroma) = buffer.split_at(h * w);
        let (u, v) = chroma.split_at(ch * cw);
        let plane = |data, h, w| Plane {
            data,
            stride: w,
            h,
            w,
        };
        let picture = Picture {
            planes: [plane(luma, h, w), plane(u, ch, cw), plane(v, ch, cw)],
            color: Color::Yuv {
                bt709: false,
                full_range: false,
            },
        };
        f(&picture)
    })
}
