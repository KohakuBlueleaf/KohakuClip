//! AV1 images (one temporal unit of raw OBUs) through libdav1d directly: one single-threaded,
//! low-latency decoder per thread, the picture's Y / Cb / Cr planes read in place.
//!
//! Only the leading fields of dav1d's structs are declared (stable across libdav1d 6 and 7);
//! each struct is allocated larger than either version's, and dav1d fills it. The sequence
//! header's color range moved between the two (an `int` before dav1d 1.3, a byte since).

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::OnceLock;

use crate::image::{Color, Picture, Plane};

/// `Dav1dSettings`: the threading fields, then the rest (filled by `dav1d_default_settings`).
#[repr(C)]
struct Settings {
    n_threads: c_int,
    max_frame_delay: c_int,
    apply_grain: c_int,
    rest: [u64; 32],
}

/// `Dav1dPictureParameters`.
#[repr(C)]
#[derive(Clone, Copy)]
struct PictureParameters {
    w: c_int,
    h: c_int,
    layout: c_uint,
    bpc: c_int,
}

/// `Dav1dPicture`: headers, planes, strides and parameters; the rest is dav1d's.
#[repr(C)]
struct RawPicture {
    seq_hdr: *const u8,
    frame_hdr: *const c_void,
    data: [*mut u8; 3],
    stride: [isize; 2],
    p: PictureParameters,
    rest: [u64; 48],
}

/// Byte offsets in `Dav1dSequenceHeader`: the matrix coefficients (both versions), the color
/// range (dav1d >= 1.3: a byte after the one-byte `hbd`; before: an `int` after the `int` one).
const MATRIX_OFFSET: usize = 24;
const RANGE_OFFSET_BYTE: usize = 33;
const RANGE_OFFSET_INT: usize = 36;

/// `Dav1dData`: a buffer dav1d owns (made by `dav1d_data_create`).
#[repr(C)]
struct Data {
    data: *const u8,
    sz: usize,
    rest: [u64; 8],
}

const LAYOUT_I400: c_uint = 0;
const LAYOUT_I420: c_uint = 1;
const LAYOUT_I422: c_uint = 2;
/// `DAV1D_MC_BT709` and `DAV1D_MC_UNKNOWN`: read as BT.709 (as for video).
const MC_BT709: c_uint = 1;
const MC_UNKNOWN: c_uint = 2;
const EAGAIN: c_int = -11;

#[link(name = "dav1d")]
unsafe extern "C" {
    fn dav1d_version() -> *const c_char;
    fn dav1d_default_settings(settings: *mut Settings);
    fn dav1d_open(context: *mut *mut c_void, settings: *const Settings) -> c_int;
    fn dav1d_close(context: *mut *mut c_void);
    fn dav1d_flush(context: *mut c_void);
    fn dav1d_data_create(data: *mut Data, size: usize) -> *mut u8;
    fn dav1d_data_unref(data: *mut Data);
    fn dav1d_send_data(context: *mut c_void, data: *mut Data) -> c_int;
    fn dav1d_get_picture(context: *mut c_void, picture: *mut RawPicture) -> c_int;
    fn dav1d_picture_unref(picture: *mut RawPicture);
}

/// A dav1d context, closed on drop.
struct Decoder(*mut c_void);

impl Decoder {
    /// One thread, no frame delay: each image is decoded within the calls that send it.
    fn open() -> Result<Self, String> {
        // SAFETY: all-zero is a valid `Settings` (integers)
        let mut settings: Settings = unsafe { std::mem::zeroed() };
        // SAFETY: `Settings` is larger than Dav1dSettings, which the call fills in
        unsafe { dav1d_default_settings(&mut settings) };
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        let mut context = ptr::null_mut();
        // SAFETY: valid settings; the context is written on success
        let ret = unsafe { dav1d_open(&mut context, &settings) };
        if ret < 0 || context.is_null() {
            return Err(format!("av1: cannot open dav1d ({ret})"));
        }
        Ok(Self(context))
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: opened by dav1d_open, closed once
        unsafe { dav1d_close(&mut self.0) };
    }
}

/// Whether the loaded libdav1d is 1.3 or newer (byte-sized sequence header fields).
fn compact_header() -> bool {
    static COMPACT: OnceLock<bool> = OnceLock::new();
    *COMPACT.get_or_init(|| {
        // SAFETY: a static NUL-terminated version string, e.g. "1.5.4"
        let version = unsafe { CStr::from_ptr(dav1d_version()) }.to_string_lossy();
        let mut parts = version
            .split(['.', '-'])
            .map(|p| p.parse::<u32>().unwrap_or(0));
        let (major, minor) = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
        (major, minor) >= (1, 3)
    })
}

/// (matrix coefficients, full range) of a picture's sequence header.
fn color_of(seq_hdr: *const u8) -> (c_uint, bool) {
    // SAFETY: a decoded picture's sequence header holds both fields at these offsets (per the
    // loaded version); unaligned reads need no alignment
    unsafe {
        let matrix = ptr::read_unaligned(seq_hdr.add(MATRIX_OFFSET).cast::<c_uint>());
        let full_range = if compact_header() {
            *seq_hdr.add(RANGE_OFFSET_BYTE) != 0
        } else {
            ptr::read_unaligned(seq_hdr.add(RANGE_OFFSET_INT).cast::<c_int>()) != 0
        };
        (matrix, full_range)
    }
}

thread_local! {
    static DECODER: RefCell<Option<Decoder>> = const { RefCell::new(None) };
}

/// The planes of a decoded picture (8-bit only).
fn planes(raw: &RawPicture) -> Result<Picture<'_>, String> {
    let p = raw.p;
    if p.bpc != 8 {
        return Err(format!("av1: {}-bit pictures are not supported", p.bpc));
    }
    let (h, w) = (p.h as usize, p.w as usize);
    let plane = |i: usize, (ph, pw): (usize, usize)| {
        let stride = raw.stride[i.min(1)] as usize;
        // SAFETY: dav1d's planes hold `ph` rows of `stride` bytes while the picture is held
        let data = unsafe { std::slice::from_raw_parts(raw.data[i], stride * (ph - 1) + pw) };
        Plane {
            data,
            stride,
            h: ph,
            w: pw,
        }
    };
    let luma = plane(0, (h, w));
    if p.layout == LAYOUT_I400 {
        return Ok(Picture {
            planes: [luma, luma, luma],
            color: Color::Gray,
        });
    }
    let chroma = match p.layout {
        LAYOUT_I420 => (h.div_ceil(2), w.div_ceil(2)),
        LAYOUT_I422 => (h, w.div_ceil(2)),
        _ => (h, w),
    };
    if raw.seq_hdr.is_null() {
        return Err("av1: picture without a sequence header".into());
    }
    let (matrix, full_range) = color_of(raw.seq_hdr);
    Ok(Picture {
        planes: [luma, plane(1, chroma), plane(2, chroma)],
        color: Color::Yuv {
            bt709: matrix == MC_BT709 || matrix == MC_UNKNOWN,
            full_range,
        },
    })
}

/// Decode one stored AV1 image (sequence header + one intra frame) and run `f` on its planes.
pub fn decode<T>(data: &[u8], f: impl FnOnce(&Picture) -> Result<T, String>) -> Result<T, String> {
    DECODER.with_borrow_mut(|slot| {
        let decoder = match slot {
            Some(decoder) => decoder,
            empty => empty.insert(Decoder::open()?),
        };
        // SAFETY: the context is open; the previous image is fully drained
        unsafe { dav1d_flush(decoder.0) };

        // SAFETY: zeroed is the empty Dav1dData
        let mut input: Data = unsafe { std::mem::zeroed() };
        // SAFETY: allocates `data.len()` bytes owned by `input`
        let buffer = unsafe { dav1d_data_create(&mut input, data.len()) };
        if buffer.is_null() {
            return Err("av1: out of memory".into());
        }
        // SAFETY: `buffer` holds data.len() bytes
        unsafe { ptr::copy_nonoverlapping(data.as_ptr(), buffer, data.len()) };

        // SAFETY: zeroed is the empty Dav1dPicture
        let mut raw: RawPicture = unsafe { std::mem::zeroed() };
        let got = loop {
            if input.sz > 0 {
                // SAFETY: the context is open; dav1d consumes (part of) `input`
                let ret = unsafe { dav1d_send_data(decoder.0, &mut input) };
                if ret < 0 && ret != EAGAIN {
                    break ret;
                }
            }
            // SAFETY: the context is open; `raw` is written on success
            let ret = unsafe { dav1d_get_picture(decoder.0, &mut raw) };
            if ret != EAGAIN || input.sz == 0 {
                break ret;
            }
        };
        // SAFETY: what is left of `input` (or nothing) is released
        unsafe { dav1d_data_unref(&mut input) };
        if got < 0 {
            return Err(format!("av1: no picture decoded ({got})"));
        }
        let result = planes(&raw).and_then(|picture| f(&picture));
        // SAFETY: the picture was returned by dav1d_get_picture and is released once
        unsafe { dav1d_picture_unref(&mut raw) };
        result
    })
}
