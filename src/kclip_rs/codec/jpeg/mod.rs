//! JPEG through libjpeg-turbo: the header; decoding straight to Y / Cb / Cr planes (no chroma
//! upsampling, no color conversion: the resize reads the planes as they are), optionally shrunk
//! by the IDCT itself (1/2, 1/4, 1/8: libjpeg-turbo's reduced-size SIMD IDCTs); encoding from
//! planes (fast). `moz`: encoding through mozjpeg (smaller files at the same quality).

mod exif;
mod ffi;
pub mod moz;

use std::borrow::Cow;
use std::ffi::{CStr, c_int};

pub use exif::orientation;

use crate::image::{Color, Picture, Plane};

/// Pictures above this many pixels are refused (decompression bombs).
const MAX_PIXELS: c_int = 1 << 28;

/// A TurboJPEG instance, destroyed on drop.
struct Handle(ffi::Handle);

impl Handle {
    fn new(init: c_int) -> Result<Self, String> {
        // SAFETY: plain constructor; null means failure
        let raw = unsafe { ffi::tj3Init(init) };
        if raw.is_null() {
            return Err("cannot create a TurboJPEG instance".into());
        }
        Ok(Self(raw))
    }

    /// The message of the last error of this instance.
    fn error(&self) -> String {
        // SAFETY: the handle is valid; the string belongs to it and is copied out here
        let message = unsafe { CStr::from_ptr(ffi::tj3GetErrorStr(self.0)) };
        format!("jpeg: {}", message.to_string_lossy())
    }

    fn check(&self, ret: c_int) -> Result<(), String> {
        if ret < 0 {
            return Err(self.error());
        }
        Ok(())
    }

    fn get(&self, param: c_int) -> c_int {
        // SAFETY: the handle is valid
        unsafe { ffi::tj3Get(self.0, param) }
    }

    fn set(&self, param: c_int, value: c_int) -> Result<(), String> {
        // SAFETY: the handle is valid
        self.check(unsafe { ffi::tj3Set(self.0, param, value) })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by tj3Init and is destroyed once
        unsafe { ffi::tj3Destroy(self.0) };
    }
}

/// What the header says about a JPEG.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub w: u32,
    pub h: u32,
    pub gray: bool,
    subsamp: c_int,
}

/// The size of a dimension decoded at 1 / 2^shift (libjpeg-turbo rounds up).
pub fn scaled(dim: u32, shift: u32) -> u32 {
    dim.div_ceil(1 << shift)
}

/// The largest IDCT shrink (1 / 2^shift, shift 0-3) of an (h, w) JPEG whose shrunk planes still
/// hold at least (need_h, need_w) samples.
pub fn max_shift(h: u32, w: u32, need_h: f64, need_w: f64) -> u32 {
    let mut shift = 0;
    while shift < 3 {
        let next = shift + 1;
        if (scaled(h, next) as f64) < need_h || (scaled(w, next) as f64) < need_w {
            break;
        }
        shift = next;
    }
    shift
}

/// Width and height of plane `component` of a (w, h) picture with the given subsampling.
fn plane_size(component: c_int, w: u32, h: u32, subsamp: c_int) -> (usize, usize) {
    // SAFETY: pure functions of their arguments
    let (pw, ph) = unsafe {
        (
            ffi::tj3YUVPlaneWidth(component, w as c_int, subsamp),
            ffi::tj3YUVPlaneHeight(component, h as c_int, subsamp),
        )
    };
    (pw.max(0) as usize, ph.max(0) as usize)
}

/// A decompressor and the planes of its last picture (one per thread, reused).
pub struct Decoder {
    handle: Handle,
    buffer: Vec<u8>,
}

impl Decoder {
    pub fn new() -> Result<Self, String> {
        let handle = Handle::new(ffi::INIT_DECOMPRESS)?;
        handle.set(ffi::PARAM_MAXPIXELS, MAX_PIXELS)?;
        Ok(Self {
            handle,
            buffer: Vec::new(),
        })
    }

    /// Read the header: size and layout. Only 8-bit lossy YCbCr or grayscale JPEGs are taken
    /// (CMYK, 12-bit and lossless JPEGs are refused).
    pub fn header(&mut self, data: &[u8]) -> Result<Header, String> {
        let handle = &self.handle;
        // SAFETY: the handle is valid and `data` is readable for its length
        handle.check(unsafe { ffi::tj3DecompressHeader(handle.0, data.as_ptr(), data.len()) })?;

        if handle.get(ffi::PARAM_LOSSLESS) != 0 || handle.get(ffi::PARAM_PRECISION) != 8 {
            return Err("jpeg: only 8-bit lossy JPEGs are supported".into());
        }
        let colorspace = handle.get(ffi::PARAM_COLORSPACE);
        if colorspace != ffi::CS_YCBCR && colorspace != ffi::CS_GRAY {
            return Err("jpeg: only YCbCr and grayscale JPEGs are supported (not CMYK)".into());
        }
        let subsamp = handle.get(ffi::PARAM_SUBSAMP);
        if subsamp < 0 {
            return Err("jpeg: unusual chroma subsampling".into());
        }
        Ok(Header {
            w: handle.get(ffi::PARAM_JPEGWIDTH) as u32,
            h: handle.get(ffi::PARAM_JPEGHEIGHT) as u32,
            gray: colorspace == ffi::CS_GRAY,
            subsamp,
        })
    }

    /// Decode at 1 / 2^shift of the size (`shift` 0-3) into planes owned by this decoder:
    /// Y, Cb, Cr at their own (subsampled) sizes, full-range BT.601; or luma alone.
    pub fn decode(&mut self, data: &[u8], shift: u32) -> Result<Picture<'_>, String> {
        let header = self.header(data)?;
        let factor = ffi::ScalingFactor {
            num: 1,
            denom: 1 << shift.min(3),
        };
        // SAFETY: the handle is valid; 1/1, 1/2, 1/4 and 1/8 are always supported
        let ret = unsafe { ffi::tj3SetScalingFactor(self.handle.0, factor) };
        self.handle.check(ret)?;
        let w = scaled(header.w, shift.min(3));
        let h = scaled(header.h, shift.min(3));

        // plane layout in the buffer: (stride, rows, offset) per component
        let components = if header.gray { 1 } else { 3 };
        let mut layout = [(0usize, 0usize, 0usize); 3];
        let mut total = 0;
        for (c, slot) in layout.iter_mut().enumerate().take(components) {
            let (stride, rows) = plane_size(c as c_int, w, h, header.subsamp);
            *slot = (stride, rows, total);
            total += stride * rows;
        }
        self.buffer.resize(total, 0);

        let base = self.buffer.as_mut_ptr();
        // SAFETY: each offset lies inside the buffer, which holds every plane
        let mut pointers = layout.map(|(_, _, offset)| unsafe { base.add(offset) });
        let mut strides = layout.map(|(stride, _, _)| stride as c_int);
        // SAFETY: the planes are sized by tj3YUVPlaneWidth / Height for this scaled picture
        let ret = unsafe {
            ffi::tj3DecompressToYUVPlanes8(
                self.handle.0,
                data.as_ptr(),
                data.len(),
                pointers.as_mut_ptr(),
                strides.as_mut_ptr(),
            )
        };
        self.handle.check(ret)?;

        let buffer = &self.buffer;
        let plane = |c: usize, ph: usize, pw: usize| {
            let (stride, rows, offset) = layout[c];
            Plane {
                data: &buffer[offset..offset + stride * rows],
                stride,
                h: ph,
                w: pw,
            }
        };
        let luma = plane(0, h as usize, w as usize);
        if header.gray {
            return Ok(Picture {
                planes: [luma, luma, luma],
                color: Color::Gray,
            });
        }
        let (cw, ch) = plane_size(1, w, h, header.subsamp);
        Ok(Picture {
            planes: [luma, plane(1, ch, cw), plane(2, ch, cw)],
            color: Color::Yuv {
                bt709: false,
                full_range: true,
            },
        })
    }
}

/// Chroma subsampling of the JPEGs the writer makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chroma {
    Yuv420,
    Yuv444,
}

impl Chroma {
    fn subsamp(self) -> c_int {
        match self {
            Chroma::Yuv420 => ffi::SAMP_420,
            Chroma::Yuv444 => ffi::SAMP_444,
        }
    }

    /// The size of the chroma planes of a (w, h) picture.
    pub fn chroma_size(self, w: u32, h: u32) -> (usize, usize) {
        plane_size(1, w, h, self.subsamp())
    }
}

/// `plane` as at least (rows, cols) samples: borrowed when it already is, else copied with its
/// last column and row repeated (rows of `max(cols, stride)` bytes).
fn pad<'a>(plane: &Plane<'a>, rows: usize, cols: usize) -> Cow<'a, [u8]> {
    if plane.stride >= cols && plane.h >= rows && plane.data.len() >= plane.stride * rows {
        return Cow::Borrowed(plane.data);
    }
    let stride = cols.max(plane.stride);
    let mut out = vec![0u8; stride * rows];
    for (y, row) in out.chunks_exact_mut(stride).enumerate() {
        let source = y.min(plane.h - 1) * plane.stride;
        let source = &plane.data[source..source + plane.w];
        row[..plane.w].copy_from_slice(source);
        let last = source[plane.w - 1];
        row[plane.w..].fill(last);
    }
    Cow::Owned(out)
}

/// A compressor at a fixed quality and subsampling (Huffman tables optimized per image).
pub struct Encoder {
    handle: Handle,
    subsamp: c_int,
    gray: Handle,
}

impl Encoder {
    pub fn new(quality: u32, chroma: Chroma) -> Result<Self, String> {
        let make = |subsamp: c_int| {
            let handle = Handle::new(ffi::INIT_COMPRESS)?;
            handle.set(ffi::PARAM_QUALITY, quality.clamp(1, 100) as c_int)?;
            handle.set(ffi::PARAM_SUBSAMP, subsamp)?;
            handle.set(ffi::PARAM_OPTIMIZE, 1)?;
            Ok::<_, String>(handle)
        };
        Ok(Self {
            handle: make(chroma.subsamp())?,
            subsamp: chroma.subsamp(),
            gray: make(ffi::SAMP_GRAY)?,
        })
    }

    /// Encode a (w, h) picture: Y, Cb, Cr planes (full-range BT.601, chroma at this encoder's
    /// subsampling), or luma alone when `gray`.
    pub fn encode(
        &mut self,
        planes: &[Plane],
        w: u32,
        h: u32,
        gray: bool,
    ) -> Result<Vec<u8>, String> {
        let (handle, subsamp) = if gray {
            (&self.gray, ffi::SAMP_GRAY)
        } else {
            (&self.handle, self.subsamp)
        };
        // TurboJPEG reads whole subsampling blocks: luma of an odd-sized 4:2:0 picture padded
        // to even (the edge repeated)
        let padded: Vec<Cow<[u8]>> = planes
            .iter()
            .enumerate()
            .map(|(c, plane)| {
                let (pw, ph) = plane_size(c as c_int, w, h, subsamp);
                pad(plane, ph, pw)
            })
            .collect();
        let pointers: Vec<*const u8> = padded.iter().map(|p| p.as_ptr()).collect();
        let strides: Vec<c_int> = planes
            .iter()
            .enumerate()
            .map(|(c, plane)| plane_size(c as c_int, w, h, subsamp).0.max(plane.stride) as c_int)
            .collect();
        let mut jpeg: *mut u8 = std::ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: each plane holds at least the rows and columns tj3YUVPlaneWidth / Height give
        // for this subsampling (padded above); TurboJPEG allocates the output
        let ret = unsafe {
            ffi::tj3CompressFromYUVPlanes8(
                handle.0,
                pointers.as_ptr(),
                w as c_int,
                strides.as_ptr(),
                h as c_int,
                &mut jpeg,
                &mut size,
            )
        };
        let result = handle.check(ret).map(|()| {
            // SAFETY: on success `jpeg` holds `size` bytes
            unsafe { std::slice::from_raw_parts(jpeg, size) }.to_vec()
        });
        // SAFETY: allocated by TurboJPEG (or null), freed once
        unsafe { ffi::tj3Free(jpeg.cast()) };
        result
    }
}
