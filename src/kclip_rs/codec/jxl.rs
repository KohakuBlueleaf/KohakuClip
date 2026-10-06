//! JPEG XL images through libjxl directly: one decoder per thread (reset between images), no
//! parallel runner (single-threaded), 8-bit RGB output split into planes; orientation and spot
//! colors are not rendered.
//!
//! Only the leading fields of `JxlBasicInfo` are declared; the struct is allocated larger.

use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::ptr;

use crate::image::{Picture, split_packed};

/// `JxlBasicInfo`: the size; the rest is libjxl's.
#[repr(C)]
struct BasicInfo {
    have_container: c_int,
    xsize: u32,
    ysize: u32,
    rest: [u32; 128],
}

/// `JxlPixelFormat`.
#[repr(C)]
struct PixelFormat {
    num_channels: u32,
    data_type: c_int,
    endianness: c_int,
    align: usize,
}

const SUCCESS: c_int = 0;
const NEED_IMAGE_OUT_BUFFER: c_int = 5;
const BASIC_INFO: c_int = 0x40;
const FULL_IMAGE: c_int = 0x1000;
const TYPE_UINT8: c_int = 2;
const NATIVE_ENDIAN: c_int = 0;
const TRUE: c_int = 1;
const FALSE: c_int = 0;

/// Packed 8-bit RGB.
const RGB8: PixelFormat = PixelFormat {
    num_channels: 3,
    data_type: TYPE_UINT8,
    endianness: NATIVE_ENDIAN,
    align: 0,
};

#[link(name = "jxl")]
unsafe extern "C" {
    fn JxlDecoderCreate(memory_manager: *const c_void) -> *mut c_void;
    fn JxlDecoderReset(dec: *mut c_void);
    fn JxlDecoderDestroy(dec: *mut c_void);
    fn JxlDecoderSubscribeEvents(dec: *mut c_void, events: c_int) -> c_int;
    fn JxlDecoderSetKeepOrientation(dec: *mut c_void, keep: c_int) -> c_int;
    fn JxlDecoderSetRenderSpotcolors(dec: *mut c_void, render: c_int) -> c_int;
    fn JxlDecoderSetInput(dec: *mut c_void, data: *const u8, size: usize) -> c_int;
    fn JxlDecoderCloseInput(dec: *mut c_void);
    fn JxlDecoderProcessInput(dec: *mut c_void) -> c_int;
    fn JxlDecoderGetBasicInfo(dec: *const c_void, info: *mut BasicInfo) -> c_int;
    fn JxlDecoderImageOutBufferSize(
        dec: *const c_void,
        format: *const PixelFormat,
        size: *mut usize,
    ) -> c_int;
    fn JxlDecoderSetImageOutBuffer(
        dec: *mut c_void,
        format: *const PixelFormat,
        buffer: *mut c_void,
        size: usize,
    ) -> c_int;
}

/// A libjxl decoder and its buffers: the packed RGB output, then the planes.
struct Decoder {
    dec: *mut c_void,
    packed: Vec<u8>,
    planes: Vec<u8>,
}

impl Decoder {
    fn new() -> Result<Self, String> {
        // SAFETY: plain constructor (default memory manager); null means failure
        let dec = unsafe { JxlDecoderCreate(ptr::null()) };
        if dec.is_null() {
            return Err("jxl: cannot create a decoder".into());
        }
        Ok(Self {
            dec,
            packed: Vec::new(),
            planes: Vec::new(),
        })
    }

    /// Decode `data` into `self.packed` (h rows of 3 * w bytes); returns (h, w).
    fn decode_packed(&mut self, data: &[u8]) -> Result<(usize, usize), String> {
        let dec = self.dec;
        let check = |ret: c_int, what: &str| {
            if ret != SUCCESS {
                return Err(format!("jxl: {what} failed"));
            }
            Ok(())
        };
        // SAFETY: the decoder is valid; `data` outlives the decoding below (input is closed and
        // fully processed before returning, and the decoder is reset before the next image)
        unsafe {
            JxlDecoderReset(dec);
            check(
                JxlDecoderSubscribeEvents(dec, BASIC_INFO | FULL_IMAGE),
                "events",
            )?;
            check(JxlDecoderSetKeepOrientation(dec, TRUE), "orientation")?;
            check(JxlDecoderSetRenderSpotcolors(dec, FALSE), "spot colors")?;
            check(JxlDecoderSetInput(dec, data.as_ptr(), data.len()), "input")?;
            JxlDecoderCloseInput(dec);
        }

        let mut size = (0, 0);
        loop {
            // SAFETY: the decoder is valid and has its input
            let status = unsafe { JxlDecoderProcessInput(dec) };
            match status {
                BASIC_INFO => {
                    // SAFETY: zeroed is a valid BasicInfo; libjxl writes the real fields
                    let mut info: BasicInfo = unsafe { std::mem::zeroed() };
                    check(
                        // SAFETY: basic info is available at this event
                        unsafe { JxlDecoderGetBasicInfo(dec, &mut info) },
                        "basic info",
                    )?;
                    size = (info.ysize as usize, info.xsize as usize);
                }
                NEED_IMAGE_OUT_BUFFER => {
                    let mut bytes = 0;
                    check(
                        // SAFETY: the decoder is valid; `bytes` is written
                        unsafe { JxlDecoderImageOutBufferSize(dec, &RGB8, &mut bytes) },
                        "buffer size",
                    )?;
                    self.packed.resize(bytes, 0);
                    let buffer = self.packed.as_mut_ptr().cast();
                    check(
                        // SAFETY: `packed` holds `bytes` bytes and is not touched until the
                        // full image event
                        unsafe { JxlDecoderSetImageOutBuffer(dec, &RGB8, buffer, bytes) },
                        "output buffer",
                    )?;
                }
                FULL_IMAGE | SUCCESS => break,
                _ => return Err(format!("jxl: decoding failed (status {status})")),
            }
        }
        if size.0 == 0 || self.packed.len() < 3 * size.0 * size.1 {
            return Err("jxl: no picture decoded".into());
        }
        Ok(size)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: created by JxlDecoderCreate, destroyed once
        unsafe { JxlDecoderDestroy(self.dec) };
    }
}

thread_local! {
    static DECODER: RefCell<Option<Decoder>> = const { RefCell::new(None) };
}

/// Decode one JPEG XL image and run `f` on its R, G, B planes.
pub fn decode<T>(data: &[u8], f: impl FnOnce(&Picture) -> Result<T, String>) -> Result<T, String> {
    DECODER.with_borrow_mut(|slot| {
        let decoder = match slot {
            Some(decoder) => decoder,
            empty => empty.insert(Decoder::new()?),
        };
        let (h, w) = decoder.decode_packed(data)?;
        let picture = split_packed(&decoder.packed, 3 * w, (h, w), 3, &mut decoder.planes);
        f(&picture)
    })
}
