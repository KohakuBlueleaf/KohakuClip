//! JPEG encoding through mozjpeg (built in statically by `mozjpeg-sys`): trellis quantization,
//! mozjpeg's quantization tables, optimized Huffman tables, baseline (sequential) scans so the
//! reader's libjpeg-turbo decodes them at full speed. Input: Y, Cb, Cr planes (full-range
//! BT.601) passed as raw data, or luma alone.
//!
//! libjpeg reports errors through `error_exit`, which panics here (mozjpeg-sys builds mozjpeg
//! for unwinding); the panic is caught around the whole encode and becomes an `Err`.

use std::ffi::c_ulong;
use std::panic::{AssertUnwindSafe, catch_unwind};

use mozjpeg_sys::*;

use super::Chroma;
use crate::image::Plane;

/// libjpeg's block size.
const DCT: usize = 8;

/// libjpeg's message buffer size.
const JMSG_LENGTH_MAX: usize = 80;

/// One component as libjpeg's raw-data input wants it: padded to whole MCUs (the edge column
/// and row repeated).
struct Padded {
    data: Vec<u8>,
    stride: usize,
    rows: usize,
}

impl Padded {
    fn new(plane: &Plane, rows: usize, stride: usize) -> Self {
        let mut data = vec![0u8; rows * stride];
        for (y, row) in data.chunks_exact_mut(stride).enumerate() {
            let from = y.min(plane.h - 1) * plane.stride;
            let source = &plane.data[from..from + plane.w];
            row[..plane.w].copy_from_slice(source);
            row[plane.w..].fill(source[plane.w - 1]);
        }
        Self { data, stride, rows }
    }

    fn row(&self, y: usize) -> *const u8 {
        self.data[y.min(self.rows - 1) * self.stride..].as_ptr()
    }
}

/// libjpeg's `format_message`, with the message buffer as the pointer libjpeg writes through.
type FormatMessage = unsafe extern "C-unwind" fn(&mut jpeg_common_struct, *mut u8);

/// libjpeg's errors become panics (caught in `encode`).
unsafe extern "C-unwind" fn error_exit(cinfo: &mut jpeg_common_struct) {
    let mut buffer = [0u8; JMSG_LENGTH_MAX];
    // SAFETY: the error manager is installed and has a format_message function
    if let Some(format) = unsafe { (*cinfo.err).format_message } {
        // SAFETY: the same function with its buffer argument as a raw pointer (same ABI);
        // libjpeg writes a NUL-terminated message of at most JMSG_LENGTH_MAX bytes
        unsafe {
            let format: FormatMessage = std::mem::transmute(format);
            format(cinfo, buffer.as_mut_ptr());
        }
    }
    let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
    std::panic::resume_unwind(Box::new(
        String::from_utf8_lossy(&buffer[..end]).into_owned(),
    ));
}

/// Encode a (w, h) picture at `quality`: Y, Cb, Cr planes at `chroma`'s subsampling, or luma
/// alone when `gray`.
pub fn encode(
    planes: &[Plane],
    w: u32,
    h: u32,
    gray: bool,
    quality: u32,
    chroma: Chroma,
) -> Result<Vec<u8>, String> {
    // SAFETY: zeroed is the state libjpeg expects before jpeg_CreateCompress; both structs
    // live until the end of this function
    let mut err: jpeg_error_mgr = unsafe { std::mem::zeroed() };
    // SAFETY: as above
    let mut cinfo: jpeg_compress_struct = unsafe { std::mem::zeroed() };
    let mut output: *mut u8 = std::ptr::null_mut();
    let mut size: c_ulong = 0;

    let result = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: plain libjpeg calls in the documented order on a struct owned here; every
        // pointer passed in stays valid for the call
        unsafe {
            cinfo.common.err = jpeg_std_error(&mut err);
            err.error_exit = Some(error_exit);
            jpeg_CreateCompress(
                &mut cinfo,
                JPEG_LIB_VERSION,
                std::mem::size_of::<jpeg_compress_struct>(),
            );
            jpeg_mem_dest(&mut cinfo, &mut output, &mut size);
            write(&mut cinfo, planes, w, h, gray, quality, chroma);
        }
    }));
    // SAFETY: created above (or still zeroed: destroying a zeroed struct is a no-op); the
    // output buffer was allocated by jpeg_mem_dest with malloc
    let data = unsafe {
        jpeg_destroy_compress(&mut cinfo);
        let data = if output.is_null() {
            Vec::new()
        } else {
            std::slice::from_raw_parts(output, size as usize).to_vec()
        };
        libc::free(output.cast());
        data
    };
    match result {
        Ok(()) => Ok(data),
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "unknown error".into());
            Err(format!("mozjpeg: {message}"))
        }
    }
}

/// Set up `cinfo` (mozjpeg's default profile, baseline scans) and write the planes.
///
/// # Safety
/// `cinfo` must be created with a destination set.
unsafe fn write(
    cinfo: &mut jpeg_compress_struct,
    planes: &[Plane],
    w: u32,
    h: u32,
    gray: bool,
    quality: u32,
    chroma: Chroma,
) {
    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = if gray { 1 } else { 3 };
    cinfo.in_color_space = if gray {
        J_COLOR_SPACE::JCS_GRAYSCALE
    } else {
        J_COLOR_SPACE::JCS_YCbCr
    };
    // SAFETY: the caller's contract (a created compressor); the component array exists after
    // jpeg_set_defaults
    unsafe {
        jpeg_c_set_int_param(
            cinfo,
            J_INT_PARAM::JINT_COMPRESS_PROFILE,
            JCP_MAX_COMPRESSION as i32,
        );
        jpeg_set_defaults(cinfo);
        jpeg_set_quality(cinfo, quality.clamp(1, 100) as i32, 1);
        let luma = &mut *cinfo.comp_info;
        let factor = if chroma == Chroma::Yuv420 && !gray {
            2
        } else {
            1
        };
        luma.h_samp_factor = factor;
        luma.v_samp_factor = factor;
    }
    cinfo.optimize_coding = 1;
    // baseline: no progressive scan script (mozjpeg's default profile would make one)
    cinfo.num_scans = 0;
    cinfo.scan_info = std::ptr::null();
    cinfo.raw_data_in = 1;

    // SAFETY: as above; start_compress sets each component's block counts
    unsafe { jpeg_start_compress(cinfo, 1) };
    let components = cinfo.num_components as usize;
    let max_v = cinfo.max_v_samp_factor as usize;
    let max_h = cinfo.max_h_samp_factor as usize;
    let mcu_rows = (h as usize).div_ceil(DCT * max_v);
    let mcu_cols = (w as usize).div_ceil(DCT * max_h);
    let padded: Vec<Padded> = (0..components)
        .map(|c| {
            // SAFETY: comp_info holds num_components entries
            let info = unsafe { &*cinfo.comp_info.add(c) };
            let rows = mcu_rows * DCT * info.v_samp_factor as usize;
            let stride = mcu_cols * DCT * info.h_samp_factor as usize;
            Padded::new(&planes[c], rows, stride)
        })
        .collect();

    // one MCU row per call: max_v * 8 luma rows, the matching chroma rows
    let lines = DCT * max_v;
    for mcu in 0..mcu_rows {
        let rows: Vec<Vec<*const u8>> = (0..components)
            .map(|c| {
                let per_mcu = padded[c].rows / mcu_rows;
                (0..per_mcu)
                    .map(|y| padded[c].row(mcu * per_mcu + y))
                    .collect()
            })
            .collect();
        let arrays: Vec<JSAMPARRAY> = rows.iter().map(|r| r.as_ptr()).collect();
        // SAFETY: each array holds one MCU row of its component, rows wide enough for the
        // padded width
        unsafe { jpeg_write_raw_data(cinfo, arrays.as_ptr(), lines as u32) };
    }
    // SAFETY: every scanline was written
    unsafe { jpeg_finish_compress(cinfo) };
}
