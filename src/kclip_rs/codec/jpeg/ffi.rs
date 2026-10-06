//! The TurboJPEG 3 functions and constants KohakuClip uses (libjpeg-turbo >= 3.0; the values
//! are those of turbojpeg.h, stable across 3.x).

use std::ffi::{c_char, c_int, c_uchar, c_void};

pub type Handle = *mut c_void;

/// A decompression scaling factor, num / denom.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalingFactor {
    pub num: c_int,
    pub denom: c_int,
}

pub const INIT_COMPRESS: c_int = 0;
pub const INIT_DECOMPRESS: c_int = 1;

pub const PARAM_QUALITY: c_int = 3;
pub const PARAM_SUBSAMP: c_int = 4;
pub const PARAM_JPEGWIDTH: c_int = 5;
pub const PARAM_JPEGHEIGHT: c_int = 6;
pub const PARAM_PRECISION: c_int = 7;
pub const PARAM_COLORSPACE: c_int = 8;
pub const PARAM_OPTIMIZE: c_int = 11;
pub const PARAM_LOSSLESS: c_int = 15;
pub const PARAM_MAXPIXELS: c_int = 24;

pub const SAMP_444: c_int = 0;
pub const SAMP_420: c_int = 2;
pub const SAMP_GRAY: c_int = 3;

pub const CS_YCBCR: c_int = 1;
pub const CS_GRAY: c_int = 2;

#[link(name = "turbojpeg")]
unsafe extern "C" {
    pub fn tj3Init(init_type: c_int) -> Handle;
    pub fn tj3Destroy(handle: Handle);
    pub fn tj3Set(handle: Handle, param: c_int, value: c_int) -> c_int;
    pub fn tj3Get(handle: Handle, param: c_int) -> c_int;
    pub fn tj3GetErrorStr(handle: Handle) -> *mut c_char;
    pub fn tj3Free(buffer: *mut c_void);

    pub fn tj3YUVPlaneWidth(component: c_int, width: c_int, subsamp: c_int) -> c_int;
    pub fn tj3YUVPlaneHeight(component: c_int, height: c_int, subsamp: c_int) -> c_int;

    pub fn tj3DecompressHeader(handle: Handle, jpeg: *const c_uchar, size: usize) -> c_int;
    pub fn tj3SetScalingFactor(handle: Handle, factor: ScalingFactor) -> c_int;
    pub fn tj3DecompressToYUVPlanes8(
        handle: Handle,
        jpeg: *const c_uchar,
        size: usize,
        planes: *mut *mut c_uchar,
        strides: *mut c_int,
    ) -> c_int;

    pub fn tj3CompressFromYUVPlanes8(
        handle: Handle,
        planes: *const *const c_uchar,
        width: c_int,
        strides: *const c_int,
        height: c_int,
        jpeg: *mut *mut c_uchar,
        size: *mut usize,
    ) -> c_int;
}
